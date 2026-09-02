#!/usr/bin/env python3
"""Focused tests for tools/convert_mlx_quant.py (Mei-owned conversion wrapper).

Deterministic, dependency-light, non-network. A tiny POSIX-shell fake
converter stands in for mlx_lm.convert so run()/provenance behavior is tested
without the real 55 GB conversion.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_mlx_quant as cq

FAKE_CONVERTER = """#!/bin/sh
# Minimal fake mlx_lm.convert for wrapper tests. Parses --mlx-path, honours
# FAKE_CONVERT_FAIL=1 to emulate a conversion failure.
out=""
fail="${FAKE_CONVERT_FAIL:-0}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mlx-path) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [ -z "$out" ]; then echo "fake: missing --mlx-path" >&2; exit 9; fi
if [ "$fail" = "1" ]; then echo "fake convert: intentional failure" >&2; exit 7; fi
mkdir -p "$out"
printf '{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}' > "$out/config.json"
printf 'FAKEDATA' > "$out/model-00001-of-00001.safetensors"
echo "fake convert: wrote $out"
exit 0
"""

REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_BYTES = 55_563_006_776  # Qwen/Qwen3.8-27B tree-verified source size
GIB = 1024**3


class ConvertWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_source(self, name="src", *, with_config=True, extra=""):
        src = self.root / name
        src.mkdir()
        if with_config:
            (src / "config.json").write_text('{"model_type":"qwen3_5"}')
        (src / "model-00001-of-00001.safetensors").write_bytes(b"W" * 16)
        if extra:
            (src / extra).write_text("x" * 8)
        return src

    def make_fake_converter(self):
        path = self.root / "fake_convert.sh"
        path.write_text(FAKE_CONVERTER)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    def out_dir(self, name="out"):
        return str(self.root / name)

    # ---- quant argument validation ------------------------------------

    def test_quant_args_accept_reference_recipe(self):
        cq.validate_quant_args(5, 64, "affine", "bfloat16")  # the note recipe
        cq.validate_quant_args(4, 64, "affine", "float16")
        cq.validate_quant_args(8, 128, "mxfp8", "float32")

    def test_quant_args_reject_invalid(self):
        for kwargs in (
            {"q_bits": 1, "q_group_size": 64, "q_mode": "affine", "q_dtype": "bfloat16"},
            {"q_bits": 9, "q_group_size": 64, "q_mode": "affine", "q_dtype": "bfloat16"},
            {"q_bits": 5, "q_group_size": 31, "q_mode": "affine", "q_dtype": "bfloat16"},
            {"q_bits": 5, "q_group_size": 0, "q_mode": "affine", "q_dtype": "bfloat16"},
            {"q_bits": 5, "q_group_size": 100, "q_mode": "affine", "q_dtype": "bfloat16"},
            {"q_bits": 5, "q_group_size": 64, "q_mode": "ud", "q_dtype": "bfloat16"},
            {"q_bits": 5, "q_group_size": 64, "q_mode": "affine", "q_dtype": "fp16"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(cq.UsageError):
                    cq.validate_quant_args(**kwargs)

    # ---- source identity validation -----------------------------------

    def test_hub_source_requires_pinned_40_hex_revision(self):
        with self.assertRaises(cq.UsageError):
            cq.validate_source("Qwen/Qwen3.8-27B", None)
        with self.assertRaises(cq.UsageError):
            cq.validate_source("Qwen/Qwen3.8-27B", "main")
        plan = cq.validate_source("Qwen/Qwen3.8-27B", REV)
        self.assertEqual(plan["kind"], "hub")
        self.assertEqual(plan["revision"], REV)

    def test_local_source_requires_config_json(self):
        with self.assertRaises(cq.UsageError):
            cq.validate_source(str(self.make_source(name="src-noconfig", with_config=False)), REV)
        plan = cq.validate_source(str(self.make_source()), REV)
        self.assertEqual(plan["kind"], "local")
        self.assertEqual(plan["revision"], REV)
        with self.assertRaises(cq.UsageError):
            cq.validate_source(str(self.make_source(name="badrev")), "not-a-hash")

    def test_missing_local_source_is_rejected(self):
        with self.assertRaises(cq.UsageError):
            cq.validate_source(str(self.root / "nope"), REV)

    # ---- source tree digest -------------------------------------------

    def test_dir_digest_is_deterministic_and_structural(self):
        a = self.make_source(name="t1")
        b = self.make_source(name="t2")
        da, bytes_a = cq.dir_digest(a)
        db, bytes_b = cq.dir_digest(b)
        self.assertEqual(da, db)
        self.assertEqual(bytes_a, bytes_b)
        # documented structural scope: name+size, not content
        (a / "config.json").write_text('{"model_type":"qwen3_6"}')  # same length
        self.assertEqual(cq.dir_digest(a)[0], da)
        # name/size changes DO change the digest
        (a / "extra.txt").write_text("y" * 16)
        self.assertNotEqual(cq.dir_digest(a)[0], da)

    # ---- output estimation + disk guard -------------------------------

    def test_estimate_output_bytes(self):
        est = cq.estimate_output_bytes(SOURCE_BYTES, 5)
        self.assertEqual(est, int(SOURCE_BYTES * 5 / 16) + cq.OUTPUT_OVERHEAD_BYTES)
        self.assertGreater(cq.estimate_output_bytes(SOURCE_BYTES, 6), est)
        self.assertGreater(est, cq.estimate_output_bytes(SOURCE_BYTES, 4))

    def test_fits_requirement_math(self):
        out = cq.estimate_output_bytes(SOURCE_BYTES, 5)
        need = SOURCE_BYTES + out
        self.assertTrue(cq.fits_requirement(SOURCE_BYTES, out, need + 20 * GIB, 20))
        self.assertFalse(cq.fits_requirement(SOURCE_BYTES, out, need + 20 * GIB - 1, 20))
        self.assertFalse(cq.fits_requirement(SOURCE_BYTES, out, need + 21 * GIB - 1, 21))

    # ---- exact command construction -----------------------------------

    def test_build_command_matches_documented_recipe(self):
        command = cq.build_command(
            "/venv/bin/mlx_lm.convert",
            "Qwen/Qwen3.8-27B",
            "/models/Qwen3.8-27B-5bit-affine-g64",
            q_bits=5,
            q_group_size=64,
            q_mode="affine",
            q_dtype="bfloat16",
        )
        self.assertEqual(
            command,
            [
                "/venv/bin/mlx_lm.convert",
                "--hf-path", "Qwen/Qwen3.8-27B",
                "--mlx-path", "/models/Qwen3.8-27B-5bit-affine-g64",
                "-q",
                "--q-bits", "5",
                "--q-group-size", "64",
                "--q-mode", "affine",
                "--dtype", "bfloat16",
            ],
        )

    # ---- provenance shape and honesty ---------------------------------

    def test_provenance_never_claims_ud_equivalence(self):
        plan = cq.validate_source(str(self.make_source()), REV)
        command = cq.build_command(
            "/x/mlx_lm.convert", plan["source"], self.out_dir(),
            q_bits=5, q_group_size=64, q_mode="affine", q_dtype="bfloat16",
        )
        payload = cq.provenance_payload(
            plan,
            versions={"mlx_lm": "0.31.3", "mlx": "0.32.0"},
            converter="/x/mlx_lm.convert",
            output_path=self.out_dir(),
            command=command,
            converted_at_utc="2026-09-02T14:00:00Z",
            output_tree_digest="abc",
            output_bytes=123,
        )
        self.assertIn("claims", payload)
        self.assertIs(payload["claims"]["ud_equivalence"], False)
        self.assertIs(payload["claims"]["gguf_derived"], False)
        self.assertIn("NOT UD", payload["claims"]["label"])

    def test_provenance_distinguishes_all_identity_fields(self):
        plan = cq.validate_source("Qwen/Qwen3.8-27B", REV)
        command = cq.build_command(
            "/x/mlx_lm.convert", plan["source"], self.out_dir(),
            q_bits=5, q_group_size=64, q_mode="affine", q_dtype="bfloat16",
        )
        payload = cq.provenance_payload(
            plan,
            versions={"mlx_lm": "0.31.3", "mlx": "0.32.0"},
            converter="/x/mlx_lm.convert",
            output_path=self.out_dir(),
            command=command,
            converted_at_utc="2026-09-02T14:00:00Z",
            output_tree_digest="abc",
            output_bytes=123,
            source_tree_digest="def",
            source_bytes=SOURCE_BYTES,
            config_sha256="cafe",
        )
        # source revision, source tree/hash, quant recipe, converter version,
        # output path and command are DISTINCT top-level fields
        self.assertEqual(payload["source"]["repo"], "Qwen/Qwen3.8-27B")
        self.assertEqual(payload["source"]["revision"], REV)
        self.assertEqual(payload["source"]["tree_digest"], "def")
        self.assertEqual(payload["source"]["bytes"], SOURCE_BYTES)
        self.assertEqual(payload["source"]["config_sha256"], "cafe")
        self.assertEqual(payload["quant_recipe"], {"bits": 5, "group_size": 64, "mode": "affine", "dtype": "bfloat16"})
        self.assertEqual(payload["converter"]["version"], {"mlx_lm": "0.31.3", "mlx": "0.32.0"})
        self.assertEqual(payload["converter"]["path"], "/x/mlx_lm.convert")
        self.assertEqual(payload["output"]["path"], self.out_dir())
        self.assertEqual(payload["command"], command)
        self.assertEqual(payload["converted_at_utc"], "2026-09-02T14:00:00Z")
        self.assertEqual(payload["schema"], cq.PROVENANCE_SCHEMA)

    def test_provenance_write_is_idempotent_json_and_refuses_overwrite(self):
        path = self.root / "out.provenance.json"
        payload = {"a": 1}
        cq.write_provenance_json(payload, path)
        self.assertEqual(json.loads(path.read_text())["a"], 1)
        with self.assertRaises(cq.UsageError):
            cq.write_provenance_json({"a": 2}, path)
        self.assertEqual(json.loads(path.read_text())["a"], 1)  # untouched
        cq.write_provenance_json({"a": 2}, path, force=True)
        self.assertEqual(json.loads(path.read_text())["a"], 2)

    def test_provenance_path_is_sibling_of_output(self):
        self.assertEqual(
            cq.provenance_path(Path("/m/Qwen3.8-27B-5bit-affine-g64")),
            Path("/m/Qwen3.8-27B-5bit-affine-g64.provenance.json"),
        )

    # ---- converter version resolution ----------------------------------

    def make_fake_converter_with_venv(self, mlx_lm_version, mlx_version):
        """A venv whose bin/python prints canned versions, + a converter
        script shebang-bound to it. Models mlx_lm 0.31.3 in a venv wrapped
        by a different interpreter."""
        venv = self.root / "fake-venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "lib").mkdir()
        py = venv / "bin" / "python3.12"
        py.write_text(
            "#!/bin/sh\n"
            f'echo "{mlx_lm_version}"\n'
            f'echo "{mlx_version}"\n'
        )
        py.chmod(py.stat().st_mode | stat.S_IEXEC)
        converter = self.root / "fake-venv-convert"
        converter.write_text(f"#!{py}\n")
        converter.chmod(converter.stat().st_mode | stat.S_IEXEC)
        return str(converter)

    def test_converter_versions_uses_shebang_interpreter(self):
        converter = self.make_fake_converter_with_venv("9.9.1", "8.8.2")
        versions = cq.converter_versions(converter)
        self.assertEqual(versions, {"mlx_lm": "9.9.1", "mlx": "8.8.2"})

    def test_converter_versions_tolerates_missing_packages(self):
        versions = cq.converter_versions("/nonexistent/converter")
        self.assertEqual(versions, {"mlx_lm": "unknown", "mlx": "unknown"})

    # ---- dry-run / planning mode --------------------------------------

    def test_dry_run_plans_without_side_effects(self):
        src = self.make_source()
        out = self.out_dir()
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", out, "--converter", "/any/mlx_lm.convert",
            "--dry-run", "--min-free-gib", "0",
        ])
        self.assertEqual(rc, 0)
        self.assertFalse(Path(out).exists(), "dry-run must not create output")
        self.assertFalse(cq.provenance_path(Path(out)).exists())
        self.assertTrue((src / "config.json").exists(), "dry-run must not delete source")

    def test_dry_run_refuses_when_disk_requirement_cannot_fit(self):
        src = self.make_source()
        out = self.out_dir()
        original_free = cq.free_bytes
        # free_bytes is a plain module-level function; the mock must be a
        # plain callable (a staticmethod object is not callable at module scope
        # in Python 3.11: 'staticmethod' object is not callable).
        cq.free_bytes = lambda path: 1  # 1 byte free
        try:
            rc = cq.run([
                "--source", str(src), "--revision", REV,
                "--output", out, "--converter", "/any/mlx_lm.convert",
                "--dry-run", "--min-free-gib", "20",
            ])
        finally:
            cq.free_bytes = original_free
        self.assertEqual(rc, 3)
        self.assertFalse(Path(out).exists())
        self.assertFalse(cq.provenance_path(Path(out)).exists())
        self.assertTrue((src / "config.json").exists(), "refusal must not delete source")

    def test_local_plan_guard_counts_source_as_sunk_cost(self):
        # A local source already occupies its bytes on disk (sunk cost); the
        # guard must require output + floor, NOT source + output + floor.
        # (Real-world case: the 55.6 GB source is staged in the HF cache, so
        # free space can never again satisfy source+output+20GiB, even though
        # output+20GiB fits and the end-state floor holds.)
        src = self.make_source()
        out = self.out_dir()
        src_bytes = cq.dir_digest(src)[1]
        output_bytes = cq.estimate_output_bytes(src_bytes, 5)
        original_free = cq.free_bytes
        cq.free_bytes = lambda path: output_bytes + 20 * GIB + 1  # >= O+F, < S+O+F
        try:
            rc = cq.run([
                "--source", str(src), "--revision", REV,
                "--output", out, "--converter", "/any/mlx_lm.convert",
                "--dry-run", "--min-free-gib", "20",
            ])
        finally:
            cq.free_bytes = original_free
        self.assertEqual(rc, 0, "local plan with output+floor free must pass the guard")
        self.assertFalse(Path(out).exists())
        self.assertTrue((src / "config.json").exists(), "dry-run must not delete source")

    def test_local_plan_guard_still_refuses_below_output_plus_floor(self):
        src = self.make_source()
        out = self.out_dir()
        src_bytes = cq.dir_digest(src)[1]
        output_bytes = cq.estimate_output_bytes(src_bytes, 5)
        original_free = cq.free_bytes
        cq.free_bytes = lambda path: output_bytes + 20 * GIB - 1
        try:
            rc = cq.run([
                "--source", str(src), "--revision", REV,
                "--output", out, "--converter", "/any/mlx_lm.convert",
                "--dry-run", "--min-free-gib", "20",
            ])
        finally:
            cq.free_bytes = original_free
        self.assertEqual(rc, 3, "local plan below output+floor must still refuse")
        self.assertFalse(Path(out).exists())

    def test_hub_plan_guard_still_counts_source(self):
        # Regression: a hub plan (source downloaded inside the run) must keep
        # the strict source+output+floor requirement even when output+floor fits.
        src_bytes = SOURCE_BYTES
        output_bytes = cq.estimate_output_bytes(src_bytes, 5)
        original_free = cq.free_bytes
        cq.free_bytes = lambda path: output_bytes + 20 * GIB + 1  # plenty for local, NOT hub
        try:
            rc = cq.run([
                "--source", "Qwen/Qwen3.8-27B", "--revision", REV,
                "--output", self.out_dir(), "--converter", "/any/mlx_lm.convert",
                "--dry-run", "--source-bytes", str(src_bytes),
                "--expect-output-bytes", str(output_bytes),
                "--min-free-gib", "20",
            ])
        finally:
            cq.free_bytes = original_free
        self.assertEqual(rc, 3, "hub plan must still require source+output+floor")

    def test_hub_plan_requires_byte_estimates_for_guard(self):
        # Mock free space so the test is deterministic and independent of the
        # machine's current free bytes (the hub requirement is ~73+ GB).
        original_free = cq.free_bytes
        cq.free_bytes = lambda path: 10**12  # 1 TB free
        try:
            rc = cq.run([
                "--source", "Qwen/Qwen3.8-27B", "--revision", REV,
                "--output", self.out_dir(), "--converter", "/any/mlx_lm.convert",
                "--dry-run",
            ])
            self.assertEqual(rc, 3)  # cannot prove the disk requirement
            rc = cq.run([
                "--source", "Qwen/Qwen3.8-27B", "--revision", REV,
                "--output", self.out_dir(), "--converter", "/any/mlx_lm.convert",
                "--dry-run", "--source-bytes", str(SOURCE_BYTES),
                "--expect-output-bytes", str(cq.estimate_output_bytes(SOURCE_BYTES, 5)),
                "--min-free-gib", "0",
            ])
            self.assertEqual(rc, 0)
            rc = cq.run([
                "--source", "Qwen/Qwen3.8-27B", "--revision", REV,
                "--output", self.out_dir(), "--converter", "/any/mlx_lm.convert",
                "--dry-run", "--source-bytes", str(SOURCE_BYTES),
                "--expect-output-bytes", str(cq.estimate_output_bytes(SOURCE_BYTES, 5)),
                "--min-free-gib", "20",
            ])
            self.assertEqual(rc, 0, "hub plan with source+output+floor free must pass")
        finally:
            cq.free_bytes = original_free

    # ---- real (fake-converter) run ------------------------------------

    def test_run_converts_then_writes_provenance(self):
        src = self.make_source()
        out = self.out_dir()
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", out, "--converter", self.make_fake_converter(),
            "--min-free-gib", "0",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue((Path(out) / "config.json").exists())
        self.assertTrue((Path(out) / "model-00001-of-00001.safetensors").exists())
        prov = json.loads(cq.provenance_path(Path(out)).read_text())
        self.assertEqual(prov["source"]["repo"], str(Path(src).resolve()))
        self.assertEqual(prov["source"]["revision"], REV)
        self.assertEqual(prov["source"]["kind"], "local")
        self.assertIs(prov["claims"]["ud_equivalence"], False)
        self.assertEqual(prov["quant_recipe"]["bits"], 5)
        self.assertTrue(prov["quant_recipe"]["group_size"] == 64)
        self.assertTrue(prov["quant_recipe"]["mode"] == "affine")
        self.assertTrue(prov["quant_recipe"]["dtype"] == "bfloat16")
        self.assertIn("version", prov["converter"])
        self.assertIn("mlx_lm", prov["converter"]["version"])
        self.assertIn("mlx", prov["converter"]["version"])
        self.assertEqual(prov["output"]["path"], str(Path(out).resolve()))
        self.assertIsInstance(prov["command"], list)
        self.assertIn("--q-bits", prov["command"])

    def test_failed_conversion_writes_no_provenance(self):
        src = self.make_source()
        out = self.out_dir()
        env = dict(os.environ, FAKE_CONVERT_FAIL="1")
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", out, "--converter", self.make_fake_converter(),
            "--min-free-gib", "0",
        ], env=env)
        self.assertNotEqual(rc, 0)
        self.assertFalse(Path(out).exists())
        self.assertFalse(cq.provenance_path(Path(out)).exists())

    def test_refuses_non_empty_existing_output(self):
        src = self.make_source()
        out = Path(self.out_dir())
        out.mkdir()
        (out / "keep.txt").write_text("keep")
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", str(out), "--converter", self.make_fake_converter(),
            "--min-free-gib", "0",
        ])
        self.assertEqual(rc, 2)
        self.assertTrue((out / "keep.txt").exists())

    def test_refuses_existing_provenance_without_force(self):
        src = self.make_source()
        out = self.out_dir()
        prov = cq.provenance_path(Path(out))
        prov.parent.mkdir(parents=True, exist_ok=True)
        prov.write_text('{"existing": true}')
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", out, "--converter", self.make_fake_converter(),
            "--min-free-gib", "0",
        ])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(prov.read_text())["existing"], True)
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", out, "--converter", self.make_fake_converter(),
            "--min-free-gib", "0", "--force",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(Path(out).exists())

    # ---- source cache deletion is explicit-only -----------------------

    def test_source_never_auto_deleted(self):
        src = self.make_source()
        out = self.out_dir()
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", out, "--converter", self.make_fake_converter(),
            "--min-free-gib", "0",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(src.exists(), "source must NOT be auto-deleted")
        prov = json.loads(cq.provenance_path(Path(out)).read_text())
        self.assertIs(prov["post_conversion"]["source_cache_deleted"], False)

    def test_delete_source_cache_requires_explicit_flag(self):
        src = self.make_source(name="src-delete")
        out = self.out_dir("out-delete")
        rc = cq.run([
            "--source", str(src), "--revision", REV,
            "--output", out, "--converter", self.make_fake_converter(),
            "--min-free-gib", "0", "--delete-source-cache",
        ])
        self.assertEqual(rc, 0)
        self.assertFalse(src.exists(), "explicit --delete-source-cache removes source")
        prov = json.loads(cq.provenance_path(Path(out)).read_text())
        self.assertIs(prov["post_conversion"]["source_cache_deleted"], True)


if __name__ == "__main__":
    unittest.main()