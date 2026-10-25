import copy, importlib.util, pathlib, unittest

spec = importlib.util.spec_from_file_location(
    "adapter", pathlib.Path(__file__).parents[1] / "prepare_flint_jangh.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ContractTests(unittest.TestCase):
    def config(self):
        q = {"bits": 8, "group_size": 64, "mode": "affine"}
        for role in ("gate_proj", "up_proj", "down_proj"):
            q["model.layers.2.mlp.switch_mlp." + role] = {
                "bits": 3 if role == "down_proj" else 2,
                "mode": "jangtq2",
                "rotation": "hadamard32",
            }
        books = {
            str(b): {
                "alpha": 0.25,
                "beta": 0.0,
                "levels": [(i - ((1 << b) - 1) / 2) * 0.25 for i in range(1 << b)],
            }
            for b in (2, 3, 4)
        }
        return {
            "model_type": "flint",
            "num_hidden_layers": 3,
            "mlp_only_layers": [0, 1],
            "hidden_size": 64,
            "moe_intermediate_size": 32,
            "num_experts": 8,
            "quantization": q,
            "quantization_config": copy.deepcopy(q),
            "jangtq": {
                "version": 2,
                "packing": "lsb-bitstream",
                "scale_dtype": "float16",
                "codebook_family": "odd-cubic",
                "rotation": "hadamard32",
                "codebooks": books,
            },
        }

    def test_exact_partition_retains_mixed_widths(self):
        c = self.config()
        ordinary = adapter.partition(c)
        self.assertNotIn("jangtq", ordinary)
        for role in ("gate_proj", "up_proj", "down_proj"):
            self.assertIs(
                ordinary["quantization"]["model.layers.2.mlp.switch_mlp." + role], False
            )
        self.assertEqual(
            c["quantization"]["model.layers.2.mlp.switch_mlp.down_proj"]["bits"], 3
        )

    def test_bad_packing_or_codebook_is_rejected(self):
        for field, value in [
            ("packing", "msb"),
            ("scale_dtype", "bfloat16"),
            ("version", 1),
        ]:
            c = self.config()
            c["jangtq"][field] = value
            with self.assertRaises(ValueError):
                adapter.partition(c)
        c = self.config()
        c["jangtq"]["codebooks"]["2"]["levels"][0] += 1
        with self.assertRaises(ValueError):
            adapter.partition(c)

    def test_conflicting_maps_or_missing_role_is_rejected(self):
        c = self.config()
        c["quantization_config"]["bits"] = 4
        with self.assertRaises(ValueError):
            adapter.partition(c)
        c = self.config()
        del c["quantization"]["model.layers.2.mlp.switch_mlp.up_proj"]
        c["quantization_config"] = copy.deepcopy(c["quantization"])
        with self.assertRaises(ValueError):
            adapter.partition(c)

    def test_unknown_width_rotation_or_module_is_rejected(self):
        for field, value in [("bits", 5), ("rotation", "none"), ("mode", "affine")]:
            c = self.config()
            c["quantization"]["model.layers.2.mlp.switch_mlp.gate_proj"][field] = value
            c["quantization_config"] = copy.deepcopy(c["quantization"])
            with self.assertRaises(ValueError):
                adapter.partition(c)

    def test_exact_real_headers_and_mutations(self):
        fixture = pathlib.Path(__file__).parents[2] / "Tests/MeiTests/Fixtures/JANGH2"
        import json

        c = json.loads((fixture / "config.json").read_text())
        t = {}
        for p in fixture.glob("*.header.json"):
            t.update(
                {
                    k: v
                    for k, v in json.loads(p.read_text())["tensors"].items()
                    if k != "__metadata__"
                }
            )
        adapter.partition(c)
        adapter.validate_headers(c, t)
        for mutation in ("missing", "extra", "dtype", "shape"):
            bad = copy.deepcopy(t)
            key = "model.layers.2.mlp.switch_mlp.gate_proj.tq2_packed"
            if mutation == "missing":
                del bad[key]
            if mutation == "extra":
                bad["unknown.weight"] = copy.deepcopy(bad[key])
            if mutation == "dtype":
                bad[key]["dtype"] = "BF16"
            if mutation == "shape":
                bad[key]["shape"][-1] += 1
            with self.assertRaises(ValueError):
                adapter.validate_headers(c, bad)


if __name__ == "__main__":
    unittest.main()
