#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mei_disk_guard import GuardError, cleanup_all, cleanup_cache, free_space_ok


class MeiDiskGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "runtime"
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def make_cache(self, name):
        path = self.root / name
        path.mkdir()
        (path / "record.safetensors").write_bytes(b"cache")
        return path

    def test_free_space_gate(self):
        self.assertTrue(free_space_ok(self.root, 20, available=20 * 1024**3))
        self.assertFalse(free_space_ok(self.root, 20, available=20 * 1024**3 - 1))

    def test_path_scoping_rejects_outside_and_protected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        with self.assertRaises(GuardError):
            cleanup_cache(self.root, outside)
        protected = self.root / "kv-cache-35b-lc80k"
        protected.mkdir()
        with self.assertRaises(GuardError):
            cleanup_cache(self.root, protected)
        self.assertTrue(protected.exists())

    def test_cleanup_removes_disposable_cache(self):
        cache = self.make_cache("kv-cache-cell-test")
        self.assertTrue(cleanup_cache(self.root, cache, active=False))
        self.assertFalse(cache.exists())
        self.assertIn("status=removed", (self.root / "mei-disk-guard.log").read_text())

    def test_retain_preserves_cache(self):
        cache = self.make_cache("kv-cache-exp-retain")
        self.assertFalse(cleanup_cache(self.root, cache, retain=True, active=False))
        self.assertTrue(cache.exists())
        self.assertIn("status=retained", (self.root / "mei-disk-guard.log").read_text())

    def test_active_cache_is_not_removed(self):
        cache = self.make_cache("kv-cache-cell-active")
        self.assertFalse(cleanup_cache(self.root, cache, active=True))
        self.assertTrue(cache.exists())

    def test_cleanup_all_is_scoped_and_keeps_protected(self):
        self.make_cache("kv-cache-sweep")
        self.make_cache("kv-cache-cell-one")
        protected = self.root / "kv-cache-qwen38"
        protected.mkdir()
        (protected / "record").write_text("keep")
        self.assertEqual(cleanup_all(self.root, active=False), 0)
        self.assertFalse((self.root / "kv-cache-sweep").exists())
        self.assertFalse((self.root / "kv-cache-cell-one").exists())
        self.assertTrue(protected.exists())


if __name__ == "__main__":
    unittest.main()
