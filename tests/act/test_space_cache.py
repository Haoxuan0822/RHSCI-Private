import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from act.rsci.checkpoints import sha256
from rsci.data.cache_config import InputCacheConfig
from rsci.data.raw_input_cache import (
    SCHEMA,
    RawInputCache,
    cache_identity,
)


class CacheTests(unittest.TestCase):
    def test_shared_frame_cache_copies_before_augmentation(self):
        from rsci.data.input_cache import InputCache

        cache = InputCache(self.cache, self.dataset)
        expected = self.pixels[[1, 0, 1]].transpose(0, 2, 3, 1).copy()
        actual = cache.frames(0, [1, 0, 1])
        np.testing.assert_array_equal(actual, expected)
        actual.fill(255)
        np.testing.assert_array_equal(cache.frames(0, [1, 0, 1]), expected)
        self.assertEqual(cache.arrays["0"].mode, "r")

    def test_shared_cache_authenticates_guardian_bytes(self):
        from rsci.data.input_cache import cache_identity as shared_identity

        with patch.dict(os.environ, {"RSCI_INPUT_CACHE": str(self.cache), "WORLD_SIZE": "1"}):
            with patch("rsci.data.ardy_sidecar.dataset_content_fingerprint", return_value="fixed"):
                shared_identity(self.dataset, config=InputCacheConfig(self.cache))
                with (self.directory / "rgb.npy").open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"\xff")
                with self.assertRaises(ValueError):
                    shared_identity(self.dataset, config=InputCacheConfig(self.cache))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dataset = root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "meta/episodes.jsonl").write_text(
            json.dumps(dict(episode_index=0, length=2)) + "\n"
        )
        self.source = self.dataset / "video.mp4"
        self.source.write_bytes(b"original")
        self.cache = root / "cache"
        self.directory = self.cache / "dataset"
        self.directory.mkdir(parents=True)
        self.pixels = np.stack([np.full((3, 360, 640), x, dtype=np.uint8) for x in [0, 40]])
        np.save(self.directory / "rgb.npy", self.pixels)
        np.save(self.directory / "pts.npy", np.array([0, 0.02], np.float64))
        torch.save(
            {"states": torch.arange(8).reshape(2, 4), "timestamp": torch.tensor([0.0, 0.02])},
            self.directory / "table.pt",
        )
        st = self.source.stat()
        self.manifest = dict(
            schema=SCHEMA,
            dataset="dataset",
            fingerprint="fixed",
            rows=2,
            columns=["states", "timestamp"],
            table="table.pt",
            episodes={
                "0": dict(
                    frames=2,
                    shape=[2, 3, 360, 640],
                    fps=50,
                    source="video.mp4",
                    source_stat=[st.st_size, st.st_mtime_ns],
                    source_sha256=sha256(self.source),
                    pixels="rgb.npy",
                    pts="pts.npy",
                )
            },
            files={
                n: dict(size=(self.directory / n).stat().st_size, sha256=sha256(self.directory / n))
                for n in ["rgb.npy", "pts.npy", "table.pt"]
            },
        )
        self.write_manifest()
        self.env = patch.dict(os.environ, {"RSCI_ACT_SPACE_CACHE": str(self.cache)})
        self.env.start()

    def write_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_duplicate_reorder_rng_mutation_and_pickle(self):
        cache = RawInputCache(self.cache, self.dataset)
        rng = torch.get_rng_state().clone()
        expected = torch.from_numpy(self.pixels[[1, 0, 1]].copy()).float() / 255
        out = cache.decode(self.source, [0.02, 0, 0.02], 0.001)
        self.assertTrue(torch.equal(out, expected))
        out.zero_()
        cache = pickle.loads(pickle.dumps(cache))
        self.assertEqual(cache.arrays, {})
        self.assertTrue(torch.equal(cache.decode(self.source, [0.02, 0, 0.02], 0.001), expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(
            sha256(self.directory / "rgb.npy"), self.manifest["files"]["rgb.npy"]["sha256"]
        )

    def test_table_selection_and_row_copy(self):
        cache = RawInputCache(self.cache, self.dataset)
        table = cache.table
        table[0]["states"].zero_()
        self.assertTrue(torch.equal(table[0]["states"], torch.arange(4)))
        selected = table.select([1, 0, 1]).select([2, 1])
        self.assertTrue(
            torch.equal(torch.stack(selected["states"]), torch.tensor([[4, 5, 6, 7], [0, 1, 2, 3]]))
        )
        self.assertEqual(len(selected), 2)

    def test_full_verification_and_fingerprint(self):
        cache_identity(self.dataset, "fixed", config=InputCacheConfig(self.cache))
        with self.assertRaises(ValueError):
            cache_identity(self.dataset, "other", config=InputCacheConfig(self.cache))

    def test_same_size_cache_corruption_rejected(self):
        path = self.directory / "rgb.npy"
        with path.open("r+b") as f:
            f.seek(-1, 2)
            f.write(b"\xff")
        with self.assertRaises(ValueError):
            cache_identity(self.dataset, "fixed", config=InputCacheConfig(self.cache))

    def test_same_size_source_corruption_rejected(self):
        cache_identity(self.dataset, "fixed", config=InputCacheConfig(self.cache))
        self.source.write_bytes(b"modified")
        with self.assertRaises(ValueError):
            cache_identity(self.dataset, "fixed", config=InputCacheConfig(self.cache))

    def test_missing_file_and_incomplete_inventory(self):
        (self.directory / "pts.npy").unlink()
        with self.assertRaises(OSError):
            RawInputCache(self.cache, self.dataset)
        self.manifest["episodes"] = {}
        self.write_manifest()
        with self.assertRaises(ValueError):
            RawInputCache(self.cache, self.dataset)

    def test_source_replacement_and_timestamp_tolerance(self):
        cache = RawInputCache(self.cache, self.dataset)
        with self.assertRaises(AssertionError):
            cache.decode(self.source, [0.009], 0.001)
        self.source.write_bytes(b"changed video")
        with self.assertRaises(ValueError):
            cache.decode(self.source, [0], 0.001)

    def test_cache_path_escape_rejected(self):
        self.manifest["files"]["../escape"] = {"size": 1}
        self.write_manifest()
        with self.assertRaises(ValueError):
            RawInputCache(self.cache, self.dataset)

    def test_disabled_config_ignores_legacy_environment(self):
        from rsci.data.input_cache import cache_identity as shared_identity

        with patch.dict(os.environ, {"RSCI_INPUT_CACHE": "/nonexistent", "WORLD_SIZE": "8"}):
            self.assertIsNone(shared_identity(self.dataset))
            self.assertIsNone(cache_identity(self.dataset, "fixed"))

    def test_verification_role_independent_of_world_size(self):
        from rsci.data.input_cache import cache_identity as shared_identity

        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            with patch("rsci.data.ardy_sidecar.dataset_content_fingerprint", return_value="fixed"):
                full = shared_identity(self.dataset, config=InputCacheConfig(self.cache))
                metadata = shared_identity(
                    self.dataset, config=InputCacheConfig(self.cache, "metadata")
                )
                self.assertEqual(full, metadata)
                with (self.directory / "rgb.npy").open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"\xff")
                shared_identity(self.dataset, config=InputCacheConfig(self.cache, "metadata"))
                with self.assertRaises(ValueError):
                    shared_identity(self.dataset, config=InputCacheConfig(self.cache))
        # worker仍拒绝缺失/不完整文件，不能把metadata误解成关闭检查。
        (self.directory / "pts.npy").unlink()
        with self.assertRaises(OSError):
            cache_identity(self.dataset, "fixed", config=InputCacheConfig(self.cache, "metadata"))


if __name__ == "__main__":
    unittest.main()
