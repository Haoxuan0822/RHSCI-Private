"""双数据集复用episode编号，验证缓存与视频解码逐像素/逐标签一致。"""

import importlib
import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow.parquet as pq
import torch

import test_pi05_gr00t_multitask as multitask
from rhsci.data.ardy_sidecar import dataset_content_fingerprint
from rhsci.data.cache_config import InputCacheConfig
from rhsci.data.input_cache import cache_identity
from rhsci.data.multitask import build_training_dataset, training_cache_identity
from rhsci.data.raw_input_cache import SCHEMA
from rhsci.runtime.artifacts import sha256


def builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts/data/build_raw_input_cache.py"
    spec = importlib.util.spec_from_file_location("raw_cache_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_fixture_cache(root, cache_root):
    """使用正式视频构建/独立解码核验；帧表来自同一批真实Parquet。"""
    directory = cache_root / root.name
    directory.mkdir(parents=True)
    episodes = [
        json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()
    ]
    rows = [
        pq.read_table(root / f"data/episode_{ep['episode_index']:06d}.parquet").to_pydict()
        for ep in episodes
    ]
    columns = {
        key: torch.as_tensor(np.concatenate([np.asarray(row[key]) for row in rows]))
        for key in rows[0]
    }
    for key in ("action", "states"):
        columns[key] = columns[key].float()
    torch.save(columns, directory / "table.pt")
    manifest = dict(
        schema=SCHEMA,
        dataset=root.name,
        fingerprint=dataset_content_fingerprint(root),
        rows=sum(ep["length"] for ep in episodes),
        columns=list(columns),
        table="table.pt",
        episodes={},
        files={"table.pt": dict(
            size=(directory / "table.pt").stat().st_size,
            sha256=sha256(directory / "table.pt"),
        )},
    )
    builder = builder_module()
    for episode in episodes:
        index = episode["episode_index"]
        key, record, files = builder.build_video(
            (root, directory, index, episode["length"], f"videos/episode_{index:06d}.mp4")
        )
        manifest["episodes"][key] = record
        manifest["files"].update(files)
    (directory / "manifest.json").write_text(json.dumps(manifest))


class MultitaskCacheTests(unittest.TestCase):
    video_size = (640, 360)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        multitask.MultitaskTests.setUpClass.__func__(cls)
        cls.cache_root = cls.root / "cache"
        for root in cls.roots:
            build_fixture_cache(root, cls.cache_root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def paths(self, config):
        return dict(
            dataset=self.roots[0], sidecar=self.labels[0], datasets=self.roots,
            sidecars=self.labels, input_cache=config,
        )

    def assert_sample_equal(self, cached, decoded, *, gr):
        if gr:
            np.testing.assert_array_equal(
                np.array(cached.images["rs_view"]), np.array(decoded.images["rs_view"])
            )
            self.assertEqual(cached.text, decoded.text)
            for kind in ("states", "actions", "metadata"):
                left, right = getattr(cached, kind), getattr(decoded, kind)
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
        else:
            self.assertEqual(cached.keys(), decoded.keys())
            for key in cached:
                np.testing.assert_array_equal(cached[key], decoded[key])

    def test_every_backbone_routes_cache_pixels_tables_and_labels_to_local_dataset(self):
        for backbone in ("openpi", "gr00t"):
            dataset_type = importlib.import_module(f"{backbone}.rhsci.data").SimpleRHSCIDataset
            for variant in ("STATE", "VGQ-AE-240", "VGQ-AE-Mo-480"):
                with self.subTest(backbone=backbone, variant=variant):
                    options = dict(pad_actions=True) if backbone == "openpi" else {}
                    cached = build_training_dataset(
                        self.paths(InputCacheConfig(self.cache_root)),
                        dataset_type, variant, **options,
                    )
                    decoded = build_training_dataset(
                        self.paths(InputCacheConfig()), dataset_type, variant, **options,
                    )
                    gr = backbone == "gr00t"
                    start = 0
                    for child in cached.datasets:
                        episode_size = int(child.ends[0])
                        for local in (
                            0, 12, 23, 24, episode_size - 1, episode_size, len(child) - 1,
                        ):
                            index = start + local
                            self.assert_sample_equal(
                                cached.raw_sample(index), decoded.raw_sample(index), gr=gr,
                            )
                        start += len(child)
                    # spawn会pickle整个子库，不能把mmap内容或前一个任务的解码器串过去。
                    restored = pickle.loads(pickle.dumps(cached))
                    boundary = len(cached.datasets[0])
                    self.assert_sample_equal(
                        restored.raw_sample(boundary + 24),
                        decoded.raw_sample(boundary + 24), gr=gr,
                    )
                    child = cached.datasets[0]
                    sample = child.raw_sample(24)
                    pixels = np.array(sample.images["rs_view"]) if gr else sample["images"]
                    pixels.fill(255)
                    self.assert_sample_equal(child.raw_sample(24), decoded.raw_sample(24), gr=gr)

    def test_contract_checks_all_sources_and_preserves_single_cache_schema(self):
        config = InputCacheConfig(self.cache_root)
        paths = self.paths(config)
        identity = training_cache_identity(paths)
        self.assertEqual(
            identity["datasets"], [cache_identity(root, config=config) for root in self.roots]
        )
        single = dict(dataset=self.roots[0], sidecar=self.labels[0], input_cache=config)
        self.assertEqual(training_cache_identity(single), identity["datasets"][0])
        self.assertIsNone(training_cache_identity(self.paths(InputCacheConfig())))
        # 第一个库不变，第二个库坏掉也必须在启动身份核验中被发现。
        payload = self.cache_root / self.roots[1].name / "episode-000000.npy"
        with payload.open("r+b") as stream:
            stream.seek(-1, 2)
            original = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([original[0] ^ 1]))
        try:
            with self.assertRaisesRegex(ValueError, "Corrupt input cache"):
                training_cache_identity(paths)
        finally:
            with payload.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(original)

    def test_missing_second_cache_and_colliding_names_fail_without_decoding_fallback(self):
        paths = self.paths(InputCacheConfig(self.cache_root))
        paths["datasets"] = [self.roots[0], self.root / "other" / self.roots[0].name]
        with self.assertRaisesRegex(ValueError, "dataset-name collision"):
            build_training_dataset(paths, object, "STATE")
        with self.assertRaisesRegex(ValueError, "dataset-name collision"):
            training_cache_identity(paths)
        manifest = self.cache_root / self.roots[1].name / "manifest.json"
        temporary = manifest.with_suffix(".temporarily-missing")
        manifest.rename(temporary)
        try:
            with self.assertRaises(FileNotFoundError):
                training_cache_identity(self.paths(InputCacheConfig(self.cache_root)))
        finally:
            temporary.rename(manifest)

    def test_backbone_contracts_save_all_cache_identities(self):
        for backbone in ("openpi", "gr00t"):
            if backbone == "openpi" and importlib.util.find_spec("numpydantic") is None:
                continue
            args, paths = multitask.MultitaskTests.options(self, backbone)
            paths["input_cache"] = InputCacheConfig(self.cache_root)
            module = importlib.import_module(f"{backbone}.rhsci.contract")
            hash_target = (
                f"{backbone}.rhsci.{'checkpoints' if backbone == 'gr00t' else 'contract'}.sha256"
            )
            with patch(hash_target, return_value="initialization-or-tokenizer"):
                identity, _ = module.build_training_contract(args, paths)
            self.assertEqual(identity["input_cache"], training_cache_identity(paths))
            self.assertEqual(len(identity["input_cache"]["datasets"]), 2)

    def test_batch_builder_routes_each_root_and_verifies_existing_cache(self):
        builder = builder_module()
        command = [
            "build", "--datasets", *map(str, self.roots),
            "--cache-root", str(self.cache_root),
        ]
        with patch.object(sys, "argv", command), patch.object(builder, "build_cache") as build:
            builder.main()
            self.assertEqual([call.args[0] for call in build.call_args_list], self.roots)
        with patch.object(
            builder, "load_raw_lerobot_dataset",
            side_effect=AssertionError("existing cache rebuilt"),
        ):
            builder.build_cache(self.roots[0], self.cache_root, 1)
        bad = self.cache_root / self.roots[0].name / "manifest.json"
        text = bad.read_text()
        value = json.loads(text)
        value["fingerprint"] = "wrong-dataset"
        bad.write_text(json.dumps(value))
        try:
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                builder.build_cache(self.roots[0], self.cache_root, 1)
        finally:
            bad.write_text(text)

    def test_builder_validates_before_publication_and_rejects_source_changes(self):
        builder = builder_module()
        reference = self.cache_root / self.roots[0].name
        manifest = json.loads((reference / "manifest.json").read_text())
        columns = torch.load(reference / "table.pt", weights_only=True)
        for scenario in ("stable", "source-changed", "corrupt-payload"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                root = directory / "source" / self.roots[0].name
                shutil.copytree(self.roots[0], root)
                cache_root = directory / "cache"
                cache_root.mkdir()
                hf = MagicMock()
                hf.column_names = list(columns)
                hf.select_columns.return_value = hf
                hf.__getitem__.return_value = {k: list(v.unbind()) for k, v in columns.items()}
                dataset = MagicMock()
                dataset.hf_dataset = hf
                dataset.__len__.return_value = manifest["rows"]
                dataset.meta = SimpleNamespace(
                    get_video_file_path=lambda ep, key: f"videos/episode_{ep:06d}.mp4"
                )

                def load(source):
                    if scenario == "source-changed":
                        # Reproduce a table snapshot becoming stale during the native read.
                        meta = source / "meta/info.json"
                        meta.write_text(meta.read_text() + "\n")
                    return dataset

                def videos(function, jobs):
                    # Video decoding is covered above; reuse those verified immutable pixels.
                    for _, pending, ep, _, _ in jobs:
                        record = manifest["episodes"][str(ep)]
                        files = {}
                        for name in (record["pixels"], record["pts"]):
                            (Path(pending) / name).hardlink_to(reference / name)
                            files[name] = manifest["files"][name]
                        yield str(ep), record, files

                verify = builder.cache_identity

                def validate(source, fingerprint, *, config):
                    if scenario == "corrupt-payload":
                        table = config.root / source.name / "table.pt"
                        with table.open("r+b") as stream:
                            stream.seek(-1, 2)
                            original = stream.read(1)
                            stream.seek(-1, 2)
                            stream.write(bytes([original[0] ^ 1]))
                    return verify(source, fingerprint, config=config)

                pool = MagicMock()
                pool.__enter__.return_value.map.side_effect = videos
                with patch.object(builder, "load_raw_lerobot_dataset", side_effect=load), patch.object(
                    builder.concurrent.futures, "ProcessPoolExecutor", return_value=pool
                ), patch.object(builder, "cache_identity", side_effect=validate):
                    if scenario == "stable":
                        builder.build_cache(root, cache_root, 1)
                        identity = verify(root, dataset_content_fingerprint(root),
                                          config=InputCacheConfig(cache_root))
                        receipt = json.loads((cache_root / f"{root.name}-verified.json").read_text())
                        self.assertEqual(receipt["identity"], identity)
                    else:
                        message = "Source dataset changed" if scenario == "source-changed" else "Corrupt"
                        with self.assertRaisesRegex(ValueError, message):
                            builder.build_cache(root, cache_root, 1)
                        self.assertFalse((cache_root / root.name).exists())
                        self.assertFalse((cache_root / f"{root.name}-verified.json").exists())
