"""官方多数据集/采样路径及RHSCI按数据集隔离ARDY标签的CPU回归。"""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from psi.config.transform import SimpleRepackTransform
from psi.data.lerobot.compat import LeRobotDataset, MultiLeRobotDataset
from psi.data.lerobot.lerobot_ext import LeRobotDatasetWrapper
from psi.rhsci.training_config import build_argument_parser, resolve_training_plan
from rhsci.data.ardy_sidecar import ArdySidecarWriter, dataset_content_fingerprint


def make_dataset(root, name, base, tasks):
    """使用真实LeRobot落盘；不同数据集故意复用episode/task编号。"""
    dataset = LeRobotDataset.create(
        repo_id=name, root=root / name, fps=50, use_videos=False,
        features={
            "states": {"dtype": "float32", "shape": (32,), "names": None},
            "action": {"dtype": "float32", "shape": (36,), "names": None},
            "observation.depth.egocentric": {"dtype": "float32", "shape": (1,), "names": None},
        },
    )
    for task in tasks:
        for frame in range(20):
            dataset.add_frame(dict(
                states=np.full(32, base + frame, np.float32),
                action=np.full(36, base + frame, np.float32),
                **{"observation.depth.egocentric": np.zeros(1, np.float32)},
            ), task=task)
        dataset.save_episode()
    return dataset


def make_sidecars(data, root, base):
    context = dict(
        dataset=data.repo_id, processed_content_fingerprint=dataset_content_fingerprint(data.root),
        teacher_verification_status="formal_verified", teacher_pipeline="synthetic-unit-test",
    )
    writer = ArdySidecarWriter(root / data.repo_id, context)
    for episode in range(data.num_episodes):
        valid = np.arange(20) >= 6
        writer.write_episode(
            episode, ardy_body_latent=np.broadcast_to(
                (base + episode * 100 + np.arange(20, dtype=np.float32) ** 2)[:, None], (20, 128),
            ).copy(),
            teacher_valid=np.ones(20, bool), visual_history_valid=valid,
            ardy_loss_mask=valid, raw_episode_index=np.int64(episode),
            raw_frame_index=np.arange(20, dtype=np.int64) + 1,
        )


class MultitaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="rhsci-joint-")
        cls.root = Path(cls.directory.name)
        cls.labels = cls.root / "labels"
        for name, base, tasks in (("a", 0, ["open trash", "turn faucet"]), ("b", 1000, ["open trash faucet", "move"])):
            dataset = make_dataset(cls.root, name, base, tasks)
            make_sidecars(dataset, cls.labels, base * 10)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def config(self, **options):
        args = build_argument_parser().parse_args([
            "--task", "joint", "--model", "VGQ-AE-Mo-240", "--data-root", str(self.root),
            "--sidecar-root", str(self.labels), "--train-repo-ids", "a", "b", "--dry-run",
        ])
        cfg = resolve_training_plan(args).config.data
        return cfg.model_copy(update=options)

    def wrapper(self, cfg, split="train"):
        # 不请求图像解码；动作/状态的时间查询和MultiLeRobotDataset均为真实实现。
        with patch.object(SimpleRepackTransform, "delta_timestamps", return_value={"action": [0, .02], "states": [0]}):
            return LeRobotDatasetWrapper(cfg, split)

    def test_native_multi_dataset_labels_actions_and_episode_boundaries(self):
        wrapper = self.wrapper(self.config())
        self.assertIsInstance(wrapper.base_dataset, MultiLeRobotDataset)
        self.assertEqual(len(wrapper), 80)
        self.assertIsNone(wrapper.ardy_reader)
        for index in (0, 8, 12, 19, 20, 32, 39, 40, 48, 52, 59, 60, 72, 79):
            sample = wrapper[index]
            dataset, local = divmod(index, 40)
            episode, frame = divmod(local, 20)
            self.assertEqual(int(sample["episode_index"]), episode)
            self.assertEqual(int(sample["dataset_index"]), dataset)
            self.assertEqual(float(sample["action"][0, 0]), dataset * 1000 + frame)
            self.assertEqual(float(sample["ardy_target"][0]), dataset * 10000 + episode * 100 + frame ** 2)
            self.assertEqual(bool(sample["motion_valid"]), frame >= 12)
            if frame >= 12:
                self.assertEqual(float(sample["motion_target"][0]), frame ** 2 - (frame - 6) ** 2)
            self.assertNotIn("observation.depth.egocentric", sample)
        self.assertEqual(wrapper[40]["task"], "open trash faucet")

    def test_official_max_matching_weight_and_single_validation_repo(self):
        cfg = self.config(val_repo_ids=["b"])
        wrapper = self.wrapper(cfg)
        weights = wrapper.frame_task_weights({"trash": 4, "faucet": 2, "move": .5})
        torch.testing.assert_close(weights, torch.tensor([4.] * 20 + [2.] * 20 + [4.] * 20 + [.5] * 20, dtype=torch.double))
        val = self.wrapper(cfg, "val")
        self.assertNotIsInstance(val.base_dataset, MultiLeRobotDataset)
        self.assertEqual(float(val[12]["ardy_target"][0]), 10144)

    def test_all_sidecars_are_validated_and_mismatches_rejected(self):
        cfg = self.config(rsci_ardy_sidecars_by_repo=False)
        with self.assertRaisesRegex(ValueError, "sidecars_by_repo"):
            self.wrapper(cfg)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "a").mkdir()
            (Path(folder) / "b").mkdir()
            for name in ("a", "b"):
                manifest = json.loads((self.labels / name / "manifest.json").read_text())
                manifest["context"]["dataset"] = "wrong"
                (Path(folder) / name / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.wrapper(self.config(rsci_ardy_sidecar_root=folder))

    def test_official_split_augmentation_and_training_only_weights(self):
        cfg = self.config(task_sample_weights=["trash=4"])
        raw = self.wrapper(cfg)
        with patch("psi.data.lerobot.LeRobotDatasetWrapper", return_value=raw):
            train, val = cfg("train"), cfg("val")
            explicit = cfg("val", transform_kwargs={"no_aug": False})
        self.assertFalse(train.transform_kwargs["no_aug"])
        self.assertTrue(val.transform_kwargs["no_aug"])
        self.assertFalse(explicit.transform_kwargs["no_aug"])
        self.assertTrue(hasattr(train, "sample_weights"))
        self.assertFalse(hasattr(val, "sample_weights"))

    def test_sampler_uses_official_seed_and_draw_sequence(self):
        from psi.trainers.finetune import FinetuneTrainer
        from torch.utils.data import WeightedRandomSampler

        trainer = FinetuneTrainer.__new__(FinetuneTrainer)
        trainer.cfg = SimpleNamespace(seed=19, train=SimpleNamespace(
            num_workers=1, pin_memory=False, prefetch_factor=2, train_batch_size=2, val_batch_size=2,
        ))
        trainer.tokenizer = SimpleNamespace(model_max_length=16, pad_token_id=0)
        dataset = SimpleNamespace(sample_weights=torch.tensor([1., 4., 2., .5]))
        with patch("torch.utils.data.DataLoader", side_effect=lambda *a, **kw: SimpleNamespace(**kw)):
            train, val = trainer.create_dataloaders(dataset, dataset)
        self.assertIsInstance(train.sampler, WeightedRandomSampler)
        expected = torch.multinomial(dataset.sample_weights.double(), 4, replacement=True, generator=torch.Generator().manual_seed(19))
        self.assertEqual(list(train.sampler), expected.tolist())
        self.assertFalse(hasattr(train, "shuffle"))
        self.assertFalse(hasattr(val, "sampler"))

    def test_cli_defaults_stats_weights_and_contract(self):
        from psi.rhsci.contract import build_training_contract

        parser = build_argument_parser()
        base = ["--task", "joint", "--model", "STATE", "--dry-run"]
        old = resolve_training_plan(parser.parse_args(base)).config
        self.assertEqual(old.data.train_repo_ids, ["joint"])
        self.assertEqual(old.data.val_repo_ids, ["joint"])
        new = resolve_training_plan(parser.parse_args(base + [
            "--train-repo-ids", "a", "b", "--val-repo-ids", "c", "--task-sample-weights", "trash=4",
            "--stats-path", "/missing/shared-stats.json",
        ])).config
        self.assertEqual(new.data.val_repo_ids, ["c"])
        self.assertEqual(new.data.transform.field.stat_path, "/missing/shared-stats.json")
        self.assertEqual(new.data.parsed_task_sample_weights(), {"trash": 4})
        identity = build_training_contract(new, 1)
        changed = copy.deepcopy(new)
        changed.data.task_sample_weights = ["trash=2"]
        self.assertNotEqual(identity, build_training_contract(changed, 1))
        for rule in ("bad", " =2", "trash=0", "trash=nan", "trash=inf"):
            with self.assertRaises(ValueError):
                new.data.model_copy(update={"task_sample_weights": [rule]}).parsed_task_sample_weights()


if __name__ == "__main__":
    unittest.main()
