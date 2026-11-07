"""联合训练真实数据路由、统计与部署；只运行 CPU，不启动训练任务。"""

import importlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from rhsci.data.ardy_sidecar import ArdySidecarWriter, dataset_content_fingerprint
from rhsci.data.multitask import (
    JointDataset,
    build_training_dataset,
    multitask_contract,
    shared_statistics,
)
from rhsci.paths import worker_arguments


def moments(array):
    return {
        key: value.tolist()
        for key, value in dict(
            min=array.min(0),
            max=array.max(0),
            mean=array.mean(0),
            std=array.std(0),
            q01=np.quantile(array, 0.01, axis=0),
            q99=np.quantile(array, 0.99, axis=0),
        ).items()
    }


@unittest.skipUnless(shutil.which("ffmpeg"), "Requires ffmpeg for real RGB fixtures")
class MultitaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="rhsci-multi-")
        cls.root = Path(cls.temp.name)
        cls.roots, cls.labels = [], []
        for name, base, length in (("a", 0, 48), ("b", 1000, 64)):
            root = cls.root / name
            (root / "meta").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "videos").mkdir()
            cls.roots.append(root)
            (root / "meta/info.json").write_text(
                json.dumps(
                    dict(
                        fps=50,
                        total_episodes=2,
                        total_frames=2 * length,
                        data_path="data/episode_{episode_index:06d}.parquet",
                        video_path="videos/episode_{episode_index:06d}.mp4",
                    )
                )
            )
            (root / "meta/episodes.jsonl").write_text(
                "\n".join(
                    json.dumps(dict(episode_index=episode, length=length)) for episode in range(2)
                )
                + "\n"
            )
            (root / "meta/tasks.jsonl").write_text(
                json.dumps(dict(task_index=0, task=f"task-{name}")) + "\n"
            )
            values = base + np.arange(length, dtype=np.float64)
            raw = {
                key: np.repeat(values[:, None], width, axis=1)
                for key, width in (("action", 36), ("states", 32))
            }
            (root / "meta/stats.json").write_text(
                json.dumps({key: moments(value) for key, value in raw.items()})
            )
            for episode in range(2):
                pq.write_table(
                    pa.table(
                        dict(
                            **{key: value.tolist() for key, value in raw.items()},
                            episode_index=[episode] * length,
                            frame_index=list(range(length)),
                            task_index=[0] * length,
                        )
                    ),
                    root / f"data/episode_{episode:06d}.parquet",
                )
                width, height = getattr(cls, "video_size", (32, 32))
                pixels = np.broadcast_to(
                    (base // 10 + np.arange(length, dtype=np.uint8))[:, None, None, None],
                    (length, height, width, 3),
                ).copy()
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        f"{width}x{height}",
                        "-framerate",
                        "50",
                        "-i",
                        "pipe:0",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-qp",
                        "0",
                        "-pix_fmt",
                        "yuv444p",
                        str(root / f"videos/episode_{episode:06d}.mp4"),
                    ],
                    input=pixels.tobytes(),
                    check=True,
                )
            label = cls.root / f"labels-{name}"
            cls.labels.append(label)
            writer = ArdySidecarWriter(
                label,
                dict(
                    dataset=name,
                    processed_content_fingerprint=dataset_content_fingerprint(root),
                    teacher_verification_status="formal_verified",
                    teacher_pipeline="synthetic-test",
                    teacher_hashes={"encoder": "same-encoder"},
                ),
            )
            for episode in range(2):
                writer.write_episode(
                    episode,
                    ardy_body_latent=np.broadcast_to(
                        (base * 10 + episode * 100 + np.arange(length, dtype=np.float32) ** 2)[
                            :, None
                        ],
                        (length, 128),
                    ).copy(),
                    teacher_valid=np.ones(length, bool),
                    visual_history_valid=np.arange(length) >= 6,
                    ardy_loss_mask=np.arange(length) >= 6,
                    raw_episode_index=np.int64(episode),
                    raw_frame_index=np.arange(length, dtype=np.int64) + 1,
                )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def options(self, backbone, model="VGQ-AE-Mo-240", extra=()):
        module = importlib.import_module(f"{backbone}.rhsci.config")
        args = module.build_argument_parser().parse_args(
            [
                "--task",
                "joint",
                "--model",
                model,
                "--datasets",
                *map(str, self.roots),
                "--sidecars",
                *map(str, self.labels),
                "--no-input-cache",
                "--output-dir",
                str(self.root / "output"),
                *(["--statistics-batch-size", "16"] if backbone == "openpi" else []),
                *extra,
            ]
        )
        return args, module.resolve_training_paths(args)

    def test_real_rgb_actions_text_labels_motion_and_boundaries(self):
        for backbone in ("openpi", "gr00t"):
            cls = importlib.import_module(f"{backbone}.rhsci.data").SimpleRHSCIDataset
            for variant in (
                "STATE",
                "NO-STATE",
                "VGQ-AE",
                "VGQ-AE-240",
                "VGQ-AE-Mo",
                "VGQ-AE-Mo-240",
            ):
                with self.subTest(backbone=backbone, variant=variant):
                    args, paths = self.options(backbone, variant)
                    options = {"seed": args.seed} if backbone == "gr00t" else {"pad_actions": True}
                    dataset = build_training_dataset(paths, cls, variant, **options)
                    self.assertIsInstance(dataset, JointDataset)
                    start = 0
                    for child_index, child in enumerate(dataset.datasets):
                        per_episode = int(child.ends[0])
                        for local in (
                            0,
                            6,
                            12,
                            per_episode - 1,
                            per_episode,
                            per_episode + 12,
                            len(child) - 1,
                        ):
                            item = dataset.raw_sample(start + local)
                            frame, episode = local % per_episode, local // per_episode
                            gr = backbone == "gr00t"
                            metadata = item.metadata if gr else item
                            self.assertEqual(metadata["episode_index"], episode)
                            self.assertEqual(metadata["frame_index"], frame)
                            self.assertEqual(
                                item.text if gr else item["text"], f"task-{'ab'[child_index]}"
                            )
                            action = item.actions["left_hand"] if gr else item["actions"]
                            self.assertEqual(float(action[0, 0]), child_index * 1000 + frame)
                            self.assertEqual(
                                float(action[-1, 0]),
                                child_index * 1000
                                + min(
                                    frame + (15 if gr else 29),
                                    child.episodes[episode]["length"] - 1,
                                ),
                            )
                            pixels = np.asarray(item.images["rs_view"] if gr else item["images"])
                            self.assertAlmostEqual(
                                float(pixels[-1].mean()), child_index * 100 + frame, delta=2
                            )
                            if variant.startswith("VGQ"):
                                self.assertEqual(
                                    float(metadata["rsci_ardy_target"][0]),
                                    child_index * 10000 + episode * 100 + frame**2,
                                )
                                offsets = np.array(
                                    [-12, -8, -4, 0] if "240" in variant else [-6, -4, -2, 0]
                                )
                                np.testing.assert_array_equal(
                                    metadata["rsci_history_valid"], offsets + frame >= 0
                                )
                                if "Mo" in variant and frame >= 12:
                                    self.assertEqual(
                                        float(metadata["rsci_motion_target"][0]),
                                        frame**2 - (frame - 6) ** 2,
                                    )
                            self.assertEqual(
                                bool(item.states) if gr else item["state"] is not None,
                                variant == "STATE",
                            )
                        start += len(child)

    @unittest.skipUnless(
        importlib.util.find_spec("numpydantic"), "Requires native OpenPI normalization dependency"
    )
    def test_pi_statistics_match_native_action_chunk_batches(self):
        from openpi.shared.normalize import RunningStats

        _, paths = self.options("openpi", "STATE")
        stats = shared_statistics(paths, "pi05")
        # Independent reference: query each numeric episode as LeRobot does,
        # concatenate all frame samples, then batch in the official script order.
        states, actions = [], []
        for root in self.roots:
            for path in sorted((root / "data").glob("*.parquet")):
                table = pq.read_table(path).to_pydict()
                states.extend(np.asarray(table["states"], dtype=np.float32))
                array = np.asarray(table["action"], dtype=np.float32)
                actions.extend(
                    [
                        array[np.minimum(np.arange(frame, frame + 30), len(array) - 1)]
                        for frame in range(len(array))
                    ]
                )
        for key, samples in (("states", states), ("action", actions)):
            reference = RunningStats()
            for start in range(0, len(samples) // 16 * 16, 16):
                reference.update(np.stack(samples[start : start + 16]))
            expected = reference.get_statistics()
            for metric in ("mean", "std", "q01", "q99"):
                np.testing.assert_array_equal(stats[key][metric], getattr(expected, metric))
        self.assertNotAlmostEqual(stats["action"]["mean"][0], np.mean(np.asarray(states)[:, 0]))
        contract = multitask_contract(paths, "pi05")
        saved = json.loads(json.dumps(contract))
        self.assertEqual(saved["statistics"], stats)

    @unittest.skipUnless(
        importlib.util.find_spec("numpydantic"), "Requires native OpenPI normalization dependency"
    )
    def test_pi_statistics_global_drop_tail_and_unlisted_files(self):
        from openpi.rhsci.normalization import joint_statistics
        from openpi.shared.normalize import RunningStats

        # 224 frames, batch 50: crosses both episode and root boundaries and
        # drops 24 terminal frames globally, rather than once per episode.
        computed = joint_statistics(self.roots, 50)
        samples = []
        for base, length in ((0, 48), (1000, 64)):
            for _ in range(2):
                for frame in range(length):
                    samples.append(
                        np.repeat(
                            (base + np.minimum(frame + np.arange(30), length - 1))[:, None],
                            36,
                            axis=1,
                        ).astype(np.float32)
                    )
        native = RunningStats()
        for start in range(0, 200, 50):
            native.update(np.stack(samples[start : start + 50]))
        expected = native.get_statistics()
        for metric in ("mean", "std", "q01", "q99"):
            np.testing.assert_array_equal(computed["action"][metric], getattr(expected, metric))
        orphan = self.roots[0] / "data/unlisted.parquet"
        try:
            orphan.write_bytes(b"not a parquet, must never be read")
            self.assertEqual(computed, joint_statistics(self.roots, 50))
        finally:
            orphan.unlink()

    @unittest.skipUnless(importlib.util.find_spec("av"), "Requires native GR00T video dependencies")
    def test_gr_partition_and_processed_shards_match_native_reader(self):
        import pandas as pd

        from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
        from gr00t.rhsci.data import GROUPS, STATE_KEYS, TAG, SimpleRHSCIDataset, modality
        from gr00t.rhsci.multitask import RHSCIShardedDataset

        samples = SimpleRHSCIDataset(self.roots[0], "STATE")

        class NativeEpisodeLoader:
            episode_lengths = [episode["length"] for episode in samples.episodes]

            def get_episode_length(self, slot):
                return self.episode_lengths[slot]

            def __getitem__(self, slot):
                table = samples._table(samples.episodes[slot])
                length = self.episode_lengths[slot]
                columns = {
                    "language.annotation.human.task_description": ["task-a"] * length,
                    "video.rs_view": [np.zeros((32, 32, 3), np.uint8)] * length,
                }
                for kind, source, keys in (
                    ("state", "states", STATE_KEYS),
                    ("action", "action", GROUPS),
                ):
                    for key in keys:
                        start, end = GROUPS[key]
                        columns[f"{kind}.{key}"] = list(table[source][:, start:end])
                return pd.DataFrame(columns)

        def process(messages):
            sample = messages[0]["content"]
            return dict(
                text=sample.text,
                actions={key: value.tolist() for key, value in sample.actions.items()},
                states={key: value.tolist() for key, value in sample.states.items()},
            )

        adapter = RHSCIShardedDataset(samples, shard_size=32, episode_sampling_rate=0.1, seed=42)
        with patch(
            "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader",
            return_value=NativeEpisodeLoader(),
        ):
            native = ShardedSingleStepDataset(
                self.roots[0], TAG, modality("STATE")[TAG.value], shard_size=32, seed=42
            )
        adapter.set_processor(process)
        native.set_processor(process)
        np.testing.assert_array_equal(adapter.shard_lengths, native.shard_lengths)
        for index in range(len(adapter)):
            self.assertEqual(adapter.get_shard(index), native.get_shard(index))

    @unittest.skipUnless(importlib.util.find_spec("av"), "Requires native GR00T video dependencies")
    def test_gr_rank_worker_schedule_and_cpu_training_use_native_plain_loader(self):
        from transformers import TrainingArguments

        from gr00t.rhsci.data import SimpleRHSCIDataset
        from gr00t.rhsci.multitask import build_sharded_mixture
        from gr00t.rhsci.trainer import create_resume_trainer_class

        args, paths = self.options(
            "gr00t",
            "VGQ-AE-Mo-240",
            extra=("--num-workers", "0", "--num-shards-per-epoch", "8", "--shard-size", "32"),
        )
        dataset = build_training_dataset(paths, SimpleRHSCIDataset, args.model)

        class Processor:
            def set_statistics(self, *args, **kwargs):
                pass

            def __call__(self, messages):
                sample = messages[0]["content"]
                frame = sample.metadata["frame_index"]
                episode = sample.metadata["episode_index"]
                base = 0 if sample.text == "task-a" else 1000
                np.testing.assert_array_equal(
                    sample.actions["left_hand"][:, 0], base + np.arange(frame, frame + 16)
                )
                self_target = base * 10 + episode * 100 + frame**2
                assert sample.metadata["rsci_ardy_target"][0] == self_target
                if frame >= 12:
                    assert sample.metadata["rsci_motion_target"][0] == frame**2 - (frame - 6) ** 2
                return {"x": torch.tensor([float(base + frame)])}

        mixed = build_sharded_mixture(paths, dataset, Processor())
        # Exact slot partition for 2 ranks × 2 workers, even when the same
        # (dataset,shard) appears more than once in the global schedule.
        original = mixed.shard_sampling_schedule
        mixed.shard_sampling_schedule = list(range(32))
        partitions = []
        for rank in range(2):
            mixed.rank, mixed.world_size = rank, 2
            for worker in range(2):
                mixed.worker_id = mixed.num_workers = None
                with patch(
                    "gr00t.data.dataset.sharded_mixture_dataset.get_worker_info",
                    return_value=SimpleNamespace(id=worker, num_workers=2),
                ):
                    part = mixed.filter_shard_sample_schedule()
                self.assertEqual(part, list(range(rank * 2 + worker, 32, 4)))
                partitions.extend(part)
        self.assertEqual(sorted(partitions), list(range(32)))
        mixed.shard_sampling_schedule = original
        mixed.worker_id = mixed.num_workers = None

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x):
                return {"loss": (self.weight - x.mean() / 1000).square()}

        training = TrainingArguments(
            output_dir=str(self.root / "cpu-native-gr"),
            use_cpu=True,
            per_device_train_batch_size=2,
            max_steps=2,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
        )
        trainer = create_resume_trainer_class(native_mixture=True)(
            model=TinyModel(),
            args=training,
            train_dataset=mixed,
            data_collator=lambda items: {"x": torch.stack([item["x"] for item in items])},
        )
        # A second Accelerate data-sharding pass would violate the native split.
        with patch.object(
            trainer.accelerator,
            "prepare_data_loader",
            side_effect=AssertionError("double data sharding"),
        ):
            loader = trainer.get_train_dataloader()
            self.assertIs(type(loader), torch.utils.data.DataLoader)
            self.assertIs(loader.dataset, mixed)
            trainer.train()
        self.assertEqual(trainer.state.global_step, 2)
        self.assertIsNone(mixed._executor)
        # Exercise the real fork/persistent-worker path too, including RGB
        # decoder reopening and automatic worker cleanup after training.
        mixed = build_sharded_mixture(paths, dataset, Processor())
        training.dataloader_num_workers = 2
        training.dataloader_persistent_workers = True
        trainer = create_resume_trainer_class(native_mixture=True)(
            model=TinyModel(),
            args=training,
            train_dataset=mixed,
            data_collator=lambda items: {"x": torch.stack([item["x"] for item in items])},
        )
        trainer.train()
        self.assertEqual(trainer.state.global_step, 2)
        self.assertIsNone(trainer._rhsci_joint_loader._iterator)

    def test_gr_export_keeps_live_training_processor_untouched(self):
        from gr00t.rhsci.data import TAG, modality
        from gr00t.rhsci.trainer import build_checkpoint_callback

        args, paths = self.options("gr00t")
        observed = []

        class Processor:
            modality_configs = modality(args.model)

            def save_pretrained(self, path):
                observed.append(self.modality_configs[TAG.value]["video"].delta_indices)
                # During the save itself, the live reader must retain 4 frames.
                self_outer.assertEqual(
                    processor.modality_configs[TAG.value]["video"].delta_indices, [-12, -8, -4, 0]
                )

        self_outer = self
        processor = Processor()
        with (
            patch("gr00t.rhsci.checkpoints.complete_checkpoint"),
            patch("gr00t.rhsci.checkpoints.prune_rolling"),
        ):
            callback = build_checkpoint_callback(args, paths, processor, 1, "export-only")
            callback.on_save(None, SimpleNamespace(global_step=1, is_world_process_zero=True), None)
        self.assertEqual(observed, [[0]])
        self.assertEqual(
            processor.modality_configs[TAG.value]["video"].delta_indices, [-12, -8, -4, 0]
        )

    @unittest.skipUnless(
        importlib.util.find_spec("sentencepiece"), "π0.5 environment requires SentencePiece"
    )
    def test_pi_saved_processor_needs_no_dataset(self):
        _, paths = self.options("openpi", "STATE")
        stats = shared_statistics(paths, "pi05")
        saved = json.loads(json.dumps(multitask_contract(paths, "pi05")))
        from openpi.rhsci.processing import Processor

        with patch("sentencepiece.SentencePieceProcessor"):
            processor = Processor(
                self.root / "unavailable",
                "unused-tokenizer",
                "STATE",
                statistics=saved["statistics"],
            )
        np.testing.assert_allclose(processor.normalize(np.zeros(36), "action"), -1)
        self.assertEqual(processor.stats, stats)

    @unittest.skipUnless(importlib.util.find_spec("av"), "Requires native GR00T video dependencies")
    def test_gr_native_statistics_and_shard_mixture(self):
        from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
        from gr00t.rhsci.data import TAG, SimpleRHSCIDataset, statistics

        args, paths = self.options("gr00t", extra=("--dataset-weights", "1", "3"))
        stats = shared_statistics(paths, "gr00t")
        originals = [json.loads((root / "meta/stats.json").read_text()) for root in self.roots]
        self.assertEqual(stats, merge_statistics(originals, [1, 3]))
        self.assertEqual(
            statistics("missing", "STATE", raw=stats)[TAG.value]["state"]["height"]["mean"],
            [stats["states"]["mean"][31]],
        )
        dataset = build_training_dataset(paths, SimpleRHSCIDataset, args.model, seed=args.seed)
        from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
        from gr00t.rhsci.multitask import build_sharded_mixture

        processor = SimpleNamespace(set_statistics=lambda *args, **kwargs: None)
        mixed = build_sharded_mixture(paths, dataset, processor)
        self.assertEqual(mixed.seed, 42)
        reference = ShardedMixtureDataset(mixed.datasets, [1, 3], processor, seed=42)
        self.assertEqual(mixed.shard_sampling_schedule, reference.shard_sampling_schedule)
        self.assertEqual(mixed.global_stats, reference.global_stats)
        # Native normalizes shard probabilities by average shard length. Check
        # frame mass, not the count of shard choices, for unequal dataset sizes.
        counts = np.bincount([item[0] for item in mixed.shard_sampling_schedule], minlength=2)
        mass = counts * np.array(
            [sum(child.shard_lengths) / len(child) for child in mixed.datasets]
        )
        self.assertAlmostEqual(mass[1] / mass.sum(), 0.75, delta=0.01)
        boundary = len(dataset.datasets[0])
        self.assertEqual(dataset[(7, boundary)].metadata["frame_index"], 0)
        identity = multitask_contract(paths, "gr00t")
        self.assertEqual(identity["weights"], [1, 3])

    @unittest.skipUnless(importlib.util.find_spec("tyro"), "Requires GR00T environment")
    def test_gr_preflight_saves_and_restores_shared_processor_statistics(self):
        from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
        from gr00t.rhsci.data import TAG
        from gr00t.rhsci.preflight import run_training_preflight

        class EagleStub:
            tokenizer = SimpleNamespace(padding_side="left")

            def apply_chat_template(self, conversation, **kwargs):
                return "task"

            def process_vision_info(self, conversations):
                return [
                    item["image"]
                    for conversation in conversations
                    for item in conversation[0]["content"]
                    if item["type"] == "image"
                ], None

            def __call__(self, text, images, **kwargs):
                ids = torch.ones((len(images), 2), dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        initialization = self.root / "gr-initialization"
        initialization.mkdir(exist_ok=True)
        (initialization / "config.json").write_text(
            json.dumps(dict(max_state_dim=128, max_action_dim=128, action_horizon=50))
        )
        (initialization / "model.safetensors.index.json").write_text(
            json.dumps(dict(weight_map={}))
        )
        (initialization / "processor_config.json").write_text(
            json.dumps(
                dict(
                    processor_kwargs=dict(
                        shortest_image_edge=24, crop_fraction=1.0, use_percentiles=True
                    )
                )
            )
        )
        args, paths = self.options("gr00t", extra=("--checkpoint", str(initialization)))
        with patch(
            "gr00t.model.gr00t_n1d6.processing_gr00t_n1d6.build_processor", return_value=EagleStub()
        ):
            processor, dataset = run_training_preflight(args, paths)
            self.assertEqual(len(dataset.datasets), 2)
            saved = self.root / "saved-gr-processor"
            processor.save_pretrained(saved)
            restored = Gr00tN1d6Processor.from_pretrained(saved)
        self.assertEqual(
            restored.state_action_processor.statistics, processor.state_action_processor.statistics
        )
        shared = shared_statistics(paths, "gr00t")
        decoded = restored.decode_action(np.zeros((1, 16, 36), np.float32), TAG)
        expected = (shared["action"]["q01"][0] + shared["action"]["q99"][0]) / 2
        np.testing.assert_allclose(decoded["left_hand"], expected, atol=1e-3)

    @unittest.skipUnless(importlib.util.find_spec("sentencepiece"), "Requires π0.5 environment")
    def test_pi_serving_uses_shared_statistics_without_source_datasets(self):
        from safetensors.torch import save_file

        from openpi.rhsci.serve import Server
        from rhsci.inference.protocol import RequestMessage, ResponseMessage
        from rhsci.runtime.artifacts import sha256

        class ZeroPolicy(torch.nn.Module):
            def sample_actions(self, *args, **kwargs):
                return torch.zeros((1, 30, 36))

        _, paths = self.options("openpi", "STATE")
        tokenizer = self.root / "test-tokenizer"
        tokenizer.write_bytes(b"synthetic tokenizer")
        checkpoint = self.root / "pi-checkpoint"
        checkpoint.mkdir(exist_ok=True)
        saved = dict(
            variant="STATE",
            horizon=30,
            action_dim=36,
            vlm="frozen",
            normalization="simple-quantile-zero-span-clipped-v2",
        )
        (checkpoint / "rsci_config.json").write_text(json.dumps(saved))
        save_file({}, str(checkpoint / "model.safetensors"))
        contract = dict(
            model="STATE", tokenizer=sha256(tokenizer), multitask=multitask_contract(paths, "pi05")
        )
        with (
            patch("openpi.rhsci.model.Pi05RHSCI", return_value=ZeroPolicy()),
            patch("sentencepiece.SentencePieceProcessor") as piece,
        ):
            piece.return_value.encode.return_value = [1]
            server = Server(checkpoint, contract, None, tokenizer, device="cpu")
            request = RequestMessage(
                image={"rgb_head_stereo_left": np.zeros((32, 32, 3), np.uint8)},
                instruction="move",
                history={},
                state={"states": np.zeros((1, 32), np.float32)},
                condition={},
                gt_action=[],
                dataset_name="unseen-task",
                timestamp="0",
            )
            result = ResponseMessage.deserialize(
                json.loads(server.predict_action(request.serialize()).body)
            )
        bounds = contract["multitask"]["statistics"]["action"]
        expected = (bounds["q01"][0] + bounds["q99"][0] + 1e-6) / 2
        self.assertEqual(result.action.shape, (30, 36))
        np.testing.assert_allclose(result.action, expected, atol=1e-3)

    def test_teacher_identity_mismatch_is_rejected(self):
        from openpi.rhsci.data import SimpleRHSCIDataset

        args, paths = self.options("openpi")
        dataset = build_training_dataset(paths, SimpleRHSCIDataset, args.model)
        dataset.datasets[1].reader.manifest["context"]["teacher_hashes"] = {"encoder": "different"}
        with self.assertRaisesRegex(ValueError, "teacher_hashes"):
            JointDataset(dataset.datasets)

    def test_training_contract_tracks_second_dataset_and_embeds_statistics(self):
        for backbone, name in (("openpi", "pi05"), ("gr00t", "gr00t")):
            if backbone == "openpi" and importlib.util.find_spec("numpydantic") is None:
                continue
            args, paths = self.options(backbone)
            module = importlib.import_module(f"{backbone}.rhsci.contract")
            hash_target = (
                f"{backbone}.rhsci.{'checkpoints' if backbone == 'gr00t' else 'contract'}.sha256"
            )
            with (
                patch(hash_target, return_value="initialization-or-tokenizer"),
                patch(
                    "rhsci.data.ardy_sidecar.dataset_content_fingerprint",
                    side_effect=lambda root: Path(root).name,
                ),
            ):
                if backbone == "gr00t":
                    with patch.object(module, "optimization_kwargs", return_value={}):
                        identity, digest = module.build_training_contract(args, paths)
                else:
                    identity, digest = module.build_training_contract(args, paths)
            self.assertEqual(identity["multitask"]["statistics"], shared_statistics(paths, name))
            self.assertEqual(
                [item["fingerprint"] for item in identity["multitask"]["datasets"]], ["a", "b"]
            )
            with (
                patch(hash_target, return_value="initialization-or-tokenizer"),
                patch(
                    "rhsci.data.ardy_sidecar.dataset_content_fingerprint",
                    side_effect=lambda root: "changed-b" if Path(root).name == "b" else "a",
                ),
            ):
                if backbone == "gr00t":
                    with patch.object(module, "optimization_kwargs", return_value={}):
                        changed, new_digest = module.build_training_contract(args, paths)
                else:
                    changed, new_digest = module.build_training_contract(args, paths)
            self.assertEqual(
                identity.get("dataset", identity.get("dataset_fingerprint")),
                changed.get("dataset", changed.get("dataset_fingerprint")),
            )
            self.assertNotEqual(digest, new_digest)

    def test_both_preflights_visit_every_dataset(self):
        from gr00t.rhsci.preflight import validate_training_data
        from openpi.rhsci.preflight import run_training_preflight

        for backbone in ("openpi", "gr00t"):
            if backbone == "openpi" and importlib.util.find_spec("numpydantic") is None:
                continue
            args, paths = self.options(backbone, extra=("--check-data",))
            if backbone == "openpi":
                _, dataset = run_training_preflight(args, paths)
                self.assertEqual(len(dataset.datasets), 2)
            else:
                validate_training_data(args, paths)

    def test_worker_forwarding_and_single_dataset_compatibility(self):
        for backbone in ("openpi", "gr00t"):
            module = importlib.import_module(f"{backbone}.rhsci.config")
            parser = module.build_argument_parser()
            args, paths = self.options(backbone)
            forwarded = parser.parse_args(worker_arguments(args, parser))
            self.assertEqual(forwarded.datasets, self.roots)
            self.assertEqual(forwarded.sidecars, self.labels)
            single = parser.parse_args(
                [
                    "--task",
                    "a",
                    "--model",
                    "NO-STATE",
                    "--dataset",
                    str(self.roots[0]),
                    "--no-input-cache",
                ]
            )
            single_paths = module.resolve_training_paths(single)
            cls = importlib.import_module(f"{backbone}.rhsci.data").SimpleRHSCIDataset
            self.assertIsInstance(build_training_dataset(single_paths, cls, single.model), cls)
            self.assertNotIn("datasets", single_paths)
            self.assertIsNone(shared_statistics(single_paths, "unused"))
            for extra in (("--dataset", str(self.roots[0])), ("--sidecars", str(self.labels[0]))):
                with self.assertRaises(ValueError):
                    self.options(backbone, extra=extra)

    @unittest.skipUnless(importlib.util.find_spec("sentencepiece"), "Requires π0.5 environment")
    def test_joint_http_entry_accepts_no_dataset_and_single_still_requires_it(self):
        import sys

        from rhsci.inference import http_server

        checkpoint = self.root / "http-checkpoint"
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "rsci_config.json").write_text(json.dumps(dict(variant="STATE")))
        argv = [
            "serve",
            "--backbone",
            "pi05",
            "--run-dir",
            str(self.root),
            "--tokenizer",
            "tokenizer",
        ]
        for joint in (True, False):
            contract = dict(model="STATE")
            if joint:
                contract["multitask"] = {"statistics": {"action": "saved"}}
            with (
                patch.object(sys, "argv", argv),
                patch.object(http_server, "inspect_run", return_value=(contract, checkpoint)),
                patch("openpi.rhsci.serve.Server") as server,
                patch("uvicorn.run") as run,
            ):
                if joint:
                    http_server.main()
                    self.assertIsNone(server.call_args.args[2])
                    run.assert_called_once()
                else:
                    with self.assertRaises(SystemExit):
                        http_server.main()
                    server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
