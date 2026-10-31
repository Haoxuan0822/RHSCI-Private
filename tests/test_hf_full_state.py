"""公共HF门禁保持骨干schema隔离、全rank恢复及先验新点再清理。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from human_plan.rsci import checkpoints as ego
from safetensors.torch import save_file

from act.rsci import checkpoints as act


class HFStateTests(unittest.TestCase):
    def checkpoint(self, output, step, protocol, world=2):
        path = output / f"checkpoint-{step}"
        path.mkdir()
        for name in protocol.required(world):
            (path / name).write_bytes(b"fixture")
        save_file({"weight": torch.ones(2)}, str(path / "model.safetensors"))
        torch.save({"state": {0: {"exp_avg": torch.zeros(2)}}}, path / "optimizer.pt")
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        protocol.publish(path, world, "recipe")
        return path

    def test_schemas_remain_incompatible_and_all_rank_rng_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for index, (own, other) in enumerate(((act, ego), (ego, act)), 1):
                path = self.checkpoint(output, index * 1000, own)
                self.assertTrue(own.checkpoint_ok(path, 2, "recipe", verify=True))
                self.assertFalse(other.checkpoint_ok(path, 2, "recipe", verify=True))
                self.assertFalse(own.checkpoint_ok(path, 1, "recipe"))
                self.assertFalse(own.checkpoint_ok(path, 2, "changed"))
                (path / "rng_state_1.pth").unlink()
                self.assertFalse(own.checkpoint_ok(path, 2, "recipe"))

    def test_broken_new_checkpoint_cannot_prune_and_milestones_survive(self):
        for own in (act, ego):
            with self.subTest(schema=own), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                milestone = self.checkpoint(output, 10000, own)
                rolling = self.checkpoint(output, 11000, own)
                new = self.checkpoint(output, 12000, own)
                original = (new / "optimizer.pt").read_bytes()
                (new / "optimizer.pt").write_bytes(b"x" * len(original))
                self.assertFalse(own.checkpoint_ok(new, 2, "recipe", verify=True))
                with self.assertRaises(ValueError):
                    own.prune(output, new, 2, "recipe")
                self.assertTrue(rolling.exists())
                (new / "optimizer.pt").write_bytes(original)
                own.prune(output, new, 2, "recipe")
                self.assertTrue(milestone.exists())
                self.assertTrue(new.exists())
                self.assertFalse(rolling.exists())
                self.assertEqual(own.latest(output, 2, "recipe"), new)

    def test_nonfinite_weights_or_adam_never_publish(self):
        for own in (act, ego):
            for target in ("model", "optimizer"):
                with (
                    self.subTest(protocol=own, target=target),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    path = self.checkpoint(Path(tmp), 1000, own)
                    (path / own.MARKER).unlink()
                    if target == "model":
                        save_file(
                            {"weight": torch.full((2,), float("nan"))},
                            str(path / "model.safetensors"),
                        )
                    else:
                        torch.save(
                            {"exp_avg": torch.full((2,), float("inf"))}, path / "optimizer.pt"
                        )
                    with self.assertRaises(FloatingPointError):
                        own.publish(path, 2, "recipe")
                    self.assertFalse((path / own.MARKER).exists())
