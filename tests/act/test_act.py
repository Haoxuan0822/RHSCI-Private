"""与原生ACT对照，而非只验证新增实现自己的形状。全部CPU，无下载。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from act.models.act import ACTConfig
from rsci.backbones.act.model import RSCIACT, masked_ardy_loss
from rsci.backbones.act.train import create_optimizer


def config(state=False):
    return ACTConfig(
        chunk_size=3,
        n_action_steps=3,
        action_dim=36,
        state_dim=36,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        use_robot_state=state,
        pretrained_backbone_weights=None,
        dropout=0.0,
    )


def batch():
    images = torch.randn(2, 1, 3, 32, 32)
    return {
        "observation.images": images,
        "observation.state": torch.randn(2, 36),
        "action": torch.randn(2, 3, 36),
        "action_is_pad": torch.tensor([[False, False, True], [False, False, False]]),
        "history_images": torch.cat((torch.randn(2, 3, 3, 32, 32), images), 1),
        "ardy_target": torch.randn(2, 128),
        "ardy_valid": torch.tensor([True, False]),
    }


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_state_native_forward_loss_and_gradients(self):
        model = RSCIACT("STATE", config(True)).train()
        native = copy.deepcopy(model.policy).train()
        inputs = batch()
        torch.manual_seed(17)
        original = native(inputs)["loss"]
        original.backward()
        torch.manual_seed(17)
        ours = model(**inputs)["loss"]
        ours.backward()
        torch.testing.assert_close(original, ours, rtol=0, atol=0)
        for (name, p), (other, q) in zip(
            native.named_parameters(), model.policy.named_parameters()
        ):
            self.assertEqual(name, other)
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        model.eval()
        native.eval()
        observations = {k: v for k, v in inputs.items() if k.startswith("observation.")}
        torch.testing.assert_close(
            model.predict_action(inputs), native.predict_action(observations), rtol=0, atol=0
        )

    def test_full_checkpoint_round_trip_preserves_parameters_and_buffers(self):
        from safetensors.torch import save_file

        from rsci.backbones.act.checkpoints import load_model_state

        model = RSCIACT("EGQ-AE", config()).eval()
        inputs = batch()
        expected = model.predict_action(inputs)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            save_file(model.state_dict(), str(path / "model.safetensors"))
            (path / "rsci_config.json").write_text(json.dumps(model.rsci_config))
            restored = RSCIACT("EGQ-AE", config()).eval()
            load_model_state(restored, path)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(expected, restored.predict_action(inputs), rtol=0, atol=0)

    def test_no_state_removes_both_projections_and_ignores_actual_state(self):
        model = RSCIACT("NO-STATE", config()).eval()
        inputs = batch()
        self.assertFalse(
            any("robot_state_input_proj" in name for name, _ in model.named_parameters())
        )
        expected = model.predict_action(inputs)
        inputs["observation.state"].fill_(float("nan"))
        torch.testing.assert_close(expected, model.predict_action(inputs), rtol=0, atol=0)
        inputs.pop("observation.state")
        torch.testing.assert_close(expected, model.predict_action(inputs), rtol=0, atol=0)
        self.assertEqual(model._native_batch(inputs)["observation.state"].numel(), 0)

    def test_history_and_ardy_cannot_change_current_actions(self):
        model = RSCIACT("EGQ-AE", config()).eval()
        inputs = batch()
        expected = model.predict_action(inputs)
        feature = model.policy.model.backbone(inputs["observation.images"][:, 0])["feature_map"]
        ardy = model.auxiliary_prediction(inputs["history_images"], feature)
        inputs["history_images"][:, :3].add_(10)
        changed = model.auxiliary_prediction(inputs["history_images"], feature)
        self.assertFalse(torch.allclose(ardy, changed))
        torch.testing.assert_close(expected, model.predict_action(inputs), rtol=0, atol=0)
        torch.manual_seed(19)
        first = model(**inputs)["ardy_loss"]
        inputs["action"].add_(100)
        torch.manual_seed(19)
        second = model(**inputs)["ardy_loss"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_auxiliary_nonzero_gradient_intersects_action_policy(self):
        model = RSCIACT("EGQ-AE", config()).train()
        inputs = batch()
        # 先独立ARDY反传，不能用总loss梯度伪证辅助可达性。
        maps = model.policy.model.backbone(inputs["observation.images"][:, 0])["feature_map"]
        pred = model.auxiliary_prediction(inputs["history_images"], maps)
        masked_ardy_loss(pred, inputs["ardy_target"], inputs["ardy_valid"]).backward()
        aux = {
            name
            for name, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
        model.zero_grad(set_to_none=True)
        state = {
            k: v
            for k, v in inputs.items()
            if k not in ("ardy_target", "ardy_valid", "history_images")
        }
        model(**state)["loss"].backward()
        action = {
            name
            for name, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
        for prefix in ("policy.model.backbone", "policy.model.encoder.", "policy.model.decoder."):
            self.assertTrue(any(name.startswith(prefix) for name in aux & action), prefix)
        self.assertFalse(any("vae_encoder" in name for name in aux))

    def test_empty_ardy_labels_zero_graph_and_invalid_nan_mask(self):
        pred = torch.randn(2, 128, requires_grad=True)
        loss = masked_ardy_loss(
            pred, torch.full_like(pred, float("nan")), torch.zeros(2, dtype=torch.bool)
        )
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(pred.grad.abs().sum(), 0)

    def test_native_action_queue_and_reset(self):
        model = RSCIACT("NO-STATE", config()).eval()
        inputs = batch()
        chunk = model.predict_action(inputs)
        for i in range(3):
            torch.testing.assert_close(model.select_action(inputs), chunk[:, i], rtol=0, atol=0)
        model.select_action(inputs)
        model.reset()
        self.assertEqual(len(model.policy._action_queue), 0)

    def test_optimizer_is_native_single_group_and_schedule(self):
        from transformers import get_scheduler

        original = torch.nn.Linear(3, 2)
        model = copy.deepcopy(original)
        ours = create_optimizer(model, cpu=True)
        native = torch.optim.AdamW(
            original.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8
        )
        self.assertEqual(len(ours.param_groups), 1)
        self.assertEqual(ours.param_groups[0]["weight_decay"], 1e-6)
        schedule = get_scheduler("cosine", ours, num_warmup_steps=1000, num_training_steps=40000)
        reference = get_scheduler("cosine", native, num_warmup_steps=1000, num_training_steps=40000)
        for _ in range(5):
            x = torch.randn(2, 3)
            model(x).square().mean().backward()
            original(x).square().mean().backward()
            ours.step()
            native.step()
            schedule.step()
            reference.step()
            ours.zero_grad()
            native.zero_grad()
            for p, q in zip(model.parameters(), original.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        for step in (0, 999, 1000, 20000, 40000):
            self.assertEqual(schedule.lr_lambdas[0](step), reference.lr_lambdas[0](step))


if __name__ == "__main__":
    unittest.main()
