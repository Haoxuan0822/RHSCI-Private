"""检验实际解析配方、参数覆盖、Adam更新/状态及EMA，不启动全尺寸训练。"""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_integration import inputs, tiny, vlm_inputs
from torch import nn

from rsci.backbones.lambda0.optimization import (
    effective_arguments,
    native_arguments,
    optimizer_report,
)
from rsci.backbones.lambda0.runtime import RSCITrainingBackend


class OptimizationTests(unittest.TestCase):
    def test_custom_bundle_paths_and_training_budget(self):
        from rsci.backbones.lambda0.train import configuration, parser

        args = parser().parse_args(
            [
                "--task",
                "custom",
                "--variant",
                "STATE",
                "--dataset",
                "/datasets/custom",
                "--bundle",
                "/bundles/custom-safe",
                "--data-root",
                "/datasets",
                "--output-dir",
                "/runs/custom",
                "--max-steps",
                "2000",
                "--seed",
                "7",
            ]
        )
        config = configuration(args)
        self.assertEqual(config["output"], "/runs/custom")
        self.assertEqual(config["data"], "/bundles/custom-safe")
        self.assertEqual(config["data_root"], "/datasets")
        self.assertEqual(config["training"]["iters"], 2000)
        self.assertEqual(config["seed"], 7)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_four_actual_recipes_against_official_profile_and_parser(self):
        from lambda0.training import engine
        from lambda0.training.runner import resolve_experiment

        from rsci.backbones.lambda0.contracts import UPSTREAM
        from rsci.backbones.lambda0.train import configuration, parser

        recipe = json.loads((UPSTREAM / "lambda0/lambda0/recipes/posttrain.json").read_text())[
            "training"
        ]
        single = json.loads(
            (UPSTREAM / "lambda0/configs/posttrain/simple_core6/single.train.json").read_text()
        )
        operational = (
            "output_dir",
            "release_stage",
            "initialization",
            "validation_metas_path",
            "data_root",
            "sampling_plan_json",
            "release_data_manifest",
            "model_revision",
        )
        for task in ("hug", "between"):
            for variant in ("STATE", "VGQ-AE"):
                args = parser().parse_args(
                    [
                        "--dataset",
                        "/datasets/test",
                        "--sidecar",
                        "/labels/test",
                        "--task",
                        task,
                        "--variant",
                        variant,
                        "--gpus",
                        "0,1,2,3",
                    ]
                )
                metadata = os.environ.get("RSCI_TEST_LAMBDA0_BUNDLE")
                if metadata:
                    args.bundle = Path(metadata).resolve().parent
                config = configuration(args)
                if (
                    not Path(config["model"]).exists()
                    or not (Path(config["data"]) / "manifest.json").is_file()
                ):
                    self.skipTest(
                        "requires the external λ₀ initialization checkpoint and prepared bundle"
                    )
                cli = SimpleNamespace(
                    config=None,
                    stage=None,
                    data=None,
                    data_root=None,
                    model=None,
                    model_revision=None,
                    output=None,
                    resume=None,
                    rtc=None,
                    prediction_type=None,
                    initialization=None,
                    batch_size=None,
                    global_batch_size=None,
                    gradient_accumulation_steps=None,
                    seed=None,
                    num_workers=None,
                )
                with tempfile.TemporaryDirectory() as folder:
                    cli.config = Path(folder) / "audit.json"
                    cli.config.write_text(json.dumps(config))
                    with patch.dict(os.environ, WORLD_SIZE="4"):
                        settings = resolve_experiment(cli)["training"]
                actual = vars(native_arguments(settings))
                reference = vars(
                    engine.get_args_parser().parse_args(
                        [
                            "--models",
                            actual["models"],
                            "--train_metas_path",
                            actual["train_metas_path"],
                        ]
                    )
                )
                reference.update(recipe)
                reference.update(single["training"])
                reference.update(
                    batch_size=single["batch_size"],
                    gradient_accumulation_steps=single["gradient_accumulation_steps"],
                )
                reference.update({key: actual[key] for key in operational})
                differences = {
                    key: (reference.get(key), actual.get(key))
                    for key in reference.keys() | actual.keys()
                    if reference.get(key) != actual.get(key)
                }
                self.assertEqual(
                    differences, {"num_domains": (8, 30), "save_interval": (10000, 1000)}
                )
                effective = effective_arguments(settings, variant)
                if variant == "VGQ-AE":
                    self.assertFalse(effective["use_proprio"])
                    self.assertEqual(
                        effective["state_conditioning_mode"], "continuous_action_expert"
                    )
                else:
                    self.assertEqual(effective, actual)
                self.assertEqual((actual["dtype"], actual["mixed_precision"]), ("auto", "bf16"))
                self.assertFalse(actual["fused_optimizer"])
                self.assertEqual((actual["ema_decay"], actual["ema_update_interval"]), (0.999, 1))
                self.assertFalse(actual["validation_use_ema"])

    def test_optimizer_coverage_rejects_missing_duplicate_or_frozen(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer_report(model, optimizer)
        parameter = optimizer.param_groups[0]["params"].pop()
        with self.assertRaisesRegex(ValueError, "exactly once"):
            optimizer_report(model, optimizer)
        optimizer.param_groups[0]["params"].append(parameter)
        optimizer.param_groups[0]["params"].append(parameter)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            optimizer_report(model, optimizer)
        optimizer.param_groups[0]["params"].pop()
        parameter.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            optimizer_report(model, optimizer)

    def test_parameter_updates_moments_accumulation_and_ema_match_native(self):
        from accelerate import Accelerator
        from lambda0.training import engine

        accelerator = Accelerator(cpu=True, mixed_precision="no")
        args = SimpleNamespace(
            weight_decay=1e-10,
            betas=[0.9, 0.95],
            fused_optimizer=False,
            learning_rate=5e-5,
            vlm_lr_coef=1.0,
            conditioning_lr_coef=1.0,
            freeze_steps=0,
            warmup_steps=0,
            new_module_warmup_steps=0,
            iters=40000,
            min_lr_ratio=0.01,
            use_cosine_decay=True,
        )
        for variant in ("STATE", "VGQ-AE"):
            native = tiny(variant)
            adapted = copy.deepcopy(native)
            for model in (native, adapted):
                engine.configure_trainable_action_spaces(model, [3])
            reference_optimizer = engine.build_optimizer(native, args)
            backend = RSCITrainingBackend(engine, variant, {})
            actual_optimizer = backend.build_optimizer(adapted, args)
            reference_report = optimizer_report(native, reference_optimizer)
            self.assertEqual(reference_report, optimizer_report(adapted, actual_optimizer))
            self.assertEqual(
                {group["name"] for group in reference_report["effective_groups"]},
                {"vlm", "action_expert", "conditioning", "action_heads"},
            )
            for group in reference_optimizer.param_groups:
                self.assertEqual(
                    (group["betas"], group["eps"], group["weight_decay"]),
                    ((0.9, 0.95), 1e-8, 1e-10),
                )
                self.assertFalse(group["amsgrad"])
                self.assertFalse(group["maximize"])
                self.assertIsNone(group["foreach"])
                self.assertFalse(group["fused"])
                self.assertFalse(group["capturable"])
                self.assertFalse(group["differentiable"])
                self.assertTrue(group["decoupled_weight_decay"])
            if variant == "VGQ-AE":
                auxiliary = [
                    name
                    for group in reference_report["effective_groups"]
                    if group["name"] == "action_expert"
                    for name in group["auxiliary_parameters"]
                ]
                self.assertEqual(
                    set(auxiliary),
                    {name for name, _ in native.named_parameters() if name.startswith("rsci_")},
                )
            emas = [
                engine.create_ema_model(model, accelerator, 0.999, 1) for model in (native, adapted)
            ]
            for step in (0, 1, 20000, 39999):
                batch = inputs(native)
                extra = (
                    {}
                    if variant == "STATE"
                    else dict(
                        history=[vlm_inputs() for _ in range(3)],
                        history_valid=torch.ones(2, 4, dtype=torch.bool),
                        ardy_target=torch.randn(2, 128),
                        ardy_valid=torch.tensor([True, False]),
                    )
                )
                for index, (model, optimizer) in enumerate(
                    ((native, reference_optimizer), (adapted, actual_optimizer))
                ):
                    engine.update_group_lrs(optimizer, step, args)
                    optimizer.zero_grad(set_to_none=True)
                    # 使用原生累计公式，两个相同microbatch检验额外门禁不改变更新。
                    for _ in range(2):
                        accelerator.backward(model(**batch, **extra)["loss"] / 2)
                    engine.clip_gradients(accelerator, model, 1.0)
                    optimizer.step()
                    emas[index].update_parameters(model)
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(value, adapted.state_dict()[key], rtol=0, atol=0)
                first, second = reference_optimizer.state_dict(), actual_optimizer.state_dict()
                self.assertEqual(first["param_groups"], second["param_groups"])
                for key, values in first["state"].items():
                    for name, value in values.items():
                        torch.testing.assert_close(
                            value, second["state"][key][name], rtol=0, atol=0
                        )
                for key, value in emas[0].state_dict().items():
                    torch.testing.assert_close(value, emas[1].state_dict()[key], rtol=0, atol=0)
                self.assertAlmostEqual(
                    reference_optimizer.param_groups[0]["lr"],
                    engine.scheduled_learning_rate(step, 0, 0, 40000, 5e-5, 0.01, True),
                    places=15,
                )
                # 未选动作头无梯度、无动量、保持冻结。
                for model, optimizer in (
                    (native, reference_optimizer),
                    (adapted, actual_optimizer),
                ):
                    for space in ("0", "1", "2", "4"):
                        for module in (
                            model.action_space.action_encoders[space],
                            model.action_space.action_decoders[space],
                        ):
                            for parameter in module.parameters():
                                self.assertFalse(parameter.requires_grad)
                                self.assertIsNone(parameter.grad)
                                self.assertNotIn(parameter, optimizer.state)

    def test_native_bfloat16_adam_has_no_implicit_fp32_moments(self):
        from lambda0.training import engine

        model = nn.Linear(2, 1).to(torch.bfloat16)
        args = SimpleNamespace(weight_decay=1e-10, betas=[0.9, 0.95], fused_optimizer=False)
        optimizer = engine.build_optimizer(model, args)
        for group in optimizer.param_groups:
            group["lr"] = 5e-5
        model(torch.ones(2, 2, dtype=torch.bfloat16)).float().sum().backward()
        optimizer.step()
        for parameter, state in optimizer.state.items():
            self.assertEqual(parameter.dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
            self.assertEqual(state["step"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
