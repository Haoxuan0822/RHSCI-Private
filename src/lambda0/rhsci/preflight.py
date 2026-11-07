"""λ₀ 数据、输出目录和真实模型预检；按调用模式显式执行。"""

import json
from pathlib import Path
from types import SimpleNamespace

from rhsci.data.visual_history import visual_history_offsets
from rhsci.paths import dataset_paths

from .contracts import sha256


def validate_training_data(args, config):
    """检查训练数据、归一化与历史帧边界，不进行模型训练更新。"""
    import numpy as np

    from lambda0.data.bundle import verify_bundle
    from lambda0.data.dataset import InfiniteDataReader
    from lambda0.data.domain_handler.registry import get_handler_cls

    from .data import RHSCICollator

    bundle = Path(config["data"])
    provenance = json.loads((bundle / "rsci_provenance.json").read_text())
    paths = dataset_paths(
        SimpleNamespace(
            task=args.task, dataset=args.dataset, sidecar=args.sidecar, model=args.variant
        )
    )
    dataset_root = paths["dataset"]
    inventory = {
        path.relative_to(dataset_root).as_posix()
        for path in list((dataset_root / "meta").glob("*"))
        + list((dataset_root / "data").rglob("*.parquet"))
        + list((dataset_root / "videos").rglob("*.mp4"))
        if path.is_file()
    }
    if inventory != set(provenance["files"]):
        raise ValueError("Dataset provenance inventory changed")
    for name, expected in provenance["files"].items():
        if sha256(dataset_root / name) != expected:
            raise ValueError(f"Dataset provenance changed: {name}")
    if (
        args.variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480")
        and sha256(paths["sidecar"] / "manifest.json") != provenance["sidecar_context"]
    ):
        raise ValueError("ARDY manifest changed")
    manifest = verify_bundle(bundle, config["data_root"], {"strategy": "frame_uniform"})
    reader = InfiniteDataReader(
        str(bundle / "train"), data_root=config["data_root"], training=False
    )
    scan_frames, maximum, roundtrip = 0, 0.0, 0.0
    for meta in reader.metas.values():
        if args.variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"):
            meta["handler"] = "lambda0.rhsci.data:RHSCIPsiSimpleHandler"
            meta.update(rsci_variant=args.variant, rsci_ardy_delta_frames=getattr(args, "ardy_delta_frames", 6))
        elif meta["normalization_contract"] != "quantile_q01_q99_v1":
            meta["handler"] = "lambda0.rhsci.data:SafePsiSimpleHandler"
        handler = get_handler_cls(meta["handler"])(meta, reader.num_views)
        # 全量训练文件扫描真实极值，保留官方不裁剪的归一化，不用随机样本掩盖异常。
        for item in meta["datalist"]:
            states, actions, _, _ = handler._load_episode(item)
            for raw, normalize in (
                (states, handler.normalizer.normalize_proprio),
                (actions, handler.normalizer.normalize_action),
            ):
                normalized = normalize(raw)
                if not np.isfinite(normalized).all():
                    raise FloatingPointError("Nonfinite official-normalized training data")
                maximum = max(maximum, float(np.abs(normalized).max()))
            recovered = handler.normalizer.denormalize_action(
                handler.normalizer.normalize_action(actions)
            )
            roundtrip = max(roundtrip, float(np.abs(recovered - actions).max()))
            scan_frames += len(actions)
        for episode in (0, len(meta["datalist"]) - 1):
            item = meta["datalist"][episode]
            for index in (0, -visual_history_offsets(args.variant)[0], item["length"] - 1):
                sample = handler.sample_episode(
                    episode,
                    num_actions=50,
                    image_aug=reader.image_aug,
                    lang_aug_map=None,
                    sample_index=index,
                )
                sample = reader._finalize_sample(meta["dataset_name"], sample)
                batch = RHSCICollator()([sample])
                assert batch["action"].shape[1] == 50
                if args.variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"):
                    assert (
                        not batch["proprio"].count_nonzero()
                        and not batch["proprio_valid_mask"].any()
                    )
                    assert batch["history"].shape[1] == 3
    if maximum > 1000:
        raise FloatingPointError(f"Unsafe normalization magnitude: {maximum}")
    print(
        json.dumps(
            dict(
                task=args.task,
                variant=args.variant,
                training_frames=scan_frames,
                normalized_max_abs=maximum,
                action_roundtrip_max=roundtrip,
                split=manifest["datasets"],
            ),
            indent=2,
        )
    )


def validate_output_directory(args, output):
    """检查既有输出目录与 resume 选择，拒绝在无完整断点时静默从零开始。

    协调日志不代表训练状态；有训练痕迹却无完整断点时不得悄悄从零开始。"""
    output = Path(output)
    if not output.exists():
        return
    coordination = {"guardian.lock", "launch.lock", "guardian.log", "worker.json"}
    artifacts = [
        path
        for path in output.iterdir()
        if path.name not in coordination
        and not (path.name.startswith("attempt-") and path.name.endswith(".log"))
    ]
    if args.resume == "none" and artifacts:
        raise FileExistsError("Existing run artifacts; choose a new run-tag")
    if args.resume == "auto" and artifacts and not (output / "last/rsci_complete.json").is_file():
        raise ValueError("Existing run has no verified complete checkpoint; choose a new run-tag")


def run_model_preflight(args, config, backend, settings, native_args):
    """CPU加载真实模型，核验处理和辅助模块回载；不执行训练更新。"""
    import torch

    from lambda0.training import engine

    engine.set_seed(0)
    model, processor = backend.load_model_and_processor(native_args)
    engine.configure_trainable_action_spaces(model, native_args.trainable_action_space_ids)
    optimizer = backend.build_optimizer(model, native_args)
    from .optimization import optimizer_report

    optimization = optimizer_report(model, optimizer)
    loader = backend.create_dataloader(
        batch_size=1,
        metas_path=settings["train_metas_path"],
        num_actions=50,
        training=True,
        num_workers=0,
        data_root=config["data_root"],
        sampling_plan={"strategy": "frame_uniform"},
        sampling_seed=0,
    )
    batch = next(iter(loader))
    prepared = backend.prepare_inputs(batch, processor, torch.device("cpu"), model)
    if args.variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"):
        assert len(prepared["history"]) == 3 and not prepared["proprio_valid_mask"].any()
    reload_exact = None
    if args.variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480"):
        # 不复制5GB基础权重：用只读symlink加新增模块小分片，验真实HF构造/严格回载。
        import tempfile

        from safetensors import safe_open
        from safetensors.torch import save_file

        from .model import Lambda0RHSCI

        with tempfile.TemporaryDirectory(prefix="rsci-lambda0-reload-") as temp:
            fixture = Path(temp)
            auxiliary = {
                name: tensor.contiguous()
                for name, tensor in model.state_dict().items()
                if name.startswith("rsci_")
            }
            save_file(auxiliary, str(fixture / "aux.safetensors"))
            original = Path(config["model"]) / "model.safetensors"
            (fixture / "base.safetensors").symlink_to(original)
            with safe_open(original, framework="pt") as stream:
                weight_map = {name: "base.safetensors" for name in stream.keys()}
            weight_map.update({name: "aux.safetensors" for name in auxiliary})
            (fixture / "model.safetensors.index.json").write_text(
                json.dumps(dict(metadata={}, weight_map=weight_map))
            )
            model.config.save_pretrained(fixture)
            restored = Lambda0RHSCI.from_pretrained(
                fixture, dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            for name, value in auxiliary.items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            assert not restored.use_proprio
            reload_exact = True
    print(
        json.dumps(
            dict(
                variant=args.variant,
                parameters=sum(p.numel() for p in model.parameters()),
                action_dim=model.action_space.specs[3].action_dim,
                num_actions=model.num_actions,
                use_proprio=model.use_proprio,
                state_conditioning_mode=model.config.state_conditioning_mode,
                processor_keys=sorted(prepared),
                auxiliary_reload_exact=reload_exact,
                optimizer=optimization,
            ),
            indent=2,
        )
    )
    return
