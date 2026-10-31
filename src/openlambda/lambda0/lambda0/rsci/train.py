"""官方 λ₀ SIMPLE 后训练入口；额外提供只读检查及 VGQ-AE。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from rsci.paths import add_data_arguments, dataset_paths
from rsci.runtime.contracts import reuse_run_contract

from .contracts import CODE as CODE
from .contracts import CONTRACT, RUNTIME, UPSTREAM_COMMIT, sha256


def build_argument_parser():
    """构造命令行解析器并读取环境默认值；返回解析器，不加载模型或启动训练。"""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(argument_parser)
    argument_parser.add_argument("--task", required=True)
    argument_parser.add_argument("--bundle", type=Path, help="Prepared RHSCI lambda0 data bundle")
    argument_parser.add_argument(
        "--data-root", type=Path, help="Root used to resolve bundle metadata paths"
    )
    argument_parser.add_argument("--max-steps", type=int, default=40000)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--variant", choices=("STATE", "VGQ-AE"), default="VGQ-AE")
    argument_parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    argument_parser.add_argument(
        "--checkpoint", type=Path, help="本地官方 Stage-II raw checkpoint，不自动下载"
    )
    argument_parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    argument_parser.add_argument("--batch-size", type=int, default=4)
    argument_parser.add_argument("--global-batch-size", type=int, default=16)
    argument_parser.add_argument("--num-workers", type=int, default=4)
    argument_parser.add_argument("--run-tag", default="main")
    argument_parser.add_argument("--resume", choices=("none", "auto"), default="auto")
    argument_parser.add_argument(
        "--check", action="store_true", help="只读配置与路径检查，不加载模型"
    )
    argument_parser.add_argument(
        "--check-data", action="store_true", help="真实数据/官方归一化/四帧检查，不训练"
    )
    argument_parser.add_argument(
        "--check-model", action="store_true", help="CPU严格加载真实Stage-II与处理器，不训练"
    )
    argument_parser.add_argument(
        "--dry-run", action="store_true", help="官方完整解析并验证资产/数据，不训练"
    )
    argument_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return argument_parser


def build_training_config(args):
    """校验 CLI 参数；返回供原生 λ₀ resolver 使用的训练配置。"""
    if Path(args.run_tag).name != args.run_tag or args.run_tag in ("", ".", ".."):
        raise ValueError("run-tag must be a plain directory name")
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError("gpus must contain distinct integer IDs")
    from lambda0.training.runner import batch_layout

    accumulation = batch_layout(args.global_batch_size, args.batch_size, len(gpus))
    if args.num_workers < 0 or args.max_steps < 1:
        raise ValueError("num-workers must be nonnegative and max-steps positive")
    dataset_paths(
        SimpleNamespace(
            task=args.task, dataset=args.dataset, sidecar=args.sidecar, model=args.variant
        )
    )
    output = (
        args.output_dir
        or args.runtime_root / "runs/lambda0" / args.task / args.variant / args.run_tag
    )
    bundle = args.bundle or args.runtime_root / "bundles/lambda0" / f"{args.task}-safe"
    model = args.checkpoint or args.runtime_root / "cache/checkpoints/lambda0-midtrain-100k"
    domains = 30
    if (model / "config.json").is_file():
        domains = max(8, int(json.loads((model / "config.json").read_text())["num_domains"]))
    config = dict(
        stage="posttrain",
        initialization="midtrain",
        midtrain_weight_source="raw",
        prediction_type="velocity",
        model=str(model),
        data=str(bundle),
        data_root=str(args.data_root or args.runtime_root / "data"),
        output=str(output),
        seed=args.seed,
        batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
        gradient_accumulation_steps=accumulation,
        resume=args.resume,
        training=dict(
            iters=args.max_steps,
            num_domains=domains,
            trainable_action_space_ids=[3],
            save_interval=1000,
            num_workers=args.num_workers,
        ),
    )
    return config


def validate_training_data(args, config):
    """检查训练数据、归一化与历史帧边界，不进行模型训练更新。"""
    import numpy as np

    from lambda0.data.bundle import verify_bundle
    from lambda0.data.dataset import InfiniteDataReader
    from lambda0.data.domain_handler.registry import get_handler_cls

    from .data import RSCICollator

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
        args.variant == "VGQ-AE"
        and sha256(paths["sidecar"] / "manifest.json") != provenance["sidecar_context"]
    ):
        raise ValueError("ARDY manifest changed")
    manifest = verify_bundle(bundle, config["data_root"], {"strategy": "frame_uniform"})
    reader = InfiniteDataReader(
        str(bundle / "train"), data_root=config["data_root"], training=False
    )
    scan_frames, maximum, roundtrip = 0, 0.0, 0.0
    for meta in reader.metas.values():
        if args.variant == "VGQ-AE":
            meta["handler"] = "lambda0.rsci.data:RSCIPsiSimpleHandler"
        elif meta["normalization_contract"] != "quantile_q01_q99_v1":
            meta["handler"] = "lambda0.rsci.data:SafePsiSimpleHandler"
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
            for index in (0, 6, item["length"] - 1):
                sample = handler.sample_episode(
                    episode,
                    num_actions=50,
                    image_aug=reader.image_aug,
                    lang_aug_map=None,
                    sample_index=index,
                )
                sample = reader._finalize_sample(meta["dataset_name"], sample)
                batch = RSCICollator()([sample])
                assert batch["action"].shape[1] == 50
                if args.variant == "VGQ-AE":
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


def execute_training(args, config):
    """将 RHSCI 配置接入原生 λ₀；按 CLI 模式执行只读检查或当前 worker 的训练。"""
    from .assets import verify_assets

    asset_manifest = verify_assets(config["model"], verify=args.check_model)
    from lambda0.training.runner import resolve_experiment

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
    # 官方 resolver 只接受文件配置；写在独立输出的同级，避免 nonempty-output 误判。
    if not args.check_model and int(os.environ.get("RANK", "0")) == 0:
        validate_output_directory(args, config["output"])
    # none的新训准入已由上面检查；原生auto允许守护器预建的协调日志。
    resolver_config = {**config, "resume": "auto"}
    location = Path(config["output"]).parent / f".{args.run_tag}.experiment.json"
    location.parent.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        temporary = location.with_suffix(".tmp")
        temporary.write_text(json.dumps(resolver_config, indent=2))
        os.replace(temporary, location)
    cli.config = location
    saved_world = os.environ.get("WORLD_SIZE")
    if args.dry_run or args.check_model:
        os.environ["WORLD_SIZE"] = str(len(args.gpus.split(",")))
    try:
        resolved = resolve_experiment(cli)
    finally:
        if args.dry_run or args.check_model:
            if saved_world is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = saved_world
    resolved["rsci"] = dict(
        variant=args.variant,
        effective_use_proprio=args.variant == "STATE",
        effective_state_conditioning_mode="discrete_vlm"
        if args.variant == "STATE"
        else "continuous_action_expert",
        history_offsets=[-6, -4, -2, 0] if args.variant == "VGQ-AE" else [0],
        normalization="rsci_q01_q99_degenerate_minmax_v1",
        initialization_weight_source="raw",
        inference="current-frame only; native Lambda0 generation; no Reader/Query",
    )
    from .optimization import effective_arguments, native_arguments

    resolved["native_training"] = effective_arguments(resolved["training"], args.variant)
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    from lambda0.training import engine

    from .runtime import RSCITrainingBackend

    # 绑定模型配方、数据统计和初始化，避免同名输出误接其他实验。
    stable = {key: value for key, value in config.items() if key != "resume"}
    signature = dict(
        contract=CONTRACT,
        upstream=UPSTREAM_COMMIT,
        variant=args.variant,
        config=stable,
        initialization=asset_manifest,
        native_training=resolved["native_training"],
        bundle={
            p.relative_to(Path(config["data"])).as_posix(): sha256(p)
            for p in sorted(Path(config["data"]).rglob("*.json"))
        },
    )
    signature = reuse_run_contract(signature, config["output"], "rsci_run.json")
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    contract = dict(schema=CONTRACT, identity=identity, variant=args.variant)
    backend = RSCITrainingBackend(engine, args.variant, contract)
    settings = resolved["training"]
    native_args = native_arguments(settings)
    if args.check_model:
        import torch

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
        if args.variant == "VGQ-AE":
            assert len(prepared["history"]) == 3 and not prepared["proprio_valid_mask"].any()
        reload_exact = None
        if args.variant == "VGQ-AE":
            # 不复制5GB基础权重：用只读symlink加新增模块小分片，验真实HF构造/严格回载。
            import tempfile

            from safetensors import safe_open
            from safetensors.torch import save_file

            from .model import Lambda0RSCI

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
                restored = Lambda0RSCI.from_pretrained(
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
    Path(native_args.output_dir).mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        provenance = Path(native_args.output_dir) / "rsci_run.json"
        if provenance.is_file() and json.loads(provenance.read_text()) != signature:
            raise ValueError(
                "Existing run differs from current config/assets; choose another run-tag"
            )
        provenance.write_text(json.dumps(signature, indent=2))
    engine.main(native_args, backend=backend)


def main():
    args = build_argument_parser().parse_args()
    config = build_training_config(args)
    if args.check:
        print(
            json.dumps(
                dict(
                    experiment=config,
                    checkpoint_exists=Path(config["model"]).is_dir(),
                    bundle_exists=Path(config["data"]).is_dir(),
                ),
                indent=2,
            )
        )
    elif args.check_data:
        validate_training_data(args, config)
    elif args.dry_run or args.worker or args.check_model:
        execute_training(args, config)
    else:
        raise SystemExit("Use scripts/train/lambda0.sh for the guarded training launcher")


if __name__ == "__main__":
    main()
