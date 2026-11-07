"""将已解析的 RHSCI 配置接入 λ₀ 原生 resolver 与训练引擎。"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from rhsci.data.ardy_motion import motion_config
from rhsci.data.visual_history import visual_history_offsets
from rhsci.runtime.contracts import compute_contract_digest, reuse_run_contract
from rhsci.runtime.environment import get_env

from .config import LambdaTrainingConfig
from .contracts import CONTRACT, UPSTREAM_COMMIT, sha256
from .preflight import run_model_preflight, validate_output_directory


def execute_training(args: argparse.Namespace, config: LambdaTrainingConfig) -> None:
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
    if not args.check_model and int(get_env("RANK", "0")) == 0:
        validate_output_directory(args, config["output"])
    # none的新训准入已由上面检查；原生auto允许守护器预建的协调日志。
    resolver_config = {**config, "resume": "auto"}
    location = Path(config["output"]).parent / f".{args.run_tag}.experiment.json"
    location.parent.mkdir(parents=True, exist_ok=True)
    if int(get_env("RANK", "0")) == 0:
        temporary = location.with_suffix(".tmp")
        temporary.write_text(json.dumps(resolver_config, indent=2))
        os.replace(temporary, location)
    cli.config = location
    saved_world = get_env("WORLD_SIZE")
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
        history_offsets=list(visual_history_offsets(args.variant))
        if args.variant in ("VGQ-AE", "VGQ-AE-240", "VGQ-AE-480", "VGQ-AE-Mo", "VGQ-AE-Mo-240", "VGQ-AE-Mo-480") else [0],
        normalization="rsci_q01_q99_degenerate_minmax_v1",
        initialization_weight_source="raw",
        inference="current-frame only; native Lambda0 generation; no Reader/Query",
    )
    from .optimization import effective_arguments, native_arguments

    resolved["native_training"] = effective_arguments(resolved["training"], args.variant)
    resolved["rsci"].update(motion_config(args))
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    from lambda0.training import engine

    from .runtime import RHSCITrainingBackend

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
    signature.update(motion_config(args))
    signature = reuse_run_contract(signature, config["output"], "rsci_run.json")
    identity = compute_contract_digest(signature)
    contract = dict(schema=CONTRACT, identity=identity, variant=args.variant)
    backend = RHSCITrainingBackend(engine, args.variant, contract, **motion_config(args))
    settings = resolved["training"]
    native_args = native_arguments(settings)
    if args.check_model:
        run_model_preflight(args, config, backend, settings, native_args)
        return
    Path(native_args.output_dir).mkdir(parents=True, exist_ok=True)
    if int(get_env("RANK", "0")) == 0:
        provenance = Path(native_args.output_dir) / "rsci_run.json"
        if provenance.is_file() and json.loads(provenance.read_text()) != signature:
            raise ValueError(
                "Existing run differs from current config/assets; choose another run-tag"
            )
        provenance.write_text(json.dumps(signature, indent=2))
    engine.main(native_args, backend=backend)
