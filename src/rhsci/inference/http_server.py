"""Serve RHSCI π0.5 or GR00T through the SIMPLE /act protocol."""

import argparse
import json
import os
from pathlib import Path

from rhsci.inference.checkpoints import (
    BACKBONES,
    discover,
    inspect_run,
    validate_evaluation_variant,
)
from rhsci.inference.http_app import ActionPolicy, create_policy_app


def create_app(server: ActionPolicy):
    """为策略对象构造 HTTP 应用；请求处理沿用该策略的动作预测与异常检查。"""
    return create_policy_app(server.predict_action)


def build_argument_parser():
    """构造命令行解析器并读取环境默认值；返回解析器，不加载模型或启动训练。"""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--backbone", choices=BACKBONES, required=True)
    source = argument_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--discover", type=Path, metavar="RUNS_ROOT")
    argument_parser.add_argument("--checkpoint-step", type=int)
    argument_parser.add_argument(
        "--expected-variant", help="Require this checkpoint method for evaluation"
    )
    argument_parser.add_argument(
        "--training-source-root",
        type=Path,
        help="π0.5: verify historical run source hashes against its original training checkout",
    )
    argument_parser.add_argument(
        "--dataset",
        type=Path,
        help="Single-task π0.5 training dataset for its processor; joint models use saved statistics",
    )
    argument_parser.add_argument("--tokenizer", type=Path, help="π0.5 training SentencePiece model")
    argument_parser.add_argument("--device", default="cuda:0")
    argument_parser.add_argument("--host", default="127.0.0.1")
    argument_parser.add_argument("--port", type=int, default=5555)
    argument_parser.add_argument(
        "--deterministic-eval", action=argparse.BooleanOptionalAction, default=False
    )
    argument_parser.add_argument("--eval-seed", type=int, default=0)
    return argument_parser


def main():
    cli = build_argument_parser()
    args = cli.parse_args()
    if args.deterministic_eval:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        from rhsci.inference.action_noise import RequestActionNoise, configure_torch

        RequestActionNoise(True, args.eval_seed)
        configure_torch()
    if args.discover:
        print(
            json.dumps(
                discover(
                    args.discover, args.backbone, training_source_root=args.training_source_root
                ),
                indent=2,
            )
        )
        return
    contract, checkpoint = inspect_run(
        args.run_dir,
        args.backbone,
        step=args.checkpoint_step,
        verify_hash=False,
        training_source_root=args.training_source_root,
    )
    if args.backbone == "pi05":
        if args.tokenizer is None:
            cli.error("π0.5 requires --tokenizer")
        if args.dataset is None and not contract.get("multitask", {}).get("statistics"):
            cli.error("Single-task π0.5 requires --dataset")
    saved = json.loads(
        (
            checkpoint / ("rsci_config.json" if args.backbone == "pi05" else "config.json")
        ).read_text()
    )
    actual = saved["variant"] if args.backbone == "pi05" else saved.get("rsci_variant", "STATE")
    validate_evaluation_variant(args.expected_variant, actual)
    validate_evaluation_variant(contract["model"], actual)
    if args.backbone == "pi05":
        from openpi.rhsci.serve import Server

        server = Server(
            checkpoint,
            contract,
            args.dataset,
            args.tokenizer,
            args.device,
            deterministic_eval=args.deterministic_eval,
            eval_seed=args.eval_seed,
        )
    else:
        from gr00t.rhsci.serve import build_server

        server = build_server(
            checkpoint,
            args.device,
            deterministic_eval=args.deterministic_eval,
            eval_seed=args.eval_seed,
        )
    import uvicorn

    uvicorn.run(create_app(server), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
