"""Serve RHSCI π0.5 or GR00T through the SIMPLE /act protocol."""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from rsci.inference.checkpoints import BACKBONES, ROOT, discover, inspect_run
from rsci.inference.state_noise import RequestStateNoise, add_arguments


def create_app(server):
    """为策略对象构造 HTTP 应用；请求处理沿用该策略的动作预测与异常检查。"""
    from fastapi import FastAPI, HTTPException

    app = FastAPI()
    lock = threading.Lock()

    @app.post("/act")
    def act(payload: dict):
        try:
            with lock:
                return server.predict_action(payload)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def build_argument_parser():
    """构造命令行解析器并读取环境默认值；返回解析器，不加载模型或启动训练。"""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--backbone", choices=BACKBONES, required=True)
    source = argument_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--discover", type=Path, metavar="RUNS_ROOT")
    argument_parser.add_argument("--checkpoint-step", type=int)
    argument_parser.add_argument(
        "--training-source-root",
        type=Path,
        help="π0.5: verify historical run source hashes against its original training checkout",
    )
    argument_parser.add_argument(
        "--dataset", type=Path, help="π0.5 training dataset for its processor"
    )
    argument_parser.add_argument("--tokenizer", type=Path, help="π0.5 training SentencePiece model")
    argument_parser.add_argument("--device", default="cuda:0")
    argument_parser.add_argument("--host", default="127.0.0.1")
    argument_parser.add_argument("--port", type=int, default=5555)
    add_arguments(argument_parser)
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
        from rsci.inference.action_noise import RequestActionNoise, configure_torch

        RequestActionNoise(True, args.eval_seed)
        configure_torch()
    noise = RequestStateNoise(args.state_inference_measurement_noise_std_rad, args.state_noise_seed)
    if noise.scale > 0 and args.backbone != "pi05":
        cli.error("Raw measured-joint inference noise is supported for PSI0 and π0.5 STATE only")
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
    if args.backbone == "pi05" and (args.dataset is None or args.tokenizer is None):
        cli.error("π0.5 requires --dataset and --tokenizer")
    contract, checkpoint = inspect_run(
        args.run_dir,
        args.backbone,
        step=args.checkpoint_step,
        verify_hash=False,
        training_source_root=args.training_source_root,
    )
    if args.backbone == "pi05":
        from openpi.rsci.serve import Server

        server = Server(
            checkpoint,
            contract,
            args.dataset,
            args.tokenizer,
            args.device,
            state_inference_measurement_noise_std_rad=noise.scale,
            state_noise_seed=noise.seed,
            deterministic_eval=args.deterministic_eval,
            eval_seed=args.eval_seed,
        )
    else:
        sys.path.insert(0, str(ROOT / "src/gr00t"))
        from gr00t.rsci.serve import build_server

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
