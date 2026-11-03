"""用各骨干现有数据预检检查真实输入；不加载全尺寸模型或启动训练。"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check_inputs(backbone, dataset, sidecar, variant, *, vlm_path=None, ae_path=None):
    """复用生产预检，禁用缓存以直接读取原视频；仅在对应骨干的环境运行。"""
    from rhsci.data.cache_config import InputCacheConfig

    paths = dict(task=dataset.name, dataset=dataset, sidecar=sidecar,
                 input_cache=InputCacheConfig())
    args = SimpleNamespace(task=dataset.name, model=variant, seed=292285, check_data=True)
    if backbone == "psi0":
        from psi.rhsci.preflight import run_training_preflight
        from psi.rhsci.training_config import build_argument_parser, resolve_training_plan
        cli = build_argument_parser().parse_args([
            "--task", dataset.name, "--model", variant, "--data-root", str(dataset.parent),
            "--sidecar-root", str(sidecar), "--vlm-path", str(vlm_path),
            "--ae-path", str(ae_path), "--num-workers", "1", "--check",
        ])
        run_training_preflight(resolve_training_plan(cli).config)
    elif backbone == "pi05":
        from openpi.rhsci.preflight import run_training_preflight
        run_training_preflight(args, paths)
    elif backbone == "gr00t":
        from gr00t.rhsci.preflight import validate_training_data
        validate_training_data(args, paths)
    elif backbone == "act":
        from act.rhsci.preflight import check_data
        check_data(args, paths)
    else:
        from human_plan.rhsci.preflight import check_data
        check_data(paths, variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("psi0", "pi05", "gr00t", "act", "egovla"), required=True)
    parser.add_argument("--data-report", type=Path, required=True,
                        help="audit_training_data.py 输出的数据/标签配对清单")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[], help="只核验指定的精确任务目录名，可重复")
    parser.add_argument("--vlm-path", type=Path, help="Ψ0 本地 VLM/处理器")
    parser.add_argument("--ae-path", type=Path, help="Ψ0 本地 Action Expert")
    args = parser.parse_args()
    if args.backbone == "psi0" and (args.vlm_path is None or args.ae_path is None):
        parser.error("psi0 requires --vlm-path and --ae-path")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    torch.set_num_threads(1)
    if args.backbone == "psi0":
        variants = ("STATE", "NO-STATE", "VGQ-AE", "VGQ-VLM", "STATE-RECONSTRUCTION")
    elif args.backbone == "egovla":
        variants = ("STATE", "VGQ-AE")
    else:
        variants = ("STATE", "NO-STATE", "EGQ-AE" if args.backbone == "act" else "VGQ-AE")
    pairs = json.loads(args.data_report.read_text())["datasets"]
    if not pairs:
        parser.error("data report contains no dataset pairs")
    missing = set(args.task) - {Path(pair["dataset"]).name for pair in pairs}
    if missing:
        parser.error(f"unknown tasks: {sorted(missing)}")
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(), backbone=args.backbone,
                  scope="existing CPU data preflight, direct source videos; may populate external HF caches; no full model/worker/GPU check",
                  selected_tasks=args.task or "all supplied pairs",
                  results=[])
    for pair in pairs:
        if args.task and Path(pair["dataset"]).name not in args.task:
            continue
        for variant in variants:
            result = dict(dataset=pair["dataset"], variant=variant,
                          content_fingerprint=pair["content_fingerprint"])
            try:
                check_inputs(args.backbone, Path(pair["dataset"]), Path(pair["sidecar"]), variant,
                             vlm_path=args.vlm_path, ae_path=args.ae_path)
                result["passed"] = True
            except Exception as error:
                result.update(passed=False, error=f"{type(error).__name__}: {error}")
                print(result["error"], flush=True)
            report["results"].append(result)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return any(not result["passed"] for result in report["results"])


if __name__ == "__main__":
    sys.exit(main())
