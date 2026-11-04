"""诊断：复用现有评测入口，恢复历史时钟、导航和手部适配，不改模型state。"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--historical-source", type=Path, required=True)
    parser.add_argument(
        "--historical-initial-pose",
        choices=("wbc-default-upper-v1", "legacy-zero-arms-v1"),
    )
    parser.add_argument("--initial-pose-only", action="store_true")
    args, remaining = parser.parse_known_args()
    if "--trace-control" not in remaining:
        parser.error("Historical controls are restricted to traced diagnostics")
    if args.initial_pose_only and not args.historical_initial_pose:
        parser.error("--initial-pose-only requires --historical-initial-pose")
    source = args.historical_source.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05_between"))
    import eval_decoupled_wbc
    import trace_control

    original_trace = trace_control.install_control_trace

    def install(path, **kwargs):
        # 在官方Kit导入后、agent构造前安装；避免提前导入仿真模块。
        if kwargs.get("control_clock") != "wall" or kwargs.get("legacy_navigation"):
            raise ValueError("Do not combine historical controls with other clock/nav adapters")
        from legacy_navigation import install as install_navigation

        receipt = {"model_state_substitution": False, "sources": {}}
        modules = []
        if not args.initial_pose_only:
            os.environ["PSI_EVAL_CONTROL_CLOCK"] = "sim-step-v1"
            os.environ["PSI_EVAL_WBC_HAND_ORDER"] = "1"
            modules.extend(("eval_control_clock", "eval_wbc_hand_order"))
        if args.historical_initial_pose:
            os.environ["PSI_EVAL_WBC_INITIAL_POSE"] = "1"
            os.environ["PSI_EVAL_WBC_INITIAL_POSE_PROTOCOL"] = args.historical_initial_pose
            modules.append("eval_wbc_initial_pose")
            receipt["initial_pose"] = args.historical_initial_pose
        for name in modules:
            filename = source / "scripts" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(f"historical_{name}", filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.install()
            receipt["sources"][name] = {
                "path": str(filename),
                "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                "protocol": module.PROTOCOL,
            }
        receipt["navigation"] = (
            {"protocol": "official-current"} if args.initial_pose_only else install_navigation()
        )
        (Path.cwd() / "historical-controls.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("HISTORICAL_CONTROLS", json.dumps(receipt), flush=True)
        return original_trace(path, **kwargs)

    trace_control.install_control_trace = install
    sys.argv = [sys.argv[0], *remaining]
    eval_decoupled_wbc.main()


if __name__ == "__main__":
    main()
