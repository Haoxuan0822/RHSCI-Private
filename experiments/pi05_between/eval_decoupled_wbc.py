"""Thin CLI for official SIMPLE evaluation with per-process renderer and DDS placement."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_id")
    parser.add_argument("policy")
    parser.add_argument("split", nargs="?", default="train")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data-format", choices=["lerobot"], default="lerobot")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sim-mode", choices=["mujoco_isaac"], default="mujoco_isaac")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--success-criteria", type=float, default=0.4)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--renderer-gpu", type=int, required=True)
    parser.add_argument("--dds-domain-id", type=int, required=True)
    args = parser.parse_args()
    # 保留材质、纹理和 USD 引用警告，便于本次资源恢复验收。
    os.environ.setdefault("SIMPLE_ISAAC_LOG_LEVEL", "warning")
    # Isaac Sim consumes these native Kit arguments after our own CLI has parsed.
    sys.argv = [
        sys.argv[0],
        f"--/renderer/activeGpu={args.renderer_gpu}",
        "--/renderer/multiGpu/enabled=False",
    ]
    from simple.cli.eval_decoupled_wbc import _make_sonic_config, run_eval
    from simple.evals.api import EvalConfig

    sonic = _make_sonic_config()
    sonic["DOMAIN_ID"] = args.dds_domain_id
    sonic["INTERFACE"] = "lo"
    config = EvalConfig(
        **{
            name: getattr(args, name)
            for name in (
                "env_id",
                "policy",
                "split",
                "host",
                "port",
                "data_format",
                "data_dir",
                "sim_mode",
                "headless",
                "num_episodes",
                "episode_start",
                "max_episode_steps",
                "success_criteria",
                "eval_dir",
                "save_video",
            )
        },
        num_workers=1,
    )
    result = run_eval(config, sonic_config=sonic, show_progress=False)
    (Path.cwd() / "result.json").write_text(
        json.dumps(
            {
                "stats": result.stats,
                "success_rate": result.success_rate,
                "episodes": len(result.stats),
                "log_path": result.log_path,
            },
            indent=2,
        )
        + "\n"
    )
    if len(result.stats) != args.num_episodes:
        raise RuntimeError(f"Incomplete evaluation: {len(result.stats)}/{args.num_episodes}")
    print("CYCLE_COMPLETE", result.stats, flush=True)


if __name__ == "__main__":
    main()
