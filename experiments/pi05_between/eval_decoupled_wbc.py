"""Thin CLI for official SIMPLE evaluation with per-process renderer and DDS placement."""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def configure_video(args, directory):
    """按完整评测轮次决定录像；兼容已启动 worker 的 repeat-NN 工作目录。"""
    match = re.fullmatch(r"repeat-(\d+)", directory.name)
    repeat = args.evaluation_round
    if repeat is None:
        repeat = int(match[1]) if match else 1
    requested_rounds = args.video_rounds
    policy = next(
        (
            parent / "video-policy.json"
            for parent in (directory, *directory.parents)
            if (parent / "video-policy.json").is_file()
        ),
        None,
    )
    # 每次新轮次启动时读取批次设置，让已启动worker无需中断即可采用新录像要求。
    if policy is not None:
        args.video_rounds = json.loads(policy.read_text())["video_rounds"]
    if repeat < 1 or not isinstance(args.video_rounds, int) or args.video_rounds < 1:
        raise ValueError("Evaluation and video rounds must be positive integers")
    requested = args.save_video
    args.save_video = requested and repeat <= args.video_rounds
    return {
        "evaluation_round": repeat,
        "requested_video_rounds": requested_rounds,
        "video_policy_file": str(policy) if policy else None,
        "video_rounds": args.video_rounds,
        "requested_save_video": requested,
        "save_video": args.save_video,
    }


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
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--evaluation-round", type=int)
    parser.add_argument("--video-rounds", type=int, default=1)
    parser.add_argument("--renderer-gpu", type=int, required=True)
    parser.add_argument("--dds-domain-id", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--repair-scene-references", action="store_true")
    parser.add_argument("--trace-control", type=Path)
    parser.add_argument("--control-clock", choices=["wall", "simulation"], default="wall")
    parser.add_argument("--diagnostic-execute-horizon", type=int, choices=[24, 30])
    parser.add_argument("--diagnostic-legacy-navigation", action="store_true")
    parser.add_argument("--trace-camera-timing", action="store_true")
    parser.add_argument("--diagnostic-noise-seed", type=int)
    parser.add_argument("--diagnostic-noise-stream")
    # simple 激活脚本会清空 PYTHONPATH；从既有公共包读取参数定义。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from rhsci.inference.measurement_noise import add_arguments, config_from_arguments

    add_arguments(parser)
    args = parser.parse_args()
    try:
        state_noise = config_from_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if state_noise is not None and args.policy != "psi0_decoupled_wbc":
        parser.error("State noise requires the psi0_decoupled_wbc HTTP client")
    if args.trace_control is not None and args.num_episodes != 1:
        parser.error("--trace-control requires --num-episodes 1 to keep step indices unambiguous")
    if args.control_clock != "wall" and args.trace_control is None:
        parser.error("--control-clock simulation is restricted to traced diagnostics")
    if args.diagnostic_execute_horizon is not None and args.trace_control is None:
        parser.error("--diagnostic-execute-horizon is restricted to traced diagnostics")
    if args.diagnostic_legacy_navigation and args.trace_control is None:
        parser.error("--diagnostic-legacy-navigation is restricted to traced diagnostics")
    if args.trace_camera_timing and args.trace_control is None:
        parser.error("--trace-camera-timing requires --trace-control")
    if (args.diagnostic_noise_seed is None) != (args.diagnostic_noise_stream is None):
        parser.error("Diagnostic noise requires both seed and stream")
    if args.diagnostic_noise_seed is not None and (
        args.trace_control is None or args.diagnostic_noise_seed < 0
    ):
        parser.error("Diagnostic noise requires single-episode tracing and a nonnegative seed")
    video = configure_video(args, Path.cwd())
    # 在 Kit 导入前记录实际配置；close() 可能提前结束 Python。
    (Path.cwd() / "effective_eval_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "video_policy": video,
                "state_noise_config": state_noise.metadata() if state_noise else {"enabled": False},
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print("VIDEO_POLICY", json.dumps(video), flush=True)
    # 保留材质、纹理和 USD 引用警告，便于本次资源恢复验收。
    os.environ.setdefault("SIMPLE_ISAAC_LOG_LEVEL", "warning")
    # Isaac Sim consumes these native Kit arguments after our own CLI has parsed.
    sys.argv = [
        sys.argv[0],
        f"--/renderer/activeGpu={args.renderer_gpu}",
        "--/renderer/multiGpu/enabled=False",
        f"--/plugins/carb.tasking.plugin/threadCount={args.cpu_threads}",
        f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={args.cpu_threads}",
    ]
    from simple.cli.eval_decoupled_wbc import _make_sonic_config, run_eval
    from simple.evals.api import EvalConfig

    if args.repair_scene_references:
        from scene_references import install_scene_reference_repair

        install_scene_reference_repair(
            args.data_dir,
            Path(os.environ["SIMPLE_HOME"]) / "data/scenes/hssd",
            Path.cwd() / "scene-reference-repair.json",
        )

    if args.diagnostic_noise_seed is not None:
        from paired_noise import install_client

        install_client(args.diagnostic_noise_seed, args.diagnostic_noise_stream)

    if args.trace_control is not None:
        from trace_control import install_control_trace

        install_control_trace(
            args.trace_control,
            control_clock=args.control_clock,
            execute_horizon=args.diagnostic_execute_horizon,
            legacy_navigation=args.diagnostic_legacy_navigation,
            camera_timing=args.trace_camera_timing,
        )

    if state_noise is not None:
        from rhsci.inference.state_noise_client import install

        install(state_noise, episode_start=args.episode_start)
        print("STATE_NOISE", json.dumps(state_noise.metadata()), flush=True)

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
