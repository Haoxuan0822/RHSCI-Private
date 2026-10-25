"""父终端仅跟随日志；守护持锁、后台存活，Ctrl+C不结束训练。"""

import fcntl
import os
import subprocess
import sys

from .train import parser, resolve


def main():
    args = parser().parse_args()
    paths = resolve(args)
    # 前台核对参数、路径和基础资产；守护必须通过正式数据验收后才会启动训练。
    checked = subprocess.run(
        [sys.executable, "-m", "rsci.backbones.egovla.train", *sys.argv[1:], "--dry-run"]
    )
    if checked.returncode:
        raise SystemExit(checked.returncode)
    from .train import assets

    assets(paths["checkpoint"])
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launch.lock").open("a+") as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "guardian.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
                active = False
            except BlockingIOError:
                active = True
        log_path = output / "guardian.log"
        if active:
            print(f"EgoVLA该run的守护已存在；只查看日志：{log_path}", flush=True)
        else:
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "rsci.backbones.egovla.train",
                        *sys.argv[1:],
                        "--guardian",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    env=environment,
                )
            print(f"EgoVLA guardian PID={process.pid}; log={log_path}", flush=True)
    try:
        subprocess.run(["tail", "-n", "50", "-F", str(log_path)])
    except KeyboardInterrupt:
        print("已退出日志查看；后台训练守护继续。")


if __name__ == "__main__":
    main()
