"""Launch a native baseline or an RHSCI adapter in its own Python environment.

Options before `--` configure the launcher; arguments after it go unchanged to
the model's trainer. --dry-run prints the command without importing ML packages.
"""

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKBONES = (
    "psi0",
    "pi05",
    "gr00t",
    "egovla",
    "lambda0",
    "act",
    "dp",
    "internvla-m1",
    "h-rdt",
    "dreamzero",
    "beingh05",
)
ADAPTED = ("psi0", "pi05", "gr00t", "egovla", "lambda0", "act")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backbone", choices=BACKBONES, required=True)
    result.add_argument("--mode", choices=("native", "rsci"), default="native")
    result.add_argument(
        "--python", default=sys.executable, help="Interpreter in the selected baseline environment"
    )
    result.add_argument(
        "--gpus", help="Visible GPU indices; otherwise inherit CUDA_VISIBLE_DEVICES"
    )
    result.add_argument(
        "--nproc-per-node", type=int, default=None, help="Default: number of selected GPUs, or 1"
    )
    result.add_argument("--nnodes", type=int, default=1)
    result.add_argument("--node-rank", type=int, default=0)
    result.add_argument("--master-addr", default="127.0.0.1")
    result.add_argument(
        "--master-port", type=int, help="Rendezvous port; single-node default uses a free port"
    )
    result.add_argument(
        "--config",
        help="Native config module/name/path (required for PSI0, ACT, DP, InternVLA and lambda0)",
    )
    result.add_argument(
        "--args-file",
        type=Path,
        help="JSON array of native trainer arguments; identity and paths can follow --",
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "trainer_args", nargs=argparse.REMAINDER, help="Native trainer arguments after --"
    )
    return result


def plan(args):
    forwarded = args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args
    if args.args_file:
        if args.mode != "native":
            raise ValueError("--args-file is for native trainers")
        defaults = json.loads(args.args_file.read_text())
        if not isinstance(defaults, list) or not all(isinstance(value, str) for value in defaults):
            raise ValueError("--args-file must contain a JSON array of strings")
        forwarded = defaults + forwarded
    devices = args.gpus if args.gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        devices is not None
        and devices != ""
        and (
            not re.fullmatch(r"\d+(,\d+)*", devices)
            or len(set(devices.split(","))) != len(devices.split(","))
        )
    ):
        raise ValueError("GPU indices must be distinct comma-separated integers")
    processes = (
        args.nproc_per_node
        if args.nproc_per_node is not None
        else len(devices.split(","))
        if devices
        else 1
    )
    if processes < 1 or args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes:
        raise ValueError("Invalid process count or node rank")
    if args.master_port is not None and not 1024 <= args.master_port <= 65535:
        raise ValueError("master-port must be between 1024 and 65535")
    if devices and processes > len(devices.split(",")):
        raise ValueError("Process count exceeds visible GPUs")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    source_paths = [ROOT / "src"]
    cwd = ROOT
    if args.mode == "rsci":
        if args.backbone not in ADAPTED:
            raise ValueError(f"{args.backbone} has no RHSCI adapter; use --mode native")
        if (
            args.config
            or args.nnodes != 1
            or args.node_rank != 0
            or args.nproc_per_node is not None
            or args.master_port is not None
            or args.master_addr != "127.0.0.1"
        ):
            raise ValueError(
                "RHSCI adapters manage their own processes; pass their options after --"
            )
        variable = dict(
            psi0="PSI0_PYTHON",
            pi05="PI05_RSCI_PYTHON",
            gr00t="GROOT_RSCI_PYTHON",
            egovla="EGOVLA_PYTHON",
            lambda0="LAMBDA0_PYTHON",
            act="ACT_PYTHON",
        )[args.backbone]
        env[variable] = args.python
        if args.gpus is not None:
            forwarded = ["--gpus", args.gpus, *forwarded]
        script = "act_rsci" if args.backbone == "act" else args.backbone
        command = ["bash", str(ROOT / f"scripts/train/{script}.sh"), *forwarded]
    else:
        backbone = args.backbone
        if backbone in ("psi0", "act", "dp"):
            if not args.config:
                raise ValueError("--config must select a psi.config.train module")
            allowed = {
                "act": ("simple_act_config", "real_act_config"),
                "dp": ("simple_dp_config", "real_dp_config"),
            }
            if backbone in allowed and args.config not in allowed[backbone]:
                raise ValueError(f"{backbone} requires one of {allowed[backbone]}")
            entry = [str(ROOT / "scripts/train.py"), args.config]
            env["PSI_RSCI_TRAIN"] = "1"
        elif backbone == "pi05":
            if not args.config:
                raise ValueError("--config must select an openpi.training.config entry")
            entry = [str(ROOT / "src/openpi/train_pytorch.py"), args.config]
        elif backbone == "gr00t":
            source_paths.insert(0, ROOT / "src/gr00t")
            entry = [
                "--module",
                "gr00t.experiment.launch_finetune",
                "--num-gpus",
                str(processes * args.nnodes),
            ]
        elif backbone == "egovla":
            source_paths[:0] = [ROOT / "src/egovla", ROOT / "src/egovla/VILA"]
            cwd = ROOT / "src/egovla"
            entry = [str(cwd / "human_plan/vila_train/train.py")]
        elif backbone == "internvla-m1":
            if not args.config:
                raise ValueError("--config must name an InternVLA training YAML")
            cwd = ROOT / "src/InternVLA-M1"
            source_paths.insert(0, cwd)
            entry = [
                str(cwd / "InternVLA/training/train_internvla.py"),
                "--config_yaml",
                str(Path(args.config).resolve()),
            ]
        elif backbone == "h-rdt":
            cwd = ROOT / "src/h_rdt"
            source_paths.insert(0, cwd)
            entry = [str(cwd / "main.py")]
            if args.config:
                entry += ["--config_path", str(Path(args.config).resolve())]
        elif backbone == "dreamzero":
            source_paths.insert(0, ROOT / "src/dreamzero")
            entry = [str(ROOT / "src/dreamzero/groot/vla/experiment/experiment.py")]
            if args.config:
                entry += [
                    "--config-path",
                    str(Path(args.config).resolve().parent),
                    "--config-name",
                    Path(args.config).stem,
                ]
        elif backbone == "beingh05":
            cwd = ROOT / "src/beingh05"
            source_paths.insert(0, cwd)
            entry = [str(cwd / "BeingH/train/train.py")]
            if args.config:
                entry += ["--dataset_config_file", str(Path(args.config).resolve())]
        else:
            if not args.config:
                raise ValueError("--config must name a lambda0 experiment JSON")
            source_paths.insert(0, ROOT / "src/openlambda/lambda0")
            entry = ["--module", "lambda0", "train", "--config", str(Path(args.config).resolve())]
        if args.config and backbone in ("gr00t", "egovla"):
            raise ValueError("This native trainer takes its configuration as arguments after --")
        topology = (
            ["--standalone"]
            if args.nnodes == 1 and args.master_port is None
            else [
                f"--node-rank={args.node_rank}",
                f"--master-addr={args.master_addr}",
                f"--master-port={args.master_port or 29500}",
            ]
        )
        command = [
            args.python,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={processes}",
            f"--nnodes={args.nnodes}",
            *topology,
            *entry,
            *forwarded,
        ]
    env["PYTHONPATH"] = os.pathsep.join(map(str, source_paths)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return command, cwd, env


def main():
    options = parser()
    args = options.parse_args()
    try:
        command, cwd, env = plan(args)
    except ValueError as error:
        options.error(str(error))
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    command=command,
                    shell=shlex.join(command),
                    cwd=str(cwd),
                    environment={
                        name: env[name]
                        for name in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")
                        if name in env
                    },
                ),
                indent=2,
            )
        )
        return
    os.chdir(cwd)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
