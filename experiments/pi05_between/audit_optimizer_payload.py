"""只读核对所选完整断点的优化器参数顺序、计步和动量；不创建优化器。"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

from openpi.rhsci.model import Pi05RHSCI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads((args.run_root / "launch_contract.json").read_text())
    checkpoint = args.run_root / "checkpoint-40000"
    config = json.loads((checkpoint / "rsci_config.json").read_text())
    source = args.run_root / "source-code/src/rsci_pi05/model.py"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != contract["code"]["src/rsci_pi05/model.py"]:
        raise ValueError("Saved training model source differs from its launch contract")
    sys.path.insert(0, str(source.parent.parent))
    archived = importlib.import_module("rsci_pi05.model")
    if Path(archived.__file__).resolve() != source.resolve():
        raise ValueError("Imported another training model")
    # 仅构造元数据；不为36亿参数分配存储或初始化真实权重。
    with torch.device("meta"):
        original = archived.Pi05RSCI(config["variant"], compile_inference=False)
        current = Pi05RHSCI(config["variant"], compile_inference=False)
    original_layout = [(n, tuple(p.shape), p.requires_grad) for n, p in original.named_parameters()]
    current_layout = [(n, tuple(p.shape), p.requires_grad) for n, p in current.named_parameters()]
    if original_layout != current_layout:
        raise ValueError("Current parameter layout differs from saved training source")
    trainable = [(name, shape) for name, shape, active in original_layout if active]
    path = checkpoint / "optimizer.pt"
    optimizer = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    identities = [index for group in optimizer["param_groups"] for index in group["params"]]
    if len(trainable) != len(identities) or set(identities) != set(optimizer["state"]):
        raise ValueError("Missing, extra or duplicated optimizer slots")
    rows = []
    for (name, shape), identity in zip(trainable, identities, strict=True):
        state = optimizer["state"][identity]
        if state["exp_avg"].shape != shape or state["exp_avg_sq"].shape != shape:
            raise ValueError(f"Optimizer shape differs from ordered parameter: {name}")
        moments = {}
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if not torch.isfinite(value).all() or (key == "exp_avg_sq" and (value < 0).any()):
                raise ValueError(f"Invalid optimizer moment: {name}/{key}")
            moments[key] = dict(
                dtype=str(value.dtype),
                nonzero_elements=int(torch.count_nonzero(value)),
                elements=value.numel(),
                l2=float(torch.linalg.vector_norm(value)),
            )
        rows.append(
            dict(name=name, shape=shape, slot=identity, step=float(state["step"]), **moments)
        )
    report = dict(
        run_root=str(args.run_root),
        variant=config["variant"],
        optimizer_bytes=path.stat().st_size,
        saved_model_source_sha256=source_sha,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        actual_training_parameter_order_matches_current=True,
        trainable_parameter_tensors=len(rows),
        trainable_elements=sum(row["exp_avg"]["elements"] for row in rows),
        all_step_40000=all(row["step"] == 40000 for row in rows),
        all_moment_tensors_finite=True,
        zero_first_moment_parameters=[
            r["name"] for r in rows if not r["exp_avg"]["nonzero_elements"]
        ],
        zero_second_moment_parameters=[
            r["name"] for r in rows if not r["exp_avg_sq"]["nonzero_elements"]
        ],
        param_groups=[
            {k: v for k, v in group.items() if k != "params"} for group in optimizer["param_groups"]
        ],
        rows=rows,
        limitation="Final local optimizer payload only. Shapes and saved source order cannot "
        "detect a past permutation of same-shaped slots or reconstruct intermediate updates. "
        "Nonzero moments/steps do not prove every update was useful or that training succeeded. "
        "No optimizer creation, stepping, full model allocation, GPU or downloads.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
