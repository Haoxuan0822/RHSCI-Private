"""逐张量比较冻结VLM或共享动作参数；CPU mmap，不构造模型或更改权重。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("frozen-vlm", "common-action"), default="frozen-vlm")
    args = parser.parse_args()
    if len(args.checkpoints) < 2 or args.output.exists():
        raise ValueError("Need multiple checkpoints and a fresh output path")
    prefixes = (
        ("paligemma_with_expert.paligemma.",)
        if args.scope == "frozen-vlm"
        else (
            "paligemma_with_expert.gemma_expert.",
            "action_in_proj.",
            "action_out_proj.",
            "time_mlp_in.",
            "time_mlp_out.",
        )
    )
    comparisons = []
    with safe_open(
        args.checkpoints[0] / "model.safetensors", framework="pt", device="cpu"
    ) as reference:
        names = sorted(name for name in reference.keys() if name.startswith(prefixes))
        if not names:
            raise ValueError("No selected tensors")
        for checkpoint in args.checkpoints[1:]:
            differences, elements = [], 0
            with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as other:
                if names != sorted(name for name in other.keys() if name.startswith(prefixes)):
                    raise ValueError("Selected tensor inventory differs")
                for name in names:
                    left, right = reference.get_tensor(name), other.get_tensor(name)
                    elements += left.numel()
                    if left.dtype != right.dtype or not torch.equal(left, right):
                        differences.append(name)
            comparisons.append(
                dict(
                    checkpoint=str(checkpoint),
                    tensors=len(names),
                    elements=elements,
                    bitwise_equal=not differences,
                    different_tensors=differences,
                )
            )
    report = dict(
        reference=str(args.checkpoints[0]),
        scope=args.scope,
        comparisons=comparisons,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="Selected actual tensors only; this does not establish correct pretrained "
        "initialization, training provenance, conditioning, or visual representation quality. "
        "common-action excludes auxiliary Reader/reconstruction parameters.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
