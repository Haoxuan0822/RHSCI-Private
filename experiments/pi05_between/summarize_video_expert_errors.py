"""对同一专家回放的PNG/视频预测核对保存的实际v2标签。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    content, reference_content = args.predictions.read_bytes(), args.reference.read_bytes()
    data, reference = json.loads(content), json.loads(reference_content)
    if data["config"]["normalization"] != reference["config"]["normalization"]:
        raise ValueError("Normalization mismatch")
    ep = next(e for e in reference["episode_results"] if e["episode"] == args.episode)
    expert = np.asarray(ep["expert_actions_v2"])[..., 14:28]
    expert_index = dict(zip(ep["frames"], expert, strict=True))
    selected = [
        r for r in data["rows"] if r["source"] == args.source and r["frame"] in expert_index
    ]
    rows, summaries = [], []
    for row in selected:
        pred = np.asarray(row["predictions"])[:, 14:28]
        error = pred - expert_index[row["frame"]]
        rows.append(
            dict(
                kind=row["kind"],
                frame=row["frame"],
                seed=row["seed"],
                arm_rmse_rad=float(np.sqrt(np.mean(error**2))),
                first_error_l2_rad=float(np.linalg.norm(error[0])),
                last_error_l2_rad=float(np.linalg.norm(error[-1])),
            )
        )
    for kind in ("png", "video"):
        for seed in sorted({r["seed"] for r in selected}):
            values = sorted(
                [r for r in selected if r["kind"] == kind and r["seed"] == seed],
                key=lambda r: r["frame"],
            )
            if [r["frame"] for r in values] != ep["frames"]:
                raise ValueError("Missing/duplicate expert frame")
            pred = np.asarray([r["predictions"] for r in values])[..., 14:28]
            jump = np.linalg.norm(pred[1:, 0] - pred[:-1, -1], axis=-1)
            inside = np.linalg.norm(np.diff(pred, axis=1), axis=-1)
            summaries.append(
                dict(
                    kind=kind,
                    seed=seed,
                    arm_rmse_rad=float(np.sqrt(np.mean((pred - expert) ** 2))),
                    boundary_l2_rad=jump.tolist(),
                    boundary_inside_ratio=float(np.median(jump) / np.median(inside)),
                )
            )
    report = dict(
        rows=rows,
        summaries=summaries,
        source=args.source,
        episode=args.episode,
        predictions_sha256=hashlib.sha256(content).hexdigest(),
        reference_sha256=hashlib.sha256(reference_content).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation="Caller must establish replay/episode identity. Expert labels "
        "are targets, not actual measured joints. Same replay frames and noise; "
        "compression is a recorded-image intervention, not a new closed-loop test "
        "or proof of a training encoder bug.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
