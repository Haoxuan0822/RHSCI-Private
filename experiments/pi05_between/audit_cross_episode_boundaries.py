"""分解已有跨轨迹预测的边界误差，不调用模型或修改控制。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = []
    for path in args.predictions:
        content = path.read_bytes()
        data = json.loads(content)
        rows, summaries = [], []
        for episode in data["episode_results"]:
            ep, frames = episode["episode"], episode["frames"]
            if frames != [0, 30, 60, 90]:
                raise ValueError("Expected four consecutive 30-frame windows")
            expert = np.asarray(episode["expert_actions_v2"])[..., 14:28]
            expert_jump = expert[1:, 0] - expert[:-1, -1]
            for condition in episode["conditions"]:
                fraction, seed = condition["fraction"], condition["seed"]
                selected = sorted(
                    [
                        r
                        for r in data["rows"]
                        if r["episode"] == ep and r["fraction"] == fraction and r["seed"] == seed
                    ],
                    key=lambda r: r["frame"],
                )
                if [r["frame"] for r in selected] != frames:
                    raise ValueError("Missing/repeated prediction")
                pred = np.asarray([r["predictions"] for r in selected])[..., 14:28]
                start_error = pred[1:, 0] - expert[1:, 0]
                end_error = pred[:-1, -1] - expert[:-1, -1]
                jump = pred[1:, 0] - pred[:-1, -1]
                error = jump - expert_jump
                if not np.allclose(error, start_error - end_error, atol=1e-12, rtol=0):
                    raise ValueError("Boundary decomposition failed")
                progress = pred[:-1, -1] - pred[:-1, 0]
                for i, frame in enumerate(frames[1:]):
                    square = float(error[i] @ error[i])
                    denominator = float(np.linalg.norm(jump[i]) * np.linalg.norm(progress[i]))
                    rows.append(
                        dict(
                            episode=ep,
                            fraction=fraction,
                            seed=seed,
                            frame=frame,
                            jump_l2_rad=float(np.linalg.norm(jump[i])),
                            expert_jump_l2_rad=float(np.linalg.norm(expert_jump[i])),
                            start_error_l2_rad=float(np.linalg.norm(start_error[i])),
                            end_error_l2_rad=float(np.linalg.norm(end_error[i])),
                            start_signed_projection=(
                                float(start_error[i] @ error[i] / square) if square else None
                            ),
                            jump_progress_cosine=(
                                float(jump[i] @ progress[i] / denominator) if denominator else None
                            ),
                        )
                    )
        for fraction in sorted({r["fraction"] for r in rows}, reverse=True):
            selected = [r for r in rows if r["fraction"] == fraction]

            def median(key):
                return float(np.median([r[key] for r in selected if r[key] is not None]))

            summaries.append(
                dict(
                    fraction=fraction,
                    count=len(selected),
                    median_jump_l2_rad=median("jump_l2_rad"),
                    median_expert_jump_l2_rad=median("expert_jump_l2_rad"),
                    median_start_error_l2_rad=median("start_error_l2_rad"),
                    median_end_error_l2_rad=median("end_error_l2_rad"),
                    median_start_signed_projection=median("start_signed_projection"),
                    start_error_larger_count=sum(
                        r["start_error_l2_rad"] > r["end_error_l2_rad"] for r in selected
                    ),
                    reversal_cos_below_minus_half_count=sum(
                        r["jump_progress_cosine"] is not None and r["jump_progress_cosine"] < -0.5
                        for r in selected
                    ),
                )
            )
        reports.append(
            dict(
                variant=data["config"]["variant"],
                input=str(path),
                input_sha256=hashlib.sha256(content).hexdigest(),
                rows=rows,
                summaries=summaries,
            )
        )
    with args.output.open("x") as stream:
        json.dump(
            dict(
                reports=reports,
                limitation="Exact finite-sample algebra on training "
                "trajectories. Signed projections are not causal percentages. Each seed "
                "shares an episode. Boundary jumps need not reverse previous progress. "
                "No closed-loop or success-rate inference.",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    for report in reports:
        print(json.dumps({"variant": report["variant"], "summaries": report["summaries"]}))


if __name__ == "__main__":
    main()
