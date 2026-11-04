"""以完整30帧专家窗口分析已有预测，检验固定时移解释，不调用模型。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.rhsci.processing import Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = []
    for path in args.predictions:
        content = path.read_bytes()
        saved = json.loads(content)
        root = Path(saved["dataset"])
        info = json.loads((root / "meta/info.json").read_text())
        processor = Processor(
            root,
            args.tokenizer,
            saved["config"]["variant"],
            normalization=saved["config"]["normalization"],
        )
        lo, hi = [np.asarray(processor.stats["action"][k], np.float32) for k in ("q01", "q99")]
        scale, offset = (hi - lo + 1e-6) / 2, lo + (hi - lo + 1e-6) / 2
        experts = {}
        for ep in saved["episode_results"]:
            episode = ep["episode"]
            parquet = root / info["data_path"].format(
                episode_index=episode,
                episode_chunk=episode // info.get("chunks_size", 1000),
            )
            raw = np.asarray(pq.read_table(parquet, columns=["action"])["action"].to_pylist())
            expert = processor.normalize(raw, "action") * scale + offset
            reference = np.asarray(ep["expert_actions_v2"])
            windows = np.stack([expert[f : f + 30] for f in ep["frames"]])
            if not np.allclose(windows, reference, atol=1e-7, rtol=0):
                raise ValueError("Expert reconstruction differs from saved labels")
            experts[episode] = expert[:, 14:28]
        # 固定时移候选±30，只使用30/60/90输入，避免负索引截断伪造零时移。
        shifts = np.arange(-30, 31)
        rows = []
        for row in saved["rows"]:
            frame, episode = row["frame"], row["episode"]
            if frame == 0:
                continue
            pred = np.asarray(row["predictions"])[:, 14:28]
            candidates = np.stack([experts[episode][frame + s : frame + s + 30] for s in shifts])
            mse = np.mean((candidates - pred) ** 2, axis=(1, 2))
            best = int(np.argmin(mse))
            rows.append(
                dict(
                    episode=episode,
                    frame=frame,
                    seed=row["seed"],
                    fraction=row["fraction"],
                    best_shift_frames=int(shifts[best]),
                    rmse_at_correct_frame_rad=float(np.sqrt(mse[30])),
                    best_window_rmse_rad=float(np.sqrt(mse[best])),
                    mse_by_shift=mse.tolist(),
                    search_edge=best in (0, len(shifts) - 1),
                )
            )
        episodes = sorted(experts)
        fit, heldout = episodes[::2], episodes[1::2]
        conditions = []
        for fraction in sorted({r["fraction"] for r in rows}, reverse=True):
            selected = [r for r in rows if r["fraction"] == fraction]
            train = np.asarray([r["mse_by_shift"] for r in selected if r["episode"] in fit])
            test = np.asarray([r["mse_by_shift"] for r in selected if r["episode"] in heldout])
            index = int(np.argmin(train.mean(0)))
            conditions.append(
                dict(
                    fraction=fraction,
                    fit_episodes=fit,
                    heldout_episodes=heldout,
                    fit_global_shift_frames=int(shifts[index]),
                    heldout_correct_frame_rmse_rad=float(np.sqrt(test[:, 30].mean())),
                    heldout_global_shift_rmse_rad=float(np.sqrt(test[:, index].mean())),
                    heldout_individual_best_shift_rmse_rad=float(np.sqrt(test.min(1).mean())),
                    all_individual_shift_median_frames=float(
                        np.median([r["best_shift_frames"] for r in selected])
                    ),
                    all_individual_shift_min_max_frames=[
                        min(r["best_shift_frames"] for r in selected),
                        max(r["best_shift_frames"] for r in selected),
                    ],
                    search_edge_count=sum(r["search_edge"] for r in selected),
                    count=len(selected),
                )
            )
        reports.append(
            dict(
                input=str(path),
                input_sha256=hashlib.sha256(content).hexdigest(),
                variant=saved["config"]["variant"],
                rows=rows,
                conditions=conditions,
            )
        )
    report = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reports=reports,
        limitation="Descriptive nearest expert 30-frame action-window matching, not model internal "
        "phase. Search limited to +/-30 frames. Five training episodes fit a global shift; five "
        "other training episodes evaluate it, not a held-out policy-success test. Seeds share "
        "episodes. Individual best shifts are an optimistic oracle. No model, simulation, "
        "parameter change, or deployment time-index change.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    for report in reports:
        print(json.dumps({"variant": report["variant"], "conditions": report["conditions"]}))


if __name__ == "__main__":
    main()
