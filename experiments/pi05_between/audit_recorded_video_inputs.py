"""在历史/当前已录视频上离线推理，另配对原PNG量化压缩影响。"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--video", nargs="+", required=True, help="label=path")
    parser.add_argument("--png-trace", nargs="*", default=[], help="matching-video-label=trace")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max-frame", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    if saved["variant"] not in ("VGQ-AE", "NO-STATE"):
        raise ValueError("Image-only variants required")
    if torch.cuda.mem_get_info()[0] < 16 * 1024**3:
        raise RuntimeError("Need an isolated GPU with 16 GiB free")
    processor = Processor(
        args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
    )
    model = Pi05RHSCI(saved["variant"], compile_inference=False)
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.cuda().eval()
    low, high = [np.asarray(processor.stats["action"][k], np.float32) for k in ("q01", "q99")]
    scale, offset = (high - low + 1e-6) / 2, low + (high - low + 1e-6) / 2
    if args.max_frame < 30:
        raise ValueError("Need at least one complete 30-frame interval")
    frames = sorted(set(range(0, args.max_frame + 1, 24)) | set(range(0, args.max_frame + 1, 30)))
    inputs, provenance = [], []
    for item in args.video:
        label, value = item.split("=", 1)
        path = Path(value)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 50) > 1e-6:
            raise ValueError("Need an open 50Hz recorded video")
        for frame in range(max(frames) + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Missing frame {frame} in {path}")
            if frame in frames:
                inputs.append((label, "video", frame, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        cap.release()
        provenance.append(
            dict(
                source=label,
                kind="video",
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    for item in args.png_trace:
        label, value = item.split("=", 1)
        path = Path(value)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] != "model_chunk" or row["step"] not in (0, 30, 60, 90):
                continue
            if row["step"] > max(frames):
                continue
            if row["instruction"] != args.instruction:
                raise ValueError("Instruction mismatch")
            image_path = Path(row["image_path"])
            inputs.append((label, "png", row["step"], np.asarray(Image.open(image_path))))
        provenance.append(
            dict(
                source=label,
                kind="png-trace",
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    rows = []
    for label, kind, frame, image in inputs:
        # 图像策略不读取state；占位不冒称从历史视频恢复了测量状态。
        sample = dict(
            images=image[None],
            state=np.zeros(32, np.float32),
            actions=np.zeros((30, 36), np.float32),
            text=args.instruction,
            rsci_history_valid=np.ones(4, bool),
            rsci_ardy_target=np.zeros(128),
            rsci_ardy_valid=False,
        )
        batch = processor(sample)
        values = [batch[k][None].cuda() for k in ("images", "tokens", "token_mask")]
        for seed in (17, 29, 43):
            noise = torch.randn(
                (1, 30, 36), device="cuda", generator=torch.Generator("cuda").manual_seed(seed)
            )
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
            ):
                pred = model.sample_actions(*values, num_steps=10, noise=noise)
            physical = pred[0].float().cpu().numpy() * scale + offset
            rows.append(
                dict(
                    source=label,
                    kind=kind,
                    frame=frame,
                    seed=seed,
                    rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    predictions=physical.tolist(),
                )
            )
        print(json.dumps(dict(source=label, kind=kind, frame=frame)), flush=True)
    summaries = []
    for label in sorted({r["source"] for r in rows}):
        for horizon in (24, 30):
            for seed in (17, 29, 43):
                selected = sorted(
                    [
                        r
                        for r in rows
                        if r["source"] == label
                        and r["kind"] == "video"
                        and r["seed"] == seed
                        and r["frame"] % horizon == 0
                    ],
                    key=lambda r: r["frame"],
                )
                p = np.asarray([r["predictions"] for r in selected])[:, :horizon, 14:28]
                boundary = np.linalg.norm(p[1:, 0] - p[:-1, -1], axis=-1)
                inside = np.linalg.norm(np.diff(p, axis=1), axis=-1)
                summaries.append(
                    dict(
                        source=label,
                        horizon=horizon,
                        seed=seed,
                        frames=[r["frame"] for r in selected],
                        boundary_l2_rad=boundary.tolist(),
                        median_boundary_l2_rad=float(np.median(boundary)),
                        boundary_inside_ratio=float(np.median(boundary) / np.median(inside)),
                    )
                )
    compression = []
    index = {(r["source"], r["kind"], r["frame"], r["seed"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate source/kind/frame/seed identity")
    for row in rows:
        if row["kind"] == "png":
            video = index[row["source"], "video", row["frame"], row["seed"]]
            delta = (np.asarray(row["predictions"]) - np.asarray(video["predictions"]))[:, 14:28]
            compression.append(
                dict(
                    source=row["source"],
                    frame=row["frame"],
                    seed=row["seed"],
                    arm_rmse_rad=float(np.sqrt(np.mean(delta**2))),
                    max_abs_arm_difference_rad=float(np.abs(delta).max()),
                )
            )
    with args.output.open("x") as stream:
        json.dump(
            dict(
                config=saved,
                checkpoint=str(args.checkpoint),
                rows=rows,
                summaries=summaries,
                compression_pairs=compression,
                provenance=provenance,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitation="Recorded paths are not generated by these predictions. "
                "No joint ground truth recovered from historical video. Sources are not "
                "paired physical states or noises. 24/30 grids are offline diagnostics, "
                "not new rollouts or success-rate tests. H264 compression may affect "
                "predictions; current-video PNG pairs only bound their own effect.",
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print(json.dumps(dict(summaries=summaries, compression_pairs=compression)))


if __name__ == "__main__":
    main()
