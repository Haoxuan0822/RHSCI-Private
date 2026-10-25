"""只读比对已训练任务的raw状态来源与两种RGB；图像不同不自动判为几何错误。"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

GROUPS = ("observation.leg_joints", "observation.arm_joints", "observation.hand_joints")


def state(path, columns):
    values = pq.read_table(path, columns=list(columns)).to_pydict()
    return np.concatenate([np.stack(values[key]) for key in columns], axis=1)


def archive_video(archive, raw_path):
    """以完整数据集后缀定位视频，兼容水龙头嵌套ZIP，不按episode编号跨批猜测。"""
    raw_relative = str(raw_path).split("/extracted/", 1)[1]
    prefix = raw_relative.split("/data/chunk-", 1)[0]
    suffix = f"{prefix}/videos/chunk-000/observation.images.ego_view/{raw_path.stem}.mp4"
    found = []
    for name in archive.namelist():
        if name.endswith(suffix) or suffix.endswith(name) and name.endswith(".mp4"):
            found.append((name, archive.read(name)))
        elif name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as nested:
                found.extend(archive_video(nested, raw_path))
    return found


def frames(source, indices):
    result = {}
    with av.open(source) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                result[index] = frame.to_ndarray(format="rgb24")
            if index >= max(indices):
                break
    if set(result) != set(indices):
        raise ValueError(f"Missing video frames: {set(indices) - set(result)}")
    return result


def audit(processed, raw_search, archive_paths, samples, output):
    candidates = []
    for info in sorted(raw_search.rglob("meta/info.json")):
        meta = json.loads(info.read_text())
        if (
            "observation.base_pose" not in meta["features"]
            or "observation.state" not in meta["features"]
        ):
            continue
        for path in sorted(info.parent.parent.glob("data/chunk-*/episode_*.parquet")):
            candidates.append((path, state(path, ("observation.state",))))
    mapping = {}
    for path in sorted(processed.glob("data/chunk-*/episode_*.parquet")):
        recorded = state(path, GROUPS)
        matches = [
            raw
            for raw, array in candidates
            if len(array) > len(recorded)
            and array.shape[1:] == recorded.shape[1:]
            and np.allclose(array[1 : len(recorded) + 1], recorded, atol=1e-6, rtol=0)
        ]
        mapping[int(path.stem.split("_")[-1])] = {
            "matches": [str(p) for p in matches],
            "frames": len(recorded),
        }
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    for episode in samples:
        entry = mapping[episode]
        if len(entry["matches"]) != 1:
            checks.append({"episode": episode, "status": "raw_not_unique", **entry})
            continue
        raw_path = Path(entry["matches"][0])
        found = []
        for archive_path in archive_paths:
            with zipfile.ZipFile(archive_path) as archive:
                found.extend(
                    (str(archive_path), name, video)
                    for name, video in archive_video(archive, raw_path)
                )
        if len(found) != 1:
            raise ValueError(f"{raw_path}: video matches={len(found)}")
        archive_path, name, video = found[0]
        indices = sorted({0, entry["frames"] // 2, entry["frames"] - 1})
        raw_images = frames(io.BytesIO(video), [i + 1 for i in indices])
        meta = json.loads((processed / "meta/info.json").read_text())
        video_path = processed / meta["video_path"].format(
            episode_chunk=episode // meta.get("chunks_size", 1000),
            episode_index=episode,
            video_key="observation.images.egocentric",
        )
        images = frames(str(video_path), indices)
        canvas = Image.new("RGB", (1280, 390 * len(indices)))
        draw = ImageDraw.Draw(canvas)
        for row, index in enumerate(indices):
            canvas.paste(Image.fromarray(raw_images[index + 1]), (0, row * 390 + 30))
            canvas.paste(Image.fromarray(images[index]), (640, row * 390 + 30))
            draw.text(
                (5, row * 390 + 5), f"raw MuJoCo ep{raw_path.stem} frame{index + 1}", fill="white"
            )
            draw.text(
                (645, row * 390 + 5), f"processed Isaac ep{episode} frame{index}", fill="white"
            )
        image_path = output / f"episode_{episode:06d}.png"
        canvas.save(image_path)
        checks.append(
            {
                "episode": episode,
                "raw_parquet": str(raw_path),
                "archive": archive_path,
                "raw_video_member": name,
                "raw_video_sha256": hashlib.sha256(video).hexdigest(),
                "processed_video": str(video_path),
                "processed_video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                "processed_frames": indices,
                "raw_frames": [i + 1 for i in indices],
                "comparison": str(image_path),
                "status": "visual_review_required",
            }
        )
    report = {
        "processed": str(processed),
        "processed_episodes": len(mapping),
        "unique_state_matches": sum(len(v["matches"]) == 1 for v in mapping.values()),
        "missing": [i for i, v in mapping.items() if not v["matches"]],
        "mapping": mapping,
        "samples": checks,
        "limitation": "State matching does not certify RGB/object/base alignment. Cross-renderer MAE is not a geometry/depth accuracy measurement.",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mapping"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--raw-search", type=Path, required=True)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--sample", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.processed, args.raw_search, args.archive, args.sample, args.output)
