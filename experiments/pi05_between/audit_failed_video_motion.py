"""只读首轮失败视频，量化末六秒画面运动；候选不等同于失败根因。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "simple_queue"))
from monitor_goal import snapshot
from summarize_results import read_round_stats


def inspect(item):
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(item["video"])
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count < fps * 6:
        raise ValueError(f"Video too short/unreadable: {item['video']}")
    frames = []
    # 只读尾段，0.5秒一帧；解码在CPU，不接触任何评测服务。
    start = max(0, count - int(fps * 6))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for index in range(start, count):
        ok, frame = capture.read()
        if not ok:
            break
        if (index - start) % max(1, round(fps / 2)) == 0:
            frames.append(cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA))
    capture.release()
    if len(frames) < 10:
        raise ValueError(f"Incomplete tail: {item['video']}")
    gray = np.asarray([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], dtype=np.float32)
    deltas = np.abs(np.diff(gray, axis=0))
    item.update(duration_s=count / fps, tail_frames=len(frames),
                tail_mean_delta=float(deltas.mean()),
                tail_max_pair_mean_delta=float(deltas.mean(axis=(1, 2)).max()),
                tail_pixel_std_mean=float(gray.std(axis=0).mean()),
                tail_end_start_mae=float(np.abs(gray[-1] - gray[0]).mean()))
    return item, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items, groups = [], []
    for key, levels in sorted(snapshot().items()):
        group = dict(backbone=key[0], variant=key[1], task=key[2], episodes=0, failures=0)
        groups.append(group)
        for level, value in sorted(levels.items()):
            round_dir = Path(value["base"]) / "repeat-01"
            stats = read_round_stats(round_dir, 10)
            group["episodes"] += len(stats)
            group["failures"] += sum(not success for success in stats.values())
            for episode, success in sorted(stats.items()):
                if success:
                    continue
                videos = sorted(round_dir.glob(f"**/episode_{episode}/head_stereo_left_failed.mp4"))
                if not videos:
                    raise ValueError(f"Missing video: {key} {level} {episode}")
                items.append(dict(backbone=key[0], variant=key[1], task=key[2],
                                  level=level, episode=episode, video=str(videos[-1])))
    # 冻结本次清单，避免在途评测让分母漂移。
    (args.output / "inventory.json").write_text(json.dumps(dict(groups=groups, items=items), indent=2))
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, (item, frames) in enumerate(pool.map(inspect, items)):
            name = f"{item['backbone']}-{item['variant']}-{item['task']}-L{item['level']}-E{item['episode']}"
            item["id"] = name
            sheet = np.full((256, 192 * 6, 3), 245, dtype=np.uint8)
            for j, frame in enumerate(frames[:12]):
                row, col = divmod(j, 6)
                y, x = row * 128, col * 192
                sheet[y:y + 108, x:x + 192] = frame
                cv2.putText(sheet, f"-{6-j*0.5:.1f}s", (x+5, y+123), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            cv2.imwrite(str(args.output / f"{name}.jpg"), sheet)
            results.append(item)
            if index % 25 == 0:
                print(f"AUDITED {index+1}/{len(items)}", flush=True)
    (args.output / "motion.json").write_text(json.dumps(dict(groups=groups, items=results), indent=2))
    # 阈值只描述8位灰度画面变化，不能证明关节静止或任务停滞的原因。
    summary = []
    for group in groups:
        rows = [r for r in results if all(r[k] == group[k] for k in ("backbone", "variant", "task"))]
        summary.append(dict(group, low_motion=sum(r["tail_pixel_std_mean"] <= 2 and
                                                 r["tail_mean_delta"] <= 2 for r in rows)))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    candidates = [r for r in results if r["tail_pixel_std_mean"] <= 2 and r["tail_mean_delta"] <= 2]
    for offset in range(0, len(candidates), 12):
        batch = candidates[offset:offset+12]
        sheet = np.full((len(batch) * 112, 1160, 3), 245, dtype=np.uint8)
        for row, item in enumerate(batch):
            source = cv2.imread(str(args.output / (item["id"] + ".jpg")))
            cv2.putText(sheet, item["id"], (4, row*112+16), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1)
            for col, frame_index in enumerate((0, 2, 4, 6, 8, 10)):
                sy, sx = divmod(frame_index, 6)
                tile = source[sy*128:sy*128+108, sx*192:sx*192+192]
                sheet[row*112+20:row*112+110, col*192:col*192+160] = cv2.resize(tile, (160,90))
        cv2.imwrite(str(args.output / f"candidates-{offset//12:02d}.jpg"), sheet)
    print(f"COMPLETE {len(results)} failed videos", flush=True)


if __name__ == "__main__":
    main()
