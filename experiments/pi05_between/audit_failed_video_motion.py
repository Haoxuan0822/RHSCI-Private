"""只读首轮失败视频，量化末六秒画面运动；候选不等同于失败根因。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from collections import defaultdict

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
    parser.add_argument("--full-timeline", action="store_true", help="读取冻结清单，导出全程接触表供人工阶段分类")
    parser.add_argument("--summarize-review", action="store_true", help="校验并汇总显式人工标注；不自动判定阶段")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize_review:
        summarize_review(args.output)
        return
    if args.full_timeline:
        full_timeline(args.output)
        return
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


def full_timeline(output):
    """沿用原始清单，每两秒取一帧；不从运动量推断任务阶段或失败原因。"""
    items = json.loads((output / "motion.json").read_text())["items"]
    cv2.setNumThreads(1)
    groups = defaultdict(list)
    for item in items:
        groups[(item["backbone"], item["variant"], item["task"])].append(item)
    for key, rows in groups.items():
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset+8]
            sheet = np.full((len(batch)*124, 160*13, 3), 245, np.uint8)
            for row, item in enumerate(batch):
                cap = cv2.VideoCapture(item["video"])
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                y = row*124
                label = item["id"] + f"  ({item['duration_s']:.1f}s)"
                cv2.putText(sheet, label, (3,y+15), cv2.FONT_HERSHEY_SIMPLEX, .44, (0,0,0), 1)
                for col, seconds in enumerate(np.linspace(0, max(0,(count-1)/fps), 13)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, round(seconds*fps))
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot seek {seconds}: {item['video']}")
                    x = col*160
                    sheet[y+18:y+108,x:x+160] = cv2.resize(frame,(160,90),interpolation=cv2.INTER_AREA)
                    cv2.putText(sheet, f"{seconds:.1f}s", (x+3,y+121), cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
                cap.release()
            name = "-".join(key) + f"-page{offset//8:02d}.jpg"
            cv2.imwrite(str(output / ("timeline-"+name)),sheet)
        print("TIMELINE", key, len(rows), flush=True)


def summarize_review(output):
    """严格匹配冻结失败清单与人工标注；未知/重复ID报错，不补造标签。"""
    data = json.loads((output / "motion.json").read_text())
    review = json.loads((output / "stage-review.json").read_text())
    groups = {r["key"]: r for r in review["groups"]}
    counts = defaultdict(lambda: dict(failures=0, stage_stall_candidate=0, uncertain=0, other=0))
    labels = []
    for item in data["items"]:
        key = "-".join(item[k] for k in ("backbone", "variant", "task"))
        group = groups[key]
        if not group.get("reviewed"):
            raise ValueError(f"Unreviewed group: {key}")
        short = f"L{item['level']}-E{item['episode']}"
        explicit = [name for name in ("stage_stall", "uncertain", "other") if short in group.get(name, [])]
        if len(explicit) > 1:
            raise ValueError(f"Conflicting labels: {key} {short}")
        label = explicit[0] if explicit else ("stage_stall" if group.get("all_stage_stall") else "other")
        label = "stage_stall_candidate" if label == "stage_stall" else label
        counts[key]["failures"] += 1
        counts[key][label] += 1
        labels.append(dict(id=item["id"], video=item["video"], label=label))
    for key, group in groups.items():
        known = {f"L{r['level']}-E{r['episode']}" for r in data["items"]
                 if "-".join(r[k] for k in ("backbone", "variant", "task")) == key}
        specified = [s for name in ("stage_stall", "uncertain", "other") for s in group.get(name, [])]
        if not known or not set(specified) <= known or len(specified) != len(set(specified)):
            raise ValueError(f"Invalid/duplicate/unknown reviewed IDs: {key}")
    result = dict(scope=review["scope"], review_method=review["review_method"],
                  definitions=review["definitions"], counts=dict(counts), labels=labels)
    (output / "stage-summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
