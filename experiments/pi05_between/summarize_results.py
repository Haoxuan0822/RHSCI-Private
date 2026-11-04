"""Report official episode counts in level-0/1/2/hard order without hiding partial runs."""

import argparse
import re
from pathlib import Path

LEVELS = ("dr-level-0", "dr-level-1", "dr-level-2", "dr-level-hard")


def read_stats(path, num_episodes=10):
    """读取官方结果，拒绝重复、越界或损坏的 episode 行；允许运行中的未完成轮次。"""
    stats = {}
    for line in path.read_text().splitlines():
        if not line.startswith("episode_"):
            continue
        match = re.fullmatch(r"episode_(\d+):\s*(True|False)\s*", line)
        if match is None:
            raise ValueError(f"Malformed episode result: {path}: {line}")
        episode = int(match[1])
        if episode in stats or not 0 <= episode < num_episodes:
            raise ValueError(f"Duplicate or out-of-range episode: {path}: {episode}")
        stats[episode] = match[2] == "True"
    return stats


def summarize(root):
    """按模型输出成功数和已完成数向量，避免把未运行样本当作失败。"""
    for variant in ("STATE", "VGQ-AE", "NO-STATE", "RGB"):
        if not (root / variant).is_dir():
            continue
        successes, completed = [], []
        for level in LEVELS:
            rounds = [
                read_stats(path)
                for path in sorted((root / variant / level).glob("repeat-*/stats/eval_stats.txt"))
            ]
            successes.append(sum(sum(stats.values()) for stats in rounds))
            completed.append(sum(len(stats) for stats in rounds))
        success_vector = "/".join(map(str, successes))
        completed_vector = "/".join(map(str, completed))
        print(f"{variant}: 成功 {success_vector}；已完成 {completed_vector}（阶段统计）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_root.is_dir():
        parser.error("Output root does not exist")
    summarize(args.output_root)
