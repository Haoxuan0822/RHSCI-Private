"""对齐已完成轮次的官方episode成绩；保留历史协议不同及未配对噪声的限制。"""

import argparse
import hashlib
import json
from pathlib import Path

from summarize_results import LEVELS, read_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-root", "new-root", "historical", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    historical_bytes = args.historical.read_bytes()
    history = json.loads(historical_bytes)
    rows, sources = [], []
    for level_index, level in enumerate(LEVELS[:3]):
        results = {}
        for name, root in (("old", args.old_root), ("new", args.new_root)):
            path = root / level / f"repeat-{args.repeat:02d}" / "stats/eval_stats.txt"
            results[name] = read_stats(path)
            if set(results[name]) != set(range(10)):
                raise ValueError(f"Only completed ten-episode rounds are comparable: {path}")
            sources.append(
                dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            )
        recorded = [
            item["record"]
            for item in history["records"]
            if item["record"]["repeat"] == args.repeat
            and item["record"]["internal_level"] == level_index
        ]
        results["historical"] = {item["episode"]: item["success"] for item in recorded}
        if len(recorded) != 10 or set(results["historical"]) != set(range(10)):
            raise ValueError("Historical round is incomplete or duplicated")
        rows.append(
            dict(
                level=level,
                completed=10,
                success_episodes={
                    name: [i for i in range(10) if values[i]] for name, values in results.items()
                },
                outcomes=[
                    dict(episode=i, **{name: values[i] for name, values in results.items()})
                    for i in range(10)
                ],
                old_new_counts={
                    f"old_{a}_new_{b}": sum(
                        results["old"][i] == a and results["new"][i] == b for i in range(10)
                    )
                    for a in (False, True)
                    for b in (False, True)
                },
            )
        )
    report = dict(
        repeat=args.repeat,
        historical_source=str(args.historical.resolve()),
        historical_sha256=hashlib.sha256(historical_bytes).hexdigest(),
        score_sources=sources,
        levels=rows,
        scope="Same episode numbers, three levels, one completed round. Old/new use current 30/30; historical used 30/24 and different navigation/clock/hand compatibility. Native random noise is unpaired and rendering processes differ. This is descriptive evidence, not an isolated spawn-only success-rate experiment or the final 150-run result. Hard is absent in the old dataset.",
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
