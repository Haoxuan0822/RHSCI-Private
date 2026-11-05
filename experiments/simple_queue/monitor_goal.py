"""Durable half-hour evaluation checks; unchanged checks only write local state."""

import ast
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

from plan import ROOT, REPO

sys.path.insert(0, str(REPO / "experiments/pi05_between"))
from summarize_results import read_round_stats

STATE = ROOT / "evals/goal-latest-protocol-20261105"
RAW = Path("/tmp/rhsci-video-upload-GHEvq5/repo")
ROOTS = [
    ROOT / "evals/pi05-between-zero-arms-pre1102-5cycles-20261104",
    ROOT / "evals/pi05-three-tasks-zero-arms-pre1102-5cycles-20261105",
    ROOT / "evals/gr00t-zero-arms-pre1102-5cycles-20261104",
    ROOT / "evals/nostate-zero-arms-pre1102-5cycles-20261105",
    ROOT / "evals/legacy-only-zero-arms-pre1102-5cycles-20261105",
]
TASKS = ("between", "hug", "oven", "handover", "faucet")
BACKBONES = {"pi05": "π0.5", "gr00t": "GR00T", "psi0": "Ψ0"}


def snapshot():
    results = {}
    for root in ROOTS:
        for metadata in root.glob("*/*/dr-level-*/worker.json"):
            info = json.loads(metadata.read_text())
            base = metadata.parent
            counts = [read_round_stats(base / f"repeat-{repeat:02d}", 10) for repeat in range(1, 6)]
            key = (info["backbone"], info["variant"], base.parent.parent.name.rsplit("-", 1)[-1])
            # The original between root has no model-key parent.
            if "BetweenTables" in info["task"]:
                key = (*key[:2], "between")
            levels = results.setdefault(key, {})
            level = int(info["level"].rsplit("-", 1)[-1])
            levels[level] = dict(successes=sum(sum(c.values()) for c in counts), episodes=sum(len(c) for c in counts), base=str(base))
    # Between-tables has variant directly below its root.
    root = ROOTS[0]
    for variant in ("STATE", "VGQ-AE"):
        for level in range(3):
            base = root / variant / f"dr-level-{level}"
            if not (base / "worker.json").exists():
                continue
            counts = [read_round_stats(base / f"repeat-{r:02d}", 10) for r in range(1, 6)]
            results.setdefault(("pi05", variant, "between"), {})[level] = dict(successes=sum(sum(c.values()) for c in counts), episodes=sum(len(c) for c in counts), base=str(base))
    return results


def git(*args):
    return subprocess.run(["git", "-C", str(RAW), *args], check=True, text=True, capture_output=True).stdout


def check_stalls(results):
    """Allow long renderer startup; resume a verified worker after an hour without results."""
    path = STATE / "worker-progress.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    now = time.time()
    current = {}
    for levels in results.values():
        for value in levels.values():
            if value["episodes"] == 50:
                continue
            base = Path(value["base"])
            info = json.loads((base / "worker.json").read_text())
            key = str(base)
            prior = old.get(key, {})
            changed = prior.get("episodes") != value["episodes"] or prior.get("pid") != info["worker_pid"]
            since = now if changed else prior["since"]
            current[key] = dict(episodes=value["episodes"], pid=info["worker_pid"], since=since)
            if now - since <= 3600:
                continue
            process = Path(f"/proc/{info['worker_pid']}/cmdline")
            try:
                command = process.read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            expected_root = str(base.parent.parent).encode()
            if any(arg.endswith(b"run_evaluation.py") for arg in command) and expected_root in command:
                os.kill(info["worker_pid"], signal.SIGTERM)
                with (STATE / "repairs.jsonl").open("a") as log:
                    log.write(json.dumps(dict(time=now, base=key, action="resume_stalled_worker", pid=info["worker_pid"], no_progress_seconds=now-since)) + "\n")
    path.write_text(json.dumps(current, indent=2) + "\n")


def repair_dispatchers():
    """Restart exited dispatchers with episode-level continuation; adopt live workers."""
    commands = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            commands.append((path / "cmdline").read_bytes().split(b"\0"))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    jobs = list(zip(("pi05", "gr00t", "nostate"), ROOTS[1:4]))
    phase2 = STATE / "phase2-started.json"
    if not phase2.exists() and all((root / "COMPLETE.json").exists() for root in ROOTS[1:4]):
        from legacy_inventory import inventory
        runs = inventory()
        with (STATE / "legacy-preparation.log").open("a") as log:
            child = subprocess.Popen(["/root/miniforge/bin/python3", "-u", str(REPO / "experiments/simple_queue/prepare_retests.py")],
                                     cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        phase2.write_text(json.dumps(dict(time=time.time(), models=len(runs), preparation_pid=child.pid)) + "\n")
    if phase2.exists():
        jobs.append(("legacy", ROOTS[4]))
    for backbone, root in jobs:
        if (root / "COMPLETE.json").exists():
            continue
        if any(any(arg.endswith(b"simple_queue/run.py") for arg in args) and str(root).encode() in args for args in commands):
            continue
        attempts_path = STATE / f"{backbone}-restarts.json"
        attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        recent = [t for t in attempts if time.time() - t < 86400]
        if len(recent) >= 3:
            raise RuntimeError(f"Repeated {backbone} failure; inspect retained logs instead of blind retries")
        command = [str(ROOT / ".venv-pi05-rsci/bin/python"), "-u", str(REPO / "experiments/simple_queue/run.py"),
                   "--backbone", backbone, "--gpus", *map(str, range(8)), "--auto-gpus", "--resume", "--resume-incomplete", "--output-root", str(root)]
        with (STATE / f"{backbone}-recovery.log").open("a") as log:
            child = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO / "src")),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        attempts_path.write_text(json.dumps([*recent, time.time()]) + "\n")
        with (STATE / "repairs.jsonl").open("a") as log:
            log.write(json.dumps(dict(time=time.time(), backbone=backbone, pid=child.pid, command=command)) + "\n")


def update_tables(results):
    """Replace only the latest-protocol section and matching training markers."""
    if git("status", "--porcelain").strip():
        raise RuntimeError("RAW checkout has edits; preserve them and defer publication")
    git("fetch", "origin", "main")
    git("merge", "--ff-only", "origin/main")
    path = RAW / "README.md"
    original = path.read_text()
    start = original.index("## 评测记录（最新协议）")
    end = original.index("## 历史评测记录", start)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    lines = ["## 评测记录（最新协议）", ""]
    incomplete = []
    for backbone, title in BACKBONES.items():
        methods = list(dict.fromkeys(k[1] for k in results if k[0] == backbone))
        if not methods:
            continue
        lines += [f"### {title}", "", "| Model | Mobile P&P | P&P&HugContainer | OpenOven | Handover | OpenFaucet |", "| :--- | :---: | :---: | :---: | :---: | :---: |"]
        for method in methods:
            cells = []
            for task in TASKS:
                levels = results.get((backbone, method, task), {})
                successes = [levels.get(n, {}).get("successes", 0) for n in range(3)]
                episodes = [levels.get(n, {}).get("episodes", 0) for n in range(3)]
                if not sum(episodes):
                    cells.append("—")
                    continue
                complete = episodes == [50, 50, 50]
                cells.append("/".join(map(str, successes)) + f"（{100 * sum(successes) / sum(episodes):.1f}%）" + ("" if complete else "*"))
                if not complete:
                    incomplete.append(f"{title} {method} {task}：" + "/".join(map(str, episodes)))
            lines.append("| " + " | ".join([method, *cells]) + " |")
        lines.append("")
    lines += ["三级成功数按 Level-0/1/2 排列；每级50次、总150次。`—` 为无结果；`*` 为未测完，百分比按已完成样本计算。中断或故障不算模型失败。", "", "最新协议：零臂、pre1102 初态包、decoupled WBC；原生 wall time、无 RTC、测量加噪关闭，仅首轮完整双目录像。pre1102 是资产版本标识。[配置与历史明细](EXPERIMENT_RECORDS.md)。", "", "未完成组的各级已完成数：" + ("；".join(incomplete) or "无"), "", f"更新：**{stamp}（北京时间）**；Goal 每半小时检查，完成后更新。", ""]
    revised = original[:start] + "\n".join(lines) + "\n" + original[end:]
    # Training tables preserve all unrelated cells and history.
    before, training = revised.split("## 训练记录", 1)
    backbone = None
    rows = training.splitlines()
    for i, row in enumerate(rows):
        if row.startswith("### "):
            backbone = next((b for b, name in BACKBONES.items() if row == f"### {name}"), None)
        if not row.startswith("| ") or backbone is None:
            continue
        fields = [f.strip() for f in row.strip("|").split("|")]
        for j, task in enumerate(TASKS, 1):
            levels = results.get((backbone, fields[0], task), {})
            if any(l["episodes"] > 0 for l in levels.values()) and j < len(fields):
                fields[j] = fields[j].strip("（）()").rstrip("*")
        rows[i] = "| " + " | ".join(fields) + " |"
    revised = before + "## 训练记录" + "\n".join(rows) + "\n"
    path.write_text(revised)
    detail = RAW / "EXPERIMENT_RECORDS.md"
    content = detail.read_text()
    tag = "<!-- goal-latest-snapshot -->"
    if tag in content:
        a = content.index(tag)
        b = content.index(tag, a + len(tag)) + len(tag)
        content = content[:a] + content[b:]
    table = [tag, "## Goal 自动核对快照", "", f"更新：{stamp}（北京时间）；逐集统计合并续测片段，历史记录保留。", "", "| Backbone | Model | Task | Level-0 | Level-1 | Level-2 |", "| --- | --- | --- | --- | --- | --- |"]
    for (b, method, task), levels in results.items():
        cells = [f"{levels.get(n, {}).get('successes', 0)}/{levels.get(n, {}).get('episodes', 0)}" for n in range(3)]
        table.append("| " + " | ".join([BACKBONES[b], method, task, *cells]) + " |")
    detail.write_text(content.rstrip() + "\n\n" + "\n".join(table) + "\n" + tag + "\n")
    git("diff", "--check")
    git("add", "README.md", "EXPERIMENT_RECORDS.md")
    if git("diff", "--cached", "--name-only").strip():
        git("commit", "-m", "Update latest-protocol evaluation progress")
        git("push", "origin", "HEAD:main")


def check_downloads():
    """记录准备收据和实时分块进度；下载一小时无增长时留下故障证据。"""
    from legacy_inventory import inventory
    from plan import PREPARATION, run_key

    runs = inventory()
    records = {row["rfilename"]: row for run in runs for row in run["records"]}
    path = STATE / "download-progress.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    now = time.time()
    active = []
    preparation_pids = []
    model_root = ROOT / "hf/RHSCI/model-59a1ea14"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if any(arg.endswith("simple_queue/prepare_retests.py") for arg in args[1:]):
                preparation_pids.append(int(proc.name))
            if len(args) < 4 or not args[1].endswith("/download_wheel.py"):
                continue
            destination = Path(args[3])
            relative = str(destination.relative_to(model_root))
            if relative not in records:
                continue
            parts = destination.parent / (destination.name + ".parts")
            downloaded = sum(part.stat().st_size for part in parts.iterdir() if part.is_file()) if parts.exists() else 0
        except (OSError, UnicodeError, ValueError):
            continue
        prior = next((v for v in old.get("active", []) if v["file"] == relative), {})
        since = prior.get("unchanged_since", now) if prior.get("bytes") == downloaded else now
        active.append(dict(file=relative, pid=int(proc.name), bytes=downloaded,
                           total=records[relative]["size"], unchanged_since=since,
                           stalled=now-since > 3600))
    ready = [run_key(run) for run in runs if (PREPARATION / f"{run_key(run)}.json").exists()]
    # 线程池会跳过失败项继续工作；父PID存活不能证明所有下载健康。
    failures = {}
    preparation_log = STATE / "legacy-preparation.log"
    if preparation_log.exists():
        for line in preparation_log.read_text().splitlines():
            if line.startswith("PREPARATION_FAILED "):
                failure = ast.literal_eval(line.split(" ", 1)[1])
                failures[failure["run"]] = failure["error"]
    active_files = {row["file"] for row in active}
    failed = [dict(run=run_key(run), error=failures[run_key(run)],
                   retry_downloading=any(record["rfilename"] in active_files for record in run["records"]))
              for run in runs if run_key(run) in failures and run_key(run) not in ready]
    payload = dict(time=now, ready=ready, total=len(runs), active=active,
                   preparation_pids=preparation_pids, failed_unprepared=failed)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    if (any(row["stalled"] for row in active)
            or any(not row["retry_downloading"] for row in failed)
            or (len(ready) < len(runs) and not preparation_pids)):
        with (STATE / "errors.log").open("a") as log:
            log.write(f"{datetime.now(timezone.utc).isoformat()} Download needs repair; see download-progress.json\n")


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    saved = STATE / "published-completions.json"
    if saved.exists():
        previous = json.loads(saved.read_text())
    while True:
        try:
            check_downloads()
            repair_dispatchers()
            results = snapshot()
            check_stalls(results)
            payload = {"|".join(k): v for k, v in results.items()}
            (STATE / "progress.json").write_text(json.dumps(dict(time=time.time(), results=payload), indent=2) + "\n")
            completed = sorted("|".join(k) for k, levels in results.items() if sum(v["episodes"] for v in levels.values()) == 150)
            if completed != previous:
                update_tables(results)
                saved.write_text(json.dumps(completed) + "\n")
                previous = completed
            (STATE / "last-check.json").write_text(json.dumps(dict(time=time.time(), completed=completed)) + "\n")
        except Exception as error:
            with (STATE / "errors.log").open("a") as log:
                log.write(f"{datetime.now(timezone.utc).isoformat()} {type(error).__name__}: {error}\n")
        time.sleep(1800)


if __name__ == "__main__":
    main()
