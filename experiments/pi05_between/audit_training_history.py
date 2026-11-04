"""核验已下载的小型历史训练元数据，不下载或加载任何模型权重。"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.checkpoint_manifest.read_text())
    identities, metadata = {}, {}
    for name in ("checkpoint-40000/trainer_state.json", "speed_resume_origin.json"):
        raw = (args.directory / Path(name).name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        record = manifest["files"][name]
        if digest != record["sha256"] or len(raw) != record["bytes"]:
            raise ValueError(f"Historical metadata differs from checkpoint manifest: {name}")
        identities[name] = dict(sha256=digest, bytes=len(raw))
        metadata[name] = json.loads(raw)
    state = metadata["checkpoint-40000/trainer_state.json"]
    if state["global_step"] != 40000 or manifest["step"] != 40000:
        raise ValueError("This diagnosis requires the specified checkpoint-40000")
    contract = json.loads(args.checkpoint_manifest.with_name("launch_contract.json").read_text())
    source = args.training_source.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if source_sha != contract["code"]["src/rsci_pi05/train.py"]:
        raise ValueError("Training source differs from the checkpoint contract")
    # 只加载已核对的纯数学函数，避免导入原训练器或启动任何训练。
    tree = ast.parse(source)
    schedule = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "simple_lr_multiplier"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[schedule], type_ignores=[]), str(args.training_source), "exec"),
        namespace,
    )
    rate_errors = [
        abs(
            row["learning_rate"]
            - contract["lr"] * namespace["simple_lr_multiplier"](row["step"] - 1)
        )
        for row in state["log_history"]
        if "learning_rate" in row
    ]
    summaries = {}
    for key in ("action_loss", "ardy_loss", "loss", "grad_norm", "learning_rate"):
        rows = [row for row in state["log_history"] if key in row]
        steps = [row["step"] for row in rows]
        component = key in ("action_loss", "ardy_loss")
        expected = list(range(0 if component else 10, 40000 if component else 40010, 10))
        if not all(math.isfinite(row[key]) for row in rows):
            raise ValueError(f"Nonfinite historical metric: {key}")
        windows = []
        for start, end in ((0, 1000), (6000, 8000), (10000, 20000), (30000, 40001)):
            values = [row[key] for row in rows if start <= row["step"] < end]
            windows.append(
                dict(
                    start_step=start,
                    end_step_exclusive=end,
                    records=len(values),
                    median=median(values) if values else None,
                )
            )
        summaries[key] = dict(
            records=len(rows),
            exact_expected_step_grid=steps == expected,
            windows=windows,
        )
    report = dict(
        metadata=identities,
        global_step=state["global_step"],
        epoch=state["epoch"],
        train_batch_size=state["train_batch_size"],
        metrics=summaries,
        learning_rate_check=dict(
            source_sha256=source_sha,
            scheduler_step_offset_from_log=-1,
            records=len(rate_errors),
            max_abs_error=max(rate_errors),
        ),
        recorded_eval_metric_keys=sorted(
            {key for row in state["log_history"] for key in row if key.startswith("eval_")}
        ),
        resume_origin=metadata["speed_resume_origin.json"],
        limitation="Historical metadata and loss logs only. Low teacher-conditioned flow-matching loss does not prove autonomous sampled-action accuracy, closed-loop continuity, or original optimizer/data order equivalence. The resume receipt describes payload preservation; this audit did not load those payloads.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
