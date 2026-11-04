"""用原始训练导航标签和真实停滞trace量化rate/flag语义差异，不推进物理。"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from audit_navigation_contract import evaluate
from legacy_navigation import SOURCE_SHA256
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def heading_rate(command, pose):
    yaw = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2]
    error = np.arctan2(np.sin(command[:, 3] - yaw), np.cos(command[:, 3] - yaw))
    rate = np.where(np.abs(error) < 0.01, 0, np.clip(error * 2, -1, 1))
    return error, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-parquet", "trace", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from decoupled_wbc.control.policy import g1_gear_wbc_policy as module

    native = module.G1GearWbcPolicy.get_action
    source = textwrap.dedent(inspect.getsource(native))
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Native method changed")
    gate = ast.parse(
        "abs(yaw_error) < YAW_ERROR_THRESHOLD or np.abs(vyaw_flag) < 0.1", mode="eval"
    ).body
    changes = []

    class RemoveFlagGate(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.dump(node.test) == ast.dump(gate):
                node.test = ast.parse("abs(yaw_error) < YAW_ERROR_THRESHOLD", mode="eval").body
                changes.append(1)
            return self.generic_visit(node)

    tree = RemoveFlagGate().visit(ast.parse(source))
    if changes != [1]:
        raise RuntimeError("Expected exactly the one-line July 21 gate change")
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree), "<gate-only-counterfactual>", "exec"), namespace)
    alternative = namespace["get_action"]
    raw = pq.read_table(args.raw_parquet).to_pydict()
    command = np.asarray(raw["teleop.navigate_command"])
    pose = np.asarray(raw["observation.base_pose"])
    error, rate = heading_rate(command, pose)
    affected = (np.abs(error) >= 0.01) & (np.abs(command[:, 2]) < 0.1)
    # 恒定target yaw期间非零第三维是rate反馈的额外证据，仍不恢复未保存的操作者输入。
    initial = command[:120]
    raw_report = dict(
        frames=len(command),
        third_dimension_vs_heading_rate_correlation=float(np.corrcoef(command[:, 2], rate)[0, 1]),
        third_dimension_vs_heading_rate_rmse=float(np.mean((command[:, 2] - rate) ** 2) ** 0.5),
        gate_changes_heading_correction=int(affected.sum()),
        first120=dict(
            heading_target_minmax=[float(initial[:, 3].min()), float(initial[:, 3].max())],
            nonzero_third_dimension=int(np.count_nonzero(initial[:, 2])),
            gate_changes_heading_correction=int(affected[:120].sum()),
        ),
        selected=[
            dict(frame=i, command=command[i].tolist(), ungated_rate=float(rate[i]))
            for i in (0, 30, 60, 90)
        ],
    )
    trace = [json.loads(line) for line in args.trace.read_text().splitlines()]
    installed = next(row for row in trace if row["kind"] == "trace_installed")
    if installed["navigation"] is not None:
        raise ValueError("Need an unmodified native-controller trace")
    os.environ["MODIFIED_POLICY_SWITCH_BEHAVIOR"] = installed["modified_policy_switch_behavior"]
    pending, entries = None, []
    for row in trace:
        if row["kind"] == "lower_input":
            pending = row
        elif row["kind"] == "lower_output":
            if pending is None:
                raise ValueError("Unmatched lower output")
            result = evaluate(native, pending)
            if (
                result["velocity"] != row["velocity"]
                or result["navigate_cmd"] != row["navigate_cmd"]
            ):
                raise ValueError("Native replay differs from live trace")
            alt = evaluate(alternative, pending)
            if pending["phase"] == "policy":
                entries.append(
                    dict(
                        step=pending["step"],
                        command=pending["navigate_cmd"],
                        native_velocity=result["velocity"],
                        ungated_velocity=alt["velocity"],
                        native_mode=result["mode"],
                        ungated_mode=alt["mode"],
                        yaw_rate_difference=float(alt["velocity"][2] - result["velocity"][2]),
                    )
                )
            pending = None
    if not entries or pending is not None:
        raise ValueError("Incomplete trace")
    differences = np.asarray([entry["yaw_rate_difference"] for entry in entries])
    report = dict(
        raw_parquet=str(args.raw_parquet),
        raw_parquet_sha256=sha(args.raw_parquet),
        trace=str(args.trace),
        trace_sha256=sha(args.trace),
        native_source_sha256=SOURCE_SHA256,
        script_sha256=sha(Path(__file__)),
        raw=raw_report,
        trace_summary=dict(
            policy_calls=len(entries),
            affected_calls=int(np.count_nonzero(differences)),
            max_abs_yaw_rate_difference=float(np.abs(differences).max()),
            network_selection_differences=sum(
                e["native_mode"] != e["ungated_mode"] for e in entries
            ),
            native_outputs_replayed_exactly=True,
        ),
        trace_rows=entries,
        limitations=[
            "Controller networks are placeholders; command conversion uses the actual native method.",
            "Only flag gating changes; this is not the full historical navigation adapter.",
            "Same recorded state for both methods, no counterfactual physical trajectory or VLA inference.",
            "Rate-like labels support a contract mismatch but do not prove it causes arm reset or success regression.",
            "Raw operator turning commands and original recorder commit are not contained in this parquet.",
        ],
    )
    if args.dataset:
        actions = [
            np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist())
            for path in sorted((args.dataset / "data").glob("chunk-*/episode_*.parquet"))
        ]
        if not actions or any(value.shape[1] != 36 for value in actions):
            raise ValueError("Expected actual 36D processed training actions")
        navigation = np.concatenate([value[:, 32:36] for value in actions])
        report["processed_dataset"] = dict(
            path=str(args.dataset),
            episodes=len(actions),
            frames=len(navigation),
            nonzero_third_dimension=int(np.count_nonzero(navigation[:, 2])),
            nonzero_below_flag_threshold=int(
                np.count_nonzero((navigation[:, 2] != 0) & (np.abs(navigation[:, 2]) < 0.1))
            ),
            zero_heading_nonzero_third_dimension=int(
                np.count_nonzero((navigation[:, 3] == 0) & (navigation[:, 2] != 0))
            ),
            third_dimension_quantiles=np.quantile(
                navigation[:, 2], [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
            ).tolist(),
            limitation="Distribution of saved labels only; no original physical poses for "
            "these additional episodes, so no global rate/controller reconstruction.",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trace_rows"}))


if __name__ == "__main__":
    main()
