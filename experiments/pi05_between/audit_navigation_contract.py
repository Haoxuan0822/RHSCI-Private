"""Replay real lower-controller inputs through native and historical navigation rules.

Only network selection and command conversion are compared. Network outputs are
replaced by placeholders and never enter a simulator or a model server.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from legacy_navigation import install


def evaluate(method, row):
    policy = SimpleNamespace(
        height_cmd=0.74,
        # 原生load_config把cmd_init转为float32，之后仅原位赋值，必须保留舍入语义。
        cmd=np.array(row["previous_velocity"], dtype=np.float32),
        target_yaw_cmd=row["previous_heading"],
        roll_cmd=0.0,
        pitch_cmd=0.0,
        yaw_cmd=0.0,
        obs_tensor=torch.zeros(1, 1),
        observation={"floating_base_pose": np.asarray(row["floating_base_pose"])},
        use_teleop_policy_cmd=True,
        use_policy_action=True,
        config={"action_scale": 1.0, "default_angles": np.zeros(1), "num_actions": 1},
    )

    def network(mode):
        def select(_):
            policy.selected = mode
            return torch.zeros(1, 1)

        return select

    policy.policy_1, policy.policy_2 = network("stand"), network("walk")
    command = row["navigate_cmd"]
    result = method(
        policy,
        0.0,
        None,
        np.array([0.74]),
        np.zeros(3),
        np.asarray(command) if command is not None else None,
    )
    return dict(
        mode=policy.selected,
        velocity=policy.cmd.tolist(),
        navigate_cmd=result["navigate_cmd"].tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy

    native = G1GearWbcPolicy.get_action
    provenance = install()
    historical = G1GearWbcPolicy.get_action
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    installed = next(row for row in rows if row["kind"] == "trace_installed")
    os.environ["MODIFIED_POLICY_SWITCH_BEHAVIOR"] = installed["modified_policy_switch_behavior"]
    active = "historical" if installed["navigation"] is not None else "current"
    results = []
    pending = selected = None
    for row in rows:
        if row["kind"] == "lower_input":
            if pending is not None:
                raise ValueError("Unclosed controller input")
            pending = row
        elif row["kind"] == "lower_policy":
            selected = row["mode"]
        elif row["kind"] == "lower_output":
            if pending is None:
                raise ValueError("Controller output without input")
            alternatives = {
                name: evaluate(method, pending)
                for name, method in (("current", native), ("historical", historical))
            }
            actual = alternatives[active]
            if actual["mode"] != selected:
                raise ValueError("Replay selected a different network than the live controller")
            for key in ("velocity", "navigate_cmd"):
                if not np.array_equal(actual[key], row[key]):
                    raise ValueError(f"Replay {key} differs from live controller")
            results.append(
                dict(
                    phase=pending["phase"],
                    step=pending["step"],
                    input_command=pending["navigate_cmd"],
                    **alternatives,
                    selected_mode_differs=alternatives["current"]["mode"]
                    != alternatives["historical"]["mode"],
                    velocity_l2_difference=float(
                        np.linalg.norm(
                            np.array(alternatives["current"]["velocity"])
                            - alternatives["historical"]["velocity"]
                        )
                    ),
                )
            )
            pending = selected = None
    if pending is not None or not results:
        raise ValueError("Incomplete or missing controller records")
    report = dict(
        trace=str(args.trace),
        provenance=provenance,
        active_protocol=active,
        actual_command_and_selection_replay_exact=True,
        counts_by_phase={
            phase: dict(
                calls=sum(row["phase"] == phase for row in results),
                selection_differences=sum(
                    row["phase"] == phase and row["selected_mode_differs"] for row in results
                ),
                max_velocity_l2_difference=max(
                    (row["velocity_l2_difference"] for row in results if row["phase"] == phase),
                    default=0.0,
                ),
            )
            for phase in ("stabilize", "policy")
        },
        rows=results,
        limitation="Identical recorded lower-controller inputs and pre-call state. Only command conversion and network selection use real methods; neural outputs are placeholders, not counterfactual physical trajectories or success rates.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
