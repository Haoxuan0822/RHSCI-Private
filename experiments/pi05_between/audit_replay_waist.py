"""用已保存qpos与原生MJCF核对专家回放腰部差异，不推进物理或生成图像。"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("simple-source", "trace", "expert-requests", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    xml = args.simple_source / "data/robots/g1_sonic/g1_29dof_with_hand.xml"
    engine_path = args.simple_source / "src/simple/engines/mujoco.py"
    engine_source = engine_path.read_bytes()
    assignment = next(
        node
        for node in ast.walk(ast.parse(engine_source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_HEAD_CAM_POSITION"
            for target in node.targets
        )
    )
    # 只求值可信本地官方源码的该常量表达式，避免导入完整Kit引擎。
    head_offset = eval(
        compile(ast.Expression(assignment.value), str(engine_path), "eval"), {"np": np}
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    waist_names = ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")
    addresses = np.array([model.jnt_qposadr[model.joint(name).id] for name in waist_names])
    torso = model.body("torso_link").id
    request_content = args.expert_requests.read_bytes()
    requests = {row["frame"]: row for row in map(json.loads, request_content.splitlines())}
    trace_content = args.trace.read_bytes()
    previous, observations = None, {}
    for row in map(json.loads, trace_content.splitlines()):
        if row["kind"] == "env_step":
            previous = row
        elif row["kind"] == "model_chunk":
            if previous is None or row["step"] in observations:
                raise ValueError("Missing or repeated preceding observation")
            observations[row["step"]] = previous
    if set(observations) != set(requests) or sorted(requests) != [0, 30, 60, 90]:
        raise ValueError("Expected matching four expert-replay requests")

    def pose(qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        rotation = data.xmat[torso].reshape(3, 3).copy()
        return rotation, data.xpos[torso] + rotation @ head_offset

    rows = []
    for frame, request in sorted(requests.items()):
        qpos = np.asarray(observations[frame]["qpos"])[: model.nq]
        actual, expert = (
            np.asarray(request[key])[28:31] for key in ("actual_state", "expert_state")
        )
        correspondence_error = float(np.max(np.abs(qpos[addresses] - actual)))
        if correspondence_error > 1.5e-6:
            raise ValueError("Trace robot qpos does not match native observed waist state")
        corrected = qpos.copy()
        corrected[addresses] = expert
        original_rotation, original_point = pose(qpos)
        corrected_rotation, corrected_point = pose(corrected)
        cosine = (np.trace(original_rotation.T @ corrected_rotation) - 1) / 2
        row = dict(
            frame=frame,
            actual_waist_rpy_rad=actual.tolist(),
            expert_waist_rpy_rad=expert.tolist(),
            waist_delta_rad=(actual - expert).tolist(),
            trace_vs_observed_waist_max_abs_rad=correspondence_error,
            torso_mount_rotation_change_deg=float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
            mujoco_head_proxy_displacement_m=float(
                np.linalg.norm(corrected_point - original_point)
            ),
            floating_base_qpos_unchanged=True,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        trace_sha256=hashlib.sha256(trace_content).hexdigest(),
        requests_sha256=hashlib.sha256(request_content).hexdigest(),
        xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256(engine_source).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        waist_names=waist_names,
        head_offset_from_native_mujoco=head_offset.tolist(),
        rows=rows,
        limitation="Only a waist-joint kinematic counterfactual on saved replay qpos, with replay floating base fixed. Training data has no complete floating-base trajectory. Native MuJoCo head offset is a proxy, not a measurement of actual Isaac USD camera pixels or proof that view differences cause the policy errors. No rendering, inference, dynamics or deployment change.",
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
