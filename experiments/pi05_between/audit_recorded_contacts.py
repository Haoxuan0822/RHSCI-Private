"""以官方碰撞几何检查已记录姿态的手臂接触；只mj_forward，不推进物理。"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from simple.assets import AssetManager
from simple.assets.primitive import Box
from simple.core.actor import ObjectActor, RobotActor
from simple.core.object import SemanticAnnotated
from simple.core.types import Pose
from simple.engines.mujoco import MujocoSimulator


def fingerprint(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build_scene(config, robot_xml):
    # 复用官方三种builder，不启动Task/控制器/DDS/renderer，也不替换碰撞形状。
    builder = SimpleNamespace(_is_sonic=True)
    spec = mujoco.MjSpec()
    # 与当前两桌任务metadata一致；这里只forward，dt不会用于积分。
    spec.option.timestep = 0.005
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 2
    robot = RobotActor(SimpleNamespace(mjcf_path=str(robot_xml)))
    robot.pose = Pose(**config["dr_state_dict"]["spatial"]["g1_sonic"])
    MujocoSimulator._build_robot(builder, spec, spec.worldbody, robot)
    roles, files = {}, [fingerprint(robot_xml)]
    for role, entry in config["layout"]["actors"].items():
        if role in ("table", "table2"):
            box = Box(entry["size"], **entry["pose"])
            MujocoSimulator._build_primitive(builder, spec, spec.worldbody, box, role)
        elif "asset" in entry:
            asset_config = entry["asset"]
            asset = AssetManager.create(f"{asset_config['res_id']}:{asset_config['uid']}")
            actor = ObjectActor(asset)
            actor.pose = Pose(**entry["pose"])
            for mesh in asset.collision_meshes_mujoco:
                files.append(fingerprint(Path(mesh)))
            MujocoSimulator._build_object(builder, spec, spec.worldbody, actor)
            roles[role] = asset.label if isinstance(asset, SemanticAnnotated) else asset.uid
    model = spec.compile()
    return model, roles, files, hashlib.sha256(spec.to_xml().encode()).hexdigest()


def audit(label, metadata, trace, robot_xml, steps):
    rows = list(map(json.loads, trace.read_text().splitlines()))
    config_path = trace.parent / "effective_eval_config.json"
    config = json.loads(config_path.read_text())
    episode = next(
        row
        for row in map(json.loads, metadata.read_text().splitlines())
        if row["episode_index"] == config["episode_start"]
    )
    environment = episode["environment_config"]
    environment = json.loads(environment) if isinstance(environment, str) else environment
    model, roles, assets, spec_sha = build_scene(environment, robot_xml)
    data = mujoco.MjData(model)
    robot_nq = mujoco.MjModel.from_xml_path(str(robot_xml)).nq
    arm_roots = {model.body(f"{side}_shoulder_pitch_link").id for side in ("left", "right")}
    robot_root = model.body("pelvis").id
    robot_bodies = set()
    for body in range(model.nbody):
        parent = body
        while parent:
            if parent == robot_root:
                robot_bodies.add(body)
                break
            parent = int(model.body_parentid[parent])
    arm_bodies = set()
    for body in range(model.nbody):
        parent = body
        while parent:
            if parent in arm_roots:
                arm_bodies.add(body)
                break
            parent = int(model.body_parentid[parent])
    role_by_body = {model.body(name).id: role for role, name in roles.items()}
    role_by_body.update({model.body(name).id: name for name in ("table", "table2")})
    addresses = {
        role: int(model.jnt_qposadr[model.body(name).jntadr[0]]) for role, name in roles.items()
    }
    stable = [r for r in rows if r["kind"] == "env_step" and r["phase"] == "stabilize"][-1]
    physical = [r for r in rows if r["kind"] == "env_step" and r["phase"] == "policy"][:steps]
    if len(physical) != steps or [r["step"] for r in physical] != list(range(steps)):
        raise ValueError("Need contiguous recorded steps from zero")
    result = []
    for frame, row in enumerate([stable, *physical]):
        data.qpos[:robot_nq] = row["qpos"][:robot_nq]
        for role, address in addresses.items():
            data.qpos[address : address + 7] = row["objects"][role]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        errors = [
            float(np.max(np.abs(data.body(name).xpos - row["objects"][role][:3])))
            for role, name in roles.items()
        ]
        if max(errors) > 1e-9 or np.max(np.abs(data.body("pelvis").xpos - row["qpos"][:3])) > 1e-9:
            raise ValueError("Reconstructed body poses differ from recorded poses")
        contacts, self_contacts = [], []
        for contact in data.contact:
            a, b = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if (a in arm_bodies and b in robot_bodies) or (b in arm_bodies and a in robot_bodies):
                self_contacts.append(
                    dict(
                        bodies=[model.body(a).name, model.body(b).name],
                        distance_m=float(contact.dist),
                        included=bool(contact.efc_address >= 0),
                    )
                )
            if a in arm_bodies and b in role_by_body:
                arm, other = a, b
            elif b in arm_bodies and a in role_by_body:
                arm, other = b, a
            else:
                continue
            contacts.append(
                dict(
                    arm_body=model.body(arm).name,
                    other_role=role_by_body[other],
                    distance_m=float(contact.dist),
                    included=bool(contact.efc_address >= 0),
                )
            )
        result.append(
            dict(frame=frame, policy_step=frame - 1, contacts=contacts, self_contacts=self_contacts)
        )
    chunks = [r for r in rows if r["kind"] == "model_chunk" and r["step"] < steps]
    if [r["step"] for r in chunks] != list(range(0, steps, 30)):
        raise ValueError("Expected 30-step model chunks")
    if any(np.asarray(r["actions"]).shape != (30, 36) for r in chunks):
        raise ValueError("Expected complete 30x36 model outputs")
    targets = np.concatenate([r["actions"] for r in chunks])[:steps, 14:28]
    if not np.isfinite(targets).all():
        raise ValueError("Nonfinite model targets")
    boundaries = []
    for left, right in zip(chunks, chunks[1:]):
        a, b = (np.asarray(r["actions"])[:, 14:28] for r in (left, right))
        progress, jump = a[-1] - a[0], b[0] - a[-1]
        denominator = np.linalg.norm(progress) * np.linalg.norm(jump)
        boundaries.append(
            dict(
                step=right["step"],
                jump_l2_rad=float(np.linalg.norm(jump)),
                jump_vs_previous_progress_cosine=float(jump @ progress / denominator)
                if denominator > 1e-12
                else None,
            )
        )
    joint_limits = []
    names = (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
    for column, (side, name) in enumerate((s, n) for s in ("left", "right") for n in names):
        joint = model.joint(f"{side}_{name}_joint")
        address = int(joint.qposadr[0])
        if address != [*range(22, 29), *range(36, 43)][column] or not joint.limited[0]:
            raise ValueError("Recorded arm mapping or joint limits differ")
        actual = np.asarray([r["qpos"][address] for r in [stable, *physical]])
        low, high = joint.range
        joint_limits.append(
            dict(
                name=joint.name,
                range_rad=[float(low), float(high)],
                actual_min_margin_rad=float(np.minimum(actual - low, high - actual).min()),
                target_min_margin_rad=float(
                    np.minimum(targets[:, column] - low, high - targets[:, column]).min()
                ),
            )
        )
    return dict(
        label=label,
        metadata=fingerprint(metadata),
        trace=fingerprint(trace),
        config=fingerprint(config_path),
        assets=assets,
        generated_spec_sha256=spec_sha,
        robot_nq=robot_nq,
        native_nq=model.nq,
        frames=result,
        arm_joint_limits=joint_limits,
        boundaries=boundaries,
        first_arm_environment_contact_frame=next(
            (r["frame"] for r in result if r["contacts"]), None
        ),
        first_arm_self_contact_frame=next((r["frame"] for r in result if r["self_contacts"]), None),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs=3, action="append", required=True, metavar=("LABEL", "METADATA", "TRACE")
    )
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(
        script=fingerprint(Path(__file__)),
        native_builder=fingerprint(Path(inspect.getfile(MujocoSimulator))),
        cases=[
            audit(label, Path(metadata), Path(trace), args.robot_xml, args.steps)
            for label, metadata, trace in args.case
        ],
        scope="Static native collision geometry at recorded poses only; no mj_step, VLA, "
        "DDS or renderer. Ground and room walls omitted; analysis covers arms vs native "
        "tables, movable objects and robot self contacts. Zero velocities do not recover actual contact "
        "forces. Policy-rate samples cannot exclude between-sample collisions. "
        "Joint limits cover only the 14 arm joints and recorded 30-step targets.",
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            [
                dict(
                    label=c["label"],
                    first_contact=c["first_arm_environment_contact_frame"],
                    contact_frames=sum(bool(r["contacts"]) for r in c["frames"]),
                )
                for c in result["cases"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
