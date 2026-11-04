"""直接调用官方SpatialDR，核对已保存的机器人位置和朝向是否完整恢复。"""

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from simple.core.actor import RobotActor
from simple.dr.spatial import SpatialDR, SpatialDRCfg


def audit(metadata):
    content = metadata.read_bytes()
    results = []
    for line in content.splitlines():
        episode = json.loads(line)
        config = episode["environment_config"]
        if isinstance(config, str):
            config = json.loads(config)
        spatial_cfg = config["dr_cfgs"]["spatial"]
        if spatial_cfg["robot_orientation_region"] is not None:
            raise ValueError("This audit isolates saved state with no orientation randomizer")
        state = copy.deepcopy(config["dr_state_dict"]["spatial"])
        expected = copy.deepcopy(state["g1_sonic"])
        # 只承载uid；实际Actor/Pose及恢复调用均为原生实现，不创建机器人/物理环境。
        actor = RobotActor(SimpleNamespace(uid="g1_sonic"))
        randomizer = SpatialDR(SpatialDRCfg(spatial_mode=spatial_cfg["spatial_mode"]))
        randomizer.load_state_dict(state)
        randomizer("train", SimpleNamespace(scene=SimpleNamespace(), actors={"robot": actor}))
        wanted, actual = np.asarray(expected["quaternion"]), np.asarray(actor.pose.quaternion)
        wanted, actual = wanted / np.linalg.norm(wanted), actual / np.linalg.norm(actual)
        angle = float(2 * np.arccos(np.clip(abs(np.dot(wanted, actual)), 0, 1)))
        position_error = float(
            np.max(np.abs(np.asarray(expected["position"]) - actor.pose.position))
        )
        results.append(
            dict(
                episode=episode["episode_index"],
                saved=expected,
                restored=actor.pose.as_dict(),
                saved_state_after_call=copy.deepcopy(randomizer.state_dict()["g1_sonic"]),
                position_max_error_m=position_error,
                orientation_error_rad=angle,
            )
        )
    return dict(
        metadata=str(metadata.resolve()),
        sha256=hashlib.sha256(content).hexdigest(),
        episodes=len(results),
        orientation_mismatch_count=sum(row["orientation_error_rad"] > 1e-7 for row in results),
        maximum_orientation_error_rad=max(row["orientation_error_rad"] for row in results),
        maximum_position_error_m=max(row["position_max_error_m"] for row in results),
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(inspect.getfile(SpatialDR))
    result = dict(
        source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Native robot-only spatial restoration using actual saved states and actual Actor/Pose. "
        "No simulation, rendering, model inference or success-rate counterfactual. "
        "Does not establish that the measured small orientation error explains task failure.",
        datasets=[audit(path) for path in args.metadata],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if key != "results"}
                for row in result["datasets"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
