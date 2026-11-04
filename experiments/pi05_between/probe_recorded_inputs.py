"""Compare live policies with expert targets on actual recorded training inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from openpi.rhsci.data import SimpleRHSCIDataset
from rhsci.inference.protocol import RequestMessage, ResponseMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames", type=int, nargs="+", help="Explicit frames within each episode")
    parser.add_argument("--state-port", type=int, default=8120)
    parser.add_argument("--vgq-port", type=int, default=8126)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = SimpleRHSCIDataset(args.dataset, "STATE")
    session = requests.Session()
    session.trust_env = False
    start = 0
    with (args.output / "predictions.jsonl").open("x", buffering=1) as stream:
        for episode in dataset.episodes[: args.episodes]:
            windows = int(episode["length"]) - 30 + 1
            frames = (
                args.frames
                if args.frames is not None
                else [int((windows - 1) * fraction) for fraction in (0.0, 0.2, 0.4, 0.6, 0.8)]
            )
            for frame in frames:
                if not 0 <= frame < windows:
                    raise ValueError(f"Frame {frame} has no complete 30-step target")
                sample = dataset.raw_sample(start + frame)
                identity = (
                    f"episode-{sample['episode_index']:03d}-frame-{sample['frame_index']:04d}"
                )
                Image.fromarray(sample["images"][-1]).save(args.output / f"{identity}.png")
                row = {
                    "episode": sample["episode_index"],
                    "frame": sample["frame_index"],
                    "instruction": sample["text"],
                    "state": sample["state"].tolist(),
                    "expert_actions": sample["actions"].tolist(),
                    "models": {},
                }
                for model, port in (("STATE", args.state_port), ("VGQ-AE", args.vgq_port)):
                    request = RequestMessage(
                        image={"rgb_head_stereo_left": sample["images"][-1]},
                        instruction=sample["text"],
                        history={"reset": True},
                        state={"states": sample["state"][None]},
                        condition={},
                        gt_action=[],
                        dataset_name="simple",
                        timestamp=f"recorded-input-probe-{identity}-{model}",
                    )
                    response = session.post(
                        f"http://127.0.0.1:{port}/act", json=request.serialize(), timeout=120
                    )
                    response.raise_for_status()
                    actions = ResponseMessage.deserialize(response.json()).action
                    if actions.shape != (30, 36) or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid {model} prediction: {actions.shape}")
                    arm_error = actions[:, 14:28] - sample["actions"][:, 14:28]
                    row["models"][model] = {
                        "actions": actions.tolist(),
                        "arm_expert_rmse_rad": float(np.sqrt(np.mean(arm_error**2))),
                        "first_arm_expert_l2_rad": float(np.linalg.norm(arm_error[0])),
                        "first_arm_state_l2_rad": float(
                            np.linalg.norm(actions[0, 14:28] - sample["state"][14:28])
                        ),
                    }
                stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps(
                        {
                            "input": identity,
                            "metrics": {
                                model: {
                                    key: value for key, value in result.items() if key != "actions"
                                }
                                for model, result in row["models"].items()
                            },
                        }
                    ),
                    flush=True,
                )
            start += windows


if __name__ == "__main__":
    main()
