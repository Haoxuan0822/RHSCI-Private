"""Isolate image/state feedback on identical saved observations without a simulator."""

import argparse
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from rhsci.inference.protocol import RequestMessage, ResponseMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-port", type=int, default=8120)
    parser.add_argument("--vgq-port", type=int, default=8126)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--instruction", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    chunks = [
        json.loads(line)
        for line in args.trace.read_text().splitlines()
        if json.loads(line)["kind"] == "model_chunk"
    ]
    initial = chunks[0]
    initial_state = np.asarray(initial["observed_state"]["states"][0], np.float32)
    initial_image = np.asarray(Image.open(initial["image_path"]))
    session = requests.Session()
    session.trust_env = False
    results = []
    for chunk in chunks[1:]:
        state = np.asarray(chunk["observed_state"]["states"][0], np.float32)
        current_image = np.asarray(Image.open(chunk["image_path"]))
        direction = state[14:28] - initial_state[14:28]
        for condition, rgb, proprio in (
            ("current_image_current_state", current_image, state),
            ("current_image_initial_state", current_image, initial_state),
            ("initial_image_current_state", initial_image, state),
        ):
            for model, port in (("STATE", args.state_port), ("VGQ-AE", args.vgq_port)):
                actions = []
                for repeat in range(args.repeats):
                    request = RequestMessage(
                        image={"rgb_head_stereo_left": rgb},
                        instruction=chunk.get("instruction", args.instruction),
                        history={"reset": True},
                        state={"states": proprio[None]},
                        condition={},
                        gt_action=[],
                        dataset_name="simple",
                        timestamp=f"feedback-{chunk['step']}-{condition}-{model}-{repeat}",
                    )
                    response = session.post(
                        f"http://127.0.0.1:{port}/act", json=request.serialize(), timeout=120
                    )
                    response.raise_for_status()
                    action = ResponseMessage.deserialize(response.json()).action
                    if action.shape != (30, 36) or not np.isfinite(action).all():
                        raise ValueError("Invalid prediction")
                    actions.append(action)
                first = np.stack(actions)[:, 0]
                row = {
                    "step": chunk["step"],
                    "condition": condition,
                    "model": model,
                    "instruction": chunk.get("instruction", args.instruction),
                    "current_state": state.tolist(),
                    "initial_state": initial_state.tolist(),
                    "first_actions": first.tolist(),
                    "left_yaw_mean_rad": float(first[:, 16].mean()),
                    "left_yaw_std_rad": float(first[:, 16].std()),
                    # 0 表示沿已发生运动方向保持当前位姿；-1 表示退回初态。
                    "progress_innovation_mean": float(
                        np.mean(
                            (first[:, 14:28] - state[14:28])
                            @ direction
                            / max(float(direction @ direction), 1e-12)
                        )
                    ),
                }
                results.append(row)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("current_state", "initial_state", "first_actions")
                        }
                    ),
                    flush=True,
                )
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
