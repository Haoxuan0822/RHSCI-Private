"""Serve recorded expert chunks and probe VGQ-AE on the resulting simulator observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import requests
import uvicorn
from PIL import Image

from openpi.rhsci.data import SimpleRHSCIDataset
from rhsci.inference.http_app import create_policy_app
from rhsci.inference.protocol import RequestMessage, ResponseMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8140)
    parser.add_argument("--compare-port", type=int, default=8126)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = SimpleRHSCIDataset(args.dataset, "STATE")
    stream = (args.output / "expert_requests.jsonl").open("x", buffering=1)
    frame = 0
    session = requests.Session()
    session.trust_env = False

    def predict(payload):
        nonlocal frame
        request = RequestMessage.deserialize(payload)
        if request.history.get("reset"):
            frame = 0
        if frame + 30 > dataset.episodes[0]["length"]:
            raise ValueError("Expert episode exhausted")
        sample = dataset.raw_sample(frame)
        # 模型预测只记录；真正交给原生控制器的是专家动作。
        response = session.post(
            f"http://127.0.0.1:{args.compare_port}/act", json=payload, timeout=120
        )
        response.raise_for_status()
        prediction = ResponseMessage.deserialize(response.json()).action
        if prediction.shape != (30, 36) or not np.isfinite(prediction).all():
            raise ValueError("Invalid comparison prediction")
        Image.fromarray(sample["images"][-1]).save(args.output / f"expert-frame-{frame:04d}.png")
        stream.write(
            json.dumps(
                {
                    "frame": frame,
                    "expert_state": sample["state"].tolist(),
                    "actual_state": request.state["states"][0].tolist(),
                    "expert_actions": sample["actions"].tolist(),
                    "model_actions": prediction.tolist(),
                }
            )
            + "\n"
        )
        print(f"Serving recorded episode0 frames {frame}:{frame + 30}", flush=True)
        frame += 30
        return ResponseMessage(sample["actions"], 0.0).serialize()

    uvicorn.run(create_policy_app(predict), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
