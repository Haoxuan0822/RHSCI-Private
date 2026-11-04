"""用真实HTTP预测核验两个独立流的同输入/同噪声输出，不启动仿真。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from rhsci.inference.action_noise import PROTOCOL
from rhsci.inference.protocol import RequestMessage, ResponseMessage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-prefix", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    image = np.asarray(Image.open(args.image))
    session = requests.Session()
    session.trust_env = False
    rows = []
    for index in range(2):
        outputs = []
        for suffix in ("a", "b"):
            stream = f"{args.stream_prefix}-{suffix}"
            metadata = dict(
                protocol=PROTOCOL,
                episode=0,
                inference_index=index,
                episode_seed=args.seed,
                stream=stream,
            )
            request = RequestMessage(
                {"rgb_head_stereo_left": image},
                " pick up the cracker box from table1,locomotion to table2,and place  on table2.",
                {"psi_eval_deterministic": metadata},
                {"states": np.zeros((1, 32), np.float32)},
                {},
                [],
                "grasp",
                "diagnostic",
            )
            # 首次原生编译可超过五分钟；超时不代表服务器已停止，不自动重试。
            response = session.post(
                f"http://127.0.0.1:{args.port}/act", json=request.serialize(), timeout=1800
            )
            response.raise_for_status()
            action = ResponseMessage.deserialize(response.json()).action
            if action.shape != (30, 36) or not np.isfinite(action).all():
                raise ValueError("Invalid real model response")
            outputs.append(action)
            rows.append(
                dict(
                    stream=stream,
                    inference_index=index,
                    action_sha256=hashlib.sha256(action.tobytes()).hexdigest(),
                )
            )
        if not np.array_equal(*outputs):
            raise ValueError(f"Paired output differs: {np.max(np.abs(outputs[0] - outputs[1]))}")
    result = dict(
        image=str(args.image.resolve()),
        image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        same_input_paired_streams_bitwise_equal=True,
        requests=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
