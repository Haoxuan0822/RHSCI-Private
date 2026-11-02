"""独立ACT HTTP服务；完整动作块和原生队列，训练统计随断点读取。"""

import argparse
import json
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from act.models.act import ACTConfig
from rhsci.inference.http_app import create_policy_app
from rhsci.inference.protocol import RequestMessage, ResponseMessage
from rhsci.inference.weights import validate_inference_weights

from .checkpoints import load_model_state
from .data import Processor
from .model import RHSCIACT


class Policy:
    def __init__(self, checkpoint, device="cuda:0"):
        checkpoint = Path(checkpoint)
        contract = json.loads((checkpoint / "launch_contract.json").read_text())
        validate_inference_weights(checkpoint)
        saved = json.loads((checkpoint / "rsci_config.json").read_text())
        config = ACTConfig(**{**saved["act"], "pretrained_backbone_weights": None})
        self.model = RHSCIACT(saved["variant"], config)
        self.model.rsci_config = saved
        load_model_state(self.model, checkpoint)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.processor = Processor(contract["normalization"])
        self.task = contract["task"]
        self.lock = threading.Lock()

    def predict(self, image, states=None, *, single_action=False):
        if self.model.variant == "STATE" and states is None:
            raise ValueError(
                "STATE requires current measured32D state; no predicted-waist fallback"
            )
        batch = self.processor.inference(image, states if self.model.variant == "STATE" else None)
        batch = {name: value.to(self.device) for name, value in batch.items()}
        with torch.inference_mode():
            # 原生官方部署为FP32；不擅自把训练bf16推广至推理精度。
            action = (
                self.model.select_action(batch)
                if single_action
                else self.model.predict_action(batch)
            )
            action = self.processor.field.denormalize(action).cpu().numpy()[0]
        if not np.isfinite(action).all():
            raise FloatingPointError("Nonfinite ACT action")
        return action

    def reset(self):
        self.model.reset()


def create_app(policy):
    """为策略对象构造 HTTP 应用；请求处理沿用该策略的动作预测与异常检查。"""

    def health():
        return {
            "status": "ok",
            "model": policy.model.variant,
            "architecture": "ACT-no-language-no-reader",
            "task": policy.task,
            "action_chunk": 16,
            "action_dim": 36,
        }

    def predict_action(payload):
        request = RequestMessage.deserialize(payload)
        if len(request.image) != 1:
            raise ValueError("ACT requires exactly one current RGB camera")
        array = np.asarray(next(iter(request.image.values())))
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Expected H×W×3 uint8 RGB")
        actions = policy.predict(Image.fromarray(array), request.state.get("states"))
        return ResponseMessage(actions, 0.0).serialize()

    return create_policy_app(
        predict_action,
        health=health,
        reset=policy.reset,
        lock=policy.lock,
        error_status=400,
        errors=(Exception,),
    )


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8126)
    args = parser.parse_args()
    uvicorn.run(create_app(Policy(args.checkpoint, args.device)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
