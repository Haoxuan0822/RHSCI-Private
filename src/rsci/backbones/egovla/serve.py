"""SIMPLE EgoVLA /act 协议：STATE 使用当前状态，返回原生 36 维动作块。"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from .checkpoints import checkpoint_ok, load_trainables


def decode_numpy(value):
    if isinstance(value, dict) and "__numpy__" in value:
        return np.frombuffer(
            base64.b64decode(value["__numpy__"]), dtype=np.dtype(value["dtype"])
        ).reshape(value["shape"])
    return value


def encode_numpy(value):
    value = np.ascontiguousarray(value)
    return dict(
        __numpy__=base64.b64encode(value.tobytes()).decode(),
        dtype=value.dtype.str,
        shape=list(value.shape),
    )


def predict(model, processor, payload):
    import torch

    image = np.asarray(payload["image"]["rgb_head_stereo_left"])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected H×W×3 uint8 RGB")
    batch = processor.inference(image, payload.get("instruction", "finish the task"))
    if model.variant == "STATE":
        state = np.asarray(payload["state"]["states"], dtype=np.float32).reshape(-1)
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError("STATE requires finite 32D current state")
        batch["states"] = torch.from_numpy(state.copy())[None]
    device = next(model.parameters()).device
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        actions = model.sample_actions(**batch).float().cpu().numpy()[0]
    if actions.shape != (30, 36) or not np.isfinite(actions).all():
        raise ValueError("Invalid native EgoVLA actions")
    # SIMPLE 原生向量头输出物理动作；不套用 π0.5 quantile 逆归一化或 MANO retarget。
    return actions


def make_handler(model, processor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/info"):
                self.reply(
                    dict(
                        status="ok",
                        model="EgoVLA-" + model.variant,
                        state=model.variant == "STATE",
                        history=False,
                        query=False,
                        horizon=30,
                        action_dim=36,
                        rtc=False,
                    )
                )
            else:
                self.reply(dict(error="not found"), 404)

        def do_POST(self):
            if self.path == "/reset":
                # 无历史、无跨请求动作缓存；reset 不改变模型或随机数。
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.reply(dict(status="ok"))
                return
            if self.path not in ("/act", "/predict"):
                self.reply(dict(error="not found"), 404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 32 * 1024 * 1024:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length), object_hook=decode_numpy)
                if self.path == "/predict":
                    from io import BytesIO

                    from PIL import Image

                    state = payload.get("state")
                    image = np.asarray(
                        Image.open(BytesIO(base64.b64decode(payload["image_b64"]))).convert("RGB")
                    )
                    payload = dict(
                        image=dict(rgb_head_stereo_left=image),
                        instruction=payload.get("task_desc", "finish the task"),
                    )
                    if model.variant == "STATE":
                        payload["state"] = dict(states=state)
                action = predict(model, processor, payload)
                if self.path == "/act":
                    self.reply(
                        dict(
                            action=encode_numpy(action),
                            err=0.0,
                            traj_image=encode_numpy(np.zeros((1, 1, 3), dtype=np.uint8)),
                        )
                    )
                else:
                    self.reply(dict(action=action.tolist()))
            except Exception as exc:
                # 不能用空动作掩盖推理失败；SIMPLE客户端应将非200记为错误。
                self.reply(dict(error=str(exc)), 500)

        def log_message(self, *args):
            pass

    return Handler


def main():
    import torch

    from .checkpoints import sha256
    from .native import load_native
    from .train import CODE, assets, digest_contract

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8011)
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve()
    value = json.loads((checkpoint / "launch_contract.json").read_text())
    if not checkpoint_ok(checkpoint, value["world"], digest_contract(value), verify=True):
        raise ValueError("Incomplete/corrupt EgoVLA checkpoint")
    for name, digest in value["code"].items():
        if sha256(CODE / name) != digest:
            raise ValueError(f"Serving source differs from trained model: {name}")
    base = Path(value["initialization"])
    if assets(base) != value["assets"]:
        raise ValueError("Frozen pretrained assets changed")
    model, processor = load_native(base, vlm_training=value["vlm"], variant=value["model"])
    load_trainables(model, checkpoint)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    print(f"EgoVLA VGQ-AE SIMPLE server {args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), make_handler(model, processor)).serve_forever()


if __name__ == "__main__":
    main()
