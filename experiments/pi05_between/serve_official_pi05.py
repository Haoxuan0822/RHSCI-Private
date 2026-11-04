"""Adapt the unmodified official OpenPI policy to the existing SIMPLE HTTP transport."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import uvicorn
from safetensors.torch import load_model
from serve_single_thread import create_app

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.policy import Policy
from openpi.policies.psi_policy import HfmInputs, HfmOutputs
from openpi.shared.normalize import NormStats
from rhsci.inference.protocol import RequestMessage, ResponseMessage


class OfficialPolicy:
    def __init__(self, checkpoint, stats):
        config = Pi0Config(pi05=True, action_dim=36, action_horizon=30, max_token_len=250)
        model = PI0Pytorch(config)
        load_model(model, str(checkpoint / "model.safetensors"), strict=True)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        raw = json.loads(stats.read_text())
        norm = {
            key: NormStats(
                **{
                    name: np.array(raw[source][name], dtype=np.float32)
                    for name in ("mean", "std", "q01", "q99")
                }
            )
            for key, source in (("state", "states"), ("actions", "action"))
        }
        # 与官方HFM部署同序：输入转换、分位归一化、缩放、状态分词、补维。
        self.policy = Policy(
            model,
            transforms=[
                HfmInputs(config.model_type),
                transforms.Normalize(norm, use_quantiles=True),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(PaligemmaTokenizer(250), discrete_state_input=True),
                transforms.PadStatesAndActions(36),
            ],
            output_transforms=[transforms.Unnormalize(norm, use_quantiles=True), HfmOutputs()],
            sample_kwargs={"num_steps": 10},
            is_pytorch=True,
            pytorch_device="cuda:0",
        )
        print("OFFICIAL_POLICY_READY", transforms.__file__, flush=True)

    def predict_action(self, payload):
        request = RequestMessage.deserialize(payload)
        # 公开训练数据和HfmInputs使用完整32维；不沿用官方客户端的:28截断。
        state = np.asarray(request.state["states"][0], dtype=np.float32)
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid SIMPLE state: {state.shape}")
        observation = {
            "observation/image": request.image["rgb_head_stereo_left"],
            "states": state,
            "prompt": request.instruction,
            "reset": bool(request.history.get("reset", False)),
        }
        with torch.inference_mode():
            action = np.asarray(self.policy.infer(observation)["actions"], dtype=np.float32)
        if action.shape != (30, 36) or not np.isfinite(action).all():
            raise ValueError(f"Invalid official prediction: {action.shape}")
        return ResponseMessage(action, 0.0).serialize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = OfficialPolicy(args.checkpoint, args.stats)
    uvicorn.run(create_app(server), host="127.0.0.1", port=args.port)
