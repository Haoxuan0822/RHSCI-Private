"""Check eager/compiled inference on recorded inputs without changing live model services."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("Need 12 GiB free for isolated inference comparison")
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    # rotary 的非持久 buffer 不在 checkpoint；沿用正常构造，不能只从 meta 赋权重。
    model = Pi05RHSCI(saved["variant"], compile_inference=False)
    model.load_state_dict(
        load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
    )
    model.to("cuda").eval()
    processor = Processor(
        args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
    )
    compiled = torch.compile(model.sample_actions, mode="max-autotune")
    chunks = [
        json.loads(line)
        for line in args.trace.read_text().splitlines()
        if json.loads(line)["kind"] == "model_chunk"
    ]
    span = np.array(processor.stats["action"]["q99"]) - np.array(processor.stats["action"]["q01"])
    results = []
    for chunk in chunks:
        sample = dict(
            text=args.instruction,
            images=[np.asarray(Image.open(chunk["image_path"]))],
            actions=np.zeros((30, 36), np.float32),
            state=np.asarray(chunk["observed_state"]["states"][0], np.float32),
            rsci_history_valid=np.ones(4, bool),
            rsci_ardy_target=np.zeros(128),
            rsci_ardy_valid=False,
        )
        batch = processor(sample)
        inputs = [batch[key][None].to("cuda") for key in ("images", "tokens", "token_mask")]
        # 固定同一噪声，只比较执行后端，不影响在线服务的随机数或控制流程。
        noise = torch.randn(
            1, 30, 36, device="cuda", generator=torch.Generator("cuda").manual_seed(17)
        )
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            eager = model.sample_actions(*inputs, num_steps=10, noise=noise.clone())
            optimized = compiled(*inputs, num_steps=10, noise=noise.clone())
        difference = (eager - optimized).abs()[0].cpu().numpy()
        arms = difference[:, 14:28] * (span[14:28] + 1e-6) / 2
        row = {
            "step": chunk["step"],
            "normalized_max_abs": float(difference.max()),
            "arms_max_abs_rad": float(arms.max()),
            "arms_max_l2_rad": float(np.linalg.norm(arms, axis=-1).max()),
            "eager_first_arm_action": (
                (eager[0, 0, 14:28].float().cpu().numpy() + 1) / 2 * (span[14:28] + 1e-6)
                + np.array(processor.stats["action"]["q01"])[14:28]
            ).tolist(),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps({"seed": 17, "num_steps": 10, "chunks": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
