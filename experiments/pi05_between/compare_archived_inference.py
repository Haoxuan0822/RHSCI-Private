"""用真实权重和固定噪声核对归档适配器与当前适配器；不连接在线服务。"""

import argparse
import gc
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from openpi.rhsci.model import Pi05RHSCI
from openpi.rhsci.processing import Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "legacy-source", "trace", "dataset", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("Need 12 GiB free for isolated comparison")
    saved = json.loads((args.checkpoint / "rsci_config.json").read_text())
    sys.path.insert(0, str(args.legacy_source.parent))
    legacy_model = importlib.import_module(f"{args.legacy_source.name}.model")
    legacy_processing = importlib.import_module(f"{args.legacy_source.name}.processing")
    processors = (
        legacy_processing.Processor(args.dataset, args.tokenizer, saved["variant"]),
        Processor(
            args.dataset, args.tokenizer, saved["variant"], normalization=saved["normalization"]
        ),
    )
    instruction = json.loads((args.dataset / "meta/tasks.jsonl").read_text().splitlines()[0])[
        "task"
    ]
    chunks = [
        json.loads(line)
        for line in args.trace.read_text().splitlines()
        if json.loads(line)["kind"] == "model_chunk"
    ]
    batches = []
    for chunk in chunks:
        sample = dict(
            text=chunk.get("instruction", instruction),
            images=[np.asarray(Image.open(chunk["image_path"]))],
            actions=np.zeros((30, 36), np.float32),
            state=np.asarray(chunk["observed_state"]["states"][0], np.float32),
            rsci_history_valid=np.ones(4, bool),
            rsci_ardy_target=np.zeros(128),
            rsci_ardy_valid=False,
        )
        old, new = (processor(sample) for processor in processors)
        for key in ("images", "tokens", "token_mask", "actions"):
            if not torch.equal(old[key], new[key]):
                raise ValueError(f"Processor differs at step {chunk['step']}: {key}")
        batches.append(new)
    predictions = []
    for label, cls in (("archived", legacy_model.Pi05RSCI), ("current", Pi05RHSCI)):
        model = cls(saved["variant"], compile_inference=False, execution=saved["execution"])
        model.load_state_dict(
            load_file(str(args.checkpoint / "model.safetensors")), strict=True, assign=True
        )
        model.to("cuda").eval()
        outputs = []
        for chunk, batch in zip(chunks, batches):
            inputs = [batch[key][None].to("cuda") for key in ("images", "tokens", "token_mask")]
            noise = torch.randn(
                1, 30, 36, device="cuda", generator=torch.Generator("cuda").manual_seed(17)
            )
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
            ):
                output = model.sample_actions(*inputs, num_steps=10, noise=noise)
            outputs.append(output.float().cpu())
            print(f"{label} step={chunk['step']} finished", flush=True)
            del inputs, noise, output
        predictions.append(outputs)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    stats = processors[1].stats["action"]
    span = torch.tensor(
        np.asarray(stats["q99"], np.float32) - np.asarray(stats["q01"], np.float32) + 1e-6
    )
    rows = []
    for chunk, old, new in zip(chunks, *predictions):
        diff = (old - new).abs()
        rows.append(
            dict(
                step=chunk["step"],
                bitwise_equal=torch.equal(old, new),
                normalized_max_abs=diff.max().item(),
                physical_max_abs=(diff * span / 2).max().item(),
            )
        )
    result = dict(
        checkpoint=str(args.checkpoint),
        variant=saved["variant"],
        legacy_source=str(args.legacy_source),
        legacy_model_sha256=hashlib.sha256(
            (args.legacy_source / "model.py").read_bytes()
        ).hexdigest(),
        limitation="Archived STATE-run adapter; not an authenticated VGQ training snapshot. Eager BF16, current installed native backbone; no success-rate claim.",
        processor_equal=True,
        seed=17,
        sampling_steps=10,
        chunks=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
