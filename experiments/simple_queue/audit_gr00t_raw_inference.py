"""用同一真实权重/记录输入/种子，比较RAW与整理版完整GR00T推理。"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def describe(value):
    """保存实际模型边界张量摘要，不改输入或随机流。"""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if hasattr(value, "items"):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("raw", "current"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if args.implementation == "raw":
        sys.path[:0] = [str(source / "src/gr00t"), str(source / "RSCI/gr00t")]
        from rsci_next.model import register_inference_model

        register_inference_model()
    else:
        sys.path.insert(0, str(source / "src"))

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.deploy.gr00t_serve_simple import Server, ServerConfig
    from psi.deploy.helpers import RequestMessage

    server = Server(
        ServerConfig(
            model_path=str(args.checkpoint.resolve()),
            embodiment_tag=EmbodimentTag.G1_LOCO_DOWNSTREAM,
            device="cuda:0",
            action_exec_horizon=16,
        )
    )
    policy = server.policy.policy
    for configs in (server.modality_configs, policy.modality_configs):
        configs["video"].delta_indices = [0]
    if policy.model.training or policy.processor.training:
        raise ValueError("Expected native eval mode")
    paths = {
        "server": Path(inspect.getfile(Server)).resolve(),
        "model": Path(inspect.getfile(type(policy.model))).resolve(),
        "processor": Path(inspect.getfile(type(policy.processor))).resolve(),
    }
    if any(not path.is_relative_to(source) for path in paths.values()):
        raise ValueError(f"Unexpected source imports: {paths}")
    evidence = {}
    native_action = policy.model.get_action

    def get_action(*positional, **keywords):
        evidence["model_inputs"] = describe({"args": positional, "kwargs": keywords})
        prediction = native_action(*positional, **keywords)
        evidence["normalized_prediction"] = describe(prediction)
        return prediction

    policy.model.get_action = get_action

    def backbone_hook(module, inputs, output):
        evidence["backbone"] = describe(output)

    hook = policy.model.backbone.register_forward_hook(backbone_hook)
    chunks = [
        row
        for line in args.trace.read_text().splitlines()
        if (row := json.loads(line))["kind"] == "model_chunk"
    ]
    reference = json.loads(args.reference.read_text()) if args.reference else None
    reference_actions = (
        np.load(args.reference.with_suffix(".npz"))["actions"] if reference else None
    )
    actions = []
    records = []
    for index in sorted(set(np.linspace(0, len(chunks) - 1, 6).astype(int))):
        chunk = chunks[index]
        image_path = Path(chunk["image_path"])
        if not image_path.is_absolute():
            image_path = args.trace.parent / image_path
        pixels = np.asarray(Image.open(image_path).convert("RGB"))
        state = np.asarray(chunk["observed_state"]["states"], dtype=np.float32)
        for seed in (17, 29, 41):
            evidence.clear()
            torch.manual_seed(seed)
            observation = server._build_observation(
                RequestMessage(
                    image={"rgb_head_stereo_left": pixels},
                    instruction=chunk["instruction"],
                    state={"states": state},
                    history={},
                    condition={},
                    gt_action=[],
                    dataset_name="simple",
                    timestamp="",
                )
            )
            action, _ = server.policy.get_action(observation)
            predicted = server._action_to_psi_format(action)[0, :16]
            if predicted.shape != (16, 36) or not np.isfinite(predicted).all():
                raise ValueError("Expected finite native 16x36 action")
            record = {"step": chunk["step"], "seed": seed, **evidence}
            if reference:
                previous = reference["records"][len(records)]
                if (record["step"], seed) != (previous["step"], previous["seed"]):
                    raise ValueError("Unpaired reference")
                record["comparison"] = {
                    f"{key}_equal": record[key] == previous[key] for key in evidence
                }
                difference = predicted - reference_actions[len(records)]
                record["comparison"].update(
                    physical_action_max_error=float(np.abs(difference).max()),
                    arm_rms_error=float(np.sqrt(np.mean(difference[:, 14:28] ** 2))),
                )
            records.append(record)
            actions.append(predicted)
            print(
                json.dumps({"step": chunk["step"], "seed": seed, **record.get("comparison", {})}),
                flush=True,
            )
    hook.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), actions=np.stack(actions))
    result = {
        "implementation": args.implementation,
        "checkpoint": str(args.checkpoint),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "sources": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in paths.items()
        },
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
