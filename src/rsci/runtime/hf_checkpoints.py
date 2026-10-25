"""Numerical checks for HF model shards and Adam state."""


def validate_numerical_state(path, names):
    """只读检查模型与优化器；分块检查大张量，避免创建整模型大小的布尔临时量。"""
    import torch
    from safetensors import safe_open

    from rsci.runtime.checkpoints import check_tensors

    for name in names:
        if name.endswith(".safetensors"):
            with safe_open(path / name, framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    check_tensors(stream.get_tensor(key), key)
    check_tensors(
        torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True), "optimizer"
    )
