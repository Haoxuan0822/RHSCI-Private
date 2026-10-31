"""Chunked numerical validation of nested tensor state."""

import math


def check_tensors(value, name="state"):
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            for chunk in value.detach().reshape(-1).split(1024 * 1024):
                if not torch.isfinite(chunk).all():
                    raise ValueError("Nonfinite checkpoint tensor: " + name)
    elif isinstance(value, dict):
        for key, child in value.items():
            check_tensors(child, f"{name}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            check_tensors(child, f"{name}/{index}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint scalar: " + name)
