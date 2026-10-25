"""RHSCI CPU fixtures for model inputs and training losses."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from psi.trainers.finetune import FinetuneTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))


def vl_inputs(index=1):
    return dict(
        input_ids=torch.full((2, 5), index, dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        pixel_values=torch.zeros(2, 4),
        image_grid_thw=torch.ones(2, 3, dtype=torch.long),
    )


def batch():
    return {
        **vl_inputs(),
        "states": torch.randn(2, 1, 36),
        "actions": torch.randn(2, 2, 2),
        "actions_mask": torch.ones(2, 2, 2),
        "history_vlm_inputs": [vl_inputs(i) for i in (2, 3, 4)],
        "history_valid": torch.ones(2, 4, dtype=torch.bool),
        "ardy_target": torch.zeros(2, 128),
        "ardy_valid": torch.tensor([True, False]),
    }


def trainer(cfg):
    obj = FinetuneTrainer.__new__(FinetuneTrainer)
    obj.cfg = SimpleNamespace(model=cfg)
    obj.device = torch.device("cpu")
    obj.noise_scheduler_name, obj.train_diffusion_steps = "flow", 10
    obj.loss_w = torch.ones(2) / 2
    return obj
