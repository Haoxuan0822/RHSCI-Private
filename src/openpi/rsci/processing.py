"""无需 JAX 的 π0.5 原生 SentencePiece/state 文本与 SIMPLE 绝对动作归一化。"""

import json
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from PIL import Image

from rsci.data.state_noise import InferenceStateNoise, add_measurement_noise, validate_scale


class Processor:
    def __init__(
        self,
        root,
        tokenizer,
        variant,
        *,
        state_measurement_noise_std_rad=0.0,
        normalization="simple-quantile-zero-span-clipped-v2",
    ):
        self.variant = variant
        if normalization not in ("simple-quantile-v1", "simple-quantile-zero-span-clipped-v2"):
            raise ValueError(f"Unsupported π0.5 normalization: {normalization}")
        self.normalization = normalization
        self.state_measurement_noise_std_rad = validate_scale(
            state_measurement_noise_std_rad, variant
        )
        self.inference_state_noise = InferenceStateNoise()
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        self.stats = json.loads((Path(root) / "meta/stats.json").read_text())

    def normalize(self, value, key):
        stats = self.stats[key]
        # 沿用本仓库 SIMPLE/OpenPI 的 quantile 保护；不能修改共享统计数组。
        lo, hi = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
        value = np.asarray(value, np.float32)
        if not (np.isfinite(value).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError(f"Nonfinite raw value or quantile statistics: {key}")
        if np.any(hi < lo):
            raise ValueError(f"Reversed quantile statistics: {key}")
        if (
            getattr(self, "normalization", "simple-quantile-zero-span-clipped-v2")
            == "simple-quantile-v1"
        ):
            return ((value - lo) / (hi - lo + 1e-6) * 2 - 1).astype(np.float32)
        ill = np.abs(hi - lo) < 1e-4 * (np.abs(hi) + np.abs(lo) + 1e-8)
        # 零跨度维度直接置零；正常维度裁剪，避免稀有动作被 1e-6 放大数百万倍。
        span = np.where(ill, 1.0, hi - lo)
        result = np.where(ill, 0.0, (value - lo) / (span + 1e-6) * 2 - 1)
        return np.clip(result, -1, 1).astype(np.float32)

    def __call__(
        self, sample, *, state_inference_measurement_noise_std_rad=0.0, state_noise_seed=None
    ):
        inference_scale = validate_scale(state_inference_measurement_noise_std_rad, self.variant)
        text = sample["text"].strip().replace("_", " ").replace("\n", " ")
        if self.variant == "STATE":
            state = np.array(sample["state"], dtype=np.float32, copy=True)
            if state.shape != (32,):
                raise ValueError("SIMPLE state requires exactly 32 values")
            scale = inference_scale or self.state_measurement_noise_std_rad
            if scale > 0:
                values = torch.from_numpy(state)
                noise = (
                    self.inference_state_noise.sample(values[:31], state_noise_seed)
                    if inference_scale > 0
                    else torch.randn_like(values[:31])
                )
                state = add_measurement_noise(values, noise, scale).numpy()
            state = self.normalize(state, "states")
            discrete = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
            prompt = f"Task: {text}, State: {' '.join(map(str, discrete))};\nAction: "
        else:
            prompt = f"Task: {text};\nAction: "
        encoded = self.tokenizer.encode(prompt, add_bos=True)
        if len(encoded) > 250:
            raise ValueError("Prompt would truncate state/instruction")
        tokens = torch.zeros(250, dtype=torch.long)
        mask = torch.arange(250) < len(encoded)
        tokens[: len(encoded)] = torch.tensor(encoded)
        frames = []
        for frame in sample["images"]:
            image = Image.fromarray(frame)
            ratio = 224 / max(image.size)
            resized = image.resize(
                tuple(max(1, round(size * ratio)) for size in image.size), Image.Resampling.BILINEAR
            )
            canvas = Image.new("RGB", (224, 224))
            canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
            frames.append(
                torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 127.5 - 1
            )
        result = dict(
            images=torch.stack(frames),
            tokens=tokens,
            token_mask=mask,
            actions=torch.from_numpy(self.normalize(sample["actions"], "action")),
        )
        if self.state_measurement_noise_std_rad > 0:
            result["state_noise_std_rad"] = torch.tensor(self.state_measurement_noise_std_rad)
        if self.variant == "VGQ-AE":
            result.update(
                history_valid=torch.as_tensor(sample["rsci_history_valid"]),
                ardy_target=torch.tensor(np.array(sample["rsci_ardy_target"], copy=True)),
                ardy_valid=torch.tensor(bool(sample["rsci_ardy_valid"])),
            )
        return result


def collate(samples):
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
