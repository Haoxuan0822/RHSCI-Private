"""Raw SIMPLE measured-joint noise, before normalization or tokenization."""

import hashlib
import math
import os

MEASURED_STATE_DIM = 31
STATE_NOISE_LOCATION = "raw-measured-joints-radians-0-31-prenorm-v3"


def validate_scale(scale, variant=None):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("State measurement noise must be finite nonnegative radians")
    if scale > 0 and variant is not None and variant != "STATE":
        raise ValueError("State measurement noise requires STATE")
    return float(scale)


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("State noise seed must be an integer in [0, 2**63)")
    return seed


def add_measurement_noise(values, noise, scale):
    import torch

    if values.shape[-1] < 32 or noise.shape != values[..., :MEASURED_STATE_DIM].shape:
        raise ValueError("SIMPLE state requires 31 measured joints and a height command")
    measured = values[..., :MEASURED_STATE_DIM] + noise * validate_scale(scale)
    return torch.cat((measured, values[..., MEASURED_STATE_DIM:]), dim=-1)


class InferenceStateNoise:
    def __init__(self):
        self.generators = {}

    def sample(self, reference, seed=None):
        import torch

        if seed is not None:
            generator = torch.Generator(device=reference.device).manual_seed(validate_seed(seed))
        else:
            key = str(reference.device)
            if key not in self.generators:
                self.generators[key] = torch.Generator(device=reference.device).manual_seed(
                    (torch.initial_seed() ^ 0x5354415445) % 2**63
                )
            generator = self.generators[key]
        return torch.randn(
            reference.shape, device=reference.device, dtype=reference.dtype, generator=generator
        )


class TrainingStateNoise:
    """Worker-local stream; its generator state is not included in training checkpoints."""

    def __init__(self):
        self.generators = {}

    def sample(self, reference, rank=0):
        import torch

        if type(rank) is not int or rank < 0:
            raise ValueError("State noise rank must be a nonnegative global process index")
        worker_seed = torch.initial_seed()
        key = (os.getpid(), worker_seed, rank, str(reference.device))
        if key not in self.generators:
            identity = f"psi0-raw-state-training-v1:{worker_seed}:{rank}".encode()
            seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % 2**63
            self.generators[key] = torch.Generator(device=reference.device).manual_seed(seed)
        return torch.randn(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=self.generators[key],
        )
