"""SIMPLE episode/request action noise, using the local SplitMix64 protocol."""

import functools
import hashlib
import json
import threading

from rsci.data.state_noise import validate_seed

PROTOCOL = "episode-noise-v1-no-rtc"
RTC_PROTOCOL = "episode-noise-v2-rtc"


def derive_seed(episode_seed, inference_index):
    value = (episode_seed + 0x9E3779B97F4A7C15 * (inference_index + 1)) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def configure_torch():
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


class RequestActionNoise:
    def __init__(self, enabled=False, base_seed=0, *, rtc=False):
        self.enabled = enabled
        self.base_seed = validate_seed(base_seed)
        self.position = (None, 0)
        self.protocol = RTC_PROTOCOL if rtc else PROTOCOL

    def metadata(self, payload):
        meta = payload.get("history", {}).get("psi_eval_deterministic")
        if not isinstance(meta, dict) or meta.get("protocol") != self.protocol:
            raise ValueError(f"Deterministic inference requires {self.protocol} request metadata")
        episode, index, seed = (meta[key] for key in ("episode", "inference_index", "episode_seed"))
        if any(type(value) is not int or value < 0 for value in (episode, index, seed)):
            raise ValueError("Episode, request index and episode seed must be nonnegative integers")
        if seed != self.base_seed + episode:
            raise ValueError("Client and server evaluation seeds differ")
        last_episode, expected = self.position
        if index != (expected if episode == last_episode else 0):
            raise ValueError("Deterministic request indices must start at zero and be consecutive")
        return meta

    def sample(self, payload, device):
        import torch

        if not self.enabled:
            return None
        meta = self.metadata(payload)
        seed, index = meta["episode_seed"], meta["inference_index"]
        generator = torch.Generator(device=device).manual_seed(derive_seed(seed, index))
        return torch.randn((1, 30, 36), device=device, dtype=torch.float32, generator=generator)

    def commit(self, payload, noise):
        if noise is None:
            return
        meta = payload["history"]["psi_eval_deterministic"]
        self.commit_traces(
            meta,
            [
                dict(
                    shape=list(noise.shape),
                    device=str(noise.device),
                    sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                )
            ],
        )

    def commit_traces(self, meta, traces):
        self.position = (meta["episode"], meta["inference_index"] + 1)
        print(
            "EVAL_DETERMINISTIC_INFERENCE",
            json.dumps(
                dict(
                    meta,
                    noise_seed=derive_seed(meta["episode_seed"], meta["inference_index"]),
                    noise=traces,
                )
            ),
            flush=True,
        )


def noise_context(seed, traces):
    """Isolate native Gaussian action draws without changing model sampling code."""
    import torch
    from torch.overrides import TorchFunctionMode

    class NoiseMode(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.generators = {}

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            # Explicit generators belong to independent streams, including state noise.
            if func not in (torch.randn, torch.randn_like) or kwargs.get("generator") is not None:
                return func(*args, **kwargs)
            if func is torch.randn_like:
                source = args[0]
                args = (tuple(source.shape),)
                kwargs.setdefault("device", source.device)
                kwargs.setdefault("dtype", source.dtype)
                kwargs.setdefault("layout", source.layout)
                kwargs.pop("memory_format", None)
                func = torch.randn
            device = torch.device(kwargs.get("device") or torch.get_default_device())
            key = str(device)
            if key not in self.generators:
                self.generators[key] = torch.Generator(device=device).manual_seed(seed)
            kwargs["generator"] = self.generators[key]
            result = func(*args, **kwargs)
            traces.append(
                dict(
                    shape=list(result.shape),
                    device=key,
                    sha256=hashlib.sha256(
                        result.detach().float().cpu().numpy().tobytes()
                    ).hexdigest(),
                )
            )
            return result

    return NoiseMode()


def install_server(server, *, enabled=False, base_seed=0, rtc=False):
    """Wrap only this server instance; validate successful requests before advancing."""
    if not enabled:
        return server
    controller = RequestActionNoise(True, base_seed, rtc=rtc)
    original = server.predict_action
    lock = threading.Lock()

    @functools.wraps(original)
    def predict(payload):
        with lock:
            meta = controller.metadata(payload)
            if rtc and meta["episode"] != controller.position[0]:
                server.previous_action = None
            traces = []
            with noise_context(derive_seed(meta["episode_seed"], meta["inference_index"]), traces):
                response = original(payload)
            body = json.loads(response.body)
            if response.status_code != 200 or not isinstance(body, dict) or "action" not in body:
                raise ValueError("Deterministic inference failed; request index was not advanced")
            if not traces:
                raise ValueError(
                    "No native action noise captured; deterministic sampling unverified"
                )
            controller.commit_traces(meta, traces)
            return response

    server.predict_action = predict
    return server
