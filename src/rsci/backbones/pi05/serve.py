"""SIMPLE HTTP inference with the RHSCI π0.5 model and training processor."""

import json

from rsci.runtime.artifacts import sha256


class Server:
    def __init__(
        self,
        checkpoint,
        contract,
        dataset,
        tokenizer,
        device="cuda:0",
        *,
        state_inference_measurement_noise_std_rad=0.0,
        state_noise_seed=0,
        deterministic_eval=False,
        eval_seed=0,
    ):
        import torch
        from safetensors.torch import load_file

        from rsci.backbones.pi05.model import Pi05RSCI
        from rsci.backbones.pi05.processing import Processor
        from rsci.data.ardy_sidecar import dataset_content_fingerprint
        from rsci.inference.action_noise import RequestActionNoise
        from rsci.inference.state_noise import RequestStateNoise

        self.action_noise = RequestActionNoise(deterministic_eval, eval_seed)

        self.state_noise = RequestStateNoise(
            state_inference_measurement_noise_std_rad, state_noise_seed, contract["model"]
        )

        if dataset_content_fingerprint(dataset) != contract["dataset"]:
            raise ValueError("π0.5 processor dataset differs from training")
        if sha256(tokenizer) != contract["tokenizer"]:
            raise ValueError("π0.5 tokenizer differs from training")
        saved = json.loads((checkpoint / "rsci_config.json").read_text())
        if (
            saved["variant"] != contract["model"]
            or saved["horizon"] != 30
            or saved["action_dim"] != 36
            or saved["vlm"] != "frozen"
        ):
            raise ValueError("Saved π0.5 architecture differs from the serving contract")
        normalization = saved.get("normalization")
        if normalization is None:
            if contract.get("preprocessing") != "RGB-bilinear-pad224-quantile-no-augmentation-v1":
                raise ValueError(
                    "Historical π0.5 checkpoint has no recognized preprocessing contract"
                )
            normalization = "simple-quantile-v1"
        self.model = Pi05RSCI(
            contract["model"],
            execution=contract.get("execution", "reference"),
            state_measurement_noise_std_rad=contract.get("state_measurement_noise_std_rad", 0.0),
        )
        index = checkpoint / "model.safetensors.index.json"
        files = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        weights = {}
        for name in files:
            weights.update(load_file(str(checkpoint / name)))
        self.model.load_state_dict(weights, strict=True, assign=True)
        for name, tensor in self.model.state_dict().items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite π0.5 weight: {name}")
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.processor = Processor(
            dataset, tokenizer, contract["model"], normalization=normalization
        )
        self.variant = contract["model"]

    def predict_action(self, payload: dict):
        import numpy as np
        import torch
        from fastapi.responses import JSONResponse

        from psi.deploy.helpers import RequestMessage, ResponseMessage

        request = RequestMessage.deserialize(payload)
        if set(request.image) != {"rgb_head_stereo_left"}:
            raise ValueError("π0.5 requires the current left-head RGB image")
        sample = dict(
            text=request.instruction,
            images=[request.image["rgb_head_stereo_left"]],
            actions=np.zeros((30, 36), np.float32),
        )
        if self.variant == "STATE":
            sample["state"] = request.state["states"][0]
        if self.variant == "VGQ-AE":
            # Processor metadata only; sample_actions never executes the auxiliary branch.
            sample.update(
                rsci_history_valid=np.ones(4, bool),
                rsci_ardy_target=np.zeros(128),
                rsci_ardy_valid=False,
            )
        noise = getattr(self, "state_noise", None)
        if noise is not None and noise.scale > 0:
            batch = self.processor(
                sample,
                state_inference_measurement_noise_std_rad=noise.scale,
                state_noise_seed=noise.request_seed(payload),
            )
        else:
            batch = self.processor(sample)
        action_noise = getattr(self, "action_noise", None)
        initial_noise = action_noise.sample(payload, self.device) if action_noise else None
        noise_kwargs = {"noise": initial_noise} if initial_noise is not None else {}
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16, cache_enabled=False),
        ):
            normalized = self.model.sample_actions(
                batch["images"][None].to(self.device),
                batch["tokens"][None].to(self.device),
                batch["token_mask"][None].to(self.device),
                num_steps=10,
                **noise_kwargs,
            )
        stats = self.processor.stats["action"]
        low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
        actions = (normalized[0].float().cpu().numpy() + 1) / 2 * (high - low + 1e-6) + low
        if actions.shape != (30, 36) or not np.isfinite(actions).all():
            raise ValueError("Invalid π0.5 action shape or nonfinite values")
        response = JSONResponse(ResponseMessage(actions[:24].astype(np.float32), 0.0).serialize())
        if action_noise is not None:
            action_noise.commit(payload, initial_noise)
        return response
