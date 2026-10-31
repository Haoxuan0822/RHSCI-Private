"""可训练权重 + 固定基础资产构成完整模型；HF 保存其余全部训练状态。"""

import json
from pathlib import Path

from rsci.runtime.artifacts import atomic_json as atomic_json
from rsci.runtime.artifacts import sha256 as sha256
from rsci.runtime.hf_full_state import (
    MARKER as MARKER,
)
from rsci.runtime.hf_full_state import (
    HFCheckpointProtocol,
)
from rsci.runtime.hf_full_state import (
    finite_state as finite_state,
)
from rsci.runtime.hf_full_state import (
    required as required,
)
from rsci.runtime.hf_full_state import (
    validate_numerics as validate_numerics,
)

# 格式隔离：ACT保存全模型及buffers，EgoVLA只保存可训练权重并绑定基础资产。
_protocol = HFCheckpointProtocol(
    "egovla-trainables-and-base-v1", lambda path: validate_numerics(path)
)
publish = _protocol.publish
checkpoint_ok = _protocol.checkpoint_ok
latest = _protocol.latest
prune = _protocol.prune


def load_trainables(model, path):
    from safetensors.torch import load_file

    state = load_file(str(Path(path) / "model.safetensors"))
    expected = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != set(expected) or any(
        state[name].shape != parameter.shape for name, parameter in expected.items()
    ):
        raise ValueError("Trainable checkpoint keys/shapes do not match this exact model")
    if json.loads(
        (Path(path) / "rsci_config.json").read_text()
    ) != model.rsci_config or not finite_state(state):
        raise ValueError("Checkpoint architecture/numerical mismatch")
    model.load_state_dict(state, strict=False)
