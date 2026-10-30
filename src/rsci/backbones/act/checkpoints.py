"""ACT完整模型（包括FrozenBN buffers）和HF完整状态；独立版本契约。"""

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
_protocol = HFCheckpointProtocol("act-full-state-v1", lambda path: validate_numerics(path))
publish = _protocol.publish
checkpoint_ok = _protocol.checkpoint_ok
latest = _protocol.latest
prune = _protocol.prune


def load_model_state(model, path):
    from safetensors.torch import load_file

    state = load_file(str(Path(path) / "model.safetensors"))
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        state[name].shape != value.shape for name, value in expected.items()
    ):
        raise ValueError("ACT full checkpoint keys/shapes mismatch")
    if json.loads(
        (Path(path) / "rsci_config.json").read_text()
    ) != model.rsci_config or not finite_state(state):
        raise ValueError("ACT checkpoint architecture/numerical mismatch")
    model.load_state_dict(state, strict=True)
