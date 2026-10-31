"""骨干声明完整断点元数据，文件/SHA/数值与轮换逻辑复用公共实现。"""

from rsci.runtime.artifacts import atomic_json as atomic_json
from rsci.runtime.artifacts import sha256 as sha256
from rsci.runtime.hf_full_state import MARKER as MARKER
from rsci.runtime.hf_full_state import HFTrainerCheckpointProtocol

_protocol = HFTrainerCheckpointProtocol(
    ("config.json", "processor_config.json", "statistics.json", "embodiment_id.json")
)
required_files = _protocol.required_files
complete_checkpoint = _protocol.publish
checkpoint_ok = _protocol.checkpoint_ok
latest_checkpoint = _protocol.latest
prune_rolling = _protocol.prune
