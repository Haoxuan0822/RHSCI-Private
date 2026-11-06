"""读取Mo归档的模型前缀，复用哈希完全一致的本地处理器，跳过优化器。"""

import hashlib
import io
import json
from pathlib import Path
import time
import tarfile

import requests
import zstandard

from plan import PREPARATION, ROOT

REVISION = "de464d121fcfc6a1056e8b7096cf0a096708d996"
ARCHIVE = "checkpoint/Gr00t/VGQ-AE-Mo/hug/gr00t-motion-hug-handover-20261105-v1/checkpoint-20000.tar.zst"
SIZE = 17828130219
KEY = "gr00t-VGQ-AE-Mo-hug"
STATE = ROOT / "evals/goal-latest-protocol-20261105/mo-inference-download"
URL = f"https://huggingface.co/ds1x/RHSCI/resolve/{REVISION}/{ARCHIVE}"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CachedRanges(io.RawIOBase):
    """只按解压器需要读取压缩前缀；失败重试保留已完成Range。"""

    block = 16 * 1024 * 1024

    def __init__(self):
        self.position = 0
        self.current_index = None
        self.payload = None
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = dict(http="http://127.0.0.1:17890", https="http://127.0.0.1:17890")

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.position >= SIZE:
            return 0
        index, offset = divmod(self.position, self.block)
        if index != self.current_index:
            start = index * self.block
            end = min(start + self.block, SIZE) - 1
            path = STATE / f"range-{index:05d}"
            if not path.exists() or path.stat().st_size != end - start + 1:
                for attempt in range(20):
                    try:
                        response = self.session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 60))
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{SIZE}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("Unexpected Range response")
                        if len(response.content) != end - start + 1:
                            raise ValueError("Incomplete Range")
                        temporary = path.with_suffix(".partial")
                        temporary.write_bytes(response.content)
                        temporary.replace(path)
                        break
                    except (requests.RequestException, ValueError):
                        if attempt == 19:
                            raise
                        time.sleep(min(5 * (attempt + 1), 60))
            self.payload = path.read_bytes()
            self.current_index = index
            progress = dict(time=time.time(), compressed_prefix_bytes=end + 1, archive_bytes=SIZE,
                            current_range=index, revision=REVISION)
            (STATE / "progress.json").write_text(json.dumps(progress) + "\n")
            print("PREFIX_BYTES", end + 1, flush=True)
        count = min(len(buffer), len(self.payload) - offset)
        buffer[:count] = self.payload[offset:offset + count]
        self.position += count
        return count

    def close(self):
        self.session.close()
        super().close()


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    receipt = PREPARATION / f"{KEY}.json"
    if receipt.exists():
        print("ALREADY_PREPARED", flush=True)
        return
    destination = ROOT / "checkpoints/followup" / KEY / "inference"
    checkpoint = destination / "checkpoint-20000"
    checkpoint.mkdir(parents=True, exist_ok=True)
    source = ROOT / "checkpoints/followup/gr00t-VGQ-AE-hug/checkpoint-20000"
    wanted = None
    extracted = set()
    manifest = None
    with CachedRanges() as ranges, zstandard.ZstdDecompressor().stream_reader(ranges) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                name = Path(member.name)
                if name.parts[0] != "checkpoint-20000" or len(name.parts) != 2 or not member.isfile():
                    raise ValueError(f"Unexpected prefix member: {member.name}")
                if name.name == "RSCI_COMPLETE.json":
                    payload = archive.extractfile(member).read()
                    manifest = json.loads(payload)
                    (checkpoint / name.name).write_bytes(payload)
                    records = manifest["files"]
                    wanted = {n for n in records if n.startswith("model") or n in {"config.json", "embodiment_id.json"}}
                    # 元数据由内容哈希认证，不用不同版本的统计或凭空改变归一化。
                    processor = (source / "processor_config.json").read_bytes().replace(b'"VGQ-AE"', b'"VGQ-AE-Mo"')
                    if hashlib.sha256(processor).hexdigest() != records["processor_config.json"]["sha256"]:
                        raise ValueError("Local processor cannot reproduce exact saved content")
                    statistics = source / "statistics.json"
                    if digest(statistics) != records["statistics.json"]["sha256"]:
                        raise ValueError("Local statistics identity differs")
                    (checkpoint / "processor_config.json").write_bytes(processor)
                    (checkpoint / "statistics.json").write_bytes(statistics.read_bytes())
                    continue
                if wanted is None or name.name not in wanted:
                    raise ValueError(f"Refusing to download training-state member: {member.name}")
                target = checkpoint / name.name
                expected = manifest["files"][name.name]
                if not target.exists() or target.stat().st_size != expected["size"] or digest(target) != expected["sha256"]:
                    temporary = target.with_suffix(target.suffix + ".partial")
                    checksum = hashlib.sha256()
                    with archive.extractfile(member) as incoming, temporary.open("wb") as output:
                        while payload := incoming.read(1024 * 1024):
                            output.write(payload)
                            checksum.update(payload)
                    if temporary.stat().st_size != expected["size"] or checksum.hexdigest() != expected["sha256"]:
                        raise ValueError(f"Model member digest differs: {name.name}")
                    temporary.replace(target)
                extracted.add(name.name)
                print("MEMBER_VERIFIED", name.name, flush=True)
                if extracted == wanted:
                    break  # 不请求optimizer及后续压缩数据。
    if wanted is None or extracted != wanted:
        raise ValueError("Incomplete model prefix")
    config = json.loads((checkpoint / "config.json").read_text())
    processor = json.loads((checkpoint / "processor_config.json").read_text())
    action = processor["processor_kwargs"]["modality_configs"]["g1_loco_downstream"]["action"]
    horizon = len(action["delta_indices"])
    if config["rsci_variant"] != "VGQ-AE-Mo" or horizon != 16:
        raise ValueError("Unexpected saved method or action horizon")
    # 只生成部署入口描述，明确不是原训练contract，也不声明完整续训状态。
    contract = dict(model=config["rsci_variant"], valid_action_horizon=horizon,
                    inference_only=True, original_training_contract_sha256=manifest["contract"],
                    provenance="Model/config/statistics verified against saved RSCI_COMPLETE; original training contract not downloaded")
    (destination / "launch_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    info = dict(run_dir=str(destination), revision=REVISION, archive_url=URL,
                horizon=horizon, saved_model=config["rsci_variant"], inference_only=True,
                verification="Per-member SHA256; full training archive intentionally not downloaded",
                files={name: manifest["files"][name] for name in sorted(wanted | {"processor_config.json", "statistics.json"})})
    receipt.write_text(json.dumps(info, indent=2) + "\n")
    print("CHECKPOINT_PREPARED", KEY, flush=True)


if __name__ == "__main__":
    main()
