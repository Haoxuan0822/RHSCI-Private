"""用HTTP Range仅读优化器ZIP索引、受限pickle结构和标量计步，不读大型动量。"""

import argparse
import collections
import hashlib
import importlib
import io
import json
import pickle
import struct
import sys
from pathlib import Path
from zipfile import ZipFile

import aiohttp
import fsspec
import torch


def tensor_metadata(storage, offset, size, stride, requires_grad, hooks, metadata=None):
    return dict(storage=storage, offset=offset, shape=list(size), stride=list(stride))


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): "float32",
            ("torch._utils", "_rebuild_tensor_v2"): tensor_metadata,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported pickle class: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, identity):
        if len(identity) != 5 or identity[0] != "storage" or identity[1] != "float32":
            raise ValueError("Unexpected storage descriptor")
        return dict(key=identity[2], device=identity[3], elements=identity[4], dtype="float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract.read_text())
    source_sha = hashlib.sha256((args.training_source / "model.py").read_bytes()).hexdigest()
    if source_sha != contract["code"]["src/rsci_pi05/model.py"]:
        raise ValueError("Training source does not match checkpoint contract")
    sys.path.insert(0, str(args.training_source.parent))
    module = importlib.import_module("rsci_pi05.model")
    if Path(module.__file__).resolve() != (args.training_source / "model.py").resolve():
        raise ValueError("Imported another model")
    with torch.device("meta"):
        model = module.Pi05RSCI("VGQ-AE", compile_inference=False)
    layout = [
        (name, list(value.shape)) for name, value in model.named_parameters() if value.requires_grad
    ]
    expected = json.loads(args.manifest.read_text())["files"]["checkpoint-40000/optimizer.pt"]
    headers = {}
    from huggingface_hub import get_token

    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    members_read, rows = [], []
    with fsspec.open(
        args.url,
        "rb",
        block_size=65536,
        headers=headers,
        client_kwargs={"trust_env": True, "timeout": aiohttp.ClientTimeout(total=60, connect=10)},
    ) as remote:
        if remote.size != expected["bytes"]:
            raise ValueError("Remote optimizer size differs from saved manifest")
        with ZipFile(remote) as archive:
            pickle_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if len(pickle_names) != 1:
                raise ValueError("Need one optimizer pickle member")
            name = pickle_names[0]
            if archive.getinfo(name).file_size > 1024**2:
                raise ValueError("Unexpectedly large optimizer metadata")
            content = archive.read(name)
            members_read.append(
                dict(name=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            )
            payload = MetadataReader(io.BytesIO(content)).load()
            identities = [index for group in payload["param_groups"] for index in group["params"]]
            if len(identities) != len(layout) or set(identities) != set(payload["state"]):
                raise ValueError("Optimizer parameter inventory differs from training source")
            prefix = name.removesuffix("data.pkl")
            for (parameter, shape), identity in zip(layout, identities, strict=True):
                state = payload["state"][identity]
                if any(state[key]["shape"] != shape for key in ("exp_avg", "exp_avg_sq")):
                    raise ValueError(f"Moment shape differs: {parameter}")
                step = state["step"]
                if step["shape"] != [] or step["offset"] != 0 or step["storage"]["elements"] != 1:
                    raise ValueError("Expected a scalar float32 step")
                member = prefix + "data/" + step["storage"]["key"]
                if archive.getinfo(member).file_size != 4:
                    raise ValueError("Unexpected step storage size")
                value = archive.read(member)
                members_read.append(
                    dict(name=member, bytes=4, sha256=hashlib.sha256(value).hexdigest())
                )
                rows.append(
                    dict(
                        name=parameter,
                        shape=shape,
                        slot=identity,
                        step=struct.unpack("<f", value)[0],
                    )
                )
    report = dict(
        url=args.url,
        expected_whole_file=expected,
        actual_remote_size=remote.size,
        model_source_sha256=source_sha,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trainable_parameter_tensors=len(layout),
        all_shapes_match=True,
        all_step_40000=all(row["step"] == 40000 for row in rows),
        selected_member_bytes=sum(item["bytes"] for item in members_read),
        members_read=members_read,
        rows=rows,
        param_groups=payload["param_groups"],
        limitations=[
            "Whole optimizer SHA is a manifest expectation, not verified by partial reads.",
            "Range transport may also fetch adjacent blocks and ZIP central directory.",
            "No momentum payloads, intermediate checkpoints, training or parameter updates read/executed.",
            "Final step counters and layout do not prove correct original optimizer restoration.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("members_read", "rows", "param_groups")
            }
        )
    )


if __name__ == "__main__":
    main()
