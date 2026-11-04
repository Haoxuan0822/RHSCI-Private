"""只读比较历史评测源码/资产/环境锁与当前环境，保存更新差异证据。"""

import argparse
import ast
import difflib
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path


def sha(content):
    return hashlib.sha256(content).hexdigest()


def tree(content):
    class StripDocs(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]
            return node

    return ast.dump(StripDocs().visit(ast.parse(content)))


def initialization(directory):
    """从已记录反馈比较稳定阶段，不生成新状态或执行动作。"""
    rows = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    stabilization = [
        row for row in rows if row["kind"] == "env_step" and row["phase"] == "stabilize"
    ]
    policy = [row for row in rows if row["kind"] == "env_step" and row["phase"] == "policy"]
    if not stabilization:
        return {"status": "initializing"}
    stats = directory / "stats/eval_stats.txt"
    result = re.findall(r"^episode_(\d+):\s*(True|False)\s*$", stats.read_text(), re.M)
    return {
        "trace_sha256": sha((directory / "trace.jsonl").read_bytes()),
        "stabilization_steps": len(stabilization),
        "first_stabilization_arms": stabilization[0]["observed_joints"][15:29],
        "first_stabilization_base": stabilization[0]["qpos"][:7],
        "stable_base": stabilization[-1]["qpos"][:7],
        "policy_steps": len(policy),
        "outcomes": {key: value == "True" for key, value in result},
    }


def noise_pairs(root, original, restored):
    def noise(name):
        prefix = "EVAL_DETERMINISTIC_INFERENCE "
        return [
            json.loads(line[len(prefix) :])
            for line in (root / f"{name}-server.log").read_text().splitlines()
            if line.startswith(prefix)
        ]

    a, b = noise(original), noise(restored)
    return {
        "original": original,
        "restored": restored,
        "common_requests": min(len(a), len(b)),
        "common_noise_equal": bool(a and b)
        and all(
            x["noise"] == y["noise"] and x["noise_seed"] == y["noise_seed"] for x, y in zip(a, b)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    current = runtime / "external/SIMPLE-official"
    bundle = runtime / "hf/RHSCI/dataset-ea9d480f/eval-data"
    source = bundle / "SIMPLE/reproduction/actual-source-main-v2-v5-SIMPLE-WBC.tar.zst"
    metadata = bundle / "_metadata/eval-metadata-and-provenance-20261027.tar.zst"
    with tarfile.open(
        fileobj=io.BytesIO(subprocess.check_output(["/root/miniforge/bin/zstd", "-dc", metadata]))
    ) as archive:
        lock = json.load(archive.extractfile("historical-eval/eval_environment.lock.json"))
    rows, diffs = [], []
    with tarfile.open(
        fileobj=io.BytesIO(subprocess.check_output(["/root/miniforge/bin/zstd", "-dc", source]))
    ) as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if not member.isfile() or not name.endswith(".py"):
                continue
            if name.startswith("SIMPLE/src/simple/"):
                path = current / "src/simple" / name.removeprefix("SIMPLE/src/simple/")
            elif name.startswith("WBC/control/"):
                path = (
                    current
                    / "third_party/decoupled_wbc/control"
                    / name.removeprefix("WBC/control/")
                )
            else:
                continue
            old = archive.extractfile(member).read()
            row = {"member": name, "old_sha256": sha(old), "current_path": str(path)}
            if not path.is_file():
                row["status"] = "missing_current"
            else:
                new = path.read_bytes()
                row["current_sha256"] = sha(new)
                row["status"] = (
                    "identical"
                    if old == new
                    else "ast_equal"
                    if tree(old) == tree(new)
                    else "changed"
                )
                if row["status"] == "changed":
                    diffs.extend(
                        difflib.unified_diff(
                            old.decode().splitlines(True),
                            new.decode().splitlines(True),
                            fromfile=f"historical/{name}",
                            tofile=str(path),
                        )
                    )
            rows.append(row)
    assets = []
    manifest = bundle / "SIMPLE/assets-20261016/assets-manifest.json"
    for row in json.loads(manifest.read_text())["entries"]:
        name = row["path"]
        if row["kind"] != "file" or "/robots/" not in name:
            continue
        if Path(name).suffix not in {".xml", ".urdf", ".yaml", ".yml", ".json"}:
            continue
        path = current / "data" / name.removeprefix("data/simple-assets/")
        digest = sha(path.read_bytes()) if path.is_file() else None
        assets.append(
            dict(
                path=str(path),
                old_sha256=row["sha256"],
                current_sha256=digest,
                equal=digest == row["sha256"],
            )
        )
    controllers = []
    for manifest in (bundle / "SIMPLE/controller-dependencies").glob("*/manifest.json"):
        for row in json.loads(manifest.read_text())["source"]:
            name = row["path"]
            prefix = "eval-code/SIMPLE-upstream-6d106287/"
            if not name.startswith(prefix) or "third_party/decoupled_wbc/" not in name:
                continue
            if Path(name).suffix not in {".onnx", ".yaml", ".yml", ".xml", ".urdf"}:
                continue
            path = current / name.removeprefix(prefix)
            digest = sha(path.read_bytes()) if path.is_file() else None
            controllers.append(
                dict(
                    path=str(path),
                    old_sha256=row["sha256"],
                    current_sha256=digest,
                    equal=digest == row["sha256"],
                )
            )
    versions = json.loads(
        subprocess.check_output(
            [
                str(runtime / ".venv-simple/bin/python"),
                "-c",
                "import importlib.metadata as m,json;print(json.dumps({"
                "d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))",
            ],
            text=True,
        )
    )
    old_versions = dict(lock["snapshot"]["environments"][".venv-simple-eval"]["packages"])
    packages = [
        dict(package=name, old=value, current=versions.get(name))
        for name, value in old_versions.items()
        if versions.get(name) != value
    ]
    report = {
        "scope": "historical artifact comparison; not proof of a successful historical GR00T run",
        "source_archive": {"path": str(source), "sha256": sha(source.read_bytes())},
        "metadata_archive": {"path": str(metadata), "sha256": sha(metadata.read_bytes())},
        "historical_protocol": lock["protocol"],
        "historical_submodules": lock["snapshot"]["submodules"],
        "current_submodules": subprocess.check_output(
            ["git", "-C", str(current), "submodule", "status"], text=True
        ),
        "sources": rows,
        "robot_configs": assets,
        "controller_assets": controllers,
        "package_differences": packages,
    }
    root = runtime / "evals/gr00t-oven-state-diagnostic-20261104"
    names = (
        "control-paired-original",
        "control-paired-historical",
        "control-paired-historical-zero-arms",
        "initial-only-ep0",
        "initial-only-ep1",
    )
    report["initialization_diagnostics"] = {
        name: initialization(root / name)
        for name in names
        if (root / name / "trace.jsonl").exists()
    }
    report["noise_pairs"] = [
        noise_pairs(root, original, restored)
        for original, restored in (
            ("control-paired-historical", "control-paired-historical-zero-arms"),
            ("control-paired-original", "initial-only-ep0"),
            ("paired-ep1-original", "initial-only-ep1"),
        )
        if (root / f"{restored}-server.log").exists()
    ]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".diff").write_text("".join(diffs))
    print(
        json.dumps(
            {
                "sources": len(rows),
                "source_changes": sum(x["status"] == "changed" for x in rows),
                "robot_configs": len(assets),
                "robot_differences": sum(not x["equal"] for x in assets),
                "controller_assets": len(controllers),
                "controller_differences": sum(not x["equal"] for x in controllers),
                "package_differences": len(packages),
            }
        )
    )


if __name__ == "__main__":
    main()
