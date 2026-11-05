"""只读核对三任务训练归档与部署路径；CPU/meta检查，不占用评测GPU。"""

import ast
import json
from pathlib import Path

import torch
from safetensors import safe_open

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
REPO = Path(__file__).resolve().parents[2]


def methods(path):
    tree = ast.parse(path.read_text())
    return {
        f"{cls.name}.{fn.name}": ast.dump(fn, include_attributes=False)
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef)
    }


def main():
    records = []
    for task in ("oven", "handover", "faucet"):
        variants = {}
        for variant in ("STATE", "VGQ-AE"):
            root = ROOT / f"checkpoints/followup/gr00t-{variant}-{task}"
            checkpoint = root / "checkpoint-20000"
            config = json.loads((checkpoint / "config.json").read_text())
            processor = json.loads((checkpoint / "processor_config.json").read_text())[
                "processor_kwargs"
            ]
            contract = json.loads((root / "launch_contract.json").read_text())
            statistics = json.loads((checkpoint / "statistics.json").read_text())
            saved = root / "source-code/src/gr00t/gr00t/model/gr00t_n1d6"
            comparisons = {}
            for filename, names in {
                "gr00t_n1d6.py": [
                    "Gr00tN1d6.get_action",
                    "Gr00tN1d6ActionHead.get_action",
                    "Gr00tN1d6ActionHead._state_features",
                    "Gr00tN1d6ActionHead._join_state_action",
                ],
                "processing_gr00t_n1d6.py": [
                    "Gr00tN1d6Processor._get_vlm_inputs",
                    "Gr00tN1d6Processor.decode_action",
                ],
            }.items():
                old = methods(saved / filename)
                current = methods(REPO / "src/gr00t/model/gr00t_n1d6" / filename)
                comparisons.update({name: old[name] == current[name] for name in names})
            # 实际权重头与生产尺寸动作头逐键/形状核对，meta上不分配权重内存。
            with torch.device("meta"):
                model = Gr00tN1d6ActionHead(Gr00tN1d6Config(**config))
            expected = {
                f"action_head.{key}": list(value.shape) for key, value in model.state_dict().items()
            }
            actual = {}
            index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
            for filename in set(index["weight_map"].values()):
                with safe_open(checkpoint / filename, framework="pt", device="cpu") as weights:
                    for key in weights.keys():
                        if key.startswith("action_head."):
                            actual[key] = weights.get_slice(key).get_shape()
            mismatch = {
                key: [expected.get(key), actual.get(key)]
                for key in expected.keys() | actual.keys()
                if expected.get(key) != actual.get(key)
            }
            logs = json.loads((checkpoint / "trainer_state.json").read_text())["log_history"]
            losses = {}
            for key in ("action_loss", "ardy_loss"):
                values = [
                    entry[key] for entry in logs if key in entry and entry.get("step", 0) >= 19000
                ]
                losses[key] = sum(values) / len(values) if values else None
            variants[variant] = dict(
                contract={
                    key: contract.get(key)
                    for key in (
                        "dataset_fingerprint",
                        "history_offsets",
                        "vlm",
                        "recipe",
                        "max_steps",
                    )
                },
                processor=processor,
                statistics=statistics,
                inference_ast_matches_saved=comparisons,
                action_head_saved_keys=len(actual),
                action_head_mismatches=mismatch,
                last_1000_steps_mean_losses=losses,
            )
            print(
                task,
                variant,
                "HEAD_KEYS",
                len(actual),
                "MISMATCH",
                len(mismatch),
                "AST_MATCH",
                all(comparisons.values()),
                flush=True,
            )
        left, right = variants["STATE"], variants["VGQ-AE"]
        records.append(
            dict(
                task=task,
                variants=variants,
                action_statistics_identical=(
                    left["statistics"]["g1_loco_downstream"]["action"]
                    == right["statistics"]["g1_loco_downstream"]["action"]
                ),
                processor_different_keys=[
                    key
                    for key in left["processor"].keys() | right["processor"].keys()
                    if left["processor"].get(key) != right["processor"].get(key)
                ],
            )
        )
    output = ROOT / "evals/gr00t-vgq-deployment-audit-20261105.json"
    output.write_text(
        json.dumps(
            dict(
                scope="CPU/meta, saved-source AST and actual tensor headers; no production numerical replay or new rollout",
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print("REPORT", output, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
