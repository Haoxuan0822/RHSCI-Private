"""仅在进程内修正场景USD的失效props引用，保留官方文件与材质内容。"""

import argparse
import hashlib
import json
from pathlib import Path

_LAYERS = []


def install_scene_reference_repair(data_dir, scene_root, output):
    """沿用原生Kit初始化顺序，在场景加载前修正进程内引用。"""
    from simple.envs.base_dual_env import BaseDualSim

    initialize = BaseDualSim._init_isaac

    def initialize_and_repair(self, headless: bool, webrtc: bool = False):
        initialize(self, headless, webrtc)
        repair_scene_references(data_dir, scene_root, output)

    BaseDualSim._init_isaac = initialize_and_repair


def reference_site_count(layer):
    """计数引用位置；两个位置可引用同一文件，不能只数去重文件名。"""
    from pxr import Sdf

    count = 0

    def visit(path):
        nonlocal count
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            # Isaac 4.5附带的USD没有GetAppliedItems；逐项计数原始list-op。
            references = spec.referenceList
            count += sum(
                len(getattr(references, field))
                for field in (
                    "explicitItems",
                    "addedItems",
                    "prependedItems",
                    "appendedItems",
                    "deletedItems",
                    "orderedItems",
                )
            )

    layer.Traverse("/", visit)
    return count


def repair_scene_references(data_dir, scene_root, output):
    """按初态中的scene name定位原包，仅重定向失效引用到本场景同名原文件。"""
    from pxr import Sdf

    names = set()
    for line in (Path(data_dir) / "meta/episodes.jsonl").read_text().splitlines():
        config = json.loads(line)["environment_config"]
        config = json.loads(config) if isinstance(config, str) else config
        names.add(config["layout"]["scene"]["name"])
    records = []
    for name in sorted(names):
        scene = Path(scene_root) / name
        usd = scene / f"{name}.usd"
        before = hashlib.sha256(usd.read_bytes()).hexdigest()
        layer = Sdf.Layer.FindOrOpen(str(usd))
        references = list(layer.GetExternalReferences())
        site_count = reference_site_count(layer)
        changes = []
        for reference in references:
            resolved = Path(reference) if Path(reference).is_absolute() else usd.parent / reference
            if resolved.exists() or "/props/" not in reference:
                continue
            target = scene / "props" / reference.rsplit("/props/", 1)[1]
            original_name = target.stem
            case_variant = False
            if not target.exists():
                matches = [
                    p
                    for p in target.parent.iterdir()
                    if p.name.casefold() == target.name.casefold()
                ]
                if len(matches) == 1:
                    candidate = Sdf.Layer.FindOrOpen(str(matches[0]))
                    material = candidate.GetPrimAtPath(f"/World/Looks/{original_name}")
                    if material and material.typeName == "Material":
                        target, case_variant = matches[0], True
            if not target.is_file() or not target.resolve().is_relative_to(scene.resolve()):
                raise ValueError(f"No exact scene-local original for {reference}: {target}")
            if not layer.UpdateExternalReference(reference, str(target.resolve())):
                raise RuntimeError(f"Could not update in-memory USD reference: {reference}")
            changes.append(
                dict(
                    original=reference,
                    target=str(target),
                    target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    filename_case_variant=case_variant,
                )
            )
        if reference_site_count(layer) != site_count:
            raise ValueError("Reference count changed")
        if hashlib.sha256(usd.read_bytes()).hexdigest() != before:
            raise ValueError("Official USD on disk changed")
        # USD Layer缓存为弱引用；持续持有，确保原生加载器复用该进程内修正。
        _LAYERS.append(layer)
        records.append(
            dict(
                scene=name,
                source_sha256=before,
                original_unique_reference_count=len(references),
                reference_site_count=site_count,
                changes=changes,
            )
        )
    Path(output).write_text(json.dumps(records, indent=2) + "\n")
    print(
        "SCENE_REFERENCES_REPAIRED_IN_MEMORY",
        sum(len(row["changes"]) for row in records),
        flush=True,
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from pxr import Usd

    result = []
    for index, directory in enumerate(args.data_dir):
        records = repair_scene_references(
            directory, args.scene_root, args.output_dir / f"references-{index}.json"
        )
        for record in records:
            usd = args.scene_root / record["scene"] / f"{record['scene']}.usd"
            stage = Usd.Stage.Open(str(usd))
            prims = list(stage.TraverseAll())
            errors = [str(error) for prim in prims for error in prim.GetPrimIndex().localErrors]
            row = dict(scene=record["scene"], prim_count=len(prims), composition_errors=errors)
            result.append(row)
            print(json.dumps(row), flush=True)
            if errors:
                raise ValueError(f"Scene composition still has errors: {record['scene']}")
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
