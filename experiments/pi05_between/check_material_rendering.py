"""Render four official initial states to check USD composition and material loading."""

# ruff: noqa: E402

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--all-episodes", action="store_true")
args = parser.parse_args()
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
os.environ["SIMPLE_ISAAC_LOG_LEVEL"] = "warning"
sys.argv = [
    sys.argv[0],
    "--/renderer/activeGpu=7",
    "--/renderer/multiGpu/enabled=False",
    "--/plugins/carb.tasking.plugin/threadCount=8",
    "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
]

import gymnasium as gym
import simple.envs  # noqa: F401 -- registers official gym environments
from PIL import Image
from simple.cli.eval_decoupled_wbc import _make_sonic_config

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
TASK = "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
OUT = ROOT / (
    "logs/pi05-between-material-rendering-all40"
    if args.all_episodes
    else "logs/pi05-between-material-rendering"
)
OUT.mkdir(exist_ok=True)
sonic = _make_sonic_config()
sonic["DOMAIN_ID"] = 199
sonic["INTERFACE"] = "lo"
env = gym.make(
    "simple/" + TASK,
    sim_mode="mujoco_isaac",
    headless=True,
    sonic_config=sonic,
    disable_env_checker=True,
)
results = []


def check_table_bindings(stage):
    from pxr import UsdShade

    records = []
    for actor_name, prim_name in (("table", "table_cuboid"), ("table2", "table2_cuboid")):
        actor = env.unwrapped.isaac.task.layout.actors.get(actor_name)
        if actor is None:
            continue
        expected = getattr(actor, "material", None)
        prim = stage.GetPrimAtPath("/World/workspace/" + prim_name)
        assert prim.IsValid(), prim_name
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        sources = []
        if material:
            for child in material.GetPrim().GetChildren():
                if child.IsA(UsdShade.Shader):
                    shader = UsdShade.Shader(child)
                    asset = shader.GetSourceAsset("mdl")
                    if asset:
                        sources.append(
                            dict(
                                path=asset.path, name=str(shader.GetSourceAssetSubIdentifier("mdl"))
                            )
                        )
        if expected:
            expected_path = Path(expected["path"])
            if not expected_path.is_absolute():
                expected_path = (
                    ROOT
                    / "external/SIMPLE-official/data"
                    / str(expected_path).removeprefix("data/")
                )
            assert any(
                Path(source["path"]).resolve() == expected_path.resolve()
                and source["name"] == expected["name"]
                for source in sources
            ), (actor_name, expected, sources)
        else:
            assert not material, (
                actor_name,
                "unexpected material binding",
                str(material.GetPath()),
            )
        records.append(dict(actor=actor_name, expected=expected, actual=sources))
    return records


try:
    for level, episode in (
        (level, episode)
        for level in ("dr-level-0", "dr-level-1", "dr-level-2", "dr-level-hard")
        for episode in range(10 if args.all_episodes else 1)
    ):
        metadata = (
            ROOT / "SIMPLE-eval-20261103/snapshot-fb67cd9a" / TASK / level / "meta/episodes.jsonl"
        )
        record = json.loads(metadata.read_text().splitlines()[episode])
        config = json.loads(record["environment_config"])
        config["dr_state_dict"]["scene"]["uid"] = config["dr_state_dict"]["scene"]["uid"].replace(
            "102344280", "scene3"
        )
        observation, _ = env.reset(options={"state_dict": config})
        from omni.usd import get_context

        stage = get_context().get_stage()
        # Isaac Sim 4.5 的 USD 尚无 Stage.GetCompositionErrors，逐 prim 汇总。
        errors = sorted(
            {
                str(error)
                for prim in stage.TraverseAll()
                for error in prim.GetPrimIndex().localErrors
            }
        )
        image = observation["head_stereo_left"]
        assert image.ndim == 3 and image.shape[-1] == 3 and image.std() > 1
        bindings = check_table_bindings(stage)
        Image.fromarray(image).save(OUT / f"{level}-{episode:02d}.png")
        results.append(
            dict(
                level=level,
                episode=episode,
                table_bindings=bindings,
                composition_errors=errors,
                image_shape=list(image.shape),
                image_std=float(image.std()),
            )
        )
        print("MATERIAL_RENDER_CHECK", results[-1], flush=True)
        (OUT / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "result.json").write_text(
        json.dumps(
            {"scope": "render/reset diagnostics only; not evaluation episodes", "results": results},
            indent=2,
        )
        + "\n"
    )
    assert len(results) == (40 if args.all_episodes else 4) and not any(
        item["composition_errors"] for item in results
    )
except BaseException:
    # Kit close 可能退出进程，先保留异常，不能让 close 掩盖检查失败。
    traceback.print_exc()
    (OUT / "failure.txt").write_text(traceback.format_exc())
    raise
finally:
    env.close()
