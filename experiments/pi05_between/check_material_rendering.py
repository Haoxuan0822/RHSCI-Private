"""Render four official initial states to check USD composition and material loading."""

# ruff: noqa: E402

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
os.environ["SIMPLE_ISAAC_LOG_LEVEL"] = "warning"
sys.argv = [sys.argv[0], "--/renderer/activeGpu=7", "--/renderer/multiGpu/enabled=False"]

import gymnasium as gym
import simple.envs  # noqa: F401 -- registers official gym environments
from PIL import Image
from simple.cli.eval_decoupled_wbc import _make_sonic_config

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
TASK = "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
OUT = ROOT / "logs/pi05-between-material-rendering"
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
try:
    for level in ("dr-level-0", "dr-level-1", "dr-level-2", "dr-level-hard"):
        metadata = (
            ROOT / "SIMPLE-eval-20261103/snapshot-fb67cd9a" / TASK / level / "meta/episodes.jsonl"
        )
        record = json.loads(metadata.read_text().splitlines()[0])
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
        Image.fromarray(image).save(OUT / f"{level}.png")
        results.append(
            dict(
                level=level,
                episode=0,
                composition_errors=errors,
                image_shape=list(image.shape),
                image_std=float(image.std()),
            )
        )
        print("MATERIAL_RENDER_CHECK", results[-1], flush=True)
    (OUT / "result.json").write_text(
        json.dumps(
            {"scope": "render/reset diagnostics only; not evaluation episodes", "results": results},
            indent=2,
        )
        + "\n"
    )
    assert len(results) == 4 and not any(item["composition_errors"] for item in results)
except BaseException:
    # Kit close 可能退出进程，先保留异常，不能让 close 掩盖检查失败。
    traceback.print_exc()
    (OUT / "failure.txt").write_text(traceback.format_exc())
    raise
finally:
    env.close()
