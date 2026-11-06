# PSI0

Native model: `src/psi`. RHSCI adapter: `src/psi/rhsci`.

## Installation

Use Linux, a CUDA-compatible PyTorch environment and [uv](https://docs.astral.sh/uv/). Run from the repository root:

```bash
export PSI_HOME="$PWD"
export PSI_RUNTIME_ROOT=/absolute/path/to/rsci-data
mkdir -p "$PSI_RUNTIME_ROOT"
uv venv --python 3.10 "$PSI_RUNTIME_ROOT/.venv-psi"
source "$PSI_RUNTIME_ROOT/.venv-psi/bin/activate"
uv sync --active --frozen --group psi --group serve --group viz
export PSI0_PYTHON="$PSI_RUNTIME_ROOT/.venv-psi/bin/python"
```

Verify the locked LeRobot 0.3.3 layout. The model's compatibility module selects `lerobot.datasets`; the historical `src/lerobot_patch/common` overlay is not installed into this environment:

```bash
PYTHONPATH="$PSI_HOME/src" "$PSI0_PYTHON" -c 'from psi.data.lerobot.compat import LEROBOT_LAYOUT; assert LEROBOT_LAYOUT == "datasets"; print("LeRobot layout:", LEROBOT_LAYOUT)'
```

Copy `.env.sample` to `.env` and set `PSI_HOME` and `PSI_RUNTIME_ROOT` to the same absolute paths. Keep `.env` untracked. The launchers use offline model loading, so download all initialization assets before running them.

## Assets and training

Prepare a SIMPLE G1 LeRobot dataset with PSI0 normalization statistics (`meta/stats_psi0.json`), matching ARDY sidecars for query variants, a compatible PSI0 VLM checkpoint and an action-expert checkpoint. EGQ variants additionally need a local DINOv2 checkpoint. See [Hugging Face assets](../../docs/ASSETS.md) for dataset and research-checkpoint locations.

Use `"$PSI0_PYTHON" scripts/data/prepare_simple_stats.py --dataset /datasets/my_task` to prepare the normalization files. See [SIMPLE data preparation](../../docs/DATA_PREPARATION.md) for schema checks and the complete ARDY preparation recipe.

Use the [root quickstart](../../README.md#getting-started) for the complete VGQ-AE command. `--data-root` is the parent of the task dataset; `--task` is its exact local repo name. `--sidecar-root` points directly to that task's sidecars. `--dry-run` prints configuration; `--check` validates local assets. Remove the check option to train.

Supported variants are STATE, NO-STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-Mo, EGQ-AE, VGQ-VLM, EGQ-VLM, VGQ-VLM*, EGQ-VLM* and STATE-RECONSTRUCTION. Nonstarred VLM query variants require `--vlm lora`. Quote starred names to prevent shell expansion. See [Training](../../docs/TRAINING.md) for launcher and runtime options.

VGQ-VLM is now the four-original-image direct VLM Query method, without a Reader; the early unstarred Reader implementation and state-teacher distillation are removed. STATE-RECONSTRUCTION keeps the four-frame VGQ-AE Reader/AE path and predicts 31 measured joints only as labels. Both use current targets and their fixed architecture, without `--no-reader` / `--ardy-offset`. Existing DIRECT-named saved runs are accepted by the saved-config loader; early same-named Reader weights are incompatible.

VGQ-AE-240 changes only visual sampling to `[-12, -8, -4, 0]`; VGQ-AE-Mo adds backward-difference supervision. Both retain the Reader and current ARDY target. Other Reader Query variants accept `--no-reader` and `--ardy-offset N`, independently or together. See [experimental switches](../../docs/TRAINING.md#experimental-switches) for defaults and episode boundaries. State robustness evaluation uses shared [client measurement perturbations](../../docs/INFERENCE.md#state-measurement-noise); training and server-side measurement noise have been removed.

## Inference

Use `scripts/deploy/serve_psi0_simple.sh` to serve a checkpoint and the official SIMPLE runtime to evaluate it. `bash scripts/deploy/serve_psi0_simple.sh --help` lists server arguments. See [Inference](../../docs/INFERENCE.md) for the four-frame client protocol required by starred variants.
