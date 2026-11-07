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

Supported variants are STATE, NO-STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-480, VGQ-AE-Mo, VGQ-AE-Mo-240, VGQ-AE-Mo-480, EGQ-AE, VGQ-VLM, EGQ-VLM, VGQ-VLM*, EGQ-VLM* and STATE-RECONSTRUCTION. Nonstarred VLM query variants require `--vlm lora`. Quote starred names to prevent shell expansion. See [Training](../../docs/TRAINING.md) for launcher and runtime options.

VGQ-VLM is now the four-original-image direct VLM Query method, without a Reader; the early unstarred Reader implementation and state-teacher distillation are removed. STATE-RECONSTRUCTION keeps the four-frame VGQ-AE Reader/AE path and predicts 31 measured joints only as labels. Both use current targets and their fixed architecture, without `--no-reader` / `--ardy-offset`. Existing DIRECT-named saved runs are accepted by the saved-config loader; early same-named Reader weights are incompatible.

VGQ-AE-240/480 change visual sampling to `[-12, -8, -4, 0]` / `[-24, -16, -8, 0]`. VGQ-AE-Mo and Mo-240 use a default 120 ms backward difference; Mo-480 uses 240 ms. All these variants retain the Reader and current ARDY target. Other Reader Query variants accept `--no-reader` and `--ardy-offset N`, independently or together. See [experimental switches](../../docs/TRAINING.md#experimental-switches) for defaults and episode boundaries. State robustness evaluation uses shared [client measurement perturbations](../../docs/INFERENCE.md#state-measurement-noise); training and server-side measurement noise have been removed.

## Joint finetuning

The PSI0 launcher supports the upstream `MultiLeRobotDataset` and optional task-weighted sampling. With explicit repo lists, `--task` names the run rather than selecting its only dataset:

```bash
bash scripts/train/psi0.sh --model VGQ-AE-Mo-240 --task joint \
  --data-root /datasets/simple --train-repo-ids task_a task_b \
  --val-repo-ids validation_a validation_b --sidecar-root /datasets/ardy \
  --task-sample-weights 'trash=4' 'faucet=2' --dry-run
```

All repo IDs are local directory names under `--data-root`. When training and validation together reference more than one repo, sidecars live at `--sidecar-root/<repo_id>/`; each must match its own dataset. Labels are selected using the native sample's dataset index and local episode/frame indices. Single-task sidecar paths retain their existing meaning. Datasets must share the sampling frequency and compatible input/action schemas. The optional RGB frame cache currently supports single-dataset loading only.

Sampling rules match case-insensitive substrings of each frame's **task description**, not repo names. Multiple matches use the largest matching weight; unmatched frames receive weight 1. Sampling uses replacement and one dataset-length draw sequence per epoch, followed by Accelerate sharding. It does not automatically balance tasks or datasets. Omit the rules for ordinary shuffled training. Validation uses fixed ordering, no weighted sampling and no augmentation by default.

Omitted validation lists use the first training repo, as upstream; this does not provide an independent validation split. Select the intended validation data explicitly.

For multiple training repos, omitted `--stats-path` now generates shared **bounds** at `<output-dir>/normalization/shared_bounds.json` before asset checking or training. Each action/state dimension uses the smallest minimum and largest maximum across the training repos' prepared `meta/stats_psi0.json` files. This exactly merges their bounds, includes no validation data and does not average ranges or quantiles. All tasks use the same scale. Source hashes are recorded; conflicting existing shared statistics are rejected rather than overwritten. `--dry-run` only prints the planned path and writes nothing.

The native run configuration embeds the resulting bounds. Inference uses those saved values without requiring the training datasets or generated JSON. Single-task statistics remain unchanged, and an explicit `--stats-path` takes precedence. This automatic shared-bounds default differs from upstream's first-repo fallback; it does not change existing checkpoints. RHSCI visual sampling, Reader, ARDY targets and motion losses remain those of the selected variant; this interface does not add joint-training options to the other backbone launchers.

## Inference

Use `scripts/deploy/serve_psi0_simple.sh` to serve a checkpoint and the official SIMPLE runtime to evaluate it. `bash scripts/deploy/serve_psi0_simple.sh --help` lists server arguments. See [Inference](../../docs/INFERENCE.md) for the four-frame client protocol required by starred variants.
