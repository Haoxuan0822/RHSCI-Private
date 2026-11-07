# RHSCI

Rethinking Humanoid State Conditioning in Vision-Language-Action Models.

RHSCI studies state conditioning and visual query supervision across Ψ0, π0.5, GR00T, EgoVLA and λ₀, with ACT as a visual action baseline. VGQ/EGQ variants use four-frame visual histories and ARDY supervision during training. Nonstarred variants need only current RGB and language at inference.

## Getting started

Use Linux, NVIDIA GPUs, Git LFS and [uv](https://docs.astral.sh/uv/getting-started/installation/). Run from the checkout root. Keep data, weights and environments outside it. These examples use the SIMPLE box-hugging task and eight training GPUs. Repeat the path exports in each terminal.

```bash
export PSI_HOME="$PWD"
export PSI_RUNTIME_ROOT=/absolute/path/to/rsci-data
export TRAIN_TASK=G1WholebodyPickAndPlaceAndHugContainerTeleop-v0
export SIM_TASK=G1WholebodyPickAndPlaceAndHugContainerTeleop-v0
export DATA_ROOT="$PSI_RUNTIME_ROOT/data/simple"
export HUG_DATASET="$DATA_ROOT/$TRAIN_TASK"
export HUG_SIDECAR="$PSI_RUNTIME_ROOT/data/simple-ardy-sidecars/$TRAIN_TASK"
export PSI_VLM="$PSI_RUNTIME_ROOT/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
export PSI_AE="$PSI_RUNTIME_ROOT/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"
export GROOT_BASE="$PSI_RUNTIME_ROOT/cache/checkpoints/GR00T-N1.6-3B"
export PSI_RUN="$PSI_RUNTIME_ROOT/runs/psi0/$TRAIN_TASK/VGQ-AE/frozen/readme"
export GROOT_RUN="$PSI_RUNTIME_ROOT/runs/gr00t/$TRAIN_TASK/VGQ-AE/top4/readme"
export PSI0_PYTHON="$PSI_RUNTIME_ROOT/.venv-psi/bin/python"
export GROOT_RHSCI_PYTHON="$PSI_RUNTIME_ROOT/.venv-gr00t-rsci/bin/python"
export SIMPLE_HOME="$PSI_RUNTIME_ROOT/external/SIMPLE-official-b75c6786"
export SIMPLE_EVAL_ROOT="$PSI_RUNTIME_ROOT/SIMPLE-eval-20261103/snapshot-fb67cd9a"
mkdir -p "$PSI_RUNTIME_ROOT"
```

For a clean machine, follow the [fresh-clone training sequence](docs/TRAINING.md#from-a-fresh-clone). Install only the environment for the selected backbone; pure training does not need the SIMPLE simulator.

### Installation

Install Ψ0 from the lockfile:

```bash
uv venv --python 3.10 "$PSI_RUNTIME_ROOT/.venv-psi"
source "$PSI_RUNTIME_ROOT/.venv-psi/bin/activate"
uv sync --active --frozen --group psi --group serve --group viz
PYTHONPATH="$PSI_HOME/src" "$PSI0_PYTHON" -c 'from psi.data.lerobot.compat import LEROBOT_LAYOUT; assert LEROBOT_LAYOUT == "datasets"; print("LeRobot layout:", LEROBOT_LAYOUT)'
if [[ ! -e .env ]]; then cp .env.sample .env; fi
deactivate
```

Edit `.env` so `PSI_HOME` and `PSI_RUNTIME_ROOT` match the exports above. LeRobot 0.3.3 uses its native `datasets` layout; no legacy overlay is needed.

Install GR00T in its own environment; FlashAttention needs a compatible CUDA toolkit and compiler:

```bash
uv venv --python 3.10 "$PSI_RUNTIME_ROOT/.venv-gr00t-rsci"
uv pip install --python "$GROOT_RHSCI_PYTHON" \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$GROOT_RHSCI_PYTHON" \
  -r baselines/gr00t/requirements.txt \
  datasets==3.6.0 click==8.1.8 gymnasium==1.2.2 \
  matplotlib==3.10.1 wandb==0.23.0 pyzmq==27.0.1 \
  setuptools wheel packaging ninja
uv pip install --python "$GROOT_RHSCI_PYTHON" --no-build-isolation \
  flash-attn==2.7.4.post1
uv pip install --python "$GROOT_RHSCI_PYTHON" --no-deps -e ./src/gr00t
"$GROOT_RHSCI_PYTHON" -c 'import torch, transformers, flash_attn; print(torch.__version__, transformers.__version__, flash_attn.__version__); assert torch.cuda.is_available()'
```

### Data and weights

Restore the processed 50 Hz SIMPLE box-hugging dataset to `$HUG_DATASET` and its matching ARDY labels to `$HUG_SIDECAR`. See [Hugging Face assets](docs/ASSETS.md) for our data and trained-checkpoint directories.

Download the initialization weights to `$PSI_VLM`, `$PSI_AE` and `$GROOT_BASE` using the [asset commands](docs/DATA_PREPARATION.md#initialization-weights). The archived data already includes normalization files. For newly prepared data, generate them before building labels:

```bash
"$PSI0_PYTHON" scripts/data/prepare_simple_stats.py --dataset "$HUG_DATASET"
```

See [data preparation](docs/DATA_PREPARATION.md) for schemas, existing-statistics handling and the reference ARDY builder’s task restrictions.

## Training

### Ψ0 VGQ-AE

```bash
psi_train=(
  --model VGQ-AE --vlm frozen --task "$TRAIN_TASK"
  --gpus 0,1,2,3,4,5,6,7 --batch-size 16 --num-workers 8
  --data-root "$DATA_ROOT" --sidecar-root "$HUG_SIDECAR"
  --vlm-path "$PSI_VLM" --ae-path "$PSI_AE"
  --output-dir "$PSI_RUN" --max-steps 40000 --seed 292285
)
bash scripts/train/psi0.sh "${psi_train[@]}" --check
bash scripts/train/psi0.sh "${psi_train[@]}"
```

### GR00T box-hugging VGQ-AE

```bash
groot_train=(
  --model VGQ-AE --vlm top4 --task "$TRAIN_TASK"
  --gpus 0,1,2,3,4,5,6,7 --batch-size 3
  --gradient-accumulation-steps 1 --num-workers 4
  --dataset "$HUG_DATASET" --sidecar "$HUG_SIDECAR"
  --checkpoint "$GROOT_BASE" --output-dir "$GROOT_RUN"
  --max-steps 20000 --seed 292285
)
bash scripts/train/gr00t.sh "${groot_train[@]}" --check
bash scripts/train/gr00t.sh "${groot_train[@]}"
```

`--check` validates local assets and one real sample; the second invocation trains. Both runs use seed 292285 and ARDY weight 0.1. GR00T starts from NVIDIA's official `nvidia/GR00T-N1.6-3B` weights and matches the [SIMPLE optimization recipe](baselines/gr00t/README.md#simple-recipe): 20000 updates, global batch 24, learning rate `1e-4`, warmup `0.05` and weight decay `1e-5`.

| Configuration | Ψ0 | GR00T |
| --- | --- | --- |
| VLM tuning | frozen | top4 |
| Global batch | 128 | 24 |
| Training updates | 40000 | 20000 |
| Final checkpoint | `$PSI_RUN/checkpoints/ckpt_40000` | `$GROOT_RUN/checkpoint-20000` |
| Actions per request | 30 | 16 |

Keep the entire run directory. See [training options](docs/TRAINING.md) for other variants, state noise, no-Reader and future ARDY switches.

### π0.5 training configuration

The reference π0.5 recipe follows the [official Psi0 SIMPLE training guide](https://github.com/physical-superintelligence-lab/Psi0/blob/main/baselines/pi05/README.md). It initializes from the PyTorch-converted `pi05_droid` checkpoint and fine-tunes on the target SIMPLE task.

| Setting | Value |
| --- | --- |
| Initialization | `pi05_droid` (PyTorch) |
| Training updates | 40000 |
| Global batch size | 128 |
| Peak learning rate | `1e-4` |
| Learning-rate schedule | Cosine decay |
| Warmup updates | 1000 |
| Decay updates | 40000 |
| Final learning rate | `1e-8` |
| Action dimension | 36 |

The current RHSCI π0.5 adapter freezes PaliGemma and returns all 30 predicted actions per request. The non-RTC client executes all 30 in the current evaluation configuration. See [π0.5](baselines/pi05/README.md) for RHSCI-specific variants and launch options.

## Evaluation

### Evaluation configuration

The current zero-arm experiments use **official SIMPLE `b75c6786` plus our
task-level zero-arm source changes**, and the **pre-November-2 evaluation ZIPs**
from `USC-PSI-Lab/psi-data@e78fb93cc28912a3031a10b8656d32d7f0a2b867`.
The newer `SIMPLE-eval/fb67cd9a` installation example below is a separate release;
do not use it as the identity of these experiments.

| Component | Current experiment |
| --- | --- |
| Simulator code | SIMPLE `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11`; apply the [zero-arm source diff](configs/simple/zero_arm_tasks.patch) as described in the [version record](docs/EVALUATION.md#simulator-source) |
| Physics / rendering | MuJoCo 3.3.6 / Isaac Sim 4.5.0.0; decoupled WBC, native wall-clock control and navigation |
| Client | SIMPLE `Psi0DecoupledWbcAgent` / `HttpActionClient`, launched through [eval_decoupled_wbc.py](experiments/pi05_between/eval_decoupled_wbc.py); 32D state = 31 measured joints + height command |
| Server | RHSCI [`rhsci.inference.http_server`](src/rhsci/inference/http_server.py), `--backbone pi05` or `gr00t`; SIMPLE HTTP `/act` and `/health`, separate model environment |
| Action execution | No RTC; π0.5 30 predicted / 30 executed, 10 sampling steps; GR00T 16 / 16, 4 sampling steps; BF16 |
| Evaluation data | Old ZIP revision above; levels 0/1/2, 10 initial states × 5 rounds per level; only the first round records complete stereo video |
| Measurement noise | Off by default; optional four kinds × four levels on the client, before normalization; [arguments and magnitudes](docs/INFERENCE.md#state-measurement-noise) |

Task limits and success thresholds are task-specific. The
[evaluation worker](experiments/pi05_between/run_evaluation.py) saves the resolved
configuration in `worker.json` and per-round `effective_eval_config.json`, alongside
the client command in `command.sh`. Configure paths for your machine before using
the worker. Download weights and scene assets separately.

### Install SIMPLE

SIMPLE is pinned to official `b75c6786` (including [PR #29](https://github.com/physical-superintelligence-lab/SIMPLE/pull/29)). Use a separate simulator environment and the pinned [SIMPLE installation guide](https://github.com/physical-superintelligence-lab/SIMPLE/blob/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11/docs/source/tutorials/installation.md) for system dependencies and Isaac Sim setup:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/physical-superintelligence-lab/SIMPLE.git "$SIMPLE_HOME"
GIT_LFS_SKIP_SMUDGE=1 git -C "$SIMPLE_HOME" checkout b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11
GIT_LFS_SKIP_SMUDGE=1 git -c url.https://github.com/.insteadOf=git@github.com: \
  -C "$SIMPLE_HOME" submodule update --init --recursive --depth 1
python3 "$PSI_HOME/tools/check_simple.py" "$SIMPLE_HOME"
cd "$SIMPLE_HOME"
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --all-groups --index-strategy unsafe-best-match
source .venv/bin/activate
bash scripts/install_curobo.sh
python -c 'import simple; print(simple.__version__)'
bash scripts/pre-minimal-download.sh

# Latest pinned evaluation release, including the 2026-11-03 instruction fixes.
python3 "$PSI_HOME/tools/sync_simple_eval.py" \
  --output-root "$PSI_RUNTIME_ROOT/SIMPLE-eval-20261103" --task "$SIM_TASK"
export SIMPLE_EVAL_ROOT="$PSI_RUNTIME_ROOT/SIMPLE-eval-20261103/snapshot-fb67cd9a"
deactivate
cd "$PSI_HOME"
```

The clone commands above are for a fresh external directory. The sync tool keeps each evaluation release in its own directory. Run it without `--task` to fetch all 17 tasks in this snapshot, or add `--verify-only` to check existing files. This snapshot includes instruction fixes absent from the older ZIPs used in the configuration above.

Some tasks in the newer snapshot require success-checker updates absent from SIMPLE `b75c6786`. Use matching task code and evaluation data. The examples below use between-tables / box-hugging; see the [version notes](docs/EVALUATION.md) for task compatibility and controller limitations.

### Ψ0

Serve on GPU 0, then run evaluation in another terminal on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/deploy/serve_psi0_simple.sh \
  --run-dir "$PSI_RUN" --ckpt-step 40000 --policy psi0 \
  --device cuda:0 --host 127.0.0.1 --port 8124 \
  --action-exec-horizon 30 --no-rtc --no-deterministic-eval
```

```bash
cd "$SIMPLE_HOME"
CUDA_VISIBLE_DEVICES=1 "$SIMPLE_HOME/.venv/bin/python" -m simple.cli.eval_decoupled_wbc \
  "simple/$SIM_TASK" psi0_decoupled_wbc level-0 \
  --host 127.0.0.1 --port 8124 --sim-mode mujoco_isaac --headless \
  --data-format lerobot --data-dir "$SIMPLE_EVAL_ROOT/$SIM_TASK/dr-level-0" \
  --eval-dir "$PSI_RUNTIME_ROOT/evals/psi0/$TRAIN_TASK/VGQ-AE/step-40000/eval-fb67cd9a/level-0" \
  --episode-start 0 --num-episodes 10 --num-workers 1 --save-video
```

### GR00T

Serve the matching GR00T checkpoint, then evaluate on the same split:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PSI_HOME/src" "$GROOT_RHSCI_PYTHON" \
  -m rhsci.inference.http_server \
  --backbone gr00t --run-dir "$GROOT_RUN" --checkpoint-step 20000 \
  --device cuda:0 --host 127.0.0.1 --port 8125 --no-deterministic-eval
```

```bash
cd "$SIMPLE_HOME"
CUDA_VISIBLE_DEVICES=1 "$SIMPLE_HOME/.venv/bin/python" -m simple.cli.eval_decoupled_wbc \
  "simple/$SIM_TASK" psi0_decoupled_wbc level-0 \
  --host 127.0.0.1 --port 8125 --sim-mode mujoco_isaac --headless \
  --data-format lerobot --data-dir "$SIMPLE_EVAL_ROOT/$SIM_TASK/dr-level-0" \
  --eval-dir "$PSI_RUNTIME_ROOT/evals/gr00t/$TRAIN_TASK/VGQ-AE/step-20000/eval-fb67cd9a/level-0" \
  --episode-start 0 --num-episodes 10 --num-workers 1 --save-video
```

Both use the `psi0_decoupled_wbc` HTTP client and `level-0`, episodes 0–9. Results are written to separate evaluation directories. These commands use ordinary sampling; fixed-noise comparisons require the [request metadata protocol](docs/INFERENCE.md#reproducible-action-sampling).

## Backbones

| Backbone | RHSCI variants implemented | Guide |
| --- | --- | --- |
| PSI0 | STATE, NO-STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-480, VGQ-AE-Mo, VGQ-AE-Mo-240, VGQ-AE-Mo-480, EGQ-AE, VGQ-VLM, EGQ-VLM, VGQ-VLM*, EGQ-VLM*, STATE-RECONSTRUCTION | [PSI0](baselines/psi0/README.md) |
| π0.5 | STATE, NO-STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-480, VGQ-AE-Mo, VGQ-AE-Mo-240, VGQ-AE-Mo-480 | [π0.5](baselines/pi05/README.md) |
| GR00T | STATE, NO-STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-480, VGQ-AE-Mo, VGQ-AE-Mo-240, VGQ-AE-Mo-480 | [GR00T](baselines/gr00t/README.md) |
| EgoVLA | STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-480, VGQ-AE-Mo, VGQ-AE-Mo-240, VGQ-AE-Mo-480 | [EgoVLA](baselines/egovla/README.md) |
| λ₀ | STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-480, VGQ-AE-Mo, VGQ-AE-Mo-240, VGQ-AE-Mo-480 | [λ₀](baselines/lambda0/README.md) |
| ACT | STATE, NO-STATE, EGQ-AE; native training also available | [ACT](baselines/act/README.md) |
| Diffusion Policy | Native training entry; no RHSCI adapter | [DP](baselines/dp/README.md) |
| InternVLA-M1 | Native training entry; no RHSCI adapter | [InternVLA-M1](baselines/internvla-m1/README.md) |
| H-RDT | Native training entry; no RHSCI adapter | [H-RDT](baselines/h-rdt/README.md) |
| DreamZero | Native training entry; no RHSCI adapter | [DreamZero](baselines/dreamzero/README.md) |
| Being-H0.5 | Native training entry; no RHSCI or SIMPLE adapter | [Being-H0.5](baselines/beingh05/README.md) |

## Documentation

[Data preparation](docs/DATA_PREPARATION.md) · [Training](docs/TRAINING.md) · [Inference](docs/INFERENCE.md) · [Official SIMPLE version](docs/EVALUATION.md) · [Repository structure](docs/DIRECTORY_STRUCTURE.md) · [Development](docs/ARCHITECTURE.md)

## Acknowledgements and license

Built on [Ψ0](https://github.com/physical-superintelligence-lab/Psi0) and the native backbone implementations. See [Sources](docs/SOURCES.md) and [Third-party notices](THIRD_PARTY_NOTICES.md). The root [Apache 2.0 notice](LICENSE) does not cover all components; bundled [GR00T N1.6](src/gr00t/LICENSE) and derivatives have non-commercial research restrictions.
