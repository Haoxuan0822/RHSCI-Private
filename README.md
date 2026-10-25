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
export GROOT_RSCI_PYTHON="$PSI_RUNTIME_ROOT/.venv-gr00t-rsci/bin/python"
export SIMPLE_HOME="$PSI_RUNTIME_ROOT/SIMPLE"
mkdir -p "$PSI_RUNTIME_ROOT"
```

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
uv pip install --python "$GROOT_RSCI_PYTHON" \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$GROOT_RSCI_PYTHON" \
  -r baselines/gr00t/requirements.txt \
  datasets==3.6.0 click==8.1.8 gymnasium==1.2.2 \
  matplotlib==3.10.1 wandb==0.23.0 pyzmq==27.0.1 \
  setuptools wheel packaging ninja
uv pip install --python "$GROOT_RSCI_PYTHON" --no-build-isolation \
  flash-attn==2.7.4.post1
uv pip install --python "$GROOT_RSCI_PYTHON" --no-deps -e ./src/gr00t
"$GROOT_RSCI_PYTHON" -c 'import torch, transformers, flash_attn; print(torch.__version__, transformers.__version__, flash_attn.__version__); assert torch.cuda.is_available()'
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

`--check` validates local assets and one real sample; the second invocation trains. Both runs use seed 292285 and ARDY weight 0.1. GR00T starts from NVIDIA's official `nvidia/GR00T-N1.6-3B` weights and matches the [SIMPLE optimization recipe](baselines/gr00t/README.md#simple-recipe): 50000 updates, global batch 24, learning rate `1e-4`, warmup `0.05` and weight decay `1e-5`.

| Configuration | Ψ0 | GR00T |
| --- | --- | --- |
| VLM tuning | frozen | top4 |
| Global batch | 128 | 24 |
| Training updates | 40000 | 20000 |
| Final checkpoint | `$PSI_RUN/checkpoints/ckpt_40000` | `$GROOT_RUN/checkpoint-20000` |
| Actions per request | 30 | 16 |

Keep the entire run directory. See [training options](docs/TRAINING.md) for other variants, state noise, no-Reader and future ARDY switches.

## Evaluation

### Install SIMPLE

Use a separate simulator environment and the pinned [SIMPLE installation guide](https://github.com/physical-superintelligence-lab/SIMPLE/blob/803db7e4c1f0fef11008c9480ee75b2f9cc0ca36/docs/source/tutorials/installation.md) for system dependencies and Isaac Sim setup:

```bash
git clone https://github.com/physical-superintelligence-lab/SIMPLE.git "$SIMPLE_HOME"
git -C "$SIMPLE_HOME" checkout 803db7e4c1f0fef11008c9480ee75b2f9cc0ca36
git -C "$SIMPLE_HOME" submodule update --init --recursive
cd "$SIMPLE_HOME"
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match
source .venv/bin/activate
bash scripts/install_curobo.sh
python -c 'import simple; print(simple.__version__)'
bash scripts/pre-minimal-download.sh

hf download USC-PSI-Lab/psi-data "simple-eval/$SIM_TASK.zip" \
  --repo-type dataset --local-dir "$PSI_RUNTIME_ROOT/downloads"
mkdir -p "$PSI_RUNTIME_ROOT/data/simple-eval"
unzip "$PSI_RUNTIME_ROOT/downloads/simple-eval/$SIM_TASK.zip" \
  -d "$PSI_RUNTIME_ROOT/data/simple-eval"
deactivate
cd "$PSI_HOME"
```

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
  --data-format lerobot --data-dir "$PSI_RUNTIME_ROOT/data/simple-eval/$SIM_TASK/level-0" \
  --eval-dir "$PSI_RUNTIME_ROOT/evals/psi0/$TRAIN_TASK/VGQ-AE/step-40000/level-0" \
  --episode-start 0 --num-episodes 10 --num-workers 1 --save-video
```

### GR00T

Serve the matching GR00T checkpoint, then evaluate on the same split:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PSI_HOME/src" "$GROOT_RSCI_PYTHON" \
  -m rsci.inference.http_server \
  --backbone gr00t --run-dir "$GROOT_RUN" --checkpoint-step 20000 \
  --device cuda:0 --host 127.0.0.1 --port 8125 --no-deterministic-eval
```

```bash
cd "$SIMPLE_HOME"
CUDA_VISIBLE_DEVICES=1 "$SIMPLE_HOME/.venv/bin/python" -m simple.cli.eval_decoupled_wbc \
  "simple/$SIM_TASK" psi0_decoupled_wbc level-0 \
  --host 127.0.0.1 --port 8125 --sim-mode mujoco_isaac --headless \
  --data-format lerobot --data-dir "$PSI_RUNTIME_ROOT/data/simple-eval/$SIM_TASK/level-0" \
  --eval-dir "$PSI_RUNTIME_ROOT/evals/gr00t/$TRAIN_TASK/VGQ-AE/step-20000/level-0" \
  --episode-start 0 --num-episodes 10 --num-workers 1 --save-video
```

Both use the `psi0_decoupled_wbc` HTTP client and `level-0`, episodes 0–9. Results are written to separate evaluation directories. These commands use ordinary sampling; fixed-noise comparisons require the [request metadata protocol](docs/INFERENCE.md#reproducible-action-sampling). Full GPU training and simulator rollouts have not been validated in this standalone checkout.

## Backbones

| Backbone | RHSCI variants implemented | Guide |
| --- | --- | --- |
| PSI0 | STATE, NO-STATE, VGQ-AE, EGQ-AE, VGQ-VLM, EGQ-VLM, VGQ-VLM*, EGQ-VLM* | [PSI0](baselines/psi0/README.md) |
| π0.5 | STATE, NO-STATE, VGQ-AE | [π0.5](baselines/pi05/README.md) |
| GR00T | STATE, NO-STATE, VGQ-AE | [GR00T](baselines/gr00t/README.md) |
| EgoVLA | STATE, VGQ-AE | [EgoVLA](baselines/egovla/README.md) |
| λ₀ | STATE, VGQ-AE | [λ₀](baselines/lambda0/README.md) |
| ACT | STATE, NO-STATE, EGQ-AE; native training also available | [ACT](baselines/act/README.md) |
| Diffusion Policy | Native training entry; no RHSCI adapter | [DP](baselines/dp/README.md) |
| InternVLA-M1 | Native training entry; no RHSCI adapter | [InternVLA-M1](baselines/internvla-m1/README.md) |
| H-RDT | Native training entry; no RHSCI adapter | [H-RDT](baselines/h-rdt/README.md) |
| DreamZero | Native training entry; no RHSCI adapter | [DreamZero](baselines/dreamzero/README.md) |
| Being-H0.5 | Native training entry; no RHSCI or SIMPLE adapter | [Being-H0.5](baselines/beingh05/README.md) |

## Documentation

[Data preparation](docs/DATA_PREPARATION.md) · [Training](docs/TRAINING.md) · [Inference](docs/INFERENCE.md) · [Repository structure](docs/DIRECTORY_STRUCTURE.md) · [Development](docs/ARCHITECTURE.md)

## Acknowledgements and license

Built on [Ψ0](https://github.com/physical-superintelligence-lab/Psi0) and the native backbone implementations. See [Sources](docs/SOURCES.md) and [Third-party notices](THIRD_PARTY_NOTICES.md). The root [Apache 2.0 notice](LICENSE) does not cover all components; bundled [GR00T N1.6](src/gr00t/LICENSE) and derivatives have non-commercial research restrictions.
