# Repository structure

The layout follows [PSI0](https://github.com/physical-superintelligence-lab/Psi0): native implementations live under `src`, backbone guides under `baselines`, and executable entry points under `scripts`. RHSCI integrations are grouped in `src/rsci`; each backbone keeps its own model dependencies and training recipe.

## Directory tree

```text
RHSCI/
├── README.md                    # Overview; PSI0/GR00T install → train → evaluate examples
├── LICENSE                      # Root license; components retain their own notices
├── THIRD_PARTY_NOTICES.md        # Component license scopes and restrictions
├── licenses/                    # Complete shared license texts
├── PUBLIC_EXPORT_MANIFEST.json  # Public source inventory and SHA256 digests
├── pyproject.toml               # Root package and dependency groups
├── uv.lock                      # PSI0 environment lock
├── ruff.toml                    # Style checks for maintained RHSCI code
├── .env.sample                  # Template for local code and runtime paths
├── baselines/                   # Per-backbone usage, dependencies and source records
│   ├── psi0/                    # Ψ0
│   ├── pi05/                    # π0.5
│   ├── gr00t/                   # GR00T
│   ├── egovla/                  # EgoVLA
│   ├── lambda0/                 # λ₀, including bundle preparation and serving wrappers
│   ├── act/                     # ACT native and RHSCI training
│   ├── dp/                      # Diffusion Policy
│   ├── internvla-m1/            # InternVLA-M1
│   ├── h-rdt/                   # H-RDT
│   ├── dreamzero/               # DreamZero
│   └── beingh05/                # Being-H0.5 native training
├── configs/train/              # Native ACT, DP and Being-H0.5 argument presets
├── scripts/
│   ├── train.py                # Upstream PSI0/ACT/DP training driver
│   ├── train/                  # Common launcher, backbone wrappers and launch configs
│   ├── deploy/                 # Ψ0 SIMPLE serving entry
│   ├── data/                   # SIMPLE statistics, ARDY tools and native conversion tools
│   ├── deepspeed/              # Native distributed-training configurations
│   └── with_nccl_runtime.py    # Explicit subprocess-only NCCL selection
├── src/
│   ├── rsci/                   # RHSCI integrations and shared utilities; see below
│   ├── psi/                    # Native Ψ0 models, processors, trainers and deployment
│   ├── openpi/                 # Native π0.5
│   ├── gr00t/                  # Native GR00T
│   ├── egovla/                 # Native EgoVLA and its VILA source
│   ├── openlambda/             # Native λ₀ package and source manifest
│   ├── act/                    # Native ACT
│   ├── dp/                     # Native Diffusion Policy
│   ├── InternVLA-M1/           # Native InternVLA-M1
│   ├── h_rdt/                  # Native H-RDT
│   ├── dreamzero/              # Native DreamZero
│   ├── beingh05/               # Official Being-H0.5 source, docs and license
│   ├── fast/                   # Native action tokenizer utilities
│   ├── lerobot_patch/          # Upstream LeRobot compatibility overlay
│   ├── egodex/                 # Native EgoDex data helpers
│   └── egomimic/               # Native EgoMimic data helpers
├── tests/
│   ├── act/                    # Native parity, preprocessing, cache and resume tests
│   ├── egovla/                 # Model, optimizer, data and resume tests
│   ├── lambda0/                # Integration, source, loss and completion tests
│   └── test_*.py               # Ψ0, π0.5, GR00T, serving and shared runtime tests
├── tools/                      # CPU suite selection, local paths and export checks
└── docs/                       # Training, inference, architecture and source guides
```

This is a navigation tree, not an exhaustive file listing. No simulator, robot controller, datasets, checkpoints or virtual environments are bundled. Native directories can retain their upstream documentation and example scripts; public RHSCI entry points are listed below.

## Backbone entry points

All eleven backbones are selectable through `scripts/train/baseline.py`. Use `--mode native` for the native trainer, or `--mode rsci` for an implemented RHSCI adapter. Each row links to its environment and asset requirements.

| Backbone | Native source | RHSCI adapter | Training wrapper |
| --- | --- | --- | --- |
| [Ψ0](../baselines/psi0/README.md) | `src/psi` | `src/rsci/backbones/psi0` | `scripts/train/psi0.sh` |
| [π0.5](../baselines/pi05/README.md) | `src/openpi` | `src/rsci/backbones/pi05` | `scripts/train/pi05.sh` |
| [GR00T](../baselines/gr00t/README.md) | `src/gr00t` | `src/rsci/backbones/gr00t` | `scripts/train/gr00t.sh` |
| [EgoVLA](../baselines/egovla/README.md) | `src/egovla` | `src/rsci/backbones/egovla` | `scripts/train/egovla.sh` |
| [λ₀](../baselines/lambda0/README.md) | `src/openlambda` | `src/rsci/backbones/lambda0` | `scripts/train/lambda0.sh` |
| [ACT](../baselines/act/README.md) | `src/act` | `src/rsci/backbones/act` | `scripts/train/act_rsci.sh`; native: `act.sh` |
| [Diffusion Policy](../baselines/dp/README.md) | `src/dp` | Not implemented | `scripts/train/dp.sh` |
| [InternVLA-M1](../baselines/internvla-m1/README.md) | `src/InternVLA-M1` | Not implemented | `scripts/train/internvla-m1.sh` |
| [H-RDT](../baselines/h-rdt/README.md) | `src/h_rdt` | Not implemented | `scripts/train/h-rdt.sh` |
| [DreamZero](../baselines/dreamzero/README.md) | `src/dreamzero` | Not implemented | `scripts/train/dreamzero.sh` |
| [Being-H0.5](../baselines/beingh05/README.md) | `src/beingh05` | Not implemented | `scripts/train/beingh05.sh` |

The first five wrappers directly launch RHSCI training. ACT has separate native and RHSCI wrappers. The remaining wrappers select native mode through the common launcher. Source availability and a native launcher do not imply a SIMPLE data adapter or RHSCI model implementation.

## RHSCI code

```text
src/rsci/
├── backbones/
│   ├── psi0/        # Reader/full-token queries, future ARDY, state noise and training hooks
│   ├── pi05/        # π0.5 integration, quantile-safe processing and execution profiles
│   ├── gr00t/       # GR00T integration, optimizer partitioning and recomputation
│   ├── egovla/      # Native vector action expert, processor and RHSCI supervision
│   ├── lambda0/     # Native flow/RTC integration, bundles and run contracts
│   └── act/         # Native ACT, no-Reader EGQ-AE and bounded video decoding cache
├── data/            # ARDY sidecars/future targets, state noise and frame utilities
├── inference/       # HTTP discovery, causal history and request-scoped noise
├── runtime/         # Artifact publication, numerical checks, resume and process cleanup
└── paths.py         # Explicit dataset/output arguments and optional NCCL wrapper
```

Within an adapter, `model.py` defines research changes, `data.py` and optional `processing.py` translate native inputs, and `train.py` owns the backbone's recipe. Checkpoint logic remains specific to each native format. `launch.py` is present where a separate guardian is needed. Shared behavior belongs in `runtime`, `data` or `inference`; for example, ACT and EgoVLA share epoch-safe HF recovery and signal-safe worker termination.

Ψ0 also uses hooks in `src/psi` and the upstream `scripts/train.py` driver. λ₀ calls the vendored native trainer through its backend interface. Native source retains upstream style; maintained RHSCI modules use the common Ruff rules. See [Code organization](ARCHITECTURE.md) for development commands.

## Serving and external assets

| Backbone | RHSCI serving entry |
| --- | --- |
| Ψ0 | `scripts/deploy/serve_psi0_simple.sh` |
| π0.5 / GR00T | `python -m rsci.inference.http_server --backbone pi05` / `gr00t` |
| EgoVLA | `python -m rsci.backbones.egovla.serve` |
| λ₀ | `baselines/lambda0/Lambda0_Serve.sh` |
| ACT | `python -m rsci.backbones.act.serve` |

These entries have different checkpoint and environment requirements; complete commands and wire protocols are in [Inference](INFERENCE.md). Being-H0.5 and the other native-only baselines retain their upstream inference APIs. They are not registered as RHSCI HTTP services.

## Following the README examples

The [root README](../README.md#getting-started) contains complete installation, training and evaluation examples for PSI0 VGQ-AE and GR00T box-hugging VGQ-AE. Both use the same SIMPLE G1 training dataset and ARDY labels. Their model environments and checkpoint formats remain separate:

| Stage | PSI0 VGQ-AE | GR00T VGQ-AE |
| --- | --- | --- |
| Install | Root `uv.lock`, groups `psi`, `serve`, `viz`; native LeRobot 0.3.3 layout | `baselines/gr00t/requirements.txt`, pinned PyTorch/FlashAttention and native `src/gr00t` package |
| Train | `scripts/train/psi0.sh` → `src/rsci/backbones/psi0/train.py` → native `scripts/train.py` | `scripts/train/gr00t.sh` → `src/rsci/backbones/gr00t/train.py` |
| Saved run | `run_config.json`, `argv.txt`, `checkpoints/ckpt_40000/` | `launch_contract.json`, `checkpoint-50000/` with model and processor |
| Serve | `scripts/deploy/serve_psi0_simple.sh` → `src/psi/deploy/psi0_serve_simple.py` | `src/rsci/inference/http_server.py` → `src/rsci/backbones/gr00t/serve.py` |
| Evaluate | External SIMPLE `simple.cli.eval_decoupled_wbc`, HTTP client `psi0_decoupled_wbc` | Same client and evaluation split; separate server port and results directory |

The SIMPLE checkout and its simulation environment live outside this repository. The evaluation client uses the server's action response length: 30 for the README's PSI0 configuration and 16 for GR00T. Nonstarred VGQ-AE inference uses current-frame RGB and language, without ARDY labels or training visual history. See the README's parameter table before changing batch sizes, training budgets or action horizons.

[SIMPLE data preparation](DATA_PREPARATION.md) describes normalization generation and the reference-verified sidecar builder's input and task restrictions. `scripts/data/prepare_simple_stats.py` reads the processed numeric action/state columns without decoding images or loading depth.

Keep datasets, ARDY labels, model assets, output runs, caches and environments outside the checkout. `PSI_RUNTIME_ROOT` controls default external paths; explicit dataset, model and output arguments are documented in [Training](TRAINING.md). Root `uv.lock` covers the Ψ0 stack; other backbones use isolated compatible environments.

Historical run directories are not part of the source release. Training recovery checks source/configuration identities and does not silently migrate cross-version runs. Existing Ψ0 and π0.5 inference compatibility is described separately in the inference guide. Component revisions and licenses are listed in [Sources](SOURCES.md).
