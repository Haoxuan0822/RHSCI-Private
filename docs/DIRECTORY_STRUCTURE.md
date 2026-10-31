# Repository structure

RHSCI follows Psi0's native package layout. Each model's RHSCI implementation is inside its own package; `src/rsci` contains only shared tools. See the [migration record](NATIVE_PACKAGE_LAYOUT_20261031.md) before using old module names or run contracts.

```text
RHSCI-Private/
├── README.md                         # Installation, training and evaluation examples
├── AGENTS.md                         # Repository collaboration rules
├── pyproject.toml / uv.lock          # Ψ0 package/environment; other models use isolated environments
├── PUBLIC_EXPORT_MANIFEST.json       # Current source inventory and SHA256
├── baselines/<model>/                # Usage, dependencies and model-specific launch helpers
├── configs/
│   ├── train/                       # Native trainer argument presets
│   └── simple/official.json          # Pinned external official SIMPLE source
├── scripts/
│   ├── train/                       # Public backbone launchers and shared launch planning
│   ├── data/                        # Data conversion, ARDY, normalization and cache builders
│   ├── deploy/                      # Ψ0 serving launcher
│   └── deepspeed/                   # Distributed launch configurations
├── src/
│   ├── psi/                         # Ψ0 models, config, data, trainers and deployment
│   │   └── rsci/                    # Ψ0 query/model/trainer extensions and experiment entry
│   ├── openpi/                      # Native π0.5
│   │   └── rsci/                    # π0.5 model/data/processing/training/serving
│   ├── gr00t/gr00t/                 # Native GR00T Python package
│   │   └── rsci/                    # GR00T data/training/serving integration
│   ├── egovla/human_plan/           # Native EgoVLA package (VILA remains in src/egovla/VILA)
│   │   └── rsci/                    # EgoVLA model/processing/training/serving
│   ├── openlambda/lambda0/lambda0/  # Native λ₀ Python package
│   │   └── rsci/                    # λ₀ model/bundles/backend/training/serving
│   ├── act/                         # Native ACT
│   │   └── rsci/                    # ACT model/data/training/serving, without Reader
│   ├── rsci/                        # Cross-model utilities only
│   │   ├── data/                    # ARDY, state noise, raw uint8 caches and frame tools
│   │   ├── runtime/                 # Artifacts, complete checkpoints, resume, numerics and processes
│   │   ├── inference/               # Shared HTTP entry, checkpoint discovery, history and noise
│   │   └── paths.py                 # Shared path/argument helpers and lazy module-location map
│   ├── dp/, InternVLA-M1/, h_rdt/, dreamzero/, beingh05/  # Other native baselines
│   └── fast/, lerobot_patch/, egodex/, egomimic/         # Native supporting packages
├── tests/                           # Shared tests and model regression suites
├── tools/                           # CPU suites, source checks and local path tools
├── docs/                            # Architecture, usage, sources and dated records
└── licenses/                        # Shared license texts; native components retain their licenses
```

The `rsci/` subdirectory within a native package is that model's research extension. It uses the adjacent original model and data implementation; there is no second parallel model tree under `src/rsci`. Algorithms and checkpoint formats remain specific to each model.

## Model entry points

Run shell launchers from the repository root in the corresponding environment. `scripts/train/baseline.py` selects native or implemented RHSCI mode; it does not load all models into one environment.

| Model | Package-local extension | Training launcher | Serving entry |
| --- | --- | --- | --- |
| [Ψ0](../baselines/psi0/README.md) | `psi.rsci` | `scripts/train/psi0.sh` | `scripts/deploy/serve_psi0_simple.sh` → `psi.deploy.psi0_serve_simple` |
| [π0.5](../baselines/pi05/README.md) | `openpi.rsci` | `scripts/train/pi05.sh` | Shared `rsci.inference.http_server --backbone pi05` → `openpi.rsci.serve` |
| [GR00T](../baselines/gr00t/README.md) | `gr00t.rsci` | `scripts/train/gr00t.sh` | Shared HTTP entry `--backbone gr00t` → `gr00t.rsci.serve` |
| [EgoVLA](../baselines/egovla/README.md) | `human_plan.rsci` | `scripts/train/egovla.sh` | `python -m human_plan.rsci.serve` |
| [λ₀](../baselines/lambda0/README.md) | `lambda0.rsci` | `scripts/train/lambda0.sh` | `baselines/lambda0/Lambda0_Serve.sh` → `lambda0.rsci.serve` |
| [ACT](../baselines/act/README.md) | `act.rsci` | `scripts/train/act_rsci.sh` | `python -m act.rsci.serve` |

Other native launchers remain `act.sh`, `dp.sh`, `internvla-m1.sh`, `h-rdt.sh`, `dreamzero.sh`, `beingh05.sh` under `scripts/train`. Native source availability does not imply an implemented RHSCI or SIMPLE adapter.

## Within each model

ACT, π0.5, GR00T and EgoVLA retain `config.py` (recipe/arguments), `preflight.py` (assets/data checks), `contract.py` (source/data/config identity), `train.py` (CLI), `worker.py` (assembly), `trainer.py` (Trainer behavior), `checkpoints.py`, and guardian/launch modules. Model and processing code is alongside those modules. Ψ0 keeps its native trainer hooks; λ₀ keeps its native backend integration. The [2026-10-30 responsibility map](TRAINING_STRUCTURE_20261030.md) describes those duties, with old paths retained as history.

## External resources

Datasets, ARDY labels, weights, checkpoints, logs and environments stay outside the checkout. On this machine, data defaults to `/mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data`, and weights/logs to its sibling `checkpoints`. Explicit CLI paths take precedence.

Official SIMPLE stays external at `$PSI_RUNTIME_ROOT/external/SIMPLE-official-b75c6786`. Its source lock and verification tool remain in `configs/simple/official.json` and `tools/check_simple.py`. No SIMPLE files or dependencies were moved by this reorganization; environment/LFS/rollout limitations are in the [SIMPLE record](SIMPLE_OFFICIAL_20261030.md).

See [Training](TRAINING.md), [Inference](INFERENCE.md), [Sources](SOURCES.md) and [Development](ARCHITECTURE.md) for model-specific commands and validation limits.
