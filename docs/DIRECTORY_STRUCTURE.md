# Repository structure

RHSCI follows Psi0's native package layout. Each model's RHSCI implementation is inside its own package; `src/rhsci` contains only shared tools. See the [migration record](NATIVE_PACKAGE_LAYOUT_20261031.md) before using old module names or run contracts.

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
│   │   └── rhsci/                   # Ψ0 query/model/trainer extensions and experiment entry
│   ├── openpi/                      # Native π0.5
│   │   └── rhsci/                   # π0.5 model/data/processing/training/serving
│   ├── gr00t/                       # Native GR00T Python package
│   │   └── rhsci/                   # GR00T data/training/serving integration
│   ├── human_plan/                  # Native EgoVLA Python package
│   │   └── rhsci/                   # EgoVLA model/processing/training/serving
│   ├── lambda0/                     # Native λ₀ Python package
│   │   └── rhsci/                   # λ₀ model/bundles/backend/training/serving
│   ├── act/                         # Native ACT
│   │   └── rhsci/                   # ACT model/data/training/serving, without Reader
│   ├── rhsci/                       # Cross-model utilities only
│   │   ├── data/                    # ARDY, state noise, raw uint8 caches and frame tools
│   │   ├── runtime/                 # Artifacts, complete checkpoints, resume, numerics and processes
│   │   ├── inference/               # Shared HTTP entry, checkpoint discovery, history and noise
│   │   └── paths.py                 # Shared path/argument helpers and lazy module-location map
│   ├── egovla/, openlambda/          # Upstream dependency/project resources; VILA, configs, collection/alignment
│   ├── dp/, InternVLA-M1/, h_rdt/, dreamzero/, beingh05/  # Other native baselines
│   └── fast/, lerobot_patch/, egodex/, egomimic/         # Native supporting packages
├── tests/                           # Shared tests and model regression suites
├── experiments/                     # Isolated exploratory probes and mechanism studies
├── tools/                           # CPU suites, source checks and local path tools
├── docs/                            # Architecture, usage, sources and dated records
└── licenses/                        # Shared license texts; native components retain their licenses
```

The `rhsci/` subdirectory within a native package is that model's research extension. It uses the adjacent original model and data implementation; there is no second parallel model tree under `src/rhsci`. Algorithms and checkpoint formats remain specific to each model.

<a id="development-boundaries-20261103"></a>

## 后续开发边界（2026-11-03）

用户确定后续训练、推理及正式功能开发以 RHSCI-Private 整理版为准，尽量保持当前框架和目录稳定。新增功能扩展 `src/<模型包>/rhsci/`，跨模型复用放在 `src/rhsci/`，训练和部署继续使用现有入口。必要的框架修改应说明原因、影响及兼容方式，不为单个实验重建一套框架。

尚未确定的探针、离线机制分析等独立工作放在 [`experiments/<主题>/`](../experiments/README.md)，调用现有模型、处理器和训练/推理接口。正式代码和默认入口不反向导入实验；确认纳入的方法再按现有框架接入。每个实验说明目的、调用入口、配置及验证状态，数据、权重、缓存和输出仍放仓库外。

本约定面向后续开发，不自动搬迁 PSI0-RAW 已有独立实验、停止活动训练、替换旧权重配套源码或宣称旧断点已兼容。

## Package depth (2026-11-01)

All six model packages now share the import root `src`: `psi`, `openpi`, `gr00t`, `human_plan`, `lambda0`, `act`. Their extensions consistently live at `src/<package>/rhsci`. The flat-layout change preserved imports; the subsequent naming correction uses `<package>.rhsci` and shared `rhsci`, while checkpoint parameter names remain unchanged; filesystem paths and editable-package discovery were updated. No symlink or duplicate compatibility package is used.

The former extra levels came from copied upstream project roots, not extra network layers: `gr00t/gr00t` was project/package; `egovla/human_plan` was repository/package; `openlambda/lambda0/lambda0` was repository/distribution/package. The Python package roots have been flattened. `src/egovla` retains VILA and its project setup/tools; `src/openlambda` retains collection/alignment tools, assets and native experiment configs. These directories are upstream resources, not additional model import roots. Only VILA still needs its independent `llava` import root at `src/egovla/VILA`.

Existing isolated editable installs should be reinstalled without dependencies from their original project locations when moving this source tree to a persistent environment. Their `pyproject.toml` discovery now points at the flat packages. This round did not install or alter environments; shell/module checks explicitly import this checkout through `PYTHONPATH=src`, adding VILA only for EgoVLA.

## Model entry points

Run shell launchers from the repository root in the corresponding environment. `scripts/train/baseline.py` selects native or implemented RHSCI mode; it does not load all models into one environment.

| Model | Package-local extension | Training launcher | Serving entry |
| --- | --- | --- | --- |
| [Ψ0](../baselines/psi0/README.md) | `psi.rhsci` | `scripts/train/psi0.sh` | `scripts/deploy/serve_psi0_simple.sh` → `psi.deploy.psi0_serve_simple` |
| [π0.5](../baselines/pi05/README.md) | `openpi.rhsci` | `scripts/train/pi05.sh` | Shared `rhsci.inference.http_server --backbone pi05` → `openpi.rhsci.serve` |
| [GR00T](../baselines/gr00t/README.md) | `gr00t.rhsci` | `scripts/train/gr00t.sh` | Shared HTTP entry `--backbone gr00t` → `gr00t.rhsci.serve` |
| [EgoVLA](../baselines/egovla/README.md) | `human_plan.rhsci` | `scripts/train/egovla.sh` | `python -m human_plan.rhsci.serve` |
| [λ₀](../baselines/lambda0/README.md) | `lambda0.rhsci` | `scripts/train/lambda0.sh` | `baselines/lambda0/Lambda0_Serve.sh` → `lambda0.rhsci.serve` |
| [ACT](../baselines/act/README.md) | `act.rhsci` | `scripts/train/act_rhsci.sh` | `python -m act.rhsci.serve` |

Other native launchers remain `act.sh`, `dp.sh`, `internvla-m1.sh`, `h-rdt.sh`, `dreamzero.sh`, `beingh05.sh` under `scripts/train`. Native source availability does not imply an implemented RHSCI or SIMPLE adapter.

## Within each model

ACT, π0.5, GR00T and EgoVLA retain `config.py` (recipe/arguments), `preflight.py` (assets/data checks), `contract.py` (model/data/training configuration), `train.py` (CLI), `worker.py` (assembly), `trainer.py` (Trainer behavior), `checkpoints.py`, and guardian/launch modules. Model and processing code is alongside those modules. Ψ0 keeps its native trainer hooks; λ₀ keeps its native backend integration. The [2026-10-30 responsibility map](TRAINING_STRUCTURE_20261030.md) describes those duties, with old paths retained as history.

## Function names and reading order

Read `train.main` first, then follow the called functions to their owning modules. Names describe the action and its object; matching responsibilities use matching names across models.

| Function | Responsibility and result |
| --- | --- |
| `build_argument_parser` | Return the CLI parser, without loading a model or launching training |
| `resolve_training_paths` | Validate arguments, fill defaults and return data/cache/output paths for ACT, π0.5, GR00T and EgoVLA |
| `build_training_arguments` | Return the model's native trainer arguments; Ψ0 returns its native CLI argument list |
| `build_training_contract` | Record model/data/recipe identity; π0.5 and GR00T also return its digest |
| `compute_contract_digest` | Hash ACT/EgoVLA contract dictionaries for checkpoint validation |
| `run_training_preflight` | Check assets and preprocessing in Ψ0, π0.5 and GR00T; return values are documented per model |
| `create_trainer_class` | Lazily construct and return a Trainer subclass, without creating a model |
| `build_checkpoint_callback` | Return the callback that validates and publishes a complete checkpoint |
| `run_training_worker` | Assemble the model/data/Trainer and execute one worker in ACT, π0.5, GR00T and EgoVLA |
| `run_training_guardian` | Supervise workers and restore only verified complete checkpoints |
| `run_checkpoint_guardian` | Shared π/GR retry loop; adapters supply checkpoint validation and worker commands |
| `run_worker_process` | Launch one isolated worker group, forward termination and reap its process |
| `create_policy_app` | Shared HTTP routes around a prediction callable, optional health/reset and existing error policy |
| `load_raw_lerobot_dataset` | Read raw frame tables and video metadata without importing a model or its Processor |

Ψ0 uses `resolve_training_plan` because it returns config, command, output and rank count. λ₀ uses `build_training_config` and `execute_training` to connect its native resolver/backend. These differences reflect their actual return values and execution flow.

GR00T's `validate_training_data` and EgoVLA's `validate_model_and_worker_preprocessing` live in `preflight.py`, so the CLI shows the execution sequence directly. Shared launch planning uses `build_launch_plan`; HTTP factories use `create_app`. Framework methods such as `forward`, `train` and Trainer hooks retain their required names. CLI flags and persisted checkpoint fields retain their existing names. Old abbreviated Python function names have no aliases; update custom imports when switching versions. The [dated record](NATIVE_PACKAGE_LAYOUT_20261031.md#可读性与函数命名整理2026-10-31) lists validation and compatibility limits.

The [2026-11-02 SIMPLE interface refactor](ARCHITECTURE.md#simple-shared-interfaces-20261102) maps actual shared callers and native model hooks. `rhsci.inference.protocol` owns SIMPLE messages; the old native `psi.deploy.helpers` import reexports the same classes. Distinct native protocols and model algorithms remain separate.

The [2026-11-03 code review](CODE_REVIEW_20261103.md) records current readability and recovery/lifecycle ownership. All four HF adapters use the shared epoch-tail recovery and worker RNG isolation. Read implementation in its owning module; existing CLI reexports remain for compatibility. Ψ/λ now also separate CLI configuration, preflight and execution/guardian duties; see the [SIMPLE-inspired interface map](CODE_REVIEW_20261103.md#参考-simple-的职责与接口整理2026-11-03). Ψ uses `training_config.py` because its existing `config.py` reads saved run configurations. Typed path dictionaries and the named Ψ training plan expose return fields while retaining their existing dictionary/tuple use. The HF internal-loop adapter and native model hooks still require more context.

## External resources

Datasets, ARDY labels, weights, checkpoints, logs and environments stay outside the checkout. On this machine, data defaults to `/mnt/cfs/cqizhh/personal/huangkailun/haoxuan/data`, and weights/logs to its sibling `checkpoints`. Explicit CLI paths take precedence.

Official SIMPLE stays external at `$PSI_RUNTIME_ROOT/external/SIMPLE-official-b75c6786`. Its source lock and verification tool remain in `configs/simple/official.json` and `tools/check_simple.py`. No SIMPLE files or dependencies were moved by this reorganization; environment/LFS/rollout limitations are in the [SIMPLE record](SIMPLE_OFFICIAL_20261030.md).

See [Training](TRAINING.md), [Inference](INFERENCE.md), [Sources](SOURCES.md) and [Development](ARCHITECTURE.md) for model-specific commands and validation limits.

Source hashes and exact environment versions are no longer runtime gates. Inference only needs its model files and processing configuration; see [runtime check policy](RUNTIME_CHECKS_20261031.md).

## RHSCI naming correction (2026-11-01)

Current package directories, imports, public classes and function names use `rhsci` / `RHSCI`. ACT's RHSCI launcher is `scripts/train/act_rhsci.sh`; the shared baseline planner accepts `--mode rhsci`. Existing checkpoint tensor/config keys and artifact names keep their saved spelling. Environment lookup prefers the corresponding `RHSCI` name and falls back to its old `RSCI` name without changing the caller's environment. Historical source records retain their original names.

Validation in this checkout: Ψ0 43, π0.5 37, GR00T 32 and ACT 29 CPU tests passed; 40 shared tests passed, including checkpoint/HTTP/official-client checks and environment fallback. All six module and six shell `--help` entries passed. The λ₀ source audit and its two regression tests passed. EgoVLA passed 28 of 29 tests; its SigLIP test remains blocked by missing `flash_attn`. The λ₀ suite ran three lightweight tests successfully; three model test modules could not import `Qwen3_5Config` from the available Transformers 4.53.2. No environment installation, full-size GPU training, old production checkpoint reload or simulation rollout was performed.
