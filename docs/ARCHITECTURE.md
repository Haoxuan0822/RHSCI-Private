# Code organization

The project name is RHSCI. Shared utilities retain the `rsci` Python namespace, `RSCI_*` environment variables and saved configuration/checkpoint field names. Model extensions now use each native package; the old `rsci.backbones.*` imports were removed. See the [2026-10-31 migration](NATIVE_PACKAGE_LAYOUT_20261031.md) for compatibility boundaries.

Model-specific RHSCI extensions live inside their native packages. `src/rsci` contains shared utilities; importing its root does not load a backbone or optional dependencies. See [Repository structure](DIRECTORY_STRUCTURE.md) for the directory tree and per-backbone entry points.

Use descriptive action names for maintained extension functions, consistently across models. The [function guide](DIRECTORY_STRUCTURE.md#function-names-and-reading-order) maps CLI parsing, path resolution, preflight, contracts, workers and guardians to their owning modules. Keep detailed preflight checks in `preflight.py` and state return values and side effects in critical function docstrings. Native framework method names, CLI options and persisted checkpoint keys follow their original interfaces.

| Directory | Responsibility |
| --- | --- |
| `src/psi/rsci`, `src/openpi/rsci`, `src/gr00t/gr00t/rsci`, `src/egovla/human_plan/rsci`, `src/openlambda/lambda0/lambda0/rsci`, `src/act/rsci` | Model-specific integration, data, training, checkpoints and serving, colocated with each native implementation |
| `src/rsci/data` | ARDY sidecars, dataset provenance, frame caching and worker cleanup |
| `src/rsci/runtime` | Artifact writes, numerical checks, process cleanup and profiling |
| `src/rsci/inference` | SIMPLE control-step RGB history protocol |
| `scripts/train` | Common launcher, shell entry points and distributed launch configuration |
| `baselines/<backbone>` | Usage, dependencies and component integration records |
| `tests` | Model, data, resume, numerical failure, serving and launcher regressions |

Native packages such as `psi`, `openpi`, `gr00t`, `egovla` and `openlambda` retain their source trees under `src`. Each backbone uses its own dependency environment. Shared utilities belong in `src/rsci`; native model dependencies belong in the corresponding adapter.

PSI0 integrates through `RSCIFinetuneTrainer` and native trainer hooks. λ₀ integrates through `RSCITrainingBackend` and explicit server arguments. ACT and EgoVLA share sampler-epoch recovery through `runtime/hf_resume.py`; its compatibility check stays local to each derived Trainer. ACT's bounded decoder/frame cache stays in its adapter because it depends on its native LeRobot pipeline. Being-H0.5 retains official code under `src/beingh05` and uses only the native common launcher. The external SIMPLE client integration is an opt-in wrapper; see [Inference](INFERENCE.md).

ACT and EgoVLA share complete HF checkpoint gates through `runtime/hf_full_state.py`, while their adapter modules retain separate schemas and model loading. Other adapters retain their native checkpoint implementations. Checkpoint contracts validate source, configuration and data identities before resuming. Use a new output directory when changing those contracts.

## Development

Maintained RHSCI code, launch tools and tests use Python 3.10 syntax, four-space indentation, double quotes, sorted imports and a 100-character target line length. `ruff.toml` scopes these checks; native source retains upstream style.

```bash
uvx --from ruff==0.17.2 ruff format --config ruff.toml --check .
uvx --from ruff==0.17.2 ruff check --config ruff.toml .
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_training_cli.py
```

With the corresponding backbone environment active:

```bash
python tools/test_cpu.py --backbone psi0
```

Replace `psi0` with `pi05`, `gr00t`, `egovla`, `lambda0` or `act`. The helper keeps temporary caches outside the checkout and excludes distributed tests. Optional λ₀ asset tests use `RSCI_TEST_LAMBDA0_STATS` and `RSCI_TEST_LAMBDA0_BUNDLE`; missing assets produce skips. CPU tests do not exercise full-model CUDA training or simulator rollouts.

For λ₀ asset tests, `RSCI_TEST_LAMBDA0_BUNDLE` points to the prepared bundle's `train` metadata directory, not its root (for example `/bundles/my_task-safe/train`). `RSCI_TEST_LAMBDA0_STATS` points to its normalization JSON. Set `PSI_RUNTIME_ROOT` to the external runtime root when metadata uses paths relative to its `data` directory.

The HTTP checkpoint and protocol tests run in the π0.5 environment with FastAPI and its test client installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_http_inference.py
```


## 2026-10-30 PSI0-RAW integration

Latest input-cache and training-recovery changes are documented in the [integration record](PSI0_RAW_INTEGRATION_20261030.md). It lists source paths, compatibility limits and CPU validation. Existing inference entry points and backbone recipes remain documented above.


## Training responsibilities (2026-10-30)

ACT, π0.5, GR00T and EgoVLA use `config`, `preflight`, `contract`, `trainer`, `worker` and guardian modules. `train.py` remains the compatible CLI entry and reexports existing helpers. See the [responsibility map and validation](TRAINING_STRUCTURE_20261030.md). Raw input caches use explicit `InputCacheConfig`; environment defaults are read only by the CLI.
