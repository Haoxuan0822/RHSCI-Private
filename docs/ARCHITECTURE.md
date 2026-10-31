# Code organization

The project and current Python namespace are **RHSCI / `rhsci`**. Model extensions use `psi.rhsci`, `openpi.rhsci`, `gr00t.rhsci`, `human_plan.rhsci`, `lambda0.rhsci`, `act.rhsci`; common tools use `rhsci`. Current CLI mode is `--mode rhsci`. `RHSCI_*` environment names take priority; historical `RSCI_*` names and `--mode rsci` are accepted as compatibility inputs. Existing checkpoint tensor/config keys, schema strings, filenames such as `rsci_config.json` / `RSCI_COMPLETE.json`, and runtime/environment directories retain their saved spelling. This avoids changing model parameter identities or old artifact locations. Historical source records describe their original names.

Model-specific RHSCI extensions live inside their native packages. `src/rhsci` contains shared utilities; importing its root does not load a backbone or optional dependencies. See [Repository structure](DIRECTORY_STRUCTURE.md) for the directory tree and per-backbone entry points.

Use descriptive action names for maintained extension functions, consistently across models. The [function guide](DIRECTORY_STRUCTURE.md#function-names-and-reading-order) maps CLI parsing, path resolution, preflight, contracts, workers and guardians to their owning modules. Keep detailed preflight checks in `preflight.py` and state return values and side effects in critical function docstrings. Native framework method names, CLI options and persisted checkpoint keys follow their original interfaces.

| Directory | Responsibility |
| --- | --- |
| `src/psi/rhsci`, `src/openpi/rhsci`, `src/gr00t/rhsci`, `src/human_plan/rhsci`, `src/lambda0/rhsci`, `src/act/rhsci` | Model-specific integration, data, training, checkpoints and serving, colocated with each native implementation |
| `src/rhsci/data` | ARDY sidecars, dataset provenance, frame caching and worker cleanup |
| `src/rhsci/runtime` | Artifact writes, numerical checks, process cleanup and profiling |
| `src/rhsci/inference` | SIMPLE control-step RGB history protocol |
| `scripts/train` | Common launcher, shell entry points and distributed launch configuration |
| `baselines/<backbone>` | Usage, dependencies and component integration records |
| `tests` | Model, data, resume, numerical failure, serving and launcher regressions |

Native packages such as `psi`, `openpi`, `gr00t`, `egovla` and `openlambda` retain their source trees under `src`. Each backbone uses its own dependency environment. Shared utilities belong in `src/rhsci`; native model dependencies belong in the corresponding adapter.

PSI0 integrates through `RHSCIFinetuneTrainer` and native trainer hooks. λ₀ integrates through `RHSCITrainingBackend` and explicit server arguments. ACT, GR00T and EgoVLA share sampler-epoch recovery through `runtime/hf_resume.py`; its compatibility check stays local to each derived Trainer. ACT's bounded decoder/frame cache stays in its adapter because it depends on its native LeRobot pipeline. Being-H0.5 retains official code under `src/beingh05` and uses only the native common launcher. The external SIMPLE client integration is an opt-in wrapper; see [Inference](INFERENCE.md).

ACT, EgoVLA, π0.5 and GR00T share the complete-checkpoint lifecycle in `runtime/hf_full_state.py`. ACT/EgoVLA bind their own schema and model loader; π0.5/GR00T bind the existing schema-2 format and their distinct metadata files. Ψ0 and λ₀ keep their native checkpoint integration. Contracts compare model, data and training configuration; source/version audit records are excluded from runtime compatibility. Inference validates model files independently of optimizer/RNG state; see [runtime check policy](RUNTIME_CHECKS_20261031.md).

## Shared functions (2026-11-01)

| Shared implementation | Consumers | Per-model responsibility |
| --- | --- | --- |
| `runtime/losses.py`: `masked_auxiliary_loss`, `masked_ardy_loss` | All six backbones | Native action loss, auxiliary head and feature construction |
| `data/sampling.py`: `EpochRandomSampler`, `EpochSampler` | Ψ0, ACT, GR00T | Whether samples carry explicit `(epoch,index)` |
| `runtime/contracts.py`: `compute_contract_digest` | ACT, EgoVLA, π0.5, GR00T, λ₀ | Model/data/recipe fields; old digest serialization is preserved |
| `runtime/hf_full_state.py`: `HFCheckpointProtocol`, `HFTrainerCheckpointProtocol` | ACT, EgoVLA, π0.5, GR00T | Existing schema, metadata files, native parameter loading |
| `runtime/hf_callbacks.py`: `build_checkpoint_callback` | ACT, EgoVLA | Protocol binding; π/GR keep their processor/config save hooks |
| `runtime/numerics.py`: `require_finite` | ACT, EgoVLA, π0.5, GR00T | Which losses/gradients are checked and when |

Native Reader layouts, model execution, normalization, action losses, optimizer/scheduler recipes and specialized resume handling remain with each backbone. Common functions are imported or bound directly; identical algorithms are not copied into each adapter. The shared loss masks invalid targets, preserves a zero-gradient graph when all labels are invalid, and normalizes uneven DDP ranks by global valid sample count. Valid nonfinite labels fail explicitly.

Validation for this refactor uses the existing CPU backbone suites, a two-rank CPU/Gloo loss-versus-centralized-gradient comparison, empty-label gradients, epoch/RNG independence, schema-2 sharded checkpoint compatibility, native package resolution and real CLI help. Actual results from `fdd7edf` plus this refactor: Ψ0 43, π0.5 37, GR00T 32, ACT 29 CPU tests passed. Shared suite: 37 passed and 2 SIMPLE tests initially skipped without `SIMPLE_HOME`; those 2 were then run against the existing pinned checkout and passed. The four new shared tests include the real two-rank CPU/Gloo comparison. λ₀ source-audit tests: 2 passed, including relocated-package changes. All 12 module/shell training help calls passed. Python/shell syntax, Ruff, package discovery, source inventories and documentation links were checked without installation. Counts overlap across suites and are not a unique-test total. Full GPU training/old large-model resume remain untested. EgoVLA's existing missing FlashAttention and λ₀'s missing Qwen3.5 API remain environment limitations.


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

Replace `psi0` with `pi05`, `gr00t`, `egovla`, `lambda0` or `act`. The helper keeps temporary caches outside the checkout and excludes distributed tests. Optional λ₀ asset tests use `RHSCI_TEST_LAMBDA0_STATS` and `RHSCI_TEST_LAMBDA0_BUNDLE`; missing assets produce skips. CPU tests do not exercise full-model CUDA training or simulator rollouts.

For λ₀ asset tests, `RHSCI_TEST_LAMBDA0_BUNDLE` points to the prepared bundle's `train` metadata directory, not its root (for example `/bundles/my_task-safe/train`). `RHSCI_TEST_LAMBDA0_STATS` points to its normalization JSON. Set `PSI_RUNTIME_ROOT` to the external runtime root when metadata uses paths relative to its `data` directory.

The HTTP checkpoint and protocol tests run in the π0.5 environment with FastAPI and its test client installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_http_inference.py
```


## 2026-10-30 PSI0-RAW integration

Latest input-cache and training-recovery changes are documented in the [integration record](PSI0_RAW_INTEGRATION_20261030.md). It lists source paths, compatibility limits and CPU validation. Existing inference entry points and backbone recipes remain documented above.


## Training responsibilities (2026-10-30)

ACT, π0.5, GR00T and EgoVLA use `config`, `preflight`, `contract`, `trainer`, `worker` and guardian modules. `train.py` remains the compatible CLI entry and reexports existing helpers. See the [responsibility map and validation](TRAINING_STRUCTURE_20261030.md). Raw input caches use explicit `InputCacheConfig`; environment defaults are read only by the CLI.
