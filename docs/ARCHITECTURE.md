# Code organization

The project name is RHSCI. The existing `rsci` Python namespace, `RSCI_*` environment variables and saved configuration/checkpoint field names remain compatibility interfaces; new help text, messages and experiment display names use RHSCI.

RHSCI extensions live in `src/rsci`. Importing the package root does not load a backbone or its optional dependencies. See [Repository structure](DIRECTORY_STRUCTURE.md) for the directory tree and per-backbone entry points.

| Directory | Responsibility |
| --- | --- |
| `src/rsci/backbones/<backbone>` | Model integration, data adaptation, training, checkpoint format and serving |
| `src/rsci/data` | ARDY sidecars, dataset provenance, frame caching and worker cleanup |
| `src/rsci/runtime` | Artifact writes, numerical checks, process cleanup and profiling |
| `src/rsci/inference` | SIMPLE control-step RGB history protocol |
| `scripts/train` | Common launcher, shell entry points and distributed launch configuration |
| `baselines/<backbone>` | Usage, dependencies and component integration records |
| `tests` | Model, data, resume, numerical failure, serving and launcher regressions |

Native packages such as `psi`, `openpi`, `gr00t`, `egovla` and `openlambda` retain their source trees under `src`. Each backbone uses its own dependency environment. Shared utilities belong in `src/rsci`; native model dependencies belong in the corresponding adapter.

PSI0 integrates through `RSCIFinetuneTrainer` and native trainer hooks. λ₀ integrates through `RSCITrainingBackend` and explicit server arguments. ACT and EgoVLA share sampler-epoch recovery through `runtime/hf_resume.py`; its compatibility check stays local to each derived Trainer. ACT's bounded decoder/frame cache stays in its adapter because it depends on its native LeRobot pipeline. Being-H0.5 retains official code under `src/beingh05` and uses only the native common launcher. The external SIMPLE client integration is an opt-in wrapper; see [Inference](INFERENCE.md).

Adapters share file digests and numerical checks but retain their native checkpoint schemas. Checkpoint contracts validate source, configuration and data identities before resuming. Use a new output directory when changing those contracts.

## Development

Maintained RHSCI code, launch tools and tests use Python 3.10 syntax, four-space indentation, double quotes, sorted imports and a 100-character target line length. `ruff.toml` scopes these checks; native source retains upstream style.

```bash
uvx --from ruff==0.17.2 ruff format --check .
uvx --from ruff==0.17.2 ruff check .
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
