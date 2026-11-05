# Repository structure

```text
RHSCI/
├── baselines/<model>/       # Model guides, dependencies and launch helpers
├── configs/                # Training presets and pinned SIMPLE source configuration
├── scripts/
│   ├── train/              # Backbone training launchers
│   ├── data/               # Conversion, ARDY, normalization and cache tools
│   └── deploy/             # Ψ0 serving launcher
├── src/
│   ├── psi/rhsci/          # Ψ0 extensions
│   ├── openpi/rhsci/       # π0.5 extensions
│   ├── gr00t/rhsci/        # GR00T extensions
│   ├── human_plan/rhsci/   # EgoVLA extensions
│   ├── lambda0/rhsci/      # λ₀ extensions
│   ├── act/rhsci/          # ACT extensions
│   └── rhsci/              # Shared data, runtime and inference utilities
├── experiments/            # Evaluation orchestration and exploratory tools
├── tests/                  # Unit and integration tests
├── tools/                  # Source checks and configuration helpers
├── docs/                   # Installation, training, inference and evaluation guides
└── licenses/               # Component license texts
```

Other native baselines and supporting packages remain under `src`. Each model
uses its own dependency environment; installing one does not require installing
all backbones. Keep datasets, weights, environments, caches and outputs outside
the checkout.

## Model entry points

| Model | Training launcher | Serving entry |
| --- | --- | --- |
| Ψ0 | `scripts/train/psi0.sh` | `scripts/deploy/serve_psi0_simple.sh` |
| π0.5 | `scripts/train/pi05.sh` | `rhsci.inference.http_server --backbone pi05` |
| GR00T | `scripts/train/gr00t.sh` | `rhsci.inference.http_server --backbone gr00t` |
| EgoVLA | `scripts/train/egovla.sh` | `human_plan.rhsci.serve` |
| λ₀ | `scripts/train/lambda0.sh` | `baselines/lambda0/Lambda0_Serve.sh` |
| ACT | `scripts/train/act_rhsci.sh` | `act.rhsci.serve` |

Run launchers from the checkout root. See the [backbone guides](../README.md#backbones)
for dependencies and required arguments.

## Within each model

Maintained adapters separate configuration (`config.py`), asset checks
(`preflight.py`), training configuration compatibility (`contract.py`), CLI
(`train.py`), worker assembly (`worker.py`), trainer behavior (`trainer.py`) and
checkpoints. Ψ0 uses native trainer hooks; λ₀ uses its native backend.

Cross-model utilities belong in `src/rhsci`. Model algorithms and processors
remain in their owning packages. See [Architecture](ARCHITECTURE.md).
