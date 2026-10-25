# GR00T

Implementation: `src/rsci/backbones/gr00t`, alongside the native GR00T model.
Entry: `bash scripts/train/gr00t.sh --help`.

This distribution includes GR00T **N1.6**, whose [NVIDIA License](../../src/gr00t/LICENSE) limits the work and derivatives to non-commercial research and imposes other use restrictions. The current upstream main branch's license does not replace these terms. See [Third-party notices](../../THIRD_PARTY_NOTICES.md).

The adapter uses native optimization defaults, FP32 Adam states with ZeRO-1 partitioning, and non-reentrant DiT recomputation. Supported variants are STATE, NO-STATE and VGQ-AE.

The separate environment requires [requirements.txt](requirements.txt) and the native GR00T dependencies. `GROOT_RSCI_PYTHON` overrides the interpreter; the default is `$PSI_RUNTIME_ROOT/.venv-gr00t-rsci/bin/python`.

See the [end-to-end README example](../../README.md#getting-started) for the isolated installation recipe and a box-hugging VGQ-AE run, followed by its matching HTTP server and SIMPLE evaluation commands. The example pins PyTorch 2.7.0, Transformers 4.51.3 and FlashAttention 2.7.4.post1, uses `top4` tuning with global batch 24 and trains for 50000 updates. Its service loads `checkpoint-50000` and returns 16 actions per request.

## SIMPLE recipe

The reference is PSI0's pinned [SIMPLE preset](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/gr00t-n1.6/presets/train/finetune_simple.yaml), which inherits [_common_finetune.yaml](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/baselines/gr00t-n1.6/presets/train/_common_finetune.yaml).

| Setting | SIMPLE reference | RHSCI README example |
| --- | --- | --- |
| Updates / global batch | 50000 / 24 | Same |
| Learning rate / warmup ratio / weight decay | `1e-4` / `0.05` / `1e-5` | Same |
| Color jitter: brightness, contrast, saturation, hue | `0.3`, `0.4`, `0.5`, `0.08` | Same |
| GPU layout | 2 GPUs | 8 GPUs × batch 3; accumulation 1 |
| Initialization |Public `nvidia/GR00T-N1.6-3B`| Same |
| Checkpoint saving | Every 10000 steps; limit 4 | Every 1000 steps; retain 10000-step anchors and the latest rolling checkpoint |

RHSCI uses NVIDIA's official `nvidia/GR00T-N1.6-3B` initialization, as in its local experiments. Alignment here covers the optimization recipe; the initialization, RHSCI query objective and checkpoint strategy differ from the reference preset.

Use matching Eagle initialization assets and the task's RGB/ARDY dataset. `--dry-run` resolves the configuration; `--check-data` checks data without loading the complete backbone. GPU memory and distributed behavior require their own validation.
