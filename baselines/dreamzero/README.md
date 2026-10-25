# DreamZero

Native source: `src/dreamzero`. RHSCI adaptation is pending.

Install the root shared package and [requirements-dreamzero.txt](requirements-dreamzero.txt) in a separate Python 3.11 environment. Prepare the DreamZero, Wan and tokenizer assets required by the chosen Hydra configuration. Dataset conversion and calibration assets are external.

Training: `bash scripts/train/dreamzero.sh --help`. The launcher forwards Hydra overrides to the native trainer. See [the training guide](../../docs/TRAINING.md) for explicit data, initialization and output options. Native training has not been validated in this release.
