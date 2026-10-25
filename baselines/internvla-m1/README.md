# InternVLA-M1

Native source and environment: `src/InternVLA-M1`. RHSCI adaptation is pending.

Prepare its separate environment using the source `pyproject.toml` / `uv.lock` and install the dependencies required by your CUDA platform. Configure local VLM, DINO and initialization checkpoints in a copied training YAML.

Training: `bash scripts/train/internvla-m1.sh --help`. The entry invokes the upstream trainer through torchrun and forwards OmegaConf overrides. See [the training guide](../../docs/TRAINING.md). Native training has not been validated in this release.
