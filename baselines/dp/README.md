# Diffusion Policy

Native implementation: `src/dp`, with shared PSI0 configs and `psi.trainers.diffusion_policy_g1`. RHSCI adaptation is pending.

Install the root shared package and [requirements-dp.txt](requirements-dp.txt) in a separate Python 3.10 environment. The baseline's torch/transformers/LeRobot versions differ from the RHSCI stacks; root `uv.lock` does not certify this environment.

Training: `bash scripts/train/dp.sh --help`. See [the training guide](../../docs/TRAINING.md) for dataset/output arguments and the configurable SIMPLE G1 argument list. Native training has not been validated in this release.
