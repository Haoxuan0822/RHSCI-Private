# H-RDT

Native source and environment: `src/h_rdt`. RHSCI adaptation is pending.

Use the component's separate Python 3.11 environment and its pinned dependencies. Configure local H-RDT, DINO/SigLIP and language assets; prepare LeRobot language embeddings explicitly when required by the selected data backend.

Training: `bash scripts/train/h-rdt.sh --help`. This launches `src/h_rdt/main.py` from its source directory and forwards native options. See [the training guide](../../docs/TRAINING.md). Native training has not been validated in this release.
