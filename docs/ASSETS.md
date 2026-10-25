# Data and checkpoints

Project assets are hosted under the Hugging Face account [ds1x](https://huggingface.co/ds1x). Historical repository and directory names retain `Psi0` and `RSCI`.

## Training data

[ds1x/Psi0-reproducibility](https://huggingface.co/datasets/ds1x/Psi0-reproducibility/tree/main) contains dataset and experiment backups:

| Directory | Contents |
| --- | --- |
| `last-and-40k-20261015/data/` | Processed SIMPLE training datasets and matching ARDY sidecars, including box-hugging, between-tables, oven and XMovePick. |
| `training-datasets-20261025/` | Paired training-data/ARDY archives for faucet (99 episodes) and handover (100 episodes). |
| `eval-environment-20261016/` | Evaluation-environment backups. |
| Other dated directories | Supplementary backups, runtime archives and cleanup records. Consult their manifests before restoring. |

For the README's box-hugging examples, use these two archives under `last-and-40k-20261015/data/`:

- Data: `simple-hug-v2-training-h264--G1WholebodyPickAndPlaceAndHugContainerTeleop-v0--0ace54e4c5.tar.zst`
- ARDY labels: `simple-hug-v2-training-ardy-h264--G1WholebodyPickAndPlaceAndHugContainerTeleop-v0--20dd4a7b08.tar.zst`

Each archive contains the task directory's contents: extract the data into `$HUG_DATASET` and labels into `$HUG_SIDECAR`, preserving the task basename. Retain the supplied statistics and metadata so the ARDY dataset fingerprint remains valid. Run the README's training `--check` before use. Other archives may use different layouts; follow their accompanying JSON manifests and checksums.

## Trained policies

[ds1x/RSCI](https://huggingface.co/ds1x/RSCI/tree/main) contains historical RHSCI checkpoints:

| Directory | Contents |
| --- | --- |
| `newcheckpoint/Psi0/` | Ψ0 runs grouped by variant, task, VLM mode and run name; includes box-hugging VGQ-AE checkpoints. |
| `newcheckpoint/Psi0-NR/` | Ψ0 without Reader; currently includes between-tables VGQ-AE. |
| `newcheckpoint/Pi05/` | π0.5 runs; currently includes between-tables VGQ-AE. |
| `backup-archives/` | Additional archived runs, including box-hugging π0.5 checkpoints. |

Keep each run's configuration, checkpoint manifest and checkpoint files together. Historical runs can contain old absolute asset paths and differ from the current README recipe; downloading them does not imply that training can resume unchanged. See [inference compatibility](INFERENCE.md) before loading them.

GR00T trained policies are not present in the directories above at the checked revision. The README's GR00T example trains from NVIDIA's official N1.6 initialization; it does not refer to an uploaded RHSCI GR00T policy.

[ds1x/Psi0-training-checkpoints](https://huggingface.co/ds1x/Psi0-training-checkpoints/tree/main) holds older Ψ0 experiments, including depth variants outside this release's supported architectures.

Initialization weights and ARDY teacher assets are listed separately in [data preparation](DATA_PREPARATION.md#initialization-weights). The directory listing above was checked at dataset revision `1c1fc584995c50b4be9232111d33788009e5f0f6` and model revision `cd89a6fb1bedc8b90272b0734444edf2f46cd26f`.
