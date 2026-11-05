# Data and checkpoints

Project assets are hosted under the Hugging Face account [ds1x](https://huggingface.co/ds1x). Historical repository and directory names retain `Psi0` and `RSCI`.

## Current RHSCI handoff

The current public handoff uses the dataset [ds1x/RHSCI](https://huggingface.co/datasets/ds1x/RHSCI): `train-data/` contains training data and ARDY labels; `eval-data/` contains evaluation inputs. Weights and resumable checkpoints use the separate model [ds1x/RHSCI](https://huggingface.co/ds1x/RHSCI) under `checkpoint/`. Initialization weights are still downloaded from the original backbone providers.

Use the asset revision specified in [Evaluation](EVALUATION.md) for evaluation runs. Training data, ARDY labels and normalization remain bound to their saved versions.

The [fixed asset manifest](https://huggingface.co/datasets/ds1x/RHSCI/blob/2bff75eb0ca574211ed95a5d88d4fc8482426369/ASSET_MANIFEST.json) records exact filenames, roles, sizes and SHA256. Task directories are `train-data/{hug,between,oven,handover,faucet,xmove}`. Some tasks have separate data/ARDY archives; handover and faucet use combined archives. Read their manifests before extracting. For box-hugging, the two archive basenames listed below are available together under `train-data/hug/` at that fixed revision.

Download only the selected task's training archives and matching metadata. Preserve archived statistics and the processed-data/ARDY pairing. Prebuilt formal labels can be used directly; training with those labels does not require installing the ARDY teacher.

Keep original demonstration identities when creating data splits: duplicated motion sources must stay in the same split. Data and sidecar format checks do not establish demonstration success.

## Historical training-data sources

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

## Historical trained-policy sources

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
