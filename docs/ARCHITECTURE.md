# Code organization

RHSCI extends each backbone inside its native Python package. Model-specific
algorithms, preprocessing, losses and checkpoint formats stay with the backbone.
Shared infrastructure lives in `src/rhsci`.

| Package | Integration |
| --- | --- |
| `psi.rhsci` | Ψ0 model extensions and native trainer hooks |
| `openpi.rhsci` | π0.5 model, data, processor, training and serving |
| `gr00t.rhsci` | GR00T data, training and serving |
| `human_plan.rhsci` | EgoVLA model, processor, training and serving |
| `lambda0.rhsci` | λ₀ bundles, backend, training and serving |
| `act.rhsci` | ACT model, data, training and serving |

## Shared infrastructure

| Module | Responsibility |
| --- | --- |
| `rhsci.data` | ARDY sidecars, dataset provenance, sampling and raw input caches |
| `rhsci.data.ardy_motion` | Mo variant configuration and same-episode backward ARDY differences |
| `rhsci.data.visual_history` | Visual frame offsets for VGQ-AE and its 240/480 ms variants, independent of ARDY teacher windows |
| `rhsci.data.multitask` | π0.5/GR00T multi-root arguments, dataset-local routing and shared normalization contracts |
| `rhsci.runtime.losses` | Masked auxiliary and ARDY losses |
| `rhsci.runtime.auxiliary_accumulation` | Valid-label counts across a complete optimizer update, including gradient accumulation and DDP |
| `rhsci.runtime.hf_full_state` | Complete checkpoint lifecycle for HF-based adapters |
| `rhsci.runtime.hf_resume` | Sampler epoch and progress restoration |
| `rhsci.runtime.contracts` | Model, data and training configuration compatibility |
| `rhsci.runtime.numerics` | Finite-value checks |
| `rhsci.runtime.processes` | Child process tracking and cleanup |
| `rhsci.inference.protocol` | SIMPLE request/response serialization |
| `rhsci.inference.http_server` | π0.5 / GR00T HTTP serving entry |
| `rhsci.inference.history_client` | Control-step RGB history for starred variants |
| `rhsci.inference.measurement_noise` | Client measurement perturbations and temporal buffers |
| `rhsci.inference.control_clock` | Optional SIMPLE/WBC simulation clock, shared CLI options and clock consistency on evaluation continuation |

## Training and inference

Shell launchers in `scripts/train` select the backbone environment and call its
package-local CLI. Configuration, preflight, worker, trainer and checkpoint
modules separate launch setup from model execution. Each backbone retains its
native optimizer and sampling recipe.

π0.5 and GR00T opt into multi-root training through `rhsci.data.multitask`, which pairs source/label roots and validates their recorded teacher identities. Child adapters own language, labels, video caches and local episode/frame IDs. π0.5 trains through PyTorch `ConcatDataset`, preserving all frames with terminal action padding; `openpi.rhsci.normalization` feeds numeric action queries and states to the native OpenPI `RunningStats`. GR00T preflight validates child map readers, then `gr00t.rhsci.multitask` adapts them to inherited native shard partitioning and the native mixture scheduler/statistics. Its trainer inherits `Gr00tTrainer` for the plain loader and native reseeding, avoiding a second rank partition by Accelerate. Existing RHSCI losses/optimizer/callbacks remain in place. Shared statistics and sampling settings are embedded in the contract; GR00T saves its native processor. Single-root paths keep their previous dataset class, sampling and statistics.

VGQ-AE-240 reuses each backbone's VGQ-AE model and processor. Dataset adapters obtain `[-12, -8, -4, 0]` from `rhsci.data.visual_history`; the original variants keep `[-6, -4, -2, 0]`. Existing current-ARDY labels are combined with the selected visual-history validity mask. The method name is saved through each backbone's existing configuration and checkpoint path; inference remains current-frame only.

VGQ-AE-Mo extends the existing VGQ-AE branch inside each of the five VLA backbones. The current-ARDY and difference heads consume one shared Query output; the existing masked auxiliary loss handles both targets. Dataset adapters read backward differences through `rhsci.data.ardy_motion` and pass labels separately from observations. Method-specific settings and head weights are saved with the backbone's existing checkpoint/configuration files.

VGQ-AE-Mo-240 uses the same Mo model and backward-difference labels, selecting `[-12, -8, -4, 0]` from the existing visual-history helper. It retains the default six-frame difference interval. Dataset adapters combine the expanded visual-history mask with each label's validity; GR00T performs this combination in its existing model and accumulation paths.

VGQ-AE-480 and VGQ-AE-Mo-480 select `[-24, -16, -8, 0]` through the same helper. Mo-480 defaults to a twelve-frame backward difference (240 ms), resolved by `rhsci.data.ardy_motion` and saved through the existing configuration/contract paths. Explicit intervals still override defaults. Network parameters, four-frame Reader and current-only inference retain the corresponding AE/Mo architecture.

Training recovery restores model, optimizer, scheduler, RNG and progress state.
Inference loads model weights, saved preprocessing and configuration without
requiring optimizer state. Existing checkpoint keys and filenames such as
`rsci_config.json` retain their original spelling for compatibility.

The simulator runs separately from the model server. The SIMPLE client sends
observations using the shared protocol; each model adapter owns prediction and
postprocessing. Exploratory scripts in `experiments` call maintained interfaces.
Production training and serving modules do not import those scripts.

See [Repository structure](DIRECTORY_STRUCTURE.md), [Training](TRAINING.md),
[Inference](INFERENCE.md) and [Evaluation](EVALUATION.md).
