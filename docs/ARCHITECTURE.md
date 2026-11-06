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
| `rhsci.data.ardy_motion` | VGQ-AE-Mo configuration and same-episode backward ARDY differences |
| `rhsci.data.visual_history` | Visual frame offsets for VGQ-AE and its 240 ms variant, independent of ARDY teacher windows |
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

## Training and inference

Shell launchers in `scripts/train` select the backbone environment and call its
package-local CLI. Configuration, preflight, worker, trainer and checkpoint
modules separate launch setup from model execution. Each backbone retains its
native optimizer and sampling recipe.

VGQ-AE-240 reuses each backbone's VGQ-AE model and processor. Dataset adapters obtain `[-12, -8, -4, 0]` from `rhsci.data.visual_history`; the original variants keep `[-6, -4, -2, 0]`. Existing current-ARDY labels are combined with the selected visual-history validity mask. The method name is saved through each backbone's existing configuration and checkpoint path; inference remains current-frame only.

VGQ-AE-Mo extends the existing VGQ-AE branch inside each of the five VLA backbones. The current-ARDY and difference heads consume one shared Query output; the existing masked auxiliary loss handles both targets. Dataset adapters read backward differences through `rhsci.data.ardy_motion` and pass labels separately from observations. Method-specific settings and head weights are saved with the backbone's existing checkpoint/configuration files.

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
