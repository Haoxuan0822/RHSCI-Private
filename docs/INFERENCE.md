# Inference

## PSI0

Copy `.env.sample` to an untracked `.env` and set paths as described in the [PSI0 guide](../baselines/psi0/README.md). Serve a checkpoint with the normal server arguments:

```bash
bash scripts/deploy/serve_psi0_simple.sh --help
```

Use the official SIMPLE evaluation program with the task, assets and simulator settings required by your checkpoint.

Starred RHSCI variants consume RGB at offsets `[-6, -4, -2, 0]` on the 50 Hz training clock. Record every control step, including steps between policy requests, and reset history at each episode boundary. The last history frame must equal the current request RGB. Invalid frames at episode start are padded and masked.

`src/rsci/inference/history_client.py` provides `CausalRgbHistory` and `install()`. Call `install()` in the simulator process before constructing the SIMPLE PSI0 agent; it wraps `Psi0DecoupledWbcAgent` and `HttpActionClient` in memory. For another client, attach `CausalRgbHistory.condition()` to the request condition after recording the current RGB. Nonstarred variants do not require this history input during inference.

Keep model source, normalization statistics, control frequency and simulator settings fixed when comparing results. CPU regression checks do not validate simulator rollouts or GPU training.

## π0.5 and GR00T

Both RHSCI adapters expose SIMPLE HTTP `POST /act` and `GET /health` through a common entry. Use the selected backbone's training environment, with FastAPI and Uvicorn installed. Run from the repository root with `PYTHONPATH=src`.

Discover complete checkpoints under an explicit run directory:

```bash
PYTHONPATH=src /path/to/pi05-env/bin/python -m rsci.inference.http_server \
  --backbone pi05 --discover /runs/pi05
```

Serve π0.5 with the training dataset's normalization statistics and matching tokenizer:

```bash
PYTHONPATH=src /path/to/pi05-env/bin/python -m rsci.inference.http_server \
  --backbone pi05 --run-dir /runs/pi05/my_task \
  --dataset /datasets/my_task --tokenizer /models/paligemma_tokenizer.model \
  --device cuda:0 --host 127.0.0.1 --port 5555
```

Serve GR00T using the model and processor saved in its checkpoint:

```bash
PYTHONPATH=src /path/to/gr00t-env/bin/python -m rsci.inference.http_server \
  --backbone gr00t --run-dir /runs/gr00t/my_task \
  --device cuda:0 --host 127.0.0.1 --port 5555
```

The run directory must contain `launch_contract.json` and complete checkpoints. By default, the server verifies source identities against this checkout. It selects the latest complete checkpoint; `--checkpoint-step N` selects one explicitly. Discovery checks completeness but leaves the full hash/numerical scan to server startup.

For an existing π0.5 run trained before the reorganization, add `--training-source-root /path/to/original/training-checkout` to serving or discovery. All recorded source hashes are checked against that explicit directory, including the original launcher scripts. The old contract and checkpoint markers remain unchanged. This inference-only compatibility path accepts schema-1 completion markers while retaining file, hash, numerical and strict weight checks; it does not authorize resuming training with changed sources. The processor restores the saved normalization version, including the early unprotected quantile recipe. Unknown preprocessing versions are rejected. No original training modules are imported or executed.

PSI0 serving reads `run_config.json` using the configuration module recorded in `argv.txt`, without replaying obsolete training flags. Historical RHSCI sidecar and RGB-encoder field names are translated to their current names. Reader selection is checked against the checkpoint schema, which also identifies older full-token runs without a saved Reader flag. Removed depth/v5 architectures are rejected when enabled; their inactive settings do not recreate those branches.

Requests and responses use `psi.deploy.helpers.RequestMessage` and `ResponseMessage`, the same NumPy serialization as PSI0. π0.5 accepts the current `rgb_head_stereo_left` image and instruction, plus `state['states'][0]` only for STATE. Ordinary VGQ-AE does not consume history or state at inference. It uses native 10-step sampling, the training quantile inverse transform and a 24-step response from the 30-step prediction. All 36 action dimensions come from that request's prediction, including the last four dimensions. This HTTP entry does not invoke the upstream openpi WebSocket server or its previous-chunk tail replacement.

GR00T uses its native SIMPLE request conversion and sampling with a 16-step action response. The saved processor and policy are set to current-frame video indices `[0]` in eval mode; the training history/query branch is not executed. Use the corresponding chunk length and disable RTC in the SIMPLE client for these native non-RTC sampling paths. The simulator and task assets remain external.

## Reproducible action sampling

Add `--deterministic-eval --eval-seed 73` to the PSI0 serving script or the π0.5/GR00T HTTP command. Configure the client with the same base evaluation seed. This is separate from `--state-noise-seed`. With this option off, native global-RNG action sampling remains unchanged.

Each request must carry `history.psi_eval_deterministic` with nonnegative integer `episode`, `episode_seed` and `inference_index`. Set `protocol` to `"episode-noise-v2-rtc"` for PSI0 with `--rtc`, or `"episode-noise-v1-no-rtc"` for PSI0 without RTC and for π0.5/GR00T. The episode seed must equal the base seed plus the episode number. Requests start at index zero for each episode and increment after each successful response. PSI0 clears its previous RTC action chunk on a new episode. Missing metadata, a seed mismatch or discontinuous indices are rejected; failed responses do not advance the index.

The servers use the local protocol's SplitMix64 mapping and private per-request Torch generators. π0.5 receives explicit `(1, 30, 36)` initial action noise; PSI0 and GR00T capture native Gaussian sampling at its original shape and dtype. These streams are independent of state perturbations and previous global RNG use. The option also enables deterministic Torch settings before model initialization; successful requests log their derived noise seed and noise hash. A native response with no captured action noise is rejected rather than reported as deterministic. Keep software, hardware, inputs and simulator settings fixed when comparing rollouts. Fixing action noise alone does not guarantee identical complete rollouts across different software or hardware.

## State measurement noise

PSI0 and π0.5 STATE serving accept `--state-inference-measurement-noise-std-rad S` and `--state-noise-seed SEED`. Noise is off by default, including for a checkpoint trained with noise. The standard deviation is in raw radians; the flag without a value selects `0.02`.

For PSI0, add those options to `scripts/deploy/serve_psi0_simple.sh`. For π0.5:

```bash
PYTHONPATH=src /path/to/pi05-env/bin/python -m rsci.inference.http_server \
  --backbone pi05 --run-dir /runs/pi05/my_state_task \
  --dataset /datasets/my_state_task --tokenizer /models/paligemma_tokenizer.model \
  --device cuda:0 --port 5555 \
  --state-inference-measurement-noise-std-rad 0.02 --state-noise-seed 73
```

Use a STATE checkpoint and SIMPLE's measured-joint state schema. Only indices `[0:31]` are perturbed before normalization; the height command and padding remain unchanged. Non-STATE variants reject enabled state noise. GR00T does not accept this option.

The noise generator is independent of action-sampling RNG. When `history.psi_eval_deterministic` contains nonnegative integer `episode` and `inference_index` values, the base seed and those indices determine the request seed, matching the local evaluation protocol. Otherwise a server-local request counter is used; restart the server to reset it. Keep the indices and base seed fixed to compare noise conditions reproducibly.

## EgoVLA

Run in the EgoVLA environment with the native VILA source on the import path:

```bash
PYTHONPATH=src:src/egovla:src/egovla/VILA \
  /path/to/egovla-env/bin/python -m rsci.backbones.egovla.serve \
  --checkpoint /runs/egovla/my_task/checkpoint-10120 --port 8011
```

The service checks the complete checkpoint, source hashes and original base assets. Keep the base checkpoint at the initialization path saved in the run contract. `/act` consumes the current `rgb_head_stereo_left` image and instruction; STATE also requires finite 32D `state["states"]`. It returns 30 × 36 native physical actions. VGQ-AE inference does not use state, Reader, query supervision or training history. It does not expose the Ψ0/π0.5/GR00T request-seed option.

## λ₀

Use `baselines/lambda0/Lambda0_Serve.sh` in the pinned λ₀ environment. Supply `--model` with a complete RHSCI checkpoint and `--normalization` with the matching file from its `data_bundle/stats` directory; `--processor`, if supplied, must point to the same model directory. `--model /path/to/checkpoint/ema` selects its EMA model. Run the wrapper with `--help` for native flow/RTC server options. The adapter validates the source, processor and normalization against the checkpoint before calling the native service.

## ACT

Use the ACT environment, including the serving dependencies, with a complete RHSCI checkpoint:

```bash
PYTHONPATH=src /path/to/act-env/bin/python -m rsci.backbones.act.serve \
  --checkpoint /runs/act/my_task/checkpoint-40000 --device cuda:0 --port 8126
```

The service exposes `/health`, `/reset` and `/act`, using PSI0's `RequestMessage` / `ResponseMessage` serialization. Send one current uint8 RGB image. STATE requires `state["states"]` containing the 32 measured SIMPLE values; NO-STATE and EGQ-AE ignore state and need no history or ARDY labels. Responses contain the full 16 × 36 action chunk. The native latent-zero policy is deterministic in evaluation mode; no action-noise seed switch is needed. This entry validates checkpoint completeness, normalization and source identity. Being-H0.5 currently uses its native inference API and is not part of this HTTP protocol.

## Query ablations

PSI0 restores Reader selection from the saved model configuration. Full-token checkpoints do not need an additional serving flag. The training-only future ARDY offset does not change inference inputs or require future labels. Starred variants still require the same four causal RGB frames. See [Training](TRAINING.md#experimental-switches) for the supported combinations.
