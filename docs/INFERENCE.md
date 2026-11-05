# Inference

Current zero-arm evaluation versions, client/server roles and action horizons
are summarized in the [README evaluation table](../README.md#current-evaluation-configuration-2026-11-05).
Those experiments use the old `psi-data@e78fb93c` ZIPs and official SIMPLE
`b75c6786` with the [documented task source changes](SIMPLE_OFFICIAL_20261030.md#zero-arm-evaluation-20261105).
The newer asset release described below is a separate evaluation condition.

Existing trained-policy directories are listed in [Hugging Face assets](ASSETS.md). Preserve the saved run configuration and consult the historical-checkpoint compatibility notes below.

See the [PSI0-RAW compatibility audit](RAW_COMPATIBILITY_AUDIT_20261103.md) for actual checkpoint evidence and limits. Ψ0 restores the original four-frame state-reconstruction method from metadata/weights and retains embedded training bounds. GR00T restores saved `rsci_memory_mode=full-vlm` without a Reader. Historical removed architectures require their matching runtime. π0.5 still needs the matching training dataset/tokenizer; EgoVLA trainable-only weights need their matching base assets.

SIMPLE now defaults to official `b75c6786` (including PR #29). Before simulation, verify the external checkout with `python3 tools/check_simple.py "$SIMPLE_HOME"` and review [version, assets and protocol compatibility](SIMPLE_OFFICIAL_20261030.md). CPU wire-format compatibility does not validate old-checkpoint behavior in the new simulator. Use the [current pinned evaluation assets](SIMPLE_UPDATES_20261103.md), including the instruction fixes in `SIMPLE-eval/fb67cd9a`; old ZIPs and the historical RHSCI handoff are different releases. Four advertised success-checker updates are absent from official `b75c6786`. The [historical WBC/MuJoCo finger-order mismatch](SIMPLE_UPDATES_20261103.md#历史手指映射问题仍未随官方更新解决) also remains in the current official path; data updates do not repair that controller interface.

SIMPLE message encoding lives in `rhsci.inference.protocol`; the native `psi.deploy.helpers` path reexports those classes for compatibility. ACT and the π/GR HTTP entry share `create_policy_app`, while prediction, error status and optional reset remain adapter-specific. EgoVLA keeps its standard-library HTTP service and reuses the NumPy codec. See the [shared interface map](ARCHITECTURE.md#simple-shared-interfaces-20261102).

## PSI0

Copy `.env.sample` to an untracked `.env` and set paths as described in the [PSI0 guide](../baselines/psi0/README.md). Serve a checkpoint with the normal server arguments:

```bash
bash scripts/deploy/serve_psi0_simple.sh --help
```

Use the official SIMPLE evaluation program with the task, assets and simulator settings required by your checkpoint.

Starred RHSCI variants consume RGB at offsets `[-6, -4, -2, 0]` on the 50 Hz training clock. Record every control step, including steps between policy requests, and reset history at each episode boundary. The last history frame must equal the current request RGB. Invalid frames at episode start are padded and masked.

`src/rhsci/inference/history_client.py` provides `CausalRgbHistory` and `install()`. Call `install()` in the simulator process before constructing the SIMPLE PSI0 agent; it wraps `Psi0DecoupledWbcAgent` and `HttpActionClient` in memory. For another client, attach `CausalRgbHistory.condition()` to the request condition after recording the current RGB. Nonstarred variants do not require this history input during inference.

Keep model source, normalization statistics, control frequency and simulator settings fixed when comparing results. CPU regression checks do not validate simulator rollouts or GPU training.

## π0.5 and GR00T

Both RHSCI adapters expose SIMPLE HTTP `POST /act` and `GET /health` through a common entry. Use the selected backbone's training environment, with FastAPI and Uvicorn installed. Run from the repository root with `PYTHONPATH=src`.

Discover complete checkpoints under an explicit run directory:

```bash
PYTHONPATH=src /path/to/pi05-env/bin/python -m rhsci.inference.http_server \
  --backbone pi05 --discover /runs/pi05
```

Serve π0.5 with the training dataset's normalization statistics and matching tokenizer:

```bash
PYTHONPATH=src /path/to/pi05-env/bin/python -m rhsci.inference.http_server \
  --backbone pi05 --run-dir /runs/pi05/my_task \
  --dataset /datasets/my_task --tokenizer /models/paligemma_tokenizer.model \
  --device cuda:0 --host 127.0.0.1 --port 5555
```

Serve GR00T using the model and processor saved in its checkpoint:

```bash
PYTHONPATH=src /path/to/gr00t-env/bin/python -m rhsci.inference.http_server \
  --backbone gr00t --run-dir /runs/gr00t/my_task \
  --device cuda:0 --host 127.0.0.1 --port 5555
```

The run directory contains `launch_contract.json` and model checkpoints. The server selects the latest checkpoint with configuration and nonempty weights; `--checkpoint-step N` selects one explicitly. Optimizer, scheduler, RNG, training completion markers and an identical training checkout are not required. Model loading still checks keys, shapes and the saved preprocessing settings.

The shared discovery entry now reads GR00T's native `config.json` and π0.5's `rsci_config.json`; this was checked against six actual GR00T 20k checkpoints. See the [research audit](RESEARCH_AUDIT_20261104.md) for evaluation-data verification, remaining official success-checker gaps and reporting requirements. Fix task/levels/episodes, initialization, action timing/horizon, inference settings, success criteria and random streams across methods. Replaying the same weight/episode/random stream does not create independent samples; retain technical failures and retries in the report.

Historical π0.5 checkpoints use the same model-only discovery path. `--training-source-root /path/to/original/training-checkout` remains an optional explicit source audit; it is not required for historical serving. Original training modules are not imported or executed. The processor restores the saved normalization version, including the early quantile recipe; unsupported preprocessing remains an error.

PSI0 serving reads `run_config.json` using the configuration module recorded in `argv.txt`, without replaying obsolete training flags. Historical RHSCI sidecar and RGB-encoder field names are translated to their current names. Reader selection is checked against the checkpoint schema, which also identifies older full-token runs without a saved Reader flag. Removed depth/v5 architectures are rejected when enabled; their inactive settings do not recreate those branches.

Requests and responses use `psi.deploy.helpers.RequestMessage` and `ResponseMessage`, the same NumPy serialization as PSI0. π0.5 accepts the current `rgb_head_stereo_left` image and instruction, plus `state['states'][0]` only for STATE. Ordinary VGQ-AE does not consume history or state at inference. It uses native 10-step sampling, the training quantile inverse transform and the complete 30-step prediction as its response. The non-RTC SIMPLE client executes all 30 actions before requesting a new prediction (30 predicted / 30 executed, per the 2026-11-04 evaluation configuration). All 36 action dimensions come from that request's prediction, including the last four dimensions. This HTTP entry does not invoke the upstream openpi WebSocket server or its previous-chunk tail replacement.

GR00T uses its native SIMPLE request conversion and sampling with a 16-step action response. The saved processor and policy are set to current-frame video indices `[0]` in eval mode; the training history/query branch is not executed. Use the corresponding chunk length and disable RTC in the SIMPLE client for these native non-RTC sampling paths. The simulator and task assets remain external.

## Reproducible action sampling

Add `--deterministic-eval --eval-seed 73` to the PSI0 serving script or the π0.5/GR00T HTTP command. Configure the client with the same base evaluation seed. This is separate from `--state-noise-seed`. With this option off, native global-RNG action sampling remains unchanged.

The unmodified official SIMPLE client does not generate the following RHSCI metadata. Keep server deterministic evaluation off when using it directly; enabling this option requires a separately validated client wrapper.

Each request must carry `history.psi_eval_deterministic` with nonnegative integer `episode`, `episode_seed` and `inference_index`. Set `protocol` to `"episode-noise-v2-rtc"` for PSI0 with `--rtc`, or `"episode-noise-v1-no-rtc"` for PSI0 without RTC and for π0.5/GR00T. The episode seed must equal the base seed plus the episode number. Requests start at index zero for each episode and increment after each successful response. PSI0 clears its previous RTC action chunk on a new episode. Missing metadata, a seed mismatch or discontinuous indices are rejected; failed responses do not advance the index.

The servers use the local protocol's SplitMix64 mapping and private per-request Torch generators. π0.5 receives explicit `(1, 30, 36)` initial action noise; PSI0 and GR00T capture native Gaussian sampling at its original shape and dtype. These streams are independent of state perturbations and previous global RNG use. The option also enables deterministic Torch settings before model initialization; successful requests log their derived noise seed and noise hash. A native response with no captured action noise is rejected rather than reported as deterministic. Keep software, hardware, inputs and simulator settings fixed when comparing rollouts. Fixing action noise alone does not guarantee identical complete rollouts across different software or hardware.

## State measurement noise

### Client noise switch and four severity levels

The existing SIMPLE wrapper `experiments/pi05_between/eval_decoupled_wbc.py` and
its multi-round worker `run_evaluation.py` accept an optional client-side switch:
`--state-noise --state-noise-type TYPE --state-noise-level LEVEL`. Supported types
are `gaussian`, `zero-offset`, `delay`, and `dropout`; levels are `low`, `medium`,
`high`, and `extreme`. With the switch absent, the native agent and HTTP client
are left untouched: no measurement cache, noise generator, or random draws are
created. Enabling requires both type and level. Supplying type, level, or seed
without the switch is an error. `--state-noise-seed` is optional (default 0).

| Parameter | low | medium | high | extreme |
| --- | --- | --- | --- | --- |
| Gaussian standard deviation | 5° | 15° | 30° | 60° |
| Uniform zero-offset bounds | ±5° | ±15° | ±30° | ±60° |
| Measurement delay | 100 ms | 300 ms | 600 ms | 1200 ms |
| Consecutive dropout duration | 200 ms | 600 ms | 1200 ms | 2400 ms |

These are user-selected stress-test levels, not sensor specifications. Angles
are converted to radians before use. Gaussian noise is independent for each
joint and each control-step measurement; it is not bounded at its standard
deviation. Zero offsets are sampled independently per joint once per episode.
There is no additional clipping or angular wrapping in the client; the model's
saved normalization and clipping still apply afterwards.

Only raw `state['states'][0, :31]` is replaced before sending the HTTP request.
The current height command at index 31, extra state fields, images, instruction,
action queue, simulation joint positions, and WBC feedback are preserved. This
works across model servers using the SIMPLE HTTP schema; a policy that does not
consume state will not gain state dependence from this switch.

The common implementation is `rhsci.inference.measurement_noise`; the small
`rhsci.inference.state_noise_client` adapter reuses the native agent, state
joint slices, and HTTP client without duplicating the evaluation/control loop.
It records measurements on every policy control step, including steps between
HTTP requests. Time comes from MuJoCo, relative to the first policy observation
after stabilization. Delay selects the newest sample at or before the target
time, padding with the first measurement when history is insufficient. Dropouts
start at 4, 8, 12… simulation seconds, freeze the last valid measured joints,
and resume with the latest measurement at the end of each interruption.

Each agent reset clears temporal history and derives an independent NumPy
generator from `[state_noise_seed, episode_index]`. The index starts at the CLI's
`--episode-start` and advances on each reset. Repeating the same seed and episode
repeats the measurement stream; it does not seed model action sampling. For
independent repetitions, explicitly change the noise seed. A worker forwards
the same chosen seed to each round and records it, rather than silently changing
the experiment. Use a separate output root for each noise condition/seed.

For example, after activating the model environment, this **launch template**
adds medium Gaussian noise to the existing worker (not an experiment started
as part of implementation):

```bash
python experiments/pi05_between/run_evaluation.py \
  --gpu 0 --variant STATE --level dr-level-0 \
  --evaluation-root "$PSI_RUNTIME_ROOT/eval-official-pre1102" \
  --output-root "$PSI_RUNTIME_ROOT/evals/pi05-between-gaussian-medium-seed73" \
  --repair-scene-references \
  --state-noise --state-noise-type gaussian --state-noise-level medium \
  --state-noise-seed 73
```

The worker records resolved parameters in `worker.json`; each client records
them in `effective_eval_config.json` before Kit initializes and prints
`STATE_NOISE` to its log. Measurement perturbations are applied only by the client;
training and server-side measurement noise have been removed. CPU
unit/integration tests validate measurement and request behavior; GPU physical
rollouts under these noise levels have not yet been run.

## EgoVLA

Run in the EgoVLA environment with the native VILA source on the import path:

```bash
PYTHONPATH=src:src/egovla/VILA \
  /path/to/egovla-env/bin/python -m human_plan.rhsci.serve \
  --checkpoint /runs/egovla/my_task/checkpoint-10120 --port 8011
```

The service checks model configuration/weights and the pretrained base assets needed by its trainable-only model. Source file hashes and optimizer/RNG files do not gate serving. Keep the base checkpoint at the initialization path saved in the run contract. `/act` consumes the current `rgb_head_stereo_left` image and instruction; STATE also requires finite 32D `state["states"]`. It returns 30 × 36 native physical actions. VGQ-AE inference does not use state, Reader, query supervision or training history. It does not expose the Ψ0/π0.5/GR00T request-seed option.

## λ₀

Use `baselines/lambda0/Lambda0_Serve.sh` in the pinned λ₀ environment. Supply `--model` with an RHSCI model directory and `--normalization` with the matching file from its `data_bundle/stats` directory; `--processor`, if supplied, must point to the same model directory. `--model /path/to/checkpoint/ema` selects its EMA model. Run the wrapper with `--help` for native flow/RTC server options. The adapter validates the model weights, processor and normalization before calling the native service; source snapshots, exact dependency versions and training state are not required.

## ACT

Use the ACT environment, including the serving dependencies, with an RHSCI model checkpoint:

```bash
PYTHONPATH=src /path/to/act-env/bin/python -m act.rhsci.serve \
  --checkpoint /runs/act/my_task/checkpoint-40000 --device cuda:0 --port 8126
```

The service exposes `/health`, `/reset` and `/act`, using PSI0's `RequestMessage` / `ResponseMessage` serialization. Send one current uint8 RGB image. STATE requires `state["states"]` containing the 32 measured SIMPLE values; NO-STATE and EGQ-AE ignore state and need no history or ARDY labels. Responses contain the full 16 × 36 action chunk. The native latent-zero policy is deterministic in evaluation mode; no action-noise seed switch is needed. This entry validates model weights/configuration and normalization; it does not require training state or identical source files. Being-H0.5 currently uses its native inference API and is not part of this HTTP protocol.

## Query ablations

PSI0 restores Reader selection from the saved model configuration. Full-token checkpoints do not need an additional serving flag. The training-only future ARDY offset does not change inference inputs or require future labels. Starred variants still require the same four causal RGB frames. See [Training](TRAINING.md#experimental-switches) for the supported combinations.
