# Evaluation

RHSCI uses the SIMPLE simulator as the client and a separate model process as
the policy server. The client sends RGB, language and state over HTTP; the server
returns action chunks. Run the two processes in their respective environments.

## Versions and configuration

| Component | Configuration |
| --- | --- |
| SIMPLE | `b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11` plus the task source changes below |
| Physics / rendering | MuJoCo 3.3.6 / Isaac Sim 4.5.0.0 |
| Controller | Decoupled WBC; native wall-clock control and navigation |
| Evaluation inputs | `USC-PSI-Lab/psi-data@e78fb93cc28912a3031a10b8656d32d7f0a2b867`, `simple-eval/<task>.zip` |
| Initial arms | Zero joint targets in all 13 G1Sonic tasks |
| Sampling | π0.5: 10 steps; GR00T: 4 steps; BF16 |
| Action chunks | π0.5: 30 predicted / 30 executed; GR00T: 16 / 16; no RTC |
| Episodes | Levels 0/1/2, 10 initial states × 5 rounds per level |
| Video | Complete stereo video for the first round |
| State noise | Disabled by default; optional client measurement perturbations |

Task step limits and success thresholds are task-specific. For the current
worker settings, between-tables uses 1200 steps / 0.4, oven 1000 / 0.5,
handover 800 / 0.9 and faucet 1000 / 0.2.
Preserve the checkpoint's processor and normalization statistics.

## Simulator source

Install the pinned [official SIMPLE source](https://github.com/physical-superintelligence-lab/SIMPLE/tree/b75c6786f87b4a7f43b52b0ac5b83a1d9ea62e11)
in a separate directory and follow its installation guide. The source lock,
including submodule revisions, is in [`configs/simple/official.json`](../configs/simple/official.json).
Set `PSI_HOME` to this repository and `SIMPLE_HOME` to that checkout.

Before changing task source, check the clean official baseline:

```bash
python3 "$PSI_HOME/tools/check_simple.py" "$SIMPLE_HOME"
```

The official between-tables task already initializes both arms to zero. Apply
the equivalent task configuration to the other 12 tasks once:

```bash
git -C "$SIMPLE_HOME" apply --check "$PSI_HOME/configs/simple/zero_arm_tasks.patch"
git -C "$SIMPLE_HOME" apply "$PSI_HOME/configs/simple/zero_arm_tasks.patch"
```

This modifies task constructors directly; there is no runtime pose override.
The diff is based on SIMPLE's MIT-licensed source; retain its `license.md`.
`check_simple.py` checks the unmodified official baseline and will report these
intentional changes after application.

The newer `SIMPLE-eval/fb67cd9a` snapshot used in the README installation example
is a separate asset release. Do not combine its results with the ZIP version
listed above. Some newer tasks also require success-checker changes absent from
the pinned SIMPLE code. The pinned controller retains its upstream finger-order
mapping; the zero-arm diff does not change controller behavior.

## Server

Use the selected model's environment and the [inference commands](INFERENCE.md).
π0.5 and GR00T use `python -m rhsci.inference.http_server --backbone pi05|gr00t`.
Ψ0 uses `scripts/deploy/serve_psi0_simple.sh`. The SIMPLE HTTP endpoints are
`GET /health` and `POST /act`; message encoding is in `rhsci.inference.protocol`.
Model weights and normalization assets are downloaded separately; see [Assets](ASSETS.md).

## Client and multi-round worker

The client uses SIMPLE's `Psi0DecoupledWbcAgent` and `HttpActionClient`.
`experiments/pi05_between/eval_decoupled_wbc.py` configures rendering and DDS,
then calls the native SIMPLE `run_eval`. The state has 32 dimensions: 31 measured
joints followed by the height command. WBC uses the simulator's actual feedback.

`experiments/pi05_between/run_evaluation.py` starts a server and runs multiple
rounds. Configure its filesystem constants for your machine before use. Example:

```bash
python experiments/pi05_between/run_evaluation.py \
  --gpu 0 --variant STATE --level dr-level-0 \
  --evaluation-root "$PSI_RUNTIME_ROOT/eval-official-pre1102" \
  --output-root "$PSI_RUNTIME_ROOT/evals/pi05-between-state" \
  --repair-scene-references
```

The worker defaults to π0.5 between-tables, checkpoint 40000, five rounds of ten
episodes and first-round video. For GR00T, specify `--backbone gr00t`, its saved
run directory, checkpoint 20000 and `--expected-horizon 16`. Supply the correct
`--task`, `--max-episode-steps` and `--success-criteria` for other tasks.

The worker saves `worker.json`, per-round `command.sh`, `effective_eval_config.json`,
logs and episode statistics. `--resume` continues unfinished episodes; use the
same experiment configuration when continuing a run.

Noise enablement, type, severity, seed and protocol must match the saved worker
and all evaluation segments when resuming. Historical records without noise
configuration are treated as noise disabled.

To evaluate `VGQ-AE-Mo`, use `--variant VGQ-AE-Mo --run-dir <saved-run>`.
The worker passes the selected method to the model server, which checks it
against the checkpoint configuration before loading model weights. A method
mismatch is rejected rather than recorded under the requested label.

Optional noise flags and the four severity levels are documented under
[State measurement noise](INFERENCE.md#state-measurement-noise). Noise changes
only the outgoing measured state, before model normalization. It does not alter
the simulation state, images or WBC feedback.
