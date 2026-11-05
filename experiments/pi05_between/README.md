# SIMPLE evaluation worker

The worker starts the RHSCI HTTP server, checks one model response, then calls
SIMPLE's native evaluation loop. It supports π0.5 and GR00T, task levels,
video recording and continuation of unfinished episodes.

See [Evaluation](../../docs/EVALUATION.md) for simulator and asset versions,
zero-arm task configuration, client/server roles and an example command.
See [Inference](../../docs/INFERENCE.md) for standalone serving and optional state
measurement noise.

| Script | Purpose |
| --- | --- |
| `run_evaluation.py` | Start one server and run evaluation rounds |
| `eval_decoupled_wbc.py` | Configure renderer/DDS and call native SIMPLE evaluation |
| `summarize_results.py` | Read episode results, including resumed segments |
| `scene_references.py` | Resolve evaluation scene asset references |

Configure the filesystem constants in `run_evaluation.py` for your installation.
Use `--help` for arguments. The worker saves configuration, commands, logs and
per-episode results under the selected output root.

Other scripts in this directory analyze recorded observations, actions and
videos; they are not required for ordinary model inference.
