# Evaluation queue

These scripts prepare selected assets and schedule SIMPLE evaluation workers.
They reuse the [shared worker](../pi05_between/README.md) and task-specific
configuration in `plan.py`.

| Script | Purpose |
| --- | --- |
| `prepare.py` | Prepare model assets and worker receipts |
| `prepare_old_evaluation.py` | Download and verify versioned evaluation ZIPs |
| `plan.py` | Task names, step limits, thresholds and worker arguments |
| `run.py` | Dispatch prepared workers and continue unfinished runs |
| `prepare_three_task_downloads.py` | Prepare selected checkpoint downloads |
| `prepare_nostate.py` | Prepare native Ψ0 / GR00T NO-STATE assets |
| `legacy_inventory.py` | Freeze historical-only training cells for retesting |
| `prepare_retests.py` | Restore selected historical checkpoints and processors |
| `monitor_goal.py` | Check progress, continue exited workers and update result tables |
| `watch_hf_free_nodes.py` | Check free download nodes hourly and fetch new HF checkpoint assets after a real transfer succeeds |

Configure filesystem roots and model paths for your machine before running.
Use each entry's `--help` to inspect its options. Preparation and evaluation are
separate operations; verify asset receipts before launching workers.
See [Evaluation](../../docs/EVALUATION.md) for the protocol and client/server setup.
