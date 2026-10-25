# Being-H0.5

Official source is in `src/beingh05`, including the Apache 2.0 license and upstream documentation. [SOURCE_MANIFEST.json](../../src/beingh05/SOURCE_MANIFEST.json) records the revision. The runtime RNG snapshot is excluded. This integration provides native training only: SIMPLE G1 adaptation, RHSCI variants and the shared RHSCI HTTP evaluation protocol are not implemented.

Create a separate Python 3.10 environment using the [official installation instructions](../../src/beingh05/README.md#installation) and `src/beingh05/requirements.txt`. Prepare the VLM, action expert, Being-H checkpoint and a dataset YAML matching the official schema outside the checkout.

```bash
bash scripts/train/beingh05.sh \
  --python /path/to/beingh05-env/bin/python --gpus 0,1,2,3 \
  --config /datasets/my_task/beingh05.yaml \
  --args-file configs/train/beingh05.json --dry-run -- \
  --mllm_path /models/InternVL3_5-2B --expert_path /models/Qwen3-0.6B \
  --resume_from /models/Being-H05-2B --resume_model_only True \
  --output_dir /runs/beingh05/my_task --logging_dir /runs/beingh05/my_task/logs
```

The [argument file](../../configs/train/beingh05.json) retains model and optimization options from the official [single-embodiment example](../../src/beingh05/scripts/train/train_libero_example.sh), including its native MPG and RTC choices. It contains no task, GPU assignment or asset paths. Copy and edit it for your checkpoint and data; image/action dimensions, token budgets and normalization still need to match the dataset YAML. For multiple embodiments, follow the [cross-embodiment example](../../src/beingh05/scripts/train/train_cross_emb_example.sh). Argument files are JSON lists of strings, without shell evaluation.

`--config` becomes native `--dataset_config_file`. Arguments after `--` go unchanged to the trainer; paths should be absolute. Dry-run prints the command, environment and working directory without importing model dependencies. Remove it to execute. Multi-node options are shared with [the common launcher](../../docs/TRAINING.md#common-launcher). Native training has not been exercised in this standalone release.

Use [upstream training documentation](../../src/beingh05/docs/training.md) for dataset metadata, normalization and checkpoint recovery. `--resume_model_only True` loads model weights, not optimizer/progress state. The official trainer may create its runtime RNG snapshot under its `configs` directory; do not include generated runtime files in a public export.
