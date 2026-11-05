# Training

Run commands from the repository root, using the selected backbone's environment. See the [backbone guides](../README.md#backbones) for dependencies. Supply data and initialization assets locally; launchers do not install packages or download assets.

## From a fresh clone

Work from the checkout root on Linux with an NVIDIA driver compatible with the chosen PyTorch/CUDA build. Install Git, uv, shared FFmpeg libraries and zstd/tar as needed. Pure training does not require the SIMPLE simulator.

1. Choose one backbone and a supported variant. Install only its environment from the corresponding `baselines/<model>/README.md`; do not combine the different Transformers versions. Ψ0 and ACT use the root Python 3.10 uv environment; π0.5 has its own Python 3.10 environment and Transformers overlay; GR00T, EgoVLA and λ₀ have dedicated environments.
2. Set `PSI_HOME` to the checkout, `PSI_RUNTIME_ROOT` to external runtime storage, and the model's Python override. Keep data, initialization weights and training outputs outside the checkout; pass explicit paths. Ψ0 also needs the local `.env` described in its guide.
3. Restore the selected task's processed data from the [current RHSCI asset handoff](ASSETS.md#current-rhsci-handoff). Query variants need the matching formal ARDY labels. STATE/NO-STATE do not. Keep the archived normalization files; compute them only for new datasets before generating labels.
4. Download the selected backbone's initialization weights and processor/tokenizer. Ψ0 needs its VLM and Action Expert; π0.5 needs native PyTorch pi05_droid and the PaliGemma tokenizer; GR00T uses NVIDIA N1.6; EgoVLA uses its base VLM; λ₀ needs its base assets and prepared bundle; ACT needs the local ImageNet ResNet18 file. Ψ0 EGQ variants also need DINOv2. Existing RHSCI-trained checkpoints are only needed for continuation or inference.
5. Run the actual entry's `--help`, then its configuration/data checks. Most adapters use `--dry-run`, `--check-data` and `--check`; Ψ0 supports `--dry-run`/`--check`, while λ₀ additionally uses bundle/model checks. A dry run only prints configuration. A CPU check does not validate GPU memory or distributed training.
6. Launch with explicit GPUs, per-GPU batch, accumulation, task, variant, seed, budget and an independent output directory. Global batch equals GPU count × per-GPU batch × accumulation. Keep data/model/recipe fixed when resuming; rerun the same command against the same output to recover a complete checkpoint where supported. Retain model, optimizer, scheduler, RNG, processing/configuration and progress files together.

The root [Ψ0 VGQ-AE example](../README.md#training) shows preflight followed by training. Model guides list environment dependencies and required flags. Substitute your own absolute asset and output paths.

## Data preparation

RHSCI adapters currently consume the SIMPLE G1 schema with 36-dimensional actions and 50 Hz timing. Prepare LeRobot data and the normalization metadata required by the selected backbone. Query variants also require matching ARDY labels; state baselines do not use those labels.

Current Ψ0 validation defaults to the training repo; ACT evaluates a prefix of training data with native augmentation. These are training diagnostics, not independent held-out estimates. Other RHSCI workers do not automatically create a validation split. For a held-out study, split by original demonstration source, keep duplicate groups together and compute statistics on training data only. The handover pack contains 100 episodes but 93 unique motion sources.

Set `HF_HOME`, `HF_DATASETS_CACHE` and `TORCH_HOME` to external storage with sufficient space before invoking native data preflight. LeRobot can materialize large Arrow caches, including unused depth columns, even for an RGB experiment. Compressed archive size is not a cache-space estimate. Keep caches separate from source data and the checkout; preserve active runs' caches.

Follow [SIMPLE data preparation](DATA_PREPARATION.md) for the processed 50 Hz schema, `scripts/data/prepare_simple_stats.py` for both normalization files, and the pinned ARDY teacher/sidecar commands. `scripts/data/raw_to_lerobot_v2.py` is an upstream real-robot converter, not a SIMPLE replay converter. The retained sidecar builder verifies the original XMovePick task and its reference latent; it does not certify arbitrary tasks. λ₀ additionally requires a prepared bundle, shown below. Existing datasets and sidecars are listed in [Hugging Face assets](ASSETS.md).

If native configurations contain `/opt/rhsci` or `/opt/rsci-data` defaults, preview local path replacement with `python3 tools/configure_paths.py --data-root /absolute/path/to/rsci-data`, then repeat with `--apply`. Keep data outside the checkout. Restore defaults with `python3 tools/configure_paths.py --reset --apply` before committing. Configure `.env` separately as described in the [PSI0 guide](../baselines/psi0/README.md#installation).

## Common launcher

```bash
python3 scripts/train/baseline.py --help
DATASET_PATH=/datasets/my-task python3 scripts/train/baseline.py --backbone gr00t --mode native \
  --python /path/to/gr00t-env/bin/python --gpus 0,1 --dry-run -- \
  --dataset-path /datasets/my-task --base-model-path /models/GR00T-N1.6-3B \
  --embodiment-tag G1_LOCO_DOWNSTREAM \
  --modality-config-path "$PWD/src/gr00t/configs/modality/g1_locomanip.py" \
  --output-dir /runs/gr00t/my-task \
  --max-steps 20000 --global-batch-size 24 \
  --learning-rate 1e-4 --warmup-ratio 0.05 --weight-decay 1e-5 \
  --dataloader-num-workers 4 --save-steps 10000 --save-total-limit 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

Arguments before `--` belong to the launcher; arguments after it are passed as separate tokens to the trainer. `--dry-run` before `--` only prints the executable, arguments, working directory and selected environment. Remove it to execute. `--help` after `--` asks the native trainer for help and requires its dependencies. Native π0.5 also requires a local `.env`.

This native GR00T example uses NVIDIA N1.6 initialization and explicitly selects the SIMPLE optimization parameters. It requires the native SIMPLE dataset layout, including `meta/modality.json`; `DATASET_PATH` points to the same dataset as `--dataset-path`. For RHSCI VGQ-AE training and its dataset adapter, use the [README example](../README.md#gr00t-box-hugging-vgq-ae).

`--gpus` sets visible devices; otherwise the launcher inherits `CUDA_VISIBLE_DEVICES`. Process count defaults to the selected GPU count, or one. Native multi-node runs also accept `--nnodes`, `--node-rank`, `--master-addr` and `--master-port`; launch the same job on each node with a distinct rank. Relative native trainer arguments are interpreted from the working directory shown by dry-run. Pass absolute data, model and output paths.

The lightweight wrappers `act.sh`, `dp.sh`, `internvla-m1.sh`, `h-rdt.sh`, `dreamzero.sh` and `beingh05.sh` select native backbones through this common launcher. `act_rhsci.sh` and the other adapted backbones' wrappers directly select RHSCI training. All eleven models can also be selected through `baseline.py`.

## RHSCI adapters

```bash
python3 scripts/train/baseline.py --backbone pi05 --mode rhsci \
  --python /path/to/pi05-rsci-env/bin/python --gpus 0,1 -- \
  --task my_task --model VGQ-AE \
  --dataset /datasets/my_task --sidecar /labels/my_task \
  --checkpoint /models/pi05 --tokenizer /models/paligemma_tokenizer.model \
  --output-dir /runs/pi05/my_task --max-steps 40000 --check-data
```

Remove `--check-data` after preflight to train. π0.5, GR00T and EgoVLA accept `--dataset`, `--sidecar`, `--checkpoint`, `--output-dir`, `--max-steps`, batch/worker settings, and a plain `--task` run identifier. Dataset paths are explicit and sizes come from metadata. Other robot schemas need corresponding adapters. π0.5 accepts a native local initialization directory and tokenizer without `RSCI_ASSETS.json`. When a manifest is supplied, `--check` can audit its full hashes. Resume checks compare model, data and training configuration; source edits and compatible dependency updates do not invalidate a run.

PSI0 takes the exact repo ID through `--task`, its parent directory through `--data-root`, labels through `--sidecar-root`, and initialization paths through `--vlm-path` / `--ae-path`. It also accepts `--output-dir` and `--max-steps`. It uses the environment's NCCL by default; `--nccl-library` selects an explicitly validated alternate runtime.

π0.5 and GR00T also use the environment's NCCL by default. Their optional `--nccl-library` or `RHSCI_NCCL_LIBRARY` selects a library checked by `scripts/with_nccl_runtime.py`; `RHSCI_NCCL_VERSION` sets its expected version (default `2.31.2`). No local NCCL installation path is required by default.

For λ₀, prepare the bundle first:

```bash
PSI_RUNTIME_ROOT=/external/runtime LAMBDA0_PYTHON=/path/to/lambda0-env/bin/python \
  bash baselines/lambda0/Lambda0_Prepare.sh \
  --task my_task --dataset /datasets/my_task --sidecar /labels/my_task \
  --output /bundles/my_task

python3 scripts/train/baseline.py --backbone lambda0 --mode rhsci \
  --python /path/to/lambda0-env/bin/python --gpus 0 -- \
  --task my_task --variant VGQ-AE \
  --dataset /datasets/my_task --sidecar /labels/my_task \
  --bundle /bundles/my_task-safe --data-root /datasets \
  --checkpoint /models/lambda0-midtrain --output-dir /runs/lambda0/my_task \
  --max-steps 40000 --check-data
```

λ₀ uses its native dependency imports and model loader; the environment lock and source manifest are reproducibility references, not runtime version gates. Its inner `--dry-run` resolves local configuration and assets without a full weight hash scan; `--check-model` additionally checks asset hashes when a source receipt is available. The common launcher's outer `--dry-run` remains command-only.

## VGQ-AE-Mo

Select `--model VGQ-AE-Mo` for Ψ0, π0.5, GR00T or EgoVLA; λ₀ uses `--variant VGQ-AE-Mo`. Start a separate output directory for this method. Existing VGQ-AE commands retain their original behavior.

VGQ-AE-Mo keeps the four visual frames `[-6, -4, -2, 0]`, Reader, shared Action Expert and current ARDY supervision. Two independent LayerNorm/Linear heads read the same auxiliary Query output: one predicts `z(t)`, the other predicts `z(t) - z(t - Δ)`. The Query and Action Expert are not duplicated. Inference uses the current RGB and language through the existing action path, without either auxiliary head.

`--ardy-delta-frames` sets Δ (default `2`, or 40 ms at 50 Hz); `--motion-weight` sets the difference MSE weight (default `0.1`). The loss is `L_action + 0.1 L_ARDY + motion_weight L_motion`; Ψ0 retains its configurable current-ARDY weight. Differences use the existing 128-dimensional ARDY coordinates, without division by elapsed time or additional normalization. No teacher rerun or sidecar regeneration is required.

Both labels must be valid and belong to the same episode. Missing history or an invalid endpoint masks only the difference loss, keeping the existing action/current-ARDY sample. With the standard six-frame ARDY history requirement and Δ=2, difference supervision starts at frame 8 (zero-based). Reader removal and future-current-target switches are unavailable for this method.

For example, adapt the π0.5 command above with:

```bash
--model VGQ-AE-Mo --ardy-delta-frames 2 --motion-weight 0.1 \
--output-dir /runs/pi05/my_task/VGQ-AE-Mo
```

## Experimental switches

| Training option | Backbones | Default | Effect |
| --- | --- | --- | --- |
| `--no-reader` | PSI0 Reader query variants, except STATE-RECONSTRUCTION and VGQ-AE-Mo | Reader enabled | Use all valid tokens from each of the four history frames, without learned Reader compression. |
| `--ardy-offset N` | PSI0 Reader query variants, except STATE-RECONSTRUCTION and VGQ-AE-Mo | `0` | Predict the ARDY target at `t + N` in the same episode. At 50 Hz, `10` is 200 ms. |

For example, configure PSI0 VGQ-AE without a Reader and with a 200 ms future target:

```bash
bash scripts/train/psi0.sh --model VGQ-AE --task my_task \
  --data-root /datasets/simple --sidecar-root /labels/my_task \
  --vlm-path /models/psi0-vlm --ae-path /models/psi0-action-expert \
  --output-dir /runs/psi0/my_task-full-token-future \
  --no-reader --ardy-offset 10 --dry-run
```

Remove `--dry-run` after asset preparation to train. Both switches can also be used independently. Visual history remains `[-6, -4, -2, 0]`: future images, states and actions are never added to the policy input. A target beyond the episode end is zero-filled and masked out of the auxiliary loss; the action sample remains in training. The future teacher validity mask and the current history validity mask must both pass. Asset preflight rejects a future offset with no valid supervision in the dataset.

`VGQ-VLM` now selects the four-original-image direct Query, requires `--vlm lora`, and has no Reader. `STATE-RECONSTRUCTION` uses its fixed Reader/AE and state31 labels. Neither takes the Reader/future-target switches.

Both specialized Ψ0 methods use the original entry's `1e-4` VLM learning rates (only relevant when the VLM is trainable). To compare against an existing PSI0-RAW run, match its actual initialization, data/statistics, world size, per-device batch, accumulation, schedule, precision and execution profile; generic CLI defaults are not a reconstruction of every historical run.

STATE and NO-STATE reject Reader and ARDY switches. Reader selection is saved in the PSI0 model configuration; strict loading rejects mismatched Reader weights. Future offset is saved in the data configuration. Both alter the run contract and default output path.

State measurement perturbations are evaluation-only and applied by the [SIMPLE client](INFERENCE.md#state-measurement-noise). Training Gaussian noise and server-side Gaussian noise have been removed. Native action sampling noise and image augmentation remain unchanged.

## ACT and Diffusion Policy

ACT also supports RHSCI STATE, NO-STATE and EGQ-AE through `scripts/train/act_rhsci.sh`, or `--backbone act --mode rhsci` in the common launcher. Supply explicit `--dataset`, `--sidecar` (EGQ-AE), `--output-dir` and external `--torch-home` containing ImageNet ResNet18 weights. See [ACT](../baselines/act/README.md) for preflight, full-checkpoint recovery and video-cache behavior. The commands below select upstream native pipelines.

The JSON argument lists retain upstream SIMPLE G1 model/normalization defaults, without task names, GPU assignments or machine paths. Copy and edit them for other robot schemas or optimization budgets. They are consumed directly, without shell evaluation.

```bash
bash scripts/train/act.sh --python /path/to/act-env/bin/python --gpus 0,1 \
  --config simple_act_config --args-file configs/train/act.json -- \
  --exp my_task --data.root-dir /datasets --data.train-repo-ids my_task \
  --train.output-dir /runs/act

bash scripts/train/dp.sh --python /path/to/dp-env/bin/python --gpus 0,1 \
  --config simple_dp_config --args-file configs/train/dp.json -- \
  --exp my_task --data.root-dir /datasets --data.train-repo-ids my_task \
  --train.output-dir /runs/dp
```

Select `real_act_config` / `real_dp_config` and supply the corresponding transforms for real robot data. Do not reuse SIMPLE dimensions without checking the schema. Change values in a copied argument file when tuning a preset; avoid duplicate native options.

## Being-H0.5

The [native training guide](../baselines/beingh05/README.md) uses `scripts/train/beingh05.sh`, or `--backbone beingh05` in the common launcher. Its `--config` selects a native dataset YAML. SIMPLE and RHSCI adaptation are not implemented; native dataset formats and checkpoint semantics apply.

## InternVLA-M1

```bash
bash scripts/train/internvla-m1.sh --python /path/to/internvla-env/bin/python --gpus 0,1 \
  --config /path/to/my_training.yaml -- \
  --datasets.vla_data.data_root_dir /datasets/my_task \
  --run_root_dir /runs/internvla --run_id my_task
```

Start from `src/InternVLA-M1/InternVLA/config/training/internvla_cotrain_humanoid.yaml`. Configure its VLM, DINO, initialization checkpoint, dataset mixture and schema before training. Native overrides use the upstream OmegaConf CLI; paths resolve from `src/InternVLA-M1`.

## H-RDT

```bash
bash scripts/train/h-rdt.sh --python /path/to/hrdt-env/bin/python --gpus 0,1 \
  --config /path/to/my_hrdt.yaml -- \
  --dataset_type finetune --dataset_name lerobot --dataset_root /datasets/my_task \
  --pretrained_model_name_or_path /models/hrdt \
  --output_dir /runs/hrdt/my_task --max_train_steps 40000
```

Configure local vision/language assets and language embeddings in the H-RDT environment. The trainer's default relative config paths resolve from `src/h_rdt`. Data preparation is explicit; the launcher does not generate embeddings or modify datasets.

## DreamZero

```bash
bash scripts/train/dreamzero.sh --python /path/to/dreamzero-env/bin/python --gpus 0,1 -- \
  data=dreamzero/simple train_architecture=full \
  simple_data_root=/datasets/dreamzero \
  pretrained_model_path=/models/DreamZero-AgiBot \
  dit_version=/models/Wan2.1-I2V-14B-480P \
  tokenizer_path=/models/umt5-xxl \
  text_encoder_pretrained_path=/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
  image_encoder_pretrained_path=/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  vae_pretrained_path=/models/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
  output_dir=/runs/dreamzero/my_task max_steps=20000
```

This invokes the native Hydra entry. Supply the dataset mixture, action/state dimensions, camera/frame settings and component-loading configuration that match the checkpoint. Native configuration files are in `src/dreamzero/groot/vla/configs`. SIMPLE data conversion requires the external camera calibration assets expected by the upstream converter. The generic launch command is not a claim that an arbitrary checkpoint/dataset pair has been validated.

## Native PSI0, π0.5, GR00T, EgoVLA and λ₀

Use `baseline.py --mode native` with the selected backbone. PSI0 takes a `psi.config.train` module through `--config`; π0.5 takes an `openpi.training.config` name. GR00T forwards finetuning options to `gr00t.experiment.launch_finetune`. λ₀ takes an experiment JSON through `--config`. EgoVLA forwards HF-style options to its native trainer; this source snapshot does not include the complete native `llava/data` pipeline, so the native path needs that upstream dependency before it can run. The RHSCI EgoVLA reader is provided separately.

## Explicit raw input caches

ACT, π0.5, GR00T and EgoVLA accept `--input-cache-root PATH` (the parent of `<dataset-name>/manifest.json`) or `--no-input-cache`. Without either option, the CLI accepts its existing cache environment variable; unset means disabled. The resolved choice is passed to child processes. Direct Python dataset users pass `cache=InputCacheConfig(...)`; datasets no longer inspect environment variables.

Automatic training recovery requires complete optimizer, scheduler, RNG and progress state. Inference only requires the model and its saved preprocessing configuration.
