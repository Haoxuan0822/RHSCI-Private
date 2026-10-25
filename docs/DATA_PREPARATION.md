# SIMPLE training data preparation

Run from the RHSCI checkout root. The [README](../README.md#getting-started) defines `PSI_HOME`, `PSI_RUNTIME_ROOT`, `PSI0_PYTHON`, `HUG_DATASET` and `HUG_SIDECAR`. Keep all data, teacher weights and external repositories outside the checkout. This guide prepares normalization and ARDY labels for an already processed SIMPLE dataset; it does not convert human/real-robot trajectories into simulated G1 trajectories.

## Initialization weights

After installing the two model environments and editing `.env` as shown in the README, download the compatible initialization weights. The PSI0 downloader flattens each selected remote directory into its matching local path:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$PSI0_PYTHON" scripts/data/download.py \
  --repo-id USC-PSI-Lab/psi-model --repo-type model \
  --remote-dir psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k --local-dir "$PSI_VLM"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$PSI0_PYTHON" scripts/data/download.py \
  --repo-id USC-PSI-Lab/psi-model --repo-type model \
  --remote-dir psi0/postpre.1by1.pad36.2601131206.ckpt.he30k --local-dir "$PSI_AE"
HF_HUB_OFFLINE=0 "$GROOT_RSCI_PYTHON" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="nvidia/GR00T-N1.6-3B", local_dir=os.environ["GROOT_BASE"])'
```

Accept any access terms and authenticate with Hugging Face if required. These are initialization weights, not RHSCI-trained policies or ARDY labels. Launchers use offline loading after preparation.

GR00T training starts from the downloaded NVIDIA official N1.6 weights. No additional HE pretraining is required. See the [optimization recipe](../baselines/gr00t/README.md#simple-recipe) for training parameters and reference differences.

## Processed training data

Supply a SIMPLE G1 LeRobot v2 dataset with `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`, episode parquet files under `data/chunk-*/` and RGB video files under `videos/`. Metadata must declare 50 FPS, a 36-dimensional `action`, and a 32-dimensional `states`. Video paths and frame identities must agree with metadata. Use training episodes here, not the evaluation split that contains simulator resets.

The `states` order is 14 hand joints, 14 arm joints, three measured waist angles and the previous commanded height. Actions use the same first 28 joint values followed by roll, pitch, yaw, height, torso velocity x/y/yaw and target yaw. Do not assemble the state by slicing the first 32 dimensions of the 43D raw G1 joint vector.

Processed SIMPLE replay/conversion must preserve this schema and its episode mapping to the original raw capture. Keep that mapping and raw base poses if rebuilding ARDY labels. The upstream `scripts/data/raw_to_lerobot_v2.py` converts real-robot recordings at 30 FPS; it is not the converter for these SIMPLE examples. Existing processed datasets and matching labels are listed in [Hugging Face assets](ASSETS.md). Preserve their archived statistics and metadata when restoring them.

## Generate both normalization files

After installing the PSI0 environment, run:

```bash
"$PSI0_PYTHON" scripts/data/prepare_simple_stats.py --dataset "$HUG_DATASET"
```

The tool reads only `action` and `states`, checks dimensions and finite values, and computes per-dimension min, max, mean, population standard deviation, q01 and q99 over the supplied training frames. It writes `meta/stats.json` for GR00T and `meta/stats_psi0.json` for PSI0/ACT. Other keys in an existing statistics object are preserved. Both files use the same physical action/state ordering; model-specific padding remains in the model processor.

Matching existing statistics are left byte-for-byte unchanged. Conflicts stop before either file is changed. Use `--overwrite` only when intentionally preparing normalization for a new training run; never replace the saved bounds used by an existing trained checkpoint. This tool holds the numeric columns in memory to compute exact quantiles, but does not load RGB or depth arrays.

Generate statistics **before** building sidecars: ARDY's dataset fingerprint covers metadata and parquet files, including the normalization JSON. If any fingerprinted file changes afterward, rebuild the labels in a new sidecar directory. Do not manually edit the fingerprint in a manifest.

## Prepare the pinned ARDY teacher

The following uses the [official ARDY source](https://github.com/nv-tlabs/ardy/tree/693f74d13b3d04a0a22ce127ee79c929dd89756b) and the model revision in `src/rsci/data/ardy_teacher_artifacts.py`. Use a separate environment: that source's Transformers dependencies differ from PSI0 and GR00T. CUDA builds must match your driver and GPU.

```bash
export ARDY_SOURCE="$PSI_RUNTIME_ROOT/third_party/ardy"
export ARDY_TEACHER="$PSI_RUNTIME_ROOT/cache/checkpoints/ARDY-G1-RP-25FPS-Horizon52"
export ARDY_PYTHON="$PSI_RUNTIME_ROOT/.venv-ardy/bin/python"
mkdir -p "$PSI_RUNTIME_ROOT/third_party"
git clone https://github.com/nv-tlabs/ardy.git "$ARDY_SOURCE"
git -C "$ARDY_SOURCE" checkout 693f74d13b3d04a0a22ce127ee79c929dd89756b
uv venv --python 3.11 "$PSI_RUNTIME_ROOT/.venv-ardy"
uv pip install --python "$ARDY_PYTHON" torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ARDY_PYTHON" -e "$ARDY_SOURCE" \
  numpy==1.26.4 pandas==2.2.3 pyarrow==20.0.0 mujoco==3.3.6
HF_HUB_OFFLINE=0 "$ARDY_PYTHON" scripts/data/download_ardy_teacher.py \
  --output-dir "$ARDY_TEACHER"
```

The downloader verifies all eight pinned tokenizer/configuration/statistics artifacts before publishing the teacher directory. The sidecar path instantiates only the tokenizer; it does not need the gated text-generation model. Follow the upstream source's installation instructions for additional platform requirements. This is an installation recipe, not a claim of a completed full teacher CUDA run in a fresh environment.

## Build reference-verified sidecars

The retained `build_simple_ardy_sidecars.py` is a **reference-specific builder for `G1WholebodyXMovePickTeleop-v0`**. Both raw and processed roots must end in that exact task name and declare 50 FPS. Raw data must contain the ordered 43D `observation.state`, 7D `observation.base_pose` in xyz + wxyz order, actions, timestamps and `meta/replay.txt`. Processed data additionally needs the 15D leg, 14D arm and 14D hand observation columns used to prove raw/processed frame alignment. The ordinary training adapters do not need all of these raw columns.

With the matching original raw capture and processed training data available, run:

```bash
export RAW_DATASET=/absolute/path/to/raw/G1WholebodyXMovePickTeleop-v0
export REFERENCE_DATASET=/absolute/path/to/processed/G1WholebodyXMovePickTeleop-v0
export REFERENCE_SIDECAR="$PSI_RUNTIME_ROOT/data/simple-ardy-sidecars/G1WholebodyXMovePickTeleop-v0"
"$PSI0_PYTHON" scripts/data/prepare_simple_stats.py --dataset "$REFERENCE_DATASET"
PYTHONPATH="$PSI_HOME/src" "$ARDY_PYTHON" scripts/data/build_simple_ardy_sidecars.py \
  --raw-root "$RAW_DATASET" --processed-root "$REFERENCE_DATASET" \
  --teacher-dir "$ARDY_TEACHER" --ardy-source-root "$ARDY_SOURCE" \
  --output-root "$REFERENCE_SIDECAR" --formal-verified
```

This checks exact action/joint episode alignment, source revision, teacher hashes and the pinned processed-episode-0/raw-episode-1 latent digest. Omitting `--formal-verified` creates candidate labels, which the training preflight rejects. Repeating an interrupted build with the same inputs and `--resume` checks the saved context before continuing. Do not use `--episode-limit` for the full training recipe.

The fixed reference digest does not certify another task or a reordered/subsampled dataset. For other tasks, supply their separately verified formal labels; this builder cannot generate them by renaming directories or copying a manifest. The README uses the box-hugging task `G1WholebodyPickAndPlaceAndHugContainerTeleop-v0`, which is distinct from XMovePick and requires its own matching data, formal labels and evaluation task. The reference commands above are separate from that box-hugging run. Use the matching box-hugging archives in [Hugging Face assets](ASSETS.md); a general formal-label generation workflow for other tasks is not provided here.

## Validate training inputs

Return to the [README training examples](../README.md#training). Their `--check` commands validate the model assets and read a real processed sample with matching formal labels before training starts. PSI0 takes the dataset parent plus task name; GR00T takes the dataset directory directly. Both examples must resolve to the same processed data and the sidecar manifest fingerprint.
