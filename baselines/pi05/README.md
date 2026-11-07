# π0.5

Implementations: `src/openpi` and `src/openpi/rhsci`.
Entry: `bash scripts/train/pi05.sh --help`.

The adapter uses quantile-safe preprocessing and native optimization. Optional execution profiles control history batching and query recomputation; the default is `reference`. Supported variants are STATE, NO-STATE, VGQ-AE, VGQ-AE-240, VGQ-AE-480, VGQ-AE-Mo, VGQ-AE-Mo-240 and VGQ-AE-Mo-480. Pass `--dataset`, `--sidecar`, `--tokenizer`, `--checkpoint`, `--output-dir` and `--max-steps` for a SIMPLE G1 task. See [Training](../../docs/TRAINING.md).

For joint training, pass `--datasets ROOT_A ROOT_B` with matching `--sidecars LABELS_A LABELS_B` for Query variants. Use `--input-cache-root CACHE_PARENT` to reuse per-dataset RGB/table caches, or `--no-input-cache` for direct decoding. Each dataset keeps its own instructions and labels. All frames are eligible; terminal action chunks repeat the last action. Shared statistics use native OpenPI `RunningStats` over action chunks/current states and are saved for inference. `--statistics-batch-size` defaults to 256 and affects histogram quantiles and the final dropped partial batch. See [joint training](../../docs/TRAINING.md#joint-training-π05-and-gr00t) for the command and remaining training-recipe differences.

Use a separate Python 3.10 environment with the versions required by `src/openpi/rhsci/train.py`, the source `transformers_replace` overlay, and compatible π0.5 weights/tokenizer. `PI05_RHSCI_PYTHON` overrides the interpreter; otherwise it uses `$PSI_RUNTIME_ROOT/.venv-pi05-rsci/bin/python`.

## Environment and assets

Run from the repository root on Linux with a CUDA 12.8-compatible driver and shared FFmpeg libraries for TorchCodec:

```bash
python3.10 -m venv /opt/rsci-data/.venv-pi05-rsci
source /opt/rsci-data/.venv-pi05-rsci/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r baselines/pi05/requirements-rhsci.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PI05_RHSCI_PYTHON="$VIRTUAL_ENV/bin/python"
python -m openpi.rhsci.prepare_assets --install-transformers-patch
python -m openpi.rhsci.prepare_assets --check-transformers-patch
python -m pip check
```

The installer copies only the repository's `transformers_replace` overlay into the selected interpreter's Transformers 4.53.2 package. It changes that dedicated environment. Keep it separate from Ψ0, EgoVLA and GR00T; reinstalling Transformers requires reinstalling the overlay. The check compares every overlay Python file by SHA256, and training repeats the same check.

Obtain the native PyTorch `config.json` and unsharded `model.safetensors` from [the PSI0 asset repository](https://huggingface.co/USC-PSI-Lab/psi-model/tree/cb1480f50d9764ab2f6261bc2e651a67ad54da17/openpi/pi05_droid), revision `cb1480f50d9764ab2f6261bc2e651a67ad54da17`, and [the PaliGemma SentencePiece tokenizer](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model). Check the downloaded weights against the provider's LFS SHA256 before registration and comply with the provider's model/tokenizer terms. Put them outside this source repository, then register them:

```bash
python -m openpi.rhsci.prepare_assets \
  --checkpoint /models/pi05_droid \
  --tokenizer /models/paligemma_tokenizer.model \
  --repo USC-PSI-Lab/psi-model \
  --revision cb1480f50d9764ab2f6261bc2e651a67ad54da17
```

This validates the checkpoint format and tokenizer, then writes `RSCI_ASSETS.json` with relative filenames, sizes and SHA256 hashes. It neither downloads nor changes weights. Repository/revision fields are operator-supplied provenance, not proof of upstream authenticity. An existing matching inventory is retained; a different one is rejected. Pass the same checkpoint and tokenizer paths to training or `--check`. No model assets are included in this source distribution.

`--dry-run` prints the configuration. `--check-data` checks the RGB/ARDY datasets; `--check` also checks initialization/processing and audits `RSCI_ASSETS.json` if supplied. The manifest is optional; native local weights and tokenizer can be loaded directly. Changed model/data/training configurations require a separate run; source edits and compatible dependency updates do not.

State robustness evaluation uses shared [client measurement perturbations](../../docs/INFERENCE.md#state-measurement-noise). Training and server-side measurement noise have been removed. Reader/future-target ablations are currently PSI0-only.
