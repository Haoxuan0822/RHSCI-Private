# π0.5

Implementations: `src/openpi` and `src/rsci/backbones/pi05`.
Entry: `bash scripts/train/pi05.sh --help`.

The adapter uses quantile-safe preprocessing and native optimization. Optional execution profiles control history batching and query recomputation; the default is `reference`. Supported variants are STATE, NO-STATE and VGQ-AE. Pass `--dataset`, `--sidecar`, `--tokenizer`, `--checkpoint`, `--output-dir` and `--max-steps` for a SIMPLE G1 task. See [Training](../../docs/TRAINING.md).

Use a separate Python 3.10 environment with the versions required by `src/rsci/backbones/pi05/train.py`, the source `transformers_replace` overlay, and the π0.5 checkpoint/tokenizer asset manifest. `PI05_RSCI_PYTHON` overrides the interpreter; otherwise it uses `$PSI_RUNTIME_ROOT/.venv-pi05-rsci/bin/python`.

## Environment and assets

Run from the repository root on Linux with a CUDA 12.8-compatible driver and shared FFmpeg libraries for TorchCodec:

```bash
python3.10 -m venv /opt/rsci-data/.venv-pi05-rsci
source /opt/rsci-data/.venv-pi05-rsci/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r baselines/pi05/requirements-rsci.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PI05_RSCI_PYTHON="$VIRTUAL_ENV/bin/python"
python -m rsci.backbones.pi05.prepare_assets --install-transformers-patch
python -m rsci.backbones.pi05.prepare_assets --check-transformers-patch
python -m pip check
```

The installer copies only the repository's `transformers_replace` overlay into the selected interpreter's Transformers 4.53.2 package. It changes that dedicated environment. Keep it separate from Ψ0, EgoVLA and GR00T; reinstalling Transformers requires reinstalling the overlay. The check compares every overlay Python file by SHA256, and training repeats the same check.

Obtain the native PyTorch `config.json` and unsharded `model.safetensors` from [the PSI0 asset repository](https://huggingface.co/USC-PSI-Lab/psi-model/tree/cb1480f50d9764ab2f6261bc2e651a67ad54da17/openpi/pi05_droid), revision `cb1480f50d9764ab2f6261bc2e651a67ad54da17`, and [the PaliGemma SentencePiece tokenizer](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model). Check the downloaded weights against the provider's LFS SHA256 before registration and comply with the provider's model/tokenizer terms. Put them outside this source repository, then register them:

```bash
python -m rsci.backbones.pi05.prepare_assets \
  --checkpoint /models/pi05_droid \
  --tokenizer /models/paligemma_tokenizer.model \
  --repo USC-PSI-Lab/psi-model \
  --revision cb1480f50d9764ab2f6261bc2e651a67ad54da17
```

This validates the checkpoint format and tokenizer, then writes `RSCI_ASSETS.json` with relative filenames, sizes and SHA256 hashes. It neither downloads nor changes weights. Repository/revision fields are operator-supplied provenance, not proof of upstream authenticity. An existing matching inventory is retained; a different one is rejected. Pass the same checkpoint and tokenizer paths to training or `--check`. No model assets are included in this source distribution.

`--dry-run` prints the configuration. `--check-data` checks the RGB/ARDY datasets; `--check` also checks initialization weights and the tokenizer against `RSCI_ASSETS.json`. Changed execution profiles or source contracts require a separate run.

STATE accepts `--state-measurement-noise-std-rad S` for raw measured-joint training noise before normalization and tokenization. The flag alone selects `0.02` rad; omission disables it. Inference noise is controlled separately through the [HTTP server](../../docs/INFERENCE.md#state-measurement-noise), and stays off regardless of the training setting. Reader/future-target ablations are currently PSI0-only.
