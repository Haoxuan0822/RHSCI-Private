# λ₀

Backbone: `src/openlambda`. Adapter: `src/rsci/backbones/lambda0` (`rsci.backbones.lambda0`).
Entry: `bash scripts/train/lambda0.sh --help`.

The adapter uses native SIMPLE preparation, processing, flow/RTC, optimization and serving. VGQ-AE adds training-only Reader/query supervision. Normalization falls back to observed bounds for nonconstant dimensions with degenerate quantiles; this differs from upstream quantile-only normalization.

Use the separate environment specified by [environment.lock.json](environment.lock.json). `LAMBDA0_PYTHON` overrides `$PSI_RUNTIME_ROOT/.venv-lambda0-rsci/bin/python`. `Lambda0_Prepare.sh` prepares local bundles; it does not start training.

The source manifest checks the vendored files against their SHA256 digests. [The integration patch](upstream.patch) records changes to the native interfaces.

VGQ-AE reports `flow + 0.1 × ARDY` as total loss in both training and validation. STATE reports native flow loss. Validation uses local masked ARDY reduction before the native metric aggregation, so uneven validation ranks do not enter collectives inside the model forward pass. Inference runs only the native action branch.
