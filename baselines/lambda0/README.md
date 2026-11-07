# λ₀

Backbone package: `src/lambda0`; upstream configs/tools: `src/openlambda`. Adapter: `src/lambda0/rhsci` (`lambda0.rhsci`).
Entry: `bash scripts/train/lambda0.sh --help`.

The adapter uses native SIMPLE preparation, processing, flow/RTC, optimization and serving. VGQ-AE adds training-only Reader/query supervision. VGQ-AE-240/480 and the corresponding Mo variants use the same architecture; see [visual history and motion targets](../../docs/TRAINING.md#vgq-ae-480-and-vgq-ae-mo-480). Normalization falls back to observed bounds for nonconstant dimensions with degenerate quantiles; this differs from upstream quantile-only normalization.

Use the separate environment specified by [environment.lock.json](environment.lock.json). `LAMBDA0_PYTHON` overrides `$PSI_RUNTIME_ROOT/.venv-lambda0-rsci/bin/python`. `Lambda0_Prepare.sh` prepares local bundles; it does not start training.

The source manifest supports an explicit audit of vendored files against their SHA256 digests; training and serving do not run this audit automatically. [The integration patch](upstream.patch) records changes to the native interfaces.

VGQ-AE reports `flow + 0.1 × ARDY` as total loss in both training and validation. STATE reports native flow loss. Validation uses local masked ARDY reduction before the native metric aggregation, so uneven validation ranks do not enter collectives inside the model forward pass. Inference runs only the native action branch.

The environment lock records a known setup; runtime does not require every exact package version. Model/API compatibility is determined by native loading.

## Guardian lifecycle

The guardian forwards SIGTERM/SIGINT to its worker process group and waits for worker/rank cleanup before releasing the training lock. The worker inherits the lock FD, preventing another launch while it remains alive after a guardian SIGKILL; SIGKILL itself cannot be forwarded. Foreground Ctrl+C still exits only the log viewer.
