# λ₀

Backbone package: `src/lambda0`; upstream configs/tools: `src/openlambda`. Adapter: `src/lambda0/rhsci` (`lambda0.rhsci`).
Entry: `bash scripts/train/lambda0.sh --help`.

The adapter uses native SIMPLE preparation, processing, flow/RTC, optimization and serving. VGQ-AE adds training-only Reader/query supervision. Normalization falls back to observed bounds for nonconstant dimensions with degenerate quantiles; this differs from upstream quantile-only normalization.

Use the separate environment specified by [environment.lock.json](environment.lock.json). `LAMBDA0_PYTHON` overrides `$PSI_RUNTIME_ROOT/.venv-lambda0-rsci/bin/python`. `Lambda0_Prepare.sh` prepares local bundles; it does not start training.

The source manifest supports an explicit audit of vendored files against their SHA256 digests; training and serving do not run this audit automatically. [The integration patch](upstream.patch) records changes to the native interfaces.

VGQ-AE reports `flow + 0.1 × ARDY` as total loss in both training and validation. STATE reports native flow loss. Validation uses local masked ARDY reduction before the native metric aggregation, so uneven validation ranks do not enter collectives inside the model forward pass. Inference runs only the native action branch.

The environment lock records a known setup; runtime does not require every exact package version. Model/API compatibility is determined by native loading. See [runtime check policy](../../docs/RUNTIME_CHECKS_20261031.md).

## Guardian lifecycle (2026-11-02)

The guardian forwards SIGTERM/SIGINT to its worker process group and waits for worker/rank cleanup before releasing the training lock. The worker inherits the lock FD, preventing another launch while it remains alive after a guardian SIGKILL; SIGKILL itself cannot be forwarded. Foreground Ctrl+C still exits only the log viewer. See the [source comparison, process tests and limits](../../docs/GUARDIAN_RECOVERY_AUDIT_20261102.md).

2026-11-03: [Lifecycle and readability review](../../docs/CODE_REVIEW_20261103.md). Stop-signal handling now spans Popen startup, launch-record callbacks and exit cleanup; rank cleanup finishes before the guardian releases its lock. Hard-kill protection still relies on the worker inheriting the lock.

2026-11-03: The CLI stays at `lambda0.rhsci.train`; `config` owns arguments and the typed native JSON configuration, `preflight` owns data/output/model checks, and `worker.execute_training` owns resolver/backend execution. The guardian imports configuration directly. See the [interface map and compatibility checks](../../docs/CODE_REVIEW_20261103.md#参考-simple-的职责与接口整理2026-11-03).
