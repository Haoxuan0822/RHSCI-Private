"""恢复时比较实验配置；源码清单和依赖版本属于记录，不是运行许可证。"""

import json
from pathlib import Path

# 中文说明：旧版本将这些审计字段混入摘要。复用旧记录可继续核对旧完整点，
# 同时允许重命名、注释修改和兼容的依赖升级，不改写历史记录。
_AUDIT_FIELDS = {
    "code",
    "sources",
    "source",
    "source_sha256",
    "dependencies",
    "versions",
    "environment",
    "upstream",
    "upstream_patch",
    "shared_ardy_reader",
    "shared_paths",
    "shared_runtime",
    "initialization_stat",
}


def training_configuration(contract):
    """返回与实验兼容性有关的字段；保留模型、数据、优化器及调度配置。"""
    configuration = {key: value for key, value in contract.items() if key not in _AUDIT_FIELDS}
    if isinstance(configuration.get("execution"), dict):
        configuration["execution"] = {
            key: value
            for key, value in configuration["execution"].items()
            if key not in {"nccl_runtime", "nccl_library", "nccl_library_sha256", "cuda_allocator"}
        }
    return configuration


def reuse_run_contract(current, output, filename="launch_contract.json"):
    """兼容时返回已保存记录以沿用断点摘要；配置变化时报错，不修改文件。"""
    record = Path(output) / filename
    if not record.is_file():
        return current
    saved = json.loads(record.read_text())
    expected = json.dumps(training_configuration(saved), sort_keys=True)
    actual = json.dumps(training_configuration(current), sort_keys=True)
    if actual != expected:
        raise ValueError("Existing run has different training configuration; use a new run-tag")
    return saved
