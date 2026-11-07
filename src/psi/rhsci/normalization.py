"""Ψ0 联合训练的共享 bounds；沿用原生归一化及配置保存格式。"""

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from psi.utils import resolve_data_path
from rhsci.runtime.artifacts import atomic_json


def merge_training_bounds(data_root, repo_ids):
    """合并训练库已准备的逐维极值；不读验证集，不平均或合并分位数。"""
    if len(repo_ids) < 2 or len(set(repo_ids)) != len(repo_ids):
        raise ValueError("Shared bounds require at least two distinct training datasets")
    merged = {}
    sources = []
    for repo in repo_ids:
        path = resolve_data_path(Path(data_root) / repo / "meta/stats_psi0.json")
        content = path.read_bytes()
        stats = json.loads(content)
        for key, size in (("action", 36), ("states", 32)):
            lo = np.asarray(stats[key]["min"], dtype=np.float64)
            hi = np.asarray(stats[key]["max"], dtype=np.float64)
            if lo.shape != (size,) or hi.shape != (size,):
                raise ValueError(f"Shared SIMPLE bounds require {key} shape ({size},): {path}")
            if not np.isfinite([lo, hi]).all() or np.any(lo > hi):
                raise ValueError(f"Invalid {key} bounds in {path}")
            if key in merged:
                lo = np.minimum(merged[key]["min"], lo)
                hi = np.maximum(merged[key]["max"], hi)
            merged[key] = {"min": lo.tolist(), "max": hi.tolist()}
        sources.append({"repo_id": repo, "stats_sha256": hashlib.sha256(content).hexdigest()})
    merged["rhsci_shared_bounds"] = {"schema": 1, "sources": sources}
    return merged


def prepare_shared_statistics(cfg):
    """准备自动共享统计并填入当前配置；已有同名文件不匹配时拒绝覆盖。"""
    field = cfg.data.transform.field
    if field.action_norm_type != "bounds":
        raise ValueError("Automatic joint statistics support bounds normalization only")
    stats = merge_training_bounds(cfg.data.root_dir, cfg.data.train_repo_ids)
    destination = Path(field.stat_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 两个启动进程不能覆盖同一实验的不同统计；发布仍用公共原子JSON写入。
    with destination.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if json.loads(destination.read_text()) != stats:
                raise ValueError("Shared normalization statistics changed; use a new output directory")
        else:
            atomic_json(destination, stats)
    field.populate_stats(stats)
    return destination
