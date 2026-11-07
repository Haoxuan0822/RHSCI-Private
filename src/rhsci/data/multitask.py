"""π0.5/GR00T 多库入口：保留各库身份，仅在采样层拼接。"""

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset

from rhsci.paths import dataset_paths


def add_multitask_arguments(parser, *, weighted=False):
    parser.add_argument(
        "--datasets", type=Path, nargs="+", help="Joint-training LeRobot roots, in order"
    )
    parser.add_argument(
        "--sidecars",
        type=Path,
        nargs="+",
        help="One ARDY sidecar per --datasets root, in the same order",
    )
    if weighted:
        parser.add_argument(
            "--dataset-weights",
            type=float,
            nargs="+",
            help="Positive dataset mixing ratios; joint-training default is equal datasets",
        )


def multitask_paths(args):
    """单库仍走原解析；多库显式逐一配对，首库字段仅兼容旧接口。"""
    roots = getattr(args, "datasets", None)
    labels = getattr(args, "sidecars", None)
    weights = getattr(args, "dataset_weights", None)
    if roots is None:
        if labels is not None or weights is not None:
            raise ValueError("--sidecars/--dataset-weights require --datasets")
        return dataset_paths(args)
    if args.dataset is not None or args.sidecar is not None:
        raise ValueError("Use either --dataset/--sidecar or --datasets/--sidecars")
    if len(roots) < 2 or len(set(root.resolve() for root in roots)) != len(roots):
        raise ValueError("--datasets requires at least two distinct roots")
    if args.model not in ("STATE", "NO-STATE") and labels is None:
        raise ValueError("Query joint training requires --sidecars")
    if labels is not None and len(labels) != len(roots):
        raise ValueError("--sidecars must have exactly one entry per dataset")
    if weights is not None and (
        len(weights) != len(roots) or not np.isfinite(weights).all() or min(weights) <= 0
    ):
        raise ValueError("--dataset-weights must contain one finite positive ratio per dataset")
    first = argparse.Namespace(**vars(args))
    first.dataset, first.sidecar = roots[0], labels[0] if labels else None
    paths = dataset_paths(first)
    paths.update(
        datasets=[root.resolve() for root in roots],
        sidecars=[label.resolve() for label in labels] if labels else [None] * len(roots),
    )
    if hasattr(args, "dataset_weights"):
        paths["dataset_weights"] = weights or [1.0] * len(roots)
    return paths


def training_sources(paths):
    return list(
        zip(paths.get("datasets", [paths["dataset"]]), paths.get("sidecars", [paths["sidecar"]]))
    )


class JointDataset(ConcatDataset):
    """原生 ConcatDataset 路由；GR00T 的 epoch 索引原样交给所属子库。"""

    def __init__(self, datasets):
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("Every training dataset needs at least one complete action window")
        super().__init__(datasets)
        contexts = [
            dataset.reader.manifest["context"]
            for dataset in datasets
            if getattr(dataset, "reader", None) is not None
        ]
        for field in (
            "teacher_pipeline",
            "teacher_hashes",
            "ardy_model_revision",
            "body_joint_order",
            "quaternion_convention",
            "skeleton_identity",
            "tokenizer_context_policy",
        ):
            if contexts and any(
                context.get(field) != contexts[0].get(field) for context in contexts[1:]
            ):
                raise ValueError(f"Joint ARDY labels have different {field}")

    def locate(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        child = bisect_right(self.cumulative_sizes, index)
        start = self.cumulative_sizes[child - 1] if child else 0
        return self.datasets[child], index - start

    def __getitem__(self, index):
        epoch, position = index if isinstance(index, tuple) else (None, index)
        dataset, local = self.locate(position)
        return dataset[(epoch, local) if epoch is not None else local]

    def raw_sample(self, index):
        dataset, local = self.locate(index)
        return dataset.raw_sample(local)


def build_training_dataset(paths, dataset_type, variant, processor=None, **kwargs):
    sources = training_sources(paths)
    if len(sources) > 1 and paths["input_cache"].root is not None:
        raise ValueError(
            "Joint training currently requires --no-input-cache; a single cache belongs to one dataset"
        )
    datasets = []
    for root, sidecar in sources:
        datasets.append(
            dataset_type(root, variant, processor, sidecar, cache=paths["input_cache"], **kwargs)
        )
    return datasets[0] if len(datasets) == 1 else JointDataset(datasets)


def dataset_children(dataset):
    return dataset.datasets if isinstance(dataset, JointDataset) else [dataset]


def preflight_indices(dataset, variant):
    """覆盖库内起始、完整历史、episode 边界；短轨迹不越界。"""
    from rhsci.data.visual_history import visual_history_offsets

    if not len(dataset):
        raise ValueError("No complete action windows in training dataset")
    candidates = [
        0,
        -visual_history_offsets(variant)[0],
        int(dataset.ends[0]) - 1,
        int(dataset.ends[0]),
    ]
    return list(dict.fromkeys(index for index in candidates if 0 <= index < len(dataset)))


def shared_statistics(paths, backbone):
    """多库共享数值尺度；单库仍读原 stats.json，不生成任何文件。"""
    if "datasets" not in paths:
        return None
    if "shared_statistics" in paths:
        return paths["shared_statistics"]
    if backbone == "gr00t":
        from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics

        # 原生函数：加权 mean/std，min/max 和 q01/q99 取各库包络。
        values = [
            {
                key: json.loads((root / "meta/stats.json").read_text())[key]
                for key in ("action", "states")
            }
            for root in paths["datasets"]
        ]
        stats = merge_statistics(values, paths["dataset_weights"])
    elif backbone == "pi05":
        from openpi.rhsci.normalization import joint_statistics

        stats = joint_statistics(paths["datasets"], paths["statistics_batch_size"])
    else:
        raise ValueError(backbone)
    for key, width in (("action", 36), ("states", 32)):
        for name in ("min", "max", "mean", "std", "q01", "q99"):
            value = np.asarray(stats[key][name])
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid joint statistics: {key}/{name}")
    paths["shared_statistics"] = stats
    return stats


def multitask_contract(paths, backbone):
    """把所有库、逐库标签和共享统计写入运行契约，部署无需源训练数据。"""
    from rhsci.data.ardy_sidecar import dataset_content_fingerprint
    from rhsci.runtime.artifacts import sha256

    return dict(
        datasets=[
            dict(
                fingerprint=dataset_content_fingerprint(root),
                sidecar=sha256(sidecar / "manifest.json") if sidecar else None,
            )
            for root, sidecar in training_sources(paths)
        ],
        sampling="native-sharded-mixture"
        if backbone == "gr00t"
        else "concatenated-all-frames-terminal-padding",
        weights=paths.get("dataset_weights"),
        statistics_method="native-quantile-envelope"
        if backbone == "gr00t"
        else "native-running-stats-action-chunks",
        statistics_batch_size=paths.get("statistics_batch_size"),
        shard_config=paths.get("shard_config"),
        statistics=shared_statistics(paths, backbone),
    )
