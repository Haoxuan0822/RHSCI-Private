"""只读对照原 PSI0-RAW 的 RGB 预处理；不加载大模型、不启动训练。"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def load_source_module(name, path):
    """加载只读参照模块；相对骨干依赖来自当前测试环境，不安装或修改原仓库。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_equal(original, current):
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(original, dict):
        assert original.keys() == current.keys(), (original.keys(), current.keys())
        for key in original:
            assert_equal(original[key], current[key])
    elif isinstance(original, (list, tuple)):
        assert len(original) == len(current)
        for left, right in zip(original, current):
            assert_equal(left, right)
    elif isinstance(original, (np.ndarray, torch.Tensor, Image.Image)):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(current))
    else:
        assert original == current, (original, current)


def check_image_outputs(source, checks):
    """比较实际图像像素、裁剪结果及增强消费的 RNG。processor 用记录图像的 spy。"""
    import numpy as np
    import torch
    from PIL import Image
    from psi.config.augmentation import CenterCrop, ResizeImage
    from test_psi0_preprocessing import ImageTransform, PreprocessingTests

    class SourceImageTransform(source.Psi0ModelTransform):
        build_qwenvl_inputs = ImageTransform.build_qwenvl_inputs

    variants = ("STATE", "NO-STATE", "VGQ-AE", "VGQ-VLM", "EGQ-AE", "STATE-RECONSTRUCTION")
    for variant in variants:
        for no_aug in (False, True):
            data = PreprocessingTests().sample(variant)
            rgb = np.random.default_rng(5).integers(0, 256, (12, 20, 3), dtype=np.uint8)
            data["observations"] = [Image.fromarray(rgb)]
            if variant not in ("STATE", "NO-STATE"):
                key = "history_rgb_images" if variant.startswith("EGQ") else "history_vlm_images"
                data[key] = np.stack([
                    np.roll(rgb, i, axis=1).transpose(2, 0, 1) for i in range(4)
                ])
                data["observations"] = [Image.fromarray(np.roll(rgb, 3, axis=1))]
            settings = dict(resize=ResizeImage(size=(10, 16)),
                            center_crop=CenterCrop(size=(8, 10)), img_aug=True)
            torch.manual_seed(104)
            original = SourceImageTransform(**settings)(copy.deepcopy(data), no_aug=no_aug)
            rng = torch.get_rng_state()
            torch.manual_seed(104)
            current = ImageTransform(**settings)(copy.deepcopy(data), no_aug=no_aug)
            assert_equal(original, current)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            checks.append(f"image pixels+geometry+RNG: {variant}, no_aug={no_aug}")


def check_repacking(source, target, checks):
    import numpy as np
    import torch

    for variant in ("STATE", "NO-STATE", "VGQ-AE", "VGQ-VLM", "EGQ-AE"):
        settings = dict(rsci_variant=variant, pad_action_dim=36, pad_state_dim=36)
        original = source.SimpleRepackTransform(**settings)
        current = target.SimpleRepackTransform(**settings)
        assert_equal(original.delta_timestamps(50), current.delta_timestamps(50))
        count = 1 if variant in ("STATE", "NO-STATE") else 4
        data = {
            "observation.images.egocentric": torch.rand(count, 3, 12, 20),
            "states": torch.randn(1, 32), "action": torch.randn(30, 36),
            "action_is_pad": torch.arange(30) > 26, "task": "MOVE BOX",
            "ardy_target": np.zeros(128, np.float32), "ardy_valid": True,
            "observation.images.egocentric_is_pad": np.zeros(count, bool),
        }
        assert_equal(original(copy.deepcopy(data)), current(copy.deepcopy(data)))
        if count == 4:
            data["observation.images.egocentric_is_pad"][:2] = True
            assert_equal(original(copy.deepcopy(data)), current(copy.deepcopy(data)))
        checks.append(f"repack/history/padding/masks/state: {variant}")


def check_normalization(source, target, raw_root, checks):
    import numpy as np
    from psi.rhsci.supervision import state_labels

    def bounds(dim):
        low, high = np.linspace(-2, -0.1, dim), np.linspace(0.2, 3, dim)
        high[0] = low[0]
        return dict(min=low.tolist(), q01=low.tolist(), max=high.tolist(), q99=high.tolist())

    for kind in ("bounds", "bounds_q99"):
        settings = dict(stat_path="/nonexistent/audit-only-stats.json", normalize_state=True,
                        action_norm_type=kind, pad_action_dim=36, pad_state_dim=36)
        original = source.ActionStateTransform(**settings)
        current = target.ActionStateTransform(**settings)
        stats = dict(action=bounds(36), states=bounds(32))
        original.populate_stats(stats)
        current.populate_stats(stats)
        value = dict(actions=np.random.default_rng(3).normal(size=(30, 36)).astype(np.float32),
                     states=np.random.default_rng(4).normal(size=(1, 36)).astype(np.float32))
        assert_equal(original(copy.deepcopy(value)), current(copy.deepcopy(value)))
        assert_equal(original.reverse_call(value["actions"]), current.reverse_call(value["actions"]))
        checks.append(f"normalization+inverse including constant dimensions: {kind}")
    labels = load_source_module("raw_state_labels_audit", raw_root / "RSCI/psi0_state_reconstruction/data.py")
    value = np.random.default_rng(8).normal(size=(1, 32)).astype(np.float32)
    value[0, 2] = np.nan
    assert_equal(labels.state_labels(value, original)[:2], state_labels(value, current))
    checks.append("four-frame state reconstruction labels31 and validity")


def canonical_ast(path):
    """忽略格式、docstring 与命名空间迁移；保留公式、默认值和调用参数。"""
    text = path.read_text().replace("RSCI", "RHSCI").replace("rsci.", "rhsci.").replace("rsci_", "rhsci_")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if hasattr(node, "body") and isinstance(node.body, list):
            node.body = [item for item in node.body if not (
                isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            )]
    return ast.dump(tree, include_attributes=False)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_backbone_sources(raw_root):
    """源码对照不替代各骨干依赖环境里的数值测试。"""
    pairs = [
        ("src/rsci_pi05/augmentation.py", "src/openpi/rhsci/augmentation.py"),
        ("src/openpi/models_pytorch/preprocessing_pytorch.py", "src/openpi/models_pytorch/preprocessing_pytorch.py"),
        ("src/gr00t/gr00t/model/gr00t_n1d6/image_augmentations.py", "src/gr00t/model/gr00t_n1d6/image_augmentations.py"),
        ("src/gr00t/gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py", "src/gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py"),
    ]
    records = []
    for original, current in pairs:
        left, right = raw_root / original, ROOT / current
        assert canonical_ast(left) == canonical_ast(right), (original, current)
        records.append(dict(original=str(left), current=str(right), ast_equal=True,
                            original_sha256=file_sha(left), current_sha256=file_sha(right)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Original PSI0-RAW checkout")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON audit record")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    from psi.config import transform as target

    raw_root = args.source_root.resolve()
    # 参照 transform 的 augmentation 导入来自当前包，因此先核对完整类定义和默认参数。
    assert canonical_ast(raw_root / "RSCI/psi0_controls/native/src/psi/config/augmentation.py") == canonical_ast(ROOT / "src/psi/config/augmentation.py")
    source_path = raw_root / "RSCI/psi0_controls/native/src/psi/config/transform.py"
    source = load_source_module("raw_transform_audit", source_path)
    checks = []
    check_image_outputs(source, checks)
    check_repacking(source, target, checks)
    check_normalization(source, target, raw_root, checks)
    files = [source_path, raw_root / "RSCI/psi0_controls/native/src/psi/config/augmentation.py",
             raw_root / "RSCI/psi0_state_reconstruction/data.py", ROOT / "src/psi/config/transform.py",
             ROOT / "src/psi/config/augmentation.py", ROOT / "src/psi/rhsci/supervision.py"]
    report = dict(
        date_utc=datetime.now(timezone.utc).isoformat(),
        scope="Own PSI0-RAW RGB training preprocessing; no GPU or physics evaluation",
        source_head=subprocess.check_output(["git", "-C", str(raw_root), "rev-parse", "HEAD"], text=True).strip(),
        checks_passed=checks, files={str(path): file_sha(path) for path in files},
        augmentation_class_ast_equal=True,
        other_backbone_source_checks=check_backbone_sources(raw_root),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(checks)} paired preprocessing groups; 4 backbone source comparisons")


if __name__ == "__main__":
    main()
