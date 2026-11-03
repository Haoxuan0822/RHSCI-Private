"""只读加载 PSI0-RAW 参照，比较微型模型的权重、损失、梯度、Adam 更新及推理。"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("src", "src/egovla/VILA", "tests", "scripts")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compare(original, current):
    import torch

    if isinstance(original, dict):
        assert original.keys() == current.keys(), (original.keys(), current.keys())
        for key in original:
            compare(original[key], current[key])
    elif isinstance(original, (list, tuple)):
        assert len(original) == len(current)
        for left, right in zip(original, current):
            compare(left, right)
    elif isinstance(original, torch.Tensor):
        torch.testing.assert_close(original, current, rtol=0, atol=0, equal_nan=False)
    else:
        assert original == current, (original, current)


def paired_updates(original, current, batch, loss, inference):
    """固定相同输入/RNG；使用相同 AdamW 验证模型更新，不代替生产优化器/多卡长跑。"""
    import torch

    old_state, new_state = original.state_dict(), current.state_dict()
    assert old_state.keys() == new_state.keys()
    assert all(old_state[k].shape == new_state[k].shape for k in old_state)
    initial_differences = [k for k in old_state if not torch.equal(old_state[k], new_state[k])]
    old_trainable = {n for n, p in original.named_parameters() if p.requires_grad}
    new_trainable = {n for n, p in current.named_parameters() if p.requires_grad}
    assert old_trainable == new_trainable, (old_trainable - new_trainable, new_trainable - old_trainable)
    current.load_state_dict(old_state, strict=True)
    optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
                  for m in (original, current)]
    for step in range(3):
        outputs, gradients, rngs = [], [], []
        for index, model in enumerate((original, current)):
            model.train()
            optimizers[index].zero_grad(set_to_none=True)
            torch.manual_seed(900 + step)
            value = loss(model, copy.deepcopy(batch), index)
            value["loss"].backward()
            outputs.append({k: v.detach().clone() for k, v in value.items()})
            gradients.append({n: None if p.grad is None else p.grad.detach().clone()
                              for n, p in model.named_parameters() if p.requires_grad})
            rngs.append(torch.get_rng_state())
            optimizers[index].step()
        compare(outputs[0], outputs[1])
        compare(gradients[0], gradients[1])
        compare(rngs[0], rngs[1])
        compare(original.state_dict(), current.state_dict())
        compare(optimizers[0].state_dict(), optimizers[1].state_dict())
    predictions = []
    for index, model in enumerate((original, current)):
        model.eval()
        torch.manual_seed(192)
        with torch.no_grad():
            predictions.append(inference(model, copy.deepcopy(batch), index))
    compare(*predictions)
    return dict(state_entries=len(old_state), trainable_tensors=len(old_trainable),
                same_seed_initialization_equal=not initial_differences,
                initial_differences=initial_differences, strict_raw_weight_load=True,
                three_steps_loss_gradient_parameters_adam_rng_exact=True,
                fixed_rng_inference_exact=True)


def audit_simple(backbone, raw):
    import torch

    if backbone == "pi05":
        import test_pi05_rsci as fixture
        from openpi.rhsci.model import Pi05RHSCI

        source = raw / "src/rsci_pi05/model.py"
        original_cls = load("openpi.rhsci._raw_model_audit", source).Pi05RSCI
        variants = ("STATE", "NO-STATE", "VGQ-AE")
        def build(cls, variant):
            return cls(variant, fixture.TinyBackbone(), vlm_width=64, ae_width=32, compile_inference=False)
        def sample(variant):
            value = fixture.fixture(variant)[1]
            if variant != "VGQ-AE":
                value["images"] = value["images"][:, -1:]
                for key in ("history_valid", "ardy_target", "ardy_valid"):
                    del value[key]
            return value
        current_cls = Pi05RHSCI
        loss = lambda m, b, i: m(**b)
        inference = lambda m, b, i: m.sample_actions(b["images"][:, -1:], b["tokens"], b["token_mask"], num_steps=3)
    elif backbone == "act":
        fixture = load("act_audit_fixture", ROOT / "tests/act/test_act.py")
        from act.rhsci.model import RHSCIACT

        source = raw / "RSCI/act/rsci_act/model.py"
        original_cls = load("act_raw_audit", source).RSCIACT
        current_cls = RHSCIACT
        variants = ("STATE", "NO-STATE", "EGQ-AE")
        build = lambda cls, v: cls(v, fixture.config(v == "STATE"))
        sample = lambda v: fixture.batch()
        loss = lambda m, b, i: m(**b)
        inference = lambda m, b, i: m.predict_action(b)
    elif backbone == "egovla":
        fixture = load("ego_audit_fixture", ROOT / "tests/egovla/test_egovla.py")
        from human_plan.rhsci.model import EgoVLARHSCI

        source = raw / "RSCI/egovla/rsci_egovla/model.py"
        original_cls = load("ego_raw_audit", source).EgoVLARSCI
        current_cls = EgoVLARHSCI
        variants = ("VGQ-AE",)
        build = lambda cls, v: cls(fixture.Backbone(), vlm_training="full", variant=v)
        sample = lambda v: fixture.batch()
        loss = lambda m, b, i: m(**b)
        inference = lambda m, b, i: m.sample_actions(b["images"][:, -1:], b["input_ids"], b["labels"], b["attention_mask"])
    else:
        import test_gr00t_rsci as fixture
        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

        source = raw / "src/gr00t/gr00t/model/gr00t_n1d6/gr00t_n1d6.py"
        load("gr00t.model.gr00t_n1d6.rsci_query", source.with_name("rsci_query.py"))
        original_module = load("gr00t.model.gr00t_n1d6._raw_audit", source)
        raw_dit = load("raw_gr00t_dit_audit", raw / "src/gr00t/gr00t/model/modules/dit.py")
        original_module.DiT, original_module.AlternateVLDiT = raw_dit.DiT, raw_dit.AlternateVLDiT
        with patch.dict(sys.modules, {"gr00t.model.gr00t_n1d6.gr00t_n1d6": original_module}):
            next_module = load("raw_gr00t_next_audit", raw / "RSCI/gr00t/rsci_next/model.py")
        original_cls, current_cls = original_module.Gr00tN1d6, Gr00tN1d6
        variants = ("STATE", "NO-STATE", "VGQ-AE", "VGQ-AE-full-vlm")
        def build(cls, variant):
            config = fixture.model("VGQ-AE" if variant.endswith("full-vlm") else variant).config
            if variant.endswith("full-vlm"):
                config.rsci_memory_mode = "full-vlm"
                if cls is original_cls:
                    cls = next_module.NextGr00t
            with patch.object(original_module, "get_backbone_cls", return_value=fixture.TinyBackbone), \
                 patch("gr00t.model.gr00t_n1d6.gr00t_n1d6.get_backbone_cls", return_value=fixture.TinyBackbone), \
                 patch("gr00t.model.gr00t_n1d6.processing_gr00t_n1d6.Gr00tN1d6DataCollator"):
                return cls(config)
        sample = lambda v: fixture.batch()
        loss = lambda m, b, i: dict(m(b))
        inference = lambda m, b, i: dict(m.get_action(b))
    records = {}
    for variant in variants:
        torch.manual_seed(251)
        original = build(original_cls, variant)
        torch.manual_seed(251)
        current = build(current_cls, variant)
        records[variant] = paired_updates(original, current, sample(variant), loss, inference)
    sources = [source]
    if backbone == "gr00t":
        sources.extend([source.with_name("rsci_query.py"), raw / "RSCI/gr00t/rsci_next/model.py",
                        raw / "src/gr00t/gr00t/model/modules/dit.py"])
    return records, sources


def audit_psi0(raw):
    import torch
    import psi.models
    import psi.utils
    from test_rsci import inputs, policy
    from test_psi0_controls import four_frame_sample, tiny_vlm
    from psi.rhsci.controls import ControlSpec, forward_loss

    native = raw / "RSCI/psi0_controls/native/src/psi"
    # 仅在本 CPU 审计进程允许参照模型解析它自己的旧扩展；不改安装路径或原文件。
    psi.models.__path__.append(str(native / "models"))
    psi.utils.__path__.append(str(native / "utils"))
    config = load("raw_psi_config_audit", native / "config/model_psi0.py")
    original_module = load("psi.models._raw_psi_audit", native / "models/psi0.py")
    sources = [native / "models/psi0.py", native / "models/rsci.py", native / "config/model_psi0.py"]
    load("vlm_query", raw / "RSCI/psi0_controls/vlm_query.py")
    results = {}
    for variant, mode, folder in (("VGQ-VLM", "lora", "psi0_controls"),
                                  ("STATE-RECONSTRUCTION", "frozen", "psi0_state_reconstruction")):
        method_file = raw / "RSCI" / folder / "methods.py"
        methods = load("raw_psi_methods_" + folder, method_file)
        sources.append(method_file)
        spec = methods.ControlSpec(variant, mode)
        torch.manual_seed(251)
        cfg, current = policy(variant, mode, vlm=tiny_vlm())
        raw_cfg = config.Psi0ModelConfig.model_validate(dict(cfg.model_dump(), rsci_variant=spec.native_variant))
        torch.manual_seed(251)
        vlm = tiny_vlm()
        constructor = original_module.ActionTransformerModel
        def small_header(**kwargs):
            return constructor(attention_head_dim=4, num_attention_heads=2,
                               caption_projection_dim=8, pooled_projection_dim=8, **kwargs)
        with patch.dict(sys.modules, {"psi.models.psi0": original_module}), \
             patch.object(original_module, "ActionTransformerModel", side_effect=small_header):
            original = methods.policy_class(spec)(raw_cfg, vlm)
        value = four_frame_sample()
        value.update(state_target=torch.randn(2, 31), state_valid=torch.ones(2, 31, dtype=torch.bool))
        def loss(model, batch, index):
            with patch.dict(sys.modules, {"psi.models.psi0": original_module}):
                if index == 0:
                    return methods.forward_loss(model, batch, spec, max_delay=2)
                return forward_loss(model, batch, ControlSpec(variant), max_delay=2)
        def inference(model, batch, index):
            # 固定噪声/时刻的速度场；不将此项冒充多步真实闭环动作评测。
            with patch.dict(sys.modules, {"psi.models.psi0": original_module}):
                return model(**inputs(batch, auxiliary=False)).action
        results[variant] = paired_updates(original, current, value, loss, inference)
        results[variant]["inference_scope"] = "one velocity evaluation with fixed noise/time; no simulator"
    return results, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=("psi0", "pi05", "act", "egovla", "gr00t"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      NO_ALBUMENTATIONS_UPDATE="1")
    import torch
    import transformers

    torch.set_num_threads(1)
    raw = args.source_root.resolve()
    results, sources = audit_psi0(raw) if args.backbone == "psi0" else audit_simple(args.backbone, raw)
    target_files = {
        "psi0": ["src/psi/models/psi0.py", "src/psi/rhsci/model.py", "src/psi/rhsci/controls.py", "src/psi/rhsci/vlm_query.py"],
        "pi05": ["src/openpi/rhsci/model.py", "src/openpi/models_pytorch/gemma_pytorch.py"],
        "act": ["src/act/rhsci/model.py", "src/act/models/act.py"],
        "gr00t": ["src/gr00t/model/gr00t_n1d6/gr00t_n1d6.py", "src/gr00t/model/gr00t_n1d6/rhsci_query.py", "src/gr00t/model/modules/dit.py"],
        "egovla": ["src/human_plan/rhsci/model.py", "src/egovla/VILA/llava/model/ego_vla_decoder/transformer.py"],
    }
    native_ast = {}
    if args.backbone in ("pi05", "act", "egovla"):
        from audit_psi0_preprocessing import canonical_ast

        native = target_files[args.backbone][-1]
        native_ast[native] = canonical_ast(raw / native) == canonical_ast(ROOT / native)
        assert all(native_ast.values())
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(), backbone=args.backbone,
                  scope="CPU tiny native layers; shared installed native dependencies; no production GPU/physics evaluation",
                  torch=torch.__version__, transformers=transformers.__version__, results=results,
                  shared_native_ast_equal=native_ast,
                  target_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in target_files[args.backbone]},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {args.backbone}: {len(results)} variants, paired 3-step AdamW and fixed-RNG inference")


if __name__ == "__main__":
    main()
