"""仅限制新的训练组合，不阻止历史配置/权重加载和评测。"""

from rhsci.runtime.contracts import reuse_run_contract


def validate_rhsci_training(variant, vlm_mode):
    """非星号VLM Query必须有可训练的共享VLM，不能静默改星号或解冻。"""
    if variant in ("VGQ-VLM", "EGQ-VLM") and vlm_mode == "frozen":
        raise ValueError(
            f"{variant} / frozen has no ARDY supervision path to the action policy: "
            "the VLM is frozen and the Query latent is excluded from AE. "
            "Choose --vlm lora explicitly for this non-starred model; "
            "a starred model changes the architecture and is a separate experiment. "
            "Historical checkpoints remain loadable for evaluation."
        )


def build_training_contract(cfg, ranks):
    """根据模型、数据与配方生成恢复身份；源码编辑和依赖版本不作为恢复门槛。返回契约字典。"""
    data = cfg.model_dump(mode="json")
    # 新增默认字段不改变旧实验身份；显式新方法/执行仍完整绑定。
    data["model"] = canonical_model_config(data["model"])
    train = dict(data["train"])
    for key in ("resume_from_checkpoint", "skip_resumed_steps"):
        train.pop(key, None)
    return reuse_run_contract(
        dict(
            schema=1,
            model=data["model"],
            data=data["data"],
            train=train,
            seed=data["seed"],
            world_size=ranks,
        ),
        cfg.train.output_dir,
    )


def canonical_model_config(model):
    """兼容旧run缺省字段；不改写旧配置，不吞掉显式非默认选择。"""
    model = dict(model)
    defaults = dict(rsci_execution="reference")
    for key, value in defaults.items():
        if model.get(key) == value:
            model.pop(key, None)
    return model
