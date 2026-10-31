"""仅限制新的训练组合，不阻止历史配置/权重加载和评测。"""


def validate_rsci_training(variant, vlm_mode):
    """非星号VLM Query必须有可训练的共享VLM，不能静默改星号或解冻。"""
    if variant in ("VGQ-VLM", "EGQ-VLM") and vlm_mode == "frozen":
        raise ValueError(
            f"{variant} / frozen has no ARDY supervision path to the action policy: "
            "the VLM is frozen and the Query latent is excluded from AE. "
            "Choose --vlm lora explicitly for this non-starred model; "
            "a starred model changes the architecture and is a separate experiment. "
            "Historical checkpoints remain loadable for evaluation."
        )
