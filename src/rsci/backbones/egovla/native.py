"""固定 SIMPLE/Psi0 的 EgoVLA baseline；不从独立 EgoVLA 仓库替换实现。"""

from pathlib import Path


def load_native(root, vlm_training="full", variant="VGQ-AE"):
    import torch
    from human_plan.vila_train.args import VLADataArguments, VLAModelArguments, VLATrainingArguments
    from llava.model import LlavaLlamaConfig, LlavaLlamaModel
    from llava.train.utils import prepare_config_for_training, vision_resolution_elevation

    from .model import EgoVLARSCI
    from .processing import Processor

    root = Path(root).resolve()
    config = LlavaLlamaConfig.from_pretrained(str(root), local_files_only=True)
    config._name_or_path = config.resume_path = str(root)
    # Resolve checkpoint components relative to the selected local asset directory.
    for key in ("llm", "vision_tower", "mm_projector"):
        component = root / key
        if not (component / "config.json").is_file():
            raise FileNotFoundError(f"Missing official pretrained component: {component}")
        setattr(config, key + "_cfg", str(component))
    model_args = VLAModelArguments(
        model_name_or_path=str(root),
        version="qwen2",
        vision_tower=str(root / "vision_tower"),
        mm_projector="mlp_downsample",
        mm_vision_select_layer=-2,
        mm_vision_select_feature="cls_patch",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        traj_decoder_type="transformer_action_vector",
        traj_decoder="",
        traj_action_output_dim=36,
        proprio_size=32,
        use_proprio=variant == "STATE",
        sep_proprio=False,
        sep_query_token=False,
    )
    data_args = VLADataArguments(
        image_aspect_ratio="resize",
        predict_future_step=30,
        future_index=0,
        mask_input=True,
        raw_action_label=True,
    )
    training_args = VLATrainingArguments(
        output_dir=str(root),
        bf16=True,
        model_max_length=4096,
        tune_language_model=vlm_training == "full",
        tune_vision_tower=vlm_training == "full",
        tune_mm_projector=vlm_training == "full",
        next_token_loss_coeff=0.0,
        loss_use_l1=False,
    )
    # SIMPLE 原脚本显式 --traj_decoder ""，因此重建4层36维AE，而不是挪用人手48维头。
    prepare_config_for_training(config, model_args, training_args, data_args)
    # The language and vision loaders both need an explicit attention implementation.
    config.rsci_vision_attn_implementation = "sdpa"
    backbone = LlavaLlamaModel(config=config, attn_implementation="sdpa", model_max_length=4096)
    vision_resolution_elevation(backbone, config)
    if (
        backbone.llm.config._attn_implementation != "sdpa"
        or backbone.get_vision_tower().vision_tower.config._attn_implementation != "sdpa"
    ):
        raise RuntimeError("EgoVLA RHSCI requires SDPA for both language and vision")
    backbone.llm.config.use_cache = False
    tokenizer = backbone.tokenizer

    def add_tokens(tokens):
        # 官方 smart_tokenizer_and_embedding_resize 的均值初始化；避免 import 缺失的 llava.data。
        added = tokenizer.add_special_tokens(tokens)
        backbone.llm.resize_token_embeddings(len(tokenizer))
        if added:
            with torch.no_grad():
                for embedding in (
                    backbone.llm.get_input_embeddings(),
                    backbone.llm.get_output_embeddings(),
                ):
                    embedding.weight[-added:] = embedding.weight[:-added].mean(0, keepdim=True)

    if tokenizer.bos_token is None:
        add_tokens(dict(bos_token="[BOS]"))
    tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        add_tokens(dict(pad_token="[PAD]"))
    backbone.llm.pad_token_id = tokenizer.pad_token_id
    backbone.llm.config.tokenizer_padding_side = tokenizer.padding_side
    backbone.llm.config.tokenizer_model_max_length = 4096
    processor = Processor(tokenizer, backbone.get_vision_tower().image_processor)
    processor.configure(config)
    return EgoVLARSCI(backbone, vlm_training=vlm_training, variant=variant), processor
