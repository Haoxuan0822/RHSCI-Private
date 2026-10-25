"""直接使用 SIMPLE EgoVLA 的提示、占位 token 和 RGB Processor。"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class Processor:
    def __init__(self, tokenizer, image_processor):
        from human_plan.utils.action_tokenizer import build_action_tokenizer
        from human_plan.vila_train.args import VLADataArguments, VLAModelArguments
        from llava import conversation as conversation_lib

        self.tokenizer = tokenizer
        tokenizer.padding_side = "right"
        # 与官方训练一样；缺失 token 由初始化步骤增补，不在数据 worker 中扩词表。
        if tokenizer.pad_token_id is None:
            raise ValueError("Initialize official pad token before constructing Processor")
        # SIMPLE 训练/服务都保留此 fallback；发布的 conversation.py 没有 qwen2 注册项。
        conversation_lib.default_conversation = conversation_lib.conv_templates.get(
            "qwen2", conversation_lib.conv_templates["vicuna_v1"]
        )
        model_args = VLAModelArguments()
        model_args.predict_future_step = 30
        model_args.sep_query_token = False
        self.data_args = VLADataArguments()
        self.data_args.image_processor = image_processor
        self.data_args.is_multimodal = True
        self.data_args.mm_use_im_start_end = False
        self.data_args.image_aspect_ratio = "resize"
        self.data_args.with_aug = False  # 官方 SIMPLE 脚本未开启，不能擅加 π0.5 增强。
        self.data_args.mask_input = True
        self.data_args.raw_action_label = True
        self.data_args.traj_action_output_dim = 36
        self.action_tokenizer = build_action_tokenizer("uniform", tokenizer, model_args)

    def configure(self, config):
        for key in (
            "invalid_token_idx",
            "input_placeholder_token_idx",
            "input_placeholder_start_token_idx",
            "input_placeholder_end_token_idx",
        ):
            setattr(config, key, getattr(self.action_tokenizer, key))

    def prompt(self, text, actions=None):
        from human_plan.preprocessing.preprocessing import preprocess_multimodal_vla, preprocess_vla
        from human_plan.preprocessing.prompting_format import preprocess_language_instruction

        instruction = preprocess_multimodal_vla(
            preprocess_language_instruction(text, 0, self.data_args), self.data_args
        )
        actions = (
            torch.zeros(30, 36)
            if actions is None
            else torch.as_tensor(np.array(actions, copy=True))
        )
        output = preprocess_vla(
            instruction,
            torch.empty(0),
            actions,
            torch.ones_like(actions, dtype=torch.bool),
            self.action_tokenizer,
            self.tokenizer,
            mask_input=True,
            raw_action_label=True,
            traj_action_output_dim=36,
            sep_query_token=False,
        )
        if len(output["input_ids"]) > 4096:
            raise ValueError("Prompt exceeds official 4096-token limit")
        return output

    def images(self, images):
        from llava.mm_utils import process_image_ndarray_v2

        # 输入是 RGB，不能套官方函数默认 BGR 翻转；SIMPLE 原服务同样显式 False。
        return torch.stack(
            [
                process_image_ndarray_v2(image, self.data_args, reverse_channel_order=False)
                for image in images
            ]
        )

    def __call__(self, sample):
        prompt = self.prompt(sample["text"], sample["actions"])
        if "states" in sample:
            return dict(
                input_ids=prompt["input_ids"],
                labels=prompt["labels"],
                images=self.images(sample["images"]),
                actions=prompt["raw_action_label"].float(),
                action_mask=prompt["raw_action_mask"].bool(),
                states=torch.as_tensor(sample["states"], dtype=torch.float32),
            )
        return dict(
            input_ids=prompt["input_ids"],
            labels=prompt["labels"],
            images=self.images(sample["images"]),
            actions=prompt["raw_action_label"].float(),
            action_mask=prompt["raw_action_mask"].bool(),
            history_valid=torch.as_tensor(sample["history_valid"], dtype=torch.bool),
            ardy_target=torch.as_tensor(sample["ardy_target"], dtype=torch.float32),
            ardy_valid=torch.tensor(sample["ardy_valid"], dtype=torch.bool),
        )

    def inference(self, image, text):
        prompt = self.prompt(text)
        return dict(
            images=self.images([image])[None],
            input_ids=prompt["input_ids"][None],
            labels=prompt["labels"][None],
            attention_mask=torch.ones_like(prompt["input_ids"][None], dtype=torch.bool),
        )


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        batch = {
            key: torch.stack([sample[key] for sample in samples])
            for key in samples[0]
            if key not in ("input_ids", "labels")
        }
        batch["input_ids"] = pad_sequence(
            [sample["input_ids"] for sample in samples],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        batch["labels"] = pad_sequence(
            [sample["labels"] for sample in samples], batch_first=True, padding_value=-100
        )
        lengths = torch.tensor([len(sample["input_ids"]) for sample in samples])
        batch["attention_mask"] = torch.arange(batch["input_ids"].shape[1])[None] < lengths[:, None]
        return batch
