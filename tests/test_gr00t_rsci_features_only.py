"""真实小尺寸Qwen3验证：跳过lm_head不会改变Eagle隐藏表征或其梯度。"""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, Qwen3Config, Qwen3ForCausalLM


class FeaturesOnlyTests(unittest.TestCase):
    def test_eagle_features_match_native_without_calling_lm_head(self):
        torch.set_num_threads(1)
        torch.manual_seed(120)
        root = (
            Path(__file__).resolve().parents[1]
            / "src/gr00t/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2"
        )
        config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
        text = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            use_cache=False,
            attention_dropout=0.0,
        )
        language = Qwen3ForCausalLM(text)
        config.text_config = text
        config._attn_implementation = "sdpa"
        config.vision_config.hidden_size = 8
        config.image_token_index = 7
        eagle = AutoModel.from_config(
            config, trust_remote_code=True, vision_model=nn.Identity(), language_model=language
        ).eval()
        pixels = [torch.zeros(1)]
        vision = torch.randn(2, 16, requires_grad=True)
        inputs = dict(
            pixel_values=pixels,
            input_ids=torch.tensor([[7, 7, 1, 2]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            output_hidden_states=True,
            use_cache=False,
        )
        with patch.object(eagle, "extract_feature", return_value=vision):
            native = eagle(**inputs)
            expected = torch.autograd.grad(
                native.hidden_states[-1].square().sum(),
                (vision, language.model.layers[0].self_attn.q_proj.weight),
                retain_graph=False,
            )
            with patch.object(
                language.lm_head, "forward", side_effect=AssertionError("Unused logits computed")
            ):
                actual = eagle(**inputs, features_only=True)
                self.assertIsNone(actual.logits)
                torch.testing.assert_close(
                    actual.hidden_states[-1], native.hidden_states[-1], rtol=0, atol=0
                )
                grads = torch.autograd.grad(
                    actual.hidden_states[-1].square().sum(),
                    (vision, language.model.layers[0].self_attn.q_proj.weight),
                )
                for first, second in zip(expected, grads):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
