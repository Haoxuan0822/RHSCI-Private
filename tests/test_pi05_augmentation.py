"""真实官方增强数值、时序共享、生产前向与推理隔离。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_pi05_rsci import fixture

from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from rsci.backbones.pi05.augmentation import augment_training_images


class AugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matches_official_current_camera_and_shared_history(self):
        image = torch.linspace(-1, 1, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
        observation = SimpleNamespace(
            images={"base_0_rgb": image.clone()},
            image_masks={},
            state=torch.zeros(2, 1),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
            token_ar_mask=None,
            token_loss_mask=None,
        )
        torch.manual_seed(721)
        expected = preprocess_observation_pytorch(
            observation, train=True, image_keys=("base_0_rgb",)
        ).images["base_0_rgb"]
        frames = image[:, None].expand(-1, 4, -1, -1, -1).clone()
        original = frames.clone()
        torch.manual_seed(721)
        actual = augment_training_images(frames)
        for index in range(4):
            torch.testing.assert_close(actual[:, index], expected, rtol=0, atol=0)
        torch.testing.assert_close(frames, original, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual[:, -1], image))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLessEqual(actual.abs().max().item(), 1.0)

    def test_train_uses_one_augmented_current_for_action_and_history(self):
        model, batch = fixture()
        batch["images"] = batch["images"].tanh()
        torch.manual_seed(72)
        expected = augment_training_images(batch["images"])
        seen = []
        encode = model.encode

        def capture(image, *args):
            seen.append(image.detach().clone())
            return encode(image, *args)

        model.encode = capture
        torch.manual_seed(72)
        result = model(**batch)
        for actual, index in zip(seen, (3, 0, 1, 2)):
            torch.testing.assert_close(actual, expected[:, index], rtol=0, atol=0)
        self.assertEqual(len(seen), 4)
        result["loss"].backward()
        self.assertTrue(
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        )

    def test_eval_forward_and_sampling_never_augment(self):
        model, batch = fixture()
        model.eval()
        with patch(
            "rsci.backbones.pi05.model.augment_training_images",
            side_effect=AssertionError("inference augmentation"),
        ):
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(**batch)["loss"]))
                result = model.sample_actions(
                    batch["images"][:, -1:],
                    batch["tokens"],
                    batch["token_mask"],
                    num_steps=2,
                    noise=torch.zeros_like(batch["actions"]),
                )
                self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
