"""PSI0 image geometry, shared color augmentation and saved-run configuration."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from psi.config.augmentation import CenterCrop, ColorJitter, ResizeImage
from psi.config.transform import Psi0ModelTransform
from rsci.backbones.psi0.config import load_run_config
from rsci.backbones.psi0.train import parser, resolved


class ImageTransform(Psi0ModelTransform):
    def build_qwenvl_inputs(self, processor, images, instruction, **kwargs):
        return {"pixels": [np.array(image) for image in images]}


class PreprocessingTests(unittest.TestCase):
    def sample(self, variant):
        rgb = np.full((12, 20, 3), [40, 80, 120], np.uint8)
        data = dict(
            observations=[Image.fromarray(rgb)],
            instruction="move",
            dataset="fixture",
            raw_actions=np.zeros((30, 36)),
            actions=np.zeros((30, 36)),
            states=np.zeros((1, 36)),
        )
        if variant.startswith(("VGQ-", "EGQ-")):
            key = "history_vlm_images" if variant.startswith("VGQ-") else "history_rgb_images"
            data[key] = np.repeat(rgb.transpose(2, 0, 1)[None], 4, axis=0)
            data.update(history_valid=np.ones(4, bool), ardy_target=np.zeros(128), ardy_valid=True)
        return data

    def test_all_training_image_paths_with_and_without_augmentation(self):
        for variant in ("STATE", "NO-STATE", "VGQ-AE", "EGQ-AE"):
            for enabled, no_aug in ((False, False), (True, False), (True, True)):
                with self.subTest(variant=variant, enabled=enabled, no_aug=no_aug):
                    transform = ImageTransform(
                        resize=ResizeImage(size=(10, 16)),
                        center_crop=CenterCrop(size=(8, 10)),
                        color_jitter=ColorJitter(
                            brightness=(0.5, 0.5),
                            contrast=(1.0, 1.0),
                            saturation=(1.0, 1.0),
                            hue=(0.0, 0.0),
                        ),
                        img_aug=enabled,
                    )
                    data = self.sample(variant)
                    before = copy.deepcopy(data)
                    output = transform(data, no_aug=no_aug)
                    expected = np.array([20, 40, 60] if enabled and not no_aug else [40, 80, 120])
                    current = output["pixels"][0]
                    self.assertEqual(current.shape, (8, 10, 3))
                    np.testing.assert_array_equal(current, np.broadcast_to(expected, current.shape))
                    np.testing.assert_array_equal(output["actions"], before["actions"])
                    if variant.startswith("VGQ-"):
                        self.assertEqual(len(output["history_vlm_inputs"]), 3)
                        for frame in output["history_vlm_inputs"]:
                            np.testing.assert_array_equal(frame["pixels"][0], current)
                    if variant.startswith("EGQ-"):
                        history = output["history_rgb_images"]
                        self.assertEqual(history.shape, (4, 3, 12, 20))
                        np.testing.assert_array_equal(
                            history[:, :, 0, 0], np.tile(expected, (4, 1))
                        )
                        np.testing.assert_array_equal(
                            data["history_rgb_images"], before["history_rgb_images"]
                        )

    def test_random_color_parameters_are_shared_across_history(self):
        transform = ImageTransform(
            resize=ResizeImage(size=(10, 16)), center_crop=CenterCrop(size=(8, 10)), img_aug=True
        )
        for variant in ("VGQ-AE", "EGQ-AE"):
            output = transform(self.sample(variant))
            pixel = output["pixels"][0][0, 0]
            if variant.startswith("VGQ-"):
                for frame in output["history_vlm_inputs"]:
                    np.testing.assert_array_equal(frame["pixels"][0][0, 0], pixel)
            else:
                np.testing.assert_array_equal(
                    output["history_rgb_images"][:, :, 0, 0], np.tile(pixel, (4, 1))
                )

    def test_saved_json_is_loaded_without_replaying_obsolete_cli(self):
        cfg, _, _, _ = resolved(parser().parse_args(["--task", "fixture", "--model", "EGQ-AE"]))
        saved = cfg.model_dump(mode="json")
        saved["model"]["depth_rgb_encoder_path"] = saved["model"].pop("rsci_rgb_encoder_path")
        saved["model"]["depth_encoder_input_size"] = saved["model"].pop("rsci_encoder_input_size")
        saved["model"].update(psi0_ardy_aux=False, psi0_ardy_aux_weight=0.1, use_depth=False)
        saved["data"]["v5_ardy_sidecar_root"] = saved["data"].pop("rsci_ardy_sidecar_root")
        argv = list(sys.argv)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "argv.txt").write_text(
                "train.py\nfinetune_simple_psi0_config\n"
                "--model.psi0-ardy-aux-weight=0.1\n--data.v5-ardy-sidecar-root=/labels\n"
            )
            (run / "run_config.json").write_text(json.dumps(saved))
            restored = load_run_config(run)
            self.assertEqual(restored.model.rsci_rgb_encoder_path, cfg.model.rsci_rgb_encoder_path)
            self.assertEqual(restored.model.rsci_variant, "EGQ-AE")
            self.assertEqual(restored.data.rsci_ardy_sidecar_root, cfg.data.rsci_ardy_sidecar_root)
            self.assertEqual(sys.argv, argv)
            import torch
            from safetensors.torch import save_file

            weights = run / "model.safetensors"
            save_file({"action_header.rsci.schema_version": torch.tensor(2)}, weights)
            with self.assertRaisesRegex(ValueError, "Reader setting disagrees"):
                load_run_config(run, weights)
            saved["model"].pop("rsci_use_reader")
            (run / "run_config.json").write_text(json.dumps(saved))
            self.assertFalse(load_run_config(run, weights).model.rsci_use_reader)
            saved["model"]["use_depth"] = True
            (run / "run_config.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "unsupported historical architecture"):
                load_run_config(run)
