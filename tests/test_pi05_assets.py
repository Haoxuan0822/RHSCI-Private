"""Local asset registration rejects invalid inputs and changed inventories."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import sentencepiece
import torch
from safetensors.torch import save_file

from rsci.backbones.pi05.prepare_assets import register_assets, transformers_overlay


class AssetRegistration(unittest.TestCase):
    def test_registration_is_portable_idempotent_and_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            weights = checkpoint / "model.safetensors"
            save_file({"action_in_proj.weight": torch.zeros(2, 3)}, weights)
            tokenizer = root / "tokenizer.model"
            output = io.BytesIO()
            sentencepiece.SentencePieceTrainer.train(
                sentence_iterator=iter(["pick the box", "move the robot"] * 20),
                model_writer=output,
                vocab_size=32,
                hard_vocab_limit=False,
                minloglevel=2,
            )
            tokenizer.write_bytes(output.getvalue())
            first = register_assets(checkpoint, tokenizer, "test/repo", "fixed-revision")
            self.assertEqual(set(first["files"]), {"config.json", "model.safetensors"})
            self.assertNotIn(temp, json.dumps(first))
            self.assertEqual(
                register_assets(checkpoint, tokenizer, "test/repo", "fixed-revision"), first
            )
            save_file({"action_in_proj.weight": torch.ones(2, 3)}, weights)
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                register_assets(checkpoint, tokenizer, "test/repo", "fixed-revision")
            self.assertEqual(json.loads((checkpoint / "RSCI_ASSETS.json").read_text()), first)

    def test_invalid_checkpoint_does_not_publish_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text("{}")
            save_file({"unrelated": torch.ones(1)}, root / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "unsharded native"):
                register_assets(root, root / "missing-tokenizer", "local", "unknown")
            self.assertFalse((root / "RSCI_ASSETS.json").exists())

    def test_installed_overlay_matches_source(self):
        self.assertTrue(transformers_overlay().is_dir())


if __name__ == "__main__":
    unittest.main()
