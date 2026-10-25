import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsci.backbones.lambda0 import contracts


class SourceManifestTests(unittest.TestCase):
    def test_single_tree_source_and_changes_are_verified_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("value = 1\n")
            manifest = {
                "upstream_commit": contracts.UPSTREAM_COMMIT,
                "files": {"model.py": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
            (root / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
            with patch.object(contracts, "UPSTREAM", root):
                self.assertEqual(contracts.verify_upstream(), contracts.UPSTREAM_COMMIT)
                source.write_text("value = 2\n")
                with self.assertRaises(ValueError):
                    contracts.verify_upstream()
                source.write_text("value = 1\n")
                extra = root / "extra.py"
                extra.write_text("value = 1\n")
                with self.assertRaises(ValueError):
                    contracts.verify_upstream()
                extra.unlink()
                source.unlink()
                with self.assertRaises(ValueError):
                    contracts.verify_upstream()
