"""固定资产下载、失败清理和历史目录保护的 CPU 回归。"""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.sync_simple_eval import MANIFEST, extract_task, fetch_file, matches_record, sync_release


class SimpleEvalAssetsTest(unittest.TestCase):
    def test_source_archives_match_published_release(self):
        manifest = json.loads(MANIFEST.read_text())
        published = manifest["releases"]["zip-38e43f22"]["files"]
        self.assertEqual(len(manifest["source"]["files"]), 17)
        for name, source in manifest["source"]["files"].items():
            self.assertEqual(source["sha256"], published[name]["sha256"])
            self.assertEqual(source["bytes"], published[name]["bytes"])

    def test_git_blob_digest_is_not_plain_file_sha1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_bytes(b"abc")
            record = {"bytes": 3, "git_blob_sha1": hashlib.sha1(b"blob 3\0abc").hexdigest()}
            self.assertTrue(matches_record(path, record))
            record["git_blob_sha1"] = hashlib.sha1(b"abc").hexdigest()
            self.assertFalse(matches_record(path, record))

    def test_corrupt_download_does_not_publish_or_leave_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"corrupt")
            record = {"bytes": 7, "sha256": hashlib.sha256(b"correct").hexdigest()}
            with patch("tools.sync_simple_eval.time.sleep"):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    fetch_file(source.as_uri(), root / "download", record)
            self.assertEqual(list(root.iterdir()), [source])

    def test_archive_rejects_traversal_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "Task.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("Task/../../escape", b"bad")
            with self.assertRaisesRegex(ValueError, "unexpected ZIP member"):
                extract_task(archive, root)
            self.assertEqual(list(root.iterdir()), [archive])

    def test_archive_preserves_changed_existing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "Task.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("Task/meta/episodes.jsonl", b"official")
            extract_task(archive, root)
            extract_task(archive, root, verify_only=True)
            metadata = root / "Task/meta/episodes.jsonl"
            metadata.write_bytes(b"historical")
            with self.assertRaisesRegex(ValueError, "differs"):
                extract_task(archive, root)
            self.assertEqual(metadata.read_bytes(), b"historical")

    def test_snapshot_reuses_verified_file_and_checks_task_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "zip-38e43f22/Task/meta/info.json"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"abc")
            record = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            release = {
                "repo": "fixture",
                "revision": "fixed",
                "prefix": "",
                "archives": False,
                "files": {"Task/meta/info.json": record, "Other/meta/info.json": record},
            }
            manifest = {"releases": {"snapshot": release}}
            with patch("tools.sync_simple_eval.fetch_file", side_effect=AssertionError("network")):
                sync_release(manifest, "snapshot", root, tasks=["Task"])
                result = sync_release(manifest, "snapshot", root, ["Task"], verify_only=True)
            self.assertEqual(result["files_verified"], 1)
            self.assertFalse((root / "snapshot/Other").exists())
            with self.assertRaisesRegex(ValueError, "unknown tasks"):
                sync_release(manifest, "snapshot", root, ["Typo"])


if __name__ == "__main__":
    unittest.main()
