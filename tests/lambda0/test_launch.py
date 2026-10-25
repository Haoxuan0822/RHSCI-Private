"""Guardian completion uses the requested budget, including nondefault runs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsci.backbones.lambda0.launch import guardian


class GuardianTests(unittest.TestCase):
    def test_completion_matches_requested_training_budget(self):
        for budget, recorded in (
            (40000, 40000),
            (50000, 50000),
            (50000, 40000),
            (40000, 50000),
            (50000, None),
        ):
            with (
                self.subTest(budget=budget, recorded=recorded),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output = Path(tmp)
                if recorded is not None:
                    (output / "training_complete.json").write_text(
                        json.dumps(dict(global_step=recorded))
                    )
                process = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda: 0)
                with patch("rsci.backbones.lambda0.launch.subprocess.Popen", return_value=process):
                    if recorded == budget:
                        guardian(SimpleNamespace(gpus="0", max_steps=budget), [], output)
                    else:
                        with self.assertRaisesRegex(RuntimeError, f"{budget}-step completion"):
                            guardian(SimpleNamespace(gpus="0", max_steps=budget), [], output)
