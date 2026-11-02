"""Guardian completion uses the requested budget, including nondefault runs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lambda0.rhsci.launch import run_training_guardian


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
                with patch("lambda0.rhsci.launch.run_worker_process", return_value=0):
                    if recorded == budget:
                        run_training_guardian(
                            SimpleNamespace(gpus="0", max_steps=budget), [], output
                        )
                    else:
                        with self.assertRaisesRegex(RuntimeError, f"{budget}-step completion"):
                            run_training_guardian(
                                SimpleNamespace(gpus="0", max_steps=budget), [], output
                            )
