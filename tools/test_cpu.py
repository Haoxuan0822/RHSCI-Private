"""Run the selected backbone's CPU regression tests in its own dependency environment."""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PATTERNS = {
    "psi0": (
        "test_rsci.py",
        "test_rsci_training_gate.py",
        "test_rsci_runtime_safety.py",
        "test_rsci_history_client.py",
        "test_rsci_serving.py",
        "test_rsci_trainer_safety.py",
        "test_rsci_switches.py",
        "test_psi0_preprocessing.py",
        "test_guardian_processes.py",
        "test_simple_preparation.py",
    ),
    "pi05": (
        "test_pi05_assets.py",
        "test_pi05_augmentation.py",
        "test_pi05_official_recipe.py",
        "test_pi05_rsci.py",
        "test_pi05_speed.py",
        "test_pi05_state_noise.py",
    ),
    "gr00t": (
        "test_gr00t_rsci.py",
        "test_gr00t_rsci_features_only.py",
        "test_gr00t_memory.py",
        "test_gr00t_official_optimization.py",
    ),
    "egovla": ("test_egovla.py",),
    "lambda0": ("test*.py",),
    "act": ("test*.py",),
}


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=PATTERNS, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    for name in (
        "src",
        "src/gr00t",
        "src/egovla",
        "src/egovla/VILA",
        "src/openlambda/lambda0",
        "scripts",
        "tests",
        ".",
    ):
        sys.path.insert(0, str(root / name))
    os.chdir(root)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    with tempfile.TemporaryDirectory(prefix="rsci-cpu-tests-") as cache:
        os.environ.update(HF_HOME=cache, HF_MODULES_CACHE=str(Path(cache) / "modules"))
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        start = (
            root / "tests" / args.backbone
            if args.backbone in ("egovla", "lambda0", "act")
            else root / "tests"
        )
        excluded = 0
        for pattern in PATTERNS[args.backbone]:
            for test in cases(loader.discover(str(start), pattern=pattern)):
                if any(
                    word in test.id()
                    for word in (
                        "two_rank",
                        "two_gloo",
                        "ddp",
                        "bad_rank",
                        "distributed",
                        "collective",
                    )
                ):
                    excluded += 1
                else:
                    suite.addTest(test)
        print(f"CPU suite: {args.backbone}; distributed tests excluded: {excluded}", flush=True)
        # CPU suites do not use DeepSpeed; HF's optional unwrap must not initialize its CUDA engine.
        with patch("accelerate.utils.other.is_deepspeed_available", return_value=False):
            result = unittest.TextTestRunner(verbosity=1).run(suite)
    return not result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(main())
