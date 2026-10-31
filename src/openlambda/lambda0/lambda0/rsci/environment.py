"""验证λ₀运行依赖与版本锁一致。"""

import json
from importlib.metadata import version

from rsci.paths import CODE_ROOT


def verify_environment():
    lock = json.loads((CODE_ROOT / "baselines/lambda0/environment.lock.json").read_text())
    mismatches = {
        name: dict(expected=expected, actual=version(name))
        for name, expected in lock["versions"].items()
        if version(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"λ₀ runtime differs from environment.lock.json: {mismatches}")
    return lock
