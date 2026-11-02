"""官方 λ₀ SIMPLE 后训练入口；额外提供只读检查及 VGQ-AE。"""

import json
from pathlib import Path

from .config import build_argument_parser as build_argument_parser
from .config import build_training_config as build_training_config
from .contracts import CODE as CODE
from .preflight import validate_output_directory as validate_output_directory
from .preflight import validate_training_data as validate_training_data
from .worker import execute_training as execute_training


def main():
    args = build_argument_parser().parse_args()
    config = build_training_config(args)
    if args.check:
        print(
            json.dumps(
                dict(
                    experiment=config,
                    checkpoint_exists=Path(config["model"]).is_dir(),
                    bundle_exists=Path(config["data"]).is_dir(),
                ),
                indent=2,
            )
        )
    elif args.check_data:
        validate_training_data(args, config)
    elif args.dry_run or args.worker or args.check_model:
        execute_training(args, config)
    else:
        raise SystemExit("Use scripts/train/lambda0.sh for the guarded training launcher")


if __name__ == "__main__":
    main()
