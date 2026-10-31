"""复用官方 HTTP/SIMPLE/RTC 服务，绑定 RHSCI 架构及 checkpoint 内统计。"""

import argparse
import json
import sys
from pathlib import Path

from .contracts import sha256


def main():

    if "--help" in sys.argv or "-h" in sys.argv:
        from lambda0.inference import server

        server.main(sys.argv[1:])
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--processor", type=Path)
    known, _ = parser.parse_known_args()
    if known.processor is not None and known.processor.resolve() != known.model.resolve():
        raise ValueError("Serving processor must be the checkpoint-bound processor")
    root = known.model.parent if known.model.name == "ema" else known.model
    from rsci.inference.weights import validate_inference_weights

    validate_inference_weights(known.model, config="config.json")
    expected = root / "data_bundle/stats" / known.normalization.name
    if not expected.is_file() or sha256(expected) != sha256(known.normalization):
        raise ValueError("Serving statistics differ from checkpoint-bound statistics")
    from lambda0.inference import server

    from .model import Lambda0RSCI

    statistics = json.loads(known.normalization.read_text())
    from .contracts import SAFE_NORMALIZATION

    normalization_contract = statistics["normalization_contract"]
    if normalization_contract not in (SAFE_NORMALIZATION, server.QUANTILE_NORMALIZATION_CONTRACT):
        raise ValueError("Unsupported checkpoint normalization contract")
    server.main(
        sys.argv[1:], model_class=Lambda0RSCI, normalization_contract=normalization_contract
    )


if __name__ == "__main__":
    main()
