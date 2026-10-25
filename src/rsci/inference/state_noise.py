"""Request-indexed state noise settings independent of action sampling RNG."""

import hashlib

from rsci.data.state_noise import validate_scale, validate_seed


class RequestStateNoise:
    def __init__(self, scale=0.0, seed=0, variant=None):
        self.scale = validate_scale(scale, variant)
        self.seed = validate_seed(seed)
        self.index = 0

    def request_seed(self, payload):
        meta = payload.get("history", {}).get("psi_eval_deterministic")
        if meta is not None:
            episode, index = meta["episode"], meta["inference_index"]
            if any(type(value) is not int or value < 0 for value in (episode, index)):
                raise ValueError("State noise episode/request indices must be nonnegative integers")
        else:
            episode, index = 0, self.index
            self.index += 1
        identity = f"state-v1:{self.seed}:{episode}:{index}".encode()
        return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % 2**63


def add_arguments(parser):
    parser.add_argument(
        "--state-inference-measurement-noise-std-rad",
        type=float,
        nargs="?",
        const=0.02,
        default=0.0,
        help="Raw measured-joint noise; flag alone selects 0.02 rad; default off",
    )
    parser.add_argument("--state-noise-seed", type=int, default=0)
