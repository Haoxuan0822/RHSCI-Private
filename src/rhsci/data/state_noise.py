"""Shared measured-joint dimensions and seed validation for client perturbations."""

MEASURED_STATE_DIM = 31


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("State noise seed must be an integer in [0, 2**63)")
    return seed
