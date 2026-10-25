"""Select future ARDY supervision without changing policy inputs or action samples."""

import numpy as np


def validate_offset(offset):
    if type(offset) is not int or offset < 0:
        raise ValueError("ARDY offset must be a nonnegative integer")
    return offset


def ardy_target_at_offset(reader, episode, frame, offset):
    validate_offset(offset)
    arrays = reader._episodes.get(episode)
    if arrays is None:
        raise IndexError(f"ARDY episode {episode} unavailable")
    length = len(arrays["ardy_body_latent"])
    if not 0 <= frame < length:
        raise IndexError(f"ARDY source frame {frame} outside episode {episode}")
    if frame + offset >= length:
        return np.zeros(128, dtype=np.float32), False
    target = reader.get(episode, frame + offset)
    return target.ardy_body_latent, target.ardy_loss_mask


def supervision_counts(reader, offset):
    validate_offset(offset)
    valid = total = tail = 0
    for arrays in reader._episodes.values():
        mask = np.asarray(arrays["ardy_loss_mask"], dtype=bool)
        length = len(mask)
        total += length
        tail += min(offset, length)
        if length > offset + 6:
            valid += int(mask[6 + offset :].sum())
    return dict(
        total_frames=total, valid_ardy_frames=valid, tail_masked_frames=tail, offset_frames=offset
    )
