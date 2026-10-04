"""Differential recovery helpers for the CV22 sigmoid accelerator operator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class SigmoidRecoveryError(ValueError):
    """The oracle/logit pair cannot establish one exact sigmoid mapping."""


def _validate_oracle_score(value: np.float32) -> int:
    """Return exact IEEE bits for one finite ADES sigmoid observation.

    Early constant-frame captures happened to fall on values exactly
    representable by Ambarella's unsigned 5.11 storage format. Natural-frame
    captures prove that the detector's exposed float32 results also contain
    additional low mantissa bits, so unsigned-5.11 representability is not a
    valid acceptance gate for the recovered output boundary.
    """

    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
    if (
        not np.isfinite(value)
        or value < np.float32(0.0)
        or value > np.float32(1.0)
    ):
        raise SigmoidRecoveryError(
            f"oracle score 0x{bits:08x} is outside the finite sigmoid range"
        )
    if bits == 0x80000000:
        raise SigmoidRecoveryError("oracle score uses negative zero")
    return bits


def merge_sigmoid_float32_mappings(
    *mappings: Mapping[int, int],
) -> dict[int, int]:
    """Merge independent ADES observations without permitting ambiguity."""

    combined: dict[int, int] = {}
    for mapping in mappings:
        for effective, bits in mapping.items():
            if not isinstance(effective, int) or not isinstance(bits, int):
                raise SigmoidRecoveryError(
                    "sigmoid mappings must contain integer bits"
                )
            if not -256 <= effective <= 255 or not 0 <= bits <= 0xFFFFFFFF:
                raise SigmoidRecoveryError(
                    "sigmoid mapping value is outside its domain"
                )
            previous = combined.setdefault(effective, bits)
            if previous != bits:
                raise SigmoidRecoveryError(
                    f"effective value {effective} produced conflicting outputs "
                    f"0x{previous:08x} and 0x{bits:08x}"
                )
    return dict(sorted(combined.items()))


def recover_sigmoid_float32_mapping(
    logits: object,
    scores: object,
    *,
    input_exponent_offset: int,
) -> dict[int, int]:
    """Pair raw int8 logits with oracle scores and return IEEE float32 bits."""

    raw_logits = np.asarray(logits)
    raw_scores = np.asarray(scores)
    if raw_logits.dtype != np.int8 or raw_scores.dtype != np.float32:
        raise SigmoidRecoveryError("logits must be int8 and scores must be float32")
    if raw_logits.shape != raw_scores.shape:
        raise SigmoidRecoveryError("logits and scores must have identical shapes")
    if input_exponent_offset not in {2, 3}:
        raise SigmoidRecoveryError("unsupported sigmoid exponent offset")
    effective = raw_logits.astype(np.int16) * (
        2 ** (3 - input_exponent_offset)
    )
    mapping: dict[int, int] = {}
    for value in np.unique(effective):
        observed = np.unique(raw_scores[effective == value])
        if len(observed) != 1:
            raise SigmoidRecoveryError(
                f"effective value {int(value)} produced multiple outputs"
            )
        bits = int(observed.view(np.uint32)[0])
        _validate_oracle_score(observed[0])
        mapping[int(value)] = bits
    return mapping
