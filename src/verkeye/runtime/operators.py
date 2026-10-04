"""Small exact operators used to verify the reference execution machinery.

These are synthetic contract-test operators.  They are not representations of
the still-opaque CB62 CV22 compiled splits.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


SUPPORTED_EXACT_OPERATORS = frozenset(
    {
        "exact.identity",
        "exact.add_saturating_int32",
    }
)


def execute_exact_operator(
    operator: str,
    inputs: Sequence[NDArray[Any]],
    attributes: Mapping[str, Any],
) -> NDArray[Any]:
    """Execute one explicitly supported operator without implicit coercion."""

    if attributes:
        raise ValueError(f"operator {operator} does not accept attributes")
    if operator == "exact.identity":
        if len(inputs) != 1:
            raise ValueError("exact.identity requires exactly one input")
        return np.ascontiguousarray(inputs[0]).copy()
    if operator == "exact.add_saturating_int32":
        if len(inputs) != 2:
            raise ValueError("exact.add_saturating_int32 requires exactly two inputs")
        left, right = (np.asarray(item) for item in inputs)
        if left.dtype != np.int32 or right.dtype != np.int32:
            raise ValueError("exact.add_saturating_int32 inputs must be int32")
        if left.shape != right.shape:
            raise ValueError("exact.add_saturating_int32 input shapes must match")
        limits = np.iinfo(np.int32)
        widened = left.astype(np.int64) + right.astype(np.int64)
        clipped = np.clip(widened, limits.min, limits.max)
        return np.ascontiguousarray(clipped.astype(np.int32))
    raise ValueError(f"unsupported exact operator: {operator}")
