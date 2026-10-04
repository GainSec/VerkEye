"""Proved host implementations of recovered Ambarella CV22 operators."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .parameter_extraction import FastconvChannel


class OperatorSemanticsError(ValueError):
    """An operator requests behavior that has not yet been proved."""


_CV22_SILU_INT8_LUT = tuple(
    max(
        -128,
        min(127, round((2 * value) / (1 + math.exp(-value / 8)))),
    )
    for value in range(-128, 128)
)

_CV22_SILU_SCALE_PAIRS = frozenset({(1, 2), (1, 3), (2, 4), (3, 3), (3, 4)})

# ADES differential captures establish this contiguous effective int8/2^3
# sigmoid domain. The original constant-frame observations were exactly
# representable in an unsigned 5.11 encoding. Natural frames expose additional
# float32 mantissa bits at the detector boundary; those directly observed values
# override the provisional 5.11 decode below.
_CV22_SIGMOID_EXP3_MINIMUM = -100
_CV22_SIGMOID_EXP3_UFLOAT16 = (
    0x5FA3, 0x60DC, 0x6209, 0x6360, 0x64E3, 0x669A, 0x6846, 0x6960,
    0x6AA0, 0x6C0A, 0x6DA4, 0x6F75, 0x70C2, 0x71ED, 0x733F, 0x74BF,
    0x7671, 0x782E, 0x7945, 0x7A81, 0x7BE7, 0x7D7D, 0x7F48, 0x80A9,
    0x81D0, 0x831E, 0x8499, 0x8647, 0x8817, 0x892A, 0x8A63, 0x8BC5,
    0x8D56, 0x8F1D, 0x908F, 0x91B3, 0x92FD, 0x9473, 0x961C, 0x97FD,
    0x990F, 0x9A44, 0x9BA2, 0x9D2D, 0x9EED, 0xA075, 0xA196, 0xA2DB,
    0xA44C, 0xA5F0, 0xA7C9, 0xA8F2, 0xAA21, 0xAB79, 0xACFE, 0xAEB9,
    0xB057, 0xB171, 0xB2B0, 0xB41C, 0xB5B5, 0xB784, 0xB8C9, 0xB9F2,
    0xBB40, 0xBCBB, 0xBE68, 0xC025, 0xC135, 0xC268, 0xC3C4, 0xC54A,
    0xC702, 0xC879, 0xC98F, 0xCAC7, 0xCC24, 0xCDAB, 0xCF62, 0xD0A6,
    0xD1B6, 0xD2E4, 0xD434, 0xD5A7, 0xD742, 0xD882, 0xD97A, 0xDA88,
    0xDBAD, 0xDCEA, 0xDE41, 0xDFAE, 0xE09B, 0xE16A, 0xE244, 0xE328,
    0xE415, 0xE508, 0xE603, 0xE6FF, 0xE800, 0xE881, 0xE8FF, 0xE97C,
    0xE9F6, 0xEA6C, 0xEADE, 0xEB4B, 0xEBB3, 0xEC15, 0xEC70, 0xECC6,
    0xED15, 0xED5E, 0xEDA2, 0xEDE0, 0xEE18, 0xEE4B, 0xEE7A, 0xEEA4,
    0xEEC9, 0xEEEB, 0xEF0A, 0xEF25, 0xEF3E, 0xEF54, 0xEF67, 0xEF78,
    0xEF88, 0xEF96, 0xEFA2, 0xEFAD, 0xEFB6, 0xEFBF, 0xEFC6, 0xEFCD,
    0xEFD3, 0xEFD8, 0xEFDD, 0xEFE1, 0xEFE5, 0xEFE8, 0xEFEB,
)
_CV22_SIGMOID_EXP2_TAIL_FLOAT32_BITS = (
    0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x30801000, 0x30801000, 0x3085D000, 0x30ABE000, 0x30DCA000,
    0x310DB000, 0x3135F000, 0x31699000, 0x31960000, 0x31C09000,
    0x31F74000, 0x321EC000, 0x324BE000, 0x3282D000, 0x32A80000,
    0x32D7C000, 0x330A8000, 0x3331E000, 0x33646000, 0x3392A000,
    0x33BC4000, 0x33F1B000, 0x341B3000, 0x34474000, 0x347FD000,
    0x34A44000, 0x34D2F000, 0x35076000, 0x352DE000, 0x355F5000,
    0x358F5000, 0x35B81000, 0x35EC5000, 0x3617B000, 0x3642D000,
    0x367A3000, 0x36A09000, 0x36CE3000, 0x37046000, 0x372A0000,
    0x375A4000, 0x378C2000, 0x37B3F000, 0x37E71000, 0x38145000,
    0x383E7000, 0x38748000, 0x389D0000, 0x38C99000, 0x39017000,
    0x39263000, 0x39556000, 0x3988F000, 0x39AFD000, 0x39E1C000,
    0x3A10F000, 0x3A3A2000, 0x3A6ED000, 0x3A996000, 0x3AC4C000,
    0x3AFC9000, 0x3B221000, 0x3B4FE000, 0x3B857000, 0x3BAB0000,
    0x3BDB5000, 0x3C0C9000, 0x3C340000, 0x3C668000, 0x3C935000,
    0x3CBC4000, 0x3CF02000, 0x3D18F000, 0x3D424000, 0x3D762000,
)
_CV22_SIGMOID_EXP3_TAIL_FLOAT32_BITS = (
    0x3400A000, 0x3408F000, 0x341B3000, 0x342FD000, 0x34474000,
    0x3461D000, 0x347FD000, 0x34910000, 0x34A44000, 0x34BA2000,
    0x34D2F000, 0x34EF0000, 0x35076000, 0x35197000, 0x352DE000,
    0x35450000, 0x355F5000, 0x357D0000, 0x358F5000, 0x35A27000,
    0x35B81000, 0x35D08000, 0x35EC5000, 0x3605E000, 0x3617B000,
    0x362BF000, 0x3642D000, 0x365CC000,
)

# Direct ADES observations from the tracked natural-frame oracle corpus. Keys
# are effective int8/2^3 logits shared by the exponent-two and exponent-three
# paths.
_CV22_SIGMOID_POSITIVE_FLOAT32_BITS = {
    1: 0x3F080800,
    2: 0x3F0FE800,
    4: 0x3F1F5800,
    8: 0x3F3B2800,
    10: 0x3F46FC00,
    12: 0x3F514C00,
    14: 0x3F5A1800,
    15: 0x3F5DF800,
    16: 0x3F617C00,
    18: 0x3F679800,
    19: 0x3F6A3800,
    20: 0x3F6C9400,
    21: 0x3F6EB400,
    22: 0x3F709E00,
    24: 0x3F73DC00,
    26: 0x3F767100,
    27: 0x3F778700,
}


@dataclass(frozen=True, slots=True)
class FastconvGeometry:
    kernel_height: int
    kernel_width: int
    stride_y: int
    stride_x: int
    padding_top: int
    padding_left: int
    padding_bottom: int | None = None
    padding_right: int | None = None

    def __post_init__(self) -> None:
        if self.kernel_height <= 0 or self.kernel_width <= 0:
            raise OperatorSemanticsError("kernel dimensions must be positive")
        if self.stride_y <= 0 or self.stride_x <= 0:
            raise OperatorSemanticsError("strides must be positive")
        if self.padding_top < 0 or self.padding_left < 0:
            raise OperatorSemanticsError("padding must be non-negative")
        if self.padding_bottom is not None and self.padding_bottom < 0:
            raise OperatorSemanticsError("padding must be non-negative")
        if self.padding_right is not None and self.padding_right < 0:
            raise OperatorSemanticsError("padding must be non-negative")

    @property
    def effective_padding_bottom(self) -> int:
        return self.padding_top if self.padding_bottom is None else self.padding_bottom

    @property
    def effective_padding_right(self) -> int:
        return self.padding_left if self.padding_right is None else self.padding_right


def channel_halves(input_tensor: object) -> tuple[object, object]:
    """Split an even-depth CHW tensor into its two ordered channel halves."""

    import numpy as np

    tensor = np.asarray(input_tensor)
    if tensor.ndim != 3 or tensor.shape[0] % 2:
        raise OperatorSemanticsError("channel shuffle input must have even CHW depth")
    midpoint = tensor.shape[0] // 2
    return tensor[:midpoint], tensor[midpoint:]


def max_filter_5x5(input_tensor: object) -> object:
    """Apply the proved split-4 5x5 maximum filter with two-cell zero padding."""

    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    tensor = np.asarray(input_tensor)
    if tensor.ndim != 3 or not np.issubdtype(tensor.dtype, np.unsignedinteger):
        raise OperatorSemanticsError("max filter input must be unsigned CHW")
    padded = np.pad(tensor, ((0, 0), (2, 2), (2, 2)), constant_values=0)
    windows = sliding_window_view(padded, (5, 5), axis=(1, 2))
    return windows.max(axis=(-1, -2))


def concatenate_channels(*input_tensors: object) -> object:
    """Concatenate compatible CHW tensors along the recovered merge depth axis."""

    import numpy as np

    tensors = tuple(np.asarray(tensor) for tensor in input_tensors)
    if len(tensors) < 2 or any(tensor.ndim != 3 for tensor in tensors):
        raise OperatorSemanticsError("merge requires at least two CHW tensors")
    if len({(tensor.shape[1:], tensor.dtype.str) for tensor in tensors}) != 1:
        raise OperatorSemanticsError("merge inputs must share spatial shape and dtype")
    return np.concatenate(tensors, axis=0)


def nearest_resample_2x(input_tensor: object) -> object:
    """Apply the proved split-4 nearest-neighbor two-times spatial resample."""

    import numpy as np

    tensor = np.asarray(input_tensor)
    if tensor.ndim != 3:
        raise OperatorSemanticsError("resample input must be CHW")
    return np.repeat(np.repeat(tensor, 2, axis=1), 2, axis=2)


def apply_bitpacked_mux_mask(input_tensor: object, packed_mask: bytes) -> object:
    """Select input values or zero using the recovered row-aligned 1-bit mask."""

    import numpy as np

    tensor = np.asarray(input_tensor)
    if tensor.ndim != 3:
        raise OperatorSemanticsError("mux input must be CHW")
    height, width = tensor.shape[1:]
    row_bytes = (width + 7) // 8
    if len(packed_mask) != height * row_bytes:
        raise OperatorSemanticsError("mux mask length does not match spatial shape")
    encoded = np.frombuffer(packed_mask, dtype=np.uint8).reshape(height, row_bytes)
    mask = np.unpackbits(encoded, axis=1, bitorder="little")[:, :width].astype(bool)
    return np.where(mask[None], tensor, np.zeros((), dtype=tensor.dtype))


def requantize_signed(input_tensor: object, *, divisor: int) -> object:
    """Round signed integers to int8, with exact half values away from zero."""

    import numpy as np

    tensor = np.asarray(input_tensor)
    if not np.issubdtype(tensor.dtype, np.integer):
        raise OperatorSemanticsError("requantizer input must be integer")
    if divisor <= 0 or divisor & (divisor - 1):
        raise OperatorSemanticsError("requantizer divisor must be a power of two")
    wide = tensor.astype(np.int64, copy=False)
    magnitude = (np.abs(wide) + divisor // 2) // divisor
    rounded = np.where(wide < 0, -magnitude, magnitude)
    return np.clip(rounded, -128, 127).astype(np.int8)


def scale_unsigned(
    input_tensor: object,
    *,
    multiplier: int,
    divisor: int,
    output_dtype: str,
) -> object:
    """Apply the proved unsigned multiplyadd scale with half-up rounding."""

    import numpy as np

    tensor = np.asarray(input_tensor)
    if not np.issubdtype(tensor.dtype, np.unsignedinteger):
        raise OperatorSemanticsError("unsigned scaler input must be unsigned integer")
    if multiplier <= 0 or multiplier & (multiplier - 1):
        raise OperatorSemanticsError("unsigned scaler multiplier must be a power of two")
    if divisor <= 0 or divisor & (divisor - 1):
        raise OperatorSemanticsError("unsigned scaler divisor must be a power of two")
    if output_dtype not in {"int8", "uint8", "uint16"}:
        raise OperatorSemanticsError("unsupported unsigned scaler output dtype")
    dtype = np.dtype(output_dtype)
    limits = np.iinfo(dtype)
    scaled = tensor.astype(np.uint64, copy=False) * multiplier
    rounded = (scaled + divisor // 2) // divisor
    return np.clip(rounded, limits.min, limits.max).astype(dtype)


def quantized_silu_int8(
    input_tensor: object,
    *,
    input_exponent_offset: int = 3,
    output_exponent_offset: int = 4,
) -> object:
    """Apply a recovered CV22 int8 SiLU scale pair.

    The CV22 graph computes ``x * sigmoid(x)`` through a sign/magnitude
    decomposition and its unsigned 16-bit floating format.  ADES differential
    evidence establishes the declared scale pairs, and every resulting value
    remains clear of an unresolved half-rounding boundary.
    """

    import numpy as np

    tensor = np.asarray(input_tensor)
    if tensor.dtype != np.int8:
        raise OperatorSemanticsError("quantized SiLU input must be int8")
    scale_pair = (input_exponent_offset, output_exponent_offset)
    if scale_pair not in _CV22_SILU_SCALE_PAIRS:
        raise OperatorSemanticsError("quantized SiLU scale pair is unproved")
    if scale_pair == (3, 4):
        lookup_values = _CV22_SILU_INT8_LUT
    else:
        input_scale = 2**input_exponent_offset
        output_scale = 2**output_exponent_offset
        lookup_values = tuple(
            max(
                -128,
                min(
                    127,
                    round(
                        (value / input_scale)
                        / (1 + math.exp(-value / input_scale))
                        * output_scale
                    ),
                ),
            )
            for value in range(-128, 128)
        )
    lookup = np.asarray(lookup_values, dtype=np.int8)
    return lookup[tensor.astype(np.int16) + 128]


def cv22_sigmoid_float32(
    input_tensor: object, *, input_exponent_offset: int
) -> object:
    """Decode proved CV22 sigmoid results to the detector's float32 format.

    The captured sigmoid chains normalize their inputs to an effective
    int8/2^3 domain.  This routine deliberately rejects values outside the
    ADES-proved lookup instead of silently substituting host ``exp`` behavior.
    """

    import numpy as np

    tensor = np.asarray(input_tensor)
    if tensor.dtype != np.int8:
        raise OperatorSemanticsError("CV22 sigmoid input must be int8")
    if input_exponent_offset not in {2, 3}:
        raise OperatorSemanticsError("CV22 sigmoid exponent offset is unproved")
    raw_input = tensor.astype(np.int16)
    effective = raw_input * (2 ** (3 - input_exponent_offset))
    minimum = _CV22_SIGMOID_EXP3_MINIMUM
    maximum = minimum + len(_CV22_SIGMOID_EXP3_UFLOAT16) - 1
    if input_exponent_offset == 2:
        proved = (raw_input >= -111) & (raw_input <= 21)
        tail = raw_input <= -11
    else:
        proved = (raw_input >= -128) & (raw_input <= 42)
        tail = raw_input <= -101
    if not bool(np.all(proved)):
        unresolved = np.unique(effective[~proved]).tolist()
        raise OperatorSemanticsError(
            f"CV22 sigmoid effective values are unproved: {unresolved}"
        )
    result = np.empty(effective.shape, dtype=np.float32)
    if input_exponent_offset == 2:
        tail_values = np.asarray(
            _CV22_SIGMOID_EXP2_TAIL_FLOAT32_BITS, dtype=np.uint32
        ).view(np.float32)
        result[tail] = tail_values[raw_input[tail] + 111]
    else:
        tail_values = np.asarray(
            _CV22_SIGMOID_EXP3_TAIL_FLOAT32_BITS, dtype=np.uint32
        ).view(np.float32)
        result[tail] = tail_values[raw_input[tail] + 128]
    contiguous = ~tail
    lookup = np.asarray(_CV22_SIGMOID_EXP3_UFLOAT16, dtype=np.uint16)
    raw = lookup[effective[contiguous] - minimum]
    exponent = raw >> 11
    mantissa = raw & 0x07FF
    result[contiguous] = np.ldexp(
        1.0 + mantissa.astype(np.float64) / 2048.0,
        exponent.astype(np.int32) - 30,
    ).astype(np.float32)
    for value, bits in _CV22_SIGMOID_POSITIVE_FLOAT32_BITS.items():
        observed = effective == value
        if bool(np.any(observed)):
            result[observed] = np.asarray(bits, dtype=np.uint32).view(np.float32)
    return result


def execute_sparse_fastconv_region(
    input_tensor: object,
    channels: tuple[FastconvChannel, ...],
    geometry: FastconvGeometry,
    *,
    output_origin: tuple[int, int],
    output_shape: tuple[int, int],
) -> object:
    """Execute the proved CB62 sparse integer convolution over a bounded region.

    The current proof covers planar unsigned-eight-bit input, zero padding,
    signed integer accumulation, arithmetic accumulator shift, per-channel
    offset, the observed ADES final rounding control value ``2`` (one-bit
    rounded right shift), and unsigned output saturation.
    """

    import numpy as np

    tensor = np.asarray(input_tensor)
    if tensor.dtype != np.uint8 or tensor.ndim != 3:
        raise OperatorSemanticsError("fastconv input must be planar CHW uint8")
    if not channels:
        raise OperatorSemanticsError("fastconv requires at least one output channel")
    origin_y, origin_x = output_origin
    output_height, output_width = output_shape
    if min(origin_y, origin_x) < 0 or min(output_height, output_width) <= 0:
        raise OperatorSemanticsError("output region must be positive and non-negative")

    outputs = np.empty((len(channels), output_height, output_width), dtype=np.uint8)
    output_y = origin_y + np.arange(output_height, dtype=np.int64)
    output_x = origin_x + np.arange(output_width, dtype=np.int64)

    for channel_index, channel in enumerate(channels):
        if channel.output_saturation_max != 255:
            raise OperatorSemanticsError(
                "only proved unsigned-eight-bit output saturation is executable"
            )
        if channel.final_shift_control != 2:
            raise OperatorSemanticsError(
                "only proved final_shift_control value 2 is executable"
            )
        accumulator = np.zeros((output_height, output_width), dtype=np.int64)
        for point in channel.points:
            if point.input_channel >= tensor.shape[0]:
                raise OperatorSemanticsError("kernel references an absent input channel")
            if not (
                0 <= point.kernel_y < geometry.kernel_height
                and 0 <= point.kernel_x < geometry.kernel_width
            ):
                raise OperatorSemanticsError("kernel point exceeds declared geometry")
            input_y = (
                output_y * geometry.stride_y
                + point.kernel_y
                - geometry.padding_top
            )
            input_x = (
                output_x * geometry.stride_x
                + point.kernel_x
                - geometry.padding_left
            )
            valid_y = (input_y >= 0) & (input_y < tensor.shape[1])
            valid_x = (input_x >= 0) & (input_x < tensor.shape[2])
            if not valid_y.any() or not valid_x.any():
                continue
            safe_y = np.clip(input_y, 0, tensor.shape[1] - 1)
            safe_x = np.clip(input_x, 0, tensor.shape[2] - 1)
            values = tensor[point.input_channel][np.ix_(safe_y, safe_x)].astype(
                np.int64, copy=False
            )
            valid = valid_y[:, None] & valid_x[None, :]
            accumulator += np.where(valid, values * point.weight, 0)

        shifted = accumulator >> channel.accumulator_shift
        # The exact split-0 oracle has shifted sums above 0x1ff.  ADES clamps
        # those values to 0x1ff before applying the offset.  A lower saturation
        # boundary is intentionally not asserted until a distinguishable oracle
        # case proves it.
        shifted = np.minimum(shifted, 0x1FF)
        biased = shifted + channel.offset
        rounded = (biased + 1) >> 1
        outputs[channel_index] = np.clip(
            rounded, 0, channel.output_saturation_max
        ).astype(np.uint8)
    return outputs
