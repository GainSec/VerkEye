"""Persistent OpenVINO execution for exact recovered CV22 convolutions."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from verkeye.cv22.operators import (
    FastconvGeometry,
    channel_halves,
    concatenate_channels,
    cv22_sigmoid_float32,
    max_filter_5x5,
    quantized_silu_int8,
)
from verkeye.cv22.parameter_extraction import (
    FastconvChannel,
    parse_fastconv_capture,
)
from verkeye.runtime.generation import GeneratedRuntime


def _capture(root: Path, call: int):
    entries = (root / f"fastconv-{call}-entries.bin").read_bytes()
    payloads = {
        index: path.read_bytes()
        for index in range(len(entries) // 96)
        if (path := root / f"fastconv-{call}-entry-{index}-offset-0.bin").is_file()
    }
    return parse_fastconv_capture(entries, payloads)


def _exact_channel_chunks(
    weights: np.ndarray,
    input_maximum: int,
) -> tuple[tuple[int, int], ...]:
    """Pack the widest sequential channel ranges with exact float32 sums."""

    if weights.ndim != 4 or weights.shape[1] <= 0 or input_maximum <= 0:
        raise ValueError("weights and input maximum must define a convolution")
    channel_bounds = np.abs(weights).sum(axis=(2, 3), dtype=np.float64)
    limit = float(2**24 - 1)
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < weights.shape[1]:
        accumulated = np.zeros(weights.shape[0], dtype=np.float64)
        stop = start
        while stop < weights.shape[1]:
            candidate = accumulated + channel_bounds[:, stop]
            if float(candidate.max(initial=0)) * input_maximum > limit:
                break
            accumulated = candidate
            stop += 1
        if stop == start:
            raise ValueError("one input channel exceeds the exact float32 range")
        chunks.append((start, stop))
        start = stop
    return tuple(chunks)


def _fastconv_node(
    input_node: object,
    channels: tuple[FastconvChannel, ...],
    geometry: FastconvGeometry,
    *,
    input_dtype: str,
    output_dtype: str,
    integer_shifts: bool,
) -> object:
    """Append one exact recovered convolution to an OpenVINO graph."""

    import openvino as ov
    from openvino import opset15 as ops

    _, input_channels, input_height, input_width = input_node.get_output_shape(0)
    weights = np.zeros(
        (
            len(channels),
            input_channels,
            geometry.kernel_height,
            geometry.kernel_width,
        ),
        dtype=np.float32,
    )
    for output_channel, channel in enumerate(channels):
        for point in channel.points:
            weights[
                output_channel,
                point.input_channel,
                point.kernel_y,
                point.kernel_x,
            ] = point.weight

    base_input = ops.convert(input_node, ov.Type.f32)
    if input_dtype == "uint16":
        high_byte = ops.floor(
            ops.divide(base_input, ops.constant(256.0, ov.Type.f32))
        )
        byte_inputs = (
            (
                ops.subtract(
                    base_input,
                    ops.multiply(high_byte, ops.constant(256.0, ov.Type.f32)),
                ),
                0,
            ),
            (high_byte, 8),
        )
    else:
        byte_inputs = ((base_input, 0),)
    input_maximum = 128 if input_dtype == "int8" else 255
    channel_chunks = _exact_channel_chunks(weights, input_maximum)
    accumulator_bound = float(
        np.abs(weights).sum(axis=(1, 2, 3), dtype=np.float64).max(initial=0)
    ) * ({"int8": 128, "uint8": 255, "uint16": 65_535}[input_dtype])
    accumulator_type = (
        ov.Type.i32 if accumulator_bound <= np.iinfo(np.int32).max else ov.Type.i64
    )
    accumulator_numpy_type = np.int32 if accumulator_type == ov.Type.i32 else np.int64
    accumulator = None
    for byte_input, byte_shift in byte_inputs:
        for start, stop in channel_chunks:
            sliced_input = ops.slice(
                byte_input,
                ops.constant([0, start, 0, 0], np.int64),
                ops.constant([1, stop, input_height, input_width], np.int64),
                ops.constant([1, 1, 1, 1], np.int64),
            )
            selected_weights = weights[:, start:stop]
            partial = ops.convolution(
                sliced_input,
                ops.constant(selected_weights),
                strides=[geometry.stride_y, geometry.stride_x],
                pads_begin=[geometry.padding_top, geometry.padding_left],
                pads_end=[
                    geometry.effective_padding_bottom,
                    geometry.effective_padding_right,
                ],
                dilations=[1, 1],
            )
            partial = ops.convert(partial, accumulator_type)
            if byte_shift:
                partial = ops.multiply(partial, ops.constant(256, accumulator_type))
            accumulator = (
                partial if accumulator is None else ops.add(accumulator, partial)
            )
    assert accumulator is not None
    shifts = ops.constant(
        np.asarray(
            [channel.accumulator_shift for channel in channels],
            dtype=accumulator_numpy_type,
        ).reshape(1, len(channels), 1, 1)
    )
    offsets = ops.constant(
        np.asarray(
            [channel.offset for channel in channels], dtype=accumulator_numpy_type
        ).reshape(1, len(channels), 1, 1)
    )
    if (
        output_dtype == "int8"
        and geometry.kernel_height == 2
        and geometry.kernel_width == 2
    ):
        accumulator = ops.maximum(
            accumulator, ops.constant(-32_768, accumulator_type)
        )
        accumulator = ops.minimum(
            accumulator, ops.constant(32_767, accumulator_type)
        )
    if integer_shifts:
        result = ops.bitwise_right_shift(accumulator, shifts)
    else:
        divisors = ops.power(
            ops.constant(2.0, ov.Type.f64), ops.convert(shifts, ov.Type.f64)
        )
        result = ops.convert(
            ops.floor(ops.divide(ops.convert(accumulator, ov.Type.f64), divisors)),
            accumulator_type,
        )
    if output_dtype == "int8":
        result = ops.minimum(result, ops.constant(0x1FF, accumulator_type))
        result = ops.add(ops.add(result, offsets), ops.constant(2, accumulator_type))
        if integer_shifts:
            result = ops.bitwise_right_shift(
                result, ops.constant(2, accumulator_type)
            )
        else:
            result = ops.convert(
                ops.floor(
                    ops.divide(
                        ops.convert(result, ov.Type.f64),
                        ops.constant(4.0, ov.Type.f64),
                    )
                ),
                accumulator_type,
            )
        result = ops.maximum(result, ops.constant(-128, accumulator_type))
        result = ops.minimum(result, ops.constant(127, accumulator_type))
        return ops.convert(result, ov.Type.i8)
    maximum = channels[0].output_saturation_max
    result = ops.minimum(
        result, ops.constant(maximum * 2 + 1, accumulator_type)
    )
    result = ops.add(ops.add(result, offsets), ops.constant(1, accumulator_type))
    if integer_shifts:
        result = ops.bitwise_right_shift(result, ops.constant(1, accumulator_type))
    else:
        result = ops.convert(
            ops.floor(
                ops.divide(
                    ops.convert(result, ov.Type.f64),
                    ops.constant(2.0, ov.Type.f64),
                )
            ),
            accumulator_type,
        )
    result = ops.maximum(result, ops.constant(0, accumulator_type))
    result = ops.minimum(result, ops.constant(maximum, accumulator_type))
    return ops.convert(result, ov.Type.u8 if output_dtype == "uint8" else ov.Type.u16)


class OpenVinoFastconvSession:
    """Compile one fixed-shape exact fast-convolution for the CPU plugin."""

    def __init__(
        self,
        channels: tuple[FastconvChannel, ...],
        geometry: FastconvGeometry,
        *,
        input_shape: tuple[int, int, int],
        input_dtype: str = "uint8",
        output_dtype: str | None = None,
        device: str = "CPU",
    ) -> None:
        if not channels:
            raise ValueError("OpenVINO fastconv requires output channels")
        if len(input_shape) != 3 or any(value <= 0 for value in input_shape):
            raise ValueError("input_shape must be positive CHW")
        if input_dtype not in {"int8", "uint8", "uint16"}:
            raise ValueError("OpenVINO input dtype must be int8, uint8, or uint16")
        output_saturations = {channel.output_saturation_max for channel in channels}
        if len(output_saturations) != 1 or next(iter(output_saturations)) not in {
            255,
            65_535,
        }:
            raise ValueError("unproved output saturation for OpenVINO fastconv")
        if output_dtype is None:
            output_dtype = "uint8" if 255 in output_saturations else "uint16"
        if output_dtype not in {"uint8", "uint16", "int8"}:
            raise ValueError("unproved output dtype for OpenVINO fastconv")
        expected_control = 3 if output_dtype == "int8" else 2
        if any(channel.final_shift_control != expected_control for channel in channels):
            raise ValueError("unproved final_shift_control for OpenVINO fastconv")

        import openvino as ov
        from openvino import opset15 as ops

        input_channels, input_height, input_width = input_shape
        element_type = {
            "int8": ov.Type.i8,
            "uint8": ov.Type.u8,
            "uint16": ov.Type.u16,
        }[input_dtype]
        parameter = ops.parameter(
            [1, input_channels, input_height, input_width],
            element_type,
            name="input",
        )
        weights = np.zeros(
            (
                len(channels),
                input_channels,
                geometry.kernel_height,
                geometry.kernel_width,
            ),
            dtype=np.float32,
        )
        for output_channel, channel in enumerate(channels):
            for point in channel.points:
                if not 0 <= point.input_channel < input_channels:
                    raise ValueError("kernel point references absent input channel")
                weights[
                    output_channel,
                    point.input_channel,
                    point.kernel_y,
                    point.kernel_x,
                ] = point.weight

        accumulator = None
        chunk_size = 64
        byte_shifts: tuple[int | None, ...]
        byte_shifts = (None,) if input_dtype == "int8" else ((0,) if input_dtype == "uint8" else (0, 8))
        for byte_shift in byte_shifts:
            if byte_shift is None:
                byte_input = ops.convert(parameter, ov.Type.f32)
            else:
                shifted = ops.bitwise_right_shift(
                    ops.convert(parameter, ov.Type.u32),
                    ops.constant(byte_shift, ov.Type.u32),
                )
                byte_input = ops.convert(
                    ops.bitwise_and(shifted, ops.constant(0xFF, ov.Type.u32)),
                    ov.Type.f32,
                )
            for start in range(0, input_channels, chunk_size):
                stop = min(start + chunk_size, input_channels)
                partial_bound = float(
                    np.abs(weights[:, start:stop]).sum(
                        axis=(1, 2, 3), dtype=np.float64
                    ).max(initial=0)
                ) * (128 if input_dtype == "int8" else 255)
                if partial_bound > 2**24 - 1:
                    raise ValueError("fastconv partial exceeds exact float32 range")
                sliced = ops.slice(
                    byte_input,
                    ops.constant([0, start, 0, 0], np.int64),
                    ops.constant([1, stop, input_height, input_width], np.int64),
                    ops.constant([1, 1, 1, 1], np.int64),
                )
                partial = ops.convolution(
                    sliced,
                    ops.constant(weights[:, start:stop]),
                    strides=[geometry.stride_y, geometry.stride_x],
                    pads_begin=[geometry.padding_top, geometry.padding_left],
                    pads_end=[
                        geometry.effective_padding_bottom,
                        geometry.effective_padding_right,
                    ],
                    dilations=[1, 1],
                )
                partial = ops.convert(partial, ov.Type.i64)
                if byte_shift:
                    partial = ops.bitwise_left_shift(
                        partial, ops.constant(byte_shift, ov.Type.i64)
                    )
                accumulator = partial if accumulator is None else ops.add(accumulator, partial)

        assert accumulator is not None
        if (
            output_dtype == "int8"
            and geometry.kernel_height == 2
            and geometry.kernel_width == 2
        ):
            accumulator = ops.maximum(
                accumulator, ops.constant(-32_768, ov.Type.i64)
            )
            accumulator = ops.minimum(
                accumulator, ops.constant(32_767, ov.Type.i64)
            )
        shifts = ops.constant(
            np.asarray(
                [channel.accumulator_shift for channel in channels], dtype=np.int64
            ).reshape(1, len(channels), 1, 1)
        )
        offsets = ops.constant(
            np.asarray([channel.offset for channel in channels], dtype=np.int64).reshape(
                1, len(channels), 1, 1
            )
        )
        result = ops.bitwise_right_shift(accumulator, shifts)
        if output_dtype == "int8":
            result = ops.minimum(result, ops.constant(0x1FF, ov.Type.i64))
            result = ops.bitwise_right_shift(
                ops.add(ops.add(result, offsets), ops.constant(2, ov.Type.i64)),
                ops.constant(2, ov.Type.i64),
            )
            result = ops.maximum(result, ops.constant(-128, ov.Type.i64))
            result = ops.minimum(result, ops.constant(127, ov.Type.i64))
            result = ops.convert(result, ov.Type.i8)
        else:
            maximum = channels[0].output_saturation_max
            result = ops.minimum(result, ops.constant(maximum * 2 + 1, ov.Type.i64))
            result = ops.bitwise_right_shift(
                ops.add(ops.add(result, offsets), ops.constant(1, ov.Type.i64)),
                ops.constant(1, ov.Type.i64),
            )
            result = ops.maximum(result, ops.constant(0, ov.Type.i64))
            result = ops.minimum(result, ops.constant(maximum, ov.Type.i64))
            result = ops.convert(
                result, ov.Type.u8 if output_dtype == "uint8" else ov.Type.u16
            )

        core = ov.Core()
        model = ov.Model([result], [parameter], "cv22-fastconv-exact")
        self._compiled = core.compile_model(
            model,
            device,
            {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": str(max(1, min(6, input_channels))),
            },
        )
        self._request = self._compiled.create_infer_request()
        self._input_shape = input_shape
        self._input_dtype = np.dtype(input_dtype)
        self._identity: Mapping[str, object] = MappingProxyType(
            {
                "provider": "openvino",
                "version": version("openvino"),
                "device": device,
                "input_dtype": input_dtype,
                "output_dtype": output_dtype,
                "input_channel_chunk_size": chunk_size,
                "numeric_path": (
                    "chunked float32 convolution with exact integer partials, "
                    "int64 accumulation, and recovered CV22 requantization"
                ),
            }
        )
        self.session_creation_count = 1
        self.inference_count = 0

    @property
    def identity(self) -> Mapping[str, object]:
        return self._identity

    def infer(self, input_tensor: object) -> np.ndarray:
        tensor = np.asarray(input_tensor)
        if tensor.dtype != self._input_dtype or tensor.shape != self._input_shape:
            raise ValueError("OpenVINO fastconv input must match fixed CHW contract")
        result = self._request.infer({0: tensor[None]})[0]
        self.inference_count += 1
        return np.asarray(result)[0]


class OpenVinoCb62Session:
    """Compile all recovered CB62 operators into one persistent CPU graph."""

    MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"

    def __init__(
        self,
        *,
        capture_root: str | Path | None = None,
        split4_root: str | Path | None = None,
        split5_root: str | Path | None = None,
        generated_runtime: GeneratedRuntime | None = None,
        device: str = "CPU",
        inference_threads: int = 6,
        diagnostics: bool = False,
    ) -> None:
        import openvino as ov
        from openvino import opset15 as ops

        if inference_threads <= 0:
            raise ValueError("inference_threads must be positive")
        if generated_runtime is None and (
            capture_root is None or split4_root is None or split5_root is None
        ):
            raise ValueError(
                "accelerated runtime requires a generated runtime or all legacy roots"
            )
        captures = (
            {call: generated_runtime.fastconv_capture(call) for call in range(55)}
            if generated_runtime is not None
            else {call: _capture(Path(capture_root), call) for call in range(55)}  # type: ignore[arg-type]
        )
        g3 = lambda stride=1: FastconvGeometry(3, 3, stride, stride, 1, 1)
        g1 = FastconvGeometry(1, 1, 1, 1, 0, 0)
        gt = FastconvGeometry(
            2, 2, 1, 1, 1, 1, padding_bottom=0, padding_right=0
        )
        specifications = {
            0: (g3(2), "uint8", "uint8"), 1: (g3(2), "uint8", "uint8"),
            2: (g3(), "uint8", "uint8"), 3: (g3(), "uint8", "uint8"),
            4: (g3(2), "uint8", "uint8"), 5: (g3(), "uint8", "uint8"),
            6: (g3(), "uint8", "uint8"), 7: (g3(), "uint8", "uint8"),
            8: (g3(), "uint8", "uint8"), 9: (g3(2), "uint8", "uint16"),
            10: (g3(), "uint16", "uint16"), 11: (g3(), "uint16", "uint16"),
            12: (g3(), "uint16", "uint16"), 13: (g3(), "uint16", "uint16"),
            14: (g3(), "uint16", "uint16"), 15: (g3(), "uint16", "uint16"),
            16: (g3(2), "uint16", "uint16"), 17: (g3(), "uint16", "uint8"),
            18: (g3(), "uint8", "uint16"), 19: (g1, "uint16", "uint8"),
            20: (g1, "uint8", "uint8"), 21: (g1, "uint8", "uint8"),
            22: (gt, "uint8", "int8"), 23: (g3(), "int8", "uint16"),
            24: (g3(), "uint16", "uint8"), 25: (g3(), "uint8", "uint16"),
            26: (g3(), "uint16", "uint16"), 27: (g1, "uint16", "uint8"),
            28: (gt, "uint8", "int8"), 29: (g3(), "int8", "uint8"),
            30: (g3(), "uint8", "uint8"), 31: (g3(), "uint8", "uint8"),
            32: (g3(), "uint8", "uint8"), 33: (g3(2), "uint8", "uint8"),
            34: (g3(), "uint8", "uint16"), 35: (g3(), "uint16", "uint16"),
            36: (g3(), "uint16", "uint16"), 37: (g3(), "uint16", "uint8"),
            38: (g3(2), "uint8", "uint8"), 39: (g3(), "uint8", "uint8"),
            40: (g3(), "uint8", "uint8"), 41: (g3(), "uint8", "uint8"),
            42: (g3(), "uint8", "uint8"), 43: (g1, "uint8", "int8"),
            44: (g3(), "int8", "int8"), 45: (g1, "int8", "int8"),
            46: (g1, "int8", "int8"), 47: (g1, "uint8", "int8"),
            48: (g3(), "int8", "int8"), 49: (g1, "int8", "int8"),
            50: (g1, "int8", "int8"), 51: (g1, "uint8", "int8"),
            52: (g3(), "int8", "int8"), 53: (g1, "int8", "int8"),
            54: (g1, "int8", "int8"),
        }
        cpu_device = device.upper().startswith("CPU")

        def fc(call: int, node: object) -> object:
            geometry, input_dtype, output_dtype = specifications[call]
            return _fastconv_node(
                node,
                captures[call].channels,
                geometry,
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                integer_shifts=not cpu_device,
            )

        def requantize_signed(node: object, divisor: int) -> object:
            wide = ops.convert(node, ov.Type.i64)
            magnitude = ops.add(
                ops.absolute(wide), ops.constant(divisor // 2, ov.Type.i64)
            )
            if cpu_device:
                magnitude = ops.convert(
                    ops.floor(
                        ops.divide(
                            ops.convert(magnitude, ov.Type.f32),
                            ops.constant(float(divisor), ov.Type.f32),
                        )
                    ),
                    ov.Type.i64,
                )
            else:
                magnitude = ops.bitwise_right_shift(
                    magnitude,
                    ops.constant(divisor.bit_length() - 1, ov.Type.i64),
                )
            rounded = ops.select(
                ops.less(wide, ops.constant(0, ov.Type.i64)),
                ops.negative(magnitude),
                magnitude,
            )
            rounded = ops.maximum(rounded, ops.constant(-128, ov.Type.i64))
            rounded = ops.minimum(rounded, ops.constant(127, ov.Type.i64))
            return ops.convert(rounded, ov.Type.i8)

        def scale_unsigned(
            node: object, multiplier: int, divisor: int, dtype: object
        ) -> object:
            wide = ops.multiply(
                ops.convert(node, ov.Type.u32),
                ops.constant(multiplier, ov.Type.u32),
            )
            if divisor > 1:
                wide = ops.add(
                    wide, ops.constant(divisor // 2, ov.Type.u32)
                )
                if cpu_device:
                    wide = ops.convert(
                        ops.floor(
                            ops.divide(
                                ops.convert(wide, ov.Type.f32),
                                ops.constant(float(divisor), ov.Type.f32),
                            )
                        ),
                        ov.Type.u32,
                    )
                else:
                    wide = ops.bitwise_right_shift(
                        wide,
                        ops.constant(divisor.bit_length() - 1, ov.Type.u32),
                    )
            maximum = 65_535 if dtype == ov.Type.u16 else 127
            wide = ops.minimum(wide, ops.constant(maximum, ov.Type.u32))
            return ops.convert(wide, dtype)

        signed_domain = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        silu_luts = {
            pair: ops.constant(
                quantized_silu_int8(
                    signed_domain,
                    input_exponent_offset=pair[0],
                    output_exponent_offset=pair[1],
                )
            )
            for pair in ((1, 2), (1, 3), (2, 4), (3, 3), (3, 4))
        }

        def silu(node: object, pair: tuple[int, int] = (3, 4)) -> object:
            indices = ops.add(
                ops.convert(node, ov.Type.i16), ops.constant(128, ov.Type.i16)
            )
            return ops.gather(silu_luts[pair], indices, ops.constant(0, np.int64))

        sigmoid_luts = {
            2: ops.constant(
                cv22_sigmoid_float32(
                    np.arange(-111, 22, dtype=np.int8), input_exponent_offset=2
                )
            ),
            3: ops.constant(
                cv22_sigmoid_float32(
                    np.arange(-128, 43, dtype=np.int16).astype(np.int8),
                    input_exponent_offset=3,
                )
            ),
        }

        def sigmoid(node: object, exponent: int) -> tuple[object, object]:
            raw_input = ops.convert(node, ov.Type.i16)
            minimum, maximum = {2: (-111, 21), 3: (-128, 42)}[exponent]
            proved = ops.logical_and(
                ops.greater_equal(raw_input, ops.constant(minimum, ov.Type.i16)),
                ops.less_equal(raw_input, ops.constant(maximum, ov.Type.i16)),
            )
            indices = ops.minimum(
                ops.maximum(
                    ops.subtract(raw_input, ops.constant(minimum, ov.Type.i16)),
                    ops.constant(0, ov.Type.i16),
                ),
                ops.constant(maximum - minimum, ov.Type.i16),
            )
            values = ops.gather(
                sigmoid_luts[exponent], indices, ops.constant(0, np.int64)
            )
            proved = ops.reduce_logical_and(
                proved,
                ops.constant([0, 1, 2, 3], np.int64),
                False,
            )
            return values, proved

        def upsample2(node: object, height: int, width: int) -> object:
            node = ops.gather(
                node,
                ops.constant(np.repeat(np.arange(height), 2), np.int64),
                ops.constant(2, np.int64),
            )
            return ops.gather(
                node,
                ops.constant(np.repeat(np.arange(width), 2), np.int64),
                ops.constant(3, np.int64),
            )

        def max_filter(node: object) -> object:
            return ops.max_pool(
                node,
                strides=[1, 1],
                dilations=[1, 1],
                pads_begin=[2, 2],
                pads_end=[2, 2],
                kernel_shape=[5, 5],
                rounding_type="floor",
                auto_pad="explicit",
            ).output(0)

        def mask(packed: bytes, height: int, width: int) -> object:
            row_bytes = (width + 7) // 8
            encoded = np.frombuffer(packed, dtype=np.uint8).reshape(height, row_bytes)
            unpacked = np.unpackbits(encoded, axis=1, bitorder="little")[:, :width]
            return ops.constant(unpacked.astype(bool)[None, None])

        parameter = ops.parameter([1, 3, 608, 1088], ov.Type.u8, name="input")
        x = parameter
        for call in range(0, 6):
            x = fc(call, x)
        split0_output = x
        for call in range(6, 11):
            x = fc(call, x)
            if call == 8:
                backbone_p3 = x
        split1_deep = x
        for call in range(11, 15):
            x = fc(call, x)
        split2_output = x
        for call in range(15, 19):
            x = fc(call, x)
            if call == 15:
                backbone_p4 = x
        backbone_p5 = x

        split4_mask_path = (
            generated_runtime.split4_mask
            if generated_runtime is not None
            else Path(split4_root) / "d9-logical.bin"  # type: ignore[arg-type]
        )
        split5_mask_path = (
            generated_runtime.split5_mask
            if generated_runtime is not None
            else Path(split5_root) / "d14-logical.bin"  # type: ignore[arg-type]
        )
        reduced_p5 = fc(19, backbone_p5)
        d15 = max_filter(reduced_p5)
        d16 = max_filter(d15)
        pooled = max_filter(d16)
        merged = ops.concat(
            [
                reduced_p5,
                d15,
                d16,
                pooled,
            ],
            1,
        )
        neck_low = fc(20, merged)
        upsample_source = fc(21, neck_low)
        upsampled = upsample2(upsample_source, 19, 34)
        split4_upsampled = upsampled
        upsampled = ops.select(
            mask(split4_mask_path.read_bytes(), 38, 68),
            upsampled,
            ops.constant(0, ov.Type.u8),
        )
        split4_masked = upsampled
        split4_transpose_raw = fc(22, upsampled)
        neck_transpose = requantize_signed(split4_transpose_raw, 8)
        backbone_p4_signed = requantize_signed(backbone_p4, 128)

        x = fc(23, ops.concat([neck_transpose, backbone_p4_signed], 1))
        x = scale_unsigned(x, 4, 1, ov.Type.u16)
        for call in range(24, 28):
            x = fc(call, x)
        reduced_medium = x
        high_upsample = upsample2(reduced_medium, 38, 68)
        high_upsample = ops.select(
            mask(split5_mask_path.read_bytes(), 76, 136),
            high_upsample,
            ops.constant(0, ov.Type.u8),
        )
        high_upsample = fc(28, high_upsample)
        backbone_p3_signed = scale_unsigned(backbone_p3, 1, 2, ov.Type.i8)

        x = fc(29, ops.concat([high_upsample, backbone_p3_signed], 1))
        for call in range(30, 33):
            x = fc(call, x)
        high_feature = x
        x = fc(33, high_feature)
        x = fc(34, ops.concat([x, reduced_medium], 1))
        x = scale_unsigned(x, 2, 1, ov.Type.u16)
        for call in range(35, 38):
            x = fc(call, x)
        medium_feature = x
        x = fc(38, medium_feature)
        low_feature = fc(39, ops.concat([x, upsample_source], 1))

        x = low_feature
        for call in range(40, 44):
            x = fc(call, x)
        low = fc(44, silu(x))
        low_score_source, low_box_source = ops.split(low, 1, 2).outputs()
        low_score_logits = fc(46, silu(low_score_source, (3, 3)))
        low_boxes = fc(45, silu(low_box_source))

        medium = fc(48, silu(fc(47, medium_feature)))
        medium_score_source, medium_box_source = ops.split(medium, 1, 2).outputs()
        medium_score_logits = fc(50, silu(medium_score_source, (1, 2)))
        medium_boxes = fc(49, silu(medium_box_source, (1, 3)))

        high_input = silu(fc(51, high_feature), (2, 4))
        high = fc(52, high_input)
        high_score_source, high_box_source = ops.split(high, 1, 2).outputs()
        high_score_logits = fc(54, silu(high_score_source, (1, 2)))
        high_boxes = fc(53, silu(high_box_source, (1, 3)))

        high_scores, high_proved = sigmoid(high_score_logits, 2)
        medium_scores, medium_proved = sigmoid(medium_score_logits, 2)
        low_scores, low_proved = sigmoid(low_score_logits, 3)
        output_nodes = [
            high_scores,
            medium_scores,
            low_scores,
            ops.divide(
                ops.convert(high_boxes, ov.Type.f32), ops.constant(4.0, ov.Type.f32)
            ),
            ops.divide(
                ops.convert(medium_boxes, ov.Type.f32), ops.constant(4.0, ov.Type.f32)
            ),
            ops.divide(
                ops.convert(low_boxes, ov.Type.f32), ops.constant(4.0, ov.Type.f32)
            ),
            ops.logical_and(ops.logical_and(high_proved, medium_proved), low_proved),
        ]
        if diagnostics:
            output_nodes.extend(
                [
                    high_score_logits,
                    medium_score_logits,
                    low_score_logits,
                    split0_output,
                    backbone_p3,
                    split1_deep,
                    split2_output,
                    backbone_p4,
                    backbone_p5,
                    reduced_p5,
                    neck_low,
                    upsample_source,
                    split4_upsampled,
                    split4_masked,
                    split4_transpose_raw,
                    neck_transpose,
                    backbone_p4_signed,
                ]
            )
        core = ov.Core()
        model = ov.Model(output_nodes, [parameter], "cb62-cv22-exact")
        compile_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "PERF_COUNT": "YES",
        }
        if device.upper().startswith("CPU"):
            compile_config["INFERENCE_NUM_THREADS"] = str(inference_threads)
        else:
            compile_config["INFERENCE_PRECISION_HINT"] = "f32"
        self._compiled = core.compile_model(
            model,
            device,
            compile_config,
        )
        self._request = self._compiled.create_infer_request()
        self._identity = MappingProxyType(
            {
                "backend": "openvino-cv22-exact",
                "provider": "openvino",
                "version": version("openvino"),
                "device": device,
                "inference_threads": inference_threads,
                "model_sha256": self.MODEL_SHA256,
                "fastconv_node_count": 55,
                "diagnostics": diagnostics,
            }
        )
        self._diagnostics = diagnostics
        self.session_creation_count = 1
        self.inference_count = 0

    @property
    def identity(self) -> Mapping[str, object]:
        return self._identity

    def infer_outputs(self, input_tensor: object) -> Mapping[str, np.ndarray]:
        tensor = np.asarray(input_tensor)
        if tensor.dtype != np.uint8 or tensor.shape != (3, 608, 1088):
            raise ValueError("CB62 input must be exactly 3x608x1088 uint8")
        result = self._request.infer({0: tensor[None]})
        values = [np.asarray(result[index]) for index in range(7)]
        if not bool(values[6].item()):
            raise ValueError("CB62 scores left the ADES-proved sigmoid domain")
        outputs = {
            f"output_{index}": values[index] for index in range(6)
        }
        for value in outputs.values():
            value.setflags(write=False)
        self.inference_count += 1
        return MappingProxyType(outputs)

    def infer_diagnostics(self, input_tensor: object) -> Mapping[str, np.ndarray]:
        """Return raw detector logits alongside currently proved score outputs."""

        if not self._diagnostics:
            raise RuntimeError("OpenVINO diagnostics were not enabled")
        tensor = np.asarray(input_tensor)
        if tensor.dtype != np.uint8 or tensor.shape != (3, 608, 1088):
            raise ValueError("CB62 input must be exactly 3x608x1088 uint8")
        result = self._request.infer({0: tensor[None]})
        values = [np.asarray(result[index]) for index in range(24)]
        names = (
            "output_0",
            "output_1",
            "output_2",
            "output_3",
            "output_4",
            "output_5",
            "sigmoid_domain_proved",
            "high_score_logits",
            "medium_score_logits",
            "low_score_logits",
            "split0_output",
            "split1_output_0",
            "split1_output_1",
            "split2_output",
            "split3_output_0",
            "split3_output_1",
            "split4_op0",
            "split4_op17",
            "split4_op18",
            "split4_op19",
            "split4_op20",
            "split4_op21",
            "split4_op22",
            "split4_op23",
        )
        outputs = dict(zip(names, values, strict=True))
        for value in outputs.values():
            value.setflags(write=False)
        self.inference_count += 1
        return MappingProxyType(outputs)
