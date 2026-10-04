"""Persistent MLX execution for proved CB62 fast-convolution semantics."""

from __future__ import annotations

from importlib.metadata import version
from types import MappingProxyType
from typing import Mapping

from verkeye.cv22.operators import FastconvGeometry
from verkeye.cv22.parameter_extraction import FastconvChannel


class MlxFastconvSession:
    """Compile one exact fast-convolution configuration once and reuse it."""

    def __init__(
        self,
        channels: tuple[FastconvChannel, ...],
        geometry: FastconvGeometry,
        *,
        input_channels: int,
        input_dtype: str = "uint8",
        output_dtype: str | None = None,
    ) -> None:
        if not channels:
            raise ValueError("MLX fastconv requires output channels")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if input_dtype not in {"int8", "uint8", "uint16"}:
            raise ValueError("MLX fastconv input dtype must be int8, uint8, or uint16")
        output_saturations = {channel.output_saturation_max for channel in channels}
        if len(output_saturations) != 1 or next(iter(output_saturations)) not in {
            255,
            65_535,
        }:
            raise ValueError("unproved output saturation for MLX fastconv")
        if output_dtype is None:
            output_dtype = "uint8" if 255 in output_saturations else "uint16"
        if output_dtype not in {"uint8", "uint16", "int8"}:
            raise ValueError("unproved output dtype for MLX fastconv")
        if output_dtype == "uint16" and channels[0].output_saturation_max != 65_535:
            raise ValueError("output dtype differs from fastconv saturation")
        if output_dtype in {"uint8", "int8"} and channels[0].output_saturation_max != 255:
            raise ValueError("output dtype differs from fastconv saturation")
        expected_control = 3 if output_dtype == "int8" else 2
        for channel in channels:
            if channel.final_shift_control != expected_control:
                raise ValueError("unproved final_shift_control for MLX fastconv")

        import mlx.core as mx
        import numpy as np

        weights = np.zeros(
            (
                len(channels),
                geometry.kernel_height,
                geometry.kernel_width,
                input_channels,
            ),
            dtype=np.float32,
        )
        for output_channel, channel in enumerate(channels):
            for point in channel.points:
                if not 0 <= point.input_channel < input_channels:
                    raise ValueError("kernel point references absent input channel")
                if not (
                    0 <= point.kernel_y < geometry.kernel_height
                    and 0 <= point.kernel_x < geometry.kernel_width
                ):
                    raise ValueError("kernel point exceeds declared geometry")
                weights[
                    output_channel,
                    point.kernel_y,
                    point.kernel_x,
                    point.input_channel,
                ] = point.weight

        input_type = np.dtype(input_dtype)
        input_limits = np.iinfo(input_type)
        input_maximum = max(abs(input_limits.min), abs(input_limits.max))
        absolute_weight_sums = np.abs(weights).sum(axis=(1, 2, 3), dtype=np.float64)
        if float(absolute_weight_sums.max(initial=0)) * input_maximum > np.iinfo(
            np.int64
        ).max:
            raise ValueError("fastconv accumulator exceeds proved int64 range")

        chunk_size = 64
        weight_chunks = []
        for start in range(0, input_channels, chunk_size):
            stop = min(start + chunk_size, input_channels)
            chunk = weights[:, :, :, start:stop]
            partial_input_maximum = 128 if input_dtype == "int8" else 255
            partial_bound = float(
                np.abs(chunk).sum(axis=(1, 2, 3), dtype=np.float64).max(initial=0)
            ) * partial_input_maximum
            if partial_bound > 2**24 - 1:
                raise ValueError("fastconv partial exceeds exact float32 integer range")
            weight_chunks.append((start, stop, mx.array(chunk)))

        self._mx = mx
        self._np = np
        self._geometry = geometry
        self._input_channels = input_channels
        self._input_dtype = np.dtype(input_dtype)
        self._output_channels = len(channels)
        self._output_saturation_max = channels[0].output_saturation_max
        self._output_dtype = output_dtype
        self._weight_chunks = tuple(weight_chunks)
        self._accumulator_shifts = mx.array(
            np.asarray(
                [channel.accumulator_shift for channel in channels], dtype=np.int32
            ).reshape(1, 1, 1, len(channels))
        )
        self._offsets = mx.array(
            np.asarray([channel.offset for channel in channels], dtype=np.int64).reshape(
                1, 1, 1, len(channels)
            )
        )

        def execute(input_array: object) -> object:
            accumulator = None
            padding = (geometry.padding_top, geometry.padding_left)
            if (
                geometry.effective_padding_bottom != geometry.padding_top
                or geometry.effective_padding_right != geometry.padding_left
            ):
                input_array = mx.pad(
                    input_array,
                    (
                        (0, 0),
                        (geometry.padding_top, geometry.effective_padding_bottom),
                        (geometry.padding_left, geometry.effective_padding_right),
                        (0, 0),
                    ),
                )
                padding = (0, 0)
            if input_dtype == "int8":
                byte_shifts: tuple[int | None, ...] = (None,)
            else:
                byte_shifts = (0,) if input_dtype == "uint8" else (0, 8)
            for byte_shift in byte_shifts:
                if byte_shift is None:
                    byte_input = input_array.astype(mx.float32)
                else:
                    byte_input = (
                        mx.right_shift(input_array.astype(mx.uint32), byte_shift) & 0xFF
                    ).astype(mx.float32)
                for start, stop, weight_chunk in self._weight_chunks:
                    partial = mx.conv2d(
                        byte_input[:, :, :, start:stop],
                        weight_chunk,
                        stride=(geometry.stride_y, geometry.stride_x),
                        padding=padding,
                    ).astype(mx.int64)
                    if byte_shift:
                        partial = mx.left_shift(partial, byte_shift)
                    accumulator = (
                        partial if accumulator is None else accumulator + partial
                    )
            if (
                self._output_dtype == "int8"
                and geometry.kernel_height == 2
                and geometry.kernel_width == 2
            ):
                accumulator = mx.clip(accumulator, -32_768, 32_767)
            result = mx.right_shift(accumulator, self._accumulator_shifts)
            if self._output_dtype == "int8":
                # ADES saturates the shifted accumulator's positive domain at
                # 0x1ff before applying the signed-output offset and final
                # two-bit rounded shift.  This is observable independently of
                # the final int8 clamp when a negative offset pulls 511 back
                # into range.
                result = mx.minimum(result, 0x1FF)
                result = mx.right_shift(result + self._offsets + 2, 2)
                return mx.clip(result, -128, 127).astype(mx.int8)
            result = mx.minimum(result, self._output_saturation_max * 2 + 1)
            result = mx.right_shift(result + self._offsets + 1, 1)
            dtype = mx.uint8 if self._output_dtype == "uint8" else mx.uint16
            return mx.clip(result, 0, self._output_saturation_max).astype(dtype)

        self._execute = mx.compile(execute)
        self._identity: Mapping[str, object] = MappingProxyType(
            {
                "provider": "mlx",
                "version": version("mlx"),
                "device": str(mx.default_device()),
                "numeric_path": (
                    (
                        "signed float32 partial convolutions"
                        if input_dtype == "int8"
                        else "byte-decomposed float32 partial convolutions"
                    )
                    + " with exact int64 accumulation and CV22 requantization; "
                    "byte-exactness must be established per registered operator"
                ),
                "input_dtype": input_dtype,
                "output_dtype": output_dtype,
                "input_channel_chunk_size": chunk_size,
            }
        )
        self.session_creation_count = 1
        self.inference_count = 0

    @property
    def input_channels(self) -> int:
        return self._input_channels

    @property
    def output_channels(self) -> int:
        return self._output_channels

    @property
    def identity(self) -> Mapping[str, object]:
        return self._identity

    def infer(self, input_tensor: object) -> object:
        tensor = self._np.asarray(input_tensor)
        if tensor.dtype != self._input_dtype or tensor.ndim != 3:
            raise ValueError(
                "MLX fastconv input dtype/shape must be planar CHW "
                f"{self._input_dtype.name}"
            )
        if tensor.shape[0] != self._input_channels:
            raise ValueError("MLX fastconv input channel count differs from session")
        input_array = self._mx.array(tensor.transpose(1, 2, 0)[None])
        result = self.infer_device(input_array)
        self._mx.eval(result)
        output = self._np.array(result).transpose(0, 3, 1, 2)[0]
        return output

    def infer_device(self, input_array: object) -> object:
        """Execute one NHWC tensor while retaining the result on the MLX device."""

        expected_dtype = {
            self._np.dtype("int8"): self._mx.int8,
            self._np.dtype("uint8"): self._mx.uint8,
            self._np.dtype("uint16"): self._mx.uint16,
        }[self._input_dtype]
        if (
            getattr(input_array, "ndim", None) != 4
            or input_array.shape[0] != 1
            or input_array.shape[3] != self._input_channels
            or input_array.dtype != expected_dtype
        ):
            raise ValueError(
                "MLX fastconv device input must be NHWC with the session dtype "
                "and channel count"
            )
        result = self._execute(input_array)
        self.inference_count += 1
        return result


class MlxFastconvChain:
    """Execute a connected sequence of persistent exact fastconv sessions."""

    def __init__(self, sessions: tuple[MlxFastconvSession, ...]) -> None:
        if not sessions:
            raise ValueError("MLX fastconv chain requires at least one session")
        for previous, current in zip(sessions, sessions[1:], strict=False):
            if previous.output_channels != current.input_channels:
                raise ValueError(
                    "MLX fastconv chain channel mismatch: "
                    f"{previous.output_channels} != {current.input_channels}"
                )
        self._sessions = sessions
        self.session_creation_count = sum(
            session.session_creation_count for session in sessions
        )
        self.inference_count = 0

    @property
    def sessions(self) -> tuple[MlxFastconvSession, ...]:
        return self._sessions

    @property
    def identity(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "provider": "mlx",
                "session_count": len(self._sessions),
                "sessions": tuple(dict(session.identity) for session in self._sessions),
            }
        )

    def infer(self, input_tensor: object) -> object:
        output = input_tensor
        for session in self._sessions:
            output = session.infer(output)
        self.inference_count += 1
        return output

    def infer_with_intermediates(self, input_tensor: object) -> tuple[object, ...]:
        """Return every layer boundary for differential-oracle validation."""

        outputs: list[object] = []
        output = input_tensor
        for session in self._sessions:
            output = session.infer(output)
            outputs.append(output)
        self.inference_count += 1
        return tuple(outputs)
