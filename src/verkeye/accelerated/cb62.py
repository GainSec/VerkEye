"""Persistent, exact host execution of the recovered CB62 detector graph."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from types import MappingProxyType
import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..compat.ades import (
    AdesExecutionResult,
    AdesSplitResult,
    AdesTensorRecord,
    assemble_cb62_predictions,
)
from ..compat.ades_runtime import AdesInference
from ..cv22.operators import (
    FastconvGeometry,
    apply_bitpacked_mux_mask,
    channel_halves,
    concatenate_channels,
    cv22_sigmoid_float32,
    max_filter_5x5,
    nearest_resample_2x,
    quantized_silu_int8,
    requantize_signed,
    scale_unsigned,
)
from ..cv22.parameter_extraction import parse_fastconv_capture
from .macos import MlxFastconvSession


class Cb62AcceleratedError(RuntimeError):
    """The accelerated CB62 package or input violates its proved contract."""


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_INPUT_BYTES = 3 * 608 * 1088


def validate_cb62_input(input_tensor: object) -> NDArray[np.uint8]:
    """Require the exact planar tensor consumed by the recovered model."""

    tensor = np.asarray(input_tensor)
    if tensor.dtype != np.uint8 or tensor.shape != (3, 608, 1088):
        raise Cb62AcceleratedError("CB62 input must be exactly 3x608x1088 uint8")
    return tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Cb62MlxSession:
    """Compile all 55 recovered fastconv nodes once and reuse them per frame."""

    MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"

    def __init__(
        self,
        *,
        capture_root: str | Path,
        split4_root: str | Path,
        split5_root: str | Path,
    ) -> None:
        capture_path = Path(capture_root)
        self._sessions: dict[int, MlxFastconvSession] = {}

        def add(
            call: int,
            geometry: FastconvGeometry,
            input_channels: int,
            input_dtype: str,
            output_dtype: str,
        ) -> None:
            entries_path = capture_path / f"fastconv-{call}-entries.bin"
            try:
                entries = entries_path.read_bytes()
            except OSError as error:
                raise Cb62AcceleratedError(
                    f"missing recovered fastconv capture {entries_path}"
                ) from error
            count = len(entries) // 96
            payloads: dict[int, bytes] = {}
            for index in range(count):
                point_path = (
                    capture_path
                    / f"fastconv-{call}-entry-{index}-offset-0.bin"
                )
                if point_path.is_file():
                    payloads[index] = point_path.read_bytes()
            capture = parse_fastconv_capture(entries, payloads)
            self._sessions[call] = MlxFastconvSession(
                capture.channels,
                geometry,
                input_channels=input_channels,
                input_dtype=input_dtype,
                output_dtype=output_dtype,
            )

        g3 = lambda stride=1: FastconvGeometry(3, 3, stride, stride, 1, 1)
        g1 = FastconvGeometry(1, 1, 1, 1, 0, 0)
        gt = FastconvGeometry(2, 2, 1, 1, 1, 1, padding_bottom=0, padding_right=0)
        specifications = (
            (0, g3(2), 3, "uint8", "uint8"),
            (1, g3(2), 22, "uint8", "uint8"),
            (2, g3(), 64, "uint8", "uint8"),
            (3, g3(), 64, "uint8", "uint8"),
            (4, g3(2), 64, "uint8", "uint8"),
            (5, g3(), 128, "uint8", "uint8"),
            (6, g3(), 127, "uint8", "uint8"),
            (7, g3(), 128, "uint8", "uint8"),
            (8, g3(), 128, "uint8", "uint8"),
            (9, g3(2), 128, "uint8", "uint16"),
            (10, g3(), 158, "uint16", "uint16"),
            (11, g3(), 168, "uint16", "uint16"),
            (12, g3(), 155, "uint16", "uint16"),
            (13, g3(), 161, "uint16", "uint16"),
            (14, g3(), 173, "uint16", "uint16"),
            (15, g3(), 186, "uint16", "uint16"),
            (16, g3(2), 256, "uint16", "uint16"),
            (17, g3(), 81, "uint16", "uint8"),
            (18, g3(), 483, "uint8", "uint16"),
            (19, g1, 83, "uint16", "uint8"),
            (20, g1, 1024, "uint8", "uint8"),
            (21, g1, 507, "uint8", "uint8"),
            (22, gt, 128, "uint8", "int8"),
            (23, g3(), 384, "int8", "uint16"),
            (24, g3(), 115, "uint16", "uint8"),
            (25, g3(), 115, "uint8", "uint16"),
            (26, g3(), 116, "uint16", "uint16"),
            (27, g1, 102, "uint16", "uint8"),
            (28, gt, 64, "uint8", "int8"),
            (29, g3(), 192, "int8", "uint8"),
            (30, g3(), 64, "uint8", "uint8"),
            (31, g3(), 64, "uint8", "uint8"),
            (32, g3(), 64, "uint8", "uint8"),
            (33, g3(2), 64, "uint8", "uint8"),
            (34, g3(), 128, "uint8", "uint16"),
            (35, g3(), 82, "uint16", "uint16"),
            (36, g3(), 75, "uint16", "uint16"),
            (37, g3(), 82, "uint16", "uint8"),
            (38, g3(2), 128, "uint8", "uint8"),
            (39, g3(), 256, "uint8", "uint8"),
            (40, g3(), 251, "uint8", "uint8"),
            (41, g3(), 256, "uint8", "uint8"),
            (42, g3(), 256, "uint8", "uint8"),
            (43, g1, 256, "uint8", "int8"),
            (44, g3(), 256, "int8", "int8"),
            (45, g1, 256, "int8", "int8"),
            (46, g1, 256, "int8", "int8"),
            (47, g1, 128, "uint8", "int8"),
            (48, g3(), 128, "int8", "int8"),
            (49, g1, 128, "int8", "int8"),
            (50, g1, 128, "int8", "int8"),
            (51, g1, 64, "uint8", "int8"),
            (52, g3(), 64, "int8", "int8"),
            (53, g1, 64, "int8", "int8"),
            (54, g1, 64, "int8", "int8"),
        )
        for specification in specifications:
            add(*specification)

        split4_path = Path(split4_root)
        split5_path = Path(split5_root)
        try:
            self._split4_mask = (split4_path / "d9-logical.bin").read_bytes()
            self._split5_mask = (split5_path / "d14-logical.bin").read_bytes()
        except (OSError, ValueError) as error:
            raise Cb62AcceleratedError("recovered static graph assets are incomplete") from error

        import mlx.core as mx

        self._mx = mx
        def device_mask(packed: bytes, height: int, width: int) -> object:
            row_bytes = (width + 7) // 8
            encoded = np.frombuffer(packed, dtype=np.uint8).reshape(
                height, row_bytes
            )
            unpacked = np.unpackbits(encoded, axis=1, bitorder="little")[:, :width]
            return mx.array(unpacked.astype(bool, copy=False)[None, :, :, None])

        self._split4_mask_device = device_mask(self._split4_mask, 38, 68)
        self._split5_mask_device = device_mask(self._split5_mask, 76, 136)
        signed_domain = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        self._silu_luts = {
            pair: mx.array(
                quantized_silu_int8(
                    signed_domain,
                    input_exponent_offset=pair[0],
                    output_exponent_offset=pair[1],
                )
            )
            for pair in ((1, 2), (1, 3), (2, 4), (3, 3), (3, 4))
        }
        self._sigmoid_luts = {
            2: mx.array(
                cv22_sigmoid_float32(
                    np.arange(-111, 22, dtype=np.int8), input_exponent_offset=2
                )
            ),
            3: mx.array(
                cv22_sigmoid_float32(
                    np.arange(-128, 43, dtype=np.int16).astype(np.int8),
                    input_exponent_offset=3,
                )
            ),
        }
        self._identity = MappingProxyType(
            {
                "backend": "mlx-cv22-exact",
                "model_sha256": self.MODEL_SHA256,
                "fastconv_session_count": len(self._sessions),
                "provider": dict(self._sessions[0].identity),
                "static_assets": {
                    "split4_mask": _sha256(split4_path / "d9-logical.bin"),
                    "split5_mask": _sha256(split5_path / "d14-logical.bin"),
                },
            }
        )
        self.session_creation_count = 1
        self.inference_count = 0

    @property
    def identity(self) -> Mapping[str, object]:
        return self._identity

    def _infer_outputs_host(
        self, input_tensor: object
    ) -> Mapping[str, NDArray[np.float32]]:
        """Reference execution retaining every recovered boundary on the host."""

        source = validate_cb62_input(input_tensor)
        s = self._sessions

        # Splits 0-3: linear recovered backbone, retaining two skip tensors.
        x = source
        for call in range(0, 6):
            x = s[call].infer(x)
        for call in range(6, 11):
            x = s[call].infer(x)
            if call == 8:
                backbone_p3 = x
        for call in range(11, 15):
            x = s[call].infer(x)
        for call in range(15, 19):
            x = s[call].infer(x)
            if call == 15:
                backbone_p4 = x
        backbone_p5 = x

        # Split 4: SPP-style branches, neck reduction, and exact signed joins.
        reduced_p5 = s[19].infer(backbone_p5)
        d15 = max_filter_5x5(reduced_p5)
        d16 = max_filter_5x5(d15)
        pooled = max_filter_5x5(d16)
        merged = concatenate_channels(
            concatenate_channels(reduced_p5, d15),
            concatenate_channels(d16, pooled),
        )
        neck_low = s[20].infer(merged)
        upsample_source = s[21].infer(neck_low)
        upsampled = apply_bitpacked_mux_mask(
            nearest_resample_2x(upsample_source), self._split4_mask
        )
        neck_transpose = s[22].infer(upsampled)
        neck_transpose = requantize_signed(neck_transpose, divisor=8)
        backbone_p4_signed = requantize_signed(backbone_p4, divisor=128)

        # Split 5: first FPN join and high-resolution upsample branch.
        split5_input = concatenate_channels(neck_transpose, backbone_p4_signed)
        x5 = s[23].infer(split5_input)
        x5 = scale_unsigned(x5, multiplier=4, divisor=1, output_dtype="uint16")
        for call in range(24, 28):
            x5 = s[call].infer(x5)
        reduced_medium = x5
        high_upsample = apply_bitpacked_mux_mask(
            nearest_resample_2x(reduced_medium), self._split5_mask
        )
        high_upsample = s[28].infer(high_upsample)
        backbone_p3_signed = scale_unsigned(
            backbone_p3, multiplier=1, divisor=2, output_dtype="int8"
        )

        # Split 6: PAN/FPN fusion and the three detector feature scales.
        x6 = s[29].infer(concatenate_channels(high_upsample, backbone_p3_signed))
        for call in range(30, 33):
            x6 = s[call].infer(x6)
        high_feature = x6
        medium_join = concatenate_channels(s[33].infer(high_feature), reduced_medium)
        x6 = s[34].infer(medium_join)
        x6 = scale_unsigned(x6, multiplier=2, divisor=1, output_dtype="uint16")
        for call in range(35, 38):
            x6 = s[call].infer(x6)
        medium_feature = x6
        x6 = s[38].infer(medium_feature)
        low_join = concatenate_channels(x6, upsample_source)
        low_feature = s[39].infer(low_join)

        # Split 7: final low-scale stem and CV22 quantized SiLU.
        x7 = low_feature
        for call in range(40, 44):
            x7 = s[call].infer(x7)
        low_detector_input = quantized_silu_int8(x7)

        # Split 8: low/medium heads plus the final high-scale stem.
        low = s[44].infer(low_detector_input)
        low_score_source, low_box_source = channel_halves(low)
        low_score_logits = s[46].infer(
            quantized_silu_int8(
                low_score_source,
                input_exponent_offset=3,
                output_exponent_offset=3,
            )
        )
        low_boxes = s[45].infer(quantized_silu_int8(low_box_source))

        medium_stem = quantized_silu_int8(s[47].infer(medium_feature))
        medium = s[48].infer(medium_stem)
        medium_score_source, medium_box_source = channel_halves(medium)
        medium_score_logits = s[50].infer(
            quantized_silu_int8(
                medium_score_source,
                input_exponent_offset=1,
                output_exponent_offset=2,
            )
        )
        medium_boxes = s[49].infer(
            quantized_silu_int8(
                medium_box_source,
                input_exponent_offset=1,
                output_exponent_offset=3,
            )
        )
        high_detector_input = quantized_silu_int8(
            s[51].infer(high_feature),
            input_exponent_offset=2,
            output_exponent_offset=4,
        )

        # Split 9: high-resolution box and score heads.
        high = s[52].infer(high_detector_input)
        high_score_source, high_box_source = channel_halves(high)
        high_score_logits = s[54].infer(
            quantized_silu_int8(
                high_score_source,
                input_exponent_offset=1,
                output_exponent_offset=2,
            )
        )
        high_boxes = s[53].infer(
            quantized_silu_int8(
                high_box_source,
                input_exponent_offset=1,
                output_exponent_offset=3,
            )
        )

        outputs = {
            "output_0": cv22_sigmoid_float32(
                high_score_logits, input_exponent_offset=2
            )[None],
            "output_1": cv22_sigmoid_float32(
                medium_score_logits, input_exponent_offset=2
            )[None],
            "output_2": cv22_sigmoid_float32(
                low_score_logits, input_exponent_offset=3
            )[None],
            "output_3": (high_boxes.astype(np.float32) / 4.0)[None],
            "output_4": (medium_boxes.astype(np.float32) / 4.0)[None],
            "output_5": (low_boxes.astype(np.float32) / 4.0)[None],
        }
        for value in outputs.values():
            value.setflags(write=False)
        self.inference_count += 1
        return MappingProxyType(outputs)

    def _device_silu(
        self,
        tensor: object,
        *,
        input_exponent_offset: int = 3,
        output_exponent_offset: int = 4,
    ) -> object:
        pair = (input_exponent_offset, output_exponent_offset)
        lookup = self._silu_luts[pair]
        indices = tensor.astype(self._mx.int16) + 128
        return self._mx.take(lookup, indices)

    def _device_sigmoid(
        self, tensor: object, *, input_exponent_offset: int
    ) -> tuple[object, object]:
        raw_input = tensor.astype(self._mx.int16)
        minimum, maximum = {2: (-111, 21), 3: (-128, 42)}[
            input_exponent_offset
        ]
        proved = (raw_input >= minimum) & (raw_input <= maximum)
        indices = self._mx.clip(raw_input - minimum, 0, maximum - minimum)
        values = self._mx.take(self._sigmoid_luts[input_exponent_offset], indices)
        return values, self._mx.all(proved)

    def _device_requantize_signed(self, tensor: object, divisor: int) -> object:
        wide = tensor.astype(self._mx.int64)
        magnitude = (self._mx.abs(wide) + divisor // 2) // divisor
        rounded = self._mx.where(wide < 0, -magnitude, magnitude)
        return self._mx.clip(rounded, -128, 127).astype(self._mx.int8)

    def _device_max_filter_5x5(self, tensor: object) -> object:
        padded = self._mx.pad(tensor, ((0, 0), (2, 2), (2, 2), (0, 0)))
        height, width = tensor.shape[1:3]
        result = padded[:, 0:height, 0:width, :]
        for row in range(5):
            for column in range(5):
                if row or column:
                    result = self._mx.maximum(
                        result,
                        padded[:, row : row + height, column : column + width, :],
                    )
        return result

    def _device_scale_unsigned(
        self, tensor: object, *, multiplier: int, divisor: int, dtype: object
    ) -> object:
        scaled = tensor.astype(self._mx.uint32) * multiplier
        rounded = (scaled + divisor // 2) // divisor
        maximum = 65_535 if dtype == self._mx.uint16 else 127
        return self._mx.clip(rounded, 0, maximum).astype(dtype)

    def infer_outputs(self, input_tensor: object) -> Mapping[str, NDArray[np.float32]]:
        """Execute the exact graph device-resident and return six detector heads."""

        source = validate_cb62_input(input_tensor)
        mx = self._mx
        s = self._sessions
        x = mx.array(source.transpose(1, 2, 0)[None])

        # Splits 0-3: backbone and two retained FPN skip tensors.
        for call in range(0, 6):
            x = s[call].infer_device(x)
        for call in range(6, 11):
            x = s[call].infer_device(x)
            if call == 8:
                backbone_p3 = x
        for call in range(11, 15):
            x = s[call].infer_device(x)
        for call in range(15, 19):
            x = s[call].infer_device(x)
            if call == 15:
                backbone_p4 = x
        backbone_p5 = x

        # Split 4: live SPPF branches and medium-resolution upsample join.
        reduced_p5 = s[19].infer_device(backbone_p5)
        d15 = self._device_max_filter_5x5(reduced_p5)
        d16 = self._device_max_filter_5x5(d15)
        pooled = self._device_max_filter_5x5(d16)
        merged = mx.concatenate(
            (
                reduced_p5,
                d15,
                d16,
                pooled,
            ),
            axis=-1,
        )
        neck_low = s[20].infer_device(merged)
        upsample_source = s[21].infer_device(neck_low)
        upsampled = mx.repeat(mx.repeat(upsample_source, 2, axis=1), 2, axis=2)
        upsampled = mx.where(self._split4_mask_device, upsampled, 0)
        neck_transpose = self._device_requantize_signed(
            s[22].infer_device(upsampled), 8
        )
        backbone_p4_signed = self._device_requantize_signed(backbone_p4, 128)

        # Split 5: first FPN join and high-resolution upsample branch.
        x = s[23].infer_device(
            mx.concatenate((neck_transpose, backbone_p4_signed), axis=-1)
        )
        x = self._device_scale_unsigned(
            x, multiplier=4, divisor=1, dtype=mx.uint16
        )
        for call in range(24, 28):
            x = s[call].infer_device(x)
        reduced_medium = x
        high_upsample = mx.repeat(mx.repeat(reduced_medium, 2, axis=1), 2, axis=2)
        high_upsample = mx.where(self._split5_mask_device, high_upsample, 0)
        high_upsample = s[28].infer_device(high_upsample)
        backbone_p3_signed = self._device_scale_unsigned(
            backbone_p3, multiplier=1, divisor=2, dtype=mx.int8
        )

        # Split 6: PAN/FPN fusion and the three detector feature scales.
        x = s[29].infer_device(
            mx.concatenate((high_upsample, backbone_p3_signed), axis=-1)
        )
        for call in range(30, 33):
            x = s[call].infer_device(x)
        high_feature = x
        x = s[33].infer_device(high_feature)
        x = s[34].infer_device(mx.concatenate((x, reduced_medium), axis=-1))
        x = self._device_scale_unsigned(
            x, multiplier=2, divisor=1, dtype=mx.uint16
        )
        for call in range(35, 38):
            x = s[call].infer_device(x)
        medium_feature = x
        x = s[38].infer_device(medium_feature)
        low_feature = s[39].infer_device(
            mx.concatenate((x, upsample_source), axis=-1)
        )

        # Splits 7-9: detector stems, box regressors, and score heads.
        x = low_feature
        for call in range(40, 44):
            x = s[call].infer_device(x)
        low = s[44].infer_device(self._device_silu(x))
        low_score_source, low_box_source = mx.split(low, 2, axis=-1)
        low_score_logits = s[46].infer_device(
            self._device_silu(
                low_score_source,
                input_exponent_offset=3,
                output_exponent_offset=3,
            )
        )
        low_boxes = s[45].infer_device(self._device_silu(low_box_source))

        medium = s[48].infer_device(
            self._device_silu(s[47].infer_device(medium_feature))
        )
        medium_score_source, medium_box_source = mx.split(medium, 2, axis=-1)
        medium_score_logits = s[50].infer_device(
            self._device_silu(
                medium_score_source,
                input_exponent_offset=1,
                output_exponent_offset=2,
            )
        )
        medium_boxes = s[49].infer_device(
            self._device_silu(
                medium_box_source,
                input_exponent_offset=1,
                output_exponent_offset=3,
            )
        )

        high_detector_input = self._device_silu(
            s[51].infer_device(high_feature),
            input_exponent_offset=2,
            output_exponent_offset=4,
        )
        high = s[52].infer_device(high_detector_input)
        high_score_source, high_box_source = mx.split(high, 2, axis=-1)
        high_score_logits = s[54].infer_device(
            self._device_silu(
                high_score_source,
                input_exponent_offset=1,
                output_exponent_offset=2,
            )
        )
        high_boxes = s[53].infer_device(
            self._device_silu(
                high_box_source,
                input_exponent_offset=1,
                output_exponent_offset=3,
            )
        )

        high_scores, high_proved = self._device_sigmoid(
            high_score_logits, input_exponent_offset=2
        )
        medium_scores, medium_proved = self._device_sigmoid(
            medium_score_logits, input_exponent_offset=2
        )
        low_scores, low_proved = self._device_sigmoid(
            low_score_logits, input_exponent_offset=3
        )
        device_outputs = {
            "output_0": high_scores,
            "output_1": medium_scores,
            "output_2": low_scores,
            "output_3": high_boxes.astype(mx.float32) / 4.0,
            "output_4": medium_boxes.astype(mx.float32) / 4.0,
            "output_5": low_boxes.astype(mx.float32) / 4.0,
        }
        proved = (high_proved, medium_proved, low_proved)
        mx.eval(*device_outputs.values(), *proved)
        if not all(bool(np.array(flag).item()) for flag in proved):
            raise Cb62AcceleratedError(
                "CB62 detector scores left the ADES-proved sigmoid domain"
            )
        outputs = {
            name: np.array(value).transpose(0, 3, 1, 2)
            for name, value in device_outputs.items()
        }
        for value in outputs.values():
            value.setflags(write=False)
        self.inference_count += 1
        return MappingProxyType(outputs)

    def infer_predictions(self, input_tensor: object) -> NDArray[np.float32]:
        """Execute the exact model and assemble its 13,566x8 ABI."""

        return assemble_cb62_predictions(self.infer_outputs(input_tensor))


class AcceleratedCb62Runtime:
    """Adapt any persistent exact CB62 session to VerkEye's media runtime."""

    def __init__(self, *, session: Cb62MlxSession | Any) -> None:
        self._session = session
        self.identity = MappingProxyType(dict(session.identity))

    def infer(
        self,
        input_tensor: bytes | bytearray | memoryview,
        *,
        run_id: str,
    ) -> AdesInference:
        if not _RUN_ID.fullmatch(run_id):
            raise Cb62AcceleratedError("invalid accelerated run id")
        payload = bytes(input_tensor)
        if len(payload) != _INPUT_BYTES:
            raise Cb62AcceleratedError(
                f"CB62 input tensor must be exactly {_INPUT_BYTES} bytes"
            )
        source = np.frombuffer(payload, dtype=np.uint8).reshape(3, 608, 1088)
        started = time.monotonic_ns()
        outputs = self._session.infer_outputs(source)
        predictions = assemble_cb62_predictions(outputs)
        elapsed_ns = time.monotonic_ns() - started

        locations = {
            "output_5": (8, 0),
            "output_2": (8, 1),
            "output_4": (8, 2),
            "output_1": (8, 3),
            "output_3": (9, 0),
            "output_0": (9, 1),
        }
        records: dict[int, list[AdesTensorRecord]] = {index: [] for index in range(10)}
        hashes: dict[str, str] = {}
        for name, tensor in outputs.items():
            split_index, output_index = locations[name]
            filename = f"accelerated-{name}.bin"
            data = tensor.tobytes(order="C")
            hashes[filename] = hashlib.sha256(data).hexdigest()
            records[split_index].append(
                AdesTensorRecord(
                    split_index=split_index,
                    output_index=output_index,
                    name=name,
                    path=filename,
                    size=len(data),
                    changed_bytes=len(data),
                    main_input_output=True,
                )
            )
        provider = self.identity.get("provider", "accelerated")
        if isinstance(provider, Mapping):
            provider = provider.get("provider", "accelerated")
        execution = AdesExecutionResult(
            cavalry_version=f"host-{provider}-exact",
            cavalry_hash="not-applicable",
            splits=tuple(
                AdesSplitResult(
                    index=index,
                    duration_us=0,
                    tensors=tuple(sorted(records[index], key=lambda item: item.output_index)),
                )
                for index in range(10)
            ),
        )
        return AdesInference(
            run_id=run_id,
            output_directory=Path("."),
            execution=execution,
            tensors=outputs,
            predictions=predictions,
            tensor_sha256=MappingProxyType(dict(sorted(hashes.items()))),
            wall_ms=max(1, elapsed_ns // 1_000_000),
            timings_ns=MappingProxyType(
                {
                    "backend_total": elapsed_ns,
                    "device_resident_exact_graph": elapsed_ns,
                }
            ),
        )


# Retained for callers created before the portable OpenVINO provider existed.
MlxCb62Runtime = AcceleratedCb62Runtime
