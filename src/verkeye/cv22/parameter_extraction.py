"""Normalize exact fast-convolution parameters captured from Ambarella ADES.

The vendor runtime expands the DVI bitstream into an in-memory sparse-kernel
representation.  Capture records contain process-local ``std::vector``
pointers, so this parser validates those pointers only as length witnesses and
then discards them.  The returned representation is deterministic and contains
only model semantics plus cryptographic provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any, Mapping


_ENTRY_SIZE = 96
_POINT_SIZE = 24


class FastconvCaptureError(ValueError):
    """A captured ADES fast-convolution record is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class SparseKernelPoint:
    input_channel: int
    kernel_y: int
    kernel_x: int
    weight: int


@dataclass(frozen=True, slots=True)
class FastconvChannel:
    points: tuple[SparseKernelPoint, ...]
    accumulator_shift: int
    offset: int
    output_saturation_max: int
    final_shift_control: int
    points_sha256: str

    def __post_init__(self) -> None:
        if len(self.points_sha256) != 64:
            raise FastconvCaptureError("points sha256 must have 64 hex digits")
        try:
            bytes.fromhex(self.points_sha256)
        except ValueError as exc:
            raise FastconvCaptureError(
                "points sha256 must have 64 hex digits"
            ) from exc


@dataclass(frozen=True, slots=True)
class FastconvCapture:
    channels: tuple[FastconvChannel, ...]
    entries_sha256: str


def parse_fastconv_capture(
    entries: bytes | bytearray | memoryview,
    point_payloads: Mapping[int, bytes | bytearray | memoryview],
) -> FastconvCapture:
    """Parse one ADES kernel capture and remove all ephemeral addresses.

    ``point_payloads`` is keyed by output-channel index.  The begin/end pointers
    in each 96-byte entry prove the expected point-payload length; their numeric
    values are deliberately absent from the normalized result.
    """

    entry_bytes = bytes(entries)
    if not entry_bytes or len(entry_bytes) % _ENTRY_SIZE:
        raise FastconvCaptureError(
            "fastconv entries must be a non-empty multiple of 96 bytes"
        )
    channel_count = len(entry_bytes) // _ENTRY_SIZE
    unexpected = sorted(set(point_payloads) - set(range(channel_count)))
    if unexpected:
        raise FastconvCaptureError(f"unexpected point payload indices: {unexpected}")

    channels: list[FastconvChannel] = []
    for channel_index in range(channel_count):
        record = entry_bytes[
            channel_index * _ENTRY_SIZE : (channel_index + 1) * _ENTRY_SIZE
        ]
        words = struct.unpack("<12Q", record)
        begin, end, capacity = words[:3]
        if end < begin or capacity < end:
            raise FastconvCaptureError(
                f"channel {channel_index} has an invalid vector pointer span"
            )
        span_size = end - begin
        if span_size % _POINT_SIZE:
            raise FastconvCaptureError(
                f"channel {channel_index} pointer span is not point-aligned"
            )
        if any(words[index] != 0 for index in (3, 4, 5, 6, 11)):
            raise FastconvCaptureError(
                f"channel {channel_index} has nonzero reserved words"
            )
        if channel_index not in point_payloads and span_size:
            raise FastconvCaptureError(
                f"channel {channel_index} is missing point payload"
            )
        payload = bytes(point_payloads.get(channel_index, b""))
        if len(payload) != span_size:
            raise FastconvCaptureError(
                f"channel {channel_index} point payload does not match pointer span"
            )

        points: list[SparseKernelPoint] = []
        for point_offset in range(0, len(payload), _POINT_SIZE):
            input_channel, kernel_y, kernel_x, weight = struct.unpack_from(
                "<IIqq", payload, point_offset
            )
            if kernel_y > 255 or kernel_x < 0 or kernel_x > 255:
                raise FastconvCaptureError(
                    f"channel {channel_index} has an invalid kernel coordinate"
                )
            points.append(
                SparseKernelPoint(
                    input_channel=input_channel,
                    kernel_y=kernel_y,
                    kernel_x=kernel_x,
                    weight=weight,
                )
            )

        channels.append(
            FastconvChannel(
                points=tuple(points),
                accumulator_shift=words[7],
                offset=struct.unpack_from("<q", record, 64)[0],
                output_saturation_max=words[9],
                final_shift_control=words[10],
                points_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )

    return FastconvCapture(
        channels=tuple(channels),
        entries_sha256=hashlib.sha256(entry_bytes).hexdigest(),
    )


def fastconv_capture_document(capture: FastconvCapture) -> dict[str, object]:
    """Return the stable, address-free JSON representation of a capture."""

    return {
        "schema": "verkeye.cv22.fastconv-capture.v1",
        "entries_sha256": capture.entries_sha256,
        "channel_count": len(capture.channels),
        "channels": [
            {
                "channel_index": channel_index,
                "point_count": len(channel.points),
                "points_sha256": channel.points_sha256,
                "accumulator_shift": channel.accumulator_shift,
                "offset": channel.offset,
                "output_saturation_max": channel.output_saturation_max,
                "final_shift_control": channel.final_shift_control,
                "points": [
                    [
                        point.input_channel,
                        point.kernel_y,
                        point.kernel_x,
                        point.weight,
                    ]
                    for point in channel.points
                ],
            }
            for channel_index, channel in enumerate(capture.channels)
        ],
    }


def encode_fastconv_capture(capture: FastconvCapture) -> bytes:
    """Encode normalized semantics without ADES process-local pointers."""

    output = bytearray(b"VKFC\x01\x00\x00\x00")
    output.extend(struct.pack("<I", len(capture.channels)))
    for channel in capture.channels:
        output.extend(
            struct.pack(
                "<QqQQI",
                channel.accumulator_shift,
                channel.offset,
                channel.output_saturation_max,
                channel.final_shift_control,
                len(channel.points),
            )
        )
        for point in channel.points:
            output.extend(
                struct.pack(
                    "<IIqq",
                    point.input_channel,
                    point.kernel_y,
                    point.kernel_x,
                    point.weight,
                )
            )
    return bytes(output)


def decode_fastconv_capture(
    payload: bytes | bytearray | memoryview,
) -> FastconvCapture:
    """Decode one deterministic, address-free ``.vkfc`` package."""

    encoded = bytes(payload)
    if len(encoded) < 12 or encoded[:8] != b"VKFC\x01\x00\x00\x00":
        raise FastconvCaptureError("invalid normalized fastconv header")
    channel_count = struct.unpack_from("<I", encoded, 8)[0]
    if channel_count == 0:
        raise FastconvCaptureError("normalized fastconv has no channels")
    offset = 12
    channels: list[FastconvChannel] = []
    channel_header_size = struct.calcsize("<QqQQI")
    for channel_index in range(channel_count):
        if offset > len(encoded) or channel_header_size > len(encoded) - offset:
            raise FastconvCaptureError(
                f"normalized fastconv channel {channel_index} is truncated"
            )
        (
            accumulator_shift,
            channel_offset,
            output_saturation_max,
            final_shift_control,
            point_count,
        ) = struct.unpack_from("<QqQQI", encoded, offset)
        offset += channel_header_size
        point_bytes = point_count * _POINT_SIZE
        if offset > len(encoded) or point_bytes > len(encoded) - offset:
            raise FastconvCaptureError(
                f"normalized fastconv channel {channel_index} points are truncated"
            )
        raw_points = encoded[offset : offset + point_bytes]
        points: list[SparseKernelPoint] = []
        for point_offset in range(0, point_bytes, _POINT_SIZE):
            input_channel, kernel_y, kernel_x, weight = struct.unpack_from(
                "<IIqq", raw_points, point_offset
            )
            if kernel_y > 255 or kernel_x < 0 or kernel_x > 255:
                raise FastconvCaptureError(
                    f"normalized channel {channel_index} has invalid kernel coordinate"
                )
            points.append(
                SparseKernelPoint(input_channel, kernel_y, kernel_x, weight)
            )
        channels.append(
            FastconvChannel(
                points=tuple(points),
                accumulator_shift=accumulator_shift,
                offset=channel_offset,
                output_saturation_max=output_saturation_max,
                final_shift_control=final_shift_control,
                points_sha256=hashlib.sha256(raw_points).hexdigest(),
            )
        )
        offset += point_bytes
    if offset != len(encoded):
        raise FastconvCaptureError("normalized fastconv has trailing data")
    return FastconvCapture(
        channels=tuple(channels),
        entries_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def catalog_fastconv_captures(
    capture_root: str | Path, graph_document: Mapping[str, Any]
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Bind sequential ADES captures to decoded split/operator identities.

    ADES executes the ten split graphs serially, and the preload hook numbers
    only ``fastconvolution`` calls.  Therefore, iterating decoded splits and
    operators in their recorded order provides the exact call-to-node mapping.
    The function verifies output depth and refuses missing or extra captures.
    """

    root = Path(capture_root)
    nodes: list[tuple[int, int, Mapping[str, Any]]] = []
    for split in graph_document.get("splits", []):
        split_index = int(split["split_index"])
        for operator in split.get("operators", []):
            if operator.get("type_name") == "fastconvolution_operator_t":
                nodes.append((split_index, int(operator["operator_id"]), operator))

    captures: list[dict[str, object]] = []
    normalized_members: dict[str, bytes] = {}
    for call_index, (split_index, operator_id, operator) in enumerate(nodes):
        entries_path = root / f"fastconv-{call_index}-entries.bin"
        if not entries_path.is_file():
            raise FastconvCaptureError(
                f"fastconv call {call_index} is missing entries capture"
            )
        entry_bytes = entries_path.read_bytes()
        channel_count = len(entry_bytes) // _ENTRY_SIZE if entry_bytes else 0
        payloads: dict[int, bytes] = {}
        for channel_index in range(channel_count):
            payload_path = (
                root
                / f"fastconv-{call_index}-entry-{channel_index}-offset-0.bin"
            )
            if payload_path.is_file():
                payloads[channel_index] = payload_path.read_bytes()
        capture = parse_fastconv_capture(entry_bytes, payloads)

        config = operator.get("config", {})
        depth_key = "voperator.fastconvolution_operator.output_depth_minus_one"
        if depth_key not in config:
            raise FastconvCaptureError(
                f"split {split_index} op {operator_id} lacks output-depth evidence"
            )
        expected_channels = int(config[depth_key]) + 1
        if len(capture.channels) != expected_channels:
            raise FastconvCaptureError(
                f"split {split_index} op {operator_id} channel count mismatch: "
                f"{len(capture.channels)} != {expected_channels}"
            )

        member_name = (
            f"fastconv-call-{call_index:03d}-split-{split_index:02d}-"
            f"op-{operator_id:03d}.vkfc"
        )
        normalized = encode_fastconv_capture(capture)
        normalized_members[member_name] = normalized
        captures.append(
            {
                "call_index": call_index,
                "split_index": split_index,
                "operator_id": operator_id,
                "source_entries_member": entries_path.name,
                "source_entries_sha256": capture.entries_sha256,
                "channel_count": len(capture.channels),
                "point_count": sum(len(channel.points) for channel in capture.channels),
                "normalized_member": member_name,
                "normalized_size": len(normalized),
                "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
                "channels": [
                    {
                        "channel_index": channel_index,
                        "point_count": len(channel.points),
                        "points_sha256": channel.points_sha256,
                        "accumulator_shift": channel.accumulator_shift,
                        "offset": channel.offset,
                        "output_saturation_max": channel.output_saturation_max,
                        "final_shift_control": channel.final_shift_control,
                    }
                    for channel_index, channel in enumerate(capture.channels)
                ],
            }
        )

    extra_entries = sorted(root.glob(f"fastconv-{len(nodes)}-entries.bin"))
    if extra_entries:
        raise FastconvCaptureError(
            f"capture root contains unbound call {len(nodes)}"
        )
    catalog: dict[str, object] = {
        "schema": "verkeye.cv22.fastconv-catalog.v1",
        "model_sha256": graph_document.get("model_sha256"),
        "mapping_rule": (
            "ascending split_index and decoded operator order, filtered to "
            "fastconvolution_operator_t"
        ),
        "capture_count": len(captures),
        "channel_count": sum(int(item["channel_count"]) for item in captures),
        "point_count": sum(int(item["point_count"]) for item in captures),
        "captures": captures,
    }
    return catalog, normalized_members
