from __future__ import annotations

import hashlib
import struct

import pytest

from verkeye.cv22.parameter_extraction import (
    decode_fastconv_capture,
    FastconvCaptureError,
    catalog_fastconv_captures,
    encode_fastconv_capture,
    fastconv_capture_document,
    parse_fastconv_capture,
)


def _entry(*, begin: int = 0x1000, point_count: int = 2) -> bytes:
    end = begin + point_count * 24
    return struct.pack(
        "<12Q",
        begin,
        end,
        end + 24,
        0,
        0,
        0,
        0,
        8,
        7,
        255,
        2,
        0,
    )


def _points() -> bytes:
    return b"".join(
        (
            struct.pack("<IIqq", 0, 1, 2, -5),
            struct.pack("<IIqq", 2, 0, 0, 35),
        )
    )


def test_parse_fastconv_capture_normalizes_pointers_and_preserves_hashes() -> None:
    entries = _entry()
    points = _points()

    capture = parse_fastconv_capture(entries, {0: points})
    document = fastconv_capture_document(capture)

    assert capture.entries_sha256 == hashlib.sha256(entries).hexdigest()
    assert capture.channels[0].points_sha256 == hashlib.sha256(points).hexdigest()
    assert capture.channels[0].accumulator_shift == 8
    assert capture.channels[0].offset == 7
    assert capture.channels[0].output_saturation_max == 255
    assert capture.channels[0].final_shift_control == 2
    assert [(point.input_channel, point.kernel_y, point.kernel_x, point.weight) for point in capture.channels[0].points] == [
        (0, 1, 2, -5),
        (2, 0, 0, 35),
    ]
    assert "4096" not in str(document)
    assert document["channels"][0]["point_count"] == 2
    assert hashlib.sha256(encode_fastconv_capture(capture)).hexdigest() == (
        "4c6ca6adffa30f75a10ca37568e1e1322dd8eae11cd4b6a92bbdfc609daa71f0"
    )


def test_normalized_fastconv_package_round_trips_semantics() -> None:
    capture = parse_fastconv_capture(_entry(), {0: _points()})

    decoded = decode_fastconv_capture(encode_fastconv_capture(capture))

    assert decoded.channels == capture.channels


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"not-vkfc", "header"),
        (b"VKFC\x01\x00\x00\x00\x01\x00\x00\x00", "truncated"),
        (encode_fastconv_capture(parse_fastconv_capture(_entry(), {0: _points()})) + b"x", "trailing"),
    ],
)
def test_normalized_fastconv_package_rejects_malformed_data(
    payload: bytes, message: str
) -> None:
    with pytest.raises(FastconvCaptureError, match=message):
        decode_fastconv_capture(payload)


def test_parse_fastconv_capture_rejects_missing_point_payload() -> None:
    with pytest.raises(FastconvCaptureError, match="missing point payload"):
        parse_fastconv_capture(_entry(), {})


def test_parse_fastconv_capture_accepts_omitted_empty_point_payload() -> None:
    capture = parse_fastconv_capture(_entry(point_count=0), {})

    assert capture.channels[0].points == ()
    assert capture.channels[0].points_sha256 == hashlib.sha256(b"").hexdigest()


def test_parse_fastconv_capture_rejects_pointer_span_mismatch() -> None:
    with pytest.raises(FastconvCaptureError, match="pointer span"):
        parse_fastconv_capture(_entry(point_count=3), {0: _points()})


def test_parse_fastconv_capture_rejects_nonzero_reserved_words() -> None:
    words = list(struct.unpack("<12Q", _entry()))
    words[4] = 1

    with pytest.raises(FastconvCaptureError, match="reserved words"):
        parse_fastconv_capture(struct.pack("<12Q", *words), {0: _points()})


def test_parse_fastconv_capture_rejects_out_of_kernel_coordinate() -> None:
    invalid = struct.pack("<IIqq", 0, 256, 0, 1) + _points()[24:]

    with pytest.raises(FastconvCaptureError, match="kernel coordinate"):
        parse_fastconv_capture(_entry(), {0: invalid})


def test_catalog_binds_capture_sequence_to_graph_identity(tmp_path) -> None:
    (tmp_path / "fastconv-0-entries.bin").write_bytes(_entry())
    (tmp_path / "fastconv-0-entry-0-offset-0.bin").write_bytes(_points())
    graph = {
        "model_sha256": "ab" * 32,
        "splits": [
            {
                "split_index": 4,
                "operators": [
                    {
                        "operator_id": 17,
                        "type_name": "fastconvolution_operator_t",
                        "config": {
                            "voperator.fastconvolution_operator.output_depth_minus_one": 0
                        },
                    }
                ],
            }
        ],
    }

    catalog, normalized = catalog_fastconv_captures(tmp_path, graph)

    assert catalog["schema"] == "verkeye.cv22.fastconv-catalog.v1"
    assert catalog["capture_count"] == 1
    assert catalog["captures"][0]["split_index"] == 4
    assert catalog["captures"][0]["operator_id"] == 17
    assert catalog["captures"][0]["point_count"] == 2
    assert catalog["captures"][0]["normalized_sha256"] == hashlib.sha256(
        normalized["fastconv-call-000-split-04-op-017.vkfc"]
    ).hexdigest()


def test_catalog_rejects_missing_capture_call(tmp_path) -> None:
    graph = {
        "model_sha256": "ab" * 32,
        "splits": [
            {
                "split_index": 0,
                "operators": [
                    {
                        "operator_id": 0,
                        "type_name": "fastconvolution_operator_t",
                        "config": {
                            "voperator.fastconvolution_operator.output_depth_minus_one": 0
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(FastconvCaptureError, match="missing entries"):
        catalog_fastconv_captures(tmp_path, graph)
