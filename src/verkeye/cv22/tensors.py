"""Lossless tensor descriptors at the front of each CV22 split payload.

The recovered artifact provides a repeated 0x480-byte record for every split
input and output.  This module names only the fields whose interpretation is
corroborated by the exact model.  All remaining words stay available through
the descriptor digest and the two raw word groups.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from .reader import BinaryReader, BoundsError
from .schema import ArtifactSpan, CV22Container


TENSOR_DESCRIPTOR_SIZE = 0x480
TENSOR_NAME_OFFSET = 0x80
TENSOR_NAME_MIRROR_OFFSET = 0x280
TENSOR_NAME_SIZE = 0x100
TENSOR_HEADER_WORD_COUNT = 8
TENSOR_AUX_WORD_OFFSET = 0x60
TENSOR_AUX_WORD_COUNT = 8
TENSOR_PITCH_OFFSET = 0x10
TENSOR_PITCH_BYTE_OFFSET = 0x14
TENSOR_MEMORY_FORMAT_OFFSET = 0x18
TENSOR_DATA_FORMAT_OFFSET = 0x74
TENSOR_BUFFER_EXTENT_OFFSET = 0x78


class TensorFormatError(ValueError):
    """Tensor metadata contradicts the repeated exact-artifact layout."""


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProvenancedValue(Generic[T]):
    """One interpreted value with its exact source and evidence boundary."""

    value: T
    offset: int
    evidence: str
    confidence: Literal["exact", "corroborated", "inferred", "unknown"]


@dataclass(frozen=True, slots=True)
class TensorMemoryFormat:
    """Memory fields proven by exact source and matching AArch64 loads."""

    pitch: ProvenancedValue[int]
    pitch_byte_offset: ProvenancedValue[int]
    pitch_bit_size: ProvenancedValue[int]
    dram_format: ProvenancedValue[int]
    bitvector: ProvenancedValue[int]
    packed_word: ProvenancedValue[int]


@dataclass(frozen=True, slots=True)
class TensorDataFormat:
    """NNCtrl public data-format fields without an invented numeric formula."""

    sign: ProvenancedValue[int]
    element_size_code: ProvenancedValue[int]
    exponent_offset: ProvenancedValue[int]
    exponent_bits: ProvenancedValue[int]

    @property
    def element_bits(self) -> int:
        return 8 << self.element_size_code.value

    @property
    def storage_dtype(self) -> str:
        prefix = "int" if self.sign.value else "uint"
        return f"{prefix}{self.element_bits}"

    @property
    def semantic_encoding(self) -> None:
        """Remain unset until the proprietary exponent encoding is proven."""

        return None


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    """One split input or output, preserving raw provenance."""

    split_index: int
    ordinal: int
    role: Literal["input", "output"]
    span: ArtifactSpan
    name: ProvenancedValue[str]
    dimensions: tuple[
        ProvenancedValue[int],
        ProvenancedValue[int],
        ProvenancedValue[int],
        ProvenancedValue[int],
    ]
    memory: TensorMemoryFormat
    data_format: TensorDataFormat
    header_words: tuple[ProvenancedValue[int], ...]
    auxiliary_words: tuple[ProvenancedValue[int], ...]
    raw_sha256: str

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(item.value for item in self.dimensions)  # type: ignore[return-value]

    @property
    def format_flags(self) -> ProvenancedValue[int]:
        """Compatibility alias for the proven packed memory-format word."""

        return self.memory.packed_word

    @property
    def layout_code(self) -> ProvenancedValue[int]:
        """Compatibility alias for the four serialized data-format bytes."""

        return self.auxiliary_words[5]

    @property
    def buffer_extent(self) -> ProvenancedValue[int]:
        return self.auxiliary_words[6]

    @property
    def connection_signature(self) -> tuple[object, ...]:
        """Fields that remain stable across producer/consumer split records."""

        return (
            self.shape,
            self.memory.pitch.value,
            self.memory.pitch_bit_size.value,
            self.memory.dram_format.value,
            self.memory.bitvector.value,
            self.data_format.sign.value,
            self.data_format.element_size_code.value,
            self.data_format.exponent_offset.value,
            self.data_format.exponent_bits.value,
            self.buffer_extent.value,
        )


@dataclass(frozen=True, slots=True)
class SplitTensorMap:
    """Tensor prefix and remaining compiled bytes for one split payload."""

    split_index: int
    payload_span: ArtifactSpan
    descriptor_span: ArtifactSpan
    compiled_graph_span: ArtifactSpan
    inputs: tuple[TensorDescriptor, ...]
    outputs: tuple[TensorDescriptor, ...]


def _fixed_name(reader: BinaryReader, offset: int) -> str:
    field = reader.read_bytes(offset, TENSOR_NAME_SIZE)
    if b"\x00" not in field:
        raise TensorFormatError(
            f"tensor name at offset {offset} has no NUL terminator"
        )
    encoded = field.partition(b"\x00")[0]
    if not encoded:
        raise TensorFormatError(f"tensor name at offset {offset} is empty")
    try:
        return encoded.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise TensorFormatError(
            f"tensor name at offset {offset} is not ASCII"
        ) from exc


def _word_group(
    reader: BinaryReader,
    offset: int,
    count: int,
    *,
    evidence: str,
    confidence: Literal["exact", "corroborated", "inferred", "unknown"],
) -> tuple[ProvenancedValue[int], ...]:
    return tuple(
        ProvenancedValue(
            value=reader.read_u32le(offset + (index * 4)),
            offset=offset + (index * 4),
            evidence=evidence,
            confidence=confidence,
        )
        for index in range(count)
    )


def _parse_descriptor(
    reader: BinaryReader,
    *,
    split_index: int,
    ordinal: int,
    role: Literal["input", "output"],
    offset: int,
) -> TensorDescriptor:
    raw = reader.read_bytes(offset, TENSOR_DESCRIPTOR_SIZE)
    primary_name = _fixed_name(reader, offset + TENSOR_NAME_OFFSET)
    mirror_name = _fixed_name(reader, offset + TENSOR_NAME_MIRROR_OFFSET)
    if primary_name != mirror_name:
        raise TensorFormatError(
            f"tensor name mirror mismatch at descriptor offset {offset}: "
            f"{primary_name!r} != {mirror_name!r}"
        )

    header_words = _word_group(
        reader,
        offset,
        TENSOR_HEADER_WORD_COUNT,
        evidence="exact serialized tensor header word",
        confidence="exact",
    )
    auxiliary_words = _word_group(
        reader,
        offset + TENSOR_AUX_WORD_OFFSET,
        TENSOR_AUX_WORD_COUNT,
        evidence="exact serialized auxiliary tensor word",
        confidence="exact",
    )
    dimensions = tuple(
        ProvenancedValue(
            value=header_words[index].value,
            offset=header_words[index].offset,
            evidence=(
                "repeated four-word tensor dimensions corroborated by model/layer "
                "names and producer-consumer pairs"
            ),
            confidence="corroborated",
        )
        for index in range(4)
    )
    if any(item.value <= 0 for item in dimensions):
        raise TensorFormatError(
            f"tensor descriptor at offset {offset} has a non-positive dimension"
        )
    if auxiliary_words[6].value <= 0:
        raise TensorFormatError(
            f"tensor descriptor at offset {offset} has no buffer extent"
        )

    abi_evidence = (
        "exact nnctrl 0.3.0 source assignment and matching recovered AArch64 "
        "gen_net_parent_port_size field load"
    )
    pitch = reader.read_u32le(offset + TENSOR_PITCH_OFFSET)
    pitch_byte_offset = reader.read_u32le(offset + TENSOR_PITCH_BYTE_OFFSET)
    packed_memory = reader.read_u64le(offset + TENSOR_MEMORY_FORMAT_OFFSET)
    sign = reader.read_u8(offset + TENSOR_DATA_FORMAT_OFFSET)
    element_size_code = reader.read_u8(offset + TENSOR_DATA_FORMAT_OFFSET + 1)
    exponent_offset = int.from_bytes(
        reader.read_bytes(offset + TENSOR_DATA_FORMAT_OFFSET + 2, 1),
        byteorder="little",
        signed=True,
    )
    exponent_bits = reader.read_u8(offset + TENSOR_DATA_FORMAT_OFFSET + 3)
    if pitch <= 0:
        raise TensorFormatError(f"tensor descriptor at offset {offset} has no pitch")
    if sign not in (0, 1):
        raise TensorFormatError(
            f"tensor descriptor at offset {offset} has invalid sign {sign}"
        )
    if element_size_code > 3:
        raise TensorFormatError(
            f"tensor descriptor at offset {offset} has invalid element size code "
            f"{element_size_code}"
        )

    def exact(value: int, relative_offset: int) -> ProvenancedValue[int]:
        return ProvenancedValue(
            value=value,
            offset=offset + relative_offset,
            evidence=abi_evidence,
            confidence="exact",
        )

    memory = TensorMemoryFormat(
        pitch=exact(pitch, TENSOR_PITCH_OFFSET),
        pitch_byte_offset=exact(pitch_byte_offset, TENSOR_PITCH_BYTE_OFFSET),
        pitch_bit_size=exact(packed_memory & 0x3F, TENSOR_MEMORY_FORMAT_OFFSET),
        dram_format=exact((packed_memory >> 6) & 0xF, TENSOR_MEMORY_FORMAT_OFFSET),
        bitvector=exact((packed_memory >> 10) & 0x1, TENSOR_MEMORY_FORMAT_OFFSET),
        packed_word=exact(packed_memory & 0xFFFFFFFF, TENSOR_MEMORY_FORMAT_OFFSET),
    )
    data_format = TensorDataFormat(
        sign=exact(sign, TENSOR_DATA_FORMAT_OFFSET),
        element_size_code=exact(element_size_code, TENSOR_DATA_FORMAT_OFFSET + 1),
        exponent_offset=exact(exponent_offset, TENSOR_DATA_FORMAT_OFFSET + 2),
        exponent_bits=exact(exponent_bits, TENSOR_DATA_FORMAT_OFFSET + 3),
    )

    return TensorDescriptor(
        split_index=split_index,
        ordinal=ordinal,
        role=role,
        span=ArtifactSpan(offset, TENSOR_DESCRIPTOR_SIZE),
        name=ProvenancedValue(
            value=primary_name,
            offset=offset + TENSOR_NAME_OFFSET,
            evidence="matching fixed-width primary and mirror names",
            confidence="exact",
        ),
        dimensions=dimensions,  # type: ignore[arg-type]
        memory=memory,
        data_format=data_format,
        header_words=header_words,
        auxiliary_words=auxiliary_words,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_tensor_map(data: Any, container: CV22Container) -> tuple[SplitTensorMap, ...]:
    """Parse the evidenced input/output prefix of every split payload."""

    reader = BinaryReader(data)
    if reader.size != container.file_size:
        raise TensorFormatError(
            f"container size {container.file_size} disagrees with data size {reader.size}"
        )

    results: list[SplitTensorMap] = []
    try:
        for record in container.records:
            input_count = record.raw_words[4]
            output_count = record.raw_words[5]
            compiled_graph_size = record.raw_words[2]
            descriptor_count = input_count + output_count
            descriptor_size = descriptor_count * TENSOR_DESCRIPTOR_SIZE
            if descriptor_size + compiled_graph_size != record.payload.size:
                raise TensorFormatError(
                    f"split {record.index} payload accounting mismatch: "
                    f"{descriptor_count} descriptors ({descriptor_size} bytes) + "
                    f"compiled graph ({compiled_graph_size} bytes) != payload "
                    f"({record.payload.size} bytes)"
                )

            descriptor_span = ArtifactSpan(record.payload.offset, descriptor_size)
            compiled_graph_span = ArtifactSpan(
                descriptor_span.end,
                compiled_graph_size,
            )
            inputs: list[TensorDescriptor] = []
            outputs: list[TensorDescriptor] = []
            for descriptor_index in range(descriptor_count):
                role: Literal["input", "output"] = (
                    "input" if descriptor_index < input_count else "output"
                )
                role_ordinal = (
                    descriptor_index
                    if role == "input"
                    else descriptor_index - input_count
                )
                descriptor = _parse_descriptor(
                    reader,
                    split_index=record.index,
                    ordinal=role_ordinal,
                    role=role,
                    offset=record.payload.offset
                    + (descriptor_index * TENSOR_DESCRIPTOR_SIZE),
                )
                (inputs if role == "input" else outputs).append(descriptor)

            results.append(
                SplitTensorMap(
                    split_index=record.index,
                    payload_span=record.payload,
                    descriptor_span=descriptor_span,
                    compiled_graph_span=compiled_graph_span,
                    inputs=tuple(inputs),
                    outputs=tuple(outputs),
                )
            )
    except BoundsError as exc:
        raise TensorFormatError(f"truncated tensor metadata: {exc}") from exc
    return tuple(results)
