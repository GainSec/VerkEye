"""Evidence-bounded Ambarella Cavalry request and memory primitives."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from ..cv22.program import ProgramPackage

from .trace import TraceRecorder


class CavalryError(ValueError):
    """A request cannot be handled without violating the fidelity boundary."""


@dataclass(frozen=True, slots=True)
class DecodedIoctl:
    raw: int
    direction: str
    type_number: int
    type_character: str
    number: int
    argument_size: int


def decode_ioctl(request: int) -> DecodedIoctl:
    """Decode the generic Linux _IOC bit layout used by recovered requests."""

    if not isinstance(request, int) or isinstance(request, bool):
        raise CavalryError("ioctl request must be an integer")
    if request < 0 or request > 0xFFFF_FFFF:
        raise CavalryError("ioctl request must fit in 32 bits")
    direction_number = (request >> 30) & 0x3
    direction = {
        0: "none",
        1: "write",
        2: "read",
        3: "read_write",
    }[direction_number]
    type_number = (request >> 8) & 0xFF
    return DecodedIoctl(
        raw=request,
        direction=direction,
        type_number=type_number,
        type_character=(chr(type_number) if 0x20 <= type_number <= 0x7E else ""),
        number=request & 0xFF,
        argument_size=(request >> 16) & 0x3FFF,
    )


class CavalryRequest(IntEnum):
    QUERY_BUF = 0xC0084300
    START_VP = 0xC0084301
    STOP_VP = 0xC0084302
    RUN_DAGS = 0xC0084303
    EARLY_QUIT = 0xC0084306
    ALLOC_MEM = 0xC0084307
    FREE_MEM = 0xC0084308
    SYNC_CACHE_MEM = 0xC0084309
    SET_HOTLINK_SLOT_CFG = 0xC008430C
    GET_HOTLINK_SLOT_CFG = 0xC008430D
    QUERY_VP_CORE_DUMP = 0xC0084314
    ALLOC_MEMFD = 0xC0084340
    SYNC_CACHE_MEMFD = 0xC0084341
    RUN_DAGS_MEMFD = 0xC0084342
    GET_AUDIO_CLK = 0x80084380
    SET_CAVALRY_CLK = 0x40084381
    GET_CAVALRY_CLK = 0xC0084382

    @classmethod
    def from_number(cls, request: int) -> "CavalryRequest":
        try:
            return cls(request)
        except ValueError as exc:
            raise CavalryError(f"unknown Cavalry ioctl 0x{request:08x}") from exc


@dataclass(frozen=True, slots=True)
class CavalryRequestSpec:
    request: int
    name: str
    artifact_sha256: str
    callsite_virtual_addresses: tuple[int, ...]
    evidence: str = "recovered AArch64 callsite and adjacent API string"


_NNCTRL_SHA256 = "647b38c0ef77558bc58bc2b32e1003bd8c4c781eaf25e195268297edc44f4c3d"
_MEM_SHA256 = "330836af26ab536d02430ea8ac338fa2975688d290c3c12b0304d4554aac0ed4"
_LOAD_SHA256 = "d19a8d03f7cbc8e371bef1f5c653de400f117cd82d15a6b4e8b9f92d7f0c1ad6"
_CORE_DUMP_SHA256 = (
    "066d91ee87d0c4b2dc65e2302946c249981fa21b736534e127b80d8704eef360"
)


def _spec(
    request: CavalryRequest,
    name: str,
    artifact_sha256: str,
    *callsites: int,
) -> CavalryRequestSpec:
    return CavalryRequestSpec(
        request=int(request),
        name=name,
        artifact_sha256=artifact_sha256,
        callsite_virtual_addresses=tuple(callsites),
    )


CAVALRY_REQUESTS: dict[int, CavalryRequestSpec] = {
    int(CavalryRequest.QUERY_BUF): _spec(
        CavalryRequest.QUERY_BUF,
        "CAVALRY_QUERY_BUF",
        _LOAD_SHA256,
        0x1B04,
        0x1C48,
    ),
    int(CavalryRequest.START_VP): _spec(
        CavalryRequest.START_VP, "CAVALRY_START_VP", _LOAD_SHA256, 0x28FC
    ),
    int(CavalryRequest.STOP_VP): _spec(
        CavalryRequest.STOP_VP,
        "CAVALRY_STOP_VP",
        _LOAD_SHA256,
        0x2764,
        0x27E8,
    ),
    int(CavalryRequest.RUN_DAGS): _spec(
        CavalryRequest.RUN_DAGS,
        "CAVALRY_RUN_DAGS",
        _NNCTRL_SHA256,
        0xE7EC,
        0xEB98,
        0xEEB4,
        0xF2DC,
    ),
    int(CavalryRequest.EARLY_QUIT): _spec(
        CavalryRequest.EARLY_QUIT,
        "CAVALRY_EARLY_QUIT",
        _NNCTRL_SHA256,
        0x2518,
    ),
    int(CavalryRequest.ALLOC_MEM): _spec(
        CavalryRequest.ALLOC_MEM, "CAVALRY_ALLOC_MEM", _MEM_SHA256, 0xE4C
    ),
    int(CavalryRequest.FREE_MEM): _spec(
        CavalryRequest.FREE_MEM, "CAVALRY_FREE_MEM", _MEM_SHA256, 0xEE8, 0x12E0
    ),
    int(CavalryRequest.SYNC_CACHE_MEM): _spec(
        CavalryRequest.SYNC_CACHE_MEM,
        "CAVALRY_SYNC_CACHE_MEM",
        _MEM_SHA256,
        0x154C,
    ),
    int(CavalryRequest.SET_HOTLINK_SLOT_CFG): _spec(
        CavalryRequest.SET_HOTLINK_SLOT_CFG,
        "CAVALRY_SET_HOTLINK_SLOT_CFG",
        _LOAD_SHA256,
        0x2370,
    ),
    int(CavalryRequest.GET_HOTLINK_SLOT_CFG): _spec(
        CavalryRequest.GET_HOTLINK_SLOT_CFG,
        "CAVALRY_GET_HOTLINK_SLOT_CFG",
        _LOAD_SHA256,
        0x216C,
    ),
    int(CavalryRequest.QUERY_VP_CORE_DUMP): _spec(
        CavalryRequest.QUERY_VP_CORE_DUMP,
        "CAVALRY_QUERY_VP_CORE_DUMP",
        _CORE_DUMP_SHA256,
        0x1D28,
    ),
    int(CavalryRequest.ALLOC_MEMFD): _spec(
        CavalryRequest.ALLOC_MEMFD, "CAVALRY_ALLOC_MEMFD", _MEM_SHA256, 0x10B0
    ),
    int(CavalryRequest.SYNC_CACHE_MEMFD): _spec(
        CavalryRequest.SYNC_CACHE_MEMFD,
        "CAVALRY_SYNC_CACHE_MEMFD",
        _MEM_SHA256,
        0x16F8,
    ),
    int(CavalryRequest.RUN_DAGS_MEMFD): _spec(
        CavalryRequest.RUN_DAGS_MEMFD,
        "CAVALRY_RUN_DAGS_MEMFD",
        _NNCTRL_SHA256,
        0x1166C,
        0x11A18,
        0x11D38,
        0x12170,
    ),
    int(CavalryRequest.GET_AUDIO_CLK): _spec(
        CavalryRequest.GET_AUDIO_CLK,
        "CAVALRY_GET_AUDIO_CLK",
        _NNCTRL_SHA256,
        0x12970,
    ),
    int(CavalryRequest.SET_CAVALRY_CLK): _spec(
        CavalryRequest.SET_CAVALRY_CLK,
        "CAVALRY_SET_CAVALRY_CLK",
        _LOAD_SHA256,
        0x258C,
    ),
    int(CavalryRequest.GET_CAVALRY_CLK): _spec(
        CavalryRequest.GET_CAVALRY_CLK,
        "CAVALRY_GET_CAVALRY_CLK",
        _LOAD_SHA256,
        0x25C8,
    ),
}


RUN_DAGS_HEADER_SIZE = 0x3C
RUN_DAGS_DESCRIPTOR_SIZE = 0xB2C
RUN_DAGS_MAX_DAGS = 0x100
RUN_DAGS_TABLE_COUNT_OFFSET = 0x24
RUN_DAGS_TABLE_OFFSET = 0x2C
RUN_DAGS_TABLE_ENTRY_SIZE = 0x10
RUN_DAGS_TABLE_MAX_ENTRIES = 0x80
RUN_DAGS_TABLE2_COUNT_OFFSET = 0x28
RUN_DAGS_TABLE2_OFFSET = 0x82C
RUN_DAGS_TABLE2_ENTRY_SIZE = 0x0C
RUN_DAGS_TABLE2_MAX_ENTRIES = 0x40
RUN_DAGS_COMMAND_HEADER_SIZE = 0x30
RUN_DAGS_RESPONSE_MAGIC = 0x8000_0001
RUN_DAGS_RESPONSE_SIZE = 0x14
RUN_DAGS_FIRMWARE_ERROR_RESULT = -53
RUN_DAGS_SEMANTIC_SOURCE_SHA256 = (
    "b6b4197498d81a8c5dfa393a0516ebb179c9be889e72d58245c7038ca74bca6e"
)
RUN_DAGS_SEMANTIC_SOURCE_COMMIT = "aa3856db0028691b688b122a11279d87efd25f65"


@dataclass(frozen=True, slots=True)
class RunDagsHeader:
    word_0x00: int
    word_0x04: int
    word_0x08: int
    word_0x0c: int
    dag_count: int
    reserved_words: tuple[int, ...]

    def to_document(self) -> dict[str, object]:
        return {
            "word_0x00": self.word_0x00,
            "word_0x04": self.word_0x04,
            "word_0x08": self.word_0x08,
            "word_0x0c": self.word_0x0c,
            "dag_count": self.dag_count,
            "reserved_words": self.reserved_words,
        }


@dataclass(frozen=True, slots=True)
class RunDagsTableEntry:
    index: int
    source_offset: int
    words: tuple[int, int, int, int]

    @property
    def port_dram_addr(self) -> int:
        return self.words[0]

    @property
    def port_boffset_in_dag(self) -> int:
        return self.words[1]

    @property
    def port_dram_size(self) -> int:
        return self.words[2]

    @property
    def port_daddr_increment(self) -> int:
        return struct.unpack("<i", struct.pack("<I", self.words[3]))[0]

    def to_document(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_offset": self.source_offset,
            "port_dram_addr": self.port_dram_addr,
            "port_boffset_in_dag": self.port_boffset_in_dag,
            "port_dram_size": self.port_dram_size,
            "port_daddr_increment": self.port_daddr_increment,
            "raw_words": self.words,
        }


@dataclass(frozen=True, slots=True)
class RunDagsSecondaryTableEntry:
    index: int
    source_offset: int
    words: tuple[int, int, int]

    @property
    def poke_val(self) -> int:
        return self.words[0]

    @property
    def poke_vaddr(self) -> int:
        return self.words[1]

    @property
    def poke_bsize(self) -> int:
        return self.words[2]

    def to_document(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_offset": self.source_offset,
            "poke_val": self.poke_val,
            "poke_vaddr": self.poke_vaddr,
            "poke_bsize": self.poke_bsize,
            "raw_words": self.words,
        }


@dataclass(frozen=True, slots=True)
class RunDagsDescriptor:
    index: int
    source_offset: int
    control_halfword: int
    words_0x00_through_0x20: tuple[int, ...]
    table_count: int
    table_entries: tuple[RunDagsTableEntry, ...]
    table2_count: int
    table2_entries: tuple[RunDagsSecondaryTableEntry, ...]
    raw_bytes: bytes

    @property
    def control_word(self) -> int:
        return self.words_0x00_through_0x20[0]

    @property
    def dvi_mode(self) -> bool:
        return bool(self.control_word & 0x1)

    @property
    def use_ping_pong_vmem(self) -> bool:
        return bool(self.control_word & 0x2)

    @property
    def reserved_control_bits(self) -> int:
        return (self.control_word >> 2) & 0x3FFF

    @property
    def dag_loop_count(self) -> int:
        return self.control_halfword

    @property
    def dvi_dram_addr(self) -> int:
        return self.words_0x00_through_0x20[1]

    @property
    def dvi_img_vaddr(self) -> int:
        return self.words_0x00_through_0x20[2]

    @property
    def dvi_img_size(self) -> int:
        return self.words_0x00_through_0x20[3]

    @property
    def dvi_dag_vaddr(self) -> int:
        return self.words_0x00_through_0x20[4]

    @property
    def unknown_words_0x14_through_0x20(self) -> tuple[int, int, int, int]:
        unknown = self.words_0x00_through_0x20[5:]
        return (unknown[0], unknown[1], unknown[2], unknown[3])

    @property
    def port_count(self) -> int:
        return self.table_count

    @property
    def port_entries(self) -> tuple[RunDagsTableEntry, ...]:
        return self.table_entries

    @property
    def poke_count(self) -> int:
        return self.table2_count

    @property
    def poke_entries(self) -> tuple[RunDagsSecondaryTableEntry, ...]:
        return self.table2_entries

    def to_document(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_offset": self.source_offset,
            "control": {
                "raw_word": self.control_word,
                "dvi_mode": self.dvi_mode,
                "use_ping_pong_vmem": self.use_ping_pong_vmem,
                "reserved_bits_2_through_15": self.reserved_control_bits,
                "dag_loop_count": self.dag_loop_count,
            },
            "dvi_dram_addr": self.dvi_dram_addr,
            "dvi_img_vaddr": self.dvi_img_vaddr,
            "dvi_img_size": self.dvi_img_size,
            "dvi_dag_vaddr": self.dvi_dag_vaddr,
            "unknown_words_0x14_through_0x20": self.unknown_words_0x14_through_0x20,
            "port_count": self.port_count,
            "ports": [entry.to_document() for entry in self.port_entries],
            "poke_count": self.poke_count,
            "pokes": [entry.to_document() for entry in self.poke_entries],
            "raw_prefix_words": self.words_0x00_through_0x20,
        }


@dataclass(frozen=True, slots=True)
class RunDagsSnapshot:
    sha256: str
    header: RunDagsHeader
    descriptors: tuple[RunDagsDescriptor, ...]
    consumed_bytes: int
    trailing_bytes: bytes

    @property
    def timeout_milliseconds(self) -> int:
        """Return the exact timeout selected by the recovered driver."""

        return (
            10_000
            if any(item.dag_loop_count > 1 for item in self.descriptors)
            else 5_000
        )

    @property
    def command_size(self) -> int:
        return RUN_DAGS_COMMAND_HEADER_SIZE + len(self.descriptors) * (
            RUN_DAGS_DESCRIPTOR_SIZE
        )


@dataclass(frozen=True, slots=True)
class RunDagsCompletion:
    request_bytes: bytes
    ioctl_result: int
    response_words: tuple[int, int, int, int, int]
    descriptor_result_words: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RunDagsPackagePlacement:
    """One descriptor's byte-proven package placement in the working arena."""

    descriptor_index: int
    descriptor_source_offset: int
    physical_address: int
    arena_offset: int
    model_offset: int
    package_size: int
    arena_sha256: str
    model_sha256: str
    status: str = "verified_byte_identical"

    def to_document(self) -> dict[str, int | str]:
        return {
            "descriptor_index": self.descriptor_index,
            "descriptor_source_offset": self.descriptor_source_offset,
            "physical_address": self.physical_address,
            "arena_offset": self.arena_offset,
            "model_offset": self.model_offset,
            "package_size": self.package_size,
            "arena_sha256": self.arena_sha256,
            "model_sha256": self.model_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class RunDagsPackagePlacementReport:
    """Fail-closed proof that descriptors select the exact recovered packages."""

    schema: str
    snapshot_sha256: str
    arena_sha256: str
    model_sha256: str
    physical_base: int
    arena_size: int
    packages: tuple[RunDagsPackagePlacement, ...]
    verified_bytes: int
    status: str

    @property
    def package_count(self) -> int:
        return len(self.packages)

    def to_document(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "snapshot_sha256": self.snapshot_sha256,
            "arena_sha256": self.arena_sha256,
            "model_sha256": self.model_sha256,
            "physical_base": self.physical_base,
            "arena_size": self.arena_size,
            "package_count": self.package_count,
            "verified_bytes": self.verified_bytes,
            "status": self.status,
            "packages": [item.to_document() for item in self.packages],
        }


@dataclass(frozen=True, slots=True)
class RunDagsBufferRegion:
    """A unique bounded arena region referenced by descriptor table entries."""

    physical_address: int
    arena_offset: int
    size: int
    references: tuple[tuple[int, int, int], ...]
    sha256: str
    all_zero: bool

    def to_document(self) -> dict[str, object]:
        return {
            "physical_address": self.physical_address,
            "arena_offset": self.arena_offset,
            "size": self.size,
            "references": [
                {
                    "descriptor_index": descriptor_index,
                    "table_index": table_index,
                    "port_boffset_in_dag": port_boffset_in_dag,
                }
                for descriptor_index, table_index, port_boffset_in_dag in (
                    self.references
                )
            ],
            "sha256": self.sha256,
            "all_zero": self.all_zero,
        }


@dataclass(frozen=True, slots=True)
class RunDagsMemoryTopologyReport:
    """Exact package and descriptor-table layout within one loaded arena."""

    schema: str
    snapshot_sha256: str
    arena_sha256: str
    physical_base: int
    alignment: int
    package_region_size: int
    package_padding_bytes: int
    buffers: tuple[RunDagsBufferRegion, ...]
    table_buffer_bytes: int
    table_buffer_padding_bytes: int
    dag_region_size: int
    working_bytes: int
    unassigned_working_bytes: int
    reported_total_dag_bytes: int
    reported_input_bytes: int
    reported_output_bytes: int
    reported_working_bytes: int
    status: str

    @property
    def table_buffer_count(self) -> int:
        return len(self.buffers)

    def to_document(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "snapshot_sha256": self.snapshot_sha256,
            "arena_sha256": self.arena_sha256,
            "physical_base": self.physical_base,
            "alignment": self.alignment,
            "package_region_size": self.package_region_size,
            "package_padding_bytes": self.package_padding_bytes,
            "table_buffer_count": self.table_buffer_count,
            "table_buffer_bytes": self.table_buffer_bytes,
            "table_buffer_padding_bytes": self.table_buffer_padding_bytes,
            "dag_region_size": self.dag_region_size,
            "working_bytes": self.working_bytes,
            "unassigned_working_bytes": self.unassigned_working_bytes,
            "reported": {
                "total_dag_bytes": self.reported_total_dag_bytes,
                "input_bytes": self.reported_input_bytes,
                "output_bytes": self.reported_output_bytes,
                "working_bytes": self.reported_working_bytes,
            },
            "status": self.status,
            "buffers": [item.to_document() for item in self.buffers],
        }


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def parse_run_dags_snapshot(data: bytes) -> RunDagsSnapshot:
    """Parse only the RUN_DAGS layout directly evidenced by NNCtrl callsites.

    Unknown words deliberately retain offset-based names.  Assigning semantic
    names to those fields before the driver/firmware contract is independently
    recovered would weaken the exact-execution gate.
    """

    if not isinstance(data, bytes):
        raise CavalryError("RUN_DAGS snapshot must be immutable bytes")
    if len(data) < RUN_DAGS_HEADER_SIZE:
        raise CavalryError("truncated RUN_DAGS snapshot header")
    dag_count = _u32(data, 0x10)
    if dag_count == 0 or dag_count > RUN_DAGS_MAX_DAGS:
        raise CavalryError(f"invalid RUN_DAGS dag count {dag_count}")
    consumed_bytes = RUN_DAGS_HEADER_SIZE + dag_count * RUN_DAGS_DESCRIPTOR_SIZE
    if len(data) < consumed_bytes:
        raise CavalryError(
            "truncated RUN_DAGS snapshot: "
            f"need {consumed_bytes} bytes for {dag_count} descriptors, "
            f"observed {len(data)}"
        )
    header = RunDagsHeader(
        word_0x00=_u32(data, 0x00),
        word_0x04=_u32(data, 0x04),
        word_0x08=_u32(data, 0x08),
        word_0x0c=_u32(data, 0x0C),
        dag_count=dag_count,
        reserved_words=tuple(
            _u32(data, offset) for offset in range(0x14, RUN_DAGS_HEADER_SIZE, 4)
        ),
    )
    descriptors: list[RunDagsDescriptor] = []
    for index in range(dag_count):
        source_offset = RUN_DAGS_HEADER_SIZE + index * RUN_DAGS_DESCRIPTOR_SIZE
        table_count = _u32(data, source_offset + RUN_DAGS_TABLE_COUNT_OFFSET)
        if table_count > RUN_DAGS_TABLE_MAX_ENTRIES:
            raise CavalryError(
                f"primary table count {table_count} exceeds recovered maximum "
                f"{RUN_DAGS_TABLE_MAX_ENTRIES} at descriptor {index}"
            )
        table2_count = _u32(data, source_offset + RUN_DAGS_TABLE2_COUNT_OFFSET)
        if table2_count > RUN_DAGS_TABLE2_MAX_ENTRIES:
            raise CavalryError(
                f"secondary table count {table2_count} exceeds recovered maximum "
                f"{RUN_DAGS_TABLE2_MAX_ENTRIES} at descriptor {index}"
            )
        for required_offset in (0x04, 0x08, 0x0C, 0x10):
            if _u32(data, source_offset + required_offset) == 0:
                raise CavalryError(
                    f"descriptor {index} required word +0x{required_offset:02x} "
                    "is zero"
                )
        table_entries = tuple(
            RunDagsTableEntry(
                index=entry_index,
                source_offset=(
                    source_offset
                    + RUN_DAGS_TABLE_OFFSET
                    + entry_index * RUN_DAGS_TABLE_ENTRY_SIZE
                ),
                words=struct.unpack_from(
                    "<IIII",
                    data,
                    source_offset
                    + RUN_DAGS_TABLE_OFFSET
                    + entry_index * RUN_DAGS_TABLE_ENTRY_SIZE,
                ),
            )
            for entry_index in range(table_count)
        )
        for entry in table_entries:
            if entry.words[2] == 0:
                raise CavalryError(
                    f"descriptor {index} primary table entry {entry.index} "
                    "word +0x08 is zero"
                )
        table2_entries = tuple(
            RunDagsSecondaryTableEntry(
                index=entry_index,
                source_offset=(
                    source_offset
                    + RUN_DAGS_TABLE2_OFFSET
                    + entry_index * RUN_DAGS_TABLE2_ENTRY_SIZE
                ),
                words=struct.unpack_from(
                    "<III",
                    data,
                    source_offset
                    + RUN_DAGS_TABLE2_OFFSET
                    + entry_index * RUN_DAGS_TABLE2_ENTRY_SIZE,
                ),
            )
            for entry_index in range(table2_count)
        )
        descriptors.append(
            RunDagsDescriptor(
                index=index,
                source_offset=source_offset,
                control_halfword=struct.unpack_from(
                    "<H", data, source_offset + 0x02
                )[0],
                words_0x00_through_0x20=struct.unpack_from(
                    "<IIIIIIIII", data, source_offset
                ),
                table_count=table_count,
                table_entries=table_entries,
                table2_count=table2_count,
                table2_entries=table2_entries,
                raw_bytes=data[
                    source_offset : source_offset + RUN_DAGS_DESCRIPTOR_SIZE
                ],
            )
        )
    return RunDagsSnapshot(
        sha256=hashlib.sha256(data).hexdigest(),
        header=header,
        descriptors=tuple(descriptors),
        consumed_bytes=consumed_bytes,
        trailing_bytes=data[consumed_bytes:],
    )


def analyze_run_dags_semantics(
    snapshot: RunDagsSnapshot,
    assignment_source: bytes,
) -> dict[str, object]:
    """Correlate named host fields with an exact pinned NNCtrl source.

    The source identifies which host values NNCtrl assigns.  Offsets and
    bounds remain grounded in the recovered CB62 binary/driver and captured
    request.  Four prefix words are therefore retained as unknown rather than
    inheriting meanings from a newer public header.
    """

    if not isinstance(assignment_source, bytes):
        raise CavalryError("RUN_DAGS semantic source must be immutable bytes")
    observed_sha256 = hashlib.sha256(assignment_source).hexdigest()
    if observed_sha256 != RUN_DAGS_SEMANTIC_SOURCE_SHA256:
        raise CavalryError(
            "RUN_DAGS semantic source sha256 mismatch: "
            f"expected {RUN_DAGS_SEMANTIC_SOURCE_SHA256}, "
            f"observed {observed_sha256}"
        )
    if not snapshot.descriptors:
        raise CavalryError("RUN_DAGS semantic analysis requires descriptors")

    return {
        "schema": "verkeye.cv22.run-dags-semantics.v1",
        "status": "verified_named_fields_unknowns_preserved",
        "source": {
            "repository": "https://github.com/cchiou-amba/nnctrl.git",
            "commit": RUN_DAGS_SEMANTIC_SOURCE_COMMIT,
            "version": "0.3.0",
            "rundags_c_sha256": observed_sha256,
        },
        "layout": {
            "descriptor_size": RUN_DAGS_DESCRIPTOR_SIZE,
            "named_offsets": {
                "control_word": 0x00,
                "dag_loop_count": 0x02,
                "dvi_dram_addr": 0x04,
                "dvi_img_vaddr": 0x08,
                "dvi_img_size": 0x0C,
                "dvi_dag_vaddr": 0x10,
                "port_count": RUN_DAGS_TABLE_COUNT_OFFSET,
                "poke_count": RUN_DAGS_TABLE2_COUNT_OFFSET,
                "port_table": RUN_DAGS_TABLE_OFFSET,
                "poke_table": RUN_DAGS_TABLE2_OFFSET,
            },
            "control_bits": {
                "dvi_mode": 0,
                "use_ping_pong_vmem": 1,
                "reserved": "2..15",
                "dag_loop_count": "16..31",
            },
            "unknown_word_offsets": ["0x14", "0x18", "0x1c", "0x20"],
            "port_entry": {
                "size": RUN_DAGS_TABLE_ENTRY_SIZE,
                "maximum_count": RUN_DAGS_TABLE_MAX_ENTRIES,
                "fields": {
                    "port_dram_addr": 0x00,
                    "port_boffset_in_dag": 0x04,
                    "port_dram_size": 0x08,
                    "port_daddr_increment": 0x0C,
                },
            },
            "poke_entry": {
                "size": RUN_DAGS_TABLE2_ENTRY_SIZE,
                "maximum_count": RUN_DAGS_TABLE2_MAX_ENTRIES,
                "fields": {
                    "poke_val": 0x00,
                    "poke_vaddr": 0x04,
                    "poke_bsize": 0x08,
                },
            },
        },
        "capture": {
            "snapshot_sha256": snapshot.sha256,
            "descriptor_count": len(snapshot.descriptors),
            "port_count": sum(item.port_count for item in snapshot.descriptors),
            "poke_count": sum(item.poke_count for item in snapshot.descriptors),
            "all_dvi_mode": all(item.dvi_mode for item in snapshot.descriptors),
            "all_unknown_words_zero": all(
                not any(item.unknown_words_0x14_through_0x20)
                for item in snapshot.descriptors
            ),
            "descriptors": [item.to_document() for item in snapshot.descriptors],
        },
        "claim_boundary": (
            "named host descriptor fields and captured values are verified; "
            "firmware-side accelerator semantics and four prefix words remain unresolved"
        ),
    }


def build_run_dags_command(
    snapshot: RunDagsSnapshot,
    previous_command_buffer: bytes,
) -> bytes:
    """Apply exactly the writes made by recovered ``cavalry_run_dags``.

    The driver reuses a shared command buffer and does not clear bytes outside
    the fixed header, fixed descriptor prefixes, and counted tables.  Requiring
    the previous buffer makes that state explicit and prevents this host model
    from inventing zero initialization that the recovered code does not perform.
    """

    if not isinstance(previous_command_buffer, bytes):
        raise CavalryError("previous command buffer must be immutable bytes")
    if len(previous_command_buffer) < snapshot.command_size:
        raise CavalryError(
            "command buffer is too small: "
            f"need {snapshot.command_size}, observed {len(previous_command_buffer)}"
        )

    command = bytearray(previous_command_buffer)
    struct.pack_into("<I", command, 0x00, 1)
    struct.pack_into("<I", command, 0x04, len(snapshot.descriptors))
    prior_flags = _u32(previous_command_buffer, 0x08)
    request_flags = snapshot.header.reserved_words[0]
    struct.pack_into("<I", command, 0x08, (prior_flags & ~0x3) | (request_flags & 0x3))
    struct.pack_into("<I", command, 0x0C, snapshot.header.reserved_words[1])

    for descriptor in snapshot.descriptors:
        destination = (
            RUN_DAGS_COMMAND_HEADER_SIZE
            + descriptor.index * RUN_DAGS_DESCRIPTOR_SIZE
        )
        command[destination : destination + 0x2C] = descriptor.raw_bytes[:0x2C]

        table_bytes = descriptor.table_count * RUN_DAGS_TABLE_ENTRY_SIZE
        command[
            destination
            + RUN_DAGS_TABLE_OFFSET : destination
            + RUN_DAGS_TABLE_OFFSET
            + table_bytes
        ] = descriptor.raw_bytes[
            RUN_DAGS_TABLE_OFFSET : RUN_DAGS_TABLE_OFFSET + table_bytes
        ]

        table2_bytes = descriptor.table2_count * RUN_DAGS_TABLE2_ENTRY_SIZE
        command[
            destination
            + RUN_DAGS_TABLE2_OFFSET : destination
            + RUN_DAGS_TABLE2_OFFSET
            + table2_bytes
        ] = descriptor.raw_bytes[
            RUN_DAGS_TABLE2_OFFSET : RUN_DAGS_TABLE2_OFFSET + table2_bytes
        ]

    return bytes(command)


def _reconstruct_run_dags_request(snapshot: RunDagsSnapshot) -> bytearray:
    request = bytearray(snapshot.consumed_bytes)
    header_words = (
        snapshot.header.word_0x00,
        snapshot.header.word_0x04,
        snapshot.header.word_0x08,
        snapshot.header.word_0x0c,
        snapshot.header.dag_count,
        *snapshot.header.reserved_words,
    )
    struct.pack_into(f"<{len(header_words)}I", request, 0, *header_words)
    for descriptor in snapshot.descriptors:
        request[
            descriptor.source_offset : descriptor.source_offset
            + RUN_DAGS_DESCRIPTOR_SIZE
        ] = descriptor.raw_bytes
    request.extend(snapshot.trailing_bytes)
    return request


def apply_run_dags_response(
    snapshot: RunDagsSnapshot,
    *,
    command: bytes,
    response: bytes,
) -> RunDagsCompletion:
    """Apply the recovered driver's successful-message copy-out contract.

    Firmware response status zero copies all four result words.  A nonzero
    status updates only request word ``+0x00`` and produces the driver's
    ``-EBADR`` result while still copying every per-DAG result word.  Message
    types other than the recovered RUN_DAGS completion are rejected instead of
    being treated as successful host inference.
    """

    if not isinstance(command, bytes) or not isinstance(response, bytes):
        raise CavalryError("RUN_DAGS command and response must be immutable bytes")
    if len(command) < snapshot.command_size:
        raise CavalryError(
            "command buffer is too small: "
            f"need {snapshot.command_size}, observed {len(command)}"
        )
    if len(response) < RUN_DAGS_RESPONSE_SIZE:
        raise CavalryError(
            "truncated RUN_DAGS response: "
            f"need {RUN_DAGS_RESPONSE_SIZE}, observed {len(response)}"
        )

    response_words = struct.unpack_from("<IIIII", response)
    if response_words[0] != RUN_DAGS_RESPONSE_MAGIC:
        raise CavalryError(
            "unexpected RUN_DAGS response magic: "
            f"expected 0x{RUN_DAGS_RESPONSE_MAGIC:08x}, "
            f"observed 0x{response_words[0]:08x}"
        )

    request = _reconstruct_run_dags_request(snapshot)
    status = response_words[1]
    struct.pack_into("<I", request, 0x00, status)
    if status == 0:
        struct.pack_into("<III", request, 0x04, *response_words[2:])

    descriptor_results: list[int] = []
    for descriptor in snapshot.descriptors:
        source = (
            RUN_DAGS_COMMAND_HEADER_SIZE
            + descriptor.index * RUN_DAGS_DESCRIPTOR_SIZE
            + 0x14
        )
        result_word = _u32(command, source)
        struct.pack_into("<I", request, descriptor.source_offset + 0x14, result_word)
        descriptor_results.append(result_word)

    return RunDagsCompletion(
        request_bytes=bytes(request),
        ioctl_result=(0 if status == 0 else RUN_DAGS_FIRMWARE_ERROR_RESULT),
        response_words=response_words,
        descriptor_result_words=tuple(descriptor_results),
    )


def verify_run_dags_package_placement(
    snapshot: RunDagsSnapshot,
    arena: bytes,
    *,
    physical_base: int,
    model: bytes,
    packages: Sequence[ProgramPackage],
) -> RunDagsPackagePlacementReport:
    """Prove named DVI fields select exact model packages.

    The pinned NNCtrl assignments name +0x04 as ``dvi_dram_addr`` and +0x0c as
    ``dvi_img_size``.  The recovered execution capture independently proves
    those values select the exact package bytes in the working arena.
    """

    if not isinstance(arena, bytes) or not isinstance(model, bytes):
        raise CavalryError("arena and model evidence must be immutable bytes")
    if not isinstance(physical_base, int) or isinstance(physical_base, bool):
        raise CavalryError("physical base must be an integer")
    if physical_base < 0 or physical_base > 0xFFFF_FFFF:
        raise CavalryError("physical base must fit in 32 bits")
    if len(snapshot.descriptors) != len(packages):
        raise CavalryError(
            "descriptor/package count mismatch: "
            f"{len(snapshot.descriptors)} descriptors, {len(packages)} packages"
        )

    verified: list[RunDagsPackagePlacement] = []
    for descriptor, package in zip(snapshot.descriptors, packages, strict=True):
        if package.split_index != descriptor.index:
            raise CavalryError(
                "descriptor/package order mismatch: "
                f"descriptor {descriptor.index}, split {package.split_index}"
            )
        physical_address = descriptor.dvi_dram_addr
        package_size = descriptor.dvi_img_size
        if physical_address < physical_base:
            raise CavalryError(
                f"descriptor {descriptor.index} address 0x{physical_address:08x} "
                f"is below physical base 0x{physical_base:08x}"
            )
        arena_offset = physical_address - physical_base
        arena_end = arena_offset + package_size
        if arena_end > len(arena):
            raise CavalryError(
                f"descriptor {descriptor.index} package range "
                f"[0x{arena_offset:x}, 0x{arena_end:x}) is outside captured arena "
                f"of 0x{len(arena):x} bytes"
            )
        if package_size != package.span.size:
            raise CavalryError(
                f"descriptor {descriptor.index} package size mismatch: "
                f"descriptor={package_size}, recovered={package.span.size}"
            )
        if package.span.offset < 0 or package.span.end > len(model):
            raise CavalryError(
                f"split {package.split_index} is outside recovered model"
            )

        model_package = model[package.span.offset : package.span.end]
        model_sha256 = hashlib.sha256(model_package).hexdigest()
        if model_sha256 != package.sha256:
            raise CavalryError(
                f"split {package.split_index} model package digest mismatch: "
                f"observed={model_sha256}, expected={package.sha256}"
            )
        arena_package = arena[arena_offset:arena_end]
        arena_sha256 = hashlib.sha256(arena_package).hexdigest()
        if arena_package != model_package:
            raise CavalryError(
                f"descriptor {descriptor.index} arena/model byte mismatch: "
                f"arena={arena_sha256}, model={model_sha256}"
            )
        verified.append(
            RunDagsPackagePlacement(
                descriptor_index=descriptor.index,
                descriptor_source_offset=descriptor.source_offset,
                physical_address=physical_address,
                arena_offset=arena_offset,
                model_offset=package.span.offset,
                package_size=package_size,
                arena_sha256=arena_sha256,
                model_sha256=model_sha256,
            )
        )

    return RunDagsPackagePlacementReport(
        schema="verkeye.cv22.run-dags-package-placement.v1",
        snapshot_sha256=snapshot.sha256,
        arena_sha256=hashlib.sha256(arena).hexdigest(),
        model_sha256=hashlib.sha256(model).hexdigest(),
        physical_base=physical_base,
        arena_size=len(arena),
        packages=tuple(verified),
        verified_bytes=sum(item.package_size for item in verified),
        status="verified_byte_identical",
    )


def verify_run_dags_memory_topology(
    snapshot: RunDagsSnapshot,
    arena: bytes,
    placement: RunDagsPackagePlacementReport,
    *,
    reported_total_dag_bytes: int,
    reported_input_bytes: int,
    reported_output_bytes: int,
    reported_working_bytes: int,
    alignment: int,
) -> RunDagsMemoryTopologyReport:
    """Account for the exact pre-execution arena layout with named port fields.

    The pinned NNCtrl source names the entry fields while the recovered capture
    proves their concrete values and arena ranges.  This does not assign input,
    output, or intermediate roles to individual regions.
    """

    if not isinstance(arena, bytes):
        raise CavalryError("arena evidence must be immutable bytes")
    if alignment <= 0 or alignment & (alignment - 1):
        raise CavalryError("memory topology alignment must be a power of two")
    reported_values = (
        reported_total_dag_bytes,
        reported_input_bytes,
        reported_output_bytes,
        reported_working_bytes,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in reported_values
    ):
        raise CavalryError("reported memory sizes must be non-negative integers")
    arena_sha256 = hashlib.sha256(arena).hexdigest()
    if placement.snapshot_sha256 != snapshot.sha256:
        raise CavalryError("placement/snapshot digest mismatch")
    if placement.arena_sha256 != arena_sha256:
        raise CavalryError("placement/arena digest mismatch")
    if placement.package_count != len(snapshot.descriptors):
        raise CavalryError("placement/descriptor count mismatch")
    if placement.verified_bytes != reported_input_bytes:
        raise CavalryError(
            "input byte count mismatch: "
            f"packages={placement.verified_bytes}, reported={reported_input_bytes}"
        )
    if len(arena) != reported_working_bytes:
        raise CavalryError(
            "working byte count mismatch: "
            f"arena={len(arena)}, reported={reported_working_bytes}"
        )

    def align(value: int) -> int:
        return (value + alignment - 1) & ~(alignment - 1)

    package_cursor = 0
    for item in placement.packages:
        if item.arena_offset != package_cursor:
            raise CavalryError(
                f"package {item.descriptor_index} is not at the evidenced aligned "
                f"cursor: observed=0x{item.arena_offset:x}, expected=0x{package_cursor:x}"
            )
        package_cursor = align(package_cursor + item.package_size)
    package_padding_bytes = package_cursor - placement.verified_bytes

    observed: dict[int, tuple[int, list[tuple[int, int, int]]]] = {}
    for descriptor in snapshot.descriptors:
        for entry in descriptor.table_entries:
            physical_address = entry.port_dram_addr
            port_boffset_in_dag = entry.port_boffset_in_dag
            size = entry.port_dram_size
            if size <= 0:
                raise CavalryError(
                    f"descriptor {descriptor.index} table {entry.index} has zero size"
                )
            prior = observed.get(physical_address)
            reference = (descriptor.index, entry.index, port_boffset_in_dag)
            if prior is None:
                observed[physical_address] = (size, [reference])
            elif prior[0] != size:
                raise CavalryError(
                    f"physical buffer 0x{physical_address:08x} has conflicting sizes "
                    f"{prior[0]} and {size}"
                )
            else:
                prior[1].append(reference)

    regions: list[RunDagsBufferRegion] = []
    cursor = package_cursor
    padding = 0
    for physical_address, (size, references) in sorted(observed.items()):
        if physical_address < placement.physical_base:
            raise CavalryError(
                f"table buffer 0x{physical_address:08x} is below physical base"
            )
        arena_offset = physical_address - placement.physical_base
        if arena_offset < cursor:
            raise CavalryError(
                f"table buffer at 0x{physical_address:08x} overlaps prior arena region"
            )
        padding += arena_offset - cursor
        end = arena_offset + size
        if end > len(arena):
            raise CavalryError(
                f"table buffer [0x{arena_offset:x}, 0x{end:x}) is outside arena"
            )
        raw = arena[arena_offset:end]
        regions.append(
            RunDagsBufferRegion(
                physical_address=physical_address,
                arena_offset=arena_offset,
                size=size,
                references=tuple(references),
                sha256=hashlib.sha256(raw).hexdigest(),
                all_zero=not any(raw),
            )
        )
        cursor = end

    table_buffer_bytes = sum(item.size for item in regions)
    if table_buffer_bytes != reported_output_bytes:
        raise CavalryError(
            "table-buffer byte count mismatch: "
            f"topology={table_buffer_bytes}, reported={reported_output_bytes}"
        )
    if cursor != reported_total_dag_bytes:
        raise CavalryError(
            "DAG region size mismatch: "
            f"topology={cursor}, reported={reported_total_dag_bytes}"
        )
    if reported_total_dag_bytes > reported_working_bytes:
        raise CavalryError("reported DAG region exceeds working memory")

    return RunDagsMemoryTopologyReport(
        schema="verkeye.cv22.run-dags-memory-topology.v1",
        snapshot_sha256=snapshot.sha256,
        arena_sha256=arena_sha256,
        physical_base=placement.physical_base,
        alignment=alignment,
        package_region_size=package_cursor,
        package_padding_bytes=package_padding_bytes,
        buffers=tuple(regions),
        table_buffer_bytes=table_buffer_bytes,
        table_buffer_padding_bytes=padding,
        dag_region_size=cursor,
        working_bytes=len(arena),
        unassigned_working_bytes=len(arena) - cursor,
        reported_total_dag_bytes=reported_total_dag_bytes,
        reported_input_bytes=reported_input_bytes,
        reported_output_bytes=reported_output_bytes,
        reported_working_bytes=reported_working_bytes,
        status="verified",
    )


@dataclass(frozen=True, slots=True)
class CavalryMemoryRegion:
    handle: int
    address: int
    requested_size: int
    mapped_size: int
    persistent: bool


class CavalryMemoryArena:
    """Deterministic bounded memory used by the compatibility device model."""

    def __init__(
        self,
        *,
        base_address: int,
        capacity: int,
        page_size: int,
        trace: TraceRecorder | None = None,
    ) -> None:
        if base_address < 0 or capacity <= 0 or page_size <= 0:
            raise CavalryError("invalid Cavalry arena geometry")
        if page_size & (page_size - 1):
            raise CavalryError("Cavalry arena page size must be a power of two")
        self.base_address = base_address
        self.capacity = capacity
        self.page_size = page_size
        self.trace = trace
        self._cursor = 0
        self._next_handle = 1
        self._regions: dict[int, CavalryMemoryRegion] = {}
        self._storage: dict[int, bytearray] = {}

    def _align(self, value: int) -> int:
        return (value + self.page_size - 1) & ~(self.page_size - 1)

    def allocate(self, size: int, *, persistent: bool = False) -> CavalryMemoryRegion:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise CavalryError("allocation size must be a positive integer")
        mapped_size = self._align(size)
        if mapped_size > self.capacity - self._cursor:
            raise CavalryError(
                f"Cavalry arena exhausted: requested 0x{mapped_size:x}, "
                f"remaining 0x{self.capacity - self._cursor:x}"
            )
        handle = self._next_handle
        self._next_handle += 1
        region = CavalryMemoryRegion(
            handle=handle,
            address=self.base_address + self._cursor,
            requested_size=size,
            mapped_size=mapped_size,
            persistent=persistent,
        )
        self._cursor += mapped_size
        self._regions[handle] = region
        self._storage[handle] = bytearray(mapped_size)
        if self.trace is not None:
            self.trace.record(
                "cavalry_memory",
                "allocate",
                "passed",
                address=region.address,
                handle=handle,
                mapped_size=mapped_size,
                persistent=persistent,
                requested_size=size,
            )
        return region

    def _region(self, handle: int) -> tuple[CavalryMemoryRegion, bytearray]:
        region = self._regions.get(handle)
        storage = self._storage.get(handle)
        if region is None or storage is None:
            raise CavalryError(f"unknown or freed allocation handle {handle}")
        return region, storage

    @staticmethod
    def _bounds(region: CavalryMemoryRegion, offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset > region.requested_size - size:
            raise CavalryError(
                f"range offset=0x{offset:x} size=0x{size:x} is outside "
                f"requested region size 0x{region.requested_size:x}"
            )

    def read(self, handle: int, offset: int, size: int) -> bytes:
        region, storage = self._region(handle)
        self._bounds(region, offset, size)
        return bytes(storage[offset : offset + size])

    def write(self, handle: int, offset: int, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise CavalryError("Cavalry writes require immutable bytes")
        region, storage = self._region(handle)
        self._bounds(region, offset, len(data))
        storage[offset : offset + len(data)] = data

    def free(self, handle: int) -> None:
        region, _ = self._region(handle)
        del self._regions[handle]
        del self._storage[handle]
        if self.trace is not None:
            self.trace.record(
                "cavalry_memory",
                "free",
                "passed",
                address=region.address,
                handle=handle,
                mapped_size=region.mapped_size,
                persistent=region.persistent,
                requested_size=region.requested_size,
            )
