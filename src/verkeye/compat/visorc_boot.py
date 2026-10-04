"""Pinned VISORC boot/register contract recovered from CB62 artifacts.

This module deliberately stops at the native-instruction boundary.  It proves
the host driver's register writes and ISR state transitions from exact AArch64
instruction bytes, while retaining the Ambarella ORC words as opaque data.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verkeye.cv22.vendor_runtime import analyze_elf

from .cavalry_firmware import parse_cavalry_firmware


class VisorcBootError(ValueError):
    """A recovered boot artifact no longer matches the pinned evidence."""


@dataclass(frozen=True, slots=True)
class VisorcMmioWrite:
    """One ordered 32-bit write into the sparse VISORC register window."""

    sequence: int
    offset: int
    value: int
    source: str

    def to_document(self) -> dict[str, int | str]:
        return {
            "sequence": self.sequence,
            "offset": self.offset,
            "value": self.value,
            "source": self.source,
        }


@dataclass(slots=True)
class VisorcBootMachine:
    """Executable sparse model of only the driver-proven boot state machine."""

    registers: dict[int, int] = field(default_factory=dict)
    writes: list[VisorcMmioWrite] = field(default_factory=list)
    state: int = 0
    response_word: int = 0
    command_code: int | None = None
    running: bool = False

    def _write(self, offset: int, value: int, *, source: str) -> None:
        if not 0 <= offset <= 0x0100_0000 - 4 or offset % 4:
            raise VisorcBootError(f"invalid VISORC MMIO offset 0x{offset:x}")
        if not 0 <= value <= 0xFFFF_FFFF:
            raise VisorcBootError(f"invalid VISORC 32-bit value 0x{value:x}")
        self.registers[offset] = value
        self.writes.append(
            VisorcMmioWrite(
                sequence=len(self.writes),
                offset=offset,
                value=value,
                source=source,
            )
        )

    def start(self, *, cavalry_address: int, ucode_address: int) -> None:
        """Replay the exact ``visorc_start`` write sequence.

        No native instruction is executed.  Startup becomes successful only
        after :meth:`on_interrupt` observes the ISR transition to state one.
        """

        if (
            not isinstance(cavalry_address, int)
            or isinstance(cavalry_address, bool)
            or not isinstance(ucode_address, int)
            or isinstance(ucode_address, bool)
            or cavalry_address < 0
            or ucode_address < 0
        ):
            raise VisorcBootError("VISORC addresses must be unsigned addresses")
        if ucode_address < cavalry_address:
            raise VisorcBootError("ucode address is below Cavalry base")
        relative_ucode = ucode_address - cavalry_address
        if (
            cavalry_address > 0xFFFF_FFFF
            or ucode_address > 0xFFFF_FFFF
            or relative_ucode > 0xFFFF_FFFF
        ):
            raise VisorcBootError("addresses exceed the recovered 32-bit VISORC offset contract")

        self.registers.clear()
        self.writes.clear()
        self.state = 0
        self.response_word = 0
        self.command_code = None
        self.running = False

        for value in (0, 0x700, 0):
            self._write(0x80228, value, source="visorc_start")
        for offset, value in (
            (0xA0040, 0),
            (0xA0044, 0x600000),
            (0xA0048, cavalry_address | 1),
            (0xA0054, 0),
            (0xA0064, 0),
            (0xA0074, 0),
            (0x10008, relative_ucode),
            (0x10004, 3),
            (0x10000, 0xF00),
        ):
            self._write(offset, value, source="visorc_start")
        for pass_value in (0x1020, 0, 0x1020, 0):
            for offset in range(0x5F1C0, 0x5F200, 4):
                self._write(offset, pass_value, source="visorc_start")
        self._write(0x10000, 0xF, source="visorc_start")

    def kick(self, *, source: str = "visorc_kick_vp") -> None:
        """Apply the post-``dsb st`` mailbox write proven by the driver."""

        self._write(0x5F1FC, 0x1400, source=source)

    def request_stop(self) -> None:
        """Model command-buffer code two followed by the inline VP kick."""

        self.command_code = 2
        self.kick(source="visorc_stop")

    def on_interrupt(self, response_word: int) -> None:
        """Apply the state transitions visible in ``cavalry_vp_isr``."""

        if (
            not isinstance(response_word, int)
            or isinstance(response_word, bool)
            or not 0 <= response_word <= 0xFFFF_FFFF
        ):
            raise VisorcBootError("VISORC response word must fit in 32 bits")
        self.response_word = response_word
        if self.state == 0:
            self.state = 1
        if self.state == 1 and response_word == 0x8000_0002:
            self.state = 0
        self.running = self.state == 1


_DRIVER_SIZE = 154_192
_DRIVER_SHA256 = "39811bf17113e78c7babe9c0a6b8bad53ca80648847c364da5fccb5dd5f0de24"
_FIRMWARE_SIZE = 101_332
_FIRMWARE_SHA256 = "889c32b8b599454b1a98dfac4e6f9ee7f25abe8f02e21030646e697b07aa45f3"

_SYMBOLS = {
    "cavalry_vp_isr": (0x01C4, 0x00AC),
    "visorc_kick_vp": (0x2ED0, 0x0020),
    "visorc_init": (0x2FC0, 0x0194),
    "visorc_start": (0x3380, 0x03AC),
    "visorc_stop": (0x3780, 0x01B0),
}

# Each proof is an exact little-endian AArch64 instruction from the pinned
# driver's .text section.  Labels describe only the behavior established by
# the instruction and its surrounding data flow; no ORC opcode is decoded.
_INSTRUCTION_PROOFS: tuple[tuple[str, int, str], ...] = (
    ("ioremap_size_16m", 0x3104, "0120a0d2"),
    ("ioremap_physical_base", 0x3108, "00a0bdd2"),
    ("start_state_clear", 0x33CC, "7fde03b9"),
    ("response_pointer_load", 0x33D0, "605a40f9"),
    ("response_word_clear", 0x33D4, "1f0000b9"),
    ("reset_offset_low", 0x33E0, "024580d2"),
    ("reset_offset_high", 0x33E4, "0201a0f2"),
    ("reset_value_0x700", 0x33FC, "03e08052"),
    ("offset_a0040_high", 0x3420, "00804291"),
    ("offset_a0040_low", 0x3424, "00000191"),
    ("value_0x600000", 0x3434, "020ca052"),
    ("offset_a0044_low", 0x343C, "00100191"),
    ("firmware_address_context_load", 0x344C, "603e40f9"),
    ("offset_a0048_low", 0x3454, "42200191"),
    ("firmware_address_enable_bit", 0x3458, "00000032"),
    ("offset_a0054_low", 0x346C, "00500191"),
    ("offset_a0064_low", 0x3480, "00900191"),
    ("offset_a0074_low", 0x3494, "00d00191"),
    ("cavalry_base_context_load", 0x349C, "623e40f9"),
    ("ucode_address_context_load", 0x34A0, "609e40f9"),
    ("ucode_relative_subtract", 0x34A4, "000002cb"),
    ("offset_10008_low", 0x34B4, "42200091"),
    ("value_3", 0x34C4, "62008052"),
    ("offset_10004_low", 0x34CC, "00100091"),
    ("value_f00", 0x34DC, "02e08152"),
    ("offset_10000", 0x34E0, "00404091"),
    ("range_start_5f1c0", 0x34E8, "02389ed2"),
    ("range_end_5f200", 0x34EC, "03409ed2"),
    ("range_start_high", 0x34F0, "a200a0f2"),
    ("range_value_1020", 0x34F4, "04048252"),
    ("range_pass_1_store", 0x350C, "040000b9"),
    ("range_pass_2_store", 0x353C, "1f0000b9"),
    ("range_pass_3_store", 0x356C, "640000b9"),
    ("range_pass_4_store", 0x359C, "7f0000b9"),
    ("value_f", 0x35B4, "e1018052"),
    ("final_offset_10000", 0x35B8, "00404091"),
    ("start_timeout_ticks", 0x35F8, "15fa80d2"),
    ("start_state_load", 0x3610, "60de43b9"),
    ("start_success_state_compare", 0x3614, "1f040071"),
    ("kick_store_barrier", 0x2ED0, "9f3e03d5"),
    ("kick_value_1400", 0x2ED8, "01808252"),
    ("kick_offset_5f000", 0x2EE0, "007c4191"),
    ("kick_offset_1fc", 0x2EE4, "00f00791"),
    ("kick_store", 0x2EE8, "010000b9"),
    ("stop_command_pointer_load", 0x37B8, "804e40f9"),
    ("stop_command_code_2", 0x37BC, "41008052"),
    ("stop_command_store", 0x37C0, "010000b9"),
    ("stop_store_barrier", 0x37C4, "9f3e03d5"),
    ("stop_kick_value_1400", 0x37CC, "01808252"),
    ("stop_kick_offset_5f000", 0x37D4, "007c4191"),
    ("stop_kick_offset_1fc", 0x37D8, "00f00791"),
    ("stop_kick_store", 0x37DC, "010000b9"),
    ("stop_timeout_ticks", 0x3804, "13fa80d2"),
    ("isr_context_load", 0x01CC, "205440f9"),
    ("isr_state_load", 0x01D4, "01dc43b9"),
    ("isr_response_pointer_load", 0x01D8, "025840f9"),
    ("isr_state_one", 0x01E0, "21008052"),
    ("isr_response_word_load", 0x0220, "420040b9"),
    ("isr_stop_magic_low", 0x0224, "41008052"),
    ("isr_stop_magic_high", 0x0228, "0100b072"),
    ("isr_response_compare", 0x022C, "5f00016b"),
    ("isr_stop_state_clear", 0x0234, "1fdc03b9"),
)


def _read_pinned(
    path: str | Path,
    *,
    role: str,
    expected_size: int,
    expected_sha256: str,
) -> tuple[Path, bytes]:
    source = Path(path).resolve()
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise VisorcBootError(f"cannot read {role} {source}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size:
        raise VisorcBootError(
            f"{role} size mismatch: expected {expected_size}, observed {len(data)}"
        )
    if digest != expected_sha256:
        raise VisorcBootError(
            f"{role} sha256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    return source, data


def _vaddr_to_offset(elf: dict[str, Any], vaddr: int, size: int) -> int:
    for section in elf["sections"]:
        start = section["virtual_address"]
        end = start + section["size"]
        if start <= vaddr and vaddr + size <= end:
            return section["file_offset"] + (vaddr - start)
    raise VisorcBootError(f"virtual address 0x{vaddr:x} is outside driver sections")


def _verify_driver_contract(path: Path, data: bytes) -> dict[str, Any]:
    elf = analyze_elf(path)
    symbols = {item["name"]: item for item in elf["defined_symbols"]}
    for name, (vaddr, size) in _SYMBOLS.items():
        observed = symbols.get(name)
        if observed is None:
            raise VisorcBootError(f"driver lacks required symbol {name}")
        if observed["virtual_address"] != vaddr or observed["size"] != size:
            raise VisorcBootError(
                f"driver symbol {name} mismatch: expected VA 0x{vaddr:x} size {size}, "
                f"observed VA 0x{observed['virtual_address']:x} size {observed['size']}"
            )

    entries: list[dict[str, int | str]] = []
    for label, vaddr, expected_hex in _INSTRUCTION_PROOFS:
        expected = bytes.fromhex(expected_hex)
        file_offset = _vaddr_to_offset(elf, vaddr, len(expected))
        observed = data[file_offset : file_offset + len(expected)]
        if observed != expected:
            raise VisorcBootError(
                f"driver instruction proof {label} mismatch at VA 0x{vaddr:x}: "
                f"expected {expected_hex}, observed {observed.hex()}"
            )
        entries.append(
            {
                "label": label,
                "virtual_address": vaddr,
                "file_offset": file_offset,
                "instruction_hex": observed.hex(),
            }
        )
    return {"proof_count": len(entries), "entries": entries}


def analyze_visorc_boot(
    driver_path: str | Path,
    firmware_path: str | Path,
) -> dict[str, Any]:
    """Recover the exact host-visible boot contract and stop at ORC semantics."""

    driver_source, driver = _read_pinned(
        driver_path,
        role="driver",
        expected_size=_DRIVER_SIZE,
        expected_sha256=_DRIVER_SHA256,
    )
    _, firmware = _read_pinned(
        firmware_path,
        role="firmware",
        expected_size=_FIRMWARE_SIZE,
        expected_sha256=_FIRMWARE_SHA256,
    )
    parsed_firmware = parse_cavalry_firmware(firmware)
    instruction_proofs = _verify_driver_contract(driver_source, driver)

    native_words = struct.unpack_from("<8I", firmware, 0x60)
    image_extent_word = struct.unpack_from("<I", firmware, 0x54)[0]

    return {
        "schema": "verkeye.compat.visorc-boot.v1",
        "driver": {
            "size": len(driver),
            "sha256": hashlib.sha256(driver).hexdigest(),
        },
        "firmware": {
            "size": len(firmware),
            "sha256": hashlib.sha256(firmware).hexdigest(),
            "version": parsed_firmware.version.to_document(),
            "metadata_prefix": {
                "start": 0,
                "end_exclusive": 0x60,
                "image_extent_word": {
                    "offset": 0x54,
                    "value": image_extent_word,
                    "semantic_status": "unresolved",
                },
            },
            "native_word_probe": {
                "start": 0x60,
                "word_size": 4,
                "byte_order": "little",
                "words": [f"0x{word:08x}" for word in native_words],
                "instruction_set": "Ambarella ORC/VP",
                "opcode_semantics": "unresolved",
            },
        },
        "register_window": {
            "physical_base": 0xED00_0000,
            "size": 0x0100_0000,
            "mapped_by": "visorc_init",
            "function_virtual_address": 0x2FC0,
        },
        "start": {
            "function_virtual_address": 0x3380,
            "context_state_offset": 0x3DC,
            "success_state": 1,
            "timeout_ticks": 2_000,
            "mmio_writes": [
                {"offset": 0x80228, "values": [0, 0x700, 0]},
                {"offset": 0xA0040, "value": 0},
                {"offset": 0xA0044, "value": 0x600000},
                {
                    "offset": 0xA0048,
                    "value_expression": "low32(context[0x78]) | 0x1",
                },
                {"offset": 0xA0054, "value": 0},
                {"offset": 0xA0064, "value": 0},
                {"offset": 0xA0074, "value": 0},
                {
                    "offset": 0x10008,
                    "value_expression": "low32(context[0x138] - context[0x78])",
                },
                {"offset": 0x10004, "value": 3},
                {"offset": 0x10000, "value": 0xF00},
                {
                    "offset_range": {
                        "start": 0x5F1C0,
                        "end_exclusive": 0x5F200,
                    },
                    "stride": 4,
                    "pass_values": [0x1020, 0, 0x1020, 0],
                },
                {"offset": 0x10000, "value": 0xF},
            ],
        },
        "kick": {
            "function_virtual_address": 0x2ED0,
            "store_barrier": "dsb st",
            "mmio_offset": 0x5F1FC,
            "value": 0x1400,
        },
        "stop": {
            "function_virtual_address": 0x3780,
            "command_buffer_pointer_context_offset": 0x98,
            "command_code": 2,
            "kick_mmio_offset": 0x5F1FC,
            "kick_value": 0x1400,
            "context_state_offset": 0x3DC,
            "success_state": 0,
            "timeout_ticks": 2_000,
        },
        "interrupt": {
            "function": "cavalry_vp_isr",
            "function_virtual_address": 0x1C4,
            "response_pointer_context_offset": 0xB0,
            "context_state_offset": 0x3DC,
            "on_interrupt_when_state_zero": {"new_state": 1},
            "startup_success_condition": "context[0x3dc] == 1",
            "startup_response_magic": None,
            "stop_response_magic": "0x80000002",
            "on_stop_response": {"new_state": 0},
        },
        "instruction_proofs": instruction_proofs,
        "gate": {
            "boot_contract": "passed",
            "instruction_execution": "blocked",
            "reason_codes": ["ORC_OPCODE_SEMANTICS_UNRESOLVED"],
        },
        "claim_boundary": (
            "The pinned AArch64 driver proves the host MMIO/state-machine contract. "
            "The firmware words beginning at 0x60 remain native Ambarella ORC/VP "
            "instructions with unresolved opcode semantics; no inference is claimed."
        ),
    }
