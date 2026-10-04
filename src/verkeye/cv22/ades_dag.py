"""Strict parser for the Ambarella ADES DAG decoder transcript.

ADES is retained as the vendor-provided structural decoder.  This module turns
its human-readable, digest-pinned output into immutable records without
claiming that decoded structure alone proves numerical equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import re
import struct
from typing import Mapping

from .program import FOOTER_CONTROL_WORD, FOOTER_MAGIC, FOOTER_SIZE


class AdesDagError(ValueError):
    """An ADES transcript is incomplete, inconsistent, or unsupported."""


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DAG_SIZE = re.compile(r"dag file,\s+[^,]+,\s+is\s+(\d+)\s+bytes long")
_OPERATOR = re.compile(
    r"operator \(op(?P<id>\d+)\) of type, "
    r"(?P<type>[A-Za-z_][A-Za-z0-9_]*) \((?P<opcode>\d+)\), is constructed\."
)
_LINK = re.compile(
    r"Link from OP\[(?P<source>\d+)\] Port (?P<source_port>\d+) "
    r"to OP\[(?P<target>\d+)\] Port (?P<target_port>\d+)"
)
_INPUT = re.compile(
    r"Dagbin loader: Input Descriptor (?P<name>[A-Za-z0-9_]+) "
    r"\((?P<shape>\d+(?:x\d+)*)\)\s+->\(op(?P<target>\d+), "
    r"V(?P<target_port>\d+)\)"
)
_OUTPUT = re.compile(
    r"Dagbin loader: Output Descriptor (?P<name>[A-Za-z0-9_]+)"
)
_CONFIG_SECTION = re.compile(
    r"\[Config Info\]\s*(?P<body>.*?)(?=\n\[Link Info\])", re.DOTALL
)
_CONFIG = re.compile(r"^(?P<name>[A-Za-z0-9_.]+):\s*(?P<value>-?(?:0x)?[0-9A-Fa-f]+)$")


@dataclass(frozen=True, slots=True)
class AdesOperator:
    identifier: str
    operator_id: int
    type_name: str
    opcode: int
    config_available: bool
    config: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class AdesLink:
    source_operator: int
    source_port: int
    target_operator: int
    target_port: int


@dataclass(frozen=True, slots=True)
class AdesInputDescriptor:
    name: str
    shape: tuple[int, ...]
    target_operator: int
    target_port: int


@dataclass(frozen=True, slots=True)
class AdesOutputDescriptor:
    name: str


@dataclass(frozen=True, slots=True)
class AdesDagGraph:
    dag_size: int
    operators: tuple[AdesOperator, ...]
    links: tuple[AdesLink, ...]
    inputs: tuple[AdesInputDescriptor, ...]
    outputs: tuple[AdesOutputDescriptor, ...]


def extract_dagbin(dvi: bytes, dag_offset: int) -> bytes:
    """Extract the ADES-loadable DAG suffix while preserving its footer."""

    if not isinstance(dvi, bytes) or len(dvi) <= FOOTER_SIZE:
        raise AdesDagError("DVI source is too short for its compiler footer")
    if not isinstance(dag_offset, int) or isinstance(dag_offset, bool):
        raise AdesDagError("DAG offset must be an integer")
    if not 0 < dag_offset < len(dvi) - FOOTER_SIZE:
        raise AdesDagError("DAG offset is outside the DVI body")
    leading, control, _raw_word, magic = struct.unpack("<IIII", dvi[-FOOTER_SIZE:])
    if leading != 0 or control != FOOTER_CONTROL_WORD or magic != FOOTER_MAGIC:
        raise AdesDagError("DVI compiler footer is invalid")
    return dvi[dag_offset:]


def parse_ades_dag_transcript(
    transcript: str, *, require_configs: bool = True
) -> AdesDagGraph:
    """Parse one ADES ``ldag``/``opinfo`` transcript fail-closed.

    The caller must request ``opinfo`` in ascending operator order.  ADES does
    not echo commands, so configuration sections are associated with operators
    by that deterministic order and their count must match exactly.
    """

    if not isinstance(transcript, str) or not transcript:
        raise AdesDagError("ADES transcript is empty")
    text = _ANSI.sub("", transcript).replace("\r\n", "\n")
    size_match = _DAG_SIZE.search(text)
    if size_match is None:
        raise AdesDagError("ADES transcript is missing DAG size")
    dag_size = int(size_match.group(1))

    operator_matches = tuple(_OPERATOR.finditer(text))
    if not operator_matches:
        raise AdesDagError("ADES transcript contains no operators")
    operator_ids = tuple(int(match.group("id")) for match in operator_matches)
    if operator_ids != tuple(range(len(operator_ids))):
        raise AdesDagError("ADES operator identifiers are not contiguous")

    configs = tuple(
        _parse_config_section(match.group("body"))
        for match in _CONFIG_SECTION.finditer(text)
    )
    if require_configs and len(configs) != len(operator_matches):
        raise AdesDagError(
            "ADES configuration sections do not match the operator count: "
            f"{len(configs)} != {len(operator_matches)}"
        )
    if not require_configs and configs:
        raise AdesDagError(
            "operator-discovery transcript unexpectedly contains configuration sections"
        )
    if not configs:
        configs = tuple((False, {}) for _ in operator_matches)

    operators = tuple(
        AdesOperator(
            identifier=f"op{operator_id}",
            operator_id=operator_id,
            type_name=match.group("type"),
            opcode=int(match.group("opcode")),
            config_available=config_available,
            config=MappingProxyType(dict(config)),
        )
        for operator_id, match, (config_available, config) in zip(
            operator_ids, operator_matches, configs
        )
    )
    links = tuple(
        AdesLink(
            source_operator=int(match.group("source")),
            source_port=int(match.group("source_port")),
            target_operator=int(match.group("target")),
            target_port=int(match.group("target_port")),
        )
        for match in _LINK.finditer(text)
    )
    inputs = tuple(
        AdesInputDescriptor(
            name=match.group("name"),
            shape=tuple(int(item) for item in match.group("shape").split("x")),
            target_operator=int(match.group("target")),
            target_port=int(match.group("target_port")),
        )
        for match in _INPUT.finditer(text)
    )
    outputs = tuple(
        AdesOutputDescriptor(name=match.group("name"))
        for match in _OUTPUT.finditer(text)
    )
    _validate_references(len(operators), links, inputs)
    if not outputs:
        raise AdesDagError("ADES transcript contains no output descriptors")
    return AdesDagGraph(
        dag_size=dag_size,
        operators=operators,
        links=links,
        inputs=inputs,
        outputs=outputs,
    )


def ades_dag_document(graph: AdesDagGraph) -> dict[str, object]:
    """Render a stable JSON-compatible representation of one decoded graph."""

    return {
        "dag_size": graph.dag_size,
        "operators": [
            {
                "identifier": operator.identifier,
                "operator_id": operator.operator_id,
                "type_name": operator.type_name,
                "opcode": operator.opcode,
                "config_available": operator.config_available,
                "config": dict(operator.config),
            }
            for operator in graph.operators
        ],
        "links": [
            {
                "source_operator": link.source_operator,
                "source_port": link.source_port,
                "target_operator": link.target_operator,
                "target_port": link.target_port,
            }
            for link in graph.links
        ],
        "inputs": [
            {
                "name": descriptor.name,
                "shape": list(descriptor.shape),
                "target_operator": descriptor.target_operator,
                "target_port": descriptor.target_port,
            }
            for descriptor in graph.inputs
        ],
        "outputs": [{"name": descriptor.name} for descriptor in graph.outputs],
    }


def _parse_config_section(body: str) -> tuple[bool, dict[str, int]]:
    config: dict[str, int] = {}
    lines = tuple(line.strip() for line in body.splitlines() if line.strip())
    if lines == ("config unavailable",):
        return False, config
    for line in lines:
        match = _CONFIG.fullmatch(line)
        if match is None:
            raise AdesDagError(f"unsupported ADES configuration line: {line}")
        name = match.group("name")
        if name in config:
            raise AdesDagError(f"duplicate ADES configuration key: {name}")
        config[name] = int(match.group("value"), 0)
    return True, config


def _validate_references(
    operator_count: int,
    links: tuple[AdesLink, ...],
    inputs: tuple[AdesInputDescriptor, ...],
) -> None:
    for link in links:
        if not (
            0 <= link.source_operator < operator_count
            and 0 <= link.target_operator < operator_count
        ):
            raise AdesDagError("ADES link references an unknown operator")
    for descriptor in inputs:
        if not 0 <= descriptor.target_operator < operator_count:
            raise AdesDagError("ADES input descriptor references an unknown operator")
