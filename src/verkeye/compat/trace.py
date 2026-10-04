"""Deterministic compatibility-layer traces.

The trace intentionally contains no implicit wall-clock timestamps.  Timing data
belongs in explicit measured fields supplied by the caller; omitting ambient
time makes identical compatibility operations byte-for-byte reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceEvent:
    sequence: int
    phase: str
    operation: str
    outcome: str
    details: dict[str, Any]

    def to_document(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "phase": self.phase,
            "operation": self.operation,
            "outcome": self.outcome,
            "details": self.details,
        }


class TraceRecorder:
    """Append-only, deterministic record of compatibility operations."""

    def __init__(self, events: tuple[TraceEvent, ...] = ()) -> None:
        self._events = list(events)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        phase: str,
        operation: str,
        outcome: str,
        **details: Any,
    ) -> TraceEvent:
        document = {
            "phase": phase,
            "operation": operation,
            "outcome": outcome,
            "details": details,
        }
        try:
            json.dumps(document, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("trace details must be JSON-serializable") from exc
        event = TraceEvent(
            sequence=len(self._events),
            phase=phase,
            operation=operation,
            outcome=outcome,
            details=details,
        )
        self._events.append(event)
        return event

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "verkeye.cv22.compatibility-trace.v1",
            "events": [event.to_document() for event in self._events],
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            self.to_document(), indent=2, sort_keys=True, allow_nan=False
        )
        destination.write_text(content + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "TraceRecorder":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("schema") != "verkeye.cv22.compatibility-trace.v1":
            raise ValueError("unsupported compatibility trace schema")
        raw_events = document.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("compatibility trace events must be an array")
        events: list[TraceEvent] = []
        for expected_sequence, item in enumerate(raw_events):
            if not isinstance(item, dict) or item.get("sequence") != expected_sequence:
                raise ValueError("compatibility trace sequence is not contiguous")
            details = item.get("details")
            if not isinstance(details, dict):
                raise ValueError("compatibility trace details must be an object")
            events.append(
                TraceEvent(
                    sequence=expected_sequence,
                    phase=str(item.get("phase")),
                    operation=str(item.get("operation")),
                    outcome=str(item.get("outcome")),
                    details=details,
                )
            )
        return cls(tuple(events))
