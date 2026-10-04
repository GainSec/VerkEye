"""Immutable runtime performance evidence and percentile calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping


def _identity_copy(value: Mapping[str, str]) -> Mapping[str, str]:
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise ValueError("identity keys and values must be strings")
    return MappingProxyType(dict(sorted(value.items())))


def _nearest_rank(values: tuple[int, ...], percentile: int) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class PerformanceSample:
    """One warmed benchmark with every measured per-frame phase sample."""

    warmup_frames: int
    measured_frames: int
    phase_ns: Mapping[str, tuple[int, ...]]
    elapsed_ns: int
    backend_identity: Mapping[str, str] = field(default_factory=dict)
    host_identity: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.warmup_frames, int) or self.warmup_frames < 0:
            raise ValueError("warmup_frames must be a non-negative integer")
        if not isinstance(self.measured_frames, int) or self.measured_frames <= 0:
            raise ValueError("measured_frames must be a positive integer")
        if not isinstance(self.elapsed_ns, int) or self.elapsed_ns <= 0:
            raise ValueError("elapsed_ns must be a positive integer")
        normalized: dict[str, tuple[int, ...]] = {}
        for name, values in self.phase_ns.items():
            samples = tuple(values)
            if not name:
                raise ValueError("phase names must not be empty")
            if len(samples) != self.measured_frames:
                raise ValueError(
                    f"phase {name!r} sample count must equal measured_frames"
                )
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in samples
            ):
                raise ValueError("phase samples must be non-negative integers")
            normalized[name] = samples
        object.__setattr__(self, "phase_ns", MappingProxyType(dict(sorted(normalized.items()))))
        object.__setattr__(self, "backend_identity", _identity_copy(self.backend_identity))
        object.__setattr__(self, "host_identity", _identity_copy(self.host_identity))

    @property
    def fps(self) -> float:
        return self.measured_frames * 1_000_000_000.0 / self.elapsed_ns

    def phase_percentiles(self, name: str) -> dict[str, int]:
        try:
            samples = self.phase_ns[name]
        except KeyError as error:
            raise KeyError(f"unknown performance phase {name!r}") from error
        return {
            "p50": _nearest_rank(samples, 50),
            "p95": _nearest_rank(samples, 95),
            "maximum": max(samples),
        }

    def to_document(self) -> dict[str, object]:
        return {
            "warmup_frames": self.warmup_frames,
            "measured_frames": self.measured_frames,
            "elapsed_ns": self.elapsed_ns,
            "fps": self.fps,
            "phase_ns": {
                name: list(values) for name, values in self.phase_ns.items()
            },
            "phase_percentiles_ns": {
                name: self.phase_percentiles(name) for name in self.phase_ns
            },
            "backend_identity": dict(self.backend_identity),
            "host_identity": dict(self.host_identity),
        }


def sample_from_frame_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    warmup_frames: int,
    elapsed_ns: int,
    backend_identity: Mapping[str, str] | None = None,
    host_identity: Mapping[str, str] | None = None,
) -> PerformanceSample:
    """Build a benchmark sample from exact-inference frame documents."""

    frames = tuple(documents)
    if not frames:
        raise ValueError("at least one measured frame document is required")
    phase_names: tuple[str, ...] | None = None
    collected: dict[str, list[int]] = {}
    for index, frame in enumerate(frames):
        try:
            raw_phases = frame["execution"]["phase_ns"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"frame {index} does not contain execution phase_ns"
            ) from error
        if not isinstance(raw_phases, Mapping):
            raise ValueError(f"frame {index} phase_ns is not a mapping")
        observed = tuple(sorted(raw_phases))
        if phase_names is None:
            phase_names = observed
            collected = {name: [] for name in observed}
        elif observed != phase_names:
            raise ValueError(f"frame {index} phase set changed")
        for name in observed:
            value = raw_phases[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"frame {index} phase {name!r} is not a non-negative integer"
                )
            collected[name].append(value)
    return PerformanceSample(
        warmup_frames=warmup_frames,
        measured_frames=len(frames),
        phase_ns={name: tuple(values) for name, values in collected.items()},
        elapsed_ns=elapsed_ns,
        backend_identity=backend_identity or {},
        host_identity=host_identity or {},
    )
