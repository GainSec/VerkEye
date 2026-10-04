"""Deterministic QEMU launch support for recovered AArch64 userspace."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .trace import TraceRecorder


class QemuCompatibilityError(ValueError):
    """A QEMU launch cannot meet its declared compatibility contract."""


@dataclass(frozen=True, slots=True)
class QemuEnvironment:
    qemu: Path
    sysroot: Path
    library_paths: tuple[Path, ...]
    required_libraries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QemuLaunch:
    argv: tuple[str, ...]
    environment: dict[str, str]
    provenance: dict[str, object]


@dataclass(frozen=True, slots=True)
class QemuResult:
    returncode: int
    stdout: str
    stderr: str


def _loader(sysroot: Path) -> Path:
    candidates = (
        sysroot / "lib/ld-linux-aarch64.so.1",
        sysroot / "lib64/ld-linux-aarch64.so.1",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise QemuCompatibilityError(
        f"AArch64 loader ld-linux-aarch64.so.1 is missing below {sysroot}"
    )


def _libraries(
    names: Sequence[str], library_paths: Sequence[Path]
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in names:
        matches = [path / name for path in library_paths if (path / name).is_file()]
        if not matches:
            raise QemuCompatibilityError(f"required library {name} is missing")
        resolved[name] = str(matches[0].resolve())
    return resolved


def build_launch(
    environment: QemuEnvironment,
    executable: str | Path,
    arguments: Sequence[str] = (),
    *,
    guest_environment: Mapping[str, str] | None = None,
) -> QemuLaunch:
    qemu = environment.qemu.resolve()
    sysroot = environment.sysroot.resolve()
    target = Path(executable).resolve()
    if not qemu.is_file() or not os.access(qemu, os.X_OK):
        raise QemuCompatibilityError(f"QEMU executable is unavailable: {qemu}")
    if not target.is_file() or not os.access(target, os.X_OK):
        raise QemuCompatibilityError(f"AArch64 executable is unavailable: {target}")
    loader = _loader(sysroot)
    paths = tuple(path.resolve() for path in environment.library_paths)
    for path in paths:
        if not path.is_dir():
            raise QemuCompatibilityError(f"library path is unavailable: {path}")
    libraries = _libraries(environment.required_libraries, paths)
    required_guest = {"LD_LIBRARY_PATH": ":".join(str(path) for path in paths)}
    supplied_guest = dict(guest_environment or {})
    for name, value in supplied_guest.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        ):
            raise QemuCompatibilityError(f"invalid guest environment name: {name!r}")
        if not isinstance(value, str) or "\x00" in value:
            raise QemuCompatibilityError(
                f"invalid guest environment value for {name}"
            )
        if name in required_guest and value != required_guest[name]:
            raise QemuCompatibilityError(
                f"guest environment may not override required {name}"
            )
    guest = {**required_guest, **supplied_guest}
    guest_arguments = tuple(
        argument
        for name, value in sorted(guest.items())
        for argument in ("-E", f"{name}={value}")
    )
    return QemuLaunch(
        argv=(
            str(qemu),
            "-L",
            str(sysroot),
            *guest_arguments,
            str(target),
            *tuple(arguments),
        ),
        environment={"LC_ALL": "C"},
        provenance={
            "qemu": str(qemu),
            "sysroot": str(sysroot),
            "loader": str(loader),
            "libraries": libraries,
            "executable": str(target),
            "guest_environment": dict(sorted(guest.items())),
        },
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_launch(
    launch: QemuLaunch,
    *,
    trace: TraceRecorder | None = None,
    inherited_environment: Mapping[str, str] | None = None,
    require_success: bool = False,
) -> QemuResult:
    process_environment = dict(
        os.environ if inherited_environment is None else inherited_environment
    )
    for host_loader_variable in (
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    ):
        process_environment.pop(host_loader_variable, None)
    process_environment.update(launch.environment)
    try:
        completed = subprocess.run(
            launch.argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=process_environment,
        )
    except (OSError, UnicodeError) as exc:
        if trace is not None:
            trace.record(
                "qemu",
                "execute",
                "failed",
                argv=list(launch.argv),
                error=str(exc),
            )
        raise QemuCompatibilityError(f"cannot execute QEMU launch: {exc}") from exc
    result = QemuResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if trace is not None:
        trace.record(
            "qemu",
            "execute",
            "passed" if result.returncode == 0 else "failed",
            argv=list(launch.argv),
            returncode=result.returncode,
            stderr_sha256=_sha256(result.stderr),
            stdout_sha256=_sha256(result.stdout),
        )
    if require_success and result.returncode != 0:
        raise QemuCompatibilityError(
            f"QEMU launch failed with exit status {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result
