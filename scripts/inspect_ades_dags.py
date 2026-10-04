#!/usr/bin/env python3
"""Decode recovered CB62 DAG suffixes with the pinned vendor ADES tooling."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Mapping

from verkeye.compat.ades import parse_cavalry_verbose
from verkeye.compat.ades_runtime import load_runtime_spec
from verkeye.cv22.ades_dag import (
    AdesDagError,
    ades_dag_document,
    extract_dagbin,
    parse_ades_dag_transcript,
)
from verkeye.cv22.program import FOOTER_SIZE


_SPLIT = re.compile(r"_split_(?P<index>\d+)\.dvi$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--docker-command", default="docker")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_runtime_spec(args.runtime)
    manifest_path = args.workspace / "cavalry-verbose.txt"
    manifest = parse_cavalry_verbose(manifest_path.read_text(encoding="utf-8"))
    split_manifest = {split.index: split for split in manifest.splits}
    indexed_paths = _verified_dvi_paths(args.workspace, spec.dvi_sha256)
    if set(indexed_paths) != set(split_manifest):
        raise SystemExit("DVI and Cavalry split sets differ")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    graphs: list[dict[str, object]] = []
    families: Counter[str] = Counter()
    opcodes: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="verkeye-ades-dag-") as temporary:
        dag_directory = Path(temporary)
        for split_index, dvi_path in sorted(indexed_paths.items()):
            manifest_split = split_manifest[split_index]
            dvi = dvi_path.read_bytes()
            if len(dvi) != manifest_split.image_size:
                raise SystemExit(
                    f"DVI size differs from manifest for split {split_index}"
                )
            dagbin = extract_dagbin(dvi, manifest_split.dag_offset)
            program = dagbin[:-FOOTER_SIZE]
            dag_path = dag_directory / f"split-{split_index:02d}.dagbin"
            dag_path.write_bytes(dagbin)

            discovery_commands = f"ldag /dag/{dag_path.name}\nquit\n"
            discovery = _run_ades(
                spec=spec,
                dag_directory=dag_directory,
                docker_command=args.docker_command,
                commands=discovery_commands,
            )
            discovered = parse_ades_dag_transcript(
                discovery.decode("utf-8", errors="strict"), require_configs=False
            )
            commands = "".join(
                [
                    f"ldag /dag/{dag_path.name}\n",
                    *(f"opinfo {operator.operator_id}\n" for operator in discovered.operators),
                    "quit\n",
                ]
            )
            transcript = _run_ades(
                spec=spec,
                dag_directory=dag_directory,
                docker_command=args.docker_command,
                commands=commands,
            )
            try:
                graph = parse_ades_dag_transcript(
                    transcript.decode("utf-8", errors="strict"),
                    require_configs=True,
                )
            except (UnicodeDecodeError, AdesDagError) as error:
                raise SystemExit(
                    f"cannot parse ADES transcript for split {split_index}: {error}"
                ) from error
            if graph.dag_size != len(dagbin):
                raise SystemExit(
                    f"ADES DAG size differs for split {split_index}: "
                    f"{graph.dag_size} != {len(dagbin)}"
                )

            discovery_path = args.raw_dir / f"split-{split_index:02d}-discovery.log"
            transcript_path = args.raw_dir / f"split-{split_index:02d}-opinfo.log"
            _atomic_bytes(discovery_path, discovery)
            _atomic_bytes(transcript_path, transcript)
            graph_document = ades_dag_document(graph)
            for operator in graph.operators:
                families[operator.type_name] += 1
                previous = opcodes.setdefault(operator.type_name, operator.opcode)
                if previous != operator.opcode:
                    raise SystemExit(
                        f"operator family {operator.type_name} has multiple opcodes"
                    )
            graph_document.update(
                {
                    "split_index": split_index,
                    "source": {
                        "name": dvi_path.name,
                        "size": len(dvi),
                        "sha256": _digest(dvi),
                        "dag_offset": manifest_split.dag_offset,
                    },
                    "dagbin_sha256": _digest(dagbin),
                    "program_size": len(program),
                    "program_sha256": _digest(program),
                    "discovery_transcript": _evidence_member(discovery_path, discovery),
                    "opinfo_transcript": _evidence_member(transcript_path, transcript),
                }
            )
            graphs.append(graph_document)
            print(
                f"split {split_index}: {len(graph.operators)} operators, "
                f"{len(graph.links)} links, {len(graph.inputs)} primary inputs"
            )

    document = {
        "schema": "verkeye.cv22.ades-dag-semantics.v1",
        "claim_boundary": (
            "The digest-pinned Ambarella ADES decoder proves DAG operator types, "
            "opcodes, connectivity, descriptor dimensions, and serialized "
            "configuration values. Numerical behavior and VMEM parameter mapping "
            "remain unresolved until independent oracle comparison passes."
        ),
        "model_sha256": spec.model_sha256,
        "runtime_spec_sha256": _digest(args.runtime.read_bytes()),
        "cavalry_verbose_sha256": _digest(manifest_path.read_bytes()),
        "ades_runtime": {
            "image": spec.image,
            "platform": spec.platform,
            "container_user": spec.container_user,
            "toolchain_env": spec.toolchain_env,
        },
        "summary": {
            "split_count": len(graphs),
            "operator_count": sum(families.values()),
            "operator_families": [
                {
                    "type_name": name,
                    "opcode": opcodes[name],
                    "count": families[name],
                }
                for name in sorted(families)
            ],
            "decoded_structure": True,
            "numerical_semantics_proved": False,
            "parameter_mapping_proved": False,
        },
        "splits": graphs,
    }
    _atomic_json(args.json_out, document)
    print(json.dumps(document["summary"], indent=2, sort_keys=True))
    return 0


def _verified_dvi_paths(
    workspace: Path, expected: Mapping[str, str]
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for name, digest in expected.items():
        match = _SPLIT.search(name)
        if match is None:
            raise SystemExit(f"cannot determine split index from {name}")
        split_index = int(match.group("index"))
        path = workspace / "parse" / name
        observed = _digest(path.read_bytes())
        if observed != digest:
            raise SystemExit(
                f"DVI SHA-256 mismatch for {name}: expected {digest}, got {observed}"
            )
        paths[split_index] = path
    return paths


def _run_ades(*, spec, dag_directory: Path, docker_command: str, commands: str) -> bytes:
    shell_command = (
        "set +u; "
        f"source {shlex.quote(spec.toolchain_env)} >/dev/null; "
        f"printf %s {shlex.quote(commands)} | ades"
    )
    argv = (
        docker_command,
        "run",
        "--rm",
        "--platform",
        spec.platform,
        "--network",
        "none",
        "--user",
        spec.container_user,
        "--volume",
        f"{dag_directory.resolve()}:/dag:ro",
        "--workdir",
        "/tmp",
        spec.image,
        "bash",
        "-lc",
        shell_command,
    )
    result = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"ADES decoder failed with status {result.returncode}: "
            f"{result.stdout.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def _evidence_member(path: Path, data: bytes) -> dict[str, object]:
    try:
        display = str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        display = path.name
    return {"path": display, "size": len(data), "sha256": _digest(data)}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, document: object) -> None:
    data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, data)


if __name__ == "__main__":
    raise SystemExit(main())
