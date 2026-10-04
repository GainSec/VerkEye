#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${VERKEYE_CORPUS_ARCHIVE:-$project_dir/../bull-cam-2-non512/evidence/approot/cb62-specimen2-approot-ml-corpus-20260927.tar.zst}"
member="var/cache/cvproc/yolov6n_hor.bin"
destination="$project_dir/fixtures/models/yolov6n_hor.bin"
expected_size="5561124"
expected_sha256="eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"

if [[ ! -f "$archive" ]]; then
  echo "missing preserved corpus: $archive" >&2
  exit 1
fi

stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/verkeye-model.XXXXXX")"
trap 'rm -rf "$stage_dir"' EXIT

tar -xf "$archive" -C "$stage_dir" "$member"
extracted="$stage_dir/$member"

actual_size="$(wc -c < "$extracted" | tr -d ' ')"
actual_sha256="$(shasum -a 256 "$extracted" | awk '{print $1}')"

if [[ "$actual_size" != "$expected_size" ]]; then
  echo "size mismatch: expected $expected_size, got $actual_size" >&2
  exit 1
fi

if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "SHA-256 mismatch: expected $expected_sha256, got $actual_sha256" >&2
  exit 1
fi

mkdir -p "$(dirname "$destination")"
install -m 0644 "$extracted" "$destination"
printf 'extracted %s\nsize=%s\nsha256=%s\n' "$destination" "$actual_size" "$actual_sha256"
