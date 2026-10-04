#!/usr/bin/env bash
set -euo pipefail

qemu_repository="https://gitlab.com/qemu-project/qemu.git"
qemu_commit="f7ada39edacaa5c26b30e98b94017b0b2ccbcf94"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_DIR BUILD_DIR TOOLS_VENV" >&2
  exit 2
fi

source_dir=$1
build_dir=$2
tools_venv=$3
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
patch_file="$project_dir/third_party/patches/qemu-or1k-cv22.patch"

if [[ ! -d "$source_dir/.git" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "source path exists but is not a git checkout: $source_dir" >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "$source_dir")"
  git clone --filter=blob:none "$qemu_repository" "$source_dir"
fi

if [[ "$(git -C "$source_dir" remote get-url origin)" != "$qemu_repository" ]]; then
  echo "unexpected QEMU origin" >&2
  exit 1
fi
git -C "$source_dir" fetch origin "$qemu_commit"
git -C "$source_dir" checkout --detach "$qemu_commit"
if ! git -C "$source_dir" diff --quiet || [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=no)" ]]; then
  echo "QEMU source has tracked modifications before patching" >&2
  exit 1
fi
git -C "$source_dir" apply --check "$patch_file"
git -C "$source_dir" apply "$patch_file"

python3 -m venv "$tools_venv"
"$tools_venv/bin/python" -m pip install --disable-pip-version-check \
  meson==1.12.1 ninja==1.13.2

mkdir -p "$build_dir"
(
  cd "$build_dir"
  PATH="$tools_venv/bin:$PATH" "$source_dir/configure" \
    --target-list=or1k-softmmu \
    --without-default-features \
    --enable-tcg \
    --enable-system \
    --disable-werror
)
PATH="$tools_venv/bin:$PATH" ninja -C "$build_dir" qemu-system-or1k

qemu_binary="$build_dir/qemu-system-or1k"
if ! "$qemu_binary" -cpu help | grep -Fq "ambarella-cv22"; then
  echo "built QEMU does not expose the ambarella-cv22 CPU model" >&2
  exit 1
fi
printf '%s\n' "$qemu_binary"
