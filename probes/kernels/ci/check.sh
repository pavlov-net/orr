#!/usr/bin/env bash
# CI toolchain check on a Linux image: fetch the pinned slangc, run it, list the glibc symbol
# versions its binaries need, then build and run the cc-built Slang C++ crate (OpenPBR wrapper
# and packing, checked against the OpenPBR C++ reference). On failure with the default asset,
# reruns with the glibc-2.28 asset.
# SLANG_DEST and CARGO_TARGET_DIR pass through; ORR_TOOLS locates openpbr-bsdf and glm.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
dest="${SLANG_DEST:-$root/.tools/slang-2026.20}"

run() {
  local asset="$1"
  echo "== asset: $asset (glibc $(ldd --version | head -1 | awk '{print $NF}') on this image)"
  SLANG_ASSET="$asset" SLANG_DEST="$dest" bash "$here/fetch-slang.sh" || return 1
  "$dest/bin/slangc" -version || return 1
  for f in "$dest"/bin/slangc "$dest"/lib/*.so*; do
    [ -f "$f" ] || continue
    need=$(objdump -T "$f" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
    echo "  $(basename "$f"): highest $need"
  done
  (cd "$here/slang-cpp" && ORR_SLANGC="$dest/bin/slangc" cargo run --release) || return 1
}

run default || run glibc-2.28
