#!/usr/bin/env bash
# Regenerates each crate's private Windows bindings with the windows-bindgen version pinned in
# tools/windows-bindings/Cargo.toml, from the filters in tools/windows-bindings/filters. Builds
# the generator in .tools (ORR_TOOLS, else the nearest .tools at or above the repository).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
tools="${ORR_TOOLS:-}"
if [ -z "$tools" ]; then
  dir="$root"
  while [ ! -d "$dir/.tools" ] && [ "$dir" != "$(dirname "$dir")" ]; do
    dir="$(dirname "$dir")"
  done
  tools="$dir/.tools"
  [ -d "$tools" ] || tools="$root/.tools"
fi

CARGO_TARGET_DIR="$tools/windows-bindings" cargo run --quiet --locked --release \
  --manifest-path "$root/tools/windows-bindings/Cargo.toml" -- "$root"
