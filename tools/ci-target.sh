#!/usr/bin/env bash
# Makes a target/ restored from CI's cache safe to build on for this checkout.
#
# A checkout gives every file a new mtime, which would dirty every workspace crate and build
# script. So every file and directory goes to one old date, and only what differs from the
# commit the cache was built from gets rebuilt: its package is cleaned, which also covers an
# input a build script doesn't declare, and the file goes to now. Directories count because
# Cargo includes them when a build script watches one.
set -euo pipefail

cd "$(dirname "$0")/.."
export LC_ALL=C
manifest=target/ci-manifest

find . \( -path ./target -o -path ./.git \) -prune -o -exec touch -h -t 200001010000 {} +

current=$(git -c core.quotePath=false ls-files -s)
if [ -f "$manifest" ]; then
  changed=$(comm -3 <(sort "$manifest") <(sort <<<"$current") | awk -F'\t' '{print $NF}' | sort -u)
else
  rm -rf target
  changed=
fi

packages=()
while IFS= read -r path; do
  [ -n "$path" ] || continue
  [ ! -e "$path" ] || touch -h "$path"
  case "$path" in
    crates/*|tests/*) dir=$(cut -d/ -f1-2 <<<"$path") ;;
    examples/*|xtask/*) dir=${path%%/*} ;;
    *) continue ;;
  esac
  [ -f "$dir/Cargo.toml" ] || continue
  packages+=("-p" "$(sed -n 's/^name = "\(.*\)"$/\1/p' "$dir/Cargo.toml" | head -1)")
done <<<"$changed"

if [ ${#packages[@]} -gt 0 ]; then
  cargo clean "${packages[@]}"
fi

mkdir -p target
printf '%s\n' "$current" > "$manifest"
