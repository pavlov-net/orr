#!/usr/bin/env bash
# Fetches the pinned third-party tools into .tools/ (ignored by git):
#   slang-2026.20/   Slang release for the host, checked against its sha256
#   openpbr-bsdf/    adobe/openpbr-bsdf at a pinned commit
#   glm/             GLM 1.0.1 at its tag commit
# Build scripts never download; run this once after cloning and after a pin changes.
#
# SLANG_ASSET=glibc-2.28 selects the Linux build for glibc older than 2.34.
# ORR_TOOLS overrides the tools directory, SLANG_DEST the Slang directory inside it
# (for example a separate Linux copy when WSL shares the tree with Windows).
set -euo pipefail

slang_version=2026.20
openpbr_commit=c91aad1d1ce1693e803f039d7c92c2965c4eb013
glm_commit=0af55ccecd98d4e5a8d1fad7de25ba429d60e863

root="$(cd "$(dirname "$0")/.." && pwd)"
tools="${ORR_TOOLS:-$root/.tools}"
dest="${SLANG_DEST:-$tools/slang-$slang_version}"

case "$(uname -s)" in
  Linux)
    case "${SLANG_ASSET:-default}" in
      default)
        asset=slang-$slang_version-linux-x86_64.tar.gz
        sum=e863e568c204810d6e8bc31852f056aef6763c55ea9925db2db7affa060347c9 ;;
      glibc-2.28)
        asset=slang-$slang_version-linux-x86_64-glibc-2.28.tar.gz
        sum=84ebd777968caceea26b21bfd014341078be6047be6c5a963a69f984f896cebd ;;
      *) echo "unknown SLANG_ASSET '$SLANG_ASSET' (default or glibc-2.28)" >&2; exit 1 ;;
    esac ;;
  MINGW*|MSYS*|CYGWIN*)
    asset=slang-$slang_version-windows-x86_64.zip
    sum=e6dd32bbddf1720ebb4e679d55e9635ce792865a5bd84f3d2cd9af28647c0174 ;;
  *) echo "unsupported host $(uname -s)" >&2; exit 1 ;;
esac

if [ "$(cat "$dest/.asset-sha256" 2>/dev/null)" = "$sum" ]; then
  echo "slang $slang_version ($asset) already in $dest"
else
  mkdir -p "$tools/dl"
  file="$tools/dl/$asset"
  if [ ! -f "$file" ]; then
    curl -fsSL --retry 3 -o "$file.part" \
      "https://github.com/shader-slang/slang/releases/download/v$slang_version/$asset"
    mv "$file.part" "$file"
  fi
  if ! echo "$sum  $file" | sha256sum -c --quiet -; then
    rm -f "$file"
    exit 1
  fi
  rm -rf "$dest"
  mkdir -p "$dest"
  case "$asset" in
    *.zip) unzip -q "$file" -d "$dest" ;;
    *) tar -xzf "$file" -C "$dest" ;;
  esac
  echo "$sum" > "$dest/.asset-sha256"
  echo "slang $slang_version ($asset) in $dest"
fi

# Git checks every object against its hash, so the commit id is the digest.
fetch_commit() {
  local url="$1" commit="$2" dir="$3"
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    echo "$(basename "$dir") $commit already in $dir"
    return
  fi
  rm -rf "$dir"
  git init -q "$dir"
  git -C "$dir" fetch -q --depth 1 "$url" "$commit"
  git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
  echo "$(basename "$dir") $commit in $dir"
}

fetch_commit https://github.com/adobe/openpbr-bsdf "$openpbr_commit" "$tools/openpbr-bsdf"
fetch_commit https://github.com/g-truc/glm "$glm_commit" "$tools/glm"
