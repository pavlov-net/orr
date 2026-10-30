#!/usr/bin/env bash
# Downloads the pinned Slang release for the host, checks its sha256, extracts it.
# SLANG_ASSET picks the Linux asset: default (the plain build) or glibc-2.28 / glibc-2.27.
# SLANG_DEST overrides the extract directory (default <workspace>/.tools/slang-2026.20).
set -euo pipefail

version=2026.20
root="$(cd "$(dirname "$0")/../../.." && pwd)"
tools="$root/.tools"
dest="${SLANG_DEST:-$tools/slang-$version}"

case "$(uname -s)" in
  Linux)
    case "${SLANG_ASSET:-default}" in
      default) asset=slang-$version-linux-x86_64.tar.gz
               sum=e863e568c204810d6e8bc31852f056aef6763c55ea9925db2db7affa060347c9 ;;
      glibc-2.28) asset=slang-$version-linux-x86_64-glibc-2.28.tar.gz
               sum=84ebd777968caceea26b21bfd014341078be6047be6c5a963a69f984f896cebd ;;
      glibc-2.27) asset=slang-$version-linux-x86_64-glibc-2.27.tar.gz
               sum=b831477e994dee2fb36a3e83c54ad89cbfeefd47b4d973b2d8bed65b64780996 ;;
      *) echo "unknown SLANG_ASSET ${SLANG_ASSET}" >&2; exit 1 ;;
    esac ;;
  MINGW*|MSYS*|CYGWIN*)
    asset=slang-$version-windows-x86_64.zip
    sum=e6dd32bbddf1720ebb4e679d55e9635ce792865a5bd84f3d2cd9af28647c0174 ;;
  *) echo "unsupported host $(uname -s)" >&2; exit 1 ;;
esac

mkdir -p "$tools/dl"
file="$tools/dl/$asset"
if [ ! -f "$file" ]; then
  curl -fL --retry 3 -o "$file.part" "https://github.com/shader-slang/slang/releases/download/v$version/$asset"
  mv "$file.part" "$file"
fi
echo "$sum  $file" | sha256sum -c -

rm -rf "$dest"
mkdir -p "$dest"
case "$asset" in
  *.zip) unzip -q "$file" -d "$dest" ;;
  *) tar -xzf "$file" -C "$dest" ;;
esac
echo "slang $version ($asset) in $dest"
