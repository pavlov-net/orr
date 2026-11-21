#!/usr/bin/env bash
# Fetches the pinned third-party tools into .tools/ (ignored by git):
#   slang-VERSION/   the Slang release slang_version names, for the host (Linux, Windows or
#                    Apple silicon macOS), checked against its sha256
#   openpbr-bsdf/    adobe/openpbr-bsdf at a pinned commit of its main branch
#   glm/             g-truc/glm at glm_tag's commit
#   display/gt7/     Polyphony Digital's GT7 tone-mapping sample (MIT), checked against its
#                    sha256
#   libdeflate-gdeflate/
#                    NVIDIA/libdeflate's gdeflate branch at a pinned commit (MIT and
#                    Apache-2.0): the GDeflate codec of orr_cook's `gdeflate` feature
#   OpenSubdiv-TAG/
#                    PixarAnimationStudios/OpenSubdiv at opensubdiv_tag's commit (Tomorrow
#                    Open Source Technology License 1.0): the Far library of orr_cook's
#                    `subd` feature
# With ORR_VALIDATION_DATA=1 it also fetches the test-only validation data:
#   precomputed_atmospheric_scattering/
#                    ebruneton/precomputed_atmospheric_scattering at a pinned commit: ASTM G173
#                    solar spectrum, ozone cross sections and CIE 1931 2-deg tables, which the
#                    committed spectrum and glare tables are checked against
#   cie/CIE_std_illum_D65.csv
#                    CIE standard illuminant D65 at 1 nm (CIE, CC BY-SA 4.0,
#                    doi:10.25039/CIE.DS.hjfjmt59), checked against its sha256, which the
#                    committed glare weights are checked against
#   sky/pragueskymodel/
#                    PetrVevoda/pragueskymodel at a pinned commit (its solar radiance table)
#   sky/prague/PragueSkyModelGround.dat
#                    the Prague sky model's ground-level dataset (103 MB), checked against its
#                    sha256
#   gltf/NormalTangentMirrorTest.glb
#                    KhronosGroup/glTF-Sample-Assets' model at a pinned commit (CC BY 4.0),
#                    checked against its sha256, whose tangents the cook computes again
#   volumes/smoke.vdb
#                    OpenVDB's smoke1 sample model from the ASWF artifacts, its zip checked
#                    against its sha256
# With ORR_FETCH_DLSS=1 on Windows it also fetches what compiling orr_dlss against NGX needs:
#   DLSS/            NVIDIA/DLSS at dlss_tag's commit: its headers and link libraries, without
#                    the DLLs that running needs, for `DLSS_SDK`
#   vulkan-headers/  KhronosGroup/Vulkan-Headers at vulkan_headers_tag's commit, for `VULKAN_SDK`
# With ORR_FETCH_NTC=1 it also fetches LibNTC for orr_cook's `ntc` feature (NVIDIA RTX SDK
# license):
#   RTXNTC/libraries/RTXNTC-Library/
#                    NVIDIA-RTX/RTXNTC at ntc_tag's commit, with only its RTXNTC-Library
#                    submodule and that library's own submodules checked out; on Windows also
#                    libntc.dll from that release's ntc_windows_zip, checked against its sha256,
#                    in bin/windows-x64/. Elsewhere libntc builds from the checkout, and
#                    ORR_NTC_LIB names it.
# Build scripts never download; run this once after cloning and after a pin changes.
#
# Renovate (renovate.json) opens a pull request when a pin below has a new upstream release
# or, for a commit pinned to a branch, a new head, and updates the commits and Slang's sums
# with it. It can't update ntc_windows_zip and ntc_windows_sum, because the zip's name doesn't
# carry the release tag: an RTXNTC pull request needs both set by hand from the new release.
# The reference data (the Bruneton sky, the Prague sky model and its dataset, the glTF sample,
# CIE D65, the GT7 sample and the smoke volume) is fixed and isn't tracked.
#
# Some SDKs come only from NVIDIA's developer site, which Renovate can't watch. Check these by
# hand:
#   Nsight Perf SDK  NVPERF_SDK_PATH
#   Reflex SDK       REFLEX_SDK
#
# SLANG_ASSET=glibc-2.28 selects the Linux build for glibc older than 2.34.
# ORR_TOOLS overrides the tools directory, SLANG_DEST the Slang directory inside it
# (for example a separate Linux copy when WSL shares the tree with Windows).
set -euo pipefail

# renovate: datasource=github-releases depName=shader-slang/slang
slang_version=2026.22.1
# renovate: datasource=git-refs depName=adobe/openpbr-bsdf packageName=https://github.com/adobe/openpbr-bsdf branch=main
openpbr_commit=c91aad1d1ce1693e803f039d7c92c2965c4eb013
# renovate: datasource=github-releases depName=g-truc/glm
glm_tag=1.0.3
glm_commit=8d1fd52e5ab5590e2c81768ace50c72bae28f2ed
# renovate: datasource=git-refs depName=NVIDIA/libdeflate packageName=https://github.com/NVIDIA/libdeflate branch=gdeflate
gdeflate_commit=8ba9502fb30d2bf728592d121f0d402e40c8cb05
# renovate: datasource=github-releases depName=PixarAnimationStudios/OpenSubdiv
opensubdiv_tag=v3_8_0
opensubdiv_commit=1d90921ea32b9dcae31d30b907cbacefc0c616bf
# renovate: datasource=github-releases depName=NVIDIA/DLSS
dlss_tag=v310.9.16
dlss_commit=9981c6350dc58c0ce21f024fd65e31e851c31689
# renovate: datasource=github-tags depName=KhronosGroup/Vulkan-Headers
vulkan_headers_tag=vulkan-sdk-1.4.363.0
vulkan_headers_commit=6802bb4733b63ed5efd3adb308a6c885ef180ea1
# renovate: datasource=github-releases depName=NVIDIA-RTX/RTXNTC
ntc_tag=v0.10.1-beta
ntc_commit=8311b54b4ae6ff54e488799f18ffd12b40e7742d
ntc_windows_zip=RTXNTC_windows_x64_v0.10.1b.zip
ntc_windows_sum=ce87104df62184fd035b4b56fe6ffea870faf82da86295ec5ee6b1909aed9eed
sky_commit=34f14e745cff948f4ca3157d1b62a445ffa7286f
d65_sum=e76f210bffff3d552ef7113025da5f325d5dfec200dd4b878b1a2f3a507032cb
gt7_sum=94df7e7e310b9423ebb5aaaa29416f376ebbae66e0948fb259f8688314b29822
prague_commit=2385c912e9051c1258013ff8c3ce2e19e10fb917
prague_ground_id=1IflyFZTJxC_N298yXq_2GK4ycIsVJZk6
prague_ground_sum=76bd619dc6dfcbc900c2996436dd6cf68197c03e3982fe5580f2b109ce1c71c2
smoke_sum=dca7a0477e77fac197a969f5f418057fc3f89b12887c35d1fc9c1b0d2892f20f
gltf_samples_commit=8f492390ddc7a2ae221a1369ef560589cf8c70fb
tangent_test_sum=31ce5f3c873fc55531a17ccacf001e3d1d482a508695aa826fa79cc196e0ce78

root="$(cd "$(dirname "$0")/.." && pwd)"
tools="${ORR_TOOLS:-$root/.tools}"
dest="${SLANG_DEST:-$tools/slang-$slang_version}"

case "$(uname -s)" in
  Linux)
    case "${SLANG_ASSET:-default}" in
      default)
        asset=slang-$slang_version-linux-x86_64.tar.gz
        # renovate: datasource=github-release-attachments depName=shader-slang/slang digestVersion=v2026.22.1
        sum=37156099ebaabf62cd92522eba55eb93b8663e65ea3be618ce015d9a8edda6bc ;;
      glibc-2.28)
        asset=slang-$slang_version-linux-x86_64-glibc-2.28.tar.gz
        # renovate: datasource=github-release-attachments depName=shader-slang/slang digestVersion=v2026.22.1
        sum=0c0e243ac15b7878237e6854313f45c791408227d5bc634f3ad1ac487edbfaf5 ;;
      *) echo "unknown SLANG_ASSET '$SLANG_ASSET' (default or glibc-2.28)" >&2; exit 1 ;;
    esac ;;
  Darwin)
    case "$(uname -m)" in
      arm64)
        asset=slang-$slang_version-macos-aarch64.tar.gz
        # renovate: datasource=github-release-attachments depName=shader-slang/slang digestVersion=v2026.22.1
        sum=1d5370a07be409b94685ea4bd84b7bd191313a6b1fae8795f451009bc0dc3fd7 ;;
      *) echo "unsupported macOS architecture $(uname -m) (Apple silicon only)" >&2; exit 1 ;;
    esac ;;
  MINGW*|MSYS*|CYGWIN*)
    asset=slang-$slang_version-windows-x86_64.zip
    # renovate: datasource=github-release-attachments depName=shader-slang/slang digestVersion=v2026.22.1
    sum=e72344f7b0323cec8b039c534caaf4680e7ba629489fc796ea4c7f7adb7deb68 ;;
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

# Git checks every object against its hash, so the commit id is the digest. Paths after the
# directory limit the checkout to them, and only their blobs download.
fetch_commit() {
  local url="$1" commit="$2" dir="$3"
  shift 3
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    echo "$(basename "$dir") $commit already in $dir"
    return
  fi
  rm -rf "$dir"
  git init -q "$dir"
  if [ $# -gt 0 ]; then
    git -C "$dir" remote add origin "$url"
    git -C "$dir" sparse-checkout set "$@"
    git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$commit"
  else
    git -C "$dir" fetch -q --depth 1 "$url" "$commit"
  fi
  git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
  echo "$(basename "$dir") $commit in $dir"
}

fetch_commit https://github.com/adobe/openpbr-bsdf "$openpbr_commit" "$tools/openpbr-bsdf"
fetch_commit https://github.com/g-truc/glm "$glm_commit" "$tools/glm"
fetch_commit https://github.com/NVIDIA/libdeflate "$gdeflate_commit" "$tools/libdeflate-gdeflate"
fetch_commit https://github.com/PixarAnimationStudios/OpenSubdiv "$opensubdiv_commit" \
  "$tools/OpenSubdiv-$opensubdiv_tag"

gt7="$tools/display/gt7/gt7_tone_mapping.cpp"
if echo "$gt7_sum  $gt7" | sha256sum -c --quiet - 2>/dev/null; then
  echo "gt7 sample already in $gt7"
else
  mkdir -p "$(dirname "$gt7")"
  curl -fsSL --retry 3 -o "$gt7.part" https://blog.selfshadow.com/publications/s2025-shading-course/pdi/supplemental/gt7_tone_mapping.cpp
  echo "$gt7_sum  $gt7.part" | sha256sum -c --quiet -
  mv "$gt7.part" "$gt7"
  echo "gt7 sample in $gt7"
fi

if [ "${ORR_FETCH_DLSS:-0}" = 1 ]; then
  fetch_commit https://github.com/NVIDIA/DLSS "$dlss_commit" "$tools/DLSS" \
    include lib/Windows_x86_64/x64
  fetch_commit https://github.com/KhronosGroup/Vulkan-Headers "$vulkan_headers_commit" \
    "$tools/vulkan-headers"
fi

if [ "${ORR_FETCH_NTC:-0}" = 1 ]; then
  # The release commit pins the library through its submodule, which git checks out at exactly
  # that commit. The sparse path leaves out the release's samples and its other submodules.
  fetch_commit https://github.com/NVIDIA-RTX/RTXNTC "$ntc_commit" "$tools/RTXNTC" \
    libraries/RTXNTC-Library
  git -C "$tools/RTXNTC" submodule update -q --init --depth 1 libraries/RTXNTC-Library
  ntc="$tools/RTXNTC/libraries/RTXNTC-Library"
  git -C "$ntc" submodule update -q --init --depth 1
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      dll="$ntc/bin/windows-x64/libntc.dll"
      if [ -f "$dll" ] && [ "$(cat "$dll.zip-sha256" 2>/dev/null)" = "$ntc_windows_sum" ]; then
        echo "libntc $ntc_tag already in $dll"
      else
        mkdir -p "$tools/dl"
        # The tag in the name makes a tag bump that leaves the zip pins stale fail to download,
        # instead of unpacking the old release's cached zip into the new checkout.
        zip="$tools/dl/${ntc_tag}_$ntc_windows_zip"
        if ! echo "$ntc_windows_sum  $zip" | sha256sum -c --quiet - 2>/dev/null; then
          curl -fsSL --retry 3 -o "$zip.part" \
            "https://github.com/NVIDIA-RTX/RTXNTC/releases/download/$ntc_tag/$ntc_windows_zip"
          echo "$ntc_windows_sum  $zip.part" | sha256sum -c --quiet -
          mv "$zip.part" "$zip"
        fi
        unzip -o -q "$zip" bin/windows-x64/libntc.dll -d "$ntc"
        echo "$ntc_windows_sum" > "$dll.zip-sha256"
        echo "libntc $ntc_tag in $dll"
      fi ;;
  esac
fi

if [ "${ORR_VALIDATION_DATA:-0}" = 1 ]; then
  fetch_commit https://github.com/ebruneton/precomputed_atmospheric_scattering "$sky_commit" \
    "$tools/precomputed_atmospheric_scattering"
  d65="$tools/cie/CIE_std_illum_D65.csv"
  if echo "$d65_sum  $d65" | sha256sum -c --quiet - 2>/dev/null; then
    echo "CIE D65 already in $d65"
  else
    mkdir -p "$(dirname "$d65")"
    curl -fsSL --retry 3 -o "$d65.part" https://files.cie.co.at/Publications-datasets/CIE_std_illum_D65.csv
    echo "$d65_sum  $d65.part" | sha256sum -c --quiet -
    mv "$d65.part" "$d65"
    echo "CIE D65 in $d65"
  fi
  fetch_commit https://github.com/PetrVevoda/pragueskymodel "$prague_commit" "$tools/sky/pragueskymodel"
  ground="$tools/sky/prague/PragueSkyModelGround.dat"
  if echo "$prague_ground_sum  $ground" | sha256sum -c --quiet - 2>/dev/null; then
    echo "prague ground dataset already in $ground"
  else
    mkdir -p "$(dirname "$ground")"
    curl -fsSL --retry 3 -o "$ground.part"       "https://drive.usercontent.google.com/download?id=$prague_ground_id&export=download&confirm=t"
    echo "$prague_ground_sum  $ground.part" | sha256sum -c --quiet -
    mv "$ground.part" "$ground"
    echo "prague ground dataset in $ground"
  fi
  tangent_test="$tools/gltf/NormalTangentMirrorTest.glb"
  if echo "$tangent_test_sum  $tangent_test" | sha256sum -c --quiet - 2>/dev/null; then
    echo "tangent test model already in $tangent_test"
  else
    mkdir -p "$(dirname "$tangent_test")"
    curl -fsSL --retry 3 -o "$tangent_test.part" \
      "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/$gltf_samples_commit/Models/NormalTangentMirrorTest/glTF-Binary/NormalTangentMirrorTest.glb"
    echo "$tangent_test_sum  $tangent_test.part" | sha256sum -c --quiet -
    mv "$tangent_test.part" "$tangent_test"
    echo "tangent test model in $tangent_test"
  fi
  smoke="$tools/volumes/smoke.vdb"
  if [ -f "$smoke" ] && [ "$(cat "$smoke.zip-sha256" 2>/dev/null)" = "$smoke_sum" ]; then
    echo "smoke1 volume already in $smoke"
  else
    mkdir -p "$tools/dl" "$tools/volumes"
    zip="$tools/dl/smoke1.vdb-1.0.0.zip"
    if ! echo "$smoke_sum  $zip" | sha256sum -c --quiet - 2>/dev/null; then
      curl -fsSL --retry 3 -o "$zip.part" https://artifacts.aswf.io/io/aswf/openvdb/models/smoke1.vdb/1.0.0/smoke1.vdb-1.0.0.zip
      echo "$smoke_sum  $zip.part" | sha256sum -c --quiet -
      mv "$zip.part" "$zip"
    fi
    unzip -o -q "$zip" smoke.vdb -d "$tools/volumes"
    echo "$smoke_sum" > "$smoke.zip-sha256"
    echo "smoke1 volume in $smoke"
  fi
fi
