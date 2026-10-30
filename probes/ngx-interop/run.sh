#!/usr/bin/env bash
# Runs every case into <out> and prints the analysis. Needs DLSS_SDK at build time, uv for
# analyze.py. With PROBE_BUSY set, refuses to run while a process matching it is listed.
set -u
out=${1:?out dir}
here=$(cd "$(dirname "$0")" && pwd)
exe="$here/../../target/probes/release/ngx-interop.exe"
mkdir -p "$out"
if [ -n "${PROBE_BUSY:-}" ] && tasklist | grep -qi "$PROBE_BUSY"; then echo "$PROBE_BUSY is running; not touching the GPU"; exit 1; fi
quiet() { grep -v "NGXSecureLoad\|IsFeatureDenied\|NGXLoadLibrary\|older API\|Loaded from" | cut -c1-1500; }
val() { VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation "$@"; }

"$exe" info 2>&1 | grep -v NGXLOG
for heap in none rebind restore restore-norebind; do echo "== interop general $heap render"; val "$exe" interop general "$heap" render "$out" 2>&1 | quiet; done
for layout in general readonly; do for q in render compute; do echo "== interop $layout restore $q"; val "$exe" interop "$layout" restore "$q" "$out" 2>&1 | quiet; done; done
for layout in general readonly; do echo "== interop $layout restore render, unifiedImageLayouts off"; PROBE_NO_UNIFIED=1 val "$exe" interop "$layout" restore render "$out" 2>&1 | quiet; done
for mode in "wide plain" "wide restore" "clip restore"; do echo "== fg $mode"; val "$exe" fg $mode "$out" 2>&1 | quiet; done
for space in 709 2020; do
    val "$exe" color "$space" 1 std "$out" 2>&1 | grep validation
    for seed in 2 3 4; do "$exe" color "$space" "$seed" std "$out" 2>&1 | grep -i "panic"; done
done
for j in flip flipx flipy; do "$exe" color 709 1 "$j" "$out" 2>&1 | grep -i panic; done
for preset in default E F; do for guide in 0 1; do
    echo "== dof $preset guide $guide"
    val "$exe" dof "$preset" "$guide" "$out" > "$out/dof_${preset}_$guide.log" 2>&1
    grep -v NGXLOG "$out/dof_${preset}_$guide.log" | quiet
done; done
"$exe" exposure "$out" 2>&1 | grep -i "panic\|validation"
"$exe" cost 2>&1 | grep -v NGXLOG | quiet
cd "$out" && for c in interop color fg dof exposure; do echo "== analyze $c"; uv run --no-project --with flip-evaluator --with numpy python -I "$here/analyze.py" "$c" "$out"; done
