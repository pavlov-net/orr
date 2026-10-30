//! CPU probes for the cook (PRB-7). One subcommand per question:
//!
//! - `clod <glb> [node]`: clodBuild per section with dilation on double-sided sections; time,
//!   peak RAM, callback copies vs the final position buffer, crack check on cuts, hierarchy bytes.
//! - `clod-zorah <gltf> <n>`: the same over the n largest Zorah meshes, plus hierarchy bytes
//!   extrapolated over every Zorah mesh.
//! - `codec <dir>`: resident pages with meshlet + vertex codec + zstd vs zstd alone; zstd chunk
//!   loss on mesh blobs.
//! - `parts <glb>`: part-locked vs unlocked simplification on a rigid-part tree.
//! - `tex <dir>`, `bc6h <dir>`: ctt encode determinism at jobs 1 and N, tile chunk loss, BC6H
//!   intel vs amd error.
//! - `gltf <files..>`: `gltf::Gltf::from_slice` coverage and the pre-parse workaround.
//! - `scatter <dir>`: placement records in the chunk files.
//! - `dup <dir>`: repeated part and component geometry.
//! - `rename <dir>`: share-delete open, rename over, old handle reads.
//!
//! Build: `cargo build --release --target-dir ../../target/probes`.

mod clod;
mod codec;
mod dup;
mod gl;
mod gltfcov;
mod parts;
mod rename;
mod scatter;
mod tex;

use std::time::Instant;

#[repr(C)]
#[derive(Default)]
struct ProcessMemoryCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set_size: usize,
    working_set_size: usize,
    quota_peak_paged_pool_usage: usize,
    quota_paged_pool_usage: usize,
    quota_peak_non_paged_pool_usage: usize,
    quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(h: isize, p: *mut ProcessMemoryCounters, cb: u32) -> i32;
}

/// (peak working set, peak private bytes) of this process, MiB.
pub fn peak_mib() -> (f64, f64) {
    let mut c = ProcessMemoryCounters { cb: size_of::<ProcessMemoryCounters>() as u32, ..Default::default() };
    // SAFETY: c is a correctly sized PROCESS_MEMORY_COUNTERS.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.peak_working_set_size as f64 / 1048576.0, c.peak_pagefile_usage as f64 / 1048576.0)
}

pub fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rest = &a[2..];
    match a.get(1).map(String::as_str) {
        Some("clod") => clod::run(rest),
        Some("clod-one") => clod::run_one(rest),
        Some("clod-zorah") => clod::run_zorah(rest),
        Some("codec") => codec::run(rest),
        Some("dict") => codec::run_dict(rest),
        Some("parts") => parts::run(rest),
        Some("tex") => tex::run(rest),
        Some("bc6h") => tex::run_bc6h(rest),
        Some("gltf") => gltfcov::run(rest),
        Some("scatter") => scatter::run(rest),
        Some("dup") => dup::run(rest),
        Some("rename") => rename::run(rest),
        _ => panic!("usage: see main.rs"),
    }
}
