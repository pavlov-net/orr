//! Times ray tracing pipeline creation: a raygen, a miss shader, a hit group and `n`
//! callables of one size, each callable a shader the driver has not seen.
//!
//! - `ray_compile sweep`: cold creation by callable size, count and joined threads.
//! - `ray_compile reuse`: a pipeline, then one linking a subset and one a superset of its
//!   callables, in one process, against cold creations of the same sizes.
//! - `ray_compile store-first DIR SALT` then `ray_compile store-second DIR SALT`: the same
//!   across two processes through a pipeline store at `DIR`.
//! - `ray_compile feedback FILE`: two devices in turn through a pipeline cache file, each
//!   creating one pipeline on one thread and one through a deferred operation, with the
//!   cache hits each creation reports.
//!
//! Every creation logs its time at info level; `RUST_LOG=orr_gpu::vulkan::ray=debug` adds the
//! concurrency the driver reports.

use std::path::Path;
use std::time::{Instant, SystemTime};

use gpu_suite::shaders::ray_compile;
use orr_gpu::{
    CompileThreads, Device, DeviceDesc, HitGroup, RayPipeline, RayPipelineDesc, RayPipelineFlags,
    ShaderRecord, ShaderStage, SpecData,
};

const SIZES: [&str; 3] = ["small", "medium", "large"];

/// A pipeline of `callables` callables of `size`, salted from `salt` on, and the seconds its
/// creation took.
fn create(d: &Device, size: &str, salt: u32, callables: u32, threads: u32) -> (RayPipeline, f64) {
    let specs: Vec<[u8; 4]> = (0..callables + 1)
        .map(|i| (salt.wrapping_add(i)).to_le_bytes())
        .collect();
    fn stage<'a>(entry: &str, spec: &'a [u8; 4]) -> ShaderStage<'a> {
        ShaderStage::new(
            &ray_compile::MODULE,
            entry,
            SpecData {
                entries: ray_compile::Spec::ENTRIES,
                bytes: spec,
            },
        )
        .unwrap()
    }
    let mut stages = vec![
        stage("raygen", &specs[0]),
        stage("miss", &specs[0]),
        stage("hit", &specs[0]),
    ];
    stages.extend(specs[1..].iter().map(|s| stage(size, s)));
    let calls: Vec<ShaderRecord<'_>> = (0..callables).map(|i| ShaderRecord::new(3 + i)).collect();
    let started = Instant::now();
    let p = d
        .create_ray_pipeline(
            &RayPipelineDesc {
                stages: &stages,
                raygens: &[ShaderRecord::new(0)],
                misses: &[ShaderRecord::new(1)],
                hit_groups: &[HitGroup {
                    closest_hit: Some(2),
                    ..HitGroup::default()
                }],
                callables: &calls,
                max_recursion: 1,
                flags: RayPipelineFlags::empty(),
                threads: CompileThreads::Spawned(threads),
            },
            "ray compile",
        )
        .unwrap();
    (p, started.elapsed().as_secs_f64())
}

fn fresh_salt() -> u32 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u32
        & 0x7fff_0000
}

fn sweep(d: &Device) {
    for (size, counts) in [
        ("small", &[8u32, 32, 128][..]),
        ("medium", &[8, 32, 128]),
        ("large", &[4, 16, 64]),
    ] {
        for &n in counts {
            for threads in [1, 4, 8, 16, 32] {
                let (_, s) = create(d, size, fresh_salt(), n, threads);
                log::info!("cold {size} x{n} threads {threads}: {:.0} ms", s * 1e3);
            }
        }
    }
}

fn reuse(d: &Device) {
    for size in SIZES {
        let n = if size == "large" { 16 } else { 32 };
        let salt = fresh_salt();
        let (first, a) = create(d, size, salt, n, 16);
        let (_, sub) = create(d, size, salt, n / 2, 16);
        let (second, sup) = create(d, size, salt, n * 2, 16);
        let (_, same) = create(d, size, salt, n, 16);
        let (_, cold) = create(d, size, fresh_salt(), n * 2, 16);
        log::info!(
            "{size}: x{n} {:.0} ms; subset x{} {:.0} ms; superset x{} {:.0} ms (cold {:.0} ms); same again {:.1} ms; hit handles equal across pipelines: {}",
            a * 1e3,
            n / 2,
            sub * 1e3,
            n * 2,
            sup * 1e3,
            cold * 1e3,
            same * 1e3,
            first.hit_handle(0) == second.hit_handle(0)
        );
    }
}

fn store_device(dir: &Path) -> Device {
    Device::new(&DeviceDesc {
        name: "ray compile",
        pipeline_store: Some(dir),
        ..DeviceDesc::default()
    })
    .unwrap()
}

fn main() {
    gpu_suite::log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("sweep") => sweep(gpu_suite::device()),
        Some("reuse") => reuse(gpu_suite::device()),
        Some("feedback") => {
            let path = Path::new(&args[1]);
            let _ = std::fs::remove_file(path);
            let salt = fresh_salt();
            for round in ["cold", "warm"] {
                let d = Device::new(&DeviceDesc {
                    name: "ray compile",
                    pipeline_cache: Some(path),
                    ..DeviceDesc::default()
                })
                .unwrap();
                for threads in [1, 16] {
                    let before = d.pipeline_cache_report().hits;
                    let (_, s) = create(&d, "small", salt + threads * 1000, 4, threads);
                    log::info!(
                        "{round}, {threads} threads: {:.1} ms, hit reported {}",
                        s * 1e3,
                        d.pipeline_cache_report().hits > before
                    );
                }
                d.save_pipeline_cache().unwrap();
            }
        }
        Some(mode @ ("store-first" | "store-second")) => {
            let d = store_device(Path::new(&args[1]));
            let salt: u32 = args[2].parse().unwrap();
            for (k, size) in SIZES.iter().enumerate() {
                let salt = salt + k as u32 * 1000;
                let n = if *size == "large" { 16 } else { 32 };
                if mode == "store-first" {
                    let (_, s) = create(&d, size, salt, n, 16);
                    log::info!("{size} x{n}: {:.0} ms", s * 1e3);
                } else {
                    for (what, count) in [("same", n), ("subset", n / 2), ("superset", n * 2)] {
                        let (_, s) = create(&d, size, salt, count, 16);
                        log::info!("{size} {what} x{count}: {:.0} ms", s * 1e3);
                    }
                }
            }
            log::info!("{:?}", d.pipeline_cache_report());
        }
        _ => log::error!(
            "usage: ray_compile sweep | reuse | store-first DIR SALT | store-second DIR SALT | feedback FILE"
        ),
    }
}
