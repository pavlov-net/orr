//! Probe: the loading path end to end (contracts.md 33), raw ash, against a concurrent
//! synthetic frame on the render queue.
//!
//! `loading <case>` with case one of:
//! - `info`: device facts the loading path depends on.
//! - `bar`: CPU writes into Upload (resizable BAR) from 1/4/8 workers; zstd decode into cached
//!   scratch then copy vs straight into BAR; page decode and tile writes into BAR; GPU read
//!   bandwidth of Upload vs DeviceLocal blocks.
//! - `hic`: host image copy of BC7 mips and tiles from 1/4/8 workers; per-format host copy
//!   performance query; memory requirements and sampling cost with and without HOST_TRANSFER
//!   usage and concurrent sharing.
//! - `clas`: loading-queue position expansion + CLAS builds + copy stream against the frame;
//!   per-page micromap builds.
//! - `stale`: address-based image copy followed by readback in one submission, 1000 runs.
//! - `queues [launches]`: three queues from three threads with concurrent copies, one fresh
//!   process per launch (`queues-child` is the child).
//! - `disk [GiB]`: IoRing vs positioned unbuffered reads of 64-256 KiB aligned blobs.
//! - `decomp`: VK_EXT_memory_decompression GDeflate exactness and throughput.
//!
//! Each case prints `ROW` lines (case | metric | value) and a `RESULT` line.
//! Build from this directory with `cargo build --release --target-dir ../../target/probes`.

#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments, clippy::missing_safety_doc)]

mod bar;
mod clas;
mod decomp;
mod disk;
mod frame;
mod gpu;
mod hic;
mod issues;

use std::sync::Mutex;

#[macro_export]
macro_rules! spv {
    ($name:literal) => {
        ash::util::read_spv(&mut std::io::Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))[..])).expect("SPIR-V")
    };
}

pub fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

static CASE: Mutex<String> = Mutex::new(String::new());

pub fn fail(msg: &str) -> ! {
    let case = CASE.lock().map(|c| c.clone()).unwrap_or_default();
    let (e, w) = gpu::validation_report();
    println!("RESULT {case} FAIL {msg} (validation errors={e} warnings={w})");
    std::process::exit(2);
}

pub fn row(metric: &str, value: impl std::fmt::Display) {
    let case = CASE.lock().unwrap().clone();
    println!("ROW {case} | {metric} | {value}");
}

pub fn median(v: &mut [f64]) -> f64 {
    pct(v, 0.5)
}

pub fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// CPU seconds (user + kernel) of the calling thread.
pub fn thread_cpu() -> f64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let mut t = [FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    unsafe {
        GetThreadTimes(GetCurrentThread(), &mut t[0], &mut t[1], &mut t[2], &mut t[3]);
    }
    let f = |x: FILETIME| ((u64::from(x.dwHighDateTime) << 32) | u64::from(x.dwLowDateTime)) as f64 * 1e-7;
    f(t[2]) + f(t[3])
}

/// SplitMix64.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Bytes moved and CPU seconds spent by a set of workers over a wall-clock window.
#[derive(Default, Clone, Copy)]
pub struct Work {
    pub bytes: f64,
    pub cpu: f64,
    pub wall: f64,
}

impl Work {
    /// Sequential windows: everything adds.
    pub fn sum(ws: &[Work]) -> Work {
        let mut s = Work::default();
        for w in ws {
            s.bytes += w.bytes;
            s.cpu += w.cpu;
            s.wall += w.wall;
        }
        s
    }
    pub fn gbps(&self) -> f64 {
        self.bytes / self.wall / 1e9
    }
    pub fn cores(&self) -> f64 {
        self.cpu / self.wall
    }
    pub fn per_core(&self) -> f64 {
        self.bytes / self.cpu.max(1e-9) / 1e9
    }
    pub fn text(&self) -> String {
        format!("{:.2} GB/s, {:.2} cores, {:.2} GB/s per core", self.gbps(), self.cores(), self.per_core())
    }
}

/// Runs `workers` threads for `dur`; each calls `step(worker, iteration)` until time is up;
/// `step` returns the bytes it produced.
pub fn run_workers(workers: usize, dur: std::time::Duration, step: &(dyn Fn(usize, u64) -> u64 + Sync)) -> Work {
    let start = std::time::Instant::now();
    let parts: Vec<Work> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers)
            .map(|w| {
                s.spawn(move || {
                    let c0 = thread_cpu();
                    let t0 = std::time::Instant::now();
                    let mut bytes = 0u64;
                    let mut i = 0u64;
                    while t0.elapsed() < dur {
                        bytes += step(w, i);
                        i += 1;
                    }
                    Work { bytes: bytes as f64, cpu: thread_cpu() - c0, wall: 0.0 }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut w = Work { wall: start.elapsed().as_secs_f64(), ..Work::default() };
    for p in parts {
        w.bytes += p.bytes;
        w.cpu += p.cpu;
    }
    w
}

/// Runs the frame on its own thread while `body` executes.
pub fn with_frame<R>(gpu: &'static gpu::Gpu, body: impl FnOnce(&'static frame::Frame) -> R) -> R {
    let frame: &'static frame::Frame = Box::leak(Box::new(frame::Frame::new(FRAME_MS)));
    let h = std::thread::spawn(move || unsafe { frame.run(gpu) });
    std::thread::sleep(std::time::Duration::from_millis(2000));
    let r = body(frame);
    frame.stop();
    h.join().unwrap();
    r
}

/// `PROBE_QUICK=1` shortens every timed loop (for validation runs).
pub fn quick() -> bool {
    std::env::var("PROBE_QUICK").is_ok_and(|v| v == "1")
}

/// Timed-loop iteration count, shortened under `PROBE_QUICK`.
pub fn iters(n: usize) -> usize {
    if quick() { 3 } else { n }
}

/// GPU time of the synthetic frame (paced at 60 Hz).
pub const FRAME_MS: f64 = 12.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).cloned().unwrap_or_else(|| "info".into());
    *CASE.lock().unwrap() = case.clone();
    match case.as_str() {
        "queues-child" => issues::queues_child(&args[2..]),
        "queues" => issues::queues_parent(&args[2..]),
        _ => {
            let gpu: &'static gpu::Gpu = Box::leak(Box::new(unsafe { gpu::Gpu::new(true) }));
            println!(
                "DEVICE {} driver {} families render={} loading={} copy={}",
                gpu.name, gpu.driver, gpu.families[0], gpu.families[1], gpu.families[2]
            );
            unsafe {
                match case.as_str() {
                    "info" => info(gpu),
                    "bar" => bar::run(gpu),
                    "gpuread" => bar::gpu_read(gpu),
                    "hic" => hic::run(gpu),
                    "clas" => clas::run(gpu),
                    "stale" => issues::stale(gpu),
                    "disk" => disk::run(gpu, &args[2..]),
                    "decomp" => decomp::run(gpu, &args[2..]),
                    _ => fail("unknown case"),
                }
            }
        }
    }
    let (e, w) = gpu::validation_report();
    println!("RESULT {case} DONE (validation errors={e} warnings={w})");
}

unsafe fn info(gpu: &gpu::Gpu) {
    for i in 0..gpu.mem.memory_type_count as usize {
        let t = gpu.mem.memory_types[i];
        let h = gpu.mem.memory_heaps[t.heap_index as usize];
        row(
            &format!("memory type {i}"),
            format!("{:?} heap {} ({:.2} GiB, {:?})", t.property_flags, t.heap_index, h.size as f64 / (1u64 << 30) as f64, h.flags),
        );
    }
    let (s, d, identical) = &gpu.hic_props;
    row("host copy src layouts", format!("{s:?}"));
    row("host copy dst layouts", format!("{d:?}"));
    row("identicalMemoryTypeRequirements", identical);
    row(
        "decompression",
        format!("{:?}, maxDecompressionIndirectCount {}", gpu.decomp_props.decompression_methods, gpu.decomp_props.max_decompression_indirect_count),
    );
}
