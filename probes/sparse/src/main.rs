//! Probe: sparse residency (contracts.md 8, 9, 33), raw ash, on the loading queue against a
//! concurrent synthetic frame on the render queue.
//!
//! `sparse <case>` with case one of:
//! - `info`: queue families, sparse properties, memory types.
//! - `pages`: sparse buffer bind and unbind of 1k/10k/100k 64 KiB pages, isolated and once per
//!   frame against the frame.
//! - `rebind`: CLAS-pool page rebind: unbind and rebind a page whose CLAS no live BLAS
//!   references while another page's BLAS stays live and traces; checked traces.
//! - `sample`: sparse BC7 images against the same images non-sparse on a resolve-style kernel
//!   and a random-walk kernel.
//! - `tiles`: tile binds and unbinds on the loading queue against the frame (100/1000/4000 per
//!   frame, saturated).
//! - `hic`: loading-queue copy of cooked tiles (padded edge tiles included) into bound tiles of
//!   sparse images, read back through a GPU copy. `hic-host`: the same by host image copy into
//!   host-transfer sparse images, a combination 621.01 reports unsupported.
//! - `tail`: `imageMipTailFirstLod` and block shapes per format and size against the cook rule.
//!
//! Each case prints `ROW` lines (case | metric | value) and a `RESULT` line.
//! Build from this directory with `cargo build --release --target-dir ../../target/probes`.

#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments, clippy::missing_safety_doc)]

mod frame;
mod gpu;
mod pages;
mod rebind;
mod sample;
mod tex;
mod tiles;

use std::sync::Mutex;
use std::time::{Duration, Instant};

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
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
    pub fn fill(&mut self, out: &mut [u8]) {
        for c in out.chunks_mut(8) {
            let b = self.next().to_le_bytes();
            c.copy_from_slice(&b[..c.len()]);
        }
    }
}

/// Runs the frame on its own thread while `body` executes.
pub fn with_frame<R>(gpu: &'static gpu::Gpu, body: impl FnOnce(&'static frame::Frame) -> R) -> R {
    let frame: &'static frame::Frame = Box::leak(Box::new(frame::Frame::new(FRAME_MS)));
    let h = std::thread::spawn(move || unsafe { frame.run(gpu) });
    std::thread::sleep(Duration::from_millis(2000));
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

/// Wall-clock milliseconds since `t`.
pub fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).cloned().unwrap_or_else(|| "info".into());
    *CASE.lock().unwrap() = case.clone();
    let gpu: &'static gpu::Gpu = Box::leak(Box::new(unsafe { gpu::Gpu::new() }));
    println!(
        "DEVICE {} driver {} families render={} loading={} copy={}",
        gpu.name, gpu.driver, gpu.families[0], gpu.families[1], gpu.families[2]
    );
    unsafe {
        match case.as_str() {
            "info" => info(gpu),
            "pages" => pages::run(gpu),
            "rebind" => rebind::run(gpu),
            "sample" => sample::run(gpu),
            "tiles" => tiles::run(gpu),
            "hic" => tex::hic(gpu, false),
            "hic-host" => tex::hic(gpu, true),
            "tail" => tex::tail(gpu),
            _ => fail("unknown case"),
        }
    }
    let (e, w) = gpu::validation_report();
    println!("RESULT {case} DONE (validation errors={e} warnings={w})");
}

unsafe fn info(gpu: &gpu::Gpu) {
    for i in 0..3 {
        row(&format!("queue {i} family {}", gpu.families[i]), format!("{:?}", gpu.queue_flags[i]));
    }
    row("sparse properties", format!("{:?}", gpu.sparse_props));
    for i in 0..gpu.mem.memory_type_count as usize {
        let t = gpu.mem.memory_types[i];
        let h = gpu.mem.memory_heaps[t.heap_index as usize];
        row(&format!("memory type {i}"), format!("{:?} heap {} ({:.2} GiB)", t.property_flags, t.heap_index, h.size as f64 / (1u64 << 30) as f64));
    }
    let (b, _, req) = gpu.sparse_buffer(1 << 30);
    row("sparse buffer 1 GiB requirements", format!("size {} align {} typeBits {:#x}", req.size, req.alignment, req.memory_type_bits));
    gpu.device.destroy_buffer(b, None);
}
