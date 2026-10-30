//! Fixed synthetic frame on the render queue: alternating bandwidth and ALU passes with a
//! barrier between each, paced at 60 Hz, three frames in flight. GPU time per frame comes
//! from timestamps around the whole frame. Load cases alternate baseline and loaded windows
//! and report the frame-time delta between them.

use crate::gpu::{Buf, Gpu, Mem, RENDER, Submitter};
use ash::vk;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameRoot {
    src: u64,
    dst: u64,
    n: u32,
    iters: u32,
    pad: [u32; 2],
}

const PASS_BYTES: u64 = 256 << 20;
const PASSES: u32 = 6;
const RING: usize = 3;
const SKIP_FRAMES: usize = 8;

pub struct Frame {
    stop: AtomicBool,
    window: AtomicU32,
    submitted: std::sync::atomic::AtomicU64,
    pub target_ms: f64,
    samples: Mutex<Vec<(u32, f64)>>,
}

pub struct Delta {
    pub base_ms: f64,
    pub load_ms: f64,
    pub base_p90: f64,
    pub load_p90: f64,
    pub frames: (usize, usize),
}

impl Delta {
    pub fn delta(&self) -> f64 {
        self.load_ms - self.base_ms
    }
    pub fn text(&self) -> String {
        format!(
            "frame {:.3} -> {:.3} ms (delta {:+.3} ms; p90 {:.3} -> {:.3}; frames {}/{})",
            self.base_ms,
            self.load_ms,
            self.delta(),
            self.base_p90,
            self.load_p90,
            self.frames.0,
            self.frames.1
        )
    }
}

impl Frame {
    pub fn new(target_ms: f64) -> Frame {
        Frame { stop: AtomicBool::new(false), window: AtomicU32::new(0), submitted: std::sync::atomic::AtomicU64::new(0), target_ms, samples: Mutex::new(Vec::new()) }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Runs on its own thread until `stop`.
    pub unsafe fn run(&self, gpu: &Gpu) {
        unsafe {
            let pipe = gpu.pipeline(&crate::spv!("frame"), gpu.layout);
            let a = gpu.buffer(PASS_BYTES, Mem::Device);
            let b = gpu.buffer(PASS_BYTES, Mem::Device);
            let mut sub = Submitter::new(gpu, RENDER, RING, 2);
            let n = (PASS_BYTES / 16) as u32;
            let record = |sub: &mut Submitter, iters: u32| {
                let cmd = sub.begin();
                sub.stamp(cmd, 0);
                gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                for p in 0..PASSES {
                    let (s, d): (&Buf, &Buf) = if p % 2 == 0 { (&a, &b) } else { (&b, &a) };
                    let it = if p % 2 == 0 { 4 } else { iters };
                    gpu.push(cmd, gpu.layout, &FrameRoot { src: s.addr, dst: d.addr, n, iters: it, pad: [0; 2] });
                    gpu.device.cmd_dispatch(cmd, n.div_ceil(256), 1, 1);
                    gpu.barrier(cmd);
                }
                sub.stamp(cmd, 1);
                sub.submit(cmd)
            };
            // Calibrate the ALU passes so the frame takes about target_ms.
            let mut iters = 256u32;
            for _ in 0..4 {
                let mut t = Vec::new();
                for _ in 0..5 {
                    let v = record(&mut sub, iters);
                    t.push(sub.ms(v, 0, 1));
                }
                let m = crate::median(&mut t);
                iters = ((f64::from(iters) * self.target_ms / m) as u32).clamp(8, 1 << 20);
            }
            let period = Duration::from_secs_f64(1.0 / 60.0);
            let mut next = Instant::now();
            let mut pending: VecDeque<(u64, u32)> = VecDeque::new();
            while !self.stop.load(Ordering::SeqCst) {
                if pending.len() >= RING - 1 {
                    let (v, w) = pending.pop_front().unwrap();
                    let ms = sub.ms(v, 0, 1);
                    self.samples.lock().unwrap().push((w, ms));
                }
                let w = self.window.load(Ordering::SeqCst);
                let v = record(&mut sub, iters);
                self.submitted.store(v, Ordering::SeqCst);
                pending.push_back((v, w));
                next += period;
                let now = Instant::now();
                if next > now {
                    std::thread::sleep(next - now);
                } else {
                    next = now;
                }
            }
            for (v, w) in pending {
                let ms = sub.ms(v, 0, 1);
                self.samples.lock().unwrap().push((w, ms));
            }
            drop(sub);
            gpu.free(a);
            gpu.free(b);
            gpu.device.destroy_pipeline(pipe, None);
        }
    }

    /// Calls `f` once per frame for `win`, right after the frame's submit, so the work lands on
    /// the loading queue while the frame runs (worst-case overlap).
    pub fn each_frame(&self, win: Duration, mut f: impl FnMut(u64)) {
        let t0 = Instant::now();
        let mut last = self.submitted.load(Ordering::SeqCst);
        let mut slot = 0;
        while t0.elapsed() < win {
            let now = self.submitted.load(Ordering::SeqCst);
            if now == last {
                std::thread::sleep(Duration::from_micros(200));
                continue;
            }
            last = now;
            f(slot);
            slot += 1;
        }
    }

    fn next_window(&self) -> u32 {
        self.window.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// `reps` pairs of (baseline window, loaded window); `load(window)` runs the load for the
    /// window's duration and returns its result.
    pub fn ab<R>(&self, reps: usize, win: Duration, mut load: impl FnMut(Duration) -> R) -> (Delta, Vec<R>) {
        let (reps, win) = if crate::quick() { (1, Duration::from_millis(300)) } else { (reps, win) };
        let mut base_w = Vec::new();
        let mut load_w = Vec::new();
        let mut out = Vec::new();
        for _ in 0..reps {
            base_w.push(self.next_window());
            std::thread::sleep(win);
            load_w.push(self.next_window());
            out.push(load(win));
        }
        self.next_window();
        std::thread::sleep(Duration::from_millis(100));
        (self.delta(&base_w, &load_w), out)
    }

    /// Frame-time delta over windows already run.
    pub fn delta(&self, base_w: &[u32], load_w: &[u32]) -> Delta {
        let s = self.samples.lock().unwrap();
        let pick = |ws: &[u32]| {
            let mut v = Vec::new();
            for &w in ws {
                v.extend(s.iter().filter(|x| x.0 == w).skip(SKIP_FRAMES).map(|x| x.1));
            }
            v
        };
        let mut b = pick(base_w);
        let mut l = pick(load_w);
        Delta {
            base_ms: crate::median(&mut b),
            load_ms: crate::median(&mut l),
            base_p90: crate::pct(&mut b, 0.9),
            load_p90: crate::pct(&mut l, 0.9),
            frames: (b.len(), l.len()),
        }
    }

    /// Marks the start of a baseline window and returns its id.
    pub fn base_window(&self) -> u32 {
        self.next_window()
    }

    /// Marks the start of a loaded window and returns its id.
    pub fn load_window(&self) -> u32 {
        self.next_window()
    }
}
