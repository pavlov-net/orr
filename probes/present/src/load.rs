//! Synthetic streaming load: 8 workers decode zstd blocks into cached scratch and copy them
//! into an Upload (BAR) pool (the PRB-1 worker path), and one loading-queue thread submits a
//! 64 MiB device copy every 8 ms on the loading queue.

use crate::gpu::{Gpu, Mem};
use ash::vk;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

pub struct Load {
    stop: Arc<AtomicBool>,
    bytes: Arc<AtomicU64>,
    copies: Arc<AtomicU64>,
    threads: Vec<JoinHandle<()>>,
    t0: std::time::Instant,
}

const BLOCK: usize = 1 << 20;

impl Load {
    /// `bar` false decodes into cached memory only (no BAR writes).
    /// `gbps` > 0 limits the decoded bytes per second across all workers.
    pub unsafe fn start(gpu: &'static Gpu, workers: usize, bar: bool, gbps: f64) -> Load {
        unsafe {
            let stop = Arc::new(AtomicBool::new(false));
            let bytes = Arc::new(AtomicU64::new(0));
            let copies = Arc::new(AtomicU64::new(0));
            // Semi-compressible input (about 2:1).
            let mut rng = crate::Rng(7);
            let raw: Vec<u8> = (0..BLOCK).map(|i| if i % 2 == 0 { (rng.next() & 0xff) as u8 } else { (i / 64) as u8 }).collect();
            let packed = Arc::new(zstd::bulk::compress(&raw, 3).unwrap());
            let pool = gpu.buffer((workers * 16 * BLOCK) as u64, Mem::Upload);
            let mut threads = Vec::new();
            for w in 0..workers {
                let (stop, bytes, packed) = (stop.clone(), bytes.clone(), packed.clone());
                let base = pool.ptr as usize;
                threads.push(std::thread::spawn(move || {
                    let mut scratch = vec![0u8; BLOCK];
                    let mut cached = vec![0u8; 16 * BLOCK];
                    let mut dec = zstd::bulk::Decompressor::new().unwrap();
                    let mut k = 0usize;
                    let t0 = std::time::Instant::now();
                    let per = gbps * 1e9 / workers as f64;
                    while !stop.load(Ordering::Relaxed) {
                        if per > 0.0 {
                            let ahead = (k * BLOCK) as f64 / per - t0.elapsed().as_secs_f64();
                            if ahead > 0.0 {
                                std::thread::sleep(std::time::Duration::from_secs_f64(ahead));
                                continue;
                            }
                        }
                        let n = dec.decompress_to_buffer(&packed[..], &mut scratch).unwrap();
                        let off = (w * 16 + k % 16) * BLOCK;
                        if bar {
                            std::ptr::copy_nonoverlapping(scratch.as_ptr(), (base as *mut u8).add(off), n);
                        } else {
                            let o = (k % 16) * BLOCK;
                            cached[o..o + n].copy_from_slice(&scratch[..n]);
                        }
                        bytes.fetch_add(n as u64, Ordering::Relaxed);
                        k += 1;
                    }
                }));
            }
            let (stop2, copies2) = (stop.clone(), copies.clone());
            threads.push(std::thread::spawn(move || {
                let src = gpu.buffer(64 << 20, Mem::Device);
                let dst = gpu.buffer(64 << 20, Mem::Device);
                let pool = gpu.command_pool(gpu.families[1]);
                let cmds = gpu.command_buffers(pool, 4);
                let tl = gpu.timeline();
                let mut v = 0u64;
                while !stop2.load(Ordering::Relaxed) {
                    let cmd = cmds[(v % 4) as usize];
                    if v >= 4 {
                        gpu.wait(tl, v - 3);
                    }
                    gpu.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                    gpu.begin(cmd);
                    gpu.device.cmd_copy_buffer(cmd, src.buffer, dst.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: 64 << 20 }]);
                    gpu.device.end_command_buffer(cmd).unwrap();
                    v += 1;
                    let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
                    let sig = [vk::SemaphoreSubmitInfo::default().semaphore(tl).value(v).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                    gpu.device
                        .queue_submit2(gpu.queues[1], &[vk::SubmitInfo2::default().command_buffer_infos(&cbs).signal_semaphore_infos(&sig)], vk::Fence::null())
                        .unwrap();
                    copies2.fetch_add(1, Ordering::Relaxed);
                    std::thread::sleep(std::time::Duration::from_millis(8));
                }
                gpu.wait(tl, v);
            }));
            Load { stop, bytes, copies, threads, t0: std::time::Instant::now() }
        }
    }

    /// (GB/s decoded into Upload, loading-queue copies per second).
    pub fn stop(self) -> (f64, f64) {
        self.stop.store(true, Ordering::SeqCst);
        let dt = self.t0.elapsed().as_secs_f64();
        for t in self.threads {
            t.join().unwrap();
        }
        (self.bytes.load(Ordering::SeqCst) as f64 / dt / 1e9, self.copies.load(Ordering::SeqCst) as f64 / dt)
    }
}
