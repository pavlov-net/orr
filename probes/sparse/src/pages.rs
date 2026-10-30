//! Case 1: sparse buffer pages. Bind and unbind 1k/10k/100k 64 KiB pages of an 8 GiB sparse
//! buffer on the loading queue, scattered (one entry per page, random resource pages onto
//! random memory slots) and ranged (one entry per contiguous run, the `bind_sparse_pages`
//! shape). CPU time of the vkQueueBindSparse call, latency to its timeline signal, and the
//! frame delta when one batch is submitted per frame.

use crate::gpu::{Gpu, LOADING};
use crate::{Rng, median, ms_since, row, with_frame};
use ash::vk;
use std::time::{Duration, Instant};

const BUFFER_PAGES: u64 = 1 << 17;
const CHUNK_PAGES: u64 = 16384;
const CHUNKS: u64 = 7;

pub struct Pool {
    pub buffer: vk::Buffer,
    pub page: u64,
    chunks: Vec<vk::DeviceMemory>,
    res_perm: Vec<u64>,
    mem_perm: Vec<u64>,
}

impl Pool {
    unsafe fn new(gpu: &Gpu) -> Pool {
        let (probe, _, req) = gpu.sparse_buffer(1 << 20);
        gpu.device.destroy_buffer(probe, None);
        let page = req.alignment;
        let (buffer, _, req) = gpu.sparse_buffer(BUFFER_PAGES * page);
        let ty = gpu.kind_type(req.memory_type_bits, crate::gpu::Mem::Device);
        let chunks = (0..CHUNKS).map(|_| gpu.alloc(CHUNK_PAGES * page, ty)).collect();
        let mut r = Rng(5);
        let mut res_perm: Vec<u64> = (0..BUFFER_PAGES).collect();
        r.shuffle(&mut res_perm);
        let mut mem_perm: Vec<u64> = (0..CHUNKS * CHUNK_PAGES).collect();
        r.shuffle(&mut mem_perm);
        row("sparse buffer", format!("{} pages of {} B ({} GiB address), backing {} x 1 GiB device-local (type {ty})", BUFFER_PAGES, page, BUFFER_PAGES * page >> 30, CHUNKS));
        Pool { buffer, page, chunks, res_perm, mem_perm }
    }

    /// Bind entries for `n` pages; `bind = false` gives the matching unbind entries.
    fn entries(&self, n: u64, scattered: bool, bind: bool) -> Vec<vk::SparseMemoryBind> {
        let mem = |slot: u64| if bind { self.chunks[(slot / CHUNK_PAGES) as usize] } else { vk::DeviceMemory::null() };
        if scattered {
            (0..n as usize)
                .map(|i| {
                    let slot = self.mem_perm[i];
                    vk::SparseMemoryBind {
                        resource_offset: self.res_perm[i] * self.page,
                        size: self.page,
                        memory: mem(slot),
                        memory_offset: if bind { (slot % CHUNK_PAGES) * self.page } else { 0 },
                        flags: vk::SparseMemoryBindFlags::empty(),
                    }
                })
                .collect()
        } else {
            (0..n.div_ceil(CHUNK_PAGES))
                .map(|c| {
                    let count = (n - c * CHUNK_PAGES).min(CHUNK_PAGES);
                    vk::SparseMemoryBind {
                        resource_offset: c * CHUNK_PAGES * self.page,
                        size: count * self.page,
                        memory: mem(c * CHUNK_PAGES),
                        memory_offset: 0,
                        flags: vk::SparseMemoryBindFlags::empty(),
                    }
                })
                .collect()
        }
    }
}

/// Submits one buffer bind on the loading queue and waits for it; returns (CPU ms of the call,
/// ms until the timeline value signalled).
pub unsafe fn submit(gpu: &Gpu, buffer: vk::Buffer, binds: &[vk::SparseMemoryBind], timeline: vk::Semaphore, value: &mut u64) -> (f64, f64) {
    let bb = [vk::SparseBufferMemoryBindInfo::default().buffer(buffer).binds(binds)];
    let info = vk::BindSparseInfo::default().buffer_binds(&bb);
    *value += 1;
    let t0 = Instant::now();
    gpu.bind_sparse(LOADING, info, &[], (timeline, *value));
    let cpu = ms_since(t0);
    if let Err(e) = gpu.wait_value(timeline, *value) {
        crate::fail(&format!("bind wait {e:?}"));
    }
    (cpu, ms_since(t0))
}

pub unsafe fn run(gpu: &'static Gpu) {
    let pool = Pool::new(gpu);
    let timeline = gpu.timeline();
    let mut value = 0u64;
    let sizes = [1000u64, 10_000, 100_000];
    for &n in &sizes {
        for scattered in [true, false] {
            // The validation layer's cost per scattered entry grows with the entry count.
            if crate::quick() && scattered && n > 1000 {
                continue;
            }
            let bind = pool.entries(n, scattered, true);
            let unbind = pool.entries(n, scattered, false);
            let (mut bc, mut bl, mut uc, mut ul) = (vec![], vec![], vec![], vec![]);
            for _ in 0..crate::iters(if n >= 100_000 { 7 } else { 21 }) {
                let (c, l) = submit(gpu, pool.buffer, &bind, timeline, &mut value);
                bc.push(c);
                bl.push(l);
                std::thread::sleep(Duration::from_millis(6));
                let (c, l) = submit(gpu, pool.buffer, &unbind, timeline, &mut value);
                uc.push(c);
                ul.push(l);
                std::thread::sleep(Duration::from_millis(6));
            }
            let (bl_m, ul_m) = (median(&mut bl), median(&mut ul));
            row(
                &format!("isolated {n} pages {}", if scattered { "scattered" } else { "ranged" }),
                format!(
                    "bind: call {:.3} ms, latency {bl_m:.3} ms ({:.2} us per page); unbind: call {:.3} ms, latency {ul_m:.3} ms ({:.2} us per page); {} entries",
                    median(&mut bc),
                    bl_m * 1e3 / n as f64,
                    median(&mut uc),
                    ul_m * 1e3 / n as f64,
                    bind.len()
                ),
            );
        }
    }
    with_frame(gpu, |frame| {
        for &n in &sizes {
            for scattered in [true, false] {
                if crate::quick() && scattered && n > 1000 {
                    continue;
                }
                let bind = pool.entries(n, scattered, true);
                let unbind = pool.entries(n, scattered, false);
                let mut lat = vec![];
                let mut cpu = vec![];
                let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
                    let mut bound = false;
                    frame.each_frame(win, |_| {
                        let (c, l) = submit(gpu, pool.buffer, if bound { &unbind } else { &bind }, timeline, &mut value);
                        bound = !bound;
                        cpu.push(c);
                        lat.push(l);
                    });
                    if bound {
                        submit(gpu, pool.buffer, &unbind, timeline, &mut value);
                    }
                });
                row(
                    &format!("per frame, overlapping it: {n} pages {}, bind and unbind alternating", if scattered { "scattered" } else { "ranged" }),
                    format!("call {:.3} ms, latency {:.3} ms (p90 {:.3}), batches {}; {}", median(&mut cpu), median(&mut lat.clone()), crate::pct(&mut lat, 0.9), lat.len(), delta.text()),
                );
            }
        }
    });
}
