//! Case 3b: tile binds on the loading queue. Four 8192^2 BC7 sparse images (concurrent,
//! device-local, mip tails bound) and a pool of 64 KiB tile slots. Each batch is one
//! vkQueueBindSparse whose entries are half binds of unbound tiles into free slots and half
//! unbinds of the oldest bound tiles (steady-state streaming). With `copy`, a loading-queue
//! submission waits the bind and copies every newly bound tile from an Upload staging ring
//! (the tile upload path, since host image copy into sparse images is unsupported). Per-frame
//! batches of 100/1000/4000 entries against the frame, then back-to-back batches.

use crate::gpu::{Buf, Gpu, LOADING, Mem, Submitter};
use crate::tex::{self, BC7, Image, Kind, layers};
use crate::{Rng, median, ms_since, pct, row, with_frame};
use ash::vk;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const SIZE: u32 = 8192;
const RESIDENT: usize = 2500;
const IMAGES: usize = 4;
const STAGING_TILES: u64 = 2048;

struct Stream<'g> {
    imgs: Vec<Image>,
    memory: vk::DeviceMemory,
    tile_bytes: u64,
    /// (image, tile) not bound, in random order.
    unbound: VecDeque<(usize, (u32, u32, u32))>,
    /// (image, tile, slot) bound, oldest first.
    bound: VecDeque<(usize, (u32, u32, u32), u64)>,
    free: Vec<u64>,
    timeline: vk::Semaphore,
    value: u64,
    staging: Buf,
    next_stage: u64,
    copy: Submitter<'g>,
}

impl<'g> Stream<'g> {
    unsafe fn new(gpu: &'g Gpu) -> Stream<'g> {
        let mips = tex::full_mips(SIZE, SIZE);
        let mut imgs = Vec::new();
        let mut all = Vec::new();
        let mut copy = Submitter::new(gpu, LOADING, 3, 2);
        for i in 0..IMAGES {
            let mut img = tex::create(gpu, BC7, SIZE, SIZE, mips, 1, Kind { sparse: true, host: false, concurrent: true });
            // Mip tail only (pinned); tiles stream.
            let ty = tex::image_type(gpu, img.req.memory_type_bits);
            let tail = gpu.alloc(img.tail_bytes(), ty);
            let s = img.sreq.unwrap();
            let b = [vk::SparseMemoryBind { resource_offset: s.image_mip_tail_offset, size: s.image_mip_tail_size, memory: tail, memory_offset: 0, flags: vk::SparseMemoryBindFlags::empty() }];
            let ob = [vk::SparseImageOpaqueMemoryBindInfo::default().image(img.image).binds(&b)];
            let sem = gpu.timeline();
            gpu.bind_sparse(LOADING, vk::BindSparseInfo::default().image_opaque_binds(&ob), &[], (sem, 1));
            gpu.wait_value(sem, 1).unwrap();
            gpu.device.destroy_semaphore(sem, None);
            copy.run(|cmd| tex::to_general(gpu, cmd, img.image, mips));
            all.extend(img.tiles().into_iter().map(|t| (i, t)));
            img.memory.push(tail);
            imgs.push(img);
        }
        Rng(3).shuffle(&mut all);
        let tile_bytes = imgs[0].tile_bytes();
        let slots = RESIDENT as u64 + 4096;
        let ty = tex::image_type(gpu, imgs[0].req.memory_type_bits);
        let memory = gpu.alloc(slots * tile_bytes, ty);
        let mut free: Vec<u64> = (0..slots).collect();
        Rng(4).shuffle(&mut free);
        let staging = gpu.buffer(STAGING_TILES * tile_bytes, Mem::Upload);
        Rng(6).fill(staging.slice());
        row("tile streaming", format!("{IMAGES} x BC7 {SIZE}^2, {} tiles above the tail, {} B tiles, pool {slots} slots (memory type {ty})", all.len(), tile_bytes));
        let mut s = Stream { imgs, memory, tile_bytes, unbound: all.into(), bound: VecDeque::new(), free, timeline: gpu.timeline(), value: 0, staging, next_stage: 0, copy };
        // Fill to the steady-state resident count.
        while s.bound.len() < RESIDENT {
            s.batch(gpu, 1000.min(2 * (RESIDENT - s.bound.len())), false, false);
        }
        s
    }

    /// One batch of `n` entries: half binds, half unbinds when `steady` (binds only otherwise),
    /// then with `copy` a staged copy of each newly bound tile. Returns (CPU ms of the bind
    /// call, ms until the bind signalled, ms until the copy finished or the bind if no copy).
    unsafe fn batch(&mut self, gpu: &Gpu, n: usize, steady: bool, copy: bool) -> (f64, f64, f64) {
        let (binds, unbinds) = if steady { (n / 2, n - n / 2) } else { (n / 2, 0) };
        let mut per: Vec<Vec<vk::SparseImageMemoryBind>> = vec![Vec::new(); self.imgs.len()];
        let mut copies: Vec<Vec<vk::BufferImageCopy>> = vec![Vec::new(); self.imgs.len()];
        for _ in 0..unbinds {
            let (i, t, slot) = self.bound.pop_front().unwrap();
            per[i].push(self.imgs[i].tile_bind(t, vk::DeviceMemory::null(), 0));
            self.free.push(slot);
            self.unbound.push_back((i, t));
        }
        let (tw, th) = BC7.tile();
        for _ in 0..binds {
            let (i, t) = self.unbound.pop_front().unwrap();
            let slot = self.free.pop().unwrap();
            per[i].push(self.imgs[i].tile_bind(t, self.memory, slot * self.tile_bytes));
            self.bound.push_back((i, t, slot));
            let (o, e) = self.imgs[i].tile_region(t);
            let stage = self.next_stage % STAGING_TILES;
            self.next_stage += 1;
            copies[i].push(vk::BufferImageCopy { buffer_offset: stage * self.tile_bytes, buffer_row_length: tw, buffer_image_height: th, image_subresource: layers(t.0), image_offset: o, image_extent: e });
        }
        let infos: Vec<_> = per
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.is_empty())
            .map(|(i, b)| vk::SparseImageMemoryBindInfo::default().image(self.imgs[i].image).binds(b))
            .collect();
        self.value += 1;
        let t0 = Instant::now();
        gpu.bind_sparse(LOADING, vk::BindSparseInfo::default().image_binds(&infos), &[], (self.timeline, self.value));
        let cpu = ms_since(t0);
        let mut done = None;
        if copy && binds > 0 {
            let cmd = self.copy.begin();
            for (i, c) in copies.iter().enumerate() {
                if !c.is_empty() {
                    gpu.device.cmd_copy_buffer_to_image(cmd, self.staging.buffer, self.imgs[i].image, vk::ImageLayout::GENERAL, c);
                }
            }
            gpu.barrier(cmd);
            done = Some(self.copy.submit_waiting(cmd, &[(self.timeline, self.value)]));
        }
        if let Err(e) = gpu.wait_value(self.timeline, self.value) {
            crate::fail(&format!("tile bind wait {e:?}"));
        }
        let bind = ms_since(t0);
        if let Some(v) = done {
            self.copy.wait(v);
        }
        (cpu, bind, ms_since(t0))
    }
}

pub unsafe fn run(gpu: &'static Gpu) {
    let mut s = Stream::new(gpu);
    let counts = [100usize, 1000, 4000];
    for copy in [false, true] {
        for &n in &counts {
            let (mut c, mut l, mut d) = (vec![], vec![], vec![]);
            for _ in 0..crate::iters(41) {
                let (a, b, e) = s.batch(gpu, n, true, copy);
                c.push(a);
                l.push(b);
                d.push(e);
                std::thread::sleep(Duration::from_millis(6));
            }
            let lm = median(&mut l);
            row(
                &format!("isolated {n} tile binds per batch{}", if copy { " + staged copies" } else { "" }),
                format!("call {:.3} ms, bind latency {lm:.3} ms (p90 {:.3}), {:.2} us per entry; done {:.3} ms", median(&mut c), pct(&mut l, 0.9), lm * 1e3 / n as f64, median(&mut d)),
            );
        }
    }
    with_frame(gpu, |frame| {
        for copy in [false, true] {
            for &n in &counts {
                let (mut c, mut l, mut d) = (vec![], vec![], vec![]);
                let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
                    frame.each_frame(win, |_| {
                        let (a, b, e) = s.batch(gpu, n, true, copy);
                        c.push(a);
                        l.push(b);
                        d.push(e);
                    });
                });
                let lm = median(&mut l);
                row(
                    &format!("per frame, overlapping it: {n} tile binds{}", if copy { " + staged copies" } else { "" }),
                    format!(
                        "call {:.3} ms, bind latency {lm:.3} ms (p90 {:.3}, max {:.3}), done {:.3} ms (max {:.3}), batches {}; {} ({:+.2}%)",
                        median(&mut c),
                        pct(&mut l, 0.9),
                        pct(&mut l, 1.0),
                        median(&mut d),
                        pct(&mut d, 1.0),
                        l.len(),
                        delta.text(),
                        delta.delta() / delta.base_ms * 100.0
                    ),
                );
            }
        }
        // Saturated: back-to-back 1000-entry batches.
        let mut tiles = 0usize;
        let mut wall = 0.0;
        let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
            let t0 = Instant::now();
            while t0.elapsed() < win {
                s.batch(gpu, 1000, true, false);
                tiles += 1000;
            }
            wall += t0.elapsed().as_secs_f64();
        });
        row(
            "saturated: back-to-back 1000-entry batches, no copies",
            format!("{:.0} entries per second; {} ({:+.2}%)", tiles as f64 / wall, delta.text(), delta.delta() / delta.base_ms * 100.0),
        );
    });
}
