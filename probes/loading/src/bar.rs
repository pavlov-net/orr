//! Case 1: resizable BAR. Worker writes into Upload (device-local host-visible) memory, zstd
//! decode paths into it, page decode and texture tile writes, and GPU reads of Upload vs
//! DeviceLocal blocks.

use crate::gpu::{Buf, Gpu, Mem, RENDER, Submitter};
use crate::{Rng, Work, row, run_workers, with_frame};
use ash::vk;
use std::time::Duration;

const REGION: usize = 64 << 20;
const CHUNK: usize = 256 << 10;
const TILE: usize = 64 << 10;
const CLUSTERS: usize = 80;
const PAGE_MAX: usize = 128 << 10;

/// Stand-in for a cooked mesh page in stored form: per-cluster records, delta-coded i16x3
/// positions, oct normals, UVs and u8x3 triangles, smooth enough to compress like mesh data.
pub fn stored_page(seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let mut out = Vec::with_capacity(PAGE_MAX);
    out.extend_from_slice(&(CLUSTERS as u32).to_le_bytes());
    let mut counts = Vec::new();
    for c in 0..CLUSTERS {
        let v = 64 + r.below(9) as u8;
        let t = 124 + r.below(5) as u8;
        counts.push((v, t));
        let mut rec = [0u8; 128];
        for (i, w) in rec.chunks_mut(4).enumerate() {
            let f = (c as f32 * 0.37 + i as f32 * 0.11).sin() * 4.0;
            w.copy_from_slice(&f.to_le_bytes());
        }
        rec[124] = v;
        rec[125] = t;
        out.extend_from_slice(&rec);
    }
    for &(v, _) in &counts {
        let (mut x, mut y, mut z) = (r.below(60000) as i16, r.below(60000) as i16, r.below(60000) as i16);
        for k in 0..v {
            let d = |r: &mut Rng| (r.below(33) as i16) - 16;
            let stored = if k == 0 {
                [x, y, z]
            } else {
                let dv = [d(&mut r), d(&mut r), d(&mut r)];
                x = x.wrapping_add(dv[0]);
                y = y.wrapping_add(dv[1]);
                z = z.wrapping_add(dv[2]);
                dv
            };
            for a in stored {
                out.extend_from_slice(&a.to_le_bytes());
            }
        }
    }
    for &(v, _) in &counts {
        let mut n = r.next() as u32;
        for _ in 0..v {
            n = n.wrapping_add((r.below(64) as u32) | ((r.below(64) as u32) << 16));
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(&((n >> 3) ^ 0x5555).to_le_bytes());
        }
    }
    for &(v, t) in &counts {
        for k in 0..t {
            let a = (k / 2) % (v - 2);
            out.extend_from_slice(&[a, a + 1 + (k & 1), a + 2]);
        }
    }
    out
}

/// Stand-in for `decode_page`: writes the resident layout (header, 128 B cluster records,
/// u16x3 positions from a per-cluster prefix sum, attributes, u8x3 triangles; each region
/// 4-aligned per cluster) sequentially into `dst`. Returns bytes written.
pub unsafe fn decode_page(src: &[u8], dst: *mut u8) -> usize {
    unsafe {
        let n = u32::from_le_bytes(src[0..4].try_into().unwrap()) as usize;
        let recs = &src[4..4 + n * 128];
        let mut s = 4 + n * 128;
        let mut o = 64usize;
        std::ptr::copy_nonoverlapping(recs.as_ptr(), dst.add(o), recs.len());
        o += recs.len();
        let counts: Vec<(usize, usize)> = (0..n).map(|c| (recs[c * 128 + 124] as usize, recs[c * 128 + 125] as usize)).collect();
        let pos0 = o;
        for &(v, _) in &counts {
            let mut acc = [0i16; 3];
            for k in 0..v {
                for slot in acc.iter_mut() {
                    let d = i16::from_le_bytes([src[s], src[s + 1]]);
                    s += 2;
                    *slot = if k == 0 { d } else { slot.wrapping_add(d) };
                    (dst.add(o) as *mut u16).write_unaligned(*slot as u16);
                    o += 2;
                }
            }
            o = o.next_multiple_of(4);
        }
        let attrs0 = o;
        for &(v, _) in &counts {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(s), dst.add(o), v * 8);
            s += v * 8;
            o += v * 8;
        }
        let tris0 = o;
        for &(_, t) in &counts {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(s), dst.add(o), t * 3);
            s += t * 3;
            o += t * 3;
            o = o.next_multiple_of(4);
        }
        let header = [n as u32, pos0 as u32, attrs0 as u32, tris0 as u32, o as u32];
        std::ptr::copy_nonoverlapping(header.as_ptr().cast::<u8>(), dst, 20);
        o
    }
}

/// BC7-like 64 KiB tile: mode bits and endpoints from a smooth field, random index bits.
pub fn bc7_tile(seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let mut out = Vec::with_capacity(TILE);
    for b in 0..TILE / 16 {
        let (bx, by) = ((b % 64) as f32, (b / 64) as f32);
        let e = ((bx * 0.05 + seed as f32).sin() * 60.0 + (by * 0.04).cos() * 60.0 + 128.0) as u8;
        out.extend_from_slice(&[0x40, e, e.wrapping_add(3), e.wrapping_add(7), e >> 1, e.wrapping_add(11), 0x80, e]);
        out.extend_from_slice(&r.next().to_le_bytes());
    }
    out
}

struct Src {
    chunks: Vec<Vec<u8>>,
    raw_chunk: Vec<u8>,
    pages: Vec<Vec<u8>>,
    raw_pages: Vec<Vec<u8>>,
    tiles: Vec<Vec<u8>>,
    raw_tiles: Vec<Vec<u8>>,
}

fn sources() -> Src {
    let level = 19;
    let raw_pages: Vec<Vec<u8>> = (0..64).map(|i| stored_page(i + 1)).collect();
    let mut joined = Vec::new();
    for p in &raw_pages {
        joined.extend_from_slice(p);
    }
    let raw_chunk = joined[..CHUNK].to_vec();
    let chunks: Vec<Vec<u8>> = joined.chunks(CHUNK).filter(|c| c.len() == CHUNK).map(|c| zstd::bulk::compress(c, level).unwrap()).collect();
    let pages = raw_pages.iter().map(|p| zstd::bulk::compress(p, level).unwrap()).collect();
    let raw_tiles: Vec<Vec<u8>> = (0..64).map(|i| bc7_tile(i + 100)).collect();
    let tiles = raw_tiles.iter().map(|t| zstd::bulk::compress(t, level).unwrap()).collect();
    Src { chunks, raw_chunk, pages, raw_pages, tiles, raw_tiles }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Variant {
    /// memcpy of a cached 256 KiB source into BAR.
    CopyBar,
    /// Same into cached host memory (reference).
    CopyHost,
    /// zstd 256 KiB chunk into cached scratch, then memcpy into BAR.
    ZstdScratchBar,
    /// zstd 256 KiB chunk straight into BAR.
    ZstdBar,
    /// zstd 256 KiB chunk into cached scratch only (decode cost alone).
    ZstdScratch,
    /// zstd page into scratch, decode_page into BAR.
    PageBar,
    /// zstd page into scratch, decode_page into cached host memory (reference).
    PageHost,
    /// zstd 64 KiB BC7 tile into scratch, memcpy into BAR.
    TileScratchBar,
    /// zstd 64 KiB BC7 tile straight into BAR.
    TileBar,
}

pub unsafe fn run(gpu: &'static Gpu) {
    let src = sources();
    let ratio = |c: &[Vec<u8>], raw: usize| (raw * c.len()) as f64 / c.iter().map(|x| x.len()).sum::<usize>() as f64;
    row("zstd ratio (level 19)", format!(
        "256 KiB mesh chunks {:.2}x, pages {:.2}x, BC7-like tiles {:.2}x",
        ratio(&src.chunks, CHUNK),
        src.raw_pages.iter().map(|p| p.len()).sum::<usize>() as f64 / src.pages.iter().map(|p| p.len()).sum::<usize>() as f64,
        ratio(&src.tiles, TILE)
    ));
    let mut resident = vec![0u8; PAGE_MAX];
    let page_bytes = decode_page(&src.raw_pages[0], resident.as_mut_ptr());
    row("page", format!("{} B stored, {} B resident, {} clusters", src.raw_pages[0].len(), page_bytes, CLUSTERS));

    let bar = gpu.buffer((REGION * 8) as u64, Mem::Upload);
    let ty = gpu.kind_type(u32::MAX, Mem::Upload);
    row("Upload memory type", format!("{ty} {:?}", gpu.mem.memory_types[ty as usize].property_flags));
    let host: Vec<Vec<u8>> = (0..8).map(|_| vec![1u8; REGION]).collect();
    let host_ptr: Vec<usize> = host.iter().map(|h| h.as_ptr() as usize).collect();
    let bar_ptr = bar.ptr as usize;

    let variants = [
        Variant::CopyBar,
        Variant::CopyHost,
        Variant::ZstdScratch,
        Variant::ZstdScratchBar,
        Variant::ZstdBar,
        Variant::PageHost,
        Variant::PageBar,
        Variant::TileScratchBar,
        Variant::TileBar,
    ];
    let src = &src;
    let host_ptr = &host_ptr;
    with_frame(gpu, |frame| {
        for v in variants {
            for workers in [1usize, 4, 8] {
                let (delta, works) = frame.ab(2, Duration::from_millis(1500), |win| {
                    let scratch: Vec<std::sync::Mutex<(Vec<u8>, zstd::bulk::Decompressor<'static>)>> =
                        (0..workers).map(|_| std::sync::Mutex::new((vec![0u8; CHUNK], zstd::bulk::Decompressor::new().unwrap()))).collect();
                    let step = |w: usize, i: u64| -> u64 {
                        let mut g = scratch[w].lock().unwrap();
                        let (sc, dec) = &mut *g;
                        let dst_base = if matches!(v, Variant::CopyHost | Variant::PageHost) { host_ptr[w] } else { bar_ptr + w * REGION };
                        let off = |len: usize| ((i as usize * len) % (REGION - len)) & !63;
                        let n = i as usize;
                        match v {
                            Variant::CopyBar | Variant::CopyHost => {
                                let d = (dst_base + off(CHUNK)) as *mut u8;
                                std::ptr::copy_nonoverlapping(src.raw_chunk.as_ptr(), d, CHUNK);
                                CHUNK as u64
                            }
                            Variant::ZstdScratch => dec.decompress_to_buffer(&src.chunks[n % src.chunks.len()], sc.as_mut_slice()).unwrap() as u64,
                            Variant::ZstdScratchBar => {
                                let len = dec.decompress_to_buffer(&src.chunks[n % src.chunks.len()], sc.as_mut_slice()).unwrap();
                                std::ptr::copy_nonoverlapping(sc.as_ptr(), (dst_base + off(CHUNK)) as *mut u8, len);
                                len as u64
                            }
                            Variant::ZstdBar => {
                                let d = std::slice::from_raw_parts_mut((dst_base + off(CHUNK)) as *mut u8, CHUNK);
                                dec.decompress_to_buffer(&src.chunks[n % src.chunks.len()], d).unwrap() as u64
                            }
                            Variant::PageBar | Variant::PageHost => {
                                dec.decompress_to_buffer(&src.pages[n % src.pages.len()], sc.as_mut_slice()).unwrap();
                                decode_page(sc, (dst_base + off(PAGE_MAX)) as *mut u8) as u64
                            }
                            Variant::TileScratchBar => {
                                let len = dec.decompress_to_buffer(&src.tiles[n % src.tiles.len()], sc.as_mut_slice()).unwrap();
                                std::ptr::copy_nonoverlapping(sc.as_ptr(), (dst_base + off(TILE)) as *mut u8, len);
                                len as u64
                            }
                            Variant::TileBar => {
                                let d = std::slice::from_raw_parts_mut((dst_base + off(TILE)) as *mut u8, TILE);
                                dec.decompress_to_buffer(&src.tiles[n % src.tiles.len()], d).unwrap() as u64
                            }
                        }
                    };
                    run_workers(workers, win, &step)
                });
                let w = Work::sum(&works);
                row(&format!("{v:?} x{workers}"), format!("{}; {}", w.text(), delta.text()));
            }
        }
    });
    check_bar(gpu, &bar, src);
    gpu_read(gpu);
}

/// Every write path leaves the same bytes in BAR as the reference decode.
unsafe fn check_bar(gpu: &Gpu, bar: &Buf, src: &Src) {
    let mut dec = zstd::bulk::Decompressor::new().unwrap();
    let mut sc = vec![0u8; CHUNK];
    let mut ok = true;
    let base = bar.ptr;
    let mut readback = vec![0u8; CHUNK];
    let mut cmp = |len: usize, expect: &[u8]| {
        std::ptr::copy_nonoverlapping(base, readback.as_mut_ptr(), len);
        ok &= &readback[..len] == expect;
    };
    let d = std::slice::from_raw_parts_mut(base, CHUNK);
    let n = dec.decompress_to_buffer(&src.chunks[0], d).unwrap();
    cmp(n, &src.raw_chunk);
    let n = dec.decompress_to_buffer(&src.tiles[3], std::slice::from_raw_parts_mut(base, TILE)).unwrap();
    cmp(n, &src.raw_tiles[3]);
    dec.decompress_to_buffer(&src.pages[5], sc.as_mut_slice()).unwrap();
    let mut reference = vec![0u8; PAGE_MAX];
    let len = decode_page(&src.raw_pages[5], reference.as_mut_ptr());
    std::ptr::write_bytes(base, 0, PAGE_MAX);
    decode_page(&sc, base);
    cmp(len, &reference[..len]);
    // The GPU sees the same bytes: copy BAR to readback on the render queue.
    let rb = gpu.buffer(CHUNK as u64, Mem::Readback);
    let mut sub = Submitter::new(gpu, RENDER, 1, 2);
    sub.run(|cmd| {
        gpu.device.cmd_copy_buffer(cmd, bar.buffer, rb.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: len as u64 }]);
        gpu.barrier(cmd);
    });
    ok &= std::slice::from_raw_parts(rb.ptr, len) == &reference[..len];
    row("BAR contents match reference decode (CPU and GPU view)", ok);
    if !ok {
        crate::fail("BAR contents differ from reference");
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReadRoot {
    src: u64,
    dst: u64,
    n_vec: u32,
    per: u32,
    gather: u32,
    pad: u32,
}

/// Kernel A/B per contracts.md 32: one warm and one timed dispatch per submission, 6 ms idle
/// between submissions, kinds interleaved, median of 41.
pub unsafe fn gpu_read(gpu: &Gpu) {
    let pipe = gpu.pipeline(&crate::spv!("read"), gpu.layout);
    let out = gpu.buffer(256, Mem::Device);
    let mut sub = Submitter::new(gpu, RENDER, 2, 4);
    for size in [16u64 << 20, 32 << 20, 64 << 20, 128 << 20, 256 << 20, 1 << 30] {
        let blocks = [gpu.buffer(size, Mem::Upload), gpu.buffer(size, Mem::Device)];
        sub.run(|cmd| {
            for b in &blocks {
                gpu.device.cmd_fill_buffer(cmd, b.buffer, 0, size, 0x0102_0304);
            }
            gpu.barrier(cmd);
        });
        // Bring clocks up before timing: 1.5 s of back-to-back reads.
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_millis(1500) {
            let root = ReadRoot { src: blocks[1].addr, dst: out.addr, n_vec: (size / 16) as u32, per: 16, gather: 0, pad: 0 };
            sub.run(|cmd| {
                gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                gpu.push(cmd, gpu.layout, &root);
                for _ in 0..8 {
                    gpu.device.cmd_dispatch(cmd, (root.n_vec / 16).div_ceil(256), 1, 1);
                }
            });
        }
        for gather in [0u32, 1, 2] {
            let per = 16u32;
            let n_vec = (size / 16) as u32;
            let mut t = [Vec::new(), Vec::new()];
            for it in 0..crate::iters(42) {
                for (k, b) in blocks.iter().enumerate() {
                    let root = ReadRoot { src: b.addr, dst: out.addr, n_vec, per, gather, pad: 0 };
                    let cmd = sub.begin();
                    gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                    gpu.push(cmd, gpu.layout, &root);
                    let groups = (n_vec / per).div_ceil(256);
                    gpu.device.cmd_dispatch(cmd, groups, 1, 1);
                    gpu.barrier(cmd);
                    sub.stamp(cmd, 0);
                    gpu.device.cmd_dispatch(cmd, groups, 1, 1);
                    sub.stamp(cmd, 1);
                    let v = sub.submit(cmd);
                    let ms = sub.ms(v, 0, 1);
                    if it > 0 {
                        t[k].push(ms);
                    }
                    std::thread::sleep(Duration::from_millis(6));
                }
            }
            let up = crate::median(&mut t[0]);
            let dl = crate::median(&mut t[1]);
            let name = ["stream", "gather 16 B", "gather 16 B within 128 KiB pages"][gather as usize];
            let gbs = |ms: f64| size as f64 / (ms * 1e-3) / 1e9;
            row(
                &format!("GPU read {name} {} MiB", size >> 20),
                format!("Upload {up:.3} ms ({:.0} GB/s) vs DeviceLocal {dl:.3} ms ({:.0} GB/s): Upload {:+.1}%", gbs(up), gbs(dl), (up / dl - 1.0) * 100.0),
            );
        }
        for b in blocks {
            gpu.free(b);
        }
    }
}
