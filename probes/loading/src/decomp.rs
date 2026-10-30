//! Case 6: VK_EXT_memory_decompression. GDeflate 64 KiB regions compressed on the CPU,
//! decompressed on the loading queue (direct and indirect-count), compared byte for byte with
//! the source; throughput isolated and against the frame.

use crate::bar::{bc7_tile, stored_page};
use crate::gpu::{Buf, Gpu, LOADING, Mem, Submitter};
use crate::{row, with_frame};
use ash::vk;
use std::time::{Duration, Instant};

const REGION: usize = 64 << 10;
const UNIQUE: usize = 1024; // 64 MiB of distinct source data
const REGIONS: usize = 4096; // 256 MiB per full command

fn source() -> Vec<u8> {
    let mut v = Vec::with_capacity(UNIQUE * REGION);
    let mut seed = 1;
    while v.len() < UNIQUE * REGION / 2 {
        v.extend_from_slice(&stored_page(seed));
        seed += 1;
    }
    v.truncate(UNIQUE * REGION / 2);
    while v.len() < UNIQUE * REGION {
        v.extend_from_slice(&bc7_tile(seed));
        seed += 1;
    }
    v
}

struct Packed {
    _src: Buf,
    regions: Vec<vk::DecompressMemoryRegionEXT>,
    compressed: usize,
}

/// Region i decompresses unique chunk i % UNIQUE into dst + i * REGION. `align` 1 packs the
/// compressed chunks back to back, as an archive blob stores them.
unsafe fn pack(gpu: &Gpu, chunks: &[Vec<u8>], dst: &Buf, align: usize) -> Packed {
    let mut bytes = Vec::new();
    let mut offs = Vec::new();
    for c in chunks {
        bytes.resize(bytes.len().next_multiple_of(align), 0);
        offs.push(bytes.len());
        bytes.extend_from_slice(c);
    }
    let src = gpu.buffer(bytes.len() as u64, Mem::Device);
    let stage = gpu.buffer(bytes.len() as u64, Mem::Upload);
    stage.write(0, &bytes);
    let mut sub = Submitter::new(gpu, LOADING, 1, 2);
    sub.run(|cmd| {
        gpu.device.cmd_copy_buffer(cmd, stage.buffer, src.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes.len() as u64 }]);
        gpu.barrier(cmd);
    });
    drop(sub);
    gpu.free(stage);
    let regions = (0..REGIONS)
        .map(|i| {
            let u = i % UNIQUE;
            vk::DecompressMemoryRegionEXT {
                src_address: src.addr + offs[u] as u64,
                dst_address: dst.addr + (i * REGION) as u64,
                compressed_size: chunks[u].len() as u64,
                decompressed_size: REGION as u64,
            }
        })
        .collect();
    Packed { _src: src, regions, compressed: chunks.iter().map(|c| c.len()).sum::<usize>() * REGIONS / UNIQUE }
}

unsafe fn record_direct(gpu: &Gpu, cmd: vk::CommandBuffer, regions: &[vk::DecompressMemoryRegionEXT]) {
    let info = vk::DecompressMemoryInfoEXT::default().decompression_method(vk::MemoryDecompressionMethodFlagsEXT::GDEFLATE_1_0).regions(regions);
    (gpu.decomp.fp().cmd_decompress_memory_ext)(cmd, &info);
    decomp_barrier(gpu, cmd);
}

unsafe fn record_indirect(gpu: &Gpu, cmd: vk::CommandBuffer, list: &Buf, count: &Buf, max: u32) {
    (gpu.decomp.fp().cmd_decompress_memory_indirect_count_ext)(
        cmd,
        vk::MemoryDecompressionMethodFlagsEXT::GDEFLATE_1_0,
        list.addr,
        count.addr,
        max,
        std::mem::size_of::<vk::DecompressMemoryRegionEXT>() as u32,
    );
    decomp_barrier(gpu, cmd);
}

unsafe fn decomp_barrier(gpu: &Gpu, cmd: vk::CommandBuffer) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::MEMORY_DECOMPRESSION_EXT)
        .src_access_mask(vk::AccessFlags2::MEMORY_DECOMPRESSION_WRITE_EXT)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
}

pub unsafe fn run(gpu: &'static Gpu, _args: &[String]) {
    row(
        "exposure",
        format!(
            "VK_EXT_memory_decompression methods {:?}, maxDecompressionIndirectCount {}",
            gpu.decomp_props.decompression_methods, gpu.decomp_props.max_decompression_indirect_count
        ),
    );
    let raw = source();
    let t0 = Instant::now();
    let mut comp = gdeflate::Compressor::new(gdeflate::CompressionLevel::Level12).unwrap();
    let chunks: Vec<Vec<u8>> = raw.chunks(REGION).map(|c| comp.compress(c, REGION).unwrap().bytes).collect();
    let ctime = t0.elapsed().as_secs_f64();
    let csize: usize = chunks.iter().map(|c| c.len()).sum();
    let zsize: usize = raw.chunks(REGION).map(|c| zstd::bulk::compress(c, 19).unwrap().len()).sum();
    row(
        "GDeflate level 12, 64 KiB regions (gdeflate 0.4.1 crate)",
        format!("ratio {:.3}x (zstd 19 on the same 64 KiB chunks {:.3}x); CPU compress {:.0} MB/s", raw.len() as f64 / csize as f64, raw.len() as f64 / zsize as f64, raw.len() as f64 / ctime / 1e6),
    );
    // CPU reference decode, and per-core decode rates for comparison.
    let mut dec = gdeflate::Decompressor::new().unwrap();
    let t0 = Instant::now();
    let mut cpu_ok = true;
    for (i, c) in chunks.iter().enumerate() {
        let tiles = [gdeflate::Tile { compressed_size: c.len() as u32, uncompressed_size: REGION as u32 }];
        let out = dec.decompress(gdeflate::CompressionResult::new(c, &tiles, REGION as u32)).unwrap();
        cpu_ok &= out == raw[i * REGION..(i + 1) * REGION];
    }
    let gd_cpu = raw.len() as f64 / t0.elapsed().as_secs_f64() / 1e9;
    let zchunks: Vec<Vec<u8>> = raw.chunks(REGION).map(|c| zstd::bulk::compress(c, 19).unwrap()).collect();
    let mut zd = zstd::bulk::Decompressor::new().unwrap();
    let mut out = vec![0u8; REGION];
    let t0 = Instant::now();
    for z in &zchunks {
        zd.decompress_to_buffer(z, &mut out).unwrap();
    }
    let z_cpu = raw.len() as f64 / t0.elapsed().as_secs_f64() / 1e9;
    row("CPU decode, one core", format!("GDeflate {gd_cpu:.2} GB/s (round trip exact: {cpu_ok}), zstd {z_cpu:.2} GB/s"));

    let dst = gpu.buffer((REGIONS * REGION) as u64, Mem::Device);
    let rb = gpu.buffer((REGIONS * REGION) as u64, Mem::Readback);
    let mut sub = Submitter::new(gpu, LOADING, 2, 2);
    let verify = |sub: &mut Submitter, what: &str| {
        sub.run(|cmd| {
            gpu.device.cmd_copy_buffer(cmd, dst.buffer, rb.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: dst.size }]);
            gpu.barrier(cmd);
        });
        let got = std::slice::from_raw_parts(rb.ptr, REGIONS * REGION);
        let bad = (0..REGIONS).filter(|&i| got[i * REGION..(i + 1) * REGION] != raw[(i % UNIQUE) * REGION..(i % UNIQUE + 1) * REGION]).count();
        row(&format!("exact: {what}"), format!("{} of {REGIONS} regions differ from the source", bad));
        bad == 0
    };
    let clear = |sub: &mut Submitter| {
        sub.run(|cmd| {
            gpu.device.cmd_fill_buffer(cmd, dst.buffer, 0, vk::WHOLE_SIZE, 0xcdcd_cdcd);
            gpu.barrier(cmd);
        });
    };
    let mut all_ok = true;
    let mut packs = Vec::new();
    for align in [1usize, 256] {
        let p = pack(gpu, &chunks, &dst, align);
        clear(&mut sub);
        sub.run(|cmd| record_direct(gpu, cmd, &p.regions));
        all_ok &= verify(&mut sub, &format!("direct, compressed chunks aligned to {align} B"));
        let list = gpu.buffer((REGIONS * std::mem::size_of::<vk::DecompressMemoryRegionEXT>()) as u64, Mem::Upload);
        list.write(0, &p.regions);
        let count = gpu.buffer(4, Mem::Upload);
        count.write(0, &[REGIONS as u32]);
        clear(&mut sub);
        sub.run(|cmd| record_indirect(gpu, cmd, &list, &count, REGIONS as u32));
        all_ok &= verify(&mut sub, &format!("indirect count, compressed chunks aligned to {align} B"));
        packs.push((align, p, list, count));
    }
    // Isolated throughput, per contracts.md 32 spacing.
    let (_, p, list, count) = &packs[0];
    for (name, n) in [("256 MiB (4096 regions)", REGIONS), ("16 MiB (256 regions)", 256)] {
        for indirect in [false, true] {
            count.write(0, &[n as u32]);
            let mut t = Vec::new();
            for it in 0..crate::iters(12) {
                let cmd = sub.begin();
                sub.stamp(cmd, 0);
                if indirect {
                    record_indirect(gpu, cmd, list, count, REGIONS as u32);
                } else {
                    record_direct(gpu, cmd, &p.regions[..n]);
                }
                sub.stamp(cmd, 1);
                let v = sub.submit(cmd);
                if it > 0 {
                    t.push(sub.ms(v, 0, 1));
                }
                std::thread::sleep(Duration::from_millis(6));
            }
            let m = crate::median(&mut t);
            let out_b = (n * REGION) as f64;
            let in_b = p.compressed as f64 * n as f64 / REGIONS as f64;
            row(
                &format!("isolated {name} {}", if indirect { "indirect count" } else { "direct" }),
                format!("{m:.3} ms: {:.1} GB/s out, {:.1} GB/s compressed in", out_b / (m * 1e-3) / 1e9, in_b / (m * 1e-3) / 1e9),
            );
        }
    }
    count.write(0, &[REGIONS as u32]);
    let stride = std::mem::size_of::<vk::DecompressMemoryRegionEXT>() as u64;
    let counts: Vec<(usize, Buf)> = [64usize, 256, 1024, REGIONS]
        .iter()
        .map(|&n| {
            let b = gpu.buffer(4, Mem::Upload);
            b.write(0, &[n as u32]);
            (n, b)
        })
        .collect();
    with_frame(gpu, |frame| {
        for indirect in [true, false] {
            for (per_frame, cbuf) in &counts {
                let per_frame = *per_frame;
                if !indirect && per_frame > 256 {
                    continue;
                }
                let mut lq = Vec::new();
                let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
                    frame.each_frame(win, |k| {
                        let k = k as usize;
                        let first = (k * per_frame) % REGIONS;
                        let cmd = sub.begin();
                        sub.stamp(cmd, 0);
                        if indirect {
                            let l = Buf { addr: list.addr + first as u64 * stride, ..*list };
                            record_indirect(gpu, cmd, &l, cbuf, (REGIONS - first) as u32);
                        } else {
                            record_direct(gpu, cmd, &p.regions[first..(first + per_frame).min(REGIONS)]);
                        }
                        sub.stamp(cmd, 1);
                        let v = sub.submit(cmd);
                        if k > 0 {
                            lq.push(sub.ms(v - 1, 0, 1));
                        }
                    });
                    sub.wait(sub.value);
                });
                let mb = (per_frame * REGION) as f64 / 1e6;
                row(
                    &format!("per frame, overlapping it: {} {mb:.0} MB out ({:.1} GB/s)", if indirect { "indirect" } else { "direct" }, mb * 60.0 / 1e3),
                    format!("loading {:.3} ms per batch; {}", crate::median(&mut lq), delta.text()),
                );
            }
        }
        let mut lq = Vec::new();
        let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
            let t0 = Instant::now();
            while t0.elapsed() < win {
                let cmd = sub.begin();
                sub.stamp(cmd, 0);
                record_indirect(gpu, cmd, list, count, REGIONS as u32);
                sub.stamp(cmd, 1);
                let v = sub.submit(cmd);
                if v > 1 {
                    lq.push(sub.ms(v - 1, 0, 1));
                }
            }
            sub.wait(sub.value);
        });
        let m = crate::median(&mut lq);
        row(
            "saturated indirect 256 MiB commands back to back",
            format!("loading {m:.3} ms per command ({:.1} GB/s out); {}", (REGIONS * REGION) as f64 / (m * 1e-3) / 1e9, delta.text()),
        );
    });
    clear(&mut sub);
    sub.run(|cmd| record_direct(gpu, cmd, &p.regions));
    all_ok &= verify(&mut sub, "direct, after the concurrent runs");
    if !all_ok {
        crate::fail("GPU GDeflate output differs from the source");
    }
}
