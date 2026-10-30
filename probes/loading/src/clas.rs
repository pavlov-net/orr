//! Case 3: loading-queue work against the frame. Stream-in batches of position expansion,
//! implicit CLAS builds and a buffer copy stream; per-page opacity micromap builds.

use crate::gpu::{Buf, Gpu, LOADING, Mem, Submitter};
use crate::{row, with_frame};
use ash::vk;
use std::time::{Duration, Instant};


const GRID: u32 = 9;
const VERTS: u32 = GRID * GRID;
const TRIS: u32 = 2 * (GRID - 1) * (GRID - 1);
const MAX_CLUSTERS: u32 = 8192;
const COPY_BYTES: u64 = 32 << 20;

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpandRoot {
    grid: u64,
    anchors: u64,
    dst: u64,
    vertices: u32,
    verts_per_cluster: u32,
    scale: f32,
    pad: u32,
}

struct Batch {
    expand: vk::Pipeline,
    grid: Buf,
    anchors: Buf,
    verts: Buf,
    infos: Buf,
    counts: Vec<(u32, Buf)>,
    implicit: Buf,
    scratch: Buf,
    addrs: Buf,
    sizes: Buf,
    copy_src: Buf,
    copy_dst: Buf,
    input: vk::ClusterAccelerationStructureTriangleClusterInputNV<'static>,
}

fn flags() -> vk::BuildAccelerationStructureFlagsKHR {
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS
}

impl Batch {
    unsafe fn new(gpu: &Gpu, clamps: &[u32]) -> Batch {
        let n = MAX_CLUSTERS;
        let grid = gpu.buffer(u64::from(n * VERTS) * 8, Mem::Upload);
        let anchors = gpu.buffer(u64::from(n) * 16, Mem::Upload);
        let mut g = Vec::with_capacity((n * VERTS) as usize * 2);
        let mut a = Vec::with_capacity(n as usize * 4);
        for c in 0..n {
            a.extend_from_slice(&[((c % 128) * 2000) as i32, ((c / 128) * 2000) as i32, 0, (c * VERTS) as i32]);
            for v in 0..VERTS {
                let (x, y) = (v % GRID, v / GRID);
                let z = ((x * 37 + y * 91 + c * 13) % 200) as u32;
                g.push((x * 250) | ((y * 250) << 16));
                g.push(z);
            }
        }
        grid.write(0, &g);
        anchors.write(0, &a);
        let verts = gpu.buffer(u64::from(n * VERTS) * 12, Mem::Device);
        let mut idx = Vec::new();
        for y in 0..GRID - 1 {
            for x in 0..GRID - 1 {
                let i = (y * GRID + x) as u8;
                idx.extend_from_slice(&[i, i + 1, i + GRID as u8, i + 1, i + 1 + GRID as u8, i + GRID as u8]);
            }
        }
        let index = gpu.buffer(idx.len() as u64, Mem::Upload);
        index.write(0, &idx);
        // VkClusterAccelerationStructureBuildTriangleClusterInfoNV, 64 B each.
        let infos = gpu.buffer(u64::from(n) * 64, Mem::Upload);
        let mut w = Vec::with_capacity(n as usize * 16);
        for c in 0..n {
            let vaddr = verts.addr + u64::from(c * VERTS) * 12;
            let words = [
                c,
                0,
                TRIS | (VERTS << 9) | (1 << 24) | (1 << 28),
                4 << 29, // geometry index 0, OPAQUE
                1 | (12 << 16),
                0,
                index.addr as u32,
                (index.addr >> 32) as u32,
                vaddr as u32,
                (vaddr >> 32) as u32,
                0,
                0,
                0,
                0,
                0,
                0,
            ];
            w.extend_from_slice(&words);
        }
        infos.write(0, &w);
        let counts = clamps
            .iter()
            .map(|&k| {
                let b = gpu.buffer(4, Mem::Upload);
                b.write(0, &[k]);
                (k, b)
            })
            .collect();
        let input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(0)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(TRIS)
            .max_cluster_vertex_count(VERTS)
            .max_total_triangle_count(n * TRIS)
            .max_total_vertex_count(n * VERTS)
            .min_position_truncate_bit_count(0);
        let mut ti = input;
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(n)
            .flags(flags())
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut ti });
        let mut sz = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut sz);
        let implicit = gpu.buffer(sz.acceleration_structure_size + 256, Mem::Device);
        let scratch = gpu.buffer(sz.build_scratch_size + 256, Mem::Device);
        row("CLAS host size estimate", format!("{} B dst, {} B scratch for {} clusters ({:.1} B per triangle)", sz.acceleration_structure_size, sz.build_scratch_size, n, sz.acceleration_structure_size as f64 / f64::from(n * TRIS)));
        Batch {
            expand: gpu.pipeline(&crate::spv!("expand"), gpu.layout),
            grid,
            anchors,
            verts,
            infos,
            counts,
            implicit,
            scratch,
            addrs: gpu.buffer(u64::from(n) * 8, Mem::Readback),
            sizes: gpu.buffer(u64::from(n) * 4, Mem::Readback),
            copy_src: gpu.buffer(COPY_BYTES, Mem::Device),
            copy_dst: gpu.buffer(COPY_BYTES, Mem::Device),
            input,
        }
    }

    /// Records expansion, build and copy for `clusters` (one of the clamps); stamps 0..3.
    unsafe fn record(&self, gpu: &Gpu, sub: &Submitter, cmd: vk::CommandBuffer, clusters: u32, copy: bool) {
        let count = &self.counts.iter().find(|c| c.0 == clusters).unwrap().1;
        sub.stamp(cmd, 0);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.expand);
        let root = ExpandRoot {
            grid: self.grid.addr,
            anchors: self.anchors.addr,
            dst: self.verts.addr,
            vertices: clusters * VERTS,
            verts_per_cluster: VERTS,
            scale: 1.0 / 1024.0,
            pad: 0,
        };
        gpu.push(cmd, gpu.layout, &root);
        gpu.device.cmd_dispatch(cmd, (clusters * VERTS).div_ceil(128), 1, 1);
        gpu.barrier(cmd);
        sub.stamp(cmd, 1);
        let mut ti = self.input;
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(MAX_CLUSTERS)
            .flags(flags())
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut ti });
        let p = gpu.cl_props;
        let al = |x: u64, a: u32| x.next_multiple_of(u64::from(a));
        let region = |addr: u64, stride: u64| vk::StridedDeviceAddressRegionKHR { device_address: addr, stride, size: stride * u64::from(MAX_CLUSTERS) };
        let c = vk::ClusterAccelerationStructureCommandsInfoNV::default()
            .input(info)
            .dst_implicit_data(al(self.implicit.addr, p.cluster_byte_alignment))
            .scratch_data(al(self.scratch.addr, p.cluster_scratch_byte_alignment))
            .dst_addresses_array(region(self.addrs.addr, 8))
            .dst_sizes_array(region(self.sizes.addr, 4))
            .src_infos_array(region(self.infos.addr, 64))
            .src_infos_count(count.addr);
        gpu.cluster.cmd_build_cluster_acceleration_structure_indirect(cmd, &c);
        gpu.barrier(cmd);
        sub.stamp(cmd, 2);
        if copy {
            gpu.device.cmd_copy_buffer(cmd, self.copy_src.buffer, self.copy_dst.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: COPY_BYTES }]);
            gpu.barrier(cmd);
        }
        sub.stamp(cmd, 3);
    }
}

pub unsafe fn run(gpu: &'static Gpu) {
    let clamps = [2048u32, 4096, 8192];
    let batch = Batch::new(gpu, &clamps);
    let mut sub = Submitter::new(gpu, LOADING, 2, 4);
    // Isolated: no frame.
    let mut e = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for it in 0..crate::iters(21) {
        let cmd = sub.begin();
        batch.record(gpu, &sub, cmd, MAX_CLUSTERS, true);
        let v = sub.submit(cmd);
        if it > 0 {
            e.push(sub.ms(v, 0, 1));
            b.push(sub.ms(v, 1, 2));
            c.push(sub.ms(v, 2, 3));
        }
        std::thread::sleep(Duration::from_millis(6));
    }
    let tris = f64::from(MAX_CLUSTERS * TRIS);
    let (em, bm, cm) = (crate::median(&mut e), crate::median(&mut b), crate::median(&mut c));
    row(
        "loading queue isolated, 8192 clusters (1.05M tris)",
        format!("expand {em:.3} ms, CLAS build {bm:.3} ms ({:.3} ms per 1M tris), copy 32 MiB {cm:.3} ms ({:.0} GB/s)", bm / tris * 1e6, COPY_BYTES as f64 / (cm * 1e-3) / 1e9),
    );
    let sizes = std::slice::from_raw_parts(batch.sizes.ptr.cast::<u32>(), MAX_CLUSTERS as usize);
    let addrs = std::slice::from_raw_parts(batch.addrs.ptr.cast::<u64>(), MAX_CLUSTERS as usize);
    let built: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
    let aligned = addrs.iter().all(|&a| a != 0 && a % u64::from(gpu.cl_props.cluster_byte_alignment) == 0);
    row("CLAS built size", format!("{built} B total, {:.1} B per triangle; every address set and aligned: {aligned}", built as f64 / tris));
    if !aligned || sizes.contains(&0) {
        crate::fail("CLAS build produced a zero size or bad address");
    }

    let omm = Omm::new(gpu);
    omm.isolated(gpu);

    with_frame(gpu, |frame| {
        // Paced: one batch per 60 Hz frame at each clamp, with and without the copy stream.
        for &k in &clamps {
            for copy in [false, true] {
                let mut lq = Vec::new();
                let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
                    frame.each_frame(win, |slot| {
                        let cmd = sub.begin();
                        batch.record(gpu, &sub, cmd, k, copy);
                        let v = sub.submit(cmd);
                        if slot > 0 {
                            lq.push(sub.ms(v - 1, 0, 3));
                        }
                    });
                    sub.wait(sub.value);
                });
                row(
                    &format!("per frame, overlapping it: {:.2}M tris{}", f64::from(k * TRIS) / 1e6, if copy { " + 32 MiB copy" } else { "" }),
                    format!("loading {:.3} ms per batch; {}", crate::median(&mut lq), delta.text()),
                );
            }
        }
        // Saturated: back-to-back batches, two in flight.
        let mut lq = Vec::new();
        let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
            let t0 = Instant::now();
            while t0.elapsed() < win {
                let cmd = sub.begin();
                batch.record(gpu, &sub, cmd, MAX_CLUSTERS, true);
                let v = sub.submit(cmd);
                if v > 1 {
                    lq.push(sub.ms(v - 1, 0, 3));
                }
            }
            sub.wait(sub.value);
        });
        row("saturated 1.05M tris + 32 MiB copy back to back", format!("loading {:.3} ms per batch; {}", crate::median(&mut lq), delta.text()));
        omm.concurrent(gpu, frame);
    });
}

/// Per-page opacity micromap builds: each page's OMM block is `tris` alpha-tested triangles
/// of 4-state data at one subdivision level.
struct Omm {
    pages: u32,
    configs: Vec<OmmConfig>,
}

struct OmmConfig {
    level: u32,
    tris: u32,
    data: Buf,
    tri_array: Buf,
    storage: Buf,
    scratch: Buf,
    maps: Vec<vk::MicromapEXT>,
    map_size: u64,
    scratch_size: u64,
    usage: vk::MicromapUsageEXT,
}

impl Omm {
    unsafe fn new(gpu: &Gpu) -> Omm {
        let pages = 64u32;
        let mut configs = Vec::new();
        for (level, tris) in [(3u32, 1024u32), (4, 1024), (4, 256)] {
            let bytes_per_tri = (1u64 << (2 * level)) / 4;
            let usage = vk::MicromapUsageEXT { count: tris, subdivision_level: level, format: vk::OpacityMicromapFormatEXT::TYPE_4_STATE.as_raw() as u32 };
            let us = [usage];
            let info = vk::MicromapBuildInfoEXT::default()
                .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
                .mode(vk::BuildMicromapModeEXT::BUILD)
                .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
                .usage_counts(&us);
            let mut sz = vk::MicromapBuildSizesInfoEXT::default();
            (gpu.omm.fp().get_micromap_build_sizes_ext)(gpu.device.handle(), vk::AccelerationStructureBuildTypeKHR::DEVICE, &info, &mut sz);
            let map_size = sz.micromap_size.next_multiple_of(256);
            let page_data = bytes_per_tri * u64::from(tris);
            let data = gpu.buffer(page_data * u64::from(pages), Mem::Upload);
            let mut r = crate::Rng(level as u64);
            let words: Vec<u32> = (0..(page_data * u64::from(pages) / 4)).map(|_| r.next() as u32).collect();
            data.write(0, &words);
            let tri_array = gpu.buffer(8 * u64::from(tris), Mem::Upload);
            let t: Vec<u32> = (0..tris).flat_map(|i| [i * bytes_per_tri as u32, level | (usage.format << 16)]).collect();
            tri_array.write(0, &t);
            let storage = gpu.buffer(map_size * u64::from(pages), Mem::Device);
            let scratch_size = sz.build_scratch_size.next_multiple_of(256);
            let scratch = gpu.buffer(scratch_size * u64::from(pages) + 256, Mem::Device);
            let maps = (0..pages)
                .map(|p| {
                    let ci = vk::MicromapCreateInfoEXT::default()
                        .buffer(storage.buffer)
                        .offset(map_size * u64::from(p))
                        .size(sz.micromap_size)
                        .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP);
                    let mut m = vk::MicromapEXT::null();
                    (gpu.omm.fp().create_micromap_ext)(gpu.device.handle(), &ci, std::ptr::null(), &mut m).result().unwrap();
                    m
                })
                .collect();
            row(&format!("OMM page level {level} x{tris} tris"), format!("{} B data, {} B micromap, {} B scratch", page_data, sz.micromap_size, sz.build_scratch_size));
            configs.push(OmmConfig { level, tris, data, tri_array, storage, scratch, maps, map_size, scratch_size, usage });
        }
        Omm { pages, configs }
    }

    /// One vkCmdBuildMicromapsEXT with one info per page (`batched`) or one call per page.
    unsafe fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, c: &OmmConfig, pages: u32, batched: bool) {
        let us = [c.usage];
        let page_data = (1u64 << (2 * c.level)) / 4 * u64::from(c.tris);
        let infos: Vec<_> = (0..pages)
            .map(|p| {
                vk::MicromapBuildInfoEXT::default()
                    .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
                    .mode(vk::BuildMicromapModeEXT::BUILD)
                    .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
                    .dst_micromap(c.maps[p as usize])
                    .usage_counts(&us)
                    .data(vk::DeviceOrHostAddressConstKHR { device_address: c.data.addr + page_data * u64::from(p) })
                    .scratch_data(vk::DeviceOrHostAddressKHR { device_address: (c.scratch.addr + c.scratch_size * u64::from(p)).next_multiple_of(256) })
                    .triangle_array(vk::DeviceOrHostAddressConstKHR { device_address: c.tri_array.addr })
                    .triangle_array_stride(8)
            })
            .collect();
        if batched {
            (gpu.omm.fp().cmd_build_micromaps_ext)(cmd, infos.len() as u32, infos.as_ptr());
        } else {
            for i in &infos {
                (gpu.omm.fp().cmd_build_micromaps_ext)(cmd, 1, i);
            }
        }
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::MICROMAP_BUILD_EXT)
            .src_access_mask(vk::AccessFlags2::MICROMAP_WRITE_EXT)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
        let _ = (c.map_size, &c.storage);
    }

    unsafe fn isolated(&self, gpu: &Gpu) {
        let mut sub = Submitter::new(gpu, LOADING, 2, 2);
        for c in &self.configs {
            for batched in [true, false] {
                let mut t = Vec::new();
                for it in 0..crate::iters(21) {
                    let cmd = sub.begin();
                    sub.stamp(cmd, 0);
                    self.record(gpu, cmd, c, self.pages, batched);
                    sub.stamp(cmd, 1);
                    let v = sub.submit(cmd);
                    if it > 0 {
                        t.push(sub.ms(v, 0, 1));
                    }
                    std::thread::sleep(Duration::from_millis(6));
                }
                let m = crate::median(&mut t);
                row(
                    &format!("OMM build level {} x{} tris, {} pages {}", c.level, c.tris, self.pages, if batched { "in one call" } else { "one call each" }),
                    format!("{m:.3} ms, {:.4} ms per page", m / f64::from(self.pages)),
                );
            }
        }
    }

    unsafe fn concurrent(&self, gpu: &Gpu, frame: &crate::frame::Frame) {
        let mut sub = Submitter::new(gpu, LOADING, 2, 2);
        let c = &self.configs[1];
        for pages in [16u32, 64] {
            let mut lq = Vec::new();
            let (delta, _) = frame.ab(3, Duration::from_millis(1500), |win| {
                frame.each_frame(win, |slot| {
                    let cmd = sub.begin();
                    sub.stamp(cmd, 0);
                    self.record(gpu, cmd, c, pages, true);
                    sub.stamp(cmd, 1);
                    let v = sub.submit(cmd);
                    if slot > 0 {
                        lq.push(sub.ms(v - 1, 0, 1));
                    }
                });
                sub.wait(sub.value);
            });
            row(
                &format!("per frame, overlapping it: OMM {pages} pages (level 4 x1024 tris)"),
                format!("loading {:.3} ms per batch; {}", crate::median(&mut lq), delta.text()),
            );
        }
    }
}
