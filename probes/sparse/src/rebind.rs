//! Case 2: CLAS-pool page rebind. The CLAS pool is a sparse buffer of 64 KiB pages. Set A (24
//! clusters x 128 triangles) lives in page 0 under BLAS A, which stays live for the whole run.
//! Each cycle retires the set in page 1 (its BLAS leaves the TLAS), unbinds page 1 on the
//! loading queue while the render queue traces TLAS {A} in a checked loop, rebinds page 1 to
//! different memory, builds a new set (new ClusterIDs and depth) implicitly into scratch, moves
//! it into page 1, builds its BLAS and traces TLAS {A, new} against the CPU reference.

use crate::gpu::{Buf, Gpu, LOADING, Mem, RENDER, Submitter};
use crate::{median, ms_since, pct, row};
use ash::vk;
use std::time::Instant;

const GRID: u32 = 9;
const VERTS: u32 = GRID * GRID;
const TRIS: u32 = 2 * (GRID - 1) * (GRID - 1);
const CLUSTERS: u32 = 24;
const NONE: u32 = u32::MAX;
const A_CUSTOM: u32 = 7;
const LOOP_REPS: u32 = 256;
const LOOP_DISPATCHES: u32 = 24;

#[repr(C)]
#[derive(Clone, Copy)]
struct TraceRoot {
    tlas: u64,
    rays: u64,
    expect: u64,
    result: u64,
    n: u32,
    reps: u32,
    pad: [u32; 2],
}

fn cluster_origin(slot: u32, j: u32) -> (f32, f32) {
    (slot as f32 * 10.0 + (j % 6) as f32 * 1.25, (j / 6) as f32 * 1.25)
}

fn depth(generation: u32) -> f32 {
    1.0 + 0.01 * (generation % 50) as f32
}

fn verts(slot: u32, generation: u32) -> Vec<f32> {
    let mut v = Vec::new();
    for j in 0..CLUSTERS {
        let (ox, oy) = cluster_origin(slot, j);
        for k in 0..VERTS {
            v.extend_from_slice(&[ox + (k % GRID) as f32 / 8.0, oy + (k / GRID) as f32 / 8.0, depth(generation)]);
        }
    }
    v
}

/// One ray per triangle of a set: (origin, expected record).
fn rays(slot: u32, generation: u32, custom: Option<u32>) -> (Vec<[f32; 2]>, Vec<[u32; 4]>) {
    let mut o = Vec::new();
    let mut e = Vec::new();
    for j in 0..CLUSTERS {
        let (ox, oy) = cluster_origin(slot, j);
        for q in 0..(GRID - 1) * (GRID - 1) {
            let (qx, qy) = (q % (GRID - 1), q / (GRID - 1));
            for t in 0..2u32 {
                let f = if t == 0 { 0.25 } else { 0.75 };
                o.push([ox + (qx as f32 + f) / 8.0, oy + (qy as f32 + f) / 8.0]);
                e.push(match custom {
                    Some(c) => [c, (generation << 8) | j, 2 * q + t, (depth(generation) + 1.0).to_bits()],
                    None => [NONE; 4],
                });
            }
        }
    }
    (o, e)
}

struct Ctx<'g> {
    gpu: &'g Gpu,
    index: Buf,
    clas_implicit: Buf,
    clas_scratch: Buf,
    clas_addrs: Buf,
    clas_sizes: Buf,
    move_src: Buf,
    move_dst: Buf,
    move_sizes: Buf,
    move_scratch: Buf,
    count: Buf,
    one: Buf,
    blas_store: [Buf; 2],
    blas_scratch: Buf,
    blas_refs: [Buf; 2],
    blas_info: [Buf; 2],
    blas_addr: Buf,
    tlas: vk::AccelerationStructureKHR,
    tlas_addr: u64,
    tlas_scratch: Buf,
    instances: Buf,
    vert_bufs: [Buf; 2],
    infos: Buf,
    pool_addr: u64,
    page: u64,
}

fn tri_input() -> vk::ClusterAccelerationStructureTriangleClusterInputNV<'static> {
    vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .max_geometry_index_value(0)
        .max_cluster_unique_geometry_count(1)
        .max_cluster_triangle_count(TRIS)
        .max_cluster_vertex_count(VERTS)
        .max_total_triangle_count(CLUSTERS * TRIS)
        .max_total_vertex_count(CLUSTERS * VERTS)
}

fn region(addr: u64, stride: u64, n: u32) -> vk::StridedDeviceAddressRegionKHR {
    vk::StridedDeviceAddressRegionKHR { device_address: addr, stride, size: stride * u64::from(n) }
}

impl<'g> Ctx<'g> {
    unsafe fn new(gpu: &'g Gpu, pool_addr: u64, page: u64) -> Ctx<'g> {
        let mut idx = Vec::new();
        for y in 0..GRID - 1 {
            for x in 0..GRID - 1 {
                let i = (y * GRID + x) as u8;
                let g = GRID as u8;
                idx.extend_from_slice(&[i, i + 1, i + g, i + 1, i + 1 + g, i + g]);
            }
        }
        let index = gpu.buffer(idx.len() as u64, Mem::Upload);
        index.write(0, &idx);
        let mut ti = tri_input();
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(CLUSTERS)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut ti });
        let mut cs = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut cs);
        let mut mv = vk::ClusterAccelerationStructureMoveObjectsInputNV::default()
            .ty(vk::ClusterAccelerationStructureTypeNV::TRIANGLE_CLUSTER)
            .no_move_overlap(true)
            .max_moved_bytes(page);
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(CLUSTERS)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::MOVE_OBJECTS)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::EXPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_move_objects: &mut mv });
        let mut ms = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut ms);
        let mut bl = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default().max_total_cluster_count(CLUSTERS).max_cluster_count_per_acceleration_structure(CLUSTERS);
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(1)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_clusters_bottom_level: &mut bl });
        let mut bs = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut bs);
        row(
            "host size estimates",
            format!("CLAS {} B for {CLUSTERS} x {TRIS} tris (scratch {}), move scratch {}, cluster BLAS {} B (scratch {})", cs.acceleration_structure_size, cs.build_scratch_size, ms.build_scratch_size, bs.acceleration_structure_size, bs.build_scratch_size),
        );
        let dev = |n: u64| gpu.buffer(n + 512, Mem::Device);

        // Classic TLAS with room for two instances.
        let instances = gpu.buffer(128, Mem::Upload);
        let geometry = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(vk::DeviceOrHostAddressConstKHR { device_address: instances.addr }),
            })];
        let build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut ts = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.accel.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build, Some(&[2]), &mut ts);
        let tlas_buf = gpu.buffer(ts.acceleration_structure_size, Mem::Device);
        let tlas = gpu
            .accel
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default().buffer(tlas_buf.buffer).size(ts.acceleration_structure_size).ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                None,
            )
            .unwrap();
        let tlas_addr = gpu.accel.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(tlas));
        let count = gpu.buffer(4, Mem::Upload);
        count.write(0, &[CLUSTERS]);
        let one = gpu.buffer(4, Mem::Upload);
        one.write(0, &[1u32]);
        Ctx {
            gpu,
            index,
            clas_implicit: dev(cs.acceleration_structure_size),
            clas_scratch: dev(cs.build_scratch_size),
            clas_addrs: gpu.buffer(8 * u64::from(CLUSTERS), Mem::Readback),
            clas_sizes: gpu.buffer(4 * u64::from(CLUSTERS), Mem::Readback),
            move_src: gpu.buffer(8 * u64::from(CLUSTERS), Mem::Upload),
            move_dst: gpu.buffer(8 * u64::from(CLUSTERS), Mem::Upload),
            move_sizes: gpu.buffer(4 * u64::from(CLUSTERS), Mem::Readback),
            move_scratch: dev(ms.build_scratch_size),
            count,
            one,
            blas_store: [dev(bs.acceleration_structure_size), dev(bs.acceleration_structure_size)],
            blas_scratch: dev(bs.build_scratch_size),
            blas_refs: [gpu.buffer(8 * u64::from(CLUSTERS), Mem::Upload), gpu.buffer(8 * u64::from(CLUSTERS), Mem::Upload)],
            blas_info: [gpu.buffer(16, Mem::Upload), gpu.buffer(16, Mem::Upload)],
            blas_addr: gpu.buffer(16, Mem::Readback),
            tlas,
            tlas_addr,
            tlas_scratch: dev(ts.build_scratch_size),
            instances,
            vert_bufs: [gpu.buffer(u64::from(CLUSTERS * VERTS) * 12, Mem::Upload), gpu.buffer(u64::from(CLUSTERS * VERTS) * 12, Mem::Upload)],
            infos: gpu.buffer(64 * u64::from(CLUSTERS), Mem::Upload),
            pool_addr,
            page,
        }
    }

    /// Records the implicit CLAS build of one set (geometry `generation` at `slot`).
    unsafe fn record_clas(&self, cmd: vk::CommandBuffer, slot: u32, generation: u32) {
        let gpu = self.gpu;
        let vb = &self.vert_bufs[slot as usize];
        vb.write(0, &verts(slot, generation));
        let mut w = Vec::with_capacity(CLUSTERS as usize * 16);
        for j in 0..CLUSTERS {
            let vaddr = vb.addr + u64::from(j * VERTS) * 12;
            w.extend_from_slice(&[
                (generation << 8) | j,
                0,
                TRIS | (VERTS << 9) | (1 << 24) | (1 << 28),
                4 << 29,
                1 | (12 << 16),
                0,
                self.index.addr as u32,
                (self.index.addr >> 32) as u32,
                vaddr as u32,
                (vaddr >> 32) as u32,
                0,
                0,
                0,
                0,
                0,
                0,
            ]);
        }
        self.infos.write(0, &w);
        let mut ti = tri_input();
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(CLUSTERS)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut ti });
        let p = gpu.cl_props;
        let c = vk::ClusterAccelerationStructureCommandsInfoNV::default()
            .input(info)
            .dst_implicit_data(self.clas_implicit.addr.next_multiple_of(u64::from(p.cluster_byte_alignment)))
            .scratch_data(self.clas_scratch.addr.next_multiple_of(u64::from(p.cluster_scratch_byte_alignment)))
            .dst_addresses_array(region(self.clas_addrs.addr, 8, CLUSTERS))
            .dst_sizes_array(region(self.clas_sizes.addr, 4, CLUSTERS))
            .src_infos_array(region(self.infos.addr, 64, CLUSTERS))
            .src_infos_count(self.count.addr);
        gpu.cluster.cmd_build_cluster_acceleration_structure_indirect(cmd, &c);
        gpu.barrier(cmd);
    }

    /// After the build: packs the built CLAS into pool page `page` at 128 B granularity and
    /// records the move. Returns the pool addresses and the bytes used.
    unsafe fn record_move(&self, cmd: vk::CommandBuffer, page: u64) -> (Vec<u64>, u64) {
        let gpu = self.gpu;
        let sizes = std::slice::from_raw_parts(self.clas_sizes.ptr.cast::<u32>(), CLUSTERS as usize);
        let src = std::slice::from_raw_parts(self.clas_addrs.ptr.cast::<u64>(), CLUSTERS as usize).to_vec();
        let align = u64::from(gpu.cl_props.cluster_byte_alignment);
        let mut off = 0u64;
        let mut dst = Vec::new();
        for &s in sizes {
            dst.push(self.pool_addr + page * self.page + off);
            off += u64::from(s).next_multiple_of(align);
        }
        if off > self.page || sizes.contains(&0) {
            crate::fail(&format!("set does not fit a page: {off} B, sizes {sizes:?}"));
        }
        self.move_src.write(0, &src);
        self.move_dst.write(0, &dst);
        let mut mv = vk::ClusterAccelerationStructureMoveObjectsInputNV::default()
            .ty(vk::ClusterAccelerationStructureTypeNV::TRIANGLE_CLUSTER)
            .no_move_overlap(true)
            .max_moved_bytes(self.page);
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(CLUSTERS)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::MOVE_OBJECTS)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::EXPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_move_objects: &mut mv });
        let c = vk::ClusterAccelerationStructureCommandsInfoNV::default()
            .input(info)
            .scratch_data(self.move_scratch.addr.next_multiple_of(u64::from(gpu.cl_props.cluster_scratch_byte_alignment)))
            .dst_addresses_array(region(self.move_dst.addr, 8, CLUSTERS))
            .dst_sizes_array(region(self.move_sizes.addr, 4, CLUSTERS))
            .src_infos_array(region(self.move_src.addr, 8, CLUSTERS))
            .src_infos_count(self.count.addr);
        gpu.cluster.cmd_build_cluster_acceleration_structure_indirect(cmd, &c);
        gpu.barrier(cmd);
        (dst, off)
    }

    /// Records the cluster BLAS for `slot` over pool addresses `clas`.
    unsafe fn record_blas(&self, cmd: vk::CommandBuffer, slot: usize, clas: &[u64]) {
        let gpu = self.gpu;
        self.blas_refs[slot].write(0, clas);
        let r = self.blas_refs[slot].addr;
        self.blas_info[slot].write(0, &[CLUSTERS, 8, r as u32, (r >> 32) as u32]);
        let mut bl = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default().max_total_cluster_count(CLUSTERS).max_cluster_count_per_acceleration_structure(CLUSTERS);
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(1)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_clusters_bottom_level: &mut bl });
        let p = gpu.cl_props;
        let c = vk::ClusterAccelerationStructureCommandsInfoNV::default()
            .input(info)
            .dst_implicit_data(self.blas_store[slot].addr.next_multiple_of(u64::from(p.cluster_bottom_level_byte_alignment)))
            .scratch_data(self.blas_scratch.addr.next_multiple_of(u64::from(p.cluster_scratch_byte_alignment)))
            .dst_addresses_array(region(self.blas_addr.addr + 8 * slot as u64, 8, 1))
            .src_infos_array(region(self.blas_info[slot].addr, 16, 1))
            .src_infos_count(self.one.addr);
        gpu.cluster.cmd_build_cluster_acceleration_structure_indirect(cmd, &c);
        gpu.barrier(cmd);
    }

    /// Records the TLAS build over (custom index, BLAS address) instances.
    unsafe fn record_tlas(&self, cmd: vk::CommandBuffer, insts: &[(u32, u64)]) {
        let gpu = self.gpu;
        let mut w: Vec<u32> = Vec::new();
        for &(custom, blas) in insts {
            w.extend_from_slice(&[1f32.to_bits(), 0, 0, 0, 0, 1f32.to_bits(), 0, 0, 0, 0, 1f32.to_bits(), 0]);
            w.extend_from_slice(&[custom | (0xff << 24), 0, blas as u32, (blas >> 32) as u32]);
        }
        self.instances.write(0, &w);
        let geometry = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(vk::DeviceOrHostAddressConstKHR { device_address: self.instances.addr }),
            })];
        let build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry)
            .dst_acceleration_structure(self.tlas)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: self.tlas_scratch.addr.next_multiple_of(256) });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(insts.len() as u32)];
        gpu.accel.cmd_build_acceleration_structures(cmd, &[build], &[Some(&ranges)]);
        gpu.barrier(cmd);
    }
}

struct Check {
    rays: Buf,
    expect: Buf,
    result: Buf,
    n: u32,
}

impl Check {
    unsafe fn new(gpu: &Gpu) -> Check {
        let (mut o, _) = rays(0, 0, None);
        o.extend(rays(1, 0, None).0);
        let rb = gpu.buffer(o.len() as u64 * 8, Mem::Upload);
        rb.write(0, &o);
        Check { rays: rb, expect: gpu.buffer(o.len() as u64 * 16, Mem::Upload), result: gpu.buffer(32, Mem::Readback), n: o.len() as u32 }
    }
    /// Expected records: set A always; slot 1 hit by `generation` or missed.
    unsafe fn set(&self, slot1: Option<u32>) {
        let (_, mut e) = rays(0, 0, Some(A_CUSTOM));
        e.extend(rays(1, slot1.unwrap_or(0), slot1.map(|g| 1000 + g)).1);
        self.expect.write(0, &e);
        self.result.write(0, &[0u32; 8]);
    }
    unsafe fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, pipe: vk::Pipeline, tlas: u64, reps: u32) {
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
        let root = TraceRoot { tlas, rays: self.rays.addr, expect: self.expect.addr, result: self.result.addr, n: self.n, reps, pad: [0; 2] };
        gpu.push(cmd, gpu.layout, &root);
        gpu.device.cmd_dispatch(cmd, (self.n * reps).div_ceil(64), 1, 1);
        gpu.barrier(cmd);
    }
    unsafe fn mismatches(&self) -> (u32, [u32; 5]) {
        let r = std::slice::from_raw_parts(self.result.ptr.cast::<u32>(), 8);
        (r[0], [r[1], r[2], r[3], r[4], r[5]])
    }
}

pub unsafe fn run(gpu: &Gpu) {
    let pipe = gpu.pipeline(&crate::spv!("trace"), gpu.layout);
    let (pool, pool_addr, req) = gpu.sparse_buffer(8 << 16);
    let page = req.alignment;
    assert_eq!(page, 64 << 10);
    let ty = gpu.kind_type(req.memory_type_bits, Mem::Device);
    let backing = [gpu.alloc(4 * page, ty), gpu.alloc(4 * page, ty)];
    // Page 1 cycles through these (memory, page) slots; page 0 stays on (0, 0).
    let slots: Vec<(usize, u64)> = vec![(1, 0), (0, 1), (1, 2), (0, 2), (1, 1), (0, 3), (1, 3)];
    let ctx = Ctx::new(gpu, pool_addr, page);
    let mut render = Submitter::new(gpu, RENDER, 2, 2);
    let mut loading = Submitter::new(gpu, LOADING, 2, 2);
    let bind_tl = gpu.timeline();
    let mut bind_v = 0u64;
    let bind_page = |p: u64, mem: Option<(usize, u64)>, wait: &[(vk::Semaphore, u64)], v: &mut u64| -> (f64, f64) {
        let b = [vk::SparseMemoryBind {
            resource_offset: p * page,
            size: page,
            memory: mem.map_or(vk::DeviceMemory::null(), |m| backing[m.0]),
            memory_offset: mem.map_or(0, |m| m.1 * page),
            flags: vk::SparseMemoryBindFlags::empty(),
        }];
        let bb = [vk::SparseBufferMemoryBindInfo::default().buffer(pool).binds(&b)];
        *v += 1;
        let t0 = Instant::now();
        gpu.bind_sparse(LOADING, vk::BindSparseInfo::default().buffer_binds(&bb), wait, (bind_tl, *v));
        let call = ms_since(t0);
        if let Err(e) = gpu.wait_value(bind_tl, *v) {
            crate::fail(&format!("page bind wait {e:?}"));
        }
        (call, ms_since(t0))
    };
    bind_page(0, Some((0, 0)), &[], &mut bind_v);
    bind_page(1, Some(slots[0]), &[], &mut bind_v);

    // Build set A into page 0 and set 1 into page 1.
    let mut clas_a = Vec::new();
    let mut clas_s = Vec::new();
    let mut used = Vec::new();
    for (slot, generation) in [(0u32, 0u32), (1, 1)] {
        loading.run(|cmd| ctx.record_clas(cmd, slot, generation));
        let mut out = (Vec::new(), 0);
        loading.run(|cmd| out = ctx.record_move(cmd, u64::from(slot)));
        used.push(out.1);
        if slot == 0 { clas_a = out.0 } else { clas_s = out.0 }
    }
    let sizes_ok = std::slice::from_raw_parts(ctx.move_sizes.ptr.cast::<u32>(), CLUSTERS as usize).iter().all(|&s| s > 0);
    render.run(|cmd| {
        ctx.record_blas(cmd, 0, &clas_a);
        ctx.record_blas(cmd, 1, &clas_s);
    });
    let blas = std::slice::from_raw_parts(ctx.blas_addr.ptr.cast::<u64>(), 2).to_vec();
    let check = Check::new(gpu);
    let live = Check::new(gpu);
    check.set(Some(1));
    render.run(|cmd| {
        ctx.record_tlas(cmd, &[(A_CUSTOM, blas[0]), (1001, blas[1])]);
        check.record(gpu, cmd, pipe, ctx.tlas_addr, 1);
    });
    let (bad0, last0) = check.mismatches();
    row(
        "initial: set A in page 0, set 1 in page 1",
        format!("{} rays, mismatches {bad0} (last {last0:?}); bytes used per page {used:?}; moved sizes nonzero {sizes_ok}; BLAS A at {:#x}", check.n, blas[0]),
    );
    if bad0 != 0 {
        crate::fail("initial trace differs from the reference");
    }
    // Checker self-test: a wrong expectation for page 1 (generation 2) must fail every page-1 ray.
    check.set(Some(2));
    render.run(|cmd| check.record(gpu, cmd, pipe, ctx.tlas_addr, 1));
    let (bad_self, _) = check.mismatches();
    row("checker self-test (wrong generation expected for page 1)", format!("{bad_self} mismatches of {} page-1 rays", CLUSTERS * TRIS));
    if bad_self != CLUSTERS * TRIS {
        crate::fail("checker self-test");
    }

    let cycles = if crate::quick() { 5 } else { 200 };
    let (mut ucall, mut ulat, mut bcall, mut blat) = (vec![], vec![], vec![], vec![]);
    let mut overlapped = 0;
    let mut loop_bad = 0u32;
    let mut live_bad = 0u32;
    let mut last_render = render.value;
    for k in 0..cycles {
        let generation = k + 2;
        // Retire the set in page 1: TLAS {A}, then a long checked trace loop.
        live.set(None);
        let cmd = render.begin();
        ctx.record_tlas(cmd, &[(A_CUSTOM, blas[0])]);
        for _ in 0..LOOP_DISPATCHES {
            live.record(gpu, cmd, pipe, ctx.tlas_addr, LOOP_REPS);
        }
        let loop_v = render.submit(cmd);
        // Unbind page 1 once the last frame that read it is done, then rebind it elsewhere.
        let (c, l) = bind_page(1, None, &[(render.timeline, last_render)], &mut bind_v);
        ucall.push(c);
        ulat.push(l);
        let (c, l) = bind_page(1, Some(slots[(k as usize + 1) % slots.len()]), &[], &mut bind_v);
        bcall.push(c);
        blat.push(l);
        if gpu.device.get_semaphore_counter_value(render.timeline).unwrap() < loop_v {
            overlapped += 1;
        }
        // New set into page 1.
        loading.run(|cmd| ctx.record_clas(cmd, 1, generation));
        let mut out = (Vec::new(), 0);
        loading.run(|cmd| out = ctx.record_move(cmd, 1));
        render.wait(loop_v);
        loop_bad += live.mismatches().0;
        render.run(|cmd| ctx.record_blas(cmd, 1, &out.0));
        let b1 = *ctx.blas_addr.ptr.cast::<u64>().add(1);
        check.set(Some(generation));
        last_render = render.run(|cmd| {
            ctx.record_tlas(cmd, &[(A_CUSTOM, blas[0]), (1000 + generation, b1)]);
            check.record(gpu, cmd, pipe, ctx.tlas_addr, 1);
        });
        let (bad, last) = check.mismatches();
        if bad != 0 && live_bad == 0 {
            row(&format!("cycle {k} first mismatch"), format!("{bad} rays, last {last:?}"));
        }
        live_bad += bad;
    }
    row(
        &format!("{cycles} cycles: unbind + rebind page 1 while BLAS A traces"),
        format!(
            "checked-loop mismatches {loop_bad} ({} rays per cycle), post-rebind mismatches {live_bad} ({} rays per cycle); binds landed while the loop ran in {overlapped}/{cycles} cycles",
            u64::from(check.n) * u64::from(LOOP_REPS * LOOP_DISPATCHES),
            check.n
        ),
    );
    row(
        "one-page unbind (waits the render timeline)",
        format!("call {:.3} ms, latency {:.3} ms (p90 {:.3})", median(&mut ucall), median(&mut ulat.clone()), pct(&mut ulat, 0.9)),
    );
    row(
        "one-page bind",
        format!("call {:.3} ms, latency {:.3} ms (p90 {:.3})", median(&mut bcall), median(&mut blat.clone()), pct(&mut blat, 0.9)),
    );
    if loop_bad != 0 || live_bad != 0 {
        crate::fail("trace differs from the reference after a page rebind");
    }
}
