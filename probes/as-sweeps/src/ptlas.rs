//! Partitioned TLAS: memory against GetSizes (`ptlas_mem`), write and update cost against
//! capacity, writes and partitions (`ptlas_cost`), and partition size with part runs against
//! build and trace cost (`ptlas_part`).
//!
//! Every PTLAS enables partition translation (orr rebases partitions). Timings follow the
//! contracts.md 32 method: one warm and one timed operation per submission, about 6 ms idle
//! between submissions, median over the samples.

use super::*;

type WriteRec = vk::PartitionedAccelerationStructureWriteInstanceDataNV;
type UpdateRec = vk::PartitionedAccelerationStructureUpdateInstanceDataNV;
type TranslateRec = vk::PartitionedAccelerationStructureWritePartitionTranslationDataNV;

const _: () = {
    assert!(size_of::<WriteRec>() == 104);
    assert!(size_of::<UpdateRec>() == 16);
    assert!(size_of::<TranslateRec>() == 16);
};

const GLOBAL: u32 = vk::PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV;
const SENTINEL: u32 = 0xa5a5_a5a5;

pub(crate) struct PtlasDesc {
    pub instances: u32,
    pub per_partition: u32,
    pub partitions: u32,
    pub global: u32,
}

pub(crate) unsafe fn sizes(gpu: &Gpu, d: &PtlasDesc) -> (u64, u64) {
    let mut flags = vk::PartitionedAccelerationStructureFlagsNV::default().enable_partition_translation(true);
    let input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .instance_count(d.instances)
        .max_instance_per_partition_count(d.per_partition)
        .partition_count(d.partitions)
        .max_instance_in_global_partition_count(d.global)
        .push(&mut flags);
    let mut s = vk::AccelerationStructureBuildSizesInfoKHR::default();
    gpu.ptlas.as_ref().unwrap().get_partitioned_acceleration_structures_build_sizes(&input, &mut s);
    (s.acceleration_structure_size, s.build_scratch_size)
}

pub(crate) fn write_rec(index: u32, partition: u32, custom: u32, m: [f32; 12], blas: u64, aabb: Option<[f32; 6]>) -> WriteRec {
    let mut flags = vk::PartitionedAccelerationStructureInstanceFlagsNV::FLAG_TRIANGLE_FACING_CULL_DISABLE;
    if aabb.is_some() {
        flags |= vk::PartitionedAccelerationStructureInstanceFlagsNV::FLAG_ENABLE_EXPLICIT_BOUNDING_BOX;
    }
    WriteRec {
        transform: vk::TransformMatrixKHR { matrix: m },
        explicit_aabb: aabb.unwrap_or_default(),
        instance_id: custom & 0xff_ffff,
        instance_mask: 0xff,
        instance_contribution_to_hit_group_index: 0,
        instance_flags: flags,
        instance_index: index,
        partition_index: partition,
        acceleration_structure: blas,
    }
}

/// An object-space box through a 3x4 instance transform: the explicit AABB is in world space
/// (instance transform applied, partition translation not).
pub(crate) fn world_aabb(m: &[f32; 12], b: [f32; 6]) -> [f32; 6] {
    let mut out = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
    for corner in 0..8 {
        let p = [b[if corner & 1 != 0 { 3 } else { 0 }], b[if corner & 2 != 0 { 4 } else { 1 }], b[if corner & 4 != 0 { 5 } else { 2 }]];
        for r in 0..3 {
            let v = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3];
            out[r] = out[r].min(v);
            out[r + 3] = out[r + 3].max(v);
        }
    }
    out
}

fn update_rec(index: u32, blas: u64) -> UpdateRec {
    UpdateRec { instance_index: index, instance_contribution_to_hit_group_index: 0, acceleration_structure: blas }
}

#[derive(Clone, Copy)]
pub(crate) struct Op {
    ty: vk::PartitionedAccelerationStructureOpTypeNV,
    addr: u64,
    count: u32,
    stride: u64,
}

impl Op {
    pub(crate) fn write(addr: u64, count: u32) -> Op {
        Op { ty: vk::PartitionedAccelerationStructureOpTypeNV::WRITE_INSTANCE, addr, count, stride: 104 }
    }
    fn update(addr: u64, count: u32) -> Op {
        Op { ty: vk::PartitionedAccelerationStructureOpTypeNV::UPDATE_INSTANCE, addr, count, stride: 16 }
    }
    fn translate(addr: u64, count: u32) -> Op {
        Op { ty: vk::PartitionedAccelerationStructureOpTypeNV::WRITE_PARTITION_TRANSLATION, addr, count, stride: 16 }
    }
}

/// The indirect command array and its count, host-written in mapped device memory.
pub(crate) struct OpList {
    cmds: Buf,
    count: Buf,
}

impl OpList {
    pub(crate) unsafe fn new(ctx: &Ctx, ops: &[Op]) -> OpList {
        let l = OpList { cmds: ctx.buf(8 * 24, Mem::Upload, 256), count: ctx.buf(16, Mem::Upload, 16) };
        l.set(ops);
        l
    }

    /// Only while the GPU is idle.
    unsafe fn set(&self, ops: &[Op]) {
        assert!(ops.len() <= 8);
        let cmds: Vec<vk::BuildPartitionedAccelerationStructureIndirectCommandNV> = ops
            .iter()
            .map(|o| vk::BuildPartitionedAccelerationStructureIndirectCommandNV {
                op_type: o.ty,
                arg_count: o.count,
                arg_data: vk::StridedDeviceAddressNV { start_address: o.addr, stride_in_bytes: o.stride },
            })
            .collect();
        self.cmds.write(0, &cmds);
        self.count.write(0, &[ops.len() as u32]);
    }
}

/// One PTLAS build (in place unless `first`) followed by a full barrier.
pub(crate) unsafe fn record(gpu: &Gpu, cmd: vk::CommandBuffer, d: &PtlasDesc, dst: u64, scratch: u64, first: bool, ops: &OpList) {
    let mut flags = vk::PartitionedAccelerationStructureFlagsNV::default().enable_partition_translation(true);
    let input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .instance_count(d.instances)
        .max_instance_per_partition_count(d.per_partition)
        .partition_count(d.partitions)
        .max_instance_in_global_partition_count(d.global)
        .push(&mut flags);
    let info = vk::BuildPartitionedAccelerationStructureInfoNV {
        input,
        src_acceleration_structure_data: if first { 0 } else { dst },
        dst_acceleration_structure_data: dst,
        scratch_data: scratch,
        src_infos: ops.cmds.addr,
        src_infos_count: ops.count.addr,
        ..Default::default()
    };
    gpu.ptlas.as_ref().unwrap().cmd_build_partitioned_acceleration_structures(cmd, &info);
    gpu.full_barrier(cmd, false);
}

#[derive(Clone, Copy)]
struct Stat {
    median: f64,
    min: f64,
    max: f64,
}

impl std::fmt::Display for Stat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3} (min {:.3} max {:.3})", self.median, self.min, self.max)
    }
}

/// GPU ms of the timed half of `samples` submissions; `rec(cmd, timed)` records one operation.
unsafe fn time_ms(gpu: &Gpu, case: &str, samples: u32, mut rec: impl FnMut(vk::CommandBuffer, bool)) -> Stat {
    let mut v = Vec::new();
    for _ in 0..samples {
        let cmd = gpu.begin();
        gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 40, 2);
        rec(cmd, false);
        gpu.full_barrier(cmd, false);
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, gpu.timestamps, 40);
        rec(cmd, true);
        gpu.full_barrier(cmd, false);
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, gpu.timestamps, 41);
        gpu.submit_wait(cmd, case, "timed sample");
        let mut ticks = [0u64; 2];
        gpu.device
            .get_query_pool_results(gpu.timestamps, 40, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            .unwrap();
        v.push((ticks[1] - ticks[0]) as f64 * f64::from(gpu.limits.timestamp_period) / 1e6);
        std::thread::sleep(std::time::Duration::from_millis(6));
    }
    v.sort_by(f64::total_cmp);
    Stat { median: v[v.len() / 2], min: v[0], max: v[v.len() - 1] }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CanopyRoot {
    out: u64,
    tlas: u64,
    width: u32,
    height: u32,
    mode: u32,
    n: u32,
    cam: [f32; 3],
    extent: f32,
    sun: [f32; 3],
    tmax: f32,
}

const _: () = assert!(size_of::<CanopyRoot>() == 64);

unsafe fn canopy(gpu: &Gpu, cmd: vk::CommandBuffer, pipe: vk::Pipeline, root: CanopyRoot) {
    gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
    gpu.push(cmd, &root);
    gpu.device.cmd_dispatch(cmd, root.n.div_ceil(64), 1, 1);
    gpu.full_barrier(cmd, false);
}

/// One quad cluster over [0.1, 0.9]^2 at z 0.5 as a cluster BLAS; returns its address.
unsafe fn quad_blas(ctx: &Ctx, case: &str) -> u64 {
    let clusters = vec![grid(7000, 1, 1, [0.1, 0.1, 0.9, 0.9], |_, _| 0.5, 1)];
    let up = ctx.upload_clusters(&clusters, 0, &[]);
    let cmd = ctx.gpu.begin();
    let clas = ctx.build_clas(cmd, &up, &clusters, data_flags(), None, None);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0]], None);
    ctx.gpu.submit_wait(cmd, case, "quad blas");
    blas.addrs.read::<u64>(1)[0]
}

/// Offset of a buffer's aligned address inside its VkBuffer.
unsafe fn offset(gpu: &Gpu, b: &Buf) -> u64 {
    b.addr - gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(b.buffer))
}

/// Last written byte (exclusive) and written dword count of a sentinel-filled readback.
unsafe fn high_water(b: &Buf, size: u64) -> (u64, u64) {
    let words = std::slice::from_raw_parts(b.ptr.cast::<u32>(), (size / 4) as usize);
    let last = words.iter().rposition(|&w| w != SENTINEL).map_or(0, |i| (i as u64 + 1) * 4);
    let written = words.iter().filter(|&&w| w != SENTINEL).count() as u64;
    (last, written * 4)
}

fn mb(b: u64) -> String {
    format!("{:.1}", b as f64 / 1e6)
}

// ---------------------------------------------------------------------------------------------
// ptlas_mem

pub(crate) unsafe fn case_mem(ctx: &Ctx, case: &str) -> (bool, String) {
    let gpu = &ctx.gpu;
    if !gpu.has_ptlas {
        finish(case, "UNSUPPORTED", "VK_NV_partitioned_acceleration_structure absent");
    }
    // Host sizes: capacity x partitions x per-partition cap x global capacity.
    for c in [131_072u32, 786_432, 1_048_576, 2_600_000, 4_194_304] {
        for p in [1024u32, 2048, 4096] {
            for per in [1000u32, c.div_ceil(p).max(1)] {
                if per * p < c {
                    continue;
                }
                for g in [0u32, 4096, 100_000] {
                    let (s, sc) = sizes(gpu, &PtlasDesc { instances: c, per_partition: per, partitions: p, global: g });
                    println!(
                        "ROW mem_sizes instances={c} partitions={p} per_partition={per} global={g} result_max={s} scratch={sc} result_B_per_instance={:.1}",
                        s as f64 / f64::from(c)
                    );
                }
            }
        }
    }
    for per in [256u32, 1000, 4000, 16000, 65536] {
        let (s, sc) = sizes(gpu, &PtlasDesc { instances: 786_432, per_partition: per, partitions: 2048, global: 4096 });
        println!("ROW mem_per_partition instances=786432 partitions=2048 per_partition={per} result_max={s} scratch={sc}");
    }

    let quad = quad_blas(ctx, case);
    let check_pipe = gpu.pipeline(&spv!("canopy"));
    let mut ok = true;
    let mut detail = Vec::new();
    // (instances, partitions, per-partition cap, instances per used partition)
    for (c, p, per, fill) in [(786_432u32, 1024u32, 768u32, 768u32), (786_432, 2048, 1000, 1000), (2_600_000, 4096, 1000, 1000)] {
        let d = PtlasDesc { instances: c, per_partition: per, partitions: p, global: 4096 };
        let (size, scratch_size) = sizes(gpu, &d);
        let side = (f64::from(c).sqrt().ceil()) as u32;
        for real in [false, true] {
            let dst = gpu.buffer(size, Mem::Device, 256);
            let scratch = gpu.buffer(scratch_size, Mem::Device, 256);
            let blas = if real { quad } else { 0 };
            let rec = |i: u32, dz: f32| {
                let m = translate([(i % side) as f32, (i / side) as f32, dz]);
                write_rec(i, i / fill, i, m, blas, Some(world_aabb(&m, [0.1, 0.1, 0.4, 0.9, 0.9, 0.6])))
            };
            let recs: Vec<WriteRec> = (0..c).map(|i| rec(i, 0.0)).collect();
            let args = gpu.upload(&recs);
            let moved: Vec<WriteRec> = (0..c).map(|i| rec(i, 0.25)).collect();
            let args2 = gpu.upload(&moved);
            drop((recs, moved));
            let ops = OpList::new(ctx, &[Op::write(args.addr, c)]);
            let ops2 = OpList::new(ctx, &[Op::write(args2.addr, c)]);
            let rb_dst = gpu.buffer(size, Mem::Readback, 16);
            let rb_scratch = gpu.buffer(scratch_size, Mem::Readback, 16);

            let cmd = gpu.begin();
            gpu.device.cmd_fill_buffer(cmd, dst.buffer, 0, vk::WHOLE_SIZE, SENTINEL);
            gpu.device.cmd_fill_buffer(cmd, scratch.buffer, 0, vk::WHOLE_SIZE, SENTINEL);
            gpu.full_barrier(cmd, false);
            gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 4);
            gpu.timestamp(cmd, 0);
            record(gpu, cmd, &d, dst.addr, scratch.addr, true, &ops);
            gpu.timestamp(cmd, 1);
            let dst_base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(dst.buffer));
            let sc_base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(scratch.buffer));
            gpu.device.cmd_copy_buffer(cmd, dst.buffer, rb_dst.buffer, &[vk::BufferCopy { src_offset: dst.addr - dst_base, dst_offset: offset(gpu, &rb_dst), size }]);
            gpu.device.cmd_copy_buffer(
                cmd,
                scratch.buffer,
                rb_scratch.buffer,
                &[vk::BufferCopy { src_offset: scratch.addr - sc_base, dst_offset: offset(gpu, &rb_scratch), size: scratch_size }],
            );
            gpu.submit_wait(cmd, case, "ptlas_mem build");
            let build_ms = gpu.read_timestamps(2)[0] / 1e3;
            let (used, written) = high_water(&rb_dst, size);
            let (sc_used, _) = high_water(&rb_scratch, scratch_size);

            // Every instance answers a ray aimed at its own cell (misses for null BLAS).
            let out = gpu.buffer(u64::from(c) * 4, Mem::Readback, 16);
            let cmd = gpu.begin();
            let root = CanopyRoot { out: out.addr, tlas: dst.addr, width: side, height: 0, mode: 2, n: c, cam: [0.0; 3], extent: 0.0, sun: [0.0; 3], tmax: 4.0 };
            canopy(gpu, cmd, check_pipe, root);
            gpu.submit_wait(cmd, case, "ptlas_mem check");
            let got = std::slice::from_raw_parts(out.ptr.cast::<u32>(), c as usize);
            let wrong = got.iter().enumerate().filter(|&(i, &v)| v != if real { i as u32 } else { u32::MAX }).count();

            // In place: rewrite every instance, then read the high-water mark again.
            let cmd = gpu.begin();
            gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 4);
            gpu.timestamp(cmd, 0);
            record(gpu, cmd, &d, dst.addr, scratch.addr, false, &ops2);
            gpu.timestamp(cmd, 1);
            gpu.device.cmd_copy_buffer(cmd, dst.buffer, rb_dst.buffer, &[vk::BufferCopy { src_offset: dst.addr - dst_base, dst_offset: offset(gpu, &rb_dst), size }]);
            gpu.submit_wait(cmd, case, "ptlas_mem rewrite");
            let rewrite_ms = gpu.read_timestamps(2)[0] / 1e3;
            let (used2, _) = high_water(&rb_dst, size);

            ok &= wrong == 0;
            let line = format!(
                "instances={c} partitions={p} per_partition={per} used_partitions={} global=4096 blas={} result_max={size} ({} MB) built_high_water={used} ({} MB, {:.1} B/instance) written_bytes={written} after_full_rewrite={used2} scratch={scratch_size} scratch_high_water={sc_used} full_build_ms={build_ms:.2} full_rewrite_ms={rewrite_ms:.2} wrong_instance_rays={wrong}",
                c.div_ceil(fill),
                if real { "real" } else { "null" },
                mb(size),
                mb(used),
                used as f64 / f64::from(c)
            );
            println!("ROW mem_built {line}");
            detail.push(format!("[{line}]"));
            for b in [&dst, &scratch, &args, &args2, &rb_dst, &rb_scratch, &out] {
                gpu.free(b);
            }
        }
    }
    (ok, detail.join(" "))
}

// ---------------------------------------------------------------------------------------------
// ptlas_cost

/// Instances written per frame and the partitions they touch.
#[derive(Clone, Copy, PartialEq)]
enum Touch {
    /// Spread evenly over min(writes, partitions) partitions.
    Spread,
    /// Contiguous instances: the fewest partitions.
    Packed,
}

fn chosen(c: u32, p: u32, w: u32, touch: Touch) -> (Vec<u32>, u32) {
    let per = c / p;
    match touch {
        Touch::Packed => ((0..w).collect(), w.div_ceil(per)),
        Touch::Spread => {
            let t = p.min(w);
            let step = p / t;
            ((0..w).map(|k| (k % t) * step * per + k / t).collect(), t)
        }
    }
}

fn grid_pos(i: u32, side: u32, dz: f32) -> [f32; 12] {
    translate([(i % side) as f32 * 1.5, (i / side) as f32 * 1.5, dz])
}

/// Write record on the cost grid with its world-space explicit AABB (both test BLAS fit).
fn grid_rec(i: u32, partition: u32, side: u32, dz: f32, blas: u64) -> WriteRec {
    let m = grid_pos(i, side, dz);
    write_rec(i, partition, i, m, blas, Some(world_aabb(&m, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])))
}

pub(crate) unsafe fn case_cost(ctx: &Ctx, case: &str, args: &[String]) -> (bool, String) {
    let gpu = &ctx.gpu;
    if !gpu.has_ptlas {
        finish(case, "UNSUPPORTED", "VK_NV_partitioned_acceleration_structure absent");
    }
    let samples: u32 = std::env::var("PRB5_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(11);
    // Arguments: capacities to sweep (default 131072 1048576 4194304), or `global` for only the
    // global-partition rows.
    let caps: Vec<u32> = match args {
        [] => vec![131_072, 1_048_576, 4_194_304],
        [g] if g == "global" => vec![],
        _ => args.iter().map(|a| a.parse().unwrap()).collect(),
    };
    // Two BLAS of different extent inside one explicit AABB, so UPDATE_INSTANCE can swap them.
    let clusters = base_clusters(ctx);
    let up = ctx.upload_clusters(&clusters, 0, &[]);
    let cmd = gpu.begin();
    let clas = ctx.build_clas(cmd, &up, &clusters, data_flags(), None, None);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0, 1, 2, 3], vec![0, 2]], None);
    gpu.submit_wait(cmd, case, "blas");
    let b: Vec<u64> = blas.addrs.read(2);
    let ops = OpList::new(ctx, &[]);
    let ops2 = OpList::new(ctx, &[]);
    let mut rows = 0;

    for &c in &caps {
        let side = f64::from(c).sqrt().ceil() as u32;
        for p in [1u32, 16, 128, 1024, 4096] {
            if p > c {
                continue;
            }
            let per = c / p;
            let d = PtlasDesc { instances: c, per_partition: per, partitions: p, global: 0 };
            let (size, scratch_size) = sizes(gpu, &d);
            let dst = gpu.buffer(size, Mem::Device, 256);
            let scratch = gpu.buffer(scratch_size, Mem::Device, 256);
            let full: Vec<WriteRec> = (0..c).map(|i| grid_rec(i, i / per, side, 0.0, b[0])).collect();
            let full_args = gpu.upload(&full);
            drop(full);
            ops.set(&[Op::write(full_args.addr, c)]);
            let full_ms = time_ms(gpu, case, samples.min(5), |cmd, _| record(gpu, cmd, &d, dst.addr, scratch.addr, true, &ops));
            println!("ROW cost_full capacity={c} partitions={p} per_partition={per} result_max={size} scratch={scratch_size} full_build_ms={full_ms}");
            for w in [10_000u32, 100_000, 1_000_000] {
                if w > c {
                    continue;
                }
                for touch in [Touch::Spread, Touch::Packed] {
                    let (idx, touched) = chosen(c, p, w, touch);
                    if touch == Touch::Packed && touched == p.min(w) {
                        continue;
                    }
                    let mode = if touch == Touch::Spread { "spread" } else { "packed" };
                    let wa: Vec<WriteRec> = idx.iter().map(|&i| grid_rec(i, i / per, side, 0.01, b[0])).collect();
                    let wb: Vec<WriteRec> = idx.iter().map(|&i| grid_rec(i, i / per, side, -0.01, b[0])).collect();
                    let (a1, a2) = (gpu.upload(&wa), gpu.upload(&wb));
                    drop((wa, wb));
                    ops.set(&[Op::write(a1.addr, w)]);
                    ops2.set(&[Op::write(a2.addr, w)]);
                    let t = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
                    println!("ROW cost op=write capacity={c} partitions={p} per_partition={per} writes={w} touch={mode} touched={touched} ms={t}");
                    gpu.free(&a1);
                    gpu.free(&a2);
                    let ua: Vec<UpdateRec> = idx.iter().map(|&i| update_rec(i, b[1])).collect();
                    let ub: Vec<UpdateRec> = idx.iter().map(|&i| update_rec(i, b[0])).collect();
                    let (u1, u2) = (gpu.upload(&ua), gpu.upload(&ub));
                    ops.set(&[Op::update(u1.addr, w)]);
                    ops2.set(&[Op::update(u2.addr, w)]);
                    let t = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
                    println!("ROW cost op=update capacity={c} partitions={p} per_partition={per} writes={w} touch={mode} touched={touched} ms={t}");
                    gpu.free(&u1);
                    gpu.free(&u2);
                    rows += 2;
                }
            }
            if p >= 1024 && c >= 1_048_576 {
                // Rebase: a translation for every partition.
                let tr: Vec<TranslateRec> =
                    (0..p).map(|i| TranslateRec { partition_index: i, partition_translation: [0.5 * (i % 2) as f32, 0.0, 0.0] }).collect();
                let tr2: Vec<TranslateRec> = (0..p).map(|i| TranslateRec { partition_index: i, partition_translation: [0.0; 3] }).collect();
                let (t1, t2) = (gpu.upload(&tr), gpu.upload(&tr2));
                ops.set(&[Op::translate(t1.addr, p)]);
                ops2.set(&[Op::translate(t2.addr, p)]);
                let t = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
                println!("ROW cost op=translate_all capacity={c} partitions={p} per_partition={per} writes={p} ms={t}");
                gpu.free(&t1);
                gpu.free(&t2);
                // Disable 100k instances (null BLAS, explicit AABB kept), then bring them back.
                let (idx, touched) = chosen(c, p, 100_000, Touch::Spread);
                let off: Vec<WriteRec> = idx.iter().map(|&i| grid_rec(i, i / per, side, 0.0, 0)).collect();
                let on: Vec<UpdateRec> = idx.iter().map(|&i| update_rec(i, b[0])).collect();
                let (n1, n2) = (gpu.upload(&off), gpu.upload(&on));
                ops.set(&[Op::write(n1.addr, 100_000)]);
                ops2.set(&[Op::update(n2.addr, 100_000)]);
                let t_off = time_ms(gpu, case, samples, |cmd, timed| {
                    if timed {
                        record(gpu, cmd, &d, dst.addr, scratch.addr, false, &ops);
                    } else {
                        record(gpu, cmd, &d, dst.addr, scratch.addr, false, &ops2);
                    }
                });
                let t_on = time_ms(gpu, case, samples, |cmd, timed| {
                    if timed {
                        record(gpu, cmd, &d, dst.addr, scratch.addr, false, &ops2);
                    } else {
                        record(gpu, cmd, &d, dst.addr, scratch.addr, false, &ops);
                    }
                });
                println!(
                    "ROW cost op=write_null capacity={c} partitions={p} per_partition={per} writes=100000 touch=spread touched={touched} ms={t_off}"
                );
                println!(
                    "ROW cost op=update_from_null capacity={c} partitions={p} per_partition={per} writes=100000 touch=spread touched={touched} ms={t_on}"
                );
                gpu.free(&n1);
                gpu.free(&n2);
            }
            gpu.free(&dst);
            gpu.free(&scratch);
            gpu.free(&full_args);
        }
    }

    // Global partition: G movers rewritten every frame, alone and beside 10k cell writes.
    let c = 1_048_576u32;
    let side = f64::from(c).sqrt().ceil() as u32;
    let p = 1024u32;
    for g in [0u32, 1000, 10_000, 100_000] {
        let per = (c - g).div_ceil(p);
        let d = PtlasDesc { instances: c, per_partition: per, partitions: p, global: g };
        let (size, scratch_size) = sizes(gpu, &d);
        let dst = gpu.buffer(size, Mem::Device, 256);
        let scratch = gpu.buffer(scratch_size, Mem::Device, 256);
        let part = |i: u32| if i >= c - g { GLOBAL } else { i / per };
        let full: Vec<WriteRec> = (0..c).map(|i| grid_rec(i, part(i), side, 0.0, b[0])).collect();
        let full_args = gpu.upload(&full);
        drop(full);
        ops.set(&[Op::write(full_args.addr, c)]);
        let cmd = gpu.begin();
        record(gpu, cmd, &d, dst.addr, scratch.addr, true, &ops);
        gpu.submit_wait(cmd, case, "global full build");
        let movers = |dz: f32| -> Vec<WriteRec> { (c - g..c).map(|i| grid_rec(i, GLOBAL, side, dz, b[0])).collect() };
        let (m1, m2) = (gpu.upload(&movers(0.01)), gpu.upload(&movers(-0.01)));
        let (idx, touched) = chosen(c - g - (c - g) % p, p, 10_000, Touch::Spread);
        let cells = |dz: f32| -> Vec<WriteRec> { idx.iter().map(|&i| grid_rec(i, part(i), side, dz, b[0])).collect() };
        let (c1, c2) = (gpu.upload(&cells(0.01)), gpu.upload(&cells(-0.01)));
        ops.set(&[Op::write(c1.addr, 10_000)]);
        ops2.set(&[Op::write(c2.addr, 10_000)]);
        let t = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
        println!("ROW cost op=global_cells_only capacity={c} partitions={p} global={g} writes=10000 touched={touched} ms={t}");
        if g == 0 {
            for bf in [&dst, &scratch, &full_args, &m1, &m2, &c1, &c2] {
                gpu.free(bf);
            }
            continue;
        }
        ops.set(&[Op::write(m1.addr, g)]);
        ops2.set(&[Op::write(m2.addr, g)]);
        let t = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
        println!("ROW cost op=global_writes capacity={c} partitions={p} global={g} writes={g} result_max={size} ms={t}");
        // One write array per build: a second command of the same type is dropped (ptlas_ops).
        let both = |dz: f32| -> Vec<WriteRec> { movers(dz).into_iter().chain(cells(dz)).collect() };
        let (b1, b2) = (gpu.upload(&both(0.01)), gpu.upload(&both(-0.01)));
        ops.set(&[Op::write(b1.addr, g + 10_000)]);
        ops2.set(&[Op::write(b2.addr, g + 10_000)]);
        let t = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
        println!("ROW cost op=global_plus_cells capacity={c} partitions={p} global={g} writes={} touched={touched} ms={t}", g + 10_000);
        rows += 2;
        for bf in [&dst, &scratch, &full_args, &m1, &m2, &c1, &c2, &b1, &b2] {
            gpu.free(bf);
        }
    }
    (rows > 0, format!("rows={rows} samples={samples}"))
}

// ---------------------------------------------------------------------------------------------
// ptlas_part

const PARTS: u32 = 32;
const SPACING: f32 = 8.0;

fn rng(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed.max(1);
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// A clump of random leaf triangles inside a sphere of radius 1.6 m: two clusters of 84.
fn clump(id: u32, seed: u32) -> Vec<Cluster> {
    let mut r = rng(seed);
    (0..2)
        .map(|k| {
            let mut verts = Vec::new();
            for _ in 0..84 {
                let c = loop {
                    let p = [r() * 2.0 - 1.0, r() * 2.0 - 1.0, r() * 2.0 - 1.0];
                    if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] < 1.0 {
                        break p.map(|v| v * 1.45);
                    }
                };
                for _ in 0..3 {
                    verts.push([c[0] + 0.15 * (r() - 0.5), c[1] + 0.15 * (r() - 0.5), c[2] + 0.15 * (r() - 0.5)]);
                }
            }
            Cluster {
                id: id + k,
                tris: (0..84).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect(),
                verts,
                index_bytes: 1,
                truncate: 0,
                base_geom: 0,
                geom_flags: GF_OPAQUE | GF_CULL_DISABLE,
                tri_gflags: None,
                omm_index: None,
            }
        })
        .collect()
}

/// A 0.5 m square trunk, 9 m tall.
fn trunk() -> Cluster {
    let h = 0.25;
    let mut verts = Vec::new();
    for z in [0.0, 9.0] {
        for (x, y) in [(-h, -h), (h, -h), (h, h), (-h, h)] {
            verts.push([x, y, z]);
        }
    }
    let tris = (0..4u32).flat_map(|s| {
        let n = (s + 1) % 4;
        [[s, n, n + 4], [s, n + 4, s + 4]]
    });
    Cluster {
        id: 9000,
        verts,
        tris: tris.collect(),
        index_bytes: 1,
        truncate: 0,
        base_geom: 0,
        geom_flags: GF_OPAQUE | GF_CULL_DISABLE,
        tri_gflags: None,
        omm_index: None,
    }
}

fn morton(x: u32, y: u32) -> u64 {
    let spread = |v: u32| {
        let mut v = u64::from(v);
        v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
        v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        (v | (v << 1)) & 0x5555_5555_5555_5555
    };
    spread(x) | (spread(y) << 1)
}

struct Forest {
    side: u32,
    copies: u32,
    /// Copy position by copy index (copies are in Morton order).
    pos: Vec<[f32; 2]>,
    /// Part j of copy c: (variant, local transform).
    parts: Vec<(u32, [f32; 12])>,
}

impl Forest {
    fn new(side: u32) -> Forest {
        let mut cells: Vec<(u32, u32)> = (0..side * side).map(|i| (i % side, i / side)).collect();
        cells.sort_by_key(|&(x, y)| morton(x, y));
        let mut r = rng(77);
        let half = side as f32 * SPACING / 2.0;
        let pos: Vec<[f32; 2]> = cells
            .iter()
            .map(|&(x, y)| [x as f32 * SPACING - half + 2.0 * (r() - 0.5), y as f32 * SPACING - half + 2.0 * (r() - 0.5)])
            .collect();
        let copies = side * side;
        let mut parts = Vec::with_capacity((copies * PARTS) as usize);
        for _ in 0..copies {
            for _ in 0..PARTS {
                let (a, rad, z) = (r() * std::f32::consts::TAU, 3.4 * r().sqrt(), 6.5 + 4.5 * r());
                let rot = r() * std::f32::consts::TAU;
                let (s, c) = rot.sin_cos();
                parts.push(((r() * 4.0) as u32 % 4, [c, -s, 0.0, rad * a.cos(), s, c, 0.0, rad * a.sin(), 0.0, 0.0, 1.0, z]));
            }
        }
        Forest { side, copies, pos, parts }
    }

    fn part_base(&self) -> u32 {
        self.copies
    }

    fn instances(&self) -> u32 {
        self.copies * (1 + PARTS)
    }

    /// World transform of part `j` of copy `c`, nudged by `dz` (a pose change).
    fn part_m(&self, c: u32, j: u32, dz: f32) -> [f32; 12] {
        let mut m = self.parts[(c * PARTS + j) as usize].1;
        m[3] += self.pos[c as usize][0];
        m[7] += self.pos[c as usize][1];
        m[11] += dz;
        m
    }
}

/// How copies and parts map to partitions.
struct Layout {
    name: String,
    copies_per_group: u32,
    /// Bodies and parts of a group in separate partitions.
    split: bool,
    /// Per copy: its parts sit in the global partition.
    global: Vec<bool>,
}

impl Layout {
    fn partitions(&self, f: &Forest) -> u32 {
        f.copies.div_ceil(self.copies_per_group) * if self.split { 2 } else { 1 }
    }
    fn per_partition(&self) -> u32 {
        if self.split { self.copies_per_group * PARTS } else { self.copies_per_group * (1 + PARTS) }
    }
    fn body(&self, c: u32) -> u32 {
        let g = c / self.copies_per_group;
        if self.split { 2 * g } else { g }
    }
    fn part(&self, c: u32) -> u32 {
        let g = c / self.copies_per_group;
        if self.global[c as usize] {
            GLOBAL
        } else if self.split {
            2 * g + 1
        } else {
            g
        }
    }
    fn global_instances(&self) -> u32 {
        self.global.iter().filter(|&&g| g).count() as u32 * PARTS
    }
}

pub(crate) unsafe fn case_part(ctx: &Ctx, case: &str, args: &[String]) -> (bool, String) {
    let gpu = &ctx.gpu;
    if !gpu.has_ptlas {
        finish(case, "UNSUPPORTED", "VK_NV_partitioned_acceleration_structure absent");
    }
    let samples: u32 = std::env::var("PRB5_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(11);
    let side: u32 = args.first().map_or(155, |a| a.parse().unwrap());
    let forest = Forest::new(side);
    let f = &forest;
    let n = f.instances();
    let pipe = gpu.pipeline(&spv!("canopy"));

    // BLAS: [trunk, clump 0..3].
    let mut clusters = vec![trunk()];
    for v in 0..4 {
        clusters.extend(clump(100 + 10 * v, 1234 + 97 * v));
    }
    let up = ctx.upload_clusters(&clusters, 0, &[]);
    let cmd = gpu.begin();
    let clas = ctx.build_clas(cmd, &up, &clusters, data_flags(), None, None);
    let groups = vec![vec![0], vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]];
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &groups, None);
    gpu.submit_wait(cmd, case, "forest blas");
    let blas_addrs: Vec<u64> = blas.addrs.read(5);
    let b = &blas_addrs;
    let clump_aabb = |v: u32| {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for c in &clusters[1 + 2 * v as usize..3 + 2 * v as usize] {
            for p in &c.verts {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
    };
    // One explicit AABB covering every clump variant, so UPDATE_INSTANCE can swap variants.
    let aabb = (0..4).map(clump_aabb).fold([f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN], |a, b| {
        [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2]), a[3].max(b[3]), a[4].max(b[4]), a[5].max(b[5])]
    });

    // Reference: a classic TLAS over the same instances in index order.
    let mut descs = Vec::with_capacity(n as usize);
    for c in 0..f.copies {
        let p = f.pos[c as usize];
        descs.push(InstanceDesc { m: translate([p[0], p[1], 0.0]), custom: c, mask: 0xff, flags: 0, blas_index: 0 });
    }
    for c in 0..f.copies {
        for j in 0..PARTS {
            let v = f.parts[(c * PARTS + j) as usize].0;
            descs.push(InstanceDesc { m: f.part_m(c, j, 0.0), custom: c, mask: 0xff, flags: 0, blas_index: 1 + v });
        }
    }
    let cmd = gpu.begin();
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &descs);
    gpu.submit_wait(cmd, case, "reference tlas");
    drop(descs);

    let (rw, rh) = (1920u32, 1080u32);
    let rays = rw * rh;
    let sun = {
        let (el, az) = (20f32.to_radians(), 30f32.to_radians());
        [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]
    };
    let root = |out: u64, tlas: u64, mode: u32| CanopyRoot {
        out,
        tlas,
        width: rw,
        height: rh,
        mode,
        n: rays,
        cam: [0.0, 0.0, 1.7],
        extent: 200.0,
        sun,
        tmax: 2000.0,
    };
    let ref_out = [gpu.buffer(u64::from(rays) * 16, Mem::Readback, 16), gpu.buffer(u64::from(rays) * 16, Mem::Readback, 16)];
    let dev_out = gpu.buffer(u64::from(rays) * 16, Mem::Device, 16);
    let mut ref_ms = Vec::new();
    for mode in 0..2 {
        let cmd = gpu.begin();
        canopy(gpu, cmd, pipe, root(ref_out[mode].addr, tlas, mode as u32));
        gpu.submit_wait(cmd, case, "reference trace");
        ref_ms.push(time_ms(gpu, case, samples, |cmd, _| canopy(gpu, cmd, pipe, root(dev_out.addr, tlas, mode as u32))));
    }
    let hit_share = |b: &Buf| {
        let r = std::slice::from_raw_parts(b.ptr.cast::<[u32; 4]>(), rays as usize);
        r.iter().filter(|o| o[0] != u32::MAX).count() as f64 / f64::from(rays)
    };
    println!(
        "ROW part_ref instances={n} copies={} parts_per_copy={PARTS} classic_tlas camera_ms={} shadow_ms={} camera_hit_share={:.3} shadow_hit_share={:.3}",
        f.copies,
        ref_ms[0],
        ref_ms[1],
        hit_share(&ref_out[0]),
        hit_share(&ref_out[1])
    );

    // Animated copies: within `radius` of the camera (f32::MAX: every copy).
    let animated = |radius: f32| -> Vec<u32> {
        (0..f.copies)
            .filter(|&c| {
                let p = f.pos[c as usize];
                p[0] * p[0] + p[1] * p[1] < radius * radius
            })
            .collect()
    };
    let sets: Vec<(&str, Vec<u32>)> = vec![("r80m", animated(80.0)), ("r175m", animated(175.0)), ("all", animated(f32::MAX))];

    let none = vec![false; f.copies as usize];
    let near: Vec<bool> = {
        let set = animated(175.0);
        let mut v = none.clone();
        for c in set {
            v[c as usize] = true;
        }
        v
    };
    let mut layouts = Vec::new();
    for s in [1000u32, 4000, 16000] {
        layouts.push(Layout { name: format!("inst{s}"), copies_per_group: s / (1 + PARTS), split: false, global: none.clone() });
    }
    layouts.push(Layout { name: "rec1000".into(), copies_per_group: 1000, split: false, global: none.clone() });
    for s in [1000u32, 4000] {
        layouts.push(Layout { name: format!("split{s}"), copies_per_group: s / (1 + PARTS), split: true, global: none.clone() });
    }
    // Morton-aligned groups (square blocks): partition shape without the cut points above.
    for cpg in [16u32, 64, 256, 1024] {
        layouts.push(Layout { name: format!("aligned{}", cpg * (1 + PARTS)), copies_per_group: cpg, split: false, global: none.clone() });
    }
    // Parts of the copies within 175 m in the global partition, the rest as inst1000.
    layouts.push(Layout { name: "global_r175m".into(), copies_per_group: 1000 / (1 + PARTS), split: false, global: near });

    let ops = OpList::new(ctx, &[]);
    let ops2 = OpList::new(ctx, &[]);
    let mut ok = true;
    let mut detail = Vec::new();
    for l in &layouts {
        let d = PtlasDesc { instances: n, per_partition: l.per_partition(), partitions: l.partitions(f), global: l.global_instances() };
        let (size, scratch_size) = sizes(gpu, &d);
        let dst = gpu.buffer(size, Mem::Device, 256);
        let scratch = gpu.buffer(scratch_size, Mem::Device, 256);
        let part_rec = move |c: u32, j: u32, dz: f32| {
            let v = f.parts[(c * PARTS + j) as usize].0;
            let m = f.part_m(c, j, dz);
            write_rec(f.part_base() + c * PARTS + j, l.part(c), c, m, b[1 + v as usize], Some(world_aabb(&m, aabb)))
        };
        let mut full: Vec<WriteRec> = (0..f.copies)
            .map(|c| {
                let p = f.pos[c as usize];
                write_rec(c, l.body(c), c, translate([p[0], p[1], 0.0]), b[0], None)
            })
            .collect();
        for c in 0..f.copies {
            for j in 0..PARTS {
                full.push(part_rec(c, j, 0.0));
            }
        }
        let full_args = gpu.upload(&full);
        drop(full);
        ops.set(&[Op::write(full_args.addr, n)]);
        let full_ms = time_ms(gpu, case, 3, |cmd, _| record(gpu, cmd, &d, dst.addr, scratch.addr, true, &ops));

        // Trace parity with the classic TLAS, then trace cost.
        let out = gpu.buffer(u64::from(rays) * 16, Mem::Readback, 16);
        let mut trace = Vec::new();
        for mode in 0..2u32 {
            let cmd = gpu.begin();
            canopy(gpu, cmd, pipe, root(out.addr, dst.addr, mode));
            gpu.submit_wait(cmd, case, "ptlas trace");
            let got = std::slice::from_raw_parts(out.ptr.cast::<[u32; 4]>(), rays as usize);
            let want = std::slice::from_raw_parts(ref_out[mode as usize].ptr.cast::<[u32; 4]>(), rays as usize);
            let mut ids = 0;
            let mut tdiff = 0;
            for (g, w) in got.iter().zip(want) {
                if g[..3] != w[..3] {
                    ids += 1;
                } else if g[0] != u32::MAX {
                    let (tg, tw) = (f32::from_bits(g[3]), f32::from_bits(w[3]));
                    if (tg - tw).abs() > 1e-5 * tw.max(1.0) {
                        tdiff += 1;
                    }
                }
            }
            // Ties between overlapping leaves can pick either: a few id differences are allowed.
            ok &= ids * 10_000 < rays && tdiff * 10_000 < rays;
            let ms = time_ms(gpu, case, samples, |cmd, _| canopy(gpu, cmd, pipe, root(dev_out.addr, dst.addr, mode)));
            trace.push((ms, ids, tdiff));
        }
        println!(
            "ROW part_trace layout={} partitions={} per_partition={} global={} result_max={size} scratch={scratch_size} full_build_ms={full_ms} camera_ms={} camera_id_diff={} camera_t_diff={} shadow_ms={} shadow_id_diff={} shadow_t_diff={}",
            l.name,
            d.partitions,
            d.per_partition,
            d.global,
            trace[0].0,
            trace[0].1,
            trace[0].2,
            trace[1].0,
            trace[1].1,
            trace[1].2
        );
        detail.push(format!("{}:camera_ms={:.3},shadow_ms={:.3}", l.name, trace[0].0.median, trace[1].0.median));

        // Pose changes: every part of each animated copy rewritten per frame.
        for (set_name, set) in &sets {
            let writes = set.len() as u32 * PARTS;
            let mut touched: Vec<u32> = set.iter().map(|&c| l.part(c)).collect();
            touched.sort_unstable();
            touched.dedup();
            let in_touched: u64 = touched.len() as u64 * u64::from(d.per_partition);
            let recs = |dz: f32| -> Vec<WriteRec> { set.iter().flat_map(|&c| (0..PARTS).map(move |j| part_rec(c, j, dz))).collect() };
            let (a1, a2) = (gpu.upload(&recs(0.05)), gpu.upload(&recs(-0.05)));
            ops.set(&[Op::write(a1.addr, writes)]);
            ops2.set(&[Op::write(a2.addr, writes)]);
            let t = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
            let ups = |v: u32| -> Vec<UpdateRec> {
                set.iter()
                    .flat_map(|&c| (0..PARTS).map(move |j| update_rec(f.part_base() + c * PARTS + j, b[1 + ((f.parts[(c * PARTS + j) as usize].0 + v) % 4) as usize])))
                    .collect()
            };
            let (u1, u2) = (gpu.upload(&ups(1)), gpu.upload(&ups(0)));
            ops.set(&[Op::update(u1.addr, writes)]);
            ops2.set(&[Op::update(u2.addr, writes)]);
            let tu = time_ms(gpu, case, samples, |cmd, timed| record(gpu, cmd, &d, dst.addr, scratch.addr, false, if timed { &ops2 } else { &ops }));
            println!(
                "ROW part_build layout={} set={set_name} animated_copies={} part_writes={writes} touched_partitions={} capacity_in_touched={in_touched} write_ms={t} update_ms={tu}",
                l.name,
                set.len(),
                touched.len()
            );
            for bf in [&a1, &a2, &u1, &u2] {
                gpu.free(bf);
            }
        }
        for bf in [&dst, &scratch, &full_args, &out] {
            gpu.free(bf);
        }
    }
    let _ = f.side;
    (ok, format!("instances={n} {}", detail.join(" ")))
}

// ---------------------------------------------------------------------------------------------
// ptlas_ops

#[derive(Clone, Copy, PartialEq)]
enum Target {
    Quad,
    /// A quad off the cell centre: the instance-check ray misses it.
    Off,
    Null,
}

/// Several indirect commands in one PTLAS build, over cell and global partitions. Each test
/// starts from every instance on the centred quad, applies its commands, and the
/// instance-check rays must hit exactly the instances left on the centred quad.
pub(crate) unsafe fn case_ops(ctx: &Ctx, case: &str) -> (bool, String) {
    let gpu = &ctx.gpu;
    if !gpu.has_ptlas {
        finish(case, "UNSUPPORTED", "VK_NV_partitioned_acceleration_structure absent");
    }
    let quad = quad_blas(ctx, case);
    let off = {
        let clusters = vec![grid(7001, 1, 1, [0.1, 0.1, 0.3, 0.3], |_, _| 0.5, 1)];
        let up = ctx.upload_clusters(&clusters, 0, &[]);
        let cmd = gpu.begin();
        let clas = ctx.build_clas(cmd, &up, &clusters, data_flags(), None, None);
        let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0]], None);
        gpu.submit_wait(cmd, case, "off blas");
        blas.addrs.read::<u64>(1)[0]
    };
    let addr = |t: Target| match t {
        Target::Quad => quad,
        Target::Off => off,
        Target::Null => 0,
    };
    let pipe = gpu.pipeline(&spv!("canopy"));
    let (c, p, g) = (131_072u32, 128u32, 1000u32);
    let per = (c - g).div_ceil(p);
    let side = f64::from(c).sqrt().ceil() as u32;
    let part = |i: u32| if i >= c - g { GLOBAL } else { i / per };
    let rec = |i: u32, blas: u64| {
        let m = translate([(i % side) as f32, (i / side) as f32, 0.0]);
        write_rec(i, part(i), i, m, blas, Some(world_aabb(&m, [0.0, 0.0, 0.4, 1.0, 1.0, 0.6])))
    };
    let d = PtlasDesc { instances: c, per_partition: per, partitions: p, global: g };
    let (size, scratch_size) = sizes(gpu, &d);
    let dst = ctx.buf(size, Mem::Device, 256);
    let scratch = ctx.buf(scratch_size, Mem::Device, 256);
    let full = ctx.upload(&(0..c).map(|i| rec(i, quad)).collect::<Vec<_>>());
    let movers: Vec<u32> = (c - g..c).collect();
    let (cells, _) = chosen(c - g - (c - g) % p, p, 10_000, Touch::Spread);
    let (cells_a, cells_b) = (cells[..5000].to_vec(), cells[5000..].to_vec());
    let ops = OpList::new(ctx, &[Op::write(full.addr, c)]);
    let restore = |label: &str| {
        ops.set(&[Op::write(full.addr, c)]);
        let cmd = gpu.begin();
        record(gpu, cmd, &d, dst.addr, scratch.addr, false, &ops);
        gpu.submit_wait(cmd, case, label);
    };
    let cmd = gpu.begin();
    record(gpu, cmd, &d, dst.addr, scratch.addr, true, &ops);
    gpu.submit_wait(cmd, case, "full");
    let out = ctx.buf(u64::from(c) * 4, Mem::Readback, 16);
    #[derive(Clone, Copy)]
    enum Kind {
        Write,
        Update,
    }
    type Test<'a> = (&'a str, Vec<(Kind, &'a [u32], Target)>);
    let tests: Vec<Test> = vec![
        ("write_cells_a+write_cells_b", vec![(Kind::Write, &cells_a, Target::Null), (Kind::Write, &cells_b, Target::Null)]),
        ("write_movers+write_cells", vec![(Kind::Write, &movers, Target::Null), (Kind::Write, &cells, Target::Null)]),
        ("write_cells+write_movers", vec![(Kind::Write, &cells, Target::Null), (Kind::Write, &movers, Target::Null)]),
        ("update_cells+write_movers", vec![(Kind::Update, &cells, Target::Off), (Kind::Write, &movers, Target::Null)]),
        ("write_movers+update_cells", vec![(Kind::Write, &movers, Target::Null), (Kind::Update, &cells, Target::Off)]),
        ("update_cells_a+update_cells_b", vec![(Kind::Update, &cells_a, Target::Off), (Kind::Update, &cells_b, Target::Off)]),
        ("one_write_of_cells_and_movers", vec![(Kind::Write, &[], Target::Null)]),
    ];
    let mut ok_single = true;
    let mut detail = Vec::new();
    for (label, cmds) in &tests {
        let mut state = vec![Target::Quad; c as usize];
        let mut bufs = Vec::new();
        let mut list = Vec::new();
        for &(kind, ids, t) in cmds {
            let ids: Vec<u32> = if ids.is_empty() { cells.iter().chain(&movers).copied().collect() } else { ids.to_vec() };
            for &i in &ids {
                state[i as usize] = t;
            }
            match kind {
                Kind::Write => {
                    let b = ctx.upload(&ids.iter().map(|&i| rec(i, addr(t))).collect::<Vec<_>>());
                    list.push(Op::write(b.addr, ids.len() as u32));
                    bufs.push(b);
                }
                Kind::Update => {
                    let b = ctx.upload(&ids.iter().map(|&i| update_rec(i, addr(t))).collect::<Vec<_>>());
                    list.push(Op::update(b.addr, ids.len() as u32));
                    bufs.push(b);
                }
            }
        }
        ops.set(&list);
        let cmd = gpu.begin();
        record(gpu, cmd, &d, dst.addr, scratch.addr, false, &ops);
        let root = CanopyRoot { out: out.addr, tlas: dst.addr, width: side, height: 0, mode: 2, n: c, cam: [0.0; 3], extent: 0.0, sun: [0.0; 3], tmax: 4.0 };
        canopy(gpu, cmd, pipe, root);
        gpu.submit_wait(cmd, case, label);
        let got = std::slice::from_raw_parts(out.ptr.cast::<u32>(), c as usize);
        let wrong = (0..c as usize).filter(|&i| got[i] != if state[i] == Target::Quad { i as u32 } else { u32::MAX }).count();
        // Unapplied commands, if any: count the ids whose expected change did not show.
        let mut unapplied = Vec::new();
        for (k, &(_, ids, _)) in cmds.iter().enumerate() {
            let ids: Vec<u32> = if ids.is_empty() { cells.iter().chain(&movers).copied().collect() } else { ids.to_vec() };
            let missed = ids.iter().filter(|&&i| got[i as usize] == i).count();
            if missed > 0 {
                unapplied.push(format!("cmd{k}:{missed}/{}", ids.len()));
            }
        }
        if cmds.len() == 1 {
            ok_single &= wrong == 0;
        }
        let line = format!("{label}: wrong_rays={wrong} not_applied=[{}]", unapplied.join(","));
        println!("ROW ops {line}");
        detail.push(line);
        restore("restore");
    }
    (ok_single, detail.join(" "))
}
