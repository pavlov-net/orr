//! Ray tracing pipelines: raygen, miss, hit and callable shaders created through a deferred
//! operation that several threads join, so the driver compiles the shaders in parallel. The
//! raygen, miss and callable regions of the binding table are the pipeline's own; the hit
//! region is the caller's ([`HitTable`]).

use std::mem::DropGuard;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use ash::vk;

use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use super::pipeline::{ShaderStage, StageParts};
use crate::{
    CompileThreads, Error, HitGroup, MemoryKind, RayPipelineDesc, RayPipelineFlags, ShaderRecord,
    Stage,
};

/// A ray tracing pipeline and the raygen, miss and callable regions of its binding table.
/// Dropping it destroys both once the GPU is done with the work submitted before the drop and
/// with the command lists open at it.
pub struct RayPipeline {
    pub(super) raw: vk::Pipeline,
    /// The size of the root the stages hold in the push data; 0 when none declares one.
    pub(super) inline_root_bytes: u32,
    /// Every group's handle, in group order.
    handles: Vec<u8>,
    handle_bytes: usize,
    first_hit: usize,
    hit_records: Vec<u32>,
    hit_stride: u64,
    pub(super) raygens: Vec<vk::StridedDeviceAddressRegionKHR>,
    pub(super) miss: vk::StridedDeviceAddressRegionKHR,
    pub(super) callable: vk::StridedDeviceAddressRegionKHR,
    table: Option<RawBuffer>,
    shared: Arc<Shared>,
}

impl Drop for RayPipeline {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Pipeline(self.raw));
        if let Some(table) = self.table.take() {
            self.shared.defer_after_submitted(Item::Buffer(table));
        }
    }
}

impl RayPipeline {
    /// The handle of hit group `group`, which starts each of its records.
    pub fn hit_handle(&self, group: u32) -> &[u8] {
        let at = group as usize * self.handle_bytes;
        &self.hit_handles()[at..at + self.handle_bytes]
    }

    /// Every hit group's handle, in group order, `Limits::ray_handle_bytes` each: what a kernel
    /// that writes hit records reads.
    pub fn hit_handles(&self) -> &[u8] {
        let at = self.first_hit * self.handle_bytes;
        &self.handles[at..at + self.hit_records.len() * self.handle_bytes]
    }

    /// The record data bytes hit group `group`'s shaders read after the handle.
    pub fn hit_record_bytes(&self, group: u32) -> u32 {
        self.hit_records[group as usize]
    }

    /// The least stride of a [`crate::HitTable`] this pipeline traces: room for the handle and
    /// every hit group's record data, rounded to `Limits::ray_handle_align`.
    pub fn hit_stride(&self) -> u64 {
        self.hit_stride
    }

    /// Writes a record of hit group `group` with record data `data` at the start of `record`,
    /// a record of a mapped hit table.
    pub fn write_hit_record(&self, record: &mut [u8], group: u32, data: &[u8]) {
        assert!(
            data.len() >= self.hit_record_bytes(group) as usize,
            "hit group {group} reads {} bytes of record data; {} given",
            self.hit_record_bytes(group),
            data.len()
        );
        let (handle, rest) = record.split_at_mut(self.handle_bytes);
        handle.copy_from_slice(self.hit_handle(group));
        rest[..data.len()].copy_from_slice(data);
    }
}

/// A raygen-only ray tracing pipeline, run by [`crate::CommandList::dispatch_rays`], for
/// kernels that trace with ray queries and reorder at coherence points.
pub struct RayDispatchPipeline(pub(super) Box<RayPipeline>);

/// One deferred creation and the threads joined to it, shared with the tasks of a pool.
struct Joined {
    shared: Arc<Shared>,
    op: vk::DeferredOperationKHR,
    done: Mutex<bool>,
    completed: Condvar,
}

impl Joined {
    /// Joins the operation until it has no more work for this thread. Returns whether it
    /// completed here.
    fn join(&self) -> bool {
        let f = &self.shared.deferred_fn;
        loop {
            // SAFETY: the operation is live while `self` is.
            match unsafe { f.deferred_operation_join(self.op) } {
                Ok(()) => {
                    *self.done.lock().unwrap() = true;
                    self.completed.notify_all();
                    return true;
                }
                Err(vk::Result::THREAD_IDLE_KHR) => std::thread::yield_now(),
                Err(_) => return false,
            }
        }
    }

    /// The operation's result once it is complete. Joins until then: a thread that runs out of
    /// work waits for the others, and joins again in case none is left.
    fn finish(&self) -> vk::Result {
        let f = &self.shared.deferred_fn;
        while !self.join() {
            let done = self.done.lock().unwrap();
            let (done, _) = self
                .completed
                .wait_timeout_while(done, Duration::from_millis(1), |d| !*d)
                .unwrap();
            if *done {
                break;
            }
        }
        // SAFETY: the operation is live and complete.
        match unsafe { f.get_deferred_operation_result(self.op) } {
            Ok(()) => vk::Result::SUCCESS,
            Err(e) => e,
        }
    }
}

impl Drop for Joined {
    fn drop(&mut self) {
        // SAFETY: the last holder: no thread is joined to the operation.
        unsafe {
            self.shared
                .deferred_fn
                .destroy_deferred_operation(self.op, None);
        }
    }
}

/// Runs `create` with a deferred operation that `threads` join, or with none when one thread
/// compiles, and returns its result once the operation is complete.
fn create_deferred(
    shared: &Arc<Shared>,
    threads: CompileThreads<'_>,
    create: impl FnOnce(vk::DeferredOperationKHR) -> vk::Result,
) -> vk::Result {
    let most = match threads {
        CompileThreads::Spawned(n) => n,
        CompileThreads::Pool { max, .. } => max,
    };
    if most <= 1 {
        return create(vk::DeferredOperationKHR::null());
    }
    let f = &shared.deferred_fn;
    // SAFETY: the device is live.
    let op = match unsafe { f.create_deferred_operation(None) } {
        Ok(op) => op,
        Err(e) => return e,
    };
    let joined = Arc::new(Joined {
        shared: Arc::clone(shared),
        op,
        done: Mutex::new(false),
        completed: Condvar::new(),
    });
    match create(op) {
        vk::Result::OPERATION_DEFERRED_KHR => {}
        vk::Result::OPERATION_NOT_DEFERRED_KHR => return vk::Result::SUCCESS,
        other => return other,
    }
    // SAFETY: the operation is pending.
    let concurrency = unsafe { f.get_deferred_operation_max_concurrency(op) };
    let helpers = concurrency.min(most).saturating_sub(1);
    log::debug!("deferred creation: concurrency {concurrency}, {helpers} helper threads");
    match threads {
        CompileThreads::Spawned(_) => std::thread::scope(|s| {
            for _ in 0..helpers {
                let joined = &joined;
                s.spawn(move || {
                    joined.join();
                });
            }
            joined.finish()
        }),
        CompileThreads::Pool { spawn, .. } => {
            for _ in 0..helpers {
                let joined = Arc::clone(&joined);
                spawn(Box::new(move || {
                    joined.join();
                }));
            }
            joined.finish()
        }
    }
}

/// The creation flags of `flags`.
fn creation_flags(flags: RayPipelineFlags) -> vk::PipelineCreateFlags2 {
    use vk::PipelineCreateFlags2 as P;
    let mut out = P::empty();
    for (flag, vk_flag) in [
        (
            RayPipelineFlags::OPACITY_MICROMAPS,
            P::RAY_TRACING_OPACITY_MICROMAP_EXT,
        ),
        (
            RayPipelineFlags::SWEPT_SPHERES,
            P::RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES_NV,
        ),
        (
            RayPipelineFlags::SKIP_TRIANGLES,
            P::RAY_TRACING_SKIP_BUILT_IN_PRIMITIVES_KHR,
        ),
        (RayPipelineFlags::SKIP_AABBS, P::RAY_TRACING_SKIP_AABBS_KHR),
        (
            RayPipelineFlags::NO_NULL_ANY_HIT,
            P::RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_KHR,
        ),
        (
            RayPipelineFlags::NO_NULL_CLOSEST_HIT,
            P::RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_KHR,
        ),
        (
            RayPipelineFlags::NO_NULL_MISS,
            P::RAY_TRACING_NO_NULL_MISS_SHADERS_KHR,
        ),
        (
            RayPipelineFlags::NO_NULL_INTERSECTION,
            P::RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_KHR,
        ),
    ] {
        if flags.contains(flag) {
            out |= vk_flag;
        }
    }
    out
}

fn vk_stage(stage: Stage) -> vk::ShaderStageFlags {
    use vk::ShaderStageFlags as S;
    match stage {
        Stage::RayGen => S::RAYGEN_KHR,
        Stage::ClosestHit => S::CLOSEST_HIT_KHR,
        Stage::AnyHit => S::ANY_HIT_KHR,
        Stage::Intersection => S::INTERSECTION_KHR,
        Stage::Miss => S::MISS_KHR,
        Stage::Callable => S::CALLABLE_KHR,
        other => unreachable!("{other:?} is not a ray tracing stage"),
    }
}

/// One region of the pipeline's own table: its records' groups and data, and its stride.
struct Region<'a> {
    first_group: usize,
    records: &'a [ShaderRecord<'a>],
    stride: u64,
}

impl Device {
    /// Creates a ray tracing pipeline from `desc`, its shaders compiled on `desc.threads`.
    /// Threads asking for one pipeline at once compile it once: the others wait for it. Fails
    /// unless `Caps::ray_dispatch`.
    pub fn create_ray_pipeline(
        &self,
        desc: &RayPipelineDesc<'_>,
        name: &str,
    ) -> Result<RayPipeline, Error> {
        let _zone = crate::zone!("create pipeline", name);
        let shared = &self.shared;
        let caps = &shared.caps;
        let Some(rt) = &shared.rt_fn else {
            return Err(Error::Unsupported {
                device: caps.device_name.clone(),
                missing: "ray tracing pipelines (VK_KHR_ray_tracing_pipeline with \
                          VK_EXT_ray_tracing_invocation_reorder)"
                    .into(),
            });
        };
        let invalid = |what: String| Error::Invalid(format!("ray pipeline {name}: {what}"));
        let limits = &caps.limits;
        if desc.raygens.is_empty() {
            return Err(invalid("no raygen".into()));
        }
        if desc.max_recursion > limits.max_ray_recursion {
            return Err(invalid(format!(
                "recursion {} is over the device's {}",
                desc.max_recursion, limits.max_ray_recursion
            )));
        }
        if desc.flags.contains(RayPipelineFlags::OPACITY_MICROMAPS) && !caps.omm {
            return Err(invalid("opacity micromaps without Caps::omm".into()));
        }
        if desc.flags.contains(RayPipelineFlags::SWEPT_SPHERES) && !caps.lss {
            return Err(invalid("swept spheres without Caps::lss".into()));
        }
        let parts = desc
            .stages
            .iter()
            .map(|s| {
                if !s.entry.stage.ray_tracing() {
                    return Err(invalid(format!(
                        "{:?} is a {:?} entry",
                        s.entry.name, s.entry.stage
                    )));
                }
                StageParts::new(shared, s, s.entry.stage, name)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut root = 0;
        for s in desc.stages {
            match (root, s.inline_root_bytes) {
                (_, 0) => {}
                (0, bytes) => root = bytes,
                (a, b) if a != b => {
                    return Err(invalid(format!(
                        "stages hold roots of {a} and {b} bytes; every stage declares one root"
                    )));
                }
                _ => {}
            }
        }
        // A stage index of the expected kind, or what is wrong with it.
        let stage = |index: u32, want: Stage| {
            let s = desc.stages.get(index as usize).ok_or_else(|| {
                invalid(format!(
                    "stage {index} is past the {} stages",
                    desc.stages.len()
                ))
            })?;
            if s.entry.stage != want {
                return Err(invalid(format!(
                    "stage {index} ({:?}) is a {:?} entry, not {want:?}",
                    s.entry.name, s.entry.stage
                )));
            }
            Ok(s)
        };
        let record = |r: &ShaderRecord<'_>, want: Stage| {
            let s = stage(r.stage, want)?;
            if r.data.len() < s.entry.record_bytes as usize {
                return Err(invalid(format!(
                    "{:?} reads {} bytes of record data; {} given",
                    s.entry.name,
                    s.entry.record_bytes,
                    r.data.len()
                )));
            }
            Ok(())
        };
        let general = |index: u32| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(index)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
        };
        let mut groups = Vec::new();
        for r in desc.raygens {
            record(r, Stage::RayGen)?;
            groups.push(general(r.stage));
        }
        for r in desc.misses {
            record(r, Stage::Miss)?;
            groups.push(general(r.stage));
        }
        let mut hit_records = Vec::with_capacity(desc.hit_groups.len());
        for g in desc.hit_groups {
            let HitGroup {
                closest_hit,
                any_hit,
                intersection,
            } = *g;
            let mut bytes = 0;
            let mut member = |index: Option<u32>, want: Stage| {
                index.map_or(Ok(vk::SHADER_UNUSED_KHR), |i| {
                    bytes = bytes.max(stage(i, want)?.entry.record_bytes);
                    Ok::<_, Error>(i)
                })
            };
            let closest = member(closest_hit, Stage::ClosestHit)?;
            let any = member(any_hit, Stage::AnyHit)?;
            let isect = member(intersection, Stage::Intersection)?;
            hit_records.push(bytes);
            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(if intersection.is_some() {
                        vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
                    } else {
                        vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
                    })
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(closest)
                    .any_hit_shader(any)
                    .intersection_shader(isect),
            );
        }
        for r in desc.callables {
            record(r, Stage::Callable)?;
            groups.push(general(r.stage));
        }
        let handle_bytes = limits.ray_handle_bytes as usize;
        let handle_align = u64::from(limits.ray_handle_align);
        let base = u64::from(limits.ray_table_align);
        let data = |records: &[ShaderRecord<'_>]| {
            records
                .iter()
                .map(|r| r.data.len() as u64)
                .max()
                .unwrap_or(0)
        };
        let stride = |records: &[ShaderRecord<'_>], align: u64| {
            (handle_bytes as u64 + data(records)).next_multiple_of(align)
        };
        let regions = [
            Region {
                first_group: 0,
                records: desc.raygens,
                stride: stride(desc.raygens, base),
            },
            Region {
                first_group: desc.raygens.len(),
                records: desc.misses,
                stride: stride(desc.misses, handle_align),
            },
            Region {
                first_group: groups.len() - desc.callables.len(),
                records: desc.callables,
                stride: stride(desc.callables, handle_align),
            },
        ];
        let hit_stride = (handle_bytes as u64
            + u64::from(hit_records.iter().copied().max().unwrap_or(0)))
        .next_multiple_of(handle_align);
        let max_stride = u64::from(limits.max_ray_record_stride);
        if let Some(stride) = regions
            .iter()
            .map(|r| r.stride)
            .chain([hit_stride])
            .find(|&s| s > max_stride)
        {
            return Err(invalid(format!(
                "records of {stride} bytes are over the device's {max_stride}"
            )));
        }
        let mut state = Vec::new();
        for g in &groups {
            for v in [
                g.ty.as_raw() as u32,
                g.general_shader,
                g.closest_hit_shader,
                g.any_hit_shader,
                g.intersection_shader,
            ] {
                state.extend_from_slice(&v.to_le_bytes());
            }
        }
        state.extend_from_slice(&desc.max_recursion.to_le_bytes());
        state.extend_from_slice(&desc.flags.bits().to_le_bytes());
        let extra = self.creation_flags | creation_flags(desc.flags);
        let clusters = desc.flags.contains(RayPipelineFlags::CLUSTERS);
        let started = std::time::Instant::now();
        let raw = shared.create_pipeline(
            &parts.iter().collect::<Vec<_>>(),
            &state,
            extra,
            true,
            name,
            "vkCreateRayTracingPipelinesKHR",
            |cache, next| {
                let mut chains: Vec<_> = parts.iter().map(StageParts::chain).collect();
                let stages: Vec<_> = chains
                    .iter_mut()
                    .zip(desc.stages)
                    .map(|(c, s)| c.info(vk_stage(s.entry.stage)))
                    .collect();
                let mut cluster =
                    vk::RayTracingPipelineClusterAccelerationStructureCreateInfoNV::default()
                        .allow_cluster_acceleration_structure(true);
                let mut info = vk::RayTracingPipelineCreateInfoKHR::default()
                    .stages(&stages)
                    .groups(&groups)
                    .max_pipeline_ray_recursion_depth(desc.max_recursion);
                if clusters {
                    cluster.p_next = next.cast_mut();
                    info.p_next = std::ptr::from_ref(&cluster).cast();
                } else {
                    info.p_next = next;
                }
                let mut pipeline = vk::Pipeline::null();
                let result = create_deferred(shared, desc.threads, |op| {
                    // SAFETY: valid create info that outlives the operation, which
                    // `create_deferred` waits for; the SPIR-V was validated by the shader
                    // build.
                    unsafe {
                        (rt.fp().create_ray_tracing_pipelines_khr)(
                            rt.device(),
                            op,
                            cache,
                            1,
                            &raw const info,
                            std::ptr::null(),
                            &raw mut pipeline,
                        )
                    }
                });
                match result {
                    vk::Result::SUCCESS => Ok(pipeline),
                    e => Err(e),
                }
            },
        )?;
        log::debug!(
            "ray pipeline {name}: {} stages, {} groups, {:.1} ms with {:?}",
            desc.stages.len(),
            groups.len(),
            started.elapsed().as_secs_f64() * 1e3,
            desc.threads
        );
        // SAFETY: nothing uses the pipeline until it is returned.
        let raw = DropGuard::new(raw, |p| unsafe { shared.device.destroy_pipeline(p, None) });
        // SAFETY: the pipeline has `groups.len()` groups.
        let handles = unsafe {
            rt.get_ray_tracing_shader_group_handles(
                *raw,
                0,
                groups.len() as u32,
                groups.len() * handle_bytes,
            )
        }
        .map_err(Error::vk("vkGetRayTracingShaderGroupHandlesKHR"))?;

        let mut starts = [0u64; 3];
        let mut size = 0;
        for (r, start) in regions.iter().zip(&mut starts) {
            *start = size;
            size = (size + r.stride * r.records.len() as u64).next_multiple_of(base);
        }
        let (table, ptr) = shared.create_buffer(
            MemoryKind::Upload,
            size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            base,
            name,
        )?;
        let table = DropGuard::new(table, |t| shared.destroy(Item::Buffer(t)));
        let ptr = ptr.ok_or_else(|| {
            Error::Invalid("ray pipeline table: upload memory is not mapped".into())
        })?;
        for (r, start) in regions.iter().zip(starts) {
            for (i, rec) in r.records.iter().enumerate() {
                let group = r.first_group + i;
                let at = (start + r.stride * i as u64) as usize;
                let handle = &handles[group * handle_bytes..][..handle_bytes];
                // SAFETY: the record lies inside the mapped buffer, which nothing else
                // reads or writes yet.
                unsafe {
                    let dst = ptr.as_ptr().add(at);
                    std::ptr::copy_nonoverlapping(handle.as_ptr(), dst, handle_bytes);
                    std::ptr::copy_nonoverlapping(
                        rec.data.as_ptr(),
                        dst.add(handle_bytes),
                        rec.data.len(),
                    );
                }
            }
        }
        let region = |r: &Region<'_>, start: u64| {
            if r.records.is_empty() {
                vk::StridedDeviceAddressRegionKHR::default()
            } else {
                vk::StridedDeviceAddressRegionKHR {
                    device_address: table.address + start,
                    stride: r.stride,
                    size: r.stride * r.records.len() as u64,
                }
            }
        };
        let raygen_stride = regions[0].stride;
        Ok(RayPipeline {
            raw: DropGuard::dismiss(raw),
            inline_root_bytes: root,
            handle_bytes,
            first_hit: desc.raygens.len() + desc.misses.len(),
            hit_records,
            hit_stride,
            raygens: (0..desc.raygens.len() as u64)
                .map(|i| vk::StridedDeviceAddressRegionKHR {
                    device_address: table.address + starts[0] + i * raygen_stride,
                    stride: raygen_stride,
                    size: raygen_stride,
                })
                .collect(),
            miss: region(&regions[1], starts[1]),
            callable: region(&regions[2], starts[2]),
            handles,
            table: Some(DropGuard::dismiss(table)),
            shared: Arc::clone(shared),
        })
    }

    /// Creates a ray-dispatch pipeline from a raygen entry point: one raygen and no miss, hit
    /// or callable records, recursion depth 0, compiled on the calling thread. The entry traces
    /// with ray queries; it cannot call `TraceRay`. Threads asking for one pipeline at once
    /// compile it once: the others wait for it. Fails unless `Caps::ray_dispatch`.
    pub fn create_ray_dispatch_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<RayDispatchPipeline, Error> {
        let stages = [*s];
        let pipeline = self.create_ray_pipeline(
            &RayPipelineDesc {
                stages: &stages,
                raygens: &[ShaderRecord::new(0)],
                misses: &[],
                hit_groups: &[],
                callables: &[],
                max_recursion: 0,
                flags: RayPipelineFlags::empty(),
                threads: CompileThreads::Spawned(1),
            },
            name,
        )?;
        Ok(RayDispatchPipeline(Box::new(pipeline)))
    }
}
