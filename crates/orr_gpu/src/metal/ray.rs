//! Ray tracing pipelines over `orr_shaders.pipeline`'s compute form. Each raygen is a launch
//! kernel of 64-thread groups; the closest-hit, miss and callable stages are `[[visible]]`
//! binary functions the kernel calls through a visible function table by stage index; any-hit
//! and intersection stages are intersection functions whose resource IDs start the hit records,
//! which traces hand Metal's intersector as an intersection function buffer.
//!
//! Every function is compiled once, apart from any pipeline and in parallel, as a binary
//! function whose handle is the same in every pipeline that links it, so one hit table serves
//! every raygen's kernel. Static linking would let `[[function_groups]]` narrow the table calls,
//! but its handles are each pipeline's own and it compiles every stage into every raygen's
//! kernel.
//!
//! The record layouts, the launch block and the argument table are the contract of
//! `orr_shaders.pipeline`'s compute form and `abi.slang`'s `PipelineLaunch`.

use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4BinaryFunctionDescriptor, MTL4BinaryFunctionOptions, MTL4CommandEncoder,
    MTL4ComputeCommandEncoder, MTL4FunctionDescriptor, MTL4LibraryFunctionDescriptor,
    MTL4PipelineStageDynamicLinkingDescriptor, MTL4VisibilityOptions, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLFunctionHandle, MTLResidencySet, MTLResource,
    MTLResourceOptions, MTLStages, MTLVisibleFunctionTable, MTLVisibleFunctionTableDescriptor,
};
use orr_shaders::abi::METAL_INTERSECTION_TAGS;

use super::cache::BinaryFunction;
use super::command::CommandList;
use super::device::{Device, Shared, metal_error};
use super::heap::id_bits;
use super::memory::{Allocation, malloc};
use super::pipeline::ShaderStage;
use super::{Obj, lock};
use crate::{
    DispatchArgs, Error, GpuPtr, HitGroup, HitTable, MemoryKind, RayPipelineDesc, RayPipelineFlags,
    ShaderRecord, Stage,
};

/// Bytes of a record's handle, which its data follows.
pub(super) const HANDLE_BYTES: u32 = 32;
/// The alignment of a record stride and of a region's start.
pub(super) const HANDLE_ALIGN: u32 = 32;
pub(super) const TABLE_ALIGN: u32 = 64;
/// The largest stride of an intersection function buffer (MSL 6.19.8).
pub(super) const MAX_RECORD_STRIDE: u32 = 4096;
/// Threads of a launch kernel's group, as the build's launch kernels declare.
const GROUP: u32 = 64;

/// The function a triangle group's records name when it has no any-hit, which accepts each
/// candidate, and the kernel that turns an indirect launch's extent into threadgroups.
fn source() -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;
using namespace metal::raytracing;
[[intersection(triangle, {METAL_INTERSECTION_TAGS})]]
bool orr_accept_triangle() {{ return true; }}
struct GroupArgs {{ device const uint* extent; device uint* groups; }};
kernel void launch_groups(constant GroupArgs& a [[buffer(0)]]) {{
    a.groups[0] = (a.extent[0] + {GROUP} - 1) / {GROUP};
    a.groups[1] = a.extent[1];
    a.groups[2] = a.extent[2];
}}
"#
    )
}

/// The arguments of `launch_groups`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupArgs {
    extent: u64,
    groups: u64,
}

/// `abi.slang`'s `PipelineLaunch`: what a launch kernel and its stages read at argument-table
/// buffer 3.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PipelineLaunch {
    table: u64,
    raygen: u64,
    hits: u64,
    misses: u64,
    callables: u64,
    extent_address: u64,
    checks: u64,
    reserved0: u64,
    hit_stride: u32,
    hit_count: u32,
    miss_stride: u32,
    miss_count: u32,
    callable_stride: u32,
    callable_count: u32,
    call_depth: u32,
    reserved1: u32,
    extent: [u32; 4],
}

/// What every ray tracing pipeline of a device shares: the default intersection functions,
/// the indirect launch's kernel and the word `ASSERTS` builds' checks set.
pub(super) struct RayShared {
    library: Obj<ProtocolObject<dyn objc2_metal::MTLLibrary>>,
    groups: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    checks: Obj<super::Buffer>,
    reported: std::sync::atomic::AtomicU32,
}

impl RayShared {
    fn get(shared: &Shared) -> Result<&RayShared, Error> {
        if let Some(r) = shared.ray.get() {
            return Ok(r);
        }
        let library = shared
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&source()), None)
            .map_err(|e| metal_error("newLibraryWithSource", &e))?;
        let f = MTL4LibraryFunctionDescriptor::new();
        f.setName(Some(&NSString::from_str("launch_groups")));
        f.setLibrary(Some(&library));
        let groups = shared.compute_state(&f, [1, 1, 1], "launch groups")?;
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        let checks = shared
            .device
            .newBufferWithLength_options(4, options)
            .ok_or_else(|| Error::Metal {
                call: "newBufferWithLength",
                message: "ray pipeline checks".to_owned(),
            })?;
        checks.setLabel(Some(&NSString::from_str("orr ray pipeline checks")));
        // SAFETY: shared storage is mapped; nothing else uses it yet.
        unsafe { checks.contents().cast::<u32>().as_ptr().write(0) };
        let made = RayShared {
            library: Obj(library),
            groups,
            checks: Obj(checks),
            reported: Default::default(),
        };
        // Of threads that race here, one's joins the residency set and serves; the others' are
        // dropped unused.
        Ok(shared.ray.get_or_init(|| {
            let mut r = lock(&shared.residency);
            r.set.addAllocation(ProtocolObject::from_ref(&*made.checks));
            r.dirty = true;
            made
        }))
    }

    /// The checks `ASSERTS` builds' launches found failing (`orr_shaders.pipeline`'s
    /// `PIPELINE_CHECK_*` bits) among completed work, logged once each.
    pub(super) fn failures(&self) -> u32 {
        // SAFETY: the word stays mapped while the buffer lives; the GPU only ORs into it.
        let bits = unsafe {
            self.checks
                .contents()
                .cast::<u32>()
                .as_ptr()
                .read_volatile()
        };
        let before = self
            .reported
            .fetch_or(bits, std::sync::atomic::Ordering::Relaxed);
        let new = bits & !before;
        if new & 1 != 0 {
            log::error!(
                "a trace's intersection function buffer index differs from the hit record \
                 its instance's contribution and geometry name: Metal does not add instance \
                 offsets across instancing levels as the backend lays them out"
            );
        }
        bits.count_ones()
    }
}

/// A raygen's launch kernel and the table of the stages it calls.
struct Raygen {
    state: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    table: Obj<ProtocolObject<dyn MTLVisibleFunctionTable>>,
    /// The address of its record.
    record: u64,
}

/// A region of the pipeline's own records.
#[derive(Clone, Copy, Default)]
struct Region {
    address: u64,
    stride: u32,
    count: u32,
}

/// A ray tracing pipeline: a launch kernel per raygen, the binary functions they link and the
/// raygen, miss and callable records. Command lists keep what they use until their submit
/// completes, so dropping it is safe at any time.
pub struct RayPipeline {
    raygens: Vec<Raygen>,
    /// The size of the root the stages hold inline; 0 when none declares one.
    inline_root_bytes: u32,
    /// Every hit group's handle, in group order.
    handles: Vec<u8>,
    hit_records: Vec<u32>,
    hit_stride: u64,
    misses: Region,
    callables: Region,
    call_depth: u32,
    /// The kernel that turns an indirect launch's extent into threadgroups.
    launch_groups: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The address of the word `ASSERTS` builds' checks set.
    checks: u64,
    records: Arc<Allocation<u8>>,
    /// The binary functions, which the kernels link and the hit records name.
    functions: Arc<Vec<Obj<BinaryFunction>>>,
    shared: Arc<Shared>,
}

impl Drop for RayPipeline {
    fn drop(&mut self) {
        let tables: Vec<_> = self.raygens.iter().map(|r| r.table.clone()).collect();
        let shared = Arc::clone(&self.shared);
        self.shared.defer(Box::new(move || {
            let mut r = lock(&shared.residency);
            for t in &tables {
                r.set.removeAllocation(ProtocolObject::from_ref(&**t));
            }
            r.dirty = true;
        }));
    }
}

impl RayPipeline {
    /// `None`: Metal compiles in a service of its own, whose CPU time no thread here takes.
    pub fn compile_cost(&self) -> Option<&crate::CompileCost> {
        None
    }

    /// The handle of hit group `group`, which starts each of its records.
    pub fn hit_handle(&self, group: u32) -> &[u8] {
        let at = group as usize * HANDLE_BYTES as usize;
        &self.handles[at..at + HANDLE_BYTES as usize]
    }

    /// Every hit group's handle, in group order, `Limits::ray_handle_bytes` each.
    pub fn hit_handles(&self) -> &[u8] {
        &self.handles
    }

    /// The record data bytes hit group `group`'s shaders read after the handle.
    pub fn hit_record_bytes(&self, group: u32) -> u32 {
        self.hit_records[group as usize]
    }

    /// The least stride of a [`crate::HitTable`] this pipeline traces.
    pub fn hit_stride(&self) -> u64 {
        self.hit_stride
    }

    /// Writes a record of hit group `group` with record data `data` at the start of `record`.
    pub fn write_hit_record(&self, record: &mut [u8], group: u32, data: &[u8]) {
        assert!(
            data.len() >= self.hit_record_bytes(group) as usize,
            "hit group {group} reads {} bytes of record data; {} given",
            self.hit_record_bytes(group),
            data.len()
        );
        let (handle, rest) = record.split_at_mut(HANDLE_BYTES as usize);
        handle.copy_from_slice(self.hit_handle(group));
        rest[..data.len()].copy_from_slice(data);
    }
}

/// What a binary function is compiled from: a stage, and the function of its library when it
/// is not the entry's own (a procedural group's intersection with its any-hit), or one of the
/// default functions.
enum Source<'a> {
    Stage(&'a ShaderStage<'a>, Option<String>),
    Default(&'static str),
}

impl Device {
    /// Creates a ray tracing pipeline from `desc`: its functions compiled on `desc.threads`,
    /// each once per device and through the pipeline cache.
    pub fn create_ray_pipeline(
        &self,
        desc: &RayPipelineDesc<'_>,
        name: &str,
    ) -> Result<RayPipeline, Error> {
        let _zone = crate::zone!("create pipeline", name);
        let shared = &self.shared;
        let caps = &shared.caps;
        if !caps.ray_dispatch {
            return Err(Error::Unsupported {
                device: caps.device_name.clone(),
                missing: "ray tracing pipelines (intersection function buffers, Apple9)".into(),
            });
        }
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
        if desc.call_depth < desc.max_recursion {
            return Err(invalid(format!(
                "call depth {} is below the recursion {}: each nested trace runs a stage",
                desc.call_depth, desc.max_recursion
            )));
        }
        for (flag, what) in [
            (RayPipelineFlags::OPACITY_MICROMAPS, "opacity micromaps"),
            (RayPipelineFlags::SWEPT_SPHERES, "swept spheres"),
        ] {
            if desc.flags.contains(flag) {
                return Err(invalid(format!("{what}, which Metal has none of")));
            }
        }
        if let Some(s) = desc.stages.iter().find(|s| !s.entry.stage.ray_tracing()) {
            return Err(invalid(format!(
                "{:?} is a {:?} entry",
                s.entry.name, s.entry.stage
            )));
        }
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
        for r in desc.raygens {
            record(r, Stage::RayGen)?;
        }
        for r in desc.misses {
            record(r, Stage::Miss)?;
        }
        for r in desc.callables.iter().flatten() {
            record(r, Stage::Callable)?;
        }

        // The binary functions: each closest-hit, miss and callable stage's visible function,
        // and each hit group's intersection function, shared between groups alike.
        let mut sources: Vec<Source<'_>> = Vec::new();
        let mut visible = Vec::new();
        for (i, s) in desc.stages.iter().enumerate() {
            if matches!(
                s.entry.stage,
                Stage::ClosestHit | Stage::Miss | Stage::Callable
            ) {
                visible.push((i, sources.len()));
                sources.push(Source::Stage(s, None));
            }
        }
        // Each hit group's intersection function, by its index among the binary functions: a
        // stage's own, a procedural pair's or the default, each compiled once.
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        enum Isect {
            Own(u32),
            Pair(u32, u32),
            Accept,
        }
        let mut made = std::collections::HashMap::new();
        let mut groups = Vec::with_capacity(desc.hit_groups.len());
        for g in desc.hit_groups {
            if let Some(i) = g.closest_hit {
                stage(i, Stage::ClosestHit)?;
            }
            if let Some(i) = g.any_hit {
                stage(i, Stage::AnyHit)?;
            }
            if let Some(i) = g.intersection {
                stage(i, Stage::Intersection)?;
            }
            let key = match (g.intersection, g.any_hit) {
                (Some(isect), Some(any)) => Isect::Pair(isect, any),
                (Some(i), None) | (None, Some(i)) => Isect::Own(i),
                (None, None) => Isect::Accept,
            };
            let f = match made.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    sources.push(match key {
                        Isect::Pair(isect, any) => {
                            let (is, a) =
                                (&desc.stages[isect as usize], &desc.stages[any as usize]);
                            // The pair's function is the intersection's library's, which runs
                            // the any-hit under the intersection's constants.
                            if !std::ptr::eq(is.module, a.module)
                                || is.spec.bytes != a.spec.bytes
                                || is.spec.entries != a.spec.entries
                            {
                                return Err(invalid(format!(
                                    "intersection {:?} and any-hit {:?} of one group come from \
                                     different modules or constants: Metal runs them as one \
                                     function of the intersection's module",
                                    is.entry.name, a.entry.name
                                )));
                            }
                            let any_name = a.entry.name.to_str().map_err(|_| {
                                invalid(format!("any-hit {:?} is not named in UTF-8", a.entry.name))
                            })?;
                            Source::Stage(is, Some(format!("{}__{any_name}", is.metal_name()?)))
                        }
                        Isect::Own(i) => Source::Stage(&desc.stages[i as usize], None),
                        Isect::Accept => Source::Default("orr_accept_triangle"),
                    });
                    *e.insert(sources.len() - 1)
                }
            };
            groups.push(f);
        }
        let started = std::time::Instant::now();
        let ray = RayShared::get(shared)?;
        let compiled = desc.threads.map(sources.len(), |i| {
            self.binary_function(ray, &sources[i], name)
        });
        let functions = compiled.into_iter().collect::<Result<Vec<_>, _>>()?;
        let mut linked_key = blake3::Hasher::new();
        for (_, key) in &functions {
            linked_key.update(key.as_bytes());
        }
        linked_key.update(&desc.call_depth.to_le_bytes());
        let states = desc.threads.map(desc.raygens.len(), |i| {
            let s = &desc.stages[desc.raygens[i].stage as usize];
            let function = s.function(shared)?;
            let d = super::pipeline::compute_descriptor(&function, [GROUP, 1, 1], name);
            d.setSupportBinaryLinking(true);
            let linking = MTL4PipelineStageDynamicLinkingDescriptor::new();
            let refs: Vec<&BinaryFunction> = functions.iter().map(|(f, _)| &**f).collect();
            linking.setBinaryLinkedFunctions(Some(&objc2_foundation::NSArray::from_slice(&refs)));
            linking.setMaxCallStackDepth(desc.call_depth as usize);
            let mut key = blake3::Hasher::new();
            key.update(b"launch");
            s.key(&mut key);
            key.update(linked_key.finalize().as_bytes());
            shared.cached_linked_compute(key.finalize(), self.cached_only, &d, &linking, name)
        });
        let states = states.into_iter().collect::<Result<Vec<_>, _>>()?;
        log::debug!(
            "ray pipeline {name}: {} functions, {} launch kernels, {:.1} ms with {:?}",
            functions.len(),
            states.len(),
            started.elapsed().as_secs_f64() * 1e3,
            desc.threads
        );

        // Intersection function handles, which every kernel that links the functions shares.
        let handle = |state: &ProtocolObject<dyn MTLComputePipelineState>, f: usize| {
            state
                .functionHandleWithBinaryFunction(&functions[f].0)
                .ok_or_else(|| invalid(format!("no handle of binary function {f}")))
        };
        let mut ids = vec![0u64; functions.len()];
        for &f in &groups {
            if ids[f] != 0 {
                continue;
            }
            let id = id_bits(handle(&states[0], f)?.gpuResourceID());
            for s in &states[1..] {
                if id_bits(handle(s, f)?.gpuResourceID()) != id {
                    return Err(invalid(format!(
                        "binary function {f} has another resource ID in another launch kernel, \
                         though compiled pipeline-independent"
                    )));
                }
            }
            ids[f] = id;
        }
        let mut handles = vec![0u8; desc.hit_groups.len() * HANDLE_BYTES as usize];
        let mut hit_records = Vec::with_capacity(desc.hit_groups.len());
        for ((g, f), h) in desc
            .hit_groups
            .iter()
            .zip(&groups)
            .zip(handles.as_chunks_mut::<{ HANDLE_BYTES as usize }>().0)
        {
            let HitGroup {
                closest_hit,
                any_hit,
                intersection,
            } = *g;
            h[0..8].copy_from_slice(&ids[*f].to_le_bytes());
            for (at, s) in [(8, closest_hit), (12, any_hit), (16, intersection)] {
                h[at..at + 4].copy_from_slice(&s.map_or(0, |s| s + 1).to_le_bytes());
            }
            hit_records.push(
                [closest_hit, any_hit, intersection]
                    .into_iter()
                    .flatten()
                    .map(|i| desc.stages[i as usize].entry.record_bytes)
                    .max()
                    .unwrap_or(0),
            );
        }
        let stride = |data: usize| {
            (u64::from(HANDLE_BYTES) + data as u64).next_multiple_of(u64::from(HANDLE_ALIGN))
        };
        let hit_stride = stride(hit_records.iter().copied().max().unwrap_or(0) as usize);
        let most = |records: &mut dyn Iterator<Item = &ShaderRecord<'_>>| {
            stride(records.map(|r| r.data.len()).max().unwrap_or(0))
        };
        let raygen_stride = most(&mut desc.raygens.iter());
        let miss_stride = most(&mut desc.misses.iter());
        let callable_stride = most(&mut desc.callables.iter().flatten());
        let max_stride = u64::from(MAX_RECORD_STRIDE);
        if let Some(s) = [raygen_stride, miss_stride, callable_stride, hit_stride]
            .into_iter()
            .find(|&s| s > max_stride)
        {
            return Err(invalid(format!(
                "records of {s} bytes are over the device's {max_stride}"
            )));
        }
        let align = u64::from(TABLE_ALIGN);
        let starts = [
            0,
            (raygen_stride * desc.raygens.len() as u64).next_multiple_of(align),
        ];
        let callable_start =
            (starts[1] + miss_stride * desc.misses.len() as u64).next_multiple_of(align);
        let size = callable_start + callable_stride * desc.callables.len() as u64;
        let mut records = malloc::<u8>(shared, MemoryKind::Upload, size.max(16), name, false)?;
        let base = records.gpu().address().0;
        {
            let bytes = records.as_mut_slice().expect("upload is mapped");
            bytes.fill(0);
            let mut put = |at: u64, r: &ShaderRecord<'_>| {
                let at = at as usize;
                bytes[at..at + 4].copy_from_slice(&(r.stage + 1).to_le_bytes());
                let data = at + HANDLE_BYTES as usize;
                bytes[data..data + r.data.len()].copy_from_slice(r.data);
            };
            for (i, r) in desc.raygens.iter().enumerate() {
                put(starts[0] + raygen_stride * i as u64, r);
            }
            for (i, r) in desc.misses.iter().enumerate() {
                put(starts[1] + miss_stride * i as u64, r);
            }
            for (i, r) in desc.callables.iter().enumerate() {
                if let Some(r) = r {
                    put(callable_start + callable_stride * i as u64, r);
                }
            }
        }

        // Each kernel's table of the visible stages, by stage index.
        let mut raygens = Vec::with_capacity(states.len());
        for (i, state) in states.into_iter().enumerate() {
            let td = MTLVisibleFunctionTableDescriptor::new();
            // SAFETY: a count, which Metal sizes the table by.
            unsafe { td.setFunctionCount(desc.stages.len()) };
            let table = state
                .newVisibleFunctionTableWithDescriptor(&td)
                .ok_or_else(|| invalid("no visible function table".into()))?;
            table.setLabel(Some(&NSString::from_str(name)));
            for &(stage, f) in &visible {
                let h = handle(&state, f)?;
                // SAFETY: the handle is of a function the kernel links, of the table's
                // signature, at an index below the table's count.
                unsafe { table.setFunction_atIndex(Some(&h), stage) };
            }
            {
                let mut r = lock(&shared.residency);
                r.set.addAllocation(ProtocolObject::from_ref(&*table));
                r.dirty = true;
            }
            raygens.push(Raygen {
                state,
                table: Obj(table),
                record: base + starts[0] + raygen_stride * i as u64,
            });
        }
        let region = |start: u64, stride: u64, count: usize| Region {
            address: if count == 0 { 0 } else { base + start },
            stride: stride as u32,
            count: count as u32,
        };
        Ok(RayPipeline {
            raygens,
            inline_root_bytes: root,
            handles,
            hit_records,
            hit_stride,
            misses: region(starts[1], miss_stride, desc.misses.len()),
            callables: region(callable_start, callable_stride, desc.callables.len()),
            call_depth: desc.call_depth,
            launch_groups: ray.groups.clone(),
            checks: ray.checks.gpuAddress(),
            records: Arc::new(records),
            functions: Arc::new(functions.into_iter().map(|(f, _)| f).collect()),
            shared: Arc::clone(shared),
        })
    }

    /// The binary function `source` describes and its cache key, compiled pipeline-independent.
    fn binary_function(
        &self,
        ray: &RayShared,
        source: &Source<'_>,
        name: &str,
    ) -> Result<(Obj<BinaryFunction>, blake3::Hash), Error> {
        let shared = &self.shared;
        let mut key = blake3::Hasher::new();
        key.update(b"binary");
        let (function, label): (Retained<MTL4FunctionDescriptor>, &str) = match source {
            Source::Stage(s, function) => {
                s.key(&mut key);
                match function {
                    Some(f) => {
                        key.update(f.as_bytes());
                        (s.function_named(shared, f)?, f)
                    }
                    None => (s.function(shared)?, s.metal_name()?),
                }
            }
            Source::Default(f) => {
                key.update(f.as_bytes());
                key.update(source_hash().as_bytes());
                let d = MTL4LibraryFunctionDescriptor::new();
                d.setName(Some(&NSString::from_str(f)));
                d.setLibrary(Some(&ray.library));
                (Retained::into_super(d), f)
            }
        };
        let key = key.finalize();
        let d = MTL4BinaryFunctionDescriptor::new();
        d.setName(&NSString::from_str(&key.to_hex()));
        d.setFunctionDescriptor(&function);
        d.setOptions(MTL4BinaryFunctionOptions::PipelineIndependent);
        let f = shared.cached_binary(key, self.cached_only, &d, &format!("{name}: {label}"))?;
        Ok((f, key))
    }
}

/// The hash of the default functions' source, which their cache keys hold.
fn source_hash() -> &'static blake3::Hash {
    static HASH: OnceLock<blake3::Hash> = OnceLock::new();
    HASH.get_or_init(|| blake3::hash(source().as_bytes()))
}

impl CommandList<'_> {
    /// Runs raygen `raygen` of `p` once per element of `extent`, with `root` its inline root,
    /// its traces choosing records of `hits`.
    pub fn trace_rays<R: Pod>(
        &mut self,
        p: &RayPipeline,
        raygen: u32,
        hits: &HitTable,
        root: &R,
        extent: [u32; 3],
    ) {
        let groups = [extent[0].div_ceil(GROUP), extent[1], extent[2]];
        if groups.contains(&0) {
            return;
        }
        let e = self.bind_rays(p, raygen, hits, root, extent, 0);
        e.dispatchThreadgroups_threadsPerThreadgroup(
            super::command::size(groups),
            super::command::size([GROUP, 1, 1]),
        );
    }

    /// As [`Self::trace_rays`], over the extent `args` holds on the GPU (width, height,
    /// depth). Order its writes before the launch with [`Hazards::DRAW_ARGUMENTS`].
    pub fn trace_rays_indirect<R: Pod>(
        &mut self,
        p: &RayPipeline,
        raygen: u32,
        hits: &HitTable,
        root: &R,
        args: GpuPtr<DispatchArgs>,
    ) {
        // The threadgroup counts, in the list's inline memory; the caller's barrier on `args`
        // holds for this dispatch too, which runs in the launch's stage.
        let at = self.inline_bytes(&[0; size_of::<DispatchArgs>()]);
        let a = GroupArgs {
            extent: args.address().0,
            groups: at,
        };
        self.dispatch_state(
            &p.launch_groups,
            bytemuck::bytes_of(&a),
            [1, 1, 1],
            [1, 1, 1],
        );
        self.compute()
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
        let e = self.bind_rays(p, raygen, hits, root, [0; 3], args.address().0);
        // SAFETY: the counts are in the list's inline memory, written above.
        unsafe {
            e.dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(
                at,
                super::command::size([GROUP, 1, 1]),
            );
        }
    }

    /// Binds raygen `raygen`'s kernel with `root` at buffers 0 and 4 and its launch block at 3:
    /// the extent `extent`, or the one at `extent_address` when that is not 0.
    fn bind_rays<R: Pod>(
        &mut self,
        p: &RayPipeline,
        raygen: u32,
        hits: &HitTable,
        root: &R,
        extent: [u32; 3],
        extent_address: u64,
    ) -> Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> {
        assert_eq!(
            size_of::<R>(),
            p.inline_root_bytes as usize,
            "the root is not the size of the stages' inline root"
        );
        if hits.count > 0 {
            assert!(
                hits.stride >= p.hit_stride()
                    && hits.stride <= u64::from(MAX_RECORD_STRIDE)
                    && hits.stride.is_multiple_of(u64::from(HANDLE_ALIGN))
                    && hits.address.0.is_multiple_of(u64::from(TABLE_ALIGN)),
                "hit table {hits:?}: the stride is at least {} and at most {MAX_RECORD_STRIDE}, \
                 a multiple of {HANDLE_ALIGN}, and the address a multiple of {TABLE_ALIGN}",
                p.hit_stride()
            );
        }
        let r = &p.raygens[raygen as usize];
        let launch = PipelineLaunch {
            table: id_bits(r.table.gpuResourceID()),
            raygen: r.record,
            hits: hits.address.0,
            misses: p.misses.address,
            callables: p.callables.address,
            extent_address,
            checks: p.checks,
            reserved0: 0,
            hit_stride: hits.stride as u32,
            hit_count: hits.count as u32,
            miss_stride: p.misses.stride,
            miss_count: p.misses.count,
            callable_stride: p.callables.stride,
            callable_count: p.callables.count,
            call_depth: p.call_depth,
            reserved1: 0,
            extent: [extent[0], extent[1], extent[2], 0],
        };
        let launch = self.inline_bytes(bytemuck::bytes_of(&launch));
        let root = self.inline_bytes(bytemuck::bytes_of(root));
        let e = self.compute();
        e.setComputePipelineState(&r.state);
        // SAFETY: the copies live in the list's inline blocks until the submit completes.
        unsafe {
            self.table.setAddress_atIndex(root, 0);
            self.table.setAddress_atIndex(launch, 3);
            self.table.setAddress_atIndex(root, 4);
        }
        e.setArgumentTable(Some(&self.table));
        self.keep.push(Box::new(r.state.clone()));
        self.keep.push(Box::new(r.table.clone()));
        self.keep.push(Box::new(Arc::clone(&p.records)));
        self.keep.push(Box::new(Arc::clone(&p.functions)));
        e
    }
}
