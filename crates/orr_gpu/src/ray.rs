//! Descriptions of ray tracing pipelines and their binding tables, alike on every backend.
//! Only the Vulkan backend creates them (`Caps::ray_dispatch`).

use crate::{GpuAddress, ShaderStage};

bitflags::bitflags! {
    /// What a [`RayPipelineDesc`]'s traces may meet, and what its binding table never holds.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
    pub struct RayPipelineFlags: u32 {
        /// Traces meet cluster acceleration structures.
        const CLUSTERS = 1 << 0;
        /// Traces see opacity micromaps; without it, traversal of micromapped geometry is
        /// undefined. Needs `Caps::omm`.
        const OPACITY_MICROMAPS = 1 << 1;
        /// Traces meet spheres and linear swept spheres. Needs `Caps::lss`.
        const SWEPT_SPHERES = 1 << 2;
        /// Traces skip triangle geometry.
        const SKIP_TRIANGLES = 1 << 3;
        /// Traces skip procedural (AABB) geometry.
        const SKIP_AABBS = 1 << 4;
        /// No hit record that a trace reaches has a zero any-hit handle.
        const NO_NULL_ANY_HIT = 1 << 5;
        /// No hit record that a trace reaches has a zero closest-hit handle.
        const NO_NULL_CLOSEST_HIT = 1 << 6;
        /// No miss record that a trace reaches is zeros.
        const NO_NULL_MISS = 1 << 7;
        /// No procedural hit record that a trace reaches has a zero intersection handle.
        const NO_NULL_INTERSECTION = 1 << 8;
    }
}

/// A raygen, miss or callable record of the pipeline's own binding table: the stage, by index
/// into [`RayPipelineDesc::stages`], and the record data its entry reads (its `uniform`
/// parameters, at least `EntryPoint::record_bytes` bytes).
#[derive(Clone, Copy, Debug)]
pub struct ShaderRecord<'a> {
    pub stage: u32,
    pub data: &'a [u8],
}

impl ShaderRecord<'_> {
    /// Stage `stage` with no record data.
    pub const fn new(stage: u32) -> Self {
        Self { stage, data: &[] }
    }
}

/// A hit group: the shaders a hit record names, by index into [`RayPipelineDesc::stages`].
/// With an intersection shader the group is procedural, for AABB geometry; without one it
/// is for triangles (and swept spheres).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HitGroup {
    pub closest_hit: Option<u32>,
    pub any_hit: Option<u32>,
    pub intersection: Option<u32>,
}

/// How many threads compile a [`RayPipelineDesc`]'s shaders: the driver compiles them in
/// parallel on threads that join the creation, up to the concurrency it reports. NVIDIA's
/// driver reports no pipeline cache hit for a creation more than one thread joins, so
/// [`crate::PipelineCacheReport`] counts such a creation from the cache file as a miss.
#[derive(Clone, Copy)]
pub enum CompileThreads<'a> {
    /// The calling thread and up to `n - 1` threads the creation spawns. 1 compiles on the
    /// calling thread alone.
    Spawned(u32),
    /// The calling thread and up to `max - 1` tasks handed to `spawn`, which runs them on the
    /// caller's workers. A task joins the creation and returns once it has no more work; one
    /// that starts after the creation finished returns at once.
    Pool {
        spawn: &'a (dyn Fn(Box<dyn FnOnce() + Send>) + Sync),
        max: u32,
    },
}

impl Default for CompileThreads<'_> {
    /// Every hardware thread.
    fn default() -> Self {
        Self::Spawned(std::thread::available_parallelism().map_or(1, |n| n.get() as u32))
    }
}

impl std::fmt::Debug for CompileThreads<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawned(n) => write!(f, "Spawned({n})"),
            Self::Pool { max, .. } => write!(f, "Pool {{ max: {max} }}"),
        }
    }
}

/// A ray tracing pipeline: the shaders, how the binding table groups them and what its traces
/// may meet. The pipeline's groups are numbered in the order given: raygens, misses, hit
/// groups, callables; [`crate::CommandList::trace_rays`] picks a raygen by its index in
/// `raygens`, a trace a miss record by its index in `misses`, a callable by its index in
/// `callables`, and a [`HitTable`] holds hit groups by their index in `hit_groups`. A stage
/// may serve several groups.
///
/// Every stage declares its root with `ORR_ROOT_INLINE` over one root type, or no root.
#[derive(Clone, Copy, Debug)]
pub struct RayPipelineDesc<'a> {
    pub stages: &'a [ShaderStage<'a>],
    pub raygens: &'a [ShaderRecord<'a>],
    pub misses: &'a [ShaderRecord<'a>],
    pub hit_groups: &'a [HitGroup],
    pub callables: &'a [ShaderRecord<'a>],
    /// How deep traces nest: 1 where only raygens trace, 2 where a closest-hit or miss shader
    /// traces too. A raygen's invoke of a hit object counts as its trace's level.
    pub max_recursion: u32,
    pub flags: RayPipelineFlags,
    pub threads: CompileThreads<'a>,
}

/// The hit region of a binding table: `count` records `stride` bytes apart from `address`,
/// each a hit group's handle ([`crate::RayPipeline::hit_handle`]) followed by its record data.
/// A trace's hit record is `instance contribution + geometry index * ray_types + ray_type`.
/// `address` is a multiple of `Limits::ray_table_align`, `stride` of
/// `Limits::ray_handle_align`, at most `Limits::max_ray_record_stride` and at least the
/// pipeline's [`crate::RayPipeline::hit_stride`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HitTable {
    pub address: GpuAddress,
    pub stride: u64,
    pub count: u64,
}

impl HitTable {
    /// No hit records, for launches whose traces hit nothing or whose pipeline has no hit
    /// groups.
    pub const NONE: Self = Self {
        address: GpuAddress(0),
        stride: 0,
        count: 0,
    };
}

/// What the driver reports of a compute pipeline's code, for checking a shader body's register
/// budget through a compute stand-in. Ray tracing pipelines report none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodeStatistics {
    /// Registers per thread.
    pub registers: u32,
    /// Bytes of local memory per thread: spill to memory.
    pub local_bytes: u64,
    /// Bytes of shared memory per workgroup, the compiler's spill among it.
    pub shared_bytes: u64,
    pub binary_bytes: u64,
}
