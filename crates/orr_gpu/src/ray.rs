//! Descriptions of ray tracing pipelines and their binding tables, alike on every backend.
//! Backends with `Caps::ray_dispatch` create them.

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Condvar, Mutex};

use crate::{Caps, GpuAddress, ShaderStage};

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

impl RayPipelineFlags {
    /// What traces of a scene's acceleration structures may meet on a device of `caps`: cluster
    /// structures, and the opacity micromaps and swept spheres it has.
    pub fn scene(caps: &Caps) -> Self {
        let mut flags = Self::CLUSTERS;
        if caps.omm {
            flags |= Self::OPACITY_MICROMAPS;
        }
        if caps.lss {
            flags |= Self::SWEPT_SPHERES;
        }
        flags
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

/// How many threads compile a [`RayPipelineDesc`]'s shaders: they fold the stages not yet folded
/// ([`ShaderStage::fold`]), and the driver compiles them in parallel on threads that join the
/// creation, up to the concurrency it reports. NVIDIA's driver reports no pipeline cache hit for
/// a creation more than one thread joins, so [`crate::PipelineCacheReport`] counts such a
/// creation from the cache file as a miss.
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

impl CompileThreads<'_> {
    /// What `f` returns for each index below `count`, in order. The calls run on these threads,
    /// the calling one among them, as a pipeline's compile does, and have all returned when this
    /// does; a call that panics panics this once the others have.
    pub fn map<T: Send>(&self, count: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        let out: Vec<Mutex<Option<T>>> = std::iter::repeat_with(|| Mutex::new(None))
            .take(count)
            .collect();
        let work = |i: usize| *out[i].lock().unwrap() = Some(f(i));
        let work: &(dyn Fn(usize) + Sync) = &work;
        let calls = Arc::new(Calls {
            // SAFETY: only the lifetime changes; `Calls::work` says when it is valid.
            work: unsafe {
                std::mem::transmute::<
                    *const (dyn Fn(usize) + Sync + '_),
                    *const (dyn Fn(usize) + Sync + 'static),
                >(work)
            },
            count,
            state: Mutex::default(),
            returned: Condvar::new(),
        });
        // Every call has returned and none starts once this drops, however `map` returns.
        let finished = Finished(&calls);
        let helpers = match *self {
            Self::Spawned(n) | Self::Pool { max: n, .. } => count.min(n as usize).saturating_sub(1),
        };
        match *self {
            Self::Spawned(_) => std::thread::scope(|s| {
                for _ in 0..helpers {
                    s.spawn(|| calls.run());
                }
                calls.run();
            }),
            Self::Pool { spawn, .. } => {
                for _ in 0..helpers {
                    let calls = Arc::clone(&calls);
                    spawn(Box::new(move || calls.run()));
                }
                calls.run();
            }
        }
        drop(finished);
        if let Some(panic) = calls.state.lock().unwrap().panic.take() {
            std::panic::resume_unwind(panic);
        }
        out.into_iter()
            .map(|o| o.into_inner().unwrap().expect("every index is called once"))
            .collect()
    }
}

/// The calls of a [`CompileThreads::map`], which its threads start in turn. A pool's helper may
/// hold it past the map, finding no call left to start.
struct Calls {
    /// The map's work, valid while a call it started has not returned.
    work: *const (dyn Fn(usize) + Sync),
    count: usize,
    state: Mutex<CallState>,
    /// Notified as a call returns.
    returned: Condvar,
}

// SAFETY: the work is Sync, and a thread calls it only while the map waits for the call.
unsafe impl Send for Calls {}
// SAFETY: as above.
unsafe impl Sync for Calls {}

#[derive(Default)]
struct CallState {
    /// The calls started, which is the next index to call.
    started: usize,
    /// The calls started that have not returned.
    running: usize,
    /// No more calls start: one panicked, or the map is done.
    closed: bool,
    /// The first panic of a call.
    panic: Option<Box<dyn std::any::Any + Send>>,
}

impl Calls {
    /// Makes calls until none is left to start.
    fn run(&self) {
        loop {
            let i = {
                let mut s = self.state.lock().unwrap();
                if s.closed || s.started == self.count {
                    return;
                }
                s.started += 1;
                s.running += 1;
                s.started - 1
            };
            // SAFETY: the call is running, so the map waits and its work is live.
            let work = unsafe { &*self.work };
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| work(i)));
            let mut s = self.state.lock().unwrap();
            s.running -= 1;
            if let Err(panic) = result {
                s.closed = true;
                s.panic.get_or_insert(panic);
            }
            drop(s);
            self.returned.notify_all();
        }
    }
}

/// Closes a map's calls and waits for those started to return, as it drops.
struct Finished<'a>(&'a Calls);

impl Drop for Finished<'_> {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap();
        s.closed = true;
        drop(self.0.returned.wait_while(s, |s| s.running > 0).unwrap());
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
/// `callables` (a record of zeros makes no group), and a [`HitTable`] holds hit groups by their
/// index in `hit_groups`. A stage
/// may serve several groups.
///
/// Every stage declares its root with `ORR_ROOT_INLINE` over one root type, or no root.
#[derive(Clone, Copy, Debug)]
pub struct RayPipelineDesc<'a> {
    pub stages: &'a [ShaderStage<'a>],
    pub raygens: &'a [ShaderRecord<'a>],
    pub misses: &'a [ShaderRecord<'a>],
    pub hit_groups: &'a [HitGroup],
    /// By callable index. `None` is a record of zeros, for an index the pipeline's shaders never
    /// call.
    pub callables: &'a [Option<ShaderRecord<'a>>],
    /// How deep traces nest: 1 where only raygens trace, 2 where a closest-hit or miss shader
    /// traces too. A raygen's invoke of a hit object counts as its trace's level.
    pub max_recursion: u32,
    /// How deep stages run one another, where a backend runs closest-hit, miss and callable
    /// stages as functions a raygen calls (Metal): a stage a raygen's trace, invoke or call runs
    /// is at depth 1, a stage that one's trace or call runs at 2. The backend reserves stack for
    /// this many nested calls. The Vulkan backend, whose driver sizes the stack, ignores it.
    pub call_depth: u32,
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
