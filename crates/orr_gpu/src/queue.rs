//! Queue kinds and pass measurements, alike on every backend.

use std::time::Instant;

/// Which queue. Each has one owner thread that submits it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QueueKind {
    /// Graphics and compute (Vulkan family 0); the render thread's. Always present.
    Render,
    /// A compute queue the render thread owns.
    Compute,
    /// A compute queue the loading thread owns.
    Loading,
    /// A transfer queue the loading thread owns.
    Copy,
    /// A second graphics queue the present thread owns.
    Present,
}

impl QueueKind {
    pub(crate) const ALL: [QueueKind; 5] = [
        QueueKind::Render,
        QueueKind::Compute,
        QueueKind::Loading,
        QueueKind::Copy,
        QueueKind::Present,
    ];

    #[cfg_attr(target_vendor = "apple", expect(dead_code))]
    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// The queues a device creates besides [`QueueKind::Render`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct QueueRequest {
    /// [`QueueKind::Compute`].
    pub render_compute: bool,
    /// [`QueueKind::Loading`].
    pub async_compute: bool,
    /// [`QueueKind::Copy`].
    pub copy: bool,
    /// [`QueueKind::Present`].
    pub present: bool,
}

/// Measurements of one pass between [`CommandList::begin_pass`] and
/// [`CommandList::end_pass`].
///
/// [`CommandList::begin_pass`]: crate::CommandList::begin_pass
/// [`CommandList::end_pass`]: crate::CommandList::end_pass
#[derive(Clone, Debug)]
pub struct PassResult {
    pub name: &'static str,
    pub gpu_ms: f64,
    /// Begin and end on the CPU clock; `None` without `Caps::calibrated_timestamps`.
    pub cpu: Option<(Instant, Instant)>,
    /// Shader invocations; `None` unless the pass asked for statistics and the backend
    /// counts them (Metal does not).
    pub stats: Option<PipelineStats>,
}

/// Shader invocation counts of a pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub compute: u64,
    pub task: u64,
    pub mesh: u64,
    pub pixel: u64,
}
