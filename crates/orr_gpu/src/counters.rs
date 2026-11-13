//! Per-pass hardware counters through NVIDIA's Nsight Perf SDK (the `nsight-perf` feature).

use std::path::Path;

/// Metrics that say why a pass is slow, chosen to fit in one frame on Blackwell GPUs (another
/// GPU's count is [`CounterSession::replays`](crate::CounterSession::replays)), by the SDK's
/// names: GPU time; how much of the time the SMs hold work; their instruction issue against
/// its peak; occupancy, the warps in flight against the most the SMs hold while active; the
/// L2 hit rate; DRAM throughput; and RT core throughput. The SDK's `sm__throughput` (each SM
/// unit against its peak) and `l1tex__t_sector_hit_rate` don't fit with them, and with
/// `sm__throughput` the frame is replayed dozens of times.
pub const DEFAULT_COUNTER_METRICS: &[&str] = &[
    "gpu__time_duration.sum",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active_realtime.avg.pct_of_peak_sustained_active",
    "lts__average_t_sector_hit_rate_realtime.pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "rtcore__throughput_realtime.avg.pct_of_peak_sustained_elapsed",
];

/// Passes a counter frame measures at most; the passes past it are
/// [`PassCounters::Dropped`].
pub const MAX_COUNTER_PASSES: usize = 4096;

/// How a device loads the Nsight Perf SDK, for
/// [`DeviceDesc::perf_counters`](crate::DeviceDesc::perf_counters).
#[derive(Clone, Copy, Debug, Default)]
pub struct CounterDesc<'a> {
    /// The SDK, or its `NvPerf` folder, as `NVPERF_SDK_PATH` names it: the host library
    /// loads from its `bin` (Windows) or `lib` (Linux) folder for the target's architecture.
    /// `None` loads `nvperf_grfx_host.dll` (`libnvperf_grfx_host.so`) through the system's
    /// library search, which on Windows looks beside the program first.
    pub sdk: Option<&'a Path>,
}

/// A metric a counter session collects
/// ([`CounterSession::metrics`](crate::CounterSession::metrics)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterMetric {
    /// As [`Queue::start_counters`](crate::Queue::start_counters) names it.
    pub name: String,
    /// False when the GPU or driver does not expose the metric: its value is then `None` in
    /// every pass.
    pub available: bool,
    /// The SDK's unit, such as `nanosecond` or `percent`; empty for a ratio of like units and
    /// for an unavailable metric.
    pub unit: String,
}

/// One pass's counters from a counter frame
/// ([`CounterSession::begin_frame`](crate::CounterSession::begin_frame)).
#[derive(Clone, Debug, PartialEq)]
pub enum PassCounters {
    /// A value per [`CounterSession::metrics`](crate::CounterSession::metrics) entry, in its
    /// order; `None` for a metric that isn't available.
    Values(Vec<Option<f64>>),
    /// The pass came after the frame's first [`MAX_COUNTER_PASSES`] and was not measured.
    Dropped,
}

/// What [`CounterSession::end_frame`](crate::CounterSession::end_frame) leaves to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterFrame {
    /// The SDK needs the same frame again, between another begin and end, to collect the
    /// rest of the metrics; this frame's passes have no counters.
    Replay,
    /// The collection is complete: the counters are on the results of this frame's passes.
    Complete,
}
