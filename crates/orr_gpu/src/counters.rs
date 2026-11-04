//! Counter frame results, alike on every backend.

/// The metrics a counter frame collects for every pass, in the order of
/// [`CounterRange::values`].
pub const COUNTER_METRICS: [&str; 6] = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
];

/// The counters of one counter frame: a range per pass, in submission order.
#[derive(Clone, Debug, Default)]
pub struct CounterReport {
    pub chip: String,
    pub ranges: Vec<CounterRange>,
}

/// One pass's counters.
#[derive(Clone, Debug)]
pub struct CounterRange {
    pub name: String,
    /// Values of [`COUNTER_METRICS`], in its order.
    pub values: Vec<f64>,
}
