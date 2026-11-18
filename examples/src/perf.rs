//! Frame and GPU pass times for an overlay.

use std::collections::BTreeMap;

use orr::{ExposureTelemetry, FrameStats, ResolvedDisplay};

/// Seconds each figure of the readout averages over.
const READOUT_PERIOD: f64 = 0.5;
/// The costliest GPU passes the readout lists.
const READOUT_PASSES: usize = 8;

/// Frame and pass times, averaged over each `READOUT_PERIOD`.
#[derive(Default)]
pub struct Perf {
    frames: u32,
    seconds: f64,
    worst: f64,
    stats_frame: u64,
    stats_frames: u32,
    cpu_ms: f64,
    passes: BTreeMap<&'static str, f64>,
    figures: String,
}

impl Perf {
    /// Counts a frame that took `delta` seconds, with `stats` the last completed frame's.
    pub fn add(&mut self, delta: f64, stats: &FrameStats) {
        self.frames += 1;
        self.seconds += delta;
        self.worst = self.worst.max(delta);
        if stats.frame != self.stats_frame {
            self.stats_frame = stats.frame;
            self.stats_frames += 1;
            self.cpu_ms += stats
                .cpu
                .iter()
                .map(|(_, d)| d.as_secs_f64() * 1000.0)
                .sum::<f64>();
            for p in &stats.gpu {
                *self.passes.entry(p.name).or_default() += p.gpu_ms;
            }
        }
        if self.seconds < READOUT_PERIOD {
            return;
        }
        let n = f64::from(self.stats_frames.max(1));
        let mut passes: Vec<(&str, f64)> = self.passes.iter().map(|(k, v)| (*k, v / n)).collect();
        passes.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut figures = format!(
            "{:.1} fps  {:.2} ms  worst {:.2} ms\nGPU passes {:.2} ms  CPU {:.2} ms",
            f64::from(self.frames) / self.seconds,
            self.seconds * 1000.0 / f64::from(self.frames),
            self.worst * 1000.0,
            passes.iter().map(|p| p.1).sum::<f64>(),
            self.cpu_ms / n,
        );
        for (name, ms) in passes.iter().take(READOUT_PASSES) {
            figures += &format!("\n{ms:6.2} ms  {name}");
        }
        *self = Self {
            stats_frame: self.stats_frame,
            figures,
            ..Self::default()
        };
    }

    /// The last period's figures.
    pub fn figures(&self) -> &str {
        &self.figures
    }

    /// The overlay's text, headed by `label`.
    pub fn readout(
        &self,
        label: &str,
        [w, h]: [u32; 2],
        resolved: &ResolvedDisplay,
        telemetry: Option<&ExposureTelemetry>,
        averaged: u32,
    ) -> String {
        let ev = telemetry.map_or_default(|t| format!("  EV100 {:.2}", t.adapted_ev100));
        format!(
            "{label}  {w} x {h}  {:?}\n{averaged} frames averaged{ev}\n{}",
            resolved.encoding, self.figures,
        )
    }
}
