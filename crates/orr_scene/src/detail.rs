//! Detail scales: factors of at least 1 on the error thresholds that geometry is chosen by,
//! steered from what completed frames wanted against what holds it.
//!
//! What a frame wants falls roughly with the square of the threshold it ran at. A sample whose
//! load (demand over capacity) is above the band moves the scale to where that model puts its
//! load at the target, and one below it relaxes the scale a little toward 1: a frame shows only
//! the frontier of what a finer threshold would want, which the rest follows once it has
//! landed. Inside the band the scale stays. Each step starts from the scale its own sample ran
//! at, so the samples read in the frames before a step's effect shows, which ran at the same
//! scale, ask for no more than it did.

/// The load a step above the band aims at.
const TARGET: f64 = 0.86;
/// Loads above this coarsen the scale.
const HIGH: f64 = 0.92;
/// Loads under this relax the scale toward 1. Finer cached cuts are admitted only while what
/// they add keeps the load under it.
pub(crate) const LOW: f64 = 0.80;
/// The factor a sample under the band relaxes the scale it ran at by.
const RELAX: f32 = 0.995;
/// Samples for which relaxing stops just above the scale of a sample above the band. A step
/// of demand wider than the band, such as one page too large for what is left of the pool,
/// would otherwise come back each time relaxing reaches it.
const FLOOR_SAMPLES: u32 = 600;

/// What one completed frame wanted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sample {
    /// The scale the frame ran at.
    pub(crate) scale: f32,
    pub(crate) demand: u64,
    pub(crate) capacity: u64,
    /// Whether a coarser threshold would want less: some of the demand is not held whatever
    /// the threshold.
    pub(crate) sheddable: bool,
}

/// One detail scale and its steering.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Detail {
    scale: f32,
    /// For [`FLOOR_SAMPLES`] samples after one above the band: the largest scale a sample
    /// above the band ran at, and the least above it one under the band ran at, between which
    /// relaxing stops.
    over: f32,
    under: f32,
    held: u32,
}

impl Detail {
    pub(crate) fn new() -> Self {
        Self {
            scale: 1.0,
            over: 1.0,
            under: f32::INFINITY,
            held: 0,
        }
    }

    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    pub(crate) fn step(&mut self, s: Sample) {
        if self.held > 0 {
            self.held -= 1;
        } else {
            (self.over, self.under) = (1.0, f32::INFINITY);
        }
        let load = s.demand as f64 / s.capacity as f64;
        if load > HIGH {
            if s.sheddable {
                let want = f64::from(s.scale) * (load / TARGET).sqrt();
                self.scale = self.scale.max(want as f32);
                self.over = self.over.max(s.scale);
                if self.under <= self.over {
                    self.under = f32::INFINITY;
                }
                self.held = FLOOR_SAMPLES;
            }
        } else if load < LOW {
            if self.held > 0 && s.scale > self.over {
                self.under = self.under.min(s.scale);
            }
            // Relaxing tries the middle of what the samples bracket, and stops once the
            // bracket is a step wide.
            let floor = if self.held == 0 {
                1.0
            } else if self.under * RELAX <= self.over {
                self.under
            } else if self.under.is_finite() {
                (self.over * self.under).sqrt()
            } else {
                self.over / RELAX
            };
            self.scale = self.scale.min((s.scale * RELAX).max(floor).max(1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAPACITY: u64 = 1 << 30;

    /// A plant whose demand is `fixed` plus `streamed` at scale 1 falling with the scale to the
    /// power `exponent`, and `cliff.1` more under scale `cliff.0`, sampled `late` frames after
    /// each frame runs.
    struct Plant {
        fixed: f64,
        streamed: f64,
        exponent: f64,
        cliff: (f32, f64),
        late: usize,
    }

    impl Plant {
        fn demand(&self, scale: f32) -> f64 {
            let cliff = if scale < self.cliff.0 {
                self.cliff.1
            } else {
                0.0
            };
            self.fixed + cliff + self.streamed / f64::from(scale).powf(self.exponent)
        }
    }

    /// Runs `frames` frames of `p` from `d`; returns each frame's scale and load.
    fn run(d: &mut Detail, p: &Plant, frames: usize, ran: &mut Vec<f32>) -> Vec<(f32, f64)> {
        let mut out = Vec::new();
        for _ in 0..frames {
            if ran.len() >= p.late {
                let scale = ran[ran.len() - p.late];
                let demand = p.demand(scale);
                d.step(Sample {
                    scale,
                    demand: demand as u64,
                    capacity: CAPACITY,
                    sheddable: p.streamed > 0.0,
                });
            }
            let scale = d.scale();
            ran.push(scale);
            let load = p.demand(scale) / CAPACITY as f64;
            out.push((scale, load));
        }
        out
    }

    /// How often the scale turns from rising to falling or back.
    fn turns(frames: &[(f32, f64)]) -> usize {
        let moves: Vec<bool> = frames
            .windows(2)
            .filter(|w| w[1].0 != w[0].0)
            .map(|w| w[1].0 > w[0].0)
            .collect();
        moves.windows(2).filter(|m| m[0] != m[1]).count()
    }

    #[test]
    fn the_scale_lands_in_the_band_once_per_sample_and_relaxes_to_one() {
        let c = CAPACITY as f64;
        for late in [1, 2, 3] {
            let mut d = Detail::new();
            let mut ran = Vec::new();
            // Three times the capacity: one step to the target, whatever the samples still in
            // flight say.
            let up = Plant {
                fixed: 0.0,
                streamed: 3.0 * c,
                exponent: 2.0,
                cliff: (0.0, 0.0),
                late,
            };
            let frames = run(&mut d, &up, 200, &mut ran);
            let want = (3.0 / TARGET).sqrt() as f32;
            assert!(
                (d.scale() / want - 1.0).abs() < 1e-5,
                "late {late}: {} against one step's {want}",
                d.scale()
            );
            assert_eq!(turns(&frames), 0, "late {late}");
            assert!(frames[late + 1..].iter().all(|f| f.1 < HIGH));

            // Demand halves: the scale relaxes down to where the load enters the band, never
            // rising on the way.
            let half = Plant {
                streamed: 1.5 * c,
                ..up
            };
            let frames = run(&mut d, &half, 2000, &mut ran);
            assert_eq!(turns(&frames), 0, "late {late}");
            let (scale, load) = *frames.last().unwrap();
            assert!(scale > 1.0 && (LOW..=HIGH).contains(&load), "late {late}");
            assert_eq!(frames[frames.len() - 100].0, scale, "settled, late {late}");

            // Little demand: back to 1.
            let low = Plant {
                streamed: 0.3 * c,
                ..up
            };
            let frames = run(&mut d, &low, 2000, &mut ran);
            assert_eq!(turns(&frames), 0, "late {late}");
            assert_eq!(d.scale(), 1.0, "late {late}");
        }
    }

    #[test]
    fn exponents_off_the_model_still_settle_in_the_band() {
        let c = CAPACITY as f64;
        for (exponent, fixed) in [(1.5, 0.0), (2.5, 0.0), (2.0, 0.5 * c)] {
            let mut d = Detail::new();
            let p = Plant {
                fixed,
                streamed: 3.0 * c,
                exponent,
                cliff: (0.0, 0.0),
                late: 2,
            };
            let frames = run(&mut d, &p, 3000, &mut Vec::new());
            let what = format!("exponent {exponent}, fixed {fixed}");
            assert!(turns(&frames) <= 1, "{what}: {}", turns(&frames));
            let (scale, load) = *frames.last().unwrap();
            assert!((LOW..=HIGH).contains(&load), "{what}: load {load}");
            assert_eq!(frames[frames.len() - 100].0, scale, "{what}: settled");
        }

        // Nothing a coarser threshold sheds: the scale stays.
        let mut d = Detail::new();
        let fixed = Plant {
            fixed: 2.0 * c,
            streamed: 0.0,
            exponent: 2.0,
            cliff: (0.0, 0.0),
            late: 2,
        };
        run(&mut d, &fixed, 100, &mut Vec::new());
        assert_eq!(d.scale(), 1.0);
    }

    #[test]
    fn a_step_of_demand_wider_than_the_band_is_found_and_held_above() {
        let c = CAPACITY as f64;
        // Under scale 2 a page too large for the room the pool has left is wanted.
        let p = Plant {
            fixed: 0.0,
            streamed: 0.6 * c * 4.0,
            exponent: 2.0,
            cliff: (2.0, 0.6 * c),
            late: 2,
        };
        let mut d = Detail::new();
        let frames = run(&mut d, &p, FLOOR_SAMPLES as usize / 2, &mut Vec::new());
        let rises = frames.windows(2).filter(|w| w[1].0 > w[0].0).count();
        assert!(rises <= 6, "{rises} steps up");
        let (scale, load) = *frames.last().unwrap();
        assert!(
            (2.0..2.0 / (RELAX * RELAX)).contains(&scale),
            "held at {scale}"
        );
        assert!(load < HIGH);
        assert_eq!(frames[frames.len() - 100].0, scale, "settled");
    }
}
