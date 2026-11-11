//! Detail scales: factors of at least 1 on the error thresholds that geometry is chosen by,
//! steered from what completed frames wanted against what holds it.
//!
//! What a frame wants falls roughly with the square of the threshold it ran at. A sample whose
//! load (demand over capacity) is above the band moves the scale to where that model puts its
//! load at the target. One below it moves the scale toward 1: by the same model when all the
//! frame wanted is resident, as its demand is then whole, and else by a little, as a frame shows
//! only the frontier of what a finer threshold would want, which the rest follows once it has
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
/// Samples for which relaxing stops just above the scale of a sample above the band that
/// relaxing led to. A step of demand wider than the band, such as one page too large for what
/// is left of the pool, would otherwise come back each time relaxing reaches it.
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
    /// Whether everything the frame wanted is resident, so the demand holds no unknown part.
    pub(crate) whole: bool,
}

/// One detail scale and its steering.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Detail {
    scale: f32,
    /// For [`FLOOR_SAMPLES`] samples after one above the band that relaxing led to: the
    /// largest scale such a sample ran at, and the least above it one under the band ran at,
    /// between which relaxing stops.
    over: f32,
    under: f32,
    held: u32,
    /// Whether the scale has relaxed since it last rose.
    relaxed: bool,
}

impl Detail {
    pub(crate) fn new() -> Self {
        Self {
            scale: 1.0,
            over: 1.0,
            under: f32::INFINITY,
            held: 0,
            relaxed: false,
        }
    }

    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    pub(crate) fn step(&mut self, s: Sample) {
        // A pool of no bytes holds nothing at any scale, so its samples steer none.
        if s.capacity == 0 {
            return;
        }
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
                // Demand that rose says nothing about the scales under this one; a sample
                // that relaxing led above the band does.
                if self.relaxed {
                    self.over = self.over.max(s.scale);
                    if self.under <= self.over {
                        self.under = f32::INFINITY;
                    }
                    self.held = FLOOR_SAMPLES;
                    self.relaxed = false;
                }
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
            let to = if s.whole {
                (f64::from(s.scale) * (load / TARGET).sqrt()) as f32
            } else {
                s.scale * RELAX
            };
            let to = to.max(floor).max(1.0);
            if to < self.scale {
                self.scale = to;
                self.relaxed = true;
            }
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
        /// Whether its samples hold everything a frame wants, or only a frontier.
        whole: bool,
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
                    whole: p.whole,
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
                whole: false,
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

    /// Demand that falls to a tenth: whole samples bring the scale into the band in a few
    /// steps without passing it, where frontier samples take hundreds, and a fall is not held
    /// off as a step of demand is.
    #[test]
    fn whole_samples_relax_by_the_model_after_demand_falls() {
        let c = CAPACITY as f64;
        for exponent in [1.0, 2.0] {
            let mut d = Detail::new();
            let mut ran = Vec::new();
            let mut p = Plant {
                fixed: 0.0,
                streamed: 40.0 * c,
                exponent,
                cliff: (0.0, 0.0),
                late: 2,
                whole: true,
            };
            run(&mut d, &p, 80, &mut ran);
            p.streamed = 4.0 * c;
            let frames = run(&mut d, &p, 40, &mut ran);
            assert_eq!(turns(&frames), 0, "exponent {exponent}");
            let (_, load) = *frames.last().unwrap();
            assert!(
                (LOW..=HIGH).contains(&load),
                "exponent {exponent}: load {load}"
            );
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
                whole: false,
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
            whole: false,
        };
        run(&mut d, &fixed, 100, &mut Vec::new());
        assert_eq!(d.scale(), 1.0);
    }

    /// A pool cut to a third under a demand with a cliff: below scale 3 the frames want two
    /// capacities more, as a detail that reads finer pages does. Each step is its own sample's,
    /// however late, so the scale rises to what fits, by the first sample's step at most, and
    /// does not come back down.
    #[test]
    fn a_capacity_cut_over_a_cliff_in_the_demand_moves_the_scale_to_what_fits() {
        let c = CAPACITY as f64;
        for late in [1, 3, 6] {
            let p = Plant {
                fixed: 0.3 * c,
                streamed: 6.0 * c,
                exponent: 2.0,
                cliff: (3.0, 2.0 * c),
                late,
                whole: false,
            };
            let mut d = Detail::new();
            let frames = run(&mut d, &p, 400, &mut Vec::new());
            assert_eq!(turns(&frames), 0, "late {late}");
            let (scale, load) = *frames.last().unwrap();
            assert!((LOW..=HIGH).contains(&load), "late {late}: load {load}");
            let fits = (6.0 / (TARGET - 0.3)).sqrt() as f32;
            let first = (8.3 / TARGET).sqrt() as f32;
            assert!(scale <= fits.max(first), "late {late}: {scale}");
            assert_eq!(frames[frames.len() - 100].0, scale, "late {late}: settled");
        }
    }

    /// A pool of no bytes: its samples leave the scale where it is.
    #[test]
    fn samples_of_a_pool_of_no_bytes_leave_the_scale() {
        let mut d = Detail::new();
        for demand in [0, 1 << 20] {
            d.step(Sample {
                scale: 1.0,
                demand,
                capacity: 0,
                sheddable: true,
                whole: true,
            });
            assert_eq!(d.scale(), 1.0);
        }
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
            whole: false,
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
