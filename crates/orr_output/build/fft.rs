//! The FFT glare's kernel: per Rec.2020 channel, the energy the diaphragm's PSF puts into each
//! pixel around a source, at any F-number. The Fraunhofer pattern scales with N, so a pixel at
//! F-number N covers a box REFERENCE_F / N times its size in the pattern at REFERENCE_F. Near
//! the source that box's energy comes from a table of the PSF's energy over small boxes at
//! REFERENCE_F; past the table, from the diffraction spikes taken as thin lines whose energy
//! per unit length falls as 1 / r^2, which carry the tail's energy (between the spikes the
//! pattern falls as 1 / r^4). `post/fft_glare.slang` reads the table and spikes that way. Shared
//! by the build script and the output suite.

use super::psf::Diaphragm;

/// The F-number the table holds: the widest pattern, so a pixel never covers less than one
/// table texel's worth of it.
pub const REFERENCE_F: f64 = 22.0;
/// Table texels across a pixel at the reference pitch.
pub const TEXELS_PER_PIXEL: f64 = 2.0;
/// Texels along each axis of the table's quadrant at level 0.
pub const TABLE: usize = 256;
/// Level k of the table sums 2^k by 2^k texels of level 0.
pub const LEVELS: usize = 9;
/// A box is summed at the coarsest level whose texels are at most 1 / BOX_TEXELS of its side.
pub const BOX_TEXELS: f64 = 8.0;

/// Gauss-Legendre nodes and weights on [-1, 1].
const GAUSS: [(f64, f64); 3] = [
    (-0.774_596_669_241_483_4, 5.0 / 9.0),
    (0.0, 8.0 / 9.0),
    (0.774_596_669_241_483_4, 5.0 / 9.0),
];

pub struct Kernel {
    /// Per level, its quadrant's texels row by row: the energy per channel in texel (i, j),
    /// which covers [i, i + 1] x [j, j + 1] in units of its texels (x, y >= 0). The pattern is
    /// mirror symmetric in both axes, which extends the quadrant to the plane.
    pub levels: Vec<Vec<[f64; 3]>>,
    /// Per channel, each spike's energy per unit length past the table times the distance
    /// squared, in level-0 texels.
    pub kappa: [f64; 3],
    pub spikes: Vec<[f64; 2]>,
}

impl Kernel {
    /// The table for `diaphragm` with pixels `pitch_um` across, integrated over wavelength with
    /// `spectral`'s per-channel weights (wavelength in um).
    pub fn compute(diaphragm: &Diaphragm, spectral: &[(f64, [f64; 3])], pitch_um: f64) -> Self {
        let texel = pitch_um / TEXELS_PER_PIXEL;
        let spectral: Vec<_> = spectral
            .iter()
            .filter(|(_, w)| w.iter().any(|&w| w > 0.0))
            .collect();
        let row = |j: usize| -> Vec<[f64; 3]> {
            (0..TABLE)
                .map(|i| {
                    let mut e = [0.0; 3];
                    for &(gx, wx) in &GAUSS {
                        for &(gy, wy) in &GAUSS {
                            let x = (i as f64 + 0.5 + 0.5 * gx) * texel;
                            let y = (j as f64 + 0.5 + 0.5 * gy) * texel;
                            for &&(lambda, w) in &spectral {
                                let scale = lambda * REFERENCE_F;
                                let p = diaphragm.psf([x / scale, y / scale]) / (scale * scale);
                                for c in 0..3 {
                                    e[c] += wx * wy * w[c] * p;
                                }
                            }
                        }
                    }
                    e.map(|e| e * texel * texel / 4.0)
                })
                .collect()
        };
        let threads = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
        let mut base = vec![[0.0; 3]; TABLE * TABLE];
        std::thread::scope(|s| {
            for (t, chunk) in base.chunks_mut(TABLE * TABLE.div_ceil(threads)).enumerate() {
                let row = &row;
                s.spawn(move || {
                    let first = t * TABLE.div_ceil(threads);
                    for (r, out) in chunk.chunks_mut(TABLE).enumerate() {
                        out.copy_from_slice(&row(first + r));
                    }
                });
            }
        });
        let mut levels = vec![base];
        for k in 1..LEVELS {
            let n = TABLE >> k;
            let prev = &levels[k - 1];
            let level = (0..n * n)
                .map(|t| {
                    let (i, j) = (t % n, t / n);
                    let mut e = [0.0; 3];
                    for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let p = prev[(2 * j + b) * 2 * n + 2 * i + a];
                        for c in 0..3 {
                            e[c] += p[c];
                        }
                    }
                    e
                })
                .collect();
            levels.push(level);
        }
        let inside = [0, 1, 2].map(|c| 4.0 * levels[LEVELS - 1][0][c]);
        let spikes = diaphragm.spikes();
        let exits: f64 = spikes
            .iter()
            .map(|s| s[0].abs().max(s[1].abs()) / TABLE as f64)
            .sum();
        Self {
            levels,
            kappa: inside.map(|e| (1.0 - e) / exits),
            spikes,
        }
    }
}
