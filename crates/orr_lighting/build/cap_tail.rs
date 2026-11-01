//! The subsurface walk's cap tail, from walks in a semi-infinite slab entered along its
//! normal: for walks still inside after `cap` steps, the throughput that would yet leave
//! through the boundary per unit throughput at the cap. Indexed by cap, anisotropy and
//! single-scattering albedo, albedo fastest; both axes are spaced where the tail bends.

use std::f64::consts::TAU;
use std::thread;

pub(crate) const ALBEDOS: usize = 32;
pub(crate) const ANISOTROPIES: usize = 8;
pub(crate) const CAPS: [u32; 9] = [4, 8, 16, 32, 64, 128, 256, 512, 1024];
/// The anisotropy axis spans -`G_MAX` to `G_MAX`, evenly in atanh(g).
pub(crate) const G_MAX: f64 = 0.95;

const WALKS: u32 = 16384;
const MAX_STEPS: u32 = 4096;

/// Albedo index `i` holds 1 - s^4 for s = i / (ALBEDOS - 1).
pub(crate) fn albedo(i: usize) -> f64 {
    let s = i as f64 / (ALBEDOS - 1) as f64;
    1.0 - s.powi(4)
}

/// Anisotropy index `j` holds tanh(x) for x from -atanh(`G_MAX`) to atanh(`G_MAX`).
pub(crate) fn anisotropy(j: usize) -> f64 {
    let x = G_MAX.atanh();
    (-x + 2.0 * x * j as f64 / (ANISOTROPIES - 1) as f64).tanh()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The cosine to the slab's normal of a direction scattered by Henyey-Greenstein `g` from
/// one at cosine `mu`.
fn scatter(mu: f64, g: f64, rng: &mut Rng) -> f64 {
    let u = rng.next();
    let c = if g.abs() < 1e-3 {
        1.0 - 2.0 * u
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        ((1.0 + g * g - s * s) / (2.0 * g)).clamp(-1.0, 1.0)
    };
    let sine = (1.0 - c * c).max(0.0).sqrt();
    let phi = TAU * rng.next();
    (mu * c + (1.0 - mu * mu).max(0.0).sqrt() * sine * phi.cos()).clamp(-1.0, 1.0)
}

/// The tails at every cap for albedo `a` and anisotropy `g`. Each flight adds the share
/// that would leave through the boundary and carries on with the rest, its length drawn
/// short of the boundary.
fn cell(a: f64, g: f64, seed: u64) -> [f32; CAPS.len()] {
    let mut rng = Rng(seed);
    // Throughput leaving within each cap, and throughput still inside after it.
    let mut left = [0.0f64; CAPS.len()];
    let mut inside = [0.0f64; CAPS.len()];
    let mut total = 0.0;
    for _ in 0..WALKS {
        let (mut depth, mut mu, mut throughput) = (0.0f64, -1.0f64, 1.0f64);
        let mut out = 0.0;
        let mut next = 0;
        for step in 1..=MAX_STEPS {
            let mut t = -(1.0 - rng.next()).ln();
            if mu > 0.0 {
                let escape = (-depth / mu).exp();
                out += throughput * escape;
                throughput *= 1.0 - escape;
                t = -(1.0 - rng.next() * (1.0 - escape)).ln();
            }
            depth -= mu * t;
            throughput *= a;
            mu = scatter(mu, g, &mut rng);
            if next < CAPS.len() && step == CAPS[next] {
                left[next] += out;
                inside[next] += throughput;
                next += 1;
            }
            if throughput < 1e-4 {
                break;
            }
        }
        // What is still inside at the last step leaves after as many steps again.
        total += out + throughput * a.powi(MAX_STEPS as i32);
        for l in &mut left[next..] {
            *l += out;
        }
    }
    std::array::from_fn(|k| {
        if inside[k] > 0.0 {
            ((total - left[k]) / inside[k]) as f32
        } else {
            0.0
        }
    })
}

/// The table, `CAPS.len()` x `ANISOTROPIES` x `ALBEDOS`.
pub(crate) fn table() -> Vec<f32> {
    let cells: Vec<(usize, usize)> = (0..ANISOTROPIES)
        .flat_map(|j| (0..ALBEDOS).map(move |i| (j, i)))
        .collect();
    let threads = thread::available_parallelism().map_or(4, usize::from);
    let per = cells.len().div_ceil(threads);
    let tails: Vec<[f32; CAPS.len()]> = thread::scope(|scope| {
        let handles: Vec<_> = cells
            .chunks(per)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&(j, i)| cell(albedo(i), anisotropy(j), (j * ALBEDOS + i) as u64))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("cap tail thread"))
            .collect()
    });
    let mut out = vec![0.0; CAPS.len() * cells.len()];
    for (c, tail) in tails.iter().enumerate() {
        for (k, &v) in tail.iter().enumerate() {
            out[k * cells.len() + c] = v;
        }
    }
    out
}
