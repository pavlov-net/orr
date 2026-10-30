//! Phase functions, their sampling, and the path random numbers.

use std::f64::consts::PI;

use glam::DVec3;

pub(super) fn rayleigh(cos: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + cos * cos)
}

pub(super) fn cornette_shanks(g: f64, cos: f64) -> f64 {
    henyey_greenstein(g, cos) * 1.5 * (1.0 + cos * cos) / (2.0 + g * g)
}

fn henyey_greenstein(g: f64, cos: f64) -> f64 {
    let d = 1.0 + g * g - 2.0 * g * cos;
    (1.0 - g * g) / (4.0 * PI * d * d.sqrt())
}

fn draine(g: f64, alpha: f64, cos: f64) -> f64 {
    henyey_greenstein(g, cos) * (1.0 + alpha * cos * cos)
        / (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0)
}

fn sample_henyey_greenstein(g: f64, rng: &mut Rng) -> f64 {
    let u = rng.next();
    if g.abs() < 1e-6 {
        return 2.0 * u - 1.0;
    }
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
    ((1.0 + g * g - s * s) / (2.0 * g)).clamp(-1.0, 1.0)
}

/// Henyey-Greenstein proposals accepted in proportion to `1 + alpha cos^2`.
fn sample_draine(g: f64, alpha: f64, rng: &mut Rng) -> f64 {
    loop {
        let cos = sample_henyey_greenstein(g, rng);
        if rng.next() * (1.0 + alpha) < 1.0 + alpha * cos * cos {
            return cos;
        }
    }
}

/// Cornette-Shanks is Draine with alpha 1; Rayleigh is Cornette-Shanks with g 0.
pub(super) fn sample_cornette_shanks(g: f64, rng: &mut Rng) -> f64 {
    sample_draine(g, 1.0, rng)
}

/// Jendersie and d'Eon's (2023) HG plus Draine fit for water droplets of a mean
/// diameter, valid from 5 to 50 um.
pub(super) struct FogPhase {
    g_hg: f64,
    g_d: f64,
    alpha: f64,
    w_d: f64,
}

impl FogPhase {
    pub(super) fn new(diameter: f64) -> Self {
        let d = diameter.clamp(5.0, 50.0);
        FogPhase {
            g_hg: (-0.099_056_7 / (d - 1.671_54)).exp(),
            g_d: (-2.206_79 / (d + 3.910_29) - 0.428_934).exp(),
            alpha: (3.624_89 - 8.292_88 / (d + 5.528_25)).exp(),
            w_d: (-0.599_085 / (d - 0.641_583) - 0.665_888).exp(),
        }
    }

    pub(super) fn eval(&self, cos: f64) -> f64 {
        (1.0 - self.w_d) * henyey_greenstein(self.g_hg, cos)
            + self.w_d * draine(self.g_d, self.alpha, cos)
    }

    pub(super) fn sample(&self, rng: &mut Rng) -> f64 {
        if rng.next() < self.w_d {
            sample_draine(self.g_d, self.alpha, rng)
        } else {
            sample_henyey_greenstein(self.g_hg, rng)
        }
    }
}

/// Concentration of the lobe around the light.
const LIGHT_LOBE: f64 = 12.0;

/// Density of the von Mises-Fisher lobe around the light at cosine `cos` to it.
pub(super) fn toward_light(cos: f64) -> f64 {
    LIGHT_LOBE / (2.0 * PI * -(-2.0 * LIGHT_LOBE).exp_m1()) * (LIGHT_LOBE * (cos - 1.0)).exp()
}

/// A cosine to the light drawn from [`toward_light`].
pub(super) fn sample_toward_light(rng: &mut Rng) -> f64 {
    let u = rng.next();
    (1.0 + (u + (1.0 - u) * (-2.0 * LIGHT_LOBE).exp()).ln() / LIGHT_LOBE).clamp(-1.0, 1.0)
}

/// A unit direction at cosine `cos` to `axis`, uniform in azimuth.
pub(super) fn around(axis: DVec3, cos: f64, rng: &mut Rng) -> DVec3 {
    let (t, b) = axis.any_orthonormal_pair();
    let sin = (1.0 - cos * cos).max(0.0).sqrt();
    let phi = 2.0 * PI * rng.next();
    (axis * cos + (t * phi.cos() + b * phi.sin()) * sin).normalize()
}

/// A cosine-weighted direction about the normal `n`.
pub(super) fn cosine_direction(n: DVec3, rng: &mut Rng) -> DVec3 {
    around(n, rng.next().sqrt(), rng)
}

/// SplitMix64 seeded from (sample, interaction): a path's numbers do not depend on the
/// other paths or on how many numbers earlier interactions drew.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(sample: u32, interaction: u32) -> Self {
        let seed = (u64::from(sample) << 32) | u64::from(interaction);
        let mut r = Rng(seed);
        r.next_u64();
        r
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub(super) fn next(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
