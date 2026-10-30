//! Deterministic random numbers, Pearson's chi-square test and histograms over the sphere.

use std::f64::consts::PI;

/// SplitMix64; fixed seeds keep every test deterministic.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// A uniformly distributed unit vector.
    pub fn unit(&mut self) -> [f32; 3] {
        loop {
            let v = [
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            ];
            let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (1e-4..=1.0).contains(&l2) {
                let l = l2.sqrt();
                return v.map(|c| c / l);
            }
        }
    }
}

/// The p-value of Pearson's chi-square test of `observed` counts against `expected` counts.
/// Cells expecting fewer than 5 are pooled into one.
pub fn chi_square_p(observed: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(observed.len(), expected.len());
    let (mut stat, mut cells) = (0.0, 0usize);
    let (mut pool_o, mut pool_e) = (0.0, 0.0);
    for (&o, &e) in observed.iter().zip(expected) {
        if e < 5.0 {
            pool_o += o;
            pool_e += e;
        } else {
            stat += (o - e) * (o - e) / e;
            cells += 1;
        }
    }
    if pool_e >= 5.0 {
        stat += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        cells += 1;
    } else if pool_o > 10.0 + 3.0 * pool_e {
        // Samples where the density has next to no mass.
        return 0.0;
    }
    assert!(cells >= 2, "too few cells for a chi-square test");
    gamma_q((cells - 1) as f64 / 2.0, stat / 2.0)
}

/// The regularised upper incomplete gamma function Q(a, x).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_front = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * ln_front.exp()
    } else {
        // Lentz's continued fraction.
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let i = f64::from(i);
            let an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        h * ln_front.exp()
    }
}

/// ln Γ(x) for x > 0 (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Equal-area cells over the sphere: `nz` bands in z = cos(theta) times `nphi` sectors.
pub struct SphereGrid {
    pub nz: usize,
    pub nphi: usize,
}

impl SphereGrid {
    pub fn cells(&self) -> usize {
        self.nz * self.nphi
    }

    pub fn cell(&self, d: [f32; 3]) -> usize {
        let z = f64::from(d[2]).clamp(-1.0, 1.0);
        let phi = f64::from(d[1]).atan2(f64::from(d[0])).rem_euclid(2.0 * PI);
        let iz = (((z + 1.0) / 2.0 * self.nz as f64) as usize).min(self.nz - 1);
        let ip = ((phi / (2.0 * PI) * self.nphi as f64) as usize).min(self.nphi - 1);
        iz * self.nphi + ip
    }

    /// The integral of `pdf` over every cell, by the midpoint rule on `sub` x `sub` points.
    pub fn integrate(&self, sub: usize, pdf: impl Fn([f32; 3]) -> f64) -> Vec<f64> {
        let (dz, dp) = (2.0 / self.nz as f64, 2.0 * PI / self.nphi as f64);
        let point_area = dz * dp / (sub * sub) as f64;
        let mut out = vec![0.0; self.cells()];
        for iz in 0..self.nz {
            for ip in 0..self.nphi {
                let mut sum = 0.0;
                for a in 0..sub {
                    let z = -1.0 + dz * (iz as f64 + (a as f64 + 0.5) / sub as f64);
                    let r = (1.0 - z * z).max(0.0).sqrt();
                    for b in 0..sub {
                        let phi = dp * (ip as f64 + (b as f64 + 0.5) / sub as f64);
                        sum += pdf([(r * phi.cos()) as f32, (r * phi.sin()) as f32, z as f32]);
                    }
                }
                out[iz * self.nphi + ip] = sum * point_area;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_q_matches_chi_square_tables() {
        // Upper-tail probabilities of chi-square with k degrees of freedom.
        for (k, x, p) in [
            (1.0, 3.841_458_8, 0.05),
            (10.0, 23.209_251, 0.01),
            (100.0, 124.342_13, 0.05),
            (4.0, 1.063_623_3, 0.9),
        ] {
            let q = gamma_q(k / 2.0, x / 2.0);
            assert!((q - p).abs() < 1e-6, "k {k} x {x}: {q} vs {p}");
        }
    }
}
