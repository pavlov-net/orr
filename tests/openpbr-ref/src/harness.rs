//! Checks on a [`Bsdf`]. Every function is deterministic for a given seed.
//!
//! - [`albedo`]: directional albedo per sampled lobe, by Monte Carlo over `sample`.
//! - [`chi_square`]: Pearson's test of `sample` against `pdf` integrated over equal-area
//!   cells of the sphere.
//! - [`reciprocity`]: f(wo, wi) against f(wi, wo).
//! - [`compare`]: eval and pdf of one BSDF against another on the same directions.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use shader_maths::stats::{Rng, SphereGrid, chi_square_p};

use crate::{Bsdf, Inputs, Lobe, V3};

/// A Monte Carlo mean with its standard error, per channel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Estimate {
    pub mean: [f64; 3],
    pub se: [f64; 3],
}

/// Directional albedo for one view. Samples that produce no direction count as zero.
#[derive(Clone, Debug, Default)]
pub struct Albedo {
    pub total: Estimate,
    /// By the lobe `sample` reports.
    pub lobes: BTreeMap<Lobe, Estimate>,
}

/// Directional albedo of `inputs` seen from `view`, from `samples` calls of `sample`.
pub fn albedo(bsdf: &mut dyn Bsdf, inputs: &Inputs, view: V3, samples: usize, seed: u64) -> Albedo {
    bsdf.prepare(inputs, view);
    let mut rng = Rng::new(seed);
    let mut sums: BTreeMap<Lobe, ([f64; 3], [f64; 3])> = BTreeMap::new();
    let u = uniforms(&mut rng, samples);
    for s in bsdf.sample_all(&u).into_iter().flatten() {
        let (sum, sq) = sums.entry(s.lobe).or_default();
        for c in 0..3 {
            let w = f64::from(s.weight[c]);
            sum[c] += w;
            sq[c] += w * w;
        }
    }
    let n = samples as f64;
    let estimate = |sum: [f64; 3], sq: [f64; 3]| {
        let mut e = Estimate::default();
        for c in 0..3 {
            let mean = sum[c] / n;
            e.mean[c] = mean;
            e.se[c] = ((sq[c] / n - mean * mean).max(0.0) / (n - 1.0)).sqrt();
        }
        e
    };
    let (mut total_sum, mut total_sq) = ([0.0; 3], [0.0; 3]);
    let mut lobes = BTreeMap::new();
    for (lobe, (sum, sq)) in sums {
        for c in 0..3 {
            total_sum[c] += sum[c];
            total_sq[c] += sq[c];
        }
        lobes.insert(lobe, estimate(sum, sq));
    }
    Albedo {
        total: estimate(total_sum, total_sq),
        lobes,
    }
}

/// `count` triples of uniform random numbers in [0, 1).
fn uniforms(rng: &mut Rng, count: usize) -> Vec<V3> {
    (0..count)
        .map(|_| [rng.f32(), rng.f32(), rng.f32()])
        .collect()
}

/// The p-value of Pearson's chi-square test of `samples` directions from `sample` against
/// `pdf` integrated over 32 x 64 equal-area cells. The BSDF must have no ideal-specular
/// lobe: such a lobe has no density to integrate.
pub fn chi_square(
    bsdf: &mut dyn Bsdf,
    inputs: &Inputs,
    view: V3,
    samples: usize,
    seed: u64,
) -> f64 {
    bsdf.prepare(inputs, view);
    let grid = SphereGrid { nz: 32, nphi: 64 };
    let mut observed = vec![0.0; grid.cells()];
    let mut rng = Rng::new(seed);
    for s in bsdf
        .sample_all(&uniforms(&mut rng, samples))
        .into_iter()
        .flatten()
    {
        assert!(
            !s.lobe.is(Lobe::SPECULAR),
            "ideal-specular sample; chi-square needs a density"
        );
        observed[grid.cell(s.wi)] += 1.0;
    }
    let n = samples as f64;
    let expected: Vec<f64> = integrate_cells_with(&grid, |d| {
        bsdf.pdf_all(d).into_iter().map(f64::from).collect()
    })
    .into_iter()
    .map(|p| p * n)
    .collect();
    chi_square_p(&observed, &expected)
}

/// The integral of `pdf` over every cell of `grid`, by adaptive Gauss-Legendre quadrature in
/// polar angle and azimuth: glossy lobes are narrow, and near the poles a cell spans many
/// degrees of polar angle.
pub fn integrate_cells(grid: &SphereGrid, pdf: impl Fn(V3) -> f64) -> Vec<f64> {
    integrate_cells_with(grid, |d| d.iter().map(|&d| pdf(d)).collect())
}

/// [`integrate_cells`] with `pdf` asked for many directions at once: one call per level of
/// refinement, covering every cell still being refined.
pub fn integrate_cells_with(grid: &SphereGrid, mut pdf: impl FnMut(&[V3]) -> Vec<f64>) -> Vec<f64> {
    let dphi = 2.0 * PI / grid.nphi as f64;
    let mut cells = Vec::with_capacity(grid.cells());
    for iz in 0..grid.nz {
        let z0 = -1.0 + 2.0 * iz as f64 / grid.nz as f64;
        let z1 = -1.0 + 2.0 * (iz + 1) as f64 / grid.nz as f64;
        let (t0, t1) = (z1.clamp(-1.0, 1.0).acos(), z0.clamp(-1.0, 1.0).acos());
        for ip in 0..grid.nphi {
            cells.push([t0, t1, ip as f64 * dphi, (ip + 1) as f64 * dphi]);
        }
    }
    let wholes = gauss_3x3(&mut pdf, &cells);
    let mut out = vec![0.0; cells.len()];
    // (cell, rectangle, its integral at the coarser level)
    let mut open: Vec<(usize, [f64; 4], f64)> = cells
        .into_iter()
        .zip(wholes)
        .enumerate()
        .map(|(i, (r, w))| (i, r, w))
        .collect();
    let mut depth = 0;
    while !open.is_empty() {
        let children: Vec<[f64; 4]> = open.iter().flat_map(|(_, r, _)| quarters(*r)).collect();
        let parts = gauss_3x3(&mut pdf, &children);
        let mut next = Vec::new();
        for (k, &(cell, _, whole)) in open.iter().enumerate() {
            let parts = &parts[4 * k..4 * k + 4];
            let sum: f64 = parts.iter().sum();
            // 1e-9 absolute is 2e-4 samples at 200,000.
            if (sum - whole).abs() <= 1e-9 || depth >= 12 {
                out[cell] += sum;
            } else {
                next.extend((0..4).map(|j| (cell, children[4 * k + j], parts[j])));
            }
        }
        open = next;
        depth += 1;
    }
    out
}

fn quarters([t0, t1, p0, p1]: [f64; 4]) -> [[f64; 4]; 4] {
    let (tm, pm) = ((t0 + t1) / 2.0, (p0 + p1) / 2.0);
    [
        [t0, tm, p0, pm],
        [t0, tm, pm, p1],
        [tm, t1, p0, pm],
        [tm, t1, pm, p1],
    ]
}

/// 3 x 3 Gauss-Legendre integrals of pdf times sin(theta) over rectangles
/// [theta0, theta1, phi0, phi1], with one call of `pdf` for every point.
fn gauss_3x3(pdf: &mut impl FnMut(&[V3]) -> Vec<f64>, rects: &[[f64; 4]]) -> Vec<f64> {
    const X: [f64; 3] = [-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4];
    const W: [f64; 3] = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0];
    let mut points = Vec::with_capacity(rects.len() * 9);
    let mut weights = Vec::with_capacity(rects.len() * 9);
    for &[t0, t1, p0, p1] in rects {
        let (tc, th) = ((t0 + t1) / 2.0, (t1 - t0) / 2.0);
        let (pc, ph) = ((p0 + p1) / 2.0, (p1 - p0) / 2.0);
        for (xi, wi) in X.iter().zip(W) {
            for (xj, wj) in X.iter().zip(W) {
                let (theta, phi) = (tc + th * xi, pc + ph * xj);
                let s = theta.sin();
                points.push([
                    (s * phi.cos()) as f32,
                    (s * phi.sin()) as f32,
                    theta.cos() as f32,
                ]);
                weights.push(wi * wj * s * th * ph);
            }
        }
    }
    let values = pdf(&points);
    values
        .chunks(9)
        .zip(weights.chunks(9))
        .map(|(v, w)| v.iter().zip(w).map(|(v, w)| v * w).sum())
        .collect()
}

/// The largest relative difference between f(wo, wi) and f(wi, wo) over `pairs` random
/// pairs above the surface (cosines at least 0.05), per channel, where the larger value
/// is above 1e-3. f is eval over the light cosine.
pub fn reciprocity(bsdf: &mut dyn Bsdf, inputs: &Inputs, pairs: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut both = Vec::with_capacity(2 * pairs);
    for _ in 0..pairs {
        let a = upper_hemisphere(&mut rng);
        let b = upper_hemisphere(&mut rng);
        both.extend([[a, b], [b, a]]);
    }
    let evals = bsdf.eval_pairs(inputs, &both);
    let mut worst: f64 = 0.0;
    for (pair, e) in both.chunks(2).zip(evals.chunks(2)) {
        let [a, b] = pair[0];
        let fab = e[0].total().map(|v| f64::from(v) / f64::from(b[2]));
        let fba = e[1].total().map(|v| f64::from(v) / f64::from(a[2]));
        for c in 0..3 {
            let scale = fab[c].abs().max(fba[c].abs());
            if scale > 1e-3 {
                worst = worst.max((fab[c] - fba[c]).abs() / scale);
            }
        }
    }
    worst
}

fn upper_hemisphere(rng: &mut Rng) -> V3 {
    loop {
        let d = rng.unit();
        if d[2].abs() >= 0.05 {
            return [d[0], d[1], d[2].abs()];
        }
    }
}

/// Relative errors of one BSDF against a reference: |a - b| / max(|b|, 1e-3) per channel.
#[derive(Clone, Debug, Default)]
pub struct Errors {
    pub max: f64,
    sum_sq: f64,
    pub count: usize,
    /// Counts by decade: [0] below 1e-7, [k] in [10^(k-8), 10^(k-7)), [7] 1e-1 and above.
    pub histogram: [usize; HISTOGRAM_BINS],
}

pub const HISTOGRAM_BINS: usize = 8;

impl Errors {
    pub fn add(&mut self, a: f64, b: f64) {
        let e = (a - b).abs() / b.abs().max(1e-3);
        let e = if e.is_nan() { f64::INFINITY } else { e };
        self.max = self.max.max(e);
        self.sum_sq += e * e;
        self.count += 1;
        let bin = if e < 1e-7 {
            0
        } else {
            ((e.log10() + 8.0) as usize).min(HISTOGRAM_BINS - 1)
        };
        self.histogram[bin] += 1;
    }

    pub fn rms(&self) -> f64 {
        (self.sum_sq / self.count.max(1) as f64).sqrt()
    }

    pub fn merge(&mut self, other: &Errors) {
        self.max = self.max.max(other.max);
        self.sum_sq += other.sum_sq;
        self.count += other.count;
        for (a, b) in self.histogram.iter_mut().zip(other.histogram) {
            *a += b;
        }
    }
}

/// Eval and pdf errors of `candidate` against `reference`.
#[derive(Clone, Debug, Default)]
pub struct Comparison {
    pub eval: Errors,
    pub pdf: Errors,
    /// Same random numbers, different direction or lobe: rare near lobe-selection
    /// thresholds; reported, not an error by itself.
    pub sample_mismatches: usize,
    pub samples: usize,
}

impl Comparison {
    pub fn merge(&mut self, other: &Comparison) {
        self.eval.merge(&other.eval);
        self.pdf.merge(&other.pdf);
        self.sample_mismatches += other.sample_mismatches;
        self.samples += other.samples;
    }
}

/// Compares eval, pdf and sample of `candidate` with `reference` for `inputs` seen from
/// `view`, on `cases` uniformly random light directions over the sphere.
pub fn compare(
    candidate: &mut dyn Bsdf,
    reference: &mut dyn Bsdf,
    inputs: &Inputs,
    view: V3,
    cases: usize,
    seed: u64,
) -> Comparison {
    candidate.prepare(inputs, view);
    reference.prepare(inputs, view);
    let mut rng = Rng::new(seed);
    let wi: Vec<V3> = (0..cases).map(|_| rng.unit()).collect();
    let u = uniforms(&mut rng, cases);
    let mut out = Comparison {
        samples: cases,
        ..Comparison::default()
    };
    for (a, b) in candidate.eval_all(&wi).iter().zip(reference.eval_all(&wi)) {
        let (a, b) = (a.total(), b.total());
        for c in 0..3 {
            out.eval.add(f64::from(a[c]), f64::from(b[c]));
        }
    }
    for (a, b) in candidate
        .pdf_all(&wi)
        .into_iter()
        .zip(reference.pdf_all(&wi))
    {
        out.pdf.add(f64::from(a), f64::from(b));
    }
    for (sa, sb) in candidate
        .sample_all(&u)
        .into_iter()
        .zip(reference.sample_all(&u))
    {
        let same = match (sa, sb) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.lobe == b.lobe && (0..3).all(|c| (a.wi[c] - b.wi[c]).abs() < 1e-3)
            }
            _ => false,
        };
        out.sample_mismatches += usize::from(!same);
    }
    out
}
