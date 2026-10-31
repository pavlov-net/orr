// Fits the glare pyramid's band weights to the diaphragm PSF: per first level, F-number and
// Rec.2020 channel, non-negative weights of the identity and each band whose kernels match
// the PSF's energy in rings around the source, by relative error, normalized to energy 1.

use crate::psf;
use crate::pyramid::{self, Plane};

// The reference pixel pitch the tables are fitted at: a 36 mm sensor 3840 pixels across.
pub const REFERENCE_PITCH_UM: f64 = 36000.0 / 3840.0;
pub const F_NUMBERS: usize = 10;
pub const F_MIN: f64 = 1.4;
pub const F_MAX: f64 = 22.0;
// Bands fitted on their own pyramid run; the kernels of coarser levels are this level's
// scaled by 2 per level, which the pyramid is to within its sampling.
const EXACT_LEVEL: u32 = 4;
// Impulse positions the exact kernels are averaged over, so each level's grid alignment is
// covered.
const IMPULSES: [[i64; 2]; 4] = [[0, 0], [5, 11], [10, 3], [13, 14]];
// The full-resolution plane the impulses run on; nothing reaches its edges.
const PLANE: u32 = 8192;
// Ring edges in pixels, by distance between pixel centers: the source pixel, then rings
// growing by sqrt(2) out to OUTER.
const FIRST_EDGE: f64 = 1.2;
const OUTER: f64 = 2048.0;

// F-number i of the tables, spaced evenly in log between F_MIN and F_MAX.
pub fn f_number(i: usize) -> f64 {
    F_MIN * (F_MAX / F_MIN).powf(i as f64 / (F_NUMBERS - 1) as f64)
}

// Rings around a pixel center: `edges[b]` is ring b's outer distance, `area_radius[b]` the
// radius of the disc with as much area as the pixels inside it.
struct Rings {
    edges: Vec<f64>,
    area_radius: Vec<f64>,
}

impl Rings {
    fn new() -> Self {
        let mut edges = vec![0.5, FIRST_EDGE];
        while *edges.last().unwrap() < OUTER {
            let next = edges.last().unwrap() * std::f64::consts::SQRT_2;
            edges.push(next);
        }
        let area_radius = edges
            .iter()
            .map(|&e| {
                let r = e.ceil() as i64;
                let inside: i64 = (-r..=r)
                    .map(|x| {
                        let rest = e * e - (x * x) as f64;
                        if rest <= 0.0 {
                            return 0;
                        }
                        // Lattice points with y^2 < rest.
                        let mut y = rest.sqrt().floor() as i64;
                        if (y * y) as f64 >= rest {
                            y -= 1;
                        }
                        2 * y + 1
                    })
                    .sum();
                (inside as f64 / std::f64::consts::PI).sqrt()
            })
            .collect();
        Self { edges, area_radius }
    }

    fn ring(&self, d: f64) -> Option<usize> {
        let b = self.edges.partition_point(|&e| e <= d);
        (b < self.edges.len()).then_some(b)
    }

    fn len(&self) -> usize {
        self.edges.len()
    }
}

// Energy per ring of each band's kernel at full resolution, for a chain starting at
// `first_level`: one entry per mip.
fn band_rings(rings: &Rings, first_level: u32) -> Vec<Vec<f64>> {
    let mips = (pyramid::LAST_LEVEL + 1 - first_level) as usize;
    let exact = (EXACT_LEVEL + 1 - first_level) as usize;
    let size = [PLANE; 2];
    let mut energy = vec![vec![0.0; rings.len()]; exact];
    // Energy of the last exact band by squared distance, for scaling to coarser levels.
    let mut shells = std::collections::BTreeMap::<i64, f64>::new();
    for offset in IMPULSES {
        let at = [
            i64::from(PLANE / 2) + offset[0],
            i64::from(PLANE / 2) + offset[1],
        ];
        let input = Plane::impulse(size, at, 1.0);
        let downs = pyramid::downsamples(&input, first_level, 1.0);
        for (m, e) in energy.iter_mut().enumerate() {
            let mut weights = vec![0.0; mips + 1];
            weights[1 + m] = 1.0;
            let kernel = pyramid::compose(&input, &downs[..=m], &weights, 1.0);
            for wy in 0..kernel.window[1] {
                for wx in 0..kernel.window[0] {
                    let v = kernel.data[wy * kernel.window[0] + wx] / IMPULSES.len() as f64;
                    let dx = kernel.origin[0] + wx as i64 - at[0];
                    let dy = kernel.origin[1] + wy as i64 - at[1];
                    if let Some(b) = rings.ring(((dx * dx + dy * dy) as f64).sqrt()) {
                        e[b] += v;
                    }
                    if m + 1 == exact {
                        *shells.entry(dx * dx + dy * dy).or_default() += v;
                    }
                }
            }
        }
    }
    // Cumulative energy of the last exact band against the radius of equal pixel area.
    let mut count = 0usize;
    let mut cumulative = vec![(0.0, 0.0)];
    let mut sum = 0.0;
    let mut lattice = std::collections::BTreeMap::<i64, usize>::new();
    let reach = shells
        .keys()
        .next_back()
        .map_or(0, |&d2| (d2 as f64).sqrt().ceil() as i64);
    for y in -reach..=reach {
        for x in -reach..=reach {
            *lattice.entry(x * x + y * y).or_default() += 1;
        }
    }
    for (d2, n) in lattice {
        count += n;
        sum += shells.get(&d2).copied().unwrap_or(0.0);
        cumulative.push(((count as f64 / std::f64::consts::PI).sqrt(), sum));
    }
    let at = |r: f64| -> f64 {
        let i = cumulative.partition_point(|&(radius, _)| radius <= r);
        if i >= cumulative.len() {
            return cumulative.last().unwrap().1;
        }
        let (r0, e0) = cumulative[i - 1];
        let (r1, e1) = cumulative[i];
        e0 + (e1 - e0) * (r - r0) / (r1 - r0)
    };
    for m in exact..mips {
        let scale = f64::from(1u32 << (m + 1 - exact));
        let mut e = Vec::with_capacity(rings.len());
        let mut inside = 0.0;
        for &r in &rings.area_radius {
            let c = at(r / scale);
            e.push(c - inside);
            inside = c;
        }
        energy.push(e);
    }
    energy
}

// Non-negative least squares by cyclic coordinate descent on the normal equations.
fn nnls(columns: &[Vec<f64>], target: &[f64], weight: &[f64]) -> Vec<f64> {
    let n = columns.len();
    let dot = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .zip(weight)
            .map(|((x, y), w)| x * y * w * w)
            .sum()
    };
    let gram: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| dot(&columns[i], &columns[j])).collect())
        .collect();
    let rhs: Vec<f64> = (0..n).map(|i| dot(&columns[i], target)).collect();
    let mut x = vec![0.0; n];
    for _ in 0..20000 {
        for i in 0..n {
            if gram[i][i] <= 0.0 {
                continue;
            }
            let rest: f64 = (0..n).filter(|&j| j != i).map(|j| gram[i][j] * x[j]).sum();
            x[i] = ((rhs[i] - rest) / gram[i][i]).max(0.0);
        }
    }
    x
}

// The fitted weights: per first level (1, 2), per F-number, LAST_LEVEL + 1 entries (w0,
// then the band of each mip; zero past the chain's last mip) of per-channel weights.
pub fn tables(ee: &[f64], tail: f64, spectral: &[(f64, [f64; 3])]) -> Vec<Vec<Vec<[f64; 3]>>> {
    let rings = Rings::new();
    let entries = pyramid::LAST_LEVEL as usize + 1;
    (1..=2)
        .map(|first_level| {
            let bands = band_rings(&rings, first_level);
            let mut identity = vec![0.0; rings.len()];
            identity[0] = 1.0;
            let columns: Vec<Vec<f64>> = std::iter::once(identity).chain(bands).collect();
            (0..F_NUMBERS)
                .map(|i| {
                    let n = f_number(i);
                    let mut inside = [0.0; 3];
                    let mut target = vec![[0.0; 3]; rings.len()];
                    for (b, &r) in rings.area_radius.iter().enumerate() {
                        let e = psf::encircled_rgb(ee, tail, spectral, r, n, REFERENCE_PITCH_UM);
                        for c in 0..3 {
                            target[b][c] = (e[c] - inside[c]).max(1e-12);
                        }
                        inside = e;
                    }
                    let mut row = vec![[0.0; 3]; entries];
                    for c in 0..3 {
                        let t: Vec<f64> = target.iter().map(|t| t[c]).collect();
                        let w: Vec<f64> = t.iter().map(|t| 1.0 / t).collect();
                        let x = nnls(&columns, &t, &w);
                        let total: f64 = x.iter().sum();
                        for (e, v) in row.iter_mut().zip(&x) {
                            e[c] = v / total;
                        }
                    }
                    row
                })
                .collect()
        })
        .collect()
}
