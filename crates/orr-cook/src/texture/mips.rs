//! Mip chains in linear float: separable box or Kaiser downsampling, per atlas cell where
//! cells allow, normal renormalisation, alpha-coverage scaling and Toksvig roughness.

use orr_pack::texture::{full_mip_count, mip_size};
use rayon::prelude::*;

use super::{Img, MipFilter, Wrap};

pub(crate) struct MipSpec {
    pub filter: MipFilter,
    pub wrap: [Wrap; 2],
    /// Atlas cells per axis.
    pub grid: Option<[u32; 2]>,
    /// RGB holds a unit vector: filter unnormalised, store renormalised.
    pub normal: bool,
    /// Channel and cutoff of alpha-tested coverage.
    pub coverage: Option<(usize, f32)>,
}

/// Normal variance from a normal map added to a roughness channel of a packed texture.
pub(crate) struct Toksvig {
    /// Channel holding perceptual roughness.
    pub channel: usize,
    /// Normal map mip 0 at the packed texture's size, unit vectors in RGB.
    pub normals: Img,
}

/// The full chain from `base` down to 1x1, finished as `spec` asks.
pub(crate) fn chain(base: Img, spec: &MipSpec) -> Vec<Img> {
    let mut base = base;
    if spec.normal {
        for p in &mut base.px {
            *p = normalized(*p);
        }
    }
    let mut out = raw_chain(base, spec);
    if spec.normal {
        for m in &mut out[1..] {
            for p in &mut m.px {
                *p = normalized(*p);
            }
        }
    }
    if let Some((channel, cutoff)) = spec.coverage {
        match_coverage(&mut out, channel, cutoff, spec.grid);
    }
    out
}

/// Each mip filtered from the one above it, before renormalising or coverage scaling.
fn raw_chain(base: Img, spec: &MipSpec) -> Vec<Img> {
    let count = full_mip_count(base.w, base.h);
    let (w, h) = (base.w, base.h);
    let mut raw = vec![base];
    for m in 1..count {
        let [dw, dh] = mip_size(w, h, m);
        let prev = &raw[m as usize - 1];
        let cells = spec
            .grid
            .filter(|&g| cell_mip(prev.w, prev.h, g) && cell_mip(dw, dh, g));
        let next = downsample(prev, dw, dh, spec.filter, spec.wrap, cells);
        raw.push(next);
    }
    raw
}

/// Whether a `w` x `h` mip has whole cells of at least one 4x4 block.
fn cell_mip(w: u32, h: u32, [gx, gy]: [u32; 2]) -> bool {
    w.is_multiple_of(gx) && h.is_multiple_of(gy) && w / gx >= 4 && h / gy >= 4
}

fn normalized(p: [f32; 4]) -> [f32; 4] {
    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if l > 0.0 {
        [p[0] / l, p[1] / l, p[2] / l, p[3]]
    } else {
        [0.0, 0.0, 1.0, p[3]]
    }
}

/// Resamples `src` to `dw` x `dh`. With `cells`, each atlas cell resamples alone with its
/// edges clamped; otherwise the whole image resamples with `wrap`.
pub(crate) fn downsample(
    src: &Img,
    dw: u32,
    dh: u32,
    filter: MipFilter,
    wrap: [Wrap; 2],
    cells: Option<[u32; 2]>,
) -> Img {
    let axis = |sl: u32, dl: u32, cells: u32, wrap: Wrap| -> Vec<Vec<(u32, f32)>> {
        let (sc, dc) = (sl / cells, dl / cells);
        let w = weights(sc, dc, filter, if cells > 1 { Wrap::Clamp } else { wrap });
        (0..dl)
            .map(|i| {
                let base = (i / dc) * sc;
                w[(i % dc) as usize]
                    .iter()
                    .map(|&(j, k)| (base + j, k))
                    .collect()
            })
            .collect()
    };
    let [gx, gy] = cells.unwrap_or([1, 1]);
    let wx = axis(src.w, dw, gx, wrap[0]);
    let wy = axis(src.h, dh, gy, wrap[1]);
    let sw = src.w as usize;
    let mut tmp = vec![[0.0f32; 4]; dw as usize * src.h as usize];
    tmp.par_chunks_mut(dw as usize)
        .zip(src.px.par_chunks(sw))
        .for_each(|(out, row)| {
            for (o, taps) in out.iter_mut().zip(&wx) {
                *o = taps.iter().fold([0.0; 4], |acc, &(j, k)| {
                    let p = row[j as usize];
                    std::array::from_fn(|c| acc[c] + p[c] * k)
                });
            }
        });
    let mut out = Img::new(dw, dh);
    out.px
        .par_chunks_mut(dw as usize)
        .zip(&wy)
        .for_each(|(row, taps)| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = taps.iter().fold([0.0; 4], |acc, &(j, k)| {
                    let p = tmp[j as usize * dw as usize + x];
                    std::array::from_fn(|c| acc[c] + p[c] * k)
                });
            }
        });
    if filter == MipFilter::Kaiser {
        // Negative lobes can overshoot; clamp values that were non-negative everywhere.
        let floor = src.px.iter().fold([f32::INFINITY; 4], |a, p| {
            std::array::from_fn(|c| a[c].min(p[c]))
        });
        for p in &mut out.px {
            for c in 0..4 {
                if floor[c] >= 0.0 {
                    p[c] = p[c].max(0.0);
                }
            }
        }
    }
    out
}

/// Normalised taps per destination texel: (source index, weight).
fn weights(sl: u32, dl: u32, filter: MipFilter, wrap: Wrap) -> Vec<Vec<(u32, f32)>> {
    let s = sl as f32 / dl as f32;
    let fold = |j: i64| -> u32 {
        let n = i64::from(sl);
        (match wrap {
            Wrap::Repeat => j.rem_euclid(n),
            Wrap::Clamp => j.clamp(0, n - 1),
            Wrap::Mirror => {
                let m = j.rem_euclid(2 * n);
                if m >= n { 2 * n - 1 - m } else { m }
            }
        }) as u32
    };
    (0..dl)
        .map(|i| {
            let c = (i as f32 + 0.5) * s;
            let mut taps: Vec<(u32, f32)> = match filter {
                MipFilter::Box => {
                    let (lo, hi) = (c - 0.5 * s, c + 0.5 * s);
                    (lo.floor() as i64..hi.ceil() as i64)
                        .filter_map(|j| {
                            let k = hi.min(j as f32 + 1.0) - lo.max(j as f32);
                            (k > 0.0).then(|| (fold(j), k))
                        })
                        .collect()
                }
                MipFilter::Kaiser => {
                    let r = 3.0 * s;
                    ((c - r).floor() as i64..=(c + r).ceil() as i64)
                        .filter_map(|j| {
                            let k = kaiser((j as f32 + 0.5 - c) / s);
                            (k != 0.0).then(|| (fold(j), k))
                        })
                        .collect()
                }
            };
            let sum: f32 = taps.iter().map(|t| t.1).sum();
            for t in &mut taps {
                t.1 /= sum;
            }
            taps
        })
        .collect()
}

/// Kaiser-windowed sinc over three lobes, alpha 4.
fn kaiser(x: f32) -> f32 {
    const WIDTH: f32 = 3.0;
    const ALPHA: f32 = 4.0;
    if x.abs() >= WIDTH {
        return 0.0;
    }
    let sinc = if x == 0.0 {
        1.0
    } else {
        let p = std::f32::consts::PI * x;
        p.sin() / p
    };
    let t = x / WIDTH;
    sinc * bessel_i0(ALPHA * (1.0 - t * t).sqrt()) / bessel_i0(ALPHA)
}

fn bessel_i0(x: f32) -> f32 {
    let (mut sum, mut term, q) = (1.0f32, 1.0f32, x * x / 4.0);
    for k in 1..32 {
        term *= q / (k * k) as f32;
        sum += term;
        if term < sum * 1e-9 {
            break;
        }
    }
    sum
}

/// A rectangle of texels.
#[derive(Copy, Clone)]
struct Rect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

fn rects(img: &Img, cells: Option<[u32; 2]>) -> Vec<Rect> {
    match cells {
        None => vec![Rect {
            x: 0,
            y: 0,
            w: img.w,
            h: img.h,
        }],
        Some([gx, gy]) => {
            let (cw, ch) = (img.w / gx, img.h / gy);
            (0..gy)
                .flat_map(|cy| {
                    (0..gx).map(move |cx| Rect {
                        x: cx * cw,
                        y: cy * ch,
                        w: cw,
                        h: ch,
                    })
                })
                .collect()
        }
    }
}

/// Sub-samples per texel axis for the coverage metric.
const SUB: u32 = 4;

/// Calls `f` with every coverage sub-sample of `channel` in row `y` of `r`, after scaling
/// texels by `scale` and clamping them to 1: 4x4 bilinear samples per texel, clamped to
/// the rectangle.
fn row_samples(img: &Img, r: Rect, y: u32, channel: usize, scale: f32, mut f: impl FnMut(f32)) {
    let at = |x: i64, y: i64| {
        let x = x.clamp(i64::from(r.x), i64::from(r.x + r.w) - 1) as usize;
        let y = y.clamp(i64::from(r.y), i64::from(r.y + r.h) - 1) as usize;
        (img.px[y * img.w as usize + x][channel] * scale).min(1.0)
    };
    for x in r.x..r.x + r.w {
        for j in 0..SUB {
            for i in 0..SUB {
                let px = x as f32 + (i as f32 + 0.5) / SUB as f32 - 0.5;
                let py = y as f32 + (j as f32 + 0.5) / SUB as f32 - 0.5;
                let (x0, y0) = (px.floor(), py.floor());
                let (fx, fy) = (px - x0, py - y0);
                let (x0, y0) = (x0 as i64, y0 as i64);
                let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
                let bot = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
                f(top * (1.0 - fy) + bot * fy);
            }
        }
    }
}

fn rect_coverage(img: &Img, r: Rect, channel: usize, cutoff: f32) -> f32 {
    scaled_coverage(img, r, channel, cutoff, 1.0)
}

fn scaled_coverage(img: &Img, r: Rect, channel: usize, cutoff: f32, scale: f32) -> f32 {
    let hit: u64 = (r.y..r.y + r.h)
        .into_par_iter()
        .map(|y| {
            let mut hit = 0;
            row_samples(img, r, y, channel, scale, |a| hit += u64::from(a >= cutoff));
            hit
        })
        .sum();
    hit as f32 / (u64::from(r.w * r.h) * u64::from(SUB * SUB)) as f32
}

/// Scales the coverage channel of every mip after the first so its coverage at `cutoff`
/// matches mip 0's, per atlas cell on mips with whole cells.
fn match_coverage(chain: &mut [Img], channel: usize, cutoff: f32, grid: Option<[u32; 2]>) {
    let (base, rest) = chain.split_first_mut().expect("mip 0");
    let whole = rect_coverage(base, rects(base, None)[0], channel, cutoff);
    let per_cell: Option<Vec<f32>> = grid.filter(|&g| cell_mip(base.w, base.h, g)).map(|g| {
        rects(base, Some(g))
            .into_iter()
            .map(|r| rect_coverage(base, r, channel, cutoff))
            .collect()
    });
    rest.par_iter_mut().for_each(|img| {
        let cells = grid.filter(|&g| per_cell.is_some() && cell_mip(img.w, img.h, g));
        let targets = match (cells, &per_cell) {
            (Some(_), Some(t)) => t.clone(),
            _ => vec![whole],
        };
        for (r, target) in rects(img, cells).into_iter().zip(targets) {
            let scale = coverage_scale(img, r, channel, cutoff, target);
            for y in r.y..r.y + r.h {
                for x in r.x..r.x + r.w {
                    let p = &mut img.px[(y * img.w + x) as usize][channel];
                    *p = (*p * scale).clamp(0.0, 1.0);
                }
            }
        }
    });
}

/// The smallest alpha scale whose coverage at `cutoff` inside `r` reaches `target`.
fn coverage_scale(img: &Img, r: Rect, channel: usize, cutoff: f32, target: f32) -> f32 {
    if target <= 0.0 || cutoff <= 0.0 {
        return 1.0;
    }
    // Before texels clamp to 1, coverage at scale s counts samples with a >= cutoff / s, so
    // a histogram gives the scale directly. Clamping only lowers coverage, so that scale is
    // a lower bound; where it falls short, bisect above it.
    const BINS: usize = 1 << 16;
    let hist = (r.y..r.y + r.h)
        .into_par_iter()
        .fold(
            || vec![0u32; BINS],
            |mut hist, y| {
                row_samples(img, r, y, channel, 1.0, |a| {
                    hist[((a.clamp(0.0, 1.0) * (BINS - 1) as f32) as usize).min(BINS - 1)] += 1;
                });
                hist
            },
        )
        .reduce(
            || vec![0u32; BINS],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                a
            },
        );
    let n = u64::from(r.w * r.h) * u64::from(SUB * SUB);
    let want = (f64::from(target) * n as f64).round() as u64;
    let mut above = 0u64;
    let mut bin = BINS;
    while bin > 0 && above < want {
        bin -= 1;
        above += u64::from(hist[bin]);
    }
    let threshold = (bin as f32 / (BINS - 1) as f32).max(1.0 / (BINS - 1) as f32);
    let first = |from: f32, goal: f32| {
        let reaches = |s: f32| scaled_coverage(img, r, channel, cutoff, s) >= goal;
        if reaches(from) {
            return from;
        }
        let (mut lo, mut hi) = (from, from * 2.0);
        while !reaches(hi) {
            if hi > 1e6 {
                return hi;
            }
            lo = hi;
            hi *= 2.0;
        }
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if reaches(mid) { hi = mid } else { lo = mid }
        }
        hi
    };
    // Every scale from `low` up to the next sample's crossing gives the same coverage; the
    // middle of that range keeps texels away from the cutoff, so encoding does not flip them.
    let low = first(cutoff / threshold, target);
    let reached = scaled_coverage(img, r, channel, cutoff, low);
    if reached >= 1.0 {
        return low;
    }
    let next = first(low, reached + 0.5 / n as f32);
    (low * next).sqrt()
}

impl Toksvig {
    /// Adds the normal variance of each mip to the roughness channel: with `r` the length
    /// of the averaged unit normals, `var = (1 - r) / r` and `alpha'^2 = alpha^2 + 2 var`,
    /// where `alpha` is roughness squared.
    pub(crate) fn apply(&self, chain: &mut [Img], spec: &MipSpec) -> Result<(), String> {
        let (w, h) = (chain[0].w, chain[0].h);
        if (self.normals.w, self.normals.h) != (w, h) {
            return Err("Toksvig normals differ in size from the roughness texture".into());
        }
        let normal_spec = MipSpec {
            filter: spec.filter,
            wrap: spec.wrap,
            grid: spec.grid,
            normal: false,
            coverage: None,
        };
        let mut base = self.normals.clone();
        for p in &mut base.px {
            *p = normalized(*p);
        }
        let raw = raw_chain(base, &normal_spec);
        for (m, n) in chain.iter_mut().zip(&raw) {
            for (p, v) in m.px.iter_mut().zip(&n.px) {
                let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                    .sqrt()
                    .clamp(1e-4, 1.0);
                let var = (1.0 - r) / r;
                let rough = p[self.channel].clamp(0.0, 1.0);
                p[self.channel] = (rough.powi(4) + 2.0 * var).powf(0.25).min(1.0);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::{linear_to_srgb, srgb_to_linear};

    fn coverage(img: &Img, channel: usize, cutoff: f32) -> f32 {
        let r = Rect {
            x: 0,
            y: 0,
            w: img.w,
            h: img.h,
        };
        rect_coverage(img, r, channel, cutoff)
    }

    fn spec(filter: MipFilter) -> MipSpec {
        MipSpec {
            filter,
            wrap: [Wrap::Repeat; 2],
            grid: None,
            normal: false,
            coverage: None,
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// Leaves as random ellipses with soft edges on a clear background.
    fn foliage(w: u32, h: u32, seed: u64) -> Img {
        let mut r = Rng(seed);
        let leaves: Vec<[f32; 4]> = (0..60)
            .map(|_| {
                [
                    r.next() * w as f32,
                    r.next() * h as f32,
                    4.0 + r.next() * 18.0,
                    1.5 + r.next() * 5.0,
                ]
            })
            .collect();
        let mut img = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut a = 0.0f32;
                for l in &leaves {
                    let (dx, dy) = ((x as f32 - l[0]) / l[2], (y as f32 - l[1]) / l[3]);
                    a = a.max((1.0 - (dx * dx + dy * dy)) * 3.0);
                }
                img.px[(y * w + x) as usize] = [0.2, 0.5, 0.1, a.clamp(0.0, 1.0)];
            }
        }
        img
    }

    #[test]
    fn coverage_matches_mip0() {
        for filter in [MipFilter::Box, MipFilter::Kaiser] {
            let mut s = spec(filter);
            s.coverage = Some((3, 0.5));
            let base = foliage(256, 128, 7);
            let target = coverage(&base, 3, 0.5);
            let plain = raw_chain(base.clone(), &spec(filter));
            let c = chain(base, &s);
            for (m, (img, raw)) in c.iter().zip(&plain).enumerate().skip(1) {
                if img.w.min(img.h) < 4 {
                    break;
                }
                let got = coverage(img, 3, 0.5);
                assert!(
                    (got - target).abs() <= 0.01,
                    "{filter:?} mip {m}: coverage {got} against {target}, unscaled {}",
                    coverage(raw, 3, 0.5)
                );
            }
        }
    }

    #[test]
    fn atlas_cells_stay_apart() {
        // A 4x2 atlas of 48x48 cells: mips 0-3 have whole cells of at least 4x4.
        let (w, h) = (192, 96);
        let make = |other: f32| {
            let mut img = foliage(w, h, 3);
            for y in 0..h {
                for x in 0..w {
                    if (x / 48, y / 48) != (1, 0) {
                        let p = &mut img.px[(y * w + x) as usize];
                        p[0] = other;
                        p[3] = if (x + y) % 3 == 0 { other } else { 1.0 - other };
                    }
                }
            }
            img
        };
        let mut s = spec(MipFilter::Kaiser);
        s.grid = Some([4, 2]);
        s.coverage = Some((3, 0.5));
        let (a, b) = (chain(make(0.0), &s), chain(make(1.0), &s));
        let cell0 = foliage(w, h, 3);
        let mut cells_checked = 0;
        for (m, (ia, ib)) in a.iter().zip(&b).enumerate() {
            if !cell_mip(ia.w, ia.h, [4, 2]) {
                break;
            }
            let (cw, ch) = (ia.w / 4, ia.h / 2);
            for y in 0..ch {
                for x in cw..2 * cw {
                    let i = (y * ia.w + x) as usize;
                    assert_eq!(ia.px[i], ib.px[i], "mip {m} texel {x},{y}");
                }
            }
            let r = Rect {
                x: cw,
                y: 0,
                w: cw,
                h: ch,
            };
            let want = rect_coverage(
                &cell0,
                Rect {
                    x: 48,
                    y: 0,
                    w: 48,
                    h: 48,
                },
                3,
                0.5,
            );
            let got = rect_coverage(ia, r, 3, 0.5);
            assert!(
                (got - want).abs() <= 0.01,
                "mip {m}: cell coverage {got} against {want}"
            );
            cells_checked += 1;
        }
        assert_eq!(cells_checked, 4);
    }

    #[test]
    fn srgb_mips_average_linear_light() {
        // Alternating black and white sRGB texels average to half the light, not half the
        // code value.
        let mut img = Img::new(64, 64);
        for (i, p) in img.px.iter_mut().enumerate() {
            let v = if (i + i / 64) % 2 == 0 { 1.0 } else { 0.0 };
            *p = [srgb_to_linear(v); 4];
        }
        for filter in [MipFilter::Box, MipFilter::Kaiser] {
            let c = chain(img.clone(), &spec(filter));
            for m in &c[1..] {
                let mean = m.px.iter().map(|p| p[0]).sum::<f32>() / m.px.len() as f32;
                assert!(
                    (mean - 0.5).abs() < 2e-3,
                    "{filter:?} {}x{}: mean {mean}",
                    m.w,
                    m.h
                );
                assert!((linear_to_srgb(mean) - 0.735).abs() < 3e-3);
            }
        }
    }

    #[test]
    fn toksvig_matches_slope_variance() {
        // Slopes of +-a per axis in a checkerboard: every block of a mip averages to zero
        // tilt, per-axis slope variance a^2, and the unit normals all share z.
        let a = 0.15f32;
        let (w, h) = (128, 128);
        let mut normals = Img::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sx = if x % 2 == 0 { a } else { -a };
                let sy = if y % 2 == 0 { a } else { -a };
                normals.px[(y * w + x) as usize] = normalized([-sx, -sy, 1.0, 1.0]);
            }
        }
        let base_rough = 0.3f32;
        let mut img = Img::new(w, h);
        for p in &mut img.px {
            *p = [base_rough, 0.0, 0.0, 1.0];
        }
        let s = spec(MipFilter::Box);
        let mut c = chain(img, &s);
        Toksvig {
            channel: 0,
            normals,
        }
        .apply(&mut c, &s)
        .unwrap();
        let analytic = (base_rough.powi(4) + 2.0 * a * a).powf(0.25);
        assert!(
            (c[0].px[0][0] - base_rough).abs() < 1e-6,
            "mip 0 keeps its roughness"
        );
        for m in &c[1..] {
            for p in &m.px {
                let rel = (p[0] - analytic).abs() / analytic;
                assert!(rel <= 0.02, "{}x{}: {} against {analytic}", m.w, m.h, p[0]);
            }
        }
    }
}
