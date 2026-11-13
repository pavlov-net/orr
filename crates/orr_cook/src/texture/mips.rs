//! Mip chains in linear float: separable box or Kaiser downsampling, per atlas cell where
//! cells allow, normal renormalization, alpha-coverage scaling and Toksvig roughness.

use crate::par::prelude::*;
use crate::simd::{self, Simd, dispatch, simd};
use fearless_simd::f32x4;
use fearless_simd::prelude::*;
use orr_math::Vec3;
use orr_pack::texture::mip_size;

use super::{Img, MipFilter, Wrap};

pub(super) struct MipSpec {
    pub(super) filter: MipFilter,
    pub(super) wrap: [Wrap; 2],
    /// Atlas cells per axis.
    pub(super) grid: Option<[u32; 2]>,
    /// RGB holds a unit vector: filter unnormalized, store renormalized.
    pub(super) normal: bool,
    /// Channel and cutoff of alpha-tested coverage.
    pub(super) coverage: Option<(usize, f32)>,
    /// Red holds heights, which [`settle_heights`] and [`keep_heights`] hold to mip 0's range.
    pub(super) height: bool,
}

/// Normal variance from a normal map added to a roughness channel of a packed texture.
pub(super) struct Toksvig {
    /// Channel holding perceptual roughness.
    pub(super) channel: usize,
    /// Normal map mip 0 at the packed texture's size, unit vectors in RGB.
    pub(super) normals: Img,
}

/// The first `count` mips from `base`, finished as `spec` asks.
pub(super) fn chain(base: Img, spec: &MipSpec, count: u32) -> Vec<Img> {
    let mut base = base;
    if spec.normal {
        normalize(&mut base.px);
    }
    let range = spec.height.then(|| settle_heights(&mut base));
    let mut out = {
        let _z = crate::profile::zone!("mips.downsample");
        raw_chain(base, spec, count)
    };
    // Only x and y are stored and z is rebuilt from them, so a shortened mean would read as a
    // flatter normal, not a shorter one.
    if spec.normal {
        let _z = crate::profile::zone!("mips.renormalize");
        for m in &mut out[1..] {
            normalize(&mut m.px);
        }
    }
    if let Some((channel, cutoff)) = spec.coverage {
        let _z = crate::profile::zone!("mips.coverage");
        match_coverage(&mut out, channel, cutoff, spec.grid);
    }
    if let Some(range) = range {
        keep_heights(&mut out, range);
    }
    out
}

/// `v` in `lo..hi`, which holds 0; NaN reads 0.
pub(super) fn settled(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(lo, hi) }
}

/// Holds `l` to `lo..hi`, the range its format stores. Filtering texels the format can't hold
/// spreads them through the mips: an infinite tap under a negative lobe makes NaN of its
/// neighbors, and a mip should average what the mip above it stores.
pub(super) fn settle(l: &mut Img, lo: f32, hi: f32) {
    simd::update(&mut l.px, |p| *p = p.map(|v| settled(v, lo, hi)));
}

/// Stores non-finite heights as a format of fixed span does, at an end of the image's finite
/// range, or at 0 without one, and returns that range. Geometry can't be displaced by them, and
/// filtering would spread them through the mips.
pub(super) fn settle_heights(l: &mut Img) -> (f32, f32) {
    let (lo, hi) = match l.finite_range(0) {
        (lo, hi) if lo <= hi => (lo, hi),
        _ => (0.0, 0.0),
    };
    keep_heights(std::slice::from_mut(l), (lo, hi));
    (lo, hi)
}

/// Holds the heights of `chain` to `lo..hi`: a mip's filter overshoots at steps, past what a
/// material declares. Unlike `clamp`, this leaves no NaN.
pub(super) fn keep_heights(chain: &mut [Img], (lo, hi): (f32, f32)) {
    for m in chain {
        simd::update(&mut m.px, |p| p[0] = p[0].max(lo).min(hi));
    }
}

/// The first `count` mips, each filtered from the one above it, before renormalizing or
/// coverage scaling.
fn raw_chain(base: Img, spec: &MipSpec, count: u32) -> Vec<Img> {
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

pub(super) fn normalized(p: [f32; 4]) -> [f32; 4] {
    let n = Vec3::from_slice(&p);
    let l = n.length();
    if l > 0.0 && l.is_finite() {
        (n / l).extend(p[3]).to_array()
    } else {
        [0.0, 0.0, 1.0, p[3]]
    }
}

/// [`normalized`] of every texel of `px`.
pub(super) fn normalize(px: &mut [[f32; 4]]) {
    px.par_chunks_mut(simd::PIECE)
        .for_each(|p| dispatch!(simd::level(), s => normalize_piece(s, p)));
}

/// [`normalized`] four texels at a time, with the same operations in the same order.
#[simd]
fn normalize_piece<S: Simd>(s: S, px: &mut [[f32; 4]]) {
    let (quads, rest) = px.as_chunks_mut::<4>();
    for q in quads {
        let flat: &mut [f32; 16] = bytemuck::cast_mut(q);
        let [x, y, z, a] = f32x4::load_four_interleaved(s, flat);
        let l = (x * x + y * y + z * z).sqrt();
        let unit = l.simd_gt(0.0) & l.simd_lt(f32::INFINITY);
        let (zero, one) = (f32x4::splat(s, 0.0), f32x4::splat(s, 1.0));
        let n = [
            unit.select(x / l, zero),
            unit.select(y / l, zero),
            unit.select(z / l, one),
        ];
        f32x4::store_four_interleaved([n[0], n[1], n[2], a], flat);
    }
    for p in rest {
        *p = normalized(*p);
    }
}

/// Resamples `src` to `dw` x `dh`. With `cells`, each atlas cell resamples alone with its
/// edges clamped; otherwise the whole image resamples with `wrap`.
fn downsample(
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
        .for_each(|(out, row)| dispatch!(simd::level(), s => filter_row(s, row, &wx, out)));
    let mut out = Img::new(dw, dh);
    out.px
        .par_chunks_mut(dw as usize)
        .zip(&wy)
        .for_each(|(row, taps)| {
            dispatch!(simd::level(), s => filter_rows(s, &tmp, dw as usize, taps, row));
        });
    if filter == MipFilter::Kaiser {
        // Negative lobes can overshoot; clamp values that were non-negative everywhere.
        let floor = src.px.iter().fold([f32::INFINITY; 4], |a, p| {
            std::array::from_fn(|c| a[c].min(p[c]))
        });
        simd::update(&mut out.px, |p| {
            for c in 0..4 {
                if floor[c] >= 0.0 {
                    p[c] = p[c].max(0.0);
                }
            }
        });
    }
    out
}

/// Each texel of `out` from the texels of `row` its taps in `taps` weigh.
#[simd]
fn filter_row<S: Simd>(_: S, row: &[[f32; 4]], taps: &[Vec<(u32, f32)>], out: &mut [[f32; 4]]) {
    for (o, taps) in out.iter_mut().zip(taps) {
        *o = taps.iter().fold([0.0; 4], |acc, &(j, k)| {
            let p = row[j as usize];
            std::array::from_fn(|c| acc[c] + p[c] * k)
        });
    }
}

/// `out` as the rows `taps` weigh of `src`, rows of `w` texels. Each texel sums its taps in
/// their order from 0, as [`filter_row`] does, a whole row per tap so the sums vectorize along
/// it.
#[simd]
fn filter_rows<S: Simd>(
    _: S,
    src: &[[f32; 4]],
    w: usize,
    taps: &[(u32, f32)],
    out: &mut [[f32; 4]],
) {
    let out = out.as_flattened_mut();
    out.fill(0.0);
    for &(j, k) in taps {
        let row = src[j as usize * w..][..w].as_flattened();
        for (o, &p) in out.iter_mut().zip(row) {
            *o += p * k;
        }
    }
}

/// Normalized taps per destination texel: (source index, weight).
fn weights(sl: u32, dl: u32, filter: MipFilter, wrap: Wrap) -> Vec<Vec<(u32, f32)>> {
    let s = sl as f32 / dl as f32;
    // `None` past a border: a tap that adds nothing but keeps its share of the weights.
    let fold = |j: i64| -> Option<u32> {
        let n = i64::from(sl);
        let at = match wrap {
            Wrap::Repeat => j.rem_euclid(n),
            Wrap::Clamp | Wrap::Constant => j.clamp(0, n - 1),
            Wrap::Mirror => {
                let m = j.rem_euclid(2 * n);
                if m >= n { 2 * n - 1 - m } else { m }
            }
            Wrap::Border if !(0..n).contains(&j) => return None,
            Wrap::Border => j,
        };
        Some(at as u32)
    };
    (0..dl)
        .map(|i| {
            let c = (i as f32 + 0.5) * s;
            let taps: Vec<(i64, f32)> = match filter {
                MipFilter::Box => {
                    let (lo, hi) = (c - 0.5 * s, c + 0.5 * s);
                    (lo.floor() as i64..hi.ceil() as i64)
                        .filter_map(|j| {
                            let k = hi.min(j as f32 + 1.0) - lo.max(j as f32);
                            (k > 0.0).then_some((j, k))
                        })
                        .collect()
                }
                MipFilter::Kaiser => {
                    let r = 3.0 * s;
                    ((c - r).floor() as i64..=(c + r).ceil() as i64)
                        .filter_map(|j| {
                            let k = kaiser((j as f32 + 0.5 - c) / s);
                            (k != 0.0).then_some((j, k))
                        })
                        .collect()
                }
            };
            let sum: f32 = taps.iter().map(|t| t.1).sum();
            taps.into_iter()
                .filter_map(|(j, k)| fold(j).map(|j| (j, k / sum)))
                .collect()
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
const SUB: usize = 4;

/// Per sub-sample position along an axis, at -3/8, -1/8, 1/8 and 3/8 texel from the
/// texel's center: the bilinear weight of the second texel of its pair, and where the pair
/// starts in the texel's 3x3 neighborhood.
const TAPS: [(f32, usize); SUB] = [(0.625, 0), (0.875, 0), (0.125, 1), (0.375, 1)];

/// A texel's 3x3 neighborhood, scaled and clamped to [0, 1].
type Neighborhood = [[f32; 3]; 3];

/// The bilinear sub-sample values of a neighborhood's center texel, row by row.
fn subsamples(a: &Neighborhood) -> [f32; SUB * SUB] {
    let h = a.map(|row| TAPS.map(|(f, x0)| row[x0] * (1.0 - f) + row[x0 + 1] * f));
    std::array::from_fn(|k| {
        let (i, (f, y0)) = (k % SUB, TAPS[k / SUB]);
        h[y0][i] * (1.0 - f) + h[y0 + 1][i] * f
    })
}

/// The sub-samples at or over `cutoff`, one bit each.
fn mask(values: &[f32; SUB * SUB], cutoff: f32) -> u16 {
    values
        .iter()
        .enumerate()
        .fold(0, |m, (k, &v)| m | (u16::from(v >= cutoff) << k))
}

/// `v` and its neighbors in `lo..lo + len`, clamped to that range.
fn around(v: u32, lo: u32, len: u32) -> [u32; 3] {
    [v.max(lo + 1) - 1, v, (v + 1).min(lo + len - 1)]
}

/// One texel's sub-samples still undecided in a coverage search.
#[derive(Copy, Clone)]
struct Pending {
    /// Texel index in the image.
    texel: u32,
    /// Sub-samples not yet known to reach or miss the cutoff.
    mask: u16,
    /// Sub-samples of `mask` that reach the cutoff at the scale last tested.
    hit: u16,
}

/// The coverage of one rectangle of a channel: 4x4 bilinear sub-samples per texel, clamped
/// to the rectangle, after scaling texels and clamping them to [0, 1]. A sub-sample that
/// reaches the cutoff at one scale reaches it at every larger scale.
struct Coverage<'a> {
    img: &'a Img,
    r: Rect,
    channel: usize,
    cutoff: f32,
    /// Every sub-sample of a neighborhood whose values are all at or over `all_hit` reaches
    /// the cutoff; none does when they are all at or under `none_hit`. A bilinear sum of
    /// non-negative values rounds four times, each within a relative 2^-24.
    all_hit: f32,
    none_hit: f32,
}

impl<'a> Coverage<'a> {
    fn new(img: &'a Img, r: Rect, channel: usize, cutoff: f32) -> Self {
        // Far below 1, weighted values can fall subnormal and lose more precision.
        let (all_hit, none_hit) = if cutoff >= 1e-30 {
            (cutoff * (1.0 + 1e-6), cutoff * (1.0 - 1e-6))
        } else {
            (f32::INFINITY, f32::NEG_INFINITY)
        };
        Self {
            img,
            r,
            channel,
            cutoff,
            all_hit,
            none_hit,
        }
    }

    fn samples(&self) -> u64 {
        u64::from(self.r.w * self.r.h) * (SUB * SUB) as u64
    }

    fn reaches(&self, hits: u64, goal: f32) -> bool {
        hits as f32 / self.samples() as f32 >= goal
    }

    /// Texel `i` of the image at `scale`; NaN reads as 1.
    #[expect(clippy::manual_clamp, reason = "clamp keeps NaN")]
    fn scaled(&self, i: u32, scale: f32) -> f32 {
        (self.img.px[i as usize][self.channel] * scale)
            .min(1.0)
            .max(0.0)
    }

    /// The sub-samples of a neighborhood's center texel at or over the cutoff, one bit
    /// each, and whether some scale could make any of them reach it.
    fn hits(&self, a: &Neighborhood) -> (u16, bool) {
        let (lo, hi) = a
            .iter()
            .flatten()
            .fold((f32::INFINITY, 0.0f32), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        if lo >= self.all_hit {
            (u16::MAX, true)
        } else if hi <= self.none_hit {
            (0, hi > 0.0)
        } else {
            (mask(&subsamples(a), self.cutoff), true)
        }
    }

    /// The neighborhood of texel `x`, `y` at `scale`.
    fn neighborhood(&self, x: u32, y: u32, scale: f32) -> Neighborhood {
        let r = self.r;
        let xs = around(x, r.x, r.w);
        around(y, r.y, r.h).map(|y| xs.map(|x| self.scaled(y * self.img.w + x, scale)))
    }

    /// Calls `f` with the neighborhood of each texel of row `y` at `scale`, left to right.
    fn row(&self, y: u32, scale: f32, mut f: impl FnMut(u32, &Neighborhood)) {
        let r = self.r;
        let scaled = around(y, r.y, r.h).map(|y| {
            let at = y * self.img.w + r.x;
            (at..at + r.w)
                .map(|i| self.scaled(i, scale))
                .collect::<Vec<f32>>()
        });
        for i in 0..r.w {
            let xs = around(i, 0, r.w).map(|x| x as usize);
            f(r.x + i, &scaled.each_ref().map(|s| xs.map(|x| s[x])));
        }
    }

    /// Sub-samples reaching the cutoff at `scale`.
    fn count(&self, scale: f32) -> u64 {
        let r = self.r;
        (r.y..r.y + r.h)
            .into_par_iter()
            .map(|y| dispatch!(simd::level(), s => self.count_row(s, y, scale)))
            .sum()
    }

    /// [`Self::count`] of row `y`.
    #[simd]
    fn count_row<S: Simd>(&self, _: S, y: u32, scale: f32) -> u64 {
        let mut hits = 0;
        self.row(y, scale, |_, a| {
            hits += u64::from(self.hits(a).0.count_ones());
        });
        hits
    }

    /// Histogram of the unscaled sub-sample values.
    fn histogram(&self, bins: usize) -> Vec<u32> {
        let r = self.r;
        // Rows per partial histogram, so each counts about a million samples.
        let rows = (1 << 16) / r.w as usize + 1;
        let add = |mut a: Vec<u32>, b: Vec<u32>| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            a
        };
        (r.y..r.y + r.h)
            .into_par_iter()
            .with_min_len(rows)
            .fold(
                || vec![0u32; bins],
                |mut hist, y| {
                    dispatch!(simd::level(), s => self.histogram_row(s, y, &mut hist));
                    hist
                },
            )
            .reduce(|| vec![0u32; bins], add)
    }

    /// The hits at `scale`, and the sub-samples that miss there but reach the cutoff at a
    /// larger scale.
    fn start(&self, scale: f32) -> (u64, Vec<Pending>) {
        let r = self.r;
        (r.y..r.y + r.h)
            .into_par_iter()
            .fold(
                || (0u64, Vec::new()),
                |(mut hits, mut pending), y| {
                    dispatch!(simd::level(), s => self.start_row(s, y, scale, &mut hits, &mut pending));
                    (hits, pending)
                },
            )
            .reduce(
                || (0, Vec::new()),
                |(h, mut a), (g, mut b)| {
                    a.append(&mut b);
                    (h + g, a)
                },
            )
    }

    /// Adds the histogram of row `y` to `hist`, [`Self::histogram`]'s bins.
    #[simd]
    fn histogram_row<S: Simd>(&self, _: S, y: u32, hist: &mut [u32]) {
        let bins = hist.len();
        let bin = |a: f32| ((a * (bins - 1) as f32) as usize).min(bins - 1);
        self.row(y, 1.0, |_, a| {
            // Bilinear weights summing to 1 keep 0 and 1 exact.
            let v = a[0][0];
            if (v == 0.0 || v == 1.0) && a.iter().flatten().all(|&a| a == v) {
                hist[bin(v)] += (SUB * SUB) as u32;
            } else {
                subsamples(a).iter().for_each(|&v| hist[bin(v)] += 1);
            }
        });
    }

    /// [`Self::start`] of row `y`, added to `hits` and `pending`.
    #[simd]
    fn start_row<S: Simd>(
        &self,
        _: S,
        y: u32,
        scale: f32,
        hits: &mut u64,
        pending: &mut Vec<Pending>,
    ) {
        self.row(y, scale, |x, a| {
            let (hit, reachable) = self.hits(a);
            *hits += u64::from(hit.count_ones());
            if hit != u16::MAX && reachable {
                pending.push(Pending {
                    texel: y * self.img.w + x,
                    mask: !hit,
                    hit,
                });
            }
        });
    }

    /// The hits of `pending` at `scale` among the sub-samples each still has undecided.
    #[simd]
    fn test<S: Simd>(&self, _: S, pending: &mut [Pending], scale: f32) -> u64 {
        let w = self.img.w;
        pending
            .iter_mut()
            .map(|p| {
                let a = self.neighborhood(p.texel % w, p.texel / w, scale);
                p.hit = self.hits(&a).0 & p.mask;
                u64::from(p.hit.count_ones())
            })
            .sum()
    }

    /// The smallest scale from `from` on, to bisection precision, whose coverage reaches
    /// `goal`, and the hits at it. Each test counts only the sub-samples not already known
    /// to hit at the bracket's low end or to miss at its high end.
    fn first(&self, from: f32, goal: f32) -> (f32, u64) {
        let (mut known, mut pending) = self.start(from);
        if self.reaches(known, goal) {
            return (from, known);
        }
        let mut test = |scale: f32| {
            let hit: u64 = pending
                .par_chunks_mut(1024)
                .map(|p| dispatch!(simd::level(), s => self.test(s, p, scale)))
                .sum();
            let hits = known + hit;
            let reached = self.reaches(hits, goal);
            if !reached {
                known = hits;
            }
            pending.retain_mut(|p| {
                p.mask = if reached { p.hit } else { p.mask & !p.hit };
                p.mask != 0
            });
            (reached, hits)
        };
        let (mut lo, mut hi) = (from, from * 2.0);
        let mut at_hi = loop {
            let (reached, hits) = test(hi);
            if reached {
                break hits;
            }
            if hi > 1e6 {
                return (hi, hits);
            }
            lo = hi;
            hi *= 2.0;
        };
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            let (reached, hits) = test(mid);
            if reached {
                hi = mid;
                at_hi = hits;
            } else {
                lo = mid;
            }
        }
        (hi, at_hi)
    }
}

fn rect_coverage(img: &Img, r: Rect, channel: usize, cutoff: f32) -> f32 {
    let c = Coverage::new(img, r, channel, cutoff);
    c.count(1.0) as f32 / c.samples() as f32
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
            let scale = coverage_scale(&Coverage::new(img, r, channel, cutoff), target);
            dispatch!(simd::level(), s => scale_coverage(s, img, r, channel, scale));
        }
    });
}

/// Scales `channel` of the texels of `r` in `img` by `scale`, held to `0..=1`.
#[simd]
fn scale_coverage<S: Simd>(_: S, img: &mut Img, r: Rect, channel: usize, scale: f32) {
    for y in r.y..r.y + r.h {
        let at = (y * img.w + r.x) as usize;
        for p in &mut img.px[at..at + r.w as usize] {
            p[channel] = (p[channel] * scale).clamp(0.0, 1.0);
        }
    }
}

/// The smallest alpha scale whose coverage reaches `target`.
fn coverage_scale(c: &Coverage, target: f32) -> f32 {
    if target <= 0.0 || c.cutoff <= 0.0 {
        return 1.0;
    }
    // Before texels clamp to 1, coverage at scale s counts samples with a >= cutoff / s, so
    // a histogram gives the scale directly. Clamping only lowers coverage, so that scale is
    // a lower bound; where it falls short, bisect above it.
    const BINS: usize = 1 << 16;
    let hist = c.histogram(BINS);
    let n = c.samples();
    let want = (f64::from(target) * n as f64).round() as u64;
    let mut above = 0u64;
    let mut bin = BINS;
    while bin > 0 && above < want {
        bin -= 1;
        above += u64::from(hist[bin]);
    }
    let threshold = (bin as f32 / (BINS - 1) as f32).max(1.0 / (BINS - 1) as f32);
    // Every scale from `low` up to the next sample's crossing gives the same coverage; the
    // middle of that range keeps texels away from the cutoff, so encoding does not flip them.
    let (low, hits) = c.first(c.cutoff / threshold, target);
    let reached = hits as f32 / n as f32;
    if reached >= 1.0 {
        return low;
    }
    let goal = reached + 0.5 / n as f32;
    let next = if c.reaches(hits, goal) {
        low
    } else {
        c.first(low, goal).0
    };
    (low * next).sqrt()
}

impl Toksvig {
    /// Adds the normal variance of each mip from mip `first` on to the roughness channel:
    /// with `r` the length of the averaged unit normals, `var = (1 - r) / r` and
    /// `alpha'^2 = alpha^2 + 2 var`, where `alpha` is roughness squared. The normals average
    /// with a box whatever `spec.filter` is: `r` measures spread only under positive weights,
    /// and a sharpening filter's negative lobes push it to 1 where the normals still vary.
    pub(super) fn apply(
        &self,
        chain: &mut [Img],
        spec: &MipSpec,
        first: usize,
    ) -> Result<(), String> {
        let _z = crate::profile::zone!("mips.toksvig");
        let (w, h) = (chain[0].w, chain[0].h);
        if (self.normals.w, self.normals.h) != (w, h) {
            return Err("Toksvig normals differ in size from the roughness texture".into());
        }
        // A border's black is no normal, and would read as spread.
        let inside = |w: Wrap| if w.bounded() { Wrap::Clamp } else { w };
        let normal_spec = MipSpec {
            filter: MipFilter::Box,
            wrap: spec.wrap.map(inside),
            grid: spec.grid,
            normal: false,
            coverage: None,
            height: false,
        };
        let mut base = self.normals.clone();
        normalize(&mut base.px);
        let raw = raw_chain(base, &normal_spec, chain.len() as u32);
        for (m, n) in chain.iter_mut().zip(&raw).skip(first) {
            simd::zip(&mut m.px, &n.px, |p, v| {
                let r = Vec3::from_slice(v).length().clamp(1e-4, 1.0);
                let var = (1.0 - r) / r;
                let rough = p[self.channel].clamp(0.0, 1.0);
                p[self.channel] = (rough.powi(4) + 2.0 * var).powf(0.25).min(1.0);
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use orr_pack::texture::full_mip_count;

    use super::*;
    use crate::texture::linear_to_srgb;

    fn full_chain(base: Img, spec: &MipSpec) -> Vec<Img> {
        let count = full_mip_count(base.w, base.h);
        chain(base, spec, count)
    }

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
            height: false,
        }
    }

    /// Past a border a filter reads black, so white fades toward the edges of a mip; a
    /// constant keeps what a clamp makes of the texels inside.
    #[test]
    fn a_border_filters_in_black_and_a_constant_filters_as_a_clamp() {
        let white = || Img::of(16, 16, vec![[1.0; 4]; 256]);
        let mip = |wrap: Wrap| {
            let spec = MipSpec {
                wrap: [wrap, Wrap::Repeat],
                ..spec(MipFilter::Kaiser)
            };
            chain(white(), &spec, 2).remove(1)
        };
        let border = mip(Wrap::Border);
        assert!(border.px[0][0] < 0.96, "{}", border.px[0][0]);
        assert!((border.px[3][0] - 1.0).abs() < 0.02, "{}", border.px[3][0]);
        assert_eq!(mip(Wrap::Constant), mip(Wrap::Clamp));
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
        let leaves: Vec<[f32; 4]> = Vec::from_fn(60, |_| {
            [
                r.next() * w as f32,
                r.next() * h as f32,
                4.0 + r.next() * 18.0,
                1.5 + r.next() * 5.0,
            ]
        });
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
            let plain = raw_chain(base.clone(), &spec(filter), full_mip_count(base.w, base.h));
            let c = full_chain(base, &s);
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
    fn pruned_search_matches_testing_every_sample() {
        // The search with every sub-sample counted at every step.
        let full = |c: &Coverage, from: f32, goal: f32| -> f32 {
            let reaches = |s: f32| c.reaches(c.count(s), goal);
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
        let base = foliage(256, 128, 5);
        let count = full_mip_count(base.w, base.h);
        let chain = raw_chain(base, &spec(MipFilter::Kaiser), count);
        for img in &chain[1..] {
            let c = Coverage::new(img, rects(img, None)[0], 3, 0.5);
            for goal in [0.05, 0.3, 0.6, 0.97] {
                for from in [0.1, 0.9, 1.3] {
                    let (a, b) = (c.first(from, goal).0, full(&c, from, goal));
                    assert_eq!(a.to_bits(), b.to_bits(), "{}x{} at {goal}", img.w, img.h);
                }
            }
        }
    }

    #[test]
    fn neighborhood_bounds_agree_with_subsamples() {
        let mut rng = Rng(9);
        let img = Img::new(1, 1);
        for cutoff in [0.5, 0.3333, 0.75, 1e-3, 1e-29] {
            let c = Coverage::new(&img, rects(&img, None)[0], 0, cutoff);
            for i in 0..60_000 {
                // Values on either side of the shortcut's thresholds, mixed ones, or
                // anywhere in [0, 1].
                let a: Neighborhood = std::array::from_fn(|_| {
                    std::array::from_fn(|_| {
                        let steps = (rng.next() * 32.0) as u32;
                        match i % 4 {
                            0 => f32::from_bits(cutoff.to_bits() + steps),
                            1 => f32::from_bits(cutoff.to_bits() - steps),
                            2 if rng.next() < 0.9 => f32::from_bits(cutoff.to_bits() + 16),
                            2 => f32::from_bits(cutoff.to_bits() - 1),
                            _ => rng.next().min(1.0),
                        }
                    })
                });
                let direct = mask(&subsamples(&a), cutoff);
                assert_eq!(c.hits(&a).0, direct, "{cutoff}: {a:?}");
            }
        }
    }

    #[test]
    fn sub_samples_are_bilinear_samples_clamped_to_the_rectangle() {
        // The metric itself: SUB x SUB bilinear samples per texel at texel coordinates
        // (centers at integers), reading scaled texels clamped to [0, 1] and to the rectangle.
        let img = foliage(37, 13, 4);
        let reference = |r: Rect, scale: f32, cutoff: f32| -> u64 {
            let at = |x: i64, y: i64| {
                let x = x.clamp(i64::from(r.x), i64::from(r.x + r.w) - 1) as usize;
                let y = y.clamp(i64::from(r.y), i64::from(r.y + r.h) - 1) as usize;
                (img.px[y * img.w as usize + x][3] * scale).clamp(0.0, 1.0)
            };
            let mut hits = 0;
            for y in r.y..r.y + r.h {
                for x in r.x..r.x + r.w {
                    for j in 0..SUB as u32 {
                        for i in 0..SUB as u32 {
                            let px = x as f32 + (i as f32 + 0.5) / SUB as f32 - 0.5;
                            let py = y as f32 + (j as f32 + 0.5) / SUB as f32 - 0.5;
                            let (fx, fy) = (px - px.floor(), py - py.floor());
                            let (x0, y0) = (px.floor() as i64, py.floor() as i64);
                            let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
                            let bot = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
                            hits += u64::from(top * (1.0 - fy) + bot * fy >= cutoff);
                        }
                    }
                }
            }
            hits
        };
        let rects = [(0, 0, 37, 13), (5, 3, 9, 7), (36, 0, 1, 13), (2, 12, 20, 1)]
            .map(|(x, y, w, h)| Rect { x, y, w, h });
        for r in rects {
            for scale in [1.0, 0.7, 1.9] {
                for cutoff in [0.5, 0.3333] {
                    let c = Coverage::new(&img, r, 3, cutoff);
                    assert_eq!(c.count(scale), reference(r, scale, cutoff));
                }
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
                        p[3] = if (x + y).is_multiple_of(3) {
                            other
                        } else {
                            1.0 - other
                        };
                    }
                }
            }
            img
        };
        let mut s = spec(MipFilter::Kaiser);
        s.grid = Some([4, 2]);
        s.coverage = Some((3, 0.5));
        let (a, b) = (full_chain(make(0.0), &s), full_chain(make(1.0), &s));
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
            let code = if (i + i / 64).is_multiple_of(2) {
                255
            } else {
                0
            };
            *p = [orr_color::srgb8_decode(code); 4];
        }
        for filter in [MipFilter::Box, MipFilter::Kaiser] {
            let c = full_chain(img.clone(), &spec(filter));
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
                let sx = if x.is_multiple_of(2) { a } else { -a };
                let sy = if y.is_multiple_of(2) { a } else { -a };
                normals.px[(y * w + x) as usize] = normalized([-sx, -sy, 1.0, 1.0]);
            }
        }
        let base_rough = 0.3f32;
        let mut img = Img::new(w, h);
        img.px.fill([base_rough, 0.0, 0.0, 1.0]);
        let s = spec(MipFilter::Box);
        let mut c = full_chain(img, &s);
        Toksvig {
            channel: 0,
            normals,
        }
        .apply(&mut c, &s, 0)
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

    #[test]
    fn toksvig_reads_ripples_the_mips_still_resolve() {
        // Slopes a sin(2 t x) along x, t = pi / 64: a footprint of w texels keeps a fraction
        // d = sin(w t) / (w sin t) of the wave in its mean, so their variance over it is
        // a^2 / 2 (1 - d^2), and for small slopes the mean alpha^2 added is that variance. A
        // sharpening mip filter keeps these waves in its mean, so its length stays nearer
        // 1 and reads too little of it.
        let (a, wave, size) = (0.2f32, 64.0f32, 256);
        let mut normals = Img::new(size, size);
        for (i, p) in normals.px.iter_mut().enumerate() {
            let x = (i as u32 % size) as f32 + 0.5;
            let slope = a * (std::f32::consts::TAU * x / wave).sin();
            *p = normalized([-slope, 0.0, 1.0, 1.0]);
        }
        let base_rough = 0.05f32;
        let mut img = Img::new(size, size);
        img.px.fill([base_rough, 0.0, 0.0, 1.0]);
        let s = spec(MipFilter::Kaiser);
        let mut c = full_chain(img, &s);
        Toksvig {
            channel: 0,
            normals,
        }
        .apply(&mut c, &s, 0)
        .unwrap();
        for (m, mip) in c.iter().enumerate().take(4).skip(1) {
            let (t, w) = (std::f32::consts::PI / wave, (1 << m) as f32);
            let want = 0.5 * a * a * (1.0 - ((w * t).sin() / (w * t.sin())).powi(2));
            let added = mip
                .px
                .iter()
                .map(|p| p[0].powi(4) - base_rough.powi(4))
                .sum::<f32>()
                / mip.px.len() as f32;
            let ratio = added / want;
            assert!(
                (0.95..=1.05).contains(&ratio),
                "mip {m}: read {ratio} of it"
            );
        }
    }

    /// The vector `normalize` gives `normalized`'s bits at every level this CPU has and the
    /// fallback, for zeros, subnormals, overflow, infinity and NaN, over lengths that leave a
    /// partial group of four.
    #[test]
    fn normalize_matches_normalized_at_every_level() {
        let v = [
            0.0,
            -0.0,
            1.0,
            -3.5,
            1e-30,
            f32::MIN_POSITIVE / 8.0,
            1e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let px: Vec<[f32; 4]> = Vec::from_fn(v.len().pow(3), |i| {
            [v[i % 10], v[i / 10 % 10], v[i / 100], v[(i * 7) % 10]]
        });
        let detected = fearless_simd::Level::new();
        #[allow(unused_mut, reason = "x86 adds the levels below the detected one")]
        let mut levels = vec![detected];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        levels.extend(
            [
                detected.as_sse2().map(Simd::level),
                detected.as_sse4_2().map(Simd::level),
                detected.as_avx2().map(Simd::level),
            ]
            .into_iter()
            .flatten(),
        );
        #[cfg(feature = "simd-fallback")]
        levels.push(fearless_simd::Level::fallback());
        for len in [0, 1, 3, 5, 6, 7, px.len()] {
            let want: Vec<_> = px[..len]
                .iter()
                .map(|&p| normalized(p).map(f32::to_bits))
                .collect();
            for &level in &levels {
                let mut got = px[..len].to_vec();
                dispatch!(level, s => normalize_piece(s, &mut got));
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.map(f32::to_bits), *w, "{level:?} texel {i}: {:?}", px[i]);
                }
            }
        }
    }
}
