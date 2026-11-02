//! Cooked textures a mesh cook reads: the opacity mip 0 that decides which triangles of an
//! alpha-tested section are opaque, and the displacement mips that bound each cluster's
//! displacement.

use crate::texture::Wrap;
use rayon::prelude::*;

/// One mip of one channel, row-major, with the sampler's edge modes and the material
/// slot's uv transform (`Material::uv_transforms` form).
pub(crate) struct Raster {
    pub w: u32,
    pub h: u32,
    pub texels: Vec<f32>,
    pub wrap: [Wrap; 2],
    pub uv: [f32; 6],
}

/// The side of the opacity cutoff a texel is on: at or over it.
pub(crate) const PASS: u8 = 1;
/// Under it.
pub(crate) const FAIL: u8 = 2;

/// Per row of an opacity mip, how many of its first `x` texels pass the cutoff, for every
/// `x` up to the row's width.
pub(crate) struct Passes {
    prefix: Vec<u16>,
}

impl Passes {
    pub(crate) fn new(mip: &Raster, cutoff: f32) -> Self {
        assert!(
            mip.w <= u32::from(u16::MAX),
            "opacity texture over 65535 texels wide"
        );
        let w = mip.w as usize;
        let mut prefix = vec![0; (w + 1) * mip.h as usize];
        prefix
            .par_chunks_mut(w + 1)
            .zip(mip.texels.par_chunks(w))
            .for_each(|(out, row)| {
                let mut n = 0u16;
                for (o, &t) in out[1..].iter_mut().zip(row) {
                    n += u16::from(t >= cutoff);
                    *o = n;
                }
            });
        Self { prefix }
    }

    /// Which sides of the cutoff the texels `mip.footprint(uv)` visits are on, as `PASS`
    /// and `FAIL` bits; stops once it has seen every side in `stop`.
    pub(crate) fn footprint(&self, mip: &Raster, uv: [[f32; 2]; 3], stop: u8) -> u8 {
        let mut sides = 0;
        let stride = mip.w as usize + 1;
        mip.footprint_runs(uv, |y, first, last| {
            let row = &self.prefix[wrapped(y, mip.h, mip.wrap[1]) * stride..];
            wrapped_runs(first, last, mip.w, mip.wrap[0], |a, b| {
                let pass = usize::from(row[b + 1] - row[a]);
                sides |= if pass > 0 { PASS } else { 0 } | if pass <= b - a { FAIL } else { 0 };
                sides & stop != stop
            })
        });
        sides
    }
}

/// Up to two inclusive index ranges; an empty one has its start past its end.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Ranges(pub [[usize; 2]; 2]);

impl Ranges {
    /// Indices the ranges hold, counting overlaps twice.
    pub(crate) fn len(&self) -> usize {
        self.0.iter().map(|r| (r[1] + 1).saturating_sub(r[0])).sum()
    }

    fn all(n: u32) -> Self {
        Self([[0, n as usize - 1], [1, 0]])
    }
}

/// The part of the run `r` where `keep` holds, given that it holds on a run at one end of
/// `r`, or everywhere or nowhere in it. The search for that run's other end starts at
/// `guess`.
fn narrow((a, b): (i64, i64), guess: f64, keep: impl Fn(i64) -> bool) -> Option<(i64, i64)> {
    match (keep(a), keep(b)) {
        (true, true) => Some((a, b)),
        (false, false) => None,
        (true, false) => {
            let mut x = (guess as i64).clamp(a, b - 1);
            while !keep(x) {
                x -= 1;
            }
            while keep(x + 1) {
                x += 1;
            }
            Some((a, x))
        }
        (false, true) => {
            let mut x = (guess as i64).clamp(a + 1, b);
            while !keep(x) {
                x += 1;
            }
            while keep(x - 1) {
                x -= 1;
            }
            Some((x, b))
        }
    }
}

/// Index `v` of an axis of `n` texels under `wrap`.
pub(crate) fn wrapped(v: i64, n: u32, wrap: Wrap) -> usize {
    let n = i64::from(n);
    (match wrap {
        Wrap::Repeat => v.rem_euclid(n),
        Wrap::Clamp => v.clamp(0, n - 1),
        Wrap::Mirror => {
            let m = v.rem_euclid(2 * n);
            if m < n { m } else { 2 * n - 1 - m }
        }
    }) as usize
}

/// Calls `f` with index runs that together hold exactly the indices `first..=last` wraps
/// to along an axis of `n` texels under `wrap`; stops when `f` returns false.
pub(crate) fn wrapped_runs(
    first: i64,
    last: i64,
    n: u32,
    wrap: Wrap,
    mut f: impl FnMut(usize, usize) -> bool,
) -> bool {
    // Within one period the wrap is monotone, so each period's part maps to one run.
    let period = i64::from(n);
    let mut a = first;
    while a <= last {
        let b = last.min((a.div_euclid(period) + 1) * period - 1);
        let (x, y) = (wrapped(a, n, wrap), wrapped(b, n, wrap));
        if !f(x.min(y), x.max(y)) {
            return false;
        }
        a = b + 1;
    }
    true
}

/// The lower and upper corners of `p`'s bounding box.
fn bounds(p: &[[f64; 2]; 3]) -> ([f64; 2], [f64; 2]) {
    let lo = [0, 1].map(|a| p.iter().map(|q| q[a]).fold(f64::INFINITY, f64::min));
    let hi = [0, 1].map(|a| p.iter().map(|q| q[a]).fold(f64::NEG_INFINITY, f64::max));
    (lo, hi)
}

impl Raster {
    /// Mesh uv `p` in texels of this mip, after the uv transform.
    pub(crate) fn texel_pos(&self, p: [f32; 2]) -> [f64; 2] {
        let q = self.mapped(p);
        [q[0] * f64::from(self.w), q[1] * f64::from(self.h)]
    }

    /// Mesh uv `p` as the texture is sampled at.
    pub(crate) fn mapped(&self, p: [f32; 2]) -> [f64; 2] {
        let [a, b, c, d, e, f] = self.uv.map(f64::from);
        let (u, v) = (f64::from(p[0]), f64::from(p[1]));
        [(1.0 + a) * u + b * v + c, d * u + (1.0 + e) * v + f]
    }

    /// `material_normal_frame`: tangent-space normal `v` read from this normal map, turned
    /// into the mesh uv's frame by the normalized columns of the inverse uv transform.
    pub(crate) fn normal_frame(&self, v: [f32; 3]) -> [f32; 3] {
        let [a, b, _, d, e, _] = self.uv;
        let (a, e) = (1.0 + a, 1.0 + e);
        let det = a * e - b * d;
        if det == 0.0 {
            return v;
        }
        let s = det.signum();
        let unit = |x: f32, y: f32| {
            let l = (x * x + y * y).sqrt();
            [x / l * s, y / l * s]
        };
        let (u, w) = (unit(e, -d), unit(-b, a));
        [u[0] * v[0] + w[0] * v[1], u[1] * v[0] + w[1] * v[1], v[2]]
    }

    fn texel(&self, x: i64, y: i64) -> f32 {
        let x = wrapped(x, self.w, self.wrap[0]);
        let y = wrapped(y, self.h, self.wrap[1]);
        self.texels[y * self.w as usize + x]
    }

    /// Calls `f` with every texel a bilinear sample anywhere on the triangle can read: the
    /// texels whose squares, grown by one texel on each side, overlap the triangle. Stops when
    /// `f` returns false. A triangle spanning more than two texture periods visits every
    /// texel.
    pub(crate) fn footprint(&self, uv: [[f32; 2]; 3], mut f: impl FnMut(f32) -> bool) {
        let w = self.w as usize;
        self.footprint_runs(uv, |y, first, last| {
            let row = &self.texels[wrapped(y, self.h, self.wrap[1]) * w..][..w];
            wrapped_runs(first, last, self.w, self.wrap[0], |a, b| {
                row[a..=b].iter().all(|&t| f(t))
            })
        });
    }

    /// Calls `f` with the texels [`Self::footprint`] visits as runs along x, rows in order:
    /// each run's row and first and last texel, before wrapping. Stops when `f` returns
    /// false.
    pub(crate) fn footprint_runs(
        &self,
        uv: [[f32; 2]; 3],
        mut f: impl FnMut(i64, i64, i64) -> bool,
    ) {
        let size = [f64::from(self.w), f64::from(self.h)];
        let p = uv.map(|t| self.texel_pos(t));
        let (lo, hi) = bounds(&p);
        if (0..2).any(|a| hi[a] - lo[a] > 2.0 * size[a]) {
            (0..i64::from(self.h)).all(|y| f(y, 0, i64::from(self.w) - 1));
            return;
        }
        // Each edge's outward normal, as n.x x + n.y y + d > 0 outside; degenerate edges
        // never reject.
        let edges = [0, 1, 2].map(|i| {
            let (a, b) = (p[i], p[(i + 1) % 3]);
            let n = [b[1] - a[1], a[0] - b[0]];
            let c = p[(i + 2) % 3];
            let side = n[0] * (c[0] - a[0]) + n[1] * (c[1] - a[1]);
            let s = if side > 0.0 { -1.0 } else { 1.0 };
            [s * n[0], s * n[1], -s * (n[0] * a[0] + n[1] * a[1])]
        });
        // Texels whose grown squares meet the box along x: the first and last candidates
        // may fall outside it, the next ones in never do.
        let mut first = lo[0].floor() as i64 - 2;
        if ((first + 2) as f64) < lo[0] {
            first += 1;
        }
        let mut last = hi[0].floor() as i64 + 2;
        if ((last - 1) as f64) > hi[0] {
            last -= 1;
        }
        if first > last {
            return;
        }
        // With finite edges and coordinates far from overflow, an edge's lowest corner value
        // comes from its lowest x and y terms, since each rounding step is monotone.
        let lowest = lo.iter().chain(&hi).all(|c| c.abs() < 1e12)
            && edges.iter().flatten().all(|v| v.is_finite());
        for y in (lo[1].floor() as i64 - 2)..=(hi[1].floor() as i64 + 2) {
            let (y0, y1) = ((y - 1) as f64, (y + 2) as f64);
            if y1 < lo[1] || y0 > hi[1] {
                continue;
            }
            // A texel is visited unless an edge has all of its square's corners outside.
            // Along the row, each edge rejects a run of x at one end, so the visited texels
            // form one run.
            let mut run = Some((first, last));
            for e in &edges {
                let ey = (e[1] * y0).min(e[1] * y1);
                let dx = if e[0] > 0.0 { -1 } else { 2 };
                let keeps = |x: i64| {
                    if lowest {
                        e[0] * (x + dx) as f64 + ey + e[2] <= 0.0
                    } else {
                        let (x0, x1) = ((x - 1) as f64, (x + 2) as f64);
                        ![[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
                            .iter()
                            .all(|c| e[0] * c[0] + e[1] * c[1] + e[2] > 0.0)
                    }
                };
                // Where the edge's line crosses the lowest corners.
                let guess = -(dx as f64) - (ey + e[2]) / e[0];
                run = run.and_then(|r| narrow(r, guess, keeps));
            }
            if let Some((a, b)) = run
                && !f(y, a, b)
            {
                return;
            }
        }
    }

    /// Per axis, the texel indices that hold every texel [`Self::footprint`] visits for `uv`,
    /// as one or two inclusive ranges; `None` for coordinates that are not finite.
    pub(crate) fn footprint_ranges(&self, uv: [[f32; 2]; 3]) -> Option<[Ranges; 2]> {
        let (lo, hi) = bounds(&uv.map(|t| self.texel_pos(t)));
        if !(lo.iter().chain(&hi).all(|c| c.is_finite())) {
            return None;
        }
        let size = [self.w, self.h];
        if (0..2).any(|a| hi[a] - lo[a] > 2.0 * f64::from(size[a])) {
            return Some(size.map(Ranges::all));
        }
        Some([0, 1].map(|a| {
            let (first, last) = (lo[a].floor() as i64 - 2, hi[a].floor() as i64 + 2);
            self.wrapped_range(first, last, size[a], self.wrap[a])
        }))
    }

    /// The indices `first..=last` wrap to.
    fn wrapped_range(&self, first: i64, last: i64, n: u32, wrap: Wrap) -> Ranges {
        let len = i64::from(n);
        match wrap {
            Wrap::Clamp => Ranges([[wrapped(first, n, wrap), wrapped(last, n, wrap)], [1, 0]]),
            _ if last - first + 1 >= len => Ranges::all(n),
            Wrap::Repeat => {
                let a = first.rem_euclid(len);
                let b = a + (last - first);
                if b < len {
                    Ranges([[a as usize, b as usize], [1, 0]])
                } else {
                    Ranges([[a as usize, n as usize - 1], [0, (b - len) as usize]])
                }
            }
            Wrap::Mirror if first.div_euclid(len) == last.div_euclid(len) => {
                let (a, b) = (wrapped(first, n, wrap), wrapped(last, n, wrap));
                Ranges([[a.min(b), a.max(b)], [1, 0]])
            }
            Wrap::Mirror => Ranges::all(n),
        }
    }

    /// Bilinear sample at `uv`, texel centres at half-integers.
    pub(crate) fn sample(&self, uv: [f32; 2]) -> f32 {
        let [x, y] = self.texel_pos(uv).map(|c| c - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let t = |dx, dy| self.texel(x0 + dx, y0 + dy);
        let top = t(0, 0) + (t(1, 0) - t(0, 0)) * fx;
        let bottom = t(0, 1) + (t(1, 1) - t(0, 1)) * fx;
        top + (bottom - top) * fy
    }
}

/// Which triangles of an alpha-tested section are opaque.
pub(crate) enum Opacity {
    /// Every triangle passes, or none does.
    Constant(bool),
    /// Opaque when every texel of the triangle's footprint, in uv set `uv_set`, is at or
    /// above the cutoff.
    Texture {
        mip0: Raster,
        cutoff: f32,
        uv_set: usize,
        passes: Passes,
    },
    /// The opacity is not known to the cook: every triangle is alpha-tested.
    Unknown,
}

impl Opacity {
    /// The uv set [`Self::opaque`] takes its uvs from.
    pub(crate) fn uv_set(&self) -> usize {
        match self {
            Self::Texture { uv_set, .. } => *uv_set,
            _ => 0,
        }
    }

    pub(crate) fn opaque(&self, uv: [[f32; 2]; 3]) -> bool {
        match self {
            Self::Constant(pass) => *pass,
            Self::Texture { mip0, passes, .. } => passes.footprint(mip0, uv, FAIL) & FAIL == 0,
            Self::Unknown => false,
        }
    }
}

/// A section's displacement: `scale x texel + offset` metres, within the declared range.
pub(crate) struct Displacement {
    pub scale: f32,
    pub offset: f32,
    pub min: f32,
    pub max: f32,
    /// Every mip of the texture, finest first; empty when the cook cannot read it, so
    /// clusters take the declared range.
    pub mips: Vec<Raster>,
    pub uv_set: usize,
}

impl Displacement {
    /// Displacement range over the triangles at every mip, metres.
    pub(crate) fn bounds(&self, tris: impl Iterator<Item = [[f32; 2]; 3]> + Clone) -> [f32; 2] {
        if self.mips.is_empty() {
            return [self.min, self.max];
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for m in &self.mips {
            for uv in tris.clone() {
                m.footprint(uv, |t| {
                    lo = lo.min(t);
                    hi = hi.max(t);
                    true
                });
            }
        }
        let (a, b) = (self.scale * lo + self.offset, self.scale * hi + self.offset);
        [a.min(b), a.max(b)]
    }

    /// Fails when a texel of any mip lies outside the declared range.
    pub(crate) fn check_range(&self) -> Result<(), String> {
        for (i, m) in self.mips.iter().enumerate() {
            for &t in &m.texels {
                let d = self.scale * t + self.offset;
                if d < self.min || d > self.max {
                    return Err(format!(
                        "displacement {d} m at mip {i} is outside the declared range {}..{} m",
                        self.min, self.max
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footprint_runs_visit_the_texels_whose_grown_squares_meet_the_triangle() {
        // The footprint by its definition: every texel in the box, tested on its own.
        fn reference(r: &Raster, uv: [[f32; 2]; 3]) -> Vec<(usize, usize)> {
            let p = uv.map(|t| r.texel_pos(t));
            let lo = [0, 1].map(|a| p.iter().map(|q| q[a]).fold(f64::INFINITY, f64::min));
            let hi = [0, 1].map(|a| p.iter().map(|q| q[a]).fold(f64::NEG_INFINITY, f64::max));
            let (w, h) = (f64::from(r.w), f64::from(r.h));
            if hi[0] - lo[0] > 2.0 * w || hi[1] - lo[1] > 2.0 * h {
                return (0..r.h as usize)
                    .flat_map(|y| (0..r.w as usize).map(move |x| (x, y)))
                    .collect();
            }
            let edges = [0, 1, 2].map(|i| {
                let (a, b) = (p[i], p[(i + 1) % 3]);
                let n = [b[1] - a[1], a[0] - b[0]];
                let c = p[(i + 2) % 3];
                let side = n[0] * (c[0] - a[0]) + n[1] * (c[1] - a[1]);
                let s = if side > 0.0 { -1.0 } else { 1.0 };
                [s * n[0], s * n[1], -s * (n[0] * a[0] + n[1] * a[1])]
            });
            let mut out = Vec::new();
            for y in (lo[1].floor() as i64 - 2)..=(hi[1].floor() as i64 + 2) {
                for x in (lo[0].floor() as i64 - 2)..=(hi[0].floor() as i64 + 2) {
                    let (x0, x1) = ((x - 1) as f64, (x + 2) as f64);
                    let (y0, y1) = ((y - 1) as f64, (y + 2) as f64);
                    if x1 < lo[0] || x0 > hi[0] || y1 < lo[1] || y0 > hi[1] {
                        continue;
                    }
                    let corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]];
                    let outside = edges.iter().any(|e| {
                        corners
                            .iter()
                            .all(|c| e[0] * c[0] + e[1] * c[1] + e[2] > 0.0)
                    });
                    if !outside {
                        out.push((wrapped(x, r.w, r.wrap[0]), wrapped(y, r.h, r.wrap[1])));
                    }
                }
            }
            out
        }
        let (w, h) = (23u32, 17u32);
        let texels: Vec<f32> = (0..w * h)
            .map(|i| ((i % w) as f32 * 0.4).sin() * ((i / w) as f32 * 0.3).cos() + 0.3)
            .collect();
        let mut seed = 7u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed as f32 / u32::MAX as f32 * 3.0 - 1.0
        };
        let wraps = [Wrap::Repeat, Wrap::Clamp, Wrap::Mirror];
        for wrap in wraps.iter().flat_map(|&a| wraps.map(|b| [a, b])) {
            let raster = Raster {
                w,
                h,
                texels: texels.clone(),
                wrap,
                uv: [0.0; 6],
            };
            let passes = Passes::new(&raster, 0.5);
            // Coincident corners, axis-aligned edges, corners on texel corners, a span of
            // exactly two periods, then random triangles.
            let (tw, th) = (1.0 / w as f32, 1.0 / h as f32);
            let mut tris = vec![
                [[0.3, 0.4]; 3],
                [[0.3, 0.4], [0.3, 0.4], [0.6, 0.45]],
                [[0.1, 0.2], [0.5, 0.2], [0.1, 0.7]],
                [
                    [3.0 * tw, 2.0 * th],
                    [9.0 * tw, 2.0 * th],
                    [3.0 * tw, 8.0 * th],
                ],
                [[-1.0, 0.1], [1.0, 0.1], [0.0, 0.3]],
                [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            ];
            for k in 0..3000 {
                let size = [2.5, 0.5, 0.05, 0.005][k % 4];
                let o = [next(), next()];
                tris.push([0, 1, 2].map(|_| [o[0] + next() * size, o[1] + next() * size]));
            }
            for tri in tris {
                let want = reference(&raster, tri);
                let mut got = Vec::new();
                raster.footprint_runs(tri, |y, first, last| {
                    let row = wrapped(y, h, wrap[1]);
                    got.extend((first..=last).map(|x| (wrapped(x, w, wrap[0]), row)));
                    true
                });
                assert_eq!(got, want, "{wrap:?} {tri:?}");
                let walked = want.iter().fold(0, |s, &(x, y)| {
                    s | if texels[y * w as usize + x] >= 0.5 {
                        PASS
                    } else {
                        FAIL
                    }
                });
                assert_eq!(
                    passes.footprint(&raster, tri, PASS | FAIL),
                    walked,
                    "{wrap:?} {tri:?}"
                );
            }
        }
    }

    #[test]
    fn wrapped_runs_hold_exactly_the_wrapped_indices() {
        for wrap in [Wrap::Repeat, Wrap::Clamp, Wrap::Mirror] {
            for (first, last) in [(-9, -7), (-3, 2), (0, 4), (3, 9), (-12, 14), (6, 30)] {
                let mut want: Vec<usize> = (first..=last).map(|v| wrapped(v, 5, wrap)).collect();
                let mut got = Vec::new();
                wrapped_runs(first, last, 5, wrap, |a, b| {
                    got.extend(a..=b);
                    true
                });
                for v in [&mut want, &mut got] {
                    v.sort_unstable();
                    v.dedup();
                }
                assert_eq!(got, want, "{wrap:?} {first}..={last}");
            }
        }
    }
}
