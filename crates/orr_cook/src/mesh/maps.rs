//! Cooked textures a mesh cook reads: the opacity mip 0 that decides which triangles of an
//! alpha-tested section are opaque, the displacement mips that bound each cluster's
//! displacement, and a small mip of the emission texture that weighs each emissive cluster.
//!
//! A slot reads one texture or a tile set ([`Map`]). A set reads as `texture_tile` does: the cell
//! under a point's mapped uv picks a tile or nothing, and the point reads that tile clamped to its
//! own square. So a read over a triangle walks its parts, one or more to each texture it reaches.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::par::prelude::*;
use crate::texture::Wrap;
use crate::texture::tiles::Cell;
use orr_math::{Mat3, Vec3};
use rustc_hash::FxHashMap;

/// One mip of one channel, row-major, with the sampler's edge modes, the material slot's uv
/// transform (`Material::uv_transforms` form) and what it reads past 0..1 on a bounded axis.
pub(crate) struct Raster {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) texels: Arc<Vec<f32>>,
    pub(crate) wrap: [Wrap; 2],
    pub(crate) uv: [f32; 6],
    pub(crate) outside: f32,
}

/// What a material slot reads across uv space, each `T` what the cook holds of one texture.
pub(crate) enum Map<T> {
    Image(T),
    Tiles(Tiles<T>),
}

/// A tile set. A tile's `T` reads its own square's 0..1, clamped and with no uv transform: the set
/// maps uvs and finds the tile.
pub(crate) struct Tiles<T> {
    /// The slot's uv transform, in `Material::uv_transforms` form.
    pub(crate) uv: [f32; 6],
    /// By row, then column.
    pub(crate) tiles: BTreeMap<(u16, u16), T>,
    /// What reads where the set has no tile.
    pub(crate) missing: T,
}

/// The box of mapped uvs a mesh's reads stay in. Splitting and simplifying keep a triangle's corners
/// among the mesh's vertices or between them, so a triangle crossing a cell no source triangle
/// touches stays in it too, and only rounding leaves it, which it holds.
pub(crate) struct Reach {
    lo: [f64; 2],
    hi: [f64; 2],
}

impl Reach {
    /// Of a mesh with vertex uvs `uvs`, under uv transform `uv`.
    pub(crate) fn new(uv: [f32; 6], uvs: impl Iterator<Item = [f32; 2]>) -> Self {
        let (lo, hi) = uvs.fold(
            ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]),
            |(lo, hi), c| {
                (
                    [lo[0].min(c[0]), lo[1].min(c[1])],
                    [hi[0].max(c[0]), hi[1].max(c[1])],
                )
            },
        );
        if lo[0] > hi[0] {
            return Self {
                lo: [f64::INFINITY; 2],
                hi: [f64::NEG_INFINITY; 2],
            };
        }
        // Rounding scales with a mapped uv's terms, which can cancel to far less than they are.
        let size = |a: usize| f64::from(lo[a].abs().max(hi[a].abs()));
        let [a, b, c, d, e, f] = uv.map(f64::from);
        let largest = f64::max(
            (1.0 + a).abs() * size(0) + b.abs() * size(1) + c.abs(),
            d.abs() * size(0) + (1.0 + e).abs() * size(1) + f.abs(),
        )
        .max(1.0);
        let corners = [lo, [hi[0], lo[1]], [lo[0], hi[1]], hi].map(|c| mapped(uv, c));
        let side = |a: usize, pick: fn(f64, f64) -> f64| {
            corners
                .iter()
                .map(|c| c[a])
                .reduce(pick)
                .expect("four corners")
        };
        let (lo, hi) = (
            [0, 1].map(|a| side(a, f64::min)),
            [0, 1].map(|a| side(a, f64::max)),
        );
        let slack = 16.0 * f64::from(f32::EPSILON) * largest;
        Self {
            lo: lo.map(|c| c - slack),
            hi: hi.map(|c| c + slack),
        }
    }

    /// Whether a read can land in tile `cell`, which covers u from `column` to `column + 1` and v
    /// from `-row` to `1 - row`.
    pub(crate) fn holds(&self, [column, row]: Cell) -> bool {
        let (u, v) = (f64::from(column), -f64::from(row));
        u <= self.hi[0] && u + 1.0 >= self.lo[0] && v <= self.hi[1] && v + 1.0 >= self.lo[1]
    }
}

/// The part of a triangle one texture reads.
pub(super) struct Part<'a, T> {
    pub(super) read: &'a T,
    /// Its corners in the uvs `read` takes.
    pub(super) uv: [[f32; 2]; 3],
    /// Its corners as weights of the whole triangle's corners.
    pub(super) weights: [Vec3; 3],
}

const WHOLE: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// A polygon corner: mapped u and v, then its weights of the whole triangle's corners.
type Corner = [f64; 5];

/// The parts of convex `poly` before `at` and from `at` on along `axis`. A point on the line lies
/// in the cell that starts there, so a polygon with nothing before `at` has no first part.
fn cut(poly: &[Corner], axis: usize, at: f64) -> [Vec<Corner>; 2] {
    let (mut before, mut from) = (Vec::new(), Vec::new());
    for (i, &a) in poly.iter().enumerate() {
        let b = poly[(i + 1) % poly.len()];
        let (da, db) = (a[axis] - at, b[axis] - at);
        if da <= 0.0 {
            before.push(a);
        }
        if da >= 0.0 {
            from.push(a);
        }
        if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
            let s = da / (da - db);
            let mut c: Corner = std::array::from_fn(|k| a[k] + (b[k] - a[k]) * s);
            c[axis] = at;
            before.push(c);
            from.push(c);
        }
    }
    if !before.iter().any(|c| c[axis] < at) {
        before.clear();
    }
    [before, from]
}

/// Cuts `rest` along `axis` into the unit cells of `cells`, which ascend, and what lies between
/// them, and calls `visit` with each piece and its cell, if it has one. Stops when `visit` returns
/// false, and returns whether it ran to the end.
fn sweep<C>(
    mut rest: Vec<Corner>,
    axis: usize,
    cells: impl Iterator<Item = (f64, C)>,
    visit: &mut impl FnMut(Option<C>, Vec<Corner>) -> bool,
) -> bool {
    for (at, cell) in cells {
        if rest.is_empty() {
            return true;
        }
        let [before, from] = cut(&rest, axis, at);
        if !before.is_empty() && !visit(None, before) {
            return false;
        }
        let [within, after] = cut(&from, axis, at + 1.0);
        if !within.is_empty() && !visit(Some(cell), within) {
            return false;
        }
        rest = after;
    }
    rest.is_empty() || visit(None, rest)
}

/// How near a cell's edge mapped uvs `p` count as on it: the rounding of `texture_tile`'s f32 uvs.
fn edge_slack<'a>(p: impl IntoIterator<Item = &'a f64>) -> f64 {
    let largest = p.into_iter().fold(1.0f64, |a, c| a.max(c.abs()));
    4.0 * f64::from(f32::EPSILON) * largest
}

/// How far corners `c` run along axis `a`.
fn spread<const N: usize>(c: &[[f64; N]], a: usize) -> f64 {
    let (lo, hi) = c
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| {
            (lo.min(c[a]), hi.max(c[a]))
        });
    hi - lo
}

impl<T> Tiles<T> {
    /// The tile in the cell at whole numbers `u`, `v`. Row 1 lies above row 0.
    fn tile(&self, u: f64, v: f64) -> Option<&T> {
        let (column, row) = (u, -v);
        let held = |c: f64| (0.0..65536.0).contains(&c);
        (held(column) && held(row))
            .then(|| self.tiles.get(&(row as u16, column as u16)))
            .flatten()
    }

    /// What reads in the cell at whole numbers `u`, `v`.
    fn cell(&self, u: f64, v: f64) -> &T {
        self.tile(u, v).unwrap_or(&self.missing)
    }

    /// What mapped uv `p` reads in the cell at whole numbers `cell`, and where in it: the cell's
    /// tile, or one before it if `p` lies on its lower edge and the cell has none.
    fn read(&self, p: [f64; 2], cell: [f64; 2]) -> (&T, [f32; 2]) {
        let place = [0, 1].map(|a| p[a] - cell[a]);
        let slack = edge_slack(&p);
        let edges = u8::from(place[0] < slack) | u8::from(place[1] < slack) << 1;
        for i in (0..4u8).filter(|i| i & !edges == 0) {
            let step = [f64::from(i & 1), f64::from(i >> 1)];
            if let Some(tile) = self.tile(cell[0] - step[0], cell[1] - step[1]) {
                return (tile, [0, 1].map(|a| (place[a] + step[a]) as f32));
            }
        }
        (&self.missing, place.map(|c| c as f32))
    }

    /// `texture_tile` of a point on no surface.
    fn at(&self, uv: [f32; 2]) -> (&T, [f32; 2]) {
        let p = mapped(self.uv, uv);
        self.read(p, p.map(f64::floor))
    }

    /// `texture_tile` of corner `k` of uv triangle `tri`, pointing inward at the triangle's center.
    fn corner(&self, tri: [[f32; 2]; 3], k: usize) -> CornerRead<'_, T> {
        let p = tri.map(|c| mapped(self.uv, c));
        let at = p[k];
        let slack = edge_slack(&at);
        let cell = [0, 1].map(|a| {
            let edge = at[a].round();
            let center = (p[0][a] + p[1][a] + p[2][a]) / 3.0;
            match (at[a] - edge).abs() <= slack {
                true if center < edge => edge - 1.0,
                true => edge,
                false => at[a].floor(),
            }
        });
        let (read, uv) = self.read(at, cell);
        let held = !std::ptr::eq(read, &self.missing);
        CornerRead { read, uv, held }
    }

    /// Calls `f` with the parts of mapped uv triangle `p`. A part flat along a cell's edge counts
    /// with `edges`, unless it reads no tile: there a point on the edge reads the tile beside it,
    /// and so does a part no wider than `texture_tile`'s rounding.
    fn walk<'a>(
        &'a self,
        p: [[f64; 2]; 3],
        edges: bool,
        f: &mut impl FnMut(Part<'a, T>) -> bool,
    ) -> bool {
        let cell = p[0].map(f64::floor);
        if p.iter().all(|c| c.map(f64::floor) == cell) {
            return f(Part {
                read: self.cell(cell[0], cell[1]),
                uv: p.map(|c| [0, 1].map(|a| (c[a] - cell[a]) as f32)),
                weights: WHOLE,
            });
        }
        let (e1, e2) = (
            [p[1][0] - p[0][0], p[1][1] - p[0][1]],
            [p[2][0] - p[0][0], p[2][1] - p[0][1]],
        );
        let slack = edge_slack(p.iter().flatten());
        // A triangle with no width to speak of reads along a line, and all its parts are flat.
        let line =
            e1[0] * e2[1] == e1[1] * e2[0] || spread(&p, 0) <= slack || spread(&p, 1) <= slack;
        let mut emit = |read: &'a T, origin: [f64; 2], poly: Vec<Corner>| {
            let missing = std::ptr::eq(read, &self.missing);
            let thin = if missing { slack } else { 0.0 };
            let flat = spread(&poly, 0) <= thin || spread(&poly, 1) <= thin;
            if flat && !line && (missing || !edges) {
                return true;
            }
            let last = poly.len() - 1;
            (1..last.max(2)).all(|i| {
                let corners = [poly[0], poly[i.min(last)], poly[(i + 1).min(last)]];
                f(Part {
                    read,
                    uv: corners.map(|c| [(c[0] - origin[0]) as f32, (c[1] - origin[1]) as f32]),
                    weights: corners.map(|c| Vec3::new(c[2] as f32, c[3] as f32, c[4] as f32)),
                })
            })
        };
        let poly = vec![
            [p[0][0], p[0][1], 1.0, 0.0, 0.0],
            [p[1][0], p[1][1], 0.0, 1.0, 0.0],
            [p[2][0], p[2][1], 0.0, 0.0, 1.0],
        ];
        let low = |a: usize| p.iter().map(|c| c[a]).fold(f64::INFINITY, f64::min).floor();
        let index = |c: f64| c.clamp(0.0, 65535.0) as u16;
        let (first_column, first_row) = (index(low(0)), index(-low(1)));
        let row_of = |(k, _): (&(u16, u16), &T)| k.0;
        // Rows run against v, so the sweep up v takes them from the last down.
        let rows = std::iter::successors(
            self.tiles
                .range(..=(first_row, u16::MAX))
                .next_back()
                .map(row_of),
            |&row| self.tiles.range(..(row, 0)).next_back().map(row_of),
        );
        sweep(
            poly,
            1,
            rows.map(|row| (-f64::from(row), row)),
            &mut |row, poly| {
                let Some(row) = row else {
                    let origin = [poly[0][0].floor(), poly[0][1].floor()];
                    return emit(&self.missing, origin, poly);
                };
                let v = -f64::from(row);
                let tiles = self.tiles.range((row, first_column)..=(row, u16::MAX));
                let cells = tiles.map(|(k, t)| (f64::from(k.1), (f64::from(k.1), t)));
                sweep(poly, 0, cells, &mut |tile, poly| match tile {
                    Some((u, t)) => emit(t, [u, v], poly),
                    None => emit(&self.missing, [poly[0][0].floor(), v], poly),
                })
            },
        )
    }
}

/// What a triangle's corner reads.
pub(super) struct CornerRead<'a, T> {
    pub(super) read: &'a T,
    /// The corner in the uvs `read` takes.
    pub(super) uv: [f32; 2],
    /// A texture holds it, and not what a set reads where it has none.
    pub(super) held: bool,
}

impl<T> Map<T> {
    /// What reads at mesh uv `uv`, and where in it, as `material_texture` picks it. Where a
    /// triangle holds the point, [`Self::corner`] and [`Self::parts`] know better.
    pub(super) fn at(&self, uv: [f32; 2]) -> (&T, [f32; 2]) {
        match self {
            Self::Image(t) => (t, uv),
            Self::Tiles(t) => t.at(uv),
        }
    }

    /// Corner `k` of mesh uv triangle `tri`. On a cell's edge it belongs to the triangle using it,
    /// so it reads toward that triangle and not where a lone point's floor would.
    pub(super) fn corner(&self, tri: [[f32; 2]; 3], k: usize) -> CornerRead<'_, T> {
        match self {
            Self::Image(read) => CornerRead {
                read,
                uv: tri[k],
                held: true,
            },
            Self::Tiles(t) => t.corner(tri, k),
        }
    }

    /// Calls `f` with the parts of mesh uv triangle `uv` that hold area, which is where a ray can
    /// land; stops when `f` returns false, and returns whether it ran to the end. A cell the
    /// triangle only touches, past the edge its corners sit on, has no part.
    pub(super) fn parts<'a>(
        &'a self,
        uv: [[f32; 2]; 3],
        mut f: impl FnMut(Part<'a, T>) -> bool,
    ) -> bool {
        match self {
            Self::Image(read) => f(Part {
                read,
                uv,
                weights: WHOLE,
            }),
            Self::Tiles(t) => t.walk(uv.map(|c| mapped(t.uv, c)), false, &mut f),
        }
    }

    /// [`Self::parts`] with every tile a read at a corner or along an edge can land in, as
    /// displacement's do: a point on a cell's edge reads the tile that starts there, and the
    /// runtime's f32 uvs round across an edge a corner lies within a few steps of. A cell with no
    /// tile counts only with area, since such a point reads the tile beside it.
    pub(super) fn reads<'a>(
        &'a self,
        uv: [[f32; 2]; 3],
        mut f: impl FnMut(Part<'a, T>) -> bool,
    ) -> bool {
        let Self::Tiles(t) = self else {
            return self.parts(uv, f);
        };
        let p = uv.map(|c| mapped(t.uv, c));
        let mut seen: Vec<*const T> = vec![&t.missing];
        let whole = t.walk(p, false, &mut |part| {
            seen.push(part.read);
            f(part)
        });
        let slack = edge_slack(p.iter().flatten());
        if !whole || !p.iter().flatten().any(|c| (c - c.round()).abs() <= slack) {
            return whole;
        }
        let shifted = |[x, y]: [f64; 2]| p.map(|c| [c[0] + x * slack, c[1] + y * slack]);
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
        .into_iter()
        .all(|s| {
            t.walk(shifted(s), true, &mut |part| {
                seen.contains(&std::ptr::from_ref(part.read)) || f(part)
            })
        })
    }

    /// Every texture, and what reads where a set has none.
    pub(super) fn each(&self) -> impl Iterator<Item = &T> {
        let (image, tiles, missing) = match self {
            Self::Image(t) => (Some(t), None, None),
            Self::Tiles(t) => (None, Some(t.tiles.values()), Some(&t.missing)),
        };
        image
            .into_iter()
            .chain(tiles.into_iter().flatten())
            .chain(missing)
    }

    /// The same map of what `f` makes of each texture, or `None` if it makes nothing of one.
    pub(super) fn try_map<'a, U>(
        &'a self,
        mut f: impl FnMut(&'a T) -> Option<U>,
    ) -> Option<Map<U>> {
        Some(match self {
            Self::Image(t) => Map::Image(f(t)?),
            Self::Tiles(t) => Map::Tiles(Tiles {
                uv: t.uv,
                tiles: t
                    .tiles
                    .iter()
                    .map(|(&k, t)| Some((k, f(t)?)))
                    .collect::<Option<_>>()?,
                missing: f(&t.missing)?,
            }),
        })
    }
}

impl Map<[Raster; 2]> {
    /// [`normal_frame`] of the slot's uv transform.
    pub(super) fn normal_frame(&self, v: [f32; 3]) -> [f32; 3] {
        normal_frame(
            match self {
                Self::Image(t) => t[0].uv,
                Self::Tiles(t) => t.uv,
            },
            v,
        )
    }
}

/// Mesh uv `p` under uv transform `uv`, as the texture is sampled at.
pub(super) fn mapped(uv: [f32; 6], p: [f32; 2]) -> [f64; 2] {
    let [a, b, c, d, e, f] = uv.map(f64::from);
    let (u, v) = (f64::from(p[0]), f64::from(p[1]));
    [(1.0 + a) * u + b * v + c, d * u + (1.0 + e) * v + f]
}

/// `material_normal_frame`: tangent-space normal `v` read from a normal map under uv transform
/// `uv`, turned into the mesh uv's frame by the normalized columns of the inverse transform.
fn normal_frame(uv: [f32; 6], v: [f32; 3]) -> [f32; 3] {
    let [a, b, _, d, e, _] = uv;
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

/// The side of the opacity cutoff a texel is on: at or over it.
pub(super) const PASS: u8 = 1;
/// Under it.
pub(super) const FAIL: u8 = 2;

/// Per row of an opacity mip, how many of its first `x` texels pass the cutoff, for every
/// `x` up to the row's width; and tiles of 2, 4, 8, ... texels a side, finest first, each
/// with the sides of the cutoff its texels are on (`PASS`, `FAIL` or both).
pub(crate) struct Passes {
    prefix: Vec<u16>,
    cutoff: f32,
    levels: Vec<SideLevel>,
}

struct SideLevel {
    /// Texels a side, log2.
    shift: u32,
    cols: usize,
    flags: Vec<u8>,
}

impl Passes {
    /// Tiles one range of a footprint's box may span per axis at the level that settles it.
    const SPAN_MAX: usize = 3;

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
        let (mut cols, mut rows) = (w.div_ceil(2), (mip.h as usize).div_ceil(2));
        let mut flags = vec![0u8; cols * rows];
        flags
            .par_chunks_mut(cols)
            .zip(mip.texels.par_chunks(2 * w))
            .for_each(|(out, band)| {
                for row in band.chunks_exact(w) {
                    for (x, &t) in row.iter().enumerate() {
                        out[x / 2] |= if t >= cutoff { PASS } else { FAIL };
                    }
                }
            });
        let mut levels = vec![SideLevel {
            shift: 1,
            cols,
            flags,
        }];
        while cols > 1 || rows > 1 {
            let fine = levels.last().expect("a level");
            let (c, r) = (cols.div_ceil(2), rows.div_ceil(2));
            // A quarter of the level below, which took one pass over the texels.
            let mut flags = vec![0u8; c * r];
            for (y, row) in fine.flags.chunks_exact(cols).enumerate() {
                let out = &mut flags[y / 2 * c..][..c];
                for (x, &f) in row.iter().enumerate() {
                    out[x / 2] |= f;
                }
            }
            levels.push(SideLevel {
                shift: fine.shift + 1,
                cols: c,
                flags,
            });
            (cols, rows) = (c, r);
        }
        Self {
            prefix,
            cutoff,
            levels,
        }
    }

    fn bytes(&self) -> usize {
        self.prefix.len() * size_of::<u16>()
            + self.levels.iter().map(|l| l.flags.len()).sum::<usize>()
    }

    /// `PASS` or `FAIL` when every texel in `xs` by `ys` is on that side; `None` when the
    /// tiles that hold them, at the finest level where each range spans at most
    /// [`Self::SPAN_MAX`] tiles a side, hold both.
    pub(super) fn uniform(&self, [xs, ys]: &[Ranges; 2]) -> Option<u8> {
        let ranges = |r: &Ranges| r.0.into_iter().filter(|r| r[0] <= r[1]);
        let fits = |l: &SideLevel| {
            ranges(xs)
                .chain(ranges(ys))
                .all(|r| (r[1] >> l.shift) - (r[0] >> l.shift) < Self::SPAN_MAX)
        };
        let l = self
            .levels
            .iter()
            .find(|l| fits(l))
            .unwrap_or_else(|| self.levels.last().expect("a level"));
        let mut sides = 0;
        for y in ranges(ys) {
            for x in ranges(xs) {
                for ty in y[0] >> l.shift..=y[1] >> l.shift {
                    let row = &l.flags[ty * l.cols..];
                    for &f in &row[x[0] >> l.shift..=x[1] >> l.shift] {
                        sides |= f;
                    }
                }
            }
        }
        (sides != PASS | FAIL).then_some(sides)
    }

    /// [`Self::footprint`], settled by the tiles when the box of texels that holds the
    /// footprint lies on one side.
    pub(super) fn sides(&self, mip: &Raster, uv: [[f32; 2]; 3], stop: u8) -> u8 {
        mip.footprint_ranges(uv)
            .and_then(|r| self.uniform(&r))
            .unwrap_or_else(|| self.footprint(mip, uv, stop))
    }

    /// Which sides of the cutoff the texels `mip.footprint(uv)` visits are on, as `PASS`
    /// and `FAIL` bits; stops once it has seen every side in `stop`.
    pub(super) fn footprint(&self, mip: &Raster, uv: [[f32; 2]; 3], stop: u8) -> u8 {
        let mut sides = 0;
        let stride = mip.w as usize + 1;
        let outside = if mip.outside >= self.cutoff {
            PASS
        } else {
            FAIL
        };
        mip.wrapped_footprint(uv, &mut |run| {
            sides |= match run {
                Some((y, a, b)) => {
                    let row = &self.prefix[y * stride..];
                    let pass = usize::from(row[b + 1] - row[a]);
                    (if pass > 0 { PASS } else { 0 }) | (if pass <= b - a { FAIL } else { 0 })
                }
                None => outside,
            };
            sides & stop != stop
        });
        sides
    }
}

/// Up to two inclusive index ranges; an empty one has its start past its end.
#[derive(Copy, Clone, Debug)]
pub(super) struct Ranges(pub(super) [[usize; 2]; 2]);

impl Ranges {
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

/// Index `v` of an axis of `n` texels under `wrap`; the nearest texel past a bounded edge.
fn wrapped(v: i64, n: u32, wrap: Wrap) -> usize {
    let n = i64::from(n);
    (match wrap {
        Wrap::Repeat => v.rem_euclid(n),
        Wrap::Clamp | Wrap::Border | Wrap::Constant => v.clamp(0, n - 1),
        Wrap::Mirror => {
            let m = v.rem_euclid(2 * n);
            if m < n { m } else { 2 * n - 1 - m }
        }
    }) as usize
}

/// Calls `f` with index runs that together hold exactly the indices `first..=last` wraps
/// to along an axis of `n` texels under `wrap`; stops when `f` returns false.
fn wrapped_runs(
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
    /// One texel of `value`, what a tile set reads where it has no tile.
    pub(crate) fn solid(value: f32) -> Self {
        Self {
            w: 1,
            h: 1,
            texels: Arc::new(vec![value]),
            wrap: [Wrap::Clamp; 2],
            uv: [0.0; 6],
            outside: value,
        }
    }

    /// Mesh uv `p` in texels of this mip, after the uv transform.
    pub(super) fn texel_pos(&self, p: [f32; 2]) -> [f64; 2] {
        let q = self.mapped(p);
        [q[0] * f64::from(self.w), q[1] * f64::from(self.h)]
    }

    /// Mesh uv `p` as the texture is sampled at.
    pub(super) fn mapped(&self, p: [f32; 2]) -> [f64; 2] {
        mapped(self.uv, p)
    }

    fn texel(&self, x: i64, y: i64) -> f32 {
        let past =
            |v: i64, n: u32, wrap: Wrap| wrap == Wrap::Border && !(0..i64::from(n)).contains(&v);
        if past(x, self.w, self.wrap[0]) || past(y, self.h, self.wrap[1]) {
            return self.outside;
        }
        let x = wrapped(x, self.w, self.wrap[0]);
        let y = wrapped(y, self.h, self.wrap[1]);
        self.texels[y * self.w as usize + x]
    }

    /// Calls `f` with every texel a bilinear sample anywhere on the triangle can read: the
    /// texels whose squares, grown by one texel on each side, overlap the triangle, and
    /// [`Self::outside`] where one lies past a bounded edge. Stops when `f` returns false. A
    /// triangle spanning more than two texture periods visits every texel.
    pub(super) fn footprint(&self, uv: [[f32; 2]; 3], mut f: impl FnMut(f32) -> bool) {
        let w = self.w as usize;
        self.wrapped_footprint(uv, &mut |run| match run {
            Some((y, a, b)) => self.texels[y * w..][a..=b].iter().all(|&t| f(t)),
            None => f(self.outside),
        });
    }

    /// Calls `f` with the runs of texels [`Self::footprint`] visits, each as its row and first
    /// and last texel, and with `None` for each row that lies, in part, past a bounded edge.
    /// Stops when `f` returns false.
    fn wrapped_footprint(
        &self,
        uv: [[f32; 2]; 3],
        f: &mut impl FnMut(Option<(usize, usize, usize)>) -> bool,
    ) {
        let (w, h) = (i64::from(self.w), i64::from(self.h));
        self.footprint_runs(uv, |y, first, last| {
            let bounded_x = self.wrap[0].bounded();
            let past_y = self.wrap[1].bounded() && !(0..h).contains(&y);
            if (past_y || bounded_x && (first < 0 || last >= w)) && !f(None) {
                return false;
            }
            let (first, last) = if bounded_x {
                (first.max(0), last.min(w - 1))
            } else {
                (first, last)
            };
            if past_y || first > last {
                return true;
            }
            let row = wrapped(y, self.h, self.wrap[1]);
            wrapped_runs(first, last, self.w, self.wrap[0], |a, b| {
                f(Some((row, a, b)))
            })
        });
    }

    /// Calls `f` with the texels [`Self::footprint`] visits as runs along x, rows in order:
    /// each run's row and first and last texel, before wrapping. Stops when `f` returns
    /// false.
    fn footprint_runs(&self, uv: [[f32; 2]; 3], mut f: impl FnMut(i64, i64, i64) -> bool) {
        let size = [f64::from(self.w), f64::from(self.h)];
        let p = uv.map(|t| self.texel_pos(t));
        let (lo, hi) = bounds(&p);
        if (0..2).any(|a| hi[a] - lo[a] > 2.0 * size[a]) {
            let [gx, gy] = self.wrap.map(|w| i64::from(w.bounded()));
            (-gy..i64::from(self.h) + gy).all(|y| f(y, -gx, i64::from(self.w) - 1 + gx));
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
    /// as one or two inclusive ranges; `None` for coordinates that are not finite, and for a
    /// footprint that may reach past a bounded edge, which no texels hold.
    pub(super) fn footprint_ranges(&self, uv: [[f32; 2]; 3]) -> Option<[Ranges; 2]> {
        let (lo, hi) = bounds(&uv.map(|t| self.texel_pos(t)));
        if !(lo.iter().chain(&hi).all(|c| c.is_finite())) {
            return None;
        }
        let size = [self.w, self.h];
        let bounded = self.wrap.map(Wrap::bounded);
        if (0..2).any(|a| hi[a] - lo[a] > 2.0 * f64::from(size[a])) {
            return (bounded == [false; 2]).then(|| size.map(Ranges::all));
        }
        let span = [0, 1].map(|a| (lo[a].floor() as i64 - 2, hi[a].floor() as i64 + 2));
        let past = |a: usize| span[a].0 < 0 || span[a].1 >= i64::from(size[a]);
        if (0..2).any(|a| bounded[a] && past(a)) {
            return None;
        }
        Some([0, 1].map(|a| self.wrapped_range(span[a].0, span[a].1, size[a], self.wrap[a])))
    }

    /// The indices `first..=last` wrap to.
    fn wrapped_range(&self, first: i64, last: i64, n: u32, wrap: Wrap) -> Ranges {
        let len = i64::from(n);
        match wrap {
            Wrap::Clamp | Wrap::Border | Wrap::Constant => {
                Ranges([[wrapped(first, n, wrap), wrapped(last, n, wrap)], [1, 0]])
            }
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

    /// Bilinear sample at `uv`, texel centers at half-integers, as `material_texture` reads it.
    pub(super) fn sample(&self, uv: [f32; 2]) -> f32 {
        let at = self.mapped(uv);
        let past = |a: usize| self.wrap[a] == Wrap::Constant && !(0.0..=1.0).contains(&at[a]);
        if past(0) || past(1) {
            return self.outside;
        }
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

/// A cooked opacity texture's mip 0 and what its cutoff makes of it, for every section that
/// reads it.
pub(crate) struct OpacityMap {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) texels: Arc<Vec<f32>>,
    pub(crate) passes: Arc<Passes>,
}

impl OpacityMap {
    pub(crate) fn bytes(&self) -> u64 {
        (self.texels.len() * size_of::<f32>() + self.passes.bytes()) as u64
    }
}

/// A cooked normal map's mip 0, its stored x and y.
pub(crate) struct NormalMap {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) texels: [Arc<Vec<f32>>; 2],
}

impl NormalMap {
    pub(crate) fn bytes(&self) -> u64 {
        self.texels
            .iter()
            .map(|t| (t.len() * size_of::<f32>()) as u64)
            .sum()
    }
}

/// An opacity texture's mip 0 under a section's cutoff.
pub(crate) struct OpacityRead {
    pub(crate) mip0: Raster,
    pub(crate) passes: Arc<Passes>,
}

impl OpacityRead {
    pub(super) fn new(mip0: Raster, cutoff: f32) -> Self {
        let passes = Arc::new(Passes::new(&mip0, cutoff));
        Self { mip0, passes }
    }

    /// What a tile set reads where it has no tile: `value`, under `cutoff`.
    pub(crate) fn solid(value: f32, cutoff: f32) -> Self {
        Self::new(Raster::solid(value), cutoff)
    }
}

/// Which triangles of an alpha-tested section are opaque.
pub(crate) enum Opacity {
    /// Every triangle passes, or none does.
    Constant(bool),
    /// Opaque when every texel of the triangle's footprint, in uv set `uv_set`, is at or
    /// above the cutoff.
    Texture {
        map: Map<OpacityRead>,
        cutoff: f32,
        uv_set: usize,
    },
    /// The opacity is not known to the cook: every triangle is alpha-tested.
    Unknown,
}

impl Opacity {
    /// The uv set [`Self::opaque`] takes its uvs from.
    pub(super) fn uv_set(&self) -> usize {
        match self {
            Self::Texture { uv_set, .. } => *uv_set,
            _ => 0,
        }
    }

    pub(super) fn opaque(&self, uv: [[f32; 2]; 3]) -> bool {
        match self {
            Self::Constant(pass) => *pass,
            Self::Texture { map, .. } => map.parts(uv, |p| {
                p.read.passes.sides(&p.read.mip0, p.uv, FAIL) & FAIL == 0
            }),
            Self::Unknown => false,
        }
    }
}

/// A cooked emission texture's luminance at the mip the mesh cook reads, for every section
/// that reads it.
pub(crate) struct EmissionMap {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) texels: Arc<Vec<f32>>,
}

impl EmissionMap {
    /// The largest side of the mip read: selection weights need the texture's broad shape.
    pub(crate) const SIDE_MAX: u32 = 512;

    pub(crate) fn bytes(&self) -> u64 {
        (self.texels.len() * size_of::<f32>()) as u64
    }
}

/// A section's emission texture or tile set, as luminance relative to the material's emission
/// factor.
pub(crate) struct Emission {
    pub(crate) map: Map<Raster>,
    pub(crate) uv_set: usize,
}

impl Emission {
    /// The mean of what a triangle at `uv` reads: each part's texels, by its share of the
    /// triangle. A triangle reading along a line has parts of no area, which count alike.
    pub(super) fn mean(&self, uv: [[f32; 2]; 3]) -> f64 {
        let mut parts = Vec::new();
        self.map.parts(uv, |p| {
            let (mut sum, mut count) = (0.0, 0u32);
            p.read.footprint(p.uv, |t| {
                sum += f64::from(t);
                count += 1;
                true
            });
            let [a, b, c] = p.weights;
            let share = f64::from(Mat3::from_cols(a, b, c).determinant().abs());
            parts.push((share, sum / f64::from(count.max(1))));
            true
        });
        let whole: f64 = parts.iter().map(|p| p.0).sum();
        if whole > 0.0 {
            parts.iter().map(|(share, mean)| share * mean).sum::<f64>() / whole
        } else {
            parts.iter().map(|p| p.1).sum::<f64>() / parts.len().max(1) as f64
        }
    }
}

/// A section's displacement: `scale x texel + offset` meters, within the declared range.
pub(crate) struct Displacement {
    pub(crate) scale: f32,
    pub(crate) offset: f32,
    pub(crate) min: f32,
    pub(crate) max: f32,
    /// Every mip of each texture, finest first; `None` when the cook cannot read them, so
    /// clusters take the declared range.
    pub(crate) map: Option<Map<Vec<Raster>>>,
    pub(crate) uv_set: usize,
}

impl Displacement {
    /// Displacement range over the triangles at every mip, meters.
    pub(super) fn bounds(&self, tris: impl Iterator<Item = [[f32; 2]; 3]>) -> [f32; 2] {
        let Some(map) = &self.map else {
            return [self.min, self.max];
        };
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for uv in tris {
            map.reads(uv, |p| {
                for m in p.read {
                    m.footprint(p.uv, |t| {
                        lo = lo.min(t);
                        hi = hi.max(t);
                        true
                    });
                }
                true
            });
        }
        let (a, b) = (self.scale * lo + self.offset, self.scale * hi + self.offset);
        [a.min(b), a.max(b)]
    }

    /// The displacement at corner `k` of uv triangle `tri` under `map`, meters, and whether a
    /// texture holds it.
    pub(super) fn corner(
        &self,
        map: &Map<Vec<Raster>>,
        tri: [[f32; 2]; 3],
        k: usize,
    ) -> (f32, bool) {
        let c = map.corner(tri, k);
        (self.scale * c.read[0].sample(c.uv) + self.offset, c.held)
    }

    /// The displacement of each vertex of triangles `tris` over vertex uvs `uvs`, meters, to bake
    /// into one position. A vertex takes the mean of the tiles its triangles' corners read, and
    /// what a set reads where it has no tile only when none of them reads a tile.
    pub(super) fn vertex_heights(
        &self,
        uvs: &[[f32; 2]],
        tris: impl Iterator<Item = [u32; 3]>,
    ) -> FxHashMap<u32, f32> {
        let Some(map) = &self.map else {
            return FxHashMap::default();
        };
        let mut reads: FxHashMap<u32, (f64, u32, f32)> = FxHashMap::default();
        for t in tris {
            let uv = t.map(|v| uvs[v as usize]);
            for (k, &v) in t.iter().enumerate() {
                let (height, held) = self.corner(map, uv, k);
                let e = reads.entry(v).or_insert((0.0, 0, height));
                if held {
                    e.0 += f64::from(height);
                    e.1 += 1;
                }
            }
        }
        let height = |(sum, held, missing): (f64, u32, f32)| match held {
            0 => missing,
            n => (sum / f64::from(n)) as f32,
        };
        reads.into_iter().map(|(v, r)| (v, height(r))).collect()
    }

    /// Fails when a texel of any mip lies outside the declared range.
    pub(super) fn check_range(&self) -> Result<(), String> {
        // A scale and offset fitted to the texels' range can round past the range declared for
        // the pair before it. That pair's rounding sets the error, and the fitted one can be
        // far smaller when it cancels, so the declared range sizes the slack too.
        let magnitude = self.scale.abs() + self.offset.abs() + self.min.abs() + self.max.abs();
        let slack = 4.0 * f32::EPSILON * magnitude;
        let textures = self.map.iter().flat_map(Map::each);
        for (i, m) in textures.flat_map(|mips| mips.iter().enumerate()) {
            let past = m.wrap.iter().any(|w| w.bounded()).then_some(&m.outside);
            for &t in m.texels.iter().chain(past) {
                let d = self.scale * t + self.offset;
                if d < self.min - slack || d > self.max + slack {
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

    /// A height at the top of a range ending at 0 rounds just past it in the fitted pair, whose
    /// offset and scale nearly cancel there.
    #[test]
    fn fitted_heights_stay_in_the_range_declared_before_the_fit() {
        for factor in [0.05f32, 0.1] {
            for lo in [249, 252, 254].map(|v| v as f32 / 255.0) {
                let texels = Arc::new(vec![0.0, 1.0]);
                let d = Displacement {
                    scale: factor * (1.0 - lo),
                    offset: factor * lo - factor,
                    min: -factor,
                    max: 0.0,
                    map: Some(Map::Image(vec![Raster {
                        w: 2,
                        h: 1,
                        texels,
                        wrap: [Wrap::Repeat; 2],
                        uv: [0.0; 6],
                        outside: 0.0,
                    }])),
                    uv_set: 0,
                };
                d.check_range().unwrap();
            }
        }
    }

    /// Texels at or over 0.5 everywhere, past edges that read 0.
    fn passing(wrap: Wrap) -> Raster {
        Raster {
            w: 8,
            h: 8,
            texels: Arc::new(vec![1.0; 64]),
            wrap: [wrap, Wrap::Repeat],
            uv: [0.0; 6],
            outside: 0.0,
        }
    }

    /// `material_opacity` reads the outside value past the edge, so a triangle reaching there
    /// is not opaque, one inside is, and a triangle spanning periods reaches it too.
    #[test]
    fn footprints_past_a_bounded_edge_see_what_reads_there() {
        let inside = [[0.4, 0.4], [0.6, 0.4], [0.4, 0.6]];
        let crossing = [[0.9, 0.4], [1.2, 0.4], [0.9, 0.6]];
        let beyond = [[1.5, 0.4], [1.7, 0.4], [1.5, 0.6]];
        let wide = [[-3.0, 0.1], [3.0, 0.1], [0.0, 0.3]];
        for wrap in [Wrap::Border, Wrap::Constant] {
            let r = passing(wrap);
            let passes = Passes::new(&r, 0.5);
            let sides = |uv| passes.footprint(&r, uv, PASS | FAIL);
            assert_eq!(sides(inside), PASS, "{wrap:?}");
            assert_eq!(sides(crossing), PASS | FAIL, "{wrap:?}");
            assert_eq!(sides(beyond), FAIL, "{wrap:?}");
            assert_eq!(sides(wide), PASS | FAIL, "{wrap:?}");
            assert!(r.footprint_ranges(inside).is_some(), "{wrap:?}");
            assert!(r.footprint_ranges(crossing).is_none(), "{wrap:?}");
            assert!(r.footprint_ranges(wide).is_none(), "{wrap:?}");
        }
        let r = passing(Wrap::Clamp);
        let passes = Passes::new(&r, 0.5);
        assert_eq!(passes.footprint(&r, crossing, PASS | FAIL), PASS);
    }

    /// A border filters in with the edge texels; a constant holds the edge texel up to the edge.
    #[test]
    fn samples_past_an_edge_read_as_the_runtime_does() {
        let edge = [1.0, 0.5];
        let border = passing(Wrap::Border);
        assert_eq!(border.sample([1.5, 0.5]), 0.0);
        assert_eq!(border.sample(edge), 0.5);
        let constant = passing(Wrap::Constant);
        assert_eq!(constant.sample(edge), 1.0);
        assert_eq!(constant.sample([1.001, 0.5]), 0.0);
    }

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
                texels: Arc::new(texels.clone()),
                wrap,
                uv: [0.0; 6],
                outside: 0.0,
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

    /// A set of `tiles`, each a row, a column and a value, reading `missing` elsewhere.
    fn solid_set(uv: [f32; 6], tiles: &[(u16, u16, f32)], missing: f32) -> Map<Raster> {
        let tiles = tiles.iter().map(|&(r, c, v)| ((r, c), Raster::solid(v)));
        Map::Tiles(Tiles {
            uv,
            tiles: tiles.collect(),
            missing: Raster::solid(missing),
        })
    }

    fn value(map: &Map<Raster>, uv: [f32; 2]) -> f32 {
        let (r, p) = map.at(uv);
        r.sample(p)
    }

    /// Reach holds every tile a read of the mesh can land in, and leaves out those it can't.
    #[test]
    fn a_mesh_reaches_the_cells_its_uvs_span() {
        let cells = |uv: [f32; 6], uvs: &[[f32; 2]]| {
            let reach = Reach::new(uv, uvs.iter().copied());
            let all = (0..4).flat_map(|row| (0..4).map(move |column| [column, row]));
            all.filter(|&c| reach.holds(c)).collect::<Vec<_>>()
        };
        let none = [0.0; 6];
        assert_eq!(cells(none, &[[1.25, 0.25], [1.75, 0.75]]), [[1, 0]]);
        assert_eq!(cells(none, &[[1.25, -0.75], [1.75, -0.25]]), [[1, 1]]);
        // Between two corners lie the cells they don't: an edge across a simplified surface.
        assert_eq!(
            cells(none, &[[0.5, 0.5], [2.5, 0.5]]),
            [[0, 0], [1, 0], [2, 0]]
        );
        assert_eq!(
            cells(none, &[[0.5, 0.5], [1.0, 0.5]]),
            [[0, 0], [1, 0]],
            "on an edge"
        );
        assert_eq!(
            cells([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], &[[0.5, 0.5]]),
            [[2, 0]]
        );
        assert_eq!(
            cells(
                [0.0, 0.0, -1000.0, 0.0, 0.0, 0.0],
                &[[1000.5, 0.5], [1000.9999, 0.5]]
            ),
            [[0, 0], [1, 0]],
            "rounding at the size of the uvs, not of what they map to"
        );
        assert!(cells(none, &[]).is_empty(), "a section of no triangles");
    }

    /// `texture_tile`: a floor picks the cell, rows climb up the image, and nothing lies past the
    /// tiles, below 0 or above the first row.
    #[test]
    fn a_triangle_over_two_tiles_weighs_them_by_area() {
        let emission = |tri| {
            let map = solid_set([0.0; 6], &[(0, 0, 1.0), (0, 1, 3.0)], 7.0);
            Emission { map, uv_set: 0 }.mean(tri)
        };
        // Three quarters of the triangle lie left of u = 1.
        assert!((emission([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]) - 1.5).abs() < 1e-5);
        // On a line in uv space it reads each tile it crosses alike.
        assert!((emission([[0.5, 0.5], [1.5, 0.5], [1.0, 0.5]]) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn a_point_reads_the_tile_of_the_cell_under_it() {
        let tiles = [(0, 0, 0.1), (0, 1, 0.2), (1, 0, 0.3), (0, 9, 0.4)];
        let set = solid_set([0.0; 6], &tiles, 0.9);
        for (uv, want) in [
            ([0.5, 0.5], 0.1),
            ([1.5, 0.5], 0.2),
            ([0.5, -0.5], 0.3),
            ([9.5, 0.5], 0.4),
            ([10.5, 0.5], 0.9),
            ([-0.5, 0.5], 0.9),
            ([0.5, 1.5], 0.9),
            ([0.5, -1.5], 0.9),
            ([1.0, 0.5], 0.2),
            ([0.5, 0.0], 0.1),
            ([0.5, 1.0], 0.1),
            ([2.0, 0.5], 0.2),
            ([10.0, 0.5], 0.4),
            ([2.0, 1.0], 0.2),
            ([0.5, -1.0], 0.3),
            ([70000.5, 0.5], 0.9),
        ] {
            assert_eq!(value(&set, uv), want, "{uv:?}");
        }
        let moved = solid_set([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &tiles, 0.9);
        assert_eq!(
            value(&moved, [0.5, 0.5]),
            0.2,
            "the slot's transform maps first"
        );
    }

    #[test]
    fn parts_hold_a_triangle_by_what_each_point_of_it_reads() {
        let tiles: Vec<_> = [
            (0, 0),
            (0, 1),
            (0, 4),
            (1, 1),
            (1, 2),
            (3, 2),
            (3, 3),
            (40, 7),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (r, c))| (r, c, (i + 1) as f32 / 16.0))
        .collect();
        let mut seed = 3u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed as f32 / u32::MAX as f32
        };
        for uv in [[0.0; 6], [0.5, 0.25, 0.3, -0.25, -0.5, -0.7]] {
            let set = solid_set(uv, &tiles, 0.9);
            let Map::Tiles(t) = &set else { unreachable!() };
            for k in 0..4000 {
                let size = [0.3, 2.0, 9.0][k % 3];
                let o = [next() * 12.0 - 2.0, next() * 8.0 - 6.0];
                let tri = [0, 1, 2].map(|_| [o[0] + next() * size, o[1] + next() * size]);
                let p = tri.map(|c| mapped(t.uv, c));
                let area = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
                    ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0
                };
                let mut parts = Vec::new();
                set.parts(tri, |q| {
                    parts.push((q.read.texels[0], q.weights));
                    true
                });
                let on = |w: Vec3| {
                    [0, 1].map(|a| {
                        f64::from(w.x) * p[0][a]
                            + f64::from(w.y) * p[1][a]
                            + f64::from(w.z) * p[2][a]
                    })
                };
                let total: f64 = parts
                    .iter()
                    .map(|(_, w)| area(on(w[0]), on(w[1]), on(w[2])))
                    .sum();
                let whole = area(p[0], p[1], p[2]);
                assert!((total - whole).abs() < 1e-3 * whole.max(1e-2), "{tri:?}");
                for _ in 0..8 {
                    let (a, b) = (next(), next());
                    let (a, b) = if a + b > 1.0 {
                        (1.0 - a, 1.0 - b)
                    } else {
                        (a, b)
                    };
                    let w = Vec3::new(1.0 - a - b, a, b);
                    let point = [0, 1].map(|i| w.x * tri[0][i] + w.y * tri[1][i] + w.z * tri[2][i]);
                    let want = value(&set, point);
                    // A point within rounding of a cell's edge may fall on either side of it.
                    let m = mapped(t.uv, point);
                    let edge = m.iter().any(|c| (c - c.round()).abs() < 1e-4);
                    let inside = |w3: &[Vec3; 3]| {
                        let q = [on(w3[0]), on(w3[1]), on(w3[2])];
                        let sum = area(m, q[0], q[1]) + area(m, q[1], q[2]) + area(m, q[2], q[0]);
                        (sum - area(q[0], q[1], q[2])).abs() < 1e-6
                    };
                    let held = parts.iter().filter(|(_, w)| inside(w));
                    assert!(
                        edge || held.clone().next().is_none() || held.clone().all(|h| h.0 == want),
                        "{tri:?} {point:?}"
                    );
                }
            }
        }
    }

    /// Corners on a cell's edge read the next cell, where a ray lands on nothing.
    #[test]
    fn a_cell_a_triangle_only_touches_has_no_area_to_hit() {
        let alone = solid_set([0.0; 6], &[(0, 0, 0.25)], 0.75);
        let beside = solid_set([0.0; 6], &[(0, 0, 0.25), (0, 1, 0.5)], 0.75);
        let seen = |set: &Map<Raster>, edges: bool, tri| {
            let mut seen = Vec::new();
            let mut f = |p: Part<'_, Raster>| {
                seen.push(p.read.texels[0]);
                true
            };
            match edges {
                true => set.reads(tri, &mut f),
                false => set.parts(tri, &mut f),
            };
            seen.sort_by(f32::total_cmp);
            seen.dedup();
            seen
        };
        let unit = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(seen(&alone, false, unit), [0.25]);
        assert_eq!(
            seen(&alone, true, unit),
            [0.25],
            "no tile lies there to read"
        );
        assert_eq!(seen(&beside, false, unit), [0.25]);
        assert_eq!(seen(&beside, true, unit), [0.25, 0.5]);
        let inside = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]];
        assert_eq!(seen(&beside, true, inside), [0.25]);
        let line = [[0.5, 0.5], [1.5, 0.5], [1.5, 0.5]];
        assert_eq!(
            seen(&alone, false, line),
            [0.25, 0.75],
            "a line reads along itself"
        );
        // A step below 1, which the runtime's f32 interpolation can round up to it.
        let close = [[0.5, 0.5], [1.0 - f32::EPSILON / 2.0, 0.5], [0.5, 0.75]];
        assert_eq!(seen(&beside, false, close), [0.25]);
        assert_eq!(seen(&beside, true, close), [0.25, 0.5]);
        // A step past 1, where a page's rounding can leave a corner of a tile's last triangle.
        let over = [[0.5, 0.5], [1.0 + f32::EPSILON, 0.5], [0.5, 0.75]];
        assert_eq!(seen(&alone, false, over), [0.25]);
    }

    /// A vertex on a tile's edge reads the tile its triangles lie on, whatever lies past it. Two
    /// tiles sharing it agree on one height, and a cell with no tile never outweighs a tile.
    #[test]
    fn a_vertex_reads_the_tiles_its_triangles_lie_on() {
        let tiles = [((0, 0), 0.25), ((0, 1), 0.75)];
        let tiles = tiles.map(|(k, v)| (k, vec![Raster::solid(v)]));
        let d = Displacement {
            scale: 1.0,
            offset: 0.0,
            min: 0.0,
            max: 1.0,
            map: Some(Map::Tiles(Tiles {
                uv: [0.0; 6],
                tiles: tiles.into(),
                missing: vec![Raster::solid(0.5)],
            })),
            uv_set: 0,
        };
        // Vertex 1 at u = 1, on the edge of both tiles, and vertex 3 at u = 2, past the last.
        let uvs = [
            [0.5, 0.5],
            [1.0, 0.5],
            [1.5, 0.5],
            [2.0, 0.5],
            [2.5, 0.5],
            [2.5, 0.75],
            [0.5, 0.75],
            [1.5, 0.75],
        ];
        let height = |tris: &[[u32; 3]], v: u32| d.vertex_heights(&uvs, tris.iter().copied())[&v];
        let on_first = [0, 1, 6];
        let on_second = [1, 2, 7];
        let to_the_end = [2, 3, 7];
        let past = [3, 4, 5];
        assert_eq!(height(&[on_first], 1), 0.25, "not the tile past it");
        assert_eq!(height(&[on_second], 1), 0.75);
        assert_eq!(
            height(&[on_first, on_second], 1),
            0.5,
            "one height at a seam"
        );
        assert_eq!(height(&[to_the_end], 3), 0.75, "not the cell past it");
        assert_eq!(height(&[past], 3), 0.75, "the tile before a cell with none");
        assert_eq!(height(&[past], 4), 0.5, "all that lies on no tile");
        assert_eq!(height(&[past, to_the_end], 3), 0.75, "a tile outweighs it");
    }

    /// A gradient across `n` texels, at the cutoff 0.5 halfway, with the edge modes of a tile.
    fn gradient(n: u32) -> Raster {
        Raster {
            w: n,
            h: n,
            texels: Arc::new(
                (0..n * n)
                    .map(|i| (i % n + i / n) as f32 / (2 * n - 2) as f32)
                    .collect(),
            ),
            wrap: [Wrap::Clamp; 2],
            uv: [0.0; 6],
            outside: 0.0,
        }
    }

    fn texture(map: Map<OpacityRead>) -> Opacity {
        Opacity::Texture {
            map,
            cutoff: 0.5,
            uv_set: 0,
        }
    }

    #[test]
    fn a_tile_gives_the_opaque_verdict_of_the_same_image() {
        let image = texture(Map::Image(OpacityRead::new(gradient(32), 0.5)));
        let tiled = |missing: f32| {
            texture(Map::Tiles(Tiles {
                uv: [0.0; 6],
                tiles: [((2, 3), OpacityRead::new(gradient(32), 0.5))].into(),
                missing: OpacityRead::solid(missing, 0.5),
            }))
        };
        let (fails, passes) = (tiled(0.0), tiled(1.0));
        let mut seed = 11u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(seed >> 22) / 1024.0
        };
        let moved = |t: [[f32; 2]; 3]| t.map(|c| [c[0] + 3.0, c[1] - 2.0]);
        let mut opaque = 0;
        for _ in 0..4000 {
            let o = [next() as f32, next() as f32];
            let side = next() as f32 / 8.0;
            let tri = [o, [o[0] + side, o[1]], [o[0], o[1] + side]];
            if tri.iter().flatten().all(|c| (0.0..1.0).contains(c)) {
                let want = image.opaque(tri);
                opaque += u32::from(want);
                assert_eq!(fails.opaque(moved(tri)), want, "{tri:?}");
                assert_eq!(passes.opaque(moved(tri)), want, "{tri:?}");
            }
        }
        assert!(opaque > 100, "{opaque} opaque triangles");
        // The bright corner of the tile, reaching past it into a cell with no tile.
        let across = [[3.9, -1.1], [4.1, -1.1], [3.9, -1.05]];
        assert!(!fails.opaque(across));
        assert!(passes.opaque(across));
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
