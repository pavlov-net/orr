//! Cooked textures a mesh cook reads: the opacity mip 0 that decides which triangles of an
//! alpha-tested section are opaque, and the displacement mips that bound each cluster's
//! displacement.

use crate::texture::Wrap;

/// One mip of one channel, row-major, with the sampler's edge modes.
pub(crate) struct Raster {
    pub w: u32,
    pub h: u32,
    pub texels: Vec<f32>,
    pub wrap: [Wrap; 2],
}

impl Raster {
    fn wrapped(&self, v: i64, n: u32, wrap: Wrap) -> usize {
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

    fn texel(&self, x: i64, y: i64) -> f32 {
        let x = self.wrapped(x, self.w, self.wrap[0]);
        let y = self.wrapped(y, self.h, self.wrap[1]);
        self.texels[y * self.w as usize + x]
    }

    /// Calls `f` with every texel a bilinear sample anywhere on the triangle can read: the
    /// texels whose squares, grown by one texel on each side, overlap the triangle. Stops when
    /// `f` returns false. A triangle spanning more than two texture periods visits every
    /// texel.
    pub(crate) fn footprint(&self, uv: [[f32; 2]; 3], mut f: impl FnMut(f32) -> bool) {
        let size = [f64::from(self.w), f64::from(self.h)];
        let p = uv.map(|t| [f64::from(t[0]) * size[0], f64::from(t[1]) * size[1]]);
        let lo = [0, 1].map(|a| p.iter().map(|q| q[a]).fold(f64::INFINITY, f64::min));
        let hi = [0, 1].map(|a| p.iter().map(|q| q[a]).fold(f64::NEG_INFINITY, f64::max));
        if (0..2).any(|a| hi[a] - lo[a] > 2.0 * size[a]) {
            self.texels.iter().all(|&t| f(t));
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
        for y in (lo[1].floor() as i64 - 2)..=(hi[1].floor() as i64 + 2) {
            for x in (lo[0].floor() as i64 - 2)..=(hi[0].floor() as i64 + 2) {
                let (x0, x1) = ((x - 1) as f64, (x + 2) as f64);
                let (y0, y1) = ((y - 1) as f64, (y + 2) as f64);
                if x1 < lo[0] || x0 > hi[0] || y1 < lo[1] || y0 > hi[1] {
                    continue;
                }
                let outside = edges.iter().any(|e| {
                    [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
                        .iter()
                        .all(|c| e[0] * c[0] + e[1] * c[1] + e[2] > 0.0)
                });
                if !outside && !f(self.texel(x, y)) {
                    return;
                }
            }
        }
    }

    /// Bilinear sample at `uv`, texel centres at half-integers.
    pub(crate) fn sample(&self, uv: [f32; 2]) -> f32 {
        let x = f64::from(uv[0]) * f64::from(self.w) - 0.5;
        let y = f64::from(uv[1]) * f64::from(self.h) - 0.5;
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
    /// Opaque when every texel of the triangle's footprint is at or above the cutoff.
    Texture { mip0: Raster, cutoff: f32 },
    /// The opacity is not known to the cook: every triangle is alpha-tested.
    Unknown,
}

impl Opacity {
    pub(crate) fn opaque(&self, uv: [[f32; 2]; 3]) -> bool {
        match self {
            Self::Constant(pass) => *pass,
            Self::Texture { mip0, cutoff } => {
                let mut pass = true;
                mip0.footprint(uv, |t| {
                    pass = t >= *cutoff;
                    pass
                });
                pass
            }
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
