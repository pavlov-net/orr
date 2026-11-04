//! Reads the FFT glare's kernel as `post/fft_glare.slang` does.

use super::fft::{BOX_TEXELS, Kernel, LEVELS, REFERENCE_F, TABLE, TEXELS_PER_PIXEL};

impl Kernel {
    /// Texels of the table per pixel at F-number `n` (reference pitch) for pixels 2^first_level
    /// reference pixels across.
    pub fn texels(n: f64, first_level: u32) -> f64 {
        f64::from(1u32 << first_level) * TEXELS_PER_PIXEL * REFERENCE_F / n
    }

    /// Energy per channel in the pixel `[dx, dy]` pixels from the source's, for pixels `u` texels
    /// across.
    pub fn energy(&self, dx: i64, dy: i64, u: f64) -> [f64; 3] {
        let x = [(dx as f64 - 0.5) * u, (dx as f64 + 0.5) * u];
        let y = [(dy as f64 - 0.5) * u, (dy as f64 + 0.5) * u];
        let t = TABLE as f64;
        let mut e = [0.0; 3];
        let cx = [x[0].max(-t), x[1].min(t)];
        let cy = [y[0].max(-t), y[1].min(t)];
        if cx[0] < cx[1] && cy[0] < cy[1] {
            let mut k = 0;
            while k + 1 < LEVELS && BOX_TEXELS * f64::from(1u32 << (k + 1)) <= u {
                k += 1;
            }
            e = self.table_box(k, cx, cy);
        }
        for s in &self.spikes {
            // The spike's chord through the box, past the table's edge.
            let mut near = t / s[0].abs().max(s[1].abs());
            let mut far = f64::INFINITY;
            let mut hit = true;
            for (d, [lo, hi]) in [(s[0], x), (s[1], y)] {
                if d == 0.0 {
                    hit &= lo < 0.0 && hi > 0.0;
                } else {
                    let (a, b) = (lo / d, hi / d);
                    near = near.max(a.min(b));
                    far = far.min(a.max(b));
                }
            }
            if hit && far > near {
                for (e, k) in e.iter_mut().zip(self.kappa) {
                    *e += k * (1.0 / near - 1.0 / far);
                }
            }
        }
        e
    }

    /// Energy in `x` by `y` (level-0 texels, inside the table) from level `k`, each texel weighted
    /// by the share of it the box covers.
    fn table_box(&self, k: usize, x: [f64; 2], y: [f64; 2]) -> [f64; 3] {
        let g = f64::from(1u32 << k);
        let n = TABLE >> k;
        let span = |[lo, hi]: [f64; 2]| {
            ((lo / g).floor() as i64..(hi / g).ceil() as i64).map(move |a| {
                let w = (hi.min((a + 1) as f64 * g) - lo.max(a as f64 * g)) / g;
                let i = if a >= 0 { a } else { -a - 1 };
                (i as usize, w)
            })
        };
        let level = &self.levels[k];
        let mut e = [0.0; 3];
        for (j, wy) in span(y) {
            for (i, wx) in span(x) {
                let p = level[j * n + i];
                for c in 0..3 {
                    e[c] += wx * wy * p[c];
                }
            }
        }
        e
    }
}
