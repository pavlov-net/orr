//! World positions as an integer cell plus an f32 offset, and the render origin.

use glam::{DVec3, IVec3, Vec3};

/// Render origins sit on a grid of 2^9 = 512 m, or on the cell grid when cells are larger.
pub const REBASE_GRID_LOG2: u32 = 9;

/// A world position: cell `cell` of edge 2^cell_log2 m plus `offset` inside it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldPos {
    pub cell: IVec3,
    pub offset: Vec3,
}

impl WorldPos {
    /// Splits `p` into its cell and the offset in that cell.
    pub fn new(p: DVec3, cell_log2: u32) -> Self {
        let edge = f64::from(1u32 << cell_log2);
        let cell = (p / edge).floor();
        Self {
            cell: cell.as_ivec3(),
            offset: (p - cell * edge).as_vec3(),
        }
    }

    pub fn to_dvec3(self, cell_log2: u32) -> DVec3 {
        self.cell.as_dvec3() * f64::from(1u32 << cell_log2) + self.offset.as_dvec3()
    }

    /// The translation from `from` to `self` as kernels compute it: the cell part is exact
    /// for cell deltas below 2^(24 - cell_log2) cells, so only the offsets round.
    pub fn relative_to(self, from: WorldPos, cell_log2: u32) -> Vec3 {
        ((self.cell - from.cell) << cell_log2).as_vec3() + (self.offset - from.offset)
    }
}

fn grid_log2(cell_log2: u32) -> u32 {
    REBASE_GRID_LOG2.max(cell_log2)
}

/// The render origin cell for a focus at `focus`: the focus rounded to the rebase grid.
pub(crate) fn origin_for(focus: DVec3, cell_log2: u32) -> IVec3 {
    let grid = f64::from(1u32 << grid_log2(cell_log2));
    ((focus / grid).round() * grid / f64::from(1u32 << cell_log2)).as_ivec3()
}

/// Whether `focus` is more than one grid step from the origin at `origin_cell` along any
/// axis.
pub(crate) fn needs_rebase(origin_cell: IVec3, focus: DVec3, cell_log2: u32) -> bool {
    let origin = origin_cell.as_dvec3() * f64::from(1u32 << cell_log2);
    (focus - origin).abs().max_element() > f64::from(1u32 << grid_log2(cell_log2))
}

/// `p` in the render space of the origin at `origin_cell`.
pub(crate) fn render_position(p: DVec3, origin_cell: IVec3, cell_log2: u32) -> Vec3 {
    WorldPos::new(p, cell_log2).relative_to(
        WorldPos {
            cell: origin_cell,
            offset: Vec3::ZERO,
        },
        cell_log2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulp(x: f32) -> f32 {
        let x = x.abs();
        f32::from_bits(x.to_bits() + 1) - x
    }

    /// Offsets spread over a cell, ends included.
    fn offsets(edge: f32) -> Vec<f32> {
        let mut v = vec![0.0, f32::from_bits(1), edge * 0.5, edge - ulp(edge)];
        let mut x = 0x9e37_79b9u32;
        for _ in 0..64 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            v.push((x >> 8) as f32 / (1u32 << 24) as f32 * edge);
        }
        v
    }

    #[test]
    fn cell_delta_translation_is_exact_to_32_km() {
        for cell_log2 in [6, 8, 10] {
            let edge = (1u32 << cell_log2) as f32;
            let cells = 32_768 >> cell_log2;
            let offs = offsets(edge);
            for delta in -cells..=cells {
                let to = |o: f32| WorldPos {
                    cell: IVec3::new(delta, 0, 0),
                    offset: Vec3::new(o, 0.0, 0.0),
                };
                let cell_part = ((delta << cell_log2) as f32) as i64;
                assert_eq!(cell_part, i64::from(delta) << cell_log2, "cell part rounds");
                for &a in &offs {
                    for &b in offs.iter().step_by(7) {
                        let from = WorldPos {
                            cell: IVec3::ZERO,
                            offset: Vec3::new(b, 0.0, 0.0),
                        };
                        let got = to(a).relative_to(from, cell_log2).x;
                        let exact =
                            f64::from(delta) * f64::from(edge) + f64::from(a) - f64::from(b);
                        // One rounding of the offset difference and one of the sum.
                        let bound = 0.5 * f64::from(ulp(edge)) + 0.5 * f64::from(ulp(got));
                        assert!(
                            (f64::from(got) - exact).abs() <= bound,
                            "delta {delta} cells at 2^{cell_log2} m: {got} vs {exact}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn render_space_error_near_the_camera_stays_under_the_grid_ulp() {
        // Render-space magnitude near the camera is bounded by the grid step plus the
        // near distance, so the f32 ulp table at that magnitude bounds the error wherever
        // the camera is.
        let cell_log2 = 8;
        let near = 64.0;
        let bound_magnitude = (1u32 << REBASE_GRID_LOG2) as f32 + near;
        let bound = ulp(bound_magnitude);
        assert!(
            bound <= 6.2e-5,
            "ulp table: {bound} m at {bound_magnitude} m"
        );
        let mut x = 0x1234_5678u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            f64::from(x) / f64::from(u32::MAX)
        };
        for _ in 0..20_000 {
            let camera = DVec3::new(next(), next(), next()) * 64_000.0 - 32_000.0;
            let origin = origin_for(camera, cell_log2);
            assert!(!needs_rebase(origin, camera, cell_log2));
            let p = camera + (DVec3::new(next(), next(), next()) * 2.0 - 1.0) * f64::from(near);
            let render = render_position(p, origin, cell_log2);
            let exact = p - origin.as_dvec3() * f64::from(1u32 << cell_log2);
            assert!(render.abs().max_element() <= bound_magnitude);
            let err = (render.as_dvec3() - exact).abs().max_element();
            // The split into a cell and offset rounds once, the translation once more.
            assert!(err <= f64::from(bound), "{err} m at {p}");
        }
    }
}
