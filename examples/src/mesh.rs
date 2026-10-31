//! Meshes the examples cook, built from simple shapes.

use orr::orr_pack::Span;
use orr_cook::{MeshInput, MeshParts};
use orr_math::{Vec2, Vec3};

/// Triangles per material section, each vertex in a rigid part.
#[derive(Default)]
pub struct MeshBuilder {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    sections: Vec<Vec<u32>>,
    parts: Vec<u16>,
    pivots: Vec<Vec3>,
    part: (u16, Vec3),
}

impl MeshBuilder {
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2) -> u32 {
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        self.parts.push(self.part.0);
        self.pivots.push(self.part.1);
        self.positions.len() as u32 - 1
    }

    /// Puts the shapes added from now on in rigid part `value`, pivoted at `pivot`. Shapes
    /// start in part 0, the body; a mesh with no other part cooks without parts.
    pub fn part(&mut self, value: u16, pivot: Vec3) {
        self.part = (value, pivot);
    }

    /// Two triangles over the grid cell `c`, `c + 1`, `c + w + 1`, `c + w`, counter-clockwise
    /// seen from the side the normals face.
    fn cell(&mut self, section: usize, c: u32, w: u32) {
        self.section(section)
            .extend([c, c + 1, c + w + 1, c, c + w + 1, c + w]);
    }

    fn section(&mut self, section: usize) -> &mut Vec<u32> {
        if self.sections.len() <= section {
            self.sections.resize(section + 1, Vec::new());
        }
        &mut self.sections[section]
    }

    /// A quad over `corners`, counter-clockwise seen from its front, in `n` x `n` cells.
    pub fn quad(&mut self, section: usize, corners: [Vec3; 4], n: u32) {
        let normal = (corners[1] - corners[0])
            .cross(corners[3] - corners[0])
            .normalize();
        let first = self.positions.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                let p = corners[0]
                    .lerp(corners[1], u)
                    .lerp(corners[3].lerp(corners[2], u), v);
                self.vertex(p, normal, Vec2::new(u, 1.0 - v));
            }
        }
        for j in 0..n {
            for i in 0..n {
                self.cell(section, first + j * (n + 1) + i, n + 1);
            }
        }
    }

    /// An axis-aligned box from `min` to `max`, each face in `n` x `n` cells.
    pub fn cuboid(&mut self, section: usize, min: Vec3, max: Vec3, n: u32) {
        let c = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let (a, b) = (min, max);
        let faces = [
            [
                c(b.x, a.y, b.z),
                c(b.x, a.y, a.z),
                c(b.x, b.y, a.z),
                c(b.x, b.y, b.z),
            ],
            [
                c(a.x, a.y, a.z),
                c(a.x, a.y, b.z),
                c(a.x, b.y, b.z),
                c(a.x, b.y, a.z),
            ],
            [
                c(a.x, b.y, b.z),
                c(b.x, b.y, b.z),
                c(b.x, b.y, a.z),
                c(a.x, b.y, a.z),
            ],
            [
                c(a.x, a.y, a.z),
                c(b.x, a.y, a.z),
                c(b.x, a.y, b.z),
                c(a.x, a.y, b.z),
            ],
            [
                c(a.x, a.y, b.z),
                c(b.x, a.y, b.z),
                c(b.x, b.y, b.z),
                c(a.x, b.y, b.z),
            ],
            [
                c(b.x, a.y, a.z),
                c(a.x, a.y, a.z),
                c(a.x, b.y, a.z),
                c(b.x, b.y, a.z),
            ],
        ];
        for corners in faces {
            self.quad(section, corners, n);
        }
    }

    /// A UV sphere, `rings` from pole to pole and `2 * rings` around.
    pub fn sphere(&mut self, section: usize, center: Vec3, radius: f32, rings: u32) {
        let segments = 2 * rings;
        let first = self.positions.len() as u32;
        for j in 0..=rings {
            let theta = j as f32 / rings as f32 * std::f32::consts::PI;
            for i in 0..=segments {
                let phi = i as f32 / segments as f32 * std::f32::consts::TAU;
                let n = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    -theta.sin() * phi.sin(),
                );
                self.vertex(
                    center + n * radius,
                    n,
                    Vec2::new(i as f32 / segments as f32, j as f32 / rings as f32),
                );
            }
        }
        for j in 0..rings {
            for i in 0..segments {
                // Rows run from the top pole down, so the cell turns the other way round.
                let c = first + j * (segments + 1) + i;
                let w = segments + 1;
                self.section(section)
                    .extend([c, c + w, c + w + 1, c, c + w + 1, c + 1]);
            }
        }
    }

    /// A (2, 3) torus knot tube around `center` in the xy plane, `segments` along it and
    /// `sides` around it.
    pub fn knot(&mut self, section: usize, center: Vec3, segments: u32, sides: u32) {
        let along = |t: f32| {
            let r = (3.0 * t).cos() + 2.0;
            Vec3::new(r * (2.0 * t).cos(), r * (2.0 * t).sin(), -(3.0 * t).sin()) * 0.33 + center
        };
        let (tube, tau) = (0.13, std::f32::consts::TAU);
        let first = self.positions.len() as u32;
        for s in 0..=segments {
            let t = s as f32 / segments as f32 * tau;
            let c = along(t);
            let tangent = (along(t + 1e-3) - along(t - 1e-3)).normalize();
            let out = c - center;
            let normal = (out - tangent * out.dot(tangent)).normalize();
            let binormal = tangent.cross(normal);
            for k in 0..=sides {
                let a = k as f32 / sides as f32 * tau;
                let n = normal * a.cos() + binormal * a.sin();
                self.vertex(
                    c + n * tube,
                    n,
                    Vec2::new(s as f32 / segments as f32, k as f32 / sides as f32),
                );
            }
        }
        for s in 0..segments {
            for k in 0..sides {
                self.cell(section, first + s * (sides + 1) + k, sides + 1);
            }
        }
    }

    /// A tapered tube from `a` (radius `ra`) to `b` (radius `rb`), `segments` around and
    /// `rings` along, open at both ends.
    #[allow(clippy::too_many_arguments)]
    pub fn tube(
        &mut self,
        section: usize,
        a: Vec3,
        b: Vec3,
        ra: f32,
        rb: f32,
        segments: u32,
        rings: u32,
    ) {
        let axis = (b - a).normalize();
        let side = axis.any_orthonormal_vector();
        let up = axis.cross(side);
        let first = self.positions.len() as u32;
        for r in 0..=rings {
            let t = r as f32 / rings as f32;
            let center = a.lerp(b, t);
            let radius = ra + (rb - ra) * t;
            for s in 0..=segments {
                let phi = s as f32 / segments as f32 * std::f32::consts::TAU;
                let n = side * phi.cos() + up * phi.sin();
                self.vertex(
                    center + n * radius,
                    n,
                    Vec2::new(s as f32 / segments as f32, t * 4.0),
                );
            }
        }
        let w = segments + 1;
        for r in 0..rings {
            for s in 0..segments {
                let c = first + r * w + s;
                self.section(section)
                    .extend([c, c + 1, c + w, c + 1, c + w + 1, c + w]);
            }
        }
    }

    /// A 3 x 3 cell patch `size` across at `center`, facing `normal` and cupped away from
    /// it; leaves under an opacity mask.
    pub fn leaf(&mut self, section: usize, center: Vec3, normal: Vec3, size: f32) {
        let u = normal.any_orthonormal_vector();
        let v = normal.cross(u);
        let first = self.positions.len() as u32;
        for j in 0..4 {
            for i in 0..4 {
                let (x, y) = (i as f32 / 3.0 - 0.5, j as f32 / 3.0 - 0.5);
                let bend = 0.25 * size * (x * x + y * y);
                let p = center + (u * x + v * y) * size - normal * bend;
                self.vertex(p, normal, Vec2::new(i as f32 / 3.0, j as f32 / 3.0));
            }
        }
        for j in 0..3 {
            for i in 0..3 {
                self.cell(section, first + j * 4 + i, 4);
            }
        }
    }

    pub fn finish(self) -> MeshInput {
        let mut m = MeshInput {
            positions: self.positions,
            normals: self.normals,
            uvs: Some(self.uvs),
            ..MeshInput::default()
        };
        for tris in self.sections {
            let first = m.indices.len() as u32;
            m.indices.extend(tris);
            m.sections.push(Span {
                first,
                count: m.indices.len() as u32 - first,
            });
        }
        let count = self.parts.iter().max().map_or(0, |&p| p + 1);
        m.parts = (count > 1).then_some(MeshParts {
            values: self.parts,
            count,
            by_section: false,
            pivots: Some(self.pivots),
        });
        m
    }
}
