//! The glare PSF: the Fraunhofer diffraction pattern of a regular polygon diaphragm. Shared by the
//! build script and the output suite.

use std::f64::consts::PI;

/// Blades of the diaphragm the glare is made for.
pub const BLADES: u32 = 6;
/// The reference pixel pitch the glare is made for: a 20.25 mm sensor 2160 pixels high.
pub const REFERENCE_PITCH_UM: f64 = 20250.0 / 2160.0;

/// The diaphragm: a regular polygon of `blades` sides with the area of the unit-diameter
/// disc, so its f-number is the disc's.
pub struct Diaphragm {
    vertices: Vec<[f64; 2]>,
    area: f64,
}

impl Diaphragm {
    pub fn new(blades: u32) -> Self {
        let n = f64::from(blades);
        let area = PI / 4.0;
        let radius = (2.0 * area / (n * (2.0 * PI / n).sin())).sqrt();
        let vertices = (0..blades)
            .map(|j| {
                let a = 2.0 * PI * f64::from(j) / n;
                [radius * a.cos(), radius * a.sin()]
            })
            .collect();
        Self { vertices, area }
    }

    /// The point spread function at frequency `nu` (cycles per diameter, so image radius
    /// over lambda * N): |F|^2 / area, which integrates to 1 over the plane.
    pub fn psf(&self, nu: [f64; 2]) -> f64 {
        let q = [2.0 * PI * nu[0], 2.0 * PI * nu[1]];
        let q2 = q[0] * q[0] + q[1] * q[1];
        if q2 < 1e-12 {
            return self.area;
        }
        // Divergence theorem: F(q) = i / |q|^2 * sum over edges of (q . n) L e^{-i q . m}
        // sinc(q . d / 2), n the outward normal, m the midpoint, d the edge vector.
        let (mut re, mut im) = (0.0, 0.0);
        let n = self.vertices.len();
        for j in 0..n {
            let a = self.vertices[j];
            let b = self.vertices[(j + 1) % n];
            let d = [b[0] - a[0], b[1] - a[1]];
            let m = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
            // (q . n) L with n L = (d.y, -d.x) for counter-clockwise vertices.
            let qn = q[0] * d[1] - q[1] * d[0];
            let x = 0.5 * (q[0] * d[0] + q[1] * d[1]);
            let sinc = if x.abs() < 1e-8 { 1.0 } else { x.sin() / x };
            let phase = q[0] * m[0] + q[1] * m[1];
            let amp = qn * sinc;
            re += amp * phase.cos();
            im -= amp * phase.sin();
        }
        (re * re + im * im) / (q2 * q2) / self.area
    }

    /// Unit directions of the diffraction spikes, which run along the edges' normals: one
    /// per blade for even counts, whose opposite edges share a normal, two for odd.
    pub fn spikes(&self) -> Vec<[f64; 2]> {
        let n = self.vertices.len();
        let mut out: Vec<[f64; 2]> = Vec::new();
        for j in 0..n {
            let a = self.vertices[j];
            let b = self.vertices[(j + 1) % n];
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let len = dx.hypot(dy);
            for s in [1.0, -1.0] {
                let d =
                    [s * dy / len, -s * dx / len].map(|c| if c.abs() < 1e-12 { 0.0 } else { c });
                if !out
                    .iter()
                    .any(|o| (o[0] - d[0]).abs() + (o[1] - d[1]).abs() < 1e-9)
                {
                    out.push(d);
                }
            }
        }
        out
    }
}
