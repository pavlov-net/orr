// The glare PSF: the Fraunhofer diffraction pattern of a regular polygon diaphragm, as
// encircled energy, integrated over wavelength per Rec.2020 channel. Shared by the build
// script and the output suite (through `include!`), so it has no inner attributes or
// doc comments.

use std::f64::consts::PI;

// Radial step of the dimensionless encircled-energy table, in units of lambda * N.
pub const EE_STEP: f64 = 0.05;
// The table ends here; past it 1 - EE falls as tail / rho.
pub const EE_END: f64 = 64.0;

// The diaphragm: a regular polygon of `blades` sides with the area of the unit-diameter
// disc, so its f-number is the disc's.
pub struct Diaphragm {
    vertices: Vec<[f64; 2]>,
    area: f64,
    // The PSF repeats every `sector` radians of direction: the polygon's symmetry, with the
    // PSF's own point symmetry for odd counts.
    sector: f64,
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
        let sector = if blades.is_multiple_of(2) {
            PI / n
        } else {
            PI / (2.0 * n)
        };
        Self {
            vertices,
            area,
            sector,
        }
    }

    // The point spread function at frequency `nu` (cycles per diameter, so image radius
    // over lambda * N): |F|^2 / area, which integrates to 1 over the plane.
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

    // The PSF averaged over directions at radius `rho`, with enough directions to resolve
    // the edge spikes, whose width falls as 1 / rho.
    fn azimuthal(&self, rho: f64) -> f64 {
        let steps = (8.0 * rho * self.sector).ceil().max(16.0) as u32;
        let sum: f64 = (0..steps)
            .map(|i| {
                let t = self.sector * (f64::from(i) + 0.5) / f64::from(steps);
                self.psf([rho * t.cos(), rho * t.sin()])
            })
            .sum();
        sum / f64::from(steps)
    }

    // Encircled energy at radii 0, EE_STEP, ... EE_END (units of lambda * N), and the tail
    // coefficient c with 1 - EE = c / rho past EE_END.
    pub fn encircled(&self) -> (Vec<f64>, f64) {
        let count = (EE_END / EE_STEP).round() as usize;
        let ring = |i: usize| {
            let rho = i as f64 * EE_STEP;
            2.0 * PI * rho * self.azimuthal(rho)
        };
        let mut ee = Vec::with_capacity(count + 1);
        ee.push(0.0);
        let mut prev = ring(0);
        let mut sum = 0.0;
        for i in 1..=count {
            // Simpson over [rho_{i-1}, rho_i] with the midpoint.
            let mid = {
                let rho = (i as f64 - 0.5) * EE_STEP;
                2.0 * PI * rho * self.azimuthal(rho)
            };
            let next = ring(i);
            sum += EE_STEP / 6.0 * (prev + 4.0 * mid + next);
            ee.push(sum);
            prev = next;
        }
        // The tail's mean over the last half of the table, where 1 - EE oscillates around
        // c / rho.
        let half = count / 2;
        let tail = (half..=count)
            .map(|i| (1.0 - ee[i]) * i as f64 * EE_STEP)
            .sum::<f64>()
            / (count - half + 1) as f64;
        (ee, tail)
    }
}

// Dimensionless encircled energy at `rho` from the table and tail.
pub fn encircled_at(ee: &[f64], tail: f64, rho: f64) -> f64 {
    let end = (ee.len() - 1) as f64 * EE_STEP;
    if rho >= end {
        return 1.0 - tail / rho;
    }
    let t = rho.max(0.0) / EE_STEP;
    let i = t as usize;
    let f = t - i as f64;
    ee[i] * (1.0 - f) + ee[i + 1] * f
}

// Encircled energy per Rec.2020 channel within `radius` pixels, for a lens at `f_number`
// with pixels `pitch_um` across. `spectral` holds (wavelength in um, per-channel weight),
// each channel's weights summing to 1.
pub fn encircled_rgb(
    ee: &[f64],
    tail: f64,
    spectral: &[(f64, [f64; 3])],
    radius: f64,
    f_number: f64,
    pitch_um: f64,
) -> [f64; 3] {
    let mut out = [0.0; 3];
    for &(lambda, w) in spectral {
        let e = encircled_at(ee, tail, radius * pitch_um / (lambda * f_number));
        for c in 0..3 {
            out[c] += w[c] * e;
        }
    }
    out
}
