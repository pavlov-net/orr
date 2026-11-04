//! The glare PSF as encircled energy, integrated over wavelength per Rec.2020 channel.

use std::f64::consts::PI;
use std::sync::LazyLock;

use super::psf::{BLADES, Diaphragm};
use super::spectral_weights::SPECTRAL_WEIGHTS;

/// Radial step of the dimensionless encircled-energy table, in units of lambda * N.
const EE_STEP: f64 = 0.05;
/// The table ends here; past it 1 - EE falls as tail / rho.
const EE_END: f64 = 64.0;

/// The PSF averaged over directions at radius `rho`, with enough directions to resolve the edge
/// spikes, whose width falls as 1 / rho. The PSF repeats every `sector` radians of direction.
fn azimuthal(diaphragm: &Diaphragm, sector: f64, rho: f64) -> f64 {
    let steps = (8.0 * rho * sector).ceil().max(16.0) as u32;
    let sum: f64 = (0..steps)
        .map(|i| {
            let t = sector * (f64::from(i) + 0.5) / f64::from(steps);
            diaphragm.psf([rho * t.cos(), rho * t.sin()])
        })
        .sum();
    sum / f64::from(steps)
}

/// Encircled energy at radii 0, `EE_STEP`, ... `EE_END` (units of lambda * N), and the tail
/// coefficient c with 1 - EE = c / rho past `EE_END`.
fn table() -> (Vec<f64>, f64) {
    let diaphragm = Diaphragm::new(BLADES);
    // The polygon's symmetry, with the PSF's own point symmetry for odd counts.
    let n = f64::from(BLADES);
    let sector = if BLADES.is_multiple_of(2) {
        PI / n
    } else {
        PI / (2.0 * n)
    };
    let ring = |rho: f64| 2.0 * PI * rho * azimuthal(&diaphragm, sector, rho);
    let count = (EE_END / EE_STEP).round() as usize;
    let mut ee = Vec::with_capacity(count + 1);
    ee.push(0.0);
    let mut prev = ring(0.0);
    let mut sum = 0.0;
    for i in 1..=count {
        // Simpson over [rho_{i-1}, rho_i] with the midpoint.
        let mid = ring((i as f64 - 0.5) * EE_STEP);
        let next = ring(i as f64 * EE_STEP);
        sum += EE_STEP / 6.0 * (prev + 4.0 * mid + next);
        ee.push(sum);
        prev = next;
    }
    // The tail's mean over the last half of the table, where 1 - EE oscillates around c / rho.
    let half = count / 2;
    let tail = (half..=count)
        .map(|i| (1.0 - ee[i]) * i as f64 * EE_STEP)
        .sum::<f64>()
        / (count - half + 1) as f64;
    (ee, tail)
}

/// Dimensionless encircled energy at `rho`.
fn at(rho: f64) -> f64 {
    static TABLE: LazyLock<(Vec<f64>, f64)> = LazyLock::new(table);
    let (ee, tail) = &*TABLE;
    if rho >= EE_END {
        return 1.0 - tail / rho;
    }
    let t = rho.max(0.0) / EE_STEP;
    let i = t as usize;
    let f = t - i as f64;
    ee[i] * (1.0 - f) + ee[i + 1] * f
}

/// Encircled energy per Rec.2020 channel within `radius` pixels, for a lens at `f_number` with
/// pixels `pitch_um` across.
pub fn encircled_rgb(radius: f64, f_number: f64, pitch_um: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    for &(lambda, w) in &SPECTRAL_WEIGHTS {
        let e = at(radius * pitch_um / (lambda * f_number));
        for c in 0..3 {
            out[c] += w[c] * e;
        }
    }
    out
}
