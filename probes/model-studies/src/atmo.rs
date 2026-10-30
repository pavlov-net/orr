//! Atmosphere model studies (atmosphere.md A1 and section 6).
//!
//! One transfer model, run either per wavelength (48 bins, 360-830 nm) or per channel
//! with band coefficients: single scattering by quadrature along the view ray, sun
//! transmittance from column tables (Bruneton r, mu mapping), multiple scattering from
//! Hillaire's isotropic transfer LUT per channel, ground albedo 0.1. Solar spectrum,
//! ozone cross sections and CIE 1931 come from the fetched Bruneton sources in .tools/.

use rayon::prelude::*;
use std::f64::consts::PI;

pub const R: f64 = 6_360_000.0;
pub const RT: f64 = 6_460_000.0;
const H_RAY: f64 = 8000.0;
const ALBEDO: f64 = 0.1;
const DL: f64 = 10.0; // nm per bin
pub const NL: usize = 48;

pub struct Spectra {
    pub lambda: [f64; NL],
    pub solar: [f64; NL],     // W m^-2 nm^-1, top of atmosphere
    pub ozone_xs: [f64; NL],  // m^2
    pub cmf: [[f64; 3]; NL],  // CIE 1931 2 deg, averaged over each 10 nm bin
}

fn parse_array(src: &str, name: &str) -> Vec<f64> {
    let i = src.find(name).unwrap_or_else(|| panic!("{name} not found"));
    let open = i + src[i..].find('{').unwrap();
    let close = open + src[open..].find('}').unwrap();
    src[open + 1..close]
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().unwrap())
        .collect()
}

impl Spectra {
    pub fn load() -> Self {
        let root = std::env::var("SKY_DATA").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../.tools/sky/precomputed_atmospheric_scattering/atmosphere").to_string()
        });
        let demo = std::fs::read_to_string(format!("{root}/demo/demo.cc")).unwrap();
        let consts = std::fs::read_to_string(format!("{root}/constants.h")).unwrap();
        let solar = parse_array(&demo, "kSolarIrradiance[48]");
        let ozone = parse_array(&demo, "kOzoneCrossSection[48]");
        let cie = parse_array(&consts, "CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[380]");
        assert_eq!((solar.len(), ozone.len(), cie.len()), (48, 48, 380));
        let cie_at = |l: f64| {
            let row = (((l - 360.0) / 5.0).round() as usize).min(94);
            [cie[4 * row + 1], cie[4 * row + 2], cie[4 * row + 3]]
        };
        let mut s = Spectra { lambda: [0.0; NL], solar: [0.0; NL], ozone_xs: [0.0; NL], cmf: [[0.0; 3]; NL] };
        for i in 0..NL {
            let l = 360.0 + 10.0 * i as f64;
            s.lambda[i] = l;
            s.solar[i] = solar[i];
            s.ozone_xs[i] = ozone[i];
            // Each solar bin averages [l, l + 10); the CMF gets the same average.
            let (a, b, c) = (cie_at(l), cie_at((l + 5.0).min(830.0)), cie_at((l + 10.0).min(830.0)));
            s.cmf[i] = [0, 1, 2].map(|k| 0.25 * a[k] + 0.5 * b[k] + 0.25 * c[k]);
        }
        s
    }
}

pub const XYZ_TO_2020: [[f64; 3]; 3] = [[1.716651, -0.355671, -0.253366], [-0.666684, 1.616481, 0.015769], [0.017640, -0.042771, 0.942103]];
pub const R2020_TO_XYZ: [[f64; 3]; 3] = [[0.636958, 0.144617, 0.168881], [0.262700, 0.677998, 0.059302], [0.000000, 0.028073, 1.060985]];
pub const LUM2020: [f64; 3] = [0.2627, 0.6780, 0.0593];

#[derive(Clone, Copy)]
pub enum AerosolProfile {
    Exp(f64),
    /// OPAC: mixing layer [0, h_mix] with scale height z_mix, free troposphere to 12 km with
    /// 8 km, stratosphere 12-35 km uniform; vertical optical-depth fractions per layer.
    Layered { h_mix: f64, z_mix: f64, f_mix: f64, f_ft: f64, f_st: f64 },
}

impl AerosolProfile {
    /// Density with unit vertical column (1/m).
    pub fn density(&self, h: f64) -> f64 {
        match *self {
            AerosolProfile::Exp(hs) => (-h / hs).exp() / hs,
            AerosolProfile::Layered { h_mix, z_mix, f_mix, f_ft, f_st } => {
                let col = |z: f64, a: f64, b: f64| z * ((-a / z).exp() - (-b / z).exp());
                if h < 0.0 {
                    0.0
                } else if h < h_mix {
                    f_mix * (-h / z_mix).exp() / col(z_mix, 0.0, h_mix)
                } else if h < 12_000.0 {
                    f_ft * (-h / 8000.0).exp() / col(8000.0, h_mix, 12_000.0)
                } else if h < 35_000.0 {
                    f_st / 23_000.0
                } else {
                    0.0
                }
            }
        }
    }
    /// Optical-depth-weighted mean height (m).
    pub fn mean_height(&self) -> f64 {
        let (mut a, mut b) = (0.0, 0.0);
        let n = 200_000;
        for i in 0..n {
            let h = (i as f64 + 0.5) * 60_000.0 / n as f64;
            let d = self.density(h);
            a += h * d;
            b += d;
        }
        a / b
    }
}

#[derive(Clone, Copy)]
pub struct Aerosol {
    pub aod: f64,
    pub angstrom: f64,
    pub ssa: f64,
    pub g: f64,
    pub profile: AerosolProfile,
}

impl Aerosol {
    pub fn preset(name: &str) -> Self {
        let (aod, angstrom, ssa, g) = match name {
            "default" => (0.12, 1.3, 0.94, 0.7),
            "default_a0.8" => (0.12, 0.8, 0.94, 0.7),
            "clean_continental" => (0.064, 1.42, 0.972, 0.709),
            "continental_average" => (0.151, 1.42, 0.925, 0.703),
            "urban" => (0.643, 1.43, 0.817, 0.689),
            "desert" => (0.286, 0.17, 0.888, 0.729),
            "maritime_clean" => (0.096, 0.08, 0.997, 0.772),
            _ => panic!("{name}"),
        };
        Aerosol { aod, angstrom, ssa, g, profile: AerosolProfile::Exp(1200.0) }
    }
    /// OPAC Table 5 layering with Table 3's free-troposphere (0.013 over 2-12 km) and
    /// stratosphere (0.005) optical depths; the mixing layer takes the rest.
    pub fn opac_layered(&self, name: &str) -> AerosolProfile {
        let (h_mix, z_mix) = match name {
            "desert" => (6000.0, 2000.0),
            "maritime_clean" => (2000.0, 1000.0),
            _ => (2000.0, 8000.0),
        };
        let col = |z: f64, a: f64, b: f64| z * ((-a / z).exp() - (-b / z).exp());
        let ft = 0.013 * col(8000.0, h_mix, 12_000.0) / col(8000.0, 2000.0, 12_000.0);
        let st = 0.005;
        AerosolProfile::Layered { h_mix, z_mix, f_mix: (self.aod - ft - st) / self.aod, f_ft: ft / self.aod, f_st: st / self.aod }
    }
}

/// Per-channel medium coefficients and the channel's XYZ weight per unit radiance.
#[derive(Clone, Copy, Debug)]
pub struct Chan {
    pub ray: f64,     // Rayleigh scattering = extinction at sea level (1/m)
    pub ray_ext: f64, // Rayleigh extinction (may differ in the fitted band model)
    pub aer_sca: f64, // aerosol scattering optical depth (vertical)
    pub aer_ext: f64, // aerosol extinction optical depth (vertical)
    pub ozone: f64,   // ozone absorption at peak density (1/m)
    pub xyz: [f64; 3],
}

const OZONE_PEAK_DENSITY: f64 = 300.0 * 2.687e20 / 15_000.0;

pub fn spectral_chans(sp: &Spectra, a: &Aerosol) -> Vec<Chan> {
    (0..NL)
        .map(|i| {
            let um = sp.lambda[i] / 1000.0;
            let ray = 1.24062e-6 / um.powi(4);
            let tau = a.aod * (sp.lambda[i] / 550.0).powf(-a.angstrom);
            Chan {
                ray,
                ray_ext: ray,
                aer_sca: a.ssa * tau,
                aer_ext: tau,
                ozone: OZONE_PEAK_DENSITY * sp.ozone_xs[i],
                xyz: sp.cmf[i].map(|c| 683.0 * sp.solar[i] * c * DL),
            }
        })
        .collect()
}

/// Rec.2020 channel responses times the solar spectrum (per bin).
fn band_weights(sp: &Spectra) -> [[f64; NL]; 3] {
    let mut w = [[0.0; NL]; 3];
    for i in 0..NL {
        for c in 0..3 {
            let rbar: f64 = (0..3).map(|j| XYZ_TO_2020[c][j] * sp.cmf[i][j]).sum();
            w[c][i] = sp.solar[i] * rbar;
        }
    }
    w
}

/// Three Rec.2020 bands. Scattering is the solar-weighted mean; extinction is the same
/// mean, or with `fit` the effective value whose exp(-m tau) matches the sun-weighted
/// spectral transmittance over air masses 1-40 (per component).
pub fn band_chans(sp: &Spectra, a: &Aerosol, fit: bool) -> Vec<Chan> {
    let spec = spectral_chans(sp, a);
    let w = band_weights(sp);
    (0..3)
        .map(|c| {
            let ws: f64 = w[c].iter().sum();
            let mean = |f: &dyn Fn(&Chan) -> f64| (0..NL).map(|i| w[c][i] * f(&spec[i])).sum::<f64>() / ws;
            let fitted = |f: &dyn Fn(&Chan) -> f64| {
                // tau_c minimising sum_m (ln T(m) + m tau_c)^2. The Rec.2020 responses
                // have negative lobes, so a channel's transmittance can reach zero or go
                // negative at high air mass; those air masses are left out.
                let (mut num, mut den) = (0.0, 0.0);
                for m in 1..=40 {
                    let m = m as f64;
                    let t = (0..NL).map(|i| w[c][i] * (-m * f(&spec[i])).exp()).sum::<f64>() / ws;
                    if t > 1e-3 {
                        num += -m * t.ln();
                        den += m * m;
                    }
                }
                num / den
            };
            let e_c = 683.0 * ws * DL;
            let xyz = [0, 1, 2].map(|j| R2020_TO_XYZ[j][c] * e_c);
            let ray = mean(&|s| s.ray);
            if fit {
                Chan {
                    ray,
                    ray_ext: fitted(&|s| s.ray * H_RAY) / H_RAY,
                    aer_sca: mean(&|s| s.aer_sca),
                    aer_ext: fitted(&|s| s.aer_ext),
                    ozone: fitted(&|s| s.ozone * 15_000.0) / 15_000.0,
                    xyz,
                }
            } else {
                Chan { ray, ray_ext: ray, aer_sca: mean(&|s| s.aer_sca), aer_ext: mean(&|s| s.aer_ext), ozone: mean(&|s| s.ozone), xyz }
            }
        })
        .collect()
}

/// K contiguous wavelength groups with solar-weighted mean coefficients.
pub fn group_chans(sp: &Spectra, a: &Aerosol, k: usize) -> Vec<Chan> {
    let spec = spectral_chans(sp, a);
    (0..k)
        .map(|g| {
            let (lo, hi) = (g * NL / k, (g + 1) * NL / k);
            let ws: f64 = (lo..hi).map(|i| sp.solar[i]).sum();
            let mean = |f: &dyn Fn(&Chan) -> f64| (lo..hi).map(|i| sp.solar[i] * f(&spec[i])).sum::<f64>() / ws;
            let mut xyz = [0.0; 3];
            for i in lo..hi {
                for j in 0..3 {
                    xyz[j] += spec[i].xyz[j];
                }
            }
            Chan { ray: mean(&|s| s.ray), ray_ext: mean(&|s| s.ray), aer_sca: mean(&|s| s.aer_sca), aer_ext: mean(&|s| s.aer_ext), ozone: mean(&|s| s.ozone), xyz }
        })
        .collect()
}

pub fn ozone_density(h: f64) -> f64 {
    (1.0 - (h - 25_000.0).abs() / 15_000.0).max(0.0)
}

fn dist_top(r: f64, mu: f64) -> f64 {
    -r * mu + (r * r * (mu * mu - 1.0) + RT * RT).max(0.0).sqrt()
}
fn hits_ground(r: f64, mu: f64) -> bool {
    mu < 0.0 && r * r * (mu * mu - 1.0) + R * R >= 0.0
}
fn dist_ground(r: f64, mu: f64) -> f64 {
    -r * mu - (r * r * (mu * mu - 1.0) + R * R).max(0.0).sqrt()
}

/// Wavelength-independent column densities to the top of the atmosphere: Rayleigh (m),
/// aerosol (unit-column density x m), ozone (m), on the Bruneton (r, mu) mapping.
pub struct Columns {
    nr: usize,
    nm: usize,
    d: Vec<[f64; 3]>,
}
impl Columns {
    pub fn new(aer: AerosolProfile) -> Self {
        let (nr, nm) = (128, 512);
        let hh = (RT * RT - R * R).sqrt();
        let d = (0..nr * nm)
            .into_par_iter()
            .map(|idx| {
                let (ir, im) = (idx / nm, idx % nm);
                let xr = ir as f64 / (nr - 1) as f64;
                let xm = im as f64 / (nm - 1) as f64;
                let rho = hh * xr;
                let r = (rho * rho + R * R).sqrt();
                let (dmin, dmax) = (RT - r, rho + hh);
                let dd = dmin + xm * (dmax - dmin);
                let mu = if dd == 0.0 { 1.0 } else { ((hh * hh - rho * rho - dd * dd) / (2.0 * r * dd)).clamp(-1.0, 1.0) };
                let len = dist_top(r, mu);
                let n = 1500;
                let mut acc = [0.0; 3];
                for i in 0..n {
                    let t = (i as f64 + 0.5) / n as f64 * len;
                    let h = (r * r + t * t + 2.0 * r * mu * t).sqrt() - R;
                    acc[0] += (-h / H_RAY).exp();
                    acc[1] += aer.density(h);
                    acc[2] += ozone_density(h);
                }
                acc.map(|v| v * len / n as f64)
            })
            .collect();
        Columns { nr, nm, d }
    }
    /// None when the ray hits the planet.
    pub fn get(&self, r: f64, mu: f64) -> Option<[f64; 3]> {
        if hits_ground(r, mu) {
            return None;
        }
        let r = r.clamp(R, RT);
        let hh = (RT * RT - R * R).sqrt();
        let rho = (r * r - R * R).max(0.0).sqrt();
        let d = dist_top(r, mu);
        let (dmin, dmax) = (RT - r, rho + hh);
        let xm = if dmax > dmin { ((d - dmin) / (dmax - dmin)).clamp(0.0, 1.0) } else { 0.0 };
        let xr = rho / hh;
        let fr = xr * (self.nr - 1) as f64;
        let fm = xm * (self.nm - 1) as f64;
        let (ir, im) = ((fr as usize).min(self.nr - 2), (fm as usize).min(self.nm - 2));
        let (ar, am) = (fr - ir as f64, fm - im as f64);
        let at = |a: usize, b: usize| self.d[a * self.nm + b];
        let mut o = [0.0; 3];
        for k in 0..3 {
            o[k] = (1.0 - ar) * ((1.0 - am) * at(ir, im)[k] + am * at(ir, im + 1)[k]) + ar * ((1.0 - am) * at(ir + 1, im)[k] + am * at(ir + 1, im + 1)[k]);
        }
        Some(o)
    }
}

pub fn rayleigh_phase(c: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + c * c)
}
pub fn cs_phase(g: f64, c: f64) -> f64 {
    3.0 / (8.0 * PI) * (1.0 - g * g) * (1.0 + c * c) / ((2.0 + g * g) * (1.0 + g * g - 2.0 * g * c).powf(1.5))
}

/// How the phase function mixes Rayleigh and aerosol.
#[derive(Clone, Copy, PartialEq)]
pub enum Phase {
    PerChannel,
    /// One mix for all channels: each component's share of sigma_s luminance (Rec.2020
    /// weights on the band coefficients).
    LuminanceShare,
}

/// How altitude and the local zenith follow from a render-space position.
#[derive(Clone, Copy, PartialEq)]
pub enum Altitude {
    Sphere,
    /// Height above the flat ground plane y = 0, zenith +y everywhere.
    LocalPlane,
}

pub struct Atmosphere {
    pub chans: Vec<Chan>,
    pub aer: Aerosol,
    pub cols: std::sync::Arc<Columns>,
    /// Psi_ms per channel on (h, mu_s), Hillaire 2020.
    ms: Vec<Vec<f64>>,
    pub phase: Phase,
}

const MS_H: usize = 32;
const MS_MU: usize = 32;

impl Atmosphere {
    pub fn new(chans: Vec<Chan>, aer: Aerosol, cols: std::sync::Arc<Columns>) -> Self {
        let mut a = Atmosphere { chans, aer, cols, ms: vec![], phase: Phase::PerChannel };
        a.ms = a.build_ms();
        a
    }

    pub fn densities(&self, h: f64) -> [f64; 3] {
        [(-h / H_RAY).exp(), self.aer.profile.density(h), ozone_density(h)]
    }
    fn tau(&self, c: &Chan, col: [f64; 3]) -> f64 {
        c.ray_ext * col[0] + c.aer_ext * col[1] + c.ozone * col[2]
    }
    /// Transmittance to space per channel from radius r with cosine mu to the zenith.
    pub fn sun_t(&self, r: f64, mu: f64) -> Vec<f64> {
        match self.cols.get(r, mu) {
            Some(col) => self.chans.iter().map(|c| (-self.tau(c, col)).exp()).collect(),
            None => vec![0.0; self.chans.len()],
        }
    }

    fn build_ms(&self) -> Vec<Vec<f64>> {
        let nc = self.chans.len();
        let dirs: Vec<[f64; 3]> = (0..64)
            .map(|i| {
                let z = 1.0 - (2.0 * i as f64 + 1.0) / 64.0;
                let phi = i as f64 * PI * (3.0 - 5f64.sqrt());
                let s = (1.0 - z * z).sqrt();
                [s * phi.cos(), s * phi.sin(), z]
            })
            .collect();
        let cells: Vec<Vec<f64>> = (0..MS_H * MS_MU)
            .into_par_iter()
            .map(|idx| {
                let (ih, im) = (idx / MS_MU, idx % MS_MU);
                let h = (RT - R) * (ih as f64 / (MS_H - 1) as f64).powi(2);
                let mu_s = -1.0 + 2.0 * im as f64 / (MS_MU - 1) as f64;
                let r = R + h.max(1.0);
                let sun = [(1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0, mu_s];
                let mut l2 = vec![0.0; nc];
                let mut fms = vec![0.0; nc];
                for d in &dirs {
                    let mu = d[2];
                    let tmax = if hits_ground(r, mu) { dist_ground(r, mu) } else { dist_top(r, mu) };
                    let n = 40;
                    let mut thr = vec![1.0; nc];
                    let mut lsum = vec![0.0; nc];
                    let mut fsum = vec![0.0; nc];
                    for i in 0..n {
                        let dt = tmax / n as f64;
                        let t = (i as f64 + 0.5) * dt;
                        let p = [d[0] * t, d[1] * t, r + d[2] * t];
                        let rr = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                        let hh = rr - R;
                        let up = [p[0] / rr, p[1] / rr, p[2] / rr];
                        let mus = up[0] * sun[0] + up[1] * sun[1] + up[2] * sun[2];
                        let dens = self.densities(hh);
                        let ts = self.sun_t(rr, mus);
                        for (k, c) in self.chans.iter().enumerate() {
                            let ss = c.ray * dens[0] + c.aer_sca * dens[1];
                            let st = c.ray_ext * dens[0] + c.aer_ext * dens[1] + c.ozone * dens[2];
                            let seg = (-st * dt).exp();
                            // Integrate source over the segment analytically.
                            let integ = if st > 0.0 { (1.0 - seg) / st } else { dt };
                            lsum[k] += thr[k] * ss * ts[k] / (4.0 * PI) * integ;
                            fsum[k] += thr[k] * ss * integ;
                            thr[k] *= seg;
                        }
                    }
                    if hits_ground(r, mu) {
                        let p = [d[0] * tmax, d[1] * tmax, r + d[2] * tmax];
                        let rr = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                        let up = [p[0] / rr, p[1] / rr, p[2] / rr];
                        let mus = up[0] * sun[0] + up[1] * sun[1] + up[2] * sun[2];
                        let ts = self.sun_t(rr, mus);
                        for k in 0..nc {
                            lsum[k] += thr[k] * ts[k] * mus.max(0.0) * ALBEDO / PI;
                        }
                    }
                    for k in 0..nc {
                        l2[k] += lsum[k] * 4.0 * PI / dirs.len() as f64 / (4.0 * PI);
                        fms[k] += fsum[k] * 4.0 * PI / dirs.len() as f64 / (4.0 * PI);
                    }
                }
                (0..nc).map(|k| l2[k] / (1.0 - fms[k])).collect()
            })
            .collect();
        (0..nc).map(|k| cells.iter().map(|c| c[k]).collect()).collect()
    }

    fn psi(&self, h: f64, mu_s: f64) -> Vec<f64> {
        let fh = ((h.max(0.0) / (RT - R)).sqrt() * (MS_H - 1) as f64).clamp(0.0, (MS_H - 1) as f64);
        let fm = ((mu_s + 1.0) * 0.5 * (MS_MU - 1) as f64).clamp(0.0, (MS_MU - 1) as f64);
        let (ih, im) = ((fh as usize).min(MS_H - 2), (fm as usize).min(MS_MU - 2));
        let (ah, am) = (fh - ih as f64, fm - im as f64);
        self.ms
            .iter()
            .map(|t| {
                let at = |a: usize, b: usize| t[a * MS_MU + b];
                (1.0 - ah) * ((1.0 - am) * at(ih, im) + am * at(ih, im + 1)) + ah * ((1.0 - am) * at(ih + 1, im) + am * at(ih + 1, im + 1))
            })
            .collect()
    }

    /// Planet frame: centre at (0, -R, 0) in render space, y up.
    fn frame(&self, p: [f64; 3], alt: Altitude) -> (f64, [f64; 3]) {
        match alt {
            Altitude::Sphere => {
                let q = [p[0], p[1] + R, p[2]];
                let rr = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                (rr - R, [q[0] / rr, q[1] / rr, q[2] / rr])
            }
            Altitude::LocalPlane => (p[1], [0.0, 1.0, 0.0]),
        }
    }

    /// In-scattered radiance (per unit top-of-atmosphere irradiance) and transmittance
    /// along a segment from `o` in direction `d` for `len` metres, in render space.
    pub fn segment(&self, o: [f64; 3], d: [f64; 3], len: f64, sun: [f64; 3], alt: Altitude, steps: usize, quadratic: bool) -> (Vec<f64>, Vec<f64>) {
        let nc = self.chans.len();
        let mut thr = vec![1.0; nc];
        let mut l = vec![0.0; nc];
        let cos = d[0] * sun[0] + d[1] * sun[1] + d[2] * sun[2];
        let (pr, pm) = (rayleigh_phase(cos), cs_phase(self.aer.g, cos));
        // Luminance shares of sigma_s at sea level-relative densities, per sample.
        for i in 0..steps {
            let (t0, t1) = if quadratic {
                (len * (i as f64 / steps as f64).powi(2), len * ((i + 1) as f64 / steps as f64).powi(2))
            } else {
                (len * i as f64 / steps as f64, len * (i + 1) as f64 / steps as f64)
            };
            let dt = t1 - t0;
            let t = 0.5 * (t0 + t1);
            let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
            let (h, up) = self.frame(p, alt);
            let mus = up[0] * sun[0] + up[1] * sun[1] + up[2] * sun[2];
            let dens = self.densities(h);
            let ts = self.sun_t(R + h.max(0.0), mus);
            let psi = self.psi(h, mus);
            let share = if self.phase == Phase::LuminanceShare {
                let lr: f64 = self.chans.iter().zip(LUM2020).map(|(c, w)| w * c.ray * dens[0]).sum();
                let la: f64 = self.chans.iter().zip(LUM2020).map(|(c, w)| w * c.aer_sca * dens[1]).sum();
                Some(la / (lr + la))
            } else {
                None
            };
            for (k, c) in self.chans.iter().enumerate() {
                let sr = c.ray * dens[0];
                let sa = c.aer_sca * dens[1];
                let st = c.ray_ext * dens[0] + c.aer_ext * dens[1] + c.ozone * dens[2];
                let ph = match share {
                    Some(fa) => (sr + sa) * ((1.0 - fa) * pr + fa * pm),
                    None => sr * pr + sa * pm,
                };
                let j = ph * ts[k] + (sr + sa) * psi[k];
                let seg = (-st * dt).exp();
                let integ = if st > 0.0 { (1.0 - seg) / st } else { dt };
                l[k] += thr[k] * j * integ;
                thr[k] *= seg;
            }
        }
        (l, thr)
    }

    /// Sky radiance per channel from an eye at altitude `h_eye` (spherical planet),
    /// marched from `start` metres to the top of the atmosphere; zero below the horizon.
    pub fn sky(&self, h_eye: f64, dir: [f64; 3], sun: [f64; 3], start: f64) -> Vec<f64> {
        let r = R + h_eye;
        let mu = dir[1];
        if hits_ground(r, mu) {
            return vec![0.0; self.chans.len()];
        }
        let len = dist_top(r, mu);
        let o = [dir[0] * start, h_eye + dir[1] * start, dir[2] * start];
        self.segment(o, dir, len - start, sun, Altitude::Sphere, 256, true).0
    }

    pub fn to_xyz(&self, l: &[f64]) -> [f64; 3] {
        let mut o = [0.0; 3];
        for (k, c) in self.chans.iter().enumerate() {
            for j in 0..3 {
                o[j] += l[k] * c.xyz[j];
            }
        }
        o
    }
}

pub fn uv(xyz: [f64; 3]) -> [f64; 2] {
    let d = xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2];
    [4.0 * xyz[0] / d, 9.0 * xyz[1] / d]
}
pub fn duv(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (p, q) = (uv(a), uv(b));
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}
pub fn to_2020(xyz: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| (0..3).map(|j| XYZ_TO_2020[i][j] * xyz[j]).sum())
}
pub fn dir_from(elev_deg: f64, azim_deg: f64) -> [f64; 3] {
    let (e, a) = (elev_deg.to_radians(), azim_deg.to_radians());
    [e.cos() * a.cos(), e.sin(), e.cos() * a.sin()]
}
