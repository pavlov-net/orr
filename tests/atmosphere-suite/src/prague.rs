//! The Prague sky model's ground-level dataset (Wilkie et al. 2021), read at ground albedo 0
//! and evaluated as its reference implementation does: the fitted sky radiance per
//! 40 nm channel, interpolated between visibility and solar-elevation nodes.
//!
//! The dataset and the model's source (for its solar radiance table) are test-only fetches:
//! `ORR_VALIDATION_DATA=1 tools/fetch-slang.sh` puts them under `.tools/sky`, found from
//! `ORR_TOOLS` or the nearest `.tools` above this crate.

use std::path::{Path, PathBuf};

use glam::DVec3;
use orr_atmosphere::spectrum::{SOLAR_ILLUMINANCE, SPECTRAL};

const PLANET_RADIUS: f64 = 6_378_000.0;
/// The model lifts the eye this far before evaluating it, m.
const SAFETY_ALTITUDE: f64 = 50.0;

/// The radiance part of the dataset at ground albedo 0, with the model's solar spectrum.
pub struct Prague {
    /// Ground-level visibility nodes, km.
    pub visibilities: Vec<f64>,
    /// Solar elevation nodes, degrees.
    pub elevations: Vec<f64>,
    channels: usize,
    channel_start: f64,
    channel_width: f64,
    rank: usize,
    sun_breaks: Vec<f64>,
    zenith_breaks: Vec<f64>,
    emph_breaks: Vec<f64>,
    /// Per configuration (channel fastest, then elevation, then visibility): `rank` pairs of
    /// sun and zenith coefficients, then the emphasis coefficients.
    data: Vec<f32>,
    /// The model's top-of-atmosphere solar irradiance averaged over each channel, W/m^2/nm.
    solar: Vec<f64>,
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let (head, rest) = self.0.split_at(N);
        self.0 = rest;
        head.try_into().unwrap()
    }

    fn int(&mut self) -> usize {
        usize::try_from(i32::from_le_bytes(self.take())).expect("a count")
    }

    fn double(&mut self) -> f64 {
        f64::from_le_bytes(self.take())
    }

    fn counted(&mut self) -> Vec<f64> {
        let n = self.int();
        (0..n).map(|_| self.double()).collect()
    }

    fn half(&mut self) -> f32 {
        half_to_f32(u16::from_le_bytes(self.take()))
    }

    fn skip(&mut self, n: usize) {
        self.0 = &self.0[n..];
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let mantissa = f32::from(h & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("ORR_TOOLS").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir).join("sky");
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|d| d.join(".tools/sky"))
        .find(|d| d.is_dir())
        .expect("no .tools/sky: run ORR_VALIDATION_DATA=1 tools/fetch-slang.sh")
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; run ORR_VALIDATION_DATA=1 tools/fetch-slang.sh",
            path.display()
        )
    })
}

/// `name`'s initialiser in C++ source: everything between `=` and `;`.
fn initialiser<'a>(src: &'a str, name: &str) -> &'a str {
    let start = src
        .find(&format!("double {name}"))
        .unwrap_or_else(|| panic!("{name} not in the model's source"));
    let rest = &src[start..];
    let eq = rest.find('=').unwrap();
    &rest[eq + 1..rest.find(';').unwrap()]
}

fn numbers(s: &str) -> Vec<f64> {
    s.split(|c: char| c == ',' || c == '{' || c == '}' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().unwrap_or_else(|_| panic!("not a number: {t}")))
        .collect()
}

impl Prague {
    /// Reads the dataset and the model's solar table from the fetched data.
    pub fn load() -> Self {
        let dir = data_dir();
        let source = String::from_utf8(read(&dir.join("pragueskymodel/src/PragueSkyModel.cpp")))
            .expect("UTF-8 source");
        Self::parse(&read(&dir.join("prague/PragueSkyModelGround.dat")), &source)
    }

    fn parse(bytes: &[u8], source: &str) -> Self {
        let mut r = Reader(bytes);
        let visibilities = r.counted();
        let albedos = r.counted();
        let altitudes = r.counted();
        let elevations = r.counted();
        let channels = r.int();
        let channel_start = r.double();
        let channel_width = r.double();
        let rank = r.int();
        let sun_breaks = r.counted();
        let zenith_breaks = r.counted();
        let emph_breaks = r.counted();
        assert_eq!(albedos[0], 0.0, "the first albedo node is 0");
        assert_eq!(altitudes, [0.0], "a ground-level dataset");

        let (sun, zenith, emph) = (sun_breaks.len(), zenith_breaks.len(), emph_breaks.len());
        let config_bytes = ((sun + zenith) * 2 + 8) * rank + emph * 2;
        let per_albedo = channels * elevations.len();
        let mut data =
            Vec::with_capacity(visibilities.len() * per_albedo * ((sun + zenith) * rank + emph));
        for _ in &visibilities {
            for _ in 0..per_albedo {
                for _ in 0..rank {
                    data.extend((0..sun).map(|_| r.half()));
                    let scale = r.double();
                    data.extend((0..zenith).map(|_| (f64::from(r.half()) / scale) as f32));
                }
                data.extend((0..emph).map(|_| r.half()));
            }
            r.skip(config_bytes * per_albedo * (albedos.len() - 1));
        }

        let table = numbers(initialiser(source, "SUN_RAD_TABLE[]"));
        let table_start = numbers(initialiser(source, "SUN_RAD_START"))[0];
        assert_eq!(numbers(initialiser(source, "SUN_RAD_STEP")), [1.0]);
        let sun_radius = numbers(initialiser(source, "SUN_RADIUS"))[0];
        let disc = std::f64::consts::TAU * (1.0 - sun_radius.cos());
        let solar = (0..channels)
            .map(|c| {
                let first = channel_start + c as f64 * channel_width - table_start;
                let span = &table[first as usize..(first + channel_width) as usize];
                span.iter().sum::<f64>() / span.len() as f64 * disc
            })
            .collect();

        Prague {
            visibilities,
            elevations,
            channels,
            channel_start,
            channel_width,
            rank,
            sun_breaks,
            zenith_breaks,
            emph_breaks,
            data,
            solar,
        }
    }

    /// Sky radiance seen from the ground along `view` (z up), with the sun at `elevation`
    /// degrees and azimuth 0 (toward +x), ground albedo 0: linear Rec.2020 nits as the
    /// model's sky would be under orr's sun. Each channel's radiance over the model's solar
    /// irradiance in that channel lights orr's 10 nm bins inside it; bins past the last
    /// channel take the last channel's ratio.
    pub fn sky(&self, view: DVec3, elevation: f64, visibility: f64) -> [f64; 3] {
        let p = Parameters::new(view.normalize(), elevation.to_radians());
        let mut out = [0.0; 3];
        for (k, w) in SPECTRAL.rec2020.iter().enumerate() {
            let c = ((SPECTRAL.wavelength[k] - self.channel_start) / self.channel_width) as usize;
            let c = c.min(self.channels - 1);
            let ratio = self.radiance(&p, visibility, c) / self.solar[c];
            for (o, w) in out.iter_mut().zip(w) {
                *o += ratio * w * SOLAR_ILLUMINANCE;
            }
        }
        out
    }

    fn radiance(&self, p: &Parameters, visibility: f64, channel: usize) -> f64 {
        let gamma = interpolation(p.gamma, &self.sun_breaks);
        let alpha = interpolation(
            if p.elevation < 0.0 { p.shadow } else { p.zero },
            &self.zenith_breaks,
        );
        let zero = interpolation(p.zero, &self.emph_breaks);
        let vis = interpolation(visibility, &self.visibilities);
        let elev = interpolation(p.elevation.to_degrees(), &self.elevations);
        let stride =
            (self.sun_breaks.len() + self.zenith_breaks.len()) * self.rank + self.emph_breaks.len();
        let at = |v: usize, e: usize| {
            let config = channel + self.channels * (e + self.elevations.len() * v);
            self.reconstruct(&self.data[config * stride..][..stride], gamma, alpha, zero)
        };
        blend(vis, |v| blend(elev, |e| at(v, e)))
    }

    fn reconstruct(
        &self,
        coefs: &[f32],
        gamma: (usize, f64),
        alpha: (usize, f64),
        zero: (usize, f64),
    ) -> f64 {
        let stride = self.sun_breaks.len() + self.zenith_breaks.len();
        let mut result = 0.0;
        for r in 0..self.rank {
            let sun = &coefs[r * stride..][..self.sun_breaks.len()];
            let zenith = &coefs[r * stride + self.sun_breaks.len()..][..self.zenith_breaks.len()];
            result += piecewise(sun, gamma) * piecewise(zenith, alpha);
        }
        let emph = &coefs[self.rank * stride..];
        (result * piecewise(emph, zero)).max(0.0)
    }
}

/// `value` at node `i`, blended toward node `i + 1` by `f`; node `i + 1` is read only when
/// `f` is above 0.
fn blend((i, f): (usize, f64), value: impl Fn(usize) -> f64) -> f64 {
    let low = value(i);
    if f > 0.0 {
        low + (value(i + 1) - low) * f
    } else {
        low
    }
}

fn piecewise(coefs: &[f32], at: (usize, f64)) -> f64 {
    blend(at, |i| f64::from(coefs[i]))
}

/// The break below `value` (clamped to the breaks) and the fraction toward the next one.
fn interpolation(value: f64, breaks: &[f64]) -> (usize, f64) {
    let v = value.clamp(breaks[0], breaks[breaks.len() - 1]);
    let next = 1 + breaks[1..].partition_point(|b| *b <= v);
    if next == breaks.len() {
        (next - 1, 0.0)
    } else {
        (
            next - 1,
            (v - breaks[next - 1]) / (breaks[next] - breaks[next - 1]),
        )
    }
}

/// The model's angles for an eye at the ground, as its `computeParameters` gives them.
struct Parameters {
    elevation: f64,
    gamma: f64,
    shadow: f64,
    /// Zenith angle of the view corrected for the lifted eye.
    zero: f64,
}

impl Parameters {
    fn new(view: DVec3, elevation: f64) -> Self {
        let sun = DVec3::new(elevation.cos(), 0.0, elevation.sin());
        let distance = PLANET_RADIUS + SAFETY_ALTITUDE;
        let correction = (distance * distance - PLANET_RADIUS * PLANET_RADIUS).sqrt() / distance;
        let corrected = (view + DVec3::Z * correction).normalize();
        let shadow_angle = elevation + std::f64::consts::FRAC_PI_2;
        let shadow = DVec3::new(shadow_angle.cos(), 0.0, shadow_angle.sin());
        Parameters {
            elevation,
            gamma: view.dot(sun).clamp(-1.0, 1.0).acos(),
            shadow: corrected.dot(shadow).clamp(-1.0, 1.0).acos(),
            zero: corrected.z.clamp(-1.0, 1.0).acos(),
        }
    }
}
