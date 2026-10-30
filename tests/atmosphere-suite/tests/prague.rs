#![allow(clippy::print_stderr)]

//! The sky against the Prague sky model's fitted ground-level dataset at its own nodes: a
//! plausibility gate, since Prague's medium (layered aerosol profiles) is not this one.
//! Both skies are lit by the same sun, ground albedo 0, eye at the ground, no night
//! background. Each preset's visibility is its ground extinction at 550 nm by Koschmieder
//! (`ln 50 / extinction`, a 2% contrast threshold), an assumption: Prague does not state its
//! definition.

use atmosphere_suite::prague::Prague;
use atmosphere_suite::{Rig, dir, sky_radiance};
use glam::{DVec3, Vec3};
use orr_atmosphere::colour::{REC2020_TO_XYZ, delta_uv, mul};
use orr_atmosphere::reference::{Channels, Geometry, Light, Precision, Reference};
use orr_atmosphere::spectrum::{SOLAR_COLOUR, SOLAR_ILLUMINANCE, SPECTRAL};
use orr_atmosphere::{Aerosol, Atmosphere};
use orr_scene::testing::assert_clean;

const EYE: DVec3 = DVec3::new(0.0, 2.0, 0.0);
/// `SceneRoot::near_range` by default.
const NEAR_RANGE: f64 = 256.0;
/// Prague's solar-elevation nodes the comparison runs at, degrees.
const ELEVATIONS: [f64; 6] = [-4.0, 0.0, 5.23, 14.79, 31.82, 58.46];
/// Horizon band: view elevations (degrees, standing for 0-10 deg) and azimuths from the
/// sun's (0-180 deg; both skies are symmetric about the solar meridian).
const BAND_ELEVATIONS: [f64; 5] = [1.0, 3.0, 5.0, 7.0, 9.0];
const BAND_AZIMUTH_STEP: f64 = 15.0;
/// The band leaves out the aerosol aureole around the light, degrees: there the two skies
/// compare their aerosol phase functions' forward peaks, and Prague's fit error is largest.
const AUREOLE: f64 = 15.0;
const LUMINANCE_GATE: f64 = 0.15;
const CHROMATICITY_GATE: f64 = 0.01;

fn presets() -> [(&'static str, Aerosol); 4] {
    [
        ("default", Aerosol::default()),
        ("clean_continental", Aerosol::clean_continental()),
        ("continental_average", Aerosol::continental_average()),
        ("maritime_clean", Aerosol::maritime_clean()),
    ]
}

fn atmosphere(aerosol: Aerosol) -> Atmosphere {
    Atmosphere {
        ground_albedo: Vec3::ZERO,
        aerosol,
        night_background: false,
        ..Atmosphere::earth(DVec3::ZERO, Vec3::Y)
    }
}

/// Ground-level visibility by Koschmieder from the extinction at 550 nm, km.
fn visibility(a: &Atmosphere) -> f64 {
    let k = SPECTRAL
        .wavelength
        .iter()
        .position(|w| *w == 550.0)
        .unwrap();
    let rayleigh = SPECTRAL.rayleigh[k] * f64::from(a.rayleigh.density);
    let aerosol = f64::from(a.aerosol.optical_depth_550) / f64::from(a.aerosol.scale_height);
    50f64.ln() / (rayleigh + aerosol) / 1000.0
}

/// The zenith, then the band's directions with their solid-angle weights; directions
/// within [`AUREOLE`] of `light` weigh nothing.
fn views(light: DVec3) -> Vec<(DVec3, f64)> {
    let steps = (180.0 / BAND_AZIMUTH_STEP) as usize;
    let mut v = vec![(DVec3::Y, 0.0)];
    for e in BAND_ELEVATIONS {
        for i in 0..=steps {
            let d = dir(e, i as f64 * BAND_AZIMUTH_STEP);
            let ends = if i == 0 || i == steps { 0.5 } else { 1.0 };
            let aureole = d.angle_between(light) < AUREOLE.to_radians();
            v.push((
                d,
                if aureole {
                    0.0
                } else {
                    e.to_radians().cos() * ends
                },
            ));
        }
    }
    v
}

/// Prague's frame is z up with the sun's azimuth toward +x.
fn prague_view(d: DVec3) -> DVec3 {
    DVec3::new(d.x, d.z, d.y)
}

/// The zenith's and the band's mean XYZ.
fn zenith_and_band(xyz: &[[f64; 3]], views: &[(DVec3, f64)]) -> [[f64; 3]; 2] {
    let total: f64 = views.iter().map(|v| v.1).sum();
    let band = std::array::from_fn(|c| {
        xyz.iter().zip(views).map(|(x, v)| x[c] * v.1).sum::<f64>() / total
    });
    [xyz[0], band]
}

/// Compares `sky` (linear Rec.2020 nits along each view, per atmosphere and light elevation)
/// with Prague at every preset and node; returns the worst relative luminance error and
/// Delta u'v' over the zenith and the horizon band.
fn compare(mut sky: impl FnMut(&Atmosphere, f64, &[DVec3]) -> Vec<[f64; 3]>) -> (f64, f64) {
    let prague = Prague::load();
    for e in ELEVATIONS {
        assert!(
            prague.elevations.iter().any(|n| (n - e).abs() < 1e-9),
            "{e} deg is not a Prague node"
        );
    }
    let (mut worst_luminance, mut worst_uv) = (0.0f64, 0.0f64);
    for (name, aerosol) in presets() {
        let atmosphere = atmosphere(aerosol);
        let vis = visibility(&atmosphere);
        assert!(
            (prague.visibilities[0]..=*prague.visibilities.last().unwrap()).contains(&vis),
            "{name}: visibility {vis:.1} km outside Prague's range"
        );
        for elevation in ELEVATIONS {
            let views = views(dir(elevation, 0.0));
            let dirs: Vec<DVec3> = views.iter().map(|v| v.0).collect();
            let xyz = |rgb: &[f64; 3]| mul(&REC2020_TO_XYZ, *rgb);
            let ours: Vec<_> = sky(&atmosphere, elevation, &dirs).iter().map(xyz).collect();
            let theirs: Vec<_> = dirs
                .iter()
                .map(|d| xyz(&prague.sky(prague_view(*d), elevation, vis)))
                .collect();
            let ours = zenith_and_band(&ours, &views);
            let theirs = zenith_and_band(&theirs, &views);
            let mut line = format!("{name:>19} ({vis:5.1} km), light {elevation:6.2} deg:");
            for ((label, o), p) in ["zenith", "horizon band"].iter().zip(ours).zip(theirs) {
                let luminance = o[1] / p[1] - 1.0;
                let uv = delta_uv(o, p);
                line += &format!(
                    " {label} {:.3e} cd/m^2 ({:+.1}%, u'v' {uv:.4})",
                    p[1],
                    luminance * 100.0
                );
                worst_luminance = worst_luminance.max(luminance.abs());
                worst_uv = worst_uv.max(uv);
            }
            eprintln!("{line}");
        }
    }
    eprintln!(
        "worst: luminance {:.1}%, u'v' {worst_uv:.4}",
        worst_luminance * 100.0
    );
    (worst_luminance, worst_uv)
}

/// Zenith and horizon-band luminance within 15% of Prague and chromaticity within
/// Delta u'v' 0.01, for the sky a view sees: the reference over [0, `near_range`] (where
/// the froxels would be) composed with `AtmosphereSky` beyond.
#[test]
#[ignore = "needs the Prague dataset (ORR_VALIDATION_DATA=1 tools/fetch-slang.sh); about 30 s"]
fn sky_against_prague() {
    let near = Precision {
        relative_error: 0.05,
        max_samples: 1 << 14,
    };
    let mut current: Option<(Atmosphere, Rig, Reference)> = None;
    let (luminance, uv) = compare(|atmosphere, elevation, dirs| {
        if current.as_ref().is_none_or(|c| c.0 != *atmosphere) {
            current = Some((
                *atmosphere,
                Rig::new(atmosphere),
                Reference::new(atmosphere),
            ));
        }
        let (_, rig, reference) = current.as_mut().unwrap();
        let sun = dir(elevation, 0.0);
        rig.frame(Some(sun), EYE);
        let lut = sky_radiance(rig, EYE, dirs);
        let groups = reference.run(Channels::Groups);
        let light = Light::solar(sun);
        dirs.iter()
            .zip(&lut)
            .map(|(d, lut)| {
                let s = groups.segment(EYE, *d, NEAR_RANGE, &light, Geometry::Sphere, &near);
                let inscatter = groups.rec2020(&s.inscatter.values);
                let t = groups.rec2020(&s.transmittance);
                std::array::from_fn(|c| {
                    inscatter[c] * SOLAR_ILLUMINANCE + t[c] / SOLAR_COLOUR[c] * lut[c]
                })
            })
            .collect()
    });
    assert!(
        luminance < LUMINANCE_GATE && uv < CHROMATICITY_GATE,
        "luminance off by {:.1}%, u'v' by {uv:.4}",
        luminance * 100.0
    );
    assert_clean();
}

/// The f64 reference's whole path (12 groups, multiple scattering) against Prague, which
/// separates the medium's difference from the LUTs' multiple-scattering model.
#[test]
#[ignore = "measurement, needs the Prague dataset; about 2 min: run with --ignored --nocapture"]
fn reference_against_prague() {
    let precision = Precision {
        relative_error: 0.02,
        max_samples: 1 << 20,
    };
    compare(|atmosphere, elevation, dirs| {
        let reference = Reference::new(atmosphere);
        let groups = reference.run(Channels::Groups);
        let light = Light::solar(dir(elevation, 0.0));
        dirs.iter()
            .map(|d| {
                let r = groups.sky(EYE, *d, &light, 0.0, &precision);
                groups.rec2020(&r.values).map(|c| c * SOLAR_ILLUMINANCE)
            })
            .collect()
    });
}
