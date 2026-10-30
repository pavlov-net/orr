//! The atmosphere experiments (atmosphere.md ATM probes run in A1).

use crate::atmo::*;
use std::sync::Arc;

fn rel(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-30 { 0.0 } else { (a - b) / b }
}
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| rel(*x, *y).abs()).fold(0.0, f64::max)
}

const PRESETS: [&str; 7] = ["default", "default_a0.8", "clean_continental", "continental_average", "urban", "desert", "maritime_clean"];

fn sky_dirs() -> Vec<(String, [f64; 3])> {
    let mut v = vec![("zenith".to_string(), [0.0, 1.0, 0.0])];
    for e in [2.0, 10.0, 30.0] {
        for a in [0.0, 90.0, 180.0] {
            v.push((format!("e{e:.0}a{a:.0}"), dir_from(e, a)));
        }
    }
    v
}

/// Three-band and K-wavelength skies and sun against the 48-wavelength run.
pub fn bands() {
    let sp = Spectra::load();
    let y: f64 = (0..NL).map(|i| 683.0 * sp.solar[i] * sp.cmf[i][1] * 10.0).sum();
    println!("TOA solar illuminance from the spectrum: {:.1} klx with bin-averaged CMF (plan 132.6)", y / 1000.0);
    let elevs = [30.0, 10.0, 5.0, 2.0, 0.0, -2.0, -4.0, -6.0, -9.0, -12.0];
    let cols = Arc::new(Columns::new(AerosolProfile::Exp(1200.0)));
    for preset in PRESETS {
        let aer = Aerosol::preset(preset);
        let spec = Atmosphere::new(spectral_chans(&sp, &aer), aer, cols.clone());
        let models: Vec<(&str, Atmosphere)> = vec![
            ("3band_mean", Atmosphere::new(band_chans(&sp, &aer, false), aer, cols.clone())),
            ("3band_fit", Atmosphere::new(band_chans(&sp, &aer, true), aer, cols.clone())),
            ("12wl", Atmosphere::new(group_chans(&sp, &aer, 12), aer, cols.clone())),
            ("15wl", Atmosphere::new(group_chans(&sp, &aer, 15), aer, cols.clone())),
            ("24wl", Atmosphere::new(group_chans(&sp, &aer, 24), aer, cols.clone())),
        ];
        if preset == "default" {
            for c in &models[1].1.chans {
                println!("  3band_fit coefficients: {c:?}");
            }
        }
        println!("\n{preset}: max sky Delta u'v' over {} directions (worst direction), sun Delta u'v', zenith B/R ratio spectral vs 3band_fit", sky_dirs().len());
        print!("{:>6}", "elev");
        for (n, _) in &models {
            print!(" | {n:>22}");
        }
        println!(" | sun | zenith B/R");
        for e in elevs {
            let sun = dir_from(e, 0.0);
            print!("{e:6.1}");
            let spec_sky: Vec<[f64; 3]> = sky_dirs().iter().map(|(_, d)| spec.to_xyz(&spec.sky(2.0, *d, sun, 0.0))).collect();
            let mut sun_err = vec![];
            for (_, m) in &models {
                let (mut worst, mut wname) = (0.0, String::new());
                for ((name, d), s) in sky_dirs().iter().zip(&spec_sky) {
                    let x = m.to_xyz(&m.sky(2.0, *d, sun, 0.0));
                    let du = duv(x, *s);
                    if du > worst {
                        worst = du;
                        wname = name.clone();
                    }
                }
                print!(" | {worst:.4} {wname:>15}");
                if e >= 0.0 {
                    let mu = (e as f64).to_radians().sin();
                    let ts = spec.to_xyz(&spec.sun_t(R + 2.0, mu));
                    let tm = m.to_xyz(&m.sun_t(R + 2.0, mu));
                    sun_err.push(duv(tm, ts));
                }
            }
            if e >= 0.0 {
                print!(" | sun 3fit {:.4} 15wl {:.4}", sun_err[1], sun_err[3]);
            }
            let zs = to_2020(spec_sky[0]);
            let zm = to_2020(models[1].1.to_xyz(&models[1].1.sky(2.0, [0.0, 1.0, 0.0], sun, 0.0)));
            println!(" | B/R {:.3} vs {:.3} | zenith {:.4} cd/m2", zs[2] / zs[0], zm[2] / zm[0], spec_sky[0][1]);
        }
    }
}

/// Spherical against local-plane altitude for flat-ground receivers.
pub fn curvature() {
    let sp = Spectra::load();
    let cols = Arc::new(Columns::new(AerosolProfile::Exp(1200.0)));
    println!("flat receiver at (d, 0, 0), eye at (0, 2 m, 0); rel diff sphere vs local plane, max over channels");
    println!("preset   d km  elev  azim | in-scatter | transmittance | receiver sun T | receiver altitude (sphere)");
    for preset in ["default", "urban"] {
        let aer = Aerosol::preset(preset);
        let a = Atmosphere::new(band_chans(&sp, &aer, true), aer, cols.clone());
        for d in [16_000.0f64, 32_000.0] {
            let eye = [0.0f64, 2.0, 0.0];
            let to = [d, 0.0, 0.0];
            let v = [to[0] - eye[0], to[1] - eye[1], to[2] - eye[2]];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dir = v.map(|x| x / len);
            let h_sph = (d * d + R * R).sqrt() - R;
            for elev in [2.0, 5.0, 10.0, 30.0, 60.0] {
                for azim in [0.0, 90.0, 180.0] {
                    let sun = dir_from(elev, azim);
                    let (ls, ts) = a.segment(eye, dir, len, sun, Altitude::Sphere, 256, false);
                    let (lp, tp) = a.segment(eye, dir, len, sun, Altitude::LocalPlane, 256, false);
                    // Receiver sun transmittance: tilted zenith on the sphere.
                    let q = [to[0], to[1] + R, to[2]];
                    let rq = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                    let mus = (q[0] * sun[0] + q[1] * sun[1] + q[2] * sun[2]) / rq;
                    let ss = a.sun_t(rq, mus);
                    let sl = a.sun_t(R, sun[1]);
                    println!(
                        "{preset:8} {:4.0} {elev:5.1} {azim:5.0} | {:9.4} | {:9.4} | {:9.4} | {h_sph:.1} m",
                        d / 1000.0,
                        max_rel(&ls, &lp),
                        max_rel(&ts, &tp),
                        max_rel(&ss, &sl)
                    );
                }
            }
        }
    }
}

/// Per-channel against luminance-share scalar phase near the light.
pub fn phase() {
    let sp = Spectra::load();
    let cols = Arc::new(Columns::new(AerosolProfile::Exp(1200.0)));
    println!("max per-channel |rel diff| of sky radiance, scalar vs per-channel phase, within 20 deg of the light (0.5 deg and out)");
    println!("preset               elev | max | at offset | by offset 0.5 1 2 5 10 20");
    for preset in ["default", "default_a0.8", "continental_average", "urban", "desert", "maritime_clean"] {
        let aer = Aerosol::preset(preset);
        let mut a = Atmosphere::new(band_chans(&sp, &aer, true), aer, cols.clone());
        for elev in [5.0, 30.0, 60.0] {
            let sun = dir_from(elev, 0.0);
            let mut by_off = vec![];
            let (mut worst, mut at) = (0.0, 0.0);
            for off in [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0] {
                let mut m = 0.0f64;
                for k in 0..8 {
                    // Ring around the sun.
                    let psi = k as f64 * 45.0f64.to_radians();
                    let (t, b) = {
                        let up = [0.0, 1.0, 0.0];
                        let t = norm(cross(up, sun));
                        (t, cross(sun, t))
                    };
                    let o = (off as f64).to_radians();
                    let d = [0, 1, 2].map(|i| sun[i] * o.cos() + (t[i] * psi.cos() + b[i] * psi.sin()) * o.sin());
                    if d[1] < 0.5f64.to_radians().sin() {
                        continue;
                    }
                    a.phase = Phase::PerChannel;
                    let lp = a.sky(2.0, d, sun, 0.0);
                    a.phase = Phase::LuminanceShare;
                    let ls = a.sky(2.0, d, sun, 0.0);
                    m = m.max(max_rel(&ls, &lp));
                }
                if m > worst {
                    worst = m;
                    at = off;
                }
                by_off.push(m);
            }
            println!(
                "{preset:20} {elev:4.0} | {worst:.4} | {at:4.1} | {}",
                by_off.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" ")
            );
        }
    }
}

fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    a.map(|x| x / l)
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Sky-view LUT made at the focus altitude, read by eyes above it.
pub fn eye() {
    let sp = Spectra::load();
    let cols = Arc::new(Columns::new(AerosolProfile::Exp(1200.0)));
    let aer = Aerosol::preset("default");
    let a = Atmosphere::new(band_chans(&sp, &aer, true), aer, cols);
    let near = 256.0;
    println!("sky from near_range 256 m; LUT at focus altitude h0, eye at h0 + offset; directions elev 0.5-90 deg, azimuth 0-180");
    println!("h0 km elev | offset m: max |rel| any channel (p95 over directions)");
    for h0 in [0.0, 1000.0] {
        for elev in [5.0, 30.0, 60.0] {
            let sun = dir_from(elev, 0.0);
            print!("{:4.1} {elev:4.0}", h0 / 1000.0);
            for off in [10.0, 25.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0] {
                let mut errs = vec![];
                for e in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0] {
                    for az in [0.0, 45.0, 90.0, 135.0, 180.0] {
                        let d = dir_from(e, az);
                        let l0 = a.sky(h0 + 2.0, d, sun, near);
                        let l1 = a.sky(h0 + 2.0 + off, d, sun, near);
                        errs.push(max_rel(&l0, &l1));
                    }
                }
                errs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let p95 = errs[(errs.len() as f64 * 0.95) as usize - 1];
                print!(" | {:.0}: {:.3} ({:.3})", off, errs.last().unwrap(), p95);
            }
            println!();
        }
    }
}

/// OPAC layered aerosol profiles against one 1.2 km exponential, and against the
/// exponential at the layered profile's optical-depth-weighted mean height.
pub fn aerosol() {
    let sp = Spectra::load();
    let exp_cols = Arc::new(Columns::new(AerosolProfile::Exp(1200.0)));
    println!("preset              | H_eq H_g km | elev | sky Delta u'v' vs layered: exp1.2 / exp(H_eq) / exp(H_g) (worst dir for exp1.2) | aerial 1/5/30 km max rel in-scatter: exp1.2 ; exp(H_eq) ; exp(H_g) | T exp1.2");
    for preset in ["clean_continental", "continental_average", "urban", "desert", "maritime_clean"] {
        let base = Aerosol::preset(preset);
        let layered = Aerosol { profile: base.opac_layered(preset), ..base };
        let heq = layered.profile.mean_height();
        let eqv = Aerosol { profile: AerosolProfile::Exp(heq), ..base };
        let hg = 1.0 / layered.profile.density(0.0);
        let gnd = Aerosol { profile: AerosolProfile::Exp(hg), ..base };
        let gnd_a = Atmosphere::new(band_chans(&sp, &gnd, true), gnd, Arc::new(Columns::new(gnd.profile)));
        let lay_a = Atmosphere::new(band_chans(&sp, &layered, true), layered, Arc::new(Columns::new(layered.profile)));
        let exp_a = Atmosphere::new(band_chans(&sp, &base, true), base, exp_cols.clone());
        let eq_a = Atmosphere::new(band_chans(&sp, &eqv, true), eqv, Arc::new(Columns::new(eqv.profile)));
        for elev in [5.0, 30.0, 60.0] {
            let sun = dir_from(elev, 0.0);
            let (mut w1, mut w2, mut w3, mut wn) = (0.0f64, 0.0f64, 0.0f64, String::new());
            for (name, d) in sky_dirs() {
                let xl = lay_a.to_xyz(&lay_a.sky(2.0, d, sun, 0.0));
                let d1 = duv(exp_a.to_xyz(&exp_a.sky(2.0, d, sun, 0.0)), xl);
                let d2 = duv(eq_a.to_xyz(&eq_a.sky(2.0, d, sun, 0.0)), xl);
                if d1 > w1 {
                    w1 = d1;
                    wn = name;
                }
                w2 = w2.max(d2);
                w3 = w3.max(duv(gnd_a.to_xyz(&gnd_a.sky(2.0, d, sun, 0.0)), xl));
            }
            let mut ap = vec![];
            for dist in [1000.0, 5000.0, 30_000.0] {
                let mut m = [0.0f64; 4];
                for az in [0.0, 90.0, 180.0] {
                    let d = dir_from(0.0, az);
                    let (ll, tl) = lay_a.segment([0.0, 2.0, 0.0], d, dist, sun, Altitude::Sphere, 256, false);
                    let (le, te) = exp_a.segment([0.0, 2.0, 0.0], d, dist, sun, Altitude::Sphere, 256, false);
                    let (lq, _) = eq_a.segment([0.0, 2.0, 0.0], d, dist, sun, Altitude::Sphere, 256, false);
                    m[0] = m[0].max(max_rel(&le, &ll));
                    m[1] = m[1].max(max_rel(&lq, &ll));
                    m[2] = m[2].max(max_rel(&te, &tl));
                    let (lg, _) = gnd_a.segment([0.0, 2.0, 0.0], d, dist, sun, Altitude::Sphere, 256, false);
                    m[3] = m[3].max(max_rel(&lg, &ll));
                }
                ap.push(m);
            }
            println!(
                "{preset:20}| {:.2} {:.2} | {elev:4.0} | {w1:.4} / {w2:.4} / {w3:.4} ({wn}) | {:.3} {:.3} {:.3} ; {:.3} {:.3} {:.3} ; {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3}",
                heq / 1000.0,
                hg / 1000.0,
                ap[0][0],
                ap[1][0],
                ap[2][0],
                ap[0][1],
                ap[1][1],
                ap[2][1],
                ap[0][3],
                ap[1][3],
                ap[2][3],
                ap[0][2],
                ap[1][2],
                ap[2][2]
            );
        }
    }
}

/// f32 altitude from render-space positions over a 32 km world, 0-10 km up.
pub fn altitude_f32() {
    println!("render origin anywhere in a 32 km x 32 km world; planet centre 6360 km below the world origin");
    println!("method                       | max |err| m | mean |err| m");
    let r32 = R as f32;
    let mut stats = [(0.0f64, 0.0f64, 0u64); 3];
    let n = 64;
    for oc in 0..9 {
        // Camera (render origin) positions across the world.
        let cam = [(-16_000.0 + 4000.0 * oc as f64), 0.0, (16_000.0 - 4000.0 * oc as f64)];
        let centre_render = [0.0 - cam[0], -R - cam[1], 0.0 - cam[2]];
        let top_render = [centre_render[0], centre_render[1] + R, centre_render[2]];
        for i in 0..n {
            for k in 0..n {
                for j in 0..21 {
                    let world = [-16_000.0 + 32_000.0 * i as f64 / (n - 1) as f64, 500.0 * j as f64, -16_000.0 + 32_000.0 * k as f64 / (n - 1) as f64];
                    let pr = [world[0] - cam[0], world[1] - cam[1], world[2] - cam[2]];
                    let exact = ((world[0]).powi(2) + (world[1] + R).powi(2) + (world[2]).powi(2)).sqrt() - R;
                    let p32 = pr.map(|v| v as f32);
                    // Naive: f32 vector to the centre, length, minus R.
                    let c32 = centre_render.map(|v| v as f32);
                    let v = [p32[0] - c32[0], p32[1] - c32[1], p32[2] - c32[2]];
                    let naive = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - r32;
                    // Same with the centre written as cell (1 km grid) plus offset first.
                    let cell = centre_render.map(|v| (v / 1024.0).round() * 1024.0);
                    let off = [0, 1, 2].map(|q| (centre_render[q] - cell[q]) as f32);
                    let cell32 = cell.map(|v| v as f32);
                    let v2 = [0, 1, 2].map(|q| (p32[q] - off[q]) - cell32[q]);
                    let celled = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt() - r32;
                    // Stable: q = p - top point (small), h = (q.q + 2 R q_y) / (|q + R up| + R).
                    let t32 = top_render.map(|v| v as f32);
                    let q = [p32[0] - t32[0], p32[1] - t32[1], p32[2] - t32[2]];
                    let num = q[0] * q[0] + q[2] * q[2] + q[1] * q[1] + 2.0 * r32 * q[1];
                    let len = (q[0] * q[0] + (q[1] + r32) * (q[1] + r32) + q[2] * q[2]).sqrt();
                    let stable = num / (len + r32);
                    for (s, val) in stats.iter_mut().zip([naive, celled, stable]) {
                        let e = (val as f64 - exact).abs();
                        s.0 = s.0.max(e);
                        s.1 += e;
                        s.2 += 1;
                    }
                }
            }
        }
    }
    for (name, s) in ["length(p - centre) - R", "centre as cell + offset", "(q.q + 2R q_y)/(|.| + R)"].iter().zip(stats) {
        println!("{name:28} | {:.3} | {:.3}", s.0, s.1 / s.2 as f64);
    }
}
