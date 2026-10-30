//! Subsurface walks in slabs and spheres.
//!
//! Reference: the openpbr-bsdf interface at entry and at every boundary hit (exact
//! layering: refraction in, internal reflection and refraction out, NEE through the
//! interface's BTDF). Candidates follow lighting.md L12: an entry direction (rough GGX
//! refraction or cosine) with throughput 1 - E_spec(v), a walk that exits at the first
//! boundary hit as a Lambert vertex times 1 - E_spec(l). The volume (extinction,
//! single-scattering albedo, g) comes from openpbr's subsurface mapping. Grey colour, so
//! one channel. Lengths are in mean free paths.

use crate::openpbr::{LOBE_TRANSMISSION, Mat, Prepared};
use crate::util::Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

type D3 = [f64; 3];
fn dot(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add(a: D3, b: D3) -> D3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: D3, s: f64) -> D3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn normalize(a: D3) -> D3 {
    scale(a, 1.0 / dot(a, a).sqrt())
}
fn to32(a: D3) -> [f32; 3] {
    [a[0] as f32, a[1] as f32, a[2] as f32]
}
fn to64(a: [f32; 3]) -> D3 {
    [a[0] as f64, a[1] as f64, a[2] as f64]
}
fn frame(n: D3) -> (D3, D3) {
    let (t, b) = crate::util::basis(to32(n));
    (to64(t), to64(b))
}
fn local_to_world(l: D3, n: D3) -> D3 {
    let (t, b) = frame(n);
    add(add(scale(t, l[0]), scale(b, l[1])), scale(n, l[2]))
}

#[derive(Clone, Copy)]
enum Shape {
    Slab(f64),
    Sphere(f64),
}

impl Shape {
    /// Distance to the boundary from an inside point and the outward normal there.
    fn exit(self, p: D3, w: D3) -> (f64, D3) {
        match self {
            Shape::Slab(d) => {
                if w[2] > 0.0 {
                    (-p[2] / w[2], [0.0, 0.0, 1.0])
                } else if w[2] < 0.0 {
                    ((-d - p[2]) / w[2], [0.0, 0.0, -1.0])
                } else {
                    (f64::INFINITY, [0.0, 0.0, 1.0])
                }
            }
            Shape::Sphere(r) => {
                let b = dot(p, w);
                let c = dot(p, p) - r * r;
                let t = -b + (b * b - c).max(0.0).sqrt();
                let q = add(p, scale(w, t));
                (t, normalize(q))
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Surface {
    rough: f32,
    ior: f32,
}

#[derive(Clone, Copy)]
struct Medium {
    albedo: f64,
    g: f64,
}

fn material(s: Surface, c: f32, g: f32, n: D3) -> Mat {
    let (t, b) = frame(n);
    Mat {
        sss_weight: 1.0,
        sss_color: [c; 3],
        sss_radius: 1.0,
        sss_radius_scale: [1.0; 3],
        sss_g: g,
        spec_rough: s.rough,
        spec_ior: s.ior,
        t: to32(t),
        b: to32(b),
        n: to32(n),
        ..Mat::default()
    }
}

fn hg_sample(w: D3, g: f64, rng: &mut Rng) -> D3 {
    let s = 2.0 * rng.f64() - 1.0;
    let ct = if g.abs() < 1e-4 { s } else { (1.0 + g * g - ((1.0 - g * g) / (1.0 + g * s)).powi(2)) / (2.0 * g) };
    let st = (1.0 - ct * ct).max(0.0).sqrt();
    let phi = 2.0 * PI * rng.f64();
    local_to_world([st * phi.cos(), st * phi.sin(), ct], w)
}
fn hg_pdf(ct: f64, g: f64) -> f64 {
    (1.0 - g * g) / (4.0 * PI * (1.0 + g * g - 2.0 * g * ct).powf(1.5))
}
fn cosine_sample(n: D3, rng: &mut Rng) -> D3 {
    let r = rng.f64().sqrt();
    let phi = 2.0 * PI * rng.f64();
    local_to_world([r * phi.cos(), r * phi.sin(), (1.0 - r * r).max(0.0).sqrt()], n)
}

/// Visible-normal GGX sample (Heitz 2018), local frame, isotropic alpha.
fn ggx_vndf(v: D3, alpha: f64, rng: &mut Rng) -> D3 {
    let vh = normalize([alpha * v[0], alpha * v[1], v[2]]);
    let lensq = vh[0] * vh[0] + vh[1] * vh[1];
    let t1 = if lensq > 0.0 { scale([-vh[1], vh[0], 0.0], 1.0 / lensq.sqrt()) } else { [1.0, 0.0, 0.0] };
    let t2 = [vh[1] * t1[2] - vh[2] * t1[1], vh[2] * t1[0] - vh[0] * t1[2], vh[0] * t1[1] - vh[1] * t1[0]];
    let r = rng.f64().sqrt();
    let phi = 2.0 * PI * rng.f64();
    let p1 = r * phi.cos();
    let s = 0.5 * (1.0 + vh[2]);
    let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * r * phi.sin();
    let nh = add(add(scale(t1, p1), scale(t2, p2)), scale(vh, (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt()));
    normalize([alpha * nh[0], alpha * nh[1], nh[2].max(1e-7)])
}
fn refract(i: D3, m: D3, eta: f64) -> Option<D3> {
    // i points away from the surface (towards the viewer); returns the transmitted direction.
    let c = dot(i, m);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    if k < 0.0 {
        return None;
    }
    Some(add(scale(i, -eta), scale(m, eta * c - k.sqrt())))
}

/// Specular reflection albedo of the interface from outside, from openpbr samples.
struct SpecAlbedo {
    table: Vec<f64>,
}
impl SpecAlbedo {
    fn new(s: Surface, c: f32, g: f32) -> Self {
        let n = 64;
        let table = (0..n)
            .into_par_iter()
            .map(|i| {
                let mu = (i as f64 + 0.5) / n as f64;
                let v = [(1.0 - mu * mu).sqrt(), 0.0, mu];
                let p = Prepared::new(&material(s, c, g, [0.0, 0.0, 1.0]), to32(v));
                let mut rng = Rng::new(17, i as u64);
                let k = 200_000;
                let mut acc = 0.0;
                for _ in 0..k {
                    if let Some(smp) = p.sample([rng.f32(), rng.f32(), rng.f32()]) {
                        if smp.lobe & LOBE_TRANSMISSION == 0 && smp.wi[2] > 0.0 {
                            acc += smp.weight[0] as f64;
                        }
                    }
                }
                acc / k as f64
            })
            .collect();
        SpecAlbedo { table }
    }
    fn at(&self, mu: f64) -> f64 {
        let n = self.table.len();
        let x = (mu.clamp(0.0, 1.0) * n as f64 - 0.5).clamp(0.0, (n - 1) as f64);
        let i = (x as usize).min(n - 2);
        let f = x - i as f64;
        self.table[i] * (1.0 - f) + self.table[i + 1] * f
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Entry {
    Exact,
    Ggx,
    Diffuse,
}

#[derive(Clone, Copy)]
struct Guide {
    p: f64,  // probability of the guided technique
    v0: f64, // diffusion eigenvalue
}

fn dwivedi_v0(albedo: f64) -> f64 {
    // 1 = albedo * v0 * atanh(1 / v0), v0 > 1.
    let f = |v: f64| albedo * v * 0.5 * ((v + 1.0) / (v - 1.0)).ln() - 1.0;
    let (mut lo, mut hi) = (1.0 + 1e-12, 1e6);
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if f(m) > 0.0 { lo = m } else { hi = m }
    }
    0.5 * (lo + hi)
}

struct WalkOut {
    radiance: f64,
    steps: u32,
    weight_peak: f64,
}

struct Ctx<'a> {
    shape: Shape,
    surf: Surface,
    c: f32,
    g_in: f32,
    med: Medium,
    spec: &'a SpecAlbedo,
    light: D3,
    max_steps: u32,
}

fn dwivedi_sample(gd: &Guide, n0: D3, rng: &mut Rng) -> D3 {
    let lnr = ((gd.v0 - 1.0) / (gd.v0 + 1.0)).ln();
    let mu = gd.v0 - (gd.v0 + 1.0) * (rng.f64() * lnr).exp();
    let st = (1.0 - mu * mu).max(0.0).sqrt();
    let phi = 2.0 * PI * rng.f64();
    local_to_world([st * phi.cos(), st * phi.sin(), mu], n0)
}
fn dwivedi_pdf(gd: &Guide, mu: f64) -> f64 {
    1.0 / (((gd.v0 + 1.0) / (gd.v0 - 1.0)).ln() * (gd.v0 - mu)) / (2.0 * PI)
}

/// One walk from an entry point `p0` with outward normal `n0`, viewed from `v`. With a
/// guide, every segment after the first scatter draws direction and distance from the
/// classic (HG, sigma_t) or the Dwivedi technique (towards n0, sigma_t (1 - mu / v0)),
/// weighted by the one-sample balance heuristic over the joint density.
fn walk(cx: &Ctx, entry: Entry, guide: Option<Guide>, p0: D3, n0: D3, v: D3, rng: &mut Rng) -> WalkOut {
    let mut out = WalkOut { radiance: 0.0, steps: 0, weight_peak: 0.0 };
    if dot(v, n0) < 1e-4 {
        return out;
    }
    let (mut w, mut thr) = match entry {
        Entry::Exact => {
            let prep = Prepared::new(&material(cx.surf, cx.c, cx.g_in, n0), to32(v));
            match prep.sample([rng.f32(), rng.f32(), rng.f32()]) {
                Some(s) if s.lobe & LOBE_TRANSMISSION != 0 && dot(to64(s.wi), n0) < 0.0 => (normalize(to64(s.wi)), s.weight[0] as f64),
                _ => return out,
            }
        }
        Entry::Ggx => {
            let (t, b) = frame(n0);
            let vl = [dot(v, t), dot(v, b), dot(v, n0)];
            let alpha = (cx.surf.rough as f64).powi(2).max(1e-4);
            let m = ggx_vndf(vl, alpha, rng);
            let Some(wt) = refract(vl, m, 1.0 / cx.surf.ior as f64) else { return out };
            if wt[2] >= 0.0 {
                return out;
            }
            (normalize(local_to_world(wt, n0)), 1.0 - cx.spec.at(dot(v, n0)))
        }
        Entry::Diffuse => (cosine_sample(scale(n0, -1.0), rng), 1.0 - cx.spec.at(dot(v, n0))),
    };
    let sigma_t = 1.0;
    let mut p = p0;
    // Densities of the current direction under (classic, guided), and the technique that
    // drew it; None while the direction did not come from the mixture.
    let mut mix: Option<(f64, f64, bool)> = None;
    loop {
        if out.steps >= cx.max_steps {
            return out;
        }
        out.steps += 1;
        let (tb, nb) = cx.shape.exit(p, w);
        let st_d = guide.map(|gd| sigma_t * (1.0 - dot(w, n0) / gd.v0)).unwrap_or(sigma_t);
        let guided_dist = matches!(mix, Some((_, _, true)));
        let t = -(1.0 - rng.f64()).ln() / if guided_dist { st_d } else { sigma_t };
        let reached = t >= tb;
        if let (Some(gd), Some((ph, pd, _))) = (guide, mix) {
            let (f, q) = if reached {
                let f = ph * (-sigma_t * tb).exp();
                (f, (1.0 - gd.p) * f + gd.p * pd * (-st_d * tb).exp())
            } else {
                let f = ph * sigma_t * (-sigma_t * t).exp();
                (f, (1.0 - gd.p) * f + gd.p * pd * st_d * (-st_d * t).exp())
            };
            thr *= f / q;
        }
        out.weight_peak = out.weight_peak.max(thr);
        if reached {
            p = add(p, scale(w, tb));
            if entry == Entry::Exact && dot(w, nb) < 1e-4 {
                w = normalize(add(w, scale(nb, -2.0 * dot(w, nb))));
                mix = None;
            } else if entry == Entry::Exact {
                let prep = Prepared::new(&material(cx.surf, cx.c, cx.g_in, nb), to32(scale(w, -1.0)));
                if dot(cx.light, nb) > 0.0 {
                    out.radiance += thr * prep.eval(to32(cx.light))[0] as f64;
                }
                match prep.sample([rng.f32(), rng.f32(), rng.f32()]) {
                    Some(s) if s.lobe & LOBE_TRANSMISSION == 0 && dot(to64(s.wi), nb) < 0.0 => {
                        w = normalize(to64(s.wi));
                        thr *= s.weight[0] as f64;
                        mix = None;
                    }
                    _ => return out,
                }
            } else {
                let cl = dot(cx.light, nb);
                if cl > 0.0 {
                    out.radiance += thr * (1.0 - cx.spec.at(cl)) * cl / PI;
                }
                return out;
            }
        } else {
            p = add(p, scale(w, t));
            thr *= cx.med.albedo;
            match guide {
                Some(gd) => {
                    let guided = rng.f64() < gd.p;
                    let nw = if guided { dwivedi_sample(&gd, n0, rng) } else { hg_sample(w, cx.med.g, rng) };
                    mix = Some((hg_pdf(dot(w, nw), cx.med.g), dwivedi_pdf(&gd, dot(nw, n0)), guided));
                    w = nw;
                }
                None => w = hg_sample(w, cx.med.g, rng),
            }
        }
    }
}

fn volume(s: Surface, c: f32, g: f32) -> Medium {
    let p = Prepared::new(&material(s, c, g, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]);
    let (e, a, g) = p.volume();
    assert!((e[0] - 1.0).abs() < 1e-5, "extinction {e:?}");
    Medium { albedo: a[0] as f64, g: g as f64 }
}

struct Stats {
    mean: f64,
    stderr: f64,
}

fn run_walks<F: Fn(&mut Rng) -> WalkOut + Sync>(n: usize, seed: u64, f: F) -> (Stats, Vec<(u32, f64, f64)>) {
    let rec: Vec<(u32, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed, i as u64);
            let o = f(&mut rng);
            (o.steps, o.radiance, o.weight_peak)
        })
        .collect();
    let m = rec.iter().map(|r| r.1).sum::<f64>() / n as f64;
    let var = rec.iter().map(|r| (r.1 - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (Stats { mean: m, stderr: (var / n as f64).sqrt() }, rec)
}

const SURFACES: [(&str, Surface); 2] = [("marble", Surface { rough: 0.1, ior: 1.5 }), ("skin", Surface { rough: 0.4, ior: 1.4 })];
const SIZES: [f64; 4] = [0.25, 1.0, 4.0, 16.0];

fn deg(a: f64) -> f64 {
    a.to_radians()
}

fn shape_walk(cx: &Ctx, e: Entry, guide: Option<Guide>, v: D3, rng: &mut Rng) -> WalkOut {
    match cx.shape {
        Shape::Slab(_) => walk(cx, e, guide, [0.0; 3], [0.0, 0.0, 1.0], v, rng),
        Shape::Sphere(r) => {
            // Orthographic view along -z: entry points uniform over the projected disc.
            let (rr, phi) = (r * rng.f64().sqrt(), 2.0 * PI * rng.f64());
            let (x, y) = (rr * phi.cos(), rr * phi.sin());
            let p0 = [x, y, (r * r - x * x - y * y).max(0.0).sqrt()];
            walk(cx, e, guide, p0, normalize(p0), v, rng)
        }
    }
}

/// Exact-layering reference traced from the light: refraction in through openpbr with
/// the light direction as view, a classic walk, and at each boundary hit openpbr's
/// interface from inside (internal reflection continues, transmission exits). Returns
/// the exit direction and weight, and whether the path scattered at least once.
fn walk_light(cx: &Ctx, l: D3, p0: D3, n0: D3, rng: &mut Rng) -> Option<(D3, f64, bool)> {
    if dot(l, n0) < 1e-4 {
        return None;
    }
    let prep = Prepared::new(&material(cx.surf, cx.c, cx.g_in, n0), to32(l));
    let s = prep.sample([rng.f32(), rng.f32(), rng.f32()])?;
    if s.lobe & LOBE_TRANSMISSION == 0 || dot(to64(s.wi), n0) >= 0.0 {
        return None;
    }
    let (mut w, mut thr) = (normalize(to64(s.wi)), s.weight[0] as f64);
    let mut p = p0;
    let mut scattered = false;
    for _ in 0..cx.max_steps {
        let (tb, nb) = cx.shape.exit(p, w);
        let t = -(1.0 - rng.f64()).ln();
        if t >= tb {
            p = add(p, scale(w, tb));
            if dot(w, nb) < 1e-4 {
                // Grazing internal hit: total internal reflection.
                w = normalize(add(w, scale(nb, -2.0 * dot(w, nb))));
                continue;
            }
            let prep = Prepared::new(&material(cx.surf, cx.c, cx.g_in, nb), to32(scale(w, -1.0)));
            let s = prep.sample([rng.f32(), rng.f32(), rng.f32()])?;
            let wi = normalize(to64(s.wi));
            thr *= s.weight[0] as f64;
            if dot(wi, nb) > 0.0 {
                return Some((wi, thr, scattered));
            }
            w = wi;
        } else {
            p = add(p, scale(w, t));
            thr *= cx.med.albedo;
            scattered = true;
            w = hg_sample(w, cx.med.g, rng);
        }
    }
    None
}

/// Camera-side candidate walk that drops paths with no scatter event.
fn walk_scattered(cx: &Ctx, e: Entry, v: D3, rng: &mut Rng) -> WalkOut {
    let o = shape_walk(cx, e, None, v, rng);
    if o.steps <= 1 { WalkOut { radiance: 0.0, ..o } } else { o }
}

/// Entry candidates against the exact-layering reference. Radiance toward v under unit
/// irradiance normal to the light, scattered paths only (the walk cannot produce the
/// unscattered back-face reflection); slabs per unit area, spheres averaged over the
/// projected disc.
pub fn run_entry() {
    let n_cam: usize = std::env::var("WALKS").ok().and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let n_ref = n_cam * 10;
    let half = 6.0f64.to_radians();
    let omega = 2.0 * PI * (1.0 - half.cos());
    println!("shape size surf C g | config | ref (rel stderr) | ggx rel | diffuse rel");
    let mut summary: Vec<(String, f64, f64)> = Vec::new();
    for (sname, surf) in SURFACES {
        for c in [0.8f32, 0.9] {
            for g in [0.0f32, 0.8] {
                let med = volume(surf, c, g);
                let spec = SpecAlbedo::new(surf, c, g);
                eprintln!("{sname} C{c} g{g}: albedo {:.5} g {:.2}, E_spec(1) {:.4} E_spec(0.5) {:.4}", med.albedo, med.g, spec.at(1.0), spec.at(0.5));
                for size in SIZES {
                    for shape in [Shape::Slab(size), Shape::Sphere(size)] {
                        // (label, view, light); the slab's light sits opposite the view in
                        // azimuth except at 45/45, which is the retro direction.
                        let configs: Vec<(String, D3, D3)> = match shape {
                            Shape::Slab(_) => [(0.0f64, 0.0f64, -1.0f64), (0.0, 60.0, -1.0), (60.0, 0.0, -1.0), (45.0, 45.0, 1.0), (70.0, 30.0, -1.0)]
                                .iter()
                                .map(|&(tv, tl, side)| {
                                    (format!("v{tv:.0}/l{tl:.0}"), [deg(tv).sin(), 0.0, deg(tv).cos()], [side * deg(tl).sin(), 0.0, deg(tl).cos()])
                                })
                                .collect(),
                            Shape::Sphere(_) => [0.0f64, 90.0, 150.0]
                                .iter()
                                .map(|&tl| (format!("l{tl:.0}"), [0.0, 0.0, 1.0], [deg(tl).sin(), 0.0, deg(tl).cos()]))
                                .collect(),
                        };
                        for (label, v, l) in configs {
                            let cx = Ctx { shape, surf, c, g_in: g, med, spec: &spec, light: l, max_steps: 4096 };
                            let hits: Vec<f64> = (0..n_ref)
                                .into_par_iter()
                                .map(|i| {
                                    let mut rng = Rng::new(5, i as u64);
                                    let (p0, n0) = match shape {
                                        Shape::Slab(_) => ([0.0; 3], [0.0, 0.0, 1.0]),
                                        Shape::Sphere(r) => {
                                            // Uniform over the disc the light sees.
                                            let (rr, phi) = (r * rng.f64().sqrt(), 2.0 * PI * rng.f64());
                                            let local = [rr * phi.cos(), rr * phi.sin(), (r * r - rr * rr).max(0.0).sqrt()];
                                            let p0 = local_to_world(local, l);
                                            (p0, normalize(p0))
                                        }
                                    };
                                    match walk_light(&cx, l, p0, n0, &mut rng) {
                                        Some((d, wt, true)) if dot(d, v) >= half.cos() => wt,
                                        _ => 0.0,
                                    }
                                })
                                .collect();
                            let m = hits.iter().sum::<f64>() / n_ref as f64;
                            let var = hits.iter().map(|h| (h - m).powi(2)).sum::<f64>() / (n_ref as f64 - 1.0);
                            let k = match shape {
                                Shape::Slab(_) => l[2] / (omega * v[2]),
                                Shape::Sphere(_) => 1.0 / omega,
                            };
                            let reference = m * k;
                            let rel_err = (var / n_ref as f64).sqrt() / m.max(1e-30);
                            let mut res = [0.0f64; 2];
                            for (j, e) in [Entry::Ggx, Entry::Diffuse].into_iter().enumerate() {
                                res[j] = run_walks(n_cam, 7 + j as u64, |rng| walk_scattered(&cx, e, v, rng)).0.mean;
                            }
                            let (rg, rd) = (res[0] / reference - 1.0, res[1] / reference - 1.0);
                            let shape_name = match shape {
                                Shape::Slab(_) => "slab",
                                Shape::Sphere(_) => "sphere",
                            };
                            println!("{shape_name} {size:5.2} {sname} {c} {g} | {label:9} | {reference:.5} ({rel_err:.3}) | {rg:+.3} | {rd:+.3}");
                            summary.push((format!("{sname} {shape_name} g{g}"), rg, rd));
                        }
                    }
                }
            }
        }
    }
    println!("\nmean |rel err| by group: ggx, diffuse; max |rel err|");
    let mut groups: Vec<String> = summary.iter().map(|s| s.0.clone()).collect();
    groups.sort();
    groups.dedup();
    for gname in groups {
        let rows: Vec<_> = summary.iter().filter(|s| s.0 == gname).collect();
        let n = rows.len() as f64;
        let mg = rows.iter().map(|r| r.1.abs()).sum::<f64>() / n;
        let md = rows.iter().map(|r| r.2.abs()).sum::<f64>() / n;
        let xg = rows.iter().map(|r| r.1.abs()).fold(0.0, f64::max);
        let xd = rows.iter().map(|r| r.2.abs()).fold(0.0, f64::max);
        let wins = rows.iter().filter(|r| r.1.abs() < r.2.abs()).count();
        println!("{gname:20} ggx {mg:.3} (max {xg:.3})  diffuse {md:.3} (max {xd:.3})  ggx closer in {wins}/{}", rows.len());
    }
}

/// Classic against Dwivedi-guided candidate walks: steps at equal energy per cap.
pub fn run_dwivedi() {
    let n_walks: usize = std::env::var("WALKS").ok().and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let surf = SURFACES[1].1;
    println!("shape C g P | uncapped E, mean steps | cap: E_classic cap_D cost_C cost_D saving | top-0.1% share C/D, max/mean C/D, peak weight C/D");
    for c in [0.8f32, 0.9] {
        for g in [0.0f32, 0.8] {
            let med = volume(surf, c, g);
            let spec = SpecAlbedo::new(surf, c, g);
            let red = med.albedo * (1.0 - med.g) / (1.0 - med.albedo * med.g);
            let v0 = dwivedi_v0(red);
            for shape in [Shape::Slab(1.0), Shape::Slab(4.0), Shape::Slab(16.0), Shape::Sphere(4.0), Shape::Sphere(16.0)] {
                for pg in [0.5, 0.75] {
                    let cx = Ctx { shape, surf, c, g_in: g, med, spec: &spec, light: [0.0, 0.0, 1.0], max_steps: 4096 };
                    let (sc, rc) = run_walks(n_walks, 11, |rng| shape_walk(&cx, Entry::Ggx, None, [0.0, 0.0, 1.0], rng));
                    let (sd, rd) = run_walks(n_walks, 12, |rng| shape_walk(&cx, Entry::Ggx, Some(Guide { p: pg, v0 }), [0.0, 0.0, 1.0], rng));
                    let energy = |rec: &[(u32, f64, f64)], cap: u32| rec.iter().filter(|r| r.0 <= cap).map(|r| r.1).sum::<f64>() / rec.len() as f64;
                    let cost = |rec: &[(u32, f64, f64)], cap: u32| rec.iter().map(|r| r.0.min(cap) as f64).sum::<f64>() / rec.len() as f64;
                    let tail = |rec: &[(u32, f64, f64)]| {
                        let mut v: Vec<f64> = rec.iter().map(|r| r.1).collect();
                        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        let tot: f64 = v.iter().sum();
                        let k = (v.len() / 1000).max(1);
                        let top: f64 = v[..k].iter().sum();
                        let peak = rec.iter().map(|r| r.2).fold(0.0, f64::max);
                        (top / tot, v[0] / (tot / v.len() as f64), peak)
                    };
                    let shape_name = match shape {
                        Shape::Slab(d) => format!("slab{d}"),
                        Shape::Sphere(r) => format!("sphere{r}"),
                    };
                    let (tc, mc, pc) = tail(&rc);
                    let (td, md, pd) = tail(&rd);
                    print!(
                        "{shape_name:9} C{c} g{g} P{pg} | {:.4}+-{:.4} vs {:.4}+-{:.4}, steps {:.1} vs {:.1}",
                        sc.mean,
                        sc.stderr,
                        sd.mean,
                        sd.stderr,
                        cost(&rc, 4096),
                        cost(&rd, 4096)
                    );
                    for cap in [16u32, 32, 64] {
                        let ec = energy(&rc, cap);
                        let cap_d = (1..=4096).find(|&k| energy(&rd, k) >= ec).unwrap_or(4096);
                        let (cc, cd) = (cost(&rc, cap), cost(&rd, cap_d));
                        print!(" | {cap}: {ec:.4} {cap_d} {cc:.1} {cd:.1} {:+.0}%", (1.0 - cd / cc) * 100.0);
                    }
                    println!(" | {tc:.4}/{td:.4} {mc:.0}/{md:.0} {pc:.2}/{pd:.2}");
                }
            }
        }
    }
}
