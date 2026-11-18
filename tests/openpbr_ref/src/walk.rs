//! A CPU reference of the real-time subsurface walk on analytic shapes.
//!
//! A walk enters along the transmission of adobe's BSDF from the view, weighted by that
//! transmission's albedo; it steps through the interior medium adobe maps from the
//! subsurface parameters, one walk per color channel, distances by the channel's extinction
//! and directions by Henyey-Greenstein; it leaves at the first boundary it reaches as Lambert
//! reflection on the boundary's normal times the transmission's albedo seen from the light,
//! over the subsurface weight. Lengths are in m.

use std::f64::consts::{PI, TAU};

use orr_math::DVec3;
use shader_math::stats::Rng;

use crate::{Bsdf, Inputs, Reference, in_parallel};

fn to_world(l: DVec3, n: DVec3) -> DVec3 {
    let (t, b) = n.any_orthonormal_pair();
    t * l.x + b * l.y + n * l.z
}

fn to_local(w: DVec3, n: DVec3) -> DVec3 {
    let (t, b) = n.any_orthonormal_pair();
    DVec3::new(w.dot(t), w.dot(b), w.dot(n))
}

/// The generator of walk `i` of a run seeded with `seed`.
pub fn rng(seed: u64, i: u64) -> Rng {
    Rng::new(seed ^ i.wrapping_mul(0xD1B5_4A32_D192_ED03))
}

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// The slab from z = 0 down to z = -thickness, unbounded in x and y.
    Slab { thickness: f64 },
    /// The sphere of `radius` around the origin.
    Sphere { radius: f64 },
}

impl Shape {
    /// Distance from inside point `p` along unit `w` to the boundary, and the outward normal
    /// there.
    pub fn exit(self, p: DVec3, w: DVec3) -> (f64, DVec3) {
        match self {
            Shape::Slab { thickness } => {
                if w.z > 0.0 {
                    ((-p.z / w.z).max(0.0), DVec3::Z)
                } else if w.z < 0.0 {
                    (((-thickness - p.z) / w.z).max(0.0), DVec3::NEG_Z)
                } else {
                    (f64::INFINITY, DVec3::Z)
                }
            }
            Shape::Sphere { radius } => {
                let b = p.dot(w);
                let c = p.length_squared() - radius * radius;
                let t = (-b + (b * b - c).max(0.0).sqrt()).max(0.0);
                (t, (p + w * t).normalize())
            }
        }
    }
}

/// A homogeneous interior medium.
#[derive(Clone, Copy, Debug)]
pub struct Medium {
    /// 1/m.
    pub extinction: DVec3,
    pub albedo: DVec3,
    pub anisotropy: f64,
    /// The surface's IOR against the outside's. Above 1 it reflects walks back by Fresnel, each
    /// reflection one step; at 1 they leave at the first boundary they reach.
    pub surface_ior: f64,
}

impl Medium {
    /// adobe's interior medium of `inputs`.
    pub fn of(inputs: &Inputs) -> Self {
        let mut r = Reference::new();
        r.prepare(inputs, [0.0, 0.0, 1.0]);
        let v = r.volume();
        Self {
            extinction: DVec3::from_array(v.extinction.map(f64::from)),
            albedo: DVec3::from_array(v.albedo.map(f64::from)),
            anisotropy: f64::from(v.anisotropy),
            surface_ior: 1.0,
        }
    }
}

/// A direction scattered from unit `w` by Henyey-Greenstein `g`.
fn sample_hg(w: DVec3, g: f64, rng: &mut Rng) -> DVec3 {
    let u = rng.f64();
    let c = if g.abs() < 1e-3 {
        1.0 - 2.0 * u
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        ((1.0 + g * g - s * s) / (2.0 * g)).clamp(-1.0, 1.0)
    };
    let sine = (1.0 - c * c).max(0.0).sqrt();
    let phi = TAU * rng.f64();
    to_world(DVec3::new(sine * phi.cos(), sine * phi.sin(), c), w).normalize()
}

/// The mean of `count` estimates over threads, each thread holding state from `make`.
fn mean_of<S>(
    count: usize,
    seed: u64,
    make: &(impl Fn() -> S + Sync),
    estimate: impl Fn(&mut S, &mut Rng) -> DVec3 + Sync,
) -> DVec3 {
    let chunks = 256;
    let per = count.div_ceil(chunks);
    let sums = in_parallel(chunks, make, |state, i| {
        let mut rng = rng(seed, i as u64);
        (0..per).map(|_| estimate(state, &mut rng)).sum::<DVec3>()
    });
    sums.into_iter().sum::<DVec3>() / (per * chunks) as f64
}

/// Where a walk leaves through a shape's boundary.
pub struct WalkExit {
    pub point: DVec3,
    /// The boundary's outward normal there.
    pub normal: DVec3,
    /// The walk's direction as it reaches the boundary.
    pub dir: DVec3,
    pub throughput: f64,
    /// The scatter the walk's last step started from and the direction the walk reached it
    /// along, where that step started from a scatter.
    pub last: Option<(DVec3, DVec3)>,
    /// Whether the walk scattered at all.
    pub scattered: bool,
}

/// Where channel `c`'s walk from `p` along unit `w` inside `shape` leaves through the boundary.
/// `None` when the walk is absorbed, or still inside after `max_steps` steps.
pub fn walk_exit(
    shape: Shape,
    medium: &Medium,
    c: usize,
    mut p: DVec3,
    mut w: DVec3,
    max_steps: u32,
    rng: &mut Rng,
) -> Option<WalkExit> {
    let (sigma, albedo) = (medium.extinction[c], medium.albedo[c]);
    let mut throughput = 1.0;
    let mut last = None;
    let mut scattered = false;
    for _ in 0..max_steps {
        let t = -(1.0 - rng.f64()).ln() / sigma;
        let (tb, nb) = shape.exit(p, w);
        if t >= tb {
            let cosine = w.dot(nb);
            let ior = medium.surface_ior;
            if ior == 1.0 || rng.f64() >= fresnel(1.0 / ior, cosine) {
                return Some(WalkExit {
                    point: p + w * tb,
                    normal: nb,
                    dir: w,
                    throughput,
                    last,
                    scattered,
                });
            }
            p += w * tb;
            w = (w - nb * (2.0 * cosine)).normalize();
            last = None;
            continue;
        }
        p += w * t;
        throughput *= albedo;
        // Russian roulette keeps long, dim walks cheap without bias.
        if throughput < 0.05 {
            if rng.f64() >= throughput / 0.05 {
                return None;
            }
            throughput = 0.05;
        }
        last = Some((p, w));
        scattered = true;
        w = sample_hg(w, medium.anisotropy, rng);
    }
    None
}

/// [`walk_exit`]'s outward normal and throughput.
pub fn walk_channel(
    shape: Shape,
    medium: &Medium,
    c: usize,
    p: DVec3,
    w: DVec3,
    max_steps: u32,
    rng: &mut Rng,
) -> Option<(DVec3, f64)> {
    walk_exit(shape, medium, c, p, w, max_steps, rng).map(|e| (e.normal, e.throughput))
}

/// GGX's density of normals at `h`, about +z.
fn ggx_d(alpha: f64, h: DVec3) -> f64 {
    let a2 = alpha * alpha;
    let d = h.z * h.z * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

/// Smith's masking of `v` over GGX normals about +z.
fn ggx_g1(alpha: f64, v: DVec3) -> f64 {
    let z2 = v.z * v.z;
    2.0 / (1.0 + (1.0 + alpha * alpha * (1.0 - z2) / z2).sqrt())
}

/// A visible GGX normal about +z seen from `v` (above the plane), by spherical caps.
fn sample_vndf(alpha: f64, v: DVec3, u: [f64; 2]) -> DVec3 {
    let vh = DVec3::new(alpha * v.x, alpha * v.y, v.z).normalize();
    let phi = TAU * u[0];
    let z = (1.0 - u[1]) * (1.0 + vh.z) - vh.z;
    let sin = (1.0 - z * z).max(0.0).sqrt();
    let c = DVec3::new(sin * phi.cos(), sin * phi.sin(), z) + vh;
    DVec3::new(alpha * c.x, alpha * c.y, c.z).normalize()
}

/// Henyey-Greenstein's density of `g` at cosine `c`, per steradian.
fn hg(g: f64, c: f64) -> f64 {
    (1.0 - g * g) / (4.0 * PI * (1.0 + g * g - 2.0 * g * c).powf(1.5))
}

/// The direction inside a smooth surface of outward normal `n`, `ior` times the outside's,
/// that refracts out along unit `out`.
fn inside_toward(n: DVec3, ior: f64, out: DVec3) -> DVec3 {
    let (cos, eta) = (n.dot(out), 1.0 / ior);
    let t = (1.0 - eta * eta * (1.0 - cos * cos)).sqrt();
    (out * eta - n * (eta * cos - t)).normalize()
}

/// The share of light crossing a subsurface material's surface into or out of its medium,
/// by the cosine to the surface: the albedo of adobe's transmission from that direction
/// over the subsurface weight, per channel.
pub struct Interface {
    table: Vec<DVec3>,
}

const INTERFACE_BINS: usize = 64;

impl Interface {
    /// From `samples` of adobe's BSDF of `inputs` per cosine bin. `inputs` has no
    /// transmission weight, so every direction below the surface enters the medium.
    pub fn new(inputs: &Inputs, samples: usize) -> Self {
        assert_eq!(inputs.transmission_weight, 0.0);
        let weight = f64::from(inputs.subsurface_weight * (1.0 - inputs.base_metalness));
        let table = in_parallel(INTERFACE_BINS, &Reference::new, |bsdf, i| {
            let mu = (i as f64 + 0.5) / INTERFACE_BINS as f64;
            bsdf.prepare(inputs, [(1.0 - mu * mu).sqrt() as f32, 0.0, mu as f32]);
            let mut rng = rng(17, i as u64);
            let mut sum = DVec3::ZERO;
            for _ in 0..samples {
                let u = [rng.f32(), rng.f32(), rng.f32()];
                if let Some(s) = bsdf.sample(u)
                    && s.wi[2] < 0.0
                {
                    sum += DVec3::from_array(s.weight.map(f64::from));
                }
            }
            sum / (samples as f64 * weight)
        });
        Self { table }
    }

    /// The share at cosine `mu`, interpolated between bins.
    pub fn at(&self, mu: f64) -> DVec3 {
        let last = (INTERFACE_BINS - 1) as f64;
        let x = (mu.clamp(0.0, 1.0) * INTERFACE_BINS as f64 - 0.5).clamp(0.0, last);
        let i = (x as usize).min(INTERFACE_BINS - 2);
        let f = x - i as f64;
        self.table[i].lerp(self.table[i + 1], f)
    }

    /// The cosine-weighted mean of the share over the hemisphere: what a Lambert exit returns
    /// under uniform unit radiance.
    pub fn hemisphere(&self) -> DVec3 {
        let n = INTERFACE_BINS as f64;
        (0..INTERFACE_BINS)
            .map(|i| self.table[i] * 2.0 * (i as f64 + 0.5) / n / n)
            .sum()
    }
}

/// What reaches a walk's exit: a directional light with nothing in its way but the shape
/// (unit direction toward it and illuminance), the same over a disc of angular `radius`, or
/// uniform radiance from every direction.
#[derive(Clone, Copy, Debug)]
pub enum Light {
    Sun {
        direction: DVec3,
        illuminance: DVec3,
    },
    Disc {
        direction: DVec3,
        illuminance: DVec3,
        radius: f64,
    },
    Uniform {
        radiance: DVec3,
    },
}

impl Light {
    /// A sun's or a disc's direction and illuminance; a disc's next-event estimates take it at
    /// its center.
    fn sun(self) -> Option<(DVec3, DVec3)> {
        match self {
            Light::Sun {
                direction,
                illuminance,
            }
            | Light::Disc {
                direction,
                illuminance,
                ..
            } => Some((direction, illuminance)),
            Light::Uniform { .. } => None,
        }
    }
}

/// Walks of one material in one shape.
pub struct Walks<'a> {
    pub inputs: &'a Inputs,
    pub shape: Shape,
    pub medium: Medium,
    pub interface: &'a Interface,
    /// Steps before a walk is dropped.
    pub max_steps: u32,
}

impl<'a> Walks<'a> {
    pub fn new(inputs: &'a Inputs, shape: Shape, interface: &'a Interface) -> Self {
        Self {
            inputs,
            shape,
            medium: Medium::of(inputs),
            interface,
            max_steps: u32::MAX,
        }
    }

    /// One estimate of the radiance toward `view` (unit) at boundary point `p0` with outward
    /// normal `n0` under `light`: one walk per channel.
    pub fn radiance(
        &self,
        bsdf: &mut Reference,
        p0: DVec3,
        n0: DVec3,
        view: DVec3,
        light: Light,
        rng: &mut Rng,
    ) -> DVec3 {
        let mut out = DVec3::ZERO;
        let local_view = to_local(view, n0);
        if local_view.z <= 1e-4 {
            return out;
        }
        bsdf.prepare(self.inputs, local_view.as_vec3().to_array());
        let entry = loop {
            let u = [rng.f32(), rng.f32(), rng.f32()];
            if let Some(s) = bsdf.sample(u)
                && s.wi[2] < 0.0
            {
                break to_world(DVec3::from_array(s.wi.map(f64::from)), n0).normalize();
            }
        };
        let weight = self.interface.at(local_view.z);
        let sss = f64::from(self.inputs.subsurface_weight * (1.0 - self.inputs.base_metalness));
        for c in 0..3 {
            let Some((n, throughput)) =
                walk_channel(self.shape, &self.medium, c, p0, entry, self.max_steps, rng)
            else {
                continue;
            };
            let exit = match light.sun() {
                Some((direction, illuminance)) => {
                    let cos = n.dot(direction);
                    if cos > 0.0 {
                        self.interface.at(cos)[c] * cos / PI * illuminance[c]
                    } else {
                        0.0
                    }
                }
                None => self.interface.hemisphere()[c] * uniform_radiance(light)[c],
            };
            out[c] = weight[c] * sss * throughput * exit;
        }
        out
    }

    /// The mean of [`Self::radiance`] over `count` estimates, each at a boundary point, its
    /// outward normal and the view there from `point`.
    pub fn mean(
        &self,
        count: usize,
        light: Light,
        seed: u64,
        point: impl Fn(&mut Rng) -> (DVec3, DVec3, DVec3) + Sync,
    ) -> DVec3 {
        mean_of(count, seed, &Reference::new, |bsdf, rng| {
            let (p, n, v) = point(rng);
            self.radiance(bsdf, p, n, v, light, rng)
        })
    }
}

/// The throughput that walks entered at the slab's top along unit `dir` carry out through
/// the top, per channel, over `walks` walks of at most `max_steps` steps.
pub fn top_exit(
    thickness: f64,
    medium: &Medium,
    dir: DVec3,
    walks: usize,
    max_steps: u32,
    seed: u64,
) -> DVec3 {
    let shape = Shape::Slab { thickness };
    mean_of(walks, seed, &|| (), |_, rng| {
        let mut sum = DVec3::ZERO;
        for c in 0..3 {
            if let Some((n, t)) = walk_channel(shape, medium, c, DVec3::ZERO, dir, max_steps, rng)
                && n.z > 0.0
            {
                sum[c] += t;
            }
        }
        sum
    })
}

/// Fresnel reflectance of a smooth interface at incidence cosine `c`, where the far side's IOR
/// is `eta` times the near side's.
pub fn fresnel(eta: f64, c: f64) -> f64 {
    let s2 = (1.0 - c * c) / (eta * eta);
    if s2 >= 1.0 {
        return 1.0;
    }
    let t = (1.0 - s2).sqrt();
    let rs = (c - eta * t) / (c + eta * t);
    let rp = (t - eta * c) / (t + eta * c);
    0.5 * (rs * rs + rp * rp)
}

/// Uniform light's radiance, and none of any other.
fn uniform_radiance(light: Light) -> DVec3 {
    match light {
        Light::Uniform { radiance } => radiance,
        _ => DVec3::ZERO,
    }
}

/// The interior of a dielectric holding `medium`, in air, with the real-time walk's rules: the
/// camera's ray refracts in, the walk reflects back or refracts out wherever it meets the surface,
/// by Fresnel where the surface is smooth and by adobe's BSDF of the rough dielectric seen from
/// inside where it is rough, its way out sees uniform light as it is, and a sun as Lambert
/// reflection on the outward normal times the Fresnel transmission from the sun over that
/// transmission's cosine-weighted mean, the exit lobe, but where the walk never scattered: the
/// camera's ray passed through, and its way out meets a sun only along itself. With `exact_sun` a
/// rough surface lets a sun
/// in as the reference does, through its BSDF from inside wherever the walk meets it, and a
/// smooth one lets a disc in where the walk's way out points into it. With `conditioned` a smooth
/// surface's way out weighs a sun by the exit lobe conditioned on the walk's last scatter
/// ([`DielectricWalks::conditioned`]), and a way out that no scatter started takes none of it,
/// as a mirror's.
pub struct DielectricWalks {
    pub shape: Shape,
    pub medium: Medium,
    /// The surface's GGX roughness; 0 for a smooth surface.
    pub roughness: f64,
    pub exact_sun: bool,
    pub conditioned: bool,
}

impl DielectricWalks {
    /// adobe's inputs of the surface: a plain transmitting dielectric of the medium's IOR, whose
    /// color lies in the medium.
    fn surface(&self) -> Inputs {
        Inputs {
            specular_roughness: self.roughness as f32,
            specular_ior: self.medium.surface_ior as f32,
            transmission_weight: 1.0,
            ..Inputs::default()
        }
    }

    /// The Fresnel transmission from a sun at cosine `cos` over its cosine-weighted mean: the
    /// real-time way out's weight of a sun.
    fn exit_lobe(&self, cos: f64) -> f64 {
        (1.0 - fresnel(self.medium.surface_ior, cos)) / self.exit_mean()
    }

    /// The Fresnel transmission from outside's cosine-weighted mean.
    fn exit_mean(&self) -> f64 {
        let ior = self.medium.surface_ior;
        (0..16)
            .map(|i| 1.0 - fresnel(ior, ((f64::from(i) + 0.5) / 16.0).sqrt()))
            .sum::<f64>()
            / 16.0
    }

    /// The exit lobe of a smooth surface seen from inside, at cosine `cos` to its normal: the
    /// Fresnel transmission out times the cosine, over its integral over the hemisphere inside,
    /// which is the outside's over the IOR squared.
    fn inner_lobe(&self, cos: f64) -> f64 {
        let ior = self.medium.surface_ior;
        (1.0 - fresnel(1.0 / ior, cos)) * cos * ior * ior / (PI * self.exit_mean())
    }

    /// What the exit lobe of channel `c`'s way out `e` is weighed by toward a sun along `s`,
    /// where the walk's last step started from the scatter at `at` reached along `into`: the
    /// density with which that scatter sends the walk out through the plane of the way out toward
    /// `s` over the density it sent it out along its own way, times the lobe's at that way over
    /// the lobe's toward `s`. Where the surface is that plane, the weighed lobe is what the
    /// scatter sends out exactly, whichever direction the walk's next steps lean.
    pub fn conditioned(&self, c: usize, e: &WalkExit, at: DVec3, into: DVec3, s: DVec3) -> f64 {
        let m = &self.medium;
        let n = e.normal;
        let toward = inside_toward(n, m.surface_ior, s);
        let (cos_toward, cos_own) = (toward.dot(n), e.dir.dot(n));
        let depth = (e.point - at).dot(n);
        let (g, sigma) = (m.anisotropy, m.extinction[c]);
        hg(g, into.dot(toward)) / hg(g, into.dot(e.dir))
            * (-sigma * depth * (1.0 / cos_toward - 1.0 / cos_own)).exp()
            * cos_own
            / cos_toward
    }

    /// One estimate, one walk per channel, of the radiance toward `view` at boundary point
    /// `p0` with outward normal `n0` that crosses the surface from inside.
    pub fn radiance(
        &self,
        bsdf: &mut Reference,
        p0: DVec3,
        n0: DVec3,
        view: DVec3,
        light: Light,
        rng: &mut Rng,
    ) -> DVec3 {
        if self.roughness > 0.0 {
            return self.rough_radiance(bsdf, p0, n0, view, light, rng);
        }
        assert!(
            !(self.exact_sun && matches!(light, Light::Sun { .. })),
            "a smooth surface lets only a disc in along its way out"
        );
        let c0 = n0.dot(view);
        let ior = self.medium.surface_ior;
        let eta = 1.0 / ior;
        let cos_t = (1.0 - eta * eta * (1.0 - c0 * c0)).sqrt();
        let entry = (-view * eta + n0 * (eta * c0 - cos_t)).normalize();
        let weight = 1.0 - fresnel(ior, c0);
        let mut out = DVec3::ZERO;
        for c in 0..3 {
            let Some(e) = walk_exit(self.shape, &self.medium, c, p0, entry, u32::MAX, rng) else {
                continue;
            };
            let n = e.normal;
            let exit = match light {
                Light::Disc {
                    direction,
                    illuminance,
                    radius,
                } if self.exact_sun => {
                    let cosine = e.dir.dot(n);
                    let t = (1.0 - ior * ior * (1.0 - cosine * cosine)).sqrt();
                    let out = (e.dir * ior + n * (t - ior * cosine)).normalize();
                    if out.dot(direction) > radius.cos() {
                        illuminance[c] / (PI * radius.sin().powi(2))
                    } else {
                        0.0
                    }
                }
                Light::Uniform { radiance } => radiance[c],
                _ => {
                    let (direction, illuminance) = light.sun().unwrap();
                    let cos = n.dot(direction);
                    let lobe = if cos > 0.0 {
                        self.exit_lobe(cos) * cos / PI * illuminance[c]
                    } else {
                        0.0
                    };
                    match (self.conditioned, e.last) {
                        (false, _) if e.scattered => lobe,
                        (true, Some((at, into))) if lobe > 0.0 => {
                            lobe * self.conditioned(c, &e, at, into, direction)
                        }
                        _ => 0.0,
                    }
                }
            };
            out[c] = weight * e.throughput * exit;
        }
        out
    }

    /// [`Self::radiance`] through a rough surface: the camera's ray enters along a transmitted
    /// draw of adobe's BSDF from the view, weighted by it.
    fn rough_radiance(
        &self,
        bsdf: &mut Reference,
        p0: DVec3,
        n0: DVec3,
        view: DVec3,
        light: Light,
        rng: &mut Rng,
    ) -> DVec3 {
        let inputs = self.surface();
        bsdf.prepare(&inputs, to_local(view, n0).as_vec3().to_array());
        let u = [rng.f32(), rng.f32(), rng.f32()];
        let Some(s) = bsdf.sample(u).filter(|s| s.wi[2] < 0.0) else {
            return DVec3::ZERO;
        };
        let entry = to_world(DVec3::from_array(s.wi.map(f64::from)), n0).normalize();
        let weight = DVec3::from_array(s.weight.map(f64::from));
        let mut out = DVec3::ZERO;
        for c in 0..3 {
            out[c] = weight[c] * self.rough_walk(bsdf, c, p0, entry, light, rng);
        }
        out
    }

    /// What channel `c`'s walk from `p` along unit `w` carries out through the rough surface
    /// under `light`, scattered by the surface's BSDF from inside wherever it meets it.
    fn rough_walk(
        &self,
        bsdf: &mut Reference,
        c: usize,
        mut p: DVec3,
        mut w: DVec3,
        light: Light,
        rng: &mut Rng,
    ) -> f64 {
        let inputs = self.surface();
        let (sigma, albedo) = (self.medium.extinction[c], self.medium.albedo[c]);
        let mut throughput = 1.0;
        let mut out = 0.0;
        // The direction the walk reached the scatter its step started from along and the walk's
        // throughput there, where its step started from one.
        let mut last: Option<(DVec3, f64)> = None;
        loop {
            let t = -(1.0 - rng.f64()).ln() / sigma;
            let (tb, nb) = self.shape.exit(p, w);
            if t >= tb {
                p += w * tb;
                bsdf.prepare(&inputs, to_local(-w, nb).as_vec3().to_array());
                if let Some((direction, illuminance)) = light.sun()
                    && self.exact_sun
                    && nb.dot(direction) > 0.0
                {
                    let f = bsdf.eval(to_local(direction, nb).as_vec3().to_array());
                    out += throughput * f64::from(f.total()[c]) * illuminance[c];
                }
                let arrived = throughput;
                let u = [rng.f32(), rng.f32(), rng.f32()];
                let Some(s) = bsdf.sample(u) else {
                    return out;
                };
                throughput *= f64::from(s.weight[c]);
                let wi = to_world(DVec3::from_array(s.wi.map(f64::from)), nb).normalize();
                if s.wi[2] > 0.0 {
                    if let Some((direction, illuminance)) = light.sun()
                        && self.conditioned
                    {
                        // Fresh draws from here until one leaves, as many as one over the chance
                        // that a draw leaves on average.
                        let mut draws = 1.0;
                        while !bsdf
                            .sample([rng.f32(), rng.f32(), rng.f32()])
                            .is_some_and(|s| s.wi[2] > 0.0)
                        {
                            draws += 1.0;
                        }
                        let sun = match last {
                            // The scatter's light through the plane over the density of the step
                            // to here, the share of it that a smooth surface's exit lobe from
                            // inside gives: none past the critical angle, where the draws are many
                            // and the steps rare.
                            Some((into, scattered)) => {
                                let density =
                                    hg(self.medium.anisotropy, into.dot(w)) * (-sigma * tb).exp();
                                scattered * self.inner_lobe(w.dot(nb)) / density
                                    * self.through_plane(
                                        bsdf,
                                        c,
                                        into,
                                        nb,
                                        tb * w.dot(nb),
                                        direction,
                                        rng,
                                    )
                            }
                            None if nb.dot(direction) > 0.0 => {
                                let f = bsdf.eval(to_local(direction, nb).as_vec3().to_array());
                                arrived * f64::from(f.total()[c])
                            }
                            None => 0.0,
                        };
                        return out
                            + draws * sun * illuminance[c]
                            + throughput * uniform_radiance(light)[c];
                    }
                    return out
                        + throughput
                            * match light.sun() {
                                None => uniform_radiance(light)[c],
                                Some(_) if self.exact_sun => 0.0,
                                Some((direction, illuminance)) => {
                                    let cos = nb.dot(direction);
                                    if cos > 0.0 {
                                        self.exit_lobe(cos) * cos / PI * illuminance[c]
                                    } else {
                                        0.0
                                    }
                                }
                            };
                }
                w = wi;
                last = None;
                continue;
            }
            p += w * t;
            throughput *= albedo;
            if throughput < 0.05 {
                if rng.f64() >= throughput / 0.05 {
                    return out;
                }
                throughput = 0.05;
            }
            last = Some((w, throughput));
            w = sample_hg(w, self.medium.anisotropy, rng);
        }
    }

    /// One estimate of what channel `c`'s walk from a scatter `depth` below the plane of the rough
    /// surface's outward normal `n`, reached along `into`, carries out through that plane toward a
    /// sun along `s` on its next step, unit illuminance: a way in from the sun, through a visible
    /// normal seen from it or, for a share of the draws, evenly over the hemisphere, where the
    /// multiple scattering's transmission reaches too, and the surface's BSDF from inside along
    /// that way.
    #[allow(clippy::too_many_arguments)]
    fn through_plane(
        &self,
        bsdf: &mut Reference,
        c: usize,
        into: DVec3,
        n: DVec3,
        depth: f64,
        s: DVec3,
        rng: &mut Rng,
    ) -> f64 {
        const EVEN: f64 = 0.1;
        let sl = to_local(s, n);
        if sl.z <= 0.0 {
            return 0.0;
        }
        let ior = self.medium.surface_ior;
        let alpha = self.roughness * self.roughness;
        // The density of the way `r` in, below the plane, as the sun refracted through a visible
        // normal.
        let visible = |r: DVec3| {
            let h = -(sl + r * ior);
            let h = (h * h.z.signum()).normalize();
            let (sh, rh) = (sl.dot(h), r.dot(h));
            if sh <= 0.0 || rh >= 0.0 {
                return 0.0;
            }
            ggx_g1(alpha, sl) * sh * ggx_d(alpha, h) / sl.z * ior * ior * -rh
                / (sh + ior * rh).powi(2)
        };
        let r = if rng.f64() < EVEN {
            let (z, phi) = (rng.f64(), TAU * rng.f64());
            let sine = (1.0 - z * z).max(0.0).sqrt();
            DVec3::new(sine * phi.cos(), sine * phi.sin(), -z)
        } else {
            let h = sample_vndf(alpha, sl, [rng.f64(), rng.f64()]);
            let sh = sl.dot(h);
            let eta = 1.0 / ior;
            let k = 1.0 - eta * eta * (1.0 - sh * sh);
            -sl * eta + h * (eta * sh - k.sqrt())
        };
        let pdf = EVEN / TAU + (1.0 - EVEN) * visible(r);
        let toward = -to_world(r, n).normalize();
        let cosine = toward.dot(n);
        if cosine <= 0.0 {
            return 0.0;
        }
        bsdf.prepare(&self.surface(), to_local(-toward, n).as_vec3().to_array());
        let f = f64::from(bsdf.eval(sl.as_vec3().to_array()).total()[c]);
        hg(self.medium.anisotropy, into.dot(toward))
            * (-self.medium.extinction[c] * depth / cosine).exp()
            * f
            / pdf
    }

    /// The mean of [`Self::radiance`] over `count` estimates, each at a boundary point, its
    /// outward normal and the view there from `point`.
    pub fn mean(
        &self,
        count: usize,
        light: Light,
        seed: u64,
        point: impl Fn(&mut Rng) -> (DVec3, DVec3, DVec3) + Sync,
    ) -> DVec3 {
        mean_of(count, seed, &Reference::new, |bsdf, rng| {
            let (p, n, v) = point(rng);
            self.radiance(bsdf, p, n, v, light, rng)
        })
    }
}

/// A cosine-distributed direction about +z.
pub fn cosine_direction(rng: &mut Rng) -> DVec3 {
    let r = rng.f64().sqrt();
    let phi = TAU * rng.f64();
    DVec3::new(r * phi.cos(), r * phi.sin(), (1.0 - r * r).max(0.0).sqrt())
}

/// In a furnace, a thick slab behind an index-matched interface returns the subsurface
/// color, as adobe's volume mapping intends. A dense interior's way out, conditioned on the walk's
/// last scatter, lets in what the walks' own ways out send into a sun's disc where its surface is
/// smooth, and what next-event estimation at every meeting takes where it is rough.
#[cfg(test)]
mod tests {
    use super::*;

    /// A rough jade slab's way out, seen from 45 degrees, weighs a sun by the walk's last scatter
    /// through the surface's plane: on a half space under a low sun opposite the view, and
    /// through a slab 0.3 m thick lit from behind, within noise of the light the surface lets in
    /// wherever a walk meets it, whose second moment is some fifteen times as large.
    #[test]
    fn a_rough_way_out_through_the_plane_lets_in_what_every_meeting_does() {
        let medium = Medium {
            extinction: DVec3::new(7.5, 4.3, 5.9),
            albedo: DVec3::new(0.64, 0.93, 0.76),
            anisotropy: 0.3,
            surface_ior: 1.6,
        };
        let view = DVec3::new(0.0, 1.0, 1.0).normalize();
        let cases = [
            (f64::INFINITY, DVec3::new(0.0, -0.87, 0.5).normalize()),
            (0.3, DVec3::NEG_Z),
        ];
        for (i, (thickness, direction)) in cases.into_iter().enumerate() {
            let walks = |exact_sun, conditioned| DielectricWalks {
                shape: Shape::Slab { thickness },
                medium,
                roughness: 0.35,
                exact_sun,
                conditioned,
            };
            let sun = Light::Sun {
                direction,
                illuminance: DVec3::ONE,
            };
            let at = |_: &mut Rng| (DVec3::ZERO, DVec3::Z, view);
            let exact = walks(true, false).mean(32_000_000, sun, 7 + i as u64, at);
            let plane = walks(false, true).mean(8_000_000, sun, 9 + i as u64, at);
            let error = (plane / exact - 1.0).abs().max_element();
            assert!(
                error < 0.015,
                "slab {thickness} m, sun {direction}: through the plane {plane}, exact {exact}"
            );
        }
    }

    /// The way out of a smooth jade slab, seen from 45 degrees, under suns the plain exit lobe
    /// gets wrong: on a half space 6 % too bright under a high sun and 10 % too dark under a low
    /// one opposite the view, and through a slab 0.3 m thick lit from behind, two to three times
    /// too bright. The conditioned lobe keeps within 3 % of what the walks' ways out send into a
    /// disc of 0.1 rad on the half space; through the thin slab the walks that reflect between its
    /// faces after their last scatter leave a sun to their way out alone, which keeps 4 %.
    #[test]
    fn a_smooth_way_out_conditioned_on_the_last_scatter_lets_in_what_the_walks_send_out() {
        let medium = Medium {
            extinction: DVec3::new(7.5, 4.3, 5.9),
            albedo: DVec3::new(0.64, 0.93, 0.76),
            anisotropy: 0.3,
            surface_ior: 1.6,
        };
        let view = DVec3::new(0.0, 1.0, 1.0).normalize();
        let sun = |x: f64, y: f64, z: f64| DVec3::new(x, y, z).normalize();
        let cases = [
            (f64::INFINITY, sun(0.0, 0.5, 0.87), 0.03),
            (f64::INFINITY, sun(0.0, -0.87, 0.5), 0.03),
            (0.3, sun(0.0, 0.0, -1.0), 0.04),
            (0.3, sun(0.0, 0.87, -0.5), 0.04),
        ];
        for (i, (thickness, direction, tolerance)) in cases.into_iter().enumerate() {
            let walks = |exact_sun, conditioned| DielectricWalks {
                shape: Shape::Slab { thickness },
                medium,
                roughness: 0.0,
                exact_sun,
                conditioned,
            };
            let illuminance = DVec3::ONE;
            let disc = Light::Disc {
                direction,
                illuminance,
                radius: 0.1,
            };
            let sun = Light::Sun {
                direction,
                illuminance,
            };
            let count = 16_000_000;
            let at = |_: &mut Rng| (DVec3::ZERO, DVec3::Z, view);
            let exact = walks(true, false).mean(count, disc, 3 + i as u64, at);
            let conditioned = walks(false, true).mean(count, sun, 5 + i as u64, at);
            let error = (conditioned / exact - 1.0).abs().max_element();
            assert!(
                error < tolerance,
                "slab {thickness} m, sun {direction}: conditioned {conditioned}, exact {exact}"
            );
        }
    }

    #[test]
    fn furnace_slab_returns_the_subsurface_color() {
        for c in [0.2f32, 0.5, 0.8, 0.95] {
            let inputs = Inputs {
                subsurface_weight: 1.0,
                subsurface_color: [c; 3],
                subsurface_radius: 0.01,
                subsurface_radius_scale: [1.0; 3],
                specular_weight: 0.0,
                ..Inputs::default()
            };
            let interface = Interface::new(&inputs, 4_000);
            let walks = Walks::new(
                &inputs,
                Shape::Slab {
                    thickness: f64::INFINITY,
                },
                &interface,
            );
            let light = Light::Uniform {
                radiance: DVec3::ONE,
            };
            let r = walks.mean(400_000, light, 3, |rng| {
                (DVec3::ZERO, DVec3::Z, cosine_direction(rng))
            });
            let error = (r.x / f64::from(c) - 1.0).abs();
            assert!(error < 0.01, "C {c}: the slab returns {r:?}");
        }
    }
}
