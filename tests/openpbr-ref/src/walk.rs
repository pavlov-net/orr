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
use shader_maths::stats::Rng;

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

/// Where channel `c`'s walk from `p` along unit `w` inside `shape` reaches the boundary: the
/// outward normal there and the throughput. `None` when the walk is absorbed, or still
/// inside after `max_steps` steps.
pub fn walk_channel(
    shape: Shape,
    medium: &Medium,
    c: usize,
    mut p: DVec3,
    mut w: DVec3,
    max_steps: u32,
    rng: &mut Rng,
) -> Option<(DVec3, f64)> {
    let (sigma, albedo) = (medium.extinction[c], medium.albedo[c]);
    let mut throughput = 1.0;
    for _ in 0..max_steps {
        let t = -(1.0 - rng.f64()).ln() / sigma;
        let (tb, nb) = shape.exit(p, w);
        if t >= tb {
            return Some((nb, throughput));
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
        w = sample_hg(w, medium.anisotropy, rng);
    }
    None
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
/// (unit direction toward it and illuminance), or uniform radiance from every direction.
#[derive(Clone, Copy, Debug)]
pub enum Light {
    Sun {
        direction: DVec3,
        illuminance: DVec3,
    },
    Uniform {
        radiance: DVec3,
    },
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
            let exit = match light {
                Light::Sun {
                    direction,
                    illuminance,
                } => {
                    let cos = n.dot(direction);
                    if cos > 0.0 {
                        self.interface.at(cos)[c] * cos / PI * illuminance[c]
                    } else {
                        0.0
                    }
                }
                Light::Uniform { radiance } => self.interface.hemisphere()[c] * radiance[c],
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
        let chunks = 256;
        let per = count.div_ceil(chunks);
        let sums = in_parallel(chunks, &Reference::new, |bsdf, i| {
            let mut rng = rng(seed, i as u64);
            let mut sum = DVec3::ZERO;
            for _ in 0..per {
                let (p, n, v) = point(&mut rng);
                sum += self.radiance(bsdf, p, n, v, light, &mut rng);
            }
            sum
        });
        sums.into_iter().sum::<DVec3>() / (per * chunks) as f64
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
    let chunks = 256;
    let per = walks.div_ceil(chunks);
    let shape = Shape::Slab { thickness };
    let sums = in_parallel(chunks, &|| (), |_, i| {
        let mut rng = rng(seed, i as u64);
        let mut sum = DVec3::ZERO;
        for _ in 0..per {
            for c in 0..3 {
                if let Some((n, t)) =
                    walk_channel(shape, medium, c, DVec3::ZERO, dir, max_steps, &mut rng)
                    && n.z > 0.0
                {
                    sum[c] += t;
                }
            }
        }
        sum
    });
    sums.into_iter().sum::<DVec3>() / (per * chunks) as f64
}

/// A cosine-distributed direction about +z.
pub fn cosine_direction(rng: &mut Rng) -> DVec3 {
    let r = rng.f64().sqrt();
    let phi = TAU * rng.f64();
    DVec3::new(r * phi.cos(), r * phi.sin(), (1.0 - r * r).max(0.0).sqrt())
}

/// In a furnace, a thick slab behind an index-matched interface returns the subsurface
/// color, as adobe's volume mapping intends.
#[cfg(test)]
mod tests {
    use super::*;

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
