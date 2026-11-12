//! Lights as the API takes them and their table records.

use bytemuck::Zeroable;
use orr_gpu::TextureIndex;
use orr_math::{DVec3, IVec3, Vec3};

use crate::LightGpu;
use crate::material::rec2020;
use crate::world::{WorldPos, render_position};

/// A light of one scene: its entry in the scene's light table, which grows to hold the highest
/// id set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LightId(pub u32);

impl LightId {
    pub const NONE: Self = Self(u32::MAX);
}

/// What a scene's light tables hold room for in a frame, as its root says: state kept per
/// light, per emissive list entry or per producer entry by whoever reads the tables is sized
/// from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LightTables {
    /// Entries of the light table: every `LightId` set is below it.
    pub lights: u32,
    /// Entries of the emissive list, `SceneRoot::emissive_capacity`.
    pub entries: u32,
    /// Entries of the producer table, `SceneRoot::producer_capacity`.
    pub producers: u32,
}

/// A light. Colors are linear Rec.709 and convert to Rec.2020 when written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    Directional(DirectionalLight),
    /// A sphere emitting `intensity` candela, optionally limited to a cone. Only the BSDF rays
    /// of paths see its body: the camera does not, and it casts no shadow.
    Sphere {
        position: DVec3,
        radius: f32,
        intensity: Vec3,
        cone: Option<Cone>,
    },
    /// A one-sided rectangle emitting `luminance` nits towards `edge_x` x `edge_y`. Only the
    /// BSDF rays of paths see it: the camera does not, and it casts no shadow.
    Rect {
        center: DVec3,
        edge_x: Vec3,
        edge_y: Vec3,
        luminance: Vec3,
    },
}

/// A distant disc of finite size: a sun, a moon or any other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Unit direction from a receiver towards the light.
    pub direction_to_light: Vec3,
    /// Lux at a receiver facing the light, before the scene's medium.
    pub illuminance: Vec3,
    /// Radians, above 0.
    pub angular_radius: f32,
    pub disc: DiscProfile,
}

/// How radiance is spread over a directional light's disc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiscProfile {
    Uniform,
    LimbDarkened,
    /// A sphere lit by up to four other directional lights (`LightId::NONE` = unused),
    /// with `albedo` mapped onto the disc and `up` orienting it.
    Textured {
        albedo: TextureIndex,
        up: Vec3,
        lit_by: [LightId; 4],
    },
}

/// A spotlight cone: full intensity within `inner` of `direction`, none beyond `outer`
/// (half-angles in radians).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cone {
    pub direction: Vec3,
    pub inner: f32,
    pub outer: f32,
}

/// `LightGpu::kind`.
const LIGHT_DIRECTIONAL: u32 = 1;
const LIGHT_SPHERE: u32 = 2;
const LIGHT_RECT: u32 = 3;

/// `EmissiveGpu::kind`.
pub(crate) const EMISSIVE_INSTANCE: u32 = 1;
pub(crate) const EMISSIVE_PRODUCER: u32 = 2;
/// `EmissiveGpu::set` of an entry that emits nothing this frame.
pub(crate) const NO_SET: u32 = !0;
/// `lights.slang` `GATHERED_CLUSTERS`: clusters an instance entry's gathered list holds.
pub(crate) const GATHERED_CLUSTERS: u32 = 16;
/// An instance entry's gathered list: its cluster ids, then their CDF (`gathered_list`).
pub(crate) type GatheredList = [u32; 2 * GATHERED_CLUSTERS as usize];
/// `InstanceGpu::first_emissive` of a copy with no emissive list entries.
pub(crate) const NO_EMISSIVE: u32 = !0;

/// The table record of `light` in the render space of the origin at `origin_cell`.
pub(crate) fn light_gpu(light: &Light, origin_cell: IVec3, cell_log2: u32) -> LightGpu {
    let mut g = LightGpu::zeroed();
    g.cone_cos_inner = -1.0;
    g.cone_cos_outer = -1.0;
    g.lit_by = [u32::MAX; 4];
    let mut place = |p: DVec3| {
        let w = WorldPos::new(p, cell_log2);
        g.cell = w.cell.to_array();
        g.offset = w.offset.to_array();
        g.position = render_position(p, origin_cell, cell_log2).to_array();
    };
    match *light {
        Light::Directional(d) => {
            g.kind = LIGHT_DIRECTIONAL;
            g.radius = d.angular_radius;
            g.direction = d.direction_to_light.normalize().to_array();
            g.color = rec2020(d.illuminance);
            match d.disc {
                DiscProfile::Uniform => g.disc = 0,
                DiscProfile::LimbDarkened => g.disc = 1,
                DiscProfile::Textured { albedo, up, lit_by } => {
                    g.disc = 2;
                    g.albedo = albedo;
                    g.up = up.to_array();
                    g.lit_by = lit_by.map(|l| l.0);
                }
            }
        }
        Light::Sphere {
            position,
            radius,
            intensity,
            cone,
        } => {
            place(position);
            g.kind = LIGHT_SPHERE;
            g.radius = radius;
            g.color = rec2020(intensity);
            if let Some(c) = cone {
                g.direction = c.direction.normalize().to_array();
                g.cone_cos_inner = c.inner.cos();
                g.cone_cos_outer = c.outer.cos();
            }
        }
        Light::Rect {
            center,
            edge_x,
            edge_y,
            luminance,
        } => {
            place(center);
            g.kind = LIGHT_RECT;
            g.edge_x = edge_x.to_array();
            g.edge_y = edge_y.to_array();
            g.color = rec2020(luminance);
        }
    }
    g
}

/// The lights a scene holds by kind, from what was last set for each id.
#[derive(Default)]
pub(crate) struct LightKinds {
    /// By id: whether the light is directional.
    set: std::collections::HashMap<u32, bool>,
    directional: u32,
}

impl LightKinds {
    /// Records what light `id` now is.
    pub(crate) fn set(&mut self, id: u32, light: Option<&Light>) {
        let directional = light.map(|l| matches!(l, Light::Directional(_)));
        let was = match directional {
            Some(d) => self.set.insert(id, d),
            None => self.set.remove(&id),
        };
        self.directional =
            self.directional + u32::from(directional == Some(true)) - u32::from(was == Some(true));
    }

    pub(crate) fn features(&self) -> crate::SceneFeatures {
        let mut out = crate::SceneFeatures::NONE;
        if self.directional > 0 {
            out |= crate::SceneFeatures::DIRECTIONAL_LIGHTS;
        }
        if self.set.len() as u32 > self.directional {
            out |= crate::SceneFeatures::LOCAL_LIGHTS;
        }
        out
    }
}
