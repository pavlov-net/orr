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

/// What a scene's light tables hold room for in a frame: state kept per light, per emissive
/// list entry, per producer entry or per producer cluster by whoever reads the tables is sized
/// from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LightTables {
    /// Entries of the light table: every `LightId` set is below it.
    pub lights: u32,
    /// Entries of the emissive list, `SceneRoot::emissive_capacity`.
    pub entries: u32,
    /// Entries of the producer table, `SceneRoot::producer_capacity`.
    pub producers: u32,
    /// Clusters the set producers' batches hold between them at most, and room to spare: it
    /// grows as the tables do, and no frame's producers emit more.
    pub producer_clusters: u32,
}

/// A light. Colors are linear Rec.709 and convert to Rec.2020 when written. A point, sphere or
/// rect light without `shadows` lights what lies behind other surfaces as if they were not
/// there: only next-event estimation carries it, no ray meets its body, and no shadow ray is
/// traced toward it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    Directional(DirectionalLight),
    /// A point emitting `intensity` candela, shaped by `shaping`. Only next-event estimation
    /// reaches it: no ray meets it, and its body casts no shadow.
    Point {
        position: DVec3,
        intensity: Vec3,
        shaping: Option<Shaping>,
        shadows: bool,
    },
    /// A sphere emitting `intensity` candela, shaped by `shaping`. Only the BSDF rays of paths
    /// see its body: the camera does not, and its body casts no shadow. It stays dark where f32
    /// cannot hold its radiance, `intensity` over π `radius`², or `radius`² as a normal number.
    Sphere {
        position: DVec3,
        radius: f32,
        intensity: Vec3,
        shaping: Option<Shaping>,
        shadows: bool,
    },
    /// A one-sided rectangle emitting `luminance` nits towards `edge_x` x `edge_y`, shaped by
    /// `shaping`. Only the BSDF rays of paths see it: the camera does not, and its body casts no
    /// shadow.
    Rect {
        center: DVec3,
        edge_x: Vec3,
        edge_y: Vec3,
        luminance: Vec3,
        shaping: Option<Shaping>,
        shadows: bool,
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

/// How a light's emission falls off around its axis, as UsdLux's `ShapingAPI` defines it. In
/// a direction `θ` off the axis, emission is scaled by `lerp(focus_tint, 1, |cos θ|^focus)` and
/// by `1 - smoothstep(θ, cone_angle (1 - cone_softness), cone_angle)`, and is 0 past
/// `cone_angle`. The direction is the one light leaves along: from the point of the light it
/// leaves toward what it lights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shaping {
    /// Unit direction the light's emission centers on.
    pub axis: Vec3,
    /// Radians off the axis past which nothing is emitted.
    pub cone_angle: f32,
    /// The share of the cone, in [0, 1], over which emission falls to 0 toward its edge.
    pub cone_softness: f32,
    /// At least 0; 0 leaves emission unscaled.
    pub focus: f32,
    pub focus_tint: Vec3,
}

/// `LightGpu::kind`.
const LIGHT_DIRECTIONAL: u32 = 1;
const LIGHT_SPHERE: u32 = 2;
const LIGHT_RECT: u32 = 3;
const LIGHT_POINT: u32 = 4;

/// `LightGpu::flags`: surfaces cast no shadow from the light.
const LIGHT_UNSHADOWED: u32 = 1;

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
    g.lit_by = [u32::MAX; 4];
    let mut place = |p: DVec3| {
        let w = WorldPos::new(p, cell_log2);
        g.cell = w.cell.to_array();
        g.offset = w.offset.to_array();
        g.position = render_position(p, origin_cell, cell_log2).to_array();
    };
    let shaping = match *light {
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
            None
        }
        Light::Point {
            position,
            intensity,
            shaping,
            shadows,
        } => {
            place(position);
            g.kind = LIGHT_POINT;
            g.color = rec2020(intensity);
            g.flags = if shadows { 0 } else { LIGHT_UNSHADOWED };
            shaping
        }
        Light::Sphere {
            position,
            radius,
            intensity,
            shaping,
            shadows,
        } => {
            place(position);
            g.kind = LIGHT_SPHERE;
            g.radius = radius;
            g.color = rec2020(intensity);
            g.flags = if shadows { 0 } else { LIGHT_UNSHADOWED };
            shaping
        }
        Light::Rect {
            center,
            edge_x,
            edge_y,
            luminance,
            shaping,
            shadows,
        } => {
            place(center);
            g.kind = LIGHT_RECT;
            g.edge_x = edge_x.to_array();
            g.edge_y = edge_y.to_array();
            g.color = rec2020(luminance);
            g.flags = if shadows { 0 } else { LIGHT_UNSHADOWED };
            shaping
        }
    };
    match shaping {
        Some(s) => {
            g.direction = s.axis.normalize().to_array();
            g.cone_angle = s.cone_angle;
            g.cone_start = s.cone_angle * (1.0 - s.cone_softness);
            g.focus = s.focus;
            g.focus_tint = rec2020(s.focus_tint);
        }
        // No direction lies past π off an axis, and a cone of no smooth span scales none.
        None if g.kind != LIGHT_DIRECTIONAL => {
            g.cone_angle = std::f32::consts::PI;
            g.cone_start = std::f32::consts::PI;
        }
        None => {}
    }
    g
}

/// What casts a shadow from `light`: whether surfaces do.
pub(crate) fn casts_shadows(light: &Light) -> bool {
    match *light {
        Light::Directional(_) => true,
        Light::Point { shadows, .. }
        | Light::Sphere { shadows, .. }
        | Light::Rect { shadows, .. } => shadows,
    }
}

/// What a scene's kernels may meet of a light: the features it brings.
fn light_features(light: &Light) -> crate::SceneFeatures {
    use crate::SceneFeatures as F;
    let (local, shaped) = match light {
        Light::Directional(_) => return F::DIRECTIONAL_LIGHTS,
        Light::Point { shaping, .. }
        | Light::Sphere { shaping, .. }
        | Light::Rect { shaping, .. } => (F::LOCAL_LIGHTS, shaping.is_some()),
    };
    let mut out = local;
    if shaped {
        out |= F::SHAPED_LIGHTS;
    }
    if !casts_shadows(light) {
        out |= F::UNSHADOWED_LIGHTS;
    }
    out
}

/// The lights a scene holds by what they bring its kernels, from what was last set for each id.
#[derive(Default)]
pub(crate) struct LightKinds {
    /// By id: the light's [`light_features`].
    set: std::collections::HashMap<u32, crate::SceneFeatures>,
    /// Per feature bit, the lights that bring it.
    counts: std::collections::HashMap<u64, u32>,
}

impl LightKinds {
    /// Records what light `id` now is.
    pub(crate) fn set(&mut self, id: u32, light: Option<&Light>) {
        let now = light.map(light_features);
        let was = match now {
            Some(f) => self.set.insert(id, f),
            None => self.set.remove(&id),
        };
        let bits = |f: Option<crate::SceneFeatures>| {
            let f = f.map_or(0, |f| f.0);
            (0..64).map(|b| 1u64 << b).filter(move |b| f & b != 0)
        };
        for b in bits(was) {
            *self.counts.get_mut(&b).expect("a counted feature") -= 1;
        }
        for b in bits(now) {
            *self.counts.entry(b).or_default() += 1;
        }
    }

    pub(crate) fn features(&self) -> crate::SceneFeatures {
        let held = self.counts.iter().filter(|(_, n)| **n > 0);
        crate::SceneFeatures(held.fold(0, |out, (&b, _)| out | b))
    }
}
