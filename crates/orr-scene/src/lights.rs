//! Lights as the API takes them and their table records.

use bytemuck::Zeroable;
use glam::{DVec3, IVec3, Vec3};
use orr_gpu::TextureIndex;

use crate::LightGpu;
use crate::world::{WorldPos, render_position};

/// A slot in a scene's light table, below `SceneDesc::light_capacity`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LightId(pub u32);

impl LightId {
    pub const NONE: Self = Self(u32::MAX);
}

/// A light. Colours are linear Rec.709 and convert to Rec.2020 when written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    Directional(DirectionalLight),
    /// A sphere emitting `intensity` candela, optionally limited to a cone.
    Sphere {
        position: DVec3,
        radius: f32,
        intensity: Vec3,
        cone: Option<Cone>,
    },
    /// A one-sided rectangle emitting `luminance` nits towards `edge_x` x `edge_y`.
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
pub(crate) const LIGHT_DIRECTIONAL: u32 = 1;
pub(crate) const LIGHT_SPHERE: u32 = 2;
pub(crate) const LIGHT_RECT: u32 = 3;

/// Linear Rec.709 to linear Rec.2020 (rows), as `orr.color`'s `REC709_TO_REC2020`.
const REC709_TO_REC2020: [[f32; 3]; 3] = [
    [0.627_403_9, 0.329_283_04, 0.043_313_066],
    [0.069_097_29, 0.919_540_4, 0.011_362_316],
    [0.016_391_438, 0.088_013_31, 0.895_595_25],
];

pub(crate) fn rec709_to_rec2020(c: Vec3) -> [f32; 3] {
    REC709_TO_REC2020.map(|row| Vec3::from(row).dot(c))
}

/// The table record of `light` in the render space of the origin at `origin_cell`.
pub(crate) fn light_gpu(light: &Light, origin_cell: IVec3, cell_log2: u32) -> LightGpu {
    let mut g = LightGpu::zeroed();
    g.cone_cos_inner = -1.0;
    g.cone_cos_outer = -1.0;
    g.albedo = TextureIndex::NONE;
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
            g.color = rec709_to_rec2020(d.illuminance);
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
            g.color = rec709_to_rec2020(intensity);
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
            g.color = rec709_to_rec2020(luminance);
        }
    }
    g
}
