//! The examples' cooked sky: a gradient from a bright horizon to a blue zenith with a glow
//! around the sun, over a dim ground, cooked as an environment map.

use std::path::PathBuf;

use orr::orr_gpu::{AddressMode, Allocation, Device, Filter, MemoryKind, SamplerDesc};
use orr::orr_lighting::EnvironmentMapGpu;
use orr::orr_pack::AssetId;
use orr::{MountId, Renderer};
use orr_cook::{AssetSpec, Quality, Source, TextureSettings, Usage};
use orr_math::Vec3;

/// A sky with its sun along unit `sun`, written to a temp file for the cook to read.
pub struct Sky {
    dir: PathBuf,
    path: PathBuf,
}

impl Sky {
    /// Writes the sky as a 1024 x 512 equirectangular Radiance HDR file into a temp
    /// directory named after `name`.
    pub fn new(name: &str, sun: Vec3) -> Self {
        let dir = std::env::temp_dir().join(format!("orr-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("sky.hdr");
        let (w, h) = (1024u32, 512u32);
        let mut bytes =
            format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                let (theta, phi) = (
                    (y as f32 + 0.5) / h as f32 * std::f32::consts::PI,
                    (x as f32 + 0.5) / w as f32 * std::f32::consts::TAU - std::f32::consts::PI,
                );
                let d = Vec3::new(
                    theta.sin() * phi.sin(),
                    theta.cos(),
                    -theta.sin() * phi.cos(),
                );
                bytes.extend(rgbe(radiance(d, sun)));
            }
        }
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
        Self { dir, path }
    }

    /// The sky as the cook's environment asset `name`.
    pub fn asset(&self, name: &str) -> AssetSpec {
        AssetSpec {
            name: name.into(),
            source: Source::Texture {
                path: self.path.clone(),
                settings: TextureSettings {
                    usage: Usage::Environment {
                        light_threshold: None,
                        max_lights: 0,
                    },
                    quality: Quality::Fast,
                    ..TextureSettings::default()
                },
            },
        }
    }
}

impl Drop for Sky {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The environment record of the sky cooked as `name` into `mount`, for
/// `SceneWriter::set_environment`.
pub fn environment(
    renderer: &Renderer,
    gpu: &Device,
    mount: MountId,
    name: &str,
) -> Allocation<EnvironmentMapGpu> {
    let mut environment = gpu
        .gpu_malloc::<EnvironmentMapGpu>(MemoryKind::Upload, 1, "sky environment")
        .expect("environment");
    environment.as_mut_slice().expect("environment")[0] = EnvironmentMapGpu {
        map: renderer
            .texture(mount, AssetId::from_name(name))
            .expect("sky texture"),
        sampler: gpu
            .sampler(&SamplerDesc {
                mag: Filter::Linear,
                min: Filter::Linear,
                mip: Filter::Linear,
                address: [AddressMode::ClampToEdge; 3],
                ..SamplerDesc::default()
            })
            .expect("sky sampler"),
        intensity: 1.0,
        primaries: orr_color::Primaries::Rec709.id().into(),
        from_world: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };
    environment
}

/// Sky radiance in nits, linear Rec.709, along unit direction `d`.
fn radiance(d: Vec3, sun: Vec3) -> Vec3 {
    let glow = 6.0 * (60.0 * (d.dot(sun) - 1.0)).exp();
    if d.y < 0.0 {
        return Vec3::new(0.55, 0.52, 0.48) * 6000.0;
    }
    let horizon = Vec3::new(0.85, 0.9, 1.0) * 5000.0;
    let zenith = Vec3::new(0.2, 0.38, 1.0) * 4000.0;
    horizon.lerp(zenith, d.y.sqrt()) * (1.0 + glow)
}

/// Radiance's shared-exponent encoding of `c`.
fn rgbe(c: Vec3) -> [u8; 4] {
    let v = c.max_element();
    if v < 1e-30 {
        return [0; 4];
    }
    let e = v.log2().floor() as i32 + 1;
    let scale = 256.0 / 2f32.powi(e);
    let m = |x: f32| (x * scale).clamp(0.0, 255.0) as u8;
    [m(c.x), m(c.y), m(c.z), (e + 128) as u8]
}
