//! Helpers for the atmosphere suite: a scene with one atmosphere, frames that light it, and
//! the probe kernels that read it back.

use std::sync::OnceLock;

use glam::DVec3;
use orr::abi::{GpuPtr, SamplerIndex};
use orr::orr_gpu::{
    ComputePipeline, Format, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData, Stages,
    Texture, TextureDesc, TextureIndex, TextureMemory, TextureRegion, TextureUsage, ViewDesc,
};
use orr::{
    DirectionalLight, FrameDesc, FrameTime, Light, LightId, Renderer, RendererDesc, SceneDesc,
    SceneId, SceneRoot,
};
use orr_atmosphere::presets::solar_disc;
use orr_atmosphere::spectrum::{SOLAR_COLOUR, SOLAR_ILLUMINANCE};
use orr_atmosphere::{Atmosphere, AtmosphereGpu, AtmosphereState};
use orr_lighting::{LightingStats, SceneLighting, SceneLightingSettings, ScenePass};
use orr_scene::testing::{device, read_buffer, wait};

/// Generated kernel modules and types of the suite's Slang.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// orr's kernels with `AtmosphereSky` linked as the environment.
pub mod linked {
    include!(concat!(env!("OUT_DIR"), "/linked/orr_kernels.rs"));
}

use shaders::probe::{MODULE, Root};

pub mod prague;

pub const SUN: LightId = LightId(0);

/// Unit direction at an elevation and an azimuth from +x toward +z, in degrees.
pub fn dir(elevation: f64, azimuth: f64) -> DVec3 {
    let (e, a) = (elevation.to_radians(), azimuth.to_radians());
    DVec3::new(e.cos() * a.cos(), e.sin(), e.cos() * a.sin())
}

/// The sun above the atmosphere: Rec.2020 lux.
pub fn solar_rec2020() -> [f64; 3] {
    SOLAR_COLOUR.map(|c| c * SOLAR_ILLUMINANCE)
}

/// A renderer with orr's kernels linked to the sky, one scene and one atmosphere whose
/// candidate is [`SUN`], and optionally lighting's scene pass after the atmosphere's.
pub struct Rig {
    pub renderer: Renderer,
    pub scene: SceneId,
    pub sky: AtmosphereState,
    pub lighting: Option<SceneLighting>,
    frames: u32,
    frame_id: u64,
    lights: Vec<LightId>,
}

impl Rig {
    pub fn new(atmosphere: &Atmosphere) -> Self {
        let desc = RendererDesc {
            kernels: &linked::KERNELS,
            ..RendererDesc::default()
        };
        let mut renderer = Renderer::new(device(), desc).unwrap();
        let scene = renderer
            .create_scene(&SceneDesc::default(), &Default::default())
            .unwrap();
        let mut sky = AtmosphereState::new(device(), atmosphere, SamplerIndex(7)).unwrap();
        sky.set_lights(&[SUN]);
        Self {
            renderer,
            scene,
            sky,
            lighting: None,
            frames: 0,
            frame_id: 0,
            lights: Vec::new(),
        }
    }

    /// The same, with lighting's scene pass baking the linked sky every frame.
    pub fn with_lighting(atmosphere: &Atmosphere) -> Self {
        let lighting = SceneLighting::new(
            device(),
            &linked::KERNELS,
            &SceneDesc::default(),
            &SceneLightingSettings::default(),
        )
        .unwrap();
        Self {
            lighting: Some(lighting),
            ..Self::new(atmosphere)
        }
    }

    /// One frame with sunlight from `sun` (none: no light) and the focus at `focus`; waits
    /// for it and returns the scene root.
    pub fn frame(&mut self, sun: Option<DVec3>, focus: DVec3) -> GpuPtr<SceneRoot> {
        let lights: Vec<_> = sun
            .map(|d| (SUN, solar_disc(d.as_vec3())))
            .into_iter()
            .collect();
        self.frame_lights(&lights, focus)
    }

    /// One frame with exactly `lights` in the light table and the focus at `focus`; waits
    /// for it and returns the scene root.
    pub fn frame_lights(
        &mut self,
        lights: &[(LightId, DirectionalLight)],
        focus: DVec3,
    ) -> GpuPtr<SceneRoot> {
        self.frames += 1;
        let listed = [self.scene];
        let mut f = self
            .renderer
            .begin_frame(FrameDesc {
                time: FrameTime {
                    absolute: f64::from(self.frames) / 60.0,
                    delta: 1.0 / 60.0,
                },
                scenes: &listed,
                display: None,
            })
            .unwrap();
        self.frame_id = f.id();
        {
            let mut w = f.scene(self.scene).unwrap();
            for id in self.lights.drain(..) {
                w.set_light(id, None).unwrap();
            }
            for (id, l) in lights {
                w.set_light(*id, Some(&Light::Directional(*l))).unwrap();
                self.lights.push(*id);
            }
            w.set_focus(focus, 1000.0);
            w.set_environment(self.sky.data());
            w.set_medium(self.sky.data());
        }
        let mut root = None;
        let (sky, lighting, frame) = (&mut self.sky, &mut self.lighting, self.frame_id);
        f.update_scenes(&mut |cx| {
            root = Some(cx.root);
            sky.record(cx).unwrap();
            if let Some(lighting) = lighting.as_mut() {
                let pass = ScenePass {
                    frame,
                    scene: cx.root,
                    focus: focus.as_vec3().to_array(),
                };
                lighting.record(cx.cmd, &pass);
            }
        })
        .unwrap();
        wait(f.end().unwrap());
        root.unwrap()
    }

    pub fn block(&self) -> GpuPtr<AtmosphereGpu> {
        // SAFETY: `data` is the address of an `AtmosphereGpu`.
        unsafe { GpuPtr::from_address(self.sky.data()) }
    }

    /// Lighting's statistics of the last frame.
    pub fn lighting_stats(&self) -> LightingStats {
        let lighting = self.lighting.as_ref().expect("a rig with lighting");
        lighting.stats(self.frame_id).unwrap()
    }
}

fn pipeline(entry: &str) -> &'static ComputePipeline {
    static PIPELINES: OnceLock<Vec<(String, ComputePipeline)>> = OnceLock::new();
    let all = PIPELINES.get_or_init(|| {
        MODULE
            .entry_points
            .iter()
            .map(|e| {
                let name = e.name.to_str().unwrap();
                let stage = ShaderStage::new(&MODULE, name, SpecData::default()).unwrap();
                let p = device().create_compute_pipeline(&stage, name).unwrap();
                (name.to_owned(), p)
            })
            .collect()
    });
    &all.iter().find(|(n, _)| n == entry).unwrap().1
}

/// What a probe dispatch reads besides the block.
#[derive(Clone, Copy, Default)]
pub struct Probe<'a> {
    /// Threads to run (the `irradiance` probe runs one group of 64).
    pub threads: u32,
    pub count: u32,
    pub input: &'a [[f32; 4]],
    pub layer: u32,
    pub light: [f32; 4],
    pub colour: [f32; 4],
    /// Lighting's light set.
    pub set: u64,
}

/// Runs probe kernel `entry` and returns its first `outputs` values.
pub fn probe(
    block: GpuPtr<AtmosphereGpu>,
    entry: &str,
    outputs: u64,
    p: Probe<'_>,
) -> Vec<[f32; 4]> {
    let d = device();
    let mut input = d
        .gpu_malloc::<[f32; 4]>(
            MemoryKind::Upload,
            p.input.len().max(1) as u64,
            "probe input",
        )
        .unwrap();
    input.as_mut_slice().unwrap()[..p.input.len()].copy_from_slice(p.input);
    let output = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::DeviceLocal, outputs, "probe output")
        .unwrap();
    let mut root = d
        .gpu_malloc::<Root>(MemoryKind::Upload, 1, "probe root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = Root {
        atmosphere: block,
        input: input.gpu,
        output: output.gpu,
        count: p.count,
        layer: p.layer,
        light: p.light,
        colour: p.colour,
        set: p.set,
        _pad0: 0,
    };
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
    list.dispatch(pipeline(entry), root.gpu, [p.threads.div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    read_buffer(output.gpu, outputs)
}

/// A probe's `light` field carrying the eye position.
pub fn eye_light(eye: DVec3) -> [f32; 4] {
    [eye.x as f32, eye.y as f32, eye.z as f32, eye.y as f32]
}

/// `AtmosphereSky.radiance` from `eye` along each of `dirs`: linear Rec.2020 nits.
pub fn sky_radiance(rig: &Rig, eye: DVec3, dirs: &[DVec3]) -> Vec<[f64; 3]> {
    let input: Vec<[f32; 4]> = dirs
        .iter()
        .map(|d| [d.x as f32, d.y as f32, d.z as f32, 0.0])
        .collect();
    let n = dirs.len() as u32;
    let out = probe(
        rig.block(),
        "sky_radiance",
        n.into(),
        Probe {
            threads: n,
            count: n,
            input: &input,
            light: eye_light(eye),
            ..Probe::default()
        },
    );
    let out: Vec<_> = out
        .into_iter()
        .map(|c| [c[0], c[1], c[2]].map(f64::from))
        .collect();
    assert!(out.iter().flatten().all(|c| c.is_finite()), "{out:?}");
    out
}

/// A sampled texture and its heap slot.
pub struct TestTexture {
    pub index: TextureIndex,
    _texture: Texture,
}

/// A 4 x 4 RGBA32F texture holding `texel` everywhere.
pub fn uniform_texture(texel: [f32; 4], name: &str) -> TestTexture {
    let d = device();
    let desc = TextureDesc::d2(Format::Rgba32Float, 4, 4, TextureUsage::SAMPLED);
    let texture = d
        .create_texture(&desc, TextureMemory::Dedicated, name)
        .unwrap();
    let mut staging = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Upload, 16, name)
        .unwrap();
    staging.as_mut_slice().unwrap().fill(texel);
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.discard(&texture);
    list.copy_to_texture(
        &texture,
        &TextureRegion::whole_mip(&desc, 0),
        staging.gpu.cast(),
    );
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    let slot = d.alloc_image_slot().unwrap();
    TestTexture {
        index: d.write_sampled(slot, &texture, &ViewDesc::default()),
        _texture: texture,
    }
}
