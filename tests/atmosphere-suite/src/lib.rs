//! Helpers for the atmosphere suite: a scene with one atmosphere, frames that light it, and
//! the probe kernels that read it back.

use std::sync::OnceLock;

use glam::{DVec3, Vec3};
use orr::abi::{GpuPtr, SamplerIndex};
use orr::orr_gpu::{
    ComputePipeline, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData, Stages,
};
use orr::{
    DirectionalLight, DiscProfile, FrameDesc, FrameTime, Light, LightId, Renderer, RendererDesc,
    SceneDesc, SceneId, SceneRoot,
};
use orr_atmosphere::colour::REC709_TO_REC2020;
use orr_atmosphere::spectrum::{SOLAR_COLOUR, SOLAR_ILLUMINANCE};
use orr_atmosphere::{Atmosphere, AtmosphereGpu, AtmosphereState};
use orr_scene::testing::{device, read_buffer, wait};

/// Generated kernel modules and types of the suite's Slang.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

use shaders::probe::{MODULE, Root};

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

/// Rec.709 illuminance that the light table converts to [`solar_rec2020`].
fn solar_rec709() -> Vec3 {
    let m = REC709_TO_REC2020;
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let target = solar_rec2020();
    // Cramer's rule for m x = target.
    let x: [f64; 3] = std::array::from_fn(|c| {
        let mut mc = m;
        for (row, t) in mc.iter_mut().zip(target) {
            row[c] = t;
        }
        det(mc) / det(m)
    });
    Vec3::new(x[0] as f32, x[1] as f32, x[2] as f32)
}

/// A renderer with one scene and one atmosphere whose candidate is [`SUN`].
pub struct Rig {
    pub renderer: Renderer,
    pub scene: SceneId,
    pub sky: AtmosphereState,
    frames: u32,
}

impl Rig {
    pub fn new(atmosphere: &Atmosphere) -> Self {
        let mut renderer = Renderer::new(device(), RendererDesc::default()).unwrap();
        let scene = renderer.create_scene(&SceneDesc::default()).unwrap();
        let mut sky = AtmosphereState::new(device(), atmosphere, SamplerIndex(7)).unwrap();
        sky.set_lights(&[SUN]);
        Self {
            renderer,
            scene,
            sky,
            frames: 0,
        }
    }

    /// One frame with sunlight from `sun` (none: no light) and the focus at `focus`; waits
    /// for it and returns the scene root.
    pub fn frame(&mut self, sun: Option<DVec3>, focus: DVec3) -> GpuPtr<SceneRoot> {
        self.frames += 1;
        let listed = [self.scene];
        let mut f = self
            .renderer
            .begin_frame(&FrameDesc {
                time: FrameTime {
                    absolute: f64::from(self.frames) / 60.0,
                    delta: 1.0 / 60.0,
                },
                scenes: &listed,
            })
            .unwrap();
        {
            let mut w = f.scene(self.scene).unwrap();
            let light = sun.map(|d| {
                Light::Directional(DirectionalLight {
                    direction_to_light: d.as_vec3(),
                    illuminance: solar_rec709(),
                    angular_radius: 4.652e-3,
                    disc: DiscProfile::LimbDarkened,
                })
            });
            w.set_light(SUN, light.as_ref()).unwrap();
            w.set_focus(focus, 1000.0);
            w.set_environment(self.sky.data());
            w.set_medium(self.sky.data());
        }
        let mut root = None;
        let sky = &mut self.sky;
        f.update_scenes(&mut |cx| {
            root = Some(cx.root);
            sky.record(cx).unwrap();
        })
        .unwrap();
        wait(f.end().unwrap());
        root.unwrap()
    }

    pub fn block(&self) -> GpuPtr<AtmosphereGpu> {
        // SAFETY: `data` is the address of an `AtmosphereGpu`.
        unsafe { GpuPtr::from_address(self.sky.data()) }
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
    };
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
    list.dispatch(pipeline(entry), root.gpu, [p.threads.div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    read_buffer(output.gpu, outputs)
}
