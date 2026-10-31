//! Headless GPU tests of orr-lighting, on gpu-suite's device with validation on.

use std::sync::OnceLock;
use std::time::Duration;

use orr_gpu::{
    AddressMode, Allocation, Filter, Format, GpuPtr, Hazards, KernelSet, SamplerDesc, SamplerIndex,
    ShaderModule, Stages, Texture, TextureDesc, TextureIndex, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};
use orr_lighting::{SceneLighting, SceneLightingSettings, ScenePass};
use orr_math::{Mat3, Vec3};
use orr_scene::{
    Affine3, FrameDesc, FrameTime, LightGpu, MaterialWriter, RendererDesc, SceneCompute, SceneDesc,
    SceneId, SceneRoot, SceneWriter, Scenes,
};

pub use gpu_suite::{assert_clean, device, pipeline, readback, render, require, run, upload};

pub mod bed;

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// orr-scene's and orr-lighting's kernels with `surfaces/context.slang` linked as surface 1.
pub mod linked {
    include!(concat!(env!("OUT_DIR"), "/linked/orr_kernels.rs"));
}

/// A kernel set in which each `(module, kernel)` stands in for orr-lighting's module compiled
/// from `kernel`, as a build linking another environment or medium makes it.
pub fn kernels(replacements: &[(&'static ShaderModule, &'static str)]) -> KernelSet {
    let modules: Vec<&'static ShaderModule> = replacements
        .iter()
        .map(|&(m, kernel)| {
            &*Box::leak(Box::new(ShaderModule {
                kernel,
                entry_points: m.entry_points,
                ..*m
            }))
        })
        .collect();
    KernelSet {
        modules: Box::leak(modules.into_boxed_slice()),
        surfaces: 0,
    }
}

/// A linear, clamping sampler slot.
pub fn linear_clamp() -> SamplerIndex {
    static SLOT: OnceLock<SamplerIndex> = OnceLock::new();
    *SLOT.get_or_init(|| {
        let slot = SamplerIndex(1);
        device().write_sampler(
            slot,
            &SamplerDesc {
                mag: Filter::Linear,
                min: Filter::Linear,
                mip: Filter::Nearest,
                address: [AddressMode::ClampToEdge; 3],
                ..SamplerDesc::default()
            },
        );
        slot
    })
}

/// An RGBA32F texture behind a sampled slot.
pub struct TestTexture {
    pub index: TextureIndex,
    _texture: Texture,
}

/// A `width` x `height` texture holding `texels`, row-major.
pub fn texture(width: u32, height: u32, texels: &[[f32; 4]], name: &str) -> TestTexture {
    let d = device();
    let desc = TextureDesc::d2(Format::Rgba32Float, width, height, TextureUsage::SAMPLED);
    let texture = d
        .create_texture(&desc, TextureMemory::Dedicated, name)
        .unwrap();
    let staging = upload(texels, name);
    let mut list = render().command_list().unwrap();
    list.discard(&texture);
    list.copy_to_texture(
        &texture,
        &TextureRegion::whole_mip(&desc, 0),
        staging.gpu.cast(),
    );
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    run(list);
    let slot = d.alloc_image_slot().unwrap();
    TestTexture {
        index: d.write_sampled(slot, &texture, &ViewDesc::default()),
        _texture: texture,
    }
}

/// Orders `list`'s compute work before its later compute work.
pub fn compute_barrier(list: &mut orr_gpu::CommandList<'_>) {
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
}

/// Makes `list`'s compute and transfer writes visible to the host, submits it and waits.
pub fn finish(mut list: orr_gpu::CommandList<'_>) {
    list.barrier(
        Stages::COMPUTE | Stages::TRANSFER,
        Stages::HOST,
        Hazards::empty(),
    );
    run(list);
}

/// Readback memory that `list` fills with `count` values from `src`.
pub fn copy_back<T: bytemuck::Pod>(
    list: &mut orr_gpu::CommandList<'_>,
    src: GpuPtr<T>,
    count: u64,
    name: &str,
) -> Allocation<T> {
    let out = readback::<T>(count, name);
    list.copy(out.gpu.cast(), src.cast(), count * size_of::<T>() as u64);
    out
}

/// One orr-scene scene on the suite's device, driven a frame at a time.
pub struct TestScene {
    pub scenes: Scenes,
    pub id: SceneId,
    pub desc: SceneDesc,
    /// The scene's root, after the first frame.
    pub root: GpuPtr<SceneRoot>,
    /// The last frame run.
    pub frame: u64,
}

impl TestScene {
    pub fn new(desc: SceneDesc) -> Self {
        let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
        let id = scenes.create_scene(&desc).unwrap();
        Self {
            scenes,
            id,
            desc,
            root: GpuPtr::NULL,
            frame: 0,
        }
    }

    /// Lighting for this scene with default settings and `kernels`.
    pub fn lighting(&self, kernels: &KernelSet) -> SceneLighting {
        SceneLighting::new(
            device(),
            kernels,
            &self.desc,
            &SceneLightingSettings::default(),
        )
        .unwrap()
    }

    /// Runs one frame: `write` changes the scene, then `compute` records its scene compute;
    /// waits for it.
    pub fn frame(
        &mut self,
        write: impl FnOnce(&mut SceneWriter<'_>, &mut MaterialWriter<'_>),
        mut compute: impl FnMut(&mut SceneCompute<'_, '_>),
    ) {
        let list = [self.id];
        let mut f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: FrameTime {
                    absolute: 0.0,
                    delta: 1.0 / 60.0,
                },
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        self.frame = f.id();
        {
            let mut materials = f.materials();
            let mut w = f.scene(self.id).unwrap();
            write(&mut w, &mut materials);
        }
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| {
            root = c.root;
            compute(c);
        })
        .unwrap();
        let point = f.end(&[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();
        self.root = root;
    }

    /// A frame that only writes the scene.
    pub fn write(&mut self, write: impl FnOnce(&mut SceneWriter<'_>, &mut MaterialWriter<'_>)) {
        self.frame(write, |_| {});
    }

    /// The scene pass of the last frame at `focus`.
    pub fn pass(&self, focus: [f32; 3]) -> ScenePass {
        ScenePass {
            frame: self.frame,
            scene: self.root,
            focus,
        }
    }

    pub fn root_value(&self) -> SceneRoot {
        read(self.root, 1)[0]
    }

    /// The scene's light table.
    pub fn lights(&self) -> Vec<LightGpu> {
        read(
            self.root_value().lights,
            u64::from(self.desc.light_capacity),
        )
    }
}

/// `m` followed by the translation `t`.
pub fn affine(m: Mat3, t: Vec3) -> Affine3 {
    let r = |i: usize| {
        let row = m.row(i);
        [row.x, row.y, row.z, t[i]]
    };
    Affine3 {
        rows: [r(0), r(1), r(2)],
    }
}

/// Copies `count` values at `src` back, after all work submitted so far.
pub fn read<T: bytemuck::Pod>(src: GpuPtr<T>, count: u64) -> Vec<T> {
    let mut list = render().command_list().unwrap();
    list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
    let out = copy_back(&mut list, src, count, "read back");
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run(list);
    out.as_slice().unwrap().to_vec()
}
