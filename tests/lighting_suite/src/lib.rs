//! Headless GPU tests of orr_lighting, on gpu_suite's device with validation on.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use orr_gpu::{
    AddressMode, Allocation, DEFAULT_MEDIUM, Filter, Format, GpuPtr, Hazards, KernelSet,
    SamplerDesc, SamplerIndex, ShaderModule, Stages, Texture, TextureDesc, TextureIndex,
    TextureMemory, TextureRegion, TextureUsage, ViewDesc,
};
use orr_lighting::{SceneLighting, SceneLightingSettings, ScenePass};
use orr_scene::{
    EnvironmentData, FrameDesc, FrameTime, LightGpu, LoadState, Loading, MaterialWriter, MountId,
    PackSource, RendererDesc, SceneCompute, SceneDesc, SceneId, SceneRoot, SceneWriter, Scenes,
};

pub use gpu_suite::{assert_clean, device, pipeline, readback, render, require, run, upload};

pub mod bed;
pub mod welch;

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// orr_scene's and orr_lighting's kernels with `surfaces/context.slang` linked as surface 1
/// and `Rings` as surface 2 (`rings.slang`).
pub mod linked {
    include!(concat!(env!("OUT_DIR"), "/linked/orr_kernels.rs"));
}

/// A kernel set in which each `(module, kernel)` stands in for orr_lighting's module compiled
/// from `kernel`, as a build linking the types `environment` and `medium` makes it.
pub fn kernels(
    environment: &'static str,
    medium: &'static str,
    replacements: &[(&'static ShaderModule, &'static str)],
) -> &'static KernelSet {
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
    Box::leak(Box::new(KernelSet {
        modules: Box::leak(modules.into_boxed_slice()),
        environment,
        medium,
        ..KernelSet::NONE
    }))
}

/// `procedural_bake.slang`'s `ProceduralSky`, which reads no record.
struct ProceduralSkyData;

impl EnvironmentData for ProceduralSkyData {
    const TYPE: &'static str = "ProceduralSky";
    const NULL_SAFE: bool = true;
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
        staging.gpu().cast(),
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
    list.copy(out.gpu().cast(), src.cast(), count * size_of::<T>() as u64);
    out
}

/// Threads lent to the loading side until dropped.
pub(crate) struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    pub(crate) fn start(scenes: &Scenes) -> Self {
        let loading = scenes.loading();
        let threads = (0..2)
            .map(|_| {
                let l = Arc::clone(&loading);
                std::thread::spawn(move || l.run_worker())
            })
            .collect();
        Self { loading, threads }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.loading.stop_workers();
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}

/// Test threads that hold scenes at once. A scene's tables are made for a world, whatever a
/// test puts in them, so each takes about 1 GB of the device's memory, and the render queue
/// runs the threads' frames one after another whatever their number: more threads add memory,
/// which the OS pages out between frames once several test processes pass what the device has.
const SCENE_THREADS: u64 = 6;

/// The threads holding scenes now.
fn scene_threads() -> &'static (Mutex<u64>, Condvar) {
    static THREADS: (Mutex<u64>, Condvar) = (Mutex::new(0), Condvar::new());
    &THREADS
}

thread_local! {
    /// The scenes this thread holds.
    static SCENES_HELD: Cell<u32> = const { Cell::new(0) };
}

/// A test thread's leave to hold scenes, taken with its first and given back with its last, so
/// a thread that holds one never waits for another. Stays on its thread.
pub(crate) struct ScenePermit(PhantomData<*const ()>);

impl ScenePermit {
    pub(crate) fn take() -> Self {
        if SCENES_HELD.get() == 0 {
            let (holding, freed) = scene_threads();
            let mut holding = holding.lock().unwrap_or_else(PoisonError::into_inner);
            while *holding >= SCENE_THREADS {
                holding = freed.wait(holding).unwrap_or_else(PoisonError::into_inner);
            }
            *holding += 1;
        }
        SCENES_HELD.set(SCENES_HELD.get() + 1);
        Self(PhantomData)
    }
}

impl Drop for ScenePermit {
    fn drop(&mut self) {
        SCENES_HELD.set(SCENES_HELD.get() - 1);
        if SCENES_HELD.get() == 0 {
            let (holding, freed) = scene_threads();
            *holding.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
            freed.notify_one();
        }
    }
}

type LinkedData = Box<dyn FnOnce(&mut SceneWriter<'_>)>;

/// One orr_scene scene on the suite's device, driven a frame at a time.
pub struct TestScene {
    pub scenes: Scenes,
    pub id: SceneId,
    pub desc: SceneDesc,
    kernels: &'static KernelSet,
    /// Sets the scene's linked data in its first frame.
    linked: Option<LinkedData>,
    /// The scene's root, after the first frame.
    pub root: GpuPtr<SceneRoot>,
    /// The last frame run.
    pub frame: u64,
    /// Loads mounts, from the first [`Self::mount`].
    workers: Option<Workers>,
    _permit: ScenePermit,
}

impl TestScene {
    pub fn new(desc: SceneDesc) -> Self {
        Self::linked(desc, &KernelSet::NONE, |_| {})
    }

    /// A scene on kernels that bake the procedural sky.
    pub fn procedural(desc: SceneDesc) -> Self {
        let linked = kernels(
            ProceduralSkyData::TYPE,
            DEFAULT_MEDIUM,
            &[(&shaders::procedural_bake::MODULE, "environment/bake")],
        );
        Self::linked(desc, linked, |w| {
            w.set_environment(GpuPtr::<ProceduralSkyData>::NULL);
        })
    }

    /// A scene on `kernels`, with `data` setting its linked data in its first frame.
    pub fn linked(
        desc: SceneDesc,
        kernels: &'static KernelSet,
        data: impl FnOnce(&mut SceneWriter<'_>) + 'static,
    ) -> Self {
        let renderer = RendererDesc {
            kernels,
            ..RendererDesc::default()
        };
        let permit = ScenePermit::take();
        let mut scenes = Scenes::new(device(), &renderer).unwrap();
        let id = scenes.create_scene(&desc).unwrap();
        Self {
            _permit: permit,
            scenes,
            id,
            desc,
            kernels,
            linked: Some(Box::new(data)),
            root: GpuPtr::NULL,
            frame: 0,
            workers: None,
        }
    }

    /// Mounts the pack `bytes`, running frames until it loads.
    pub fn mount(&mut self, bytes: Arc<[u8]>) -> MountId {
        if self.workers.is_none() {
            self.workers = Some(Workers::start(&self.scenes));
        }
        let m = self.scenes.mount(PackSource::Memory(bytes));
        while orr_scene::testing::mount_pending(&self.scenes, m) {
            self.write(|_, _| {});
            std::thread::sleep(Duration::from_millis(1));
        }
        if let LoadState::Failed(e) = self.scenes.mount_state(m) {
            panic!("mount failed: {e}");
        }
        m
    }

    /// Lighting for this scene with default settings, on the scene's kernels.
    pub fn lighting(&self) -> SceneLighting {
        SceneLighting::new(
            device(),
            self.kernels,
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
            if let Some(linked) = self.linked.take() {
                linked(&mut w);
            }
            write(&mut w, &mut materials);
        }
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| {
            root = c.root;
            compute(c);
        })
        .unwrap();
        let point = f.end(&[]).unwrap();
        render()
            .wait_while_completing(point, Duration::from_secs(10))
            .unwrap();
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

/// Copies `count` values at `src` back, after all work submitted so far.
pub fn read<T: bytemuck::Pod>(src: GpuPtr<T>, count: u64) -> Vec<T> {
    let mut list = render().command_list().unwrap();
    list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
    let out = copy_back(&mut list, src, count, "read back");
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run(list);
    out.as_slice().unwrap().to_vec()
}
