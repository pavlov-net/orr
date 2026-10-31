//! Camera effects after RR: motion blur against shutter-accumulated references.

use half::f16;
use orr_output::{ChainImages, MotionBlurPass, OutputKernels, OutputView, ViewOutput, srgb_encode};
use orr_scene::ViewConstants;
use orr_scene::orr_gpu::{
    Allocation, ComputePipeline, Format, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData,
    Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};
use orr_scene::testing::{
    FlipError, Image, Pixels, assert_clean, device, flip, read_buffer, read_texture, wait,
};
use output_suite::shaders::blur_scene;

/// Not a multiple of the 20-pixel tile either way.
const OUT: [u32; 2] = [336, 190];
const SHUTTER: f32 = 0.5;
const SUBFRAMES: u32 = 64;

#[derive(Clone, Copy, Debug)]
enum Scene {
    Wheel,
    Pan,
}

/// One view's chain, depth and motion at `OUT` (render extent equal to output, no jitter),
/// filled by `blur_scene.slang`.
struct Blur {
    kernels: OutputKernels,
    output: OutputView,
    scene: ComputePipeline,
    reference: ComputePipeline,
    root: Allocation<blur_scene::Root>,
    constants: Allocation<ViewConstants>,
    rr_output: Texture,
    linear: Texture,
    depth: TextureIndex,
    motion: TextureIndex,
    radiance_store: StorageImageIndex,
    depth_store: StorageImageIndex,
    motion_store: StorageImageIndex,
    chain: ChainImages,
    reference_out: Allocation<f32>,
    frame: u64,
    _targets: [Texture; 2],
}

impl Blur {
    fn new() -> Self {
        let gpu = device();
        let texture = |format, name| {
            let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
            let desc = TextureDesc::d2(format, OUT[0], OUT[1], usage);
            gpu.create_texture(&desc, TextureMemory::Dedicated, name)
                .unwrap()
        };
        let rr_output = texture(Format::Rgba16Float, "rr output");
        let linear = texture(Format::Rgba16Float, "display linear");
        let depth = texture(Format::R32Float, "rr depth");
        let motion = texture(Format::Rg16Float, "rr motion");
        let view = ViewDesc::default();
        let slot = || gpu.alloc_image_slot().unwrap();
        let chain = ChainImages {
            rr_output: gpu.write_sampled(slot(), &rr_output, &view),
            rr_output_store: gpu.write_storage(slot(), &rr_output, &view),
            linear: gpu.write_sampled(slot(), &linear, &view),
            linear_store: gpu.write_storage(slot(), &linear, &view),
        };
        let pipeline = |entry, name| {
            let stage = ShaderStage::new(&blur_scene::MODULE, entry, SpecData::default()).unwrap();
            gpu.create_compute_pipeline(&stage, name).unwrap()
        };
        let mut constants = gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap();
        let mut c: ViewConstants = bytemuck::Zeroable::zeroed();
        c.extent = OUT;
        constants.as_mut_slice().unwrap()[0] = c;
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        for t in [&rr_output, &linear, &depth, &motion] {
            list.discard(t);
        }
        wait(queue.submit(vec![list], &[]).unwrap());
        Self {
            kernels: OutputKernels::new(gpu).unwrap(),
            output: OutputView::new(gpu, 2).unwrap(),
            scene: pipeline("frame", "blur scene"),
            reference: pipeline("reference", "blur reference"),
            root: gpu.gpu_malloc(MemoryKind::Upload, 1, "scene root").unwrap(),
            constants,
            radiance_store: chain.rr_output_store,
            depth: gpu.write_sampled(slot(), &depth, &view),
            motion: gpu.write_sampled(slot(), &motion, &view),
            depth_store: gpu.write_storage(slot(), &depth, &view),
            motion_store: gpu.write_storage(slot(), &motion, &view),
            chain,
            reference_out: gpu
                .gpu_malloc(
                    MemoryKind::DeviceLocal,
                    4 * u64::from(OUT[0] * OUT[1]),
                    "reference",
                )
                .unwrap(),
            frame: 0,
            rr_output,
            linear,
            _targets: [depth, motion],
        }
    }

    fn put_root(&mut self, scene: Scene, motion_scale: f32) {
        self.root.as_mut_slice().unwrap()[0] = blur_scene::Root {
            reference: self.reference_out.gpu,
            radiance: self.radiance_store,
            depth: self.depth_store,
            motion: self.motion_store,
            width: OUT[0],
            height: OUT[1],
            scene: scene as u32,
            subframes: SUBFRAMES,
            shutter: SHUTTER,
            motion_scale,
            _pad: 0,
        };
    }

    /// The scene at time 0 into RR's output, blurred with `shutter` and `generated`. Returns
    /// the image the chain ends on and the frame's pass names.
    fn render(
        &mut self,
        scene: Scene,
        motion_scale: f32,
        pass: impl FnOnce(&mut MotionBlurPass<'_>),
    ) -> (Vec<[f32; 4]>, Vec<&'static str>) {
        let gpu = device();
        self.frame += 1;
        self.put_root(scene, motion_scale);
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        list.dispatch(&self.scene, self.root.gpu, groups(OUT));
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        let output = ViewOutput::default();
        let mut p = MotionBlurPass {
            rr_depth: self.depth,
            rr_motion: self.motion,
            extent: OUT,
            view: self.constants.gpu,
            shutter: SHUTTER,
            generated: 0,
            history_reset: false,
            frame: self.frame,
            output: &output,
        };
        pass(&mut p);
        let mut chain = self.chain;
        let blurred = self
            .kernels
            .motion_blur(&mut list, &mut self.output, &p, &mut chain)
            .unwrap();
        assert_eq!(blurred, chain != self.chain);
        let point = queue.submit(vec![list], &[]).unwrap();
        wait(point);
        let names = gpu
            .pass_results(point)
            .unwrap()
            .iter()
            .map(|p| p.name)
            .collect();
        let image = if blurred {
            &self.linear
        } else {
            &self.rr_output
        };
        (read_rgba16f(image), names)
    }

    /// The scene averaged over the open shutter.
    fn reference(&mut self, scene: Scene) -> Vec<[f32; 4]> {
        let gpu = device();
        self.put_root(scene, 1.0);
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        list.dispatch(&self.reference, self.root.gpu, groups(OUT));
        wait(queue.submit(vec![list], &[]).unwrap());
        read_buffer(self.reference_out.gpu, 4 * u64::from(OUT[0] * OUT[1]))
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }
}

fn groups(extent: [u32; 2]) -> [u32; 3] {
    [extent[0].div_ceil(8), extent[1].div_ceil(8), 1]
}

fn read_rgba16f(t: &Texture) -> Vec<[f32; 4]> {
    read_texture(t, &TextureRegion::whole_mip(t.desc(), 0))
        .chunks_exact(8)
        .map(|p| std::array::from_fn(|c| f16::from_le_bytes([p[2 * c], p[2 * c + 1]]).to_f32()))
        .collect()
}

fn srgb8(linear: &[[f32; 4]]) -> Image {
    let pixels = linear
        .iter()
        .flat_map(|p| {
            let c = |v: f32| (srgb_encode(f64::from(v).clamp(0.0, 1.0)) * 255.0).round() as u8;
            [c(p[0]), c(p[1]), c(p[2]), 255]
        })
        .collect();
    Image {
        width: OUT[0],
        height: OUT[1],
        pixels: Pixels::Rgba8(pixels),
    }
}

fn flip_against(reference: &[[f32; 4]], image: &[[f32; 4]]) -> FlipError {
    flip(&srgb8(reference), &srgb8(image))
}

/// Blurred FLIP against the reference at most `mean`, and well under the unblurred frame's.
fn check_scene(scene: Scene, mean: f32) {
    let mut b = Blur::new();
    let reference = b.reference(scene);
    let (sharp, _) = b.render(scene, 1.0, |p| p.shutter = 0.0);
    let (blurred, names) = b.render(scene, 1.0, |_| {});
    assert_eq!(names, ["motion_blur"]);
    let sharp = flip_against(&reference, &sharp);
    let blurred = flip_against(&reference, &blurred);
    log::info!("{scene:?}: FLIP blurred {blurred:?}, unblurred {sharp:?}");
    assert!(blurred.mean <= mean, "{scene:?}: {blurred:?}");
    assert!(
        blurred.mean < 0.6 * sharp.mean,
        "{scene:?}: {blurred:?} against {sharp:?}"
    );
    assert_clean();
}

#[test]
fn spinning_wheel_matches_shutter_reference() {
    check_scene(Scene::Wheel, 0.05);
}

#[test]
fn camera_pan_matches_shutter_reference() {
    check_scene(Scene::Pan, 0.05);
}

/// At one display rate, frame generation renders 2x or 4x the motion per frame; the streak
/// per presented frame stays the same.
#[test]
fn frame_generation_keeps_streak_per_presented_frame() {
    let mut b = Blur::new();
    let (base, _) = b.render(Scene::Wheel, 1.0, |_| {});
    for generated in [1, 3] {
        let scale = (generated + 1) as f32;
        let (image, _) = b.render(Scene::Wheel, scale, |p| p.generated = generated);
        assert!(image == base, "{generated} generated frames");
    }
    assert_clean();
}

#[test]
fn closed_shutter_and_history_reset_record_nothing() {
    let mut b = Blur::new();
    let (sharp, names) = b.render(Scene::Pan, 1.0, |p| p.shutter = 0.0);
    assert!(names.is_empty(), "{names:?}");
    assert_eq!(b.output.motion_blur_bytes(), 0);
    let (reset, names) = b.render(Scene::Pan, 1.0, |p| p.history_reset = true);
    assert!(names.is_empty(), "{names:?}");
    assert!(reset == sharp);
    b.render(Scene::Pan, 1.0, |_| {});
    assert!(b.output.motion_blur_bytes() > 0);
    b.render(Scene::Pan, 1.0, |p| p.shutter = 0.0);
    assert_eq!(b.output.motion_blur_bytes(), 0);
    assert_clean();
}
