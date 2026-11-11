//! Camera effects after RR: motion blur against shutter-accumulated references, depth of
//! field against lens-disc references, glare against its CPU model and PSF, and lens
//! distortion against its analytic map.

use orr_color::srgb_encode;
use orr_output::{
    ChainImages, MotionBlurPass, OutputKernels, OutputView, OverlayFront, ViewOutput,
};
use orr_scene::ViewConstants;
use orr_scene::orr_gpu::{
    Allocation, ComputePipeline, Format, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData,
    Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage,
    ViewDesc,
};
use orr_scene::testing::{FlipError, Image, Pixels, assert_clean, device, flip, read_buffer, wait};
use output_suite::read_rgba16f;
use output_suite::shaders::blur_scene;

/// Not a multiple of the 20-pixel tile either way.
const OUT: [u32; 2] = [336, 190];
const SHUTTER: f32 = 0.5;
const SUBFRAMES: u32 = 64;

#[derive(Clone, Copy, Debug)]
enum Scene {
    Wheel,
    Pan,
    /// Translucent particles crossing a still background.
    Particles = 3,
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
    overlay: OverlayFront,
    overlay_store: [StorageImageIndex; 3],
    chain: ChainImages,
    reference_out: Allocation<f32>,
    frame: u64,
    _targets: [Texture; 5],
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
        let overlay_depth = texture(Format::R32Float, "overlay depth");
        let overlay_motion = texture(Format::Rg16Float, "overlay motion");
        let overlay_weight = texture(Format::R16Float, "overlay weight");
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
        for t in [
            &rr_output,
            &linear,
            &depth,
            &motion,
            &overlay_depth,
            &overlay_motion,
            &overlay_weight,
        ] {
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
            overlay: OverlayFront {
                depth: gpu.write_sampled(slot(), &overlay_depth, &view),
                motion: gpu.write_sampled(slot(), &overlay_motion, &view),
                weight: gpu.write_sampled(slot(), &overlay_weight, &view),
            },
            overlay_store: [
                gpu.write_storage(slot(), &overlay_depth, &view),
                gpu.write_storage(slot(), &overlay_motion, &view),
                gpu.write_storage(slot(), &overlay_weight, &view),
            ],
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
            _targets: [depth, motion, overlay_depth, overlay_motion, overlay_weight],
        }
    }

    fn put_root(&mut self, scene: Scene, motion_scale: f32) {
        self.root.as_mut_slice().unwrap()[0] = blur_scene::Root {
            reference: self.reference_out.gpu(),
            radiance: self.radiance_store,
            depth: self.depth_store,
            motion: self.motion_store,
            overlay_depth: self.overlay_store[0],
            overlay_motion: self.overlay_store[1],
            overlay_weight: self.overlay_store[2],
            width: OUT[0],
            height: OUT[1],
            scene: scene as u32,
            subframes: SUBFRAMES,
            shutter: SHUTTER,
            motion_scale,
            time: 0.0,
            _pad0: 0,
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
        list.dispatch(&self.scene, self.root.gpu(), groups(OUT));
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        let output = ViewOutput::default();
        let mut p = MotionBlurPass {
            rr_depth: self.depth,
            rr_motion: self.motion,
            overlay: Some(self.overlay),
            extent: OUT,
            view: self.constants.gpu(),
            shutter: SHUTTER,
            generated: 0,
            camera_cut: false,
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
        list.dispatch(&self.reference, self.root.gpu(), groups(OUT));
        wait(queue.submit(vec![list], &[]).unwrap());
        read_buffer(self.reference_out.gpu(), 4 * u64::from(OUT[0] * OUT[1]))
            .as_chunks::<4>()
            .0
            .to_vec()
    }
}

fn groups(extent: [u32; 2]) -> [u32; 3] {
    [extent[0].div_ceil(8), extent[1].div_ceil(8), 1]
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

/// The surface behind holds still; only the front layer's motion can blur the particles.
#[test]
fn particles_over_a_still_background_match_shutter_reference() {
    check_scene(Scene::Particles, 0.05);
}

/// A front layer no pixel holds changes nothing.
#[test]
fn empty_front_layer_equals_no_front_layer() {
    let mut b = Blur::new();
    let (with, _) = b.render(Scene::Wheel, 1.0, |_| {});
    let (without, _) = b.render(Scene::Wheel, 1.0, |p| p.overlay = None);
    assert!(with == without);
    assert_clean();
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
fn closed_shutter_and_a_cut_record_nothing() {
    let mut b = Blur::new();
    let (sharp, names) = b.render(Scene::Pan, 1.0, |p| p.shutter = 0.0);
    assert!(names.is_empty(), "{names:?}");
    assert_eq!(b.output.motion_blur_bytes(), 0);
    let (cut, names) = b.render(Scene::Pan, 1.0, |p| p.camera_cut = true);
    assert!(names.is_empty(), "{names:?}");
    assert!(cut == sharp);
    b.render(Scene::Pan, 1.0, |_| {});
    assert!(b.output.motion_blur_bytes() > 0);
    b.render(Scene::Pan, 1.0, |p| p.shutter = 0.0);
    assert_eq!(b.output.motion_blur_bytes(), 0);
    assert_clean();
}

/// Depth of field on both paths against lens-disc accumulation references.
mod depth_of_field {
    use orr_color::srgb_encode;
    use orr_output::{
        ChainImages, DepthOfFieldPass, DofPath, OutputKernels, OutputView, OverlayFront,
        ViewOutput, ev100_from_exposure,
    };
    use orr_scene::orr_gpu::{
        Allocation, ComputePipeline, Format, GpuPtr, Hazards, MemoryKind, QueueKind, ShaderStage,
        SpecData, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
        TextureRegion, TextureUsage, ViewDesc,
    };
    use orr_scene::testing::{
        FlipError, Image, Pixels, assert_clean, device, flip, flip_mean_where, read_buffer,
        read_texture, wait,
    };
    use orr_scene::{DepthOfField, ExposureState, Focus, Lens, Projection, ViewConstants};
    use output_suite::shaders::dof_scene;

    use super::{groups, read_rgba16f};

    const OUT: [u32; 2] = [320, 180];
    const LENS_SAMPLES: u32 = 256;
    const SENSOR_MM: f32 = 36.0;
    /// tan(20 degrees): a 40-degree horizontal field of view.
    const TAN_X: f32 = 0.363_970_23;
    const NEAR: f32 = 0.1;
    const FOCUS: f32 = 1.0;

    #[derive(Clone, Copy, Debug)]
    enum Scene {
        Foreground,
        Background,
        Highlights,
        Water,
        /// A fence in front of the focal plane hiding, in the pinhole image, a warm stripe
        /// behind each bar.
        Fence,
        /// Nearly opaque particles on the focal plane over a far background.
        SharpParticles,
    }

    fn lens(blades: u32, f_number: f32) -> DepthOfField {
        DepthOfField {
            f_number,
            focus: Focus::Manual { distance_m: FOCUS },
            blades,
            round_at_f_number: 0.7,
            max_coc: 0.12,
        }
    }

    fn tan_y() -> f32 {
        TAN_X * OUT[1] as f32 / OUT[0] as f32
    }

    /// The aperture radius, m, of the pinhole camera whose lens-disc rays blur by the
    /// thin-lens CoC.
    fn lens_radius(dof: &DepthOfField) -> f32 {
        let focal = 0.5 * SENSOR_MM / TAN_X;
        let aperture = focal / dof.f_number;
        let focus = 1000.0 * FOCUS;
        let radius_px = 0.5 * aperture * focal / (focus - focal) / SENSOR_MM * OUT[0] as f32;
        radius_px * FOCUS * 2.0 * TAN_X / OUT[0] as f32
    }

    fn projection() -> Projection {
        Projection {
            fov_y: 2.0 * tan_y().atan(),
            near: NEAR,
        }
    }

    /// One view's chain and render targets at `OUT` (render extent equal to output, no
    /// jitter, no motion), filled by `dof_scene.slang`. With `holes` on, ray hole filling
    /// reads `dof_scene.slang`'s stand-in for lighting's background layer, traced from the
    /// view's mask.
    struct Bed {
        kernels: OutputKernels,
        output: OutputView,
        scene: ComputePipeline,
        reference: ComputePipeline,
        root: Allocation<dof_scene::Root>,
        constants: Allocation<ViewConstants>,
        /// The view's exposure slots, both holding the frame's state: the pass reads the
        /// adapted exposure from either, by its path.
        exposure: Allocation<ExposureState>,
        /// The pre-exposure the next frame is lit with. The scenes are in exposed units: the
        /// adapted exposure stays 1.
        pre_exposure: f32,
        rr_output: Texture,
        linear: Texture,
        depth: TextureIndex,
        depth_store: StorageImageIndex,
        motion: TextureIndex,
        motion_store: StorageImageIndex,
        overlay: OverlayFront,
        overlay_store: [StorageImageIndex; 3],
        /// The surfaces and front layer lighting would write, read by the pass; else the
        /// opaque depth alone.
        layered: bool,
        chain: ChainImages,
        reference_out: Allocation<f32>,
        frame: u64,
        layer: Texture,
        layer_index: TextureIndex,
        layer_store: StorageImageIndex,
        background_layer: ComputePipeline,
        holes: bool,
        reset: bool,
        /// Depth, motion and the front layer's depth, motion and weight.
        targets: [Texture; 5],
    }

    impl Bed {
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
            let layer = texture(Format::Rgba16Float, "background layer");
            let depth = texture(Format::R32Float, "rr depth");
            let motion = texture(Format::Rg16Float, "rr motion");
            let overlay_depth = texture(Format::R32Float, "overlay depth");
            let overlay_motion = texture(Format::Rg16Float, "overlay motion");
            let overlay_weight = texture(Format::R16Float, "overlay weight");
            let view = ViewDesc::default();
            let slot = || gpu.alloc_image_slot().unwrap();
            let chain = ChainImages {
                rr_output: gpu.write_sampled(slot(), &rr_output, &view),
                rr_output_store: gpu.write_storage(slot(), &rr_output, &view),
                linear: gpu.write_sampled(slot(), &linear, &view),
                linear_store: gpu.write_storage(slot(), &linear, &view),
            };
            let pipeline = |entry, name| {
                let stage =
                    ShaderStage::new(&dof_scene::MODULE, entry, SpecData::default()).unwrap();
                gpu.create_compute_pipeline(&stage, name).unwrap()
            };
            let mut constants = gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap();
            let mut exposure = gpu
                .gpu_malloc::<ExposureState>(MemoryKind::Upload, 2, "exposure")
                .unwrap();
            exposure.as_mut_slice().unwrap().fill(ExposureState {
                pre_exposure: 1.0,
                prev_pre_exposure: 1.0,
                adapted_ev100: ev100_from_exposure(1.0),
                initialized: 1,
                ..ExposureState::default()
            });
            let mut c: ViewConstants = bytemuck::Zeroable::zeroed();
            c.exposure = exposure.gpu();
            c.clip_from_view[0][0] = 1.0 / TAN_X;
            c.clip_from_view[1][1] = -1.0 / tan_y();
            c.clip_from_view[3][2] = -1.0;
            c.clip_from_view[2][3] = NEAR;
            c.clip_from_view_jittered = c.clip_from_view;
            c.prev_clip_from_view = c.clip_from_view;
            c.view_from_clip[0][0] = TAN_X;
            c.view_from_clip[1][1] = -tan_y();
            c.view_from_clip[2][3] = -1.0;
            c.view_from_clip[3][2] = 1.0 / NEAR;
            for i in 0..3 {
                c.view_from_world_rotation[i][i] = 1.0;
                c.prev_view_from_world_rotation[i][i] = 1.0;
            }
            c.extent = OUT;
            c.delta_time = 1.0 / 60.0;
            constants.as_mut_slice().unwrap()[0] = c;
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            for t in [
                &rr_output,
                &linear,
                &depth,
                &motion,
                &layer,
                &overlay_depth,
                &overlay_motion,
                &overlay_weight,
            ] {
                list.discard(t);
            }
            wait(queue.submit(vec![list], &[]).unwrap());
            Self {
                kernels: OutputKernels::new(gpu).unwrap(),
                output: OutputView::new(gpu, 2).unwrap(),
                scene: pipeline("frame", "dof scene"),
                reference: pipeline("reference", "dof reference"),
                root: gpu.gpu_malloc(MemoryKind::Upload, 1, "scene root").unwrap(),
                constants,
                exposure,
                pre_exposure: 1.0,
                depth: gpu.write_sampled(slot(), &depth, &view),
                depth_store: gpu.write_storage(slot(), &depth, &view),
                motion: gpu.write_sampled(slot(), &motion, &view),
                motion_store: gpu.write_storage(slot(), &motion, &view),
                overlay: OverlayFront {
                    depth: gpu.write_sampled(slot(), &overlay_depth, &view),
                    motion: gpu.write_sampled(slot(), &overlay_motion, &view),
                    weight: gpu.write_sampled(slot(), &overlay_weight, &view),
                },
                overlay_store: [
                    gpu.write_storage(slot(), &overlay_depth, &view),
                    gpu.write_storage(slot(), &overlay_motion, &view),
                    gpu.write_storage(slot(), &overlay_weight, &view),
                ],
                layered: false,
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
                layer_index: gpu.write_sampled(slot(), &layer, &view),
                layer_store: gpu.write_storage(slot(), &layer, &view),
                layer,
                background_layer: pipeline("background_layer", "dof background layer"),
                holes: false,
                reset: false,
                targets: [depth, motion, overlay_depth, overlay_motion, overlay_weight],
            }
        }

        fn put_root(&mut self, scene: Scene, dof: &DepthOfField) {
            let polygon = dof.blades >= 3 && dof.f_number > dof.round_at_f_number;
            self.root.as_mut_slice().unwrap()[0] = dof_scene::Root {
                reference: self.reference_out.gpu(),
                radiance: self.chain.rr_output_store,
                depth: self.depth_store,
                motion: self.motion_store,
                overlay_depth: self.overlay_store[0],
                overlay_motion: self.overlay_store[1],
                overlay_weight: self.overlay_store[2],
                layered: u32::from(self.layered),
                width: OUT[0],
                height: OUT[1],
                scene: scene as u32,
                lens_samples: LENS_SAMPLES,
                tan_x: TAN_X,
                tan_y: tan_y(),
                near: NEAR,
                focus: FOCUS,
                lens_radius: lens_radius(dof),
                blades: if polygon { dof.blades } else { 0 },
                roundness: (dof.round_at_f_number / dof.f_number).min(1.0),
                mask: TextureIndex::NONE,
                background: self.layer_store,
                layer_focus: GpuPtr::NULL,
                layer_radius: 0.0,
                sample: 0,
                pre_exposure: self.pre_exposure,
                _pad1: 0,
            };
        }

        /// The scene through `dof` on `path`: the image the chain ends on (on the guide
        /// path the blurred color for RR), the frame's pass names, and on the guide path the
        /// guide.
        fn render(
            &mut self,
            scene: Scene,
            dof: Option<DepthOfField>,
            path: DofPath,
        ) -> (Vec<[f32; 4]>, Vec<&'static str>, Option<Vec<f32>>) {
            let gpu = device();
            self.frame += 1;
            for e in self.exposure.as_mut_slice().unwrap() {
                e.prev_pre_exposure = e.pre_exposure;
                e.pre_exposure = self.pre_exposure;
            }
            self.put_root(scene, &dof.unwrap_or_else(|| lens(0, 1.4)));
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            list.dispatch(&self.scene, self.root.gpu(), groups(OUT));
            let lens = Lens {
                sensor_width_mm: SENSOR_MM,
                depth_of_field: dof,
                ..Lens::default()
            };
            let mask = self.output.dof_hole_mask(&lens, &projection(), OUT, OUT);
            let background = match mask.filter(|_| self.holes) {
                Some(mask) => {
                    let root = &mut self.root.as_mut_slice().unwrap()[0];
                    root.mask = mask.mask;
                    root.layer_focus = mask.focus_distance;
                    root.layer_radius = mask.aperture_radius;
                    root.sample = self.frame as u32;
                    list.dispatch(&self.background_layer, self.root.gpu(), groups(OUT));
                    Some(self.layer_index)
                }
                None => None,
            };
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
            let mut output = ViewOutput::default();
            output.post.dof_path = path;
            output.post.dof_ray_hole_fill = self.holes;
            let pass = DepthOfFieldPass {
                rr_depth: self.depth,
                rr_motion: self.motion,
                overlay: self.layered.then_some(self.overlay),
                extent: OUT,
                view: self.constants.gpu(),
                sensor_width_mm: SENSOR_MM,
                depth_of_field: dof,
                history_reset: self.reset,
                frame: self.frame,
                output: &output,
                render_extent: OUT,
                background,
            };
            let mut chain = self.chain;
            let recorded = match path {
                DofPath::AfterRr => self
                    .kernels
                    .depth_of_field(&mut list, &mut self.output, &pass, &mut chain)
                    .unwrap(),
                DofPath::Guide => self
                    .kernels
                    .depth_of_field_guide(&mut list, &mut self.output, &pass, chain.rr_output)
                    .unwrap(),
            };
            let point = queue.submit(vec![list], &[]).unwrap();
            wait(point);
            let names = gpu
                .pass_results(point)
                .unwrap()
                .iter()
                .map(|p| p.name)
                .collect();
            if path == DofPath::Guide && recorded {
                let images = self.output.depth_of_field_guide().unwrap();
                let guide = read_r16f(images.guide);
                return (read_rgba16f(images.color), names, Some(guide));
            }
            let image = if chain.rr_output == self.chain.linear {
                &self.linear
            } else {
                &self.rr_output
            };
            (read_rgba16f(image), names, None)
        }

        /// Whether each pixel holds a translucent layer, as of the last frame.
        fn overlay_pixels(&self) -> Vec<bool> {
            read_r16f(&self.targets[4])
                .iter()
                .map(|w| *w > 0.0)
                .collect()
        }

        /// The stand-in background layer the last frame traced.
        fn layer(&self) -> Vec<[f32; 4]> {
            read_rgba16f(&self.layer)
        }

        /// The scene's reverse-Z depth.
        fn depth(&self) -> Vec<f32> {
            read_texture(
                &self.targets[0],
                &TextureRegion::whole_mip(self.targets[0].desc(), 0),
            )
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect()
        }

        fn reference(&mut self, scene: Scene, dof: &DepthOfField) -> Vec<[f32; 4]> {
            let gpu = device();
            self.put_root(scene, dof);
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            list.dispatch(&self.reference, self.root.gpu(), groups(OUT));
            wait(queue.submit(vec![list], &[]).unwrap());
            read_buffer(self.reference_out.gpu(), 4 * u64::from(OUT[0] * OUT[1]))
                .as_chunks::<4>()
                .0
                .to_vec()
        }
    }

    fn read_r16f(t: &Texture) -> Vec<f32> {
        read_texture(t, &TextureRegion::whole_mip(t.desc(), 0))
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| half::f16::from_le_bytes(*p).to_f32())
            .collect()
    }

    /// Reinhard per channel, then sRGB, so highlights keep their spread.
    fn display(linear: &[[f32; 4]]) -> Image {
        let pixels = linear
            .iter()
            .flat_map(|p| {
                let c = |v: f32| {
                    let v = f64::from(v.max(0.0));
                    (srgb_encode(v / (1.0 + v)) * 255.0).round() as u8
                };
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
        flip(&display(reference), &display(image))
    }

    /// FLIP of the blurred frame against the reference at most `mean` on both paths, and
    /// well under the sharp frame's.
    fn check(scene: Scene, dof: DepthOfField, mean: f32) {
        for path in [DofPath::AfterRr, DofPath::Guide] {
            let mut b = Bed::new();
            let reference = b.reference(scene, &dof);
            let (sharp, names, _) = b.render(scene, None, path);
            assert!(names.is_empty(), "{names:?}");
            let (blurred, names, _) = b.render(scene, Some(dof), path);
            let name = match path {
                DofPath::AfterRr => "depth_of_field",
                DofPath::Guide => "depth_of_field_guide",
            };
            assert_eq!(names, [name]);
            let sharp = flip_against(&reference, &sharp);
            let blurred = flip_against(&reference, &blurred);
            log::info!("{scene:?} {path:?}: FLIP blurred {blurred:?}, sharp {sharp:?}");
            assert!(blurred.mean <= mean, "{scene:?} {path:?}: {blurred:?}");
            assert!(
                blurred.mean < 0.6 * sharp.mean,
                "{scene:?} {path:?}: {blurred:?} against {sharp:?}"
            );
        }
        assert_clean();
    }

    #[test]
    fn foreground_over_focus_matches_lens_reference() {
        check(Scene::Foreground, lens(0, 1.4), 0.05);
    }

    /// The mirrored fill guesses what a foreground pixel hides from the background around
    /// it; the background layer sees it once it has gathered samples.
    #[test]
    fn ray_hole_fill_sees_what_the_foreground_hides() {
        const THRESHOLD: f32 = 0.05;
        for f_number in [2.0, 1.4, 1.0] {
            let dof = lens(0, f_number);
            for path in [DofPath::AfterRr, DofPath::Guide] {
                let mut b = Bed::new();
                let reference = b.reference(Scene::Fence, &dof);
                let (mirrored, _, _) = b.render(Scene::Fence, Some(dof), path);
                b.holes = true;
                let mut traced = Vec::new();
                for _ in 0..8 {
                    traced = b.render(Scene::Fence, Some(dof), path).0;
                }
                let mirrored = flip_against(&reference, &mirrored);
                let traced = flip_against(&reference, &traced);
                log::info!("f/{f_number} {path:?}: FLIP traced {traced:?}, mirrored {mirrored:?}");
                assert!(
                    mirrored.mean > THRESHOLD,
                    "f/{f_number} {path:?}: {mirrored:?}"
                );
                assert!(
                    traced.mean <= THRESHOLD,
                    "f/{f_number} {path:?}: {traced:?}"
                );
            }
        }
        assert_clean();
    }

    /// The mask marks every foreground pixel and starts its ray past the foreground; turning
    /// the fill off frees it.
    #[test]
    fn hole_mask_starts_rays_past_the_foreground() {
        let mut b = Bed::new();
        b.holes = true;
        let dof = lens(0, 2.0);
        let lens = Lens {
            sensor_width_mm: SENSOR_MM,
            depth_of_field: Some(dof),
            ..Lens::default()
        };
        b.render(Scene::Foreground, Some(dof), DofPath::AfterRr);
        let with_holes = b.output.depth_of_field_bytes();
        let mask = b
            .output
            .dof_hole_mask(&lens, &projection(), OUT, OUT)
            .unwrap();
        assert!(
            b.output
                .dof_hole_mask(&lens, &projection(), [OUT[0] / 2, OUT[1]], OUT)
                .is_none()
        );
        let expected = 0.5 * (0.5 * SENSOR_MM / TAN_X) / dof.f_number / 1000.0;
        assert!((mask.aperture_radius - expected).abs() < 1e-6 * expected);
        b.render(Scene::Foreground, Some(dof), DofPath::AfterRr);
        let (foreground, wall) = (NEAR / 0.35, NEAR / 1.0);
        let (depth, layer) = (b.depth(), b.layer());
        let mut marked = 0;
        for (d, t) in depth.iter().zip(&layer) {
            if (d - foreground).abs() < 1e-3 * foreground {
                assert!(t[3] >= 0.0, "an unmarked foreground pixel");
            }
            if t[3] >= 0.0 {
                marked += 1;
                assert!(
                    (t[3] - wall).abs() < 2e-3 * wall,
                    "a ray stopped at {}",
                    t[3]
                );
            } else {
                assert_eq!(*t, [0.0, 0.0, 0.0, -1.0]);
            }
        }
        assert!(
            marked > 0 && marked < layer.len(),
            "{marked} of {}",
            layer.len()
        );
        b.holes = false;
        b.render(Scene::Foreground, Some(dof), DofPath::AfterRr);
        assert!(
            b.output
                .dof_hole_mask(&lens, &projection(), OUT, OUT)
                .is_none()
        );
        assert!(b.output.depth_of_field_bytes() < with_holes);
        assert_clean();
    }

    /// The layer traced from the mask before a cut is not used: the frame after it fills
    /// holes by mirroring, as a view's first frame does.
    #[test]
    fn cut_keeps_the_mirrored_fill() {
        let dof = Some(lens(0, 1.4));
        let (first, _, _) = Bed::new().render(Scene::Fence, dof, DofPath::AfterRr);
        let mut b = Bed::new();
        b.holes = true;
        for _ in 0..4 {
            b.render(Scene::Fence, dof, DofPath::AfterRr);
        }
        b.reset = true;
        let (cut, _, _) = b.render(Scene::Fence, dof, DofPath::AfterRr);
        assert_eq!(first, cut);
        assert_clean();
    }

    #[test]
    fn focus_over_background_matches_lens_reference() {
        check(Scene::Background, lens(0, 1.4), 0.06);
    }

    #[test]
    fn highlights_through_foliage_match_lens_reference() {
        // Highlights hidden behind leaves in the pinhole image still reach the lens around
        // them; post-process depth of field cannot recover them.
        check(Scene::Highlights, lens(6, 1.0), 0.22);
    }

    /// The water plane and particles are blurred at the opaque depth behind them.
    #[test]
    fn water_and_particles_match_lens_reference() {
        check(Scene::Water, lens(0, 1.4), 0.06);
    }

    /// FLIP over the pixels of a front layer against the reference, and of the same pixels
    /// blurred at the opaque depth behind them.
    fn over_the_front_layer(scene: Scene, path: DofPath) -> (f32, f32) {
        let dof = lens(0, 1.4);
        let mut behind = Bed::new();
        let reference = behind.reference(scene, &dof);
        let (behind, _, _) = behind.render(scene, Some(dof), path);
        let mut b = Bed::new();
        b.layered = true;
        let (layered, _, _) = b.render(scene, Some(dof), path);
        let mask = b.overlay_pixels();
        assert!(mask.iter().any(|&m| m) && !mask.iter().all(|&m| m));
        let measure =
            |image: &[[f32; 4]]| flip_mean_where(&display(&reference), &display(image), &mask);
        let (layered, behind) = (measure(&layered), measure(&behind));
        log::info!(
            "{scene:?} {path:?}: FLIP over the layer {layered}, at the depth behind {behind}"
        );
        (layered, behind)
    }

    #[test]
    fn water_and_particles_front_layer_matches_lens_reference() {
        for path in [DofPath::AfterRr, DofPath::Guide] {
            let (layered, _) = over_the_front_layer(Scene::Water, path);
            assert!(layered <= 0.06, "{path:?}: {layered}");
        }
        assert_clean();
    }

    /// Particles on the focal plane stay sharp over a background the lens blurs.
    #[test]
    fn particles_in_focus_keep_their_own_depth() {
        for path in [DofPath::AfterRr, DofPath::Guide] {
            let (layered, behind) = over_the_front_layer(Scene::SharpParticles, path);
            assert!(layered <= 0.09, "{path:?}: {layered}");
            assert!(
                layered < 0.75 * behind,
                "{path:?}: {layered} against {behind}"
            );
        }
        assert_clean();
    }

    #[test]
    fn static_camera_holds_still_and_off_frees_images() {
        let mut b = Bed::new();
        let dof = Some(lens(6, 1.0));
        let (first, _, _) = b.render(Scene::Highlights, dof, DofPath::AfterRr);
        assert!(b.output.depth_of_field_bytes() > 0);
        // Sprites are added in whatever order they were found, so frames match to rounding.
        let still = |a: &[[f32; 4]], b: &[[f32; 4]]| {
            a.iter()
                .zip(b)
                .all(|(a, b)| (0..3).all(|c| (a[c] - b[c]).abs() <= 2e-3 * a[c].abs().max(1.0)))
        };
        for _ in 0..2 {
            let (next, _, _) = b.render(Scene::Highlights, dof, DofPath::AfterRr);
            assert!(still(&first, &next));
        }
        let (_, names, _) = b.render(Scene::Highlights, None, DofPath::AfterRr);
        assert!(names.is_empty(), "{names:?}");
        assert_eq!(b.output.depth_of_field_bytes(), 0);
        assert_clean();
    }

    /// The stabilized image and the accumulated background layer rescale their histories by
    /// the change of pre-exposure, and highlights are picked and weighted at their exposed
    /// luminance: on a still scene at one adapted exposure, a pre-exposure that moves every
    /// frame shows what a fixed one does. The highlights scene's pre-exposure falls, which
    /// takes its highlights under the scatter threshold where that is read in pre-exposed
    /// units.
    #[test]
    fn a_moving_pre_exposure_shows_what_a_fixed_one_does() {
        for (scene, dof, stops) in [
            (Scene::Fence, lens(0, 1.4), 0.15),
            (Scene::Highlights, lens(6, 1.0), -0.15),
        ] {
            let run = |stops_per_frame: f32| {
                let mut b = Bed::new();
                b.holes = true;
                let mut image = Vec::new();
                for f in 0..24 {
                    b.pre_exposure = (stops_per_frame * f as f32).exp2();
                    let (shown, _, _) = b.render(scene, Some(dof), DofPath::AfterRr);
                    image = shown
                        .iter()
                        .map(|p| p.map(|c| c / b.pre_exposure))
                        .collect();
                }
                image
            };
            let (fixed, moving) = (run(0.0), run(stops));
            let worst = fixed
                .iter()
                .zip(&moving)
                .flat_map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).abs() / a[c].abs().max(0.05)))
                .fold(0.0, f32::max);
            log::info!("{scene:?}: largest change from the fixed pre-exposure's image: {worst}");
            assert!(worst < 1e-2, "{scene:?}: {worst}");
        }
        assert_clean();
    }

    #[test]
    fn autofocus_finds_the_surface_under_its_point() {
        let mut b = Bed::new();
        let (manual, _, _) = b.render(Scene::Foreground, Some(lens(0, 1.4)), DofPath::AfterRr);
        let auto = DepthOfField {
            focus: Focus::Auto {
                uv: [0.85, 0.8],
                speed: 4.0,
            },
            ..lens(0, 1.4)
        };
        let (focused, _, _) = b.render(Scene::Foreground, Some(auto), DofPath::AfterRr);
        let largest = manual
            .iter()
            .zip(&focused)
            .flat_map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).abs()))
            .fold(0.0, f32::max);
        assert!(largest < 0.01, "{largest}");
        assert_clean();
    }

    /// The guide is `luminance(after - before)` with luminance `(r + 2g + b) / 4`, 0 where
    /// nothing blurs.
    #[test]
    fn guide_holds_the_luminance_change() {
        let mut b = Bed::new();
        let (before, _, _) = b.render(Scene::Foreground, None, DofPath::Guide);
        let (after, _, guide) = b.render(Scene::Foreground, Some(lens(0, 1.4)), DofPath::Guide);
        let guide = guide.unwrap();
        let luminance = |p: &[f32; 4]| (p[0] + 2.0 * p[1] + p[2]) / 4.0;
        let mut unchanged = 0;
        for ((a, b), g) in after.iter().zip(&before).zip(&guide) {
            let expected = luminance(a) - luminance(b);
            assert!(
                (g - expected).abs() < 4e-3,
                "guide {g}, expected {expected}"
            );
            unchanged += usize::from(*g == 0.0);
        }
        assert!(
            unchanged > guide.len() / 4,
            "{unchanged} of {}",
            guide.len()
        );
        assert_clean();
    }
}

/// Glare: the band pyramid against its CPU model, the fitted tables and the diaphragm PSF
/// they are fitted to; FFT glare against direct convolution with its kernel, and its kernel
/// against the PSF.
mod glare {
    use orr_color::srgb_encode;
    use orr_output::{
        Exposure, GlareMethod, OutputDebug, OutputFormat, ResolvedDisplay, ViewOutput,
    };
    use orr_scene::Glare;
    use orr_scene::testing::{Image, Pixels, assert_clean, flip};
    use output_suite::glare_model::{
        F_NUMBERS, TABLES, encircled_rgb, fft, fft_kernel, psf, pyramid,
    };
    use output_suite::{Harness, read_rgba16f};

    use pyramid::Plane;

    /// An emitter's exposed value, under the HDR ceiling.
    const EXPOSED: f32 = 8000.0;

    /// The EV100 at which the lens is at table F-number `i` at the reference pitch.
    fn ev_at(i: usize) -> f32 {
        (100.0 * F_NUMBERS[i] * F_NUMBERS[i]).log2() as f32
    }

    /// The nits a surface has per unit it exposes to at `ev`.
    fn nits_per_exposed(ev: f32) -> f32 {
        1.0 / orr_output::exposure_from_ev100(ev)
    }

    /// An scRGB display; Linear debug output on it is not clamped.
    fn hdr(peak_nits: f32) -> ResolvedDisplay {
        ResolvedDisplay {
            encoding: OutputFormat::ScRgb,
            paper_white_nits: 250.0,
            signal_white_nits: 250.0,
            peak_nits,
            black_nits: 0.0,
        }
    }

    fn sdr() -> ResolvedDisplay {
        ResolvedDisplay {
            encoding: OutputFormat::Sdr,
            paper_white_nits: 250.0,
            signal_white_nits: 250.0,
            peak_nits: 250.0,
            black_nits: 0.0,
        }
    }

    fn output(ev100: f32, debug: OutputDebug) -> ViewOutput {
        ViewOutput {
            exposure: Exposure::Manual { ev100 },
            debug,
            ..ViewOutput::default()
        }
    }

    /// The harness's view at `extent` with the reference pixel pitch, glare `strength`.
    fn harness(extent: [u32; 2], glare: Option<f32>) -> Harness {
        let mut h = Harness::new(extent[0], extent[1]);
        h.sensor_height_mm = (psf::REFERENCE_PITCH_UM * f64::from(extent[1]) / 1000.0) as f32;
        h.glare = glare.map(|strength| Glare { strength });
        h
    }

    /// Runs frames until pre-exposure settles; returns the last display-linear image in
    /// linear Rec.2020, the exposure the passes applied and the pre-exposure the radiance was
    /// rendered with.
    fn settle(
        h: &mut Harness,
        nits: &[f32],
        out: &ViewOutput,
        display: &ResolvedDisplay,
    ) -> (Vec<[f64; 3]>, f64, f64) {
        h.set_image(nits);
        let mut telemetry = h.frame(1.0, out, Some(display));
        for _ in 0..3 {
            telemetry = h.frame(1.0, out, Some(display));
        }
        let rec2020 = h
            .display_linear()
            .iter()
            .map(|p| orr_color::rec709_to_rec2020([p[0], p[1], p[2]].map(f64::from)).to_array())
            .collect();
        let exposure = f64::from(orr_output::exposure_from_ev100(telemetry.adapted_ev100));
        (rec2020, exposure, f64::from(telemetry.pre_exposure))
    }

    fn weights(h: &Harness) -> Vec<[f64; 4]> {
        h.glare_weights()
            .expect("glare ran")
            .iter()
            .map(|w| w.map(f64::from))
            .collect()
    }

    /// The weights the pass writes at F-number `n` (reference pitch) for a chain of `mips`
    /// levels from `first_level`: the tables interpolated in log N, folded into the last mip.
    fn expected_weights(first_level: usize, n: f64, mips: usize) -> Vec<[f64; 3]> {
        let f = &F_NUMBERS;
        let n = n.clamp(f[0], f[f.len() - 1]);
        let t = (n / f[0]).ln() / (f[f.len() - 1] / f[0]).ln() * (f.len() - 1) as f64;
        let i = (t as usize).min(f.len() - 2);
        let rows = &TABLES[first_level - 1];
        let mut out = vec![[0.0; 3]; mips + 1];
        for c in 0..3 {
            let lerped: Vec<f64> = (0..rows[i].len())
                .map(|e| rows[i][e][c] + (rows[i + 1][e][c] - rows[i][e][c]) * (t - i as f64))
                .collect();
            for (o, w) in out.iter_mut().zip(pyramid::fold(&lerped, mips)) {
                o[c] = w;
            }
        }
        out
    }

    fn assert_weights(gpu: &[[f64; 4]], expected: &[[f64; 3]], what: &str) {
        for (e, w) in expected.iter().enumerate() {
            for c in 0..3 {
                assert!(
                    (gpu[e][c] - w[c]).abs() < 1e-5,
                    "{what}: entry {e}: {gpu:?}"
                );
            }
        }
        assert!(
            gpu[expected.len()..].iter().all(|w| w[..3] == [0.0; 3]),
            "{what}: {gpu:?}"
        );
    }

    /// `glare_extract`: the part of an exposed color glare spreads.
    fn extract(exposed: [f64; 3]) -> [f64; 3] {
        const KNEE: f64 = 1.5;
        const WIDTH: f64 = 0.5;
        let m = exposed[0].max(exposed[1]).max(exposed[2]);
        if m <= KNEE - WIDTH {
            return [0.0; 3];
        }
        let over = m - KNEE;
        let over = if over < WIDTH {
            (over + WIDTH) * (over + WIDTH) / (4.0 * WIDTH)
        } else {
            over
        };
        exposed.map(|v| v * over / m)
    }

    /// A pre-exposed pixel exposed by `scale` as the glare pass reads it.
    fn exposed_pixel(p: [f32; 4], scale: f64) -> [f64; 3] {
        [p[0], p[1], p[2]].map(|v| {
            let v = f64::from(v) * scale;
            if v > 0.0 { v.min(65504.0) } else { 0.0 }
        })
    }

    /// The CPU model of the GPU run: per channel, the pyramid of what glare spreads of the
    /// pre-exposed radiance the harness rendered, scaled by `scale`, at the frame's weights,
    /// over the pixel less the part the bands take. Also returns that kept image.
    fn model(
        h: &Harness,
        extent: [u32; 2],
        first_level: u32,
        scale: f64,
        w: &[[f64; 4]],
    ) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let radiance = h.radiance();
        let spread: Vec<[f64; 3]> = radiance
            .iter()
            .map(|&p| extract(exposed_pixel(p, scale)))
            .collect();
        let kept: Vec<[f64; 3]> = radiance
            .iter()
            .zip(&spread)
            .map(|(p, s)| [0, 1, 2].map(|c| f64::from(p[c]) * scale - (1.0 - w[0][c]) * s[c]))
            .collect();
        let mut out = vec![[0.0; 3]; radiance.len()];
        for c in 0..3 {
            let part = Plane::dense(extent, spread.iter().map(|p| p[c]).collect());
            let input = Plane::dense(extent, kept.iter().map(|p| p[c]).collect());
            let mut wc: Vec<f64> = w.iter().map(|e| e[c]).collect();
            wc[0] = 1.0;
            let downs = pyramid::downsamples(&part, first_level);
            let glare = pyramid::compose(&input, &downs, &wc);
            for (o, v) in out.iter_mut().zip(&glare.data) {
                o[c] = *v;
            }
        }
        (out, kept)
    }

    /// Point sources and a gradient on a dim ground, `extent[0] * extent[1]` nits.
    fn scene(extent: [u32; 2], ground: f32, lights: &[([u32; 2], f32)]) -> Vec<f32> {
        let [w, h] = extent;
        let mut nits: Vec<f32> = (0..w * h)
            .map(|i| ground * (1.0 + (i % w) as f32 / w as f32))
            .collect();
        for &([x, y], l) in lights {
            nits[(y * w + x) as usize] = l;
        }
        nits
    }

    #[test]
    fn pyramid_matches_cpu_model() {
        // Odd, so level 1 is more than twice as coarse per texel.
        let extent = [161, 97];
        let mut h = harness(extent, Some(1.0));
        let ev = ev_at(3) + 0.37;
        // On the knee's curve and past it.
        let k = nits_per_exposed(ev);
        let nits = scene(
            extent,
            200.0,
            &[
                ([40, 30], 4e6),
                ([41, 30], 2e6),
                ([120, 70], 6e6),
                ([159, 95], 1e6),
                ([80, 20], 1.2 * k),
                ([84, 20], 1.6 * k),
                ([88, 20], 1.9 * k),
                ([92, 20], 2.4 * k),
            ],
        );
        for first_level in [1, 2] {
            let mut out = output(ev, OutputDebug::Linear);
            out.post.glare_first_level = first_level;
            let (gpu, exposure, pre) = settle(&mut h, &nits, &out, &hdr(250.0));
            let w = weights(&h);
            let scale = exposure / pre;
            let (cpu, kept) = model(&h, extent, first_level, scale, &w);
            let mut worst: f64 = 0.0;
            for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
                for k in 0..3 {
                    let glare = (c[k] - kept[i][k]).abs();
                    // RGBA16F rounding of the output and of the chain.
                    let error = (g[k] - c[k]).abs() / (1e-3 * c[k].abs() + 0.02 * glare + 1e-6);
                    worst = worst.max(error);
                }
            }
            log::info!(
                "first level {first_level}: worst error against the CPU model {worst:.3} of its bound"
            );
            assert!(worst <= 1.0, "first level {first_level}: {worst}");
        }
        assert_clean();
    }

    #[test]
    fn no_glare_frees_the_chain() {
        let extent = [64, 64];
        let nits = vec![1.0; 64 * 64];
        let mut h = harness(extent, Some(1.0));
        let out = output(10.0, OutputDebug::None);
        settle(&mut h, &nits, &out, &sdr());
        assert!(h.output().glare_bytes() > 0);
        h.glare = None;
        settle(&mut h, &nits, &out, &sdr());
        assert_eq!(h.output().glare_bytes(), 0);
        assert!(h.glare_weights().is_none());
        assert_clean();
    }

    /// The worst relative change glare makes to `nits` under `out`, over all channels.
    fn worst_change(extent: [u32; 2], nits: &[f32], out: &ViewOutput) -> f64 {
        let (plain, ..) = settle(&mut harness(extent, None), nits, out, &hdr(250.0));
        let (glared, ..) = settle(&mut harness(extent, Some(1.0)), nits, out, &hdr(250.0));
        let mut worst: f64 = 0.0;
        for (a, b) in plain.iter().zip(&glared) {
            for k in 0..3 {
                worst = worst.max((b[k] / a[k] - 1.0).abs());
            }
        }
        worst
    }

    /// A uniform field below the knee, on its curve and past it stays as it is under `out`.
    fn assert_uniform_fields_unchanged(out: impl Fn(f32) -> ViewOutput, ev: f32) {
        let extent = [200, 120];
        for exposed in [0.05, 1.5, 6.0] {
            let nits = vec![exposed * nits_per_exposed(ev); 200 * 120];
            let worst = worst_change(extent, &nits, &out(ev));
            log::info!("uniform field at {exposed}: worst relative change {worst:.2e}");
            assert!(worst < 1e-3, "{exposed}: {worst}");
        }
        assert_clean();
    }

    #[test]
    fn uniform_field_is_unchanged() {
        assert_uniform_fields_unchanged(|ev| output(ev, OutputDebug::Linear), ev_at(5));
    }

    /// Fine detail a display can show as it is stays sharp at the widest F-number.
    #[test]
    fn detail_below_the_knee_is_unchanged() {
        let extent = [200, 120];
        let ev = ev_at(9);
        let k = nits_per_exposed(ev);
        let nits: Vec<f32> = (0..200 * 120)
            .map(|i| {
                if (i % 200 + i / 200) % 2 == 0 {
                    0.02 * k
                } else {
                    0.9 * k
                }
            })
            .collect();
        let worst = worst_change(extent, &nits, &output(ev, OutputDebug::Linear));
        log::info!("detail below the knee: worst relative change {worst:.2e}");
        assert!(worst < 1e-3, "{worst}");
        assert_clean();
    }

    #[test]
    fn point_source_keeps_energy() {
        let extent = [1024, 1024];
        let ev = ev_at(4);
        let mut nits = vec![0.0; 1024 * 1024];
        let source = EXPOSED * nits_per_exposed(ev);
        nits[512 * 1024 + 512] = source;
        let out = output(ev, OutputDebug::Linear);
        for strength in [1.0, 2.0] {
            let (image, exposure, _) = settle(
                &mut harness(extent, Some(strength)),
                &nits,
                &out,
                &hdr(250.0),
            );
            let energy = f64::from(source) * exposure;
            for k in 0..3 {
                let sum: f64 = image.iter().map(|p| p[k]).sum();
                log::info!(
                    "point source at strength {strength}: channel {k} keeps {:.5} of its energy",
                    sum / energy
                );
                assert!((sum / energy - 1.0).abs() < 0.01, "{k}: {}", sum / energy);
            }
        }
        assert_clean();
    }

    /// Encodes per-channel energies over `energy` in log10 from 1e-8 to 1 as 8-bit gray
    /// levels, which shows the PSF's tail.
    fn log_image(side: u32, pixels: &[[f64; 3]], energy: f64) -> Image {
        let level = |v: f64| {
            let t = ((v.max(1e-12) / energy).log10() + 8.0) / 8.0;
            (srgb_encode(t.clamp(0.0, 1.0)) * 255.0).round() as u8
        };
        Image {
            width: side,
            height: side,
            pixels: Pixels::Rgba8(
                pixels
                    .iter()
                    .flat_map(|p| [level(p[0]), level(p[1]), level(p[2]), 255])
                    .collect(),
            ),
        }
    }

    /// The PSF at `f_number` around a pixel center, per pixel of a `side` square, from its
    /// encircled energy over lattice shells of equal area.
    fn psf_image(side: u32, f_number: f64) -> Vec<[f64; 3]> {
        let half = i64::from(side / 2);
        let reach = 2 * half;
        let mut shells = std::collections::BTreeMap::<i64, u64>::new();
        for y in -reach..=reach {
            for x in -reach..=reach {
                *shells.entry(x * x + y * y).or_default() += 1;
            }
        }
        let ee = |n: u64| {
            let r = (n as f64 / std::f64::consts::PI).sqrt();
            encircled_rgb(r, f_number, psf::REFERENCE_PITCH_UM)
        };
        let mut inside = 0;
        let mut per_pixel = std::collections::BTreeMap::new();
        for (&d2, &n) in shells.range(..=half * half * 2) {
            let (a, b) = (ee(inside), ee(inside + n));
            per_pixel.insert(d2, [0, 1, 2].map(|c| (b[c] - a[c]) / n as f64));
            inside += n;
        }
        (0..i64::from(side) * i64::from(side))
            .map(|i| {
                let (x, y) = (i % i64::from(side) - half, i / i64::from(side) - half);
                per_pixel[&(x * x + y * y)]
            })
            .collect()
    }

    #[test]
    fn fitted_kernel_matches_psf() {
        let extent = [2048, 2048];
        let side = 512;
        let i = 4;
        let ev = ev_at(i);
        let source = EXPOSED * nits_per_exposed(ev);
        let mut nits = vec![0.0; 2048 * 2048];
        nits[1024 * 2048 + 1024] = source;
        let mut h = harness(extent, Some(1.0));
        let out = output(ev, OutputDebug::Linear);
        let (image, exposure, _) = settle(&mut h, &nits, &out, &hdr(250.0));
        assert_weights(
            &weights(&h),
            &expected_weights(1, F_NUMBERS[i], 8),
            &format!("table F-number {i}"),
        );
        let energy = f64::from(source) * exposure;
        let crop: Vec<[f64; 3]> = (0..side * side)
            .map(|p| {
                let (x, y) = (p % side + 1024 - side / 2, p / side + 1024 - side / 2);
                image[(y * 2048 + x) as usize]
            })
            .collect();
        let fitted = log_image(side, &crop, energy);
        let reference = log_image(side, &psf_image(side, F_NUMBERS[i]), 1.0);
        let error = flip(&reference, &fitted);
        log::info!(
            "fitted kernel against the PSF at f/{:.2}: FLIP {error:?}",
            F_NUMBERS[i]
        );
        assert!(error.mean < 0.03, "{error:?}");
        assert_clean();
    }

    #[test]
    fn f_number_follows_exposure_and_hdr_peak() {
        let extent = [64, 64];
        let nits = vec![1.0; 64 * 64];
        let mut h = harness(extent, Some(1.0));
        let mut last_tail = 0.0;
        for ev in [6.0, 9.0, 12.0, 15.0] {
            settle(&mut h, &nits, &output(ev, OutputDebug::None), &sdr());
            let w = weights(&h);
            let lens = f64::from(ev).exp2().sqrt() / 10.0;
            assert_weights(&w, &expected_weights(1, lens, 6), &format!("EV {ev}"));
            let tail: f64 = w[1..].iter().map(|w| w[1]).sum();
            assert!(tail >= last_tail, "{ev}: {tail} after {last_tail}");
            last_tail = tail;
            settle(
                &mut h,
                &nits,
                &output(ev + 2.0, OutputDebug::None),
                &hdr(1000.0),
            );
            assert_weights(
                &weights(&h),
                &expected_weights(1, lens, 6),
                &format!("EV {} at 1000 nits", ev + 2.0),
            );
        }
        assert_clean();
    }

    /// At one height and sensor, bands and FFT take the same weights at any width.
    #[test]
    fn weights_ignore_the_aspect_ratio() {
        let ev = ev_at(4);
        // The reference pitch at the height below, whatever the width.
        let sensor_mm = (psf::REFERENCE_PITCH_UM * 128.0 / 1000.0) as f32;
        let weights_at = |extent: [u32; 2], out: &ViewOutput| {
            let mut h = Harness::new(extent[0], extent[1]);
            h.glare = Some(Glare { strength: 1.0 });
            h.sensor_height_mm = sensor_mm;
            let nits = vec![1.0; (extent[0] * extent[1]) as usize];
            settle(&mut h, &nits, out, &sdr());
            weights(&h)
        };
        let bands = output(ev, OutputDebug::None);
        let wide = weights_at([512, 128], &bands);
        assert_eq!(wide, weights_at([128, 128], &bands));
        assert_weights(&wide, &expected_weights(1, F_NUMBERS[4], 7), "bands");
        let fft = fft_output(ev, OutputDebug::None, 1);
        assert_eq!(
            weights_at([512, 128], &fft)[0],
            weights_at([128, 128], &fft)[0]
        );
        assert_clean();
    }

    /// Distance from `center` along its row to the last pixel at or above `level` (green,
    /// display-linear).
    fn reach(image: &[[f32; 4]], width: u32, center: [u32; 2], level: f32) -> u32 {
        let row = (center[1] * width) as usize;
        (center[0]..width)
            .take_while(|&x| image[row + x as usize][1] >= level)
            .count() as u32
    }

    /// The HDR rendition's halo at the SDR clip level (250 nits) has the SDR rendition's
    /// extent within one pyramid level, by either method.
    #[test]
    fn hdr_halo_at_sdr_clip_matches_sdr_halo() {
        let extent = [512, 512];
        let ev = 8.0;
        let mut nits = vec![0.05 * 1.2 * 2f32.powf(ev); 512 * 512];
        nits[256 * 512 + 256] = EXPOSED * 1.2 * 2f32.powf(ev);
        let hdr = ResolvedDisplay {
            encoding: OutputFormat::Hdr10Pq,
            paper_white_nits: 250.0,
            signal_white_nits: 250.0,
            peak_nits: 1000.0,
            black_nits: 0.0,
        };
        for method in [GlareMethod::Bands, GlareMethod::Fft] {
            let mut out = output(ev, OutputDebug::None);
            out.post.glare = method;
            let mut h = harness(extent, Some(1.0));
            settle(&mut h, &nits, &out, &sdr());
            let sdr_reach = reach(&h.display_linear(), 512, [256, 256], 0.99);
            settle(&mut h, &nits, &out, &hdr);
            let hdr_reach = reach(&h.display_linear(), 512, [256, 256], 1.0);
            log::info!(
                "{method:?} halo reach: SDR clip {sdr_reach} px, HDR 250 nits {hdr_reach} px"
            );
            let ratio = f64::from(hdr_reach.max(1)) / f64::from(sdr_reach.max(1));
            assert!(
                ratio.log2().abs() <= 1.0,
                "{method:?}: SDR {sdr_reach}, HDR {hdr_reach}"
            );
        }
        assert_clean();
    }

    /// At night in SDR, brighter emitters show wider halos; without glare every emitter is
    /// one clipped pixel.
    #[test]
    fn night_street_shows_emitter_brightness_in_sdr() {
        let extent = [480, 160];
        let ev = 3.0;
        let k = 1.2 * 2f32.powf(ev);
        let lights = [[80, 80], [240, 80], [400, 80]];
        let exposed = [300.0, 2000.0, EXPOSED];
        let nits = scene(
            extent,
            0.02 * k,
            &[
                (lights[0], exposed[0] * k),
                (lights[1], exposed[1] * k),
                (lights[2], exposed[2] * k),
            ],
        );
        let out = output(ev, OutputDebug::None);
        let reaches = |glare| {
            let mut h = harness(extent, glare);
            settle(&mut h, &nits, &out, &sdr());
            let image = h.display_linear();
            lights.map(|p| reach(&image, 480, p, 0.5))
        };
        let lit = reaches(Some(1.0));
        let plain = reaches(None);
        log::info!("night street halo reach: {lit:?} with glare, {plain:?} without");
        assert!(lit[0] < lit[1] && lit[1] < lit[2], "{lit:?}");
        assert!(plain[0] == plain[1] && plain[1] == plain[2], "{plain:?}");
        assert_clean();
    }

    /// The FFT's image size and kernel radius for `extent`: 4/9 of the image's height at its
    /// resolution, as far as power-of-two padding up to 4096 allows.
    fn fft_layout(extent: [u32; 2], first_level: u32) -> ([u32; 2], i64) {
        let size = extent.map(|n| n >> first_level);
        let share = (4 * size[1]).div_ceil(9);
        let room = |n: u32| ((n + 2 * share).next_power_of_two().min(4096) - n) / 2;
        (size, i64::from(share.min(room(size[0])).min(room(size[1]))))
    }

    /// The lens's F-number at the reference pitch on SDR or a 250-nit peak.
    fn lens_f_number(ev100: f32) -> f64 {
        let f = &F_NUMBERS;
        (f64::from(ev100).exp2() / 100.0)
            .sqrt()
            .clamp(f[0], f[f.len() - 1])
    }

    /// The glare kernel the GPU builds: per pixel within `radius`, the PSF's energy, the
    /// source pixel's own left out.
    fn glare_kernel(n: f64, first_level: u32, radius: i64) -> Vec<[f64; 3]> {
        let kernel = fft_kernel();
        let u = fft::Kernel::texels(n, first_level);
        let side = 2 * radius + 1;
        (0..side * side)
            .map(|i| {
                let (dx, dy) = (i % side - radius, i / side - radius);
                if dx * dx + dy * dy > radius * radius || (dx, dy) == (0, 0) {
                    [0.0; 3]
                } else {
                    kernel.energy(dx, dy, u)
                }
            })
            .collect()
    }

    /// The exposed radiance the harness rendered, averaged over 2^first_level pixel blocks.
    fn exposed_blocks(
        h: &Harness,
        extent: [u32; 2],
        first_level: u32,
        scale: f64,
    ) -> Vec<[f64; 3]> {
        let radiance = h.radiance();
        let f = 1u32 << first_level;
        let size = extent.map(|n| n >> first_level);
        (0..size[0] * size[1])
            .map(|i| {
                let (x, y) = (i % size[0], i / size[0]);
                let mut sum = [0.0; 3];
                for b in 0..f {
                    for a in 0..f {
                        let p = radiance[((y * f + b) * extent[0] + x * f + a) as usize];
                        let part = extract(exposed_pixel(p, scale));
                        for c in 0..3 {
                            sum[c] += part[c];
                        }
                    }
                }
                sum.map(|s| s / f64::from(f * f))
            })
            .collect()
    }

    /// Direct convolution of `image` (`size`) with `kernel` (`radius`), edge pixels repeated.
    fn convolve(
        image: &[[f64; 3]],
        size: [u32; 2],
        kernel: &[[f64; 3]],
        radius: i64,
    ) -> Vec<[f64; 3]> {
        let [w, h] = size.map(i64::from);
        let side = 2 * radius + 1;
        let taps: Vec<(i64, i64, [f64; 3])> = (0..side * side)
            .filter(|&i| kernel[i as usize] != [0.0; 3])
            .map(|i| (i % side - radius, i / side - radius, kernel[i as usize]))
            .collect();
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let mut sum = [0.0; 3];
                for &(dx, dy, k) in &taps {
                    let sx = (x - dx).clamp(0, w - 1);
                    let sy = (y - dy).clamp(0, h - 1);
                    let v = image[(sy * w + sx) as usize];
                    for c in 0..3 {
                        sum[c] += k[c] * v[c];
                    }
                }
                sum
            })
            .collect()
    }

    fn fft_output(ev100: f32, debug: OutputDebug, first_level: u32) -> ViewOutput {
        let mut out = output(ev100, debug);
        out.post.glare = GlareMethod::Fft;
        out.post.glare_first_level = first_level;
        out
    }

    #[test]
    fn fft_convolution_matches_direct_convolution() {
        let extent = [320, 192];
        let mut h = harness(extent, Some(0.8));
        let ev = ev_at(3) + 0.37;
        // On the knee's curve and past it.
        let k = nits_per_exposed(ev);
        let nits = scene(
            extent,
            200.0,
            &[
                ([80, 60], 4e6),
                ([81, 60], 2e6),
                ([240, 140], 6e6),
                ([319, 191], 1e6),
                ([3, 100], 3e6),
                ([160, 30], 1.2 * k),
                ([164, 30], 1.6 * k),
                ([168, 30], 1.9 * k),
                ([172, 30], 2.4 * k),
            ],
        );
        for first_level in [1, 2] {
            let out = fft_output(ev, OutputDebug::Linear, first_level);
            let (_, exposure, pre) = settle(&mut h, &nits, &out, &hdr(250.0));
            let (size, radius) = fft_layout(extent, first_level);
            let kernel = glare_kernel(lens_f_number(ev), first_level, radius);
            let input = exposed_blocks(&h, extent, first_level, exposure / pre);
            let cpu = convolve(&input, size, &kernel, radius);
            let gpu = read_rgba16f(h.output().glare().expect("glare ran"));
            let mut error = [0.0; 3];
            let mut energy = [0.0; 3];
            for ((g, c), i) in gpu.iter().zip(&cpu).zip(&input) {
                for k in 0..3 {
                    error[k] += (f64::from(g[k]) - 0.8 * c[k]).abs();
                    energy[k] += i[k];
                }
            }
            let w0 = weights(&h)[0];
            for k in 0..3 {
                let relative = error[k] / energy[k];
                log::info!(
                    "first level {first_level}, channel {k}: error {relative:.2e} of the energy"
                );
                assert!(
                    relative < 1e-3,
                    "first level {first_level}, channel {k}: {relative}"
                );
                let sum: f64 = kernel.iter().map(|e| e[k]).sum();
                assert!(
                    (w0[k] - (1.0 - 0.8 * sum)).abs() < 1e-5,
                    "w0 {w0:?}, kernel {sum}"
                );
            }
        }
        assert_clean();
    }

    #[test]
    fn fft_uniform_field_is_unchanged() {
        assert_uniform_fields_unchanged(|ev| fft_output(ev, OutputDebug::Linear, 1), ev_at(5));
    }

    #[test]
    fn fft_point_source_keeps_energy() {
        let extent = [512, 512];
        let ev = ev_at(4);
        let mut nits = vec![0.0; 512 * 512];
        let source = EXPOSED * nits_per_exposed(ev);
        nits[256 * 512 + 256] = source;
        let out = fft_output(ev, OutputDebug::Linear, 1);
        let mut h = harness(extent, Some(1.0));
        let (image, exposure, pre) = settle(&mut h, &nits, &out, &hdr(250.0));
        // What the source holds after RGBA16F rounding, which is up to 5e-4 of it.
        let radiance = h.radiance();
        for k in 0..3 {
            let energy: f64 =
                radiance.iter().map(|p| f64::from(p[k])).sum::<f64>() * exposure / pre;
            let sum: f64 = image.iter().map(|p| p[k]).sum();
            log::info!(
                "fft point source: channel {k} keeps {:.5} of its energy",
                sum / energy
            );
            assert!((sum / energy - 1.0).abs() < 1e-3, "{k}: {}", sum / energy);
        }
        assert_clean();
    }

    /// A point source's glare peaks once per spike around it, inside the kernel table's
    /// reach and past it, along the edges' normals.
    #[test]
    fn fft_spikes_match_the_blades() {
        let extent = [512, 512];
        let ev = ev_at(4);
        let mut nits = vec![0.0; 512 * 512];
        nits[256 * 512 + 256] = EXPOSED * nits_per_exposed(ev);
        let mut h = harness(extent, Some(1.0));
        let out = fft_output(ev, OutputDebug::Linear, 1);
        settle(&mut h, &nits, &out, &hdr(250.0));
        let glare = read_rgba16f(h.output().glare().expect("glare ran"));
        let side = 256;
        let at = |x: f64, y: f64| {
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let p = |x: f64, y: f64| f64::from(glare[y as usize * side + x as usize][1]);
            p(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + p(x0 + 1.0, y0) * fx * (1.0 - fy)
                + p(x0, y0 + 1.0) * (1.0 - fx) * fy
                + p(x0 + 1.0, y0 + 1.0) * fx * fy
        };
        let table = fft::TABLE as f64 / fft::Kernel::texels(lens_f_number(ev), 1);
        let steps = 1440;
        for radius in [0.6 * table, 1.5 * table] {
            let ring: Vec<f64> = (0..steps)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / steps as f64;
                    at(128.0 + radius * a.cos(), 128.0 + radius * a.sin())
                })
                .collect();
            let max = ring.iter().copied().fold(0.0, f64::max);
            let peaks: Vec<f64> = (0..steps)
                .filter(|&i| {
                    let v = ring[i];
                    v > 0.05 * max
                        && (1..=8).all(|d| {
                            v > ring[(i + d) % steps] && v >= ring[(i + steps - d) % steps]
                        })
                })
                .map(|i| 360.0 * i as f64 / steps as f64)
                .collect();
            log::info!("fft spikes at radius {radius:.1}: peaks at {peaks:?} degrees");
            assert_eq!(peaks.len(), psf::BLADES as usize, "{radius}: {peaks:?}");
            for p in &peaks {
                let off = (p - 30.0).rem_euclid(60.0);
                assert!(off.min(60.0 - off) < 1.0, "{radius}: {peaks:?}");
            }
        }
        assert_clean();
    }

    #[test]
    fn fft_glare_frees_its_buffers() {
        let extent = [128, 128];
        let nits = vec![1.0; 128 * 128];
        let mut h = harness(extent, Some(1.0));
        settle(&mut h, &nits, &output(10.0, OutputDebug::None), &sdr());
        let bands = h.output().glare_bytes();
        let fft = fft_output(10.0, OutputDebug::None, 1);
        settle(&mut h, &nits, &fft, &sdr());
        let fft_bytes = h.output().glare_bytes();
        assert!(
            fft_bytes > bands && bands > 0,
            "bands {bands}, fft {fft_bytes}"
        );
        assert_eq!(h.glare_weights().map(|w| w.len()), Some(1));
        h.glare = None;
        settle(&mut h, &nits, &fft, &sdr());
        assert_eq!(h.output().glare_bytes(), 0);
        assert_clean();
    }
}

/// Lens distortion and chromatic aberration in tonemap's gather, and the distortion field it
/// writes, against the analytic map.
mod lens_distortion {
    use orr_output::{
        DistortionField, Exposure, ExposurePass, OutputDebug, OutputFormat, OutputKernels,
        OutputView, ResolvedDisplay, TonemapPass, ViewOutput,
    };
    use orr_scene::orr_gpu::{
        Allocation, Format, Hazards, MemoryKind, QueueKind, Stages, StorageImageIndex, Texture,
        TextureDesc, TextureIndex, TextureMemory, TextureRegion, TextureUsage, ViewDesc,
    };
    use orr_scene::testing::{assert_clean, device, read_texture, wait};
    use orr_scene::{ExposureState, LensDistortion, SkyStats, ViewConstants};
    use output_suite::{LensModel, read_rgba16f};

    /// Not a multiple of the 8-pixel group, and odd high, so the field rounds up.
    const OUT: [u32; 2] = [484, 271];

    const BARREL: LensDistortion = LensDistortion {
        k1: 0.12,
        k2: 0.03,
        chromatic_aberration: 0.006,
    };
    const PINCUSHION: LensDistortion = LensDistortion {
        k1: -0.1,
        k2: 0.0,
        chromatic_aberration: -0.004,
    };

    /// RGBA32F radiance tonemapped at unit exposure with the linear debug view on an scRGB
    /// display, so each output channel is the Rec.709 of the gathered Rec.2020 taps.
    struct Lens {
        kernels: OutputKernels,
        output: OutputView,
        display: ResolvedDisplay,
        exposure: Allocation<ExposureState>,
        constants: Allocation<ViewConstants>,
        sky: Allocation<SkyStats>,
        upload: Allocation<[f32; 4]>,
        source: Texture,
        target: Texture,
        source_index: TextureIndex,
        target_store: StorageImageIndex,
        frame: u64,
    }

    impl Lens {
        fn new() -> Self {
            let gpu = device();
            let texture = |name| {
                let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
                let desc = TextureDesc::d2(Format::Rgba32Float, OUT[0], OUT[1], usage);
                gpu.create_texture(&desc, TextureMemory::Dedicated, name)
                    .unwrap()
            };
            let source = texture("lens source");
            let target = texture("lens target");
            let view = ViewDesc::default();
            let mut exposure = gpu.gpu_malloc(MemoryKind::Upload, 2, "exposure").unwrap();
            exposure.as_mut_slice().unwrap().fill(ExposureState {
                pre_exposure: 1.0,
                prev_pre_exposure: 1.0,
                ..ExposureState::default()
            });
            let mut sky = gpu.gpu_malloc(MemoryKind::Upload, 1, "sky").unwrap();
            sky.as_mut_slice().unwrap()[0] = bytemuck::Zeroable::zeroed();
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.discard(&source);
            list.discard(&target);
            wait(queue.submit(vec![list], &[]).unwrap());
            Self {
                kernels: OutputKernels::new(gpu).unwrap(),
                output: OutputView::new(gpu, 2).unwrap(),
                display: ResolvedDisplay {
                    encoding: OutputFormat::ScRgb,
                    paper_white_nits: 250.0,
                    signal_white_nits: 250.0,
                    peak_nits: 1000.0,
                    black_nits: 0.0,
                },
                exposure,
                constants: gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap(),
                sky,
                upload: gpu
                    .gpu_malloc(
                        MemoryKind::Upload,
                        u64::from(OUT[0] * OUT[1]),
                        "lens upload",
                    )
                    .unwrap(),
                source_index: gpu.write_sampled(gpu.alloc_image_slot().unwrap(), &source, &view),
                target_store: gpu.write_storage(gpu.alloc_image_slot().unwrap(), &target, &view),
                source,
                target,
                frame: 0,
            }
        }

        /// Tonemaps `image` (linear Rec.2020, row by row) through `lens`, writing `field`;
        /// returns the output in linear Rec.2020.
        fn render(
            &mut self,
            image: &[[f32; 4]],
            lens: LensDistortion,
            field: Option<&mut DistortionField>,
        ) -> Vec<[f64; 3]> {
            let gpu = device();
            self.frame += 1;
            self.upload.as_mut_slice().unwrap().copy_from_slice(image);
            let mut c: ViewConstants = bytemuck::Zeroable::zeroed();
            c.exposure = self.exposure.gpu();
            c.exposure_slot = (self.frame % 2) as u32;
            c.extent = OUT;
            c.delta_time = 1.0 / 60.0;
            self.constants.as_mut_slice().unwrap()[0] = c;
            // exposure_from_ev100 is 1 / (1.2 * 2^ev100).
            let output = ViewOutput {
                exposure: Exposure::Manual {
                    ev100: -(1.2f32.log2()),
                },
                debug: OutputDebug::Linear,
                ..ViewOutput::default()
            };
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            let whole = TextureRegion::whole_mip(self.source.desc(), 0);
            list.copy_to_texture(&self.source, &whole, self.upload.gpu().cast());
            list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
            self.kernels
                .exposure(
                    &mut list,
                    &mut self.output,
                    &ExposurePass {
                        source: self.source_index,
                        extent: OUT,
                        view: self.constants.gpu(),
                        sky: self.sky.gpu(),
                        frame: self.frame,
                        output: &output,
                    },
                )
                .unwrap();
            let roots = self.output.roots(self.frame);
            self.kernels.tonemap(
                &mut list,
                roots,
                &TonemapPass {
                    source: self.source_index,
                    target: self.target_store,
                    extent: OUT,
                    view: self.constants.gpu(),
                    display: &self.display,
                    output: &output,
                    hook_source: None,
                    glare: None,
                    distortion: Some(lens),
                    frame: self.frame,
                },
                field,
            );
            wait(queue.submit(vec![list], &[]).unwrap());
            read_texture(&self.target, &whole)
                .as_chunks::<4>()
                .0
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| {
                    let c: [f64; 3] = std::array::from_fn(|i| f64::from(f32::from_le_bytes(p[i])));
                    orr_color::rec709_to_rec2020(c).to_array()
                })
                .collect()
        }
    }

    /// Gray radiance holding each texel's center along `axis` as a fraction of the extent,
    /// so a bilinear tap returns its own position.
    fn ramp(axis: usize) -> Vec<[f32; 4]> {
        (0..OUT[1])
            .flat_map(|y| (0..OUT[0]).map(move |x| [x, y]))
            .map(|p| {
                let v = (p[axis] as f32 + 0.5) / OUT[axis] as f32;
                [v, v, v, 1.0]
            })
            .collect()
    }

    /// Whether a tap at `s` lies between the outermost texel centers, where bilinear
    /// filtering reproduces the ramp.
    fn inside(s: [f64; 2]) -> bool {
        (0..2).all(|a| s[a] >= 0.5 && s[a] <= f64::from(OUT[a]) - 0.5)
    }

    /// The red, green and blue taps of every output pixel, measured from the ramps.
    fn measured_taps(v: &mut Lens, lens: LensDistortion) -> Vec<[[f64; 2]; 3]> {
        let x = v.render(&ramp(0), lens, None);
        let y = v.render(&ramp(1), lens, None);
        x.iter()
            .zip(&y)
            .map(|(x, y)| {
                std::array::from_fn(|c| [x[c] * f64::from(OUT[0]), y[c] * f64::from(OUT[1])])
            })
            .collect()
    }

    fn check_taps(lens: LensDistortion) {
        let mut v = Lens::new();
        let model = LensModel::new(&lens, OUT);
        let taps = measured_taps(&mut v, lens);
        let (mut worst, mut checked) = (0.0f64, 0usize);
        let mut reaches_edge = false;
        for (i, measured) in taps.iter().enumerate() {
            let p = [
                (i % OUT[0] as usize) as f64 + 0.5,
                (i / OUT[0] as usize) as f64 + 0.5,
            ];
            let g = model.forward(p);
            let (r, b) = model.red_blue(p);
            for (want, got) in [r, g, b].into_iter().zip(measured) {
                reaches_edge |= want
                    .iter()
                    .zip(OUT)
                    .any(|(&s, n)| s < 1.0 || s > f64::from(n) - 1.0);
                assert!(
                    (0..2).all(|a| want[a] > 0.0 && want[a] < f64::from(OUT[a])),
                    "{lens:?}: pixel {p:?} samples {want:?}, outside the frame"
                );
                if inside(want) {
                    worst = worst.max((want[0] - got[0]).hypot(want[1] - got[1]));
                    checked += 1;
                }
            }
        }
        log::info!("{lens:?}: taps within {worst:.4} px of the analytic map");
        assert!(
            worst <= 0.1,
            "{lens:?}: a tap is {worst:.3} px from the analytic map"
        );
        assert!(checked > taps.len() * 3 * 9 / 10, "{checked} taps checked");
        assert!(
            reaches_edge,
            "{lens:?}: no border pixel samples the frame's edge"
        );

        // Red and blue sit at the set scale of green's offset from the center, at each corner.
        let center = OUT.map(|n| 0.5 * f64::from(n));
        let ca = f64::from(lens.chromatic_aberration);
        for [x, y] in [
            [3, 3],
            [OUT[0] - 4, 3],
            [3, OUT[1] - 4],
            [OUT[0] - 4, OUT[1] - 4],
        ] {
            let [r, g, b] = taps[(y * OUT[0] + x) as usize];
            for (tap, scale) in [(r, ca), (b, -ca)] {
                for a in 0..2 {
                    let want = scale * (g[a] - center[a]);
                    let got = tap[a] - g[a];
                    assert!(
                        (got - want).abs() <= 0.1,
                        "{lens:?}: shift at corner pixel {x},{y}: {got:.3} px, expected {want:.3}"
                    );
                }
            }
        }
        assert_clean();
    }

    #[test]
    fn barrel_taps_follow_the_analytic_map() {
        check_taps(BARREL);
    }

    #[test]
    fn pincushion_taps_follow_the_analytic_map() {
        check_taps(PINCUSHION);
    }

    /// The field's forward map, then its inverse sampled bilinearly where the forward map
    /// lands, returns every texel to itself.
    #[test]
    fn field_round_trips() {
        let mut v = Lens::new();
        for lens in [BARREL, PINCUSHION] {
            let mut field = DistortionField::new(device(), OUT).unwrap();
            v.render(&ramp(0), lens, Some(&mut field));
            let texels: Vec<[f64; 4]> = read_rgba16f(field.current_texture())
                .into_iter()
                .map(|t| t.map(f64::from))
                .collect();
            let [fw, fh] = field.extent();
            let size = [f64::from(fw), f64::from(fh)];
            let texel = |x: f64, y: f64| {
                let x = x.clamp(0.0, size[0] - 1.0) as usize;
                let y = y.clamp(0.0, size[1] - 1.0) as usize;
                texels[y * fw as usize + x]
            };
            let inverse_at = |uv: [f64; 2]| -> [f64; 2] {
                let at: [f64; 2] = std::array::from_fn(|a| uv[a] * size[a] - 0.5);
                let [i, j] = at.map(f64::floor);
                let [tx, ty] = [at[0] - i, at[1] - j];
                std::array::from_fn(|c| {
                    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
                    let top = lerp(texel(i, j)[2 + c], texel(i + 1.0, j)[2 + c], tx);
                    let bottom = lerp(texel(i, j + 1.0)[2 + c], texel(i + 1.0, j + 1.0)[2 + c], tx);
                    lerp(top, bottom, ty)
                })
            };
            let (mut worst, mut checked) = (0.0f64, 0);
            for (i, f) in texels.iter().enumerate() {
                let t = [(i % fw as usize) as f64, (i / fw as usize) as f64];
                let uv = [(t[0] + 0.5) / size[0], (t[1] + 0.5) / size[1]];
                let q = [uv[0] + f[0], uv[1] + f[1]];
                if !(0..2).all(|a| q[a] >= 0.5 / size[a] && q[a] <= 1.0 - 0.5 / size[a]) {
                    continue;
                }
                let back = inverse_at(q);
                let dx = (q[0] + back[0] - uv[0]) * f64::from(OUT[0]);
                let dy = (q[1] + back[1] - uv[1]) * f64::from(OUT[1]);
                worst = worst.max(dx.hypot(dy));
                checked += 1;
            }
            log::info!("{lens:?}: field round trip within {worst:.4} px");
            assert!(checked > texels.len() * 9 / 10, "{checked} texels checked");
            assert!(
                worst <= 0.05,
                "{lens:?}: field round trip off by {worst:.4} px"
            );
        }
        assert_clean();
    }
}
