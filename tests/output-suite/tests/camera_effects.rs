//! Camera effects after RR: motion blur against shutter-accumulated references, depth of
//! field against lens-disc references and glare against its CPU model and PSF.

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

/// Depth of field on both paths against lens-disc accumulation references.
mod depth_of_field {
    use orr_output::{
        ChainImages, DepthOfFieldPass, DofPath, OutputKernels, OutputView, ViewOutput, srgb_encode,
    };
    use orr_scene::orr_gpu::{
        Allocation, ComputePipeline, Format, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData,
        Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
        TextureRegion, TextureUsage, ViewDesc,
    };
    use orr_scene::testing::{
        FlipError, Image, Pixels, assert_clean, device, flip, read_buffer, read_texture, wait,
    };
    use orr_scene::{DepthOfField, Focus, ViewConstants};
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

    /// One view's chain and render targets at `OUT` (render extent equal to output, no
    /// jitter, no motion), filled by `dof_scene.slang`.
    struct Bed {
        kernels: OutputKernels,
        output: OutputView,
        scene: ComputePipeline,
        reference: ComputePipeline,
        root: Allocation<dof_scene::Root>,
        constants: Allocation<ViewConstants>,
        rr_output: Texture,
        linear: Texture,
        depth: TextureIndex,
        depth_store: StorageImageIndex,
        motion: TextureIndex,
        motion_store: StorageImageIndex,
        chain: ChainImages,
        reference_out: Allocation<f32>,
        frame: u64,
        _targets: [Texture; 2],
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
                let stage =
                    ShaderStage::new(&dof_scene::MODULE, entry, SpecData::default()).unwrap();
                gpu.create_compute_pipeline(&stage, name).unwrap()
            };
            let mut constants = gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap();
            let mut c: ViewConstants = bytemuck::Zeroable::zeroed();
            c.clip_from_view[0][0] = 1.0 / TAN_X;
            c.clip_from_view[1][1] = -1.0 / tan_y();
            c.clip_from_view[3][2] = -1.0;
            c.clip_from_view[2][3] = NEAR;
            c.extent = OUT;
            c.delta_time = 1.0 / 60.0;
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
                scene: pipeline("frame", "dof scene"),
                reference: pipeline("reference", "dof reference"),
                root: gpu.gpu_malloc(MemoryKind::Upload, 1, "scene root").unwrap(),
                constants,
                depth: gpu.write_sampled(slot(), &depth, &view),
                depth_store: gpu.write_storage(slot(), &depth, &view),
                motion: gpu.write_sampled(slot(), &motion, &view),
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

        fn put_root(&mut self, scene: Scene, dof: &DepthOfField) {
            let polygon = dof.blades >= 3 && dof.f_number > dof.round_at_f_number;
            self.root.as_mut_slice().unwrap()[0] = dof_scene::Root {
                reference: self.reference_out.gpu,
                radiance: self.chain.rr_output_store,
                depth: self.depth_store,
                motion: self.motion_store,
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
            self.put_root(scene, &dof.unwrap_or(lens(0, 1.4)));
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            list.dispatch(&self.scene, self.root.gpu, groups(OUT));
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
            let mut output = ViewOutput::default();
            output.post.dof_path = path;
            let pass = DepthOfFieldPass {
                rr_depth: self.depth,
                rr_motion: self.motion,
                extent: OUT,
                view: self.constants.gpu,
                sensor_width_mm: SENSOR_MM,
                depth_of_field: dof,
                history_reset: false,
                frame: self.frame,
                output: &output,
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

        fn reference(&mut self, scene: Scene, dof: &DepthOfField) -> Vec<[f32; 4]> {
            let gpu = device();
            self.put_root(scene, dof);
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

    fn read_r16f(t: &Texture) -> Vec<f32> {
        read_texture(t, &TextureRegion::whole_mip(t.desc(), 0))
            .chunks_exact(2)
            .map(|p| half::f16::from_le_bytes([p[0], p[1]]).to_f32())
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
/// they are fitted to.
mod glare {
    use orr_output::{
        Exposure, OutputDebug, OutputFormat, ResolvedDisplay, ViewOutput, srgb_encode,
    };
    use orr_scene::Glare;
    use orr_scene::testing::{Image, Pixels, assert_clean, flip};
    use output_suite::{Harness, REC709_TO_XYZ, XYZ_TO_REC2020, mul3};

    #[allow(dead_code, unreachable_pub)]
    mod pyramid {
        include!(concat!(env!("ORR_GLARE_MODEL"), "/pyramid.rs"));
    }
    #[allow(dead_code, unreachable_pub)]
    mod psf {
        include!(concat!(env!("ORR_GLARE_MODEL"), "/psf.rs"));
    }
    #[allow(
        dead_code,
        unreachable_pub,
        clippy::unreadable_literal,
        clippy::excessive_precision
    )]
    mod data {
        include!(env!("ORR_GLARE_PSF"));
    }

    use pyramid::Plane;

    /// An emitter's exposed value: within the HDR ceiling at any whole-stop pre-exposure.
    const EXPOSED: f32 = 8000.0;

    /// The EV100 at which the lens is at table F-number `i` at the reference pitch.
    fn ev_at(i: usize) -> f32 {
        (100.0 * data::F_NUMBERS[i] * data::F_NUMBERS[i]).log2() as f32
    }

    /// An scRGB display; Linear debug output on it is not clamped.
    fn hdr(peak_nits: f32) -> ResolvedDisplay {
        ResolvedDisplay {
            encoding: OutputFormat::ScRgb,
            paper_white_nits: 250.0,
            peak_nits,
            black_nits: 0.0,
        }
    }

    fn sdr() -> ResolvedDisplay {
        ResolvedDisplay {
            encoding: OutputFormat::Sdr,
            paper_white_nits: 250.0,
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
        h.sensor_width_mm = (data::REFERENCE_PITCH_UM * f64::from(extent[0]) / 1000.0) as f32;
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
            .map(|p| {
                let c = [p[0], p[1], p[2]].map(f64::from);
                mul3(&XYZ_TO_REC2020, mul3(&REC709_TO_XYZ, c))
            })
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
        let f = &data::F_NUMBERS;
        let n = n.clamp(f[0], f[f.len() - 1]);
        let t = (n / f[0]).ln() / (f[f.len() - 1] / f[0]).ln() * (f.len() - 1) as f64;
        let i = (t as usize).min(f.len() - 2);
        let rows = &data::TABLES[first_level - 1];
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

    /// The CPU model of the GPU run: per channel, the pyramid of the pre-exposed radiance the
    /// harness rendered, scaled by `scale`, at the frame's weights.
    fn model(
        h: &Harness,
        extent: [u32; 2],
        first_level: u32,
        scale: f64,
        w: &[[f64; 4]],
    ) -> Vec<[f64; 3]> {
        let radiance = h.radiance();
        let mut out = vec![[0.0; 3]; radiance.len()];
        for c in 0..3 {
            let input = Plane::dense(extent, radiance.iter().map(|p| f64::from(p[c])).collect());
            let wc: Vec<f64> = w.iter().map(|e| e[c]).collect();
            let downs = pyramid::downsamples(&input, first_level, scale);
            let glare = pyramid::compose(&input, &downs, &wc, scale);
            for (o, v) in out.iter_mut().zip(&glare.data) {
                o[c] = *v;
            }
        }
        out
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
    fn tables_are_non_negative_and_keep_energy() {
        for first in &data::TABLES {
            for row in first {
                for c in 0..3 {
                    assert!(row.iter().all(|w| w[c] >= 0.0), "{row:?}");
                    let sum: f64 = row.iter().map(|w| w[c]).sum();
                    assert!((sum - 1.0).abs() < 1e-9, "{sum}");
                }
            }
        }
    }

    #[test]
    fn cpu_fit_keeps_uniform_fields_and_point_energy() {
        let w: Vec<f64> = data::TABLES[0][6].iter().map(|w| w[1]).collect();
        let size = [96, 64];
        let input = Plane::dense(size, vec![3.0; 96 * 64]);
        let downs = pyramid::downsamples(&input, 1, 0.5);
        let out = pyramid::compose(&input, &downs, &pyramid::fold(&w, downs.len()), 0.5);
        for v in &out.data {
            assert!((v / 1.5 - 1.0).abs() < 1e-9, "{v}");
        }
        let size = [4096; 2];
        let input = Plane::impulse(size, [2048, 2048], 1.0);
        let out = pyramid::compose(&input, &pyramid::downsamples(&input, 1, 1.0), &w, 1.0);
        assert!((out.sum() - 1.0).abs() < 0.01, "{}", out.sum());
    }

    #[test]
    fn pyramid_matches_cpu_model() {
        let extent = [160, 96];
        let mut h = harness(extent, Some(1.0));
        let ev = ev_at(5) + 0.37;
        let nits = scene(
            extent,
            200.0,
            &[
                ([40, 30], 4e6),
                ([41, 30], 2e6),
                ([120, 70], 6e6),
                ([159, 95], 1e6),
            ],
        );
        for first_level in [1, 2] {
            let mut out = output(ev, OutputDebug::Linear);
            out.post.glare_first_level = first_level;
            let (gpu, exposure, pre) = settle(&mut h, &nits, &out, &hdr(250.0));
            let w = weights(&h);
            let scale = exposure / pre;
            let cpu = model(&h, extent, first_level, scale, &w);
            let radiance = h.radiance();
            let mut worst: f64 = 0.0;
            for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
                for k in 0..3 {
                    let identity = w[0][k] * scale * f64::from(radiance[i][k]);
                    let glare = (c[k] - identity).abs();
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

    #[test]
    fn uniform_field_is_unchanged() {
        let extent = [200, 120];
        let nits = vec![900.0; 200 * 120];
        let out = output(ev_at(7), OutputDebug::Linear);
        let (plain, ..) = settle(&mut harness(extent, None), &nits, &out, &hdr(250.0));
        let (glared, ..) = settle(&mut harness(extent, Some(1.0)), &nits, &out, &hdr(250.0));
        let mut worst: f64 = 0.0;
        for (a, b) in plain.iter().zip(&glared) {
            for k in 0..3 {
                worst = worst.max((b[k] / a[k] - 1.0).abs());
            }
        }
        log::info!("uniform field: worst relative change {worst:.2e}");
        assert!(worst < 1e-3, "{worst}");
        assert_clean();
    }

    #[test]
    fn point_source_keeps_energy() {
        let extent = [1024, 1024];
        let ev = ev_at(6);
        let mut nits = vec![0.0; 1024 * 1024];
        let source = EXPOSED * 1.2 * ev.exp2();
        nits[512 * 1024 + 512] = source;
        let out = output(ev, OutputDebug::Linear);
        let (image, exposure, _) =
            settle(&mut harness(extent, Some(1.0)), &nits, &out, &hdr(250.0));
        let energy = f64::from(source) * exposure;
        for k in 0..3 {
            let sum: f64 = image.iter().map(|p| p[k]).sum();
            log::info!(
                "point source: channel {k} keeps {:.5} of its energy",
                sum / energy
            );
            assert!((sum / energy - 1.0).abs() < 0.01, "{k}: {}", sum / energy);
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
            psf::encircled_rgb(
                &data::EE,
                data::EE_TAIL,
                &data::SPECTRAL,
                r,
                f_number,
                data::REFERENCE_PITCH_UM,
            )
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
        let i = 6;
        let ev = ev_at(i);
        let source = EXPOSED * 1.2 * ev.exp2();
        let mut nits = vec![0.0; 2048 * 2048];
        nits[1024 * 2048 + 1024] = source;
        let mut h = harness(extent, Some(1.0));
        let out = output(ev, OutputDebug::Linear);
        let (image, exposure, _) = settle(&mut h, &nits, &out, &hdr(250.0));
        assert_weights(
            &weights(&h),
            &expected_weights(1, data::F_NUMBERS[i], 8),
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
        let reference = log_image(side, &psf_image(side, data::F_NUMBERS[i]), 1.0);
        let error = flip(&reference, &fitted);
        log::info!(
            "fitted kernel against the PSF at f/{:.2}: FLIP {error:?}",
            data::F_NUMBERS[i]
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

    /// Distance from `center` along its row to the last pixel at or above `level` (green,
    /// display-linear).
    fn reach(image: &[[f32; 4]], width: u32, center: [u32; 2], level: f32) -> u32 {
        let row = (center[1] * width) as usize;
        (center[0]..width)
            .take_while(|&x| image[row + x as usize][1] >= level)
            .count() as u32
    }

    /// The HDR rendition's halo at the SDR clip level (250 nits) has the SDR rendition's
    /// extent within one pyramid level.
    #[test]
    fn hdr_halo_at_sdr_clip_matches_sdr_halo() {
        let extent = [512, 512];
        let ev = 8.0;
        let mut nits = vec![0.05 * 1.2 * 2f32.powf(ev); 512 * 512];
        nits[256 * 512 + 256] = EXPOSED * 1.2 * 2f32.powf(ev);
        let hdr = ResolvedDisplay {
            encoding: OutputFormat::Hdr10Pq,
            paper_white_nits: 250.0,
            peak_nits: 1000.0,
            black_nits: 0.0,
        };
        let out = output(ev, OutputDebug::None);
        let mut h = harness(extent, Some(1.0));
        settle(&mut h, &nits, &out, &sdr());
        let sdr_reach = reach(&h.display_linear(), 512, [256, 256], 0.99);
        settle(&mut h, &nits, &out, &hdr);
        let hdr_reach = reach(&h.display_linear(), 512, [256, 256], 1.0);
        log::info!("halo reach: SDR clip {sdr_reach} px, HDR 250 nits {hdr_reach} px");
        let ratio = f64::from(hdr_reach.max(1)) / f64::from(sdr_reach.max(1));
        assert!(
            ratio.log2().abs() <= 1.0,
            "SDR {sdr_reach}, HDR {hdr_reach}"
        );
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
}
