//! A cooked scene path traced by orr: a gold knot, a glossy and a mirror sphere, a block and
//! a glowing panel on a ground, under a cooked sky and a 100 klx sun. Each frame traces one
//! path per pixel, auto exposure meters and adapts to the result, and on Windows an egui
//! overlay shows the exposure. The camera orbits the scene; a still camera averages its
//! frames in a scene-referred hook.
//!
//! `viewer [--still] [--frames N] [--headless] [--out file.png]`. Headless runs keep the
//! camera still, render 480 x 270 for 120 frames unless told otherwise and write the last
//! frame as a PNG; windowed runs with `--out` write their last frame as the window shows
//! it, overlay included.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use examples::mesh::MeshBuilder;
use examples::shaders::accumulate;
use examples::window::Window;
use examples::{Capture, Options, Workers, assert_clean, device, fixed_time};
use orr::orr_gpu::{
    AddressMode, Allocation, Filter, Format, MemoryKind, QueueKind, SamplerDesc, SpecData,
    TextureDesc, TextureMemory, TextureUsage, TimelinePoint,
};
use orr::orr_lighting::EnvironmentMapGpu;
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    Affine3, DirectionalLight, DiscProfile, Display, DisplayRequest, FrameDesc, FrameTime,
    GroupWriter, HookPass, InstanceFlags, InstancesDesc, Lens, Light, LightId, LoadState,
    PackSource, Projection, Renderer, RendererDesc, SceneDesc, SceneLightingSettings,
    ScreenshotRequest, TransformGroupDesc, ViewDesc, ViewHooks, ViewLighting, ViewOutput,
    ViewTarget,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Quality, Source, TextureSettings,
    Usage,
};
use orr_math::camera::rh::view;
use orr_math::{DVec3, UVec2, Vec2, Vec3};

const OBJECTS: &str = "objects";
const GROUND: &str = "ground";
const SKY: &str = "sky";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
const FOV_Y: f32 = 0.8;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.453_413_4, 0.654_930_5, 0.604_551_3);
/// Lux, linear Rec.709.
const SUN_ILLUMINANCE: Vec3 = Vec3::new(100_000.0, 95_000.0, 88_000.0);
/// The sun's angular radius, radians.
const SUN_RADIUS: f32 = 0.004_65;
const TARGET: Vec3 = Vec3::new(0.0, 0.9, 0.0);
/// Radians per second.
const ORBIT_RATE: f64 = 0.2;
/// Where a still camera stands on the orbit, radians.
const STILL_ANGLE: f64 = 1.3;
/// Frames a still camera renders before it starts averaging, while exposure settles.
const ACCUMULATE_AFTER: u64 = 30;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("viewer");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Runs the example; headless runs and runs with `--out` return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr viewer", 1280, 720));
    let desc = RendererDesc::default();
    let mut display = window.as_ref().map(|w| {
        Display::new(
            gpu,
            w.window_handle(),
            w.display_handle(),
            &DisplayRequest::default(),
            None,
            desc.frames_in_flight,
        )
        .expect("display")
    });
    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    let _workers = Workers::new(renderer.loading(), 2);

    let mount = renderer.mount(PackSource::Memory(cook()));
    let mut frame = 0;
    let start = Instant::now();
    // Empty frames until the pack is mounted and the frame id reaches a fixed value, since
    // the light sampling's random sequence follows the frame id and headless runs repeat.
    while matches!(renderer.mount_state(mount), LoadState::Pending) || frame < FIRST_FRAME {
        frame += 1;
        let f = renderer
            .begin_frame(FrameDesc {
                time: fixed_time(frame),
                scenes: &[],
                display: None,
            })
            .expect("begin frame");
        f.end().expect("end frame");
        assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
        if matches!(renderer.mount_state(mount), LoadState::Pending) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    if frame > FIRST_FRAME {
        log::warn!("mounting took {frame} frames; this run will not repeat others exactly");
    }
    if let LoadState::Failed(e) = renderer.mount_state(mount) {
        panic!("mount failed: {e}");
    }

    let scene = renderer
        .create_scene(
            &SceneDesc {
                instance_capacity: 16,
                group_capacity: 4,
                light_capacity: 1,
                material_set_capacity: 16,
                emissive_capacity: 16,
                part_capacity: 0,
                ptlas_partitions: 16,
                global_instances: 16,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    let models = [OBJECTS, GROUND].map(|name| {
        renderer
            .model(mount, AssetId::from_name(name))
            .unwrap_or_else(|e| panic!("model {name}: {e}"))
    });
    let mut environment = gpu
        .gpu_malloc::<EnvironmentMapGpu>(MemoryKind::Upload, 1, "viewer sky")
        .expect("environment");
    environment.as_mut_slice().expect("environment")[0] = EnvironmentMapGpu {
        map: renderer
            .texture(mount, AssetId::from_name(SKY))
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

    frame += 1;
    let listed = [scene];
    let f = renderer
        .begin_frame(FrameDesc {
            time: fixed_time(frame),
            scenes: &listed,
            display: None,
        })
        .expect("begin frame");
    let mut w = f.scene(scene).expect("scene writer");
    let group = w
        .create_transform_group(&TransformGroupDesc {
            count: models.len() as u32,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .expect("transform group");
    w.transforms(group, 0, models.len() as u32)
        .expect("transforms")
        .fill(Affine3::IDENTITY);
    for (i, &model) in models.iter().enumerate() {
        w.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first: i as u32,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .expect("instances");
    }
    w.set_light(
        LightId(0),
        Some(&Light::Directional(DirectionalLight {
            direction_to_light: SUN,
            illuminance: SUN_ILLUMINANCE,
            angular_radius: SUN_RADIUS,
            disc: DiscProfile::LimbDarkened,
        })),
    )
    .expect("sun");
    w.set_environment(environment.gpu.address());
    drop(w);
    f.end().expect("end frame");

    let headless_target = options.headless.then(|| {
        let [w, h] = HEADLESS_EXTENT;
        gpu.create_texture(
            &TextureDesc::d2(
                Format::Rgba8Unorm,
                w,
                h,
                TextureUsage::STORAGE | TextureUsage::SAMPLED,
            ),
            TextureMemory::Dedicated,
            "viewer target",
        )
        .expect("headless target")
    });
    let mut overlay = Overlay::new(desc.frames_in_flight);
    let capture = options.headless || options.out.is_some();
    let still = options.headless || options.has("--still");
    let accumulate = renderer
        .compute_pipeline(&accumulate::MODULE, "main", SpecData::default())
        .expect("accumulate pipeline");
    let slots = u64::from(desc.frames_in_flight);
    let mut roots = gpu
        .gpu_malloc::<accumulate::Root>(MemoryKind::Upload, slots, "viewer accumulate roots")
        .expect("accumulate roots");
    let mut sum: Option<(Allocation<[f32; 4]>, [u32; 2])> = None;
    let mut accumulated = 0;
    let frames = options
        .frames
        .or(options.headless.then_some(HEADLESS_FRAMES));
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<(u64, TimelinePoint)> = None;
    for i in 0.. {
        if frames.is_some_and(|n| i >= n) {
            break;
        }
        if let Some(w) = window.as_mut()
            && !w.pump()
        {
            break;
        }
        let final_frame = frames.is_some_and(|n| i + 1 == n);
        // Headless runs step a fixed clock, so they repeat exactly.
        let t = if options.headless {
            fixed_time(i).absolute
        } else {
            started.elapsed().as_secs_f64()
        };
        let time = FrameTime {
            absolute: fixed_time(frame + 1).absolute + t,
            delta: (t - previous) as f32,
        };
        previous = t;
        let angle = if still {
            STILL_ANGLE
        } else {
            0.9 + t * ORBIT_RATE
        };
        let eye = TARGET + Vec3::new(angle.sin() as f32 * 6.5, 1.1, angle.cos() as f32 * 6.5);

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let id = f.id();
        let telemetry = f.exposure_telemetry(history);
        let (extent, target) = match (&headless_target, f.display()) {
            (Some(t), _) => (HEADLESS_EXTENT, ViewTarget::Texture(t)),
            (None, Some((resolved, extent))) => {
                overlay.layout(extent, time, &resolved, telemetry.as_ref());
                (extent, ViewTarget::Display)
            }
            (None, None) => {
                last = Some((id, f.end().expect("end frame")));
                continue;
            }
        };
        f.scene(scene)
            .expect("scene writer")
            .set_focus(eye.as_dvec3(), extent[1] as f32 / FOV_Y);
        if let Some(t) = headless_target.as_ref().filter(|_| i == 0) {
            f.list().expect("frame list").discard(t);
        }
        let output = ViewOutput {
            screenshot: (capture && final_frame).then_some(ScreenshotRequest::Display {
                include_ui: !options.headless,
            }),
            ..ViewOutput::default()
        };
        if sum.as_ref().is_none_or(|(_, e)| *e != extent) {
            let texels = u64::from(extent[0]) * u64::from(extent[1]);
            let buffer = gpu
                .gpu_malloc(MemoryKind::DeviceLocal, texels, "viewer radiance sum")
                .expect("radiance sum");
            sum = Some((buffer, extent));
            accumulated = 0;
        }
        let sum_ptr = sum.as_ref().expect("radiance sum").0.gpu;
        let frames_in_sum = if i < ACCUMULATE_AFTER { 0 } else { accumulated };
        let pipeline = f.pipeline(accumulate);
        let slot = id % slots;
        let mut average = |pass: &mut HookPass<'_, '_>| {
            roots.as_mut_slice().expect("accumulate roots")[slot as usize] = accumulate::Root {
                view: pass.view,
                sum: sum_ptr,
                image: pass.image,
                width: pass.extent[0],
                height: pass.extent[1],
                frames: frames_in_sum,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            pass.cmd.begin_pass("accumulate", false);
            pass.cmd.dispatch(
                pipeline,
                roots.gpu.offset(slot),
                [pass.extent[0].div_ceil(8), pass.extent[1].div_ceil(8), 1],
            );
            pass.cmd.end_pass();
        };
        let mut ui = |pass: &mut orr::UiPass<'_, '_>| overlay.paint(pass, id);
        let mut hooks = ViewHooks {
            scene_referred: still.then_some(&mut average as &mut dyn FnMut(&mut HookPass<'_, '_>)),
            ui: Some(&mut ui),
            ..ViewHooks::default()
        };
        f.render_view(
            &ViewDesc {
                scene,
                history,
                position: eye.as_dvec3(),
                rotation: view::look_at_quat(eye, TARGET, Vec3::Y).inverse(),
                projection: Projection {
                    fov_y: FOV_Y,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: UVec2::from(extent),
                jitter: Vec2::ZERO,
                reset_history: i == 0,
                mip_bias: 0.0,
                target,
            },
            &ViewLighting::default(),
            &output,
            &mut hooks,
        )
        .expect("render view");
        last = Some((id, f.end().expect("end frame")));
        if i >= ACCUMULATE_AFTER {
            accumulated += 1;
        }
    }

    if let Some(d) = display.as_ref() {
        d.wait_idle();
    }
    if let Some((_, point)) = last {
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(10))
            .expect("wait");
    }
    let captured = last
        .filter(|_| capture)
        .and_then(|(id, _)| renderer.screenshots(history, id).pop())
        .map(|shot| Capture {
            width: shot.size[0],
            height: shot.size[1],
            rgba8: shot
                .rgba
                .iter()
                .enumerate()
                .map(|(k, &v)| {
                    let v = f64::from(v.clamp(0.0, 1.0));
                    let v = if k % 4 == 3 {
                        v
                    } else {
                        orr_color::srgb_encode(v)
                    };
                    (v * 255.0).round() as u8
                })
                .collect(),
        });
    drop(sum);
    drop(roots);
    drop(overlay);
    drop(headless_target);
    drop(environment);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// The exposure readout drawn over the view, through egui on Windows.
#[cfg(windows)]
struct Overlay {
    ctx: egui::Context,
    painter: orr_egui::Painter,
    pending: egui::TexturesDelta,
    primitives: Vec<egui::ClippedPrimitive>,
    pixels_per_point: f32,
}

#[cfg(windows)]
impl Overlay {
    fn new(frames_in_flight: u32) -> Self {
        Self {
            ctx: egui::Context::default(),
            painter: orr_egui::Painter::new(device(), frames_in_flight).expect("painter"),
            pending: egui::TexturesDelta::default(),
            primitives: Vec::new(),
            pixels_per_point: 1.0,
        }
    }

    fn layout(
        &mut self,
        [w, h]: [u32; 2],
        time: FrameTime,
        resolved: &orr::ResolvedDisplay,
        telemetry: Option<&orr::ExposureTelemetry>,
    ) {
        use egui::{Color32, Pos2, RawInput, Rect, RichText, vec2};
        let text = match telemetry {
            Some(t) => format!(
                "{w} x {h}  {:?}  paper white {:.0} nits\n\
                 EV100 {:.2} (target {:.2})  pre-exposure {:.3e}\n\
                 frame time {:.1} ms",
                resolved.encoding,
                resolved.paper_white_nits,
                t.adapted_ev100,
                t.target_ev100,
                t.pre_exposure,
                time.delta * 1000.0,
            ),
            None => format!("{w} x {h}  {:?}", resolved.encoding),
        };
        let out = self.ctx.run_ui(
            RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(w as f32, h as f32))),
                time: Some(time.absolute),
                predicted_dt: time.delta,
                ..RawInput::default()
            },
            |ui| {
                egui::Frame::new()
                    .fill(Color32::from_black_alpha(200))
                    .inner_margin(8)
                    .show(ui, |ui| {
                        ui.label(RichText::new(&text).size(16.0).color(Color32::WHITE));
                    });
            },
        );
        self.primitives = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        self.pixels_per_point = out.pixels_per_point;
        self.pending.append(out.textures_delta);
    }

    fn paint(&mut self, pass: &mut orr::UiPass<'_, '_>, frame: u64) {
        self.painter
            .update_textures(pass.cmd, frame, std::mem::take(&mut self.pending))
            .expect("egui textures");
        self.painter
            .paint(
                pass.cmd,
                frame,
                pass.layer,
                &self.primitives,
                self.pixels_per_point,
            )
            .expect("egui paint");
    }
}

#[cfg(not(windows))]
struct Overlay;

#[cfg(not(windows))]
impl Overlay {
    fn new(_: u32) -> Self {
        Self
    }

    fn layout(
        &mut self,
        _: [u32; 2],
        _: FrameTime,
        _: &orr::ResolvedDisplay,
        _: Option<&orr::ExposureTelemetry>,
    ) {
    }

    fn paint(&mut self, _: &mut orr::UiPass<'_, '_>, _: u64) {}
}

/// The objects, the ground and the sky, cooked into an in-memory pack.
fn cook() -> Arc<[u8]> {
    let gold = Material {
        base_color: [1.0, 0.77, 0.34],
        metalness: 1.0,
        specular_roughness: 0.25,
        ..Material::default()
    };
    let glossy = Material {
        base_color: [0.6, 0.05, 0.04],
        specular_roughness: 0.12,
        ..Material::default()
    };
    let mirror = Material {
        base_color: [0.95, 0.95, 0.95],
        metalness: 1.0,
        specular_roughness: 0.02,
        ..Material::default()
    };
    let block = Material {
        base_color: [0.8, 0.8, 0.78],
        specular_roughness: 0.6,
        ..Material::default()
    };
    let glow = Material {
        base_color: [0.1, 0.1, 0.1],
        emission_color: [1.0, 0.55, 0.25],
        emission_luminance: 20_000.0,
        ..Material::default()
    };
    let ground = Material {
        base_color: [0.5, 0.5, 0.48],
        specular_roughness: 0.75,
        ..Material::default()
    };

    let mut objects = MeshBuilder::default();
    objects.knot(0, Vec3::new(0.0, 1.4, 0.0), 512, 32);
    objects.sphere(1, Vec3::new(-2.3, 0.8, 1.0), 0.8, 48);
    objects.sphere(2, Vec3::new(2.3, 0.7, 1.2), 0.7, 48);
    objects.cuboid(3, Vec3::new(-1.6, 0.0, -2.6), Vec3::new(-0.2, 0.9, -1.6), 4);
    objects.quad(
        4,
        [
            Vec3::new(0.8, 0.1, -2.4),
            Vec3::new(2.4, 0.1, -1.8),
            Vec3::new(2.4, 1.3, -1.8),
            Vec3::new(0.8, 1.3, -2.4),
        ],
        2,
    );
    let mut floor = MeshBuilder::default();
    let h = 60.0;
    floor.quad(
        0,
        [
            Vec3::new(-h, 0.0, h),
            Vec3::new(h, 0.0, h),
            Vec3::new(h, 0.0, -h),
            Vec3::new(-h, 0.0, -h),
        ],
        24,
    );
    let mesh = |name: &str, mesh: MeshInput, materials: Vec<Material>| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings::default(),
        },
    };
    let sky = sky_hdr();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: SKY.into(),
                    source: Source::Texture {
                        path: sky.clone(),
                        settings: TextureSettings {
                            usage: Usage::Environment {
                                light_threshold: None,
                                max_lights: 0,
                            },
                            quality: Quality::Fast,
                            ..TextureSettings::default()
                        },
                    },
                },
                mesh(
                    OBJECTS,
                    objects.finish(),
                    vec![gold, glossy, mirror, block, glow],
                ),
                mesh(GROUND, floor.finish(), vec![ground]),
            ],
        )],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    let _ = std::fs::remove_dir_all(sky.parent().expect("temp dir"));
    pack
}

/// Sky radiance in nits, linear Rec.709, along unit direction `d`: a gradient from a bright
/// horizon to a blue zenith with a glow around the sun, over a dim ground.
fn sky(d: Vec3) -> Vec3 {
    let glow = 6.0 * (60.0 * (d.dot(SUN) - 1.0)).exp();
    if d.y < 0.0 {
        return Vec3::new(0.55, 0.52, 0.48) * 6000.0;
    }
    let horizon = Vec3::new(0.85, 0.9, 1.0) * 5000.0;
    let zenith = Vec3::new(0.2, 0.38, 1.0) * 4000.0;
    horizon.lerp(zenith, d.y.sqrt()) * (1.0 + glow)
}

/// The sky as a 1024 x 512 equirectangular Radiance HDR file in a temp directory, for the
/// cook to read.
fn sky_hdr() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-viewer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("sky.hdr");
    let (w, h) = (1024u32, 512u32);
    let mut bytes = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n").into_bytes();
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
            bytes.extend(rgbe(sky(d)));
        }
    }
    std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    path
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
