//! One cooked model as clusters: a knotted tube and an alpha-tested card of rings, cooked in
//! memory with a ground and mounted. orr rasters the clusters into its G-buffer; the
//! example's own kernel shades it under a sun, tracing one shadow ray per pixel through the
//! scene's TLAS, where the card's rings cast ring-shaped shadows. The camera orbits the
//! model.
//!
//! `cluster-rt [--clusters] [--frames N] [--headless] [--out file.png]`: `--clusters`
//! colors each cluster instead of its material. Headless runs render 480 x 270 for 120
//! frames unless told otherwise and write the last frame as a PNG.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::orr_gpu::{QueueKind, TimelinePoint};
use orr::orr_pack::material::{Material, slot};
use orr::orr_pack::{AssetId, Target};
use orr::{
    Display, DisplayRequest, FrameDesc, FrameTime, GroupWriter, InstanceFlags, InstancesDesc, Lens,
    LoadState, PackSource, Projection, Renderer, RendererDesc, SceneDesc, SceneLightingSettings,
    TextureFilter, TransformGroupDesc, ViewDesc,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshSettings, Source, TextureSettings,
    Usage,
};
use orr_examples::mesh::MeshBuilder;
use orr_examples::sun_shade::SunShade;
use orr_examples::window::Window;
use orr_examples::{Capture, Options, Workers, assert_clean, device, fixed_time, gray_png};
use orr_math::camera::rh::view;
use orr_math::{Affine3, DVec3, UVec2, Vec2, Vec3};

const MODEL: &str = "knot and card";
const GROUND: &str = "ground";
const RINGS: &str = "rings";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
const FOV_Y: f32 = 0.8;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.453_413_4, 0.654_930_5, 0.604_551_3);
const TARGET: Vec3 = Vec3::new(0.4, 1.0, 0.0);
/// Radians per second.
const ORBIT_RATE: f64 = 0.25;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("cluster-rt");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Runs the example; headless runs return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let clusters = options.has("--clusters");
    let mut window = (!options.headless).then(|| Window::new("orr cluster-rt", 1920, 1080));
    let desc = RendererDesc::default();
    let mut display = window.as_ref().map(|w| {
        Display::new(
            gpu,
            w.window_handle(),
            w.display_handle(),
            &DisplayRequest::default(),
            desc.frames_in_flight,
        )
        .expect("display")
    });
    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    let _workers = Workers::new(renderer.loading(), 2);

    let mount = renderer.mount(PackSource::Memory(cook()));
    let mut frame = 0;
    let start = Instant::now();
    while matches!(renderer.mount_state(mount), LoadState::Pending) {
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
        std::thread::sleep(Duration::from_millis(1));
    }
    if let LoadState::Failed(e) = renderer.mount_state(mount) {
        panic!("mount failed: {e}");
    }

    let scene = renderer
        .create_scene(
            &SceneDesc {
                global_instances: 16,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_raster_view_history().expect("view history");
    let models = [MODEL, GROUND].map(|name| {
        renderer
            .model(mount, AssetId::from_name(name))
            .unwrap_or_else(|e| panic!("model {name}: {e}"))
    });

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
        .fill(Affine3::IDENTITY.into());
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
    drop(w);
    f.end().expect("end frame");

    let mut shading = SunShade::new(
        &mut renderer,
        "cluster-rt",
        desc.frames_in_flight,
        options.headless.then_some(HEADLESS_EXTENT),
        SUN,
    );

    let frames = options
        .frames
        .or_else(|| options.headless.then_some(HEADLESS_FRAMES));
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<TimelinePoint> = None;
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
        let angle = 0.6 + t * ORBIT_RATE;
        let eye = TARGET + Vec3::new(angle.sin() as f32 * 5.5, 1.4, angle.cos() as f32 * 5.5);

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let extent = if options.headless {
            HEADLESS_EXTENT
        } else {
            match f.display() {
                Some((_, extent)) => extent,
                None => {
                    last = Some(f.end().expect("end frame"));
                    continue;
                }
            }
        };
        f.scene(scene)
            .expect("scene writer")
            .set_focus(eye.as_dvec3());
        let view = f
            .raster_view(&ViewDesc {
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
                output_extent: UVec2::from(extent),
                jitter: Vec2::ZERO,
                camera_cut: false,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
            })
            .expect("raster view");

        shading.record(&mut f, &view, extent, u32::from(clusters), final_frame);
        last = Some(f.end().expect("end frame"));
    }

    if let Some(d) = display.as_ref() {
        d.wait_idle();
    }
    if let Some(point) = last {
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(10))
            .expect("wait");
    }
    let captured = shading.capture().filter(|_| last.is_some());
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// The model and the ground, cooked into an in-memory pack.
fn cook() -> Arc<[u8]> {
    let knot = Material {
        base_color: [0.75, 0.42, 0.18],
        specular_roughness: 0.4,
        ..Material::default()
    };
    let mut card = Material {
        base_color: [0.3, 0.62, 0.22],
        alpha_mode: Material::MASK,
        alpha_cutoff: 0.5,
        double_sided: 1,
        ..Material::default()
    };
    card.textures[slot::OPACITY].texture = AssetId::from_name(RINGS);
    let ground = Material {
        base_color: [0.55, 0.55, 0.52],
        specular_roughness: 0.8,
        ..Material::default()
    };

    let mut model = MeshBuilder::default();
    model.knot(0, Vec3::new(0.0, 1.4, 0.0), 512, 32);
    model.quad(
        1,
        [
            Vec3::new(1.6, 0.2, -0.7),
            Vec3::new(2.4, 0.2, 0.3),
            Vec3::new(2.4, 1.6, 0.3),
            Vec3::new(1.6, 1.6, -0.7),
        ],
        1,
    );
    let mut floor = MeshBuilder::default();
    let h = 8.0;
    floor.quad(
        0,
        [
            Vec3::new(-h, 0.0, h),
            Vec3::new(h, 0.0, h),
            Vec3::new(h, 0.0, -h),
            Vec3::new(-h, 0.0, -h),
        ],
        8,
    );
    let mesh = |name: &str, mesh: MeshInput, materials: Vec<Material>, lod: LodMode| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings {
                lod,
                ..MeshSettings::default()
            },
        },
    };
    let rings = rings_png();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: RINGS.into(),
                    source: Source::Texture {
                        path: rings.clone(),
                        settings: TextureSettings {
                            usage: Usage::Linear1,
                            ..TextureSettings::default()
                        },
                    },
                },
                mesh(
                    MODEL,
                    model.finish(),
                    vec![knot, card],
                    LodMode::Discrete {
                        errors: vec![0.004, 0.016],
                    },
                ),
                mesh(GROUND, floor.finish(), vec![ground], LodMode::default()),
            ],
        )],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    let _ = std::fs::remove_dir_all(rings.parent().expect("temp dir"));
    pack
}

/// A grayscale PNG of concentric rings, opaque on the inner 60% of each ring, written to a
/// temp directory for the cook to read.
fn rings_png() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-cluster-rt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("rings.png");
    gray_png(&path, 256, |u, v| {
        let (u, v) = (u - 0.5, v - 0.5);
        if ((u * u + v * v).sqrt() * 8.0).fract() < 0.6 {
            255
        } else {
            0
        }
    });
    path
}
