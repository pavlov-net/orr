//! A corridor of textured panels under the sun, flown past at 2.8 m/s, twice walking speed,
//! with the texture pool at half the bytes the panels' textures take fully resident. Each
//! panel model has its own 1024 x 1024 cooked texture; only its mip tail loads at mount, and
//! its tiles stream on the frames' texture feedback and leave once the camera has passed.
//! Each frame checks the texture pool against its budget; once a second the run logs
//! residency.
//!
//! `streaming-stress [--frames N] [--headless] [--out file.png]`. Headless runs render
//! 960 x 540 for 240 frames unless told otherwise and write the last frame as a PNG.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use examples::mesh::MeshBuilder;
use examples::window::Window;
use examples::{Capture, Options, Workers, assert_clean, device, fixed_time, mount_at};
use orr::orr_gpu::{
    Device, Format, QueueKind, TextureDesc, TextureKind, TextureMemory, TextureUsage, TimelinePoint,
};
use orr::orr_pack::material::{Material, slot};
use orr::orr_pack::{AssetId, Target};
use orr::{
    Affine3, Budgets, DirectionalLight, DiscProfile, Display, DisplayRequest, FrameDesc, FrameTime,
    GroupWriter, InstanceFlags, InstancesDesc, Lens, Light, LightId, PackSource, Projection,
    Renderer, RendererDesc, ResidencyClass, SceneDesc, SceneLightingSettings, ScreenshotRequest,
    TextureFilter, TransformGroupDesc, ViewDesc, ViewHooks, ViewLighting, ViewOutput, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshSettings, Source, TextureSettings, Usage};
use orr_math::camera::rh::view;
use orr_math::{DVec3, UVec2, Vec2, Vec3};

const HEADLESS_EXTENT: [u32; 2] = [960, 540];
const HEADLESS_FRAMES: u64 = 240;
/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 300;
const FOV_Y: f32 = 0.9;
/// Panel models, each with its own texture, and the texture edge.
const TEXTURES: usize = 12;
const TEXTURE_SIZE: u32 = 1024;
/// Panels per side of the corridor and their spacing along it, m.
const PANELS: usize = 48;
const SPACING: f32 = 3.0;
/// m/s along the corridor.
const SPEED: f32 = 2.8;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.35, 0.8, 0.48);

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("streaming-stress");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

fn panel(i: usize) -> String {
    format!("panel/{i}")
}

fn texture(i: usize) -> String {
    format!("panel/{i}/texture")
}

/// Writes texture `i`: a grid of numbered cells in its own colors, fine enough that each
/// mip differs.
fn write_texture(path: &PathBuf, i: usize) {
    let n = TEXTURE_SIZE;
    let hue = |k: usize| -> [u8; 3] {
        let t = k as f32 * 0.618;
        let c = |o: f32| (127.0 + 110.0 * ((t + o) * std::f32::consts::TAU).cos()) as u8;
        [c(0.0), c(0.33), c(0.67)]
    };
    let (a, b) = (hue(i), hue(i + 5));
    let data: Vec<u8> = (0..n * n)
        .flat_map(|p| {
            let (x, y) = (p % n, p / n);
            let cell = (x / 64 + y / 64) % 2 == 0;
            let line = x % 64 < 2 || y % 64 < 2 || (x / 8 + y / 8) % 16 == i as u32 % 16;
            let c = if line {
                [20, 20, 20]
            } else if cell {
                a
            } else {
                b
            };
            [c[0], c[1], c[2], 255]
        })
        .collect();
    let file = std::fs::File::create(path).expect("texture file");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), n, n);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&data))
        .expect("writing a texture");
}

/// The panels, their textures and the floor, cooked into an in-memory pack.
fn cook() -> Arc<[u8]> {
    let dir = std::env::temp_dir().join(format!("orr-streaming-stress-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut assets = Vec::new();
    for i in 0..TEXTURES {
        let path = dir.join(format!("{i}.png"));
        write_texture(&path, i);
        assets.push(AssetSpec {
            name: texture(i),
            source: Source::Texture {
                path,
                settings: TextureSettings {
                    usage: Usage::Color,
                    ..TextureSettings::default()
                },
            },
        });
        let mut m = Material {
            base_color: [1.0; 3],
            specular_roughness: 0.6,
            double_sided: 1,
            ..Material::default()
        };
        m.textures[slot::BASE_COLOR].texture = AssetId::from_name(&texture(i));
        let mut quad = MeshBuilder::default();
        quad.quad(
            0,
            [
                Vec3::new(0.0, 0.0, -1.2),
                Vec3::new(0.0, 0.0, 1.2),
                Vec3::new(0.0, 2.4, 1.2),
                Vec3::new(0.0, 2.4, -1.2),
            ],
            4,
        );
        assets.push(AssetSpec {
            name: panel(i),
            source: Source::Mesh {
                mesh: quad.finish(),
                materials: vec![m],
                settings: MeshSettings::default(),
            },
        });
    }
    let mut floor = MeshBuilder::default();
    let (w, l) = (4.0, SPACING * PANELS as f32 + 20.0);
    floor.quad(
        0,
        [
            Vec3::new(-w, 0.0, 10.0),
            Vec3::new(w, 0.0, 10.0),
            Vec3::new(w, 0.0, -l),
            Vec3::new(-w, 0.0, -l),
        ],
        16,
    );
    assets.push(AssetSpec {
        name: "floor".into(),
        source: Source::Mesh {
            mesh: floor.finish(),
            materials: vec![Material {
                base_color: [0.45; 3],
                specular_roughness: 0.8,
                ..Material::default()
            }],
            settings: MeshSettings::default(),
        },
    });
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    let _ = std::fs::remove_dir_all(&dir);
    pack
}

/// Bytes one panel texture takes resident: its mip tail alone, and every tile above it
/// with the tail.
fn texture_bytes(gpu: &Device) -> (u64, u64) {
    let desc = TextureDesc {
        kind: TextureKind::D2,
        format: Format::Bc7Srgb,
        extent: [TEXTURE_SIZE, TEXTURE_SIZE, 1],
        mips: orr::orr_pack::texture::full_mip_count(TEXTURE_SIZE, TEXTURE_SIZE),
        layers: 1,
        usage: TextureUsage::SAMPLED,
    };
    let probe = gpu
        .create_texture(&desc, TextureMemory::Sparse, "tile layout probe")
        .expect("sparse texture");
    let l = *probe.sparse_layout().expect("sparse layout");
    let tiles: u64 = (0..l.tail_first_mip)
        .map(|m| {
            let [x, y] = l.tiles(&desc, m);
            u64::from(x * y)
        })
        .sum();
    let tail = u64::from(l.tails(&desc)) * l.tail_bytes;
    (tail, tail + tiles * l.tile_bytes)
}

/// Runs the example; headless runs and runs with `--out` return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let pack = cook();
    let (tail, whole) = texture_bytes(gpu);
    // Every mip tail stays; the tiles get half the bytes they would take.
    let budget = TEXTURES as u64 * (tail + (whole - tail) / 2);
    let desc = RendererDesc {
        budgets: Budgets {
            texture_pool: budget,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut window = (!options.headless).then(|| Window::new("orr streaming stress", 1280, 720));
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
    let _workers = Workers::new(renderer.loading(), 4);
    let (mount, mut frame) = mount_at(&mut renderer, PackSource::Memory(pack), FIRST_FRAME);
    let scene = renderer
        .create_scene(
            &SceneDesc {
                instance_capacity: 2 * PANELS as u32 + 1,
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
    let model = |name: &str| {
        renderer
            .model(mount, AssetId::from_name(name))
            .unwrap_or_else(|e| panic!("model {name}: {e}"))
    };
    let panels: Vec<_> = (0..TEXTURES).map(|i| model(&panel(i))).collect();
    let floor = model("floor");

    // Panels face the corridor from both sides, each side cycling through the models.
    let mut placed = vec![(floor, Affine3::IDENTITY)];
    for k in 0..PANELS {
        let z = -(k as f32) * SPACING;
        for (side, x, flip) in [(0, -1.6, 1.0f32), (1, 1.6, -1.0)] {
            let t = Affine3 {
                rows: [
                    [flip, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, flip, z],
                ],
            };
            placed.push((panels[(2 * k + side) % TEXTURES], t));
        }
    }
    frame += 1;
    let listed = [scene];
    {
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
                count: placed.len() as u32,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .expect("transform group");
        let transforms = w
            .transforms(group, 0, placed.len() as u32)
            .expect("transforms");
        for (out, (_, t)) in transforms.iter_mut().zip(&placed) {
            *out = *t;
        }
        for (i, &(model, _)) in placed.iter().enumerate() {
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
                illuminance: Vec3::new(100_000.0, 95_000.0, 88_000.0),
                angular_radius: 0.004_65,
                disc: DiscProfile::LimbDarkened,
            })),
        )
        .expect("sun");
        drop(w);
        f.end().expect("end frame");
    }

    let history = renderer.create_view_history().expect("view history");
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
            "streaming stress target",
        )
        .expect("headless target")
    });
    let capture = options.headless || options.out.is_some();
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
        // The camera walks the corridor and starts again at its end.
        let length = SPACING * PANELS as f32;
        let along = (t as f32 * SPEED) % length;
        let eye = Vec3::new(0.0, 1.4, 4.0 - along);
        let target = eye + Vec3::new(0.25 * (t as f32 * 0.7).sin(), -0.1, -1.0);

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let id = f.id();
        let (extent, view_target) = match (&headless_target, f.display()) {
            (Some(t), _) => (HEADLESS_EXTENT, ViewTarget::Texture(t)),
            (None, Some((_, extent))) => (extent, ViewTarget::Display),
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
            screenshot: (capture && final_frame)
                .then_some(ScreenshotRequest::Display { include_ui: false }),
            ..ViewOutput::default()
        };
        f.render_view(
            &ViewDesc {
                scene,
                history,
                position: eye.as_dvec3(),
                rotation: view::look_at_quat(eye, target, Vec3::Y).inverse(),
                projection: Projection {
                    fov_y: FOV_Y,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: UVec2::from(extent),
                jitter: Vec2::ZERO,
                reset_history: i == 0,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
                target: view_target,
            },
            &ViewLighting::default(),
            &output,
            &mut ViewHooks::default(),
        )
        .expect("render view");
        last = Some((id, f.end().expect("end frame")));

        let r = *renderer.residency();
        let pool = *r.class(ResidencyClass::Textures);
        assert!(
            pool.resident_bytes + pool.loading_bytes <= budget,
            "the texture pool holds {} of its {budget} bytes",
            pool.resident_bytes + pool.loading_bytes
        );
        if i % 60 == 0 {
            log::info!(
                "{:.1} m: texture pool {:.1} of {:.1} MiB ({} units, {:.1} MiB loading), {} loads, {} evictions, {} dropped",
                along,
                pool.resident_bytes as f64 / f64::from(1 << 20),
                budget as f64 / f64::from(1 << 20),
                pool.resident_units,
                pool.loading_bytes as f64 / f64::from(1 << 20),
                r.loads,
                r.evictions,
                r.dropped,
            );
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
        .map(|shot| Capture::from_screenshot(&shot));
    drop(headless_target);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}
