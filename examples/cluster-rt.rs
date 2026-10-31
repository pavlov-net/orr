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

use examples::shaders::{capture, sun_shadow};
use examples::window::Window;
use examples::{Capture, Options, Workers, assert_clean, device, fixed_time};
use orr::orr_gpu::{Format, Hazards, MemoryKind, QueueKind, SpecData, Stages, TimelinePoint};
use orr::orr_output::Image;
use orr::orr_pack::material::{Material, slot};
use orr::orr_pack::{AssetId, Span, Target};
use orr::{
    Affine3, Display, DisplayRequest, FrameDesc, FrameTime, GroupWriter, InstanceFlags,
    InstancesDesc, Lens, LoadState, PackSource, Projection, Renderer, RendererDesc, SceneDesc,
    TransformGroupDesc, ViewDesc, ViewTarget,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshSettings, Quality, Source,
    TextureSettings, Usage,
};
use orr_math::camera::rh::view;
use orr_math::{DVec3, UVec2, Vec2, Vec3};

const MODEL: &str = "knot and card";
const GROUND: &str = "ground";
const RINGS: &str = "rings";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
const FOV_Y: f32 = 0.8;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.453_413_4, 0.654_930_5, 0.604_551_3);
/// Sun illuminance and sky radiance, relative to paper white.
const SUN_ILLUMINANCE: f32 = 3.0;
const SKY: [f32; 3] = [0.18, 0.24, 0.34];
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
    let mut window = (!options.headless).then(|| Window::new("orr cluster-rt", 1280, 720));
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
        .create_scene(&SceneDesc {
            instance_capacity: 16,
            group_capacity: 4,
            light_capacity: 1,
            material_set_capacity: 16,
            emissive_capacity: 16,
            part_capacity: 0,
            ptlas_partitions: 16,
            global_instances: 16,
            ..SceneDesc::default()
        })
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
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
    drop(w);
    f.end().expect("end frame");

    let shade = renderer
        .compute_pipeline(&sun_shadow::MODULE, "main", SpecData::default())
        .expect("shade pipeline");
    let encode = renderer
        .compute_pipeline(&capture::MODULE, "main", SpecData::default())
        .expect("capture pipeline");
    let slots = u64::from(desc.frames_in_flight);
    let mut roots = gpu
        .gpu_malloc::<sun_shadow::Root>(MemoryKind::Upload, slots, "cluster-rt roots")
        .expect("roots");
    let mut capture_root = gpu
        .gpu_malloc::<capture::Root>(MemoryKind::Upload, 1, "cluster-rt capture root")
        .expect("capture root");
    let pixels = u64::from(HEADLESS_EXTENT[0] * HEADLESS_EXTENT[1]);
    let readback = options.headless.then(|| {
        gpu.gpu_malloc::<u32>(MemoryKind::Readback, pixels, "cluster-rt capture")
            .expect("readback")
    });
    let mut image: Option<Image> = None;

    let frames = options
        .frames
        .or(options.headless.then_some(HEADLESS_FRAMES));
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
            .set_focus(eye.as_dvec3(), extent[1] as f32 / FOV_Y);
        let view = f
            .render_view(&ViewDesc {
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
                target: if options.headless {
                    ViewTarget::AfterRr
                } else {
                    ViewTarget::Display
                },
            })
            .expect("render view");

        let fresh = image
            .as_ref()
            .is_none_or(|im| im.texture().desc().extent[..2] != extent);
        if fresh {
            image = Some(
                Image::new(gpu, Format::Rgba16Float, extent, "cluster-rt shaded").expect("image"),
            );
        }
        let target = image.as_ref().expect("image");
        let slot = f.id() % slots;
        roots.as_mut_slice().expect("roots")[slot as usize] = sun_shadow::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            target: target.storage,
            mode: u32::from(clusters),
            sun: [SUN.x, SUN.y, SUN.z, SUN_ILLUMINANCE],
            sky: [SKY[0], SKY[1], SKY[2], 1.0],
        };
        let (shade, encode) = (f.pipeline(shade), f.pipeline(encode));
        let list = f.list().expect("frame list");
        if fresh {
            list.discard(target.texture());
        }
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.begin_pass("sun shadow", false);
        list.dispatch(
            shade,
            roots.gpu.offset(slot),
            [extent[0].div_ceil(8), extent[1].div_ceil(8), 1],
        );
        list.end_pass();
        if let Some(out) = readback.as_ref().filter(|_| final_frame) {
            capture_root.as_mut_slice().expect("capture root")[0] = capture::Root {
                pixels: out.gpu,
                source: target.sampled,
                width: extent[0],
                height: extent[1],
                pad0: 0,
                pad1: 0,
            };
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            list.dispatch(
                encode,
                capture_root.gpu,
                [extent[0].div_ceil(8), extent[1].div_ceil(8), 1],
            );
            list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        }
        f.compose(target.sampled).expect("compose");
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
    let captured = readback.filter(|_| last.is_some()).map(|out| Capture {
        width: HEADLESS_EXTENT[0],
        height: HEADLESS_EXTENT[1],
        rgba8: rgba8(out.as_slice().expect("readback")),
    });
    drop(image);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// Bytes of packed RGBA8 words.
fn rgba8(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
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
    model.knot(0, 512, 32);
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
                            quality: Quality::Fast,
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
    let n = 256;
    let data: Vec<u8> = (0..n * n)
        .map(|i| {
            let (u, v) = (
                (i % n) as f32 / n as f32 - 0.5,
                (i / n) as f32 / n as f32 - 0.5,
            );
            if ((u * u + v * v).sqrt() * 8.0).fract() < 0.6 {
                255
            } else {
                0
            }
        })
        .collect();
    let file = std::fs::File::create(&path).expect("rings.png");
    let mut enc = png::Encoder::new(file, n, n);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&data))
        .expect("rings.png");
    path
}

/// Triangles per material section.
#[derive(Default)]
struct MeshBuilder {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    sections: Vec<Vec<u32>>,
}

impl MeshBuilder {
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2) -> u32 {
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        self.positions.len() as u32 - 1
    }

    /// Two triangles over the grid cell `c`, `c + 1`, `c + w + 1`, `c + w`, counter-clockwise
    /// seen from the side the normals face.
    fn cell(&mut self, section: usize, c: u32, w: u32) {
        if self.sections.len() <= section {
            self.sections.resize(section + 1, Vec::new());
        }
        self.sections[section].extend([c, c + 1, c + w + 1, c, c + w + 1, c + w]);
    }

    /// A quad over `corners`, counter-clockwise seen from its front, in `n` x `n` cells.
    fn quad(&mut self, section: usize, corners: [Vec3; 4], n: u32) {
        let normal = (corners[1] - corners[0])
            .cross(corners[3] - corners[0])
            .normalize();
        let first = self.positions.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                let p = corners[0]
                    .lerp(corners[1], u)
                    .lerp(corners[3].lerp(corners[2], u), v);
                self.vertex(p, normal, Vec2::new(u, 1.0 - v));
            }
        }
        for j in 0..n {
            for i in 0..n {
                self.cell(section, first + j * (n + 1) + i, n + 1);
            }
        }
    }

    /// A (2, 3) torus knot tube standing in the xy plane, `segments` along it and `sides`
    /// around it.
    fn knot(&mut self, section: usize, segments: u32, sides: u32) {
        let center = |t: f32| {
            let r = (3.0 * t).cos() + 2.0;
            Vec3::new(r * (2.0 * t).cos(), r * (2.0 * t).sin(), -(3.0 * t).sin()) * 0.33
                + Vec3::new(0.0, 1.4, 0.0)
        };
        let (tube, tau) = (0.13, std::f32::consts::TAU);
        let first = self.positions.len() as u32;
        for s in 0..=segments {
            let t = s as f32 / segments as f32 * tau;
            let c = center(t);
            let tangent = (center(t + 1e-3) - center(t - 1e-3)).normalize();
            let out = c - Vec3::new(0.0, 1.4, 0.0);
            let normal = (out - tangent * out.dot(tangent)).normalize();
            let binormal = tangent.cross(normal);
            for k in 0..=sides {
                let a = k as f32 / sides as f32 * tau;
                let n = normal * a.cos() + binormal * a.sin();
                self.vertex(
                    c + n * tube,
                    n,
                    Vec2::new(s as f32 / segments as f32, k as f32 / sides as f32),
                );
            }
        }
        for s in 0..segments {
            for k in 0..sides {
                self.cell(section, first + s * (sides + 1) + k, sides + 1);
            }
        }
    }

    fn finish(self) -> MeshInput {
        let mut m = MeshInput {
            positions: self.positions,
            normals: self.normals,
            uvs: Some(self.uvs),
            ..MeshInput::default()
        };
        for tris in self.sections {
            let first = m.indices.len() as u32;
            m.indices.extend(tris);
            m.sections.push(Span {
                first,
                count: m.indices.len() as u32 - first,
            });
        }
        m
    }
}
