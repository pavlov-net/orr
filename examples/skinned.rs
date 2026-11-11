//! A field of 100 skinned tentacles swaying through their cooked `wave` clip. The creature
//! is cooked from glTF with its skeleton and clip; the example's pose kernel samples the clip
//! in scene compute for every copy at its own phase and writes the copies' skinning matrices
//! into a GPU-written palette group. orr skins each copy at one cut chosen by its distance
//! from the camera, rasters it and traces it through a cluster BLAS built from the cut's
//! cluster templates, so the tentacles cast and receive the sun's traced shadows as they
//! bend. The camera orbits the field.
//!
//! `skinned [--frames N] [--headless] [--out file.png]`: headless runs render 480 x 270 for
//! 120 frames unless told otherwise and write the last frame as a PNG.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::orr_gpu::{QueueKind, SpecData, TimelinePoint};
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    Display, DisplayRequest, FrameDesc, FrameTime, GroupWriter, InstanceFlags, InstancesDesc, Lens,
    PackSource, Projection, Renderer, RendererDesc, SceneDesc, SceneLightingSettings, SkinnedDesc,
    TextureFilter, TransformGroupDesc, ViewDesc,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MeshInput, MeshSettings, Source,
};
use orr_examples::creature;
use orr_examples::mesh::MeshBuilder;
use orr_examples::shaders::pose;
use orr_examples::sun_shade::SunShade;
use orr_examples::window::Window;
use orr_examples::{Capture, Options, Workers, assert_clean, device, fixed_time, mount_at};
use orr_math::camera::rh::view;
use orr_math::{Affine3, DVec3, Mat3, UVec2, Vec2, Vec3};

const CREATURE: &str = "tentacle";
const GROUND: &str = "ground";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the first rendered frame follows, so headless runs repeat exactly.
const FIRST_FRAME: u64 = 60;
const FOV_Y: f32 = 0.9;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.512_092_8, 0.716_929_9, 0.473_016_4);
/// Copies per side of the field, and the spacing between them in m.
const SIDE: u32 = 10;
const SPACING: f32 = 1.6;
/// Radians per second.
const ORBIT_RATE: f64 = 0.12;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("skinned");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Runs the example; headless runs return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr skinned", 1920, 1080));
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
    let (mount, mut frame) = mount_at(&mut renderer, PackSource::Memory(cook()), FIRST_FRAME);

    let copies = SIDE * SIDE;
    let joints = creature::JOINTS as u32;
    let scene = renderer
        .create_scene(
            &SceneDesc {
                light_capacity: 1,
                emissive_capacity: 16,
                global_instances: 16,
                skinned_capacity: copies,
                skinned_clusters: copies * 16,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_raster_view_history().expect("view history");
    let id = AssetId::from_name;
    let [tentacle, ground] = [CREATURE, GROUND].map(|name| {
        renderer
            .model(mount, id(name))
            .unwrap_or_else(|e| panic!("model {name}: {e}"))
    });
    let skeleton = renderer
        .skeleton(mount, id(&format!("{CREATURE}/skeleton")))
        .expect("skeleton");
    let clip = renderer
        .clip(mount, id(&format!("{CREATURE}/clip/wave")))
        .expect("clip");
    assert_eq!(skeleton.joint_count, joints);

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
            count: copies + 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .expect("transform group");
    let palette = w
        .create_transform_group(&TransformGroupDesc {
            count: copies * joints,
            writer: GroupWriter::Gpu,
            anchor: DVec3::ZERO,
        })
        .expect("palette group");
    let transforms = w.transforms(group, 0, copies + 1).expect("transforms");
    for (i, t) in transforms.iter_mut().take(copies as usize).enumerate() {
        let (x, z) = ((i as u32 % SIDE) as f32, (i as u32 / SIDE) as f32);
        let h = (i as u32).wrapping_mul(2_654_435_761);
        let jitter = |shift: u32| ((h >> shift) & 255) as f32 / 255.0 - 0.5;
        let at = Vec3::new(
            (x - (SIDE - 1) as f32 * 0.5 + 0.6 * jitter(0)) * SPACING,
            0.0,
            (z - (SIDE - 1) as f32 * 0.5 + 0.6 * jitter(8)) * SPACING,
        );
        let scale = 0.8 + 0.4 * (jitter(24) + 0.5);
        *t = Affine3::from_mat3_translation(Mat3::from_rotation_y(jitter(16) * 6.0) * scale, at)
            .into();
    }
    transforms[copies as usize] = Affine3::IDENTITY.into();
    w.add_skinned(&SkinnedDesc {
        model: tentacle,
        copies,
        transforms: group,
        first: 0,
        palette,
        palette_first: 0,
        material_set: None,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 1000.0,
        user_tag: 0,
    })
    .expect("tentacles");
    w.add_instances(&InstancesDesc {
        model: ground,
        copies: 1,
        transforms: group,
        first: copies,
        pose: None,
        material_set: None,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 1000.0,
        animation_range: 0.0,
        user_tag: 0,
    })
    .expect("ground");
    drop(w);
    f.end().expect("end frame");

    let poser = renderer
        .compute_pipeline(&pose::MODULE, "main", SpecData::default())
        .expect("pose pipeline");
    let mut shading = SunShade::new(
        &mut renderer,
        "skinned",
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
        let eye = Vec3::new(angle.sin() as f32 * 9.0, 3.6, angle.cos() as f32 * 9.0);
        let target = Vec3::new(0.0, 1.0, 0.0);

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
        let desc = ViewDesc {
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
            output_extent: UVec2::from(extent),
            jitter: Vec2::ZERO,
            camera_cut: false,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        };
        f.prepare_raster_view(&desc).expect("prepare view");
        f.update_scenes(&mut |c| {
            let root = c
                .put_root(&pose::Root {
                    scene: c.root,
                    skeleton: skeleton.blob,
                    clip_header: clip.header,
                    clip_samples: clip.samples,
                    group: palette.0,
                    copies,
                    time: t as f32,
                    _pad: 0,
                })
                .expect("pose root");
            c.cmd
                .dispatch(c.pipeline(poser), root, [copies.div_ceil(64), 1, 1]);
        })
        .expect("update scenes");
        let view = f.raster_view(&desc).expect("raster view");

        shading.record(&mut f, &view, extent, 0, final_frame);
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

/// The creature, cooked from the glTF the example writes to a temp directory, and the
/// ground, into an in-memory pack.
fn cook() -> Arc<[u8]> {
    let dir = std::env::temp_dir().join(format!("orr-skinned-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path: PathBuf = dir.join("tentacle.glb");
    creature::write_glb(&path, 36, 16, 0.22, [0.75, 0.32, 0.42]);
    let soil = Material {
        base_color: [0.42, 0.38, 0.3],
        specular_roughness: 0.9,
        ..Material::default()
    };
    let mut floor = MeshBuilder::default();
    let h = 30.0;
    floor.quad(
        0,
        [
            Vec3::new(-h, 0.0, h),
            Vec3::new(h, 0.0, h),
            Vec3::new(h, 0.0, -h),
            Vec3::new(-h, 0.0, -h),
        ],
        16,
    );
    let ground: MeshInput = floor.finish();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: CREATURE.into(),
                    source: Source::Gltf {
                        path,
                        settings: GltfSettings {
                            mesh: MeshSettings {
                                deformation_bound: creature::REACH,
                                ..MeshSettings::default()
                            },
                            ..GltfSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: GROUND.into(),
                    source: Source::Mesh {
                        mesh: ground,
                        materials: vec![soil],
                        settings: MeshSettings {
                            lod: LodMode::default(),
                            ..MeshSettings::default()
                        },
                    },
                },
            ],
        )],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    let _ = std::fs::remove_dir_all(&dir);
    pack
}
