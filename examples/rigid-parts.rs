//! A stand of 400 trees whose branches sway in the wind as rigid parts. Each tree is one
//! cooked model: the trunk is its body and each branch, with its leaves, a part pivoted at
//! its base. The example's own kernel writes every copy's pose block into a GPU-written
//! pose group in scene compute; trees inside the animation range take their pose in the
//! raster and in ray tracing, where each branch is a PTLAS instance, and the rest stand
//! still. Shading traces one sun shadow ray per pixel, so swaying branches move their
//! shadows. The camera orbits inside the stand.
//!
//! `rigid-parts [--frames N] [--headless] [--out file.png]`: headless runs render 480 x 270
//! for 120 frames unless told otherwise and write the last frame as a PNG.

use std::sync::Arc;
use std::time::{Duration, Instant};

use examples::mesh::MeshBuilder;
use examples::shaders::{capture, sun_shadow, wind};
use examples::window::Window;
use examples::{Capture, Options, Workers, affine, assert_clean, device, fixed_time};
use orr::orr_gpu::{Format, Hazards, MemoryKind, QueueKind, SpecData, Stages, TimelinePoint};
use orr::orr_output::Image;
use orr::orr_pack::Target;
use orr::orr_pack::material::Material;
use orr::{
    Affine3, Display, DisplayRequest, FrameDesc, FrameTime, GroupWriter, InstanceFlags,
    InstancesDesc, Lens, LoadState, PackSource, Pose, Projection, Renderer, RendererDesc,
    SceneDesc, SceneLightingSettings, TransformGroupDesc, ViewDesc, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshSettings, Source};
use orr_math::camera::rh::view;
use orr_math::{DVec3, Mat3, UVec2, Vec2, Vec3};

const TREE: &str = "tree";
const GROUND: &str = "ground";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
const FOV_Y: f32 = 0.9;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.512_092_8, 0.716_929_9, 0.473_016_4);
/// Sun illuminance and sky radiance, relative to paper white.
const SUN_ILLUMINANCE: f32 = 3.0;
const SKY: [f32; 3] = [0.18, 0.24, 0.34];
/// Trees per side of the stand, and the spacing between them in m.
const SIDE: u32 = 20;
const SPACING: f32 = 5.0;
const BRANCHES: u32 = 12;
/// m from the camera within which trees sway.
const ANIMATION_RANGE: f32 = 22.0;
/// Largest branch tilt, rad.
const STRENGTH: f32 = 0.3;
/// Radians per second.
const ORBIT_RATE: f64 = 0.15;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("rigid-parts");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Branch `b`: its direction out of the trunk, its height on it and its length.
fn branch(b: u32) -> (Vec3, f32, f32) {
    let dirs = [Vec3::X, Vec3::Z, Vec3::NEG_X, Vec3::NEG_Z];
    (
        dirs[(b % 4) as usize],
        1.7 + b as f32 * 0.22,
        1.5 - 0.07 * b as f32,
    )
}

/// Runs the example; headless runs return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr rigid-parts", 1280, 720));
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

    let copies = SIDE * SIDE;
    let parts = BRANCHES + 1;
    let scene = renderer
        .create_scene(
            &SceneDesc {
                instance_capacity: copies + 1,
                group_capacity: 4,
                light_capacity: 1,
                material_set_capacity: 16,
                emissive_capacity: 16,
                part_capacity: copies * 16,
                ptlas_partitions: 64,
                global_instances: 16,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    let [tree, ground] = [TREE, GROUND].map(|name| {
        renderer
            .model(mount, orr::orr_pack::AssetId::from_name(name))
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
            count: copies + 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .expect("transform group");
    // One pose block per copy, so each tree sways in its own phase.
    let poses = w
        .create_transform_group(&TransformGroupDesc {
            count: copies * parts,
            writer: GroupWriter::Gpu,
            anchor: DVec3::ZERO,
        })
        .expect("pose group");
    let transforms = w.transforms(group, 0, copies + 1).expect("transforms");
    for (i, t) in transforms.iter_mut().take(copies as usize).enumerate() {
        let (x, z) = ((i as u32 % SIDE) as f32, (i as u32 / SIDE) as f32);
        // A little disorder: each tree turned and nudged off the grid.
        let h = (i as u32).wrapping_mul(2_654_435_761);
        let jitter = |shift: u32| ((h >> shift) & 255) as f32 / 255.0 - 0.5;
        let at = Vec3::new(
            (x - (SIDE - 1) as f32 * 0.5 + jitter(0)) * SPACING,
            0.0,
            (z - (SIDE - 1) as f32 * 0.5 + jitter(8)) * SPACING,
        );
        *t = affine(Mat3::from_rotation_y(jitter(16) * 6.0), at);
    }
    transforms[copies as usize] = Affine3::IDENTITY;
    w.add_instances(&InstancesDesc {
        model: tree,
        copies,
        transforms: group,
        first: 0,
        pose: Some(Pose {
            group: poses,
            index: 0,
            variants: 1,
            stride: 1,
        }),
        material_set: None,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 1000.0,
        animation_range: ANIMATION_RANGE,
        user_tag: 0,
    })
    .expect("trees");
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

    let sway = renderer
        .compute_pipeline(&wind::MODULE, "main", SpecData::default())
        .expect("wind pipeline");
    let shade = renderer
        .compute_pipeline(&sun_shadow::MODULE, "main", SpecData::default())
        .expect("shade pipeline");
    let encode = renderer
        .compute_pipeline(&capture::MODULE, "main", SpecData::default())
        .expect("capture pipeline");
    let mut branches = gpu
        .gpu_malloc::<wind::Branch>(MemoryKind::Upload, u64::from(parts), "branches")
        .expect("branches");
    for (b, out) in branches
        .as_mut_slice()
        .expect("branches")
        .iter_mut()
        .enumerate()
        .skip(1)
    {
        let (dir, height, _) = branch(b as u32 - 1);
        *out = wind::Branch {
            pivot: [0.0, height, 0.0],
            _pad0: 0.0,
            axis: Vec3::Y.cross(dir).to_array(),
            _pad1: 0.0,
        };
    }
    let slots = u64::from(desc.frames_in_flight);
    let mut roots = gpu
        .gpu_malloc::<sun_shadow::Root>(MemoryKind::Upload, slots, "rigid-parts roots")
        .expect("roots");
    let mut capture_root = gpu
        .gpu_malloc::<capture::Root>(MemoryKind::Upload, 1, "rigid-parts capture root")
        .expect("capture root");
    let pixels = u64::from(HEADLESS_EXTENT[0] * HEADLESS_EXTENT[1]);
    let readback = options.headless.then(|| {
        gpu.gpu_malloc::<u32>(MemoryKind::Readback, pixels, "rigid-parts capture")
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
        let angle = 0.4 + t * ORBIT_RATE;
        let eye = Vec3::new(angle.sin() as f32 * 13.0, 6.5, angle.cos() as f32 * 13.0);
        let target = Vec3::new(0.0, 1.5, 0.0);

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
        let wind_root = |root| wind::Root {
            scene: root,
            branches: branches.gpu,
            group: poses.0,
            copies,
            parts,
            time: t as f32,
            strength: STRENGTH,
            _pad0: 0,
            _pad1: 0,
        };
        f.update_scenes(&mut |c| {
            let root = c.put_root(&wind_root(c.root)).expect("wind root");
            c.cmd.dispatch(
                c.pipeline(sway),
                root,
                [(copies * parts).div_ceil(64), 1, 1],
            );
        })
        .expect("update scenes");
        let view = f
            .raster_view(&ViewDesc {
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
                target: if options.headless {
                    ViewTarget::AfterRr
                } else {
                    ViewTarget::Display
                },
            })
            .expect("raster view");

        let fresh = image
            .as_ref()
            .is_none_or(|im| im.texture().desc().extent[..2] != extent);
        if fresh {
            image = Some(
                Image::new(gpu, Format::Rgba16Float, extent, "rigid-parts shaded").expect("image"),
            );
        }
        let target = image.as_ref().expect("image");
        let slot = f.id() % slots;
        roots.as_mut_slice().expect("roots")[slot as usize] = sun_shadow::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            target: target.storage,
            mode: 0,
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
        rgba8: out
            .as_slice()
            .expect("readback")
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    });
    drop(image);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// The tree and the ground, cooked into an in-memory pack. The tree's trunk is part 0 and
/// branch `b`, with its leaves, part `b + 1`, pivoted where it leaves the trunk's axis.
fn cook() -> Arc<[u8]> {
    let bark = Material {
        base_color: [0.36, 0.26, 0.18],
        specular_roughness: 0.8,
        ..Material::default()
    };
    let leaves = Material {
        base_color: [0.22, 0.48, 0.16],
        specular_roughness: 0.6,
        ..Material::default()
    };
    let soil = Material {
        base_color: [0.42, 0.38, 0.3],
        specular_roughness: 0.9,
        ..Material::default()
    };
    let mut tree = MeshBuilder::default();
    tree.cuboid(
        0,
        Vec3::new(-0.16, 0.0, -0.16),
        Vec3::new(0.16, 4.6, 0.16),
        2,
    );
    for b in 0..BRANCHES {
        let (dir, height, length) = branch(b);
        tree.part(b as u16 + 1, Vec3::new(0.0, height, 0.0));
        let side = Vec3::Y.cross(dir);
        let base = Vec3::Y * height + dir * 0.1;
        let tip = base + dir * length;
        let corners = |a: Vec3, b: Vec3, half: Vec3| (a.min(b) - half, a.max(b) + half);
        let (lo, hi) = corners(base, tip, side.abs() * 0.05 + Vec3::Y * 0.05);
        tree.cuboid(0, lo, hi, 1);
        let leaf = Vec3::new(0.3, 0.08, 0.3);
        tree.cuboid(1, tip - leaf, tip + leaf, 1);
    }
    let mut floor = MeshBuilder::default();
    let h = 70.0;
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
    let mesh = |name: &str, mesh: MeshInput, materials: Vec<Material>| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings {
                lod: LodMode::default(),
                ..MeshSettings::default()
            },
        },
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                mesh(TREE, tree.finish(), vec![bark, leaves]),
                mesh(GROUND, floor.finish(), vec![soil]),
            ],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0)
}
