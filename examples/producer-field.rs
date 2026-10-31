//! A field of producer entries: a heightfield tile of grid patches emitted once, grass
//! re-emitted every frame around the camera from template clusters per blade topology, and
//! a pond of a wave-displaced dielectric slab. The grass kernel writes a word per cluster
//! (its clump) and per vertex (height along the blade); a pass after shading finds each
//! grass pixel's hit again and colors it from those words. Shading traces one sun shadow
//! ray per pixel, so the swaying blades shade each other and the ground; the pond is drawn
//! by tracing camera rays against the dielectric mask. The camera orbits the pond.
//!
//! `producer-field [--frames N] [--headless] [--out file.png]`: headless runs render
//! 480 x 270 for 120 frames unless told otherwise and write the last frame as a PNG.

use std::time::{Duration, Instant};

use examples::field::{Grass, Heights, Slab, Terrain, blade_triangles, patch_triangles};
use examples::shaders::{capture, grass, grass_tint, pond, slab, sun_shadow, terrain};
use examples::window::Window;
use examples::{Options, assert_clean, device, fixed_time};
use orr::orr_gpu::{
    Format, Hazards, MemoryKind, QueueKind, SamplerIndex, SpecData, Stages, TimelinePoint,
};
use orr::orr_output::Image;
use orr::orr_pack::material::Material;
use orr::{
    Affine3, BoundTexture, ClusterTopology, Display, DisplayRequest, FrameDesc, FrameTime,
    GroupWriter, Lens, MaterialGpu, ProducerId, Projection, Renderer, RendererDesc, SceneDesc,
    SceneLightingSettings, TransformGroupDesc, ViewDesc, ViewTarget,
};
use orr_math::camera::rh::view;
use orr_math::{DVec3, UVec2, Vec2, Vec3};

use examples::Capture;

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
const FOV_Y: f32 = 0.9;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.512_092_8, 0.616_929_9, 0.473_016_4);
const SUN_ILLUMINANCE: f32 = 3.0;
const SKY: [f32; 3] = [0.18, 0.24, 0.34];
const TERRAIN: ProducerId = ProducerId(0);
const GRASS: ProducerId = ProducerId(1);
const POND: ProducerId = ProducerId(2);
/// The tile: 8 x 8 patches of 8 m from (-32, -32).
const TILE: f32 = 64.0;
const POND_AT: [f32; 2] = [4.0, -3.0];
/// Radians per second.
const ORBIT_RATE: f64 = 0.12;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("producer-field");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Rolling ground with a basin for the pond.
fn ground(x: f32, z: f32) -> f32 {
    let (dx, dz) = (x - POND_AT[0], z - POND_AT[1]);
    0.6 * (0.13 * x).sin() * (0.11 * z).cos() + 0.25 * (0.31 * x + 0.2 * z).sin()
        - 1.4 * (-(dx * dx + dz * dz) / 18.0).exp()
}

/// Runs the example; headless runs return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr producer-field", 1280, 720));
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
    let scene = renderer
        .create_scene(
            &SceneDesc {
                producer_capacity: 4,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    let [
        terrain_kernel,
        grass_kernel,
        slab_kernel,
        shade,
        tint,
        water,
        encode,
    ] = [
        (&terrain::MODULE, "main"),
        (&grass::MODULE, "main"),
        (&slab::MODULE, "main"),
        (&sun_shadow::MODULE, "main"),
        (&grass_tint::MODULE, "main"),
        (&pond::MODULE, "main"),
        (&capture::MODULE, "main"),
    ]
    .map(|(m, e)| {
        renderer
            .compute_pipeline(m, e, SpecData::default())
            .expect("pipeline")
    });

    let heights = Heights::new(gpu, 129, TILE / 128.0, [-TILE / 2.0; 2], ground).expect("heights");
    let land = Terrain {
        patches: 8,
        patch_size: TILE / 8.0,
        corner: [-TILE / 2.0; 2],
    };
    let grass_field = Grass {
        cells: 128,
        cell_size: TILE / 128.0,
        corner: [-TILE / 2.0; 2],
        radius: 16.0,
        capacity: 4096,
    };
    let water_slab = Slab {
        patches: 2,
        patch_size: 5.0,
        corner: [POND_AT[0] - 5.0, POND_AT[1] - 5.0],
        level: ground(POND_AT[0], POND_AT[1]) + 0.9,
        amplitude: 0.03,
    };
    let patch = patch_triangles();
    let patch_topology = [ClusterTopology {
        vertex_count: 81,
        triangles: &patch,
    }];
    let (tall, short) = (blade_triangles(true), blade_triangles(false));
    let blade_topologies = [
        ClusterTopology {
            vertex_count: 40,
            triangles: &tall,
        },
        ClusterTopology {
            vertex_count: 24,
            triangles: &short,
        },
    ];

    // Materials, the transform and the entries; the terrain is static and emits in the
    // frame it is set.
    let listed = [scene];
    let frame = 1;
    let mut f = renderer
        .begin_frame(FrameDesc {
            time: fixed_time(frame),
            scenes: &listed,
            display: None,
        })
        .expect("begin frame");
    let material = |m: Material| MaterialGpu::new(&m, &[BoundTexture::NONE; 12], SamplerIndex(0));
    let (soil, blades, water_material) = {
        let mut mats = f.materials();
        let soil = Material {
            base_color: [0.42, 0.36, 0.26],
            specular_roughness: 0.9,
            ..Material::default()
        };
        let grass = Material {
            base_color: [0.32, 0.58, 0.2],
            specular_roughness: 0.6,
            double_sided: 1,
            ..Material::default()
        };
        let water = Material {
            base_color: [1.0; 3],
            specular_roughness: 0.05,
            specular_ior: 1.333,
            transmission_weight: 1.0,
            ..Material::default()
        };
        (
            mats.create(&material(soil)).expect("soil"),
            mats.create(&material(grass)).expect("grass"),
            mats.create(&material(water)).expect("water"),
        )
    };
    let mut w = f.scene(scene).expect("scene writer");
    let group = w
        .create_transform_group(&TransformGroupDesc {
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .expect("transform group");
    w.transforms(group, 0, 1).expect("transforms")[0] = Affine3::IDENTITY;
    let (soil, blades, water_material) = ([soil], [blades], [water_material]);
    w.set_producer(
        TERRAIN,
        Some(&land.desc(&patch_topology, &soil, (group, 0))),
    )
    .expect("terrain");
    w.set_producer(
        GRASS,
        Some(&grass_field.desc(&blade_topologies, &blades, (group, 0))),
    )
    .expect("grass");
    w.set_producer(
        POND,
        Some(&water_slab.desc(&patch_topology, &water_material, (group, 0))),
    )
    .expect("pond");
    drop(w);
    f.update_scenes(&mut |c| land.record(c, TERRAIN, terrain_kernel, &heights))
        .expect("update scenes");
    f.end().expect("end frame");

    let slots = u64::from(desc.frames_in_flight);
    let mut roots = gpu
        .gpu_malloc::<sun_shadow::Root>(MemoryKind::Upload, slots, "producer-field roots")
        .expect("roots");
    let mut tint_roots = gpu
        .gpu_malloc::<grass_tint::Root>(MemoryKind::Upload, slots, "producer-field tint roots")
        .expect("tint roots");
    let mut water_roots = gpu
        .gpu_malloc::<pond::Root>(MemoryKind::Upload, slots, "producer-field pond roots")
        .expect("pond roots");
    let mut capture_root = gpu
        .gpu_malloc::<capture::Root>(MemoryKind::Upload, 1, "producer-field capture root")
        .expect("capture root");
    let pixels = u64::from(HEADLESS_EXTENT[0] * HEADLESS_EXTENT[1]);
    let readback = options.headless.then(|| {
        gpu.gpu_malloc::<u32>(MemoryKind::Readback, pixels, "producer-field capture")
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
        let previous_t = previous;
        previous = t;
        let angle = 0.6 + t * ORBIT_RATE;
        let (x, z) = (
            POND_AT[0] + angle.sin() as f32 * 9.0,
            POND_AT[1] + angle.cos() as f32 * 9.0,
        );
        let eye = Vec3::new(x, heights.at(x, z) + 1.4, z);
        let target = Vec3::new(
            POND_AT[0],
            heights.at(POND_AT[0], POND_AT[1]) + 0.6,
            POND_AT[1],
        );

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
        f.update_scenes(&mut |c| {
            grass_field.record(
                c,
                GRASS,
                grass_kernel,
                &heights,
                [eye.x, eye.z],
                t as f32,
                previous_t as f32,
            );
            water_slab.record(c, POND, slab_kernel, t as f32);
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
                Image::new(gpu, Format::Rgba16Float, extent, "producer-field shaded")
                    .expect("image"),
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
        tint_roots.as_mut_slice().expect("tint roots")[slot as usize] = grass_tint::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            target: target.storage,
            entry: GRASS.0,
        };
        water_roots.as_mut_slice().expect("pond roots")[slot as usize] = pond::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            target: target.storage,
            _pad0: 0,
            sky: [SKY[0] * 1.6, SKY[1] * 1.6, SKY[2] * 1.6, 0.0],
            tint: [0.35, 0.55, 0.6, 0.0],
        };
        let (shade, tint, water, encode) = (
            f.pipeline(shade),
            f.pipeline(tint),
            f.pipeline(water),
            f.pipeline(encode),
        );
        let list = f.list().expect("frame list");
        if fresh {
            list.discard(target.texture());
        }
        let groups = [extent[0].div_ceil(8), extent[1].div_ceil(8), 1];
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
        list.begin_pass("shade", false);
        list.dispatch(shade, roots.gpu.offset(slot), groups);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(tint, tint_roots.gpu.offset(slot), groups);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(water, water_roots.gpu.offset(slot), groups);
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
            list.dispatch(encode, capture_root.gpu, groups);
            list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        }
        list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
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
    drop(heights);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}
