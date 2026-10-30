//! orr-gpu on its own: a device, GPU memory, the texture heap, a compute pass that paints a
//! texture and a mesh pass that draws a textured, depth-tested cube sampling it.
//!
//! Runs headless for a fixed number of frames and writes the last one as a binary PPM:
//! `gpu-basics [frames] [out.ppm]` (default 120 frames, `gpu-basics.ppm`).

use std::time::Duration;

use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};
use gpu_suite::shaders::basics;
use gpu_suite::{device, readback, render, upload};
use orr_gpu::{
    ClearColor, ColorAttachment, ColorLoad, ColorTarget, CompareOp, Cull, DepthAttachment,
    DepthLoad, DepthTarget, Filter, Format, FrontFace, Hazards, MemoryKind, MeshPipelineDesc,
    RasterState, Rendering, SamplerDesc, SamplerIndex, ShaderStage, SpecData, Stages, TextureDesc,
    TextureMemory, TextureRegion, TextureUsage, TimelinePoint, ViewDesc,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const PAINT: u32 = 256;
/// Frames whose roots may be in flight at once.
const SLOTS: usize = 2;

/// The cube's 24 vertices: four per face, wound counter-clockwise seen from outside.
fn cube() -> Vec<basics::CubeVertex> {
    let mut out = Vec::new();
    for (n, t) in [
        (Vec3::X, Vec3::Z),
        (Vec3::NEG_X, Vec3::NEG_Z),
        (Vec3::Y, Vec3::X),
        (Vec3::NEG_Y, Vec3::NEG_X),
        (Vec3::Z, Vec3::NEG_X),
        (Vec3::NEG_Z, Vec3::X),
    ] {
        let b = n.cross(t);
        for (s, r) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            let p = n + t * s + b * r;
            out.push(basics::CubeVertex {
                position: p.to_array(),
                u: (s + 1.0) * 0.5,
                normal: n.to_array(),
                v: (1.0 - r) * 0.5,
            });
        }
    }
    out
}

/// Row-major rows, as Slang's `float4x4` reads them.
fn rows(m: Mat4) -> [[f32; 4]; 4] {
    m.transpose().to_cols_array_2d()
}

#[allow(clippy::print_stdout)]
fn main() {
    let mut args = std::env::args().skip(1);
    let frames: u64 = args.next().map_or(120, |a| a.parse().expect("frame count"));
    let out_path = args.next().unwrap_or_else(|| "gpu-basics.ppm".into());

    let d = device();
    let queue = render();

    let paint_desc = TextureDesc::d2(
        Format::Rgba8Unorm,
        PAINT,
        PAINT,
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
    );
    let paint = d
        .create_texture(&paint_desc, TextureMemory::Dedicated, "paint")
        .unwrap();
    let paint_target = d.write_storage(d.alloc_image_slot().unwrap(), &paint, &ViewDesc::default());
    let texture = d.write_sampled(d.alloc_image_slot().unwrap(), &paint, &ViewDesc::default());
    let sampler = SamplerIndex(1);
    d.write_sampler(
        sampler,
        &SamplerDesc {
            mip: Filter::Nearest,
            max_anisotropy: 8.0,
            ..SamplerDesc::default()
        },
    );

    // The colour target and depth share one device-local block.
    let color_desc = TextureDesc::d2(Format::Rgba8Srgb, WIDTH, HEIGHT, TextureUsage::COLOR);
    let depth_desc = TextureDesc::d2(Format::D32Float, WIDTH, HEIGHT, TextureUsage::DEPTH);
    let (color_size, _) = d.texture_requirements(&color_desc);
    let (depth_size, depth_align) = d.texture_requirements(&depth_desc);
    let depth_offset = color_size.next_multiple_of(depth_align);
    let block = d
        .gpu_malloc::<u8>(
            MemoryKind::DeviceLocal,
            depth_offset + depth_size,
            "targets",
        )
        .unwrap();
    let placed = |offset| TextureMemory::Placed {
        block: &block,
        offset,
    };
    let color = d.create_texture(&color_desc, placed(0), "color").unwrap();
    let depth = d
        .create_texture(&depth_desc, placed(depth_offset), "depth")
        .unwrap();

    let stage = |e| ShaderStage::new(&basics::MODULE, e, SpecData::default()).unwrap();
    let paint_pipeline = d
        .create_compute_pipeline(&stage("paint_main"), "paint")
        .unwrap();
    let cube_pipeline = d
        .create_mesh_pipeline(
            &MeshPipelineDesc {
                pixel: Some(stage("pixel_main")),
                colors: &[ColorTarget::new(Format::Rgba8Srgb)],
                depth: Some(DepthTarget {
                    format: Format::D32Float,
                    compare: CompareOp::Greater,
                    write: true,
                }),
                raster: RasterState {
                    cull: Cull::Back,
                    front_face: FrontFace::CounterClockwise,
                    ..RasterState::default()
                },
                ..MeshPipelineDesc::new(stage("mesh_main"))
            },
            "cube",
        )
        .unwrap();

    let vertices = upload(&cube(), "cube vertices");
    let mut roots = d
        .gpu_malloc::<basics::Root>(MemoryKind::Upload, SLOTS as u64, "roots")
        .unwrap();
    let pixels = readback::<[u8; 4]>(u64::from(WIDTH * HEIGHT), "pixels");
    // Reverse-Z with an infinite far plane; clip-space y points down the target.
    let projection =
        proj::vulkan::perspective_infinite_reverse(0.8, WIDTH as f32 / HEIGHT as f32, 0.1);
    let view = view::look_at_mat4(Vec3::new(0.0, 1.6, 4.2), Vec3::ZERO, Vec3::Y);

    let mut in_flight: [Option<TimelinePoint>; SLOTS] = [None; SLOTS];
    let mut last = None;
    for frame in 0..frames {
        let slot = frame as usize % SLOTS;
        if let Some(p) = in_flight[slot] {
            queue.wait(p, Duration::from_secs(10)).unwrap();
        }
        d.collect_garbage();
        let angle = frame as f32 * 0.03;
        let model = Mat4::from_rotation_y(angle) * Mat4::from_rotation_x(angle * 0.6);
        roots.as_mut_slice().unwrap()[slot] = basics::Root {
            clip_from_object: rows(projection * view * model),
            world_from_object: rows(model),
            vertices: vertices.gpu,
            texture,
            sampler,
            paint_target,
            frame: frame as u32,
            size: PAINT,
            pad: 0,
        };
        let root = roots.gpu.offset(slot as u64);

        let mut list = queue.command_list().unwrap();
        if frame == 0 {
            for t in [&paint, &color, &depth] {
                list.discard(t);
            }
        } else {
            // The last frame's cube pass sampled the texture and wrote the targets.
            list.barrier(
                Stages::PIXEL | Stages::COLOR_OUT | Stages::DEPTH,
                Stages::COMPUTE | Stages::COLOR_OUT | Stages::DEPTH,
                Hazards::empty(),
            );
        }
        list.begin_pass("paint", false);
        list.dispatch(
            &paint_pipeline,
            root,
            [PAINT.div_ceil(8), PAINT.div_ceil(8), 1],
        );
        list.end_pass();
        list.barrier(Stages::COMPUTE, Stages::PIXEL, Hazards::empty());
        list.begin_pass("cube", true);
        list.begin_rendering(&Rendering {
            extent: [WIDTH, HEIGHT],
            colors: &[ColorAttachment {
                texture: &color,
                load: ColorLoad::Clear(ClearColor::Float([0.02, 0.025, 0.035, 1.0])),
                store: true,
            }],
            depth: Some(DepthAttachment {
                texture: &depth,
                load: DepthLoad::Clear(0.0),
                store: false,
            }),
        });
        list.draw_mesh_tasks(&cube_pipeline, root, [1, 1, 1]);
        list.end_rendering();
        list.end_pass();
        if frame + 1 == frames {
            list.barrier(Stages::COLOR_OUT, Stages::TRANSFER, Hazards::empty());
            list.copy_from_texture(
                pixels.gpu.cast(),
                &color,
                &TextureRegion::whole_mip(&color_desc, 0),
            );
            list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
        }
        let point = queue.submit(vec![list], &[]).unwrap();
        queue.mark_frame_end(frame).unwrap();
        in_flight[slot] = Some(point);
        last = Some(point);
    }
    let last = last.expect("at least one frame");
    queue.wait(last, Duration::from_secs(10)).unwrap();
    for pass in d.pass_results(last).unwrap_or_default() {
        println!("{}: {:.3} ms {:?}", pass.name, pass.gpu_ms, pass.stats);
    }

    let mut ppm = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
    for p in pixels.as_slice().unwrap() {
        ppm.extend_from_slice(&p[..3]);
    }
    std::fs::write(&out_path, ppm).unwrap();
    println!(
        "{frames} frames; last frame written to {out_path}; validation messages: {}",
        d.validation_messages()
    );
    assert_eq!(
        d.validation_messages(),
        0,
        "validation messages (see the log)"
    );
}
