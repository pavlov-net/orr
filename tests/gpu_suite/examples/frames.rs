//! Runs compute frames that write and sample heap textures and trace a ray query, each
//! submit ending a frame for capture tools. For a headless Nsight capture:
//! `ngfx-capture --exe <this> --delimiter-vk-frame-boundary-ext --capture-frame 60`.

use std::time::Duration;

use gpu_suite::shaders::{heap_views, rays};
use gpu_suite::{device, pipeline, readback, render, triangle_scene, upload};
use orr_gpu::{
    Filter, Format, Hazards, SamplerDesc, SamplerIndex, Stages, TextureDesc, TextureMemory,
    TextureUsage, ViewDesc,
};

const FRAMES: u64 = 120;

fn main() {
    let d = device();
    let scene = triangle_scene();
    let (w, h) = (512u32, 512u32);
    let desc = TextureDesc::d2(
        Format::Rgba16Float,
        w,
        h,
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
    );
    let tex = d
        .create_texture(&desc, TextureMemory::Dedicated, "frames texture")
        .unwrap();
    let target = d.write_storage(d.alloc_image_slot().unwrap(), &tex, &ViewDesc::default());
    let source = d.write_sampled(d.alloc_image_slot().unwrap(), &tex, &ViewDesc::default());
    let nearest = SamplerIndex(0);
    d.write_sampler(
        nearest,
        &SamplerDesc {
            mag: Filter::Nearest,
            min: Filter::Nearest,
            mip: Filter::Nearest,
            ..SamplerDesc::default()
        },
    );

    let sampled = readback::<[f32; 4]>(u64::from(w * h), "frames sampled");
    let hits = readback::<u32>(u64::from(w * h), "frames hits");
    let scene_ptr = upload(&[scene.tlas.tlas()], "frames scene pointer");
    let heap_root = upload(
        &[heap_views::Root {
            out: sampled.gpu,
            source,
            nearest,
            target,
            width: w,
            height: h,
            pad: 0,
        }],
        "frames heap root",
    );
    let ray_root = upload(
        &[rays::Root {
            out: hits.gpu,
            scene: scene.tlas.tlas(),
            scene_ptr: scene_ptr.gpu,
            width: w,
            height: h,
        }],
        "frames ray root",
    );
    let store = pipeline(&heap_views::MODULE, "store_main");
    let sample = pipeline(&heap_views::MODULE, "sample_main");
    let trace = pipeline(&rays::MODULE, "main");
    let groups = [w.div_ceil(8), h.div_ceil(4), 1];

    let queue = render();
    let mut last = None;
    for frame in 0..FRAMES {
        let mut list = queue.command_list().unwrap();
        if frame == 0 {
            list.discard(&tex);
        }
        list.begin_pass("heap textures", true);
        list.dispatch(&store, heap_root.gpu, groups);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&sample, heap_root.gpu, groups);
        list.end_pass();
        list.begin_pass("ray query", true);
        list.dispatch(&trace, ray_root.gpu, groups);
        list.end_pass();
        list.barrier(
            Stages::COMPUTE,
            Stages::HOST | Stages::COMPUTE,
            Hazards::empty(),
        );
        let point = queue.submit(vec![list], &[]).unwrap();
        queue.mark_frame_end(frame).unwrap();
        queue.wait(point, Duration::from_secs(10)).unwrap();
        d.collect_garbage();
        last = Some(point);
    }
    let last = last.unwrap();
    for p in d.pass_results(last).unwrap() {
        log::info!("{}: {:.4} ms, {:?}", p.name, p.gpu_ms, p.stats);
    }
    let hit = hits.as_slice().unwrap().iter().filter(|&&v| v != 0).count();
    log::info!(
        "{FRAMES} frames, {hit} of {} rays hit, frame boundaries {}, validation messages {}",
        w * h,
        d.caps().frame_boundary,
        d.validation_messages()
    );
    d.free_image_slot(target.0, last);
    d.free_image_slot(source.0, last);
}
