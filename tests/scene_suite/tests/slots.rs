//! Removed instance slots are reused only once the frame that removed them has completed.
//! In its own process: a host timeline holds the render queue.

use orr_gpu::GpuPtr;
use orr_scene::testing::{assert_clean, device, read_buffer, wait};
use orr_scene::{
    FrameDesc, FrameTime, InstanceFlags, InstanceRange, InstancesDesc, RendererDesc, SceneDesc,
    SceneRoot, Scenes, TransformGroupId,
};

#[test]
fn removed_slots_wait_for_their_frame() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let (_workers, model) = scene_suite::mount_quad(&mut scenes);
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let list = [s];
    let desc = FrameDesc {
        time: FrameTime::default(),
        scenes: &list,
        low_latency: None,
    };
    let add = |w: &mut orr_scene::SceneWriter<'_>| {
        w.add_instances(&InstancesDesc {
            model,
            copies: 100,
            transforms: TransformGroupId(0),
            first: 0,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER,
            rt_range: 0.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap()
    };

    let mut f = scenes.begin_frame(&desc).unwrap();
    let a = add(&mut f.scene(s).unwrap());
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    f.end(&[]).unwrap();

    // The frame removing `a` cannot complete until the gate opens.
    let gate = d.create_host_timeline("gate").unwrap();
    let f = scenes.begin_frame(&desc).unwrap();
    let b = {
        let mut w = f.scene(s).unwrap();
        w.remove(a);
        add(&mut w)
    };
    let removal = f.end(&[gate.point(1)]).unwrap();

    // Frame 3 waits only for frame 1, so `a`'s slots are still held.
    let f = scenes.begin_frame(&desc).unwrap();
    let c = add(&mut f.scene(s).unwrap());
    f.end(&[]).unwrap();
    gate.signal(1);
    wait(removal);
    assert_eq!((a.first, b.first, c.first), (0, 100, 200));
    let root: SceneRoot = read_buffer(root, 1)[0];
    let removed = read_buffer(root.instances, 100);
    assert!(removed.iter().all(is_empty), "removed slots are not empty");

    // Frame 4 waits for frame 2, after which `a`'s slots are free again.
    let f = scenes.begin_frame(&desc).unwrap();
    let e = add(&mut f.scene(s).unwrap());
    wait(f.end(&[]).unwrap());
    assert_eq!(
        e,
        InstanceRange {
            first: 0,
            count: 100
        }
    );

    let slots = read_buffer(root.instances, 300);
    assert!(slots.iter().all(|i| i.flags == 1 && i.model == model.0));
    assert_clean();
}

fn is_empty(i: &orr_scene::InstanceGpu) -> bool {
    bytemuck::bytes_of(i).iter().all(|&b| b == 0)
}
