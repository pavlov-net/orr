//! Removed instance slots are reused only once the frame that removed them has completed.
//! In its own process: the gate holds the render queue.

use orr_gpu::{GpuPtr, MemoryKind};
use orr_scene::testing::{assert_clean, device, read_buffer, wait};
use orr_scene::{
    FrameDesc, FrameTime, InstanceFlags, InstanceRange, InstancesDesc, ModelId, RendererDesc,
    SceneDesc, SceneRoot, Scenes, TransformGroupId,
};
use scene_suite::pipeline;
use scene_suite::shaders::gate;

#[test]
fn removed_slots_wait_for_their_frame() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let list = [s];
    let desc = FrameDesc {
        time: FrameTime::default(),
        scenes: &list,
    };
    let add = |w: &mut orr_scene::SceneWriter<'_>| {
        w.add_instances(&InstancesDesc {
            model: ModelId(1),
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
    let mut flag = d
        .gpu_malloc::<u32>(MemoryKind::Upload, 1, "gate flag")
        .unwrap();
    flag.as_mut_slice().unwrap()[0] = 0;
    let out = d
        .gpu_malloc::<u32>(MemoryKind::Readback, 1, "gate out")
        .unwrap();
    let mut gate_root = d
        .gpu_malloc::<gate::Root>(MemoryKind::Upload, 1, "gate root")
        .unwrap();
    gate_root.as_mut_slice().unwrap()[0] = gate::Root {
        flag: flag.gpu,
        out: out.gpu,
    };
    let p = pipeline(&gate::MODULE, "main");
    let mut f = scenes.begin_frame(&desc).unwrap();
    let b = {
        let mut w = f.scene(s).unwrap();
        w.remove(a);
        add(&mut w)
    };
    f.update_scenes(&mut |c| c.cmd.dispatch(&p, gate_root.gpu, [1, 1, 1]))
        .unwrap();
    let removal = f.end(&[]).unwrap();

    // Frame 3 waits only for frame 1, so `a`'s slots are still held.
    let f = scenes.begin_frame(&desc).unwrap();
    let c = add(&mut f.scene(s).unwrap());
    f.end(&[]).unwrap();
    flag.as_mut_slice().unwrap()[0] = 1;
    wait(removal);
    assert_eq!(out.as_slice().unwrap()[0], 1, "the gate timed out");
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
    assert!(slots.iter().all(|i| i.flags == 1 && i.model == 1));
    assert_clean();
}

fn is_empty(i: &orr_scene::InstanceGpu) -> bool {
    orr_shaders::bytemuck::bytes_of(i).iter().all(|&b| b == 0)
}
