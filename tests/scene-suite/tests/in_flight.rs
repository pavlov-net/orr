//! Frames in flight: the CPU records frame N + 1 while the GPU still has frame N, and an
//! upload window is reused only after its frame has completed. In its own process: the
//! gate holds the render queue.

use std::time::{Duration, Instant};

use glam::DVec3;
use orr_gpu::{GpuPtr, MemoryKind, QueueKind};
use orr_scene::testing::{assert_clean, device, read_buffer, wait};
use orr_scene::{
    Affine3, FrameDesc, FrameTime, GroupWriter, RendererDesc, SceneDesc, SceneRoot, Scenes,
    TransformGroupDesc, TransformGroupId,
};
use scene_suite::pipeline;
use scene_suite::shaders::gate;

fn filled(salt: f32) -> Affine3 {
    Affine3 {
        rows: [[salt; 4]; 3],
    }
}

#[test]
fn cpu_records_ahead_and_windows_wait_for_their_frame() {
    let d = device();
    let render = d.queue(QueueKind::Render).unwrap();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let list = [s];
    let desc = FrameDesc {
        time: FrameTime::default(),
        scenes: &list,
    };

    // Three groups, one written by each of the next three frames.
    let mut groups = [TransformGroupId(0); 3];
    let mut f = scenes.begin_frame(&desc).unwrap();
    {
        let mut w = f.scene(s).unwrap();
        for g in &mut groups {
            *g = w
                .create_transform_group(&TransformGroupDesc {
                    count: 1024,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .unwrap();
        }
    }
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());

    // Hold the render queue so frame N cannot start.
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
    let mut cmd = render.command_list().unwrap();
    cmd.dispatch(&p, gate_root.gpu, [1, 1, 1]);
    let gate_point = render.submit(vec![cmd], &[]).unwrap();

    let write = |scenes: &mut Scenes, g: TransformGroupId, salt: f32| {
        let mut f = scenes.begin_frame(&desc).unwrap();
        f.scene(s)
            .unwrap()
            .transforms(g, 0, 1024)
            .unwrap()
            .fill(filled(salt));
        let written = Instant::now();
        f.update_scenes(&mut |_| {}).unwrap();
        (f.end(&[]).unwrap(), written)
    };
    let (n, _) = write(&mut scenes, groups[0], 1.0);
    let started = Instant::now();
    let (_, written_n1) = write(&mut scenes, groups[1], 2.0);
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "frame N + 1 waited for the GPU"
    );
    assert!(render.completed() < n.value(), "frame N ran past the gate");

    // Frame N + 2 reuses frame N's window, so it waits for frame N.
    let flag_ptr = flag.cpu.unwrap().as_ptr() as usize;
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        // SAFETY: the flag allocation outlives this thread (joined below) and the GPU only
        // reads it.
        unsafe { (flag_ptr as *mut u32).write_volatile(1) };
    });
    let waited = Instant::now();
    let (n2, _) = write(&mut scenes, groups[2], 3.0);
    assert!(
        waited.elapsed() >= Duration::from_millis(250),
        "frame N + 2 did not wait"
    );
    release.join().unwrap();
    wait(n2);
    assert_eq!(out.as_slice().unwrap()[0], 1, "the gate timed out");
    assert!(render.completed() >= gate_point.value());

    // Each frame's data came from its own window.
    let root: SceneRoot = read_buffer(root, 1)[0];
    let records = read_buffer(root.groups, 4);
    for (g, salt) in groups.iter().zip([1.0, 2.0, 3.0]) {
        let t = read_buffer::<Affine3>(records[g.0 as usize].current.cast(), 1024);
        assert!(t.iter().all(|t| *t == filled(salt)), "{g:?}");
    }

    // The GPU timestamps of frame N start after the CPU wrote frame N + 1.
    if d.caps().calibrated_timestamps {
        let passes = d.pass_results(n).unwrap();
        let (start, _) = passes[0].cpu.unwrap();
        assert!(
            start > written_n1,
            "frame N started before frame N + 1 was written"
        );
    }
    assert_clean();
}
