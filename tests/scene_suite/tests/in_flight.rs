//! Frames in flight: the CPU records frame N + 1 while the GPU still has frame N, and an
//! upload window is reused only after its frame has completed. In its own process: a host
//! timeline holds frame N, and the render queue behind it.

use std::time::{Duration, Instant};

use orr_gpu::{GpuPtr, QueueKind, TimelinePoint};
use orr_math::DVec3;
use orr_scene::testing::{assert_clean, device, read_buffer, wait};
use orr_scene::{
    AffineRows, FrameDesc, FrameTime, GroupWriter, RendererDesc, SceneDesc, SceneRoot, Scenes,
    TransformGroupDesc, TransformGroupId,
};

fn filled(salt: f32) -> AffineRows {
    AffineRows {
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
        low_latency: None,
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

    // Frame N waits for a gate the CPU opens, so it cannot start until then.
    let gate = d.create_host_timeline("gate").unwrap();

    let write = |scenes: &mut Scenes, g: TransformGroupId, salt: f32, wait: &[TimelinePoint]| {
        let mut f = scenes.begin_frame(&desc).unwrap();
        f.scene(s)
            .unwrap()
            .transforms(g, 0, 1024)
            .unwrap()
            .fill(filled(salt));
        let written = Instant::now();
        f.update_scenes(&mut |_| {}).unwrap();
        (f.end(wait).unwrap(), written)
    };
    let (n, _) = write(&mut scenes, groups[0], 1.0, &[gate.point(1)]);
    let started = Instant::now();
    let (_, written_n1) = write(&mut scenes, groups[1], 2.0, &[]);
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "frame N + 1 waited for the GPU"
    );
    assert!(render.completed() < n.value(), "frame N ran past the gate");

    // Frame N + 2 reuses frame N's window, so it waits for frame N.
    let waited = Instant::now();
    let n2 = std::thread::scope(|t| {
        t.spawn(|| {
            std::thread::sleep(Duration::from_millis(300));
            gate.signal(1);
        });
        write(&mut scenes, groups[2], 3.0, &[]).0
    });
    assert!(
        waited.elapsed() >= Duration::from_millis(250),
        "frame N + 2 did not wait"
    );
    wait(n2);
    assert!(render.completed() >= n.value());

    // Each frame's data came from its own window.
    let root: SceneRoot = read_buffer(root, 1)[0];
    let records = read_buffer(root.groups, 4);
    for (g, salt) in groups.iter().zip([1.0, 2.0, 3.0]) {
        let t = read_buffer::<AffineRows>(records[g.0 as usize].current.cast(), 1024);
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
