//! The frame through `orr::Renderer`: frames with no scene and no view, and a scene frame
//! with a view.

use glam::{DVec3, Quat, UVec2, Vec2};
use orr::{
    FrameDesc, FrameTime, Lens, Projection, Renderer, RendererDesc, SceneDesc, ViewDesc, ViewTarget,
};
use orr_scene::testing::{assert_clean, device, wait};

#[test]
fn empty_frames_and_scene_frames_submit_in_order() {
    let mut r = Renderer::new(device(), RendererDesc::default()).unwrap();
    let mut ids = Vec::new();
    let mut last = None;
    for i in 0..4 {
        let f = r
            .begin_frame(FrameDesc {
                time: FrameTime {
                    absolute: f64::from(i) / 60.0,
                    delta: 1.0 / 60.0,
                },
                scenes: &[],
                display: None,
            })
            .unwrap();
        ids.push(f.id());
        last = Some(f.end().unwrap());
    }
    assert_eq!(ids, [1, 2, 3, 4]);

    let s = r.create_scene(&SceneDesc::default()).unwrap();
    let history = r.create_view_history().unwrap();
    let list = [s];
    for i in 0..3 {
        let mut f = r
            .begin_frame(FrameDesc {
                time: FrameTime::default(),
                scenes: &list,
                display: None,
            })
            .unwrap();
        f.scene(s)
            .unwrap()
            .set_focus(DVec3::new(f64::from(i) * 700.0, 0.0, 0.0), 1000.0);
        f.update_scenes(&mut |_| {}).unwrap();
        f.render_view(&ViewDesc {
            scene: s,
            history,
            position: DVec3::new(f64::from(i) * 700.0, 2.0, 0.0),
            rotation: Quat::IDENTITY,
            projection: Projection {
                fov_y: 1.0,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::new(1280, 720),
            jitter: Vec2::ZERO,
            reset_history: false,
            target: ViewTarget::AfterRr,
        })
        .unwrap();
        last = Some(f.end().unwrap());
    }
    wait(last.unwrap());
    let f = r
        .begin_frame(FrameDesc {
            time: FrameTime::default(),
            scenes: &[],
            display: None,
        })
        .unwrap();
    wait(f.end().unwrap());
    let stats = r.stats();
    assert_eq!(stats.frame, 7);
    let steps: Vec<&str> = stats.cpu.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        steps,
        [
            "begin_frame",
            "write",
            "update_scenes",
            "raster_view",
            "end"
        ]
    );
    assert!(stats.gpu.iter().any(|p| p.name == "rebase"));
    assert_clean();
}
