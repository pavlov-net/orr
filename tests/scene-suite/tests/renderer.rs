//! The frame through `orr::Renderer`: frames with no scene and no view, a scene frame with
//! a view, and a consumer pipeline that hot reload replaces.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{DVec3, Quat, UVec2, Vec2};
use orr::{
    FrameDesc, FrameTime, Lens, PipelineId, Projection, ReloadReport, Renderer, RendererDesc,
    SceneDesc, SceneId, TextureFilter, ViewDesc, ViewTarget,
};
use orr_gpu::{GpuPtr, MemoryKind, SpecData};
use orr_scene::testing::{assert_clean, device, read_buffer, wait};
use scene_suite::hot::fill;

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

    let s = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
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
        f.raster_view(&ViewDesc {
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
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
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

/// Writes a file's original text back when dropped.
struct Restore {
    path: PathBuf,
    text: String,
}

impl Drop for Restore {
    fn drop(&mut self) {
        fs::write(&self.path, &self.text).unwrap();
    }
}

const FILLED: u64 = 64;

/// One frame that dispatches `fill` into `out`: what hot reload applied at its step 1 and
/// the values it wrote.
fn fill_frame(
    r: &mut Renderer,
    s: SceneId,
    pipeline: PipelineId,
    out: GpuPtr<u32>,
) -> (Vec<ReloadReport>, Vec<u32>) {
    let list = [s];
    let mut f = r
        .begin_frame(FrameDesc {
            time: FrameTime::default(),
            scenes: &list,
            display: None,
        })
        .unwrap();
    let reloads = f.reloads().to_vec();
    f.update_scenes(&mut |c| {
        let root = c
            .put_root(&fill::Root {
                out,
                count: FILLED as u32,
                pad: 0,
            })
            .unwrap();
        c.cmd.dispatch(c.pipeline(pipeline), root, [1, 1, 1]);
    })
    .unwrap();
    wait(f.end().unwrap());
    (reloads, read_buffer(out, FILLED))
}

/// Writes `text` over the kernel and runs frames until hot reload reports on it.
fn edit(
    path: &Path,
    text: &str,
    r: &mut Renderer,
    s: SceneId,
    pipeline: PipelineId,
    out: GpuPtr<u32>,
) -> (ReloadReport, Vec<u32>) {
    fs::write(path, text).unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        let (reports, values) = fill_frame(r, s, pipeline, out);
        let report = reports.into_iter().find(|rep| match rep {
            ReloadReport::Swapped { kernel, .. } | ReloadReport::Rejected { kernel, .. } => {
                *kernel == fill::MODULE.name
            }
        });
        if let Some(report) = report {
            return (report, values);
        }
    }
    panic!("no reload of {} within 60 s", fill::MODULE.name);
}

#[test]
fn editing_a_kernel_body_swaps_its_pipeline_and_bad_edits_keep_the_old_one() {
    let path = Path::new(scene_suite::HOT_SRC).join("fill.slang");
    let original = fs::read_to_string(&path).unwrap();
    let _restore = Restore {
        path: path.clone(),
        text: original.clone(),
    };
    let mut r = Renderer::new(device(), RendererDesc::default()).unwrap();
    let loading = r.loading();
    let worker = std::thread::spawn({
        let loading = Arc::clone(&loading);
        move || loading.run_worker()
    });
    let pipeline = r
        .compute_pipeline(&fill::MODULE, "main", SpecData::default())
        .unwrap();
    let s = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let out = device()
        .gpu_malloc::<u32>(MemoryKind::DeviceLocal, FILLED, "fill")
        .unwrap();
    let (_, values) = fill_frame(&mut r, s, pipeline, out.gpu);
    assert!(values.iter().all(|&v| v == 1), "{values:?}");

    let two = original.replace("root.out[id.x] = 1;", "root.out[id.x] = 2;");
    assert_ne!(two, original);
    let (report, values) = edit(&path, &two, &mut r, s, pipeline, out.gpu);
    assert_eq!(
        report,
        ReloadReport::Swapped {
            kernel: "fill",
            pipelines: 1
        }
    );
    assert!(values.iter().all(|&v| v == 2), "{values:?}");

    let broken = two.replace("= 2;", "= 2 +;");
    let (report, values) = edit(&path, &broken, &mut r, s, pipeline, out.gpu);
    let ReloadReport::Rejected { reason, .. } = report else {
        panic!("a syntax error swapped the pipeline");
    };
    assert!(reason.contains("fill.slang:19:"), "{reason}");
    assert!(values.iter().all(|&v| v == 2), "{values:?}");

    let wider = two.replace(
        "uint pad;",
        "uint pad;
    float4 tint;",
    );
    let (report, values) = edit(&path, &wider, &mut r, s, pipeline, out.gpu);
    let ReloadReport::Rejected { reason, .. } = report else {
        panic!("a root edit swapped the pipeline");
    };
    assert!(reason.contains("rebuild required"), "{reason}");
    assert!(values.iter().all(|&v| v == 2), "{values:?}");

    loading.stop_workers();
    worker.join().unwrap();
    drop(r);
    assert_clean();
}
