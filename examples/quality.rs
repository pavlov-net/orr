//! The showcase scene's image quality, measured the same way for every change: noise, flicker,
//! error and bias against the converged reference, how correlated the noise is, and how long a
//! change of lighting takes to reach the image, with and without DLSS ray reconstruction.
//! Headless. Prints a table, writes every figure to `target/orr-reports/quality-<git rev>.json`
//! and images of where the noise and bias are beside it, in `quality-<git rev>/`.
//!
//! `quality [--extent WxH] [--frames N] [--reference-frames N]`. The defaults are 1280 x 720,
//! 64 frames per measure and 512 reference frames.
#![allow(clippy::print_stdout)]

use std::path::Path;
use std::time::Duration;

use orr::orr_gpu::QueueKind;
use orr::{
    FrameDesc, Lens, Light, LightId, LitView, Projection, Reconstruction, ReferenceSettings,
    Renderer, RendererDesc, ScreenshotRequest, TextureFilter, ThinLens, ViewHistoryId, ViewHooks,
    ViewLighting, ViewOutput, ViewTarget,
};
use orr_examples::linked::KERNELS;
use orr_examples::quality::{Bias, Series, TILE, error_map, luminance, welch};
use orr_examples::scene::{FOV_Y, Showcase, TARGET, orbit_eye};
use orr_examples::{
    Capture, Options, Workers, assert_clean, device, enable_dlss, fixed_time, git_rev, report_dir,
};
use orr_math::camera::rh::view;
use orr_math::{DVec3, Vec3};
use serde_json::{Value, json};

/// The camera's place on its orbit for still measures.
const STILL_ANGLE: f64 = 0.9;
/// Radians per frame of the moving camera: the dlss example's orbit at 60 frames a second.
const ORBIT_STEP: f64 = 0.4 / 60.0;
/// Frames the moving camera orbits before each measured frame arrives at the still camera.
const ORBIT_FRAMES: u32 = 48;
/// Moving runs averaged.
const ORBIT_RUNS: u32 = 8;
/// Frames dropped before a measure, for the histories to fill.
const WARM_FRAMES: u32 = 32;
/// Frames of the step response before and after a light comes on.
const STEP_BEFORE: u32 = 16;
const STEP_AFTER: u32 = 96;
/// The step response's light: a slot the scene leaves free, between the objects.
const STEP_LIGHT: LightId = LightId(1);
const STEP_POSITION: DVec3 = DVec3::new(-0.3, 1.8, 2.2);
/// Candela: about a third of the sun's illuminance on the ground below it.
const STEP_INTENSITY: f32 = 1.2e5;
/// Two-sided significance of a tile's bias.
const BIAS_SIGNIFICANCE: f64 = 1e-3;

struct Bench {
    showcase: Showcase,
    history: ViewHistoryId,
    frame: u64,
    extent: [u32; 2],
    _workers: Workers,
    renderer: Renderer,
}

/// One frame's settings.
#[derive(Clone, Copy)]
struct Shot {
    angle: f64,
    camera_cut: bool,
    reconstruction: Reconstruction,
    reference: bool,
    capture: bool,
    /// Turns the step response's light on or off.
    step_light: Option<bool>,
}

impl Bench {
    fn new(extent: [u32; 2]) -> Self {
        let gpu = device();
        let desc = RendererDesc {
            kernels: &KERNELS,
            ..RendererDesc::default()
        };
        let mut renderer = Renderer::new(gpu, desc).expect("renderer");
        let workers = Workers::new(renderer.loading(), 2);
        let showcase = Showcase::load(&mut renderer, gpu);
        let history = renderer.create_view_history().expect("view history");
        renderer.wait_paths().expect("path kernels");
        let frame = showcase.frame;
        Self {
            showcase,
            history,
            frame,
            extent,
            _workers: workers,
            renderer,
        }
    }

    /// Renders a frame and returns its scene-referred luminance if `shot` captures it.
    fn frame(&mut self, shot: Shot) -> Option<Vec<f32>> {
        self.frame += 1;
        let eye = orbit_eye(shot.angle);
        let listed = [self.showcase.scene];
        let mut f = self
            .renderer
            .begin_frame(FrameDesc {
                time: fixed_time(self.frame),
                scenes: &listed,
                display: None,
            })
            .expect("begin frame");
        let id = f.id();
        let mut writer = f.scene(self.showcase.scene).expect("scene writer");
        writer.set_focus(eye.as_dvec3());
        if let Some(on) = shot.step_light {
            let light = Light::Sphere {
                position: STEP_POSITION,
                radius: 0.1,
                intensity: Vec3::splat(STEP_INTENSITY),
                cone: None,
            };
            writer
                .set_light(STEP_LIGHT, on.then_some(&light))
                .expect("step light");
        }
        drop(writer);
        let output = ViewOutput {
            reconstruction: shot.reconstruction,
            ..ViewOutput::default()
        };
        let lighting = ViewLighting {
            reference: shot.reference.then(|| ReferenceSettings {
                lens: Some(ThinLens {
                    aperture_radius: 1e-4,
                    focus_distance: eye.distance(TARGET),
                }),
            }),
            ..ViewLighting::default()
        };
        let view = LitView {
            scene: self.showcase.scene,
            history: self.history,
            position: eye.as_dvec3(),
            rotation: view::look_at_quat(eye, TARGET, Vec3::Y).inverse(),
            projection: Projection {
                fov_y: FOV_Y,
                near: 0.05,
            },
            // Measured images hold no motion blur.
            lens: Lens {
                shutter: 0.0,
                ..Lens::default()
            },
            camera_cut: shot.camera_cut,
            not_rendering_scene: false,
            screenshot: shot
                .capture
                .then_some(ScreenshotRequest::SceneReferred { rec709: true }),
            texture_filter: TextureFilter::Hardware,
            target: ViewTarget::Radiance {
                extent: self.extent,
            },
        };
        f.prepare_view(&view, &lighting, &output)
            .expect("prepare view");
        let sky = &mut self.showcase.sky;
        f.update_scenes(&mut |c| sky.record(c, &[]).expect("sky passes"))
            .expect("update scenes");
        f.render_view(&view, &lighting, &output, &mut ViewHooks::default())
            .expect("render view");
        let point = f.end().expect("end frame");
        if !shot.capture {
            return None;
        }
        device()
            .queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(30))
            .expect("wait");
        let screenshots = self.renderer.screenshots(self.history, id);
        let capture = screenshots.last().expect("screenshot");
        Some(luminance(&capture.rgba))
    }

    fn series(&self, keep: usize) -> Series {
        Series::new(self.extent[0], self.extent[1], keep)
    }

    /// `frames` frames of the still camera after the warm frames.
    fn still(&mut self, frames: u32, keep: usize, reconstruction: Reconstruction) -> Series {
        let reference = false;
        self.run_still(frames, keep, reconstruction, reference, WARM_FRAMES)
    }

    /// The reference, whose frames share nothing, so it needs no warm frames.
    fn reference(&mut self, frames: u32) -> Series {
        self.run_still(frames, 0, Reconstruction::Off, true, 0)
    }

    fn run_still(
        &mut self,
        frames: u32,
        keep: usize,
        reconstruction: Reconstruction,
        reference: bool,
        warm: u32,
    ) -> Series {
        let mut series = self.series(keep);
        for i in 0..warm + frames {
            let shot = Shot {
                angle: STILL_ANGLE,
                camera_cut: i == 0,
                reconstruction,
                reference,
                capture: i >= warm,
                step_light: None,
            };
            if let Some(frame) = self.frame(shot) {
                series.add(frame);
            }
        }
        series
    }

    /// Frames that arrive at the still camera after orbiting to it, each from its own orbit.
    fn moving(&mut self, reconstruction: Reconstruction) -> Series {
        let mut series = self.series(ORBIT_RUNS as usize);
        for _ in 0..ORBIT_RUNS {
            for i in 0..ORBIT_FRAMES {
                let left = ORBIT_FRAMES - 1 - i;
                let shot = Shot {
                    angle: STILL_ANGLE + f64::from(left) * ORBIT_STEP,
                    camera_cut: i == 0,
                    reconstruction,
                    reference: false,
                    capture: left == 0,
                    step_light: None,
                };
                if let Some(frame) = self.frame(shot) {
                    series.add(frame);
                }
            }
        }
        series
    }

    /// The still camera as a light comes on in frame `STEP_BEFORE`; it goes off again after.
    fn step(&mut self, reconstruction: Reconstruction) -> Series {
        let mut series = self.series(0);
        let shot = Shot {
            angle: STILL_ANGLE,
            camera_cut: false,
            reconstruction,
            reference: false,
            capture: false,
            step_light: None,
        };
        for i in 0..WARM_FRAMES + STEP_BEFORE + STEP_AFTER {
            let on = i == WARM_FRAMES + STEP_BEFORE;
            let frame = self.frame(Shot {
                camera_cut: i == 0,
                capture: i >= WARM_FRAMES,
                step_light: on.then_some(true),
                ..shot
            });
            if let Some(frame) = frame {
                series.add(frame);
            }
        }
        self.frame(Shot {
            step_light: Some(false),
            ..shot
        });
        series
    }
}

/// `values`, each mapped to gray by its share of `scale`.
fn gray(values: &[f32], extent: [u32; 2], scale: f32) -> Capture {
    let rgba8 = values
        .iter()
        .flat_map(|&v| {
            let g = ((v / scale).clamp(0.0, 1.0).sqrt() * 255.0).round() as u8;
            [g, g, g, 255]
        })
        .collect();
    Capture {
        width: extent[0],
        height: extent[1],
        rgba8,
    }
}

/// `values` tone mapped for looking at.
fn tone_mapped(values: &[f32], extent: [u32; 2]) -> Capture {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let white = sorted[sorted.len() * 95 / 100].max(1e-6);
    let mapped: Vec<f32> = values.iter().map(|&v| v / (v + white)).collect();
    gray(&mapped, extent, 1.0)
}

/// A tile map as an image at the full extent: red where a tile runs bright, blue where dark.
fn bias_image(bias: &Bias, extent: [u32; 2]) -> Capture {
    let columns = extent[0].div_ceil(TILE);
    let mut rgba8 = Vec::with_capacity((extent[0] * extent[1] * 4) as usize);
    for y in 0..extent[1] {
        for x in 0..extent[0] {
            let v = bias.map[((y / TILE) * columns + x / TILE) as usize];
            let level = ((v.abs() * 4.0).clamp(0.0, 1.0) * 255.0) as u8;
            rgba8.extend_from_slice(&if v > 0.0 {
                [level, 0, 0, 255]
            } else {
                [0, 0, level, 255]
            });
        }
    }
    Capture {
        width: extent[0],
        height: extent[1],
        rgba8,
    }
}

fn percentile(sorted: &[u32], p: f64) -> Option<u32> {
    (!sorted.is_empty()).then(|| sorted[((sorted.len() - 1) as f64 * p).round() as usize])
}

/// Every measure of one reconstruction setting.
fn measure(
    bench: &mut Bench,
    name: &str,
    reconstruction: Reconstruction,
    frames: u32,
    reference: &Series,
    images: &Path,
) -> Value {
    let extent = bench.extent;
    let truth = reference.mean();
    let still = bench.still(frames, frames as usize, reconstruction);
    let moving = bench.moving(reconstruction);
    let step = bench.step(reconstruction);
    let bias = welch(&still, reference, BIAS_SIGNIFICANCE);
    let lag = step.step_response(STEP_BEFORE as usize);

    let noise = still.noise_map();
    gray(&noise, extent, 0.5).save_png(&images.join(format!("{name}-noise.png")));
    tone_mapped(&still.mean(), extent).save_png(&images.join(format!("{name}-mean.png")));
    bias_image(&bias, extent).save_png(&images.join(format!("{name}-bias.png")));
    tone_mapped(moving.last(), extent).save_png(&images.join(format!("{name}-moving.png")));
    gray(&error_map(moving.last(), &truth), extent, 0.5)
        .save_png(&images.join(format!("{name}-moving-error.png")));
    json!({
        "name": name,
        "noise": still.noise(),
        "flicker": still.flicker(),
        "error_still": still.mean_relative_error(&truth),
        "error_moving": moving.mean_relative_error(&truth),
        "bias_tiles": bias.flagged,
        "tiles": bias.tiles,
        "bias_mean": bias.mean_difference,
        "correlation_right_1": still.spatial_correlation(1),
        "correlation_right_8": still.spatial_correlation(8),
        "correlation_frame_1": still.temporal_correlation(1),
        "lag_median": percentile(&lag, 0.5),
        "lag_p90": percentile(&lag, 0.9),
        "lag_tiles": lag.len(),
    })
}

fn print_row(r: &Value) {
    let number = |k: &str| r[k].as_f64().unwrap_or(f64::NAN);
    println!(
        "{:<8} {:>7.3} {:>8.3} {:>8.4} {:>8.4} {:>6}/{:<6} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>4}/{}",
        r["name"].as_str().unwrap_or(""),
        number("noise"),
        number("flicker"),
        number("error_still"),
        number("error_moving"),
        r["bias_tiles"],
        r["tiles"],
        number("bias_mean"),
        number("correlation_right_1"),
        number("correlation_right_8"),
        number("correlation_frame_1"),
        r["lag_median"],
        r["lag_p90"],
    );
}

fn main() {
    let options = Options::from_args();
    let value = |flag: &str| {
        let at = options.rest.iter().position(|a| a == flag)?;
        Some(options.rest[at + 1].clone())
    };
    let extent = value("--extent").map_or([1280, 720], |v| {
        let (w, h) = v.split_once('x').expect("--extent WxH");
        [w.parse().expect("width"), h.parse().expect("height")]
    });
    let frames = options.frames.map_or(64, |n| n as u32);
    let reference_frames = value("--reference-frames").map_or(512, |v| v.parse().expect("count"));

    let rev = git_rev();
    let images = report_dir().join(format!("quality-{rev}"));
    std::fs::create_dir_all(&images).expect("image directory");

    let mut bench = Bench::new(extent);
    let dlss = enable_dlss(&mut bench.renderer);
    let reference = bench.reference(reference_frames);
    tone_mapped(&reference.mean(), extent).save_png(&images.join("reference.png"));

    let mut rows = vec![measure(
        &mut bench,
        "raw",
        Reconstruction::Off,
        frames,
        &reference,
        &images,
    )];
    if dlss {
        rows.push(measure(
            &mut bench,
            "rr",
            Reconstruction::default(),
            frames,
            &reference,
            &images,
        ));
    }

    println!(
        "{:<8} {:>7} {:>8} {:>8} {:>8} {:>13} {:>7} {:>7} {:>7} {:>7} {:>6}",
        "",
        "noise",
        "flicker",
        "err",
        "err mov",
        "bias tiles",
        "bias",
        "corr1",
        "corr8",
        "corr t1",
        "lag",
    );
    for row in &rows {
        print_row(row);
    }
    let report = json!({
        "rev": rev,
        "extent": extent,
        "frames": frames,
        "reference_frames": reference_frames,
        "rows": rows,
    });
    let path = report_dir().join(format!("quality-{rev}.json"));
    let text = serde_json::to_string_pretty(&report).expect("report") + "\n";
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    println!(
        "wrote {} and images in {}",
        path.display(),
        images.display()
    );
    drop(bench);
    device().collect_garbage();
    assert_clean();
}
