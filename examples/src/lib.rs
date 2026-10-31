//! Code orr's examples share: the device, a window, command-line options and headless
//! captures, and the examples' kernels.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orr::orr_gpu::{Device, DeviceDesc, QueueRequest};
use orr::orr_output::orr_dlss::Uuid;
use orr::{
    Affine3, DlssSdk, FrameDesc, FrameTime, LoadState, Loading, MountId, PackSource, Renderer,
    Screenshot,
};
use orr_math::{Mat3, Vec3};

pub mod field;
pub mod mesh;
pub mod overlay;
pub mod scene;
pub mod sky;
pub mod vegetation;
pub mod window;

/// The examples' compiled kernels and the Rust types of their Slang structs.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Info
    }

    #[allow(clippy::print_stderr)]
    fn log(&self, r: &log::Record<'_>) {
        if self.enabled(r.metadata()) {
            eprintln!("{} {}", r.level(), r.args());
        }
    }

    fn flush(&self) {}
}

/// The examples' NGX project id.
pub const PROJECT_ID: Uuid = Uuid::from_u128(0x5d1e_7a40_c2b9_4f86_9e3d_61a0_b8f2_47c5);

/// The process's device, created on first use with every queue, NGX's extensions where
/// NGX offers them, and the validation messenger on (the layer itself loads through
/// `VK_INSTANCE_LAYERS`). Logs to stderr.
pub fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        if log::set_logger(&StderrLog).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
        let hooks = DlssSdk::device_hooks(PROJECT_ID);
        let desc = DeviceDesc {
            name: "orr-examples",
            hooks: Some(&hooks),
            validation: true,
            queues: QueueRequest {
                render_compute: true,
                async_compute: true,
                copy: true,
                present: true,
            },
            ..DeviceDesc::default()
        };
        Device::new(&desc).expect("device")
    })
}

/// Panics if the validation layer has reported anything.
pub fn assert_clean() {
    let messages = device().validation_messages();
    assert_eq!(messages, 0, "validation messages (see the log)");
}

/// How an example runs.
#[derive(Clone, Debug)]
pub struct Options {
    /// Frames to render; `None` runs until the window closes.
    pub frames: Option<u64>,
    /// No window: render offscreen and capture the last frame.
    pub headless: bool,
    /// Where `main` writes the headless capture.
    pub out: Option<PathBuf>,
    /// Arguments the example reads itself.
    pub rest: Vec<String>,
}

impl Options {
    /// Reads `[--headless] [--frames N] [--out file.png] [other...]` from the command line.
    pub fn from_args() -> Self {
        let mut o = Options {
            frames: None,
            headless: false,
            out: None,
            rest: Vec::new(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--headless" => o.headless = true,
                "--frames" => {
                    o.frames = Some(
                        args.next()
                            .and_then(|n| n.parse().ok())
                            .expect("--frames takes a count"),
                    );
                }
                "--out" => o.out = Some(args.next().expect("--out takes a path").into()),
                _ => o.rest.push(a),
            }
        }
        o
    }

    /// A headless run of `frames` frames with `rest` as the example's own arguments.
    pub fn headless(frames: u64, rest: &[&str]) -> Self {
        Options {
            frames: Some(frames),
            headless: true,
            out: None,
            rest: rest.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// Whether `flag` is one of the example's own arguments.
    pub fn has(&self, flag: &str) -> bool {
        self.rest.iter().any(|a| a == flag)
    }

    /// `out`, or `<name>.png` beside the executable.
    pub fn out_path(&self, name: &str) -> PathBuf {
        self.out.clone().unwrap_or_else(|| {
            std::env::current_exe()
                .expect("executable path")
                .with_file_name(format!("{name}.png"))
        })
    }
}

/// An 8-bit sRGB RGBA image, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Capture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl Capture {
    /// A display screenshot (linear, relative to paper white) encoded as sRGB.
    pub fn from_screenshot(shot: &Screenshot) -> Self {
        let rgba8 = shot
            .rgba
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let v = f64::from(v.clamp(0.0, 1.0));
                let v = if k % 4 == 3 {
                    v
                } else {
                    orr_color::srgb_encode(v)
                };
                (v * 255.0).round() as u8
            })
            .collect();
        Capture {
            width: shot.size[0],
            height: shot.size[1],
            rgba8,
        }
    }

    pub fn save_png(&self, path: &Path) {
        let file = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("creating {}: {e}", path.display()));
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), self.width, self.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .and_then(|mut w| w.write_image_data(&self.rgba8))
            .unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    }
}

/// Rows [R | t] of rotation `m` and translation `t`.
pub fn affine(m: Mat3, t: Vec3) -> Affine3 {
    let r = |i: usize| {
        let row = m.row(i);
        [row.x, row.y, row.z, t[i]]
    };
    Affine3 {
        rows: [r(0), r(1), r(2)],
    }
}

/// Writes an 8-bit grayscale PNG of `f` over an `n` x `n` grid of unit coordinates to
/// `path`, for the cook to read.
pub fn gray_png(path: &Path, n: u32, f: impl Fn(f32, f32) -> u8) {
    let data: Vec<u8> = (0..n * n)
        .map(|i| f((i % n) as f32 / n as f32, (i / n) as f32 / n as f32))
        .collect();
    let file =
        std::fs::File::create(path).unwrap_or_else(|e| panic!("creating {}: {e}", path.display()));
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), n, n);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(&data))
        .unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
}

/// Frame `n` of a fixed 60 Hz clock, so headless runs repeat exactly.
pub fn fixed_time(n: u64) -> FrameTime {
    FrameTime {
        absolute: n as f64 / 60.0,
        delta: 1.0 / 60.0,
    }
}

/// Mounts `source` and renders empty frames until it is mounted and the frame id reaches
/// `first_frame`, so the frames after it, whose light sampling follows the frame id, repeat
/// across runs. Returns the mount and the last frame id. Panics if the mount fails or takes
/// longer than a minute.
pub fn mount_at(renderer: &mut Renderer, source: PackSource, first_frame: u64) -> (MountId, u64) {
    let mount = renderer.mount(source);
    let mut frame = 0;
    let start = Instant::now();
    while matches!(renderer.mount_state(mount), LoadState::Pending) || frame < first_frame {
        frame += 1;
        let f = renderer
            .begin_frame(FrameDesc {
                time: fixed_time(frame),
                scenes: &[],
                display: None,
            })
            .expect("begin frame");
        f.end().expect("end frame");
        assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
        if matches!(renderer.mount_state(mount), LoadState::Pending) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    if frame > first_frame {
        log::warn!("mounting took {frame} frames; this run will not repeat others exactly");
    }
    if let LoadState::Failed(e) = renderer.mount_state(mount) {
        panic!("mount failed: {e}");
    }
    (mount, frame)
}

/// Threads lent to orr's loading side until dropped.
pub struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    pub fn new(loading: Arc<Loading>, count: usize) -> Self {
        let threads = (0..count)
            .map(|_| {
                let l = Arc::clone(&loading);
                std::thread::spawn(move || l.run_worker())
            })
            .collect();
        Workers { loading, threads }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.loading.stop_workers();
        for t in self.threads.drain(..) {
            t.join().expect("loading worker");
        }
    }
}
