//! Code orr's examples share: the device, a window, command-line options and headless
//! captures, and the examples' kernels.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;

use orr::orr_gpu::{Device, DeviceDesc, QueueRequest};
use orr::orr_output::orr_dlss::Uuid;
use orr::{DlssSdk, FrameTime, Loading};

pub mod field;
pub mod mesh;
pub mod overlay;
pub mod scene;
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

/// Frame `n` of a fixed 60 Hz clock, so headless runs repeat exactly.
pub fn fixed_time(n: u64) -> FrameTime {
    FrameTime {
        absolute: n as f64 / 60.0,
        delta: 1.0 / 60.0,
    }
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
