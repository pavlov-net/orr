//! Code orr's examples share: the device, a window, command-line options and headless
//! captures, and the examples' kernels.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(dlss)]
use orr::DlssSdk;
use orr::orr_gpu::{Allocation, Device, DeviceDesc, QueueRequest};
#[cfg(dlss)]
use orr::orr_output::orr_dlss::Uuid;
use orr::orr_pack::{Archive, AssetId};
use orr::{
    FrameDesc, FrameTime, Lens, LoadState, Loading, MountId, PackSource, Renderer, Screenshot,
};

pub mod average;
pub mod creature;
pub mod field;
pub mod mesh;
pub mod overlay;
pub mod quality;
pub mod scene;
pub mod sun_shade;
pub mod vegetation;
pub mod window;

/// The examples' compiled kernels and the Rust types of their Slang structs.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// orr's kernels with `AtmosphereSky` linked as the environment and `AtmosphereMedium` as the
/// medium.
pub mod linked {
    include!(concat!(env!("OUT_DIR"), "/linked/orr_kernels.rs"));
}

/// The examples' NGX project id.
#[cfg(dlss)]
pub const PROJECT_ID: Uuid = Uuid::from_u128(0x5d1e_7a40_c2b9_4f86_9e3d_61a0_b8f2_47c5);

/// Asks [`device`] for counter frames (`DeviceDesc::perf_counters`); call before its first use.
pub fn request_perf_counters() {
    PERF_COUNTERS.store(true, Ordering::Relaxed);
}

static PERF_COUNTERS: AtomicBool = AtomicBool::new(false);

/// The process's device, created on first use with every queue, NGX's extensions where
/// NGX offers them, and validation when the environment enables the layer. Logs info and
/// above to stderr, with `RUST_LOG` adding to or overriding that.
pub fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        let _ = env_logger::Builder::new()
            .parse_filters("info")
            .parse_default_env()
            .is_test(true)
            .try_init();
        #[cfg(dlss)]
        let hooks = DlssSdk::device_hooks(PROJECT_ID);
        // The test runner sets it, so the screenshot tests share compiled pipelines.
        let store = std::env::var_os("ORR_TEST_PIPELINE_STORE").map(PathBuf::from);
        let desc = DeviceDesc {
            name: "orr-examples",
            pipeline_store: store.as_deref(),
            #[cfg(dlss)]
            hooks: Some(&hooks),
            queues: QueueRequest {
                render_compute: true,
                async_compute: true,
                copy: true,
                present: true,
            },
            perf_counters: PERF_COUNTERS.load(Ordering::Relaxed),
            ..DeviceDesc::default()
        };
        Device::new(&desc).expect("device")
    })
}

/// Gives `renderer` DLSS ray reconstruction where NGX offers it; returns whether it does.
#[cfg(not(dlss))]
pub fn enable_dlss(_renderer: &mut Renderer) -> bool {
    false
}

/// Gives `renderer` DLSS ray reconstruction where NGX offers it; returns whether it does.
#[cfg(dlss)]
pub fn enable_dlss(renderer: &mut Renderer) -> bool {
    match DlssSdk::new(device(), PROJECT_ID) {
        Ok(sdk) => renderer.set_reconstruction(sdk),
        Err(e) => {
            log::warn!("no DLSS: {e}");
            false
        }
    }
}

/// Panics if the validation layer has reported anything.
pub fn assert_clean() {
    let messages = device().validation_messages();
    assert_eq!(messages, 0, "validation messages (see the log)");
}

/// The lens effects a run can compare by eye: `--no-glare` and `--no-motion-blur` turn them off
/// from the start, the `G` and `B` keys switch them while it runs.
#[derive(Debug)]
pub struct LensSwitches {
    glare: bool,
    motion_blur: bool,
}

impl LensSwitches {
    pub fn new(options: &Options) -> Self {
        Self {
            glare: !options.has("--no-glare"),
            motion_blur: !options.has("--no-motion-blur"),
        }
    }

    /// Switches the effect `key` belongs to, if any, and logs it.
    pub fn press(&mut self, key: window::KeyCode) {
        match key {
            window::KeyCode::KeyG => self.glare = !self.glare,
            window::KeyCode::KeyB => self.motion_blur = !self.motion_blur,
            _ => return,
        }
        log::info!("{self}");
    }

    /// The default lens with the switched-off effects removed.
    pub fn lens(&self) -> Lens {
        let mut lens = Lens::default();
        if !self.glare {
            lens.glare = None;
        }
        if !self.motion_blur {
            lens.shutter = 0.0;
        }
        lens
    }
}

impl std::fmt::Display for LensSwitches {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let on = |b| if b { "on" } else { "off" };
        write!(
            f,
            "glare {}, motion blur {}",
            on(self.glare),
            on(self.motion_blur)
        )
    }
}

/// `git rev-parse --short HEAD` in the repository, or "unknown".
pub fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".into(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        )
}

/// Where examples write their reports, made if needed.
pub fn report_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("target")
        .join("orr-reports");
    std::fs::create_dir_all(&dir).expect("report directory");
    dir
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

    /// Waits for the path tracing kernels when the run counts or captures frames, so each is lit.
    /// A window left to run stays live while they compile.
    pub fn wait_for_paths(&self, renderer: &mut Renderer) {
        if self.headless || self.frames.is_some() || self.out.is_some() {
            renderer.wait_paths().expect("path kernels");
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
    /// A capture of packed RGBA8 words read back from an image.
    pub fn from_readback(extent: [u32; 2], words: &Allocation<u32>) -> Self {
        Capture {
            width: extent[0],
            height: extent[1],
            rgba8: bytemuck::cast_slice(words.as_slice().expect("readback")).to_vec(),
        }
    }

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
    let (mounts, frame) = mount_all_at(renderer, vec![source], first_frame);
    (mounts[0], frame)
}

/// [`mount_at`] for several sources at once.
pub fn mount_all_at(
    renderer: &mut Renderer,
    sources: Vec<PackSource>,
    first_frame: u64,
) -> (Vec<MountId>, u64) {
    let mounts: Vec<MountId> = sources.into_iter().map(|s| renderer.mount(s)).collect();
    let pending = |r: &Renderer| {
        mounts
            .iter()
            .any(|&m| matches!(r.mount_state(m), LoadState::Pending))
    };
    let mut frame = 0;
    let start = Instant::now();
    while pending(renderer) || frame < first_frame {
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
        if pending(renderer) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    if frame > first_frame {
        log::warn!("mounting took {frame} frames; this run will not repeat others exactly");
    }
    for &m in &mounts {
        if let LoadState::Failed(e) = renderer.mount_state(m) {
            panic!("mount failed: {e}");
        }
    }
    (mounts, frame)
}

/// Blob 0 of entry `id` in `archive`, decoded: the header a CPU reader parses.
pub fn entry_head(archive: &Archive, id: AssetId) -> Option<Vec<u8>> {
    let b = archive.blobs(archive.find(id)?).first()?;
    let mut bytes = vec![0; b.raw_len as usize];
    archive.read(b, &mut bytes).ok()?;
    Some(bytes)
}

/// How long dropping [`Workers`] waits for them, which covers the jobs that take milliseconds.
const WORKER_STOP_WAIT: Duration = Duration::from_millis(100);

/// Threads lent to orr's loading side until dropped.
pub struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
    /// Disconnects when every thread has ended; nothing is sent on it.
    ended: Receiver<()>,
}

impl Workers {
    pub fn new(loading: Arc<Loading>, count: usize) -> Self {
        let (running, ended) = mpsc::channel();
        let threads = (0..count)
            .map(|_| {
                let l = Arc::clone(&loading);
                let running = running.clone();
                std::thread::spawn(move || {
                    l.run_worker();
                    drop(running);
                })
            })
            .collect();
        Workers {
            loading,
            threads,
            ended,
        }
    }
}

impl Drop for Workers {
    /// Stops the workers. One still in a job after [`WORKER_STOP_WAIT`] finishes it on its own or
    /// dies with the process: a pipeline compile cannot be interrupted, and closing a window
    /// should not wait minutes for one the process no longer needs.
    fn drop(&mut self) {
        self.loading.stop_workers();
        let all_ended = matches!(
            self.ended.recv_timeout(WORKER_STOP_WAIT),
            Err(RecvTimeoutError::Disconnected)
        );
        for t in self
            .threads
            .drain(..)
            .filter(|t| all_ended || t.is_finished())
        {
            t.join().expect("loading worker");
        }
    }
}
