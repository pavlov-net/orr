//! GPU tests of orr-gpu. Every test runs on one shared device with validation on and fails
//! if the layer reports anything.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

use orr_gpu::{
    Allocation, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance,
    ClassicKind, CommandList, ComputePipeline, Device, DeviceDesc, Diagnostics, GpuPtr,
    InstanceFlags, LatencyTimings, MemoryKind, OutputFormat, Queue, QueueKind, QueueRequest,
    RtasFlags, ShaderModule, ShaderStage, SpecData, SwapchainFrame, TimelinePoint,
};
use orr_shaders::bytemuck::Pod;

pub mod clusters;
#[cfg(windows)]
pub mod window;

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// Kernels compiled with `-fp-mode precise`, whose raygen and compute entries must agree.
pub mod parity {
    include!(concat!(env!("OUT_DIR"), "/parity/orr_shaders.rs"));
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

/// Logs info and above to stderr.
pub fn log_to_stderr() {
    let _ = log::set_logger(&StderrLog);
    log::set_max_level(log::LevelFilter::Info);
}

static DEVICE: OnceLock<Device> = OnceLock::new();

/// The suite's device: every queue kind, validation on. `ORR_CAPTURE=1` creates it for an
/// Nsight Graphics capture ([`Diagnostics::Capture`]).
pub fn device() -> &'static Device {
    let diagnostics = if std::env::var_os("ORR_CAPTURE").is_some_and(|v| v == "1") {
        Diagnostics::Capture
    } else {
        Diagnostics::Off
    };
    device_with(diagnostics)
}

/// The suite's device created with `diagnostics`, for test binaries that need them; the
/// first call in a process decides.
pub fn device_with(diagnostics: Diagnostics<'static>) -> &'static Device {
    DEVICE.get_or_init(|| {
        log_to_stderr();
        let desc = DeviceDesc {
            name: "gpu-suite",
            validation: true,
            diagnostics,
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

/// Submits `list` on the render queue and waits for it.
pub fn run(list: CommandList<'_>) -> TimelinePoint {
    let queue = device().queue(QueueKind::Render).unwrap();
    let point = queue.submit(vec![list], &[]).expect("submit");
    queue.wait(point, Duration::from_secs(10)).expect("wait");
    point
}

/// `cond`; when false, logs one line saying the calling test is skipped because of `why`.
pub fn require(cond: bool, why: &str) -> bool {
    if !cond {
        log::warn!("skipped on {}: {why}", device().caps().device_name);
    }
    cond
}

/// Fails the calling test if the validation layer has reported anything.
pub fn assert_clean() {
    assert_eq!(
        device().validation_messages(),
        0,
        "validation messages (see the log)"
    );
}

/// Upload memory holding `values`.
pub fn upload<T: Pod>(values: &[T], name: &str) -> Allocation<T> {
    let mut a = device()
        .gpu_malloc::<T>(MemoryKind::Upload, values.len() as u64, name)
        .unwrap();
    a.as_mut_slice().unwrap().copy_from_slice(values);
    a
}

/// A deterministic stream of floats in [0, 1) seeded by `seed`.
pub fn noise(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Readback memory for `count` values.
pub fn readback<T: Pod>(count: u64, name: &str) -> Allocation<T> {
    device()
        .gpu_malloc::<T>(MemoryKind::Readback, count, name)
        .unwrap()
}

/// The compute pipeline of `module`'s `entry`.
pub fn pipeline(module: &'static ShaderModule, entry: &str) -> ComputePipeline {
    let stage = ShaderStage::new(module, entry, SpecData::default()).unwrap();
    device()
        .create_compute_pipeline(&stage, module.name)
        .unwrap()
}

/// The render queue.
pub fn render() -> &'static Queue {
    device().queue(QueueKind::Render).unwrap()
}

/// One triangle in the unit square, in a BLAS under one instance of a classic TLAS.
pub struct Scene {
    pub tlas: ClassicAs,
    _blas: ClassicAs,
    _blocks: Vec<Allocation<u8>>,
}

/// The custom index of the scene's one instance.
pub const CUSTOM_INDEX: u32 = 7;

/// Builds the scene on the render queue and waits for it.
pub fn triangle_scene() -> Scene {
    let d = device();
    let positions = upload(
        &[[0.1f32, 0.1, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
        "triangle",
    );
    let blas_desc = ClassicDesc {
        kind: ClassicKind::Blas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Triangles {
            positions: positions.gpu,
            vertex_count: 3,
            indices: GpuPtr::NULL,
            triangle_count: 1,
            opaque: true,
        },
    };
    let blas_sizes = d.classic_sizes(&blas_desc);
    let blas_block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, blas_sizes.result_max, "blas")
        .unwrap();
    let blas_scratch = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, blas_sizes.scratch, "blas scratch")
        .unwrap();
    let blas = d
        .create_classic_as(ClassicKind::Blas, blas_sizes.result_max, &blas_block, 0)
        .unwrap();
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let instance = ClassicInstance::new(
        identity,
        CUSTOM_INDEX,
        0xff,
        InstanceFlags::CULL_DISABLE,
        blas.address(),
    );
    let instances = upload(&[instance], "instances");
    let tlas_desc = ClassicDesc {
        kind: ClassicKind::Tlas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Instances {
            records: instances.gpu.address(),
            capacity: 1,
        },
    };
    let tlas_sizes = d.classic_sizes(&tlas_desc);
    let tlas_block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, tlas_sizes.result_max, "tlas")
        .unwrap();
    let tlas_scratch = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, tlas_sizes.scratch, "tlas scratch")
        .unwrap();
    let tlas = d
        .create_classic_as(ClassicKind::Tlas, tlas_sizes.result_max, &tlas_block, 0)
        .unwrap();
    let mut list = render().command_list().unwrap();
    list.build_classic(&ClassicBuild {
        desc: &blas_desc,
        dst: &blas,
        scratch: blas_scratch.gpu.address(),
    });
    list.build_classic(&ClassicBuild {
        desc: &tlas_desc,
        dst: &tlas,
        scratch: tlas_scratch.gpu.address(),
    });
    run(list);
    Scene {
        tlas,
        _blas: blas,
        _blocks: vec![blas_block, tlas_block],
    }
}

/// Records the swapchain test card into `frame`'s image, with its root in `device`'s upload
/// memory. Keep the root until the list is submitted.
pub fn fill_frame<'q>(
    device: &Device,
    queue: &'q Queue,
    fill: &ComputePipeline,
    frame: &SwapchainFrame,
    frame_no: u32,
) -> (CommandList<'q>, Allocation<shaders::swapchain_fill::Root>) {
    let [width, height] = frame.extent();
    let encoding = match frame.format() {
        OutputFormat::Sdr => 0,
        OutputFormat::Hdr10Pq => 1,
        OutputFormat::ScRgb => 2,
    };
    let mut root = device
        .gpu_malloc::<shaders::swapchain_fill::Root>(MemoryKind::Upload, 1, "swapchain fill root")
        .unwrap();
    root.as_mut_slice()
        .unwrap()
        .copy_from_slice(&[shaders::swapchain_fill::Root {
            target: frame.storage(),
            width,
            height,
            encoding,
            frame: frame_no,
            paper_white_nits: 200.0,
            pad0: 0,
            pad1: 0,
        }]);
    let mut list = queue.command_list().unwrap();
    list.dispatch(fill, root.gpu, [width.div_ceil(8), height.div_ceil(8), 1]);
    (list, root)
}

/// Asserts that every frame id up to `frames` has a latency report with every marker of the
/// render and present threads, in order, and a GPU span. The first frames run before the
/// driver's history fills and the last are in flight. 621.01 never reports ids that are
/// multiples of 128 (measured.md PR-10).
pub fn check_latency_reports(reports: &BTreeMap<u64, LatencyTimings>, frames: u64) {
    for id in (64..frames - 64).filter(|id| id % 128 != 0) {
        let t = reports
            .get(&id)
            .unwrap_or_else(|| panic!("no latency report for frame {id}"));
        let ordered = [
            t.simulation_start,
            t.simulation_end,
            t.render_submit_start,
            t.render_submit_end,
            t.present_start,
            t.present_end,
        ];
        assert!(
            ordered.iter().all(|&v| v != 0),
            "frame {id}: a marker is missing: {t:?}"
        );
        // Markers set within the same microsecond can read 1 us out of order.
        assert!(
            ordered.windows(2).all(|w| w[0] <= w[1] + 1),
            "frame {id}: markers out of order: {t:?}"
        );
        assert!(
            t.gpu_render_start != 0 && t.gpu_render_start <= t.gpu_render_end,
            "frame {id}: no GPU span: {t:?}"
        );
    }
}
