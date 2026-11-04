//! The disk pipeline cache: warm loads, files from another driver or device, files the driver
//! refuses and creation from several threads.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use gpu_suite::shaders::{cached, cached_rays};
use gpu_suite::{log_to_stderr, require};
use orr_gpu::{
    ClearColor, ColorAttachment, ColorLoad, ColorTarget, ComputePipeline, Cull, Device, DeviceDesc,
    Error, Format, Hazards, MemoryKind, MeshPipeline, MeshPipelineDesc, QueueKind, RasterState,
    RayDispatchPipeline, Rendering, ShaderStage, SpecData, Stages, TextureDesc, TextureMemory,
    TextureRegion, TextureUsage,
};

fn cache_path(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr_gpu_suite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{test}.bin"));
    let _ = std::fs::remove_file(&path);
    path
}

fn open(path: &Path) -> Device {
    log_to_stderr();
    Device::new(&DeviceDesc {
        name: "gpu_suite pipeline cache",
        pipeline_cache: Some(path),
        ..DeviceDesc::default()
    })
    .unwrap()
}

/// One pipeline of each kind, all specialized on `value`.
struct Set {
    value: u32,
    compute: ComputePipeline,
    mesh: MeshPipeline,
    raygen: Option<RayDispatchPipeline>,
    /// Whether the compute, mesh and ray-dispatch pipelines came from the cache, when created
    /// alone; the last is true on a device without ray dispatch, and where a hit is not known to
    /// be reported for one: only NVIDIA's driver has been measured.
    cached: [bool; 3],
}

impl Set {
    fn new(d: &Device, value: u32) -> Set {
        let spec = cached::Spec { value }.bytes();
        let data = SpecData {
            entries: cached::Spec::ENTRIES,
            bytes: &spec,
        };
        let stage = |e| ShaderStage::new(&cached::MODULE, e, data).unwrap();
        let hits = || d.pipeline_cache_report().hits;
        let before = hits();
        let compute = d
            .create_compute_pipeline(&stage("compute_main"), "cached compute")
            .unwrap();
        let after_compute = hits();
        let mesh = d
            .create_mesh_pipeline(
                &MeshPipelineDesc {
                    pixel: Some(stage("pixel_main")),
                    colors: &[ColorTarget::new(Format::R32Uint)],
                    ..MeshPipelineDesc::new(stage("mesh_main"))
                },
                "cached mesh",
            )
            .unwrap();
        let after_mesh = hits();
        let raygen = d.caps().ray_dispatch.then(|| {
            let spec = cached_rays::Spec { value }.bytes();
            let data = SpecData {
                entries: cached_rays::Spec::ENTRIES,
                bytes: &spec,
            };
            let stage = ShaderStage::new(&cached_rays::MODULE, "raygen_main", data).unwrap();
            d.create_ray_dispatch_pipeline(&stage, "cached raygen")
                .unwrap()
        });
        let cached = [
            after_compute > before,
            after_mesh > after_compute,
            raygen.is_none() || !d.caps().device_name.contains("NVIDIA") || hits() > after_mesh,
        ];
        Set {
            value,
            compute,
            mesh,
            raygen,
            cached,
        }
    }

    fn kinds(&self) -> u32 {
        2 + u32::from(self.raygen.is_some())
    }

    /// Runs every pipeline and checks what it wrote.
    fn check(&self, d: &Device) {
        let mut out = d
            .gpu_malloc::<u32>(MemoryKind::Readback, 2, "cached out")
            .unwrap();
        out.as_mut_slice().unwrap().fill(0);
        let mut root = d
            .gpu_malloc::<cached::Root>(MemoryKind::Upload, 1, "cached root")
            .unwrap();
        root.as_mut_slice().unwrap()[0] = cached::Root {
            out: out.gpu(),
            pad0: 0,
            pad1: 0,
        };
        let desc = TextureDesc::d2(Format::R32Uint, 4, 4, TextureUsage::COLOR);
        let tex = d
            .create_texture(&desc, TextureMemory::Dedicated, "cached target")
            .unwrap();
        let texels = d
            .gpu_malloc::<u32>(MemoryKind::Readback, 16, "cached texels")
            .unwrap();
        let queue = d.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.discard(&tex);
        list.dispatch(&self.compute, root.gpu(), [1, 1, 1]);
        if let Some(r) = &self.raygen {
            let rays_root = cached_rays::Root {
                out: out.gpu(),
                pad0: 0,
                pad1: 0,
            };
            list.dispatch_rays(r, &rays_root, [1, 1, 1]);
        }
        list.begin_rendering(&Rendering {
            extent: [4, 4],
            colors: &[ColorAttachment {
                texture: &tex,
                load: ColorLoad::Clear(ClearColor::Uint([0; 4])),
                store: true,
            }],
            depth: None,
        });
        list.draw_mesh_tasks(&self.mesh, root.gpu(), [1, 1, 1]);
        list.end_rendering();
        list.barrier(Stages::COLOR_OUT, Stages::TRANSFER, Hazards::empty());
        list.copy_from_texture(
            texels.gpu().cast(),
            &tex,
            &TextureRegion::whole_mip(&desc, 0),
        );
        list.barrier(Stages::ALL, Stages::HOST, Hazards::empty());
        let point = queue.submit(vec![list], &[]).unwrap();
        queue.wait(point, Duration::from_secs(10)).unwrap();
        let v = self.value;
        let ray = if self.raygen.is_some() { v + 1 } else { 0 };
        assert_eq!(out.as_slice().unwrap(), [v, ray]);
        assert!(texels.as_slice().unwrap().iter().all(|&t| t == v + 2));
    }
}

/// Rewrites a little-endian u32 in the file header and fixes the trailing checksum, as a
/// file written by another driver or device would carry.
fn rewrite_header(path: &Path, offset: usize, f: impl FnOnce(u32) -> u32) {
    let mut bytes = std::fs::read(path).unwrap();
    let field = &mut bytes[offset..offset + 4];
    let v = f(u32::from_le_bytes(field.try_into().unwrap()));
    field.copy_from_slice(&v.to_le_bytes());
    write_checksummed(path, bytes);
}

/// Writes a cache file after fixing its trailing checksum.
fn write_checksummed(path: &Path, mut bytes: Vec<u8>) {
    let body = bytes.len() - CHECKSUM;
    let sum = blake3::hash(&bytes[..body]);
    bytes[body..].copy_from_slice(sum.as_bytes());
    std::fs::write(path, bytes).unwrap();
}

/// Header offsets of the device id and driver version.
const DEVICE_ID: usize = 16;
const DRIVER_VERSION: usize = 20;
/// Bytes before the cache data: magic, format, vendor, device, driver version, cache UUID.
const HEADER: usize = 24 + 16;
const CHECKSUM: usize = 32;

fn assert_clean(d: &Device) {
    assert_eq!(
        d.validation_messages(),
        0,
        "validation messages (see the log)"
    );
}

/// A handle that only takes cached pipelines refuses a pipeline no cache holds, without compiling
/// or counting a miss, and takes it once it is compiled. The spec value is new each run, since the
/// driver's disk cache outlives the process.
#[test]
fn a_cached_only_handle_compiles_nothing() {
    let path = &cache_path("cached_only");
    let d = open(path);
    let value = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let spec = cached::Spec { value }.bytes();
    let data = SpecData {
        entries: cached::Spec::ENTRIES,
        bytes: &spec,
    };
    let stage = ShaderStage::new(&cached::MODULE, "compute_main", data).unwrap();
    let only = d.cached_pipelines_only();
    assert!(matches!(
        only.create_compute_pipeline(&stage, "cached compute"),
        Err(Error::CompileRequired)
    ));
    assert_eq!(d.pipeline_cache_report().misses, 0);
    d.create_compute_pipeline(&stage, "cached compute").unwrap();
    only.create_compute_pipeline(&stage, "cached compute")
        .unwrap();
    if d.caps().ray_dispatch {
        let spec = cached_rays::Spec { value }.bytes();
        let data = SpecData {
            entries: cached_rays::Spec::ENTRIES,
            bytes: &spec,
        };
        let stage = ShaderStage::new(&cached_rays::MODULE, "raygen_main", data).unwrap();
        assert!(matches!(
            only.create_ray_dispatch_pipeline(&stage, "cached raygen"),
            Err(Error::CompileRequired)
        ));
        d.create_ray_dispatch_pipeline(&stage, "cached raygen")
            .unwrap();
        only.create_ray_dispatch_pipeline(&stage, "cached raygen")
            .unwrap();
    }
    assert_clean(&d);
}

/// Writes a cache, loads it warm, then loads it as another driver version and as another
/// device: both are refused and every pipeline is compiled again.
#[test]
fn round_trip_and_refuse_foreign_files() {
    let path = &cache_path("round_trip");
    let d = open(path);
    assert!(!d.pipeline_cache_report().loaded);
    let set = Set::new(&d, 1);
    let n = set.kinds();
    assert_eq!(d.pipeline_cache_report().misses, n);
    set.check(&d);
    d.save_pipeline_cache().unwrap();
    assert_clean(&d);
    drop((set, d));

    let d = open(path);
    assert!(d.pipeline_cache_report().loaded);
    let set = Set::new(&d, 1);
    assert_eq!(
        set.cached,
        [true, true, true],
        "warm: {:?}",
        d.pipeline_cache_report()
    );
    set.check(&d);
    assert_clean(&d);
    drop((set, d));

    for (offset, what) in [(DRIVER_VERSION, "driver version"), (DEVICE_ID, "device")] {
        let saved = std::fs::read(path).unwrap();
        rewrite_header(path, offset, |v| v ^ 1);
        let d = open(path);
        let report = d.pipeline_cache_report();
        assert!(
            report.rejected.as_deref().is_some_and(|r| r.contains(what)),
            "{what}: {report:?}"
        );
        assert!(!report.loaded);
        let set = Set::new(&d, 1);
        let report = d.pipeline_cache_report();
        assert_eq!((report.hits, report.misses), (0, n), "{what}: {report:?}");
        set.check(&d);
        assert_clean(&d);
        drop((set, d));
        std::fs::write(path, saved).unwrap();
    }
}

/// A second save in one session keeps what the first saved: both sets load warm.
#[test]
fn a_second_save_keeps_the_first() {
    let path = &cache_path("second_save");
    let d = open(path);
    let first = Set::new(&d, 3);
    d.save_pipeline_cache().unwrap();
    let second = Set::new(&d, 4);
    d.save_pipeline_cache().unwrap();
    assert_clean(&d);
    drop((first, second, d));

    let d = open(path);
    for value in [3, 4] {
        let set = Set::new(&d, value);
        assert_eq!(
            set.cached,
            [true, true, true],
            "set {value}: {:?}",
            d.pipeline_cache_report()
        );
        set.check(&d);
    }
    assert_clean(&d);
}

/// A file with good headers and a good checksum whose cache data the driver refuses is removed:
/// the device starts empty, and the run after the next save hits. Valid usage does not allow data
/// like this, so the test runs only where the driver is known to return an error for it.
#[test]
fn a_file_the_driver_refuses_is_replaced() {
    let path = &cache_path("refused");
    let d = open(path);
    if !require(
        d.caps().device_name.contains("NVIDIA"),
        "only NVIDIA's driver is known to refuse bad cache data",
    ) {
        return;
    }
    let set = Set::new(&d, 1);
    d.save_pipeline_cache().unwrap();
    drop((set, d));

    let mut bytes = std::fs::read(path).unwrap();
    let body = bytes.len() - CHECKSUM;
    let driver_header = u32::from_le_bytes(bytes[HEADER..HEADER + 4].try_into().unwrap()) as usize;
    for (i, byte) in bytes[HEADER + driver_header..body].iter_mut().enumerate() {
        *byte = (i * 31) as u8;
    }
    write_checksummed(path, bytes);

    let d = open(path);
    let report = d.pipeline_cache_report();
    assert!(!report.loaded);
    assert!(
        report
            .rejected
            .as_deref()
            .is_some_and(|r| r.contains("refused")),
        "{report:?}"
    );
    assert!(!path.exists());
    let set = Set::new(&d, 1);
    set.check(&d);
    d.save_pipeline_cache().unwrap();
    assert_clean(&d);
    drop((set, d));

    let d = open(path);
    assert!(d.pipeline_cache_report().loaded);
    let set = Set::new(&d, 1);
    assert_eq!(
        set.cached,
        [true, true, true],
        "after replacing: {:?}",
        d.pipeline_cache_report()
    );
    set.check(&d);
    assert_clean(&d);
}

/// A file with good headers whose cache data names another device in the driver's own header is
/// not reported as loaded, and the next save replaces it. A driver may ignore such data or
/// refuse it.
#[test]
fn a_file_the_driver_ignores_is_replaced() {
    let path = &cache_path("ignored");
    let d = open(path);
    let set = Set::new(&d, 1);
    let n = set.kinds();
    d.save_pipeline_cache().unwrap();
    drop((set, d));

    let mut bytes = std::fs::read(path).unwrap();
    // The cache UUID in VkPipelineCacheHeaderVersionOne.
    bytes[HEADER + 16] ^= 1;
    write_checksummed(path, bytes);

    let d = open(path);
    let report = d.pipeline_cache_report();
    assert!(!report.loaded, "{report:?}");
    assert!(report.rejected.is_some(), "{report:?}");
    let set = Set::new(&d, 1);
    let report = d.pipeline_cache_report();
    assert_eq!((report.hits, report.misses), (0, n), "{report:?}");
    set.check(&d);
    d.save_pipeline_cache().unwrap();
    assert_clean(&d);
    drop((set, d));

    let d = open(path);
    assert!(d.pipeline_cache_report().loaded);
}

/// Threads asking for one pipeline at once compile it once.
#[test]
fn one_pipeline_requested_at_once_compiles_once() {
    let path = cache_path("at_once");
    let d = open(&path);
    let spec = cached::Spec { value: 77 }.bytes();
    let data = SpecData {
        entries: cached::Spec::ENTRIES,
        bytes: &spec,
    };
    let stage = ShaderStage::new(&cached::MODULE, "compute_main", data).unwrap();
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| d.create_compute_pipeline(&stage, "at once").unwrap());
        }
    });
    let report = d.pipeline_cache_report();
    assert_eq!((report.misses, report.hits), (1, 7), "{report:?}");
    assert_clean(&d);
}

/// Threads asking for mesh pipelines with one set of stages and two raster states compile each
/// state once. Only NVIDIA's driver is known to keep the states apart in its cache.
#[test]
fn mesh_pipelines_that_differ_in_state_compile_apart() {
    let d = open(&cache_path("mesh_state"));
    if !require(
        d.caps().device_name.contains("NVIDIA"),
        "only NVIDIA's driver is known to cache each cull mode apart",
    ) {
        return;
    }
    let spec = cached::Spec { value: 88 }.bytes();
    let data = SpecData {
        entries: cached::Spec::ENTRIES,
        bytes: &spec,
    };
    let stage = |e| ShaderStage::new(&cached::MODULE, e, data).unwrap();
    let colors = [ColorTarget::new(Format::R32Uint)];
    std::thread::scope(|s| {
        for cull in [Cull::None, Cull::Back] {
            let desc = MeshPipelineDesc {
                pixel: Some(stage("pixel_main")),
                colors: &colors,
                raster: RasterState {
                    cull,
                    ..RasterState::default()
                },
                ..MeshPipelineDesc::new(stage("mesh_main"))
            };
            for _ in 0..4 {
                let d = &d;
                s.spawn(move || d.create_mesh_pipeline(&desc, "mesh state").unwrap());
            }
        }
    });
    let report = d.pipeline_cache_report();
    assert_eq!((report.misses, report.hits), (2, 6), "{report:?}");
    assert_clean(&d);
}

#[test]
fn concurrent_creation() {
    let path = cache_path("concurrent");
    for warm in [false, true] {
        let d = open(&path);
        // Threads 0 and 1 compile the same first set as threads 2 and 3.
        let sets: Vec<(Set, Set)> = std::thread::scope(|s| {
            let threads: Vec<_> = (0..4u32)
                .map(|t| {
                    let d = &d;
                    s.spawn(move || (Set::new(d, t % 2), Set::new(d, 10 + t)))
                })
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        for (a, b) in &sets {
            a.check(&d);
            b.check(&d);
        }
        let report = d.pipeline_cache_report();
        let created = 8 * sets[0].0.kinds();
        assert_eq!(report.hits + report.misses, created);
        if warm {
            assert!(report.hits >= 16, "compute and mesh warm: {report:?}");
        }
        d.save_pipeline_cache().unwrap();
        assert_clean(&d);
    }
}
