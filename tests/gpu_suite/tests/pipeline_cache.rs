//! The disk pipeline cache: warm loads, files from another driver or device, files the driver
//! refuses and creation from several threads.

use std::path::{Path, PathBuf};
use std::time::Duration;

use gpu_suite::shaders::cached;
use gpu_suite::{log_to_stderr, require};
use orr_gpu::{
    ClearColor, ColorAttachment, ColorLoad, ColorTarget, ComputePipeline, Device, DeviceDesc,
    Format, Hazards, MemoryKind, MeshPipeline, MeshPipelineDesc, QueueKind, RayDispatchPipeline,
    Rendering, ShaderStage, SpecData, Stages, TextureDesc, TextureMemory, TextureRegion,
    TextureUsage,
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
    /// Whether the compute and the mesh pipeline came from the cache, when created alone.
    cached: [bool; 2],
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
        let cached = [after_compute > before, hits() > after_compute];
        let raygen = d.caps().ray_dispatch.then(|| {
            d.create_ray_dispatch_pipeline(&stage("raygen_main"), "cached raygen")
                .unwrap()
        });
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
            list.dispatch_rays(r, root.gpu(), [1, 1, 1]);
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

    // 621.01 reports cache hits for compute and mesh pipelines but never for ray-dispatch
    // pipelines, so only the first two are asserted.
    let d = open(path);
    assert!(d.pipeline_cache_report().loaded);
    let set = Set::new(&d, 1);
    assert_eq!(
        set.cached,
        [true, true],
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
        [true, true],
        "after replacing: {:?}",
        d.pipeline_cache_report()
    );
    set.check(&d);
    assert_clean(&d);
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
            assert!(
                report.hits >= 16,
                "compute and mesh pipelines warm: {report:?}"
            );
        }
        d.save_pipeline_cache().unwrap();
        assert_clean(&d);
    }
}
