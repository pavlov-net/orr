//! The disk pipeline cache: warm loads, files from another driver or device, the
//! `VkPipelineCache` fallback and creation from several threads.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use gpu_suite::log_to_stderr;
use gpu_suite::shaders::cached;
use orr_gpu::{
    ClearColor, ColorAttachment, ColorLoad, ColorTarget, ComputePipeline, Device, DeviceDesc,
    Format, Hazards, MemoryKind, MeshPipeline, MeshPipelineDesc, PipelineCacheKind, QueueKind,
    RayDispatchPipeline, Rendering, ShaderStage, SpecData, Stages, TextureDesc, TextureMemory,
    TextureRegion, TextureUsage,
};

fn cache_path(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-gpu-suite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{test}.bin"));
    let _ = std::fs::remove_file(&path);
    path
}

fn open(path: &Path, skip: &[&CStr]) -> Device {
    log_to_stderr();
    Device::new(&DeviceDesc {
        name: "gpu-suite pipeline cache",
        validation: true,
        pipeline_cache: Some(path),
        skip_extensions: skip,
        ..DeviceDesc::default()
    })
    .unwrap()
}

/// One pipeline of each kind, all specialised on `value`.
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
            out: out.gpu,
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
        list.dispatch(&self.compute, root.gpu, [1, 1, 1]);
        if let Some(r) = &self.raygen {
            list.dispatch_rays(r, root.gpu, [1, 1, 1]);
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
        list.draw_mesh_tasks(&self.mesh, root.gpu, [1, 1, 1]);
        list.end_rendering();
        list.barrier(Stages::COLOR_OUT, Stages::TRANSFER, Hazards::empty());
        list.copy_from_texture(texels.gpu.cast(), &tex, &TextureRegion::whole_mip(&desc, 0));
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
    let body = bytes.len() - 32;
    let sum = blake3::hash(&bytes[..body]);
    bytes[body..].copy_from_slice(sum.as_bytes());
    std::fs::write(path, bytes).unwrap();
}

/// Header offsets of the device id and driver version.
const DEVICE_ID: usize = 20;
const DRIVER_VERSION: usize = 24;

fn assert_clean(d: &Device) {
    assert_eq!(
        d.validation_messages(),
        0,
        "validation messages (see the log)"
    );
}

/// Writes a cache, loads it warm, then loads it as another driver version and as another
/// device: both are refused and every pipeline is compiled again.
fn round_trip(path: &Path, skip: &[&CStr], kind: PipelineCacheKind) {
    let d = open(path, skip);
    let report = d.pipeline_cache_report();
    assert_eq!((report.kind, report.loaded), (kind, 0));
    let set = Set::new(&d, 1);
    let n = set.kinds();
    assert_eq!(d.pipeline_cache_report().misses, n);
    set.check(&d);
    d.save_pipeline_cache().unwrap();
    assert_clean(&d);
    drop((set, d));

    // 621.01 takes back compute binaries, compiles mesh pipelines from the VkPipelineCache
    // and caches ray-dispatch pipelines through neither, so only the first two are asserted.
    let d = open(path, skip);
    assert!(d.pipeline_cache_report().loaded > 0);
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
        let d = open(path, skip);
        let report = d.pipeline_cache_report();
        assert!(
            report.rejected.as_deref().is_some_and(|r| r.contains(what)),
            "{what}: {report:?}"
        );
        assert_eq!(report.loaded, 0);
        let set = Set::new(&d, 1);
        let report = d.pipeline_cache_report();
        assert_eq!((report.hits, report.misses), (0, n), "{what}: {report:?}");
        set.check(&d);
        assert_clean(&d);
        drop((set, d));
        std::fs::write(path, saved).unwrap();
    }
}

#[test]
fn binaries_round_trip_and_refuse_foreign_files() {
    let path = cache_path("binaries");
    let probe = open(&path, &[]);
    if !probe.caps().pipeline_binary {
        log::warn!("no VK_KHR_pipeline_binary; skipped");
        return;
    }
    drop(probe);
    round_trip(&path, &[], PipelineCacheKind::Binaries);
}

#[test]
fn pipeline_cache_file_round_trip_and_refuse_foreign_files() {
    let path = cache_path("file");
    round_trip(&path, &[c"VK_KHR_pipeline_binary"], PipelineCacheKind::File);
}

#[test]
fn concurrent_creation() {
    let path = cache_path("concurrent");
    for warm in [false, true] {
        let d = open(&path, &[]);
        // Threads 0 and 1 share their first set's key with threads 2 and 3.
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
