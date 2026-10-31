//! Packs mounted from files and from memory: every asset placed on the loading side and
//! read back equal to the archive's CPU decode, lookups by mount, the threads lent to the
//! loading side, failures, the archive handles, and hierarchy pages streamed on request
//! and on the traversal's feedback.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bytemuck::Pod;
use glam::{DVec3, Quat, Vec2, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, Instances, LodMode, Manifest, MaterialPatch, MeshInput, MeshSettings,
    Placement, PrototypeSpec, Quality, Source, TextureSettings,
};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{MeshHeader, MeshInfo, Node, NodePage, decode_page};
use orr_pack::model::ModelInfo;
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, ArchiveWriter, AssetId, BlobIn, Class, Entry, Span, Target};
use orr_scene::orr_gpu::{
    Device, DeviceDesc, GpuAddress, GpuPtr, QueueKind, QueueRequest, Texture, TextureIndex,
    TextureRegion,
};
use orr_scene::testing::{
    assert_clean, device, global_tables, mounted_texture, read_buffer_on, read_texture_on,
};
use orr_scene::{
    Affine3, BoundTexture, Budgets, EmissiveGpu, Error, FrameDesc, FrameTime, GroupWriter,
    InstanceFlags, InstancesDesc, LoadState, Loading, MaterialGpu, MeshGpu, ModelGpu, ModelId,
    MountId, PackSource, RendererDesc, ResidencyClass, SceneDesc, SceneId, SceneRoot, SceneWriter,
    Scenes, TransformGroupDesc, material_flags,
};

const ALBEDO: &str = "kit/albedo";

/// One test at a time, so the memory report sees only the running test's allocations.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-scene-mount-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn albedo_png() -> &'static PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let path = temp("albedo.png");
        let (w, h) = (520u32, 300u32);
        let texels: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [
                    (x * 255 / w) as u8,
                    (y * 255 / h) as u8,
                    ((x ^ y) & 255) as u8,
                    255,
                ]
            })
            .collect();
        let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .unwrap()
            .write_image_data(&texels)
            .unwrap();
        path
    })
}

/// An `n` x `n` quad grid, gently curved, in two sections split at its middle row.
fn grid(n: u32) -> MeshInput {
    let mut m = MeshInput::default();
    let mut uvs = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            let (x, z) = ((u - 0.5) * 4.0, (v - 0.5) * 4.0);
            m.positions
                .push(Vec3::new(x, 0.2 * (x * 1.9).sin() * (z * 1.3).cos(), z));
            m.normals.push(Vec3::Y);
            uvs.push(Vec2::new(u, v));
        }
    }
    m.uvs = Some(uvs);
    let w = n + 1;
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            let quad = [c, c + w, c + 1, c + 1, c + w, c + w + 1];
            if j < n / 2 { &mut lower } else { &mut upper }.extend(quad);
        }
    }
    m.sections = vec![
        Span {
            first: 0,
            count: lower.len() as u32,
        },
        Span {
            first: lower.len() as u32,
            count: upper.len() as u32,
        },
    ];
    m.indices = [lower, upper].concat();
    m
}

/// A pack with a texture, a discrete-level model, a DAG model with a paged hierarchy, a
/// material no model binds and an instance page. `tint` sets the textured material's red.
fn cook(tint: f32) -> Arc<[u8]> {
    let mut textured = Material {
        base_color: [tint, 0.5, 0.25],
        ..Material::default()
    };
    textured.textures[slot::BASE_COLOR].texture = AssetId::from_name(ALBEDO);
    let glow = Material {
        emission_color: [1.0, 0.6, 0.3],
        emission_luminance: 200.0,
        ..Material::default()
    };
    let mesh = |name: &str, n, materials, lod, hierarchy_resident_max| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh: grid(n),
            materials,
            settings: MeshSettings {
                lod,
                hierarchy_resident_max,
                ..MeshSettings::default()
            },
        },
    };
    let records = (0..6)
        .map(|i| Placement {
            position: DVec3::new(f64::from(i) * 9.0, 0.0, 3.0),
            rotation: Quat::from_rotation_y(i as f32),
            scale: Vec3::ONE,
            prototype: 0,
        })
        .collect();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: ALBEDO.into(),
                    source: Source::Texture {
                        path: albedo_png().clone(),
                        settings: TextureSettings {
                            quality: Quality::Fast,
                            ..TextureSettings::default()
                        },
                    },
                },
                mesh(
                    "kit/panel",
                    40,
                    vec![textured, glow],
                    LodMode::Discrete { errors: vec![0.05] },
                    512 << 10,
                ),
                mesh("kit/dome", 48, vec![glow, textured], LodMode::Dag, 64),
                AssetSpec {
                    name: "loose".into(),
                    source: Source::Material(MaterialPatch {
                        metalness: Some(1.0),
                        ..MaterialPatch::default()
                    }),
                },
                AssetSpec {
                    name: "scatter".into(),
                    source: Source::Instances(Instances {
                        records,
                        prototypes: vec![PrototypeSpec {
                            model: "kit/panel".into(),
                            ..PrototypeSpec::default()
                        }],
                        ..Instances::default()
                    }),
                },
            ],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0)
}

fn pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| cook(0.8))
}

fn write_pack(name: &str, bytes: &[u8]) -> PathBuf {
    let path = temp(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn frame(scenes: &mut Scenes) {
    let f = scenes
        .begin_frame(&FrameDesc {
            time: FrameTime::default(),
            scenes: &[],
            low_latency: None,
        })
        .unwrap();
    f.end(&[]).unwrap();
}

/// Runs frames until `m` is no longer pending.
fn settle(scenes: &mut Scenes, m: MountId) -> LoadState {
    let start = Instant::now();
    loop {
        frame(scenes);
        let state = scenes.mount_state(m);
        if !matches!(state, LoadState::Pending) {
            return state;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "{m:?} still pending"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn ready(scenes: &mut Scenes, m: MountId) {
    if let LoadState::Failed(e) = settle(scenes, m) {
        panic!("{m:?} failed: {e}");
    }
}

/// Threads lent to the loading side until dropped.
struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    fn start(scenes: &Scenes, n: usize) -> Self {
        let loading = scenes.loading();
        let threads = (0..n)
            .map(|_| {
                let l = Arc::clone(&loading);
                std::thread::spawn(move || l.run_worker())
            })
            .collect();
        Self { loading, threads }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.loading.stop_workers();
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}

/// Readback on one device, after every submit so far on its render queue.
struct Gpu<'a>(&'a Device);

impl Gpu<'_> {
    fn bytes(&self, address: u64, len: usize) -> Vec<u8> {
        // SAFETY: the address names `len` bytes of a live allocation.
        let src = unsafe { GpuPtr::<u8>::from_address(GpuAddress(address)) };
        read_buffer_on(self.0, src, len as u64)
    }

    fn values<T: Pod>(&self, ptr: GpuPtr<T>, count: u32) -> Vec<T> {
        if count == 0 {
            return Vec::new();
        }
        read_buffer_on(self.0, ptr, u64::from(count))
    }

    fn value<T: Pod>(&self, ptr: GpuPtr<T>) -> T {
        self.values(ptr, 1)[0]
    }

    fn texture(&self, t: &Texture, region: &TextureRegion) -> Vec<u8> {
        read_texture_on(self.0, t, region)
    }
}

fn blob(a: &Archive, e: &Entry, i: usize) -> Vec<u8> {
    let b = a.blobs(e)[i];
    let mut out = vec![0; b.raw_len as usize];
    a.read(&b, &mut out).unwrap();
    out
}

fn entry(a: &Archive, id: AssetId) -> &Entry {
    a.find(id).unwrap()
}

/// Every asset of mount `m` as it reads back from the GPU, each checked against the CPU
/// decode of `a`. Mount-specific ids and addresses are left out, and texture indices are
/// replaced by the texture's position in `a`, so two mounts of the same bytes compare equal.
fn snapshot(scenes: &Scenes, gpu: &Gpu<'_>, m: MountId, a: &Archive) -> Vec<Vec<u8>> {
    let (models, meshes, materials) = global_tables(scenes);
    let class_of = |c: Class| move |e: &&Entry| e.class == c as u16;
    let textures: Vec<(AssetId, TextureIndex)> = a
        .entries()
        .iter()
        .filter(class_of(Class::Texture))
        .map(|e| (e.id, scenes.texture(m, e.id).unwrap()))
        .collect();
    let material = |id: AssetId| -> Vec<u8> {
        let mid = scenes.material(m, id).unwrap();
        let mut got = gpu.value(materials.offset(u64::from(mid.0)));
        let cooked = Material::from_bytes(&blob(a, entry(a, id), 0)).unwrap();
        let bound = cooked.textures.map(|b| {
            if b.texture == AssetId::NONE {
                return BoundTexture::NONE;
            }
            let header = TextureInfo::parse(&blob(a, entry(a, b.texture), 0))
                .unwrap()
                .header;
            BoundTexture {
                index: scenes.texture(m, b.texture).unwrap(),
                primaries: header.primaries,
            }
        });
        let want = MaterialGpu::new(&cooked, &bound, got.sampler);
        assert_eq!(
            bytemuck::bytes_of(&got),
            bytemuck::bytes_of(&want),
            "material record"
        );
        for t in &mut got.textures {
            if *t != TextureIndex::NONE {
                *t = TextureIndex(textures.iter().position(|(_, i)| i == t).unwrap() as u32);
            }
        }
        bytemuck::bytes_of(&got).to_vec()
    };
    let mut out = Vec::new();
    for e in a.entries() {
        match Class::from_u16(e.class) {
            Some(Class::Model) => {
                let id = scenes.model(m, e.id).unwrap();
                let model: ModelGpu = gpu.value(models.offset(u64::from(id.0)));
                let info = ModelInfo::parse(&blob(a, e, 0)).unwrap();
                assert_eq!(model.section_count as usize, info.materials.len());
                assert_eq!(
                    (model.bounds_min, model.bounds_max),
                    (info.header.bounds.min, info.header.bounds.max)
                );
                let section_materials = gpu.values(model.materials, model.section_count);
                for (&got, &asset) in section_materials.iter().zip(&info.materials) {
                    assert_eq!(got, scenes.material(m, asset).unwrap().0);
                    out.push(material(asset));
                }
                let mesh: MeshGpu = gpu.value(meshes.offset(u64::from(model.mesh)));
                let me = entry(a, info.header.mesh);
                let blob0 = blob(a, me, 0);
                let mi = MeshInfo::parse(&blob0).unwrap();
                assert_eq!(gpu.bytes(mesh.info.address().0, blob0.len()), blob0);
                assert_eq!(mesh.page_count as usize, mi.pages.len());
                assert_eq!(mesh.node_page_count as usize, mi.node_pages.len());
                let pages = gpu.values(mesh.pages, mesh.page_count);
                for (p, &address) in mi.pages.iter().zip(&pages) {
                    let mut want = vec![0; p.resident_len as usize];
                    decode_page(&blob(a, me, p.blob as usize), &mut want).unwrap();
                    let got = gpu.bytes(address, want.len());
                    assert!(got == want, "page {} of {}", p.blob, a.name(me));
                    out.push(got);
                }
                let nodes = gpu.values(mesh.node_pages, mesh.node_page_count);
                for (p, &address) in mi.node_pages.iter().zip(&nodes) {
                    let want = blob(a, me, p.blob as usize);
                    let got = gpu.bytes(address, want.len());
                    assert!(got == want, "node page {} of {}", p.blob, a.name(me));
                    out.push(got);
                }
                out.push(bytemuck::bytes_of(&[model.section_count, mesh.cluster_count]).to_vec());
            }
            Some(Class::Texture) => {
                let info = TextureInfo::parse(&blob(a, e, 0)).unwrap();
                let h = info.header;
                let t = mounted_texture(scenes, m, e.id).unwrap();
                for mip in 0..u32::from(h.mip_count) {
                    let mip_blob = blob(a, e, h.mip_blob(mip));
                    let [w, hh] = h.mip_size(mip);
                    for layer in 0..u32::from(h.layers) {
                        let region = TextureRegion {
                            mip,
                            base_layer: layer,
                            layers: 1,
                            offset: [0; 3],
                            extent: [w, hh, 1],
                        };
                        let got = gpu.texture(&t, &region);
                        assert!(
                            got == info.mip_texels(&mip_blob, mip, layer).unwrap(),
                            "mip {mip}"
                        );
                        out.push(got);
                    }
                }
            }
            Some(Class::Material) => out.push(material(e.id)),
            Some(Class::InstancePage) => {
                let page = scenes.page(scenes.instance_page(m, e.id).unwrap()).unwrap();
                let want = blob(a, e, 1);
                assert_eq!(page.record_count as usize * 16, want.len());
                let got = gpu.bytes(page.records.address().0, want.len());
                assert!(got == want, "records of {}", a.name(e));
                out.push(got);
            }
            _ => {}
        }
    }
    out
}

#[test]
fn file_and_memory_mounts_place_the_same_assets() {
    let _serial = serial();
    let bytes = pack();
    let path = write_pack("same.orr", bytes);
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    let mut pages_seen = (0, 0);
    for positioned_reads in [false, true] {
        let desc = RendererDesc {
            positioned_reads,
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(device(), &desc).unwrap();
        let _workers = Workers::start(&scenes, 3);
        let file = scenes.mount(PackSource::File(path.clone()));
        let memory = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
        ready(&mut scenes, file);
        ready(&mut scenes, memory);
        let gpu = Gpu(device());
        let from_file = snapshot(&scenes, &gpu, file, &a);
        assert!(
            from_file == snapshot(&scenes, &gpu, memory, &a),
            "file and memory differ"
        );

        let (_, _, materials) = global_tables(&scenes);
        let panel =
            ModelInfo::parse(&blob(&a, entry(&a, AssetId::from_name("kit/panel")), 0)).unwrap();
        let section = |i: usize| -> MaterialGpu {
            let id = scenes.material(memory, panel.materials[i]).unwrap();
            gpu.value(materials.offset(u64::from(id.0)))
        };
        let (textured, glow) = (section(0), section(1));
        let albedo = scenes.texture(memory, AssetId::from_name(ALBEDO)).unwrap();
        assert_eq!(textured.textures[slot::BASE_COLOR], albedo);
        assert_eq!(textured.flags & material_flags::EMISSIVE, 0);
        assert_ne!(glow.flags & material_flags::EMISSIVE, 0);
        assert_eq!(glow.textures, [TextureIndex::NONE; 12]);
        let dome = entry(&a, AssetId::from_name("kit/dome/mesh"));
        let dome = MeshInfo::parse(&blob(&a, dome, 0)).unwrap();
        pages_seen = (dome.pages.len(), dome.node_pages.len());
        assert!(matches!(
            scenes.model(memory, AssetId::from_name("kit/albedo")),
            Err(Error::Missing { .. })
        ));
    }
    assert!(
        pages_seen.0 > 1 && pages_seen.1 >= 1,
        "the DAG mesh pages its hierarchy"
    );
    assert_clean();
}

#[test]
fn mounts_wait_for_workers() {
    let _serial = serial();
    let path = write_pack("idle.orr", pack());
    let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let file = scenes.mount(PackSource::File(path));
    let memory = scenes.mount(PackSource::Memory(Arc::clone(pack())));
    for _ in 0..50 {
        frame(&mut scenes);
        std::thread::sleep(Duration::from_millis(2));
    }
    for m in [file, memory] {
        assert!(matches!(scenes.mount_state(m), LoadState::Pending));
        let lookup = scenes.model(m, AssetId::from_name("kit/panel"));
        assert!(matches!(lookup, Err(Error::NotReady(_))));
    }
    let _workers = Workers::start(&scenes, 2);
    ready(&mut scenes, file);
    ready(&mut scenes, memory);
    scenes.model(file, AssetId::from_name("kit/panel")).unwrap();
    assert_clean();
}

#[test]
fn truncated_archives_fail() {
    let _serial = serial();
    let bytes = pack();
    let index_len = Archive::index_range(&bytes[..Archive::HEAD_LEN])
        .unwrap()
        .len as usize;
    let short: Arc<[u8]> = bytes[..index_len * 3 / 4].into();
    let path = write_pack("short.orr", &short);
    for positioned_reads in [false, true] {
        let desc = RendererDesc {
            positioned_reads,
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(device(), &desc).unwrap();
        let _workers = Workers::start(&scenes, 1);
        let file = scenes.mount(PackSource::File(path.clone()));
        let memory = scenes.mount(PackSource::Memory(Arc::clone(&short)));
        for m in [file, memory] {
            let LoadState::Failed(e) = settle(&mut scenes, m) else {
                panic!("{m:?} mounted a truncated archive");
            };
            assert!(
                matches!(*e, Error::Pack(orr_pack::Error::Truncated { .. })),
                "{e}"
            );
            assert!(matches!(
                scenes.model(m, AssetId::from_name("kit/panel")),
                Err(Error::MountFailed(_))
            ));
        }
    }
    assert_clean();
}

#[test]
fn stop_workers_returns_each_worker_after_its_job() {
    let _serial = serial();
    let scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let loading = scenes.loading();
    let signals = Arc::new(AtomicUsize::new(0));
    let s = Arc::clone(&signals);
    loading.set_job_signal(move || {
        s.fetch_add(1, Ordering::Relaxed);
    });
    let (started, finished) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    const JOBS: usize = 64;
    for _ in 0..JOBS {
        let (s, f) = (Arc::clone(&started), Arc::clone(&finished));
        loading.spawn(move || {
            s.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(20));
            f.fetch_add(1, Ordering::SeqCst);
        });
    }
    assert_eq!(signals.load(Ordering::Relaxed), JOBS);
    let workers = Workers::start(&scenes, 4);
    while started.load(Ordering::SeqCst) < 4 {
        std::thread::yield_now();
    }
    drop(workers);
    let (s, f) = (
        started.load(Ordering::SeqCst),
        finished.load(Ordering::SeqCst),
    );
    assert_eq!(s, f, "a worker returned in the middle of a job");
    assert!(s < JOBS, "workers ran every job before stopping");
    let mut rest = 0;
    while loading.run_one() {
        rest += 1;
    }
    assert_eq!(s + rest, JOBS);
    assert!(!loading.run_one());
    loading.spawn(|| {});
    loading.run_worker();
    assert!(loading.run_one(), "a stopped run_worker ran a job");
}

#[test]
fn a_file_renamed_over_a_mount_is_not_read() {
    let _serial = serial();
    let old = pack();
    let new = cook(0.2);
    let (a_old, a_new) = (
        Archive::from_memory(Arc::clone(old)).unwrap(),
        Archive::from_memory(Arc::clone(&new)).unwrap(),
    );
    for positioned_reads in [false, true] {
        let path = write_pack(&format!("recook-{positioned_reads}.orr"), old);
        let desc = RendererDesc {
            positioned_reads,
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(device(), &desc).unwrap();
        let m = scenes.mount(PackSource::File(path.clone()));
        for _ in 0..25 {
            frame(&mut scenes);
            std::thread::sleep(Duration::from_millis(2));
        }
        let next = path.with_extension("next");
        std::fs::write(&next, &new).unwrap();
        std::fs::rename(&next, &path).unwrap();
        let _workers = Workers::start(&scenes, 2);
        ready(&mut scenes, m);
        let gpu = Gpu(device());
        snapshot(&scenes, &gpu, m, &a_old);
        let again = scenes.mount(PackSource::File(path));
        ready(&mut scenes, again);
        snapshot(&scenes, &gpu, again, &a_new);
    }
    assert_clean();
}

/// A second device without resizable BAR or host image copy, so every write is staged.
fn staged_device() -> Device {
    let _ = device();
    let staged = Device::new(&DeviceDesc {
        name: "orr-test-staged",
        validation: true,
        queues: QueueRequest {
            render_compute: true,
            async_compute: true,
            copy: true,
            present: false,
        },
        force_staging: true,
        ..DeviceDesc::default()
    })
    .unwrap();
    assert!(!staged.caps().rebar && !staged.caps().host_image_copy);
    staged
}

#[test]
fn staged_device_places_the_same_assets() {
    let _serial = serial();
    let staged = staged_device();
    let bytes = pack();
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    {
        let mut scenes = Scenes::new(&staged, &RendererDesc::default()).unwrap();
        let _workers = Workers::start(&scenes, 2);
        let m = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
        ready(&mut scenes, m);
        let on_staged = snapshot(&scenes, &Gpu(&staged), m, &a);

        let mut direct = Scenes::new(device(), &RendererDesc::default()).unwrap();
        let _w = Workers::start(&direct, 2);
        let d = direct.mount(PackSource::Memory(Arc::clone(bytes)));
        ready(&mut direct, d);
        assert!(on_staged == snapshot(&direct, &Gpu(device()), d, &a));
    }
    assert_eq!(
        staged.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

fn upload_bytes() -> u64 {
    device().memory_report().kinds[0].bytes
}

#[test]
fn models_emit_and_unmounted_packs_free_after_frames_in_flight() {
    let _serial = serial();
    let bytes = pack();
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            emissive_capacity: 64,
            ..SceneDesc::default()
        })
        .unwrap();
    let m = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
    ready(&mut scenes, m);
    let panel_id = AssetId::from_name("kit/panel");
    let panel = scenes.model(m, panel_id).unwrap();

    // One copy of the panel lists the finest level's EMISSIVE clusters of its glowing
    // section, summed as the archive's pages hold them.
    let list = [s];
    let mut f = scenes
        .begin_frame(&FrameDesc {
            time: FrameTime::default(),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut w = f.scene(s).unwrap();
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    w.transforms(g, 0, 1).unwrap()[0] = Affine3::IDENTITY;
    w.add_instances(&InstancesDesc {
        model: panel,
        copies: 1,
        transforms: g,
        first: 0,
        pose: None,
        material_set: None,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 100.0,
        animation_range: 0.0,
        user_tag: 0,
    })
    .unwrap();
    drop(w);
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    let point = f.end(&[]).unwrap();
    let render = device().queue(QueueKind::Render).unwrap();
    render.wait(point, Duration::from_secs(10)).unwrap();
    let gpu = Gpu(device());
    let root: SceneRoot = gpu.value(root);
    let entries: Vec<EmissiveGpu> = gpu.values(
        root.emissive,
        root.producer_capacity + root.emissive_capacity,
    );
    let instances = &entries[root.producer_capacity as usize..];
    let live: Vec<&EmissiveGpu> = instances.iter().filter(|e| e.kind != 0).collect();
    let info = ModelInfo::parse(&blob(&a, entry(&a, panel_id), 0)).unwrap();
    let me = entry(&a, info.header.mesh);
    let mi = MeshInfo::parse(&blob(&a, me, 0)).unwrap();
    let finest = mi.levels[0].clusters;
    let (mut index, mut area) = (0u32, 0f32);
    for p in &mi.pages {
        let mut page = vec![0; p.resident_len as usize];
        decode_page(&blob(&a, me, p.blob as usize), &mut page).unwrap();
        let view = orr_pack::mesh::PageView::new(&page, &mi.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            let in_finest = (finest.first..finest.first + finest.count).contains(&index);
            if in_finest && c.flags & orr_pack::mesh::Cluster::EMISSIVE != 0 {
                assert_eq!(c.section, 1, "only the glowing section emits");
                area += c.area;
            }
            index += 1;
        }
    }
    assert!(area > 0.0 && !live.is_empty());
    let glow = scenes.material(m, info.materials[1]).unwrap();
    assert!(
        live.iter()
            .all(|e| e.section == 1 && e.material == glow.0 && e.sections == 2)
    );
    let listed: f32 = live.iter().map(|e| e.model_area).sum();
    assert!(
        (listed - area).abs() <= 1e-4 * area,
        "{listed} m² listed, {area} m² cooked"
    );

    // The pack's pages alone are a lower bound of what the unmount frees.
    let pages: u64 = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::Mesh as u16)
        .flat_map(|e| MeshInfo::parse(&blob(&a, e, 0)).unwrap().pages)
        .map(|p| u64::from(p.resident_len))
        .sum();
    let mounted = upload_bytes();
    scenes.unmount(m);
    assert!(matches!(
        scenes.model(m, panel_id),
        Err(Error::UnknownMount(_))
    ));
    assert_eq!(
        upload_bytes(),
        mounted,
        "freed while frames that read it are in flight"
    );
    let start = Instant::now();
    while upload_bytes() > mounted - pages {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "an unmounted pack stays allocated"
        );
        frame(&mut scenes);
    }
    let again = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
    ready(&mut scenes, again);
    assert_eq!(
        scenes.model(again, panel_id).unwrap(),
        panel,
        "freed ids are reused"
    );
    assert_clean();
}

const DOME_MESH: &str = "kit/dome/mesh";

/// [`pack`] with the dome's hierarchy split into a top page holding the roots and one
/// streamed page holding every other node: the layout the cook writes for hierarchies over
/// one node page, at a size a test can cook quickly.
fn paged_hierarchy() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let a = Archive::from_memory(Arc::clone(pack())).unwrap();
        let mut w = ArchiveWriter::create(Target::Memory).unwrap();
        for e in a.entries() {
            if e.id != AssetId::from_name(DOME_MESH) {
                w.copy_entry(&a, e).unwrap();
                continue;
            }
            let mut info = MeshInfo::parse(&blob(&a, e, 0)).unwrap();
            assert_eq!(info.node_pages.len(), 1);
            let nodes = blob(&a, e, info.node_pages[0].blob as usize);
            let count = info.node_pages[0].node_count;
            let top = info
                .roots
                .iter()
                .filter(|&&r| r != MeshHeader::NO_ROOT)
                .max()
                .unwrap()
                + 1;
            assert!(top < count, "the dome has nodes below its roots");
            let first_node_blob = 1 + info.pages.len() as u32;
            info.node_pages = vec![
                NodePage {
                    first_node: 0,
                    node_count: top,
                    blob: first_node_blob,
                    parent_page: NodePage::TOP,
                },
                NodePage {
                    first_node: top,
                    node_count: count - top,
                    blob: first_node_blob + 1,
                    parent_page: 0,
                },
            ];
            let blob0 = info.to_bytes();
            let pages: Vec<Vec<u8>> = (1..first_node_blob as usize)
                .map(|i| blob(&a, e, i))
                .collect();
            let (top_nodes, rest) = nodes.split_at(top as usize * size_of::<Node>());
            let mount = |bytes| BlobIn {
                mount: true,
                ..BlobIn::new(bytes)
            };
            let mut blobs = vec![mount(&blob0)];
            blobs.extend(pages.iter().map(|p| BlobIn::new(p)));
            blobs.extend([mount(top_nodes), BlobIn::new(rest)]);
            w.add(a.name(e), e.class, e.schema, e.cook_key, &blobs)
                .unwrap();
        }
        w.finish().unwrap().unwrap()
    })
}

/// The node page table of `model`'s mesh.
fn node_table(scenes: &Scenes, gpu: &Gpu<'_>, model: ModelId) -> Vec<u64> {
    let (models, meshes, _) = global_tables(scenes);
    let model: ModelGpu = gpu.value(models.offset(u64::from(model.0)));
    let mesh: MeshGpu = gpu.value(meshes.offset(u64::from(model.mesh)));
    gpu.values(mesh.node_pages, mesh.node_page_count)
}

/// Runs frames, prefetching `model`, until its streamed node page is resident; returns the
/// frames it took.
fn stream_in(scenes: &mut Scenes, gpu: &Gpu<'_>, model: ModelId, budget: u64) -> u32 {
    let start = Instant::now();
    let mut frames = 0;
    while node_table(scenes, gpu, model)[1] == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the node page never streamed in"
        );
        scenes.prefetch(model, 0);
        frame(scenes);
        frames += 1;
        let pool = *scenes.residency().class(ResidencyClass::HierarchyPages);
        assert!(pool.resident_bytes + pool.loading_bytes <= budget);
    }
    frames
}

#[test]
fn hierarchy_pages_stream_on_request_and_evict_after_their_hold() {
    let _serial = serial();
    let bytes = paged_hierarchy();
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    let mesh = entry(&a, AssetId::from_name(DOME_MESH));
    let info = MeshInfo::parse(&blob(&a, mesh, 0)).unwrap();
    let pages: Vec<Vec<u8>> = info
        .node_pages
        .iter()
        .map(|p| blob(&a, mesh, p.blob as usize))
        .collect();
    let path = write_pack("paged.orr", bytes);
    let staged = staged_device();
    // Room for both mounts' top pages and one streamed page.
    let budget = 2 * pages[0].len() as u64 + pages[1].len() as u64;
    let desc = |positioned_reads| RendererDesc {
        positioned_reads,
        budgets: Budgets {
            hierarchy_page_pool: budget,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    for (dev, positioned_reads) in [(device(), false), (device(), true), (&staged, false)] {
        let mut scenes = Scenes::new(dev, &desc(positioned_reads)).unwrap();
        let _workers = Workers::start(&scenes, 2);
        let gpu = Gpu(dev);
        let mounts = [
            scenes.mount(PackSource::File(path.clone())),
            scenes.mount(PackSource::Memory(Arc::clone(bytes))),
        ];
        let models = mounts.map(|m| {
            ready(&mut scenes, m);
            scenes.model(m, AssetId::from_name("kit/dome")).unwrap()
        });
        for &m in &models {
            let table = node_table(&scenes, &gpu, m);
            assert_eq!(gpu.bytes(table[0], pages[0].len()), pages[0]);
            assert_eq!(table[1], 0, "streamed pages wait for a request");
        }
        let pool = *scenes.residency().class(ResidencyClass::HierarchyPages);
        assert_eq!(pool.resident_units, 2);

        stream_in(&mut scenes, &gpu, models[0], budget);
        let first = node_table(&scenes, &gpu, models[0]);
        assert_eq!(gpu.bytes(first[1], pages[1].len()), pages[1]);

        // The second mount's page fits only once the first one's 30-frame hold has passed.
        let frames = stream_in(&mut scenes, &gpu, models[1], budget);
        assert!(frames > 30, "evicted after {frames} frames");
        assert_eq!(node_table(&scenes, &gpu, models[0])[1], 0, "evicted");
        let second = node_table(&scenes, &gpu, models[1]);
        assert_eq!(gpu.bytes(second[1], pages[1].len()), pages[1]);
        assert_eq!(scenes.residency().failed, 0);
    }
    assert_eq!(
        staged.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

/// Runs one frame of `scene` with its focus 3 m above the origin, after `write`; returns the
/// figures of the last completed frame.
fn traverse_frame(
    scenes: &mut Scenes,
    scene: SceneId,
    write: impl FnOnce(&mut SceneWriter<'_>),
) -> orr_scene::CutStats {
    let list = [scene];
    let f = scenes
        .begin_frame(&FrameDesc {
            time: FrameTime::default(),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(scene).unwrap();
        s.set_focus(DVec3::new(0.0, 3.0, 0.0), 1000.0);
        write(&mut s);
    }
    f.end(&[]).unwrap();
    scenes.stats().cuts
}

#[test]
fn the_traversal_requests_the_node_pages_it_needs_and_keeps_reading_them() {
    let _serial = serial();
    let bytes = paged_hierarchy();
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    let mesh = entry(&a, AssetId::from_name(DOME_MESH));
    let info = MeshInfo::parse(&blob(&a, mesh, 0)).unwrap();
    let page_bytes = |p: usize| blob(&a, mesh, info.node_pages[p].blob as usize).len() as u64;
    // Room for both mounts' top pages and one streamed page.
    let budget = 2 * page_bytes(0) + page_bytes(1);
    let desc = RendererDesc {
        budgets: Budgets {
            hierarchy_page_pool: budget,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let dev = device();
    let mut scenes = Scenes::new(dev, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let gpu = Gpu(dev);
    let mounts = [
        scenes.mount(PackSource::Memory(Arc::clone(bytes))),
        scenes.mount(PackSource::Memory(Arc::clone(bytes))),
    ];
    let models = mounts.map(|m| {
        ready(&mut scenes, m);
        scenes.model(m, AssetId::from_name("kit/dome")).unwrap()
    });
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    let mut range = None;
    traverse_frame(&mut scenes, scene, |s| {
        let group = s
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        s.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY;
        range = Some(
            s.add_instances(&InstancesDesc {
                model: models[0],
                copies: 1,
                transforms: group,
                first: 0,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 100.0,
                animation_range: 0.0,
                user_tag: 0,
            })
            .unwrap(),
        );
    });
    let range = range.unwrap();

    // Below its roots the copy's cut waits for the page, which the traversal requests.
    let start = Instant::now();
    let (mut requests, mut before) = (0, Vec::new());
    while node_table(&scenes, &gpu, models[0])[1] == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the node page never streamed in"
        );
        let c = traverse_frame(&mut scenes, scene, |_| {});
        requests += c.page_requests;
        before.push(c.candidate_clusters);
    }
    assert!(requests > 0, "the traversal requested the page");
    assert!(
        before.iter().all(|&n| n == 0),
        "nothing drawn before the page: {before:?}"
    );
    let mut drawn = 0;
    for _ in 0..3 {
        drawn = traverse_frame(&mut scenes, scene, |_| {}).candidate_clusters;
    }
    assert!(drawn > 0, "the copy draws once the page is resident");

    // Read every frame, the page outlives its hold: the other mount's page never finds room.
    for _ in 0..60 {
        scenes.prefetch(models[1], 0);
        traverse_frame(&mut scenes, scene, |_| {});
    }
    assert_ne!(
        node_table(&scenes, &gpu, models[0])[1],
        0,
        "the page stayed"
    );
    assert_eq!(
        node_table(&scenes, &gpu, models[1])[1],
        0,
        "no room for the other page"
    );

    // Once no traversal reads it, it gives way after its hold.
    traverse_frame(&mut scenes, scene, |s| s.remove(range));
    let frames = stream_in(&mut scenes, &gpu, models[1], budget);
    assert!(frames > 30, "evicted after {frames} frames");
    assert_eq!(node_table(&scenes, &gpu, models[0])[1], 0, "evicted");
    assert_clean();
}
