//! Packs mounted from files and from memory: every asset placed on the loading side and
//! read back equal to the archive's CPU decode, lookups by mount, the threads lent to the
//! loading side, failures, the archive handles, and hierarchy pages streamed on request
//! and on the traversal's feedback.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
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
    Device, DeviceDesc, GpuAddress, GpuPtr, Hazards, QueueKind, QueueRequest, Stages, Texture,
    TextureIndex, TextureRegion,
};
use orr_scene::testing::{
    assert_clean, device, global_tables, mounted_texture, read_buffer_on, read_texture_on,
};
use orr_scene::{
    Affine3, BoundTexture, Budgets, EmissiveGpu, Error, FrameDesc, FrameTime, GroupWriter,
    InstanceFlags, InstancesDesc, LoadState, MaterialGpu, MeshGpu, ModelGpu, ModelId, MountId,
    PackSource, RendererDesc, ResidencyClass, SceneDesc, SceneId, SceneRoot, SceneWriter, Scenes,
    TransformGroupDesc, material_flags,
};
use scene_suite::Workers;

const ALBEDO: &str = "kit/albedo";

/// One test at a time, so the memory report sees only the running test's allocations.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr_scene_mount_{}", std::process::id()));
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

/// Runs frames until `m` is ready and every tile of its streamed textures (those of `a`)
/// is resident.
fn ready_whole(scenes: &mut Scenes, m: MountId, a: &Archive) {
    ready(scenes, m);
    let textures: Vec<TextureIndex> = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::Texture as u16)
        .map(|e| scenes.texture(m, e.id).unwrap())
        .collect();
    let start = Instant::now();
    for frames in 1.. {
        assert!(start.elapsed() < Duration::from_secs(60), "tiles pending");
        for &t in &textures {
            scenes.prefetch_texture(m, t, 0);
        }
        frame(scenes);
        let r = *scenes.residency();
        if frames > 1 && r.loads == 0 && r.class(ResidencyClass::Textures).loading_bytes == 0 {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
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
/// replaced by the texture's position in `a`, so two mounts of the same bytes compare equal,
/// also across renderers.
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
        let want = MaterialGpu {
            texture_streams: scenes.texture_streams(),
            ..MaterialGpu::new(&cooked, &bound, got.sampler)
        };
        assert_eq!(
            bytemuck::bytes_of(&got),
            bytemuck::bytes_of(&want),
            "material record"
        );
        got.texture_streams = GpuPtr::NULL;
        for t in &mut got.textures {
            if *t != TextureIndex::NONE {
                *t = TextureIndex(1 + textures.iter().position(|(_, i)| i == t).unwrap() as u32);
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
        ready_whole(&mut scenes, file, &a);
        ready_whole(&mut scenes, memory, &a);
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
        ready_whole(&mut scenes, m, &a_old);
        let gpu = Gpu(device());
        snapshot(&scenes, &gpu, m, &a_old);
        let again = scenes.mount(PackSource::File(path));
        ready_whole(&mut scenes, again, &a_new);
        snapshot(&scenes, &gpu, again, &a_new);
    }
    assert_clean();
}

/// A second device without resizable BAR or host image copy, so every write is staged.
fn staged_device() -> Device {
    let _ = device();
    let staged = Device::new(&DeviceDesc {
        name: "orr-test-staged",
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
        ready_whole(&mut scenes, m, &a);
        let on_staged = snapshot(&scenes, &Gpu(&staged), m, &a);

        let mut direct = Scenes::new(device(), &RendererDesc::default()).unwrap();
        let _w = Workers::start(&direct, 2);
        let d = direct.mount(PackSource::Memory(Arc::clone(bytes)));
        ready_whole(&mut direct, d, &a);
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

/// RGBA8 textures streamed by tile: (name, width, height, layers, cube).
const TILED: [(&str, u32, u32, u32, bool); 3] = [
    ("tiles/flat", 520, 300, 1, false),
    ("tiles/array", 520, 300, 3, false),
    ("tiles/cube", 256, 256, 6, true),
];

/// The texels of `mip` of `layer` of `TILED[t]`: every byte from its texel, mip, layer and
/// channel.
fn tiled_texels(t: usize, mip: u32, layer: u32) -> Vec<u8> {
    let (_, w, h, _, _) = TILED[t];
    let [mw, mh] = orr_pack::texture::mip_size(w, h, mip);
    (0..mw * mh)
        .flat_map(|i| {
            let (x, y) = (i % mw, i / mw);
            (0..4u32).map(move |c| {
                let v = x
                    .wrapping_mul(0x9e37)
                    .wrapping_add(y.wrapping_mul(0x85eb))
                    .wrapping_add(mip * 0x1b5 + layer * 0x3d + c * 0x5f + t as u32 * 0x11);
                (v ^ v >> 7) as u8
            })
        })
        .collect()
}

/// A memory pack of the `TILED` textures, laid out as the cook lays out a texture (blob 0
/// and the mip tail resident from mount, one 64 KiB tile per chunk in each mip blob above
/// the tail, edge tiles padded), a material binding each, and a consumer asset.
fn tiled_pack() -> &'static Arc<[u8]> {
    use orr_pack::texture::{
        FormatInfo, TILE_BYTES, TextureHeader, TextureInfo, full_mip_count, tail_first_mip, tile,
        vk,
    };
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let f = FormatInfo::of(vk::R8G8B8A8_UNORM).unwrap();
        let mut w = ArchiveWriter::create(Target::Memory).unwrap();
        for (t, &(name, width, height, layers, cube)) in TILED.iter().enumerate() {
            let mips = full_mip_count(width, height);
            let tail = tail_first_mip(f, width, height, mips);
            assert_eq!(tail, 2, "{name} streams two mips");
            let header = TextureHeader {
                vk_format: vk::R8G8B8A8_UNORM,
                width,
                height,
                layers: layers as u16,
                kind: if cube {
                    TextureHeader::CUBE
                } else {
                    TextureHeader::D2
                },
                mip_count: mips as u8,
                tail_first_mip: tail as u8,
                primaries: TextureHeader::REC709,
                _pad: [0; 2],
                alpha_cutoff: f32::NAN,
                lights: Span::default(),
            };
            let head = TextureInfo {
                header,
                lights: Vec::new(),
            }
            .to_bytes();
            let mut mip_blobs = Vec::new();
            for mip in 0..tail {
                let [mw, mh] = header.mip_size(mip);
                let mut out = Vec::new();
                for layer in 0..layers {
                    tile(f, mw, mh, &tiled_texels(t, mip, layer), &mut out);
                }
                mip_blobs.push(out);
            }
            let tail_blob: Vec<u8> = (tail..mips)
                .flat_map(|mip| (0..layers).flat_map(move |layer| tiled_texels(t, mip, layer)))
                .collect();
            let mut blobs = vec![BlobIn {
                mount: true,
                ..BlobIn::new(&head)
            }];
            blobs.extend(mip_blobs.iter().map(|b| BlobIn {
                chunk_len: TILE_BYTES as u32,
                ..BlobIn::new(b)
            }));
            blobs.push(BlobIn {
                mount: true,
                ..BlobIn::new(&tail_blob)
            });
            w.add(
                name,
                Class::Texture as u16,
                orr_pack::texture::SCHEMA,
                orr_pack::Hash32([0; 32]),
                &blobs,
            )
            .unwrap();
        }
        // The textures stream because a material binds them.
        for (name, ..) in TILED {
            let mut m = Material::default();
            m.textures[slot::BASE_COLOR].texture = AssetId::from_name(name);
            let bytes = bytemuck::bytes_of(&m);
            w.add(
                &format!("{name}/material"),
                Class::Material as u16,
                orr_pack::material::SCHEMA,
                orr_pack::Hash32([0; 32]),
                &[BlobIn {
                    mount: true,
                    ..BlobIn::new(bytes)
                }],
            )
            .unwrap();
        }
        let (texels, data) = consumer_blobs();
        let head = [0u8; 16];
        w.add(
            CONSUMER,
            CONSUMER_CLASS,
            1,
            orr_pack::Hash32([0; 32]),
            &[
                BlobIn {
                    mount: true,
                    ..BlobIn::new(&head)
                },
                BlobIn::new(&texels),
                BlobIn::new(&data),
            ],
        )
        .unwrap();
        w.finish().unwrap().unwrap()
    })
}

/// A consumer asset of the tiled pack: a class orr does not place, whose blob 1 holds
/// `CONSUMER_TEXELS` RGBA8 texels and blob 2 other bytes.
const CONSUMER: &str = "cells/0";
const CONSUMER_CLASS: u16 = 0x4000;
const CONSUMER_TEXELS: [u32; 2] = [64, 32];

fn consumer_blobs() -> (Vec<u8>, Vec<u8>) {
    let [w, h] = CONSUMER_TEXELS;
    let texels = (0..4 * w * h).map(|i| (i * 7 + i / 13) as u8).collect();
    let data = (0..100_000u32).map(|i| (i ^ i >> 9) as u8).collect();
    (texels, data)
}

/// Tiles above the tail and bytes of the mip tails of `TILED[t]` on `dev`.
fn tiled_layout(dev: &Device, t: usize) -> (u64, u64) {
    use orr_scene::orr_gpu::{Format, TextureDesc, TextureKind, TextureMemory, TextureUsage};
    let (_, w, h, layers, cube) = TILED[t];
    let desc = TextureDesc {
        kind: if cube {
            TextureKind::Cube
        } else {
            TextureKind::D2
        },
        format: Format::Rgba8Unorm,
        extent: [w, h, 1],
        mips: orr_pack::texture::full_mip_count(w, h),
        layers,
        usage: TextureUsage::SAMPLED,
    };
    let probe = dev
        .create_texture(&desc, TextureMemory::Sparse, "tile layout probe")
        .unwrap();
    let l = *probe.sparse_layout().unwrap();
    let tiles: u32 = (0..l.tail_first_mip)
        .map(|m| {
            let [x, y] = l.tiles(&desc, m);
            x * y * layers
        })
        .sum();
    (u64::from(tiles), u64::from(l.tails(&desc)) * l.tail_bytes)
}

/// The mips `mips` of every layer of `TILED[t]` as mount `m` holds them, against the
/// cooked texels.
fn assert_tiled(scenes: &Scenes, dev: &Device, m: MountId, t: usize, mips: std::ops::Range<u32>) {
    let (name, w, h, layers, _) = TILED[t];
    let texture = mounted_texture(scenes, m, AssetId::from_name(name)).unwrap();
    for mip in mips {
        let [mw, mh] = orr_pack::texture::mip_size(w, h, mip);
        for layer in 0..layers {
            let region = TextureRegion {
                mip,
                base_layer: layer,
                layers: 1,
                offset: [0; 3],
                extent: [mw, mh, 1],
            };
            assert!(
                read_texture_on(dev, &texture, &region) == tiled_texels(t, mip, layer),
                "{name} mip {mip} layer {layer} differs from the cooked texels"
            );
        }
    }
}

/// Runs frames, prefetching every tile of the `TILED` textures in `wanted`, until none is
/// loading; asserts the texture pool stays under `budget` and returns the frames it took.
fn stream_tiles(scenes: &mut Scenes, m: MountId, wanted: &[usize], budget: u64) -> u32 {
    let start = Instant::now();
    let indices: Vec<TextureIndex> = wanted
        .iter()
        .map(|&t| scenes.texture(m, AssetId::from_name(TILED[t].0)).unwrap())
        .collect();
    for frames in 1.. {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the tiles never settled"
        );
        for &i in &indices {
            scenes.prefetch_texture(m, i, 0);
        }
        frame(scenes);
        let r = *scenes.residency();
        let pool = *r.class(ResidencyClass::Textures);
        assert!(pool.resident_bytes + pool.loading_bytes <= budget);
        if frames > 1 && r.loads == 0 && r.dropped == 0 && pool.loading_bytes == 0 {
            return frames;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    unreachable!()
}

/// The residency map kernels read for layer 0 of texture `index`.
fn residency_map(scenes: &Scenes, gpu: &Gpu<'_>, index: TextureIndex) -> Vec<u32> {
    let record: GpuPtr<orr_scene::TextureStreamGpu> =
        gpu.value(scenes.texture_streams().offset(u64::from(index.0)));
    assert!(record != GpuPtr::NULL, "a streamed texture has a record");
    let r = gpu.value(record);
    gpu.values(r.map, r.cells[0] * r.cells[1])
}

#[test]
fn streamed_texture_tiles_load_exact_and_evict_after_their_hold() {
    let _serial = serial();
    let bytes = tiled_pack();
    let path = write_pack("tiled.orr", bytes);
    let staged = staged_device();
    for (dev, positioned_reads) in [(device(), false), (device(), true), (&staged, false)] {
        let layouts: Vec<(u64, u64)> = (0..TILED.len()).map(|t| tiled_layout(dev, t)).collect();
        let tile = orr_pack::texture::TILE_BYTES as u64;
        let tails: u64 = layouts.iter().map(|l| l.1).sum();
        // Room for the tails and the array's tiles, which the other two's tiles fit.
        let budget = tails + layouts[1].0 * tile;
        assert!(layouts[0].0 + layouts[2].0 <= layouts[1].0);
        let desc = RendererDesc {
            positioned_reads,
            budgets: Budgets {
                texture_pool: budget,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(dev, &desc).unwrap();
        let _workers = Workers::start(&scenes, 2);
        let gpu = Gpu(dev);
        let m = scenes.mount(PackSource::File(path.clone()));
        ready(&mut scenes, m);
        let pool = *scenes.residency().class(ResidencyClass::Textures);
        assert_eq!(
            pool.resident_bytes, tails,
            "only the mip tails load at mount"
        );
        let index =
            |scenes: &Scenes, t: usize| scenes.texture(m, AssetId::from_name(TILED[t].0)).unwrap();
        assert!(
            residency_map(&scenes, &gpu, index(&scenes, 0))
                .iter()
                .all(|&v| v == 2),
            "the map starts at the tail"
        );
        for t in 0..TILED.len() {
            assert_tiled(&scenes, dev, m, t, 2..9);
        }

        stream_tiles(&mut scenes, m, &[1], budget);
        assert_tiled(&scenes, dev, m, 1, 0..2);

        // The others' tiles fit once the array's have gone 30 frames untouched.
        let frames = stream_tiles(&mut scenes, m, &[0, 2], budget);
        assert!(frames > 30, "settled after {frames} frames");
        assert_tiled(&scenes, dev, m, 0, 0..9);
        assert_tiled(&scenes, dev, m, 2, 0..9);
        assert!(
            residency_map(&scenes, &gpu, index(&scenes, 0))
                .iter()
                .all(|&v| v == 0),
            "every mip 0 tile of the flat texture is resident"
        );
        assert_eq!(scenes.residency().failed, 0);
        scenes.unmount(m);
        for _ in 0..4 {
            frame(&mut scenes);
        }
    }
    assert_eq!(
        staged.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

#[test]
fn taps_of_a_streamed_texture_read_its_resident_mips_and_load_the_tiles_they_want() {
    use orr_scene::orr_gpu::{AddressMode, Filter, MemoryKind, SamplerDesc};
    use scene_suite::pipeline;
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};
    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(tiled_pack())));
    ready(&mut scenes, m);
    let (_, w, h, _, _) = TILED[0];
    let texture = scenes.texture(m, AssetId::from_name(TILED[0].0)).unwrap();
    // One tap per mip 0 texel center, its footprint one texel wide.
    let footprints: Vec<Footprint> = (0..w * h)
        .map(|i| Footprint {
            uv: [
                ((i % w) as f32 + 0.5) / w as f32,
                ((i / w) as f32 + 0.5) / h as f32,
            ],
            dx: [1.0 / w as f32, 0.0],
            dy: [0.0, 1.0 / h as f32],
            hardware: 1,
            pad: 0,
        })
        .collect();
    let mut input = dev
        .gpu_malloc::<Footprint>(MemoryKind::Upload, footprints.len() as u64, "footprints")
        .unwrap();
    input.as_mut_slice().unwrap().copy_from_slice(&footprints);
    let out = dev
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, footprints.len() as u64, "taps")
        .unwrap();
    let sampler = dev
        .sampler(&SamplerDesc {
            mag: Filter::Nearest,
            min: Filter::Nearest,
            mip: Filter::Nearest,
            address: [AddressMode::ClampToEdge; 3],
            ..SamplerDesc::default()
        })
        .unwrap();
    let mut root = dev
        .gpu_malloc::<Root>(MemoryKind::Upload, 1, "texture filter root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = Root {
        footprints: input.gpu(),
        out: out.gpu(),
        texture,
        sampler,
        count: footprints.len() as u32,
        taps: 1,
        streams: scenes.texture_streams(),
        pad: 0,
    };
    let p = pipeline(&texture_filter::MODULE, "mean");
    // The texels nearest sampling of `mip` reads at each tap.
    let nearest = |mip: u32| -> Vec<[f32; 4]> {
        let [mw, mh] = orr_pack::texture::mip_size(w, h, mip);
        let texels = tiled_texels(0, mip, 0);
        footprints
            .iter()
            .map(|f| {
                let x = ((f.uv[0] * mw as f32) as u32).min(mw - 1);
                let y = ((f.uv[1] * mh as f32) as u32).min(mh - 1);
                let at = 4 * (y * mw + x) as usize;
                std::array::from_fn(|c| f32::from(texels[at + c]) / 255.0)
            })
            .collect()
    };
    let render = dev.queue(QueueKind::Render).unwrap();
    let tap_frame = |scenes: &mut Scenes| -> Vec<[f32; 4]> {
        let mut f = scenes
            .begin_frame(&FrameDesc {
                time: FrameTime::default(),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        let list = f.list().unwrap();
        list.dispatch(&p, root.gpu(), [footprints.len().div_ceil(64) as u32, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        let point = f.end(&[]).unwrap();
        render.wait(point, Duration::from_secs(10)).unwrap();
        out.as_slice().unwrap().to_vec()
    };
    assert!(
        tap_frame(&mut scenes) == nearest(2),
        "before any tile streams, taps read the tail"
    );
    let start = Instant::now();
    let finest = nearest(0);
    while tap_frame(&mut scenes) != finest {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the taps' tiles never arrived"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let tiles = orr_pack::texture::TILE_BYTES as u64 * 21;
    let (_, tails) = tiled_layout(dev, 0);
    let pool = *scenes.residency().class(ResidencyClass::Textures);
    let others: u64 = (1..TILED.len()).map(|t| tiled_layout(dev, t).1).sum();
    assert_eq!(
        pool.resident_bytes,
        tails + others + tiles,
        "the taps loaded every tile of mips 0 and 1 of the flat texture, and nothing else"
    );
    assert_clean();
}

/// Runs frames until blob `id` is no longer pending.
fn settle_blob(scenes: &mut Scenes, id: orr_scene::BlobId) -> LoadState {
    let start = Instant::now();
    loop {
        frame(scenes);
        let state = scenes.blob_state(id);
        if !matches!(state, LoadState::Pending) {
            return state;
        }
        assert!(start.elapsed() < Duration::from_secs(30), "blob pending");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn requested_blobs_land_in_consumer_buffers_and_images_under_their_budget() {
    use orr_scene::BlobTarget;
    use orr_scene::orr_gpu::{Format, MemoryKind, TextureDesc, TextureMemory, TextureUsage};
    let _serial = serial();
    let (texels, data) = consumer_blobs();
    let path = write_pack("tiled.orr", tiled_pack());
    let staged = staged_device();
    for (dev, positioned_reads) in [(device(), false), (device(), true), (&staged, false)] {
        let desc = RendererDesc {
            positioned_reads,
            budgets: Budgets {
                user_blob_pool: data.len() as u64,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(dev, &desc).unwrap();
        let _workers = Workers::start(&scenes, 2);
        let gpu = Gpu(dev);
        let m = scenes.mount(PackSource::File(path.clone()));
        ready(&mut scenes, m);
        let id = AssetId::from_name(CONSUMER);
        let buffer = dev
            .gpu_malloc::<u8>(MemoryKind::DeviceLocal, data.len() as u64, "consumer data")
            .unwrap();
        let [w, h] = CONSUMER_TEXELS;
        let image = Arc::new(
            dev.create_texture(
                &TextureDesc::d2(Format::Rgba8Unorm, w, h, TextureUsage::SAMPLED),
                TextureMemory::Dedicated,
                "consumer image",
            )
            .unwrap(),
        );
        let render = dev.queue(QueueKind::Render).unwrap();
        let mut list = render.command_list().unwrap();
        list.discard(&image);
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        render
            .wait(
                render.submit(vec![list], &[]).unwrap(),
                Duration::from_secs(10),
            )
            .unwrap();
        let region = TextureRegion::whole_mip(image.desc(), 0);

        let into_buffer = scenes
            .request_blob(
                m,
                id,
                2,
                BlobTarget::Buffer {
                    at: buffer.gpu(),
                    len: buffer.size(),
                },
            )
            .unwrap();
        let into_image = scenes
            .request_blob(
                m,
                id,
                1,
                BlobTarget::Texture {
                    texture: Arc::clone(&image),
                    region,
                },
            )
            .unwrap();
        assert!(matches!(
            settle_blob(&mut scenes, into_buffer),
            LoadState::Ready
        ));
        assert!(gpu.bytes(buffer.gpu().address().0, data.len()) == data);
        for _ in 0..5 {
            frame(&mut scenes);
        }
        assert!(
            matches!(scenes.blob_state(into_image), LoadState::Pending),
            "the second blob waits for room in the budget"
        );
        let blobs = *scenes.residency().class(ResidencyClass::UserBlobs);
        assert_eq!(blobs.resident_bytes, data.len() as u64);

        scenes.release_blob(into_buffer);
        assert!(matches!(
            settle_blob(&mut scenes, into_image),
            LoadState::Ready
        ));
        assert!(gpu.texture(&image, &region) == texels);
        assert!(matches!(
            scenes.blob_state(into_buffer),
            LoadState::Failed(e) if matches!(*e, Error::UnknownBlob(_))
        ));
        scenes.release_blob(into_image);
        frame(&mut scenes);
        let blobs = *scenes.residency().class(ResidencyClass::UserBlobs);
        assert_eq!((blobs.resident_bytes, blobs.loading_bytes), (0, 0));

        let too_small = scenes
            .request_blob(
                m,
                id,
                2,
                BlobTarget::Buffer {
                    at: buffer.gpu(),
                    len: 16,
                },
            )
            .unwrap();
        assert!(matches!(
            settle_blob(&mut scenes, too_small),
            LoadState::Failed(_)
        ));
        assert!(matches!(
            scenes.request_blob(
                m,
                AssetId::from_name("cells/none"),
                0,
                BlobTarget::Buffer {
                    at: buffer.gpu(),
                    len: 16,
                },
            ),
            Err(Error::Missing { .. })
        ));
    }
    assert_eq!(
        staged.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

const SHEET: &str = "kit/sheet";

/// A pack holding one DAG model over a fine grid with a plain material, whose pages past
/// its pinned coarsest ones stream once `Budgets::mesh_resident_max` is below its size.
fn sheet_pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::Memory,
                vec![AssetSpec {
                    name: SHEET.into(),
                    source: Source::Mesh {
                        mesh: grid(160),
                        materials: vec![Material::default(), Material::default()],
                        settings: MeshSettings {
                            lod: LodMode::Dag,
                            ..MeshSettings::default()
                        },
                    },
                }],
            )],
        };
        orr_cook::cook(&manifest, &Default::default())
            .unwrap()
            .memory
            .remove(0)
    })
}

/// The mesh of `model`: its record, its page table and every streamed page's index.
fn mesh_of(scenes: &Scenes, gpu: &Gpu<'_>, model: ModelId) -> (MeshGpu, Vec<u64>) {
    let (models, meshes, _) = global_tables(scenes);
    let model: ModelGpu = gpu.value(models.offset(u64::from(model.0)));
    let mesh: MeshGpu = gpu.value(meshes.offset(u64::from(model.mesh)));
    let pages = gpu.values(mesh.pages, mesh.page_count);
    (mesh, pages)
}

/// Runs one frame of `scene` with its focus 3 m above the origin, after `write`; returns
/// the scene root of the frame and the cut figures of the last completed frame.
fn streamed_frame(
    scenes: &mut Scenes,
    scene: SceneId,
    write: impl FnOnce(&mut SceneWriter<'_>),
) -> (GpuPtr<SceneRoot>, orr_scene::CutStats) {
    let list = [scene];
    let mut f = scenes
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
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    f.end(&[]).unwrap();
    (root, scenes.stats().cuts)
}

/// Places one traced copy of each of `models` at `positions`.
fn place_copies(s: &mut SceneWriter<'_>, models: &[ModelId]) -> orr_scene::TransformGroupId {
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: models.len() as u32,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    s.transforms(group, 0, models.len() as u32)
        .unwrap()
        .fill(Affine3::IDENTITY);
    for (i, &model) in models.iter().enumerate() {
        s.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first: i as u32,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    }
    group
}

/// Moves copy `i` of `group` to `x` m along the x axis.
fn move_copy(s: &mut SceneWriter<'_>, group: orr_scene::TransformGroupId, i: u32, x: f32) {
    let mut t = Affine3::IDENTITY;
    t.rows[0][3] = x;
    s.transforms(group, i, 1).unwrap()[0] = t;
}

#[test]
fn streamed_mesh_pages_stop_the_cut_at_resident_groups_and_evict_beside_traced_copies() {
    let _serial = serial();
    let staged = staged_device();
    for dev in [device(), &staged] {
        stream_sheets(dev);
    }
    assert_eq!(
        staged.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

/// Two mounts of [`sheet_pack`] under a mesh page budget that holds one mount's streamed
/// pages: the near copy streams its pages in on the traversal's requests, then the other
/// mount's pages take their place.
fn stream_sheets(dev: &Device) {
    let bytes = sheet_pack();
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    let e = entry(&a, AssetId::from_name("kit/sheet/mesh"));
    let info = MeshInfo::parse(&blob(&a, e, 0)).unwrap();
    let pinned = info.header.pinned_pages as usize;
    assert!(pinned < info.pages.len(), "the sheet has streamed pages");
    let unit = |p: &orr_pack::mesh::PageInfo| u64::from(p.resident_len).next_multiple_of(256);
    let pinned_bytes: u64 = info.pages[..pinned].iter().map(unit).sum();
    let streamed_bytes: u64 = info.pages[pinned..].iter().map(unit).sum();
    // Both mounts' pinned pages and one mount's streamed pages.
    let budget = 2 * pinned_bytes + streamed_bytes;
    let desc = RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            mesh_page_pool: budget,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let gpu = Gpu(dev);
    let traced = dev.caps().cluster.max_tris > 0;
    let mut scenes = Scenes::new(dev, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let mounts = [
        scenes.mount(PackSource::Memory(Arc::clone(bytes))),
        scenes.mount(PackSource::Memory(Arc::clone(bytes))),
    ];
    let models = mounts.map(|m| {
        ready(&mut scenes, m);
        scenes.model(m, AssetId::from_name(SHEET)).unwrap()
    });
    for &m in &models {
        let (mesh, pages) = mesh_of(&scenes, &gpu, m);
        assert_ne!(mesh.page_stamps, GpuPtr::NULL);
        assert!(
            pages[..pinned].iter().all(|&p| p != 0),
            "pinned pages placed"
        );
        assert!(
            pages[pinned..].iter().all(|&p| p == 0),
            "streamed pages wait"
        );
    }
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    let mut group = None;
    streamed_frame(&mut scenes, scene, |s| {
        group = Some(place_copies(s, &models));
        move_copy(s, group.unwrap(), 1, 800.0);
    });
    let group = group.unwrap();

    // Before its fine groups land the near copy draws its coarser clusters, never nothing.
    let start = Instant::now();
    let (mut requests, mut drawn) = (0, Vec::new());
    let mut idle = 0;
    while idle < 8 {
        assert!(start.elapsed() < Duration::from_secs(60), "pages pending");
        let (_, c) = streamed_frame(&mut scenes, scene, |_| {});
        requests += c.page_requests;
        drawn.push(c.candidate_clusters);
        let r = *scenes.residency();
        idle = if r.loads == 0 && r.class(ResidencyClass::MeshPages).loading_bytes == 0 {
            idle + 1
        } else {
            0
        };
    }
    assert!(requests > 0, "the traversal requested pages");
    assert!(
        drawn.iter().skip(2).all(|&n| n > 0),
        "drawn while streaming: {drawn:?}"
    );
    let (_, near) = mesh_of(&scenes, &gpu, models[0]);
    let landed: Vec<usize> = (pinned..near.len()).filter(|&p| near[p] != 0).collect();
    assert!(!landed.is_empty(), "streamed pages landed");
    // Streamed pages decode on the loading queue into the bytes of the CPU decode.
    for &p in &landed {
        let stored = blob(&a, e, info.pages[p].blob as usize);
        let mut resident = vec![0; info.pages[p].resident_len as usize];
        decode_page(&stored, &mut resident).unwrap();
        assert_eq!(gpu.bytes(near[p], resident.len()), resident, "page {p}");
    }
    assert!(
        *drawn.last().unwrap() > drawn[2],
        "finer once resident: {drawn:?}"
    );
    // The rest of its pages on request.
    let start = Instant::now();
    while mesh_of(&scenes, &gpu, models[0]).1.contains(&0) {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "prefetch pending"
        );
        scenes.prefetch(models[0], 0);
        streamed_frame(&mut scenes, scene, |_| {});
    }
    let (root, _) = streamed_frame(&mut scenes, scene, |_| {});
    let root: SceneRoot = gpu.value(root);
    let (mesh, _) = mesh_of(&scenes, &gpu, models[0]);
    let first = landed[0];
    let firsts: Vec<u32> = gpu.values(mesh.page_clusters, mesh.page_count + 1);
    let refs: Vec<orr_scene::ClusterRef> = gpu.values(
        root.clusters
            .offset(u64::from(mesh.cluster_base + firsts[first])),
        firsts[first + 1] - firsts[first],
    );
    for r in &refs {
        assert_eq!(r.page.address().0, near[first]);
        assert_eq!(r.clas != 0, traced, "CLAS built for traced devices");
    }
    if traced {
        assert!(scenes.residency().clas_pool_bytes > 0, "the CLAS pool grew");
    }

    // The far copy keeps tracing its cached cut while its pages move in near the focus and
    // the near copy's streamed pages give way after their hold.
    streamed_frame(&mut scenes, scene, |s| {
        move_copy(s, group, 0, 800.0);
        move_copy(s, group, 1, 0.0);
    });
    let start = Instant::now();
    loop {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the second mount's pages never streamed in"
        );
        scenes.prefetch(models[1], 0);
        streamed_frame(&mut scenes, scene, |_| {});
        let pool = *scenes.residency().class(ResidencyClass::MeshPages);
        assert!(pool.resident_bytes + pool.loading_bytes <= budget);
        if !mesh_of(&scenes, &gpu, models[1]).1.contains(&0) {
            break;
        }
    }
    let (_, first_mount) = mesh_of(&scenes, &gpu, models[0]);
    assert!(
        first_mount[pinned..].iter().all(|&p| p == 0),
        "the first mount's streamed pages were evicted"
    );
    for _ in 0..4 {
        streamed_frame(&mut scenes, scene, |_| {});
    }
    assert_eq!(scenes.residency().failed, 0);
}

#[test]
fn the_clas_pool_allocator_places_like_a_first_fit_reference() {
    use orr_scene::memory::RangeAllocator;
    use orr_scene::orr_gpu::{MemoryKind, RtasOp};
    const PAGES: u32 = 12;
    const WORDS: u32 = 16;
    const UNITS: u32 = 32 * WORDS;
    const N: usize = 96;
    let dev = device();
    let module = orr_scene::MODULES
        .iter()
        .find(|m| m.name == "clas_pool")
        .unwrap();
    let pipeline = scene_suite::pipeline(module, "allocate");
    let up = MemoryKind::Upload;
    let bits = dev
        .gpu_malloc::<u32>(MemoryKind::DeviceLocal, u64::from(PAGES * WORDS), "bits")
        .unwrap();
    let summary = dev
        .gpu_malloc::<[u32; 2]>(MemoryKind::DeviceLocal, u64::from(PAGES), "runs")
        .unwrap();
    let mut frees = dev
        .gpu_malloc::<[u32; 2]>(up, (N + PAGES as usize) as u64, "frees")
        .unwrap();
    let mut takes = dev.gpu_malloc::<u32>(up, 16, "takes").unwrap();
    let taken = dev.gpu_malloc::<u32>(up, 16, "taken").unwrap();
    let mut sizes = dev.gpu_malloc::<u32>(up, N as u64, "sizes").unwrap();
    let mut sources = dev.gpu_malloc::<u64>(up, N as u64, "sources").unwrap();
    let units = dev.gpu_malloc::<u32>(up, N as u64, "units").unwrap();
    let destinations = dev.gpu_malloc::<u64>(up, N as u64, "destinations").unwrap();
    let stride = u64::from(dev.rtas_args_stride(RtasOp::Move));
    let moves = dev
        .gpu_malloc::<u8>(up, stride * N as u64, "moves")
        .unwrap();
    let count = dev.gpu_malloc::<u32>(up, 1, "move count").unwrap();
    let mut root = dev
        .gpu_malloc::<orr_scene::PoolRoot>(up, 1, "root")
        .unwrap();
    let base = 1u64 << 40;

    let mut reference = RangeAllocator::new(PAGES * UNITS);
    assert!(reference.take(0, PAGES * UNITS));
    let mut bound = [false; PAGES as usize];
    let mut live: Vec<(u32, u32)> = Vec::new();
    let mut rand = {
        let mut x = 0x2545_f491u32;
        move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        }
    };
    let render = dev.queue(QueueKind::Render).unwrap();
    for round in 0..40 {
        // Bind some pages, free some CLAS, ask to take some pages, then allocate.
        let mut free_list: Vec<[u32; 2]> = Vec::new();
        for p in 0..PAGES {
            if !bound[p as usize] && rand() % 4 == 0 {
                bound[p as usize] = true;
                free_list.push([p * UNITS, UNITS]);
            }
        }
        live.retain(|&(u, n)| {
            let keep = rand() % 3 != 0;
            if !keep {
                free_list.push([u, n]);
            }
            keep
        });
        let take_list: Vec<u32> = (0..PAGES)
            .filter(|&p| bound[p as usize] && rand() % 5 == 0)
            .collect();
        let size_list: Vec<u32> = (0..N).map(|_| 128 + rand() % 6000).collect();
        for &[f, n] in &free_list {
            reference.free(f, n);
        }
        let expect_taken: Vec<u32> = take_list
            .iter()
            .map(|&p| {
                let t = reference.take(p * UNITS, UNITS);
                if t {
                    bound[p as usize] = false;
                }
                u32::from(t)
            })
            .collect();
        let expect_units: Vec<u32> = size_list
            .iter()
            .map(|&s| {
                let need = s.div_ceil(128);
                reference
                    .alloc(need)
                    .inspect(|&u| live.push((u, need)))
                    .unwrap_or(u32::MAX)
            })
            .collect();

        frees.as_mut_slice().unwrap()[..free_list.len()].copy_from_slice(&free_list);
        takes.as_mut_slice().unwrap()[..take_list.len()].copy_from_slice(&take_list);
        sizes.as_mut_slice().unwrap().copy_from_slice(&size_list);
        for (i, s) in sources.as_mut_slice().unwrap().iter_mut().enumerate() {
            *s = 0x1000 * (i as u64 + 1);
        }
        root.as_mut_slice().unwrap()[0] = orr_scene::PoolRoot {
            bits: bits.gpu(),
            summary: summary.gpu().cast(),
            frees: frees.gpu().cast(),
            takes: takes.gpu(),
            taken: taken.gpu(),
            sizes: sizes.gpu(),
            sources: sources.gpu(),
            units: units.gpu(),
            destinations: destinations.gpu(),
            moves: moves.gpu(),
            move_count: count.gpu(),
            base,
            pages: PAGES,
            words: WORDS,
            free_count: free_list.len() as u32,
            take_count: take_list.len() as u32,
            clas_count: N as u32,
            flags: u32::from(round == 0),
            _pad: 0,
        };
        let mut list = render.command_list().unwrap();
        list.dispatch(&pipeline, root.gpu(), [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        orr_scene::testing::wait(render.submit(vec![list], &[]).unwrap());

        assert_eq!(
            &taken.as_slice().unwrap()[..take_list.len()],
            expect_taken.as_slice(),
            "takes of round {round}"
        );
        assert_eq!(
            units.as_slice().unwrap(),
            expect_units.as_slice(),
            "units of round {round}"
        );
        let placed: Vec<u64> = expect_units
            .iter()
            .filter(|&&u| u != u32::MAX)
            .map(|&u| base + 128 * u64::from(u))
            .collect();
        assert_eq!(count.as_slice().unwrap()[0] as usize, placed.len());
        assert_eq!(
            &destinations.as_slice().unwrap()[..placed.len()],
            placed.as_slice()
        );
    }
    assert_clean();
}
