//! Packs mounted from files and from memory: every asset placed on the loading side and
//! read back equal to the archive's CPU decode, lookups by mount, the threads lent to the
//! loading side, failures, the archive handles, hierarchy pages streamed on request and on
//! the traversal's feedback, and textures decoded from neural texture sets.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use orr_cook::{
    ArchiveSpec, AssetSpec, Instances, Linear, LodMode, Manifest, MaterialPatch, MeshInput,
    MeshSettings, Placement, PrototypeSpec, Source, TextureSettings,
};
use orr_math::{Affine3, DVec3, Quat, Vec2, Vec3};
use orr_pack::material::{Material, TextureBinding, filter, slot, wrap};
use orr_pack::mesh::{MeshHeader, MeshInfo, Node, NodePage, decode_page, in_cut};
use orr_pack::model::ModelInfo;
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, ArchiveWriter, AssetId, BlobIn, Class, Entry, Span, Target};
use orr_scene::PageId;
use orr_scene::orr_gpu::{
    Device, DeviceDesc, GpuAddress, GpuPtr, Hazards, QueueKind, QueueRequest, Stages, Texture,
    TextureIndex, TextureRegion,
};
use orr_scene::testing::{
    assert_clean, device, global_tables, mounted_texture, read_buffer_on, read_texture_on,
};
use orr_scene::{
    BoundTexture, Budgets, EmissiveGpu, Error, FrameDesc, FrameTime, GroupWriter, InstanceFlags,
    InstancesDesc, LoadState, MaterialGpu, MeshGpu, ModelGpu, ModelId, MountId, NeuralTextures,
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
            position: DVec3::new(f64::from(i) * 9.0, 0.0, 3.0).to_array(),
            linear: Linear::Scaled {
                rotation: Quat::from_rotation_y(i as f32).to_array(),
                scale: [1.0; 3],
            },
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
    let pages: Vec<PageId> = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
        .map(|e| scenes.instance_page(m, e.id).unwrap())
        .collect();
    let start = Instant::now();
    for frames in 1.. {
        assert!(start.elapsed() < Duration::from_secs(60), "tiles pending");
        for &t in &textures {
            scenes.prefetch_texture(m, t, 0);
        }
        for &p in &pages {
            scenes.prefetch_page(p, 0);
        }
        frame(scenes);
        let r = *scenes.residency();
        let landed = pages.iter().all(|&p| {
            let page = scenes.page(p).unwrap();
            page.info.header.record_count == 0 || page.records != GpuPtr::NULL
        });
        if frames > 1
            && landed
            && r.loads == 0
            && r.class(ResidencyClass::Textures).loading_bytes == 0
        {
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
        let bound = std::array::from_fn(|s| {
            let b = cooked.textures[s];
            if b.texture == AssetId::NONE {
                return BoundTexture::NONE;
            }
            let header = TextureInfo::parse(&blob(a, entry(a, b.texture), 0))
                .unwrap()
                .header;
            BoundTexture {
                index: scenes.texture(m, b.texture).unwrap(),
                primaries: header.primaries,
                sampler: got.samplers[s].sampler,
                ..BoundTexture::NONE
            }
        });
        let want = MaterialGpu {
            texture_streams: scenes.texture_streams(),
            ..MaterialGpu::new(&cooked, &bound)
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
                assert_eq!(page.bytes as usize, want.len());
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
        assert_eq!(glow.textures, [TextureIndex::NONE; slot::COUNT]);
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

    // One copy of the panel takes one entry for its glowing section, the cooked set of the
    // level it traces, whose weight the archive's pages add up to.
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
    w.transforms(g, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
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
    assert_eq!(live.len(), 1);
    let (e, sets) = (live[0], &mi.emissive);
    assert_eq!(sets.rank_count(), 1);
    let set = sets.sets[e.set as usize];
    assert_eq!(set.section, 1, "only the glowing section emits");
    assert_eq!(e.cluster_count, set.count);
    let level = mi.levels[e.set as usize].clusters;
    let (mut index, mut area) = (0u32, 0f32);
    for p in &mi.pages {
        let mut page = vec![0; p.resident_len as usize];
        decode_page(&blob(&a, me, p.blob as usize), &mut page).unwrap();
        let view = orr_pack::mesh::PageView::new(&page, &mi.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            if (level.first..level.first + level.count).contains(&index) && c.section == 1 {
                area += c.area;
            }
            index += 1;
        }
    }
    assert!(area > 0.0);
    assert!(
        (e.area - area).abs() <= 1e-4 * area,
        "{} m² listed, {area} m² cooked",
        e.area
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

/// A model `kit/panel` in one archive and a page of it in another.
fn split_packs() -> (Arc<[u8]>, Arc<[u8]>) {
    let glow = Material {
        emission_color: [1.0, 1.0, 1.0],
        emission_luminance: 50.0,
        ..Material::default()
    };
    let panel = AssetSpec {
        name: "kit/panel".into(),
        source: Source::Mesh {
            mesh: grid(8),
            materials: vec![glow, glow],
            settings: MeshSettings::default(),
        },
    };
    let scatter = AssetSpec {
        name: "scatter".into(),
        source: Source::Instances(Instances {
            records: (0..3)
                .map(|i| Placement {
                    position: DVec3::new(f64::from(i) * 9.0, 0.0, 3.0).to_array(),
                    linear: Linear::Scaled {
                        rotation: Quat::IDENTITY.to_array(),
                        scale: [1.0; 3],
                    },
                    prototype: 0,
                })
                .collect(),
            prototypes: vec![PrototypeSpec {
                model: "kit/panel".into(),
                ..PrototypeSpec::default()
            }],
            ..Instances::default()
        }),
    };
    let manifest = Manifest {
        archives: vec![
            ArchiveSpec::new(Target::Memory, vec![panel]),
            ArchiveSpec::new(Target::Memory, vec![scatter]),
        ],
    };
    let mut memory = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory;
    let page = memory.remove(1);
    (memory.remove(0), page)
}

#[test]
fn an_unmount_takes_copies_and_pages_of_its_models_from_a_scene_that_keeps_rendering() {
    let _serial = serial();
    let (models, pages) = split_packs();
    let scatter = Archive::from_memory(Arc::clone(&pages)).unwrap();
    let page_id = scatter
        .entries()
        .iter()
        .find(|e| e.class == Class::InstancePage as u16)
        .unwrap()
        .id;
    let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            emissive_capacity: 64,
            ..SceneDesc::default()
        })
        .unwrap();
    let m = scenes.mount(PackSource::Memory(models));
    ready(&mut scenes, m);
    let p = scenes.mount(PackSource::Memory(pages));
    ready(&mut scenes, p);
    let panel = scenes.model(m, AssetId::from_name("kit/panel")).unwrap();
    let page = scenes.instance_page(p, page_id).unwrap();

    let list = [s];
    let desc = FrameDesc {
        time: FrameTime::default(),
        scenes: &list,
        low_latency: None,
    };
    let frame_with = |scenes: &mut Scenes, write: &mut dyn FnMut(&mut SceneWriter<'_>)| {
        let mut f = scenes.begin_frame(&desc).unwrap();
        write(&mut f.scene(s).unwrap());
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| root = c.root).unwrap();
        let point = f.end(&[]).unwrap();
        let render = device().queue(QueueKind::Render).unwrap();
        render.wait(point, Duration::from_secs(10)).unwrap();
        root
    };
    let (mut copies, mut placed) = (None, None);
    let mut group = None;
    let root = frame_with(&mut scenes, &mut |w| {
        let g = w
            .create_transform_group(&TransformGroupDesc {
                count: 2,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        w.transforms(g, 0, 2)
            .unwrap()
            .fill(Affine3::IDENTITY.into());
        let range = w
            .add_instances(&InstancesDesc {
                model: panel,
                copies: 2,
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
        placed = Some(
            w.add_instance_page(&orr_scene::InstancePageRef {
                page,
                frame: None,
                poses: &[],
                material_sets: &[],
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 100.0,
                animation_range: 0.0,
                user_tag: 0,
            })
            .unwrap(),
        );
        copies = Some(range);
        group = Some(g);
    });
    let (copies, placed, group) = (copies.unwrap(), placed.unwrap(), group.unwrap());
    let gpu = Gpu(device());
    let root: SceneRoot = gpu.value(root);
    let state = |gpu: &Gpu<'_>| -> (Vec<u8>, usize) {
        let instances: Vec<orr_scene::InstanceGpu> =
            gpu.values(root.instances, root.instance_capacity);
        let entries: Vec<EmissiveGpu> = gpu.values(
            root.emissive,
            root.producer_capacity + root.emissive_capacity,
        );
        let flags = instances.iter().map(|i| i.flags).collect();
        (flags, entries.iter().filter(|e| e.kind != 0).count())
    };

    // The page's records land and write its slots and entries.
    let start = Instant::now();
    while state(&gpu).1 < 2 * (2 + 3) {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the page's records never landed"
        );
        frame_with(&mut scenes, &mut |_| {});
    }
    scenes.unmount(m);
    for _ in 0..3 {
        frame_with(&mut scenes, &mut |_| {});
    }
    let (flags, entries) = state(&gpu);
    assert!(flags.iter().all(|&f| f == 0), "copies outlive their model");
    assert_eq!(entries, 0, "emissive entries outlive their model");

    // Their ranges stay the consumer's, whose slots serve new copies of a mount that came back.
    let again = scenes.mount(PackSource::Memory(split_packs().0));
    ready(&mut scenes, again);
    let panel = scenes
        .model(again, AssetId::from_name("kit/panel"))
        .unwrap();
    frame_with(&mut scenes, &mut |w| {
        w.remove(copies);
        w.remove(placed);
        w.add_instances(&InstancesDesc {
            model: panel,
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
        .unwrap();
    });
    frame_with(&mut scenes, &mut |_| {});
    assert_eq!(state(&gpu).1, 2);
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
        s.set_focus(DVec3::new(0.0, 3.0, 0.0));
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
        s.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
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

/// Adds texture entry `name` laid out as the cook lays out a texture: blob 0 and the mip
/// tail resident from mount, one 64 KiB tile per chunk in each mip blob above the tail,
/// edge tiles padded. `texels(mip, layer)` are a mip's tight texels. Returns the first mip
/// of the tail.
fn add_texture(
    w: &mut ArchiveWriter,
    name: &str,
    vk_format: u32,
    [width, height]: [u32; 2],
    layers: u32,
    cube: bool,
    texels: impl Fn(u32, u32) -> Vec<u8>,
) -> u32 {
    use orr_pack::texture::{
        FormatInfo, TILE_BYTES, TextureHeader, TextureInfo, full_mip_count, tail_first_mip, tile,
    };
    let f = FormatInfo::of(vk_format).unwrap();
    let mips = full_mip_count(width, height);
    let tail = tail_first_mip(f, width, height, mips);
    let header = TextureHeader {
        vk_format,
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
        primaries: orr_color::Primaries::Rec709.id(),
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
            tile(f, mw, mh, &texels(mip, layer), &mut out);
        }
        mip_blobs.push(out);
    }
    let tail_blob: Vec<u8> = (tail..mips)
        .flat_map(|mip| {
            (0..layers)
                .flat_map(|layer| texels(mip, layer))
                .collect::<Vec<_>>()
        })
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
    tail
}

/// A memory pack of the `TILED` textures ([`add_texture`]), a material binding each, and a
/// consumer asset.
fn tiled_pack() -> &'static Arc<[u8]> {
    use orr_pack::texture::vk;
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let mut w = ArchiveWriter::create(Target::Memory).unwrap();
        for (t, &(name, width, height, layers, cube)) in TILED.iter().enumerate() {
            let tail = add_texture(
                &mut w,
                name,
                vk::R8G8B8A8_UNORM,
                [width, height],
                layers,
                cube,
                |mip, layer| tiled_texels(t, mip, layer),
            );
            assert_eq!(tail, 2, "{name} streams two mips");
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

/// A texture of one mip larger than a tile, which has no mip tail to stand in for tiles not
/// yet loaded, places whole even where a material binds it.
#[test]
fn a_texture_without_a_mip_tail_places_whole() {
    use orr_pack::texture::{FormatInfo, TextureHeader, tail_first_mip, tile, vk};
    let _serial = serial();
    let (name, width, height, ..) = TILED[0];
    let f = FormatInfo::of(vk::R8G8B8A8_UNORM).unwrap();
    assert_eq!(tail_first_mip(f, width, height, 1), 1, "no mip tail");
    let header = TextureHeader {
        vk_format: vk::R8G8B8A8_UNORM,
        width,
        height,
        layers: 1,
        kind: TextureHeader::D2,
        mip_count: 1,
        tail_first_mip: 1,
        primaries: orr_color::Primaries::Rec709.id(),
        _pad: [0; 2],
        alpha_cutoff: f32::NAN,
        lights: Span::default(),
    };
    let head = TextureInfo {
        header,
        lights: Vec::new(),
    }
    .to_bytes();
    let texels = tiled_texels(0, 0, 0);
    let mut tiles = Vec::new();
    tile(f, width, height, &texels, &mut tiles);
    let mut m = Material::default();
    m.textures[slot::BASE_COLOR].texture = AssetId::from_name(name);
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    let blobs = [
        BlobIn {
            mount: true,
            ..BlobIn::new(&head)
        },
        BlobIn {
            chunk_len: orr_pack::texture::TILE_BYTES as u32,
            ..BlobIn::new(&tiles)
        },
    ];
    let key = orr_pack::Hash32([0; 32]);
    w.add(
        name,
        Class::Texture as u16,
        orr_pack::texture::SCHEMA,
        key,
        &blobs,
    )
    .unwrap();
    let material = [BlobIn {
        mount: true,
        ..BlobIn::new(bytemuck::bytes_of(&m))
    }];
    w.add(
        "material",
        Class::Material as u16,
        orr_pack::material::SCHEMA,
        key,
        &material,
    )
    .unwrap();
    let pack = w.finish().unwrap().unwrap();

    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let gpu = Gpu(dev);
    let mount = scenes.mount(PackSource::Memory(pack));
    ready(&mut scenes, mount);
    let index = scenes.texture(mount, AssetId::from_name(name)).unwrap();
    let record: GpuPtr<orr_scene::TextureStreamGpu> =
        gpu.value(scenes.texture_streams().offset(u64::from(index.0)));
    assert!(record == GpuPtr::NULL, "the texture does not stream");
    let texture = mounted_texture(&scenes, mount, AssetId::from_name(name)).unwrap();
    let region = TextureRegion {
        mip: 0,
        base_layer: 0,
        layers: 1,
        offset: [0; 3],
        extent: [width, height, 1],
    };
    assert!(gpu.texture(&texture, &region) == texels);
    scenes.unmount(mount);
    for _ in 0..4 {
        frame(&mut scenes);
    }
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
            material: 0,
            inward: [0.0; 2],
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
        modes: 0,
        pad: 0,
        materials: GpuPtr::NULL,
        pad2: 0,
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

/// A tile set mounts as a table of its tiles' heap indices, its cells without one left empty, and
/// each tile streams as a texture a material binds does.
#[test]
fn a_tile_set_mounts_as_a_table_of_streamed_textures() {
    use orr_pack::texture::vk;
    use orr_pack::tiles::TextureTiles;
    use orr_scene::orr_gpu::MemoryKind;
    use scene_suite::pipeline;
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};
    let _serial = serial();
    let dev = device();
    let (_, width, height, ..) = TILED[0];
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    let names = ["set/a", "set/b"];
    for name in names {
        let texels = |mip, layer| tiled_texels(0, mip, layer);
        add_texture(
            &mut w,
            name,
            vk::R8G8B8A8_UNORM,
            [width, height],
            1,
            false,
            texels,
        );
    }
    let named = AssetId::from_name;
    let set = TextureTiles::new(&[(0, 0, named(names[0])), (2, 1, named(names[1]))]);
    let bytes = set.to_bytes();
    let blob = [BlobIn {
        mount: true,
        ..BlobIn::new(&bytes)
    }];
    let schema = orr_pack::tiles::SCHEMA;
    let no_key = orr_pack::Hash32([0; 32]);
    w.add("set", Class::TextureTiles as u16, schema, no_key, &blob)
        .unwrap();
    let mut material = Material::default();
    material.textures[slot::BASE_COLOR].texture = AssetId::from_name("set");
    let blob = [BlobIn {
        mount: true,
        ..BlobIn::new(bytemuck::bytes_of(&material))
    }];
    let schema = orr_pack::material::SCHEMA;
    w.add("material", Class::Material as u16, schema, no_key, &blob)
        .unwrap();
    let pack = w.finish().unwrap().unwrap();

    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(pack));
    ready(&mut scenes, m);
    let gpu = Gpu(dev);
    let (_, _, materials) = global_tables(&scenes);
    let id = scenes.material(m, AssetId::from_name("material")).unwrap();
    let got = gpu.value(materials.offset(u64::from(id.0)));
    let table = got.tiles[slot::BASE_COLOR];
    assert!(table != GpuPtr::NULL, "the slot reads a table");
    assert_ne!(got.samplers[slot::BASE_COLOR].modes & (1 << 10), 0);
    let index = |name: &str| scenes.texture(m, AssetId::from_name(name)).unwrap();
    let [a, b] = names.map(index);
    let want = set.table(|t| Some(if t == named(names[0]) { a.0 } else { b.0 }));
    assert_eq!(gpu.values(table, set.words() as u32), want.unwrap());
    let at_tail = |scenes: &Scenes, t| residency_map(scenes, &gpu, t).iter().all(|&mip| mip == 2);
    assert!(at_tail(&scenes, a) && at_tail(&scenes, b));

    // Feedback goes to the tile a tap lands on.
    let tap = Footprint {
        uv: [2.5, -0.5],
        dx: [1.0 / width as f32, 0.0],
        dy: [0.0, 1.0 / height as f32],
        hardware: 1,
        material: id.0,
        inward: [0.0; 2],
    };
    let mut input = dev
        .gpu_malloc::<Footprint>(MemoryKind::Upload, 1, "footprint")
        .unwrap();
    input.as_mut_slice().unwrap()[0] = tap;
    let out = dev
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 1, "tap")
        .unwrap();
    let mut root = dev
        .gpu_malloc::<Root>(MemoryKind::Upload, 1, "texture filter root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = Root {
        footprints: input.gpu(),
        out: out.gpu(),
        texture: b,
        sampler: orr_gpu::SamplerIndex(0),
        count: 1,
        taps: 1,
        streams: GpuPtr::NULL,
        modes: 0,
        pad: 0,
        materials,
        pad2: 0,
    };
    let p = pipeline(&texture_filter::MODULE, "slot_mean");
    let render = dev.queue(QueueKind::Render).unwrap();
    let start = Instant::now();
    while at_tail(&scenes, b) {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the tile's tiles never arrived"
        );
        let mut f = scenes
            .begin_frame(&FrameDesc {
                time: FrameTime::default(),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        let list = f.list().unwrap();
        list.dispatch(&p, root.gpu(), [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        let point = f.end(&[]).unwrap();
        render.wait(point, Duration::from_secs(10)).unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(at_tail(&scenes, a), "no tap landed on the other tile");
    assert_clean();
}

/// Cooks a MaterialX material on `file`, a name of tiles among `tiles` (each a file and its color)
/// with a `default` of `DEFAULT`, mounts it, and checks what it reads at each st of `cases`.
fn cooked_tiles_read(
    dir: &str,
    file: &str,
    tiles: &[(&str, [u8; 3])],
    cases: &[([f32; 2], [f32; 3])],
) {
    use orr_scene::orr_gpu::MemoryKind;
    use scene_suite::pipeline;
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};
    let _serial = serial();
    let dev = device();
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(dir);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for (name, rgb) in tiles {
        let file = std::fs::File::create(d.join(name)).unwrap();
        let mut enc = png::Encoder::new(file, 8, 8);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let texels: Vec<u8> = rgb.iter().copied().cycle().take(3 * 64).collect();
        enc.write_header()
            .unwrap()
            .write_image_data(&texels)
            .unwrap();
    }
    let path = d.join("m.mtlx");
    std::fs::write(
        &path,
        format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
  <image name="i" type="color3">
    <input name="file" type="filename" value="{file}"/>
    <input name="default" type="color3" value="0.25, 0.5, 0.75"/>
  </image>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" nodename="i"/>
  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
  </surfacematerial>
</materialx>
"#
        ),
    )
    .unwrap();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "u".into(),
                source: Source::MaterialX {
                    path,
                    material: "M".into(),
                    textures: Default::default(),
                },
            }],
        )],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);

    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(pack));
    ready(&mut scenes, m);
    let id = scenes.material(m, AssetId::from_name("u")).unwrap();
    let (_, _, materials) = global_tables(&scenes);
    let footprints: Vec<Footprint> = cases
        .iter()
        .map(|([s, t], _)| Footprint {
            uv: [*s, 1.0 - t],
            dx: [1.0 / 8.0, 0.0],
            dy: [0.0, 1.0 / 8.0],
            hardware: 1,
            material: id.0,
            inward: [0.0; 2],
        })
        .collect();
    let mut input = dev
        .gpu_malloc::<Footprint>(MemoryKind::Upload, footprints.len() as u64, "footprints")
        .unwrap();
    input.as_mut_slice().unwrap().copy_from_slice(&footprints);
    let out = dev
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, footprints.len() as u64, "taps")
        .unwrap();
    let mut root = dev
        .gpu_malloc::<Root>(MemoryKind::Upload, 1, "texture filter root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = Root {
        footprints: input.gpu(),
        out: out.gpu(),
        texture: TextureIndex(0),
        sampler: orr_gpu::SamplerIndex(0),
        count: footprints.len() as u32,
        taps: 1,
        streams: GpuPtr::NULL,
        modes: 0,
        pad: 0,
        materials,
        pad2: 0,
    };
    let p = pipeline(&texture_filter::MODULE, "slot_mean");
    let render = dev.queue(QueueKind::Render).unwrap();
    let mut f = scenes
        .begin_frame(&FrameDesc {
            time: FrameTime::default(),
            scenes: &[],
            low_latency: None,
        })
        .unwrap();
    let list = f.list().unwrap();
    list.dispatch(&p, root.gpu(), [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = f.end(&[]).unwrap();
    render.wait(point, Duration::from_secs(10)).unwrap();
    for ((st, want), got) in cases.iter().zip(out.as_slice().unwrap()) {
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 0.02,
                "at st {st:?}: {got:?} against {want:?}"
            );
        }
    }
    assert_clean();
}

/// What the `default` of `cooked_tiles_read` reads.
const DEFAULT: [f32; 3] = [0.25, 0.5, 0.75];

fn color(rgb: [u8; 3]) -> [f32; 3] {
    rgb.map(|c| f32::from(c) / 255.0)
}

/// Tile 1001 + U + 10 V fills st square U, V, which is uv u U..U + 1, v -V..1 - V under the cook's
/// top-left origin. A square without a file reads `default`.
#[test]
fn a_cooked_udim_material_reads_each_square_of_st_from_its_tile() {
    let (a, b, c) = ([200, 40, 40], [40, 200, 40], [40, 40, 200]);
    cooked_tiles_read(
        "cooked_udim",
        "t.&lt;UDIM&gt;.png",
        &[("t.1001.png", a), ("t.1002.png", b), ("t.1011.png", c)],
        &[
            ([0.5, 0.5], color(a)),
            ([1.5, 0.5], color(b)),
            ([0.5, 1.5], color(c)),
            ([1.5, 1.5], DEFAULT),
            ([2.5, 0.5], DEFAULT),
            ([-0.5, 0.5], DEFAULT),
            ([0.5, -0.5], DEFAULT),
        ],
    );
}

/// A `<UVTILE>` far from the rest reads where its name says. Its table is two tiles' worth:
/// `a_far_uvtile_takes_no_more_table_than_a_near_one` holds that.
#[test]
fn a_cooked_far_uvtile_reads_at_its_square_of_st() {
    let (a, b) = ([200, 40, 40], [40, 200, 40]);
    cooked_tiles_read(
        "cooked_far_uvtile",
        "t.&lt;UVTILE&gt;.png",
        &[("t.u1_v1.png", a), ("t.u40000_v40000.png", b)],
        &[
            ([0.5, 0.5], color(a)),
            ([39_999.5, 39_999.5], color(b)),
            ([39_998.5, 39_999.5], DEFAULT),
            ([39_999.5, 39_998.5], DEFAULT),
            ([1.5, 0.5], DEFAULT),
        ],
    );
}

/// Taps past the unit square of a streamed texture read, and load, the tiles the wrap takes
/// them to. Reading a tile takes its neighbors, so each case taps those too; frac(u) would name
/// other tiles, and the checked texels would never read.
#[test]
fn taps_past_the_edges_of_a_streamed_texture_load_the_tiles_they_read() {
    use scene_suite::shaders::texture_filter::Footprint;
    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(tiled_pack())));
    ready(&mut scenes, m);
    let (_, w, h, _, _) = TILED[0];
    let texture = scenes.texture(m, AssetId::from_name(TILED[0].0)).unwrap();
    let texels = tiled_texels(0, 0, 0);
    let at = |x: u32, y: u32| -> [f32; 4] {
        let i = 4 * (y * w + x) as usize;
        std::array::from_fn(|c| f32::from(texels[i + c]) / 255.0)
    };
    let tap = |u: f32, y: u32| Footprint {
        uv: [u, (y as f32 + 0.5) / h as f32],
        dx: [1.0 / w as f32, 0.0],
        dy: [0.0, 1.0 / h as f32],
        hardware: 1,
        material: 0,
        inward: [0.0; 2],
    };
    let center = |x: u32| (x as f32 + 0.5) / w as f32;
    // Per wrap, taps at u that read texel column x, and the first of them not to check.
    let mirrored: Vec<(f32, u32)> = (0..384).step_by(8).map(|x| (2.0 - center(x), x)).collect();
    let cases = [
        // Past the right edge, the first columns of tiles.
        (wrap::MIRROR, mirrored, 256 / 8),
        // Past the left edge, the first texel column.
        (wrap::CLAMP, vec![(-0.4, 0)], 1),
        // Past the right edge, the last, with the tiles beside it.
        (wrap::CLAMP, vec![(1.4, w - 1), (center(400), 400)], 2),
    ];
    for (mode, columns, checked) in cases {
        let binding = TextureBinding {
            wrap: [mode, wrap::CLAMP],
            filter: filter::MAG_NEAREST | filter::MIN_NEAREST | filter::MIP_NEAREST,
            ..TextureBinding::zeroed()
        };
        let rows = || (0..h).step_by(8);
        let footprints: Vec<Footprint> = columns
            .iter()
            .flat_map(|&(u, _)| rows().map(move |y| tap(u, y)))
            .collect();
        let want: Vec<[f32; 4]> = columns[..checked]
            .iter()
            .flat_map(|&(_, x)| rows().map(move |y| at(x, y)))
            .collect();
        let start = Instant::now();
        while bound_taps(&mut scenes, texture, &footprints, 1, &binding)[..want.len()] != want {
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "the tiles of {columns:?} never arrived under wrap {mode}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    assert_clean();
}

/// A repeating read filters across the seam, into tiles nothing else wants: taps on the first
/// tiles alone must load the last ones, and read no finer until they have.
#[test]
fn taps_at_a_repeating_edge_load_the_tiles_across_the_seam() {
    use scene_suite::shaders::texture_filter::Footprint;
    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(tiled_pack())));
    ready(&mut scenes, m);
    let (_, w, h, _, _) = TILED[0];
    let texture = scenes.texture(m, AssetId::from_name(TILED[0].0)).unwrap();
    let texels = tiled_texels(0, 0, 0);
    let at = |x: u32, y: u32, c: usize| f32::from(texels[4 * (y * w + x) as usize + c]) / 255.0;
    let footprints: Vec<Footprint> = (0..h)
        .map(|y| Footprint {
            uv: [0.0, (y as f32 + 0.5) / h as f32],
            dx: [1.0 / w as f32, 0.0],
            dy: [0.0, 1.0 / h as f32],
            hardware: 1,
            material: 0,
            inward: [0.0; 2],
        })
        .collect();
    let want: Vec<[f32; 4]> = (0..h)
        .map(|y| std::array::from_fn(|c| 0.5 * (at(0, y, c) + at(w - 1, y, c))))
        .collect();
    let binding = TextureBinding::zeroed();
    let start = Instant::now();
    loop {
        let got = bound_taps(&mut scenes, texture, &footprints, 1, &binding);
        if differences(&got, &want).0 < 0.01 {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the tiles across the seam never arrived"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
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
        s.set_focus(DVec3::new(0.0, 3.0, 0.0));
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
        .fill(Affine3::IDENTITY.into());
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
    s.transforms(group, i, 1).unwrap()[0] =
        Affine3::from_translation(Vec3::new(x, 0.0, 0.0)).into();
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

/// A streaming sheet's budgets: everything resident fits, nothing keeps whole.
fn streaming_desc() -> RendererDesc {
    RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            ..Budgets::default()
        },
        // Every copy traces a cached cut.
        rt_merge_pixels: 1e9,
        ..RendererDesc::default()
    }
}

#[test]
fn finer_cached_cuts_build_once_wanted_lock_their_pages_and_leave_after_their_hold() {
    let _serial = serial();
    let dev = device();
    if dev.caps().cluster.max_tris == 0 || dev.caps().cluster.max_partitions == 0 {
        return;
    }
    let gpu = Gpu(dev);
    let mut scenes = Scenes::new(dev, &streaming_desc()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(sheet_pack())));
    ready(&mut scenes, m);
    let model = scenes.model(m, AssetId::from_name(SHEET)).unwrap();
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    let mut group = None;
    streamed_frame(&mut scenes, scene, |s| {
        group = Some(place_copies(s, &[model]));
    });
    let group = group.unwrap();
    let (mesh, _) = mesh_of(&scenes, &gpu, model);
    assert!(mesh.fine_cuts > 0, "a streaming mesh has finer cuts");
    let built = |scenes: &Scenes| -> Vec<orr_scene::RtCutGpu> {
        let (mesh, _) = mesh_of(scenes, &gpu, model);
        let cuts: Vec<orr_scene::RtCutGpu> = gpu.values(mesh.rt_cuts, mesh.fine_cuts);
        cuts.into_iter().filter(|c| c.blas != 0).collect()
    };
    let a = Archive::from_memory(Arc::clone(sheet_pack())).unwrap();
    let me = entry(&a, AssetId::from_name(&format!("{SHEET}/mesh")));
    let info = MeshInfo::parse(&blob(&a, me, 0)).unwrap();
    let mut records = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        decode_page(&blob(&a, me, p.blob as usize), &mut page).unwrap();
        let view = orr_pack::mesh::PageView::new(&page, &info.header).unwrap();
        records.extend((0..view.cluster_count()).map(|i| view.cluster(i)));
    }
    // A cut's row of the cooked sets is the one holding its clusters.
    let check_rows = |cuts: &[orr_scene::RtCutGpu]| {
        for c in cuts {
            let t = info.cut_rows[c.row as usize];
            let held = records
                .iter()
                .filter(|r| {
                    let merge = info.header.merge_error;
                    in_cut(r.self_error, r.parent_error, r.flags, t, merge)
                })
                .count();
            assert_eq!(c.clusters as usize, held, "row {}", c.row);
        }
    };
    let cached: Vec<orr_scene::RtCutGpu> = gpu.values(mesh.rt_cuts, mesh.rt_cut_count);
    let first_cached = cached[mesh.fine_cuts as usize];
    assert_eq!(first_cached.row, 0);
    check_rows(&cached[mesh.fine_cuts as usize..]);
    let start = Instant::now();
    while built(&scenes).is_empty() {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "no finer cut was built"
        );
        streamed_frame(&mut scenes, scene, |_| {});
    }
    for c in built(&scenes) {
        assert!(c.clusters > 0, "a built cut holds clusters");
    }
    check_rows(&built(&scenes));
    // The copy leaves: once its hold passes the cut goes, and its pages may evict.
    streamed_frame(&mut scenes, scene, |s| move_copy(s, group, 0, 800.0));
    let start = Instant::now();
    while !built(&scenes).is_empty() {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the finer cut never left"
        );
        streamed_frame(&mut scenes, scene, |_| {});
    }
    assert_eq!(scenes.residency().failed, 0);
}

#[test]
fn an_edit_of_a_streaming_mesh_follows_its_landings_and_evictions() {
    let _serial = serial();
    let dev = device();
    let gpu = Gpu(dev);
    let bytes = sheet_pack();
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    let e = entry(&a, AssetId::from_name("kit/sheet/mesh"));
    let info = MeshInfo::parse(&blob(&a, e, 0)).unwrap();
    let unit = |p: &orr_pack::mesh::PageInfo| u64::from(p.resident_len).next_multiple_of(256);
    let pinned = info.header.pinned_pages as usize;
    let pinned_bytes: u64 = info.pages[..pinned].iter().map(unit).sum();
    let streamed_bytes: u64 = info.pages[pinned..].iter().map(unit).sum();
    let mut scenes = Scenes::new(
        dev,
        &RendererDesc {
            budgets: Budgets {
                mesh_resident_max: 0,
                mesh_page_pool: 2 * pinned_bytes + streamed_bytes,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        },
    )
    .unwrap();
    let _workers = Workers::start(&scenes, 2);
    let mounts = [
        scenes.mount(PackSource::Memory(Arc::clone(bytes))),
        scenes.mount(PackSource::Memory(Arc::clone(bytes))),
    ];
    let [source, other] = mounts.map(|m| {
        ready(&mut scenes, m);
        scenes.model(m, AssetId::from_name(SHEET)).unwrap()
    });
    let touched = [[0.0f32, 0.0, 0.0, 0.05]];
    let edit = scenes
        .create_edit(&orr_scene::EditDesc {
            model: source,
            touched: Some(&touched),
            bounds: None,
        })
        .unwrap();
    let start = Instant::now();
    while !matches!(scenes.edit_state(edit), LoadState::Ready) {
        assert!(start.elapsed() < Duration::from_secs(60), "edit pending");
        frame(&mut scenes);
    }
    let model = scenes.edit(edit).unwrap().model;
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    streamed_frame(&mut scenes, scene, |s| {
        place_copies(s, &[model]);
    });
    // Per page of the edit: a copy of its own, or what the source holds.
    let copies = |scenes: &Scenes| -> Vec<usize> {
        let (_, src) = mesh_of(scenes, &gpu, source);
        let (_, own) = mesh_of(scenes, &gpu, model);
        (0..src.len())
            .filter(|&p| {
                if own[p] == src[p] {
                    return false;
                }
                assert!(own[p] != 0, "page {p}: a copy is always resident");
                true
            })
            .collect()
    };
    let streamed_shared = |scenes: &Scenes, held: &[usize]| {
        let (_, src) = mesh_of(scenes, &gpu, source);
        (pinned..src.len())
            .filter(|p| !held.contains(p) && src[*p] != 0)
            .count()
    };
    let held = copies(&scenes);
    assert!(!held.is_empty(), "the edit holds copies");
    assert!(
        held.len() < info.pages.len(),
        "the edit shares untouched pages"
    );
    // The source's pages land: the edit's table follows them.
    let start = Instant::now();
    while streamed_shared(&scenes, &held) == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the source's pages never landed"
        );
        scenes.prefetch(source, 0);
        streamed_frame(&mut scenes, scene, |_| {});
        assert_eq!(copies(&scenes), held, "a landing reached the edit's table");
    }
    // Another mesh's pages push them out after their hold: the edit keeps its copies and
    // its shared entries go null with the source's.
    let start = Instant::now();
    while streamed_shared(&scenes, &held) > 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the source's pages never left"
        );
        scenes.prefetch(other, 0);
        streamed_frame(&mut scenes, scene, |_| {});
        assert_eq!(
            copies(&scenes),
            held,
            "an eviction reached the edit's table"
        );
    }
    scenes.destroy_edit(edit);
    assert_eq!(scenes.residency().failed, 0);
}

#[test]
fn tiles_apart_from_their_textures_others_move_beside_them_and_stay_exact() {
    let _serial = serial();
    let bytes = tiled_pack();
    let dev = device();
    let layouts: Vec<(u64, u64)> = (0..TILED.len()).map(|t| tiled_layout(dev, t)).collect();
    let tile = orr_pack::texture::TILE_BYTES as u64;
    let tails: u64 = layouts.iter().map(|l| l.1).sum();
    // Room for the flat texture's tiles, a few of the cube's and some others.
    let budget = tails + (layouts[0].0 + 14) * tile;
    let desc = RendererDesc {
        budgets: Budgets {
            texture_pool: budget,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(dev, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
    ready(&mut scenes, m);
    let index =
        |scenes: &Scenes, t: usize| scenes.texture(m, AssetId::from_name(TILED[t].0)).unwrap();
    let (a, b) = (index(&scenes, 0), index(&scenes, 2));
    // Mips one at a time, alternating, so each texture's tiles lie in several runs.
    let settle = |scenes: &mut Scenes, wanted: &[(TextureIndex, u32)]| {
        let start = Instant::now();
        for frames in 1.. {
            assert!(start.elapsed() < Duration::from_secs(60), "tiles pending");
            for &(t, finest) in wanted {
                scenes.prefetch_texture(m, t, finest);
            }
            frame(scenes);
            let r = *scenes.residency();
            let pool = *r.class(ResidencyClass::Textures);
            if frames > 1 && r.loads == 0 && r.dropped == 0 && pool.loading_bytes == 0 {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    settle(&mut scenes, &[(a, 1)]);
    settle(&mut scenes, &[(b, 1)]);
    settle(&mut scenes, &[(a, 0)]);
    // The array's coarser tiles push out the cube's and then some of the flat texture's; what
    // stays of that texture sits in two runs, and once the cube's slots are free it moves.
    let array = index(&scenes, 1);
    let start = Instant::now();
    while scenes.residency().tiles_moved == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "no tile apart from its texture's others moved"
        );
        scenes.prefetch_texture(m, array, 1);
        frame(&mut scenes);
        std::thread::sleep(Duration::from_millis(1));
    }
    stream_tiles(&mut scenes, m, &[0], budget);
    assert_tiled(&scenes, dev, m, 0, 0..9);
    assert_eq!(scenes.residency().failed, 0);
    assert_clean();
}

/// The neural pack's set: color mip 0 size, the latents' mip 0 size and layers.
const NEURAL_SIZE: [u32; 2] = [1024, 512];
const LATENT_SIZE: [u32; 2] = [512, 256];
const LATENT_LAYERS: u32 = 2;
const NEURAL_SET: &str = "neural/set";
const NEURAL_LATENTS: &str = "neural/set/latents";
/// The textures the set stands in for: base color (sRGB, channels 0..3) and a normal map
/// (channels 3..5).
const NEURAL_BASE: &str = "neural/base";
const NEURAL_NORMAL: &str = "neural/normal";

fn mix(i: u32) -> u32 {
    let mut x = i.wrapping_mul(0x9e37_79b9) ^ 0x85eb_ca6b;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ x >> 16
}

fn uniform(i: u32) -> f32 {
    (mix(i) >> 8) as f32 / (1u32 << 24) as f32
}

/// Latent texels of `mip` of `layer`: A4R4G4B4, little-endian.
fn latent_texels(mip: u32, layer: u32) -> Vec<u8> {
    let [w, h] = orr_pack::texture::mip_size(LATENT_SIZE[0], LATENT_SIZE[1], mip);
    (0..w * h)
        .flat_map(|i| (mix(i ^ mip << 24 ^ layer << 28) as u16).to_le_bytes())
        .collect()
}

fn latent_mips() -> u32 {
    orr_pack::texture::full_mip_count(LATENT_SIZE[0], LATENT_SIZE[1])
}

fn neural_constants() -> orr_pack::neural::Constants {
    use orr_pack::neural::{ColorMip, Constants};
    let mips = orr_pack::texture::full_mip_count(NEURAL_SIZE[0], NEURAL_SIZE[1]);
    let mut c: Constants = Zeroable::zeroed();
    for m in 0..mips {
        c.color_mips[m as usize] = ColorMip {
            latent_mip: m.saturating_sub(1).min(latent_mips() - 2) as i32,
            position_lod: m as f32 / 4.0,
            position_scale: 0.125 / (1u32 << m.min(3)) as f32,
            _pad: 0,
        };
    }
    c.width = NEURAL_SIZE[0] as i32;
    c.height = NEURAL_SIZE[1] as i32;
    c.mips = mips as i32;
    c.valid_channels = 0x1f;
    c.color_spaces = 0b01_01_01;
    c
}

/// Random weights: hidden matrices of E4M3 values within 0.25, small FP16 biases, and an
/// output layer centered on 0.5.
fn neural_net() -> orr_pack::neural::Fp8Network {
    use orr_pack::neural::{Fp8Network, LAYERS, OUTPUTS, e4m3_to_f32};
    let mut seed = 0x51u32;
    let mut next = || {
        seed += 1;
        mix(seed)
    };
    let hidden = std::array::from_fn(|l| {
        let (rows, cols) = LAYERS[l];
        let m = (0..rows * cols)
            .map(|_| {
                loop {
                    let b = next() as u8;
                    let v = e4m3_to_f32(b);
                    if v.abs() <= 0.25 {
                        break b;
                    }
                }
            })
            .collect();
        let b = (0..rows)
            .map(|_| half::f16::from_f32((next() >> 8) as f32 / (1u32 << 24) as f32 - 0.5))
            .collect();
        (m, b)
    });
    let output = (0..OUTPUTS * LAYERS[3].1)
        .map(|_| (next() % 255) as i32 - 127)
        .map(|v| v as i8)
        .collect();
    let scale: Vec<f32> = (0..OUTPUTS)
        .map(|r| 1.0 / (40_000.0 + 1000.0 * r as f32))
        .collect();
    let bias = scale.iter().map(|s| (0.5 / s) as i32).collect();
    Fp8Network {
        hidden,
        output,
        scale,
        bias,
    }
}

/// A memory pack of one neural set standing in for a base color and a normal map, their BC
/// entries, the set's latents and a material binding both textures.
fn neural_pack() -> &'static Arc<[u8]> {
    use orr_pack::neural::{NeuralHeader, NeuralInfo, NeuralTexture, NeuralWeights};
    use orr_pack::texture::vk;
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let mut w = ArchiveWriter::create(Target::Memory).unwrap();
        let tail = add_texture(
            &mut w,
            NEURAL_LATENTS,
            vk::A4R4G4B4_UNORM_PACK16,
            LATENT_SIZE,
            LATENT_LAYERS,
            false,
            latent_texels,
        );
        assert_eq!(tail, 2, "the latents stream two mips");
        for (name, format, channels) in [
            (NEURAL_BASE, vk::R8G8B8A8_SRGB, 4),
            (NEURAL_NORMAL, vk::R8G8_UNORM, 2),
        ] {
            add_texture(&mut w, name, format, NEURAL_SIZE, 1, false, |mip, _| {
                let [mw, mh] = orr_pack::texture::mip_size(NEURAL_SIZE[0], NEURAL_SIZE[1], mip);
                (0..mw * mh * channels).map(|i| bc_texel(i, mip)).collect()
            });
        }
        let constants = neural_constants();
        let info = NeuralInfo {
            header: NeuralHeader {
                library: [0, 10, 0],
                _pad: 0,
                interface: 0,
                file_schema: 0,
                width: NEURAL_SIZE[0],
                height: NEURAL_SIZE[1],
                mip_count: constants.mips as u8,
                channel_count: 5,
                latent_features: (4 * LATENT_LAYERS) as u8,
                _pad1: 0,
                bits_per_pixel: 0.0,
                latents: AssetId::from_name(NEURAL_LATENTS),
                textures: Span::default(),
                weights: Span::default(),
            },
            textures: vec![
                NeuralTexture {
                    texture: AssetId::from_name(NEURAL_BASE),
                    first_channel: 0,
                    channel_count: 3,
                    stored: NeuralTexture::SRGB,
                    _pad: 0,
                    psnr: 40.0,
                },
                NeuralTexture {
                    texture: AssetId::from_name(NEURAL_NORMAL),
                    first_channel: 3,
                    channel_count: 2,
                    stored: NeuralTexture::NORMAL,
                    _pad: 0,
                    psnr: 40.0,
                },
            ],
            weights: vec![(
                NeuralWeights {
                    kind: NeuralWeights::GENERIC_FP8,
                    blob: 1,
                    converted_bytes: 0,
                    constants: Span::default(),
                },
                bytemuck::bytes_of(&constants).to_vec(),
            )],
        };
        let head = info.to_bytes();
        let weights = neural_net().to_bytes();
        w.add(
            NEURAL_SET,
            Class::NeuralTextureSet as u16,
            orr_pack::neural::SCHEMA,
            orr_pack::Hash32([0; 32]),
            &[
                BlobIn {
                    mount: true,
                    ..BlobIn::new(&head)
                },
                BlobIn {
                    mount: true,
                    ..BlobIn::new(&weights)
                },
            ],
        )
        .unwrap();
        let mut m = Material::default();
        m.textures[slot::BASE_COLOR].texture = AssetId::from_name(NEURAL_BASE);
        m.textures[slot::NORMAL].texture = AssetId::from_name(NEURAL_NORMAL);
        w.add(
            "neural/material",
            Class::Material as u16,
            orr_pack::material::SCHEMA,
            orr_pack::Hash32([0; 32]),
            &[BlobIn {
                mount: true,
                ..BlobIn::new(bytemuck::bytes_of(&m))
            }],
        )
        .unwrap();
        w.finish().unwrap().unwrap()
    })
}

/// Byte `i` of mip `mip` of the neural pack's BC entries.
fn bc_texel(i: u32, mip: u32) -> u8 {
    mix(i.wrapping_add(mip << 26)) as u8
}

/// `x` rounded to the nearest E4M3 value, ties to even, as cooperative vectors convert;
/// values past 448 saturate.
fn to_e4m3(x: f32) -> f32 {
    let a = x.abs().min(448.0);
    if a == 0.0 {
        return 0.0;
    }
    let e = a.log2().floor().max(-6.0);
    let step = 2f32.powi(e as i32 - 3);
    let q = a / step;
    let f = q.floor();
    let d = q - f;
    let r = if d > 0.5 || (d == 0.5 && f % 2.0 == 1.0) {
        f + 1.0
    } else {
        f
    };
    (r * step).min(448.0).copysign(x)
}

fn to_f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// The set's decoder on the CPU, as the kernels run it: FP16 vectors, the hidden layers'
/// inputs converted to E4M3 with `fp8`, the output layer's scale folded into FP16 weights.
struct NeuralCpu {
    net: orr_pack::neural::Fp8Network,
    constants: orr_pack::neural::Constants,
    /// Per latent mip and layer, the texels.
    latents: Vec<Vec<Vec<u16>>>,
    fp8: bool,
}

impl NeuralCpu {
    fn new(fp8: bool) -> Self {
        let latents = (0..latent_mips())
            .map(|mip| {
                (0..LATENT_LAYERS)
                    .map(|layer| {
                        latent_texels(mip, layer)
                            .chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            net: neural_net(),
            constants: neural_constants(),
            latents,
            fp8,
        }
    }

    /// Features 0..4 of `layer` of latent mip `mip` filtered bilinearly at `uv`, repeating.
    fn latent(&self, mip: u32, layer: u32, uv: [f32; 2]) -> [f32; 4] {
        let [w, h] = orr_pack::texture::mip_size(LATENT_SIZE[0], LATENT_SIZE[1], mip);
        let texels = &self.latents[mip as usize][layer as usize];
        let x = uv[0] * w as f32 - 0.5;
        let y = uv[1] * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |dx: i32, dy: i32| {
            let tx = (x0 as i32 + dx).rem_euclid(w as i32) as u32;
            let ty = (y0 as i32 + dy).rem_euclid(h as i32) as u32;
            let t = texels[(ty * w + tx) as usize];
            // Blue, green, red, alpha: the order the decoder reads features in.
            [t & 15, t >> 4 & 15, t >> 8 & 15, t >> 12].map(|v| f32::from(v) / 15.0)
        };
        let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
        std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bottom = c[k] + (d[k] - c[k]) * fx;
            top + (bottom - top) * fy
        })
    }

    fn layer(&self, l: usize, x: &[f32]) -> Vec<f32> {
        use orr_pack::neural::{LAYERS, e4m3_to_f32};
        let (rows, cols) = LAYERS[l];
        let x: Vec<f32> = x
            .iter()
            .map(|&v| if self.fp8 { to_e4m3(v) } else { v })
            .collect();
        let (m, b) = &self.net.hidden[l];
        (0..rows)
            .map(|r| {
                let sum: f32 = (0..cols).map(|c| e4m3_to_f32(m[r * cols + c]) * x[c]).sum();
                let y = to_f16(sum + b[r].to_f32());
                let third = to_f16(1.0 / 3.0);
                to_f16(y.min(3.0) * to_f16(to_f16(y * third) + 0.5).clamp(0.0, 1.0))
            })
            .collect()
    }

    /// The outputs at `texel` (repeating) of color mip `mip`, sRGB channels decoded.
    fn decode(&self, texel: [i32; 2], mip: u32) -> [f32; 16] {
        use orr_pack::neural::{LAYERS, OUTPUTS};
        let [w, h] = orr_pack::texture::mip_size(NEURAL_SIZE[0], NEURAL_SIZE[1], mip);
        let tx = texel[0].rem_euclid(w as i32);
        let ty = texel[1].rem_euclid(h as i32);
        let uv = [(tx as f32 + 0.5) / w as f32, (ty as f32 + 0.5) / h as f32];
        let c = self.constants.color_mips[mip as usize];
        let mut x = vec![0.0f32; orr_pack::neural::INPUTS];
        for level in 0..2 {
            for layer in 0..LATENT_LAYERS {
                let f = self.latent(c.latent_mip as u32 + level, layer, uv);
                for k in 0..4 {
                    x[(16 * level + 4 * layer) as usize + k] = to_f16(f[k] * 2.0 - 1.0);
                }
            }
        }
        let mut p = [tx as f32 * c.position_scale, ty as f32 * c.position_scale];
        for wave in 0..3 {
            let e = [
                p[0].fract(),
                p[1].fract(),
                (p[0] + 0.25).fract(),
                (p[1] + 0.25).fract(),
            ];
            for k in 0..4 {
                x[32 + 4 * wave + k] = to_f16(e[k] * 2.0 - 1.0);
            }
            p = [p[0] * 2.0, p[1] * 2.0];
        }
        x[44] = to_f16(c.position_lod);
        x[45] = to_f16(c.position_lod);
        let h0 = self.layer(0, &x);
        let h1 = self.layer(1, &h0);
        let h2 = self.layer(2, &h1);
        let q: Vec<f32> = h2
            .iter()
            .map(|&v| (v * 80.0 - 113.0).round().clamp(-128.0, 127.0))
            .collect();
        let cols = LAYERS[3].1;
        let n = &self.net;
        std::array::from_fn(|r| {
            if r >= OUTPUTS {
                return 0.0;
            }
            let s = n.scale[r];
            let sum: f32 = (0..cols)
                .map(|c| to_f16(f32::from(n.output[r * cols + c]) * s) * q[c])
                .sum();
            let v = to_f16(sum + to_f16(n.bias[r] as f32 * s));
            if r < 3 {
                orr_color::srgb_decode(f64::from(v.clamp(0.0, 1.0))) as f32
            } else {
                v
            }
        })
    }

    /// Channels `first..first + count` as a texture reads them, at `texel` of `mip`.
    fn texel(&self, texel: [i32; 2], mip: u32, first: usize, count: usize) -> [f32; 4] {
        let o = self.decode(texel, mip);
        std::array::from_fn(|k| {
            if k < count {
                o[first + k]
            } else if k == 3 {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Bilinear over the texels of `mip` around `uv`.
    fn bilinear(&self, uv: [f32; 2], mip: u32, first: usize, count: usize) -> [f32; 4] {
        let [w, h] = orr_pack::texture::mip_size(NEURAL_SIZE[0], NEURAL_SIZE[1], mip);
        let x = uv[0] * w as f32 - 0.5;
        let y = uv[1] * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |dx: i32, dy: i32| self.texel([x0 as i32 + dx, y0 as i32 + dy], mip, first, count);
        let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
        std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bottom = c[k] + (d[k] - c[k]) * fx;
            top + (bottom - top) * fy
        })
    }

    /// The cubic B-spline filter of mip 0 at `uv`.
    fn bicubic(&self, uv: [f32; 2], first: usize, count: usize) -> [f32; 4] {
        let [w, h] = NEURAL_SIZE;
        let x = uv[0] * w as f32 - 0.5;
        let y = uv[1] * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let weights = |t: f32| {
            let g = 1.0 - t;
            [
                g * g * g / 6.0,
                (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0,
                (1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t * t * t) / 6.0,
                t * t * t / 6.0,
            ]
        };
        let (wx, wy) = (weights(x - x0), weights(y - y0));
        let mut out = [0.0f32; 4];
        for (j, wy) in wy.iter().enumerate() {
            for (i, wx) in wx.iter().enumerate() {
                let v = self.texel(
                    [x0 as i32 + i as i32 - 1, y0 as i32 + j as i32 - 1],
                    0,
                    first,
                    count,
                );
                for k in 0..4 {
                    out[k] += wx * wy * v[k];
                }
            }
        }
        out
    }
}

/// Runs the texture filter kernel once over `footprints` of `texture` with `taps` taps each.
fn neural_taps(
    scenes: &mut Scenes,
    texture: TextureIndex,
    footprints: &[scene_suite::shaders::texture_filter::Footprint],
    taps: u32,
) -> Vec<[f32; 4]> {
    // Modes of 0 leave the set's own filtering, and the sampler reads a texture's exact texels.
    let exact = TextureBinding {
        wrap: [wrap::CLAMP; 2],
        filter: filter::MAG_NEAREST | filter::MIN_NEAREST | filter::MIP_NEAREST,
        ..TextureBinding::zeroed()
    };
    filter_taps(scenes, texture, footprints, taps, &exact, 0)
}

/// `TextureSamplerGpu::modes` of `b`.
fn modes_of(b: &TextureBinding) -> u32 {
    u32::from(b.wrap[0]) | u32::from(b.wrap[1]) << 3 | u32::from(b.filter) << 6
}

/// The texture filter kernel, once over `footprints` of `texture` with `taps` taps each, read
/// as `binding` says.
fn bound_taps(
    scenes: &mut Scenes,
    texture: TextureIndex,
    footprints: &[scene_suite::shaders::texture_filter::Footprint],
    taps: u32,
    binding: &TextureBinding,
) -> Vec<[f32; 4]> {
    filter_taps(
        scenes,
        texture,
        footprints,
        taps,
        binding,
        modes_of(binding),
    )
}

/// The texture filter kernel, once over `footprints` of `texture` with `taps` taps each, through
/// `binding`'s sampler under `modes`.
fn filter_taps(
    scenes: &mut Scenes,
    texture: TextureIndex,
    footprints: &[scene_suite::shaders::texture_filter::Footprint],
    taps: u32,
    binding: &TextureBinding,
    modes: u32,
) -> Vec<[f32; 4]> {
    use orr_scene::orr_gpu::MemoryKind;
    use scene_suite::pipeline;
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};
    let dev = device();
    let mut input = dev
        .gpu_malloc::<Footprint>(MemoryKind::Upload, footprints.len() as u64, "footprints")
        .unwrap();
    input.as_mut_slice().unwrap().copy_from_slice(footprints);
    let out = dev
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, footprints.len() as u64, "taps")
        .unwrap();
    let mut root = dev
        .gpu_malloc::<Root>(MemoryKind::Upload, 1, "texture filter root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = Root {
        footprints: input.gpu(),
        out: out.gpu(),
        texture,
        sampler: dev.sampler(&orr_scene::binding_sampler(binding)).unwrap(),
        count: footprints.len() as u32,
        taps,
        streams: scenes.texture_streams(),
        modes,
        pad: 0,
        materials: GpuPtr::NULL,
        pad2: 0,
    };
    let p = pipeline(&texture_filter::MODULE, "mean");
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
    dev.queue(QueueKind::Render)
        .unwrap()
        .wait(point, Duration::from_secs(10))
        .unwrap();
    out.as_slice().unwrap().to_vec()
}

/// Footprints one mip 0 texel wide at the centers of a spread of mip 0 texels, read with
/// hardware filtering.
fn texel_centers() -> Vec<scene_suite::shaders::texture_filter::Footprint> {
    let [w, h] = NEURAL_SIZE;
    (0..2048u32)
        .map(|i| {
            let (x, y) = (mix(2 * i) % w, mix(2 * i + 1) % h);
            scene_suite::shaders::texture_filter::Footprint {
                uv: [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32],
                dx: [1.0 / w as f32, 0.0],
                dy: [0.0, 1.0 / h as f32],
                hardware: 1,
                material: 0,
                inward: [0.0; 2],
            }
        })
        .collect()
}

/// The largest and mean absolute difference over every channel.
fn differences(got: &[[f32; 4]], want: &[[f32; 4]]) -> (f32, f32) {
    let d: Vec<f32> = got
        .iter()
        .zip(want)
        .flat_map(|(g, w)| (0..4).map(move |k| (g[k] - w[k]).abs()))
        .collect();
    let max = d.iter().copied().fold(0.0, f32::max);
    (max, d.iter().sum::<f32>() / d.len() as f32)
}

/// Mounts the neural pack with `mode` and runs frames until it is ready.
fn mount_neural(mode: NeuralTextures) -> (Scenes, Workers, MountId) {
    let desc = RendererDesc {
        neural_textures: mode,
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(device(), &desc).unwrap();
    let workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(neural_pack())));
    ready(&mut scenes, m);
    (scenes, workers, m)
}

/// Prefetches every latent tile through `textures` until none is loading.
fn stream_latents(scenes: &mut Scenes, m: MountId, textures: &[TextureIndex]) {
    let start = Instant::now();
    for frames in 1.. {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "latent tiles pending"
        );
        for &t in textures {
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

#[test]
fn textures_a_neural_set_stands_in_for_decode_like_the_cpu_decoder_and_stream_only_the_latents() {
    let _serial = serial();
    let fp8 = device().caps().shader_float8;
    for (mode, fp8) in [
        (NeuralTextures::Fp16, false),
        (NeuralTextures::Fastest, fp8),
    ] {
        let (mut scenes, _workers, m) = mount_neural(mode);
        let base = scenes.texture(m, AssetId::from_name(NEURAL_BASE)).unwrap();
        let normal = scenes
            .texture(m, AssetId::from_name(NEURAL_NORMAL))
            .unwrap();
        let cpu = NeuralCpu::new(fp8);
        let footprints = texel_centers();
        let want = |mip: u32, first: usize, count: usize| -> Vec<[f32; 4]> {
            footprints
                .iter()
                .map(|f| cpu.bilinear(f.uv, mip, first, count))
                .collect()
        };
        // FP8 inputs differ from the CPU's where filtering rounds a value across an E4M3
        // step, which moves single outputs; their mean stays small.
        let (max_allowed, mean_allowed) = if fp8 { (0.15, 4e-3) } else { (0.02, 1e-3) };
        let close = |got: &[[f32; 4]], want: &[[f32; 4]]| {
            let (max, mean) = differences(got, want);
            (max < max_allowed && mean < mean_allowed, max, mean)
        };
        // Before any latent tile streams, taps read the color mips over the latents' tail:
        // latent mip 2 is first read by color mip 3.
        let (ok, max, mean) = close(
            &neural_taps(&mut scenes, base, &footprints, 1),
            &want(3, 0, 3),
        );
        assert!(
            ok,
            "{mode:?}: base color over the tail differs by {max} at most, {mean} on average"
        );
        // The taps' feedback streams the latent tiles they want, and no tile of the BC
        // entries, whose mip 0 alone is larger than every latent mip.
        for (texture, first, count, what) in [(base, 0, 3, "base color"), (normal, 3, 2, "normal")]
        {
            let finest = want(0, first, count);
            let start = Instant::now();
            loop {
                let (ok, max, mean) =
                    close(&neural_taps(&mut scenes, texture, &footprints, 1), &finest);
                if ok {
                    break;
                }
                assert!(
                    start.elapsed() < Duration::from_secs(30),
                    "{mode:?}: {what} at mip 0 still differs by {max} at most, {mean} on average"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let resident = scenes
            .residency()
            .class(ResidencyClass::Textures)
            .resident_bytes;
        let bc_mip0 = u64::from(NEURAL_SIZE[0] * NEURAL_SIZE[1]) * 4;
        assert!(
            resident < bc_mip0 / 2,
            "{mode:?}: {resident} bytes of textures resident, under the BC base color's mip 0"
        );
    }
    assert_clean();
}

#[test]
fn the_resolve_writes_the_color_a_neural_set_decodes() {
    use orr_color::RgbPrimaries;
    use orr_color::srgb_encode;
    use orr_scene::{Lens, Projection, TextureFilter, ViewDesc};

    const EXTENT: u32 = 128;
    let _serial = serial();
    let (mut scenes, _workers, m) = mount_neural(NeuralTextures::Fp16);
    let mesh = MeshInput {
        positions: vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: Some(vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ]),
        indices: vec![0, 1, 2, 0, 2, 3],
        sections: vec![Span { first: 0, count: 6 }],
        ..MeshInput::default()
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "quad".into(),
                source: Source::Mesh {
                    mesh,
                    materials: vec![Material::default()],
                    settings: MeshSettings::default(),
                },
            }],
        )],
    };
    let quad_pack = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let quad_mount = scenes.mount(PackSource::Memory(quad_pack));
    ready(&mut scenes, quad_mount);
    let model = scenes
        .model(quad_mount, AssetId::from_name("quad"))
        .unwrap();
    let material = scenes
        .material(m, AssetId::from_name("neural/material"))
        .unwrap();
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = scenes.create_view_history().unwrap();
    let list = [scene];
    let desc = FrameDesc {
        time: FrameTime::default(),
        scenes: &list,
        low_latency: None,
    };
    {
        let f = scenes.begin_frame(&desc).unwrap();
        let mut s = f.scene(scene).unwrap();
        s.material_sets(0, 1).unwrap()[0] = material;
        let group = s
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        s.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
        s.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first: 0,
            pose: None,
            material_set: Some(0),
            flags: InstanceFlags::RASTER,
            rt_range: 0.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
        drop(s);
        orr_scene::testing::wait(f.end(&[]).unwrap());
    }
    // A pixel covers two mip 0 texels along u and one along v, so the view reads color mip 1.
    let half_tan = 0.45f32.tan();
    let eye = 2.0 * EXTENT as f32 / (NEURAL_SIZE[0] as f32 * half_tan);
    let cpu = NeuralCpu::new(false);
    let to_rec2020 = RgbPrimaries::REC709
        .matrix_to(RgbPrimaries::REC2020)
        .as_mat3();
    let word = |c: [f32; 4]| {
        let linear = to_rec2020 * (Vec3::new(c[0], c[1], c[2]) * Material::default().base_color[0]);
        linear
            .to_array()
            .map(|v| srgb_encode(f64::from(v.clamp(0.0, 1.0))) as f32)
    };
    let want: Vec<[f32; 3]> = (0..EXTENT * EXTENT)
        .map(|i| {
            let at = |p: u32| (2.0 * (p as f32 + 0.5) / EXTENT as f32 - 1.0) * eye * half_tan;
            let uv = [0.5 * (at(i % EXTENT) + 1.0), 0.5 * (at(i / EXTENT) + 1.0)];
            word(cpu.bilinear(uv, 1, 0, 3))
        })
        .collect();
    let start = Instant::now();
    loop {
        let mut f = scenes.begin_frame(&desc).unwrap();
        f.scene(scene)
            .unwrap()
            .set_focus(DVec3::new(0.0, 0.0, f64::from(eye)));
        let view = f
            .raster_view(&ViewDesc {
                scene,
                history,
                position: DVec3::new(0.0, 0.0, f64::from(eye)),
                rotation: Quat::IDENTITY,
                projection: Projection {
                    fov_y: 0.9,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: orr_math::UVec2::splat(EXTENT),
                output_extent: orr_math::UVec2::splat(EXTENT),
                jitter: Vec2::ZERO,
                camera_cut: true,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
            })
            .unwrap();
        orr_scene::testing::wait(f.end(&[]).unwrap());
        let bytes = read_texture_on(
            device(),
            &view.gbuf_a,
            &TextureRegion::whole_mip(view.gbuf_a.desc(), 0),
        );
        let a: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
        let got: Vec<[f32; 3]> = a
            .iter()
            .map(|w| [0, 8, 16].map(|s| ((w[0] >> s) & 255) as f32 / 255.0))
            .collect();
        let d: Vec<f32> = got
            .iter()
            .zip(&want)
            .flat_map(|(g, w)| (0..3).map(move |k| (g[k] - w[k]).abs()))
            .collect();
        let max = d.iter().copied().fold(0.0, f32::max);
        let mean = d.iter().sum::<f32>() / d.len() as f32;
        if max < 0.03 && mean < 4e-3 {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the resolve's base color differs from the decoded texels by {max} at most, {mean} on average"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_clean();
}

#[test]
fn stochastic_taps_of_a_neural_texture_average_to_its_filters() {
    use scene_suite::shaders::texture_filter::Footprint;
    let _serial = serial();
    let (mut scenes, _workers, m) = mount_neural(NeuralTextures::Fp16);
    let base = scenes.texture(m, AssetId::from_name(NEURAL_BASE)).unwrap();
    stream_latents(&mut scenes, m, &[base]);
    let cpu = NeuralCpu::new(false);
    let [w, h] = NEURAL_SIZE.map(|v| v as f32);
    let at = |i: u32, texels: f32| Footprint {
        uv: [uniform(3 * i), uniform(3 * i + 1)],
        dx: [texels / w, 0.0],
        dy: [0.0, texels / h],
        hardware: 0,
        material: 0,
        inward: [0.0; 2],
    };
    // A quarter texel: the bicubic B-spline of mip 0. Two texels: bilinear mip 1. Three:
    // trilinear between mips 1 and 2 by log2(3).
    type Filter<'a> = &'a dyn Fn([f32; 2]) -> [f32; 4];
    let cases: [(f32, Filter<'_>); 3] = [
        (0.25, &|uv| cpu.bicubic(uv, 0, 3)),
        (2.0, &|uv| cpu.bilinear(uv, 1, 0, 3)),
        (3.0, &|uv| {
            let t = 3f32.log2() - 1.0;
            let (a, b) = (cpu.bilinear(uv, 1, 0, 3), cpu.bilinear(uv, 2, 0, 3));
            std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
        }),
    ];
    for (texels, filter) in cases {
        let footprints: Vec<Footprint> = (0..64).map(|i| at(i, texels)).collect();
        let got = neural_taps(&mut scenes, base, &footprints, 4096);
        let want: Vec<[f32; 4]> = footprints.iter().map(|f| filter(f.uv)).collect();
        let (max, mean) = differences(&got, &want);
        assert!(
            max < 0.02 && mean < 4e-3,
            "a footprint {texels} texels wide averages {max} from its filter at most, {mean} on average"
        );
    }
    assert_clean();
}

/// A neural set reads as the sampler of the binding it stands in for: taps past the edges fold
/// back to the texels the wrap names, or read black past a border, and a nearest filter reads
/// the one texel under the tap.
#[test]
fn neural_taps_follow_the_wrap_and_filter_of_their_binding() {
    let _serial = serial();
    let (mut scenes, _workers, m) = mount_neural(NeuralTextures::Fp16);
    let base = scenes.texture(m, AssetId::from_name(NEURAL_BASE)).unwrap();
    stream_latents(&mut scenes, m, &[base]);
    let bound = |wrap: u8, filter: u8| TextureBinding {
        wrap: [wrap; 2],
        filter,
        ..TextureBinding::zeroed()
    };
    let centers = texel_centers();
    let moved = |to: &dyn Fn(f32) -> f32| -> Vec<_> {
        let mut f = centers.clone();
        for p in &mut f {
            p.uv = p.uv.map(to);
        }
        f
    };
    let plain = neural_taps(&mut scenes, base, &centers, 1);
    let cases: [(u8, &dyn Fn(f32) -> f32); 2] =
        [(wrap::REPEAT, &|u| u + 3.0), (wrap::MIRROR, &|u| 2.0 - u)];
    for (mode, to) in cases {
        let got = bound_taps(&mut scenes, base, &moved(to), 1, &bound(mode, 0));
        let (max, mean) = differences(&got, &plain);
        assert!(
            max < 0.01 && mean < 1e-3,
            "wrap {mode} moves taps by {max} at most, {mean} on average"
        );
    }
    let past = bound_taps(
        &mut scenes,
        base,
        &moved(&|u| u + 3.0),
        1,
        &bound(wrap::BORDER, 0),
    );
    assert!(past.iter().flatten().all(|&c| c == 0.0), "past a border");
    let edge = moved(&|u| u.min(0.0) - 1.0);
    let clamped = bound_taps(&mut scenes, base, &edge, 1, &bound(wrap::CLAMP, 0));
    assert!(
        clamped.iter().all(|c| c == &clamped[0]),
        "every tap past a corner reads the corner texel"
    );
    let off_center = moved(&|u| u + 0.3 / NEURAL_SIZE[0] as f32);
    let nearest = bound_taps(
        &mut scenes,
        base,
        &off_center,
        1,
        &bound(wrap::REPEAT, filter::MAG_NEAREST | filter::MIN_NEAREST),
    );
    let (max, mean) = differences(&nearest, &plain);
    assert!(
        max < 0.01 && mean < 1e-3,
        "a nearest filter leaves {max} at most, {mean} on average, of its neighbors"
    );
    assert_clean();
}

#[test]
fn with_neural_textures_off_textures_sample_their_bc_entries() {
    let _serial = serial();
    let (mut scenes, _workers, m) = mount_neural(NeuralTextures::Off);
    assert!(
        scenes
            .texture(m, AssetId::from_name(NEURAL_LATENTS))
            .is_err(),
        "the latents of a set the mount does not decode stay out of it"
    );
    let normal = scenes
        .texture(m, AssetId::from_name(NEURAL_NORMAL))
        .unwrap();
    stream_latents(&mut scenes, m, &[normal]);
    let footprints = texel_centers();
    let got = neural_taps(&mut scenes, normal, &footprints, 1);
    let [w, _] = NEURAL_SIZE;
    for (f, g) in footprints.iter().zip(&got) {
        let x = (f.uv[0] * NEURAL_SIZE[0] as f32) as u32;
        let y = (f.uv[1] * NEURAL_SIZE[1] as f32) as u32;
        let i = 2 * (y * w + x);
        let want = [bc_texel(i, 0), bc_texel(i + 1, 0)].map(|b| f32::from(b) / 255.0);
        assert!(
            (g[0] - want[0]).abs() < 1e-3 && (g[1] - want[1]).abs() < 1e-3,
            "the normal map reads its BC texels: {g:?} against {want:?}"
        );
    }
    assert_clean();
}
