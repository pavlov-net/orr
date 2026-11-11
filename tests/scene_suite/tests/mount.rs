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
    MeshSettings, Placement, PrototypeSpec, Source, TextureSettings, Usage,
};
use orr_math::{Affine3, DVec3, Quat, Vec2, Vec3};
use orr_pack::instance::InstancePageInfo;
use orr_pack::material::{Material, TextureBinding, filter, slot, wrap};
use orr_pack::mesh::{MeshInfo, decode_page, in_cut};
use orr_pack::model::ModelInfo;
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, ArchiveWriter, AssetId, BlobIn, Class, Codec, Entry, Span, Target};
use orr_scene::PageId;
use orr_scene::orr_gpu::{
    Device, DeviceDesc, GpuAddress, GpuPtr, Hazards, HostTimeline, QueueKind, QueueRequest, Stages,
    Texture, TextureIndex, TextureRegion,
};
use orr_scene::testing::{
    assert_clean, device, global_tables, mounted_texture, read_buffer_on, read_texture_on,
};
use orr_scene::{
    BoundTexture, Budgets, EmissiveGpu, Error, FrameDesc, FrameTime, GroupWriter, InstanceFlags,
    InstancesDesc, LoadState, MaterialGpu, MaterialId, MeshGpu, ModelGpu, ModelId, MountId,
    NeuralTextures, PackSource, PoolSize, RendererDesc, ResidencyClass, SceneDesc, SceneId,
    SceneRoot, SceneWriter, Scenes, TransformGroupDesc, material_flags,
};
use scene_suite::Workers;

mod paged;

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
    let lists = orr_scene::testing::model_materials(scenes);
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
                let section_materials = gpu.values(
                    lists.offset(u64::from(model.materials)),
                    model.section_count,
                );
                for (&got, &asset) in section_materials.iter().zip(&info.materials) {
                    assert_eq!(got, scenes.material(m, asset).unwrap().0);
                    out.push(material(asset));
                }
                let mesh: MeshGpu = gpu.value(meshes.offset(u64::from(model.mesh)));
                let me = entry(a, info.header.mesh);
                let blob0 = blob(a, me, 0);
                let mi = MeshInfo::parse(&blob0).unwrap();
                let h = &mi.header;
                let sets = [(&mesh.emissive, &h.emissive), (&mesh.casters, &h.casters)];
                let tables = [
                    (mesh.levels.address().0, h.levels),
                    (mesh.nodes.address().0, h.nodes),
                    (mesh.node_table.address().0, h.node_pages),
                    (mesh.groups.address().0, h.groups),
                    (mesh.roots.address().0, h.roots),
                ]
                .into_iter()
                .chain(sets.into_iter().flat_map(|(t, s)| {
                    [
                        (t.sets.address().0, s.sets),
                        (t.ranks.address().0, s.ranks),
                        (t.clusters.address().0, s.clusters),
                        (t.cdf.address().0, s.cdf),
                        (t.index.address().0, s.index),
                        (t.nodes.address().0, s.nodes),
                    ]
                }));
                let mut bases: Vec<u64> = tables
                    .filter(|(_, s)| s.count != 0)
                    .map(|(address, s)| address - u64::from(s.first))
                    .collect();
                bases.dedup();
                let [base] = bases[..] else {
                    panic!("tables of {} off one blob 0: {bases:x?}", a.name(me));
                };
                assert_eq!(gpu.bytes(base, blob0.len()), blob0);
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
    let short: Arc<[u8]> = bytes[..bytes.len() - index_len / 4].into();
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

/// A memory pack of `count` whole textures, which no material binds: every mip of each is
/// staged at mount. Returns it with the bytes of their texels.
fn whole_textures(count: usize) -> (Arc<[u8]>, u64) {
    use orr_pack::texture::{mip_size, vk};
    const EDGE: u32 = 256;
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    let mut bytes = 0;
    for t in 0..count {
        let texels = |mip: u32, _| -> Vec<u8> {
            let [mw, mh] = mip_size(EDGE, EDGE, mip);
            (0..4 * mw * mh)
                .map(|i| (i ^ (i >> 9) ^ (mip * 31) ^ (t as u32 * 7)) as u8)
                .collect()
        };
        let name = format!("whole/{t}");
        let tail = add_texture(
            &mut w,
            &name,
            vk::R8G8B8A8_UNORM,
            [EDGE; 2],
            1,
            false,
            texels,
        );
        assert!(tail > 0, "{name} has mips above its tail");
        bytes += (0..orr_pack::texture::full_mip_count(EDGE, EDGE))
            .map(|m| texels(m, 0).len() as u64)
            .sum::<u64>();
    }
    (w.finish().unwrap().unwrap(), bytes)
}

#[test]
fn staged_mounts_past_the_in_flight_limit_copy_in_batches_and_place_the_same_assets() {
    use orr_scene::orr_gpu::MemoryKind;
    let _serial = serial();
    let staged = staged_device();
    let bytes = pack();
    let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
    let (whole, texel_bytes) = whole_textures(64);
    let b = Archive::from_memory(Arc::clone(&whole)).unwrap();
    {
        let desc = RendererDesc {
            budgets: Budgets {
                in_flight_bytes: 512 << 10,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(&staged, &desc).unwrap();
        let _workers = Workers::start(&scenes, 2);
        let upload = || staged.memory_report().kind(MemoryKind::Upload).bytes;
        for _ in 0..4 {
            frame(&mut scenes);
        }
        let base = upload();
        let m = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
        let n = scenes.mount(PackSource::Memory(Arc::clone(&whole)));
        // Staging frees once its copies complete and a frame has collected it, so frames run
        // back to back.
        let mut peak = 0;
        let start = Instant::now();
        while [m, n]
            .iter()
            .any(|&k| matches!(scenes.mount_state(k), LoadState::Pending))
        {
            assert!(start.elapsed() < Duration::from_secs(60), "mounts pending");
            frame(&mut scenes);
            peak = peak.max(upload().saturating_sub(base));
        }
        assert!(
            peak < texel_bytes / 4,
            "{peak} bytes of upload memory at once while staging {texel_bytes}"
        );
        ready(&mut scenes, n);
        snapshot(&scenes, &Gpu(&staged), n, &b);
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

/// The resident cluster table holds no memory with nothing mounted, binds memory for a
/// mount's clusters, and gives it all back some frames after the mount has gone.
#[test]
fn the_resident_cluster_table_binds_what_mounts_hold_and_unbinds_it_after_they_go() {
    let _serial = serial();
    let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let table = |s: &Scenes| {
        let pools = s.memory().pools;
        *pools
            .iter()
            .find(|p| p.name == "resident clusters")
            .expect("the resident cluster table's memory")
    };
    frame(&mut scenes);
    assert_eq!((table(&scenes).bytes, table(&scenes).used), (0, 0));
    let m = scenes.mount(PackSource::Memory(Arc::clone(pack())));
    ready(&mut scenes, m);
    frame(&mut scenes);
    let held = table(&scenes);
    assert!(held.used > 0, "{held:?}");
    scenes.unmount(m);
    let start = Instant::now();
    while table(&scenes).bytes > 0 {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the table kept {:?} after the mount went",
            table(&scenes)
        );
        frame(&mut scenes);
    }
    assert_eq!(table(&scenes).used, 0);
    assert_clean();
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
    let copy = InstancesDesc {
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
    };
    w.add_instances(&copy).unwrap();
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
    // The model table no longer holds the model's record.
    let mut f = scenes
        .begin_frame(&FrameDesc {
            time: FrameTime::default(),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let placed = f.scene(s).unwrap().add_instances(&copy);
    assert!(matches!(placed, Err(Error::NotResident(id)) if id == panel));
    f.update_scenes(&mut |_| {}).unwrap();
    f.end(&[]).unwrap();
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
    let state = |gpu: &Gpu<'_>| -> (Vec<u8>, usize) {
        // The slot tables move when the scene takes more slots than they hold.
        let root: SceneRoot = gpu.value(root);
        let instances: Vec<orr_scene::InstanceGpu> = gpu.values(root.instances, root.part_base);
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

/// More copies at one position than a PTLAS partition holds cook into runs of records in
/// order, no run past a partition, and a scene traces every copy from a partition per run.
#[test]
fn copies_at_one_position_past_a_partition_take_one_partition_a_run() {
    const COPIES: u32 = 2500;
    let _serial = serial();
    let panel = AssetSpec {
        name: "kit/panel".into(),
        source: Source::Mesh {
            mesh: grid(8),
            materials: vec![Material::default(); 2],
            settings: MeshSettings::default(),
        },
    };
    let scatter = AssetSpec {
        name: "scatter".into(),
        source: Source::Instances(Instances {
            records: (0..COPIES)
                .map(|_| Placement {
                    position: [3.0, 0.0, 3.0],
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
        archives: vec![ArchiveSpec::new(Target::Memory, vec![panel, scatter])],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let entry = archive
        .entries()
        .iter()
        .find(|e| e.class == Class::InstancePage as u16)
        .unwrap();
    let head = &archive.blobs(entry)[0];
    let mut blob = vec![0; head.raw_len as usize];
    archive.read(head, &mut blob).unwrap();
    let info = InstancePageInfo::parse(&blob).unwrap();
    assert_eq!(u32::from(info.header.record_count), COPIES);
    assert_eq!(info.partitions, [0, 1000, 2000]);

    let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let m = scenes.mount(PackSource::Memory(bytes));
    ready(&mut scenes, m);
    let page = scenes.instance_page(m, entry.id).unwrap();
    // One frame of the scene, then one that takes in its figures.
    let frame_with = |scenes: &mut Scenes, write: &mut dyn FnMut(&mut SceneWriter<'_>)| {
        let mut f = scenes
            .begin_frame(&FrameDesc {
                time: FrameTime::default(),
                scenes: &[s],
                low_latency: None,
            })
            .unwrap();
        {
            let mut w = f.scene(s).unwrap();
            w.set_focus(DVec3::new(3.0, 3.0, 3.0));
            write(&mut w);
        }
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| root = c.root).unwrap();
        orr_scene::testing::wait(f.end(&[]).unwrap());
        let f = scenes
            .begin_frame(&FrameDesc {
                time: FrameTime::default(),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        orr_scene::testing::wait(f.end(&[]).unwrap());
        (root, scenes.stats().rt)
    };
    let (root, added) = frame_with(&mut scenes, &mut |w| {
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
        .unwrap();
    });
    let gpu = Gpu(device());
    let root: SceneRoot = gpu.value(root);
    // Every record takes a slot once the page's records land, and then a PTLAS write.
    let start = Instant::now();
    let mut written = added.writes;
    loop {
        let (_, rt) = frame_with(&mut scenes, &mut |_| {});
        written += rt.writes;
        assert_eq!((rt.unplaced, rt.dropped), (0, 0));
        assert_eq!(rt.partitions, 3, "a partition per run");
        let instances: Vec<orr_scene::InstanceGpu> = gpu.values(root.instances, root.part_base);
        let placed = instances.iter().filter(|i| i.flags != 0).count();
        if placed == COPIES as usize && written >= COPIES && rt.deferred == 0 {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "{placed} copies placed, {written} PTLAS writes, {} deferred",
            rt.deferred
        );
    }
    assert_clean();
}

const DOME_MESH: &str = "kit/dome/mesh";

/// [`pack`] with the dome's hierarchy split into its top page and one streamed page.
fn paged_hierarchy() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| paged::split_hierarchy(pack(), AssetId::from_name(DOME_MESH)))
}

/// The paged dome's blob 0 and the bytes of its two node pages.
fn paged_dome() -> (MeshInfo, Vec<Vec<u8>>) {
    let a = Archive::from_memory(Arc::clone(paged_hierarchy())).unwrap();
    let mesh = entry(&a, AssetId::from_name(DOME_MESH));
    let info = MeshInfo::parse(&blob(&a, mesh, 0)).unwrap();
    let nodes = info
        .node_pages
        .iter()
        .map(|p| blob(&a, mesh, p.blob as usize))
        .collect();
    (info, nodes)
}

/// A mesh page budget on `dev` for two mounts of [`paged_hierarchy`], whose dome is `info`
/// with node pages `nodes`: what both mounts place, both streamed node pages, and the dome's
/// streamed pages of one, with a CLAS pool their CLAS as [`sheet_budget`] counts them.
fn dome_budget(info: &MeshInfo, nodes: &[Vec<u8>], dev: &Device) -> u64 {
    let desc = RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(dev, &desc).unwrap();
    let workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(paged_hierarchy())));
    ready(&mut scenes, m);
    let placed = scenes
        .residency()
        .class(ResidencyClass::MeshPages)
        .resident_bytes;
    drop(workers);
    let pinned = info.header.pinned_pages as usize;
    let pinned_bytes: u64 = info.pages[..pinned]
        .iter()
        .map(|p| u64::from(p.resident_len).next_multiple_of(256))
        .sum();
    2 * (placed + nodes[1].len() as u64) + sheet_budget(info, dev, 2) - 2 * pinned_bytes
}

/// The node page table and the mesh page table of `model`'s mesh.
fn tables(scenes: &Scenes, gpu: &Gpu<'_>, model: ModelId) -> (Vec<u64>, Vec<u64>) {
    let (models, meshes, _) = global_tables(scenes);
    let model: ModelGpu = gpu.value(models.offset(u64::from(model.0)));
    let mesh: MeshGpu = gpu.value(meshes.offset(u64::from(model.mesh)));
    (
        gpu.values(mesh.node_pages, mesh.node_page_count),
        gpu.values(mesh.pages, mesh.page_count),
    )
}

/// Panics unless the node page of every group of the paged dome `info` whose page is resident
/// in `models`' tables is resident beside it, and the mesh page budget holds what is resident
/// and loading.
fn check_dome(scenes: &Scenes, gpu: &Gpu<'_>, models: &[ModelId], info: &MeshInfo, budget: u64) {
    for &m in models {
        let (nodes, pages) = tables(scenes, gpu, m);
        for g in &info.groups {
            assert!(
                pages[g.page as usize] == 0 || nodes[g.node_page as usize] != 0,
                "page {} is resident without node page {}",
                g.page,
                g.node_page
            );
        }
    }
    let pool = *scenes.residency().class(ResidencyClass::MeshPages);
    assert!(pool.resident_bytes + pool.loading_bytes <= budget);
}

/// Runs frames, prefetching `model`, until its streamed node page and pages are resident,
/// checking `models` as [`check_dome`] does each frame; returns the frames it took.
fn stream_in(
    scenes: &mut Scenes,
    gpu: &Gpu<'_>,
    model: ModelId,
    models: &[ModelId],
    (info, budget): (&MeshInfo, u64),
) -> u32 {
    let start = Instant::now();
    let mut frames = 0;
    loop {
        let (nodes, pages) = tables(scenes, gpu, model);
        if !nodes.contains(&0) && !pages.contains(&0) {
            return frames;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the dome's pages never streamed in"
        );
        scenes.prefetch(model, 0);
        frame(scenes);
        frames += 1;
        check_dome(scenes, gpu, models, info, budget);
    }
}

/// Twelve mounts of the paged dome stream their node pages into three blocks of four. The
/// mesh page budget then falls to three domes' worth while one dome of each block is asked
/// for: the others' pages go, and the unit pool empties a block by moving its node page. A
/// moved page's table entry names its new slot, which holds the cooked page.
#[test]
fn node_pages_move_out_of_draining_blocks_and_their_tables_follow() {
    let _serial = serial();
    let dev = device();
    let gpu = Gpu(dev);
    let bytes = paged_hierarchy();
    let (info, nodes) = paged_dome();
    let slot = (nodes[1].len() as u64).next_multiple_of(256);
    let desc = RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            mesh_page_pool: PoolSize::Bytes(12 * dome_budget(&info, &nodes, dev)),
            texture_pool: PoolSize::Bytes(64 << 20),
            instance_page_pool: 1 << 20,
            in_flight_bytes: 1 << 20,
            stream_block: 4 * slot,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(dev, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let class = |scenes: &Scenes| *scenes.residency().class(ResidencyClass::MeshPages);
    let models: Vec<ModelId> = (0..12)
        .map(|_| {
            let m = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
            ready(&mut scenes, m);
            scenes.model(m, AssetId::from_name("kit/dome")).unwrap()
        })
        .collect();
    let pinned = class(&scenes).resident_bytes;
    for &m in &models {
        stream_in(&mut scenes, &gpu, m, &models, (&info, u64::MAX));
    }
    let unit_pool = |scenes: &Scenes| {
        let memory = scenes.memory();
        *memory
            .pools
            .iter()
            .find(|p| p.name == "stream units")
            .expect("the unit pool")
    };
    let full = unit_pool(&scenes);
    // A device that allows few memory allocations takes larger blocks.
    if (full.blocks, full.bytes) != (3, 12 * slot) {
        return;
    }
    let before: Vec<u64> = models
        .iter()
        .map(|&m| tables(&scenes, &gpu, m).0[1])
        .collect();
    let streamed = (class(&scenes).resident_bytes - pinned) / 12;
    let budget = pinned + 3 * streamed + streamed / 2;
    scenes.lower_budget(ResidencyClass::MeshPages, budget);
    let kept = [models[0], models[4], models[8]];
    let start = Instant::now();
    let (mut settled, mut drained) = (0, 0);
    while settled < 8 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the unit pool never settled: {:?}",
            unit_pool(&scenes)
        );
        for &m in &kept {
            scenes.prefetch(m, 0);
        }
        frame(&mut scenes);
        for &m in &kept {
            let (table, pages) = tables(&scenes, &gpu, m);
            assert!(!pages.contains(&0), "a page that was asked for left");
            assert_eq!(gpu.bytes(table[1], nodes[1].len()), nodes[1]);
        }
        let fill = unit_pool(&scenes);
        drained = drained.max(fill.draining);
        let idle = class(&scenes).resident_bytes <= budget && fill.draining == 0;
        settled = if idle && fill.retired == 0 {
            settled + 1
        } else {
            0
        };
    }
    let fill = unit_pool(&scenes);
    assert!(drained > 0, "no block drained: {fill:?}");
    assert!(fill.blocks < 3, "{fill:?}");
    let moved = [0, 4, 8]
        .iter()
        .filter(|&&i| tables(&scenes, &gpu, models[i]).0[1] != before[i])
        .count();
    assert!(moved > 0, "no node page moved");
    assert_eq!(scenes.residency().failed, 0);
    assert_clean();
}

#[test]
fn hierarchy_pages_stream_with_the_pages_below_them_and_leave_after_them() {
    let _serial = serial();
    let bytes = paged_hierarchy();
    let (info, nodes) = paged_dome();
    let pinned = info.header.pinned_pages as usize;
    assert!(pinned < info.pages.len(), "the dome has streamed pages");
    assert!(
        info.groups.iter().any(|g| g.node_page == 1),
        "groups below the top page"
    );
    let path = write_pack("paged.orr", bytes);
    let staged = staged_device();
    for (dev, positioned_reads) in [(device(), false), (device(), true), (&staged, false)] {
        let budget = dome_budget(&info, &nodes, dev);
        let desc = RendererDesc {
            positioned_reads,
            budgets: Budgets {
                mesh_resident_max: 0,
                mesh_page_pool: PoolSize::Bytes(budget),
                ..Budgets::default()
            },
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(dev, &desc).unwrap();
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
            let (table, pages) = tables(&scenes, &gpu, m);
            assert_eq!(gpu.bytes(table[0], nodes[0].len()), nodes[0]);
            assert_eq!(table[1], 0, "streamed pages wait for a request");
            assert!(pages[pinned..].iter().all(|&p| p == 0));
        }

        stream_in(&mut scenes, &gpu, models[0], &models, (&info, budget));
        let first = tables(&scenes, &gpu, models[0]).0;
        assert_eq!(gpu.bytes(first[1], nodes[1].len()), nodes[1]);

        // The second mount's pages fit once the first one's have left after their hold, each
        // node page after the pages below it.
        let frames = stream_in(&mut scenes, &gpu, models[1], &models, (&info, budget));
        assert!(frames > 30, "evicted after {frames} frames");
        let (_, pages) = tables(&scenes, &gpu, models[0]);
        assert!(pages[pinned..].contains(&0), "the first mount's pages left");
        let second = tables(&scenes, &gpu, models[1]).0;
        assert_eq!(gpu.bytes(second[1], nodes[1].len()), nodes[1]);
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
    let (info, nodes) = paged_dome();
    let dev = device();
    let budget = dome_budget(&info, &nodes, dev);
    let desc = RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            mesh_page_pool: PoolSize::Bytes(budget),
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
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
    traverse_frame(&mut scenes, scene, |s| {
        let group = s
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        s.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
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
        .unwrap();
    });

    // While the traversal requests the node page the copy draws the groups of the pinned
    // pages, then the finer groups below it as their pages land.
    let start = Instant::now();
    let (mut requests, mut drawn) = (0, Vec::new());
    while tables(&scenes, &gpu, models[0]).0[1] == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the node page never streamed in"
        );
        let c = traverse_frame(&mut scenes, scene, |_| {});
        requests += c.page_requests;
        drawn.push(c.candidate_clusters);
        check_dome(&scenes, &gpu, &models, &info, budget);
    }
    assert!(requests > 0, "the traversal requested the page");
    assert!(
        drawn.iter().skip(2).all(|&n| n > 0),
        "drawn before the page: {drawn:?}"
    );
    let before = drawn.len();
    for _ in 0..8 {
        drawn.push(traverse_frame(&mut scenes, scene, |_| {}).candidate_clusters);
        check_dome(&scenes, &gpu, &models, &info, budget);
    }
    assert!(
        drawn[before..].iter().max() > drawn[..before].iter().max(),
        "finer once resident: {drawn:?}"
    );

    // Read every frame, the node page outlives its hold beside the other mount's prefetch.
    for _ in 0..60 {
        scenes.prefetch(models[1], 0);
        traverse_frame(&mut scenes, scene, |_| {});
        check_dome(&scenes, &gpu, &models, &info, budget);
    }
    assert_ne!(tables(&scenes, &gpu, models[0]).0[1], 0, "the page stayed");
    assert_eq!(scenes.residency().failed, 0);
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
                // Stored as it decodes, so file mounts read it straight into staging.
                BlobIn {
                    codec: Codec::None,
                    ..BlobIn::new(&data)
                },
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
        if frames > 1 && r.loads == 0 && r.unserved() == 0 && pool.loading_bytes == 0 {
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
                texture_pool: PoolSize::Bytes(budget),
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

/// The taps raise more cells than the first texture feedback records hold, and still load
/// every tile they want.
#[test]
fn taps_of_a_streamed_texture_read_its_resident_mips_and_load_the_tiles_they_want() {
    use orr_scene::orr_gpu::{AddressMode, Filter, MemoryKind, SamplerDesc};
    use scene_suite::pipeline;
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};
    let _serial = serial();
    let dev = device();
    let desc = RendererDesc {
        budgets: Budgets {
            initial_texture_feedback_records: 1,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(dev, &desc).unwrap();
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
    let mut lost = Vec::new();
    loop {
        let taps = tap_frame(&mut scenes);
        lost.push(scenes.residency().texture_feedback_lost);
        if taps == finest && lost.last() == Some(&0) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the taps' tiles never arrived; raises lost by frame: {lost:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        lost.iter().any(|&n| n > 0),
        "the first records hold fewer raises than the taps make"
    );
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

/// A quad of 2 m a side facing +z, its uv's v running down.
fn unit_quad() -> MeshInput {
    MeshInput {
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
    }
}

const LEAF_OPACITY: &str = "leaf/opacity";

/// A pack of `unit_quad` as "quad", of the default material, as "leaf", alpha-tested at 0.5 by
/// `LEAF_OPACITY` in its opacity slot, and as "veil", a presence layer of that opacity.
/// `LEAF_OPACITY` is 1024 x 1024 texels of which a hashed quarter are 0 and the rest 1, so every
/// micro-triangle of the leaf's micromaps and both its triangles are unsure and the alpha test
/// reads the texture wherever a ray or pixel meets it.
fn leaf_pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let path = temp("leaf_opacity.png");
        let size = 1024u32;
        let texels: Vec<u8> = (0..size * size)
            .map(|i| {
                let mut h =
                    (i % size).wrapping_mul(0x9e37_79b9) ^ (i / size).wrapping_mul(0x85eb_ca6b);
                h ^= h >> 15;
                h = h.wrapping_mul(0x2c1b_3c6d);
                h ^= h >> 12;
                if h & 3 == 0 { 0 } else { 255 }
            })
            .collect();
        let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), size, size);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .unwrap()
            .write_image_data(&texels)
            .unwrap();
        let mut leaf = Material {
            alpha_mode: Material::MASK,
            alpha_cutoff: 0.5,
            ..Material::default()
        };
        leaf.textures[slot::OPACITY].texture = AssetId::from_name(LEAF_OPACITY);
        let mut veil = Material {
            alpha_mode: Material::BLEND,
            ..Material::default()
        };
        veil.textures[slot::OPACITY].texture = AssetId::from_name(LEAF_OPACITY);
        let mesh = |name: &str, material| AssetSpec {
            name: name.into(),
            source: Source::Mesh {
                mesh: unit_quad(),
                materials: vec![material],
                settings: MeshSettings::default(),
            },
        };
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::Memory,
                vec![
                    AssetSpec {
                        name: LEAF_OPACITY.into(),
                        source: Source::Texture {
                            path,
                            settings: TextureSettings {
                                usage: Usage::Linear1,
                                ..TextureSettings::default()
                            },
                        },
                    },
                    mesh("quad", Material::default()),
                    mesh("leaf", leaf),
                    mesh("veil", veil),
                ],
            )],
        };
        orr_cook::cook(&manifest, &Default::default())
            .unwrap()
            .memory
            .remove(0)
    })
}

/// The bytes of streamed texture `id` of mount `m`: its mip tails, and every tile of the mips
/// above them.
fn streamed_bytes(scenes: &Scenes, m: MountId, id: AssetId) -> (u64, u64) {
    let texture = mounted_texture(scenes, m, id).unwrap();
    let desc = texture.desc();
    let l = texture.sparse_layout().unwrap();
    let tiles: u32 = (0..l.tail_first_mip)
        .map(|mip| {
            let [x, y] = l.tiles(desc, mip);
            x * y * desc.layers
        })
        .sum();
    (
        u64::from(l.tails(desc)) * l.tail_bytes,
        u64::from(tiles) * l.tile_bytes,
    )
}

/// The bytes the texture pool holds and is loading.
fn texture_bytes(scenes: &Scenes) -> u64 {
    let pool = *scenes.residency().class(ResidencyClass::Textures);
    pool.resident_bytes + pool.loading_bytes
}

/// Runs `frame` until the texture pool holds `bytes` and loads nothing, then four frames more,
/// after which it holds `bytes` still: `what` loaded, and nothing else.
fn fill_textures(scenes: &mut Scenes, bytes: u64, what: &str, mut frame: impl FnMut(&mut Scenes)) {
    let start = Instant::now();
    while texture_bytes(scenes) < bytes || scenes.residency().loads > 0 {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "{what} never arrived"
        );
        frame(scenes);
    }
    for _ in 0..4 {
        frame(scenes);
    }
    assert_eq!(
        texture_bytes(scenes),
        bytes,
        "{what} loaded, and nothing else"
    );
}

/// Frames of a camera at the origin looking down -z at a mirror of model `mirror` turned 45
/// degrees, which sends its rays along +x to a wall of model `wall` (in `material` where given),
/// outside the view and placed for rays alone. The wall fills most of the mirrored view, at
/// under a texel of mip 0 per pixel. Eight frames whose mirror rays report no texture feedback,
/// more than `reach` of them meeting the wall in each, leave the texture pool holding `before`;
/// frames whose mirror rays report then fill it to `after`, where it stays.
fn mirror_feedback(
    scenes: &mut Scenes,
    mirror: ModelId,
    wall: ModelId,
    material: Option<MaterialId>,
    reach: usize,
    [before, after]: [u64; 2],
) {
    use orr_math::Mat3;
    use orr_scene::orr_gpu::MemoryKind;
    use orr_scene::{Lens, Projection, TextureFilter, ViewDesc};
    use scene_suite::pipeline;
    use scene_suite::shaders::trace;

    const EXTENT: u32 = 512;
    let dev = device();
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
        if let Some(material) = material {
            s.material_sets(0, 1).unwrap()[0] = material;
        }
        let group = s
            .create_transform_group(&TransformGroupDesc {
                count: 2,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        let turn = |angle: f32, scale: f32| Mat3::from_rotation_y(angle) * scale;
        let placed = s.transforms(group, 0, 2).unwrap();
        placed[0] = Affine3::from_mat3_translation(
            turn(std::f32::consts::FRAC_PI_4, 2.0),
            Vec3::new(0.0, 0.0, -1.0),
        )
        .into();
        placed[1] = Affine3::from_mat3_translation(
            turn(-std::f32::consts::FRAC_PI_2, 1.2),
            Vec3::new(2.0, 0.0, -1.0),
        )
        .into();
        for (first, model, flags, material_set) in [
            (0, mirror, InstanceFlags::RASTER | InstanceFlags::RAY, None),
            (1, wall, InstanceFlags::RAY, material.map(|_| 0)),
        ] {
            s.add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: group,
                first,
                pose: None,
                material_set,
                flags,
                rt_range: 1000.0,
                animation_range: 0.0,
                user_tag: 0,
            })
            .unwrap();
        }
        drop(s);
        orr_scene::testing::wait(f.end(&[]).unwrap());
    }
    let n = u64::from(EXTENT * EXTENT);
    let cones = dev
        .gpu_malloc::<trace::Cone>(MemoryKind::Readback, n, "cones")
        .unwrap();
    let mut root = dev
        .gpu_malloc::<trace::Root>(MemoryKind::Upload, 1, "trace root")
        .unwrap();
    let p = pipeline(&trace::MODULE, "cones");
    // One frame tracing the mirror's rays, which report feedback with `report`; returns how
    // many reached the wall.
    let mut mirrored = |scenes: &mut Scenes, report: bool| -> usize {
        let mut f = scenes.begin_frame(&desc).unwrap();
        f.scene(scene).unwrap().set_focus(DVec3::ZERO);
        let view = f
            .raster_view(&ViewDesc {
                scene,
                history,
                position: DVec3::ZERO,
                rotation: Quat::IDENTITY,
                projection: Projection {
                    fov_y: 0.9,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: orr_math::UVec2::splat(EXTENT),
                output_extent: orr_math::UVec2::splat(EXTENT),
                jitter: Vec2::ZERO,
                camera_cut: false,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
            })
            .unwrap();
        root.as_mut_slice().unwrap()[0] = trace::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            traced: GpuPtr::NULL,
            seen: GpuPtr::NULL,
            cones: cones.gpu(),
            mode: u32::from(report),
            pad0: 0,
            pad1: 0,
        };
        let list = f.list().unwrap();
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
        list.dispatch(&p, root.gpu(), [n.div_ceil(64) as u32, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        orr_scene::testing::wait(f.end(&[]).unwrap());
        cones
            .as_slice()
            .unwrap()
            .iter()
            .filter(|c| c.hits == 2)
            .count()
    };
    for _ in 0..8 {
        assert!(mirrored(scenes, false) > reach, "the rays reach the wall");
    }
    assert_eq!(
        texture_bytes(scenes),
        before,
        "nothing loads without feedback from the mirror rays"
    );
    fill_textures(scenes, after, "the tiles the mirror rays want", |scenes| {
        mirrored(scenes, true);
    });
}

/// A streamed texture on a wall that rays alone see, in a mirror, loads the tiles its reflected
/// hits want once those hits report texture feedback: they want mip 0 everywhere, so every tile
/// of mips 0 and 1. Nothing loads while they do not report.
#[test]
fn a_streamed_texture_seen_only_in_a_mirror_loads_the_tiles_its_reflected_hits_want() {
    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(tiled_pack())));
    ready(&mut scenes, m);
    let leaf = scenes.mount(PackSource::Memory(Arc::clone(leaf_pack())));
    ready(&mut scenes, leaf);
    let quad = scenes.model(leaf, AssetId::from_name("quad")).unwrap();
    let wall_material = scenes
        .material(m, AssetId::from_name(&format!("{}/material", TILED[0].0)))
        .unwrap();
    let (tails, tiles) = streamed_bytes(&scenes, m, AssetId::from_name(TILED[0].0));
    let others: u64 = (1..TILED.len()).map(|t| tiled_layout(dev, t).1).sum();
    let (leaf_tails, _) = streamed_bytes(&scenes, leaf, AssetId::from_name(LEAF_OPACITY));
    let before = tails + others + leaf_tails;
    mirror_feedback(
        &mut scenes,
        quad,
        quad,
        Some(wall_material),
        512 * 512 / 2,
        [before, before + tiles],
    );
    assert_clean();
}

/// An alpha-tested wall that rays alone see, in a mirror, its opacity a streamed texture, loads
/// every tile of its opacity's mips 0 and 1 once the mirror rays report texture feedback: their
/// alpha tests read mip 0 wherever the wall's micromaps leave it unsure, whether they keep the
/// hit or not. Nothing loads while they do not report.
#[test]
fn an_alpha_tested_wall_seen_only_in_a_mirror_loads_the_opacity_tiles_its_ray_alpha_tests_read() {
    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let leaf = scenes.mount(PackSource::Memory(Arc::clone(leaf_pack())));
    ready(&mut scenes, leaf);
    let quad = scenes.model(leaf, AssetId::from_name("quad")).unwrap();
    let wall = scenes.model(leaf, AssetId::from_name("leaf")).unwrap();
    let (tails, tiles) = streamed_bytes(&scenes, leaf, AssetId::from_name(LEAF_OPACITY));
    // A quarter of the wall's texels let the rays through.
    mirror_feedback(
        &mut scenes,
        quad,
        wall,
        None,
        512 * 512 / 4,
        [tails, tails + tiles],
    );
    assert_clean();
}

/// A presence layer that rays alone see, in a mirror, its opacity a streamed texture, loads every
/// tile of its opacity's mips 0 and 1 once the mirror rays report texture feedback: each
/// candidate of theirs on it reads its opacity at mip 0. Nothing loads while they do not report.
#[test]
fn a_presence_layer_seen_only_in_a_mirror_loads_the_opacity_tiles_its_ray_candidates_read() {
    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let leaf = scenes.mount(PackSource::Memory(Arc::clone(leaf_pack())));
    ready(&mut scenes, leaf);
    let quad = scenes.model(leaf, AssetId::from_name("quad")).unwrap();
    let veil = scenes.model(leaf, AssetId::from_name("veil")).unwrap();
    let (tails, tiles) = streamed_bytes(&scenes, leaf, AssetId::from_name(LEAF_OPACITY));
    mirror_feedback(
        &mut scenes,
        quad,
        veil,
        None,
        512 * 512 / 2,
        [tails, tails + tiles],
    );
    assert_clean();
}

/// An alpha-tested quad in view, its opacity a streamed texture, loads every tile of its
/// opacity's mips 0 and 1: the raster's alpha test reads mip 0 wherever the quad's triangles
/// are unsure, and reports those reads whether it keeps the pixel or not.
#[test]
fn an_alpha_tested_quad_in_view_loads_the_opacity_tiles_its_raster_alpha_test_reads() {
    use orr_scene::{Lens, Projection, TextureFilter, ViewDesc};

    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let leaf = scenes.mount(PackSource::Memory(Arc::clone(leaf_pack())));
    ready(&mut scenes, leaf);
    let model = scenes.model(leaf, AssetId::from_name("leaf")).unwrap();
    let (tails, tiles) = streamed_bytes(&scenes, leaf, AssetId::from_name(LEAF_OPACITY));
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = scenes.create_view_history().unwrap();
    let list = [scene];
    let desc = FrameDesc {
        time: FrameTime::default(),
        scenes: &list,
        low_latency: None,
    };
    // The camera at the origin looks down -z at the quad, which fills most of the view at two
    // texels of mip 0 per pixel.
    {
        let f = scenes.begin_frame(&desc).unwrap();
        let mut s = f.scene(scene).unwrap();
        let group = s
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        s.transforms(group, 0, 1).unwrap()[0] =
            Affine3::from_translation(Vec3::new(0.0, 0.0, -2.2)).into();
        s.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first: 0,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
        drop(s);
        orr_scene::testing::wait(f.end(&[]).unwrap());
    }
    let viewed = |scenes: &mut Scenes| {
        let mut f = scenes.begin_frame(&desc).unwrap();
        f.scene(scene).unwrap().set_focus(DVec3::ZERO);
        f.raster_view(&ViewDesc {
            scene,
            history,
            position: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            projection: Projection {
                fov_y: 0.9,
                near: 0.05,
            },
            lens: Lens::default(),
            render_extent: orr_math::UVec2::splat(512),
            output_extent: orr_math::UVec2::splat(512),
            jitter: Vec2::ZERO,
            camera_cut: false,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
        orr_scene::testing::wait(f.end(&[]).unwrap());
    };
    fill_textures(
        &mut scenes,
        tails + tiles,
        "every tile of the quad's opacity above its mip tails",
        viewed,
    );
    assert_clean();
}

/// Pixel probes of a quad in view whose material reads a streamed texture: the first frame's
/// tap wants a mip above the tail while only the tail is resident, and the probe says so, names
/// the texture and finds the tile it wants absent; once the tap's feedback has loaded that tile
/// the probe reads it resident and the tap no coarser than the mip it wants.
#[test]
fn pixel_probes_report_the_mip_a_tap_wants_and_the_finest_resident_under_it() {
    use orr_scene::{Lens, Projection, TextureFilter, UnitState, ViewDesc};

    let _serial = serial();
    let dev = device();
    let mut scenes = Scenes::new(dev, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(tiled_pack())));
    ready(&mut scenes, m);
    let leaf = scenes.mount(PackSource::Memory(Arc::clone(leaf_pack())));
    ready(&mut scenes, leaf);
    let quad = scenes.model(leaf, AssetId::from_name("quad")).unwrap();
    let name = TILED[0].0;
    let texture = scenes.texture(m, AssetId::from_name(name)).unwrap();
    let material = scenes
        .material(m, AssetId::from_name(&format!("{name}/material")))
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
        s.transforms(group, 0, 1).unwrap()[0] =
            Affine3::from_translation(Vec3::new(0.0, 0.0, -2.2)).into();
        s.add_instances(&InstancesDesc {
            model: quad,
            copies: 1,
            transforms: group,
            first: 0,
            pose: None,
            material_set: Some(0),
            flags: InstanceFlags::RASTER,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
        drop(s);
        orr_scene::testing::wait(f.end(&[]).unwrap());
    }
    let probed = |scenes: &mut Scenes| {
        let mut f = scenes.begin_frame(&desc).unwrap();
        f.scene(scene).unwrap().set_focus(DVec3::ZERO);
        let view = f
            .raster_view(&ViewDesc {
                scene,
                history,
                position: DVec3::ZERO,
                rotation: Quat::IDENTITY,
                projection: Projection {
                    fov_y: 0.9,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: orr_math::UVec2::splat(512),
                output_extent: orr_math::UVec2::splat(512),
                jitter: Vec2::ZERO,
                camera_cut: false,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
            })
            .unwrap();
        f.probe(&view, [256, 256]).unwrap();
        let id = f.id();
        orr_scene::testing::wait(f.end(&[]).unwrap());
        let mut probes = scenes.probes(history, id);
        assert_eq!(probes.len(), 1);
        let probe = probes.remove(0);
        assert!(probe.hit(), "{probe}");
        let tap = probe
            .textures
            .iter()
            .find(|t| t.gpu.texture == texture.0)
            .unwrap_or_else(|| panic!("no tap of {name}:\n{probe}"))
            .clone();
        assert_eq!(tap.name.as_deref(), Some(name), "{probe}");
        (probe, tap)
    };
    let (probe, tap) = probed(&mut scenes);
    let tail = tap.gpu.tail_mip as f32;
    assert!(tail > 0.0 && tap.gpu.wanted < tail, "{probe}");
    assert_eq!(
        tap.gpu.resident, tail,
        "only the tail is resident at first:\n{probe}"
    );
    let tile = tap
        .tile
        .unwrap_or_else(|| panic!("no tile state:\n{probe}"));
    assert_eq!(tile.mip, tap.gpu.wanted.floor().max(0.0) as u32, "{probe}");
    assert!(
        !tile.mapped && tile.status.state != UnitState::Resident,
        "{probe}"
    );
    let start = Instant::now();
    loop {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "the tap's tile never arrived"
        );
        let (probe, tap) = probed(&mut scenes);
        let tile = tap
            .tile
            .unwrap_or_else(|| panic!("no tile state:\n{probe}"));
        if tap.gpu.resident <= tap.gpu.wanted.floor().max(0.0) {
            assert!(tile.mapped, "{probe}");
            assert_eq!(tile.status.state, UnitState::Resident, "{probe}");
            break;
        }
    }
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

/// Holds the loading queue behind a host timeline until dropped.
struct Gate(HostTimeline);

impl Drop for Gate {
    fn drop(&mut self) {
        self.0.signal(1);
    }
}

#[test]
fn a_requested_blob_lands_while_a_mount_build_waits_on_the_loading_queue() {
    use orr_scene::BlobTarget;
    use orr_scene::orr_gpu::MemoryKind;
    let _serial = serial();
    let dev = device();
    let (Some(loading), Some(_)) = (dev.queue(QueueKind::Loading), dev.queue(QueueKind::Copy))
    else {
        return;
    };
    // Without cluster acceleration structures the mount builds nothing to wait on.
    if dev.caps().cluster.max_tris == 0 || dev.caps().cluster.max_partitions == 0 {
        return;
    }
    let (_, data) = consumer_blobs();
    let desc = RendererDesc {
        budgets: Budgets {
            user_blob_pool: data.len() as u64,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(dev, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let tiled = scenes.mount(PackSource::Memory(Arc::clone(tiled_pack())));
    ready(&mut scenes, tiled);

    // The loading thread is idle, so nothing else submits its queue meanwhile.
    let gate = Gate(dev.create_host_timeline("loading gate").unwrap());
    let list = loading.command_list().unwrap();
    loading.submit(vec![list], &[gate.0.point(1)]).unwrap();
    // Without textures it stages no copies, which would hold the copy queue behind the gate.
    let meshes = scenes.mount(PackSource::Memory(Arc::clone(sheet_pack())));
    // Time for the mount's reads and decodes, so its build reaches the gate first.
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(500) {
        frame(&mut scenes);
        std::thread::sleep(Duration::from_millis(1));
    }
    let buffer = dev
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, data.len() as u64, "consumer data")
        .unwrap();
    let target = BlobTarget::Buffer {
        at: buffer.gpu(),
        len: buffer.size(),
    };
    let id = AssetId::from_name(CONSUMER);
    let blob = scenes.request_blob(tiled, id, 2, target).unwrap();
    assert!(matches!(settle_blob(&mut scenes, blob), LoadState::Ready));
    assert!(Gpu(dev).bytes(buffer.gpu().address().0, data.len()) == data);
    assert!(
        matches!(scenes.mount_state(meshes), LoadState::Pending),
        "the mount's build waits behind the gate"
    );
    drop(gate);
    ready(&mut scenes, meshes);
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

/// Places one copy of each of `models` at the origin, drawn and traced.
fn place_copies(s: &mut SceneWriter<'_>, models: &[ModelId]) -> orr_scene::TransformGroupId {
    place_copies_as(s, models, InstanceFlags::RASTER | InstanceFlags::RAY)
}

/// Places one copy of each of `models` at the origin with `flags`.
fn place_copies_as(
    s: &mut SceneWriter<'_>,
    models: &[ModelId],
    flags: InstanceFlags,
) -> orr_scene::TransformGroupId {
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
            flags,
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

/// A mesh page budget on `dev` for `mounts` mounts of [`sheet_pack`]'s mesh `info`: every
/// mount's pinned pages and one mount's streamed pages, which with a CLAS pool also hold the
/// CLAS they build, at 32 B a triangle and 256 B a cluster until they have built them.
fn sheet_budget(info: &MeshInfo, dev: &Device, mounts: u64) -> u64 {
    let c = dev.caps().cluster;
    let traced = c.max_tris > 0 && c.max_partitions > 0;
    let pinned = info.header.pinned_pages as usize;
    let page = |p: &orr_pack::mesh::PageInfo| u64::from(p.resident_len).next_multiple_of(256);
    let clas = |p: &orr_pack::mesh::PageInfo| {
        let bytes = 32 * u64::from(p.tri_count) + 256 * u64::from(p.cluster_count);
        if traced {
            bytes.next_multiple_of(128)
        } else {
            0
        }
    };
    let pinned_bytes: u64 = info.pages[..pinned].iter().map(page).sum();
    let streamed_bytes: u64 = info.pages[pinned..].iter().map(|p| page(p) + clas(p)).sum();
    mounts * pinned_bytes + streamed_bytes
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
    let traced = dev.caps().cluster.max_tris > 0;
    let budget = sheet_budget(&info, dev, 2);
    let desc = RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            mesh_page_pool: PoolSize::Bytes(budget),
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let gpu = Gpu(dev);
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
        assert!(mesh.pinned_pages < mesh.page_count, "its pages stream");
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
    let bases: Vec<u32> = gpu.values(mesh.page_bases, mesh.page_count);
    let refs: Vec<orr_scene::ClusterRef> = gpu.values(
        root.clusters.offset(u64::from(bases[first])),
        firsts[first + 1] - firsts[first],
    );
    for (r, i) in refs.iter().zip(0..) {
        assert_eq!(r.page.address().0, near[first]);
        assert_eq!((r.local, r.mesh_cluster), (i, firsts[first] + i));
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
    // Pages that built less than their estimate leave room for some of the first mount's.
    let (_, first_mount) = mesh_of(&scenes, &gpu, models[0]);
    assert!(
        first_mount[pinned..].contains(&0),
        "the first mount's streamed pages were evicted"
    );
    for _ in 0..4 {
        streamed_frame(&mut scenes, scene, |_| {});
    }
    assert_eq!(scenes.residency().failed, 0);
}

/// Pools of a page or less per allocator thread, of several pages per thread, where runs cross
/// from one thread's pages to the next, and of a 6 GiB pool's 98,304 pages with a batch the
/// size of what one step of 1,024 streamed loads lands.
#[test]
fn the_clas_pool_allocator_places_like_a_first_fit_reference() {
    let _serial = serial();
    first_fit_rounds(12, 96, 40);
    first_fit_rounds(1000, 512, 20);
    first_fit_rounds(98_304, 40_000, 3);
    first_fit_rounds(98_304, 96, 3);
    first_fit_rounds(12, 40_000, 3);
    first_fit_rounds(1024, 40_000, 3);
    assert_clean();
}

/// `rounds` batches of `clas_count` CLAS in streamed pages of up to 400 into a pool of
/// `page_count` pages, each against [`orr_scene::ClasOrder::place`]. A few CLAS have no bytes
/// or more than a page, a streamed page's CLAS free together, and some pages are held, their
/// free units marked used, for a round.
fn first_fit_rounds(page_count: u32, clas_count: usize, rounds: u32) {
    use orr_scene::ClasOrder;
    use orr_scene::memory::RangeAllocator;
    use orr_scene::orr_gpu::{MemoryKind, RtasOp};
    const WORDS: u32 = 16;
    const UNITS: u32 = 32 * WORDS;
    let dev = device();
    let module = orr_scene::MODULES
        .iter()
        .find(|m| m.name == "clas_pool")
        .unwrap();
    let pipeline = scene_suite::pipeline(module, "allocate");
    let up = MemoryKind::Upload;
    let bits = dev
        .gpu_malloc::<u32>(
            MemoryKind::DeviceLocal,
            u64::from(page_count * WORDS),
            "bits",
        )
        .unwrap();
    let summary = dev
        .gpu_malloc::<[u32; 2]>(MemoryKind::DeviceLocal, u64::from(page_count), "runs")
        .unwrap();
    let mut frees = dev
        .gpu_malloc::<[u32; 2]>(up, (clas_count + page_count as usize) as u64, "frees")
        .unwrap();
    let mut takes = dev
        .gpu_malloc::<u32>(up, u64::from(page_count), "takes")
        .unwrap();
    let taken = dev
        .gpu_malloc::<u32>(up, u64::from(page_count), "taken")
        .unwrap();
    let mut holds = dev
        .gpu_malloc::<u32>(up, u64::from(page_count), "holds")
        .unwrap();
    let mut order = dev
        .gpu_malloc::<u32>(up, clas_count as u64, "order")
        .unwrap();
    let mut groups = dev
        .gpu_malloc::<[u32; 2]>(up, clas_count as u64, "groups")
        .unwrap();
    let mut sizes = dev
        .gpu_malloc::<u32>(up, clas_count as u64, "sizes")
        .unwrap();
    let mut sources = dev
        .gpu_malloc::<u64>(up, clas_count as u64, "sources")
        .unwrap();
    let units = dev
        .gpu_malloc::<u32>(up, clas_count as u64, "units")
        .unwrap();
    let destinations = dev
        .gpu_malloc::<u64>(up, clas_count as u64, "destinations")
        .unwrap();
    let stride = dev.rtas_args_stride(RtasOp::Move) as usize;
    let moves = dev
        .gpu_malloc::<u8>(up, (stride * clas_count) as u64, "moves")
        .unwrap();
    let count = dev.gpu_malloc::<u32>(up, 1, "move count").unwrap();
    let mut root = dev
        .gpu_malloc::<orr_scene::PoolRoot>(up, 1, "root")
        .unwrap();
    let base = 1u64 << 40;
    for (i, s) in sources.as_mut_slice().unwrap().iter_mut().enumerate() {
        *s = 0x1000 * (i as u64 + 1);
    }

    let mut reference = RangeAllocator::new(page_count * UNITS);
    assert!(reference.take(0, page_count * UNITS));
    let mut bound = vec![false; page_count as usize];
    // Per streamed page, the unit ranges its CLAS take, joined as the pool joins them, so a
    // free can reach across pool pages and allocator threads' blocks.
    let mut live: Vec<Vec<[u32; 2]>> = Vec::new();
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
    // The units the last round's holds marked used.
    let mut held: Vec<(u32, u32)> = Vec::new();
    for round in 0..rounds {
        // Bind some pages, free some CLAS and what the last holds marked, ask to take some
        // pages, hold some, then allocate.
        let mut free_list: Vec<[u32; 2]> = held.drain(..).map(|(f, n)| [f, n]).collect();
        for p in 0..page_count {
            if !bound[p as usize] && rand() % 4 == 0 {
                bound[p as usize] = true;
                free_list.push([p * UNITS, UNITS]);
            }
        }
        live.retain(|ranges| {
            let keep = rand() % 3 != 0;
            if !keep {
                free_list.extend_from_slice(ranges);
            }
            keep
        });
        free_list.sort_unstable_by_key(|f| f[0]);
        let take_list: Vec<u32> = (0..page_count)
            .filter(|&p| bound[p as usize] && rand() % 5 == 0)
            .collect();
        let hold_list: Vec<u32> = (0..page_count)
            .filter(|&p| bound[p as usize] && rand() % 7 == 0)
            .collect();
        let size_list: Vec<u32> = (0..clas_count)
            .map(|_| match rand() % 1000 {
                0 => 0,
                1 => 128 * UNITS + 1,
                _ => 128 + rand() % 6000,
            })
            .collect();
        // Results indices shuffled, as a batch's builds order them apart from its pages.
        let mut results: Vec<u32> = (0..clas_count as u32).collect();
        for i in (1..results.len()).rev() {
            results.swap(i, rand() as usize % (i + 1));
        }
        let mut pages = ClasOrder::default();
        let mut page_clas: Vec<&[u32]> = Vec::new();
        let mut rest = results.as_slice();
        while !rest.is_empty() {
            let wide = if rand() % 8 == 0 { 400 } else { 160 };
            let (page, after) = rest.split_at((1 + rand() as usize % wide).min(rest.len()));
            pages.push_page(page);
            page_clas.push(page);
            rest = after;
        }
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
        for &p in &hold_list {
            held.extend(reference.take_within(p * UNITS, UNITS));
        }
        let expect_units = pages.place(&mut reference, UNITS, &size_list);
        let placed: Vec<u32> = pages
            .order()
            .iter()
            .copied()
            .filter(|&i| expect_units[i as usize] != u32::MAX)
            .collect();
        for page in &page_clas {
            let mut ranges: Vec<[u32; 2]> = Vec::new();
            for &i in page
                .iter()
                .filter(|&&i| expect_units[i as usize] != u32::MAX)
            {
                let (u, n) = (
                    expect_units[i as usize],
                    size_list[i as usize].div_ceil(128),
                );
                match ranges.last_mut() {
                    Some(r) if r[0] + r[1] == u => r[1] += n,
                    _ => ranges.push([u, n]),
                }
            }
            live.push(ranges);
        }

        frees.as_mut_slice().unwrap()[..free_list.len()].copy_from_slice(&free_list);
        takes.as_mut_slice().unwrap()[..take_list.len()].copy_from_slice(&take_list);
        holds.as_mut_slice().unwrap()[..hold_list.len()].copy_from_slice(&hold_list);
        order.as_mut_slice().unwrap().copy_from_slice(pages.order());
        groups.as_mut_slice().unwrap()[..pages.groups().len()].copy_from_slice(pages.groups());
        sizes.as_mut_slice().unwrap().copy_from_slice(&size_list);
        root.as_mut_slice().unwrap()[0] = orr_scene::PoolRoot {
            bits: bits.gpu(),
            summary: summary.gpu().cast(),
            frees: frees.gpu().cast(),
            takes: takes.gpu(),
            taken: taken.gpu(),
            holds: holds.gpu(),
            order: order.gpu(),
            groups: groups.gpu().cast(),
            sizes: sizes.gpu(),
            sources: sources.gpu(),
            units: units.gpu(),
            destinations: destinations.gpu(),
            moves: moves.gpu(),
            move_count: count.gpu(),
            base,
            pages: page_count,
            words: WORDS,
            free_count: free_list.len() as u32,
            take_count: take_list.len() as u32,
            hold_count: hold_list.len() as u32,
            group_count: pages.groups().len() as u32,
            flags: u32::from(round == 0),
            _pad0: 0,
            _pad1: 0,
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
        assert_eq!(count.as_slice().unwrap()[0] as usize, placed.len());
        let records = moves.as_slice().unwrap();
        for (k, &i) in placed.iter().enumerate() {
            let at = base + 128 * u64::from(expect_units[i as usize]);
            assert_eq!(destinations.as_slice().unwrap()[k], at, "move {k}");
            let src = u64::from_le_bytes(records[k * stride..][..8].try_into().unwrap());
            assert_eq!(src, 0x1000 * (u64::from(i) + 1), "move {k}'s source");
        }
    }
}

/// Wanted mesh page table entries past the request capacity, with ties in the bucket the
/// capacity fills up in: the kernels keep that many of the largest errors, each once with its
/// error.
#[test]
fn the_traversal_requests_kept_are_those_of_the_largest_errors() {
    use orr_scene::orr_gpu::MemoryKind;
    // `cut.slang`: the counters, CUT_REQUESTS, CUT_FEEDBACK, FEEDBACK_BUCKETS and its shift.
    const COUNTERS: u64 = 25;
    const REQUESTS: usize = 16;
    const KEPT: usize = 17;
    const BUCKETS: u64 = 4096;
    let bucket = |error: u32| error >> 19;
    let dev = device();
    let module = orr_scene::MODULES
        .iter()
        .find(|m| m.name == "feedback")
        .unwrap();
    let kernels = ["count", "select", "write"].map(|e| scene_suite::pipeline(module, e));
    let entries = 50_003u32;
    let words = entries.div_ceil(32);
    let up = MemoryKind::Upload;
    let mut counters = dev.gpu_malloc::<u32>(up, COUNTERS, "counters").unwrap();
    let mut wanted = dev
        .gpu_malloc::<u32>(up, u64::from(words), "wanted")
        .unwrap();
    let mut errors = dev
        .gpu_malloc::<u32>(up, u64::from(entries), "errors")
        .unwrap();
    let mut histogram = dev.gpu_malloc::<u32>(up, BUCKETS, "histogram").unwrap();
    let mut cut = dev.gpu_malloc::<orr_scene::CutGpu>(up, 1, "cut").unwrap();
    let mut root = dev
        .gpu_malloc::<orr_scene::FeedbackRoot>(up, 1, "root")
        .unwrap();
    let mut rand = {
        let mut x = 0x9e37_79b9u32;
        move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        }
    };
    let render = dev.queue(QueueKind::Render).unwrap();
    // Errors from a few buckets, so that many tie in the one the capacity fills up in.
    let levels = [0.0f32, 0.3, 1.0, 1.01, 4.0, 250.0, f32::INFINITY];
    for capacity in [0u32, 1, 700, 9_000, 40_000] {
        let mut want = vec![None; entries as usize];
        for w in want.iter_mut() {
            if rand() % 4 == 0 {
                *w = Some(levels[rand() as usize % levels.len()].to_bits());
            }
        }
        let total = want.iter().flatten().count();
        let mut bits = vec![0u32; words as usize];
        for (i, w) in want.iter().enumerate() {
            if w.is_some() {
                bits[i / 32] |= 1 << (i % 32);
            }
        }
        wanted.as_mut_slice().unwrap().copy_from_slice(&bits);
        for (e, w) in errors.as_mut_slice().unwrap().iter_mut().zip(&want) {
            *e = w.unwrap_or(0);
        }
        counters.as_mut_slice().unwrap().fill(0);
        histogram.as_mut_slice().unwrap().fill(0);
        let feedback = dev
            .gpu_malloc::<[u32; 2]>(up, u64::from(capacity.max(1)), "feedback")
            .unwrap();
        cut.as_mut_slice().unwrap()[0] = orr_scene::CutGpu {
            counters: counters.gpu(),
            wanted: wanted.gpu(),
            wanted_errors: errors.gpu(),
            histogram: histogram.gpu(),
            feedback: feedback.gpu().cast(),
            feedback_capacity: capacity,
            feedback_words: words,
            ..Zeroable::zeroed()
        };
        root.as_mut_slice().unwrap()[0] = orr_scene::FeedbackRoot {
            cut: cut.gpu(),
            _pad: 0,
        };
        let mut list = render.command_list().unwrap();
        let groups = [words.div_ceil(64), 1, 1];
        for (k, grid) in kernels.iter().zip([groups, [1, 1, 1], groups]) {
            list.dispatch(k, root.gpu(), grid);
            list.barrier(
                Stages::COMPUTE,
                Stages::COMPUTE | Stages::HOST,
                Hazards::empty(),
            );
        }
        orr_scene::testing::wait(render.submit(vec![list], &[]).unwrap());

        let c = counters.as_slice().unwrap();
        assert_eq!(
            c[REQUESTS] as usize, total,
            "requests at capacity {capacity}"
        );
        let kept = c[KEPT] as usize;
        assert_eq!(
            kept,
            total.min(capacity as usize),
            "kept at capacity {capacity}"
        );
        let mut seen = vec![false; entries as usize];
        let mut lowest = u32::MAX;
        for &[entry, error] in &feedback.as_slice().unwrap()[..kept] {
            let i = entry as usize;
            assert_eq!(want[i], Some(error), "entry {entry} with its error");
            assert!(!seen[i], "entry {entry} kept twice");
            seen[i] = true;
            lowest = lowest.min(bucket(error));
        }
        for (i, w) in want.iter().enumerate() {
            if let Some(error) = *w
                && !seen[i]
            {
                assert!(
                    bucket(error) <= lowest,
                    "entry {i} dropped above a kept bucket at capacity {capacity}"
                );
            }
        }
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
        let r = scenes.residency();
        assert_eq!(
            (r.failed_cuts, r.out_of_memory),
            (0, 0),
            "a finer cut failed to build"
        );
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

/// A copy that only traces loads no page through the traversal, so the finer cut it wants loads
/// its own. The mesh page budget holds the mesh's pages at their CLAS estimates with room to
/// spare, but its half, the share finer cuts may lock, holds only what the cut locks once they
/// have built their CLAS, not the cut's pages at those estimates.
#[test]
fn a_finer_cut_whose_pages_no_view_loaded_builds_once_they_land() {
    let _serial = serial();
    let dev = device();
    if dev.caps().cluster.max_tris == 0 || dev.caps().cluster.max_partitions == 0 {
        return;
    }
    let gpu = Gpu(dev);
    let a = Archive::from_memory(Arc::clone(sheet_pack())).unwrap();
    let info = MeshInfo::parse(&blob(
        &a,
        entry(&a, AssetId::from_name("kit/sheet/mesh")),
        0,
    ))
    .unwrap();
    let desc = RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            mesh_page_pool: PoolSize::Bytes(sheet_budget(&info, dev, 1) * 3 / 2),
            ..Budgets::default()
        },
        ..streaming_desc()
    };
    let mut scenes = Scenes::new(dev, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let m = scenes.mount(PackSource::Memory(Arc::clone(sheet_pack())));
    ready(&mut scenes, m);
    let model = scenes.model(m, AssetId::from_name(SHEET)).unwrap();
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    streamed_frame(&mut scenes, scene, |s| {
        place_copies_as(s, &[model], InstanceFlags::RAY);
    });
    let start = Instant::now();
    loop {
        let (mesh, _) = mesh_of(&scenes, &gpu, model);
        let cuts: Vec<orr_scene::RtCutGpu> = gpu.values(mesh.rt_cuts, mesh.fine_cuts);
        if cuts.iter().any(|c| c.blas != 0) {
            break;
        }
        let r = scenes.residency();
        assert_eq!(
            (r.failed_cuts, r.out_of_memory),
            (0, 0),
            "a finer cut failed to build"
        );
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "no finer cut was built"
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
    let pinned = info.header.pinned_pages as usize;
    let mut scenes = Scenes::new(
        dev,
        &RendererDesc {
            budgets: Budgets {
                mesh_resident_max: 0,
                mesh_page_pool: PoolSize::Bytes(sheet_budget(&info, dev, 2)),
                ..Budgets::default()
            },
            ..RendererDesc::default()
        },
    )
    .unwrap();
    let _workers = Workers::start(&scenes, 2);
    let mounts = [(); 3].map(|()| scenes.mount(PackSource::Memory(Arc::clone(bytes))));
    let [source, others @ ..] = mounts.map(|m| {
        ready(&mut scenes, m);
        scenes.model(m, AssetId::from_name(SHEET)).unwrap()
    });
    // By an edge: the spheres of the coarser groups all reach the sheet's middle, so an edit
    // there copies every page another depends on.
    let touched = [[0.0f32, 0.0, 1.8, 0.05]];
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
    // The source's resident streamed pages, past `skip`.
    let streamed = |scenes: &Scenes, skip: &[usize]| {
        let (_, src) = mesh_of(scenes, &gpu, source);
        (pinned..src.len())
            .filter(|p| !skip.contains(p) && src[*p] != 0)
            .count()
    };
    let held = copies(&scenes);
    assert!(!held.is_empty(), "the edit holds copies");
    assert!(
        held.len() < info.pages.len(),
        "the edit shares untouched pages"
    );
    // The source's pages the copies depend on, and what they depend on.
    let deps = |p: usize| {
        let d = info.pages[p].deps;
        &info.page_deps[d.first as usize..][..d.count as usize]
    };
    let mut needed: Vec<usize> = Vec::new();
    let mut todo: Vec<usize> = held.clone();
    while let Some(p) = todo.pop() {
        for &d in deps(p) {
            let d = d as usize;
            if !held.contains(&d) && !needed.contains(&d) {
                needed.push(d);
                todo.push(d);
            }
        }
    }
    assert!(
        needed.iter().any(|&p| p >= pinned),
        "the copies depend on streamed pages"
    );
    // Every page of the edit's table is drawn with the coarser pages it depends on.
    let whole = |scenes: &Scenes| {
        let (_, own) = mesh_of(scenes, &gpu, model);
        for p in (0..own.len()).filter(|&p| own[p] != 0) {
            assert!(
                deps(p).iter().all(|&d| own[d as usize] != 0),
                "page {p} of the edit without the pages it depends on"
            );
        }
    };
    whole(&scenes);
    let skip: Vec<usize> = held.iter().chain(&needed).copied().collect();
    // What the source keeps for them: those pages and the pages they depend on.
    let mut kept = needed.clone();
    let mut todo = needed.clone();
    while let Some(p) = todo.pop() {
        for &d in deps(p) {
            if !kept.contains(&(d as usize)) {
                kept.push(d as usize);
                todo.push(d as usize);
            }
        }
    }
    // The source's pages land: the edit's table follows them.
    let start = Instant::now();
    while streamed(&scenes, &skip) == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the source's pages never landed"
        );
        scenes.prefetch(source, 0);
        streamed_frame(&mut scenes, scene, |_| {});
        assert_eq!(copies(&scenes), held, "a landing reached the edit's table");
        whole(&scenes);
    }
    // Two other meshes' pages, more than the budget holds, push the rest out after their
    // hold, those the edit reads its copies of in place of them too: the edit keeps its
    // copies and the pages they depend on, and its shared entries go null with the source's.
    let start = Instant::now();
    while streamed(&scenes, &kept) > 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the source's pages never left"
        );
        for &other in &others {
            scenes.prefetch(other, 0);
        }
        streamed_frame(&mut scenes, scene, |_| {});
        assert_eq!(
            copies(&scenes),
            held,
            "an eviction reached the edit's table"
        );
        whole(&scenes);
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
            texture_pool: PoolSize::Bytes(budget),
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
            if frames > 1 && r.loads == 0 && r.unserved() == 0 && pool.loading_bytes == 0 {
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

const CUBE: usize = 2;
/// Tiles of the coarser streamed mip of the flat texture and of the cube.
const COARSE_TILES: u64 = 6;

/// Scenes over one mount of the tiled pack per entry of `kinds`, each with its `TILED` texture
/// of that index loaded whole, a coarse tile then the fine ones under it, in blocks of four
/// tiles.
struct Tiled {
    scenes: Scenes,
    _workers: Workers,
    mounts: Vec<MountId>,
    kinds: Vec<usize>,
    textures: Vec<TextureIndex>,
    /// Bytes of a tile, and of one mount's mip tails.
    tile: u64,
    tails: u64,
}

impl Tiled {
    fn new(dev: &Device, kinds: &[usize]) -> Self {
        let bytes = tiled_pack();
        let tile = orr_pack::texture::TILE_BYTES as u64;
        let layouts: Vec<(u64, u64)> = (0..TILED.len()).map(|t| tiled_layout(dev, t)).collect();
        let tails = layouts.iter().map(|l| l.1).sum::<u64>();
        let tiles: u64 = kinds.iter().map(|&k| layouts[k].0).sum();
        let desc = RendererDesc {
            budgets: Budgets {
                texture_pool: PoolSize::Bytes(kinds.len() as u64 * tails + tiles * tile),
                mesh_page_pool: PoolSize::Bytes(16 << 20),
                instance_page_pool: 16 << 20,
                in_flight_bytes: 16 << 20,
                stream_block: 4 * tile,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        };
        let mut scenes = Scenes::new(dev, &desc).unwrap();
        let workers = Workers::start(&scenes, 2);
        let mut mounts = Vec::new();
        let mut textures = Vec::new();
        for &kind in kinds {
            let m = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
            ready(&mut scenes, m);
            stream_tiles(&mut scenes, m, &[kind], u64::MAX);
            assert_tiled(&scenes, dev, m, kind, 0..9);
            let name = AssetId::from_name(TILED[kind].0);
            textures.push(scenes.texture(m, name).unwrap());
            mounts.push(m);
        }
        Self {
            scenes,
            _workers: workers,
            mounts,
            kinds: kinds.to_vec(),
            textures,
            tile,
            tails,
        }
    }

    fn pool(&self) -> orr_scene::PoolFill {
        let memory = self.scenes.memory();
        *memory
            .pools
            .iter()
            .find(|p| p.name == "texture tile pool")
            .expect("the tile pool")
    }

    /// Lowers the texture budget to the mounts' coarse tiles: the fine ones go once they leave
    /// their hold, and leave each block a tile.
    fn keep_coarse(&mut self) {
        let mounts = self.mounts.len() as u64;
        let coarse = mounts * COARSE_TILES * self.tile;
        self.scenes
            .lower_budget(ResidencyClass::Textures, mounts * self.tails + coarse);
    }

    /// A frame that asks for the coarse tiles of the cubes of `mounts`.
    fn frame(&mut self, mounts: &[usize]) {
        for &i in mounts {
            self.scenes
                .prefetch_texture(self.mounts[i], self.textures[i], 1);
        }
        frame(&mut self.scenes);
    }

    /// Frames of `mounts`' coarse tiles, each checked exact, until the pool has drained and
    /// rested for eight of them; returns the most it had draining.
    fn drain(&mut self, dev: &Device, mounts: &[usize]) -> u64 {
        let start = Instant::now();
        let (mut settled, mut drained) = (0, 0);
        while settled < 8 {
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "the tile pool never settled: {:?}",
                self.pool()
            );
            self.frame(mounts);
            for &i in mounts {
                assert_tiled(&self.scenes, dev, self.mounts[i], self.kinds[i], 1..9);
            }
            let fill = self.pool();
            drained = drained.max(fill.draining);
            let coarse = mounts.len() as u64 * COARSE_TILES * self.tile;
            let idle = fill.used == coarse && fill.draining == 0 && fill.retired == 0;
            settled = if idle { settled + 1 } else { 0 };
        }
        drained
    }

    /// Frames of every cube's coarse tiles until a step has started moving tiles.
    fn until_moving(&mut self) {
        let all: Vec<usize> = (0..self.mounts.len()).collect();
        let start = Instant::now();
        while self.pool().draining == 0 {
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "no block drained"
            );
            self.frame(&all);
        }
    }
}

/// The texture budget falls to the cube's coarse tiles: the fine ones go and leave each block
/// one tile, and the pool empties half of the blocks into the others. The moved tiles are
/// bound to their new slots between frames: they never leave the residency map, nothing is
/// read again, and every frame's texels are the cooked ones.
#[test]
fn tiles_move_out_of_draining_blocks_bound_and_exact_at_every_frame() {
    let _serial = serial();
    let dev = device();
    if !dev.binds_sparse_on(QueueKind::Render) {
        return;
    }
    let gpu = Gpu(dev);
    let mut c = Tiled::new(dev, &[CUBE]);
    let full = c.pool();
    assert_eq!(
        (full.used, full.bytes),
        (30 * c.tile, 32 * c.tile),
        "{full:?}"
    );
    let read = c.scenes.residency().bytes_read;
    c.keep_coarse();
    assert!(c.drain(dev, &[0]) > 0, "no block drained");
    let map = residency_map(&c.scenes, &gpu, c.textures[0]);
    assert!(
        map.iter().all(|&mip| mip == 1),
        "a coarse tile left: {map:?}"
    );
    let fill = c.pool();
    assert!(
        fill.bytes <= 12 * c.tile,
        "at most a block past what six tiles need: {fill:?}"
    );
    assert_eq!(
        c.scenes.residency().bytes_read,
        read,
        "a tile was read again"
    );
    assert_eq!(c.scenes.residency().failed, 0);
    assert_eq!(
        dev.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

/// Three packs' fine tiles go in one step, and the blocks of the first two drain into the
/// third's. The first pack is unmounted while its tiles and the second's move in one batch:
/// the second's take the texels copied out of them, not those of the tiles that left.
#[test]
fn a_pack_unmounted_while_tiles_move_leaves_the_other_packs_texels() {
    let _serial = serial();
    let dev = device();
    if !dev.binds_sparse_on(QueueKind::Render) {
        return;
    }
    let mut c = Tiled::new(dev, &[0, CUBE, CUBE]);
    // Every tile leaves its hold at one frame.
    for i in 0..3 {
        c.scenes.prefetch_texture(c.mounts[i], c.textures[i], 0);
    }
    frame(&mut c.scenes);
    c.keep_coarse();
    c.until_moving();
    c.scenes.unmount(c.mounts[0]);
    let kept = 2 * (c.tails + COARSE_TILES * c.tile);
    c.scenes.lower_budget(ResidencyClass::Textures, kept);
    c.drain(dev, &[1, 2]);
    assert_eq!(c.scenes.residency().failed, 0);
    assert_eq!(
        dev.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

/// Frames begun and never submitted while tiles move: the copies their lists would have run
/// wait for the next frame that is, and the tiles keep the cooked texels.
#[test]
fn frames_dropped_while_tiles_move_leave_their_copies_to_the_next() {
    let _serial = serial();
    let dev = device();
    if !dev.binds_sparse_on(QueueKind::Render) {
        return;
    }
    let mut c = Tiled::new(dev, &[CUBE]);
    c.keep_coarse();
    c.until_moving();
    for _ in 0..3 {
        c.scenes.prefetch_texture(c.mounts[0], c.textures[0], 1);
        let dropped = c
            .scenes
            .begin_frame(&FrameDesc {
                time: FrameTime::default(),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        drop(dropped);
        c.frame(&[0]);
        assert_tiled(&c.scenes, dev, c.mounts[0], CUBE, 1..9);
    }
    c.drain(dev, &[0]);
    assert_eq!(c.scenes.residency().failed, 0);
    assert_eq!(
        dev.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

/// Moving tiles are evicted while their bind waits: they leave their move, stay bound where
/// they were until their eviction unbinds them, and the slots they would have taken return.
#[test]
fn moving_tiles_evicted_while_their_bind_waits_leave_their_move() {
    let _serial = serial();
    let dev = device();
    if !dev.binds_sparse_on(QueueKind::Render) {
        return;
    }
    let mut c = Tiled::new(dev, &[CUBE]);
    c.keep_coarse();
    c.until_moving();
    c.scenes.hold_tile_binds(true);
    c.scenes.lower_budget(ResidencyClass::Textures, c.tails);
    let start = Instant::now();
    while c
        .scenes
        .residency()
        .class(ResidencyClass::Textures)
        .resident_bytes
        > c.tails
    {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the tiles stayed"
        );
        frame(&mut c.scenes);
    }
    c.scenes.hold_tile_binds(false);
    let start = Instant::now();
    loop {
        frame(&mut c.scenes);
        let fill = c.pool();
        if fill.blocks == 0 {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "slots stayed taken: {fill:?}"
        );
    }
    assert_tiled(&c.scenes, dev, c.mounts[0], CUBE, 2..9);
    assert_eq!(c.scenes.residency().failed, 0);
    assert_eq!(
        dev.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

/// The scratch memory of tile moves runs out as a block drains: the frames go on with every
/// tile exact and the budgets hear of it, and the block empties once memory returns.
#[test]
fn tile_moves_without_scratch_memory_wait_and_leave_the_frames_exact() {
    let _serial = serial();
    let dev = device();
    if !dev.binds_sparse_on(QueueKind::Render) {
        return;
    }
    let mut c = Tiled::new(dev, &[CUBE]);
    let failures = c.scenes.residency().out_of_memory;
    c.scenes.fail_tile_scratch(true);
    c.keep_coarse();
    c.until_moving();
    for _ in 0..4 {
        c.frame(&[0]);
        assert_tiled(&c.scenes, dev, c.mounts[0], CUBE, 1..9);
    }
    assert!(c.scenes.residency().out_of_memory > failures);
    c.scenes.fail_tile_scratch(false);
    c.drain(dev, &[0]);
    assert_eq!(c.scenes.residency().failed, 0);
    assert_eq!(
        dev.validation_messages(),
        0,
        "validation messages (see the log)"
    );
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
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|c| u16::from_le_bytes(*c))
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
