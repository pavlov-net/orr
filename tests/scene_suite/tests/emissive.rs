//! The emissive list: instance entries and producer entries, their power summed on the GPU.

use std::sync::Arc;

use orr_cook::{
    ArchiveSpec, AssetSpec, Instances, Linear, LodMode, Manifest, MeshInput, MeshSettings,
    Placement, PrototypeSpec, Source,
};
use orr_gpu::{GpuPtr, MemoryKind};
use orr_math::{Affine3, DMat3, DVec3, Mat3, Quat, Vec2, Vec3};
use orr_pack::material::Material;
use orr_pack::mesh::{Cluster, MeshInfo};
use orr_pack::{Archive, AssetId, Class, Span, Target};
use orr_scene::testing::{assert_clean, device, mount_pending, read_buffer, wait};
use orr_scene::{
    Budgets, ClusterRef, ClusterTopology, CopyVariant, EditDesc, EmissiveGpu, FrameDesc, FrameTime,
    GroupWriter, InstanceFlags, InstanceGpu, InstancePageRef, InstanceRange, InstancesDesc,
    LoadState, MaterialGpu, MaterialId, MeshGpu, ModelGpu, ModelId, MountId, PackSource, PoolSize,
    ProducerDesc, ProducerId, ProducerUpdate, RendererDesc, RtCutGpu, SceneCompute, SceneDesc,
    SceneId, SceneRoot, Scenes, SlotStateGpu, TransformGroupDesc, TransformGroupId, material_flags,
};
use orr_shaders::bytemuck::Zeroable;
use scene_suite::shaders::emit;
use scene_suite::{Workers, pipeline};

const PRODUCERS: u32 = 4;

fn frame(scenes: &[SceneId]) -> FrameDesc<'_> {
    FrameDesc {
        time: FrameTime {
            absolute: 0.0,
            delta: 1.0 / 60.0,
        },
        scenes,
        low_latency: None,
    }
}

fn emitting(color: [f32; 3], luminance: f32) -> MaterialGpu {
    let mut m = MaterialGpu::zeroed();
    m.emission_color = color;
    m.emission_luminance = luminance;
    m.flags = material_flags::EMISSIVE;
    m
}

fn entries(root: &SceneRoot) -> Vec<EmissiveGpu> {
    read_buffer(
        root.emissive,
        u64::from(root.producer_capacity + root.emissive_capacity),
    )
}

fn close(got: f32, expected: f64, what: &str) {
    let tolerance = 1e-3 * expected.abs().max(1e-6);
    assert!(
        (f64::from(got) - expected).abs() <= tolerance,
        "{what}: {got} against {expected}"
    );
}

fn close3(got: [f32; 3], expected: DVec3, what: &str) {
    for i in 0..3 {
        close(got[i], expected[i], what);
    }
}

/// A mesh of `n` by `n` quads in two sections, the first the quads of its first `block.0`
/// columns and `block.1` rows, wavy so its clusters differ.
fn sheet(n: u32, block: (u32, u32)) -> MeshInput {
    let mut m = MeshInput::default();
    for j in 0..=n {
        for i in 0..=n {
            let (x, z) = (i as f32 / n as f32 * 3.0, j as f32 / n as f32 * 3.0);
            m.positions.push(Vec3::new(x, 0.2 * (x * 2.0).sin(), z));
            m.normals.push(Vec3::Y);
        }
    }
    m.uvs = Some(vec![Vec2::ZERO; m.positions.len()]);
    let w = n + 1;
    for inside in [true, false] {
        let first = m.indices.len() as u32;
        for j in 0..n {
            for i in (0..n).filter(|&i| (i < block.0 && j < block.1) == inside) {
                let c = j * w + i;
                m.indices.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
            }
        }
        m.sections.push(Span {
            first,
            count: m.indices.len() as u32 - first,
        });
    }
    m
}

fn cooked(color: [f32; 3], luminance: f32) -> Material {
    Material {
        emission_color: color,
        emission_luminance: luminance,
        ..Material::default()
    }
}

fn pack_of(assets: Vec<AssetSpec>) -> Arc<[u8]> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0)
}

fn sheet_asset(name: &str, mesh: MeshInput, materials: Vec<Material>, lod: LodMode) -> AssetSpec {
    AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings {
                lod,
                ..MeshSettings::default()
            },
        },
    }
}

fn lamp_materials() -> Vec<Material> {
    vec![
        cooked([1.0, 0.0, 0.0], 100.0),
        cooked([0.0, 0.0, 1.0], 10.0),
    ]
}

/// A model `lamp` of two emissive sections, and `plain` of none.
fn pack() -> Arc<[u8]> {
    pack_of(vec![
        sheet_asset(
            "lamp",
            sheet(24, (24, 12)),
            lamp_materials(),
            LodMode::default(),
        ),
        sheet_asset(
            "plain",
            sheet(24, (24, 12)),
            vec![Material::default(), Material::default()],
            LodMode::default(),
        ),
    ])
}

fn read_blob(a: &Archive, b: &orr_pack::Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

#[test]
fn instance_entries_follow_scale_material_sets_hidden_sections_and_removal() {
    let d = device();
    let bytes = pack();
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let entry = archive.find(AssetId::from_name("lamp/mesh")).unwrap();
    let info = MeshInfo::parse(&read_blob(&archive, &archive.blobs(entry)[0])).unwrap();
    let sets = &info.emissive.sets;
    assert_eq!(sets.len(), 2);

    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            producer_capacity: PRODUCERS,
            emissive_capacity: 12,
            ..SceneDesc::default()
        })
        .unwrap();
    let m = scenes.mount(PackSource::Memory(bytes));
    while mount_pending(&scenes, m) {
        let f = scenes.begin_frame(&frame(&[])).unwrap();
        f.end(&[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let model = |name: &str| scenes.model(m, AssetId::from_name(name)).unwrap();
    let (lamp_model, plain) = (model("lamp"), model("plain"));
    let material = |name: &str| scenes.material(m, AssetId::from_name(name)).unwrap();
    let (lamp, shade) = (material("lamp/material/0"), material("lamp/material/1"));

    let green = [0.0, 1.0, 0.0];
    let list = [s];
    let f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut mats = f.materials();
    let other = mats.create(&emitting(green, 1000.0)).unwrap();
    let dark = mats.create(&MaterialGpu::zeroed()).unwrap();
    wait(f.end(&[]).unwrap());

    let scales = [
        Mat3::IDENTITY,
        Mat3::from_diagonal(Vec3::splat(2.0)),
        Mat3::from_diagonal(Vec3::new(2.0, 1.0, 0.5)),
        Mat3::from_rotation_y(0.7) * Mat3::from_diagonal(Vec3::splat(3.0)),
        Mat3::IDENTITY,
        Mat3::IDENTITY,
    ];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut w = f.scene(s).unwrap();
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 8,
            writer: GroupWriter::Cpu,
            anchor: DVec3::new(5000.0, 0.0, 0.0),
        })
        .unwrap();
    for (t, m) in w.transforms(g, 0, 6).unwrap().iter_mut().zip(scales) {
        *t = Affine3::from_mat3_translation(m, Vec3::new(1.0, 2.0, 3.0)).into();
    }
    let desc = |model, copies, material_set| InstancesDesc {
        model,
        copies,
        transforms: g,
        first: 0,
        pose: None,
        material_set,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 100.0,
        animation_range: 0.0,
        user_tag: 0,
    };
    // A model without emissive sections adds no entries.
    w.add_instances(&desc(plain, 2, None)).unwrap();
    let lamps = w.add_instances(&desc(lamp_model, 4, None)).unwrap();
    let mut set = desc(lamp_model, 1, Some(10));
    set.first = 4;
    let with_set = w.add_instances(&set).unwrap();
    // Set 1 of the copy's material sets holds (other, dark).
    w.material_sets(10, 4)
        .unwrap()
        .copy_from_slice(&[lamp, lamp, other, dark]);
    let set_one = CopyVariant {
        pose: 0,
        material_set: 1,
        section_mask: 0,
    };
    w.copy_variants(with_set, 0, &[set_one]).unwrap();
    // Copy 3 hides section 1.
    let hidden = CopyVariant {
        pose: 0,
        material_set: 0,
        section_mask: 0b10,
    };
    w.copy_variants(lamps, 3, &[hidden]).unwrap();
    // A copy outside ray tracing keeps its entries and emits nothing.
    let mut untraced = desc(lamp_model, 1, None);
    untraced.first = 5;
    untraced.flags = InstanceFlags::RASTER;
    let untraced = w.add_instances(&untraced).unwrap();
    // Twelve entries hold six copies of two sections; a seventh does not fit, and the failed
    // call leaves its slot free.
    let full = w.add_instances(&desc(lamp_model, 1, None));
    assert!(full.is_err());
    let next = w.add_instances(&desc(plain, 1, None)).unwrap();
    assert_eq!(next.first, untraced.first + 1);
    drop(w);
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());
    let root: SceneRoot = read_buffer(root, 1)[0];
    let emission = |id: MaterialId| -> DVec3 {
        let m: MaterialGpu = read_buffer(root.materials.offset(u64::from(id.0)), 1)[0];
        DVec3::from(m.emission_color.map(f64::from)) * f64::from(m.emission_luminance)
    };

    let det23 = |m: Mat3| f64::from(m.determinant().abs()).powf(2.0 / 3.0);
    // (slot, copy, rank, emission), None when the entry emits nothing.
    let check = |all: &[EmissiveGpu], live: &[(u32, usize, u32, Option<DVec3>)]| {
        let instances: Vec<InstanceGpu> = read_buffer(root.instances, u64::from(root.part_base));
        let mut seen = 0;
        for (at, e) in all.iter().enumerate().skip(PRODUCERS as usize) {
            if e.kind == 0 {
                continue;
            }
            let &(_, copy, rank, material) = live
                .iter()
                .find(|l| l.0 == e.source && l.2 == e.rank)
                .unwrap_or_else(|| panic!("unexpected entry {e:?}"));
            assert_eq!(e.kind, 1);
            let inst = instances[e.source as usize];
            assert_eq!(inst.emissive_count, 2);
            assert_eq!(inst.first_emissive + rank, at as u32);
            let set = sets[rank as usize];
            let area = f64::from(set.total) * det23(scales[copy]);
            match material {
                Some(l) => {
                    assert_eq!((e.set, e.cluster_count), (rank, set.count));
                    close(e.area, area, "area");
                    close3(e.power, l * area, "power");
                }
                None => assert_eq!((e.area, e.power), (0.0, [0.0; 3]), "{e:?}"),
            }
            seen += 1;
        }
        assert_eq!(seen, live.len());
    };
    let (red, blue) = (emission(lamp), emission(shade));
    let mut live = Vec::new();
    for copy in 0..4 {
        let slot = lamps.first + copy as u32;
        live.push((slot, copy, 0, Some(red)));
        live.push((slot, copy, 1, (copy != 3).then_some(blue)));
    }
    live.push((with_set.first, 4, 0, Some(emission(other))));
    live.push((with_set.first, 4, 1, Some(DVec3::ZERO)));
    live.push((untraced.first, 5, 0, None));
    live.push((untraced.first, 5, 1, None));
    check(&entries(&root), &live);

    // Removing copies 1 and 2 clears their entries; the rest keep their power.
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    f.scene(s).unwrap().remove(InstanceRange {
        first: lamps.first + 1,
        count: 2,
    });
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    live.retain(|&(slot, ..)| slot != lamps.first + 1 && slot != lamps.first + 2);
    check(&entries(&root), &live);

    // Once that frame has completed, the freed entries take two new copies.
    for _ in 0..2 {
        let mut f = scenes.begin_frame(&frame(&list)).unwrap();
        f.update_scenes(&mut |_| {}).unwrap();
        wait(f.end(&[]).unwrap());
    }
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let again = f
        .scene(s)
        .unwrap()
        .add_instances(&desc(lamp_model, 2, None))
        .unwrap();
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    for copy in 0..2 {
        live.push((again.first + copy as u32, copy, 0, Some(red)));
        live.push((again.first + copy as u32, copy, 1, Some(blue)));
    }
    check(&entries(&root), &live);
    assert_clean();
}

/// The consumer's material sets hold every entry a write, a copy or a variant names, and keep
/// what was written as they grow. A base past the 2^31 a copy's record names is refused.
#[test]
fn material_sets_hold_every_entry_a_write_or_a_copy_names() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let m = scenes.mount(PackSource::Memory(pack()));
    while mount_pending(&scenes, m) {
        let f = scenes.begin_frame(&frame(&[])).unwrap();
        f.end(&[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // A model of two sections.
    let plain = scenes.model(m, AssetId::from_name("plain")).unwrap();
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let [a, b] = [0, 1].map(|_| f.materials().create(&MaterialGpu::zeroed()).unwrap());
    let mut w = f.scene(s).unwrap();
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 3,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let desc = |first, material_set| InstancesDesc {
        model: plain,
        copies: 1,
        transforms: g,
        first,
        pose: None,
        material_set,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 100.0,
        animation_range: 0.0,
        user_tag: 0,
    };
    let set = |material_set| {
        [CopyVariant {
            pose: 0,
            material_set,
            section_mask: 0,
        }]
    };
    w.material_sets(0, 2).unwrap().copy_from_slice(&[a, b]);
    let low = w.add_instances(&desc(0, Some(0))).unwrap();
    w.copy_variants(low, 0, &set(1)).unwrap();
    w.add_instances(&desc(1, Some(3))).unwrap();
    assert!(matches!(
        w.add_instances(&desc(2, Some(1 << 31))),
        Err(orr_scene::Error::Capacity { .. })
    ));
    drop(w);
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());

    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut w = f.scene(s).unwrap();
    w.copy_variants(low, 0, &set(2)).unwrap();
    w.material_sets(4, 2).unwrap().copy_from_slice(&[b, a]);
    drop(w);
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());
    let root: SceneRoot = read_buffer(root, 1)[0];
    assert_eq!(
        read_buffer(root.material_sets, 6),
        [a.0, b.0, 0, 0, b.0, a.0]
    );
    assert_clean();
}

#[test]
fn an_edit_of_a_dag_mesh_takes_the_sets_of_level_0() {
    let d = device();
    let bytes = pack_of(vec![sheet_asset(
        "lamp",
        sheet(64, (64, 32)),
        lamp_materials(),
        LodMode::Dag,
    )]);
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let entry = archive.find(AssetId::from_name("lamp/mesh")).unwrap();
    let info = MeshInfo::parse(&read_blob(&archive, &archive.blobs(entry)[0])).unwrap();
    assert_eq!(info.cut_rows[1], 0.0);
    let ranks = info.emissive.rank_count();
    let level0 = &info.emissive.sets[ranks as usize..][..ranks as usize];
    assert!(level0[0].count > info.emissive.sets[0].count);

    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            emissive_capacity: ranks,
            ..SceneDesc::default()
        })
        .unwrap();
    let m = scenes.mount(PackSource::Memory(bytes));
    while mount_pending(&scenes, m) {
        let f = scenes.begin_frame(&frame(&[])).unwrap();
        f.end(&[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let edit = scenes
        .create_edit(&EditDesc {
            model: scenes.model(m, AssetId::from_name("lamp")).unwrap(),
            touched: None,
            bounds: None,
        })
        .unwrap();
    while !matches!(scenes.edit_state(edit), LoadState::Ready) {
        let f = scenes.begin_frame(&frame(&[])).unwrap();
        wait(f.end(&[]).unwrap());
    }
    let model = scenes.edit(edit).unwrap().model;

    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
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
        model,
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
    wait(f.end(&[]).unwrap());
    let root: SceneRoot = read_buffer(root, 1)[0];

    let all = entries(&root);
    assert_eq!(all.iter().filter(|e| e.kind != 0).count(), ranks as usize);
    for e in all.iter().filter(|e| e.kind != 0) {
        let set = level0[e.rank as usize];
        assert_eq!((e.set, e.cluster_count), (ranks + e.rank, set.count));
        close(e.area, f64::from(set.total), "area");
    }
    assert_clean();
}

#[test]
fn a_copy_takes_the_sets_of_the_cached_cut_it_traces() {
    let d = device();
    if d.caps().cluster.max_tris == 0 || d.caps().cluster.max_partitions == 0 {
        return;
    }
    let bytes = pack_of(vec![sheet_asset(
        "lamp",
        sheet(64, (64, 32)),
        lamp_materials(),
        LodMode::Dag,
    )]);
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let entry = archive.find(AssetId::from_name("lamp/mesh")).unwrap();
    let info = MeshInfo::parse(&read_blob(&archive, &archive.blobs(entry)[0])).unwrap();
    let ranks = info.emissive.rank_count();

    // Every copy traces a cached cut.
    let desc = RendererDesc {
        rt_merge_pixels: 1e9,
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(d, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            emissive_capacity: ranks,
            ..SceneDesc::default()
        })
        .unwrap();
    let m = scenes.mount(PackSource::Memory(bytes));
    while mount_pending(&scenes, m) {
        let f = scenes.begin_frame(&frame(&[])).unwrap();
        f.end(&[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let model = scenes.model(m, AssetId::from_name("lamp")).unwrap();

    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut w = f.scene(s).unwrap();
    w.set_focus(DVec3::new(0.0, 3.0, 0.0));
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    w.transforms(g, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
    let copy = w
        .add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: g,
            first: 0,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    drop(w);
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());
    let root: SceneRoot = read_buffer(root, 1)[0];
    let record: ModelGpu = read_buffer(root.models.offset(u64::from(model.0)), 1)[0];
    let mesh: MeshGpu = read_buffer(root.meshes.offset(u64::from(record.mesh)), 1)[0];
    let cuts: Vec<RtCutGpu> = read_buffer(mesh.rt_cuts, u64::from(mesh.rt_cut_count));

    // Near the focus the copy traces a fine cut; far from it, a coarser one.
    let mut rows = Vec::new();
    for x in [0.0, 800.0] {
        for _ in 0..8 {
            let mut f = scenes.begin_frame(&frame(&list)).unwrap();
            let mut w = f.scene(s).unwrap();
            w.transforms(g, 0, 1).unwrap()[0] =
                Affine3::from_translation(Vec3::new(x, 0.0, 0.0)).into();
            drop(w);
            f.update_scenes(&mut |_| {}).unwrap();
            wait(f.end(&[]).unwrap());

            let all = entries(&root);
            let live: Vec<&EmissiveGpu> = all.iter().filter(|e| e.kind != 0).collect();
            assert_eq!(live.len(), ranks as usize);
            if live.iter().all(|e| e.cluster_count == 0) {
                continue;
            }
            let st: SlotStateGpu = read_buffer(root.slot_state.offset(u64::from(copy.first)), 1)[0];
            let row = cuts[usize::from(st.rt_lod)].row;
            for e in live {
                let at = row * ranks + e.rank;
                let set = info.emissive.sets[at as usize];
                assert_eq!((e.set, e.cluster_count), (at, set.count), "row {row}");
                close(e.area, f64::from(set.total), "area");
            }
            rows.push(row);
        }
    }
    let (first, last) = (rows[0], *rows.last().unwrap());
    assert!(last > first, "rows {rows:?}");
    assert_clean();
}

/// Area of the triangle `a b c` after `m`.
fn area(m: DMat3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    0.5 * (m * (b - a)).cross(m * (c - a)).length()
}

/// What `emit.slang` emits: per material index, the transformed area of `count` quads.
fn emitted(m: DMat3, count: u32, size: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    for c in 0..count {
        let s = size * f64::from(1 + c % 3);
        let o = DVec3::new(2.0 * f64::from(c), 0.0, 0.0);
        let p = [
            o,
            o + DVec3::new(s, 0.0, 0.0),
            o + DVec3::new(s, 0.5 * s, s),
            o + DVec3::new(0.0, 0.0, s),
        ];
        out[(c % 3) as usize] += area(m, p[0], p[1], p[2]) + area(m, p[0], p[2], p[3]);
    }
    out
}

#[test]
fn producer_power_follows_emitted_clusters_every_frame() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            producer_capacity: PRODUCERS,
            emissive_capacity: 0,
            ..SceneDesc::default()
        })
        .unwrap();
    let p = pipeline(&emit::MODULE, "main");
    let mut quad = d
        .gpu_malloc::<u8>(MemoryKind::Upload, 8, "quad triangles")
        .unwrap();
    quad.as_mut_slice()
        .unwrap()
        .copy_from_slice(&[0, 1, 2, 0, 2, 3, 0, 0]);
    let topology = [ClusterTopology {
        vertex_count: 4,
        triangles: &[[0, 1, 2], [0, 2, 3]],
    }];
    let linear = Mat3::from_rotation_z(0.4)
        * Mat3::from_cols(
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(0.3, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.75),
        );
    let m = linear.as_dmat3();
    let (warm, cool) = ([1.0, 0.5, 0.25], [0.25, 0.5, 1.0]);
    let luminance = [
        DVec3::new(1.0, 0.5, 0.25) * 50.0,
        DVec3::ZERO,
        DVec3::new(0.25, 0.5, 1.0) * 4.0,
    ];
    let list = [s];
    let dispatch = |c: &mut SceneCompute<'_, '_>, emits: &[(ProducerId, u32, f32)]| {
        for &(producer, count, size) in emits {
            let r = c
                .put_root(&emit::Root {
                    scene: c.root,
                    triangles: quad.gpu(),
                    producer: producer.0,
                    count,
                    size,
                    _pad: 0,
                })
                .unwrap();
            c.cmd.dispatch(&p, r, [count.div_ceil(64).max(1), 1, 1]);
        }
    };
    let run = |scenes: &mut Scenes,
               write: &mut dyn FnMut(&mut orr_scene::SceneWriter<'_>),
               emits: &[(ProducerId, u32, f32)]|
     -> SceneRoot {
        let mut f = scenes.begin_frame(&frame(&list)).unwrap();
        write(&mut f.scene(s).unwrap());
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| {
            root = c.root;
            dispatch(c, emits);
        })
        .unwrap();
        wait(f.end(&[]).unwrap());
        read_buffer(root, 1)[0]
    };

    let mut materials = [MaterialId(0); 3];
    let mut group = TransformGroupId(0);
    run(
        &mut scenes,
        &mut |w| {
            group = w
                .create_transform_group(&TransformGroupDesc {
                    count: 2,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::new(-3000.0, 10.0, 7000.0),
                })
                .unwrap();
            w.transforms(group, 0, 2).unwrap().copy_from_slice(&[
                Affine3::from_mat3_translation(linear, Vec3::new(4.0, 0.0, -2.0)).into(),
                Affine3::IDENTITY.into(),
            ]);
        },
        &[],
    );
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut mats = f.materials();
    materials[0] = mats.create(&emitting(warm, 50.0)).unwrap();
    materials[1] = mats.create(&MaterialGpu::zeroed()).unwrap();
    materials[2] = mats.create(&emitting(cool, 4.0)).unwrap();
    let mut w = f.scene(s).unwrap();
    let producer = |update, max_clusters, transform| ProducerDesc {
        max_clusters,
        max_vertices: 4 * max_clusters,
        update,
        topologies: &topology,
        materials: &materials,
        transform: (group, transform),
        previous_positions: false,
        vertex_user: false,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        user_tag: 0,
        ..ProducerDesc::default()
    };
    w.set_producer(
        ProducerId(1),
        Some(&producer(ProducerUpdate::EveryFrame, 100, 0)),
    )
    .unwrap();
    w.set_producer(
        ProducerId(2),
        Some(&producer(ProducerUpdate::Static, 16, 1)),
    )
    .unwrap();
    drop(w);
    // A static entry emits in the frame of its `set_producer` only.
    f.update_scenes(&mut |c| dispatch(c, &[(ProducerId(2), 11, 0.5)]))
        .unwrap();
    wait(f.end(&[]).unwrap());

    let expected = |m: DMat3, count: u32, size: f64| {
        let a = emitted(m, count, size);
        let power = (0..3).map(|i| luminance[i] * a[i]).sum::<DVec3>();
        let clusters = (0..count).filter(|c| c % 3 != 1).count() as u32;
        (a[0] + a[2], power, clusters)
    };
    let check = |e: &EmissiveGpu, (area, power, clusters): (f64, DVec3, u32), what: &str| {
        assert_eq!(e.kind, 2, "{what}");
        assert_eq!(e.cluster_count, clusters, "{what}");
        close(e.area, area, what);
        close3(e.power, power, what);
    };
    // The static entry emits once; the other changes count and size every frame with no
    // write to the list.
    let statics = expected(DMat3::IDENTITY, 11, 0.5);
    for &(count, size) in &[
        (11, 0.5),
        (7, 1.0),
        (64, 0.25),
        (99, 2.0),
        (3, 0.125),
        (0, 1.0),
    ] {
        let root = run(&mut scenes, &mut |_| {}, &[(ProducerId(1), count, size)]);
        let all = entries(&root);
        check(
            &all[1],
            expected(m, count, f64::from(size)),
            "every-frame entry",
        );
        check(&all[2], statics, "static entry");
        assert_eq!((all[0].kind, all[3].kind), (0, 0));
    }

    // Past capacity the batch keeps 100 clusters and drops the rest.
    let root = run(&mut scenes, &mut |_| {}, &[(ProducerId(1), 150, 1.0)]);
    let all = entries(&root);
    let full = expected(m, 150, 1.0);
    assert!(all[1].cluster_count <= 100 && all[1].cluster_count >= 50);
    assert!(f64::from(all[1].area) < full.0 && all[1].area > 0.0);

    // Moving the entry's transform changes its power with no write to the list.
    let root = run(
        &mut scenes,
        &mut |w| {
            w.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
        },
        &[(ProducerId(1), 20, 1.0)],
    );
    check(
        &entries(&root)[1],
        expected(DMat3::IDENTITY, 20, 1.0),
        "moved entry",
    );

    // A cleared entry leaves the list.
    let root = run(
        &mut scenes,
        &mut |w| w.set_producer(ProducerId(1), None).unwrap(),
        &[],
    );
    let all = entries(&root);
    assert_eq!((all[1].kind, all[1].power), (0, [0.0; 3]));
    check(&all[2], statics, "static entry");
    assert_clean();
}

/// `lights.slang`'s `GATHERED_SET` and `GATHERED_CLUSTERS`.
const GATHERED_SET: u32 = 0xffff_fffe;
const GATHERED_CLUSTERS: u32 = 16;

/// Mounts `bytes` on new scenes with one scene of `emissive_capacity` instance entries, and the
/// workers that load it.
fn mounted(bytes: Arc<[u8]>, emissive_capacity: u32) -> (Scenes, SceneId, MountId, Workers) {
    let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            emissive_capacity,
            ..SceneDesc::default()
        })
        .unwrap();
    let m = scenes.mount(PackSource::Memory(bytes));
    while mount_pending(&scenes, m) {
        let f = scenes.begin_frame(&frame(&[])).unwrap();
        f.end(&[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    (scenes, s, m, workers)
}

/// One copy of `model` at `transform`, traced, around a focus at the origin.
fn place(
    scenes: &mut Scenes,
    s: SceneId,
    model: ModelId,
    transform: Affine3,
) -> (InstanceRange, TransformGroupId) {
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut w = f.scene(s).unwrap();
    w.set_focus(DVec3::ZERO);
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    w.transforms(g, 0, 1).unwrap()[0] = transform.into();
    let copy = w
        .add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: g,
            first: 0,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1e5,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    drop(w);
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    (copy, g)
}

/// Runs a frame of `s`, moving group `g`'s transform to `transform` first where given, and
/// returns the scene's root.
fn run_frame(
    scenes: &mut Scenes,
    s: SceneId,
    moved: Option<(TransformGroupId, Affine3)>,
) -> SceneRoot {
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    if let Some((g, t)) = moved {
        let mut w = f.scene(s).unwrap();
        w.transforms(g, 0, 1).unwrap()[0] = t.into();
    }
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());
    read_buffer(root, 1)[0]
}

/// A copy of a DAG mesh whose sets hold no tree gathers its cut, fine near the focus and coarse
/// far from it; where the cut it traces is one built that frame, each entry samples the list
/// its traversal gathered: the traced clusters of its section, at most `GATHERED_CLUSTERS`, with
/// a fixed-point CDF by cluster area ending at the top. The sheet's one emissive section is a
/// corner block of few enough clusters for a gathered list, and the sheet is large enough for
/// its DAG to hold many groups, so its cut changes across it. Its pages stream, so the cut is
/// first gathered while some are not resident.
#[test]
fn a_copy_tracing_a_cut_it_built_samples_its_gathered_clusters() {
    let d = device();
    if d.caps().cluster.max_tris == 0 || d.caps().cluster.max_partitions == 0 {
        return;
    }
    let bytes = pack_of(vec![sheet_asset(
        "lamp",
        sheet(128, (16, 16)),
        vec![cooked([1.0, 0.0, 0.0], 100.0), Material::default()],
        LodMode::Dag,
    )]);
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let entry = archive.find(AssetId::from_name("lamp/mesh")).unwrap();
    let info = MeshInfo::parse(&read_blob(&archive, &archive.blobs(entry)[0])).unwrap();
    assert!(
        info.emissive.sets.iter().all(|s| s.node_count == 0),
        "the sheet's sets hold no tree"
    );
    assert!(
        info.emissive.widest <= GATHERED_CLUSTERS,
        "the block's cuts fit a gathered list"
    );
    let ranks = info.emissive.rank_count();
    let (mut scenes, s, m, _workers) = mounted(bytes, ranks);
    let model = scenes.model(m, AssetId::from_name("lamp")).unwrap();
    // 100 m across, its near corner at the focus.
    let (copy, _) = place(
        &mut scenes,
        s,
        model,
        Affine3::from_scale(Vec3::splat(33.0)),
    );
    let mut gathered = 0;
    for _ in 0..8 {
        let root = run_frame(&mut scenes, s, None);
        let st: SlotStateGpu = read_buffer(root.slot_state.offset(u64::from(copy.first)), 1)[0];
        let record: ModelGpu = read_buffer(root.models.offset(u64::from(model.0)), 1)[0];
        let mesh: MeshGpu = read_buffer(root.meshes.offset(u64::from(record.mesh)), 1)[0];
        let live: Vec<(u32, EmissiveGpu)> = (0..)
            .zip(entries(&root))
            .filter(|(_, e)| e.kind != 0)
            .collect();
        assert_eq!(live.len(), ranks as usize);
        if st.rt_lod != 255 {
            assert!(
                live.iter().all(|(_, e)| e.set != GATHERED_SET),
                "a cached cut's entries"
            );
            continue;
        }
        gathered += 1;
        for (i, e) in live {
            assert_eq!(e.set, GATHERED_SET, "rank {}", e.rank);
            let n = e.gathered.min(GATHERED_CLUSTERS);
            assert!(n > 0, "rank {}: nothing gathered", e.rank);
            assert_eq!(e.cluster_count, n);
            let at = (i - root.producer_capacity) * 2 * GATHERED_CLUSTERS;
            let list: Vec<u32> = read_buffer(
                root.emissive_lists.offset(u64::from(at)),
                u64::from(2 * GATHERED_CLUSTERS),
            );
            let section = info
                .emissive
                .ranks
                .iter()
                .position(|&r| r == e.rank)
                .unwrap();
            let areas: Vec<f64> = list[..n as usize]
                .iter()
                .map(|&c| {
                    let r: ClusterRef = read_buffer(root.clusters.offset(u64::from(c)), 1)[0];
                    assert!(
                        r.mesh_cluster < mesh.cluster_count,
                        "rank {}: cluster id {c}",
                        e.rank
                    );
                    let at = 96 + u64::from(r.local) * size_of::<Cluster>() as u64;
                    let cl: Cluster = read_buffer(r.page.offset(at).cast(), 1)[0];
                    assert_eq!(
                        usize::from(cl.section),
                        section,
                        "rank {}: cluster {c}",
                        e.rank
                    );
                    f64::from(cl.area)
                })
                .collect();
            let cdf = &list[GATHERED_CLUSTERS as usize..(GATHERED_CLUSTERS + n) as usize];
            assert_eq!(
                *cdf.last().unwrap(),
                u32::MAX,
                "rank {}: last value",
                e.rank
            );
            let total: f64 = areas.iter().sum();
            let mut sum = 0.0;
            for (k, a) in areas.iter().enumerate() {
                sum += a;
                let expected = sum / total * f64::from(u32::MAX);
                assert!(
                    (f64::from(cdf[k]) - expected).abs() <= 1e-5 * f64::from(u32::MAX),
                    "rank {}: CDF value {k}",
                    e.rank
                );
            }
        }
    }
    assert!(gathered > 0, "the copy never traced a cut it built");
    assert_clean();
}

/// A chain whose farther model has more emissive sections than the model gives every live copy
/// an entry per section of the most emissive step, so the copy emits at that step.
#[test]
fn a_chain_with_a_more_emissive_step_grows_its_copies_entries() {
    let d = device();
    if d.caps().cluster.max_tris == 0 || d.caps().cluster.max_partitions == 0 {
        return;
    }
    let (mut scenes, s, m, _workers) = mounted(pack(), 8);
    let lamp = scenes.model(m, AssetId::from_name("lamp")).unwrap();
    let plain = scenes.model(m, AssetId::from_name("plain")).unwrap();
    let (copy, g) = place(&mut scenes, s, plain, Affine3::IDENTITY);
    let root = run_frame(&mut scenes, s, None);
    let record: InstanceGpu = read_buffer(root.instances.offset(u64::from(copy.first)), 1)[0];
    assert_eq!(
        (record.first_emissive, record.emissive_count),
        (u32::MAX, 0)
    );

    scenes.set_chain(plain, &[(lamp, 10.0)]).unwrap();
    let root = run_frame(&mut scenes, s, None);
    let record: InstanceGpu = read_buffer(root.instances.offset(u64::from(copy.first)), 1)[0];
    assert_eq!(record.emissive_count, 2, "entries after the chain");
    let all = entries(&root);
    for k in 0..2 {
        let e = all[(record.first_emissive + k) as usize];
        assert_eq!((e.kind, e.source, e.rank), (1, copy.first, k), "entry {k}");
    }
    // Past the switch distance the copy shows the lamp, whose sections emit.
    let far = Affine3::from_translation(Vec3::new(40.0, 0.0, 0.0));
    let mut root = root;
    for _ in 0..24 {
        root = run_frame(&mut scenes, s, Some((g, far)));
    }
    let all = entries(&root);
    let emitting = (0..2)
        .map(|k| all[(record.first_emissive + k) as usize])
        .filter(|e| e.cluster_count > 0 && e.power.iter().any(|&p| p > 0.0))
        .count();
    assert_eq!(emitting, 2, "the lamp's sections at its step");
    assert_clean();
}

/// The prototype of each record of [`page_pack`]'s page, in x order: 0 `lamp`, 1 `plain`.
const PAGE_PROTOTYPES: [u16; 6] = [0, 1, 1, 0, 1, 0];

/// [`pack`] and a page of its `lamp` and `plain` copies along x.
fn page_pack() -> Arc<[u8]> {
    let prototype = |model: &str| PrototypeSpec {
        model: model.into(),
        ..PrototypeSpec::default()
    };
    let scatter = AssetSpec {
        name: "scatter".into(),
        source: Source::Instances(Instances {
            records: PAGE_PROTOTYPES
                .iter()
                .zip(0..)
                .map(|(&prototype, i)| Placement {
                    position: [f64::from(i) * 9.0, 0.0, 0.0],
                    linear: Linear::Scaled {
                        rotation: Quat::IDENTITY.to_array(),
                        scale: [1.0; 3],
                    },
                    prototype,
                })
                .collect(),
            prototypes: vec![prototype("lamp"), prototype("plain")],
            ..Instances::default()
        }),
    };
    pack_of(vec![
        sheet_asset(
            "lamp",
            sheet(24, (24, 12)),
            lamp_materials(),
            LodMode::default(),
        ),
        sheet_asset(
            "plain",
            sheet(24, (24, 12)),
            vec![Material::default(), Material::default()],
            LodMode::default(),
        ),
        scatter,
    ])
}

/// A page reserves entries for each prototype's copies alone, and each copy takes its own by
/// its rank among its prototype's records; a chain that grows a prototype's copies moves the
/// page's entries, which its slots take again from its records, keeping the rest of their
/// records.
#[test]
fn a_page_takes_entries_for_each_prototype_s_copies() {
    let d = device();
    if d.caps().cluster.max_tris == 0 || d.caps().cluster.max_partitions == 0 {
        return;
    }
    let bytes = page_pack();
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let page_id = archive
        .entries()
        .iter()
        .find(|e| e.class == Class::InstancePage as u16)
        .unwrap()
        .id;
    let (lamps, all) = (3, PAGE_PROTOTYPES.len() as u32);
    // Room for the page's entries with the chain, and for the ones it holds before.
    let (mut scenes, s, m, _workers) = mounted(bytes, 2 * (lamps + all));
    let lamp = scenes.model(m, AssetId::from_name("lamp")).unwrap();
    let plain = scenes.model(m, AssetId::from_name("plain")).unwrap();
    let page = scenes.instance_page(m, page_id).unwrap();
    assert_eq!(scenes.page_needs(page).unwrap().emissive, 2 * lamps);

    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let range = f
        .scene(s)
        .unwrap()
        .add_instance_page(&InstancePageRef {
            page,
            frame: None,
            poses: &[],
            material_sets: &[],
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1e5,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());

    // The page's slots and the list once its records have written `want` entries.
    let settle = |scenes: &mut Scenes, want: u32| {
        for _ in 0..600 {
            let root = run_frame(scenes, s, None);
            let slots: Vec<InstanceGpu> = read_buffer(
                root.instances.offset(u64::from(range.first)),
                u64::from(all),
            );
            let held: u32 = slots.iter().map(|i| i.emissive_count).sum();
            if slots.iter().all(|i| i.flags != 0) && held == want {
                return (slots, entries(&root));
            }
        }
        panic!("the page's slots never held {want} entries");
    };
    let check = |slots: &[InstanceGpu], list: &[EmissiveGpu], strides: [u32; 2], what: &str| {
        let mut taken = Vec::new();
        for (i, (inst, &p)) in slots.iter().zip(&PAGE_PROTOTYPES).enumerate() {
            let model = [lamp, plain][usize::from(p)];
            assert_eq!(inst.model, model.0, "{what}: slot {i}'s model");
            assert_eq!(inst.group_index, i as u32, "{what}: slot {i}'s record");
            let stride = strides[usize::from(p)];
            assert_eq!(inst.emissive_count, stride, "{what}: slot {i}'s entries");
            if stride == 0 {
                assert_eq!(inst.first_emissive, u32::MAX, "{what}: slot {i}");
                continue;
            }
            let slot = range.first + i as u32;
            for k in 0..stride {
                let at = inst.first_emissive + k;
                let e = list[at as usize];
                assert_eq!(
                    (e.kind, e.source, e.rank),
                    (1, slot, k),
                    "{what}: entry {at}"
                );
                taken.push(at);
            }
        }
        let n = taken.len();
        taken.sort_unstable();
        taken.dedup();
        assert_eq!(taken.len(), n, "{what}: slots share entries");
    };
    let (slots, list) = settle(&mut scenes, 2 * lamps);
    check(&slots, &list, [2, 0], "placed");

    // Flags set after the records wrote the slots stay through the move.
    let listed = [s];
    let mut f = scenes.begin_frame(&frame(&listed)).unwrap();
    f.scene(s)
        .unwrap()
        .set_instance_flags(range, InstanceFlags::RAY);
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    scenes.set_chain(plain, &[(lamp, 10.0)]).unwrap();
    let (slots, list) = settle(&mut scenes, 2 * all);
    check(&slots, &list, [2, 2], "grown");
    assert!(
        slots.iter().all(|i| i.flags == InstanceFlags::RAY.0),
        "the move rewrote the slots' flags"
    );
    let live = list.iter().filter(|e| e.kind != 0).count() as u32;
    assert_eq!(
        live,
        2 * all,
        "the entries the page held before are emptied"
    );
    assert_clean();
}

/// A material set a page placement names for a prototype reaches that prototype's copies, and
/// only theirs.
#[test]
fn a_page_s_material_set_reaches_its_prototype_s_copies() {
    const BASE: u32 = 4;
    let d = device();
    if d.caps().cluster.max_tris == 0 || d.caps().cluster.max_partitions == 0 {
        return;
    }
    let bytes = page_pack();
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let page_id = archive
        .entries()
        .iter()
        .find(|e| e.class == Class::InstancePage as u16)
        .unwrap()
        .id;
    let (mut scenes, s, m, _workers) = mounted(bytes, 2 * 3);
    let page = scenes.instance_page(m, page_id).unwrap();
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let range = f
        .scene(s)
        .unwrap()
        .add_instance_page(&InstancePageRef {
            page,
            frame: None,
            poses: &[],
            material_sets: &[(1, BASE)],
            flags: InstanceFlags::RASTER,
            rt_range: 0.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    let all = PAGE_PROTOTYPES.len() as u64;
    for _ in 0..600 {
        let root = run_frame(&mut scenes, s, None);
        let slots: Vec<InstanceGpu> =
            read_buffer(root.instances.offset(u64::from(range.first)), all);
        if slots.iter().any(|i| i.flags == 0) {
            continue;
        }
        for (i, (inst, &p)) in slots.iter().zip(&PAGE_PROTOTYPES).enumerate() {
            let base = if p == 1 { BASE } else { u32::MAX };
            assert_eq!(inst.material_base, base, "slot {i}");
        }
        assert_clean();
        return;
    }
    panic!("the page's records never landed");
}

/// The clusters of streamed pages that `root`'s instance entries list this frame, after checking
/// that every cluster an entry lists, by a cooked set of `sets` or a gathered list, is the cluster
/// its id names, on a page resident this frame.
fn listed_streamed_clusters(root: &SceneRoot, sets: &orr_pack::mesh::ClusterSets) -> u32 {
    let mut streamed = 0;
    let all = entries(root);
    for (i, e) in (0..).zip(&all).skip(root.producer_capacity as usize) {
        if e.kind != 1 || e.set == u32::MAX {
            continue;
        }
        let inst: InstanceGpu = read_buffer(root.instances.offset(u64::from(e.source)), 1)[0];
        let model: ModelGpu = read_buffer(root.models.offset(u64::from(inst.model)), 1)[0];
        let mesh: MeshGpu = read_buffer(root.meshes.offset(u64::from(model.mesh)), 1)[0];
        let n = u64::from(mesh.page_count);
        let pages: Vec<u64> = read_buffer(mesh.pages, n);
        let firsts: Vec<u32> = read_buffer(mesh.page_clusters, n + 1);
        let bases: Vec<u32> = read_buffer(mesh.page_bases, n);
        let page_of = |c: u32| firsts.partition_point(|&f| f <= c) - 1;
        let ids: Vec<u32> = if e.set == GATHERED_SET {
            let at = (i - root.producer_capacity) * 2 * GATHERED_CLUSTERS;
            let n = e.gathered.min(GATHERED_CLUSTERS);
            read_buffer(root.emissive_lists.offset(u64::from(at)), u64::from(n))
        } else {
            let k = sets.sets[e.set as usize];
            sets.clusters[k.first as usize..(k.first + k.count) as usize]
                .iter()
                .map(|&c| {
                    let p = page_of(c);
                    assert_ne!(pages[p], 0, "a cooked set lists page {p}, which is absent");
                    bases[p] + c - firsts[p]
                })
                .collect()
        };
        assert_eq!(ids.len() as u32, e.cluster_count, "entry {i}'s clusters");
        assert!(e.area > 0.0, "entry {i} lists clusters of no area");
        for id in ids {
            let r: ClusterRef = read_buffer(root.clusters.offset(u64::from(id)), 1)[0];
            let p = page_of(r.mesh_cluster);
            assert_ne!(pages[p], 0, "entry {i}: id {id} is on absent page {p}");
            assert_eq!(r.page.address().0, pages[p], "entry {i}: id {id}'s page");
            assert_eq!(
                (bases[p] + r.local, firsts[p] + r.local),
                (id, r.mesh_cluster),
                "entry {i}: id {id}"
            );
            if p >= mesh.pinned_pages as usize {
                streamed += 1;
            }
        }
    }
    streamed
}

/// Copies of an emissive mesh in two mounts, under a mesh page budget that holds one mount's
/// streamed pages: as the near copy's pages land, and as they give way to the other mount's,
/// each frame's entries list only clusters of pages resident in that frame, by their ids.
#[test]
fn a_streaming_emissive_mesh_lists_only_resident_clusters() {
    let d = device();
    let c = d.caps().cluster;
    if c.max_tris == 0 || c.max_partitions == 0 {
        return;
    }
    let bytes = pack_of(vec![sheet_asset(
        "lamp",
        sheet(160, (160, 80)),
        lamp_materials(),
        LodMode::Dag,
    )]);
    let archive = Archive::from_memory(Arc::clone(&bytes)).unwrap();
    let entry = archive.find(AssetId::from_name("lamp/mesh")).unwrap();
    let info = MeshInfo::parse(&read_blob(&archive, &archive.blobs(entry)[0])).unwrap();
    let pinned = info.header.pinned_pages as usize;
    assert!(pinned < info.pages.len(), "the lamp has streamed pages");
    // Three mounts' pinned pages and one mount's streamed pages with their CLAS estimates, and
    // the room the detail keeps free beside what the view wants, under which a finer cut is
    // admitted: the second and third mounts' streamed pages, as built, take more.
    let page = |p: &orr_pack::mesh::PageInfo| u64::from(p.resident_len).next_multiple_of(256);
    let clas = |p: &orr_pack::mesh::PageInfo| {
        (32 * u64::from(p.tri_count) + 256 * u64::from(p.cluster_count)).next_multiple_of(128)
    };
    let wanted = 3 * info.pages[..pinned].iter().map(page).sum::<u64>()
        + info.pages[pinned..]
            .iter()
            .map(|p| page(p) + clas(p))
            .sum::<u64>();
    let budget = wanted * 5 / 4;
    let desc = RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            mesh_page_pool: PoolSize::Bytes(budget),
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut scenes = Scenes::new(d, &desc).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let s = scenes
        .create_scene(&SceneDesc {
            emissive_capacity: 4,
            ..SceneDesc::default()
        })
        .unwrap();
    let mut models = [ModelId(0); 3];
    for model in &mut models {
        let m = scenes.mount(PackSource::Memory(Arc::clone(&bytes)));
        while mount_pending(&scenes, m) {
            let f = scenes.begin_frame(&frame(&[])).unwrap();
            f.end(&[]).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        *model = scenes.model(m, AssetId::from_name("lamp")).unwrap();
    }
    let list = [s];
    // 100 m across, its near corner `x` m along x from the focus.
    let at = |x: f32| {
        Affine3::from_scale_rotation_translation(
            Vec3::splat(33.0),
            Quat::IDENTITY,
            Vec3::new(x, 0.0, 0.0),
        )
    };
    let step = |scenes: &mut Scenes, moves: &[(u32, f32)]| {
        let mut f = scenes.begin_frame(&frame(&list)).unwrap();
        let mut w = f.scene(s).unwrap();
        w.set_focus(DVec3::new(0.0, 3.0, 0.0));
        for &(i, x) in moves {
            w.transforms(TransformGroupId(1), i, 1).unwrap()[0] = at(x).into();
        }
        drop(w);
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| root = c.root).unwrap();
        wait(f.end(&[]).unwrap());
        let root: SceneRoot = read_buffer(root, 1)[0];
        (listed_streamed_clusters(&root, &info.emissive), root)
    };
    // The streamed pages of the first mount's mesh that are resident.
    let resident = |root: &SceneRoot| -> Vec<usize> {
        let model: ModelGpu = read_buffer(root.models.offset(u64::from(models[0].0)), 1)[0];
        let mesh: MeshGpu = read_buffer(root.meshes.offset(u64::from(model.mesh)), 1)[0];
        let pages: Vec<u64> = read_buffer(mesh.pages, u64::from(mesh.page_count));
        (pinned..pages.len()).filter(|&p| pages[p] != 0).collect()
    };
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut w = f.scene(s).unwrap();
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 2,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    assert_eq!(g, TransformGroupId(1));
    w.transforms(g, 0, 2).unwrap().fill(at(0.0).into());
    for (i, &model) in (0..).zip(&models[..2]) {
        w.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: g,
            first: i,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    }
    drop(w);
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());

    // Until the near copy traces a finer cut over its streamed pages, and a few frames on.
    let (mut streamed, mut root) = step(&mut scenes, &[(1, 800.0)]);
    let mut after = 0;
    for _ in 0..2000 {
        let (n, r) = step(&mut scenes, &[]);
        (streamed, root) = (streamed + n, r);
        if streamed > 0 {
            after += 1;
        }
        if after == 8 {
            break;
        }
    }
    assert_eq!(after, 8, "the near copy never listed a streamed cluster");
    let landed = resident(&root);
    assert!(!landed.is_empty(), "the near copy's pages landed");

    // The far mount's pages move in near the focus beside a third mount's, and the first mount's give way.
    step(&mut scenes, &[(0, 800.0), (1, 0.0)]);
    let mut after = 0;
    for _ in 0..2000 {
        scenes.prefetch(models[1], 0);
        scenes.prefetch(models[2], 0);
        let (_, root) = step(&mut scenes, &[]);
        let now = resident(&root);
        if after > 0 || landed.iter().any(|p| !now.contains(p)) {
            after += 1;
        }
        if after == 8 {
            break;
        }
    }
    assert_eq!(
        after, 8,
        "none of the first mount's streamed pages gave way"
    );
    assert_clean();
}
