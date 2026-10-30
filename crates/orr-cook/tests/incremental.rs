//! Incremental cook: keys, links from the replaced archive and the cache, determinism.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use glam::{DVec3, Quat, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, CookOptions, Cooked, DataFormat, GltfSettings, Instances, LodMode,
    Manifest, MipFilter, Outcome, Placement, PrototypeSpec, Quality, Source, TextureRules,
    TextureSettings, Usage,
};
use orr_pack::{Archive, AssetId, Class, Target};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_png(path: &Path, n: u32, seed: u32) {
    let img = image::RgbaImage::from_fn(n, n, |x, y| {
        image::Rgba([
            (x * 7 + seed) as u8,
            (y * 5 + 2 * seed) as u8,
            ((x ^ y) + seed) as u8,
            255,
        ])
    });
    img.save(path).unwrap();
}

/// A quad per material: positions scaled by `size`.
fn write_bin(path: &Path, size: f32) {
    let pos: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
    ];
    let nrm = [[0.0f32, 0.0, 1.0]; 4];
    let uv: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let idx: [u32; 6] = [0, 1, 2, 0, 2, 3];
    let mut bin = Vec::new();
    bin.extend_from_slice(bytemuck::cast_slice(&pos));
    bin.extend_from_slice(bytemuck::cast_slice(&nrm));
    bin.extend_from_slice(bytemuck::cast_slice(&uv));
    bin.extend_from_slice(bytemuck::cast_slice(&idx));
    std::fs::write(path, bin).unwrap();
}

/// `quad.gltf` with its buffer and three images as files beside it: material 0 reads a
/// base colour and a normal map, material 1 a base colour.
fn write_scene(d: &Path) {
    write_bin(&d.join("quad.bin"), 1.0);
    write_png(&d.join("a.png"), 64, 1);
    write_png(&d.join("b.png"), 64, 2);
    write_png(&d.join("c.png"), 64, 3);
    write_png(&d.join("tex.png"), 32, 4);
    let prim = |m: u32| {
        format!(
            r#"{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":{m}}}"#
        )
    };
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{},{}]}}],
"materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}},"normalTexture":{{"index":1}}}},
 {{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":2}}}}}}],
"textures":[{{"source":0}},{{"source":1}},{{"source":2}}],
"images":[{{"uri":"a.png"}},{{"uri":"b.png"}},{{"uri":"c.png"}}],
"buffers":[{{"uri":"quad.bin","byteLength":152}}],
"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":48}},
 {{"buffer":0,"byteOffset":96,"byteLength":32}},{{"buffer":0,"byteOffset":128,"byteLength":24}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
 {{"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"}},
 {{"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"}},
 {{"bufferView":3,"componentType":5125,"count":6,"type":"SCALAR"}}]}}"#,
        prim(0),
        prim(1)
    );
    std::fs::write(d.join("quad.gltf"), json).unwrap();
}

fn tex_settings(quality: Quality) -> TextureSettings {
    TextureSettings {
        quality,
        ..Default::default()
    }
}

fn assets(d: &Path, tex_quality: Quality) -> Vec<AssetSpec> {
    let records = (0..6)
        .map(|i| Placement {
            position: DVec3::new(f64::from(i) * 300.0 - 700.0, 2.0, f64::from(i % 2) * 300.0),
            rotation: Quat::from_rotation_y(i as f32),
            scale: Vec3::splat(1.0 + i as f32 * 0.1),
            prototype: 0,
        })
        .collect();
    vec![
        AssetSpec {
            name: "quad".into(),
            source: Source::Gltf {
                path: d.join("quad.gltf"),
                settings: GltfSettings {
                    textures: TextureRules {
                        quality: Quality::Fast,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        },
        AssetSpec {
            name: "tex".into(),
            source: Source::Texture {
                path: d.join("tex.png"),
                settings: tex_settings(tex_quality),
            },
        },
        AssetSpec {
            name: "data".into(),
            source: Source::User {
                class: Class::USER_FIRST + 3,
                schema: 2,
                version: 1,
                bytes: (0..5000u32).flat_map(|v| (v * v).to_le_bytes()).collect(),
            },
        },
        AssetSpec {
            name: "scatter".into(),
            source: Source::Instances(Instances {
                records,
                prototypes: vec![PrototypeSpec {
                    model: "quad".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
        },
    ]
}

fn manifest(d: &Path, output: Target, tex_quality: Quality) -> Manifest {
    Manifest {
        archives: vec![ArchiveSpec::new(output, assets(d, tex_quality))],
    }
}

fn opts(jobs: usize) -> CookOptions {
    CookOptions {
        jobs,
        ..Default::default()
    }
}

fn names(c: &Cooked, outcome: Outcome) -> BTreeSet<String> {
    c.report
        .assets
        .iter()
        .flat_map(|a| &a.entries)
        .filter(|e| e.outcome == outcome)
        .map(|e| e.name.clone())
        .collect()
}

fn entry_count(c: &Cooked) -> usize {
    c.report.assets.iter().map(|a| a.entries.len()).sum()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

fn set_mtime(path: &Path, t: SystemTime) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(t)
        .unwrap();
}

#[test]
fn same_inputs_give_the_same_bytes() {
    let d = dir("incremental-same");
    write_scene(&d);
    let memory = |o: &CookOptions| {
        orr_cook::cook(&manifest(&d, Target::Memory, Quality::Fast), o)
            .unwrap()
            .memory
            .remove(0)
    };
    let one = memory(&opts(1));
    assert_eq!(*one, *memory(&opts(8)));
    // Every job waits for the one before it to be written.
    let tight = CookOptions {
        jobs: 4,
        working_set: 1,
        reorder_window: 0,
        ..Default::default()
    };
    assert_eq!(*one, *memory(&tight));

    // The same assets in two archives cook once and give both archives the same bytes.
    let twice = Manifest {
        archives: vec![
            ArchiveSpec::new(Target::Memory, assets(&d, Quality::Fast)),
            ArchiveSpec::new(Target::Memory, assets(&d, Quality::Fast)),
        ],
    };
    let cache = CookOptions {
        jobs: 8,
        cache: Some(d.join("cache")),
        ..Default::default()
    };
    let both = orr_cook::cook(&twice, &cache).unwrap();
    assert_eq!(*one, *both.memory[0]);
    assert_eq!(*one, *both.memory[1]);

    let file = d.join("out.orr");
    orr_cook::cook(
        &manifest(&d, Target::File(file.clone()), Quality::Fast),
        &opts(3),
    )
    .unwrap();
    assert_eq!(*one, *std::fs::read(&file).unwrap());

    let a = Archive::from_memory(one).unwrap();
    let pages = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
        .count();
    assert!(pages >= 3, "{pages} instance pages");
    for e in a.entries() {
        a.verify(e).unwrap();
    }
}

#[test]
fn cache_links_equal_a_fresh_cook() {
    let d = dir("incremental-cache");
    write_scene(&d);
    let cache = d.join("cache");
    let with_cache = CookOptions {
        cache: Some(cache.clone()),
        ..Default::default()
    };
    let m = manifest(&d, Target::Memory, Quality::Fast);
    let fresh = orr_cook::cook(&m, &opts(0)).unwrap();
    let first = orr_cook::cook(&m, &with_cache).unwrap();
    assert_eq!(names(&first, Outcome::Cooked).len(), entry_count(&first));
    let linked = orr_cook::cook(&m, &with_cache).unwrap();
    assert!(names(&linked, Outcome::Cooked).is_empty());
    assert_eq!(names(&linked, Outcome::Cache).len(), entry_count(&linked));
    assert_eq!(*fresh.memory[0], *first.memory[0]);
    assert_eq!(*fresh.memory[0], *linked.memory[0]);

    // A file cook from the cache, as after an interrupted run, writes the same bytes.
    let file = d.join("out.orr");
    let resumed = orr_cook::cook(
        &manifest(&d, Target::File(file.clone()), Quality::Fast),
        &with_cache,
    )
    .unwrap();
    assert!(names(&resumed, Outcome::Cooked).is_empty());
    assert_eq!(*fresh.memory[0], *std::fs::read(&file).unwrap());

    let evict = CookOptions {
        cache_cap: Some(0),
        ..with_cache
    };
    orr_cook::cook(&m, &evict).unwrap();
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 0);
}

#[test]
fn rebuilds_cook_only_what_changed() {
    let d = dir("incremental-rebuild");
    write_scene(&d);
    let file = d.join("out.orr");
    let cook = |q: Quality| {
        orr_cook::cook(&manifest(&d, Target::File(file.clone()), q), &opts(0)).unwrap()
    };
    let fresh = |q: Quality| {
        orr_cook::cook(&manifest(&d, Target::Memory, q), &opts(0))
            .unwrap()
            .memory
            .remove(0)
    };

    let first = cook(Quality::Fast);
    assert_eq!(names(&first, Outcome::Cooked).len(), entry_count(&first));
    let bytes = std::fs::read(&file).unwrap();

    // Nothing changed: every entry matches and the archive is left alone.
    let again = cook(Quality::Fast);
    assert!(!again.report.archives[0].written);
    assert_eq!(names(&again, Outcome::Archive).len(), entry_count(&again));
    assert_eq!(bytes, std::fs::read(&file).unwrap());

    // New times, same bytes.
    let past = SystemTime::UNIX_EPOCH + Duration::from_secs(1_500_000_000);
    for f in [
        "quad.gltf",
        "quad.bin",
        "a.png",
        "b.png",
        "c.png",
        "tex.png",
    ] {
        set_mtime(&d.join(f), past);
    }
    let touched = cook(Quality::Fast);
    assert!(!touched.report.archives[0].written);
    assert!(names(&touched, Outcome::Cooked).is_empty());

    // One image of material 1: only its texture cooks.
    write_png(&d.join("c.png"), 64, 9);
    let image = cook(Quality::Fast);
    assert!(image.report.archives[0].written);
    assert_eq!(
        names(&image, Outcome::Cooked),
        set(&["quad/texture/1/base_color"])
    );
    assert_eq!(*fresh(Quality::Fast), *std::fs::read(&file).unwrap());

    // Settings of one texture source.
    let settings = cook(Quality::Release);
    assert_eq!(names(&settings, Outcome::Cooked), set(&["tex"]));
    assert_eq!(*fresh(Quality::Release), *std::fs::read(&file).unwrap());

    // Geometry: the mesh, then what depends on it: the model and the instance pages.
    write_bin(&d.join("quad.bin"), 2.0);
    let geometry = cook(Quality::Release);
    let cooked = names(&geometry, Outcome::Cooked);
    let pages: BTreeSet<String> = cooked
        .iter()
        .filter(|n| n.starts_with("scatter/"))
        .cloned()
        .collect();
    assert!(!pages.is_empty());
    let mut expected = set(&["quad", "quad/mesh"]);
    expected.extend(pages);
    assert_eq!(cooked, expected);
    let all_pages = geometry.report.assets[3].entries.len();
    assert_eq!(cooked.len(), 2 + all_pages);
    assert_eq!(*fresh(Quality::Release), *std::fs::read(&file).unwrap());
}

#[test]
fn recook_replaces_an_archive_a_reader_holds() {
    let d = dir("incremental-reader");
    write_scene(&d);
    let file = d.join("out.orr");
    let m = manifest(&d, Target::File(file.clone()), Quality::Fast);
    orr_cook::cook(&m, &opts(0)).unwrap();
    let old_bytes = std::fs::read(&file).unwrap();
    // std opens files with delete sharing.
    let reader = Archive::open(&file).unwrap();

    write_png(&d.join("a.png"), 64, 7);
    orr_cook::cook(&m, &opts(0)).unwrap();
    let new_bytes = std::fs::read(&file).unwrap();
    assert_ne!(old_bytes, new_bytes);

    let old = Archive::from_memory(old_bytes.into()).unwrap();
    let id = AssetId::from_name("quad/texture/0/base_color");
    let read_all = |a: &Archive| {
        let e = a.find(id).unwrap();
        a.blobs(e)
            .iter()
            .map(|b| {
                let mut v = vec![0; b.raw_len as usize];
                a.read(b, &mut v).unwrap();
                v
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(read_all(&reader), read_all(&old));
    for e in reader.entries() {
        reader.verify(e).unwrap();
    }
    let new = Archive::open(&file).unwrap();
    assert_ne!(read_all(&new), read_all(&old));
}

#[test]
fn manifests_load_from_toml() {
    let d = dir("incremental-toml");
    write_scene(&d);
    let path = d.join("cook.toml");
    std::fs::write(
        &path,
        r#"
[[archive]]
output = "out.orr"
cell_log2 = 7

[[archive.asset]]
name = "quad"
gltf = { path = "quad.gltf", settings = { textures = { quality = "fast" }, mesh = { position_step_log2 = -10, lod = { discrete = { errors = [0.01, 0.1] } }, merge_level = 2 }, models = [{ node = "quad" }], joined = ["quad"], parts_from_attribute = { name = "_ID", parts = 4, by_section = true }, mesh_overrides = [["quad", { max_vertices = 64 }]], material_overrides = [["", { surface = 2, subsurface_radius = 0.01 }]] } }

[[archive.asset]]
name = "tex"
texture = { path = "tex.png", settings = { usage = { data = "rgba8" }, filter = "box", wrap = ["clamp", "mirror"] } }

[[archive.asset]]
name = "sky"
texture = { path = "tex.png", settings = { usage = { environment = { max_lights = 2 } } } }

[[archive.asset]]
name = "data"
user = { class = 32771, schema = 2, version = 1, bytes = [1, 2, 3] }

[[archive.asset]]
name = "scatter"
[archive.asset.instances]
prototypes = [{ model = "quad", sections = [{ hidden = true }] }]
records = [{ position = [1.0, 2.0, 3.0], rotation = [0.0, 0.0, 0.0, 1.0], scale = [1.0, 1.0, 1.0], prototype = 0 }]

[[archive.asset]]
name = "paint"
material = { base_color = [0.5, 0.25, 0.125], surface = 4 }
"#,
    )
    .unwrap();
    let m = Manifest::load(&path).unwrap();
    let a = &m.archives[0];
    assert!(matches!(&a.output, Target::File(p) if *p == d.join("out.orr")));
    assert_eq!(a.cell_log2, 7);
    let Source::Gltf { path: p, settings } = &a.assets[0].source else {
        panic!("gltf");
    };
    assert_eq!(*p, d.join("quad.gltf"));
    assert_eq!(settings.textures.quality, Quality::Fast);
    assert_eq!(settings.mesh.position_step_log2, -10);
    assert_eq!(settings.mesh.max_triangles, 128);
    assert_eq!(
        settings.mesh.lod,
        LodMode::Discrete {
            errors: vec![0.01, 0.1]
        }
    );
    assert_eq!(settings.mesh.merge_level, Some(2));
    assert_eq!(settings.models[0].node, "quad");
    assert_eq!(settings.joined, ["quad"]);
    let parts = settings.parts_from_attribute.as_ref().unwrap();
    assert_eq!(
        (parts.name.as_str(), parts.parts, parts.by_section),
        ("_ID", 4, true)
    );
    assert_eq!(settings.mesh_overrides[0].1.max_vertices, Some(64));
    let patch = &settings.material_overrides[0].1;
    assert_eq!(
        (patch.surface, patch.subsurface_radius),
        (Some(2), Some(0.01))
    );
    let Source::Texture { settings, .. } = &a.assets[1].source else {
        panic!("texture");
    };
    assert_eq!(settings.usage, Usage::Data(DataFormat::Rgba8));
    assert_eq!(settings.filter, MipFilter::Box);
    let Source::Texture { settings, .. } = &a.assets[2].source else {
        panic!("texture");
    };
    assert_eq!(
        settings.usage,
        Usage::Environment {
            light_threshold: None,
            max_lights: 2
        }
    );
    assert!(
        matches!(&a.assets[3].source, Source::User { class: 32771, bytes, .. } if *bytes == [1, 2, 3])
    );
    let Source::Instances(i) = &a.assets[4].source else {
        panic!("instances");
    };
    assert_eq!(i.records[0].position, DVec3::new(1.0, 2.0, 3.0));
    assert!(i.prototypes[0].sections[0].hidden);
    let Source::Material(paint) = &a.assets[5].source else {
        panic!("material");
    };
    assert_eq!(
        (paint.base_color, paint.surface),
        (Some([0.5, 0.25, 0.125]), Some(4))
    );

    std::fs::write(&path, "[[archive]]\noutput = \"x\"\nbogus = 1\n").unwrap();
    assert!(matches!(
        Manifest::load(&path),
        Err(orr_cook::Error::Manifest { .. })
    ));
}

#[test]
fn accessors_past_their_views_fail_to_import() {
    let d = dir("incremental-accessor");
    write_scene(&d);
    let gltf = d.join("quad.gltf");
    let json = std::fs::read_to_string(&gltf).unwrap();
    let bad = json.replace(
        r#""count":6,"type":"SCALAR""#,
        r#""count":60,"type":"SCALAR""#,
    );
    assert_ne!(json, bad);
    std::fs::write(&gltf, bad).unwrap();
    let err = orr_cook::cook(&manifest(&d, Target::Memory, Quality::Fast), &opts(1)).unwrap_err();
    assert!(
        matches!(&err, orr_cook::Error::Import { message, .. } if message.contains("runs past")),
        "{err}"
    );
}
