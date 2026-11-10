//! Cooked bytes per class against recorded hashes. A change to a class's output fails here
//! until the class's cook version (`src/key.rs`) is bumped and its row updated, so cooks
//! with old keys never link stale entries.
//!
//! The fixtures avoid the BCn encoders and transcendental math, whose results can differ
//! between CPUs and platforms. The C++ libraries compile without fused multiply-adds
//! (`.cargo/config.toml`), which round differently.

mod common;
#[path = "common/skinned.rs"]
mod common_skinned;

use orr_cook::{
    ArchiveSpec, AssetSpec, DataFormat, GltfSettings, GridInput, GridVoxels, Instances, Linear,
    LodMode, Manifest, MeshSettings, MipFilter, Placement, PrototypeSpec, RasterInput,
    RasterLayerSpec, RasterSet, Reduction, Source, TextureSettings, Usage, VolumeGrids,
    VolumeInput,
};
use orr_math::{DVec3, Quat, Vec3};
use orr_pack::{Archive, Class, Target};

const USER: u16 = Class::USER_FIRST + 1;

/// (class, cook version, blake3 over the content hashes of its entries in name order).
const GOLDEN: &[(u16, u32, &str)] = &[
    (
        Class::Mesh as u16,
        18,
        "93bc2dcb4bcf2af6f174b2475473447fd21e94d194efe1fe92a29b77832c1686",
    ),
    (
        Class::Material as u16,
        11,
        "331006336e726360d430c6f64f05fb32191ceba5a496942261396a18cdd414e8",
    ),
    (
        Class::Model as u16,
        3,
        "59bb63af5e4d5e7730cdb0f2fe30eb8494af01c027cc804883efc9598be430f9",
    ),
    (
        Class::Texture as u16,
        7,
        "d83172baa2c2d7e66fa9860f32a19a099ffbea2bdfec9eb8a788b2b4ea7abf93",
    ),
    (
        Class::InstancePage as u16,
        6,
        "40ea40760c7833b1f7409dcb66c44c27c0c4caf9b902ce57e94271daef5e948e",
    ),
    (
        Class::RasterSet as u16,
        2,
        "552868a7a913b6a2812e5d834cbba45fb4613839e85d744cfdc358ab53dba9be",
    ),
    (
        Class::Skeleton as u16,
        2,
        "fc0354c88262c2490c85a666dfce169baa14192124373bdad1c32f493363428f",
    ),
    (
        Class::Clip as u16,
        2,
        "6ad26f11b39caeead765da2d904ec647c92edb6fac170a1e80dcdfdc8ea601f8",
    ),
    (
        Class::Volume as u16,
        1,
        "2718b024a9f6568a23c9ca2b26bfd551d0951735505dc4b0427e414865989af4",
    ),
    (
        USER,
        1,
        "4ba4fdc45927952e6db847bca8db82d192871fc2f541209f7c4f776ada77d9e4",
    ),
];

/// A 32 x 32 quad height grid in two sections, 4 x 4 quad blocks of each in a checkerboard, so
/// a DAG of it has clusters of each section and clusters that mix them.
fn sections_grid_glb() -> std::path::PathBuf {
    const N: u32 = 32;
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden-grid.glb");
    let mut bin = Vec::new();
    for y in 0..=N {
        for x in 0..=N {
            let z = ((x * x + 3 * y) % 7) as f32 * 0.03125;
            bin.extend_from_slice(bytemuck::cast_slice(&[x as f32 * 0.25, z, y as f32 * 0.25]));
        }
    }
    let mut sections = [Vec::new(), Vec::new()];
    for y in 0..N {
        for x in 0..N {
            let v = y * (N + 1) + x;
            let s = &mut sections[((x / 4 + y / 4) % 2) as usize];
            s.extend_from_slice(&[v, v + N + 1, v + 1, v + 1, v + N + 1, v + N + 2]);
        }
    }
    let positions = bin.len();
    for s in &sections {
        bin.extend_from_slice(bytemuck::cast_slice(s));
    }
    let count = sections[0].len();
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}},{{"attributes":{{"POSITION":0}},"indices":2,"material":1}}]}}],
"materials":[{{"name":"a"}},{{"name":"b"}}],
"buffers":[{{"byteLength":{len}}}],
"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{positions}}},{{"buffer":0,"byteOffset":{positions},"byteLength":{indices}}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":{verts},"type":"VEC3","min":[0,0,0],"max":[{max},{zmax},{max}]}},
 {{"bufferView":1,"byteOffset":0,"componentType":5125,"count":{count},"type":"SCALAR"}},
 {{"bufferView":1,"byteOffset":{half},"componentType":5125,"count":{count},"type":"SCALAR"}}]}}"#,
        len = bin.len(),
        indices = bin.len() - positions,
        verts = (N + 1) * (N + 1),
        max = N as f32 * 0.25,
        zmax = 6.0 * 0.03125,
        half = count * 4,
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    std::fs::write(&path, glb).unwrap();
    path
}

#[test]
fn cooked_bytes_match_their_cook_versions() {
    let png = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden.png");
    image::RgbaImage::from_fn(40, 24, |x, y| {
        image::Rgba([(x * 6) as u8, (y * 10) as u8, (x ^ y) as u8, (x * y) as u8])
    })
    .save(&png)
    .unwrap();
    let raster = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden.r32");
    let heights: Vec<f32> = (0..32 * 32).map(|i| ((i * 7) % 61) as f32 - 20.0).collect();
    std::fs::write(&raster, bytemuck::cast_slice(&heights)).unwrap();
    std::fs::write(
        raster.with_extension("json"),
        r#"{"width": 32, "height": 32, "format": "r32f"}"#,
    )
    .unwrap();
    let records = (0..40)
        .map(|i| Placement {
            position: DVec3::new(f64::from(i) * 37.25 - 600.0, 1.5, f64::from(i % 7) * 90.0)
                .to_array(),
            linear: Linear::Scaled {
                rotation: (if i % 2 == 0 {
                    Quat::IDENTITY
                } else {
                    Quat::from_xyzw(0.0, 0.6, 0.0, 0.8)
                })
                .to_array(),
                scale: Vec3::ONE.to_array(),
            },
            prototype: 0,
        })
        .collect();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: "block".into(),
                    source: Source::Gltf {
                        path: common::block_glb(),
                        settings: GltfSettings::default(),
                    },
                },
                AssetSpec {
                    name: "block-dag".into(),
                    source: Source::Gltf {
                        path: common::block_glb(),
                        settings: GltfSettings {
                            mesh: MeshSettings {
                                lod: LodMode::Dag,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    },
                },
                AssetSpec {
                    name: "grid-dag".into(),
                    source: Source::Gltf {
                        path: sections_grid_glb(),
                        settings: GltfSettings {
                            mesh: MeshSettings {
                                lod: LodMode::Dag,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    },
                },
                AssetSpec {
                    name: "rig".into(),
                    source: Source::Gltf {
                        path: common_skinned::skinned_glb("golden-skinned", false),
                        settings: GltfSettings::default(),
                    },
                },
                AssetSpec {
                    name: "data-texture".into(),
                    source: Source::Texture {
                        path: png,
                        settings: TextureSettings {
                            usage: Usage::Data(DataFormat::Rgba8),
                            filter: MipFilter::Box,
                            ..Default::default()
                        },
                    },
                },
                AssetSpec {
                    name: "scatter".into(),
                    source: Source::Instances(Instances {
                        records,
                        prototypes: vec![PrototypeSpec {
                            model: "block".into(),
                            flags: 3,
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                },
                AssetSpec {
                    name: "terrain".into(),
                    source: Source::Raster(RasterSet {
                        world_min: [-16.0, -16.0],
                        world_size: 32.0,
                        layers: vec![RasterLayerSpec {
                            name: "height".into(),
                            input: RasterInput::Whole {
                                path: raster,
                                tile_size: 8,
                            },
                            border: 1,
                            reduction: Reduction::Mean,
                            srgb: false,
                            compress: None,
                        }],
                    }),
                },
                AssetSpec {
                    name: "smoke".into(),
                    source: Source::Volume {
                        input: VolumeInput::Grids(VolumeGrids {
                            origin: [-1.0, 0.0, 2.0],
                            voxel_size: [0.25; 3],
                            grids: ["density", "heat"]
                                .into_iter()
                                .enumerate()
                                .map(|(g, name)| GridInput {
                                    name: name.into(),
                                    background: g as f32,
                                    voxels: GridVoxels::Dense {
                                        min: [-3, 2, 5],
                                        size: [12, 9, 10],
                                        values: (0..12 * 9 * 10)
                                            .map(|i| ((i * (7 + g)) % 23) as f32 * 0.125)
                                            .collect(),
                                    },
                                })
                                .collect(),
                        }),
                        settings: Default::default(),
                    },
                },
                AssetSpec {
                    name: "user".into(),
                    source: Source::User {
                        class: USER,
                        schema: 1,
                        version: 1,
                        bytes: b"golden".to_vec(),
                    },
                },
            ],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let a = Archive::from_memory(bytes).unwrap();
    let mut failures = Vec::new();
    for &(class, version, expected) in GOLDEN {
        let mut entries: Vec<_> = a.entries().iter().filter(|e| e.class == class).collect();
        assert!(!entries.is_empty(), "no entry of class {class:#06x}");
        entries.sort_by_key(|e| a.name(e));
        let mut h = blake3::Hasher::new();
        for e in entries {
            h.update(&e.content_hash.0);
        }
        let hash = h.finalize().to_hex().to_string();
        let current = orr_cook::cook_version(class);
        if current != version {
            failures.push(format!(
                "class {class:#06x}: cook version {current}, table {version}: update the row to ({version} -> {current}, \"{hash}\")"
            ));
        } else if hash != expected {
            failures.push(format!(
                "class {class:#06x}: output changed ({hash}); bump its cook version in src/key.rs and update the row"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// (cook version, hash) of the subdivision cage class, as the rows above.
#[cfg(feature = "subd")]
const SUBD_GOLDEN: (u32, &str) = (
    4,
    "e52d04e32fe4cc7eee855e185c21728678cb01321a4013155088c7abe3087988",
);

#[cfg(feature = "subd")]
#[test]
fn cooked_cages_match_their_cook_version() {
    use orr_cook::{CageInput, Crease};
    let positions = (0..8)
        .map(|i| {
            let b = |k: u32| if i & (1 << k) != 0 { 1.0 } else { -1.0 };
            Vec3::new(b(0), b(1), b(2))
        })
        .collect();
    let cage = CageInput {
        positions,
        face_counts: vec![4; 6],
        face_indices: vec![
            0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5,
        ],
        creases: vec![Crease {
            edge: [0, 1],
            sharpness: 1.5,
        }],
        ..Default::default()
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "cage".into(),
                source: Source::Subd {
                    cage,
                    materials: vec![Default::default()],
                    settings: Default::default(),
                },
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let a = Archive::from_memory(bytes).unwrap();
    let class = Class::Subd as u16;
    let e = a.entries().iter().find(|e| e.class == class).unwrap();
    let hash = blake3::hash(&e.content_hash.0).to_hex().to_string();
    let current = orr_cook::cook_version(class);
    assert_eq!(
        (current, hash.as_str()),
        SUBD_GOLDEN,
        "bump the subd cook version in src/key.rs when its output changes, then update the row"
    );
}
