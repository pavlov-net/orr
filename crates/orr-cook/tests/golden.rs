//! Cooked bytes per class against recorded hashes. A change to a class's output fails here
//! until the class's cook version (`src/key.rs`) is bumped and its row updated, so cooks
//! with old keys never link stale entries.
//!
//! The fixtures avoid the BCn encoders and transcendental maths, whose results can differ
//! between CPUs and platforms.

mod common;
#[path = "common/skinned.rs"]
mod common_skinned;

use glam::{DVec3, Quat, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, DataFormat, GltfSettings, Instances, LodMode, Manifest, MeshSettings,
    MipFilter, Placement, PrototypeSpec, RasterInput, RasterLayerSpec, RasterSet, Reduction,
    Source, TextureSettings, Usage,
};
use orr_pack::{Archive, Class, Target};

const USER: u16 = Class::USER_FIRST + 1;

/// (class, cook version, blake3 over the content hashes of its entries in name order).
const GOLDEN: &[(u16, u32, &str)] = &[
    (
        Class::Mesh as u16,
        6,
        "a1562f19e331661eac71746507990b636868211e008b5b74e9713c095f03c871",
    ),
    (
        Class::Material as u16,
        1,
        "b3daf76e851eda198a25e9f55d102fb1f6b63c16beb9ce685ce77dfabf1c2984",
    ),
    (
        Class::Model as u16,
        3,
        "f88a9bd90ab88d29ffab4f1e5b374b686704e3bbee80f0808c8017d103883757",
    ),
    (
        Class::Texture as u16,
        1,
        "43735be6c283598fbb5f7a16384b3f7ac1ad607f834e3de5494294703863242e",
    ),
    (
        Class::InstancePage as u16,
        2,
        "a251a6855f4d9ece691a2b1e0c5b8324970d386350f71a6fdd03c7087c72e194",
    ),
    (
        Class::RasterSet as u16,
        1,
        "552868a7a913b6a2812e5d834cbba45fb4613839e85d744cfdc358ab53dba9be",
    ),
    (
        Class::Skeleton as u16,
        1,
        "04c4a01f5c1b0373b9ea75d8b848be7697eda558e82adb029a9efe011f8aebc1",
    ),
    (
        Class::Clip as u16,
        1,
        "517b3a610d36451309d0b6313cbda0b2339f05b2b247c3bcbfa0fd7880a15c44",
    ),
    (
        USER,
        1,
        "4ba4fdc45927952e6db847bca8db82d192871fc2f541209f7c4f776ada77d9e4",
    ),
];

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
            position: DVec3::new(f64::from(i) * 37.25 - 600.0, 1.5, f64::from(i % 7) * 90.0),
            rotation: if i % 2 == 0 {
                Quat::IDENTITY
            } else {
                Quat::from_xyzw(0.0, 0.6, 0.0, 0.8)
            },
            scale: Vec3::ONE,
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
                            quality: Default::default(),
                        }],
                    }),
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
    1,
    "35fdeae53cf41d76dd2a238f813ed73c4375bd5a11fe007c534d73e5a7fce9da",
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
