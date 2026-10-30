//! Cooked bytes per class against recorded hashes. A change to a class's output fails here
//! until the class's cook version (`src/key.rs`) is bumped and its row updated, so cooks
//! with old keys never link stale entries.
//!
//! The fixtures avoid the BCn encoders and transcendental maths, whose results can differ
//! between CPUs and platforms.

mod common;

use glam::{DVec3, Quat, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, DataFormat, GltfSettings, Instances, Manifest, MipFilter, Placement,
    PrototypeSpec, Source, TextureSettings, Usage,
};
use orr_pack::{Archive, Class, Target};

const USER: u16 = Class::USER_FIRST + 1;

/// (class, cook version, blake3 over the content hashes of its entries in name order).
const GOLDEN: &[(u16, u32, &str)] = &[
    (
        Class::Mesh as u16,
        1,
        "c4a5bf69abadcfa8e8c64d5bbc8495a72896e86c28619f26cdaf20266cf769fb",
    ),
    (
        Class::Material as u16,
        1,
        "d371e265b6acd6338980e7b2ea5ed66950f8ff6679f4fe7053d9e4ee5b667846",
    ),
    (
        Class::Model as u16,
        1,
        "d8b1f18f6c0f4b480415597df31be497f2f1611be06c4f711bb3db3b8b0ea56a",
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
