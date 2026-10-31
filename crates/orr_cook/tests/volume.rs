//! Volumes: cooked bricks, voxels and trilinear samples against the input within one
//! quantization step, the brick store rule and the majorant bound, on grids built in memory
//! and on OpenVDB's smoke1 sample.
#![allow(clippy::print_stdout)]

use std::collections::HashMap;
use std::path::PathBuf;

use orr_cook::{
    ArchiveSpec, AssetSpec, Block, GridInput, GridVoxels, Manifest, Source, VolumeFormat,
    VolumeGridSpec, VolumeGrids, VolumeInput, VolumeSettings,
};
use orr_pack::volume::{BRICK, BRICK_TEXEL_COUNT, BRICK_TEXELS, GridSampler, VolumeInfo};
use orr_pack::{Archive, AssetId, Class, Target};

/// A volume entry read back: blob 0, the majorants and each grid's decoded pages.
struct Cooked {
    info: VolumeInfo,
    majorants: Vec<u8>,
    pages: Vec<Vec<Vec<u8>>>,
}

fn read_volume(a: &Archive, name: &str) -> Cooked {
    let e = a
        .expect(AssetId::from_name(name), Class::Volume as u16, 1)
        .unwrap();
    let blobs = a.blobs(e);
    let read = |i: usize| {
        let mut out = vec![0; blobs[i].raw_len as usize];
        a.read(&blobs[i], &mut out).unwrap();
        out
    };
    let info = VolumeInfo::parse(&read(0)).unwrap();
    let pages = info
        .grids
        .iter()
        .map(|g| {
            (g.pages.first..g.pages.first + g.pages.count)
                .map(|b| read(b as usize))
                .collect()
        })
        .collect();
    Cooked {
        majorants: read(1),
        info,
        pages,
    }
}

/// Small xorshift for repeatable sample points.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Checks grid `gi` of `c` against `reference` (the input value of a voxel).
fn check_grid(c: &Cooked, gi: usize, reference: &dyn Fn([i32; 3]) -> f32) {
    let info = &c.info;
    let g = &info.grids[gi];
    let name = &info.names[gi];
    let step = g.step() * 1.0001;
    let h = info.header;
    let s = GridSampler::new(info, gi, &c.pages[gi]).unwrap();
    let n = BRICK_TEXELS as i32;
    let voxel_of = |b: [u32; 3], t: usize| {
        let t = [t as i32 % n, t as i32 / n % n, t as i32 / (n * n)];
        [0, 1, 2].map(|a| h.brick_min[a] + BRICK as i32 * b[a] as i32 + t[a] - 1)
    };

    // A brick is stored exactly when a texel differs from the background; every stored
    // texel matches its voxel.
    let majorants = info.majorants(&c.majorants, gi).unwrap();
    let mut stored = 0;
    for bz in 0..h.bricks[2] {
        for by in 0..h.bricks[1] {
            for bx in 0..h.bricks[0] {
                let b = [bx, by, bz];
                let needed =
                    (0..BRICK_TEXEL_COUNT).any(|t| reference(voxel_of(b, t)) != g.background);
                let slot = info.brick(gi, b.map(|x| x as u16));
                assert_eq!(slot.is_some(), needed, "{name}: brick {b:?}");
                let m = majorants[info.majorant_index(b)];
                let Some(slot) = slot else {
                    assert!(m >= g.background, "{name}: majorant of absent brick {b:?}");
                    continue;
                };
                stored += 1;
                let per_page = g.bricks_per_page();
                let bytes = &c.pages[gi][slot / per_page][slot % per_page * g.brick_bytes()..]
                    [..g.brick_bytes()];
                for t in 0..BRICK_TEXEL_COUNT {
                    let v = g.decode(g.texel(bytes, t));
                    let want = reference(voxel_of(b, t));
                    assert!(
                        (v - want).abs() <= step,
                        "{name}: brick {b:?} texel {t}: {v} vs {want}"
                    );
                    assert!(m >= v, "{name}: majorant {m} under {v} in brick {b:?}");
                }
            }
        }
    }
    assert_eq!(stored, g.bricks.count as usize, "{name}: stored bricks");

    // Every voxel over the brick grid and a margin.
    let margin = 3;
    for z in h.brick_min[2] - margin..h.brick_min[2] + (h.bricks[2] * BRICK) as i32 + margin {
        for y in h.brick_min[1] - margin..h.brick_min[1] + (h.bricks[1] * BRICK) as i32 + margin {
            for x in h.brick_min[0] - margin..h.brick_min[0] + (h.bricks[0] * BRICK) as i32 + margin
            {
                let (v, want) = (s.voxel([x, y, z]), reference([x, y, z]));
                assert!(
                    (v - want).abs() <= step,
                    "{name}: voxel {:?}: {v} vs {want}",
                    [x, y, z]
                );
            }
        }
    }

    // Trilinear samples over the bounds and a margin, all eight taps from one brick.
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ gi as u64);
    let b = h.bounds;
    for _ in 0..20_000 {
        let p = [0, 1, 2].map(|a| {
            let pad = 2.0 * h.voxel_size[a];
            b.min[a] - pad + rng.unit() * (b.max[a] - b.min[a] + 2.0 * pad)
        });
        let u = info.to_index(p);
        let f = u.map(f32::floor);
        let mut want = 0.0;
        for corner in 0..8 {
            let d = [corner & 1, (corner >> 1) & 1, corner >> 2];
            let w: f32 = (0..3)
                .map(|a| {
                    if d[a] == 1 {
                        u[a] - f[a]
                    } else {
                        1.0 - (u[a] - f[a])
                    }
                })
                .product();
            want += w * reference([0, 1, 2].map(|a| f[a] as i32 + d[a]));
        }
        let v = s.sample(p);
        assert!(
            (v - want).abs() <= step * 1.001 + 1e-6 * want.abs(),
            "{name}: sample {p:?}: {v} vs {want}"
        );
    }
}

fn density(v: [i32; 3]) -> f32 {
    let r = [v[0] as f32 - 3.5, v[1] as f32 + 6.0, v[2] as f32 - 1.0];
    let d = (r[0] * r[0] + r[1] * r[1] * 0.5 + r[2] * r[2]).sqrt();
    (1.0 - d / 14.0).max(0.0) * 3.0
}

/// Blocks of `density` around the origin, some left out, so neighbors of given blocks and
/// gaps inside the volume both occur.
fn density_blocks() -> Vec<([i32; 3], Box<Block>)> {
    let mut out = Vec::new();
    for z in -2i32..=1 {
        for y in -3..=1 {
            for x in -2..=2 {
                if (x + 2 * y + 3 * z).rem_euclid(5) == 0 {
                    continue;
                }
                let mut b = Box::new([0.0; 512]);
                for (i, v) in b.iter_mut().enumerate() {
                    *v = density([
                        x * 8 + (i % 8) as i32,
                        y * 8 + (i / 8 % 8) as i32,
                        z * 8 + (i / 64) as i32,
                    ]);
                }
                out.push(([x, y, z], b));
            }
        }
    }
    out
}

fn density_reference(blocks: &[([i32; 3], Box<Block>)]) -> impl Fn([i32; 3]) -> f32 + '_ {
    let by: HashMap<[i32; 3], &Block> = blocks.iter().map(|(c, b)| (*c, &**b)).collect();
    move |v| match by.get(&v.map(|x| x.div_euclid(8))) {
        Some(b) => {
            let l = v.map(|x| x.rem_euclid(8) as usize);
            b[l[0] + 8 * l[1] + 64 * l[2]]
        }
        None => 0.0,
    }
}

const T_MIN: [i32; 3] = [-5, 3, -9];
const T_SIZE: [u32; 3] = [21, 13, 17];
const T_BACKGROUND: f32 = 290.0;

fn temperature(v: [i32; 3]) -> f32 {
    let d = [0, 1, 2].map(|a| v[a] - T_MIN[a]);
    if (0..3).all(|a| d[a] >= 0 && d[a] < T_SIZE[a] as i32) {
        300.0 + (d[0] * 7 + d[1] * 13 + d[2] * 3) as f32 * 0.37 - 20.0
    } else {
        T_BACKGROUND
    }
}

#[test]
fn memory_grids_match_their_input() {
    let blocks = density_blocks();
    let n = T_SIZE.map(|s| s as i32);
    let mut values = Vec::new();
    for z in 0..n[2] {
        for y in 0..n[1] {
            for x in 0..n[0] {
                values.push(temperature([T_MIN[0] + x, T_MIN[1] + y, T_MIN[2] + z]));
            }
        }
    }
    let grids = VolumeGrids {
        origin: [1.5, -2.0, 0.25],
        voxel_size: [0.5, 0.25, 0.5],
        grids: vec![
            GridInput {
                name: "temperature".into(),
                background: T_BACKGROUND,
                voxels: GridVoxels::Dense {
                    min: T_MIN,
                    size: T_SIZE,
                    values,
                },
            },
            GridInput {
                name: "density".into(),
                background: 0.0,
                voxels: GridVoxels::Blocks(blocks.clone()),
            },
        ],
    };
    for format in [VolumeFormat::R8, VolumeFormat::R16f] {
        let settings = VolumeSettings {
            grids: vec![
                VolumeGridSpec {
                    name: "density".into(),
                    format: Some(format),
                },
                VolumeGridSpec {
                    name: "temperature".into(),
                    format: None,
                },
            ],
        };
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::Memory,
                vec![AssetSpec {
                    name: "cloud".into(),
                    source: Source::Volume {
                        input: VolumeInput::Grids(grids.clone()),
                        settings,
                    },
                }],
            )],
        };
        let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
        let report = cooked.report.assets[0].entries[0].volume.clone().unwrap();
        let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
        let c = read_volume(&a, "cloud");
        assert_eq!(c.info.names, ["density", "temperature"]);
        assert_eq!(report.grids[1].vk_format, orr_pack::texture::vk::R16_SFLOAT);
        check_grid(&c, 0, &density_reference(&blocks));
        check_grid(&c, 1, &temperature);
    }
}

/// OpenVDB's smoke1 sample: `ORR_VDB_SMOKE`, else `volumes/smoke.vdb` under `ORR_TOOLS`,
/// else under the nearest `.tools`.
fn smoke_path() -> PathBuf {
    if let Some(p) = std::env::var_os("ORR_VDB_SMOKE").filter(|p| !p.is_empty()) {
        return p.into();
    }
    let rel = "volumes/smoke.vdb";
    if let Some(t) = std::env::var_os("ORR_TOOLS").filter(|p| !p.is_empty()) {
        return PathBuf::from(t).join(rel);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|d| d.join(".tools").join(rel))
        .find(|p| p.exists())
        .expect("smoke.vdb not fetched: ORR_VALIDATION_DATA=1 tools/fetch-slang.sh")
}

#[test]
#[ignore = "needs OpenVDB's smoke1 sample: ORR_VALIDATION_DATA=1 tools/fetch-slang.sh; run with --release"]
fn smoke_vdb_matches_vdb_rs() {
    let path = smoke_path();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("volume-smoke");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let toml = format!(
        "[[archive]]\noutput = \"smoke.orr\"\n\n[[archive.asset]]\nname = \"smoke\"\nvolume = {{ input = {{ vdb = {{ path = {:?} }} }} }}\n",
        path.to_str().unwrap()
    );
    std::fs::write(dir.join("smoke.toml"), toml).unwrap();
    let manifest = Manifest::load(&dir.join("smoke.toml")).unwrap();
    let cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let report = cooked.report.assets[0].entries[0].volume.clone().unwrap();
    let a = Archive::open(&dir.join("smoke.orr")).unwrap();
    let c = read_volume(&a, "smoke");

    let mut r =
        vdb_rs::VdbReader::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()))
            .unwrap();
    let grid = r.read_grid::<f32>("density").unwrap();
    let mut voxels = HashMap::new();
    for (p, v, level) in grid.iter() {
        let s = level.scale() as i32;
        let o = [p.x as i32, p.y as i32, p.z as i32];
        for z in 0..s {
            for y in 0..s {
                for x in 0..s {
                    voxels.insert([o[0] + x, o[1] + y, o[2] + z], v);
                }
            }
        }
    }
    println!("{} active voxels; {report:?}", voxels.len());
    println!(
        "archive {} B",
        std::fs::metadata(dir.join("smoke.orr")).unwrap().len()
    );
    check_grid(&c, 0, &|v| voxels.get(&v).copied().unwrap_or(0.0));
}
