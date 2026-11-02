//! `orr_pack.volume` on a resident volume laid out by `VolumeInfo::resident_layout` against
//! `GridSampler`: voxels and trilinear samples within float rounding, majorants at least
//! every sample, for an R8 and an R16F grid, and the majorant cells a ray walks through a
//! sparse volume at least every sample along it.

use std::collections::HashMap;

use orr_cook::{
    ArchiveSpec, AssetSpec, Block, GridInput, GridVoxels, Manifest, Source, VolumeFormat,
    VolumeGridSpec, VolumeGrids, VolumeInput, VolumeSettings,
};
use orr_gpu::{Allocation, GpuAddress, GpuPtr, MemoryKind};
use orr_pack::volume::{GridSampler, VolumeGpu, VolumeInfo};
use orr_pack::{Archive, AssetId, Class};
use orr_scene::testing::{assert_clean, device};
use pack_suite::shaders::volume_sample::{MODULE, Root};
use pack_suite::{dispatch, dispatch_entry, read};

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A puff of density in 8^3 blocks with gaps between them, and a dense temperature ramp on
/// a non-zero background.
fn grids() -> VolumeGrids {
    let mut blocks: HashMap<[i32; 3], Box<Block>> = HashMap::new();
    for bz in -1..2 {
        for by in 0..2 {
            for bx in -2..2 {
                if (bx + by + bz) % 3 == 0 {
                    continue;
                }
                let mut b = Box::new([0.0f32; 512]);
                for (i, v) in b.iter_mut().enumerate() {
                    let p = [i % 8, i / 8 % 8, i / 64].map(|c| c as f32);
                    *v = ((p[0] * 0.7 + p[1] * 1.3 + p[2] * 0.4 + (bx * 5 + bz) as f32).sin()
                        * 0.5
                        + 0.5)
                        * 3.0;
                }
                blocks.insert([bx, by, bz], b);
            }
        }
    }
    let size = [19u32, 11, 14];
    let values = (0..size.iter().product::<u32>())
        .map(|i| 280.0 + (i % 19) as f32 * 1.5 + (i / 209) as f32 * 0.75)
        .collect();
    VolumeGrids {
        origin: [-1.0, 0.5, 2.0],
        voxel_size: [0.25, 0.5, 0.25],
        grids: vec![
            GridInput {
                name: "density".into(),
                background: 0.0,
                voxels: GridVoxels::Blocks(blocks.into_iter().collect()),
            },
            GridInput {
                name: "temperature".into(),
                background: 275.0,
                voxels: GridVoxels::Dense {
                    min: [-6, 2, -4],
                    size,
                    values,
                },
            },
        ],
    }
}

/// Bricks of density far apart: majorant cells of 4^3 bricks, many of them empty.
fn sparse() -> VolumeGrids {
    let mut blocks = Vec::new();
    let at = [
        [-6, 0, -4],
        [-5, 0, -4],
        [3, 5, 2],
        [4, 5, 2],
        [-1, 2, -1],
        [9, 1, 6],
    ];
    for (k, b) in at.into_iter().enumerate() {
        let mut v = Box::new([0.0f32; 512]);
        for (i, x) in v.iter_mut().enumerate() {
            *x = 1.5 + k as f32 + ((i % 8) as f32 * 0.9 + (i / 64) as f32 * 0.4).sin();
        }
        blocks.push((b, v));
    }
    VolumeGrids {
        origin: [0.5, -1.0, 0.25],
        voxel_size: [0.25, 0.25, 0.5],
        grids: vec![GridInput {
            name: "density".into(),
            background: 0.0,
            voxels: GridVoxels::Blocks(blocks),
        }],
    }
}

/// A cooked volume: blob 0 parsed, blob 1 and each grid's decoded pages.
struct Cooked {
    info: VolumeInfo,
    majorants: Vec<u8>,
    pages: Vec<Vec<Vec<u8>>>,
}

/// `grids` cooked, a grid named `density` in `density`'s format.
fn cook(grids: VolumeGrids, density: VolumeFormat) -> Cooked {
    let specs = grids
        .grids
        .iter()
        .map(|g| VolumeGridSpec {
            name: g.name.clone(),
            format: (g.name == "density").then_some(density),
        })
        .collect();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            orr_pack::Target::Memory,
            vec![AssetSpec {
                name: "puff".into(),
                source: Source::Volume {
                    input: VolumeInput::Grids(grids),
                    settings: VolumeSettings { grids: specs },
                },
            }],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    let e = a
        .expect(AssetId::from_name("puff"), Class::Volume as u16, 1)
        .unwrap();
    let blobs = a.blobs(e);
    let info = VolumeInfo::parse(&read(&a, &blobs[0])).unwrap();
    let pages = info
        .grids
        .iter()
        .map(|g| {
            (g.pages.first..g.pages.first + g.pages.count)
                .map(|b| read(&a, &blobs[b as usize]))
                .collect()
        })
        .collect();
    Cooked {
        majorants: read(&a, &blobs[1]),
        info,
        pages,
    }
}

/// `c` resident in upload memory, and its record.
fn resident(c: &Cooked) -> (Allocation<u8>, GpuPtr<VolumeGpu>) {
    let layout = c.info.resident_layout();
    let mut memory = device()
        .gpu_malloc::<u8>(MemoryKind::Upload, layout.size, "resident volume")
        .unwrap();
    let base = memory.gpu().address().0;
    let bytes = memory.as_mut_slice().unwrap();
    let head = c.info.resident_head(&layout, base);
    bytes[..head.len()].copy_from_slice(&head);
    let m = layout.majorants as usize;
    bytes[m..m + c.majorants.len()].copy_from_slice(&c.majorants);
    for (g, grid_pages) in c.pages.iter().enumerate() {
        for (p, page) in grid_pages.iter().enumerate() {
            let at = layout.page(&c.info, g, p) as usize;
            bytes[at..at + page.len()].copy_from_slice(page);
        }
    }
    // SAFETY: the record is resident at the start of the allocation.
    let record = unsafe { GpuPtr::from_address(GpuAddress(base)) };
    (memory, record)
}

fn upload<T: bytemuck::Pod>(values: &[T], name: &str) -> Allocation<T> {
    let mut a = device()
        .gpu_malloc::<T>(MemoryKind::Upload, values.len() as u64, name)
        .unwrap();
    a.as_mut_slice().unwrap().copy_from_slice(values);
    a
}

/// A point inside `info`'s bounds grown by `margin` m.
fn point(info: &VolumeInfo, rng: &mut Rng, margin: f32) -> [f32; 4] {
    let b = info.header.bounds;
    let p = [0, 1, 2].map(|a| {
        let (lo, hi) = (b.min[a] - margin, b.max[a] + margin);
        lo + (hi - lo) * rng.unit()
    });
    [p[0], p[1], p[2], 0.0]
}

#[test]
fn resident_volumes_sample_as_the_cpu_reference() {
    for format in [VolumeFormat::R8, VolumeFormat::R16f] {
        let c = cook(grids(), format);
        let (_memory, volume) = resident(&c);
        let info = &c.info;
        let h = info.header;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let n = 20_000usize;
        let points: Vec<[f32; 4]> = (0..n).map(|_| point(info, &mut rng, 0.6)).collect();
        let voxels: Vec<[i32; 4]> = (0..n)
            .map(|_| {
                let v = [0, 1, 2].map(|a| {
                    let span = (h.bricks[a] * 8 + 4) as f32;
                    h.brick_min[a] - 2 + (span * rng.unit()) as i32
                });
                [v[0], v[1], v[2], 0]
            })
            .collect();
        let pts = upload(&points, "points");
        let vox = upload(&voxels, "voxels");
        for g in 0..info.grids.len() {
            let out = device()
                .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, n as u64, "volume reads")
                .unwrap();
            let root = upload(
                &[Root {
                    volume,
                    points: pts.gpu(),
                    voxels: vox.gpu(),
                    out: out.gpu(),
                    dirs: GpuPtr::NULL,
                    spans: GpuPtr::NULL,
                    count: n as u32,
                    grid: g as u32,
                    _pad: 0,
                }],
                "root",
            );
            dispatch(&MODULE, root.gpu(), n.div_ceil(64) as u32);
            assert_clean();
            let s = GridSampler::new(info, g, &c.pages[g]).unwrap();
            let grid = &info.grids[g];
            let reach = grid.value_range[0].abs().max(grid.value_range[1].abs());
            let tolerance = reach * 2e-5;
            for ((r, p), v) in out.as_slice().unwrap().iter().zip(&points).zip(&voxels) {
                let p = [p[0], p[1], p[2]];
                let want = s.sample(p);
                assert!(
                    (r[0] - want).abs() <= tolerance,
                    "grid {g} {format:?} sample at {p:?}: {} against {want}",
                    r[0]
                );
                assert!(
                    r[1] >= r[0] - tolerance,
                    "grid {g} {format:?} majorant {} under sample {} at {p:?}",
                    r[1],
                    r[0]
                );
                let voxel = s.voxel([v[0], v[1], v[2]]);
                assert!(
                    (r[2] - voxel).abs() <= reach * f32::EPSILON,
                    "grid {g} {format:?} voxel {v:?}: {} against {voxel}",
                    r[2]
                );
            }
        }
    }
}

#[test]
fn majorant_cells_along_a_ray_bound_every_sample() {
    const SAMPLES: usize = 64;
    let c = cook(sparse(), VolumeFormat::R8);
    let (_memory, volume) = resident(&c);
    let info = &c.info;
    let cells = info.header.majorant_cells;
    assert!(cells.iter().product::<u32>() >= 8, "{cells:?} cells");
    let b = info.header.bounds;
    let diagonal = (0..3)
        .map(|a| (b.max[a] - b.min[a]).powi(2))
        .sum::<f32>()
        .sqrt();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let n = 4096usize;
    let points: Vec<[f32; 4]> = (0..n).map(|_| point(info, &mut rng, 1.0)).collect();
    let dirs: Vec<[f32; 4]> = (0..n)
        .map(|_| {
            let z = 2.0 * rng.unit() - 1.0;
            let phi = std::f32::consts::TAU * rng.unit();
            let r = (1.0 - z * z).sqrt();
            [r * phi.cos(), r * phi.sin(), z, diagonal]
        })
        .collect();
    let pts = upload(&points, "points");
    let dir = upload(&dirs, "directions");
    let spans = device()
        .gpu_malloc::<f32>(MemoryKind::Readback, (n * SAMPLES) as u64, "spans")
        .unwrap();
    let root = upload(
        &[Root {
            volume,
            points: pts.gpu(),
            voxels: GpuPtr::NULL,
            out: GpuPtr::NULL,
            dirs: dir.gpu(),
            spans: spans.gpu(),
            count: n as u32,
            grid: 0,
            _pad: 0,
        }],
        "root",
    );
    dispatch_entry(&MODULE, "spans", root.gpu(), n.div_ceil(64) as u32);
    assert_clean();
    let s = GridSampler::new(info, 0, &c.pages[0]).unwrap();
    let mut empty = 0;
    let found = spans.as_slice().unwrap().chunks_exact(SAMPLES);
    for ((p, d), m) in points.iter().zip(&dirs).zip(found) {
        for (k, &m) in m.iter().enumerate() {
            let t = (k as f32 + 0.5) / SAMPLES as f32 * d[3];
            let x = [0, 1, 2].map(|a| p[a] + d[a] * t);
            let v = s.sample(x);
            assert!(
                v <= m + 1e-5,
                "sample {v} over its cell's majorant {m} at {x:?}"
            );
            empty += usize::from(m == 0.0);
        }
    }
    assert!(empty > n * SAMPLES / 4, "{empty} samples in empty cells");
}
