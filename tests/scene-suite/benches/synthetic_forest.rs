//! A synthetic forest rendered through `orr::Renderer`: stand-in trees of four species and
//! three variants with rigid parts and discrete levels, grass clumps along the camera path
//! and a ground tile per world cell, cooked in process. It prints the figures that set the
//! PTLAS clamp in rebuilt instances, the partition count and the software-raster threshold,
//! and the render thread's step times while a window presents and a pack streams in and out.
//!
//! `cargo bench -p scene-suite --bench synthetic_forest -- [--validate] [--trees N]
//! [--extent WxH] [--frames N] [--ptlas-cap-ms MS]`, through `.tools/gpu-run.ps1`.
//! `--validate` runs a small forest with the validation layer on and fails on any message.
#![allow(clippy::print_stdout)]

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orr::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, DeviceDesc, GpuPtr, Hazards, MemoryKind,
    QueueKind, QueueRequest, ShaderStage, SpecData, Stages, TimelinePoint,
};
use orr::{
    Affine3, Budgets, Display, FrameDesc, FrameStats, FrameTime, GroupWriter, InstanceFlags,
    InstanceRange, InstancesDesc, Lens, LoadState, Loading, ModelId, MountId, PackSource,
    Projection, RasterView, Renderer, RendererDesc, SceneDesc, SceneId, TextureFilter,
    TransformGroupDesc, ViewDesc, ViewHistoryId, ViewTarget,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshParts, MeshSettings, Quality, Source,
    TextureSettings, Usage,
};
use orr_math::{DVec3, Mat3, Quat, UVec2, Vec2, Vec3};
use orr_pack::material::{Material, slot};
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::read_buffer_on;
use scene_suite::shaders::trace;

const SPECIES: u32 = 4;
const VARIANTS: u32 = 3;
const TREE_SPACING: f32 = 10.0;
const CELL: f32 = 256.0;
const CELL_LOG2: u32 = 8;
const GRASS_HALF_WIDTH: f32 = 60.0;
const GRASS_SPACING: f32 = 2.0;
const EYE_HEIGHT: f32 = 1.7;
const FOV_Y: f32 = 0.9;
/// Room the partition count keeps over what the forest fills.
const PARTITION_ROOM: f32 = 1.25;
/// Software-raster thresholds swept, render pixels; 0 is the mesh-shader path alone.
const THRESHOLDS: [f32; 9] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

#[derive(Clone, Copy)]
struct Args {
    validate: bool,
    trees: u32,
    extent: [u32; 2],
    frames: u32,
    ptlas_cap_ms: f64,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            validate: false,
            trees: 10_000,
            extent: [1920, 1080],
            frames: 120,
            ptlas_cap_ms: 0.5,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut value = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
            match arg.as_str() {
                "--validate" => {
                    a.validate = true;
                    a.trees = 400;
                    a.extent = [320, 180];
                    a.frames = 3;
                }
                "--trees" => a.trees = value().parse().expect("--trees N"),
                "--frames" => a.frames = value().parse().expect("--frames N"),
                "--ptlas-cap-ms" => a.ptlas_cap_ms = value().parse().expect("--ptlas-cap-ms MS"),
                "--extent" => {
                    let v = value();
                    let (w, h) = v.split_once('x').expect("--extent WxH");
                    a.extent = [w.parse().expect("width"), h.parse().expect("height")];
                }
                // cargo bench passes --bench.
                "--bench" => {}
                _ => panic!("unknown argument {arg}"),
            }
        }
        a
    }

    /// The forest's side, m.
    fn side(&self) -> f32 {
        (self.trees as f32).sqrt().ceil() * TREE_SPACING
    }
}

// --- Content ----------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

/// A mesh under construction: one index list per section, a part per vertex.
#[derive(Default)]
struct Builder {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    parts: Vec<u16>,
    pivots: Vec<Vec3>,
    sections: Vec<Vec<u32>>,
}

impl Builder {
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2, part: u16, pivot: Vec3) -> u32 {
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        self.parts.push(part);
        self.pivots.push(pivot);
        self.positions.len() as u32 - 1
    }

    /// A triangle wound so its front faces along its vertex normals.
    fn triangle(&mut self, section: usize, t: [u32; 3]) {
        let [a, b, c] = t.map(|i| self.positions[i as usize]);
        let n: Vec3 = t.iter().map(|&i| self.normals[i as usize]).sum();
        let t = if (b - a).cross(c - a).dot(n) >= 0.0 {
            t
        } else {
            [t[0], t[2], t[1]]
        };
        if self.sections.len() <= section {
            self.sections.resize(section + 1, Vec::new());
        }
        self.sections[section].extend(t);
    }

    /// A tapered tube from `a` (radius `ra`) to `b` (radius `rb`).
    #[allow(clippy::too_many_arguments)]
    fn tube(&mut self, a: Vec3, b: Vec3, ra: f32, rb: f32, segments: u32, rings: u32, part: u16) {
        let axis = (b - a).normalize();
        let side = axis.any_orthonormal_vector();
        let up = axis.cross(side);
        let first = self.positions.len() as u32;
        for r in 0..=rings {
            let t = r as f32 / rings as f32;
            let center = a.lerp(b, t);
            let radius = ra + (rb - ra) * t;
            for s in 0..=segments {
                let phi = s as f32 / segments as f32 * std::f32::consts::TAU;
                let n = side * phi.cos() + up * phi.sin();
                self.vertex(
                    center + n * radius,
                    n,
                    Vec2::new(s as f32 / segments as f32, t * 4.0),
                    part,
                    a,
                );
            }
        }
        let w = segments + 1;
        for r in 0..rings {
            for s in 0..segments {
                let c = first + r * w + s;
                self.triangle(0, [c, c + 1, c + w]);
                self.triangle(0, [c + 1, c + w + 1, c + w]);
            }
        }
    }

    /// A bent 3 x 3 leaf patch of `size` facing `normal`.
    fn patch(&mut self, center: Vec3, normal: Vec3, size: f32, part: u16, pivot: Vec3) {
        let u = normal.any_orthonormal_vector();
        let v = normal.cross(u);
        let first = self.positions.len() as u32;
        for j in 0..4 {
            for i in 0..4 {
                let (x, y) = (i as f32 / 3.0 - 0.5, j as f32 / 3.0 - 0.5);
                let bend = 0.25 * size * (x * x + y * y);
                let p = center + (u * x + v * y) * size - normal * bend;
                let uv = Vec2::new(i as f32 / 3.0, j as f32 / 3.0);
                self.vertex(p, normal, uv, part, pivot);
            }
        }
        for j in 0..3 {
            for i in 0..3 {
                let c = first + j * 4 + i;
                self.triangle(1, [c, c + 4, c + 1]);
                self.triangle(1, [c + 1, c + 4, c + 5]);
            }
        }
    }

    fn finish(self, parts: Option<u16>) -> MeshInput {
        let mut m = MeshInput {
            positions: self.positions,
            normals: self.normals,
            uvs: Some(self.uvs),
            ..MeshInput::default()
        };
        for s in self.sections {
            m.sections.push(Span {
                first: m.indices.len() as u32,
                count: s.len() as u32,
            });
            m.indices.extend(s);
        }
        m.parts = parts.map(|count| MeshParts {
            values: self.parts,
            count,
            by_section: false,
            pivots: Some(self.pivots),
        });
        m
    }
}

/// One species' shape: height, crown radius, branches and leaf patches.
struct Species {
    height: f32,
    crown: f32,
    branches: u32,
    patches: u32,
    patch_size: f32,
}

const SPECIES_SHAPES: [Species; SPECIES as usize] = [
    Species {
        height: 22.0,
        crown: 3.0,
        branches: 32,
        patches: 1600,
        patch_size: 0.45,
    },
    Species {
        height: 16.0,
        crown: 6.0,
        branches: 24,
        patches: 1400,
        patch_size: 0.6,
    },
    Species {
        height: 12.0,
        crown: 4.5,
        branches: 20,
        patches: 1000,
        patch_size: 0.55,
    },
    Species {
        height: 7.0,
        crown: 3.5,
        branches: 12,
        patches: 700,
        patch_size: 0.5,
    },
];

/// A tree: section 0 bark (trunk and branches), section 1 leaf patches; part 0 is the
/// trunk and part `b + 1` branch `b` with its leaves, pivoted at the branch's base.
fn tree(species: u32, variant: u32) -> MeshInput {
    let s = &SPECIES_SHAPES[species as usize];
    let mut rng = Rng::new(u64::from(species * 16 + variant + 1));
    let mut m = Builder::default();
    let trunk_radius = s.height * 0.02;
    m.tube(
        Vec3::ZERO,
        Vec3::Y * s.height,
        trunk_radius,
        trunk_radius * 0.3,
        16,
        12,
        0,
    );
    let crown_base = s.height * 0.35;
    let mut branches = Vec::new();
    for b in 0..s.branches {
        let t = b as f32 / s.branches as f32;
        let y = crown_base + (s.height - crown_base) * t * 0.95;
        let yaw = b as f32 * 2.399 + rng.range(-0.3, 0.3);
        // Wider low in the crown, narrow at the top.
        let reach = s.crown * (1.0 - 0.7 * t) * rng.range(0.8, 1.1);
        let base = Vec3::new(0.0, y, 0.0);
        let dir = Vec3::new(yaw.cos(), rng.range(0.2, 0.6), yaw.sin()).normalize();
        let tip = base + dir * reach;
        m.tube(
            base,
            tip,
            trunk_radius * 0.35,
            trunk_radius * 0.08,
            8,
            4,
            b as u16 + 1,
        );
        branches.push((base, tip));
    }
    for p in 0..s.patches {
        let b = p % s.branches;
        let (base, tip) = branches[b as usize];
        let along = base.lerp(tip, rng.range(0.4, 1.0));
        let offset = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-0.6, 0.8),
            rng.range(-1.0, 1.0),
        ) * s.crown
            * 0.25;
        let normal = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-0.2, 1.0),
            rng.range(-1.0, 1.0),
        )
        .normalize_or(Vec3::Y);
        m.patch(along + offset, normal, s.patch_size, b as u16 + 1, base);
    }
    m.finish(Some(s.branches as u16 + 1))
}

/// A clump of 24 curved blades of four segments each.
fn grass() -> MeshInput {
    let mut rng = Rng::new(77);
    let mut m = Builder::default();
    for _ in 0..24 {
        let root = Vec3::new(rng.range(-0.4, 0.4), 0.0, rng.range(-0.4, 0.4));
        let yaw = rng.range(0.0, std::f32::consts::TAU);
        let side = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let lean = Vec3::new(-yaw.sin(), 0.0, yaw.cos()) * rng.range(0.1, 0.3);
        let height = rng.range(0.3, 0.6);
        let normal = side.cross(Vec3::Y).normalize();
        let first = m.positions.len() as u32;
        for k in 0..=4 {
            let t = k as f32 / 4.0;
            let center = root + Vec3::Y * height * t + lean * t * t;
            let half = 0.015 * (1.0 - t) + 0.002;
            for e in [-1.0, 1.0] {
                let uv = Vec2::new(0.5 + 0.5 * e, t);
                m.vertex(center + side * half * e, normal, uv, 0, Vec3::ZERO);
            }
        }
        for k in 0..4 {
            let c = first + 2 * k;
            m.triangle(0, [c, c + 1, c + 2]);
            m.triangle(0, [c + 1, c + 3, c + 2]);
        }
    }
    m.finish(None)
}

/// Ground height at world `(x, z)`: rolling hills of the cell size's period, so every
/// cell's tile is the same mesh.
fn height(x: f32, z: f32) -> f32 {
    let k = std::f32::consts::TAU / CELL;
    2.5 * (k * x).sin() * (k * z).sin() + 0.8 * (3.0 * k * x + 1.0).sin() * (2.0 * k * z).cos()
}

/// One cell's ground, centred on the cell: 128 x 128 quads at 2 m.
fn ground() -> MeshInput {
    let n = 128u32;
    let mut m = Builder::default();
    let half = CELL / 2.0;
    for j in 0..=n {
        for i in 0..=n {
            let (x, z) = (i as f32 * 2.0 - half, j as f32 * 2.0 - half);
            let (wx, wz) = (x + half, z + half);
            let e = 0.5;
            let dx = (height(wx + e, wz) - height(wx - e, wz)) / (2.0 * e);
            let dz = (height(wx, wz + e) - height(wx, wz - e)) / (2.0 * e);
            let normal = Vec3::new(-dx, 1.0, -dz).normalize();
            let uv = Vec2::new(x, z) / 4.0;
            m.vertex(Vec3::new(x, height(wx, wz), z), normal, uv, 0, Vec3::ZERO);
        }
    }
    let w = n + 1;
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            m.triangle(0, [c, c + w, c + 1]);
            m.triangle(0, [c + 1, c + w, c + w + 1]);
        }
    }
    m.finish(None)
}

fn tree_name(species: u32, variant: u32) -> String {
    format!("tree {species} {variant}")
}

/// A leaf-shaped opacity mask as a PNG in the temp directory.
fn leaf_mask() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-synthetic-forest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("leaf.png");
    let n = 128u32;
    let data: Vec<u8> = (0..n * n)
        .map(|i| {
            let (u, v) = (
                (i % n) as f32 / n as f32 - 0.5,
                (i / n) as f32 / n as f32 - 0.5,
            );
            // Four leaves around the patch centre.
            let a = v.atan2(u) * 2.0;
            let r = (u * u + v * v).sqrt();
            if r < 0.45 * a.cos().abs().sqrt() && r > 0.04 {
                255
            } else {
                0
            }
        })
        .collect();
    let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), n, n);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&data).unwrap();
    path
}

fn cook() -> Arc<[u8]> {
    let started = Instant::now();
    let bark = Material {
        base_color: [0.35, 0.27, 0.2],
        specular_roughness: 0.8,
        ..Material::default()
    };
    let mut leaf = Material {
        base_color: [0.18, 0.4, 0.12],
        alpha_mode: Material::MASK,
        alpha_cutoff: 0.5,
        double_sided: 1,
        ..Material::default()
    };
    leaf.textures[slot::OPACITY].texture = AssetId::from_name("tex/leaf");
    let blade = Material {
        base_color: [0.25, 0.45, 0.15],
        double_sided: 1,
        ..Material::default()
    };
    let soil = Material {
        base_color: [0.3, 0.25, 0.18],
        specular_roughness: 0.9,
        ..Material::default()
    };
    let model = |name: String, mesh: MeshInput, materials, errors: Vec<f32>| AssetSpec {
        name,
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings {
                lod: LodMode::Discrete { errors },
                merge_level: Some(2),
                ..MeshSettings::default()
            },
        },
    };
    let mut assets = vec![AssetSpec {
        name: "tex/leaf".into(),
        source: Source::Texture {
            path: leaf_mask(),
            settings: TextureSettings {
                usage: Usage::Linear1,
                quality: Quality::Fast,
                ..TextureSettings::default()
            },
        },
    }];
    for s in 0..SPECIES {
        for v in 0..VARIANTS {
            assets.push(model(
                tree_name(s, v),
                tree(s, v),
                vec![bark, leaf],
                vec![0.01, 0.05, 0.2, 0.8],
            ));
        }
    }
    assets.push(model("grass".into(), grass(), vec![blade], vec![0.02, 0.1]));
    assets.push(model(
        "ground".into(),
        ground(),
        vec![soil],
        vec![0.05, 0.25, 1.0],
    ));
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    println!(
        "cooked {} MB in {:.1} s",
        pack.len() >> 20,
        started.elapsed().as_secs_f64()
    );
    pack
}

// --- Placement --------------------------------------------------------------------------------

/// One copy: its model, its cell and its transform relative to the cell's anchor.
struct Placed {
    model: usize,
    cell: (i32, i32),
    at: Affine3,
    rt_range: f32,
    tree: bool,
}

fn anchor(cell: (i32, i32)) -> DVec3 {
    DVec3::new(
        (f64::from(cell.0) + 0.5) * f64::from(CELL),
        0.0,
        (f64::from(cell.1) + 0.5) * f64::from(CELL),
    )
}

fn cell_of(x: f32, z: f32) -> (i32, i32) {
    ((x / CELL).floor() as i32, (z / CELL).floor() as i32)
}

fn affine(m: Mat3, world: Vec3, cell: (i32, i32)) -> Affine3 {
    let t = world - anchor(cell).as_vec3();
    let c = m.to_cols_array_2d();
    Affine3 {
        rows: [
            [c[0][0], c[1][0], c[2][0], t.x],
            [c[0][1], c[1][1], c[2][1], t.y],
            [c[0][2], c[1][2], c[2][2], t.z],
        ],
    }
}

/// Model indices: trees first by species and variant, then grass, then ground.
const GRASS: usize = (SPECIES * VARIANTS) as usize;
const GROUND: usize = GRASS + 1;

/// The forest's copies, sorted by cell; within a cell trees first, in placement order.
fn copies(args: &Args) -> Vec<Placed> {
    let side = args.side();
    let mut rng = Rng::new(5);
    let mut out = Vec::new();
    let per_row = (side / TREE_SPACING) as u32;
    for i in 0..args.trees {
        let (gx, gz) = ((i % per_row) as f32, (i / per_row) as f32);
        let x = (gx + 0.5) * TREE_SPACING + rng.range(-4.0, 4.0);
        let z = (gz + 0.5) * TREE_SPACING + rng.range(-4.0, 4.0);
        let pick = (rng.next() * (SPECIES * VARIANTS) as f32) as usize;
        let m = Mat3::from_rotation_y(rng.range(0.0, std::f32::consts::TAU)) * rng.range(0.8, 1.2);
        let cell = cell_of(x, z);
        out.push(Placed {
            model: pick.min(GRASS - 1),
            cell,
            at: affine(m, Vec3::new(x, height(x, z), z), cell),
            rt_range: 400.0,
            tree: true,
        });
    }
    let mid = side / 2.0;
    let rows = (2.0 * GRASS_HALF_WIDTH / GRASS_SPACING) as u32;
    let cols = (side / GRASS_SPACING) as u32;
    for r in 0..rows {
        for c in 0..cols {
            let x = (c as f32 + rng.next()) * GRASS_SPACING;
            let z = mid - GRASS_HALF_WIDTH + (r as f32 + rng.next()) * GRASS_SPACING;
            let m = Mat3::from_rotation_y(rng.range(0.0, std::f32::consts::TAU));
            let cell = cell_of(x, z);
            out.push(Placed {
                model: GRASS,
                cell,
                at: affine(m, Vec3::new(x, height(x, z), z), cell),
                rt_range: 60.0,
                tree: false,
            });
        }
    }
    let cells = (side / CELL).ceil() as i32;
    for cz in 0..cells {
        for cx in 0..cells {
            let cell = (cx, cz);
            out.push(Placed {
                model: GROUND,
                cell,
                at: affine(Mat3::IDENTITY, anchor(cell).as_vec3(), cell),
                rt_range: 4000.0,
                tree: false,
            });
        }
    }
    out.sort_by_key(|c| (c.cell, !c.tree));
    out
}

/// Threads lent to the loading side until dropped.
struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    fn new(loading: Arc<Loading>) -> Self {
        let threads = (0..4)
            .map(|_| {
                let l = Arc::clone(&loading);
                std::thread::spawn(move || l.run_worker())
            })
            .collect();
        Self { loading, threads }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.loading.stop_workers();
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}

fn time(frame: u64) -> FrameTime {
    FrameTime {
        absolute: frame as f64 / 60.0,
        delta: 1.0 / 60.0,
    }
}

fn wait(device: &Device, point: TimelinePoint) {
    let render = device.queue(QueueKind::Render).unwrap();
    render.wait(point, Duration::from_secs(30)).expect("wait");
}

/// A renderer with the forest mounted and placed in one scene.
struct Forest {
    device: &'static Device,
    r: Renderer,
    scene: SceneId,
    history: ViewHistoryId,
    /// One per tree, in copy order.
    trees: Vec<InstanceRange>,
    frame: u64,
    /// Device-local bytes the mount added, and the wall time it took.
    mount_bytes: u64,
    mount_ms: f64,
    cells: usize,
    /// The frame that placed the forest and built the PTLAS in full.
    placed: FrameStats,
    _workers: Workers,
}

fn mount(r: &mut Renderer, pack: &Arc<[u8]>) -> MountId {
    let m = r.mount(PackSource::Memory(Arc::clone(pack)));
    let start = Instant::now();
    while matches!(r.mount_state(m), LoadState::Pending) {
        let f = r
            .begin_frame(FrameDesc {
                time: time(0),
                scenes: &[],
                display: None,
            })
            .unwrap();
        f.end().unwrap();
        assert!(start.elapsed() < Duration::from_secs(600), "mount pending");
        std::thread::sleep(Duration::from_millis(1));
    }
    if let LoadState::Failed(e) = r.mount_state(m) {
        panic!("mount failed: {e}");
    }
    m
}

impl Forest {
    fn new(device: &'static Device, desc: RendererDesc, pack: &Arc<[u8]>, args: &Args) -> Self {
        let mut r = Renderer::new(device, desc).unwrap();
        let workers = Workers::new(r.loading());
        // Earlier forests' memory goes first, so the mount's own shows.
        device.collect_garbage();
        let before = device.memory_report().kind(MemoryKind::DeviceLocal).bytes;
        let start = Instant::now();
        let m = mount(&mut r, pack);
        let mount_ms = start.elapsed().as_secs_f64() * 1e3;
        let mount_bytes = device
            .memory_report()
            .kind(MemoryKind::DeviceLocal)
            .bytes
            .saturating_sub(before);
        let mut names: Vec<String> = (0..SPECIES)
            .flat_map(|s| (0..VARIANTS).map(move |v| tree_name(s, v)))
            .collect();
        names.extend(["grass".into(), "ground".into()]);
        let models: Vec<ModelId> = names
            .iter()
            .map(|n| r.model(m, AssetId::from_name(n)).unwrap())
            .collect();
        let copies = copies(args);
        let capacity = (copies.len() as u32 + 1024).next_power_of_two();
        let scene = r
            .create_scene(
                &SceneDesc {
                    instance_capacity: capacity,
                    ..SceneDesc::default()
                },
                &Default::default(),
            )
            .unwrap();
        let history = r.create_view_history().unwrap();
        let list = [scene];
        let f = r
            .begin_frame(FrameDesc {
                time: time(1),
                scenes: &list,
                display: None,
            })
            .unwrap();
        let mut trees = Vec::new();
        let mut cells = 0;
        {
            let mut s = f.scene(scene).unwrap();
            for cell in copies.chunk_by(|a, b| a.cell == b.cell) {
                cells += 1;
                let group = s
                    .create_transform_group(&TransformGroupDesc {
                        count: cell.len() as u32,
                        writer: GroupWriter::Cpu,
                        anchor: anchor(cell[0].cell),
                    })
                    .unwrap();
                let transforms = s.transforms(group, 0, cell.len() as u32).unwrap();
                for (t, c) in transforms.iter_mut().zip(cell) {
                    *t = c.at;
                }
                for (i, c) in cell.iter().enumerate() {
                    let range = s
                        .add_instances(&InstancesDesc {
                            model: models[c.model],
                            copies: 1,
                            transforms: group,
                            first: i as u32,
                            pose: None,
                            material_set: None,
                            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                            rt_range: c.rt_range,
                            animation_range: 0.0,
                            user_tag: 0,
                        })
                        .unwrap();
                    if c.tree {
                        trees.push(range);
                    }
                }
            }
        }
        wait(device, f.end().unwrap());
        let mut forest = Self {
            device,
            r,
            scene,
            history,
            trees,
            frame: 1,
            mount_bytes,
            mount_ms,
            cells,
            placed: FrameStats::default(),
            _workers: workers,
        };
        forest.placed = forest.settle();
        forest
    }
}

// --- Frames -----------------------------------------------------------------------------------

/// One frame of the camera script.
struct Shot {
    eye: Vec3,
    target: Vec3,
    focus: Vec3,
    /// Trees whose RAY flag this frame sets (`true`) or clears.
    flags: Vec<(usize, bool)>,
    trace: bool,
}

/// The camera script: down the middle of the forest along x at eye height, looking ahead.
fn path(args: &Args, i: u32, frames: u32) -> (Vec3, Vec3) {
    let side = args.side();
    let mid = side / 2.0;
    let t = if frames > 1 {
        i as f32 / (frames - 1) as f32
    } else {
        0.0
    };
    let x = 20.0 + (side - 60.0).max(0.0) * t;
    let eye = Vec3::new(x, height(x, mid) + EYE_HEIGHT, mid);
    let target = Vec3::new(
        x + 30.0,
        height(x + 30.0, mid) + 4.0,
        mid + 6.0 * (t * 9.0).sin(),
    );
    (eye, target)
}

/// World from view for a camera at `eye` looking at `target`, y up.
fn look(eye: Vec3, target: Vec3) -> Quat {
    let back = (eye - target).normalize();
    let right = Vec3::Y.cross(back).normalize();
    Quat::from_mat3(&Mat3::from_cols(right, back.cross(right), back))
}

/// What one frame measured.
struct Measured {
    stats: FrameStats,
    /// `RasterView::counts`.
    counts: Vec<u32>,
}

/// GPU ms of the passes named `pass`.
fn pass_ms(stats: &FrameStats, pass: &str) -> f64 {
    stats
        .gpu
        .iter()
        .filter(|p| p.name == pass)
        .map(|p| p.gpu_ms)
        .sum()
}

/// Camera rays through the PTLAS, one per pixel, timed as the frame's `trace` pass.
struct Tracer {
    pipeline: ComputePipeline,
    traced: Allocation<trace::Traced>,
    root: Allocation<trace::Root>,
}

impl Tracer {
    fn new(device: &Device, extent: [u32; 2]) -> Self {
        let stage = ShaderStage::new(&trace::MODULE, "pixels", SpecData::default()).unwrap();
        let pipeline = device
            .create_compute_pipeline(&stage, trace::MODULE.name)
            .unwrap();
        let n = u64::from(extent[0] * extent[1]);
        Self {
            pipeline,
            traced: device
                .gpu_malloc(MemoryKind::DeviceLocal, n, "traced")
                .unwrap(),
            root: device
                .gpu_malloc(MemoryKind::Upload, 1, "trace root")
                .unwrap(),
        }
    }
}

impl Forest {
    fn shoot(&mut self, args: &Args, shot: &Shot, tracer: Option<&mut Tracer>) -> Measured {
        self.frame += 1;
        let list = [self.scene];
        let mut f = self
            .r
            .begin_frame(FrameDesc {
                time: time(self.frame),
                scenes: &list,
                display: None,
            })
            .unwrap();
        {
            let mut s = f.scene(self.scene).unwrap();
            s.set_focus(shot.focus.as_dvec3(), args.extent[1] as f32 / FOV_Y);
            for &(tree, on) in &shot.flags {
                let flags = if on {
                    InstanceFlags::RASTER | InstanceFlags::RAY
                } else {
                    InstanceFlags::RASTER
                };
                s.set_instance_flags(self.trees[tree], flags);
            }
        }
        let view = f
            .raster_view(&view_desc(self.scene, self.history, args, shot, self.frame))
            .unwrap();
        if shot.trace
            && let Some(t) = tracer
        {
            record_trace(f.list().unwrap(), t, &view);
        }
        let point = f.end().unwrap();
        wait(self.device, point);
        let counts = read_buffer_on(self.device, view.counts, 8);
        Measured {
            stats: self.settle(),
            counts,
        }
    }

    /// Runs an empty frame so the last one's figures are taken in, and returns them.
    fn settle(&mut self) -> FrameStats {
        self.frame += 1;
        let f = self
            .r
            .begin_frame(FrameDesc {
                time: time(self.frame),
                scenes: &[],
                display: None,
            })
            .unwrap();
        wait(self.device, f.end().unwrap());
        self.r.stats().clone()
    }
}

fn view_desc<'a>(
    scene: SceneId,
    history: ViewHistoryId,
    args: &Args,
    shot: &Shot,
    frame: u64,
) -> ViewDesc<'a> {
    ViewDesc {
        scene,
        history,
        position: shot.eye.as_dvec3(),
        rotation: look(shot.eye, shot.target),
        projection: Projection {
            fov_y: FOV_Y,
            near: 0.1,
        },
        lens: Lens::default(),
        render_extent: UVec2::from(args.extent),
        jitter: Vec2::ZERO,
        reset_history: frame <= 3,
        mip_bias: 0.0,
        texture_filter: TextureFilter::Hardware,
        target: ViewTarget::AfterRr,
    }
}

fn record_trace(list: &mut CommandList<'_>, t: &mut Tracer, view: &RasterView) {
    t.root.as_mut_slice().unwrap()[0] = trace::Root {
        scene: view.scene,
        view: view.view,
        gbuffer: view.gbuffer,
        traced: t.traced.gpu,
        seen: GpuPtr::NULL,
        cones: GpuPtr::NULL,
        mode: 0,
        pad0: 0,
        pad1: 0,
    };
    let n = t.traced.size / size_of::<trace::Traced>() as u64;
    list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
    list.begin_pass("trace", false);
    list.dispatch(&t.pipeline, t.root.gpu, [n.div_ceil(64) as u32, 1, 1]);
    list.end_pass();
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

// --- Sweeps -----------------------------------------------------------------------------------

/// Visibility-buffer raster and resolve per software-raster threshold, and the share of
/// visible clusters at or below each threshold: the cluster screen-size histogram.
fn raster_sweep(device: &'static Device, pack: &Arc<[u8]>, args: &Args) {
    println!(
        "\nraster: median ms over {} frames at {:?}",
        args.frames, args.extent
    );
    println!(
        "{:>9} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8} {:>9} {:>10} {:>9}",
        "threshold",
        "cull",
        "raster",
        "saving",
        "resolve",
        "hzb",
        "depth",
        "visible",
        "software",
        "<= share"
    );
    let thresholds: &[f32] = if args.validate {
        &[0.0, 32.0]
    } else {
        &THRESHOLDS
    };
    let mut base = None;
    for &t in thresholds {
        let desc = RendererDesc {
            software_raster_pixels: t,
            cell_log2: CELL_LOG2,
            ..RendererDesc::default()
        };
        let mut forest = Forest::new(device, desc, pack, args);
        let runs: Vec<Measured> = (0..args.frames)
            .map(|i| {
                let (eye, target) = path(args, i, args.frames);
                let shot = Shot {
                    eye,
                    target,
                    focus: eye,
                    flags: Vec::new(),
                    trace: false,
                };
                forest.shoot(args, &shot, None)
            })
            .collect();
        let ms = |pass: &str| median(runs.iter().map(|m| pass_ms(&m.stats, pass)).collect());
        let count = |i: usize| median(runs.iter().map(|m| f64::from(m.counts[i])).collect());
        let (visible, software) = (count(0), count(7));
        let raster = ms("raster");
        // Raster saving against the mesh-shader path alone, threshold 0.
        let base = *base.get_or_insert(raster);
        let saving = if base > 0.0 { 1.0 - raster / base } else { 0.0 };
        println!(
            "{t:>9} {:>8.3} {raster:>8.3} {:>6.1}% {:>8.3} {:>8.3} {:>8.3} {visible:>9} {software:>10} {:>9.3}",
            ms("cull"),
            saving * 100.0,
            ms("resolve"),
            ms("hzb"),
            ms("depth"),
            if visible > 0.0 {
                software / visible
            } else {
                0.0
            },
        );
        if args.validate {
            assert!(visible > 0.0, "the forest is visible");
        }
    }
}

/// PTLAS ms against rebuilt instances with writes and BLAS updates in one build, the trace
/// after those in-place writes against the trace after the first full build, the
/// partitions the forest fills and the PTLAS bytes.
fn ptlas_sweep(device: &'static Device, pack: &Arc<[u8]>, args: &Args) {
    let desc = RendererDesc {
        cell_log2: CELL_LOG2,
        budgets: Budgets {
            ptlas_rebuilt: u32::MAX,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut forest = Forest::new(device, desc, pack, args);
    let mut tracer = Tracer::new(device, args.extent);
    let (eye, target) = path(args, 0, 1);
    // The focus jumps between two points so cuts change and BLAS updates join the writes.
    let near = eye;
    let far = eye + Vec3::new(10.0, 0.0, 0.0);
    let shot = |focus, flags, trace| Shot {
        eye,
        target,
        focus,
        flags,
        trace,
    };
    let rt = forest.placed.rt;
    println!(
        "\nmount: {:.0} ms, {:.1} MB device-local; {} trees in {} cells",
        forest.mount_ms,
        forest.mount_bytes as f64 / 1e6,
        forest.trees.len(),
        forest.cells
    );
    println!(
        "full build: {} writes, ptlas {:.3} ms",
        rt.writes,
        pass_ms(&forest.placed, "ptlas")
    );
    println!(
        "partitions: {} filled of {} at {} instances each, {} unplaced; {} with {}% room",
        rt.partitions,
        SceneDesc::default().ptlas_partitions,
        SceneDesc::default().partition_instances,
        rt.unplaced,
        (rt.partitions as f32 * PARTITION_ROOM).ceil(),
        ((PARTITION_ROOM - 1.0) * 100.0).round()
    );
    println!("PTLAS bytes: {:.1} MB", rt.bytes as f64 / 1e6);
    if args.validate {
        assert!(rt.writes > 0 && rt.partitions > 0, "the forest is traced");
    }
    // The focus's cut changes land, then frames with no PTLAS work trace the built PTLAS.
    forest.shoot(args, &shot(near, Vec::new(), false), None);
    forest.shoot(args, &shot(near, Vec::new(), true), Some(&mut tracer));
    let steady = forest.shoot(args, &shot(near, Vec::new(), true), Some(&mut tracer));
    println!(
        "steady frame: {} writes, {} updates, trace {:.3} ms",
        steady.stats.rt.writes,
        steady.stats.rt.updates,
        pass_ms(&steady.stats, "trace")
    );

    let trees = forest.trees.len();
    let counts: Vec<usize> = if args.validate {
        vec![10]
    } else {
        [100, 300, 1000, 3000, 10_000, 30_000]
            .into_iter()
            .filter(|&k| k <= trees)
            .collect()
    };
    println!(
        "\nptlas: {:>9} {:>7} {:>8} {:>8} {:>9} {:>9} {:>9}",
        "layout", "trees", "writes", "updates", "rebuilt", "ptlas ms", "trace ms"
    );
    let mut fit = Vec::new();
    for &k in &counts {
        for (layout, chosen) in [
            // One tree per stride: writes spread over every partition.
            (
                "scattered",
                (0..k).map(|i| i * trees / k).collect::<Vec<_>>(),
            ),
            // Consecutive trees: writes fill whole partitions.
            ("aligned", (0..k).collect()),
        ] {
            for on in [false, true] {
                let focus = if on { near } else { far };
                let flags = chosen.iter().map(|&t| (t, on)).collect();
                let m = forest.shoot(args, &shot(focus, flags, true), Some(&mut tracer));
                let rt = m.stats.rt;
                let ptlas = pass_ms(&m.stats, "ptlas");
                fit.push((f64::from(rt.rebuilt), ptlas));
                println!(
                    "       {layout:>9} {k:>7} {:>8} {:>8} {:>9} {ptlas:>9.3} {:>9.3}",
                    rt.writes,
                    rt.updates,
                    rt.rebuilt,
                    pass_ms(&m.stats, "trace")
                );
                if args.validate {
                    assert!(rt.writes > 0 && rt.deferred == 0, "the toggles are written");
                }
            }
        }
    }
    // Least squares: ms = a + b x rebuilt; the clamp is the rebuilt count at the cap.
    let n = fit.len() as f64;
    let (sx, sy) = fit.iter().fold((0.0, 0.0), |(x, y), p| (x + p.0, y + p.1));
    let sxx: f64 = fit.iter().map(|p| (p.0 - sx / n).powi(2)).sum();
    let sxy: f64 = fit.iter().map(|p| (p.0 - sx / n) * (p.1 - sy / n)).sum();
    if sxx > 0.0 {
        let b = sxy / sxx;
        let a = sy / n - b * sx / n;
        let clamp = if b > 0.0 {
            (args.ptlas_cap_ms - a) / b
        } else {
            f64::INFINITY
        };
        println!(
            "ptlas ms = {a:.3} + {:.3} per million rebuilt; {clamp:.0} rebuilt at {} ms (Budgets::ptlas_rebuilt {})",
            b * 1e6,
            args.ptlas_cap_ms,
            Budgets::default().ptlas_rebuilt
        );
    }
}

/// Render-thread time per step on frames that present to a window while a second mount
/// of the pack streams in and out on the loading side.
#[cfg(windows)]
fn cpu_sweep(device: &'static Device, pack: &Arc<[u8]>, args: &Args) {
    use gpu_suite::window::Window;
    use orr::DisplayRequest;

    let mut window = Window::new("synthetic forest", 1280, 720);
    let desc = RendererDesc {
        cell_log2: CELL_LOG2,
        ..RendererDesc::default()
    };
    let mut display = Display::new(
        device,
        window.window_handle(),
        window.display_handle(),
        &DisplayRequest {
            vsync: false,
            ..DisplayRequest::default()
        },
        None,
        desc.frames_in_flight,
    )
    .unwrap();
    let pack = Arc::clone(pack);
    let frames = args.frames.max(30);
    let a = *args;
    let render = std::thread::spawn(move || {
        let mut forest = Forest::new(device, desc, &pack, &a);
        let list = [forest.scene];
        let mut steps: Vec<(&'static str, Vec<f64>)> = Vec::new();
        let mut streaming: Option<MountId> = None;
        let mut last = None;
        for i in 0..frames {
            if i % 20 == 0 {
                match streaming.take() {
                    Some(m) => forest.r.unmount(m),
                    None => streaming = Some(forest.r.mount(PackSource::Memory(Arc::clone(&pack)))),
                }
            }
            let (eye, target) = path(&a, i, frames);
            let shot = Shot {
                eye,
                target,
                focus: eye,
                flags: Vec::new(),
                trace: false,
            };
            forest.frame += 1;
            let mut f = forest
                .r
                .begin_frame(FrameDesc {
                    time: time(forest.frame),
                    scenes: &list,
                    display: Some(&mut display),
                })
                .unwrap();
            f.scene(forest.scene)
                .unwrap()
                .set_focus(eye.as_dvec3(), a.extent[1] as f32 / FOV_Y);
            f.raster_view(&view_desc(
                forest.scene,
                forest.history,
                &a,
                &shot,
                forest.frame,
            ))
            .unwrap();
            last = Some(f.end().unwrap());
            // The first frames' figures are the mount's and the warm-up's.
            if i < 10 {
                continue;
            }
            for &(name, d) in &forest.r.stats().cpu {
                let ms = d.as_secs_f64() * 1e3;
                match steps.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, v)) => v.push(ms),
                    None => steps.push((name, vec![ms])),
                }
            }
        }
        if let Some(m) = streaming {
            forest.r.unmount(m);
        }
        display.wait_idle();
        if let Some(p) = last {
            wait(device, p);
        }
        steps
    });
    while !render.is_finished() {
        window.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    let steps = render.join().unwrap();
    println!("\nrender thread: ms per step over {frames} presented frames, median and p95");
    let mut total = 0.0;
    for (name, v) in steps {
        let m = median(v.clone());
        if name != "begin_frame" && name != "write" {
            total += m;
        }
        println!("{name:>14} {m:>8.3} {:>8.3}", percentile(v, 0.95));
    }
    println!(
        "{:>14} {total:>8.3} (begin_frame and write excluded)",
        "sum"
    );
}

#[cfg(not(windows))]
fn cpu_sweep(_: &'static Device, _: &Arc<[u8]>, _: &Args) {
    println!("\nrender thread: skipped, the presenting window needs Windows");
}

fn main() {
    let args = Args::parse();
    #[cfg(windows)]
    gpu_suite::log_to_stderr();
    let device: &'static Device = Box::leak(Box::new(
        Device::new(&DeviceDesc {
            name: "synthetic forest",
            validation: args.validate,
            queues: QueueRequest {
                render_compute: true,
                async_compute: true,
                copy: true,
                present: true,
            },
            ..DeviceDesc::default()
        })
        .expect("device"),
    ));
    let caps = device.caps();
    println!(
        "{} {}, {} trees over {:.0} m",
        caps.device_name,
        caps.driver,
        args.trees,
        args.side()
    );
    let pack = cook();
    raster_sweep(device, &pack, &args);
    ptlas_sweep(device, &pack, &args);
    cpu_sweep(device, &pack, &args);
    device.collect_garbage();
    if args.validate {
        assert_eq!(device.validation_messages(), 0, "validation messages");
        println!("validation clean");
    }
}
