//! The raygen and compute entries of one kernel body trace the same paths.

use gpu_suite::parity::paths;
use gpu_suite::{assert_clean, device, readback, render, run, upload};
use orr_gpu::{
    Allocation, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance,
    ClassicKind, GpuPtr, Hazards, InstanceFlags, MemoryKind, RtasFlags, ShaderStage, SpecData,
    Stages,
};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const LEAVES: usize = 300;

/// A small deterministic generator for scene layout.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: V3) -> V3 {
    scale(a, 1.0 / (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt())
}

/// A ground quad, a trunk and a crown of leaf quads, each as unindexed triangles. Leaf
/// quads are two triangles, corners (0,0) (1,0) (1,1) then (0,0) (1,1) (0,1), as the
/// kernel's alpha test reads them.
fn meshes(rng: &mut Lcg) -> [(Vec<V3>, bool); 3] {
    let g = 200.0;
    let ground = vec![
        [-g, 0.0, -g],
        [g, 0.0, -g],
        [g, 0.0, g],
        [-g, 0.0, -g],
        [g, 0.0, g],
        [-g, 0.0, g],
    ];
    let mut trunk = Vec::new();
    let sides = 8;
    let ring = |i: usize, y: f32| {
        let a = i as f32 / sides as f32 * std::f32::consts::TAU;
        [0.25 * a.cos(), y, 0.25 * a.sin()]
    };
    for i in 0..sides {
        let (a0, a1, b0, b1) = (
            ring(i, 0.0),
            ring(i + 1, 0.0),
            ring(i, 6.0),
            ring(i + 1, 6.0),
        );
        trunk.extend([a0, a1, b1, a0, b1, b0]);
    }
    let mut leaves = Vec::new();
    for _ in 0..LEAVES {
        let centre = loop {
            let p = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if p.iter().map(|v| v * v).sum::<f32>() <= 1.0 {
                break add(scale(p, 2.5), [0.0, 6.0, 0.0]);
            }
        };
        let u = normalize([
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ]);
        let w = normalize(cross(u, [rng.range(-1.0, 1.0), 1.0, rng.range(-1.0, 1.0)]));
        let (u, v) = (scale(u, 0.6), scale(cross(w, u), 0.6));
        let corner = |a: f32, b: f32| add(centre, add(scale(u, a - 0.5), scale(v, b - 0.5)));
        let (c00, c10, c11, c01) = (
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(1.0, 1.0),
            corner(0.0, 1.0),
        );
        leaves.extend([c00, c10, c11, c00, c11, c01]);
    }
    [(ground, true), (trunk, true), (leaves, false)]
}

struct Scene {
    tlas: ClassicAs,
    transforms: Allocation<[[f32; 4]; 3]>,
    _blas: Vec<ClassicAs>,
    _blocks: Vec<Allocation<u8>>,
    _positions: Vec<Allocation<V3>>,
    _instances: Allocation<ClassicInstance>,
}

fn build(
    desc: &ClassicDesc,
    kind: ClassicKind,
    blocks: &mut Vec<Allocation<u8>>,
) -> (ClassicAs, Allocation<u8>) {
    let d = device();
    let sizes = d.classic_sizes(desc);
    let block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "parity as")
        .unwrap();
    let scratch = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.scratch, "parity scratch")
        .unwrap();
    let accel = d
        .create_classic_as(kind, sizes.result_max, &block, 0)
        .unwrap();
    blocks.push(block);
    (accel, scratch)
}

/// A ground, a grid of 144 trees (trunk and leaf crown, custom indices 1 and 2) and the
/// instance transforms the kernel reads normals through.
fn forest() -> Scene {
    let mut rng = Lcg(0x5eed);
    let mut blocks = Vec::new();
    let mut positions = Vec::new();
    let mut blas = Vec::new();
    let mut list = render().command_list().unwrap();
    let mut scratch = Vec::new();
    for (vertices, opaque) in meshes(&mut rng) {
        let p = upload(&vertices, "parity positions");
        let desc = ClassicDesc {
            kind: ClassicKind::Blas,
            flags: RtasFlags::FAST_TRACE | RtasFlags::ALLOW_DATA_ACCESS,
            geometry: ClassicGeometry::Triangles {
                positions: p.gpu,
                vertex_count: vertices.len() as u32,
                indices: GpuPtr::NULL,
                triangle_count: vertices.len() as u32 / 3,
                opaque,
            },
        };
        let (accel, s) = build(&desc, ClassicKind::Blas, &mut blocks);
        list.build_classic(&ClassicBuild {
            desc: &desc,
            dst: &accel,
            scratch: s.gpu.address(),
        });
        scratch.push(s);
        positions.push(p);
        blas.push(accel);
    }

    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let mut transforms = vec![identity];
    let mut instances = vec![ClassicInstance::new(
        identity,
        0,
        0xff,
        InstanceFlags::CULL_DISABLE,
        blas[0].address(),
    )];
    for i in 0..12 {
        for j in 0..12 {
            let x = -38.5 + 7.0 * i as f32 + rng.range(-2.0, 2.0);
            let z = -30.0 + 7.0 * j as f32 + rng.range(-2.0, 2.0);
            let (s, a) = (rng.range(0.8, 1.2), rng.range(0.0, std::f32::consts::TAU));
            let (c, n) = (s * a.cos(), s * a.sin());
            let t = [[c, 0.0, n, x], [0.0, s, 0.0, 0.0], [-n, 0.0, c, z]];
            for (custom, b) in [(1, &blas[1]), (2, &blas[2])] {
                transforms.push(t);
                instances.push(ClassicInstance::new(
                    t,
                    custom,
                    0xff,
                    InstanceFlags::CULL_DISABLE,
                    b.address(),
                ));
            }
        }
    }
    let records = upload(&instances, "parity instances");
    let desc = ClassicDesc {
        kind: ClassicKind::Tlas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Instances {
            records: records.gpu.address(),
            capacity: instances.len() as u32,
        },
    };
    let (tlas, s) = build(&desc, ClassicKind::Tlas, &mut blocks);
    list.build_classic(&ClassicBuild {
        desc: &desc,
        dst: &tlas,
        scratch: s.gpu.address(),
    });
    scratch.push(s);
    run(list);
    Scene {
        tlas,
        transforms: upload(&transforms, "parity transforms"),
        _blas: blas,
        _blocks: blocks,
        _positions: positions,
        _instances: records,
    }
}

/// The hash of a path whose camera ray misses everything (the kernel's `hash_mix`).
fn sky_hash() -> u32 {
    (2_166_136_261u32 ^ 0xffff_fffe).wrapping_mul(16_777_619)
}

/// One entry's dispatch: its root and outputs.
struct Run {
    rad: Allocation<[f32; 4]>,
    hash: Allocation<u32>,
    root: Allocation<paths::Root>,
}

impl Run {
    fn new(scene: &Scene, bounces: u32) -> Run {
        let pixels = u64::from(WIDTH * HEIGHT);
        let fwd = normalize([0.0, -0.05, 1.0]);
        let right = normalize(cross([0.0, 1.0, 0.0], fwd));
        let up = cross(fwd, right);
        let sun = normalize([0.4, 0.8, 0.3]);
        let rad = readback::<[f32; 4]>(pixels, "parity radiance");
        let hash = readback::<u32>(pixels, "parity hashes");
        let root = upload(
            &[paths::Root {
                cam_pos: [0.0, 2.5, -45.0, 30.0f32.to_radians().tan()],
                cam_fwd: [fwd[0], fwd[1], fwd[2], WIDTH as f32 / HEIGHT as f32],
                cam_right: [right[0], right[1], right[2], 0.0],
                cam_up: [up[0], up[1], up[2], 0.0],
                sun: [sun[0], sun[1], sun[2], 3.0],
                out_rad: rad.gpu,
                out_hash: hash.gpu,
                transforms: scene.transforms.gpu,
                scene: scene.tlas.tlas(),
                width: WIDTH,
                height: HEIGHT,
                frame: 3,
                bounces,
            }],
            "parity root",
        );
        Run { rad, hash, root }
    }
}

/// How far the raygen run strays from the compute run.
#[derive(Debug)]
struct Diff {
    /// Pixels whose path hashes differ.
    paths: usize,
    /// Pixels with equal paths whose radiance differs by more than 1%.
    radiance: usize,
    /// Largest relative radiance difference over equal paths.
    max_relative: f32,
}

fn compare(compute: &Run, raygen: &Run) -> Diff {
    let (ha, hb) = (
        compute.hash.as_slice().unwrap(),
        raygen.hash.as_slice().unwrap(),
    );
    let (ra, rb) = (
        compute.rad.as_slice().unwrap(),
        raygen.rad.as_slice().unwrap(),
    );
    let mut d = Diff {
        paths: 0,
        radiance: 0,
        max_relative: 0.0,
    };
    for i in 0..ha.len() {
        if ha[i] != hb[i] {
            d.paths += 1;
            continue;
        }
        let rel = (0..3)
            .map(|c| (ra[i][c] - rb[i][c]).abs() / ra[i][c].abs().max(rb[i][c].abs()).max(1e-6))
            .fold(0.0f32, f32::max);
        d.max_relative = d.max_relative.max(rel);
        if rel > 0.01 {
            d.radiance += 1;
        }
    }
    d
}

/// The raygen entry (reordering at the coherence point) and the compute entry of one body
/// trace the same paths. Camera rays hit the same triangles in every pixel. The rest agrees
/// up to the floating-point differences left between the two stages under `-fp-mode
/// precise`: Slang leaves `lerp`, `dot`, `cross`, `normalize` and matrix products
/// undecorated, and the driver evaluates some product chains differently per stage, so a
/// few shadow rays and bounces flip near edges.
#[test]
fn raygen_and_compute_entries_trace_the_same_paths() {
    let d = device();
    if !d.caps().ray_dispatch {
        log::warn!(
            "skipped: {} has no ray-dispatch pipelines",
            d.caps().device_name
        );
        return;
    }
    let scene = forest();
    let stage = |entry| ShaderStage::new(&paths::MODULE, entry, SpecData::default()).unwrap();
    let compute = d
        .create_compute_pipeline(&stage("compute_main"), "parity compute")
        .unwrap();
    let rays = d
        .create_ray_dispatch_pipeline(&stage("raygen_main"), "parity raygen")
        .unwrap();
    let runs: Vec<(Run, Run)> = [0, 2]
        .into_iter()
        .map(|b| (Run::new(&scene, b), Run::new(&scene, b)))
        .collect();
    let [gx, gy, _] = compute.workgroup();
    let mut list = render().command_list().unwrap();
    for (c, r) in &runs {
        list.dispatch(&compute, c.root.gpu, [WIDTH / gx, HEIGHT / gy, 1]);
        list.dispatch_rays(&rays, r.root.gpu, [WIDTH, HEIGHT, 1]);
    }
    list.barrier(
        Stages::COMPUTE | Stages::RAY,
        Stages::HOST,
        Hazards::empty(),
    );
    run(list);

    let pixels = (WIDTH * HEIGHT) as usize;
    let sky = runs[1]
        .0
        .hash
        .as_slice()
        .unwrap()
        .iter()
        .filter(|&&h| h == sky_hash())
        .count();
    let primary = compare(&runs[0].0, &runs[0].1);
    let bounced = compare(&runs[1].0, &runs[1].1);
    log::info!(
        "entry parity over {pixels} pixels ({sky} see the sky): primary {primary:?}, two bounces {bounced:?}"
    );
    assert!(
        sky > pixels / 100 && sky < pixels * 9 / 10,
        "the view should hold both sky and geometry ({sky} sky pixels)"
    );
    assert_eq!(primary.paths, 0, "primary paths differ");
    assert!(
        primary.radiance < pixels / 10_000,
        "{} primary paths differ in radiance",
        primary.radiance
    );
    assert!(
        bounced.paths < pixels / 100,
        "{} bounced paths differ",
        bounced.paths
    );
    assert!(
        bounced.radiance < pixels / 1000,
        "{} equal bounced paths differ in radiance",
        bounced.radiance
    );
    assert_clean();
}
