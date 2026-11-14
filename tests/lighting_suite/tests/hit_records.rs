//! Rays through a scene's hit records: the same rays traced inline through orr_lighting's
//! candidate handlers and through a ray tracing pipeline whose any-hit and intersection shaders
//! the scene's hit table selects (`orr_scene.candidates`), on copies opaque, alpha-tested with and
//! without micromaps, presence layers, solid glass, a thin window, a mixed-class model, particles,
//! strands and a medium proxy; and those shaders' bodies against their register budget.

use std::sync::OnceLock;

use lighting_suite::bed::{
    BARE_LEAF, Bed, FLOOR, Item, LEAF, MASKED_PANE, MIXED_SLAB, PRESENCE_VEIL, Particle,
    ParticleLook, SUN, Spec, StrandRow, TINTED_SLAB,
};
use lighting_suite::shaders::types::{HitRaysRoot, PathOut, RayIn, ShadowOut};
use lighting_suite::shaders::{hit_inline, hit_records};
use lighting_suite::{assert_clean, device, pipeline, readback, require, upload};
use orr_gpu::{
    Allocation, CompileThreads, GpuPtr, Hazards, HitGroup, RayPipeline, RayPipelineDesc,
    RayPipelineFlags, ShaderRecord, ShaderStage, SpecData, Stages,
};
use orr_lighting::Light;
use orr_math::{Mat3, Quat, Vec3};
use orr_scene::{HitGroups, MediumProxy, MediumProxyId, Opacity, ProxyDensity};

// orr_shaders.trace masks and classes, orr_scene.candidates policies, modes and flags.
const MASK_OPAQUE: u32 = 0x01;
const MASK_PARTICLE: u32 = 0x02;
const MASK_MEDIUM: u32 = 0x04;
const MASK_DIELECTRIC: u32 = 0x80;
const SHADOW_MASK: u32 = MASK_OPAQUE | MASK_PARTICLE | MASK_MEDIUM | MASK_DIELECTRIC;
const CLASS_DIELECTRIC: u32 = 1;
const COVER_ALPHA: u32 = 0;
const COVER_LAYERS: u32 = 1;
const COVER_STACK: u32 = 2;
const SHADOW_PATH: u32 = 0;
const SHADOW_PHOTON: u32 = 1;
const SHADOW_UNBENT: u32 = 2;
const SHADOW_OVERFLOW: u32 = 1 << 15;
const SHADOW_SOLID: u32 = 1 << 17;
const SHADOW_CROSSES: u32 = 1 << 18;
const RAY_FLAG_FORCE_NON_OPAQUE: u32 = 0x02;
const RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE: u32 = 0x400;

// The suite pipeline's stages and groups.
const STAGES: [&str; 7] = [
    "pipeline_paths",
    "pipeline_shadows",
    "miss",
    "cover_path_hit",
    "cover_shadow_hit",
    "box_medium_hit",
    "proxy_shadow_hit",
];
const RAYGEN_PATHS: u32 = 0;
const RAYGEN_SHADOWS: u32 = 1;

fn pipeline_flags() -> RayPipelineFlags {
    let caps = device().caps();
    let mut flags = RayPipelineFlags::CLUSTERS;
    if caps.omm {
        flags |= RayPipelineFlags::OPACITY_MICROMAPS;
    }
    if caps.lss {
        flags |= RayPipelineFlags::SWEPT_SPHERES;
    }
    flags
}

/// The pipeline, and the groups the scene's records name: one for the process.
fn ray_pipeline() -> &'static (RayPipeline, HitGroups) {
    static PIPELINE: OnceLock<(RayPipeline, HitGroups)> = OnceLock::new();
    PIPELINE.get_or_init(create_pipeline)
}

fn create_pipeline() -> (RayPipeline, HitGroups) {
    let stages: Vec<ShaderStage<'_>> = STAGES
        .iter()
        .map(|e| ShaderStage::new(&hit_records::MODULE, e, SpecData::default()).unwrap())
        .collect();
    let desc = RayPipelineDesc {
        stages: &stages,
        raygens: &[ShaderRecord::new(0), ShaderRecord::new(1)],
        misses: &[ShaderRecord::new(2)],
        hit_groups: &[
            HitGroup {
                any_hit: Some(3),
                ..HitGroup::default()
            },
            HitGroup {
                any_hit: Some(4),
                ..HitGroup::default()
            },
            HitGroup {
                intersection: Some(5),
                any_hit: Some(6),
                closest_hit: None,
            },
        ],
        callables: &[],
        max_recursion: 1,
        flags: pipeline_flags(),
        threads: CompileThreads::default(),
    };
    let p = device().create_ray_pipeline(&desc, "hit records").unwrap();
    let groups = HitGroups {
        path: Some(0),
        producer_path: Some(0),
        strand_path: Some(0),
        shadow: Some(1),
        medium_shadow: Some(2),
        ..HitGroups::default()
    };
    (p, groups)
}

/// A small generator, so the rays repeat run to run.
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }

    fn pick<T: Copy>(&mut self, of: &[T]) -> T {
        of[self.next() as usize % of.len()]
    }

    fn direction(&mut self) -> Vec3 {
        let z = 2.0 * self.unit() - 1.0;
        let a = std::f32::consts::TAU * self.unit();
        let r = (1.0 - z * z).sqrt();
        Vec3::new(r * a.cos(), z, r * a.sin())
    }
}

/// Rays from points around `center` within `extent` in random directions, each a path ray
/// under a random policy and mask or a shadow ray in a random mode, with micromaps forced to two
/// states or not.
fn rays_around(d: &mut Draw, center: Vec3, extent: Vec3, n: usize, out: &mut Vec<RayIn>) {
    for _ in 0..n {
        let o = center
            + extent
                * Vec3::new(
                    2.0 * d.unit() - 1.0,
                    2.0 * d.unit() - 1.0,
                    2.0 * d.unit() - 1.0,
                );
        let dir = d.direction();
        out.push(ray(d, o, dir, 8.0));
    }
}

fn ray(d: &mut Draw, origin: Vec3, dir: Vec3, t_max: f32) -> RayIn {
    let policy = d.pick(&[COVER_ALPHA, COVER_LAYERS, COVER_STACK]);
    let path_mask = if policy == COVER_ALPHA {
        d.pick(&[MASK_OPAQUE, MASK_DIELECTRIC, MASK_OPAQUE | MASK_DIELECTRIC])
    } else {
        MASK_OPAQUE | MASK_DIELECTRIC
    };
    let mode = d.pick(&[SHADOW_PATH, SHADOW_PHOTON, SHADOW_UNBENT]);
    // The two traces `surface_visibility` makes in the unbent mode, and the other modes' one.
    let (shadow_mask, shadow_flags) = if mode == SHADOW_UNBENT && d.unit() < 0.5 {
        (MASK_DIELECTRIC, RAY_FLAG_FORCE_NON_OPAQUE)
    } else if mode == SHADOW_UNBENT {
        (SHADOW_MASK & !MASK_DIELECTRIC, 0)
    } else {
        (SHADOW_MASK, 0)
    };
    let two_state = if d.unit() < 0.5 {
        RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE
    } else {
        0
    };
    // A path's shadow ray forced non-opaque meets solid glass as a fading copy shows it.
    let shadow_flags = if mode == SHADOW_PATH && d.unit() < 0.25 {
        RAY_FLAG_FORCE_NON_OPAQUE
    } else {
        shadow_flags
    };
    RayIn {
        origin: origin.to_array(),
        t_max,
        dir: dir.normalize().to_array(),
        mask: path_mask | shadow_mask << 8,
        policy,
        presence: d.pick(&[0, 1, 2]),
        seed: d.next(),
        mode,
        flags: two_state | shadow_flags,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

/// What one launch of each side found.
struct Traced {
    paths: Vec<PathOut>,
    shadows: Vec<ShadowOut>,
}

/// Traces `rays` inline and through `p` over the scene's hit table, at the bed's next frame.
fn trace_both(bed: &mut Bed, p: &RayPipeline, groups: &HitGroups, rays: &[RayIn]) -> [Traced; 2] {
    let n = rays.len() as u64;
    let input = upload(rays, "hit record rays");
    let out = |name| (readback::<PathOut>(n, name), readback::<ShadowOut>(n, name));
    let (inline_paths, inline_shadows) = out("inline");
    let (piped_paths, piped_shadows) = out("pipeline");
    let paths = pipeline(&hit_inline::MODULE, "inline_paths");
    let shadows = pipeline(&hit_inline::MODULE, "inline_shadows");
    bed.accumulate_hits(p, groups, &mut |list, f| {
        let pass = f.pass();
        let hits = f.hits.expect("the bed's hit table");
        let root = |paths: &Allocation<PathOut>, shadows: &Allocation<ShadowOut>| HitRaysRoot {
            scene: pass.scene,
            view: pass.view,
            rays: input.gpu(),
            paths: paths.gpu(),
            shadows: shadows.gpu(),
            stand_ins: GpuPtr::NULL,
            count: n as u32,
            depth: 1,
            cone_width: 1e-3,
            cone_spread: 2e-3,
        };
        list.barrier(
            Stages::ALL,
            Stages::COMPUTE | Stages::RAY,
            Hazards::ACCEL_STRUCT,
        );
        let r = root(&inline_paths, &inline_shadows);
        list.dispatch_inline(&paths, &r, [n.div_ceil(64) as u32, 1, 1]);
        list.dispatch_inline(&shadows, &r, [n.div_ceil(64) as u32, 1, 1]);
        let r = root(&piped_paths, &piped_shadows);
        list.trace_rays(p, RAYGEN_PATHS, &hits, &r, [n as u32, 1, 1]);
        list.trace_rays(p, RAYGEN_SHADOWS, &hits, &r, [n as u32, 1, 1]);
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::HOST,
            Hazards::empty(),
        );
    });
    let read = |p: &Allocation<PathOut>, s: &Allocation<ShadowOut>| Traced {
        paths: p.as_slice().unwrap().to_vec(),
        shadows: s.as_slice().unwrap().to_vec(),
    };
    [
        read(&inline_paths, &inline_shadows),
        read(&piped_paths, &piped_shadows),
    ]
}

/// What the comparison saw, to check the rays reached every case.
#[derive(Default, Debug)]
struct Seen {
    hits: usize,
    hit_classes: [usize; 6],
    overflows: usize,
    solid: usize,
    crossed: usize,
    products: usize,
}

/// Checks the pipeline's results against the inline ones ray by ray.
fn compare(rays: &[RayIn], [inline, piped]: &[Traced; 2]) -> Seen {
    let mut seen = Seen::default();
    for (i, r) in rays.iter().enumerate() {
        let (a, b) = (&inline.paths[i], &piped.paths[i]);
        assert_eq!(
            (
                a.hit,
                a.t.to_bits(),
                a.instance,
                a.primitive,
                a.geometry,
                a.class_id
            ),
            (
                b.hit,
                b.t.to_bits(),
                b.instance,
                b.primitive,
                b.geometry,
                b.class_id
            ),
            "path ray {i} {r:?}: inline hit, pipeline hit"
        );
        if a.hit != 0 {
            seen.hits += 1;
            seen.hit_classes[a.class_id as usize] += 1;
        }
        let (a, b) = (&inline.shadows[i], &piped.shadows[i]);
        let what = format!("shadow ray {i} {r:?}: inline {a:?}, pipeline {b:?}");
        assert_eq!(
            a.flags & SHADOW_OVERFLOW,
            b.flags & SHADOW_OVERFLOW,
            "{what}"
        );
        assert_eq!(a.flags & SHADOW_SOLID, b.flags & SHADOW_SOLID, "{what}");
        assert_eq!(a.committed_class, b.committed_class, "{what}");
        if a.committed_class != u32::MAX || a.flags & SHADOW_SOLID != 0 {
            assert_eq!(a.committed_t.to_bits(), b.committed_t.to_bits(), "{what}");
        }
        // The pipeline passes what bends light to the raygen's walk.
        let crosses = b.flags & SHADOW_CROSSES != 0;
        if r.mode == SHADOW_UNBENT {
            assert_eq!(a.crossed > 0, crosses, "{what}");
        }
        if r.mode == SHADOW_PHOTON {
            assert!(!crosses, "{what}");
        }
        seen.overflows += usize::from(a.flags & SHADOW_OVERFLOW != 0);
        seen.solid += usize::from(a.flags & SHADOW_SOLID != 0);
        seen.crossed += usize::from(crosses);
        if crosses && r.mode == SHADOW_PATH {
            continue;
        }
        // The same candidates in another order multiply in another order.
        for c in 0..3 {
            let (x, y) = (a.product[c], b.product[c]);
            assert!((x - y).abs() <= 1e-5 * x.abs().max(y.abs()), "{what}");
        }
        if a.product.iter().any(|&x| x > 0.0 && x < 1.0) {
            seen.products += 1;
        }
    }
    seen
}

/// A copy of `model` at `at`, turned by `linear`.
fn item(model: &'static str, at: Vec3, linear: Mat3) -> Item {
    Item { model, linear, at }
}

/// Path and shadow rays through the scene's hit records find what the inline handlers find:
/// the same closest hit of every policy and mask, with the record's class the trace's; and the
/// same shadow product, overflow past `SHADOW_CANDIDATES`, solid glass and committed hit, with
/// what the inline handling bends left to the raygen's walk. On opaque, alpha-tested (with and
/// without micromaps), presence, solid and thin dielectric and mixed-class copies, a stack of
/// presence layers deeper than the candidate cap, particles and a medium proxy.
#[test]
fn rays_through_the_hit_records_meet_what_inline_traces_meet() {
    if !require(device().caps().ray_dispatch, "ray tracing pipelines") {
        return;
    }
    let up = Mat3::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let mut copies = vec![
        item(FLOOR, Vec3::ZERO, Mat3::IDENTITY),
        item(LEAF, Vec3::new(0.0, 1.1, 0.0), Mat3::IDENTITY),
        item(BARE_LEAF, Vec3::new(3.0, 1.1, 0.0), Mat3::IDENTITY),
        item(MASKED_PANE, Vec3::new(6.0, 2.5, 0.0), Mat3::IDENTITY),
        item(TINTED_SLAB, Vec3::new(9.0, 1.1, 0.0), Mat3::IDENTITY),
        item(MIXED_SLAB, Vec3::new(12.0, 1.1, 0.0), Mat3::IDENTITY),
        item(PRESENCE_VEIL, Vec3::new(15.0, 1.1, 0.0), up),
    ];
    // Deeper than the candidates a shadow ray counts.
    let stack: Vec<f32> = (0..20).map(|k| 0.5 + 0.1 * k as f32).collect();
    for &y in &stack {
        copies.push(item(PRESENCE_VEIL, Vec3::new(18.0, y, 0.0), Mat3::IDENTITY));
    }
    let spec = Spec {
        copies,
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(12.0, 6.0, 16.0),
        target: Vec3::new(12.0, 1.0, 0.0),
        fov_y: 1.0,
        focus: None,
    };
    let mut bed = Bed::with(spec);
    let particles: Vec<Particle> = (0..64)
        .map(|k| Particle {
            center: Vec3::new(
                21.0 + 0.25 * (k % 8) as f32,
                0.6 + 0.25 * (k / 8) as f32,
                0.0,
            ),
            size: 0.2,
        })
        .collect();
    bed.set_particles(
        &particles,
        ParticleLook {
            albedo: [0.5; 3],
            emission: [0.0; 3],
            opacity: 0.4,
            anisotropy: 0.0,
        },
        Opacity::Particle,
    );
    bed.set_medium_proxy(
        MediumProxyId(0),
        Some(MediumProxy {
            position: Vec3::new(24.0, 1.1, 0.0).as_dvec3(),
            rotation: Quat::from_rotation_y(0.4),
            scale: 1.0,
            density: ProxyDensity::Uniform {
                half_extent: Vec3::new(1.0, 0.8, 0.6),
            },
            extinction: Vec3::new(0.5, 0.8, 1.1),
            albedo: Vec3::splat(0.5),
            anisotropy: 0.0,
        }),
    );
    let mut d = Draw(0x5eed);
    let mut rays = Vec::new();
    for x in [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 21.0, 24.0] {
        rays_around(
            &mut d,
            Vec3::new(x + 1.0, 1.3, 0.0),
            Vec3::new(2.0, 1.5, 1.5),
            6000,
            &mut rays,
        );
    }
    // Down through the stack of layers.
    for k in 0..2000 {
        let o = Vec3::new(17.2 + 1.6 * d.unit(), 4.0, -0.8 + 1.6 * d.unit());
        let lean = 0.02 * d.unit();
        let mut r = ray(&mut d, o, Vec3::new(lean, -1.0, 0.0), 8.0);
        if k % 2 == 0 {
            r.mode = SHADOW_PATH;
            r.mask = r.mask & 0xff | SHADOW_MASK << 8;
            r.flags &= RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
        }
        rays.push(r);
    }
    let (p, groups) = ray_pipeline();
    // The first frame builds the scene's structures; the second traces them.
    bed.accumulate(1, 0, &mut |_, _| {});
    let traced = trace_both(&mut bed, p, groups, &rays);
    let seen = compare(&rays, &traced);
    assert!(seen.hits > 10_000, "{seen:?}");
    assert!(
        seen.hit_classes[0] > 1000 && seen.hit_classes[CLASS_DIELECTRIC as usize] > 100,
        "{seen:?}"
    );
    assert!(
        seen.overflows > 100 && seen.solid > 100 && seen.crossed > 100 && seen.products > 1000,
        "{seen:?}"
    );
    assert_clean();
}

/// Path rays through a row of strands, opaque swept spheres whose records run no shader, hit
/// what inline traces hit.
#[test]
fn strands_through_the_hit_records_match_inline_traces() {
    if !require(device().caps().ray_dispatch, "ray tracing pipelines") {
        return;
    }
    let spec = Spec {
        copies: vec![item(FLOOR, Vec3::ZERO, Mat3::IDENTITY)],
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(0.0, 1.0, 4.0),
        target: Vec3::new(0.0, 0.5, 0.0),
        fov_y: 1.0,
        focus: None,
    };
    let mut bed = Bed::with(spec);
    bed.set_strands(StrandRow {
        count: 32,
        first: Vec3::new(-1.6, 0.0, 0.0),
        step: Vec3::new(0.1, 0.0, 0.0),
        height: 1.0,
        radius: 0.02,
        albedo: [0.5; 3],
    });
    let mut d = Draw(0x57a2d);
    let mut rays = Vec::new();
    rays_around(
        &mut d,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(2.0, 0.6, 0.5),
        8000,
        &mut rays,
    );
    let (p, groups) = ray_pipeline();
    bed.accumulate(1, 0, &mut |_, _| {});
    let traced = trace_both(&mut bed, p, groups, &rays);
    let seen = compare(&rays, &traced);
    assert!(seen.hits > 1000, "{seen:?}");
    assert_clean();
}

/// The medium proxy's intersection and any-hit shaders fit their budget as compute stand-ins: 64
/// registers, no local or shared memory. The standard coverage's any-hits do not yet: the hit
/// reconstruction their presence resolve shares with the inline handlers holds more.
#[test]
fn proxy_shaders_fit_their_register_budget() {
    for entry in ["box_medium_stand_in", "proxy_shadow_stand_in"] {
        let Some(stats) = pipeline(&hit_records::MODULE, entry).statistics() else {
            return;
        };
        assert!(
            stats.registers <= 64 && stats.local_bytes == 0 && stats.shared_bytes == 0,
            "{entry}: {stats:?}"
        );
    }
}
