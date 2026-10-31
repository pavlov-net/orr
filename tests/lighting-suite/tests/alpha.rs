//! Alpha in traversal: shadow and bounce rays through an alpha-tested quad with and without
//! opacity micromaps, through linked coverage, and through fractional linked coverage, on a
//! real ray-query candidate loop; and the shadows they cast in the path kernel.

use lighting_suite::bed::{
    Accumulated, Analytic, BARE_LEAF, Bed, Item, LEAF, Paths, RINGS_LEAF, SUN, Spec, VEIL,
};
use lighting_suite::shaders::alpha_probe;
use lighting_suite::{assert_clean, device, linked, pipeline, readback, require, upload};
use orr_gpu::{Allocation, Hazards, KernelSet, Stages};
use orr_lighting::{Light, LightingDebugView, PathEntry, ReferenceSettings, ViewLighting};
use orr_math::{Mat3, Vec2, Vec3};
use orr_scene::RendererDesc;

/// Rays a side aimed across each quad.
const GRID: u32 = 192;
/// The rings texture's texels a side.
const TEXELS: f32 = 128.0;

fn at(model: &'static str, x: f32) -> Item {
    Item {
        model,
        linear: Mat3::IDENTITY,
        at: Vec3::new(x, 1.1, 0.0),
    }
}

/// Where the rings of the opacity texture put a uv: the position across a ring, 0 to 1,
/// with the inner 0.6 opaque.
fn ring_phase(uv: Vec2) -> f32 {
    ((uv - Vec2::splat(0.5 + 0.5 / TEXELS)).length() * 8.0).fract()
}

/// Whether a point `margin` texels or more from every ring edge is opaque; `None` nearer.
fn sure(uv: Vec2, margin: f32) -> Option<bool> {
    let m = margin / TEXELS * 8.0;
    let p = ring_phase(uv);
    if p >= m && p <= 0.6 - m {
        Some(true)
    } else if p >= 0.6 + m && p <= 1.0 - m {
        Some(false)
    } else {
        None
    }
}

/// One quad's rays: their uv and what the probe found.
struct Rays {
    uv: Vec<Vec2>,
    stopped: Vec<u32>,
    /// Alpha test calls under 4-state and forced 2-state.
    calls: [u32; 2],
}

impl Rays {
    const VISIBILITY: u32 = 1;
    const FOUR_STATE: u32 = 2;
    const TWO_STATE: u32 = 4;

    fn count(&self) -> u32 {
        self.uv.len() as u32
    }

    fn stopped(&self, bit: u32) -> usize {
        self.stopped.iter().filter(|&&s| s & bit != 0).count()
    }

    /// Checks `bit` against the texture on rays `margin` texels from a ring edge; returns
    /// how many were checked opaque and clear.
    fn check(&self, what: &str, bit: u32, margin: f32) -> (usize, usize) {
        let (mut opaque, mut clear) = (0, 0);
        for (uv, &s) in self.uv.iter().zip(&self.stopped) {
            let Some(want) = sure(*uv, margin) else {
                continue;
            };
            assert_eq!(s & bit != 0, want, "{what}: ray at uv {uv}");
            if want {
                opaque += 1;
            } else {
                clear += 1;
            }
        }
        assert!(
            opaque > 500 && clear > 500,
            "{what}: {opaque} opaque and {clear} clear rays checked"
        );
        (opaque, clear)
    }
}

/// Traces the probe's rays toward the sun across each of `models`, placed side by side.
fn trace(models: &[&'static str]) -> Vec<Rays> {
    let xs: Vec<f32> = (0..models.len()).map(|i| 3.0 * i as f32).collect();
    let spec = Spec {
        copies: models.iter().zip(&xs).map(|(&m, &x)| at(m, x)).collect(),
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(1.5 * xs[xs.len() - 1], 1.5, 6.0),
        target: Vec3::new(1.5 * xs[xs.len() - 1], 1.1, 0.0),
        fov_y: 1.0,
        focus: None,
    };
    let dir = SUN.direction_to_light.normalize();
    let mut bed = Bed::with(spec);
    let probe = pipeline(&alpha_probe::MODULE, "main");
    let mut uvs = Vec::new();
    let mut origins = Vec::new();
    for &x in &xs {
        let (mut uv, mut o) = (Vec::new(), Vec::new());
        for j in 0..GRID {
            for i in 0..GRID {
                // Inside the quad, clear of its border.
                let lx = -0.97 + 1.94 * (i as f32 + 0.5) / GRID as f32;
                let ly = -0.97 + 1.94 * (j as f32 + 0.37) / GRID as f32;
                let q = Vec3::new(x + lx, 1.1 + ly, 0.0);
                uv.push(Vec2::new(0.5 * (lx + 1.0), 0.5 * (1.0 - ly)));
                o.push((q - dir * 0.1).extend(0.0).to_array());
            }
        }
        uvs.push(uv);
        origins.push(upload(&o, "alpha probe origins"));
    }
    let n = u64::from(GRID * GRID);
    let stopped: Vec<Allocation<u32>> = xs.iter().map(|_| readback(n, "stopped")).collect();
    let mut calls: Vec<Allocation<u32>> = xs.iter().map(|_| readback(2, "calls")).collect();
    for c in &mut calls {
        c.as_mut_slice().unwrap().fill(0);
    }
    bed.accumulate(1, 0, &mut |list, f| {
        let pass = f.pass();
        for k in 0..xs.len() {
            let root = f.put(&alpha_probe::Root {
                scene: pass.scene,
                view: pass.view,
                origins: origins[k].gpu,
                stopped: stopped[k].gpu,
                calls: calls[k].gpu,
                _pad0: 0,
                dir: dir.extend(0.0).to_array(),
                count: n as u32,
                _pad1: 0,
                _pad2: 0,
            });
            list.dispatch(&probe, root, [n.div_ceil(64) as u32, 1, 1]);
        }
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    });
    uvs.into_iter()
        .zip(stopped.iter().zip(&calls))
        .map(|(uv, (s, c))| {
            let c = c.as_slice().unwrap();
            Rays {
                uv,
                stopped: s.as_slice().unwrap().to_vec(),
                calls: [c[0], c[1]],
            }
        })
        .collect()
}

fn micromaps() -> bool {
    let caps = device().caps();
    caps.omm && caps.cluster.omm_in_clusters
}

#[test]
fn alpha_candidates_stop_rays_where_the_texture_is_opaque_with_and_without_micromaps() {
    let rays = trace(&[LEAF, BARE_LEAF, RINGS_LEAF]);
    let [leaf, bare, rings] = &rays[..] else {
        unreachable!()
    };
    for (what, r) in [("leaf", leaf), ("bare leaf", bare), ("rings leaf", rings)] {
        assert_eq!(
            r.stopped(Rays::VISIBILITY),
            r.stopped(Rays::FOUR_STATE),
            "{what}: visibility against its own 4-state trace"
        );
        r.check(&format!("{what} 4-state"), Rays::FOUR_STATE, 1.5);
    }
    // Without micromaps every ray asks the alpha test about the quad once, in both modes.
    for (what, r) in [("bare leaf", bare), ("rings leaf", rings)] {
        r.check(&format!("{what} 2-state"), Rays::TWO_STATE, 1.5);
        for (mode, calls) in ["4-state", "2-state"].iter().zip(r.calls) {
            assert!(
                calls >= r.count() && calls <= r.count() + r.count() / 100,
                "{what} {mode}: {calls} alpha tests for {} rays",
                r.count()
            );
        }
    }
    // Micromap states are conservative: 4-state rays stop exactly where the alpha test
    // alone stops them.
    let differ = leaf
        .stopped
        .iter()
        .zip(&bare.stopped)
        .filter(|&(a, b)| (a ^ b) & Rays::FOUR_STATE != 0)
        .count();
    assert_eq!(
        differ, 0,
        "4-state rays with and without micromaps differ on {differ} rays"
    );
    if require(
        micromaps(),
        "no opacity micromaps in cluster acceleration structures",
    ) {
        // 4-state asks only about unknown micro-triangles (most of them on rings this fine);
        // forced 2-state never asks and keeps to the texture away from the micro-triangles
        // crossing a ring edge.
        assert!(
            leaf.calls[0] < leaf.count() * 9 / 10,
            "leaf: {} 4-state alpha tests for {} rays",
            leaf.calls[0],
            leaf.count()
        );
        assert_eq!(leaf.calls[1], 0, "leaf: forced 2-state ran alpha tests");
        leaf.check("leaf 2-state", Rays::TWO_STATE, 3.0);
    }
    assert_clean();
}

#[test]
fn fractional_linked_coverage_lets_that_share_of_rays_through() {
    let rays = trace(&[VEIL]);
    let veil = &rays[0];
    let n = f64::from(veil.count());
    for (mode, bit) in [("4-state", Rays::FOUR_STATE), ("2-state", Rays::TWO_STATE)] {
        let share = veil.stopped(bit) as f64 / n;
        assert!(
            (share - 0.25).abs() < 0.02,
            "{mode}: {share} of rays stopped by a quarter coverage"
        );
    }
    assert_clean();
}

fn direct(bed: &mut Bed, paths: &Paths, seed: u32) -> Accumulated {
    bed.accumulate(32, seed, &mut |list, f| paths.record(list, f))
}

fn luminance(p: &[f32; 4]) -> f64 {
    0.2627 * f64::from(p[0]) + 0.6780 * f64::from(p[1]) + 0.0593 * f64::from(p[2])
}

/// The alpha quad scene with `model` for its quad.
fn quad_scene(model: &'static str) -> Spec {
    let mut spec = Analytic::AlphaQuad.spec();
    spec.copies[1].model = model;
    spec
}

/// FLIP of `a` against `b`, and of `b` against `noise` (the same scene at another seed),
/// shown at an exposure putting `b`'s mean at 0.18.
fn against_noise(a: &Accumulated, b: &Accumulated, noise: &Accumulated) -> (f32, f32) {
    let mean = b.mean.iter().map(luminance).sum::<f64>() / b.mean.len() as f64;
    let exposure = (0.18 / mean) as f32;
    (a.flip(b, exposure).mean, noise.flip(b, exposure).mean)
}

fn direct_paths(kernels: &KernelSet) -> Paths {
    Paths::new(
        kernels,
        PathEntry::Fastest,
        ViewLighting {
            max_bounces: 1,
            reference: Some(ReferenceSettings::default()),
            debug_view: LightingDebugView::DirectOnly,
            ..ViewLighting::default()
        },
    )
}

#[test]
fn an_alpha_quad_casts_the_same_shadow_with_and_without_micromaps() {
    let paths = direct_paths(&KernelSet::default());
    let bare = direct(&mut Bed::with(quad_scene(BARE_LEAF)), &paths, 1);
    let noise = direct(&mut Bed::with(quad_scene(BARE_LEAF)), &paths, 2);
    let leaf = direct(&mut Bed::with(quad_scene(LEAF)), &paths, 1);
    let (e, floor) = against_noise(&leaf, &bare, &noise);
    assert!(
        e <= floor,
        "FLIP {e} with against without micromaps, {floor} between seeds"
    );
    assert_clean();
}

#[test]
fn linked_coverage_casts_the_shadow_of_the_equivalent_alpha_texture() {
    let paths = direct_paths(&linked::KERNELS);
    let renderer = RendererDesc {
        kernels: &linked::KERNELS,
        ..RendererDesc::default()
    };
    let bed = |model| Bed::with_renderer(quad_scene(model), &renderer);
    let leaf = direct(&mut bed(BARE_LEAF), &paths, 1);
    let noise = direct(&mut bed(BARE_LEAF), &paths, 2);
    let rings = direct(&mut bed(RINGS_LEAF), &paths, 1);
    let (e, floor) = against_noise(&rings, &leaf, &noise);
    assert!(
        e <= floor,
        "FLIP {e} with linked coverage against the texture, {floor} between seeds"
    );
    assert_clean();
}
