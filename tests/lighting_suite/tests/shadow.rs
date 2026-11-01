//! The shadow product on a real ray-query candidate loop: dielectric slabs and water against
//! Fresnel and Beer's law for receivers above, below and inside them, presence layers by
//! their opacity, and the candidate limit.

use lighting_suite::bed::{
    Bed, Item, PRESENCE_VEIL, PRESENCE_VEIL_OPACITY, SLAB_THICKNESS, SUN, Spec, TINTED_SLAB,
    WATER_BOX, fresnel, water_transmission,
};
use lighting_suite::shaders::shadow_probe;
use lighting_suite::{assert_clean, pipeline, readback, upload};
use orr_gpu::{Allocation, Hazards, Stages};
use orr_lighting::Light;
use orr_math::{Mat3, Vec3};

fn at(model: &'static str, at: Vec3) -> Item {
    Item {
        model,
        linear: Mat3::IDENTITY,
        at,
    }
}

/// What the probe found per ray, and the rays past the candidate limit.
struct Traced {
    transmittance: Vec<[f32; 3]>,
    overflows: u32,
}

/// Traces one shadow ray from each of `origins` along unit `dir` through `copies`.
fn trace(copies: Vec<Item>, origins: &[Vec3], dir: Vec3) -> Traced {
    let spec = Spec {
        copies,
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(0.0, 3.0, 12.0),
        target: Vec3::new(0.0, 1.0, 0.0),
        fov_y: 1.0,
        focus: None,
    };
    let mut bed = Bed::with(spec);
    let probe = pipeline(&shadow_probe::MODULE, "main");
    let o: Vec<[f32; 4]> = origins.iter().map(|p| p.extend(0.0).to_array()).collect();
    let o = upload(&o, "shadow probe origins");
    let n = origins.len() as u64;
    let out: Allocation<[f32; 4]> = readback(n, "shadow probe transmittance");
    let mut overflows: Allocation<u32> = readback(1, "shadow probe overflows");
    overflows.as_mut_slice().unwrap()[0] = 0;
    bed.accumulate(1, 0, &mut |list, f| {
        let pass = f.pass();
        let root = f.put(&shadow_probe::Root {
            scene: pass.scene,
            view: pass.view,
            origins: o.gpu(),
            transmittance: out.gpu(),
            overflows: overflows.gpu(),
            _pad0: 0,
            dir: dir.extend(0.0).to_array(),
            count: n as u32,
            _pad1: 0,
            _pad2: 0,
        });
        list.dispatch(&probe, root, [n.div_ceil(64) as u32, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    });
    Traced {
        transmittance: out
            .as_slice()
            .unwrap()
            .iter()
            .map(|t| [t[0], t[1], t[2]])
            .collect(),
        overflows: overflows.as_slice().unwrap()[0],
    }
}

/// Origins `back` meters before the points of a `side` x `side` grid of half-widths `half`
/// around `center` in its horizontal plane, along `dir`.
fn grid(center: Vec3, half: [f32; 2], back: f32, dir: Vec3) -> Vec<Vec3> {
    const SIDE: u32 = 8;
    let mut v = Vec::new();
    for j in 0..SIDE {
        for i in 0..SIDE {
            let u = (i as f32 + 0.5) / SIDE as f32 * 2.0 - 1.0;
            let w = (j as f32 + 0.5) / SIDE as f32 * 2.0 - 1.0;
            v.push(center + Vec3::new(u * half[0], 0.0, w * half[1]) - dir * back);
        }
    }
    v
}

fn assert_all(what: &str, got: &[[f32; 3]], wanted: [f64; 3], tolerance: f64) {
    for t in got {
        for c in 0..3 {
            let e = (f64::from(t[c]) - wanted[c]).abs();
            assert!(
                e <= tolerance * wanted[c].max(1e-3),
                "{what}: {t:?}, wanted {wanted:?}"
            );
        }
    }
}

/// Sunlight through the tinted slab reaches a receiver below it through both faces' Fresnel
/// transmission and Beer's law over the unbent path inside; a receiver above it is unshadowed.
#[test]
fn a_slab_shadows_receivers_below_it_by_fresnel_and_beer_law() {
    let sun = SUN.direction_to_light.normalize();
    let mid = Vec3::new(0.0, 1.0, 0.0);
    let copies = vec![at(TINTED_SLAB, mid)];
    let below = trace(
        copies.clone(),
        &grid(mid, [0.4, 0.3], 0.7 / sun.y, sun),
        sun,
    );
    let mu = f64::from(sun.y);
    let path = f64::from(SLAB_THICKNESS) / mu;
    let through =
        (1.0 - fresnel(1.5, mu)) * (1.0 - fresnel(1.0 / 1.5, mu)) * 0.5f64.powf(path / 0.3);
    assert_all("below the slab", &below.transmittance, [through; 3], 2e-3);
    let above = trace(
        copies,
        &grid(mid + Vec3::Y * 0.5, [0.4, 0.3], 0.0, sun),
        sun,
    );
    assert_all("above the slab", &above.transmittance, [1.0; 3], 0.0);
    assert_eq!(below.overflows + above.overflows, 0);
    assert_clean();
}

/// Receivers inside a water box take its Beer's law to the surface along the sun and
/// Fresnel where the light enters; receivers under the box take both faces and its whole
/// depth; receivers above it are unshadowed.
#[test]
fn water_shadows_receivers_inside_and_under_it_by_their_depth() {
    let sun = SUN.direction_to_light.normalize();
    let mu = f64::from(sun.y);
    let enter = 1.0 - fresnel(1.33, mu);
    let leave = 1.0 - fresnel(1.0 / 1.33, mu);
    // The box spans y from 0 to 4 and x and z from -3 to 3.
    let copies = vec![at(WATER_BOX, Vec3::new(0.0, 2.0, 0.0))];
    let surface = Vec3::new(0.0, 4.0, 0.0);
    for depth in [1.0f32, 3.0] {
        let inside = trace(
            copies.clone(),
            &grid(surface, [0.5, 0.45], depth / sun.y, sun),
            sun,
        );
        let wanted = water_transmission().map(|c| c.powf(f64::from(depth) / mu) * leave);
        assert_all(
            &format!("{depth} m deep"),
            &inside.transmittance,
            wanted,
            2e-3,
        );
    }
    let under = trace(
        copies.clone(),
        &grid(surface, [0.5, 0.45], 4.5 / sun.y, sun),
        sun,
    );
    let wanted = water_transmission().map(|c| c.powf(4.0 / mu) * enter * leave);
    assert_all("under the box", &under.transmittance, wanted, 2e-3);
    let above = trace(
        copies,
        &grid(surface + Vec3::Y * 0.5, [0.5, 0.45], 0.0, sun),
        sun,
    );
    assert_all("above the box", &above.transmittance, [1.0; 3], 0.0);
    assert_clean();
}

/// Stacked presence veils each pass 1 - opacity; past 16 non-opaque candidates a ray keeps
/// the product so far and counts an overflow.
#[test]
fn presence_layers_multiply_up_to_the_candidate_limit() {
    let pass = 1.0 - f64::from(PRESENCE_VEIL_OPACITY);
    let origins = grid(Vec3::ZERO, [0.8, 0.7], 0.0, Vec3::Y);
    for (veils, wanted, overflows) in [
        (3, pass.powi(3), 0),
        (16, pass.powi(16), 0),
        (20, pass.powi(16), origins.len() as u32),
    ] {
        let copies = (0..veils)
            .map(|k| at(PRESENCE_VEIL, Vec3::new(0.0, 0.5 + 0.1 * k as f32, 0.0)))
            .collect();
        let t = trace(copies, &origins, Vec3::Y);
        assert_all(
            &format!("{veils} veils"),
            &t.transmittance,
            [wanted; 3],
            2e-3,
        );
        assert_eq!(t.overflows, overflows, "{veils} veils");
    }
    assert_clean();
}
