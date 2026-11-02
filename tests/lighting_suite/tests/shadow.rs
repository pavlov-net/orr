//! The shadow product on a real ray-query candidate loop: dielectric slabs and water against
//! Fresnel and Beer's law for receivers above, below and inside them, presence layers by
//! their opacity, the candidate limit, and a smoke volume's medium proxy against a fine
//! march of its cooked grid; only rays with the medium mask bit see proxies.

use lighting_suite::bed::{
    Bed, FLOOR, Item, PLUME, PRESENCE_VEIL, PRESENCE_VEIL_OPACITY, SLAB_THICKNESS, SUN, Spec,
    TINTED_SLAB, WATER_BOX, fresnel, plume_volume, water_transmission,
};
use lighting_suite::shaders::{proxy_rays, shadow_probe};
use lighting_suite::{assert_clean, pipeline, readback, upload};
use orr_gpu::{Allocation, Hazards, Stages};
use orr_lighting::Light;
use orr_math::{DVec3, Mat3, Quat, Vec3};
use orr_pack::volume::GridSampler;
use orr_scene::{MediumProxy, MediumProxyId, ProxyDensity, SceneDesc};

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

fn spec(copies: Vec<Item>) -> Spec {
    Spec {
        copies,
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(0.0, 3.0, 12.0),
        target: Vec3::new(0.0, 1.0, 0.0),
        fov_y: 1.0,
        focus: None,
    }
}

/// Traces one shadow ray from each of `origins` along unit `dir` through `copies`.
fn trace(copies: Vec<Item>, origins: &[Vec3], dir: Vec3) -> Traced {
    let mut bed = Bed::with(spec(copies));
    trace_on(&mut bed, origins, dir, 1)
}

/// The mean over `frames` frames of the shadow ray from each of `origins` along unit `dir`
/// through `bed`'s scene, and the rays past the candidate limit.
fn trace_on(bed: &mut Bed, origins: &[Vec3], dir: Vec3, frames: u32) -> Traced {
    let probe = pipeline(&shadow_probe::MODULE, "main");
    let o: Vec<[f32; 4]> = origins.iter().map(|p| p.extend(0.0).to_array()).collect();
    let o = upload(&o, "shadow probe origins");
    let n = origins.len() as u64;
    let out: Allocation<[f32; 4]> = readback(n, "shadow probe transmittance");
    let mut overflows: Allocation<u32> = readback(1, "shadow probe overflows");
    overflows.as_mut_slice().unwrap()[0] = 0;
    bed.accumulate(frames, 0, &mut |list, f| {
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
            add: u32::from(f.index > 0),
            _pad2: 0,
        });
        list.dispatch(&probe, root, [n.div_ceil(64) as u32, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    });
    let mean = 1.0 / frames as f32;
    Traced {
        transmittance: out
            .as_slice()
            .unwrap()
            .iter()
            .map(|t| [t[0] * mean, t[1] * mean, t[2] * mean])
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

/// The plume's volume space turned 0.4 rad about y, scaled 1.25 and moved, with an extinction
/// that differs per channel.
fn plume_proxy(bed: &Bed) -> MediumProxy {
    let v = bed.volume(PLUME);
    MediumProxy {
        position: DVec3::new(0.3, 0.0, -0.2),
        rotation: Quat::from_rotation_y(0.4),
        scale: 1.25,
        density: v.density(v.grid("density").unwrap()),
        extinction: Vec3::new(1.6, 1.2, 0.9),
        albedo: Vec3::splat(0.9),
        anisotropy: 0.5,
    }
}

/// Transmittance from `origin` along unit `dir` through the proxy, by a 1 mm march of the
/// cooked grid.
fn plume_reference(s: &GridSampler, proxy: &MediumProxy, origin: Vec3, dir: Vec3) -> [f64; 3] {
    let to_volume = proxy.rotation.inverse();
    let at = |t: f32| {
        let w = origin + dir * t - proxy.position.as_vec3();
        (to_volume * w / proxy.scale).to_array()
    };
    let dt = 1e-3;
    let tau: f64 = (0..12_000)
        .map(|k| f64::from(s.sample(at((k as f32 + 0.5) * dt)).max(0.0)))
        .sum::<f64>()
        * f64::from(dt);
    proxy
        .extinction
        .to_array()
        .map(|e| (-f64::from(e) * tau).exp())
}

/// Sunlight through a smoke plume's medium proxy reaches the ground under it, on average,
/// within 5% of a fine march of the cooked density, and in full beside it.
#[test]
fn a_smoke_volume_shadows_as_a_fine_march_of_its_grid() {
    const RAYS: usize = 8;
    const FRAMES: u32 = 16;
    let sun = SUN.direction_to_light.normalize();
    let mut bed = Bed::with(spec(vec![at(FLOOR, Vec3::ZERO)]));
    let proxy = plume_proxy(&bed);
    bed.set_medium_proxy(MediumProxyId(0), Some(proxy));
    // Receivers on the ground whose sun rays cross the plume 2 m up, and beyond its edge.
    let foot = proxy.position.as_vec3() + Vec3::Y * 0.02 - sun * (2.0 / sun.y);
    let receivers = grid(Vec3::new(foot.x, 0.02, foot.z), [1.6, 1.6], 0.0, sun);
    // Each receiver traces RAYS rays a few micrometers apart, whose origins seed different
    // samples.
    let origins: Vec<Vec3> = receivers
        .iter()
        .flat_map(|r| (0..RAYS).map(move |k| *r + Vec3::X * (k as f32 * 2e-6)))
        .collect();
    let traced = trace_on(&mut bed, &origins, sun, FRAMES);
    let (info, pages) = plume_volume();
    let s = GridSampler::new(&info, 0, &pages).unwrap();
    let mut shadowed = 0;
    let mut worst = (0.0f64, String::new());
    for (r, rays) in receivers
        .iter()
        .zip(traced.transmittance.chunks_exact(RAYS))
    {
        let want = plume_reference(&s, &proxy, *r, sun);
        if want[0] < 0.9 {
            shadowed += 1;
        }
        for c in 0..3 {
            let mean = rays.iter().map(|t| f64::from(t[c])).sum::<f64>() / RAYS as f64;
            let e = (mean - want[c]).abs() / want[c].max(0.05);
            if e > worst.0 {
                worst = (
                    e,
                    format!("receiver {r}: channel {c} {mean}, wanted {want:?}"),
                );
            }
        }
    }
    assert!(worst.0 <= 0.05, "error {:.4} at {}", worst.0, worst.1);
    assert!(
        shadowed >= receivers.len() / 4,
        "{shadowed} receivers in the shadow"
    );
    assert!(shadowed < receivers.len(), "every receiver in the shadow");
    assert_eq!(traced.overflows, 0);
    assert_clean();
}

/// A box proxy turned and scaled: rays traced with the medium mask bit meet it once, as
/// class 4 at its PTLAS index, entering its cube where its placement says; rays with any
/// other mask never do, and once cleared no ray does.
#[test]
fn only_rays_with_the_medium_mask_see_medium_proxies() {
    let mut bed = Bed::with(spec(vec![at(FLOOR, Vec3::ZERO)]));
    let half = Vec3::new(1.0, 0.5, 0.75);
    let proxy = MediumProxy {
        position: DVec3::new(0.5, 2.0, -0.3),
        rotation: Quat::from_rotation_y(0.6),
        scale: 2.0,
        density: ProxyDensity::Uniform { half_extent: half },
        extinction: Vec3::ONE,
        albedo: Vec3::ONE,
        anisotropy: 0.0,
    };
    let id = MediumProxyId(3);
    bed.set_medium_proxy(id, Some(proxy));
    let d = SceneDesc::default();
    let proxy_base = d.instance_capacity + d.part_capacity + d.producer_capacity;
    let origins: Vec<Vec3> = (0..256)
        .map(|k| {
            Vec3::new(
                (k % 16) as f32 * 0.4 - 3.0,
                6.0,
                (k / 16) as f32 * 0.4 - 3.0,
            )
        })
        .collect();
    let probe = pipeline(&proxy_rays::MODULE, "main");
    let o: Vec<[f32; 4]> = origins.iter().map(|p| p.extend(0.0).to_array()).collect();
    let o = upload(&o, "proxy ray origins");
    let n = origins.len() as u64;
    let run = |bed: &mut Bed, mask: u32| {
        let out: Allocation<proxy_rays::Seen> = readback(n, "proxy rays");
        bed.accumulate(1, 0, &mut |list, f| {
            let root = f.put(&proxy_rays::Root {
                scene: f.pass().scene,
                origins: o.gpu(),
                seen: out.gpu(),
                dir: [0.0, -1.0, 0.0, 0.0],
                count: n as u32,
                mask,
            });
            list.dispatch(&probe, root, [n.div_ceil(64) as u32, 1, 1]);
            list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        });
        out.as_slice().unwrap().to_vec()
    };
    let to_object = |p: Vec3| {
        let v = proxy.rotation.inverse() * (p - proxy.position.as_vec3()) / proxy.scale;
        v / half
    };
    let (medium, opaque, particle, dielectric) = (0x04, 0x01, 0x02, 0x80);
    let mut inside = 0;
    for mask in [medium, 0xff, opaque, particle, dielectric, 0xff & !medium] {
        let seen = run(&mut bed, mask);
        for (p, s) in origins.iter().zip(&seen) {
            let x = to_object(*p);
            let within = x.x.abs() < 1.0 && x.z.abs() < 1.0;
            if (x.x.abs() - 1.0).abs() < 1e-3 || (x.z.abs() - 1.0).abs() < 1e-3 {
                continue;
            }
            if mask & medium == 0 || !within {
                assert_eq!(s.count, 0, "mask {mask:#x}, ray at {p}");
                continue;
            }
            inside += usize::from(mask == medium);
            assert_eq!(s.count, 1, "mask {mask:#x}, ray at {p}");
            assert_eq!(s.instance - proxy_base, id.0);
            assert_eq!((s.class_id, s.proxy_flags), (4, 1));
            let entry = Vec3::new(s.entry[0], s.entry[1], s.entry[2]);
            let wanted = Vec3::new(x.x, 1.0, x.z);
            assert!(
                s.entry[3] == 1.0 && (entry - wanted).abs().max_element() < 1e-4,
                "ray at {p} entered at {entry}, wanted {wanted}"
            );
        }
    }
    assert!(inside >= 20, "{inside} rays cross the box");
    bed.set_medium_proxy(id, None);
    let seen = run(&mut bed, 0xff);
    assert!(
        seen.iter().all(|s| s.count == 0),
        "a cleared proxy is traced"
    );
    assert_clean();
}
