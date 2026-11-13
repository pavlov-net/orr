//! The shadow product on a real ray-query candidate loop. The unbent product that froxels,
//! particle lighting and media apply take: dielectric slabs and water against Fresnel and Beer's
//! law for receivers above, below and inside them, and nested interiors, each face's Fresnel
//! against the medium around it. A path vertex's shadow rays end at solid glass. Presence layers
//! by their opacity, the candidate limit, particles by what their opacity lets through, and a
//! smoke volume's medium proxy against a fine march of its cooked grid; only rays with the
//! medium mask bit see proxies.

use lighting_suite::bed::{
    Bed, DEPTH_TEXTURED_SHEET, DEPTH_TEXTURED_SLAB, FLOOR, Item, PLUME, PRESENCE_VEIL,
    PRESENCE_VEIL_OPACITY, Particle, ParticleLook, SHEET_DEPTH, SHEET_THICKNESS, SLAB_THICKNESS,
    SUN, Spec, TINTED_SLAB, WATER_BOX, fresnel, plume_volume, water_transmission,
};
use lighting_suite::shaders::{proxy_rays, shadow_probe};
use lighting_suite::{assert_clean, pipeline, readback, upload};
use orr_gpu::{Allocation, Hazards, Stages};
use orr_lighting::Light;
use orr_math::{DVec3, Mat3, Quat, Vec3};
use orr_pack::volume::GridSampler;
use orr_scene::{MediumProxy, MediumProxyId, Opacity, ProxyDensity};

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

/// `MATERIAL_NONE`.
const NONE: u32 = u32::MAX;

/// `shadow.slang`'s `SHADOW_PATH` and `SHADOW_UNBENT`.
const PATH: u32 = 0;
const UNBENT: u32 = 2;

/// Traces one unbent shadow ray from each of `origins` along unit `dir` through `copies`.
fn trace(copies: Vec<Item>, origins: &[Vec3], dir: Vec3) -> Traced {
    let mut bed = Bed::with(spec(copies));
    trace_on(&mut bed, origins, dir, 0.0, 1)
}

/// The mean over `frames` frames of the unbent shadow ray from each of `origins` along unit
/// `dir` to `t_max` m (0: infinity) through `bed`'s scene, and the rays past the candidate limit.
fn trace_on(bed: &mut Bed, origins: &[Vec3], dir: Vec3, t_max: f32, frames: u32) -> Traced {
    trace_inside(bed, origins, dir, t_max, frames, NONE, UNBENT)
}

/// [`trace_on`] in shadow mode `mode`, from inside material `inside`'s interior, or none for
/// [`NONE`].
fn trace_inside(
    bed: &mut Bed,
    origins: &[Vec3],
    dir: Vec3,
    t_max: f32,
    frames: u32,
    inside: u32,
    mode: u32,
) -> Traced {
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
            dir: dir.extend(t_max).to_array(),
            count: n as u32,
            add: u32::from(f.index > 0),
            inside,
            mode,
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
    let through = (1.0 - fresnel(1.5, mu)).powi(2) * 0.5f64.powf(path / 0.3);
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

/// OpenPBR's `transmission_depth` sets "the extinction coefficient": a texture of it reads
/// at the hits of a shadow ray as at a path's, so half the depth takes twice the extinction.
#[test]
fn a_textured_depth_shadows_as_its_texels_say() {
    let sun = SUN.direction_to_light.normalize();
    let mid = Vec3::new(0.0, 1.0, 0.0);
    let copies = vec![at(DEPTH_TEXTURED_SLAB, mid)];
    let below = trace(copies, &grid(mid, [0.4, 0.3], 0.7 / sun.y, sun), sun);
    let mu = f64::from(sun.y);
    let depth = f64::from(SLAB_THICKNESS) * 128.0 / 255.0;
    let path = f64::from(SLAB_THICKNESS) / mu;
    let through = (1.0 - fresnel(1.5, mu)).powi(2) * 0.5f64.powf(path / depth);
    assert_all("below the slab", &below.transmittance, [through; 3], 2e-3);
    assert_clean();
}

/// Each textured interior on a ray takes its own distance at its own extinction: a mean over the
/// two would put the slab's length at the sheet's sigma.
#[test]
fn a_textured_depth_of_each_interior_stays_its_own() {
    let sun = SUN.direction_to_light.normalize();
    let mid = Vec3::new(0.0, 1.0, 0.0);
    let copies = vec![
        at(DEPTH_TEXTURED_SLAB, mid),
        at(DEPTH_TEXTURED_SHEET, mid + sun * 0.6),
    ];
    let below = trace(copies, &grid(mid, [0.4, 0.3], 0.7 / sun.y, sun), sun);
    let mu = f64::from(sun.y);
    let texel = 128.0 / 255.0;
    let slab = 1.0 / texel;
    let sheet = f64::from(SHEET_THICKNESS) / (f64::from(SHEET_DEPTH) * texel);
    let through = (1.0 - fresnel(1.5, mu)).powi(4) * 0.5f64.powf((slab + sheet) / mu);
    assert_all("below both", &below.transmittance, [through; 3], 2e-3);
    assert_clean();
}

/// A ray from inside a textured interior no list names knows the medium only where it leaves it,
/// so it reads the texels there.
#[test]
fn a_textured_depth_reads_at_the_face_a_ray_leaves_by() {
    let sun = SUN.direction_to_light.normalize();
    let mid = Vec3::new(0.0, 1.0, 0.0);
    let top = mid + Vec3::Y * (0.5 * SLAB_THICKNESS);
    let inside = 0.1 / sun.y;
    let copies = vec![at(DEPTH_TEXTURED_SLAB, mid)];
    let got = trace(copies, &grid(top, [0.2, 0.15], inside, sun), sun);
    let depth = f64::from(SLAB_THICKNESS) * 128.0 / 255.0;
    let through = (1.0 - fresnel(1.5, f64::from(sun.y))) * 0.5f64.powf(f64::from(inside) / depth);
    assert_all("inside the slab", &got.transmittance, [through; 3], 2e-3);
    assert_clean();
}

/// A light inside the slab reaches a receiver above it through one face: Beer's law over the part
/// of the ray inside only, and the radiance scale between the two ends, which the paraxial
/// distances through the slab give.
#[test]
fn a_light_inside_a_slab_takes_beer_over_the_part_of_the_ray_inside() {
    let mid = Vec3::new(0.0, 1.0, 0.0);
    let (above, inside) = (0.4f32, 0.5 * SLAB_THICKNESS);
    let top = mid + Vec3::Y * inside;
    let mut bed = Bed::with(spec(vec![at(TINTED_SLAB, mid)]));
    let origins = grid(top + Vec3::Y * above, [0.4, 0.3], 0.0, Vec3::NEG_Y);
    let got = trace_on(&mut bed, &origins, Vec3::NEG_Y, above + inside, 1);
    let (a, d) = (f64::from(above), f64::from(inside));
    let scale = ((a + d) / (1.5 * (a + d / 1.5))).powi(2);
    let wanted = (1.0 - fresnel(1.5, 1.0)) * 0.5f64.powf(d / f64::from(SLAB_THICKNESS)) * scale;
    assert_all("a light inside", &got.transmittance, [wanted; 3], 2e-3);
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
    // The box spans y from 0 to 4 and x and z from -3 to 3.
    let copies = vec![at(WATER_BOX, Vec3::new(0.0, 2.0, 0.0))];
    let surface = Vec3::new(0.0, 4.0, 0.0);
    for depth in [1.0f32, 3.0] {
        let inside = trace(
            copies.clone(),
            &grid(surface, [0.5, 0.45], depth / sun.y, sun),
            sun,
        );
        let wanted = water_transmission().map(|c| c.powf(f64::from(depth) / mu) * enter);
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
    let wanted = water_transmission().map(|c| c.powf(4.0 / mu) * enter * enter);
    assert_all("under the box", &under.transmittance, wanted, 2e-3);
    let above = trace(
        copies,
        &grid(surface + Vec3::Y * 0.5, [0.5, 0.45], 0.0, sun),
        sun,
    );
    assert_all("above the box", &above.transmittance, [1.0; 3], 0.0);
    assert_clean();
}

/// Rays straight up inside a water box charge each stretch to the innermost interior around it,
/// once, whether or not the list at their origin names the water: from a slab in the water, the
/// slab and then the water; from the water under the slab, the water, the slab and the water. A
/// ray ending in the water it starts in takes the water when the list names it, with no surface
/// to show it. The slab's faces take Fresnel against the water around them.
#[test]
fn nested_interiors_shadow_each_stretch_of_a_ray_once() {
    let transmission = water_transmission();
    let (slab, top) = (2.0f32, 4.0f32);
    let thick = f64::from(SLAB_THICKNESS);
    let [glass, wet] = [1.5 / 1.33, 1.33].map(|n| 1.0 - fresnel(n, 1.0));
    let mut bed = Bed::with(spec(vec![
        at(WATER_BOX, Vec3::new(0.0, 2.0, 0.0)),
        at(TINTED_SLAB, Vec3::new(0.0, slab, 0.0)),
    ]));
    let water = bed.material(WATER_BOX).0;
    let through = |in_water: f64, in_slab: f64, faces: i32| {
        transmission
            .map(|c| c.powf(in_water) * 0.5f64.powf(in_slab / thick) * glass.powi(faces) * wet)
    };
    let row = |y: f32| grid(Vec3::new(0.0, y, 0.0), [0.4, 0.3], 0.0, Vec3::Y);
    for inside in [NONE, water] {
        let from_slab = trace_inside(&mut bed, &row(slab), Vec3::Y, 0.0, 1, inside, UNBENT);
        let wanted = through(f64::from(top - slab) - thick * 0.5, thick * 0.5, 1);
        assert_all("in the slab", &from_slab.transmittance, wanted, 2e-3);
        let under = trace_inside(&mut bed, &row(1.0), Vec3::Y, 0.0, 1, inside, UNBENT);
        let wanted = through(f64::from(top - 1.0) - thick, thick, 2);
        assert_all("under the slab", &under.transmittance, wanted, 2e-3);
    }
    let held = trace_inside(&mut bed, &row(0.5), Vec3::Y, 0.5, 1, water, UNBENT);
    let wanted = transmission.map(|c| c.powf(0.5));
    assert_all("within the water", &held.transmittance, wanted, 2e-3);
    assert_clean();
}

/// A path vertex's shadow rays end at solid glass, where light bends, from outside it and from
/// inside it alike, and still take the Beer transmittance of the interior they stay in.
#[test]
fn path_shadow_rays_end_at_solid_glass() {
    let sun = SUN.direction_to_light.normalize();
    let mid = Vec3::new(0.0, 1.0, 0.0);
    let mut bed = Bed::with(spec(vec![at(TINTED_SLAB, mid)]));
    let slab = bed.material(TINTED_SLAB).0;
    let below = trace_inside(
        &mut bed,
        &grid(mid, [0.4, 0.3], 0.7 / sun.y, sun),
        sun,
        0.0,
        1,
        NONE,
        PATH,
    );
    assert_all("below the slab", &below.transmittance, [0.0; 3], 0.0);
    let top = mid + Vec3::Y * (0.5 * SLAB_THICKNESS);
    let inside = trace_inside(
        &mut bed,
        &grid(top, [0.2, 0.15], 0.1 / sun.y, sun),
        sun,
        0.0,
        1,
        slab,
        PATH,
    );
    assert_all("inside the slab", &inside.transmittance, [0.0; 3], 0.0);
    let held = trace_inside(
        &mut bed,
        &grid(top, [0.2, 0.15], 0.1, Vec3::Y),
        Vec3::Y,
        0.05,
        1,
        slab,
        PATH,
    );
    let wanted = 0.5f64.powf(0.05 / f64::from(SLAB_THICKNESS));
    assert_all("within the slab", &held.transmittance, [wanted; 3], 2e-3);
    assert_clean();
}

/// A sun 15 degrees above the horizon, past the critical angle from inside the water, still
/// lights a receiver in it, by Fresnel where the light enters. The shadow ray stands for a path
/// that bends at the surface.
#[test]
fn a_low_sun_reaches_a_receiver_under_water() {
    let sun = Vec3::new(0.966, 0.259, 0.0).normalize();
    let mu = f64::from(sun.y);
    let enter = 1.0 - fresnel(1.33, mu);
    let copies = vec![at(WATER_BOX, Vec3::new(0.0, 2.0, 0.0))];
    let depth = 0.5f32;
    let inside = trace(
        copies,
        &grid(Vec3::new(0.0, 4.0, 0.0), [0.5, 0.45], depth / sun.y, sun),
        sun,
    );
    let wanted = water_transmission().map(|c| c.powf(f64::from(depth) / mu) * enter);
    assert_all("under a low sun", &inside.transmittance, wanted, 2e-3);
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
        let copies = Vec::from_fn(veils, |k| {
            at(PRESENCE_VEIL, Vec3::new(0.0, 0.5 + 0.1 * k as f32, 0.0))
        });
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
    let traced = trace_on(&mut bed, &origins, sun, 0.0, FRAMES);
    let (info, pages) = plume_volume();
    let s = GridSampler::new(&info, 0, &pages).unwrap();
    let mut shadowed = 0;
    let mut worst = (0.0f64, String::new());
    for (r, rays) in receivers
        .iter()
        .zip(traced.transmittance.as_chunks::<RAYS>().0)
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
    let origins: Vec<Vec3> = Vec::from_fn(256, |k| {
        Vec3::new(
            (k % 16) as f32 * 0.4 - 3.0,
            6.0,
            (k / 16) as f32 * 0.4 - 3.0,
        )
    });
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
            assert_eq!(s.proxy, id.0);
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

/// Sunlight through one particle square of opacity 0.3, and through two behind each other,
/// keeps 1 - 0.3 of itself per particle.
#[test]
fn particles_shadow_by_one_minus_their_opacity() {
    let opacity = 0.3;
    let look = ParticleLook {
        albedo: [0.5; 3],
        emission: [0.0; 3],
        opacity,
        anisotropy: 0.0,
    };
    let origins: Vec<Vec3> = Vec::from_fn(16, |k| Vec3::new(-0.75 + 0.1 * k as f32, 1.0, -0.5));
    for count in [1, 2] {
        let mut bed = Bed::with(spec(vec![at(FLOOR, Vec3::ZERO)]));
        let particles: Vec<Particle> = (0..count)
            .map(|k| Particle {
                center: Vec3::new(0.0, 2.0, 0.5 * k as f32),
                size: 3.0,
            })
            .collect();
        bed.set_particles(&particles, look, Opacity::Particle);
        let t = trace_on(&mut bed, &origins, SUN.direction_to_light, 0.0, 1);
        let wanted = (1.0 - f64::from(opacity)).powi(count);
        assert_all(
            &format!("{count} particles"),
            &t.transmittance,
            [wanted; 3],
            2e-3,
        );
    }
    assert_clean();
}
