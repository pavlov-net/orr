//! Caustic photons: the grid's irradiance under a slab against Fresnel and Beer's law, under a
//! ball lens against its 1D map from impact parameter to radius, and the reference's photons
//! against BSDF rays alone, by Welch's test, through a glass sphere from a light photons own and
//! from the environment, which they do not. A mirror ball's caustic on a floor at a voxel boundary
//! of every grid level against the reference's map, and the sun off two mirrors, and off one
//! mirror of a producer whose other mirrors the sun does not reach, against its illuminance.

use lighting_suite::assert_clean;
use lighting_suite::bed::{
    Accumulated, BLACK_SPHERE, Bed, CLEAR_POOL, EMITTER, FINE_GLASS_SPHERE, FROSTED_POOL,
    FloorDecal, GLASS_PANE, GLASS_SPHERE, Item, MIRROR, MIRROR_FIELD, MIRROR_FIELD_FACING,
    MIRROR_SPHERE, POOL_IOR, Particle, Paths, ROUGH_POOL, SKY, SLAB_THICKNESS, SUN, Spec,
    TINTED_SLAB, WATER_BOX, WATER_SIGMA, WHITE_FLOOR, WHITE_FLOOR_ALBEDO, WIDE_SLAB, fresnel,
    pixel_of, reference,
};
use lighting_suite::welch::{assert_mean, assert_same_mean};
use orr_gpu::KernelSet;
use orr_lighting::{
    CausticSettings, CausticStats, DirectionalLight, DiscProfile, Light, LightingDebugView,
    SceneLightingSettings, ViewLighting,
};
use orr_math::{Affine3, DVec2, DVec3, Mat3, Vec2, Vec3};

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), settings)
}

fn render_paths(bed: &mut Bed, paths: &Paths, frames: u32, seed: u32) -> Accumulated {
    bed.accumulate(frames, seed, &mut |list, f| paths.record(list, f))
}

fn at(model: &'static str, at: Vec3) -> Item {
    Item {
        model,
        linear: Mat3::IDENTITY,
        at,
    }
}

/// What the first vertex gathers of the grid, in real time.
fn gathered() -> ViewLighting {
    ViewLighting {
        debug_view: LightingDebugView::Caustics,
        ..ViewLighting::default()
    }
}

/// The first vertex's light in real time: its next-event estimation, what its BSDF ray finds and
/// what it gathers of the grid.
fn first_vertex() -> ViewLighting {
    ViewLighting {
        max_bounces: 0,
        ..ViewLighting::default()
    }
}

/// The white floor's irradiance where it shows radiance `radiance` by its diffuse albedo.
fn irradiance(radiance: f64) -> f64 {
    radiance * std::f64::consts::PI / f64::from(WHITE_FLOOR_ALBEDO)
}

/// The irradiance across the beam a point light of intensity `intensity` gives a point under a
/// flat interface of IOR `n` along the chain Snell's law gives, with the light `h1` m off the
/// interface's plane, the point `h2` m on the other side and `r` m apart along it, and the cosine
/// of the chain's angle at the point: Fresnel's transmittance over the area across the beam at
/// the point per solid angle it leaves the light in.
fn refracted_lamp(intensity: f64, n: f64, h1: f64, h2: f64, r: f64) -> (f64, f64) {
    let reach = |a: f64| h1 * a.tan() + h2 * (a.sin() / n).asin().tan();
    let (mut lo, mut hi) = (0.0, 0.5 * std::f64::consts::PI - 1e-9);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if reach(mid) < r {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = 0.5 * (lo + hi);
    let b = (a.sin() / n).asin();
    let spread = h1 / a.cos().powi(2) + h2 / b.cos().powi(2) * a.cos() / (n * b.cos());
    (
        intensity * (1.0 - fresnel(n, a.cos())) * a.sin() / (r * spread * b.cos()),
        b.cos(),
    )
}

/// A point light 0.3 m over a clear pool and 5.2 m aside from a patch of its floor lights the
/// patch along the chain that meets the pool's top at 86 degrees and leaves it under the water
/// within a degree of the critical angle, as the CPU's refracted beam says, with caustic photons
/// or without: the floor's walk solves for that chain on the generalized half vector, which stays
/// smooth there, where the refracted direction a shooting step compares turns past the critical
/// angle.
/// The view shows direct light alone: past it the real-time paths end in the world cache, which
/// brings back the floor's own light that the water's top reflects totally.
#[test]
fn a_grazing_lamp_lights_a_pool_floor_by_its_refracted_beam() {
    let (pool_top, pool_floor) = (1.5f64, 0.7f64);
    let (h1, r) = (0.3f64, 5.2f64);
    let intensity = 40.0f32;
    let (beam, cos_below) =
        refracted_lamp(f64::from(intensity), POOL_IOR, h1, pool_top - pool_floor, r);
    let wanted = beam * cos_below;
    for photons in [false, true] {
        let below_floor = Vec3::new(0.0, pool_floor as f32, 0.0);
        let spec = Spec {
            copies: vec![
                at(WHITE_FLOOR, below_floor),
                at(CLEAR_POOL, Vec3::new(0.0, (pool_top - 0.5) as f32, 0.0)),
            ],
            lights: vec![Light::Point {
                position: DVec3::new(r, pool_top + h1, 0.0),
                intensity: Vec3::splat(intensity),
                shaping: None,
                shadows: true,
            }],
            environment: None,
            eye: Vec3::new(-3.0, pool_floor as f32 + 0.5, 0.01),
            target: below_floor,
            fov_y: 0.3,
            focus: None,
        };
        let patch = pixel_of(&spec, below_floor);
        let mut bed = Bed::with(spec);
        if !photons {
            let no_photons = SceneLightingSettings {
                caustics: CausticSettings {
                    photons: 0,
                    ..bed.lighting.settings().caustics
                },
                ..*bed.lighting.settings()
            };
            bed.lighting.set_settings(&no_photons).unwrap();
        }
        let view = ViewLighting {
            camera_medium: Some(bed.material(CLEAR_POOL)),
            debug_view: LightingDebugView::DirectOnly,
            ..first_vertex()
        };
        let image = render_paths(&mut bed, &paths(view), 64, 1);
        for (c, &radiance) in image.around(patch, 4).iter().enumerate() {
            let e = irradiance(radiance);
            assert!(
                (e - wanted).abs() <= 0.04 * wanted,
                "photons {photons}, channel {c}: {e} lux on the floor, wanted {wanted}"
            );
        }
    }
    assert_clean();
}

/// A point lamp 1.5 m over a clear pool and 3 m aside from a patch of its floor, with a black ball
/// in the water across the patch's straight way to the lamp: the chain from the lamp to the patch,
/// which passes the ball, is the caustic photons' that deposit there, which the patch's estimate
/// leaves to them, or without photons its seeded walks', which start where the ball ends that
/// straight way. The patch takes it as the CPU's refracted beam says, with photons within what
/// their gather blurs across the edge of the floor where estimates walk.
#[test]
fn a_ball_across_a_floors_straight_way_to_a_lamp_passes_its_chain_on() {
    let (pool_top, pool_floor) = (1.5f64, 0.7f64);
    let (h1, r) = (1.5f64, 3.0f64);
    let intensity = 40.0f32;
    let (beam, cos_below) =
        refracted_lamp(f64::from(intensity), POOL_IOR, h1, pool_top - pool_floor, r);
    let wanted = beam * cos_below;
    let below_floor = Vec3::new(0.0, pool_floor as f32, 0.0);
    // On the straight way from the patch to the lamp just under the pool's top, 0.35 m from the
    // chain, which leaves the water 0.65 m along.
    let ball = Vec3::new(0.94, 1.42, 0.0);
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, below_floor),
            at(CLEAR_POOL, Vec3::new(0.0, (pool_top - 0.5) as f32, 0.0)),
            Item {
                model: BLACK_SPHERE,
                linear: Mat3::from_diagonal(Vec3::splat(0.06)),
                at: ball,
            },
        ],
        lights: vec![Light::Point {
            position: DVec3::new(r, pool_top + h1, 0.0),
            intensity: Vec3::splat(intensity),
            shaping: None,
            shadows: true,
        }],
        environment: None,
        // Near enough that the patch's pixels' straight ways all meet the ball.
        eye: Vec3::new(-1.41, pool_floor as f32 + 0.51, 0.01),
        target: below_floor,
        fov_y: 0.3,
        focus: None,
    };
    let patch = pixel_of(&spec, below_floor);
    for (photons, tolerance) in [(true, 0.15), (false, 0.04)] {
        let mut bed = Bed::with(spec.clone());
        if !photons {
            let no_photons = SceneLightingSettings {
                caustics: CausticSettings {
                    photons: 0,
                    ..bed.lighting.settings().caustics
                },
                ..*bed.lighting.settings()
            };
            bed.lighting.set_settings(&no_photons).unwrap();
        }
        let mut sum = [0.0; 3];
        for debug_view in [LightingDebugView::DirectOnly, LightingDebugView::Caustics] {
            let view = ViewLighting {
                camera_medium: Some(bed.material(CLEAR_POOL)),
                debug_view,
                ..first_vertex()
            };
            // The few photons the ball's shade on the floor gets a frame converge over these.
            let image = render_paths(&mut bed, &paths(view), 256, 1);
            for (s, radiance) in sum.iter_mut().zip(image.around(patch, 4)) {
                *s += radiance;
            }
        }
        for (c, &radiance) in sum.iter().enumerate() {
            let e = irradiance(radiance);
            assert!(
                (e - wanted).abs() <= tolerance * wanted,
                "photons {photons}, channel {c}: {e} lux on the floor, wanted {wanted}"
            );
        }
    }
    assert_clean();
}

/// The ball of `a_ball_across_a_floors_straight_way_to_a_lamp_passes_its_chain_on` across the
/// straight way from the pool's floor to the lamp: where that way meets the ball the photons
/// own the lamp's chain, and elsewhere the floor's estimate does, so a cell the gather reads
/// across that edge holds photons on one side only. Each patch of a strip across the edge takes
/// the chain whole, as the CPU's refracted beam says, its estimate taking the light through its
/// diffuse lobe where it would own it at a point the gather draws in the cell it read. The grid's
/// cells are coarse here, so its reads straddle the edge over several patches, and dense, so their
/// photons' noise lies well under the share of the light a read that straddles the edge moves. The
/// strip's light matches the reference's by Welch's test.
#[test]
fn a_strip_across_where_a_ball_meets_a_floors_way_to_a_lamp_takes_the_lamps_chain_whole() {
    let (pool_top, pool_floor) = (1.5f64, 0.7f64);
    let (h1, r) = (1.5f64, 3.0f64);
    let intensity = 40.0f32;
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, Vec3::new(0.0, pool_floor as f32, 0.0)),
            at(CLEAR_POOL, Vec3::new(0.0, (pool_top - 0.5) as f32, 0.0)),
            Item {
                model: BLACK_SPHERE,
                linear: Mat3::from_diagonal(Vec3::splat(0.06)),
                at: Vec3::new(0.94, 1.42, 0.0),
            },
        ],
        lights: vec![Light::Point {
            position: DVec3::new(r, pool_top + h1, 0.0),
            intensity: Vec3::splat(intensity),
            shaping: None,
            shadows: true,
        }],
        environment: None,
        eye: Vec3::new(-1.41, pool_floor as f32 + 0.51, 0.01),
        target: Vec3::new(0.0, pool_floor as f32, 0.0),
        fov_y: 0.3,
        focus: None,
    };
    // Along z from x = 0, where the straight way to the lamp leaves the ball about 0.087 m out.
    let strip: Vec<f64> = (2..9).map(|k| 0.025 * f64::from(k)).collect();
    let patches: Vec<Vec2> = strip
        .iter()
        .map(|&z| pixel_of(&spec, Vec3::new(0.0, pool_floor as f32, z as f32)))
        .collect();
    let mut bed = Bed::with(spec);
    let coarse = SceneLightingSettings {
        caustics: CausticSettings {
            photons: 1 << 20,
            cell_photons: 256.0,
            base_cell_log2: -5,
            ..bed.lighting.settings().caustics
        },
        ..*bed.lighting.settings()
    };
    bed.lighting.set_settings(&coarse).unwrap();
    let medium = Some(bed.material(CLEAR_POOL));
    let views = [LightingDebugView::DirectOnly, LightingDebugView::Caustics].map(|debug_view| {
        paths(ViewLighting {
            camera_medium: medium,
            debug_view,
            ..first_vertex()
        })
    });
    let converged = paths(ViewLighting {
        camera_medium: medium,
        ..reference(0)
    });
    // The strip's irradiance in one render of `frames` frames from `seed`: what the estimates
    // bring and what the gather does, apart, summed.
    let strip_light = |bed: &mut Bed, frames: u32, seed: u32| {
        let mut irradiances = vec![0.0; patches.len()];
        for view in &views {
            let image = render_paths(bed, view, frames, seed);
            for (e, &patch) in irradiances.iter_mut().zip(&patches) {
                *e += irradiance(image.around(patch, 2)[1]);
            }
        }
        irradiances
    };
    let sums = strip_light(&mut bed, 256, 1);
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    for k in 0..4 {
        ours.push(strip_light(&mut bed, 64, 10 + k).iter().sum::<f64>());
        let image = render_paths(&mut bed, &converged, 64, 20 + k);
        theirs.push(
            patches
                .iter()
                .map(|&p| irradiance(image.around(p, 2)[1]))
                .sum::<f64>(),
        );
    }
    for (e, &z) in sums.iter().zip(&strip) {
        let (beam, cos_below) = refracted_lamp(
            f64::from(intensity),
            POOL_IOR,
            h1,
            pool_top - pool_floor,
            (r * r + z * z).sqrt(),
        );
        let wanted = beam * cos_below;
        assert!(
            (e - wanted).abs() <= 0.07 * wanted,
            "{z} m across: {e} lux on the floor, wanted {wanted}"
        );
    }
    assert_same_mean("the strip against the reference", &ours, &theirs);
    assert_clean();
}

/// A point light outside a water box beyond its top edge lights a patch of the floor inside it along
/// two chains, one through the top and one through the side, which meet the patch from above and
/// from the side: the patch takes both, as a CPU sum of the two refracted beams says, each through
/// the water's absorption along its leg in it and the eye's way to the patch. Without caustic
/// photons the patch's estimate owns every chain to the light, the one its walk converges on and
/// the other by seeded walks; with them the estimate owns its own chain and the photons the other.
/// The views show direct light and the gathered photons alone: past them the real-time paths end
/// in the world cache, which brings back the floor's own light that the water's top reflects
/// totally.
#[test]
fn a_lamp_beyond_a_water_box_edge_lights_its_floor_through_two_faces() {
    let (lamp, patch_at) = (DVec3::new(3.5, 2.5, 0.0), DVec3::new(1.0, 0.0, 0.0));
    let intensity = 40.0f32;
    let (half_x, half_y) = (3.0f64, 2.0f64);
    let eye = Vec3::new(-1.0, 0.8, 0.3);
    let to_eye = eye.as_dvec3().distance(patch_at);
    let sigma = orr_color::rec709_to_rec2020(WATER_SIGMA.map(|s| f64::from(-s).exp()))
        .to_array()
        .map(|c| -c.ln());
    // Through the top: the lamp 0.5 m over it, the patch 2 m under it; through the side alike,
    // turned.
    let i = f64::from(intensity);
    let top = refracted_lamp(
        i,
        POOL_IOR,
        lamp.y - half_y,
        half_y - patch_at.y,
        lamp.x - patch_at.x,
    );
    let side = refracted_lamp(
        i,
        POOL_IOR,
        lamp.x - half_x,
        half_x - patch_at.x,
        lamp.y - patch_at.y,
    );
    // The legs in the water, and the floor's cosine: the top's chain comes down at its angle
    // below the top's normal, the side's at its angle off the side's.
    let wanted = |c: usize| {
        let top_leg = (half_y - patch_at.y) / top.1;
        let side_leg = (half_x - patch_at.x) / side.1;
        (top.0 * top.1 * (-sigma[c] * top_leg).exp()
            + side.0 * (1.0 - side.1 * side.1).sqrt() * (-sigma[c] * side_leg).exp())
            * (-sigma[c] * to_eye).exp()
    };
    for photons in [false, true] {
        let floor = Vec3::new(0.0, patch_at.y as f32, 0.0);
        let target = patch_at.as_vec3();
        let spec = Spec {
            copies: vec![at(WHITE_FLOOR, floor), at(WATER_BOX, Vec3::ZERO)],
            lights: vec![Light::Point {
                position: lamp,
                intensity: Vec3::splat(intensity),
                shaping: None,
                shadows: true,
            }],
            environment: None,
            eye,
            target,
            fov_y: 0.3,
            focus: None,
        };
        let patch = pixel_of(&spec, target);
        let mut bed = Bed::with(spec);
        if !photons {
            let no_photons = SceneLightingSettings {
                caustics: CausticSettings {
                    photons: 0,
                    ..bed.lighting.settings().caustics
                },
                ..*bed.lighting.settings()
            };
            bed.lighting.set_settings(&no_photons).unwrap();
        }
        let mut sum = [0.0; 3];
        for debug_view in [LightingDebugView::DirectOnly, LightingDebugView::Caustics] {
            let view = ViewLighting {
                camera_medium: Some(bed.material(WATER_BOX)),
                debug_view,
                ..first_vertex()
            };
            let image = render_paths(&mut bed, &paths(view), 128, 1);
            for (s, radiance) in sum.iter_mut().zip(image.around(patch, 4)) {
                *s += radiance;
            }
        }
        for (c, &radiance) in sum.iter().enumerate() {
            let e = irradiance(radiance);
            let w = wanted(c);
            assert!(
                (e - w).abs() <= 0.05 * w,
                "photons {photons}, channel {c}: {e} lux on the floor, wanted {w} (top {}, side {})",
                top.0,
                side.0
            );
        }
    }
    assert_clean();
}

/// A wall beside a pool sees, off the pool's rough top at a grazing angle, a lamp in a second pool
/// whose frosted top tilts toward it and which lies under the first top's plane. The reflection
/// is a drawn ray that starts a track at the second top, whose job toward the lamp asks whether an
/// estimate at the first top's vertex owns its chain. That vertex leaves its lights past its
/// surface to the checks of the ray that reached it and estimates none of them, so the job takes
/// the lamp's light. Without caustic photons a point light's estimate owns every chain where it
/// walks, so the wall would lose that light; it takes what it takes with photons, where the job
/// asks the same and the estimate owns its own chain alone.
#[test]
fn a_rough_reflection_that_starts_a_track_takes_the_lights_past_its_surface() {
    // The second pool's top tilts 26 degrees toward the wall; the wall's patch sees the first
    // top 1.5 m off and 0.3 m down, whose reflection meets the second top 0.5 m in from its edge,
    // and goes on through it toward the lamp, 0.2 m under the first top's plane.
    let tilt = 26.0f32.to_radians();
    let along = Vec3::new(tilt.cos(), tilt.sin(), 0.0);
    let up = Vec3::new(-tilt.sin(), tilt.cos(), 0.0);
    let edge = Vec3::new(1.6, 0.29, 0.0);
    let patch_at = Vec3::new(-1.99, 0.3, 0.0);
    let spec = Spec {
        copies: vec![
            Item {
                model: WHITE_FLOOR,
                linear: Mat3::from_rotation_z(-90.0f32.to_radians())
                    * Mat3::from_diagonal(Vec3::splat(0.4)),
                at: Vec3::new(-2.0, 0.3, 0.0),
            },
            Item {
                model: ROUGH_POOL,
                linear: Mat3::from_diagonal(Vec3::new(0.1, 1.0, 0.15)),
                at: Vec3::new(0.0, -0.5, 0.0),
            },
            Item {
                model: FROSTED_POOL,
                linear: Mat3::from_rotation_z(tilt)
                    * Mat3::from_diagonal(Vec3::new(0.25, 2.0, 0.15)),
                at: edge + along * 2.5 - up * 1.0,
            },
        ],
        lights: vec![Light::Point {
            position: DVec3::new(4.34, -0.2, 0.0),
            intensity: Vec3::splat(20.0),
            shaping: None,
            shadows: true,
        }],
        environment: None,
        eye: Vec3::new(-1.0, 0.6, 0.3),
        target: patch_at,
        fov_y: 0.3,
        focus: None,
    };
    let patch = pixel_of(&spec, patch_at);
    let [with_photons, without] = [true, false].map(|photons| {
        let mut bed = Bed::with(spec.clone());
        if !photons {
            let no_photons = SceneLightingSettings {
                caustics: CausticSettings {
                    photons: 0,
                    ..bed.lighting.settings().caustics
                },
                ..*bed.lighting.settings()
            };
            bed.lighting.set_settings(&no_photons).unwrap();
        }
        render_paths(&mut bed, &paths(first_vertex()), 256, 1).around(patch, 4)
    });
    for c in 0..3 {
        assert!(
            (without[c] - with_photons[c]).abs() <= 0.05 * with_photons[c],
            "channel {c}: the wall shows {} without photons, {} with them",
            without[c],
            with_photons[c]
        );
    }
    assert_clean();
}

/// A slope on a clear pool's floor turned 30 degrees away from a sun 10 degrees up faces away from
/// the sun's own direction but toward the steeper one the pool's top bends it to: it takes the sun
/// by that bent direction, Fresnel's transmittance and the beam's widening below the top, with
/// caustic photons or without. Its estimate draws the sun by the direction its walk leaves the
/// slope along, so a BSDF ray tracked from the slope through the top to the sun, or a photon that
/// deposits there, which an estimate's chain owns, leaves no light unowned.
/// The view shows direct light alone: past it the real-time paths end in the world cache, which
/// brings back the floor's own light that the water's top reflects totally.
#[test]
fn a_slope_facing_away_from_the_sun_under_water_takes_the_bent_sun() {
    let elevation = 10.0f32.to_radians();
    let direction_to_light = Vec3::new(elevation.cos(), elevation.sin(), 0.0);
    let sun = DirectionalLight {
        direction_to_light,
        ..SUN
    };
    // The floor's 10 m square at 1.5 m, so it lies within the pool.
    let linear =
        Mat3::from_rotation_z(30.0f32.to_radians()) * Mat3::from_diagonal(Vec3::splat(0.15));
    let normal = (linear * Vec3::Y).normalize().as_dvec3();
    let c = f64::from(direction_to_light.y);
    let across = f64::from(direction_to_light.x) / POOL_IOR;
    let bent = DVec3::new(across, (1.0 - across * across).sqrt(), 0.0);
    assert!(normal.dot(direction_to_light.as_dvec3()) < 0.0 && normal.dot(bent) > 0.0);
    let beam = (1.0 - fresnel(POOL_IOR, c)) * c / bent.y;
    let (pool_top, floor) = (1.5f32, 0.95f32);
    for photons in [false, true] {
        let at_floor = Vec3::new(0.0, floor, 0.0);
        let spec = Spec {
            copies: vec![
                Item {
                    model: WHITE_FLOOR,
                    linear,
                    at: at_floor,
                },
                at(CLEAR_POOL, Vec3::new(0.0, pool_top - 0.5, 0.0)),
            ],
            lights: vec![Light::Directional(sun)],
            environment: None,
            // In the water, in front of the slope.
            eye: Vec3::new(-1.2, floor + 0.25, 0.01),
            target: at_floor,
            fov_y: 0.3,
            focus: None,
        };
        let patch = pixel_of(&spec, at_floor);
        let mut bed = Bed::with(spec);
        if !photons {
            let no_photons = SceneLightingSettings {
                caustics: CausticSettings {
                    photons: 0,
                    ..bed.lighting.settings().caustics
                },
                ..*bed.lighting.settings()
            };
            bed.lighting.set_settings(&no_photons).unwrap();
        }
        let view = ViewLighting {
            camera_medium: Some(bed.material(CLEAR_POOL)),
            debug_view: LightingDebugView::DirectOnly,
            ..first_vertex()
        };
        let image = render_paths(&mut bed, &paths(view), 32, 1);
        for (c, &radiance) in image.around(patch, 4).iter().enumerate() {
            let e = irradiance(radiance);
            let wanted = f64::from(sun.illuminance[c]) * beam * normal.dot(bent);
            assert!(
                (e - wanted).abs() <= 0.04 * wanted,
                "photons {photons}, channel {c}: {e} lux on the slope, wanted {wanted}"
            );
        }
    }
    assert_clean();
}

/// A column of 20 small sphere lights over a clear pool lights its floor by each one's apparent
/// depth, as one lamp does, with no caustic photons: a BSDF ray tracked up through the pool's
/// ideal top meets every light of the column, more than a round of owned lights the walks' stage
/// lists, and takes the light of none that the floor's estimates own, so none comes twice.
#[test]
fn a_column_of_lamps_over_a_pool_lights_its_floor_once() {
    let (pool_top, pool_floor) = (1.5f64, 0.7f64);
    let (radius, intensity) = (0.05f32, 4.0f32);
    let heights: Vec<f64> = (0..20).map(|k| 2.0 + 0.12 * f64::from(k)).collect();
    let lights = heights
        .iter()
        .map(|&h| Light::Sphere {
            position: DVec3::new(0.0, h, 0.0),
            radius,
            intensity: Vec3::splat(intensity),
            shaping: None,
            shadows: true,
        })
        .collect();
    let below_floor = Vec3::new(0.0, pool_floor as f32, 0.0);
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, below_floor),
            at(CLEAR_POOL, Vec3::new(0.0, (pool_top - 0.5) as f32, 0.0)),
        ],
        lights,
        environment: None,
        eye: Vec3::new(3.0, pool_floor as f32 + 0.5, 0.01),
        target: below_floor,
        fov_y: 0.3,
        focus: None,
    };
    let patch = pixel_of(&spec, below_floor);
    let mut bed = Bed::with(spec);
    let no_photons = SceneLightingSettings {
        caustics: CausticSettings {
            photons: 0,
            ..bed.lighting.settings().caustics
        },
        ..*bed.lighting.settings()
    };
    bed.lighting.set_settings(&no_photons).unwrap();
    let view = ViewLighting {
        camera_medium: Some(bed.material(CLEAR_POOL)),
        ..first_vertex()
    };
    let image = render_paths(&mut bed, &paths(view), 256, 1);
    let through = 1.0 - fresnel(POOL_IOR, 1.0);
    let wanted: f64 = heights
        .iter()
        .map(|&h| {
            f64::from(intensity) * through
                / (h - pool_top + (pool_top - pool_floor) / POOL_IOR).powi(2)
        })
        .sum();
    for (c, &radiance) in image.around(patch, 4).iter().enumerate() {
        let e = irradiance(radiance);
        assert!(
            (e - wanted).abs() <= 0.04 * wanted,
            "channel {c}: {e} lux on the floor, wanted {wanted}"
        );
    }
    assert_clean();
}

/// Sunlight through the tinted slab reaches the floor below it through both faces' Fresnel and
/// Beer's law along the refracted path, by manifold next-event estimation through the slab, and
/// through the internal reflections between the faces, which the grid holds, on a
/// level floor and on one tilted 30 degrees toward the sun, whose plane crosses the grid's
/// voxels. The patch lies under a point of the slab past the bands where the faces facing away
/// from the sun fold light onto the floor by total internal reflection.
#[test]
fn a_slab_passes_the_sun_to_the_floor_by_fresnel_and_beer_law() {
    let sun = SUN.direction_to_light.normalize();
    let height = 1.0f32;
    let mu = f64::from(sun.y);
    let mu_t = (1.0 - (1.0 - mu * mu) / (1.5 * 1.5)).sqrt();
    let f = fresnel(1.5, mu);
    let t = 0.5f64.powf(1.0 / mu_t);
    let through = (1.0 - f) * (1.0 - f) * t / (1.0 - f * f * t * t);
    for tilt in [0.0f32, 30.0] {
        let linear = Mat3::from_rotation_x(tilt.to_radians());
        let normal = linear * Vec3::Y;
        let spec = Spec {
            copies: vec![
                Item {
                    model: WHITE_FLOOR,
                    linear,
                    at: Vec3::ZERO,
                },
                at(TINTED_SLAB, Vec3::new(0.0, height, 0.0)),
            ],
            lights: vec![Light::Directional(SUN)],
            environment: None,
            // Off the slab's footprint, so the eye sees the patch past the slab, not through it,
            // and narrow, so the patch's pixels on the receding floor stay clear of the folded
            // bands.
            eye: Vec3::new(6.0, 4.0, 0.01),
            target: Vec3::ZERO,
            fov_y: 0.3,
            focus: None,
        };
        // Where the sun through `through_at`, in the slab's middle plane, meets the floor.
        let cosine = normal.dot(sun);
        let through_at = Vec3::new(0.4, height, 0.27);
        let under = through_at - sun * (normal.dot(through_at) / cosine);
        let patch = pixel_of(&spec, under);
        let mut bed = Bed::with(spec);
        let image = render_paths(&mut bed, &paths(first_vertex()), 32, 1);
        let got = image.around(patch, 4);
        for (c, &radiance) in got.iter().enumerate() {
            let wanted = f64::from(SUN.illuminance[c]) * f64::from(cosine) * through;
            let e = irradiance(radiance);
            assert!(
                (e - wanted).abs() <= 0.05 * wanted,
                "tilt {tilt}, channel {c}: {e} lux under the slab, wanted {wanted}"
            );
        }
        let stats = bed.lighting.stats(bed.last_frame()).unwrap().caustics;
        assert!(stats.deposits > 0, "tilt {tilt}: no photon deposited");
        // At any load a few new keys find every slot of their probe full: such a cell sits out
        // the frame, and reads take the level beside it.
        assert!(
            stats.overflows * 10_000 <= stats.deposits,
            "tilt {tilt}: {} of {} deposits overflowed the grid",
            stats.overflows,
            stats.deposits
        );
    }
    assert_clean();
}

/// A point, a small sphere and a small rect light light a floor under solid glass through its
/// faces' Fresnel, Beer's law and the glass's apparent thickness: straight below the light the
/// bundle of rays from it that reaches a patch of the floor through a face spreads as from a
/// light the glass's thickness over its IOR nearer than the glass's thickness, so the floor's
/// irradiance is the light's intensity times the transmittance over the square of that distance.
/// Under a wide tinted slab, from outside every interior, it takes the light the faces reflect
/// inside too, which goes as far again in the slab per pair of reflections; on the floor of a
/// clear pool, from inside it, a single face. The glass is wide enough that what its sides
/// reflect inside lands far off. The floor's estimate through the glass owns the chains it finds:
/// without caustic photons, a ray tracked through the glass takes a sphere's or rect's light
/// only where it does not, and no ray meets a point light; with them the photons take the rest.
/// Either way the floor's light is the same, and the scene's own pipelines walk as the general
/// ones do.
#[test]
fn local_lights_over_glass_light_the_floor_by_its_apparent_depth() {
    let light_height = 2.0f64;
    let n = 1.5f64;
    let t = f64::from(SLAB_THICKNESS);
    let (slab_height, pool_top, pool_floor) = (1.0f64, 1.5f64, 0.7f64);
    let f = |n: f64| ((n - 1.0) / (n + 1.0)).powi(2);
    // Each pair of reflections inside the slab takes the light through it twice more.
    let (above, below) = (light_height - slab_height - 0.5 * t, slab_height - 0.5 * t);
    let slab = (0..4)
        .map(|k| {
            let crossings = f64::from(2 * k + 1);
            (1.0 - f(n)).powi(2) * f(n).powi(2 * k) * 0.5f64.powf(crossings)
                / (above + below + crossings * t / n).powi(2)
        })
        .sum::<f64>();
    let pool = (1.0 - f(POOL_IOR))
        / (light_height - pool_top + (pool_top - pool_floor) / POOL_IOR).powi(2);
    let position = DVec3::new(0.0, light_height, 0.0);
    // Small enough that their light falls off as a point's within a part in a thousand.
    let (radius, side, intensity) = (0.02f32, 0.03f32, 40.0f32);
    let lights = [
        (
            "point",
            Light::Point {
                position,
                intensity: Vec3::splat(intensity),
                shaping: None,
                shadows: true,
            },
        ),
        (
            "sphere",
            Light::Sphere {
                position,
                radius,
                intensity: Vec3::splat(intensity),
                shaping: None,
                shadows: true,
            },
        ),
        (
            "rect",
            Light::Rect {
                center: position,
                edge_x: Vec3::new(side, 0.0, 0.0),
                edge_y: Vec3::new(0.0, 0.0, side),
                luminance: Vec3::splat(intensity / (side * side)),
                shaping: None,
                shadows: true,
            },
        ),
    ];
    let scenes = [
        ("slab", WIDE_SLAB, slab_height, 0.0, slab),
        ("pool", CLEAR_POOL, pool_top - 0.5, pool_floor, pool),
    ];
    for (scene, glass, glass_at, floor, through) in scenes {
        for (name, light) in lights {
            for photons in [false, true] {
                let below_floor = Vec3::new(0.0, floor as f32, 0.0);
                let spec = Spec {
                    copies: vec![
                        at(WHITE_FLOOR, below_floor),
                        at(glass, Vec3::new(0.0, glass_at as f32, 0.0)),
                    ],
                    lights: vec![light],
                    environment: None,
                    // Under the glass's top face, so the eye sees the patch past no face.
                    eye: Vec3::new(3.0, floor as f32 + 0.5, 0.01),
                    target: below_floor,
                    fov_y: 0.3,
                    focus: None,
                };
                let patch = pixel_of(&spec, below_floor);
                let mut bed = Bed::with(spec);
                if !photons {
                    let no_photons = SceneLightingSettings {
                        caustics: CausticSettings {
                            photons: 0,
                            ..bed.lighting.settings().caustics
                        },
                        ..*bed.lighting.settings()
                    };
                    bed.lighting.set_settings(&no_photons).unwrap();
                }
                let camera_medium = (glass == CLEAR_POOL).then(|| bed.material(CLEAR_POOL));
                let view = ViewLighting {
                    camera_medium,
                    ..first_vertex()
                };
                let mut walks = paths(view);
                let image = render_paths(&mut bed, &walks, 64, 1);
                // The scene's own pipelines, which a scene of local lights alone gives the walks,
                // bring the same.
                if name == "point" && !photons {
                    walks.specialize = true;
                    let general = image.around(patch, 4);
                    let specialized = render_paths(&mut bed, &walks, 64, 1).around(patch, 4);
                    for c in 0..3 {
                        assert!(
                            (specialized[c] - general[c]).abs() <= 1e-3 * general[c],
                            "{scene}, channel {c}: the scene's own pipelines bring {}, the                              general ones {}",
                            specialized[c],
                            general[c]
                        );
                    }
                }
                let wanted = f64::from(intensity) * through;
                for (c, &radiance) in image.around(patch, 4).iter().enumerate() {
                    let e = irradiance(radiance);
                    assert!(
                        (e - wanted).abs() <= 0.04 * wanted,
                        "{scene}, {name}, photons {photons}, channel {c}: {e} lux on the floor, \
                         wanted {wanted}"
                    );
                }
            }
        }
    }
    assert_clean();
}

/// Ideal thin walls above the tinted slab, the faces of a thin producer box of the glass pane's
/// material, pass the sun to the floor under the slab once: the floor's estimate takes it through
/// the slab and the walls, and a caustic photon or a BSDF ray tracked through the slab that the
/// walls passed on straight is the estimate's, as the walk that checks it, whose leg out of the
/// slab takes the estimate's shadow ray past the walls too, says. So the walls cut the floor's
/// whole light as they cut the estimate's. A disc 0.15 rad wide lets the floor's BSDF rays meet
/// the sun often enough to show a second count.
#[test]
fn walls_over_solid_glass_pass_the_sun_once() {
    let wide = DirectionalLight {
        angular_radius: 0.15,
        ..SUN
    };
    let sun = wide.direction_to_light.normalize();
    let height = 1.0f32;
    let through_at = Vec3::new(0.4, height, 0.27);
    let under = through_at - sun * (through_at.y / sun.y);
    let render = |walls: bool, debug_view: LightingDebugView| {
        // Below the walls, so the eye's own rays pass none, and seeing the patch past the slab.
        let spec = Spec {
            copies: vec![
                at(WHITE_FLOOR, Vec3::ZERO),
                at(TINTED_SLAB, Vec3::new(0.0, height, 0.0)),
            ],
            lights: vec![Light::Directional(wide)],
            environment: None,
            eye: Vec3::new(6.0, 2.0, 0.01),
            target: Vec3::ZERO,
            fov_y: 0.3,
            focus: None,
        };
        let patch = pixel_of(&spec, under);
        let mut bed = Bed::with(spec);
        if walls {
            let glass = bed.material(GLASS_PANE);
            bed.set_producer_slab(Vec3::new(0.0, 3.0, 0.0), Vec3::new(5.0, 0.01, 5.0), glass);
        }
        let settings = ViewLighting {
            debug_view,
            ..ViewLighting::default()
        };
        render_paths(&mut bed, &paths(settings), 64, 1).around(patch, 4)
    };
    let estimated = |walls| render(walls, LightingDebugView::NeeOnly);
    let whole = |walls| render(walls, LightingDebugView::None);
    let (estimate_open, estimate_walled) = (estimated(false), estimated(true));
    let (open, walled) = (whole(false), whole(true));
    for c in 0..3 {
        let wanted = estimate_walled[c] / estimate_open[c];
        let ratio = walled[c] / open[c];
        assert!(
            (ratio / wanted - 1.0).abs() <= 0.05,
            "channel {c}: the walls pass {ratio} of the floor's light, {wanted} of its estimate"
        );
    }
    assert_clean();
}

/// Pipelines compiled for what a scene may hold bend the sun through solid glass: manifold
/// next-event estimation reaches the floor under the slab through it by the pipelines that hold
/// everything, and by the scene's own alike, with a decal listed anywhere too, which a manifold
/// walk takes as any surface.
#[test]
fn specialized_pipelines_bend_the_sun_through_solid_glass() {
    let sun = SUN.direction_to_light.normalize();
    let height = 1.0f32;
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, Vec3::ZERO),
            at(TINTED_SLAB, Vec3::new(0.0, height, 0.0)),
        ],
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(6.0, 4.0, 0.01),
        target: Vec3::ZERO,
        fov_y: 0.3,
        focus: None,
    };
    let through_at = Vec3::new(0.4, height, 0.27);
    let patch = pixel_of(&spec, through_at - sun * (through_at.y / sun.y));
    let mut paths = paths(ViewLighting {
        debug_view: LightingDebugView::NeeOnly,
        max_bounces: 0,
        ..ViewLighting::default()
    });
    let decal = FloorDecal {
        material: BLACK_SPHERE,
        center: Vec3::new(-3.0, 0.0, -3.0),
        half_extent: Vec3::splat(0.3),
    };
    let mut render = |specialize, decals: &[FloorDecal]| {
        paths.specialize = specialize;
        let mut bed = Bed::with_features(spec.clone(), decals);
        render_paths(&mut bed, &paths, 8, 1).around(patch, 4)
    };
    for decals in [&[][..], &[decal]] {
        let general = render(false, decals);
        assert!(
            general[1] > 0.0,
            "{} decals: {} under the slab",
            decals.len(),
            general[1]
        );
        let specialized = render(true, decals);
        for c in 0..3 {
            assert!(
                (specialized[c] - general[c]).abs() <= 1e-3 * general[c],
                "{} decals, channel {c}: the scene's own pipelines bring {} under the slab, the general ones {}",
                decals.len(),
                specialized[c],
                general[c]
            );
        }
    }
    assert_clean();
}

/// `d` (unit) refracted through a surface of unit normal `n` facing it, from IOR `from` to `to`;
/// `None` past the critical angle.
fn refract(d: DVec2, n: DVec2, from: f64, to: f64) -> Option<DVec2> {
    let eta = from / to;
    let c = -n.dot(d);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    (k >= 0.0).then(|| eta * d + (eta * c - k.sqrt()) * n)
}

/// Where a vertical ray `b` m from the axis of a unit ball lens of IOR 1.5 centered `height` m
/// over the floor meets the floor after its refraction in, `reflections` reflections inside and
/// its refraction out, as m from the axis, and the share of its light those let through. Every
/// chord of a sphere meets its faces at one angle, so each face takes the first's Fresnel.
fn ball_lens(b: f64, height: f64, reflections: u32) -> Option<(f64, f64)> {
    let center = DVec2::new(0.0, height);
    let d0 = DVec2::new(0.0, -1.0);
    let p1 = DVec2::new(b, height + (1.0 - b * b).sqrt());
    let n1 = p1 - center;
    let mut d = refract(d0, n1, 1.0, 1.5)?;
    // Each chord's other end.
    let mut p = p1 - 2.0 * d.dot(n1) * d;
    for _ in 0..reflections {
        let n = p - center;
        d -= 2.0 * d.dot(n) * n;
        p -= 2.0 * d.dot(p - center) * d;
    }
    let out = refract(d, -(p - center), 1.5, 1.0)?;
    if out.y >= 0.0 {
        return None;
    }
    let floor = p + out * (-p.y / out.y);
    let f = fresnel(1.5, -d0.dot(n1));
    Some((
        floor.x.abs(),
        (1.0 - f) * (1.0 - f) * f.powi(reflections as i32),
    ))
}

/// Where that ray meets the floor once the lens's outside reflects it, as m from the axis, and the
/// share of its light the reflection keeps; `None` where it goes up.
fn ball_lens_reflected(b: f64, height: f64) -> Option<(f64, f64)> {
    let d0 = DVec2::new(0.0, -1.0);
    let p = DVec2::new(b, height + (1.0 - b * b).sqrt());
    let n = p - DVec2::new(0.0, height);
    let d = d0 - 2.0 * d0.dot(n) * n;
    if d.y >= 0.0 {
        return None;
    }
    Some(((p + d * (-p.y / d.y)).x.abs(), fresnel(1.5, -d0.dot(n))))
}

/// A ball lens under a sun straight overhead lights the floor below it by its 1D maps from impact
/// parameter to radius, of the light refracted through, reflected inside up to twice, and
/// reflected off its outside: the first vertex's light in rings where those maps are smooth, by
/// manifold next-event estimation through the lens and the grid's photons and the BSDF rays for
/// the light the estimate does not own, matches the flux the maps send into each ring over its
/// area, with the sun's own outside the lens's shadow, within 10%. The eye sees each ring past the
/// lens, not through it, from 43 m, where a ray origin lies 4 cm off the surface it leaves, so a
/// photon that left the lens off its own chain would count beside the estimate that owns it. The
/// lens of 128 rings bends the light as the sphere does within 2%: one of 32 rings bands its
/// caustic by 6%.
#[test]
fn a_ball_lens_lights_the_floor_by_its_impact_parameter_map() {
    const HEIGHT: f64 = 2.2;
    const ILLUMINANCE: f32 = 100_000.0;
    let sun = DirectionalLight {
        direction_to_light: Vec3::Y,
        illuminance: Vec3::splat(ILLUMINANCE),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    };
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, Vec3::ZERO),
            at(FINE_GLASS_SPHERE, Vec3::new(0.0, HEIGHT as f32, 0.0)),
        ],
        lights: vec![Light::Directional(sun)],
        environment: None,
        eye: Vec3::new(0.0, 25.0, 35.0),
        target: Vec3::ZERO,
        fov_y: 0.06,
        focus: None,
    };
    // The map's flux per ring of 1 cm, over the ring's area.
    const RING: f64 = 0.01;
    const RINGS: usize = 150;
    let mut flux = vec![0.0f64; RINGS];
    let steps = 2_000_000u32;
    for k in 0..steps {
        let b = (f64::from(k) + 0.5) / f64::from(steps);
        let ways = (0..3)
            .map(|reflections| ball_lens(b, HEIGHT, reflections))
            .chain([ball_lens_reflected(b, HEIGHT)]);
        for (r, t) in ways.flatten() {
            let ring = (r / RING) as usize;
            if ring < RINGS {
                flux[ring] += t * 2.0 * std::f64::consts::PI * b / f64::from(steps);
            }
        }
    }
    let wanted: Vec<f64> = flux
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let (r0, r1) = (i as f64 * RING, (i + 1) as f64 * RING);
            let sunlit = if r0 >= 1.0 { 1.0 } else { 0.0 };
            f64::from(ILLUMINANCE) * (f / (std::f64::consts::PI * (r1 * r1 - r0 * r0)) + sunlit)
        })
        .collect();
    let mut bed = Bed::with(spec.clone());
    // Under the HDR ceiling where the lens focuses the sun.
    let pre_exposure = 1.0 / 16.0;
    bed.set_pre_exposure(pre_exposure);
    let image = render_paths(&mut bed, &paths(first_vertex()), 48, 1);
    let mut compared = 0;
    for ring in (20..RINGS - 10).step_by(10) {
        // A ring the map changes little across, so the grid's cells blur it little.
        let near = &wanted[ring - 3..=ring + 3];
        let (lo, hi) = near.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &e| {
            (lo.min(e), hi.max(e))
        });
        if lo <= 0.0 || hi > 1.2 * lo {
            continue;
        }
        let r = (ring as f64 + 0.5) * RING;
        let samples: Vec<f64> = (0..16u32)
            .map(|k| {
                let a = f64::from(k) / 16.0 * std::f64::consts::TAU;
                let p = Vec3::new((r * a.cos()) as f32, 0.0, (r * a.sin()) as f32);
                irradiance(image.around(pixel_of(&spec, p), 1)[1]) / f64::from(pre_exposure)
            })
            .collect();
        assert_mean(&format!("ring at {r:.2} m"), &samples, wanted[ring], 0.1);
        compared += 1;
    }
    assert!(compared >= 3, "{compared} smooth rings compared");
    assert_clean();
}

/// Per-render means of green over the 12 pixel square around `at`, each of `frames` frames from
/// its own seed.
fn patch_means(
    bed: &mut Bed,
    paths: &Paths,
    at: Vec2,
    renders: u32,
    frames: u32,
    seed: u32,
) -> Vec<f64> {
    (0..renders)
        .map(|k| render_paths(bed, paths, frames, seed * 1000 + k).around(at, 6)[1])
        .collect()
}

/// A mirror ball on the white floor, which lies at height 0 on a voxel boundary of every grid
/// level, under a sun behind it, seen from low over the floor: what the grid gives the floor where
/// the ball throws the sun onto it, over the view, within 10 % of the reference map's. Photons and
/// reads round either side of that boundary, and the grid takes a receiver's cell from its plane
/// moved off it, so both meet in one cell.
#[test]
fn a_floor_on_the_grids_voxel_boundaries_takes_its_whole_caustic() {
    let ball = Vec3::new(0.0, 0.7, 0.0);
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, Vec3::ZERO),
            Item {
                model: MIRROR_SPHERE,
                linear: Mat3::from_diagonal(Vec3::splat(0.7)),
                at: ball,
            },
        ],
        lights: vec![Light::Directional(DirectionalLight {
            direction_to_light: Vec3::new(-0.453_413_4, 0.654_930_5, 0.604_551_3),
            ..SUN
        })],
        environment: None,
        eye: ball + Vec3::new(4.03, 1.3, 0.54),
        target: Vec3::new(-2.3, 0.9, -1.2),
        fov_y: 0.8,
        focus: None,
    };
    let mut bed = Bed::with(spec);
    let total = |a: &Accumulated| a.mean.iter().map(|p| f64::from(p[1])).sum::<f64>();
    let grid = total(&render_paths(&mut bed, &paths(gathered()), 32, 1));
    let map = paths(ViewLighting {
        debug_view: LightingDebugView::Caustics,
        ..reference(2)
    });
    let map = total(&render_paths(&mut bed, &map, 32, 2));
    assert!(
        (grid / map - 1.0).abs() < 0.1,
        "the grid's caustic {grid:.1}, the map's {map:.1}"
    );
    assert_clean();
}

/// The floor under a glass sphere, seen beside it, takes the same light from the reference
/// with its photons as from BSDF rays alone, by Welch's test: from an emitter above, a light
/// photons own and BSDF rays through the sphere then leave to them, and from the sky, which
/// stays with BSDF rays.
#[test]
fn photons_and_bsdf_rays_carry_the_same_light_through_a_glass_sphere() {
    let sphere = at(GLASS_SPHERE, Vec3::new(0.0, 1.3, 0.0));
    let floor = at(WHITE_FLOOR, Vec3::ZERO);
    for (what, copies, environment) in [
        (
            "owned emitter",
            vec![floor, sphere, at(EMITTER, Vec3::new(0.0, 3.0, 0.0))],
            None,
        ),
        ("sky, not owned", vec![floor, sphere], Some(SKY)),
    ] {
        let spec = Spec {
            copies,
            lights: Vec::new(),
            environment,
            eye: Vec3::new(0.0, 1.0, 4.0),
            target: Vec3::new(0.0, 0.2, 0.0),
            fov_y: 0.9,
            focus: None,
        };
        let under = pixel_of(&spec, Vec3::ZERO);
        let beside = pixel_of(&spec, Vec3::new(0.5, 0.0, 0.4));
        let mut bed = Bed::with(spec);
        let full_paths = paths(reference(2));
        let bsdf_paths = paths(ViewLighting {
            debug_view: LightingDebugView::BsdfOnly,
            ..reference(2)
        });
        for (where_, patch) in [("under", under), ("beside", beside)] {
            let full = patch_means(&mut bed, &full_paths, patch, 16, 8, 1);
            let bsdf = patch_means(&mut bed, &bsdf_paths, patch, 16, 32, 2);
            assert!(
                full.iter().sum::<f64>() > 0.0,
                "{what}, {where_}: nothing lit"
            );
            assert_same_mean(
                &format!("{what}, {where_}: full against BSDF"),
                &full,
                &bsdf,
            );
        }
    }
    assert_clean();
}

/// Pipelines compiled for what a scene may hold gather the grid's light where it holds a
/// caster: the floor under a ball lens shows the sun's focus through the pipelines that hold
/// everything, and the scene's own render it alike. Without the lens the scene's own hold no
/// caustics and start no photon.
#[test]
fn specialized_pipelines_gather_the_grid_s_light_under_a_caster() {
    let sun = DirectionalLight {
        direction_to_light: Vec3::Y,
        illuminance: Vec3::splat(100_000.0),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    };
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, Vec3::ZERO),
            at(GLASS_SPHERE, Vec3::new(0.0, 2.2, 0.0)),
        ],
        lights: vec![Light::Directional(sun)],
        environment: None,
        eye: Vec3::new(0.0, 7.0, 0.01),
        target: Vec3::ZERO,
        fov_y: 0.45,
        focus: None,
    };
    let focus = pixel_of(&spec, Vec3::ZERO);
    let mut paths = paths(gathered());
    let mut render = |spec: &Spec, specialize| {
        paths.specialize = specialize;
        let mut bed = Bed::with(spec.clone());
        let image = render_paths(&mut bed, &paths, 8, 1);
        let stats = bed.lighting.stats(bed.last_frame()).unwrap().caustics;
        (image.around(focus, 8)[1], stats)
    };
    let (general, _) = render(&spec, false);
    assert!(general > 0.0, "the lens lights nothing");
    let (specialized, _) = render(&spec, true);
    assert!(
        (specialized - general).abs() <= 0.02 * general,
        "the scene's own pipelines gather {specialized} under the lens, the general ones {general}"
    );
    let bare = Spec {
        copies: vec![at(WHITE_FLOOR, Vec3::ZERO)],
        ..spec
    };
    let (dark, stats) = render(&bare, true);
    assert_eq!(dark, 0.0, "light with no caster");
    assert_eq!(stats, CausticStats::default(), "photons with no caster");
    assert_clean();
}

/// A scene holds its caustics' grid while it holds a caster. A floor under the sun alone
/// starts no photon and gathers nothing. A mirror set over it, facing the eye, throws the sun
/// on the floor. The frame after the mirror is cleared starts none again and shows that patch
/// dark: the grid went with its cells' light, which no path reads. Three mirrors set elsewhere
/// then light their own patches, the third's among them, which the producers' caster lists
/// hold room for only once they have grown, and in their first frame the first mirror's patch
/// is dark: the grid that came with them is empty.
#[test]
fn the_grid_comes_with_a_scene_s_first_caster_and_goes_with_its_last() {
    let spec = Spec {
        copies: vec![at(WHITE_FLOOR, Vec3::ZERO)],
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(0.0, 3.0, 4.0),
        target: Vec3::new(0.0, 0.0, -0.5),
        fov_y: 0.9,
        focus: None,
    };
    // Where the sun off the middle of a mirror at `center`, which faces the eye, meets the
    // floor.
    let eye = spec.eye;
    let thrown = |center: Vec3| {
        let normal = (eye - center).normalize();
        let sun = -SUN.direction_to_light.normalize();
        let off = sun - 2.0 * sun.dot(normal) * normal;
        center - off * (center.y / off.y)
    };
    let mirror = |x: f32| Particle {
        center: Vec3::new(x, 0.5, -1.0),
        size: 0.3,
    };
    let first = pixel_of(&spec, thrown(mirror(0.0).center));
    let moved = [-2.0f32, 2.0, 3.0].map(mirror);
    let third = pixel_of(&spec, thrown(moved[2].center));
    let mut bed = Bed::with(spec);
    let gathered = paths(gathered());
    let none = CausticStats::default();
    let stats = |bed: &Bed| bed.lighting.stats(bed.last_frame()).unwrap().caustics;

    let image = render_paths(&mut bed, &gathered, 4, 1);
    assert_eq!(stats(&bed), none, "photons with no caster");
    assert_eq!(image.around(first, 6), [0.0; 3], "light with no caster");

    bed.set_mirror_quads(&[mirror(0.0)], Affine3::IDENTITY);
    let image = render_paths(&mut bed, &gathered, 8, 2);
    assert!(stats(&bed).deposits > 0, "no photon off the mirror");
    assert!(image.around(first, 6)[1] > 0.0, "the mirror lights nothing");

    bed.clear_quads();
    let image = render_paths(&mut bed, &gathered, 1, 3);
    assert_eq!(stats(&bed), none, "photons once the caster is gone");
    assert_eq!(image.around(first, 6), [0.0; 3], "its light stayed");

    bed.set_mirror_quads(&moved, Affine3::IDENTITY);
    let image = render_paths(&mut bed, &gathered, 1, 4);
    assert!(stats(&bed).deposits > 0, "no photon off the new mirrors");
    assert_eq!(
        image.around(first, 6),
        [0.0; 3],
        "the new grid holds the first mirror's light"
    );
    let image = render_paths(&mut bed, &gathered, 8, 5);
    assert!(
        image.around(third, 6)[1] > 0.0,
        "the third mirror lights nothing"
    );
    assert_clean();
}

/// A producer's photons find the one mirror of its 4,097 that the sun reaches: the rest lie under
/// the floor, in its shadow, so the lit mirror holds 0.024% of the producer's area, the share of
/// its photons it would take by area alone. The mirror is white metal, which reflects all of the
/// sun, so the patch it throws the sun on takes the sun's illuminance across the beam slanted
/// onto the floor: measured within 3% over 8 frames, after 16 in which the photons find it. The
/// first frame aims them blind, and the few that reach the patch carry a flux thousands of times
/// a later frame's, which the grid weighs by and lets go of.
#[test]
fn photons_find_the_one_lit_mirror_of_a_producer() {
    let spec = Spec {
        copies: vec![at(WHITE_FLOOR, Vec3::ZERO)],
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(0.0, 3.0, 4.0),
        target: Vec3::new(0.0, 0.0, -0.5),
        fov_y: 0.9,
        focus: None,
    };
    let center = Vec3::new(0.0, 0.5, -1.0);
    let normal = (spec.eye - center).normalize();
    let sun = -SUN.direction_to_light.normalize();
    let off = sun - 2.0 * sun.dot(normal) * normal;
    let lit = pixel_of(&spec, center - off * (center.y / off.y));
    let side = 64;
    let cell = |k: u32| (k as f32 + 0.5) / side as f32 * 8.8 - 4.4;
    let mirrors: Vec<Particle> = std::iter::once(Particle { center, size: 0.3 })
        .chain((0..side * side).map(|i| Particle {
            center: Vec3::new(cell(i % side), -0.5, cell(i / side)),
            size: 0.3,
        }))
        .collect();
    let mut bed = Bed::with(spec);
    let pre_exposure = 1.0 / 64.0;
    bed.set_pre_exposure(pre_exposure);
    bed.set_mirror_quads(&mirrors, Affine3::IDENTITY);
    let gathered = paths(gathered());
    render_paths(&mut bed, &gathered, 16, 1);
    let got = render_paths(&mut bed, &gathered, 8, 2).around(lit, 2)[1];
    let want = f64::from(pre_exposure) * f64::from(WHITE_FLOOR_ALBEDO) / std::f64::consts::PI
        * f64::from(SUN.illuminance.y)
        * f64::from(-off.y);
    assert!(
        (got / want - 1.0).abs() < 0.03,
        "the lit mirror's patch shows {got}, not {want}"
    );
    assert_clean();
}

/// A copy's photons find the one mirror of its caster section that the sun reaches: the section
/// is a cooked mesh, a 1,008 m^2 sheet under a floor scaled to cover it, in its shadow, and a
/// 0.36 m^2 square above the floor, which its cooked set gives 0.036% of the photons by area.
/// The square is white metal, which reflects all of the sun, so the patch it throws the sun on
/// takes the sun's illuminance across the beam slanted onto the floor: measured within 3% over
/// 8 frames, after 16 in which the photons find it.
#[test]
fn photons_find_the_lit_square_of_a_cooked_caster() {
    let field = Vec3::new(0.0, -0.5, 0.0);
    let spec = Spec {
        copies: vec![
            Item {
                model: WHITE_FLOOR,
                linear: Mat3::from_diagonal(Vec3::new(4.0, 1.0, 4.0)),
                at: Vec3::ZERO,
            },
            at(MIRROR_FIELD, field),
        ],
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(0.0, 3.0, 4.0),
        target: Vec3::new(0.0, 0.0, -0.5),
        fov_y: 0.9,
        focus: None,
    };
    let center = field + Vec3::Y;
    let normal = MIRROR_FIELD_FACING.normalize();
    let sun = -SUN.direction_to_light.normalize();
    let off = sun - 2.0 * sun.dot(normal) * normal;
    let lit = pixel_of(&spec, center - off * (center.y / off.y));
    let mut bed = Bed::with(spec);
    let pre_exposure = 1.0 / 64.0;
    bed.set_pre_exposure(pre_exposure);
    let gathered = paths(gathered());
    render_paths(&mut bed, &gathered, 16, 1);
    let got = render_paths(&mut bed, &gathered, 8, 2).around(lit, 2)[1];
    let want = f64::from(pre_exposure) * f64::from(WHITE_FLOOR_ALBEDO) / std::f64::consts::PI
        * f64::from(SUN.illuminance.y)
        * f64::from(-off.y);
    assert!(
        (got / want - 1.0).abs() < 0.03,
        "the lit square's patch shows {got}, not {want}"
    );
    assert_clean();
}

/// The floor before a mirror takes the light of two emitters above by the reference's photons
/// as it does by BSDF rays alone, by Welch's test. The mirror is a producer set once frames
/// have run, after a producer that emits nothing: the two emitters' copies hold the emissive
/// list's first entries, so its emissive list entry is not its producer table entry, and it
/// grows the producer table while the emissive list holds room for it, so lighting keeps what
/// it holds per entry and runs the producers' caster lists over one entry more.
#[test]
fn photons_reflect_off_a_producer_set_as_the_producer_table_alone_grows() {
    let emitter = |x: f32| at(EMITTER, Vec3::new(x, 3.0, 0.0));
    let spec = Spec {
        copies: vec![at(WHITE_FLOOR, Vec3::ZERO), emitter(-0.4), emitter(0.4)],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 1.0, 4.0),
        target: Vec3::new(0.0, 0.2, 0.0),
        fov_y: 0.9,
        focus: None,
    };
    // Where the floor sees the emitters in the mirror.
    let lit = pixel_of(&spec, Vec3::new(0.0, 0.0, -0.6));
    let mut bed = Bed::with(spec);
    bed.set_idle_producer();
    let full_paths = paths(reference(2));
    let bsdf_paths = paths(ViewLighting {
        debug_view: LightingDebugView::BsdfOnly,
        ..reference(2)
    });
    render_paths(&mut bed, &full_paths, 2, 1);
    let before = bed.table_entries();
    bed.set_mirror_quads(
        &[Particle {
            center: Vec3::new(0.0, 0.85, -1.0),
            size: 0.8,
        }],
        Affine3::IDENTITY,
    );
    let photons = render_paths(&mut bed, &paths(gathered()), 8, 3).around(lit, 6)[1];
    let after = bed.table_entries();
    assert!(
        after.0 > before.0 && after.1 == before.1,
        "the producer table alone grows: {before:?} to {after:?}"
    );
    assert!(photons > 0.0, "no photon reached the floor by the mirror");
    let full = patch_means(&mut bed, &full_paths, lit, 16, 8, 1);
    let bsdf = patch_means(&mut bed, &bsdf_paths, lit, 16, 32, 2);
    assert_same_mean("full against BSDF", &full, &bsdf);
    assert_clean();
}

/// A photon's chain goes on at opaque mirrors: the sun off one mirror, level to a second and
/// down from it lights the floor where nothing else throws it, so a scene with a mirror among
/// its copies gives its photons the rounds of a chain of opaque vertices. The mirrors are white
/// metal, which reflects all of the sun, so the patch takes the sun's illuminance across the beam
/// slanted onto the floor along its way down, in the grid and in the reference's map: measured
/// within 1.5%. The grid takes it again with cells 8 times as wide, wider than a pixel's
/// footprint, so its reads take the finer of their two levels where the coarser holds photons
/// enough. At a pre-exposure of 1 the patch passes the HDR ceiling, which would clip the
/// frames that draw it brightest; the frame that sets the pre-exposure runs no photons, which the
/// grid's frames after it count as no frame at all.
#[test]
fn photons_chain_through_two_mirrors() {
    let sun = -SUN.direction_to_light.normalize();
    // A [`MIRROR`] of edge `edge` at `center`, facing `normal`.
    let mirror = |center: Vec3, normal: Vec3, edge: f32| {
        let x = Vec3::Y.cross(normal).normalize();
        Item {
            model: MIRROR,
            linear: Mat3::from_cols(x, normal.cross(x), normal) * (0.5 * edge),
            at: center,
        }
    };
    let first = Vec3::new(-1.5, 1.0, -1.0);
    let level = Vec3::X;
    let second = first + 2.0 * level;
    let patch = Vec3::new(1.5, 0.0, -0.5);
    let down = (patch - second).normalize();
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, Vec3::ZERO),
            mirror(first, (level - sun).normalize(), 0.8),
            mirror(second, (down - level).normalize(), 0.8),
        ],
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(0.0, 3.0, 4.0),
        target: Vec3::new(0.5, 0.0, -0.5),
        fov_y: 0.9,
        focus: None,
    };
    let lit = pixel_of(&spec, patch);
    let mut bed = Bed::with(spec);
    let pre_exposure = 1.0 / 64.0;
    bed.set_pre_exposure(pre_exposure);
    let want = f64::from(pre_exposure) * f64::from(WHITE_FLOOR_ALBEDO) / std::f64::consts::PI
        * f64::from(SUN.illuminance.y)
        * f64::from(-down.y);
    let map = paths(ViewLighting {
        debug_view: LightingDebugView::Caustics,
        ..reference(2)
    });
    for (what, settings, frames) in [("grid", &paths(gathered()), 8), ("map", &map, 32)] {
        // The beam's footprint on the floor is a few pixels across, so its middle alone.
        let got = render_paths(&mut bed, settings, frames, 1).around(lit, 1)[1];
        assert!(
            (got / want - 1.0).abs() < 0.03,
            "the {what}'s patch by both mirrors shows {got}, not {want}"
        );
    }
    let wide = SceneLightingSettings {
        caustics: CausticSettings {
            base_cell_log2: -5,
            ..bed.lighting.settings().caustics
        },
        ..*bed.lighting.settings()
    };
    bed.lighting.set_settings(&wide).unwrap();
    let got = render_paths(&mut bed, &paths(gathered()), 8, 2).around(lit, 1)[1];
    assert!(
        (got / want - 1.0).abs() < 0.03,
        "the grid of wide cells' patch shows {got}, not {want}"
    );
    assert_clean();
}
