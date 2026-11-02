//! The froxel volume: the slice mapping, the froxel lookup of points, froxel in-scatter
//! against the reference camera march in open haze and under a canopy, silhouettes against
//! fogged sky and distant surfaces, camera segments ending at dielectric layers, and
//! directional lights past the shadowed ones. Aerial perspective: its slice mapping,
//! surfaces at 1, 5 and 30 km against the reference march, no seam where it meets the
//! froxels, and long path segments' march against a fine one. Blended particles: in shade
//! against in sun inside and beyond the froxel volume, fogged by the single scattering in
//! front of them and through the aerial-perspective LUT past the froxels, and steady at a
//! still camera. Medium proxies on the camera ray: a plume against the reference march and, per
//! pixel, against a fine march of its grid; overlapping boxes, surfaces inside boxes and eyes
//! inside boxes against Beer's law, inside and past the froxels; sky pixels past the froxels;
//! and proxies off screen. Smoke: its opacity on the camera ray, and a column across a shaft
//! against the reference march.

use lighting_suite::bed::{
    Accumulated, Bed, CANOPY, EXTENT, FLOOR, GLASS_SPHERE, Item, LEAF, MIRROR, PLUME, Particle,
    ParticleLook, Paths, SKY, SUN, Spec, WHITE_SPHERE, look, pixel_of, plume_volume,
};
use lighting_suite::shaders::{
    fog_aerial, fog_apply, fog_directional, fog_froxel, fog_particles, froxel_lookup, froxel_probe,
    segment_probe,
};
use lighting_suite::{assert_clean, finish, kernels, pipeline, readback, render, upload};
use orr_color::luminance;
use orr_gpu::{Allocation, Hazards, KernelSet, Stages};
use orr_lighting::{
    AERIAL_SIZE, DirectionalLight, DiscProfile, Light, LightId, LightingDebugView, MediaSettings,
    PathEntry, ReferenceSettings, ViewLighting,
};
use orr_math::{DVec3, Mat3, Quat, Vec3};
use orr_pack::volume::GridSampler;
use orr_scene::{MediumProxy, MediumProxyId, Opacity, ProxyDensity, RendererDesc};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

/// Kernels with the stock height fog linked as the medium (`media_types.slang`).
fn fog_kernels() -> &'static KernelSet {
    static SET: OnceLock<KernelSet> = OnceLock::new();
    SET.get_or_init(|| {
        kernels(&[
            (&fog_froxel::MODULE, "media/froxel"),
            (&fog_aerial::MODULE, "media/aerial"),
            (&fog_apply::MODULE, "media/apply"),
            (&fog_directional::MODULE, "light_sampling/directional"),
            (&fog_particles::MODULE, "media/particle_draw"),
        ])
    })
}

/// `media_types`' `FogData`: extinction at `base_height`, 1/m, falling by e every
/// `falloff_height` m.
struct Fog {
    density: f32,
    base_height: f32,
    falloff_height: f32,
}

const HAZE: Fog = Fog {
    density: 2e-3,
    base_height: 0.0,
    falloff_height: 100.0,
};

fn directional(elevation: f32, azimuth: f32, illuminance: f32) -> Light {
    let (e, a) = (elevation.to_radians(), azimuth.to_radians());
    Light::Directional(DirectionalLight {
        direction_to_light: Vec3::new(e.cos() * a.sin(), e.sin(), -e.cos() * a.cos()),
        illuminance: Vec3::splat(illuminance),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    })
}

fn at(model: &'static str, p: Vec3) -> Item {
    Item {
        model,
        linear: Mat3::IDENTITY,
        at: p,
    }
}

/// A bed of `copies`, `lights` and a uniform `environment` in `fog`, its camera at `eye`
/// looking at `target`. A sphere behind the camera stands in when `copies` is empty.
fn foggy(
    fog: &Fog,
    mut copies: Vec<Item>,
    lights: Vec<Light>,
    environment: Option<[f32; 3]>,
    eye: Vec3,
    target: Vec3,
) -> (Bed, Allocation<[f32; 4]>) {
    if copies.is_empty() {
        copies.push(at(WHITE_SPHERE, eye + (eye - target).normalize() * 30.0));
    }
    let spec = Spec {
        copies,
        lights,
        environment,
        eye,
        target,
        fov_y: 0.8,
        focus: None,
    };
    let renderer = RendererDesc {
        kernels: fog_kernels(),
        ..RendererDesc::default()
    };
    let mut bed = Bed::with_renderer(spec, &renderer);
    let data = upload(
        &[[fog.density, fog.base_height, fog.falloff_height, 0.0]],
        "fog data",
    );
    bed.set_medium(data.gpu().address());
    (bed, data)
}

/// The fog kernels' paths, compiled once for the whole binary.
fn fog_paths() -> MutexGuard<'static, Paths> {
    static PATHS: OnceLock<Mutex<Paths>> = OnceLock::new();
    PATHS
        .get_or_init(|| {
            Mutex::new(Paths::new(
                fog_kernels(),
                PathEntry::Fastest,
                ViewLighting::default(),
            ))
        })
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

fn settings(reference: bool, media: MediaSettings) -> ViewLighting {
    ViewLighting {
        media,
        debug_view: LightingDebugView::Media,
        reference: reference.then(ReferenceSettings::default),
        ..ViewLighting::default()
    }
}

/// In-scatter over `frames` frames: froxels, or the reference camera march in 64 steps.
fn inscatter(bed: &mut Bed, reference: bool, media: MediaSettings, frames: u32) -> Accumulated {
    let media = if reference {
        MediaSettings {
            segment_steps: 64,
            ..media
        }
    } else {
        media
    };
    shown(bed, settings(reference, media), frames)
}

/// What media apply writes under `lighting` over `frames` frames.
fn shown(bed: &mut Bed, lighting: ViewLighting, frames: u32) -> Accumulated {
    let mut paths = fog_paths();
    paths.settings = lighting;
    bed.accumulate(frames, 3, &mut |list, f| paths.record(list, f))
}

/// Mean luminance of each 8 x 8 block.
fn block_luminance(a: &Accumulated) -> Vec<f64> {
    let [w, h] = EXTENT;
    let mut out = Vec::new();
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let mut sum = 0.0;
            for y in by * 8..by * 8 + 8 {
                for x in bx * 8..bx * 8 + 8 {
                    let p = a.mean[(y * w + x) as usize];
                    sum += luminance([p[0], p[1], p[2]].map(f64::from));
                }
            }
            out.push(sum / 64.0);
        }
    }
    out
}

/// The largest and the mean relative error of `a`'s blocks against `reference`'s.
fn block_errors(a: &Accumulated, reference: &Accumulated) -> (f64, f64) {
    let (a, r) = (block_luminance(a), block_luminance(reference));
    let errors: Vec<f64> = a
        .iter()
        .zip(&r)
        .map(|(a, r)| (a - r).abs() / r.max(1e-9))
        .collect();
    let worst = errors.iter().copied().fold(0.0, f64::max);
    (worst, errors.iter().sum::<f64>() / errors.len() as f64)
}

/// The froxels' slices over [0, `near_range`] and the aerial-perspective LUT's over
/// [`near_range`, `ap_range`]: each distance to its slice coordinate and back.
#[test]
fn slices_map_distances_and_back() {
    let (near_range, ap_range) = (256.0f32, 128_000.0f32);
    for (entry, start, end, slices) in [
        ("main", 0.0, near_range, 64),
        ("aerial", near_range, ap_range, AERIAL_SIZE),
    ] {
        let distances: Vec<f32> = (0..=1000)
            .map(|k| start + (end - start) * (k as f32 / 1000.0))
            .collect();
        let input = upload(&distances, "distances");
        let out = readback::<[f32; 2]>(distances.len() as u64, "slices");
        let root = upload(
            &[froxel_probe::Root {
                distances: input.gpu(),
                out: out.gpu(),
                count: distances.len() as u32,
                slices,
                near_range,
                ap_range,
            }],
            "froxel probe root",
        );
        let p = pipeline(&froxel_probe::MODULE, entry);
        let mut list = render().command_list().unwrap();
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(
            &p,
            root.gpu(),
            [(distances.len() as u32).div_ceil(64), 1, 1],
        );
        finish(list);
        for (t, [c, back]) in distances.iter().zip(out.as_slice().unwrap()) {
            let expected = slices as f32 * ((t - start) / (end - start)).sqrt();
            assert!(
                (c - expected).abs() < 1e-4 * slices as f32,
                "{entry}: {t} m maps to slice {c}, not {expected}"
            );
            assert!(
                (back - t).abs() <= 1e-4 * end,
                "{entry}: {t} m comes back as {back}"
            );
        }
    }
    assert_clean();
}

/// One shadowed sun at 10 degrees in open height fog, seen 18 to 80 degrees away in a wide
/// view and 1 to 9 degrees away in a narrow one whose 8 x 8 tiles span what they do at
/// 1080p: each 8 x 8 block's froxel in-scatter within 5% of the reference camera march.
#[test]
fn open_haze_matches_the_reference_march() {
    let sun = directional(10.0, 0.0, 1_000.0);
    let eye = Vec3::new(0.0, 2.0, 0.0);
    for (azimuth, elevation, fov_y) in [(48.0f32, 5.0f32, 0.8f32), (5.0, 10.0, 0.1)] {
        let (a, e) = (azimuth.to_radians(), elevation.to_radians());
        let target = eye + Vec3::new(a.sin() * e.cos(), e.sin(), -a.cos() * e.cos());
        let (mut bed, _fog) = foggy(&HAZE, Vec::new(), vec![sun], None, eye, target);
        bed.spec.fov_y = fov_y;
        let froxels = inscatter(&mut bed, false, MediaSettings::default(), 64);
        let reference = inscatter(&mut bed, true, MediaSettings::default(), 256);
        let (worst, mean) = block_errors(&froxels, &reference);
        assert!(
            worst < 0.05,
            "a block's in-scatter is {worst:.3} off the reference at {azimuth} deg (mean {mean:.3})"
        );
    }
    assert_clean();
}

const DENSE: Fog = Fog {
    density: 2e-2,
    base_height: 0.0,
    falloff_height: 100.0,
};

fn mean_luminance(a: &Accumulated) -> f64 {
    let n = a.mean.len() as f64;
    a.mean
        .iter()
        .map(|p| luminance([p[0], p[1], p[2]].map(f64::from)))
        .sum::<f64>()
        / n
}

/// Two suns at 10 degrees from either side over a slatted canopy, both shadowed, in a view
/// whose 8 x 8 tiles span what they do at 1080p: each block's froxel in-scatter, shafts
/// included, within 5% of the reference camera march, and the fog under the canopy darker
/// than without it.
#[test]
fn canopy_shafts_match_the_reference_march() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 3.0, -10.0);
    let suns = vec![
        directional(10.0, 90.0, 1_000.0),
        directional(10.0, -90.0, 800.0),
    ];
    let media = MediaSettings {
        froxel_shadowed_lights: 2,
        ..MediaSettings::default()
    };
    let canopy = vec![at(CANOPY, Vec3::new(0.0, 6.0, 0.0))];
    let (mut bed, _fog) = foggy(&DENSE, canopy, suns.clone(), None, eye, target);
    bed.spec.fov_y = 0.1;
    let froxels = inscatter(&mut bed, false, media, 64);
    let reference = inscatter(&mut bed, true, media, 256);
    let (worst, mean) = block_errors(&froxels, &reference);
    assert!(
        worst < 0.05,
        "a block's in-scatter is {worst:.3} off the reference (mean {mean:.3})"
    );

    let (mut open, _fog) = foggy(&DENSE, Vec::new(), suns, None, eye, target);
    open.spec.fov_y = 0.1;
    let open = inscatter(&mut open, false, media, 64);
    let (under, outside) = (mean_luminance(&froxels), mean_luminance(&open));
    assert!(
        under < 0.75 * outside,
        "fog under the canopy is {under:.3}, in the open {outside:.3}"
    );
    assert_clean();
}

/// An alpha-tested leaf and an opaque square against fogged sky and against a sphere 1 km
/// away, lit by a sun and the sky's environment: the pixels within two pixels of their
/// silhouettes take the in-scatter they take without them.
#[test]
fn silhouettes_against_fog_have_no_halos() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 2.0, -10.0);
    let sun = vec![directional(30.0, 20.0, 1_000.0)];
    let front = [
        at(LEAF, Vec3::new(-1.2, 2.0, -6.0)),
        at(MIRROR, Vec3::new(1.4, 2.0, -6.0)),
    ];
    let fogged = |copies: Vec<Item>| {
        let (mut bed, _fog) = foggy(&HAZE, copies, sun.clone(), Some(SKY), eye, target);
        inscatter(&mut bed, false, MediaSettings::default(), 32)
    };
    let a = fogged(front.to_vec());
    let [w, h] = EXTENT;
    let on = |x: i32, y: i32| {
        (0..w as i32).contains(&x)
            && (0..h as i32).contains(&y)
            && a.mean[(y as u32 * w + x as u32) as usize][3] > 0.5
    };
    let lum = |p: [f32; 4]| luminance([p[0], p[1], p[2]].map(f64::from));
    for background in [None, Some(distant(1_000.0))] {
        let with = fogged(front.iter().copied().chain(background).collect());
        let without = fogged(background.into_iter().collect());
        let mut worst = 0.0f64;
        let mut checked = 0;
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let beside = (-2..=2).any(|dy| (-2..=2).any(|dx| on(x + dx, y + dy)));
                if on(x, y) || !beside {
                    continue;
                }
                let i = (y as u32 * w + x as u32) as usize;
                let open = lum(without.mean[i]);
                worst = worst.max((lum(with.mean[i]) - open).abs() / open);
                checked += 1;
            }
        }
        assert!(checked > 100, "{checked} pixels beside the silhouettes");
        assert!(
            worst < 0.03,
            "{background:?}: beside a silhouette the in-scatter is {worst:.3} off its own without it"
        );
    }
    assert_clean();
}

/// Seven directional lights, the brightest shadowed: every froxel in front of the surfaces
/// traces at most two rays, and each unshadowed light adds its share to the fog.
#[test]
fn unshadowed_directional_lights_each_light_the_fog() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 2.0, -10.0);
    let mut lights = vec![directional(40.0, 20.0, 1_000.0)];
    lights.extend((0..6).map(|k| directional(45.0, 60.0 * k as f32, 300.0)));
    let copies = vec![at(FLOOR, Vec3::ZERO)];
    let (mut bed, _fog) = foggy(&HAZE, copies, lights.clone(), None, eye, target);
    let media = MediaSettings::default();
    let all = mean_luminance(&inscatter(&mut bed, false, media, 32));
    let stats = bed.view().media_stats(bed.last_frame()).unwrap();
    // One shadow ray each, and one more unless every candidate came out 0.
    let active = stats.active_froxels;
    assert!(active > 0);
    assert!(
        (2 * active - active / 100..=2 * active).contains(&stats.froxel_rays),
        "{stats:?}"
    );

    for k in 1..7 {
        bed.set_light(LightId(k), None);
    }
    let shadowed = mean_luminance(&inscatter(&mut bed, false, media, 32));
    let share = (all - shadowed) / 6.0;
    assert!(share > 0.0, "the unshadowed lights add {share}");
    for k in 1..7 {
        for (j, l) in lights.iter().enumerate().skip(1) {
            bed.set_light(LightId(j as u32), (j != k as usize).then_some(*l));
        }
        let without = mean_luminance(&inscatter(&mut bed, false, media, 32));
        let added = all - without;
        assert!(
            (0.5 * share..1.5 * share).contains(&added),
            "light {k} adds {added:.3} to the fog; an equal share is {share:.3}"
        );
    }
    assert_clean();
}

/// Points between their froxel's column rays and inside its slice, seen from the bed's
/// view, map back to that froxel.
#[test]
fn froxel_lookup_finds_the_froxel_of_each_point() {
    let near_range = 256.0;
    let (mut bed, _fog) = foggy(
        &HAZE,
        Vec::new(),
        Vec::new(),
        Some(SKY),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(3.0, 1.0, -10.0),
    );
    let grid = [EXTENT[0].div_ceil(8), EXTENT[1].div_ceil(8), 64];
    let count = u64::from(grid[0] * grid[1] * grid[2]);
    let out = readback::<[u32; 4]>(count, "froxels found");
    let p = pipeline(&froxel_lookup::MODULE, "main");
    let mut roots = Vec::new();
    bed.accumulate(1, 0, &mut |list, f| {
        let root = upload(
            &[froxel_lookup::Root {
                view: f.pass().view,
                out: out.gpu(),
                columns: grid[0],
                rows: grid[1],
                slices: grid[2],
                near_range,
            }],
            "froxel lookup root",
        );
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&p, root.gpu(), grid.map(|n| n.div_ceil(4)));
        roots.push(root);
    });
    let found = out.as_slice().unwrap();
    for z in 0..grid[2] {
        for y in 0..grid[1] {
            for x in 0..grid[0] {
                let f = found[((z * grid[1] + y) * grid[0] + x) as usize];
                assert_eq!(f, [x, y, z, 1], "froxel {x}, {y}, {z}");
            }
        }
    }
    assert_clean();
}

/// A clear glass sphere of radius 1 m at 6 m and at 16 m in uniform haze against fogged
/// sky: the in-scatter at the image center grows with the distance to the glass's front
/// face (5 m and 15 m), not to the sky behind it.
#[test]
fn camera_segments_end_at_dielectric_layers() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 2.0, -10.0);
    let sun = vec![directional(30.0, 20.0, 1_000.0)];
    let center = |copies| {
        let (mut bed, _fog) = foggy(&HAZE, copies, sun.clone(), Some(SKY), eye, target);
        let a = inscatter(&mut bed, false, MediaSettings::default(), 16);
        let [w, h] = EXTENT;
        let mut sum = 0.0;
        for y in h / 2 - 4..h / 2 + 4 {
            for x in w / 2 - 4..w / 2 + 4 {
                let p = a.mean[(y * w + x) as usize];
                sum += luminance([p[0], p[1], p[2]].map(f64::from));
            }
        }
        sum / 64.0
    };
    let near = center(vec![at(GLASS_SPHERE, Vec3::new(0.0, 2.0, -6.0))]);
    let far = center(vec![at(GLASS_SPHERE, Vec3::new(0.0, 2.0, -16.0))]);
    let sky = center(Vec::new());
    assert!(
        far < 0.2 * sky,
        "in-scatter to the glass {far:.4}, to the sky {sky:.4}"
    );
    let ratio = far / near;
    assert!(
        (ratio - 3.0).abs() < 0.15,
        "in-scatter to the glass at 15 m is {ratio:.3} times that at 5 m"
    );
    assert_clean();
}

/// Haze for kilometers: extinction 3e-5 /m at the ground, falling by e every km.
const AIR: Fog = Fog {
    density: 3e-5,
    base_height: 0.0,
    falloff_height: 1000.0,
};

/// A white sphere whose near face lies `distance` m straight ahead of an eye at (0, 2, 0)
/// looking along -z, filling the middle of the view.
fn distant(distance: f32) -> Item {
    let radius = 0.15 * distance;
    Item {
        model: WHITE_SPHERE,
        linear: Mat3::from_diagonal(Vec3::splat(radius)),
        at: Vec3::new(0.0, 2.0, -(distance + radius)),
    }
}

/// Mean luminance of the four 8 x 8 blocks at the middle of the image.
fn middle_blocks(a: &Accumulated) -> [f64; 4] {
    let blocks = block_luminance(a);
    let (columns, rows) = (EXTENT[0] / 8, EXTENT[1] / 8);
    let at = |bx: u32, by: u32| blocks[(by * columns + bx) as usize];
    let (x, y) = (columns / 2, rows / 2);
    [at(x - 1, y - 1), at(x, y - 1), at(x - 1, y), at(x, y)]
}

/// Surfaces 1, 5 and 30 km away in haze thinning with altitude, under a sun at 20 degrees
/// beside the view: the in-scatter in front of them, the froxels' to `near_range` and the
/// aerial-perspective LUT's past it, within 5% of the reference camera march per 8 x 8 block
/// at the middle of the view.
#[test]
fn distant_surfaces_match_the_reference_march() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 2.0, -10.0);
    let sun = vec![directional(20.0, 100.0, 1_000.0)];
    for distance in [1_000.0, 5_000.0, 30_000.0] {
        let copies = vec![distant(distance)];
        let (mut bed, _fog) = foggy(&AIR, copies, sun.clone(), None, eye, target);
        let lut = middle_blocks(&inscatter(&mut bed, false, MediaSettings::default(), 16));
        let reference = middle_blocks(&inscatter(&mut bed, true, MediaSettings::default(), 256));
        for (l, r) in lut.iter().zip(&reference) {
            assert!(
                (l - r).abs() < 0.05 * r,
                "at {distance} m a block's in-scatter is {l:.2}, the reference's {r:.2}"
            );
        }
    }
    assert_clean();
}

/// Surfaces a quarter meter either side of `near_range` in haze under a sun: the in-scatter
/// in front of them, from the froxels alone and from the froxels and the aerial-perspective
/// LUT, within 1% of each other.
#[test]
fn aerial_perspective_meets_the_froxels_without_a_seam() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 2.0, -10.0);
    let sun = vec![directional(20.0, 100.0, 1_000.0)];
    let near_range = 256.0;
    let [inside, past] = [near_range - 0.25, near_range + 0.25].map(|d| {
        let (mut bed, _fog) = foggy(&HAZE, vec![distant(d)], sun.clone(), None, eye, target);
        middle_blocks(&inscatter(&mut bed, false, MediaSettings::default(), 16))
    });
    for (a, b) in inside.iter().zip(&past) {
        assert!(
            (b - a).abs() < 0.01 * a,
            "in-scatter {a:.3} just inside near_range, {b:.3} just past it"
        );
    }
    assert_clean();
}

/// Path segments from a mirror on the ground up to terrain 5 and 20 km away, rising at 0.5 to
/// 5 degrees, in haze thinning with altitude under a low sun: `segment_inscatter`'s march,
/// averaged over its jitter, within 5% of a fine march of the same segment.
#[test]
fn long_segments_match_a_fine_march() {
    let (mut bed, _fog) = foggy(
        &AIR,
        Vec::new(),
        vec![directional(10.0, 20.0, 1_000.0)],
        None,
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, -10.0),
    );
    let mut segments = Vec::new();
    for t in [5_000.0f32, 20_000.0] {
        for elevation in [0.5f32, 2.0, 5.0] {
            let e = elevation.to_radians();
            segments.push(segment_probe::MarchSegment {
                origin: [0.0, 0.0, -50.0],
                t,
                dir: [0.0, e.sin(), -e.cos()],
                _pad: 0.0,
            });
        }
    }
    let input = upload(&segments, "segments");
    let out = readback::<[f32; 4]>(2 * segments.len() as u64, "segment in-scatter");
    let p = pipeline(&segment_probe::MODULE, "main");
    let mut roots = Vec::new();
    bed.accumulate(1, 0, &mut |list, f| {
        let root = upload(
            &[segment_probe::Root {
                set: f.lighting.light_set(),
                segments: input.gpu(),
                out: out.gpu(),
                count: segments.len() as u32,
                steps: MediaSettings::default().segment_steps,
                jitters: 64,
                fine_steps: 4096,
                _pad0: 0,
                _pad1: 0,
            }],
            "segment probe root",
        );
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&p, root.gpu(), [1, 1, 1]);
        roots.push(root);
    });
    let lum = |p: [f32; 4]| luminance([p[0], p[1], p[2]].map(f64::from));
    for (s, pair) in segments.iter().zip(out.as_slice().unwrap().chunks(2)) {
        let (march, fine) = (lum(pair[0]), lum(pair[1]));
        assert!(fine > 0.0);
        assert!(
            (march - fine).abs() < 0.05 * fine,
            "over {} m rising {:.3}: the march gives {march:.3}, the fine march {fine:.3}",
            s.t,
            s.dir[1]
        );
    }
    assert_clean();
}

/// White particles 0.4 m across facing the camera, opaque.
const WHITE_PARTICLES: ParticleLook = ParticleLook {
    albedo: [0.8; 3],
    emission: [0.0; 3],
    opacity: 1.0,
    anisotropy: 0.0,
};

/// Two particles 3 m apart over the floor under the sun and the sky, the right one in the
/// shadow of a white sphere 3 m toward the sun from it.
fn sun_and_shade() -> (Spec, [Vec3; 2]) {
    let (lit, shaded) = (Vec3::new(-1.5, 1.0, 0.0), Vec3::new(1.5, 1.0, 0.0));
    let blocker = shaded + SUN.direction_to_light * 3.0;
    let spec = Spec {
        copies: vec![at(FLOOR, Vec3::ZERO), at(WHITE_SPHERE, blocker)],
        lights: vec![Light::Directional(SUN)],
        environment: Some(SKY),
        eye: Vec3::new(0.0, 1.2, 6.0),
        target: Vec3::new(0.0, 1.0, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    (spec, [lit, shaded])
}

/// A particle in the shadow of a sphere is darker than one in the sun, lit by the froxels
/// inside their range and by its own shadow ray and the world cache beyond it.
#[test]
fn particles_in_shade_are_darker_than_in_the_sun() {
    let paths = Paths::new(
        &KernelSet::default(),
        PathEntry::Fastest,
        ViewLighting::default(),
    );
    for near_range in [256.0, 2.0] {
        let (spec, points) = sun_and_shade();
        let mut bed = Bed::with(spec.clone());
        bed.set_near_range(near_range);
        let particles = points.map(|center| Particle { center, size: 0.2 });
        bed.set_particles(&particles, WHITE_PARTICLES, Opacity::Particle);
        let a = bed.accumulate(16, 5, &mut |list, f| paths.record(list, f));
        let [lit, shaded] = points.map(|p| luminance(a.around(pixel_of(&spec, p), 2)));
        assert!(
            lit > 0.0 && shaded < 0.6 * lit,
            "near_range {near_range}: in the sun {lit:.1}, in the shade {shaded:.1}"
        );
    }
    assert_clean();
}

/// An emissive opaque particle 40 m away in height fog under a uniform sky: what reaches the
/// eye from it is its emission through the fog's transmittance plus the sky's single
/// scattering over the segment in front of it, L (1 - T), not its emission pasted over the
/// fogged sky.
#[test]
fn particles_in_fog_take_the_fog_in_front_of_them() {
    let fog = Fog {
        density: 0.02,
        base_height: 0.0,
        falloff_height: 100.0,
    };
    let eye = Vec3::new(0.0, 1.0, 0.0);
    let target = Vec3::new(0.0, 1.0, -40.0);
    let emission = 2000.0f32;
    let (mut bed, _fog) = foggy(&fog, Vec::new(), Vec::new(), Some(SKY), eye, target);
    let look = ParticleLook {
        albedo: [0.0; 3],
        emission: [emission; 3],
        opacity: 1.0,
        anisotropy: 0.0,
    };
    bed.set_particles(
        &[Particle {
            center: target,
            size: 1.0,
        }],
        look,
        Opacity::Particle,
    );
    let mut paths = fog_paths();
    paths.settings = ViewLighting::default();
    let a = bed.accumulate(16, 3, &mut |list, f| paths.record(list, f));
    let got = luminance(a.around(pixel_of(&bed.spec, target), 2));
    let sigma = f64::from(fog.density) * (-f64::from(eye.y) / f64::from(fog.falloff_height)).exp();
    let transmittance = (-sigma * 40.0).exp();
    let sky = luminance(SKY.map(f64::from));
    let inscatter = sky * (1.0 - transmittance);
    let wanted = f64::from(emission) * transmittance + inscatter;
    assert!(
        (got - wanted).abs() <= 0.05 * wanted,
        "fogged particle {got:.1}, wanted {wanted:.1}: emission through the fog {:.1}, in-scatter {inscatter:.1}",
        f64::from(emission) * transmittance
    );
    assert_clean();
}

/// An emissive opaque particle 40 m away in height fog with nothing lit, the froxels ending at
/// 16 m: what reaches the eye is its emission through the fog's transmittance, the froxels'
/// to 16 m and the aerial-perspective LUT's past it.
#[test]
fn particles_past_the_froxels_take_the_aerial_transmittance() {
    let fog = Fog {
        density: 0.02,
        base_height: 0.0,
        falloff_height: 100.0,
    };
    let eye = Vec3::new(0.0, 1.0, 0.0);
    let target = Vec3::new(0.0, 1.0, -40.0);
    let emission = 2000.0f32;
    let (mut bed, _fog) = foggy(&fog, Vec::new(), Vec::new(), None, eye, target);
    bed.set_near_range(16.0);
    let look = ParticleLook {
        albedo: [0.0; 3],
        emission: [emission; 3],
        opacity: 1.0,
        anisotropy: 0.0,
    };
    bed.set_particles(
        &[Particle {
            center: target,
            size: 1.0,
        }],
        look,
        Opacity::Particle,
    );
    let mut paths = fog_paths();
    paths.settings = ViewLighting::default();
    let a = bed.accumulate(4, 3, &mut |list, f| paths.record(list, f));
    let got = luminance(a.around(pixel_of(&bed.spec, target), 2));
    let sigma = f64::from(fog.density) * (-f64::from(eye.y) / f64::from(fog.falloff_height)).exp();
    let wanted = f64::from(emission) * (-sigma * 40.0).exp();
    assert!(
        (got - wanted).abs() <= 0.03 * wanted,
        "fogged particle {got:.1}, its emission through the fog {wanted:.1}"
    );
    assert_clean();
}

/// At a still camera a sunlit particle in haze does not flicker: from one frame to the next it
/// changes by under 1% (root mean square), the noise its froxels' blended light keeps, even while
/// its level still drifts slowly.
#[test]
fn particles_stay_steady_at_a_still_camera() {
    let (spec, [lit, _]) = sun_and_shade();
    let (mut bed, _fog) = foggy(
        &HAZE,
        spec.copies.clone(),
        spec.lights.clone(),
        spec.environment,
        spec.eye,
        spec.target,
    );
    bed.set_particles(
        &[Particle {
            center: lit,
            size: 0.2,
        }],
        WHITE_PARTICLES,
        Opacity::Particle,
    );
    let mut paths = fog_paths();
    paths.settings = ViewLighting::default();
    let mut frames = Vec::new();
    for k in 0..48 {
        let a = bed.follow(spec.eye, spec.target, k, &mut |list, f| {
            paths.record(list, f);
        });
        if k >= 32 {
            frames.push(luminance(a.around(pixel_of(&spec, lit), 2)));
        }
    }
    let mean = frames.iter().sum::<f64>() / frames.len() as f64;
    let steps = frames.windows(2).map(|w| (w[1] - w[0]) * (w[1] - w[0]));
    let change = (steps.sum::<f64>() / (frames.len() - 1) as f64).sqrt() / mean;
    assert!(
        change < 0.01,
        "frame to frame the particle changes by {:.2}% around {mean:.1}",
        100.0 * change
    );
    assert_clean();
}

/// No medium outside the proxies.
const CLEAR: Fog = Fog {
    density: 0.0,
    base_height: 0.0,
    falloff_height: 100.0,
};

/// Transmittance from the eye per pixel, in the froxel debug view's red channel.
fn camera_transmittance(bed: &mut Bed, frames: u32) -> Vec<f64> {
    let lighting = ViewLighting {
        debug_view: LightingDebugView::Froxels,
        ..ViewLighting::default()
    };
    shown(bed, lighting, frames)
        .mean
        .iter()
        .map(|p| f64::from(p[0]))
        .collect()
}

fn plume_proxy(bed: &Bed, extinction: f32) -> MediumProxy {
    let v = bed.volume(PLUME);
    MediumProxy {
        position: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: 1.0,
        density: v.density(v.grid("density").unwrap()),
        extinction: Vec3::splat(extinction),
        albedo: Vec3::splat(0.9),
        anisotropy: 0.3,
    }
}

fn plume_view(fog: &Fog) -> (Bed, Allocation<[f32; 4]>) {
    foggy(
        fog,
        Vec::new(),
        vec![directional(40.0, 60.0, 1_000.0)],
        None,
        Vec3::new(0.0, 1.5, 5.0),
        Vec3::new(0.0, 1.5, 0.0),
    )
}

/// The plume in haze under a sun: each 8 x 8 block's in-scatter within 5% of the reference
/// camera march, whose steps trace their own shadow rays.
#[test]
fn a_plume_on_the_camera_ray_matches_the_reference_march() {
    let (mut bed, _fog) = plume_view(&HAZE);
    let proxy = plume_proxy(&bed, 1.0);
    bed.set_medium_proxy(MediumProxyId(0), Some(proxy));
    let media = MediaSettings {
        segment_steps: 32,
        ..MediaSettings::default()
    };
    let froxels = inscatter(&mut bed, false, media, 32);
    let reference = inscatter(&mut bed, true, media, 32);
    let (worst, mean) = block_errors(&froxels, &reference);
    assert!(
        worst < 0.05,
        "a block's in-scatter is {worst:.3} off the reference (mean {mean:.3})"
    );
    assert_clean();
}

/// The plume's transmittance per pixel within 5% of a fine march of its cooked grid along the
/// pixel's ray, with detail inside an 8 x 8 tile that the froxels could not hold.
#[test]
fn a_plume_keeps_detail_finer_than_a_froxel() {
    let (mut bed, _fog) = plume_view(&CLEAR);
    let proxy = plume_proxy(&bed, 1.5);
    bed.set_medium_proxy(MediumProxyId(0), Some(proxy));
    let got = camera_transmittance(&mut bed, 32);

    let (info, pages) = plume_volume();
    let sampler = GridSampler::new(&info, 0, &pages).unwrap();
    let [w, h] = EXTENT;
    let to_world = look(bed.spec.eye, bed.spec.target);
    let tan = (0.5 * bed.spec.fov_y).tan();
    let aspect = w as f32 / h as f32;
    let dt = 5e-3;
    let want: Vec<f64> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let ndc = [
                (x as f32 + 0.5) / w as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / h as f32 * 2.0,
            ];
            let dir = (to_world * Vec3::new(ndc[0] * tan * aspect, ndc[1] * tan, -1.0)).normalize();
            let tau: f64 = (0..800)
                .map(|k| {
                    let p = bed.spec.eye + dir * (3.0 + (k as f32 + 0.5) * dt);
                    f64::from(sampler.sample(p.to_array()).max(0.0))
                })
                .sum::<f64>()
                * f64::from(dt);
            (-1.5 * tau).exp()
        })
        .collect();

    let mut worst = (0.0f64, 0, 0.0, 0.0);
    for (i, (g, t)) in got.iter().zip(&want).enumerate() {
        let e = (g - t).abs() / t.max(0.05);
        if e > worst.0 {
            worst = (e, i, *g, *t);
        }
    }
    assert!(
        worst.0 <= 0.05,
        "pixel {}: transmittance {:.4}, a fine march's {:.4}",
        worst.1,
        worst.2,
        worst.3
    );
    let tile_spread = (0..h / 8)
        .flat_map(|ty| (0..w / 8).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| {
            let tile = (0..64).map(|k| want[((ty * 8 + k / 8) * w + tx * 8 + k % 8) as usize]);
            let (lo, hi) = tile.fold((1.0f64, 0.0f64), |(l, h), v| (l.min(v), h.max(v)));
            hi - lo
        })
        .fold(0.0f64, f64::max);
    assert!(tile_spread > 0.2, "the widest tile spans {tile_spread:.3}");
    assert_clean();
}

fn box_proxy(z: f32, half_z: f32, extinction: f32, albedo: f32) -> MediumProxy {
    MediumProxy {
        position: DVec3::new(0.0, 2.0, f64::from(z)),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        density: ProxyDensity::Uniform {
            half_extent: Vec3::new(4.0, 4.0, half_z),
        },
        extinction: Vec3::splat(extinction),
        albedo: Vec3::splat(albedo),
        anisotropy: 0.0,
    }
}

/// Transmittance at the middle of the view.
fn middle_transmittance(a: &[f64]) -> f64 {
    let [w, h] = EXTENT;
    a[(h / 2 * w + w / 2) as usize]
}

/// Boxes of [8, 12] m at extinction 0.2 and [11, 15] m at 0.1 along the center ray, in haze:
/// Beer's law over their spans and the haze's own, whether they overlap a surface or the eye
/// and whether they lie inside or past the froxels.
#[test]
fn proxy_boxes_on_the_camera_ray_take_beer_law() {
    let sigma = f64::from(HAZE.density) * (-2.0 / f64::from(HAZE.falloff_height)).exp();
    let sun = vec![directional(30.0, 20.0, 1_000.0)];
    let boxes = [
        box_proxy(-10.0, 2.0, 0.2, 0.5),
        box_proxy(-13.0, 2.0, 0.1, 0.5),
    ];
    let cases = [
        // Surface past the boxes, from outside and from inside the first, inside and past the
        // froxels.
        (
            20.0,
            vec![
                (256.0, 0.0, 0.2 * 4.0 + 0.1 * 4.0),
                (4.0, 0.0, 0.2 * 4.0 + 0.1 * 4.0),
                (256.0, -10.0, 0.2 * 2.0 + 0.1 * 4.0),
            ],
        ),
        // Surface inside their overlap.
        (
            11.5,
            vec![
                (256.0, 0.0, 0.2 * 3.5 + 0.1 * 0.5),
                (4.0, 0.0, 0.2 * 3.5 + 0.1 * 0.5),
            ],
        ),
    ];
    for (surface, views) in cases {
        let radius = 0.15 * surface;
        let copy = Item {
            model: WHITE_SPHERE,
            linear: Mat3::from_diagonal(Vec3::splat(radius)),
            at: Vec3::new(0.0, 2.0, -(surface + radius)),
        };
        let eye = Vec3::new(0.0, 2.0, 0.0);
        let target = Vec3::new(0.0, 2.0, -10.0);
        let (mut bed, _fog) = foggy(&HAZE, vec![copy], sun.clone(), None, eye, target);
        for (k, b) in boxes.iter().enumerate() {
            bed.set_medium_proxy(MediumProxyId(k as u32), Some(*b));
        }
        for (near_range, eye_z, tau) in views {
            bed.set_near_range(near_range);
            bed.spec.eye = Vec3::new(0.0, 2.0, eye_z);
            bed.spec.target = bed.spec.eye - Vec3::Z * 10.0;
            let got = middle_transmittance(&camera_transmittance(&mut bed, 8));
            let distance = f64::from(surface + eye_z);
            let want = (-tau - sigma * distance).exp();
            assert!(
                (got - want).abs() <= 0.01 * want,
                "near_range {near_range}, eye at {eye_z}, surface at {surface}: {got:.4}, wanted {want:.4}"
            );
        }
    }
    assert_clean();
}

/// A box lit by a sun in haze, in front of a surface: the in-scatter at the middle of the view
/// within 5% of the reference march inside the froxels, and within 10% past them, where the
/// aerial-perspective LUT's haze takes no shadow from the box.
#[test]
fn a_proxy_box_lights_like_the_reference_inside_and_past_the_froxels() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 2.0, -10.0);
    let sun = vec![directional(30.0, 20.0, 1_000.0)];
    let (mut bed, _fog) = foggy(&HAZE, vec![distant(20.0)], sun, None, eye, target);
    bed.set_medium_proxy(MediumProxyId(0), Some(box_proxy(-10.0, 2.0, 0.3, 0.9)));
    let media = MediaSettings {
        segment_steps: 32,
        ..MediaSettings::default()
    };
    for (near_range, tolerance) in [(256.0, 0.05), (4.0, 0.1)] {
        bed.set_near_range(near_range);
        let got = middle_blocks(&inscatter(&mut bed, false, media, 16));
        let reference = middle_blocks(&inscatter(&mut bed, true, media, 32));
        for (g, r) in got.iter().zip(&reference) {
            assert!(
                (g - r).abs() < tolerance * r,
                "near_range {near_range}: in-scatter {g:.2}, the reference's {r:.2}"
            );
        }
    }
    assert_clean();
}

/// On sky pixels the environment holds the medium past `near_range`. A box past it that lets
/// everything through leaves the in-scatter as without it, and a black box that lets nothing
/// through ends it as a surface at its near face does, with no light through.
#[test]
fn sky_pixels_past_the_froxels_keep_the_medium_in_front_of_a_box() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 2.0, -10.0);
    let sun = vec![directional(30.0, 20.0, 1_000.0)];
    let near_range = 4.0;
    let middle = |bed: &mut Bed, proxy: Option<MediumProxy>| {
        bed.set_near_range(near_range);
        bed.set_medium_proxy(MediumProxyId(0), proxy);
        let s = middle_blocks(&inscatter(bed, false, MediaSettings::default(), 16));
        let t = middle_transmittance(&camera_transmittance(bed, 4));
        (s[0], t)
    };
    let (mut open_sky, _fog) = foggy(&HAZE, Vec::new(), sun.clone(), None, eye, target);
    let (open, open_t) = middle(&mut open_sky, None);
    let (clear, clear_t) = middle(&mut open_sky, Some(box_proxy(-30.0, 10.0, 1e-7, 0.0)));
    assert!(
        (clear - open).abs() <= 0.01 * open && (clear_t - open_t).abs() <= 0.01 * open_t,
        "a clear box past the froxels: in-scatter {clear:.3} and transmittance {clear_t:.3}, without it {open:.3} and {open_t:.3}"
    );
    let (black, black_t) = middle(&mut open_sky, Some(box_proxy(-30.0, 10.0, 8.0, 0.0)));
    let (mut wall, _fog) = foggy(&HAZE, vec![distant(20.0)], sun, None, eye, target);
    let (surface, _) = middle(&mut wall, None);
    assert!(
        (black - surface).abs() <= 0.05 * surface && black_t < 1e-3,
        "a black box at 20 m: in-scatter {black:.3} and transmittance {black_t:.5}, a surface there {surface:.3}"
    );
    assert_clean();
}

/// Proxies behind the eye or beside the view leave media apply without its ray; one on screen,
/// or holding the eye, counts.
#[test]
fn only_proxies_in_the_frustum_count_as_visible() {
    let (mut bed, _fog) = foggy(
        &CLEAR,
        Vec::new(),
        Vec::new(),
        Some(SKY),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 2.0, -10.0),
    );
    let at = |x: f64, z: f64, half: f32| MediumProxy {
        position: DVec3::new(x, 2.0, z),
        ..box_proxy(0.0, half, 0.5, 0.5)
    };
    for (proxy, visible) in [
        (at(0.0, 20.0, 2.0), 0),
        (at(60.0, -10.0, 2.0), 0),
        (at(0.0, -20.0, 2.0), 1),
        (at(0.0, 0.0, 2.0), 1),
    ] {
        bed.set_medium_proxy(MediumProxyId(0), Some(proxy));
        let _ = inscatter(&mut bed, false, MediaSettings::default(), 2);
        let stats = bed.view().media_stats(bed.last_frame()).unwrap();
        assert_eq!(stats.visible_proxies, visible, "{proxy:?}");
    }
    assert_clean();
}

fn smoke_look(opacity: f32, albedo: f32) -> ParticleLook {
    ParticleLook {
        albedo: [albedo; 3],
        emission: [0.0; 3],
        opacity,
        anisotropy: 0.0,
    }
}

/// Smoke between the eye and the fog behind it takes from the camera ray what its opacity takes:
/// a sheet wider than the view, two of them, and specks far smaller than a froxel, whose share
/// of each 8 x 8 block is what they let through on average.
#[test]
fn smoke_takes_its_opacity_from_the_camera_ray() {
    let eye = Vec3::new(0.0, 1.5, 5.0);
    let target = Vec3::new(0.0, 1.5, 0.0);
    for (distances, wanted) in [(vec![4.0], 0.6), (vec![4.0, 6.0], 0.36)] {
        let (mut bed, _fog) = foggy(&CLEAR, Vec::new(), Vec::new(), None, eye, target);
        let sheets: Vec<Particle> = distances
            .iter()
            .map(|&d| Particle {
                center: eye + (target - eye).normalize() * d,
                size: 40.0,
            })
            .collect();
        bed.set_particles(&sheets, smoke_look(0.4, 0.0), Opacity::Smoke);
        let got = middle_transmittance(&camera_transmittance(&mut bed, 1));
        assert!(
            (got - wanted).abs() < 0.01,
            "{} sheets let through {got:.3}, not {wanted}",
            distances.len()
        );
    }

    // In a square 8 m away, 2 m across, a grid of specks 2 cm across.
    let n = 40;
    let half = 0.01;
    let specks: Vec<Particle> = (0..n * n)
        .map(|i| Particle {
            center: Vec3::new(
                ((i % n) as f32 + 0.5) / n as f32 * 2.0 - 1.0,
                1.5 + ((i / n) as f32 + 0.5) / n as f32 * 2.0 - 1.0,
                -3.0,
            ),
            size: half,
        })
        .collect();
    let opacity = 0.8;
    let (mut bed, _fog) = foggy(&CLEAR, Vec::new(), Vec::new(), None, eye, target);
    bed.set_particles(&specks, smoke_look(opacity, 0.0), Opacity::Smoke);
    let got = middle_transmittance(&camera_transmittance(&mut bed, 1));
    let share = f64::from((2.0 * half) * (2.0 * half) * (n * n) as f32 / 4.0);
    let wanted = 1.0 - share * f64::from(opacity);
    assert!(
        (got - wanted).abs() < 0.01,
        "specks covering {share:.3} let through {got:.3}, not {wanted:.3}"
    );
    assert_clean();
}

/// A column of smoke crossing a shaft under the canopy, against the reference camera march, which
/// takes each sheet on the ray with its own shadow ray. Averaged over the height of the view inside
/// the column, in-scatter is within 5%: froxels blur the sun's hard pattern on the sheets over
/// their columns, so single blocks there are not. Beside it every block is. Black smoke darkens
/// the shaft behind it; white smoke adds to it.
#[test]
fn smoke_across_a_shaft_matches_the_reference_march() {
    let eye = Vec3::new(0.0, 2.0, 0.0);
    let target = Vec3::new(0.0, 3.0, -10.0);
    let sun = vec![directional(10.0, 90.0, 1_000.0)];
    let canopy = vec![at(CANOPY, Vec3::new(0.0, 6.0, 0.0))];
    let dir = (target - eye).normalize();
    let half = 0.1;
    let column: Vec<Particle> = (0..16)
        .map(|k| 3.0 + 0.25 * k as f32)
        .flat_map(|d| {
            (-8..=8).map(move |y| Particle {
                center: eye + dir * d + Vec3::Y * (y as f32 * 2.0 * half * d / 5.0),
                size: half * d / 5.0,
            })
        })
        .collect();
    let (mut clear, _fog) = foggy(&DENSE, canopy.clone(), sun.clone(), None, eye, target);
    clear.spec.fov_y = 0.1;
    let open = block_luminance(&inscatter(&mut clear, false, MediaSettings::default(), 64));
    // The column covers about a third of the view's width around its center.
    let w = (EXTENT[0] / 8) as usize;
    let inside = |b: usize| (2 * (b % w) + 1).abs_diff(w) <= 1;
    let beside = |b: usize| (2 * (b % w) + 1).abs_diff(w) >= 12;
    let inside_mean = |v: &[f64]| {
        let kept: Vec<f64> = (0..v.len()).filter(|&b| inside(b)).map(|b| v[b]).collect();
        kept.iter().sum::<f64>() / kept.len() as f64
    };
    for (albedo, name) in [(0.0, "black"), (0.8, "white")] {
        let (mut bed, _fog) = foggy(&DENSE, canopy.clone(), sun.clone(), None, eye, target);
        bed.spec.fov_y = 0.1;
        bed.set_particles(&column, smoke_look(0.15, albedo), Opacity::Smoke);
        let froxels = block_luminance(&inscatter(&mut bed, false, MediaSettings::default(), 64));
        let reference = block_luminance(&inscatter(&mut bed, true, MediaSettings::default(), 256));
        let (f, r) = (inside_mean(&froxels), inside_mean(&reference));
        assert!(
            (f - r).abs() < 0.05 * r,
            "{name} smoke: in-scatter inside the column is {f:.3e}, the reference's {r:.3e}"
        );
        for b in (0..froxels.len()).filter(|&b| beside(b)) {
            let (f, r) = (froxels[b], reference[b]);
            assert!(
                (f - r).abs() < 0.05 * r.max(1e-9),
                "{name} smoke: block {b} beside the column is {f:.3e}, the reference's {r:.3e}"
            );
        }
        let without = inside_mean(&open);
        if albedo == 0.0 {
            assert!(
                f < 0.8 * without,
                "black smoke leaves {f:.3e} of {without:.3e}"
            );
        } else {
            assert!(
                f > without,
                "white smoke gives {f:.3e}, without it {without:.3e}"
            );
        }
    }
    assert_clean();
}
