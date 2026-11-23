//! The code of the path tracer that not every scene runs: the pipelines made for exactly
//! what a scene and its settings may meet hold it behind each of its `specialize.slang` bits.

use std::any::Any;

use lighting_suite::bed::{
    Analytic, Bed, EXTENT, FrostGuide, GLOSSY_FLOOR, Gate, Item, NEAR, Paths, SUN, Spec, WATER_BOX,
    assert_gate, glass_scene, light_bed, particle_bed, plume_bed, producer_slab_bed, reference,
    shaped_bed, slab_spec, unshadowed_bed,
};
use lighting_suite::{assert_clean, texture, upload};
use orr_gpu::KernelSet;
use orr_lighting::path_may::{
    MAY_BACKGROUND, MAY_CACHE, MAY_CAUSTICS, MAY_COVERAGE, MAY_DEBUG, MAY_DIRECTIONAL,
    MAY_DISPERSION, MAY_EMISSIVE, MAY_FROST, MAY_INTERIORS, MAY_LAYERS, MAY_LENS, MAY_LOCAL_LIGHTS,
    MAY_PARTICLES, MAY_PHOTON, MAY_PROXIES, MAY_REFERENCE, MAY_SCATTERING, MAY_SUBSURFACE,
    MAY_UPDATE, MAY_VIEW, MAY_WALK,
};
use orr_lighting::{
    DofBackgroundPass, Light, LightingDebugView, ReferenceSettings, ThinLens, ViewLighting,
};
use orr_math::{Mat3, Vec3};
use orr_scene::SceneFeatures;

/// A gate of [`gates`]: [`Gate`] with a scene and a setup that capture nothing.
struct Row {
    what: &'static str,
    held: Vec<u64>,
    bed: fn() -> Bed,
    setup: fn(&mut Bed, &mut Paths) -> Box<dyn Any>,
    settings: ViewLighting,
}

/// The bit of `path_may2`, in the high word, that `feature` decides.
fn may2(feature: SceneFeatures) -> u64 {
    u64::from(!SceneFeatures(!feature.0).surface_may()) << 32
}

/// A scene's frames at a pre-exposure that keeps sunlit ground under the HDR ceiling.
fn dim(bed: &mut Bed, _: &mut Paths) -> Box<dyn Any> {
    bed.set_pre_exposure(1.0 / 64.0);
    Box::new(())
}

/// The code of the path tracer that not every scene runs, by the `path_may` and
/// `path_may2` bits it sits behind, each with a scene whose paths run it.
fn gates() -> Vec<Row> {
    let view = u64::from(MAY_VIEW);
    let manifolds = may2(SceneFeatures::INTERIORS | SceneFeatures::DIRECTIONAL_LIGHTS);
    let real_time = ViewLighting::default();
    let row = |what, held: &[u64], bed, setup, settings| Row {
        what,
        held: held.to_vec(),
        bed,
        setup,
        settings,
    };
    let plain: fn(&mut Bed, &mut Paths) -> Box<dyn Any> = |_, _| Box::new(());
    let in_water: fn(&mut Bed, &mut Paths) -> Box<dyn Any> = |bed, paths| {
        paths.settings.camera_medium = Some(bed.material(WATER_BOX));
        dim(bed, paths)
    };
    let thin_lens = ViewLighting {
        reference: Some(ReferenceSettings {
            lens: Some(ThinLens {
                aperture_radius: 0.02,
                focus_distance: 2.0,
            }),
            ..ReferenceSettings::default()
        }),
        ..reference(4)
    };
    vec![
        row(
            "the sun's disc in a glossy floor",
            &[
                view | u64::from(MAY_DIRECTIONAL | MAY_CACHE),
                u64::from(MAY_UPDATE),
            ],
            || {
                // The eye looks down the sun's reflection, so the floor's BSDF rays meet its disc.
                let to_sun = SUN.direction_to_light;
                Bed::with(Spec {
                    copies: vec![Item {
                        model: GLOSSY_FLOOR,
                        linear: Mat3::IDENTITY,
                        at: Vec3::ZERO,
                    }],
                    lights: vec![Light::Directional(SUN)],
                    environment: None,
                    eye: 4.0 * Vec3::new(-to_sun.x, to_sun.y, -to_sun.z),
                    target: Vec3::ZERO,
                    fov_y: 0.8,
                    focus: None,
                })
            },
            dim,
            real_time,
        ),
        row(
            "emissive quad",
            &[view | u64::from(MAY_EMISSIVE)],
            || Bed::new(Analytic::CornellBox),
            plain,
            real_time,
        ),
        row(
            "alpha-tested quad",
            &[view | u64::from(MAY_COVERAGE)],
            || Bed::new(Analytic::AlphaQuad),
            dim,
            real_time,
        ),
        row(
            "subsurface walks",
            &[view | u64::from(MAY_WALK | MAY_SUBSURFACE)],
            || Bed::new(Analytic::SubsurfaceSlab),
            dim,
            real_time,
        ),
        row(
            "rays of subsurface walks",
            &[view | u64::from(MAY_DEBUG | MAY_WALK)],
            || Bed::new(Analytic::SubsurfaceSlab),
            dim,
            ViewLighting {
                debug_view: LightingDebugView::RaysPerPixel,
                ..real_time
            },
        ),
        row(
            "the world cache at the first vertex",
            &[view | u64::from(MAY_DEBUG)],
            || Bed::new(Analytic::SunAndGround),
            dim,
            ViewLighting {
                debug_view: LightingDebugView::WorldCache,
                ..real_time
            },
        ),
        row(
            "manifold walk through solid glass",
            &[view | u64::from(MAY_LAYERS) | manifolds],
            || Bed::with(slab_spec(false)),
            dim,
            real_time,
        ),
        row(
            "coated emitter",
            &[view | u64::from(MAY_EMISSIVE) | may2(SceneFeatures::COATED_EMITTERS)],
            || Bed::with(slab_spec(true)),
            dim,
            real_time,
        ),
        row(
            "scattering interiors",
            &[view | u64::from(MAY_INTERIORS | MAY_LAYERS | MAY_SCATTERING)],
            || Bed::with(glass_scene("milk")),
            dim,
            real_time,
        ),
        row(
            "thin panes",
            &[view | u64::from(MAY_LAYERS) | may2(SceneFeatures::THIN_DIELECTRICS)],
            || Bed::with(glass_scene("pane")),
            dim,
            real_time,
        ),
        row(
            "dispersive sphere",
            &[view | u64::from(MAY_DISPERSION)],
            || Bed::with(glass_scene("prism")),
            dim,
            real_time,
        ),
        row(
            "frosted slab",
            &[view | u64::from(MAY_FROST)],
            || Bed::with(glass_scene("frosted")),
            |bed, paths| {
                paths.frost_guide = Some(FrostGuide::new(0.25));
                dim(bed, paths)
            },
            real_time,
        ),
        row(
            "caustic of a sphere",
            &[view | u64::from(MAY_CAUSTICS), u64::from(MAY_PHOTON)],
            || Bed::with(glass_scene("lens")),
            dim,
            real_time,
        ),
        row(
            "the reference's photon map",
            &[
                view | u64::from(MAY_REFERENCE | MAY_CAUSTICS),
                u64::from(MAY_PHOTON),
            ],
            || Bed::with(glass_scene("lens")),
            dim,
            reference(4),
        ),
        row(
            "eye in water",
            &[
                view | u64::from(MAY_INTERIORS),
                u64::from(MAY_UPDATE | MAY_INTERIORS),
            ],
            || Bed::with(glass_scene("water")),
            in_water,
            real_time,
        ),
        row(
            "lens in water",
            &[view | u64::from(MAY_LENS | MAY_REFERENCE | MAY_INTERIORS)],
            || Bed::with(glass_scene("water")),
            in_water,
            thin_lens,
        ),
        row(
            "particles",
            &[view | u64::from(MAY_PARTICLES)],
            particle_bed,
            plain,
            real_time,
        ),
        row(
            "sphere and rect lights",
            &[view | u64::from(MAY_LOCAL_LIGHTS)],
            light_bed,
            plain,
            real_time,
        ),
        row(
            "smoke plume",
            &[view | u64::from(MAY_PROXIES)],
            plume_bed,
            dim,
            real_time,
        ),
        row(
            "the background layer",
            &[u64::from(MAY_BACKGROUND)],
            || Bed::new(Analytic::SunAndGround),
            |bed, paths| {
                // The layer is read from frames whose paths record it, past the exposure's own.
                dim(bed, paths);
                bed.accumulate_background_layer(true);
                // The left half's background rays start past the near plane.
                let mask: Vec<[f32; 4]> = (0..EXTENT[0] * EXTENT[1])
                    .map(|i| {
                        let start = if i % EXTENT[0] < EXTENT[0] / 2 {
                            NEAR / 2.0
                        } else {
                            0.0
                        };
                        [start, 0.0, 0.0, 0.0]
                    })
                    .collect();
                let mask = texture(EXTENT[0], EXTENT[1], &mask, "dof mask");
                let focus = upload(&[5.0f32], "focus distance");
                paths.dof_background = Some(DofBackgroundPass {
                    mask: mask.index,
                    aperture_radius: 0.05,
                    focus_distance: focus.gpu(),
                });
                Box::new((mask, focus))
            },
            reference(2),
        ),
        row(
            "a light a cone shapes",
            &[view | u64::from(MAY_LOCAL_LIGHTS) | may2(SceneFeatures::SHAPED_LIGHTS)],
            shaped_bed,
            plain,
            real_time,
        ),
        row(
            "a light surfaces cast no shadow from",
            &[view | u64::from(MAY_LOCAL_LIGHTS) | may2(SceneFeatures::UNSHADOWED_LIGHTS)],
            unshadowed_bed,
            plain,
            real_time,
        ),
        row(
            "manifold walk through solid glass on a producer entry",
            &[view | u64::from(MAY_LAYERS) | manifolds | may2(SceneFeatures::PRODUCERS)],
            producer_slab_bed,
            dim,
            real_time,
        ),
    ]
}

/// The gates of [`gates`] in `run`, each by [`assert_gate`].
fn gate_rows(run: std::ops::Range<usize>) {
    for row in gates().into_iter().skip(run.start).take(run.len()) {
        assert_gate(
            &KernelSet::default(),
            &Gate {
                what: row.what,
                held: &row.held,
                bed: &row.bed,
                setup: &row.setup,
                settings: row.settings,
            },
        );
    }
    assert_clean();
}

/// The pipelines made for exactly what a scene and its settings may meet hold the code its
/// paths run behind each gate of [`gates`]: on opaque scenes.
#[test]
fn specialized_pipelines_launch_the_paths_the_general_one_does() {
    gate_rows(0..8);
}

/// As above, through glass.
#[test]
fn specialized_pipelines_launch_the_paths_the_general_one_does_through_glass() {
    gate_rows(8..16);
}

/// As above, on what lies along a ray, and from the background layer.
#[test]
fn specialized_pipelines_launch_the_paths_the_general_one_does_through_layers() {
    gate_rows(16..20);
}

/// As above, through a producer entry's glass, and from a shaped light and a light surfaces
/// cast no shadow from.
#[test]
fn specialized_pipelines_launch_the_paths_the_general_one_does_through_producer_glass() {
    gate_rows(20..23);
}
