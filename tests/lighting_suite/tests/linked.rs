//! Linked kernels through the path kernel, kept in one target because each target compiles
//! its own linked path pipelines: linked coverage against the equivalent alpha texture, and a
//! linked surface in a mirror, behind glass and at each bounce. A scene's data against its linked
//! types. Linked surfaces and coverage, geometry kinds and rich materials in real time against
//! the reference.

use lighting_suite::bed::{
    Accumulated, BARE_LEAF, Bed, CONVERGES, FLOOR, GLASS_SLAB, Item, LAYERED_SPHERE, MIRROR, PANEL,
    Particle, ParticleLook, Paths, RINGS_LEAF, SKINNED_EMITTER, SKY, SUN, Spec, StrandRow,
    WHITE_SPHERE, assert_close, assert_converges, assert_within_noise, luminance, mean_where,
    quad_scene, reference, render_paths, surface_mean,
};
use lighting_suite::{TestScene, assert_clean, linked};
use orr_gpu::{GpuPtr, KernelSet};
use orr_lighting::{EnvironmentMapGpu, Light, LightingDebugView, ReferenceSettings, ViewLighting};
use orr_math::{Affine3, Mat3, Vec3};
use orr_scene::{EnvironmentData, RendererDesc, SceneDesc, SceneWriter};

fn at(model: &'static str, linear: Mat3, at: Vec3) -> Item {
    Item { model, linear, at }
}

fn debug(view: LightingDebugView, max_bounces: u32) -> ViewLighting {
    ViewLighting {
        debug_view: view,
        ..reference(max_bounces)
    }
}

/// The bed with `spec`, its panels' surface linked as `surfaces/context.slang`.
fn linked_bed(spec: Spec) -> Bed {
    Bed::with_renderer(
        spec,
        &RendererDesc {
            kernels: &linked::KERNELS,
            ..RendererDesc::default()
        },
    )
}

fn linked_paths(settings: ViewLighting) -> Paths {
    Paths::new(&linked::KERNELS, settings)
}

/// `Dome`, a linked environment that reads no record.
struct DomeData;

impl EnvironmentData for DomeData {
    const TYPE: &'static str = "Dome";
    const NULL_SAFE: bool = true;
}

/// A record `Dome` would read, if it read one.
struct DomeRecord;

impl EnvironmentData for DomeRecord {
    const TYPE: &'static str = "Dome";
}

static DOME: KernelSet = KernelSet {
    environment: DomeData::TYPE,
    ..KernelSet::NONE
};

/// One frame of a scene on [`DOME`], with `data` setting its linked data.
fn dome_frame(data: impl FnOnce(&mut SceneWriter<'_>) + 'static) {
    TestScene::linked(SceneDesc::default(), &DOME, data).write(|_, _| {});
}

#[test]
fn null_data_serves_a_linked_type_that_reads_none() {
    dome_frame(|w| w.set_environment(GpuPtr::<DomeData>::NULL));
    assert_clean();
}

#[test]
#[should_panic(
    expected = "the kernels link Dome as the environment; this data is for EnvironmentMap"
)]
fn data_for_another_type_panics() {
    dome_frame(|w| w.set_environment(GpuPtr::<EnvironmentMapGpu>::NULL));
}

#[test]
#[should_panic(expected = "Dome reads its data, and this data is null")]
fn null_data_for_a_type_that_reads_it_panics() {
    dome_frame(|w| w.set_environment(GpuPtr::<DomeRecord>::NULL));
}

#[test]
#[should_panic(expected = "has no data for the environment its kernels link: call set_environment")]
fn a_linked_environment_never_set_panics() {
    dome_frame(|_| {});
}

#[test]
fn a_linked_surface_looks_the_same_in_a_mirror() {
    let panel_facing_back = Mat3::from_rotation_y(std::f32::consts::PI);
    let direct = Spec {
        copies: vec![at(PANEL, Mat3::IDENTITY, Vec3::new(0.0, 1.0, 0.0))],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 1.0, 3.0),
        target: Vec3::new(0.0, 1.0, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let mirrored = Spec {
        copies: vec![
            at(MIRROR, Mat3::IDENTITY, Vec3::new(0.0, 1.0, 0.0)),
            at(PANEL, panel_facing_back, Vec3::new(0.0, 1.0, 6.0)),
        ],
        ..direct.clone()
    };
    let color = |spec| {
        let mut bed = linked_bed(spec);
        let p = linked_paths(reference(2));
        let image = render_paths(&mut bed, &p, 4, 1);
        let m = mean_where(&image, |p| luminance(p) > 1.0);
        let total = m[0] + m[1] + m[2];
        m.map(|c| c / total)
    };
    let seen = color(direct);
    let reflected = color(mirrored);
    assert!(
        seen[0] > 0.7,
        "the panel seen directly is not red: {seen:?}"
    );
    assert_close("panel in the mirror", reflected, seen, 0.02);
    assert_clean();
}

/// The panel behind glass is reached only by the layer kernels' rays, whose first-layer ray and
/// guides resolve through the linked surface. The slab's two faces put it at depth 2.
#[test]
fn a_linked_surface_shows_through_glass() {
    let spec = Spec {
        copies: vec![
            at(PANEL, Mat3::IDENTITY, Vec3::new(0.0, 1.0, 0.0)),
            at(
                GLASS_SLAB,
                Mat3::from_rotation_x(std::f32::consts::FRAC_PI_2),
                Vec3::new(0.0, 1.0, 1.5),
            ),
        ],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 1.0, 3.0),
        target: Vec3::new(0.0, 1.0, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let mut bed = linked_bed(spec);
    let image = render_paths(&mut bed, &linked_paths(reference(4)), 4, 1);
    let m = mean_where(&image, |p| luminance(p) > 1.0);
    let total = m[0] + m[1] + m[2];
    assert!(
        m[1] > 0.7 * total,
        "the panel behind glass is not in its far form: {m:?}"
    );
    assert_clean();
}

#[test]
fn a_linked_surface_switches_form_only_from_the_second_bounce() {
    let spec = Spec {
        copies: vec![
            at(FLOOR, Mat3::IDENTITY, Vec3::ZERO),
            at(PANEL, Mat3::IDENTITY, Vec3::new(0.0, 0.8, -1.0)),
            at(WHITE_SPHERE, Mat3::IDENTITY, Vec3::new(1.6, 1.0, 0.0)),
        ],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 2.0, 4.0),
        target: Vec3::new(0.0, 0.5, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let mut bed = linked_bed(spec);
    let mut light = |view, specialize| {
        let mut p = linked_paths(debug(view, 3));
        p.specialize = specialize;
        render_paths(&mut bed, &p, 32, 1)
    };
    let direct = light(LightingDebugView::DirectOnly, false);
    let indirect = light(LightingDebugView::IndirectOnly, false);
    // The scene's own pipelines shade a linked surface's vertex in the stage that holds the
    // linked surfaces, not in the plain opaque stage, and render the same image.
    let bits =
        |a: &Accumulated| -> Vec<[u32; 4]> { a.mean.iter().map(|p| p.map(f32::to_bits)).collect() };
    assert!(
        bits(&light(LightingDebugView::IndirectOnly, true)) == bits(&indirect),
        "the scene's own pipelines changed the image"
    );
    let (direct, indirect) = (surface_mean(&direct), surface_mean(&indirect));
    assert!(
        direct[1] < 0.15 * direct[0],
        "light from the first bounce holds the far form: {direct:?}"
    );
    assert!(
        indirect[0] < 0.15 * indirect[1],
        "light from later bounces holds the near form: {indirect:?}"
    );
    assert_clean();
}

#[test]
fn linked_coverage_casts_the_shadow_of_the_equivalent_alpha_texture() {
    let paths = linked_paths(ViewLighting {
        max_bounces: 1,
        reference: Some(ReferenceSettings::default()),
        debug_view: LightingDebugView::DirectOnly,
        ..ViewLighting::default()
    });
    let render = |model, seed| {
        let mut bed = linked_bed(quad_scene(model));
        // At a pre-exposure of 1 the sunlit ground passes the HDR ceiling, which would clip the
        // samples that draw the sun.
        bed.set_pre_exposure(1.0 / 64.0);
        render_paths(&mut bed, &paths, 32, seed)
    };
    let leaf = render(BARE_LEAF, 1);
    let noise = render(BARE_LEAF, 2);
    let rings = render(RINGS_LEAF, 1);
    // The linked coverage is the texture's alpha test, so the two differ as two seeds do. The
    // sampler's fixed-point bilinear weights still move a pixel whose center reads within a
    // weight step of the cutoff to the other side.
    assert_within_noise(
        "linked coverage against the texture",
        &rings,
        &leaf,
        &noise,
        1.1,
    );
    assert_clean();
}

/// Real-time paths converge to the reference under the sun and the sky through linked types,
/// geometry kinds and rich materials: a surface of its own (`Context`, whose resolve the pipeline
/// makes for its id) beside a floor and a coated, fuzzy, filmed sphere, the rings quad of linked
/// coverage (its id's any-hit shaders) casting its shadow, a skinned emitter and a row of strands;
/// then a producer entry's opaque quads with a back of their own.
#[test]
fn real_time_paths_converge_through_linked_types_geometry_kinds_and_rich_materials() {
    let real_time = linked_paths(ViewLighting::default());
    let truth_paths = linked_paths(reference(12));
    let lit = |copies: Vec<Item>| Spec {
        copies: [vec![at(FLOOR, Mat3::IDENTITY, Vec3::ZERO)], copies].concat(),
        lights: vec![Light::Directional(SUN)],
        environment: Some(SKY),
        eye: Vec3::new(0.3, 1.6, 4.2),
        target: Vec3::new(0.0, 0.7, 0.0),
        fov_y: 0.9,
        focus: None,
    };
    let mixed = linked_bed(lit(vec![
        at(PANEL, Mat3::IDENTITY, Vec3::new(-1.4, 0.7, -0.4)),
        at(
            LAYERED_SPHERE,
            Mat3::from_diagonal(Vec3::splat(0.45)),
            Vec3::new(1.3, 0.45, 0.2),
        ),
        at(
            RINGS_LEAF,
            Mat3::from_diagonal(Vec3::splat(0.6)),
            Vec3::new(0.0, 0.7, -1.2),
        ),
    ]));
    let quads = linked_bed(lit(Vec::new()));
    let mut beds = [
        ("linked, layered, skinned and strands", mixed),
        ("producer quads", quads),
    ];
    {
        let bed = &mut beds[0].1;
        bed.add_skinned(
            SKINNED_EMITTER,
            Affine3::from_translation(Vec3::new(0.6, 1.5, 1.0)),
            &[Affine3::IDENTITY, Affine3::from_rotation_z(-0.5)],
        );
        bed.set_strands(StrandRow {
            count: 12,
            first: Vec3::new(-0.7, 0.0, 1.0),
            step: Vec3::new(0.09, 0.0, 0.0),
            height: 0.8,
            radius: 0.02,
            albedo: [0.6, 0.45, 0.2],
        });
    }
    {
        let quad = |x: f32, y: f32| Particle {
            center: Vec3::new(x, y, 0.0),
            size: 0.35,
        };
        let look = |albedo: [f32; 3]| ParticleLook {
            albedo,
            emission: [0.0; 3],
            opacity: 1.0,
            anisotropy: 0.0,
        };
        beds[1].1.set_sided_quads(
            &[quad(-0.9, 0.5), quad(0.0, 0.9), quad(0.9, 0.5)],
            look([0.7, 0.6, 0.2]),
            look([0.1, 0.3, 0.7]),
            Affine3::IDENTITY,
        );
    }
    for (what, bed) in &mut beds {
        // At a pre-exposure of 1 the sunlit ground passes the HDR ceiling, which would clip the
        // samples that draw the sun.
        bed.set_pre_exposure(1.0 / 64.0);
        let truth = render_paths(bed, &truth_paths, 1024, 7);
        assert_converges(what, bed, &real_time, &truth, CONVERGES);
    }
    assert_clean();
}

/// The most a scene's own pipeline may meet: every bit but the last of `path_may2`, which no
/// feature sets. Only the pipeline that may meet everything has that one.
const SCENE_MOST: u64 = u64::MAX >> 1;

/// The largest pipeline a scene's own mask makes, from the kernels that link the suite's surface
/// types: the standard surface's stages hold their caps as where nothing links, and a stage made
/// for one linked type holds that type's code alone. One that shades its own hits is the smaller
/// for the standard surface it leaves out. One that only covers shades with the standard
/// surface's closest-hits, and one that only shades covers with the standard surface's any-hits,
/// so neither has a copy of a standard stage of its own. Neither declares its other side, so no
/// stage serves any type.
#[test]
fn the_stages_of_a_pipeline_with_linked_surfaces_hold_their_caps() {
    use linked::surface::{CONTEXT, RINGS};
    let sizes = orr_lighting::path_stage_sizes(&linked::KERNELS, SCENE_MOST).unwrap();
    assert_eq!(orr_lighting::over_cap(&sizes), None);
    let own = |entry: &str, surface: u8| {
        let stage = sizes
            .iter()
            .find(|s| (s.entry, s.geometry, s.hit_surface) == (entry, ".copies", surface.into()));
        stage
            .and_then(|s| s.own)
            .expect("a stage made for the type")
    };
    assert!(own("surface_hit", CONTEXT) < 0);
    assert!((1..1500).contains(&own("cover_hit", RINGS)));
    let made_for = |surface: u8, role: orr_lighting::Role| {
        sizes
            .iter()
            .filter(|s| s.role == role && s.hit_surface == u32::from(surface))
            .count()
    };
    assert_eq!(made_for(RINGS, orr_lighting::Role::Hit), 0);
    assert_eq!(made_for(CONTEXT, orr_lighting::Role::AnyHit), 0);
    // Where any record's own weights may give it a lobe, a type that shades every hit itself
    // leaves them out.
    let rich = sizes
        .iter()
        .find(|s| {
            (s.entry, s.geometry, s.hit_surface) == ("surface_hit", ".copies", CONTEXT.into())
        })
        .map(|s| s.rich);
    assert_eq!(rich, Some(false));
    assert!(
        sizes
            .iter()
            .all(|s| s.hit_surface != u32::MAX || s.geometry.is_empty())
    );
}

/// The pipeline that may meet everything stands in for any scene, linked types and all, so it
/// shades every linked surface through the stage made for any id, one per kind and kernel, and
/// holds the pipeline's cap however many types link: it holds no stage made for one type.
#[test]
fn the_pipeline_that_may_meet_everything_holds_no_stage_per_linked_type() {
    let sizes = orr_lighting::path_stage_sizes(&linked::KERNELS, u64::MAX).unwrap();
    assert_eq!(orr_lighting::over_cap(&sizes), None);
    let shaded = |s: &&orr_lighting::StageSize| !s.geometry.is_empty();
    assert!(
        sizes
            .iter()
            .filter(shaded)
            .all(|s| s.hit_surface == u32::MAX)
    );
    let scene = orr_lighting::path_stage_sizes(&linked::KERNELS, SCENE_MOST).unwrap();
    assert!(
        sizes.iter().filter(shaded).count() < scene.iter().filter(shaded).count(),
        "the general pipeline's hit stages against a scene's"
    );
}
