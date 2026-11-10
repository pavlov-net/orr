//! Linked kernels through the path kernel, kept in one target because each target compiles
//! its own linked path pipelines: linked coverage against the equivalent alpha texture, and a
//! linked surface in a mirror, behind glass and at each bounce. A scene's data against its linked
//! types.

use lighting_suite::bed::{
    Accumulated, BARE_LEAF, Bed, FLOOR, GLASS_SLAB, Item, MIRROR, PANEL, Paths, RINGS_LEAF, Spec,
    WHITE_SPHERE, assert_close, assert_within_noise, luminance, mean_where, quad_scene, reference,
    surface_mean,
};
use lighting_suite::{TestScene, assert_clean, linked};
use orr_gpu::{GpuPtr, KernelSet};
use orr_lighting::{
    EnvironmentMapGpu, LightingDebugView, PathEntry, ReferenceSettings, ViewLighting,
};
use orr_math::{Mat3, Vec3};
use orr_scene::{EnvironmentData, RendererDesc, SceneDesc, SceneWriter};

fn at(model: &'static str, linear: Mat3, at: Vec3) -> Item {
    Item { model, linear, at }
}

fn render_paths(bed: &mut Bed, paths: &Paths, frames: u32, seed: u32) -> Accumulated {
    bed.accumulate(frames, seed, &mut |list, f| paths.record(list, f))
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
    Paths::new(&linked::KERNELS, PathEntry::Compute, settings)
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
    let mut light = |view| {
        let p = linked_paths(debug(view, 3));
        surface_mean(&render_paths(&mut bed, &p, 32, 1))
    };
    let direct = light(LightingDebugView::DirectOnly);
    let indirect = light(LightingDebugView::IndirectOnly);
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
