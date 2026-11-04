//! Linked kernels through the path kernel, kept in one target because each target compiles
//! its own linked path pipelines: linked coverage against the equivalent alpha texture, and a
//! linked surface in a mirror and at each bounce.

use lighting_suite::bed::{
    Accumulated, BARE_LEAF, Bed, FLOOR, Item, MIRROR, PANEL, Paths, RINGS_LEAF, Spec, WHITE_SPHERE,
    against_noise, assert_close, luminance, mean_where, quad_scene, reference, surface_mean,
};
use lighting_suite::{assert_clean, linked};
use orr_lighting::{LightingDebugView, PathEntry, ReferenceSettings, ViewLighting};
use orr_math::{Mat3, Vec3};
use orr_scene::RendererDesc;

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
    let leaf = render_paths(&mut linked_bed(quad_scene(BARE_LEAF)), &paths, 32, 1);
    let noise = render_paths(&mut linked_bed(quad_scene(BARE_LEAF)), &paths, 32, 2);
    let rings = render_paths(&mut linked_bed(quad_scene(RINGS_LEAF)), &paths, 32, 1);
    let (e, floor) = against_noise(&rings, &leaf, &noise);
    assert!(
        e <= floor,
        "FLIP {e} with linked coverage against the texture, {floor} between seeds"
    );
    assert_clean();
}
