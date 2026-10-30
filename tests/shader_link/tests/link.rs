use orr_shaders::{KernelSet, Mismatch};
use shader_link::renderer::MODULES as RENDERER;
use shader_link::{AirData, edited, linked, unlinked};

const PRIMARY: u32 = 0;
const SECONDARY: u32 = 1;

#[test]
fn every_surface_method_reaches_the_type_linked_for_its_id() {
    use linked::surface::{GRASS, GROUND};
    for kind in [PRIMARY, SECONDARY] {
        for depth in [0, 3] {
            let call = |id| shader_link::surface(id, kind, depth, 0.01, 0.002);
            let ground = call(GROUND);
            assert_eq!(ground.albedo, [1.0, kind as f32, depth as f32, 0.01]);
            assert_eq!(ground.depth_offset, 0.125);
            assert_eq!(ground.coverage, if kind == PRIMARY { 0.75 } else { 0.5 });

            let grass = call(GRASS);
            assert_eq!(grass.albedo, [2.0, kind as f32, depth as f32, 0.002]);
            assert_eq!(grass.depth_offset, if depth == 0 { 0.5 } else { 0.0 });
            // Grass omits coverage, so the interface's body answers.
            assert_eq!(grass.coverage, 1.0);

            let standard = call(0);
            assert_eq!(standard.albedo, [0.5, 0.5, 0.5, 1.0]);
            assert_eq!(standard.depth_offset, 0.0);
            assert_eq!(standard.coverage, 0.25);
            assert_eq!(
                call(3),
                standard,
                "an id nothing links to falls back to the default"
            );
        }
    }
}

#[test]
fn displacement_environment_and_medium_reach_the_linked_types() {
    let (p, n) = ([3.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert_eq!(
        shader_link::displace(linked::displacement::WAVES, p, n),
        [0.0, 3.0, 2.0]
    );
    assert_eq!(shader_link::displace(0, p, n), [0.0, 0.0, 0.5]);
    assert_eq!(shader_link::environment([0.0, 0.25, 0.0]), [4.0, 0.25, 6.0]);

    let air = AirData {
        scattering: [0.01, 0.02, 0.04],
        absorption: 0.005,
        anisotropy: 0.7,
        pad: [0.0; 3],
    };
    let m = shader_link::medium(&air, 10.0);
    assert_eq!(&m[0..3], &air.scattering);
    assert_eq!(&m[3..6], &air.scattering.map(|s| s + air.absorption));
    assert_eq!(m[6], 0.7);
    for c in 0..3 {
        let expected = (-(air.scattering[c] + air.absorption) * 10.0).exp();
        assert!(
            (m[7 + c] - expected).abs() < 1e-6,
            "{} {expected}",
            m[7 + c]
        );
    }
}

#[test]
fn coherence_key_widens_with_the_highest_surface_id() {
    // Two surfaces: 3 + bit length of 2.
    assert_eq!(shader_link::key_bits(), 5);
}

#[test]
fn kernel_set_holds_the_kernels_that_declare_a_linked_point() {
    let mut kernels: Vec<&str> = linked::KERNELS.modules.iter().map(|m| m.kernel).collect();
    kernels.sort_unstable();
    assert_eq!(kernels, ["resolve", "sky", "tessellate"]);
    linked::KERNELS
        .check(RENDERER)
        .unwrap_or_else(|e| panic!("{e}"));
    for &own in RENDERER {
        let m = linked::KERNELS.module(own).unwrap();
        if own.kernel == "plain" {
            assert!(core::ptr::eq(m, own));
        } else {
            assert_ne!(m.spirv, own.spirv, "{}", own.kernel);
        }
    }
}

#[test]
fn nothing_linked_keeps_the_renderer_defaults() {
    assert!(unlinked::KERNELS.modules.is_empty());
    for &own in RENDERER {
        assert!(core::ptr::eq(unlinked::KERNELS.module(own).unwrap(), own));
    }
    assert!(KernelSet::default().check(RENDERER).is_ok());
}

#[test]
fn a_set_built_against_an_edited_root_is_rejected() {
    let e = edited::KERNELS.check(RENDERER).unwrap_err();
    assert_eq!(e.kernel, "resolve");
    assert!(matches!(e.reason, Mismatch::Interface { .. }), "{e}");
}
