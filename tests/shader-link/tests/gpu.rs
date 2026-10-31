//! The linked switches dispatched on the GPU and read back, and `orr::Renderer::new` checking
//! kernel sets built against orr's own kernels.

use orr::{Renderer, RendererDesc};
use orr_gpu::{
    Allocation, ComputePipeline, GpuPtr, Hazards, MemoryKind, QueueKind, ShaderModule, ShaderStage,
    SpecData, Stages,
};
use orr_scene::testing::{assert_clean, device, wait};
use orr_shaders::bytemuck::Pod;
use orr_shaders::{KernelSet, Mismatch};
use shader_link::renderer::types::{DisplacementInput, HitContext, SurfaceInput};
use shader_link::renderer::{resolve, sky, tessellate};
use shader_link::{AirData, linked, orr_edited, orr_linked, unlinked};

fn pipeline(module: &'static ShaderModule) -> ComputePipeline {
    let stage = ShaderStage::new(module, "main", SpecData::default()).unwrap();
    device()
        .create_compute_pipeline(&stage, module.name)
        .unwrap()
}

fn upload<T: Pod>(values: &[T]) -> Allocation<T> {
    let mut b = device()
        .gpu_malloc::<T>(MemoryKind::Upload, values.len() as u64, "inputs")
        .unwrap();
    b.as_mut_slice().unwrap().copy_from_slice(values);
    b
}

/// Runs one workgroup of `set`'s module in place of `own` with `root`, then returns `count`
/// outputs.
fn run<R: Pod, O: Pod>(
    set: &KernelSet,
    own: &'static ShaderModule,
    count: u64,
    root: impl FnOnce(GpuPtr<O>) -> R,
) -> Vec<O> {
    let d = device();
    let out = d
        .gpu_malloc::<O>(MemoryKind::Readback, count, "out")
        .unwrap();
    let root = upload(&[root(out.gpu)]);
    let p = pipeline(set.module(own).unwrap());
    let render = d.queue(QueueKind::Render).unwrap();
    let mut cmd = render.command_list().unwrap();
    cmd.dispatch(&p, root.gpu, [1, 1, 1]);
    cmd.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![cmd], &[]).unwrap());
    out.as_slice().unwrap().to_vec()
}

const PRIMARY: u32 = 0;
const SECONDARY: u32 = 1;

/// Every surface id 0-3 with each hit kind and depths 0 and 3.
fn surface_inputs() -> Vec<SurfaceInput> {
    let mut v = Vec::new();
    for surface in 0..4 {
        for kind in [PRIMARY, SECONDARY] {
            for depth in [0, 3] {
                v.push(SurfaceInput {
                    context: HitContext {
                        kind,
                        depth,
                        cone_width: 0.01,
                        cone_spread: 0.002,
                    },
                    surface,
                    material: 0,
                    data: 0,
                });
            }
        }
    }
    v
}

fn resolve_on_gpu(set: &KernelSet) -> Vec<(SurfaceInput, resolve::SurfaceOut)> {
    let inputs = surface_inputs();
    let count = inputs.len() as u32;
    let gpu_inputs = upload(&inputs);
    let out = run(set, &resolve::MODULE, count.into(), |out| resolve::Root {
        inputs: gpu_inputs.gpu,
        out,
        count,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    });
    inputs.into_iter().zip(out).collect()
}

#[test]
fn every_surface_method_reaches_the_type_linked_for_its_id() {
    use linked::surface::{GRASS, GROUND};
    for (i, o) in resolve_on_gpu(&linked::KERNELS) {
        let c = i.context;
        let (kind, depth) = (c.kind as f32, c.depth as f32);
        let (albedo, depth_offset, coverage) = match i.surface {
            s if s == u32::from(GROUND) => (
                [1.0, kind, depth, c.cone_width],
                0.125,
                if c.kind == PRIMARY { 0.75 } else { 0.5 },
            ),
            // Grass omits coverage, so the interface's body answers.
            s if s == u32::from(GRASS) => (
                [2.0, kind, depth, c.cone_spread],
                if c.depth == 0 { 0.5 } else { 0.0 },
                1.0,
            ),
            // StandardSurface, for id 0 and for an id nothing links to.
            _ => ([0.5, 0.5, 0.5, 1.0], 0.0, 0.25),
        };
        assert_eq!(o.albedo, albedo, "{i:?}");
        assert_eq!(o.depth_offset, depth_offset, "{i:?}");
        assert_eq!(o.coverage, coverage, "{i:?}");
        assert_eq!(o.key_bits, 5);
        assert_eq!(o.key, 1 | (i.surface << 3));

        let cpu = shader_link::surface(i.surface as u8, c.kind, c.depth, 0.01, 0.002);
        assert_eq!(
            (cpu.albedo, cpu.depth_offset, cpu.coverage),
            (o.albedo, o.depth_offset, o.coverage),
            "GPU and CPU differ for {i:?}"
        );
    }
    assert_clean();
}

#[test]
fn nothing_linked_runs_the_defaults() {
    for set in [&unlinked::KERNELS, &KernelSet::default()] {
        for (i, o) in resolve_on_gpu(set) {
            assert_eq!(o.albedo, [0.5, 0.5, 0.5, 1.0], "{i:?}");
            assert_eq!((o.depth_offset, o.coverage), (0.0, 0.25), "{i:?}");
            assert_eq!(o.key_bits, 3);
        }
        let d = displace_on_gpu(set);
        assert_eq!(d, [[0.0, 0.0, 0.5, 0.0]; 2]);
        let s = sky_on_gpu(set);
        assert_eq!(s[0], [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(s[1], [0.0; 4]);
        assert_eq!(s[2], [1.0, 1.0, 1.0, 0.0]);
    }
    assert_clean();
}

/// Displacement ids 0 and 1 at the same point.
fn displace_on_gpu(set: &KernelSet) -> Vec<[f32; 4]> {
    let inputs: Vec<DisplacementInput> = (0..2)
        .map(|displacement| DisplacementInput {
            position: [3.0, 0.0, 0.0],
            displacement,
            normal: [0.0, 0.0, 1.0],
            material: 0,
        })
        .collect();
    let gpu_inputs = upload(&inputs);
    // The kernel has no bounds check: one workgroup writes 64 outputs.
    let mut out = run(set, &tessellate::MODULE, 64, |out| tessellate::Root {
        inputs: gpu_inputs.gpu,
        out,
    });
    out.truncate(2);
    out
}

#[test]
fn displacement_switches_on_its_id() {
    assert_eq!(
        usize::from(linked::displacement::WAVES),
        1,
        "Waves is the first displacement"
    );
    let d = displace_on_gpu(&linked::KERNELS);
    assert_eq!(d[0], [0.0, 0.0, 0.5, 0.0]);
    assert_eq!(d[1], [0.0, 3.0, 2.0, 0.0]);
    assert_eq!(
        d[1][..3],
        shader_link::displace(1, [3.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    );
    assert_clean();
}

const AIR: AirData = AirData {
    scattering: [0.01, 0.02, 0.04],
    absorption: 0.005,
    anisotropy: 0.7,
    pad: [0.0; 3],
};

/// The sky kernel's three outputs, with the medium reading `AIR` through its address.
fn sky_on_gpu(set: &KernelSet) -> Vec<[f32; 4]> {
    let air = upload(&[AIR]);
    run(set, &sky::MODULE, 3, |out| sky::Root {
        out,
        environment: 0,
        medium: air.gpu.address().0,
        pad: 0,
    })
}

#[test]
fn environment_and_medium_read_their_data_through_the_address() {
    let out = sky_on_gpu(&linked::KERNELS);
    assert_eq!(out[0], [4.0, 1.0, 6.0, 0.0]);
    assert_eq!(
        out[1],
        [
            AIR.scattering[0],
            AIR.scattering[1],
            AIR.scattering[2],
            AIR.anisotropy
        ]
    );
    for c in 0..3 {
        let expected = (-(AIR.scattering[c] + AIR.absorption) * 2.0).exp();
        assert!((out[2][c] - expected).abs() < 1e-6, "{:?}", out[2]);
    }
    assert_clean();
}

fn renderer_error(kernels: &'static KernelSet) -> Option<orr::Error> {
    Renderer::new(
        device(),
        RendererDesc {
            kernels,
            ..RendererDesc::default()
        },
    )
    .err()
}

#[test]
fn renderer_accepts_a_set_built_against_its_kernels() {
    let mut kernels: Vec<&str> = orr_linked::KERNELS
        .modules
        .iter()
        .map(|m| m.kernel)
        .collect();
    kernels.sort_unstable();
    assert_eq!(kernels, ["environment/bake", "light_sampling/directional"]);
    if let Some(e) = renderer_error(&orr_linked::KERNELS) {
        panic!("{e}");
    }
    // The linked modules create pipelines as orr's own do.
    for &m in orr_linked::KERNELS.modules {
        pipeline(m);
    }
    assert!(renderer_error(&unlinked::KERNELS).is_none());
    assert_clean();
}

#[test]
fn renderer_rejects_a_set_built_against_other_kernels() {
    match renderer_error(&orr_edited::KERNELS) {
        Some(orr::Error::Scene(orr::SceneError::Kernels(e))) => {
            assert_eq!(e.kernel, "environment/bake");
            assert!(matches!(e.reason, Mismatch::Interface { .. }), "{e}");
        }
        other => panic!("expected an interface mismatch, got {other:?}"),
    }
    // The fixture's `resolve` shares its kernel name with orr-scene's material resolve, so
    // its entry points are compared and differ.
    match renderer_error(&linked::KERNELS) {
        Some(orr::Error::Scene(orr::SceneError::Kernels(e))) => {
            assert_eq!(e.kernel, "resolve");
            assert_eq!(e.reason, Mismatch::EntryPoints, "{e}");
        }
        other => panic!("expected different entry points, got {other:?}"),
    }
    assert_clean();
}
