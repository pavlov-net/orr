//! Headless GPU tests of orr-pack's Slang modules. Every test runs on
//! `orr_scene::testing`'s device with validation on and fails if the layer reports anything.

use orr_gpu::{GpuPtr, Hazards, QueueKind, ShaderModule, ShaderStage, SpecData, Stages};
use orr_pack::{Archive, Blob};
use orr_scene::testing::{device, wait};

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// Runs `module`'s `main` once over `groups` workgroups with `root` and waits for it.
pub fn dispatch<R>(module: &'static ShaderModule, root: GpuPtr<R>, groups: u32) {
    let stage = ShaderStage::new(module, "main", SpecData::default()).unwrap();
    let p = device()
        .create_compute_pipeline(&stage, module.name)
        .unwrap();
    let render = device().queue(QueueKind::Render).unwrap();
    let mut cmd = render.command_list().unwrap();
    cmd.dispatch(&p, root, [groups, 1, 1]);
    cmd.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![cmd], &[]).unwrap());
}

/// The decoded bytes of blob `b`.
pub fn read(a: &Archive, b: &Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}
