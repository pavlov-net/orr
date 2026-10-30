//! Headless GPU tests of orr-lighting, on gpu-suite's device with validation on.

use std::sync::OnceLock;

use orr_gpu::{
    AddressMode, Allocation, Filter, Format, GpuPtr, Hazards, KernelSet, SamplerDesc, SamplerIndex,
    ShaderModule, Stages, Texture, TextureDesc, TextureIndex, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};

pub use gpu_suite::{assert_clean, device, pipeline, readback, render, run, upload};

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// A kernel set in which each `(module, kernel)` stands in for orr-lighting's module compiled
/// from `kernel`, as a build linking another environment or medium makes it.
pub fn kernels(replacements: &[(&'static ShaderModule, &'static str)]) -> KernelSet {
    let modules: Vec<&'static ShaderModule> = replacements
        .iter()
        .map(|&(m, kernel)| {
            &*Box::leak(Box::new(ShaderModule {
                kernel,
                entry_points: m.entry_points,
                ..*m
            }))
        })
        .collect();
    KernelSet {
        modules: Box::leak(modules.into_boxed_slice()),
    }
}

/// A linear, clamping sampler slot.
pub fn linear_clamp() -> SamplerIndex {
    static SLOT: OnceLock<SamplerIndex> = OnceLock::new();
    *SLOT.get_or_init(|| {
        let slot = SamplerIndex(1);
        device().write_sampler(
            slot,
            &SamplerDesc {
                mag: Filter::Linear,
                min: Filter::Linear,
                mip: Filter::Nearest,
                address: [AddressMode::ClampToEdge; 3],
                ..SamplerDesc::default()
            },
        );
        slot
    })
}

/// An RGBA32F texture behind a sampled slot.
pub struct TestTexture {
    pub index: TextureIndex,
    _texture: Texture,
}

/// A `width` x `height` texture holding `texels`, row-major.
pub fn texture(width: u32, height: u32, texels: &[[f32; 4]], name: &str) -> TestTexture {
    let d = device();
    let desc = TextureDesc::d2(Format::Rgba32Float, width, height, TextureUsage::SAMPLED);
    let texture = d
        .create_texture(&desc, TextureMemory::Dedicated, name)
        .unwrap();
    let staging = upload(texels, name);
    let mut list = render().command_list().unwrap();
    list.discard(&texture);
    list.copy_to_texture(
        &texture,
        &TextureRegion::whole_mip(&desc, 0),
        staging.gpu.cast(),
    );
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    run(list);
    let slot = d.alloc_image_slot().unwrap();
    TestTexture {
        index: d.write_sampled(slot, &texture, &ViewDesc::default()),
        _texture: texture,
    }
}

/// Readback memory that `list` fills with `count` values from `src`.
pub fn copy_back<T: bytemuck::Pod>(
    list: &mut orr_gpu::CommandList<'_>,
    src: GpuPtr<T>,
    count: u64,
    name: &str,
) -> Allocation<T> {
    let out = readback::<T>(count, name);
    list.copy(out.gpu.cast(), src.cast(), count * size_of::<T>() as u64);
    out
}
