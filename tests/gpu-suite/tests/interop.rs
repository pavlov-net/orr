//! Raw Vulkan commands inside orr command lists, and device-creation hooks.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;
use gpu_suite::shaders::heap_views;
use gpu_suite::{assert_clean, device, log_to_stderr, pipeline, readback, render, run, upload};
use orr_gpu::vulkan::{self, Features, RawDevice};
use orr_gpu::{
    Device, DeviceDesc, DeviceHooks, Filter, Format, Hazards, SamplerDesc, SamplerIndex, Stages,
    TextureDesc, TextureMemory, TextureUsage, ViewDesc,
};

/// A raw clear between two dispatches of one pipeline: the second dispatch rebinds heaps,
/// pipeline and root, and samples the cleared texels.
#[test]
fn heap_commands_after_raw_commands() {
    let d = device();
    let (w, h) = (64u32, 64u32);
    let desc = TextureDesc::d2(
        Format::Rgba32Float,
        w,
        h,
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
    );
    let tex = d
        .create_texture(&desc, TextureMemory::Dedicated, "raw clear")
        .unwrap();
    let slots = d.alloc_image_slots(2).unwrap();
    let target = d.write_storage(slots, &tex, &ViewDesc::default());
    let source = d.write_sampled(slots + 1, &tex, &ViewDesc::default());
    let nearest = SamplerIndex(3);
    d.write_sampler(
        nearest,
        &SamplerDesc {
            mag: Filter::Nearest,
            min: Filter::Nearest,
            mip: Filter::Nearest,
            ..SamplerDesc::default()
        },
    );
    let outs = [0, 1].map(|_| readback::<[f32; 4]>(u64::from(w * h), "raw clear out"));
    let roots = outs.each_ref().map(|out| {
        upload(
            &[heap_views::Root {
                out: out.gpu,
                source,
                nearest,
                target,
                width: w,
                height: h,
                pad: 0,
            }],
            "raw clear root",
        )
    });
    let sample = pipeline(&heap_views::MODULE, "sample_main");
    let groups = [w.div_ceil(8), h.div_ceil(4), 1];
    let value = [0.25, -2.0, 7.5, 1.0];
    let handles = vulkan::image_handles(&tex).unwrap();
    assert_eq!(
        vulkan::image_handles(&tex).unwrap().view,
        handles.view,
        "the view is made once"
    );

    let mut list = render().command_list().unwrap();
    list.discard(&tex);
    list.dispatch(&sample, roots[0].gpu, groups);
    list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
    let raw = list.vulkan_raw();
    // SAFETY: the buffer is recording; the image is in GENERAL and allows transfer writes.
    unsafe {
        raw.device.cmd_clear_color_image(
            raw.buffer,
            handles.image,
            vk::ImageLayout::GENERAL,
            &vk::ClearColorValue { float32: value },
            &[handles.range],
        );
    }
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&sample, roots[1].gpu, groups);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = run(list);

    let wrong = outs[1]
        .as_slice()
        .unwrap()
        .iter()
        .filter(|&&t| t != value)
        .count();
    assert_eq!(wrong, 0, "texels sampled after the raw clear");
    d.free_image_slot(slots, point);
    d.free_image_slot(slots + 1, point);
    assert_clean();
}

struct ShaderClock {
    created: AtomicBool,
}

impl DeviceHooks for ShaderClock {
    fn device_extensions(&self, _: &ash::Instance, _: vk::PhysicalDevice) -> Vec<CString> {
        vec![ash::khr::shader_clock::NAME.to_owned()]
    }

    fn device_features(&self, _: vk::PhysicalDevice, features: &mut Features<'_>) {
        features
            .push(vk::PhysicalDeviceShaderClockFeaturesKHR::default().shader_subgroup_clock(true));
        features.vulkan_1_4.push_descriptor = vk::TRUE;
    }

    fn device_created(&self, raw: &RawDevice<'_>) {
        assert!(raw.has_extension(ash::khr::shader_clock::NAME));
        self.created.store(true, Ordering::Relaxed);
    }
}

/// A hook's extension and chained feature struct reach device creation, validation clean.
#[test]
fn hooks_add_extensions_and_features() {
    log_to_stderr();
    let hooks = ShaderClock {
        created: AtomicBool::new(false),
    };
    let d = Device::new(&DeviceDesc {
        name: "gpu-suite hooks",
        hooks: Some(&hooks),
        validation: true,
        ..DeviceDesc::default()
    })
    .unwrap();
    assert!(hooks.created.load(Ordering::Relaxed));
    assert_eq!(
        d.validation_messages(),
        0,
        "validation messages (see the log)"
    );
}
