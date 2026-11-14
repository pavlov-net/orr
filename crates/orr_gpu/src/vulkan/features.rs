//! The required device features and extensions (contracts.md 7) and the optional ones that
//! become `Caps` flags.

use std::any::Any;
use std::ffi::{CStr, c_void};

use ash::vk;

use super::diagnostics::Diagnostics;
use super::interop::{DeviceHooks, Features};
use crate::Error;

const REQUIRED_EXTENSIONS: &[&CStr] = &[
    ash::ext::descriptor_heap::NAME,
    ash::khr::shader_untyped_pointers::NAME,
    ash::khr::acceleration_structure::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::ray_tracing_position_fetch::NAME,
    ash::ext::mesh_shader::NAME,
    ash::nv::cluster_acceleration_structure::NAME,
    ash::nv::partitioned_acceleration_structure::NAME,
    ash::ext::memory_budget::NAME,
    ash::khr::maintenance8::NAME,
    ash::khr::shader_maximal_reconvergence::NAME,
    ash::khr::shader_clock::NAME,
    ash::ext::shader_atomic_float::NAME,
    ash::ext::shader_image_atomic_int64::NAME,
    ash::khr::shader_quad_control::NAME,
    ash::khr::shader_subgroup_uniform_control_flow::NAME,
    ash::khr::shader_relaxed_extended_instruction::NAME,
];

/// Required on devices that present (`QueueRequest::present`).
const SURFACE_EXTENSIONS: &[&CStr] = &[
    ash::khr::swapchain::NAME,
    ash::khr::swapchain_maintenance1::NAME,
];

/// Instance extensions of devices that present.
pub(super) const SURFACE_INSTANCE_EXTENSIONS: &[&CStr] = &[
    ash::khr::surface::NAME,
    #[cfg(windows)]
    ash::khr::win32_surface::NAME,
    ash::khr::get_surface_capabilities2::NAME,
    ash::khr::surface_maintenance1::NAME,
    ash::ext::swapchain_colorspace::NAME,
];

/// Optional extensions Nsight Graphics captures do not support; [`Diagnostics::Capture`]
/// leaves them off.
const CAPTURE_UNSUPPORTED: &[&CStr] = &[
    ash::khr::device_address_commands::NAME,
    ash::khr::device_fault::NAME,
    ash::khr::opacity_micromap::NAME,
    ash::khr::shader_abort::NAME,
    ash::khr::shader_constant_data::NAME,
];

/// Spec version of `VK_NV_cluster_acceleration_structure` with OMM in clusters.
const CLUSTER_SPEC_VERSION: u32 = 4;

/// Optional extensions found on the device.
#[derive(Default, Clone, Copy)]
pub(super) struct Optional {
    pub(super) ray_dispatch: bool,
    /// `VK_NV_ray_tracing_linear_swept_spheres`, with `VK_KHR_ray_tracing_pipeline` enabled
    /// beside it even where ray dispatch is off.
    pub(super) lss: bool,
    pub(super) omm: bool,
    pub(super) device_address_commands: bool,
    pub(super) indirect_copy: bool,
    pub(super) unified_image_layouts: bool,
    pub(super) memory_priority: bool,
    /// `VK_EXT_pageable_device_local_memory`, only beside `memory_priority`.
    pub(super) pageable_memory: bool,
    pub(super) shader_abort: bool,
    pub(super) cooperative_vector: bool,
    pub(super) shader_float8: bool,
    pub(super) memory_decompression: bool,
    /// `VK_EXT_external_memory_host`.
    pub(super) host_memory: bool,
    pub(super) calibrated_timestamps: bool,
    pub(super) frame_boundary: bool,
    pub(super) executable_info: bool,
    pub(super) host_image_copy: bool,
    pub(super) conservative_raster: bool,
    pub(super) occupancy_priority: bool,
    pub(super) generated_commands: bool,
    pub(super) ray_tracing_validation: bool,
    /// `VK_NV_device_diagnostic_checkpoints`.
    pub(super) checkpoints: bool,
    /// `VK_NV_device_diagnostics_config`, configured for Aftermath.
    pub(super) aftermath: bool,
    /// `VK_AMD_buffer_marker`, for the breadcrumb buffer.
    pub(super) buffer_marker: bool,
    /// `VK_KHR_device_fault`.
    pub(super) device_fault: bool,
    /// `VK_EXT_device_fault`, where the KHR extension is off.
    pub(super) device_fault_ext: bool,
    /// The device presents: swapchain and swapchain_maintenance1 are enabled.
    pub(super) surface: bool,
    pub(super) present_wait: bool,
    pub(super) low_latency: bool,
    pub(super) present_metering: bool,
    pub(super) present_timing: bool,
    pub(super) hdr_metadata: bool,
}

/// Exposed only when `NV_ALLOW_RAYTRACING_VALIDATION=1` is set before instance creation.
const RAY_TRACING_VALIDATION: &CStr = c"VK_NV_ray_tracing_validation";

/// Every feature struct orr reads or enables, chained through raw pointers.
#[derive(Default)]
pub(crate) struct FeatureChain {
    core: vk::PhysicalDeviceFeatures2<'static>,
    v11: vk::PhysicalDeviceVulkan11Features<'static>,
    v12: vk::PhysicalDeviceVulkan12Features<'static>,
    v13: vk::PhysicalDeviceVulkan13Features<'static>,
    v14: vk::PhysicalDeviceVulkan14Features<'static>,
    heap: vk::PhysicalDeviceDescriptorHeapFeaturesEXT<'static>,
    untyped: vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR<'static>,
    accel: vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    position_fetch: vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR<'static>,
    mesh: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    ray_pipeline: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'static>,
    reorder: vk::PhysicalDeviceRayTracingInvocationReorderFeaturesEXT<'static>,
    lss: vk::PhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV<'static>,
    cluster: vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV<'static>,
    ptlas: vk::PhysicalDevicePartitionedAccelerationStructureFeaturesNV<'static>,
    maintenance8: vk::PhysicalDeviceMaintenance8FeaturesKHR<'static>,
    reconvergence: vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR<'static>,
    clock: vk::PhysicalDeviceShaderClockFeaturesKHR<'static>,
    atomic_float: vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT<'static>,
    image_int64: vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT<'static>,
    quad_control: vk::PhysicalDeviceShaderQuadControlFeaturesKHR<'static>,
    uniform_control_flow: vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR<'static>,
    relaxed_extended: vk::PhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR<'static>,
    omm: vk::PhysicalDeviceOpacityMicromapFeaturesKHR<'static>,
    address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'static>,
    indirect_copy: vk::PhysicalDeviceCopyMemoryIndirectFeaturesKHR<'static>,
    unified: vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR<'static>,
    abort: vk::PhysicalDeviceShaderAbortFeaturesKHR<'static>,
    constant_data: vk::PhysicalDeviceShaderConstantDataFeaturesKHR<'static>,
    coop: vk::PhysicalDeviceCooperativeVectorFeaturesNV<'static>,
    replicated: vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT<'static>,
    float8: vk::PhysicalDeviceShaderFloat8FeaturesEXT<'static>,
    decompression: vk::PhysicalDeviceMemoryDecompressionFeaturesEXT<'static>,
    frame_boundary: vk::PhysicalDeviceFrameBoundaryFeaturesEXT<'static>,
    occupancy: vk::PhysicalDeviceComputeOccupancyPriorityFeaturesNV<'static>,
    generated: vk::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT<'static>,
    executable: vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR<'static>,
    rt_validation: vk::PhysicalDeviceRayTracingValidationFeaturesNV<'static>,
    swapchain_maintenance: vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR<'static>,
    present_id2: vk::PhysicalDevicePresentId2FeaturesKHR<'static>,
    present_wait2: vk::PhysicalDevicePresentWait2FeaturesKHR<'static>,
    present_metering: vk::PhysicalDevicePresentMeteringFeaturesNV<'static>,
    present_timing: vk::PhysicalDevicePresentTimingFeaturesEXT<'static>,
    fault: vk::PhysicalDeviceFaultFeaturesKHR<'static>,
    fault_ext: vk::PhysicalDeviceFaultFeaturesEXT<'static>,
    priority: vk::PhysicalDeviceMemoryPriorityFeaturesEXT<'static>,
    pageable: vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT<'static>,
    /// Device creation only.
    diagnostics_config: vk::DeviceDiagnosticsConfigCreateInfoNV<'static>,
}

impl FeatureChain {
    /// Links the structs the device supports into one chain headed by `core`; `create` adds
    /// the device-creation-only structs. The chain points into `self`, so `self` must not
    /// move while it is in use.
    fn link(&mut self, opt: &Optional, create: bool) {
        let mut chain: Vec<*mut vk::BaseOutStructure<'static>> = Vec::new();
        fn ptr<T>(s: &mut T) -> *mut vk::BaseOutStructure<'static> {
            std::ptr::from_mut(s).cast()
        }
        chain.extend([
            ptr(&mut self.core),
            ptr(&mut self.v11),
            ptr(&mut self.v12),
            ptr(&mut self.v13),
            ptr(&mut self.v14),
            ptr(&mut self.heap),
            ptr(&mut self.untyped),
            ptr(&mut self.accel),
            ptr(&mut self.ray_query),
            ptr(&mut self.position_fetch),
            ptr(&mut self.mesh),
            ptr(&mut self.cluster),
            ptr(&mut self.ptlas),
            ptr(&mut self.maintenance8),
            ptr(&mut self.reconvergence),
            ptr(&mut self.clock),
            ptr(&mut self.atomic_float),
            ptr(&mut self.image_int64),
            ptr(&mut self.quad_control),
            ptr(&mut self.uniform_control_flow),
            ptr(&mut self.relaxed_extended),
        ]);
        if opt.ray_dispatch {
            chain.push(ptr(&mut self.ray_pipeline));
            chain.push(ptr(&mut self.reorder));
        }
        if opt.lss {
            chain.push(ptr(&mut self.lss));
        }
        if opt.omm {
            chain.push(ptr(&mut self.omm));
        }
        if opt.device_address_commands {
            chain.push(ptr(&mut self.address_commands));
        }
        if opt.indirect_copy {
            chain.push(ptr(&mut self.indirect_copy));
        }
        if opt.unified_image_layouts {
            chain.push(ptr(&mut self.unified));
        }
        if opt.shader_abort {
            chain.push(ptr(&mut self.abort));
            chain.push(ptr(&mut self.constant_data));
        }
        if opt.cooperative_vector {
            chain.push(ptr(&mut self.coop));
            chain.push(ptr(&mut self.replicated));
        }
        if opt.shader_float8 {
            chain.push(ptr(&mut self.float8));
        }
        if opt.memory_decompression {
            chain.push(ptr(&mut self.decompression));
        }
        if opt.frame_boundary {
            chain.push(ptr(&mut self.frame_boundary));
        }
        if opt.occupancy_priority {
            chain.push(ptr(&mut self.occupancy));
        }
        if opt.generated_commands {
            chain.push(ptr(&mut self.generated));
        }
        if opt.executable_info {
            chain.push(ptr(&mut self.executable));
        }
        if opt.ray_tracing_validation {
            chain.push(ptr(&mut self.rt_validation));
        }
        if opt.surface {
            chain.push(ptr(&mut self.swapchain_maintenance));
        }
        if opt.present_wait {
            chain.push(ptr(&mut self.present_id2));
            chain.push(ptr(&mut self.present_wait2));
        }
        if opt.present_metering {
            chain.push(ptr(&mut self.present_metering));
        }
        if opt.present_timing {
            chain.push(ptr(&mut self.present_timing));
        }
        if opt.device_fault {
            chain.push(ptr(&mut self.fault));
        }
        if opt.device_fault_ext {
            chain.push(ptr(&mut self.fault_ext));
        }
        if opt.memory_priority {
            chain.push(ptr(&mut self.priority));
        }
        if opt.pageable_memory {
            chain.push(ptr(&mut self.pageable));
        }
        if create && opt.aftermath {
            chain.push(ptr(&mut self.diagnostics_config));
        }
        for pair in chain.windows(2) {
            // SAFETY: every pointer is a live field of `self` whose header is a
            // `VkBaseOutStructure`.
            unsafe { (*pair[0]).p_next = pair[1] };
        }
        // SAFETY: as above.
        unsafe { (**chain.last().unwrap()).p_next = std::ptr::null_mut() };
    }

    pub(super) fn head(&mut self) -> *mut c_void {
        std::ptr::from_mut(&mut self.core).cast()
    }

    /// The structs of the chain, in order.
    ///
    /// # Safety
    ///
    /// The chain is linked and its pointers are live.
    unsafe fn links(&mut self) -> Vec<*mut vk::BaseOutStructure<'static>> {
        let mut out = Vec::new();
        let mut p = self.head().cast::<vk::BaseOutStructure<'static>>();
        while !p.is_null() {
            out.push(p);
            // SAFETY: forwarded from the caller.
            p = unsafe { (*p).p_next };
        }
        out
    }

    /// Lets `hooks` set feature bits and add structs, which are chained after orr's. The
    /// returned boxes hold the added structs and must outlive device creation.
    pub(super) fn hook(
        &mut self,
        hooks: &dyn DeviceHooks,
        adapter: vk::PhysicalDevice,
    ) -> Vec<Box<dyn Any>> {
        // SAFETY: `select` linked the chain.
        let links = unsafe { self.links() };
        let chained = links
            .iter()
            // SAFETY: every link is a live struct of the chain.
            .map(|&p| unsafe { (*p).s_type })
            .collect();
        let mut features = Features {
            core: &mut self.core.features,
            vulkan_1_1: &mut self.v11,
            vulkan_1_2: &mut self.v12,
            vulkan_1_3: &mut self.v13,
            vulkan_1_4: &mut self.v14,
            chained,
            extra: Vec::new(),
        };
        hooks.device_features(adapter, &mut features);
        let mut extra = features.extra;
        // SAFETY: as above; hooks leave `p_next` alone (`DeviceHooks::device_features`).
        let mut tail = *unsafe { self.links() }.last().unwrap();
        for b in &mut extra {
            let s = std::ptr::from_mut::<dyn Any>(&mut **b).cast::<vk::BaseOutStructure<'static>>();
            // SAFETY: `tail` is the chain's last struct and `s` a pushed struct, whose type
            // `Features::push` bounds to device-creation structs with a Vulkan header.
            unsafe {
                (*s).p_next = std::ptr::null_mut();
                (*tail).p_next = s;
            }
            tail = s;
        }
        extra
    }
}

/// What a physical device offers orr: the extensions to enable and the features to turn on.
pub(super) struct Selection {
    pub(super) extensions: Vec<&'static CStr>,
    /// Every extension the device offers.
    offered: Vec<vk::ExtensionProperties>,
    pub(super) optional: Optional,
    /// Boxed so the chain's pointers stay valid.
    pub(super) enable: Box<FeatureChain>,
}

impl Selection {
    pub(super) fn offers(&self, name: &CStr) -> bool {
        has(&self.offered, name).is_some()
    }
}

fn has(list: &[vk::ExtensionProperties], name: &CStr) -> Option<u32> {
    list.iter()
        .find(|e| e.extension_name_as_c_str() == Ok(name))
        .map(|e| e.spec_version)
}

/// Checks `physical` against orr's required set and picks the optional items it offers,
/// leaving out the optional extensions named in `skip`. `surface` adds the extensions and
/// features presenting needs; `diagnostics` the ones its mode uses. The error names the
/// first missing item.
pub(super) fn select(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    skip: &[&CStr],
    surface: bool,
    diagnostics: Diagnostics<'_>,
) -> Result<Selection, String> {
    // SAFETY: `physical` came from this instance.
    let list = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map_err(|e| format!("extension list ({e})"))?;
    let surface_names = if surface { SURFACE_EXTENSIONS } else { &[] };
    for name in REQUIRED_EXTENSIONS.iter().chain(surface_names) {
        if has(&list, name).is_none() {
            return Err(name.to_string_lossy().into_owned());
        }
    }
    let cluster = has(&list, ash::nv::cluster_acceleration_structure::NAME).unwrap_or(0);
    if cluster < CLUSTER_SPEC_VERSION {
        return Err(format!(
            "VK_NV_cluster_acceleration_structure spec version {CLUSTER_SPEC_VERSION} (has {cluster})"
        ));
    }

    let capture = matches!(diagnostics, Diagnostics::Capture);
    let aftermath = matches!(diagnostics, Diagnostics::Aftermath { .. });
    let breadcrumbs = matches!(
        diagnostics,
        Diagnostics::Checkpoints | Diagnostics::Aftermath { .. }
    );
    let present = |n: &CStr| {
        has(&list, n).is_some()
            && !skip.contains(&n)
            && !(capture && CAPTURE_UNSUPPORTED.contains(&n))
    };
    let device_fault = present(ash::khr::device_fault::NAME);
    let mut opt = Optional {
        ray_dispatch: present(ash::khr::ray_tracing_pipeline::NAME)
            && present(ash::ext::ray_tracing_invocation_reorder::NAME),
        lss: present(ash::nv::ray_tracing_linear_swept_spheres::NAME)
            && present(ash::khr::ray_tracing_pipeline::NAME),
        omm: present(ash::khr::opacity_micromap::NAME),
        device_address_commands: present(ash::khr::device_address_commands::NAME),
        indirect_copy: present(ash::khr::copy_memory_indirect::NAME),
        unified_image_layouts: present(ash::khr::unified_image_layouts::NAME),
        memory_priority: present(ash::ext::memory_priority::NAME),
        pageable_memory: present(ash::ext::pageable_device_local_memory::NAME),
        // shader_abort requires device_fault and shader_constant_data enabled with it.
        shader_abort: cfg!(debug_assertions)
            && present(ash::khr::shader_abort::NAME)
            && device_fault
            && present(ash::khr::shader_constant_data::NAME),
        cooperative_vector: present(ash::nv::cooperative_vector::NAME)
            && present(ash::ext::shader_replicated_composites::NAME),
        shader_float8: present(ash::ext::shader_float8::NAME),
        memory_decompression: present(ash::ext::memory_decompression::NAME),
        host_memory: present(ash::ext::external_memory_host::NAME),
        calibrated_timestamps: present(ash::khr::calibrated_timestamps::NAME),
        frame_boundary: present(ash::ext::frame_boundary::NAME),
        executable_info: present(ash::khr::pipeline_executable_properties::NAME),
        host_image_copy: false,
        conservative_raster: present(ash::ext::conservative_rasterization::NAME),
        occupancy_priority: present(ash::nv::compute_occupancy_priority::NAME),
        generated_commands: present(ash::ext::device_generated_commands::NAME),
        // Aftermath hides the extension.
        ray_tracing_validation: !aftermath && present(RAY_TRACING_VALIDATION),
        checkpoints: breadcrumbs && present(ash::nv::device_diagnostic_checkpoints::NAME),
        aftermath: aftermath
            && present(ash::nv::device_diagnostic_checkpoints::NAME)
            && present(ash::nv::device_diagnostics_config::NAME),
        buffer_marker: breadcrumbs && present(ash::amd::buffer_marker::NAME),
        device_fault,
        device_fault_ext: !device_fault && present(ash::ext::device_fault::NAME),
        surface,
        present_wait: surface
            && present(ash::khr::present_id2::NAME)
            && present(ash::khr::present_wait2::NAME),
        low_latency: surface && present(ash::nv::low_latency2::NAME),
        present_metering: surface && present(ash::nv::present_metering::NAME),
        // Present timing requires calibrated timestamps enabled with it.
        present_timing: surface
            && present(ash::ext::present_timing::NAME)
            && present(ash::khr::calibrated_timestamps::NAME),
        hdr_metadata: surface && present(ash::ext::hdr_metadata::NAME),
    };

    let mut q = Box::<FeatureChain>::default();
    q.link(&opt, false);
    // SAFETY: the chain holds only structs of extensions the device lists.
    unsafe {
        instance.get_physical_device_features2(physical, &mut q.core);
    };

    let c = &q.core.features;
    let required: [(&str, vk::Bool32); 76] = [
        ("shaderInt64", c.shader_int64),
        // Slang declares the Geometry capability for SV_PrimitiveID in mesh and pixel stages.
        ("geometryShader", c.geometry_shader),
        ("fillModeNonSolid", c.fill_mode_non_solid),
        ("depthBiasClamp", c.depth_bias_clamp),
        ("shaderInt16", c.shader_int16),
        ("samplerAnisotropy", c.sampler_anisotropy),
        ("pipelineStatisticsQuery", c.pipeline_statistics_query),
        ("multiDrawIndirect", c.multi_draw_indirect),
        ("fragmentStoresAndAtomics", c.fragment_stores_and_atomics),
        (
            "shaderStorageImageReadWithoutFormat",
            c.shader_storage_image_read_without_format,
        ),
        (
            "shaderStorageImageWriteWithoutFormat",
            c.shader_storage_image_write_without_format,
        ),
        ("sparseBinding", c.sparse_binding),
        ("sparseResidencyBuffer", c.sparse_residency_buffer),
        ("sparseResidencyImage2D", c.sparse_residency_image2_d),
        ("shaderResourceMinLod", c.shader_resource_min_lod),
        (
            "storageBuffer16BitAccess",
            q.v11.storage_buffer16_bit_access,
        ),
        ("bufferDeviceAddress", q.v12.buffer_device_address),
        ("scalarBlockLayout", q.v12.scalar_block_layout),
        (
            "uniformBufferStandardLayout",
            q.v12.uniform_buffer_standard_layout,
        ),
        ("timelineSemaphore", q.v12.timeline_semaphore),
        ("vulkanMemoryModel", q.v12.vulkan_memory_model),
        (
            "vulkanMemoryModelDeviceScope",
            q.v12.vulkan_memory_model_device_scope,
        ),
        ("shaderFloat16", q.v12.shader_float16),
        ("shaderInt8", q.v12.shader_int8),
        ("storageBuffer8BitAccess", q.v12.storage_buffer8_bit_access),
        ("drawIndirectCount", q.v12.draw_indirect_count),
        (
            "shaderBufferInt64Atomics",
            q.v12.shader_buffer_int64_atomics,
        ),
        ("hostQueryReset", q.v12.host_query_reset),
        ("synchronization2", q.v13.synchronization2),
        (
            "pipelineCreationCacheControl",
            q.v13.pipeline_creation_cache_control,
        ),
        ("dynamicRendering", q.v13.dynamic_rendering),
        ("maintenance4", q.v13.maintenance4),
        ("shaderIntegerDotProduct", q.v13.shader_integer_dot_product),
        ("maintenance5", q.v14.maintenance5),
        ("descriptorHeap", q.heap.descriptor_heap),
        ("shaderUntypedPointers", q.untyped.shader_untyped_pointers),
        ("accelerationStructure", q.accel.acceleration_structure),
        ("rayQuery", q.ray_query.ray_query),
        (
            "rayTracingPositionFetch",
            q.position_fetch.ray_tracing_position_fetch,
        ),
        ("taskShader", q.mesh.task_shader),
        ("meshShader", q.mesh.mesh_shader),
        ("meshShaderQueries", q.mesh.mesh_shader_queries),
        (
            "clusterAccelerationStructure",
            q.cluster.cluster_acceleration_structure,
        ),
        (
            "partitionedAccelerationStructure",
            q.ptlas.partitioned_acceleration_structure,
        ),
        // Without it OpSRem and OpSMod of a negative operand are poison (Vulkan SPIR-V
        // environment, "Precision and Operation of SPIR-V Instructions").
        ("maintenance8", q.maintenance8.maintenance8),
        // The software raster declares MaximallyReconvergesKHR: its shuffles read other lanes
        // after branches that split its lanes and subgroups.
        (
            "shaderMaximalReconvergence",
            q.reconvergence.shader_maximal_reconvergence,
        ),
        ("shaderImageGatherExtended", c.shader_image_gather_extended),
        (
            "shaderStorageImageExtendedFormats",
            c.shader_storage_image_extended_formats,
        ),
        ("shaderDrawParameters", q.v11.shader_draw_parameters),
        (
            "uniformAndStorageBuffer16BitAccess",
            q.v11.uniform_and_storage_buffer16_bit_access,
        ),
        (
            "shaderSubgroupExtendedTypes",
            q.v12.shader_subgroup_extended_types,
        ),
        (
            "uniformAndStorageBuffer8BitAccess",
            q.v12.uniform_and_storage_buffer8_bit_access,
        ),
        (
            "shaderSharedInt64Atomics",
            q.v12.shader_shared_int64_atomics,
        ),
        (
            "subgroupBroadcastDynamicId",
            q.v12.subgroup_broadcast_dynamic_id,
        ),
        ("samplerFilterMinmax", q.v12.sampler_filter_minmax),
        (
            "shaderDemoteToHelperInvocation",
            q.v13.shader_demote_to_helper_invocation,
        ),
        (
            "shaderTerminateInvocation",
            q.v13.shader_terminate_invocation,
        ),
        ("subgroupSizeControl", q.v13.subgroup_size_control),
        ("computeFullSubgroups", q.v13.compute_full_subgroups),
        (
            "shaderZeroInitializeWorkgroupMemory",
            q.v13.shader_zero_initialize_workgroup_memory,
        ),
        ("shaderSubgroupRotate", q.v14.shader_subgroup_rotate),
        (
            "shaderSubgroupRotateClustered",
            q.v14.shader_subgroup_rotate_clustered,
        ),
        ("shaderFloatControls2", q.v14.shader_float_controls2),
        ("shaderExpectAssume", q.v14.shader_expect_assume),
        ("shaderSubgroupClock", q.clock.shader_subgroup_clock),
        ("shaderDeviceClock", q.clock.shader_device_clock),
        (
            "shaderBufferFloat32Atomics",
            q.atomic_float.shader_buffer_float32_atomics,
        ),
        (
            "shaderBufferFloat32AtomicAdd",
            q.atomic_float.shader_buffer_float32_atomic_add,
        ),
        (
            "shaderSharedFloat32Atomics",
            q.atomic_float.shader_shared_float32_atomics,
        ),
        (
            "shaderSharedFloat32AtomicAdd",
            q.atomic_float.shader_shared_float32_atomic_add,
        ),
        (
            "shaderImageFloat32Atomics",
            q.atomic_float.shader_image_float32_atomics,
        ),
        (
            "shaderImageFloat32AtomicAdd",
            q.atomic_float.shader_image_float32_atomic_add,
        ),
        (
            "shaderImageInt64Atomics",
            q.image_int64.shader_image_int64_atomics,
        ),
        ("shaderQuadControl", q.quad_control.shader_quad_control),
        (
            "shaderSubgroupUniformControlFlow",
            q.uniform_control_flow.shader_subgroup_uniform_control_flow,
        ),
        (
            "shaderRelaxedExtendedInstruction",
            q.relaxed_extended.shader_relaxed_extended_instruction,
        ),
    ];
    if let Some((name, _)) = required.iter().find(|(_, v)| *v == vk::FALSE) {
        return Err((*name).to_owned());
    }
    if surface && q.swapchain_maintenance.swapchain_maintenance1 == vk::FALSE {
        return Err("swapchainMaintenance1".to_owned());
    }
    // Height maps of every kind are read through the material's linear sampler.
    for format in [
        vk::Format::R16_UNORM,
        vk::Format::R16_SNORM,
        vk::Format::R32_SFLOAT,
    ] {
        // SAFETY: `physical` came from this instance.
        let p = unsafe { instance.get_physical_device_format_properties(physical, format) };
        if !p
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            return Err(format!("linear filtering of {format:?} images"));
        }
    }

    opt.ray_dispatch &= q.ray_pipeline.ray_tracing_pipeline != 0
        && q.ray_pipeline.ray_tracing_pipeline_trace_rays_indirect != 0
        && q.reorder.ray_tracing_invocation_reorder != 0;
    opt.lss &= q.lss.linear_swept_spheres != 0;
    opt.device_address_commands &= q.address_commands.device_address_commands != 0;
    // Micromap builds take their addresses through device address commands.
    opt.omm &= q.omm.micromap != 0 && opt.device_address_commands;
    if cfg!(orr_omm) && !opt.omm {
        let (why, fix) = if capture {
            (
                "capture sessions leave micromaps off",
                "ORR_SHADER_CAPTURE=1 or ORR_SHADER_OMM=0",
            )
        } else {
            (
                "no VK_KHR_opacity_micromap with micromap, or no VK_KHR_device_address_commands",
                "ORR_SHADER_OMM=0",
            )
        };
        return Err(format!(
            "micromap support ({why}), and the shaders were built with micromaps (ORR_SHADER_OMM, on by default); rebuild with {fix}"
        ));
    }
    opt.omm &= cfg!(orr_omm);
    opt.indirect_copy &= q.indirect_copy.indirect_memory_copy != 0
        && q.indirect_copy.indirect_memory_to_image_copy != 0;
    opt.unified_image_layouts &= q.unified.unified_image_layouts != 0;
    opt.shader_abort &= q.abort.shader_abort != 0 && q.constant_data.shader_constant_data != 0;
    opt.cooperative_vector &=
        q.coop.cooperative_vector != 0 && q.replicated.shader_replicated_composites != 0;
    opt.shader_float8 &= q.float8.shader_float8 != 0;
    opt.memory_decompression &= q.decompression.memory_decompression != 0;
    opt.frame_boundary &= q.frame_boundary.frame_boundary != 0;
    opt.occupancy_priority &= q.occupancy.compute_occupancy_priority != 0;
    opt.generated_commands &= q.generated.device_generated_commands != 0;
    opt.executable_info &= q.executable.pipeline_executable_info != 0;
    opt.host_image_copy = q.v14.host_image_copy != 0;
    opt.ray_tracing_validation &= q.rt_validation.ray_tracing_validation != 0;
    opt.present_wait &= q.present_id2.present_id2 != 0 && q.present_wait2.present_wait2 != 0;
    // Low latency keys its frames by present id.
    opt.low_latency &= opt.present_wait;
    opt.present_metering &= q.present_metering.present_metering != 0;
    opt.present_timing &= q.present_timing.present_timing != 0;
    opt.device_fault &= q.fault.device_fault != 0;
    opt.shader_abort &= opt.device_fault;
    opt.device_fault_ext &= q.fault_ext.device_fault != 0;
    opt.memory_priority &= q.priority.memory_priority != 0;
    opt.pageable_memory &= opt.memory_priority && q.pageable.pageable_device_local_memory != 0;

    let mut e = Box::<FeatureChain>::default();
    e.core.features = vk::PhysicalDeviceFeatures {
        shader_int64: 1,
        geometry_shader: 1,
        fill_mode_non_solid: 1,
        depth_bias_clamp: 1,
        shader_int16: 1,
        sampler_anisotropy: 1,
        pipeline_statistics_query: 1,
        multi_draw_indirect: 1,
        fragment_stores_and_atomics: 1,
        shader_storage_image_read_without_format: 1,
        shader_storage_image_write_without_format: 1,
        sparse_binding: 1,
        sparse_residency_buffer: 1,
        sparse_residency_image2_d: 1,
        shader_resource_min_lod: 1,
        shader_image_gather_extended: 1,
        shader_storage_image_extended_formats: 1,
        ..Default::default()
    };
    e.v11.storage_buffer16_bit_access = 1;
    e.v11.uniform_and_storage_buffer16_bit_access = 1;
    e.v11.shader_draw_parameters = 1;
    e.v12 = e
        .v12
        .buffer_device_address(true)
        .scalar_block_layout(true)
        .uniform_buffer_standard_layout(true)
        .timeline_semaphore(true)
        .vulkan_memory_model(true)
        .vulkan_memory_model_device_scope(true)
        .shader_float16(true)
        .shader_int8(true)
        .storage_buffer8_bit_access(true)
        .draw_indirect_count(true)
        .shader_buffer_int64_atomics(true)
        .host_query_reset(true)
        .shader_subgroup_extended_types(true)
        .uniform_and_storage_buffer8_bit_access(true)
        .shader_shared_int64_atomics(true)
        .subgroup_broadcast_dynamic_id(true)
        .sampler_filter_minmax(true);
    e.v13 = e
        .v13
        .synchronization2(true)
        .pipeline_creation_cache_control(true)
        .dynamic_rendering(true)
        .maintenance4(true)
        .shader_integer_dot_product(true)
        .shader_demote_to_helper_invocation(true)
        .shader_terminate_invocation(true)
        .subgroup_size_control(true)
        .compute_full_subgroups(true)
        .shader_zero_initialize_workgroup_memory(true);
    e.v14 = e
        .v14
        .maintenance5(true)
        .shader_subgroup_rotate(true)
        .shader_subgroup_rotate_clustered(true)
        .shader_float_controls2(true)
        .shader_expect_assume(true)
        .host_image_copy(opt.host_image_copy);
    e.heap.descriptor_heap = 1;
    e.untyped.shader_untyped_pointers = 1;
    e.accel.acceleration_structure = 1;
    e.ray_query.ray_query = 1;
    e.position_fetch.ray_tracing_position_fetch = 1;
    e.mesh = e
        .mesh
        .task_shader(true)
        .mesh_shader(true)
        .mesh_shader_queries(true);
    e.cluster.cluster_acceleration_structure = 1;
    e.ptlas.partitioned_acceleration_structure = 1;
    e.maintenance8.maintenance8 = 1;
    e.reconvergence.shader_maximal_reconvergence = 1;
    e.clock = e
        .clock
        .shader_subgroup_clock(true)
        .shader_device_clock(true);
    e.atomic_float = e
        .atomic_float
        .shader_buffer_float32_atomics(true)
        .shader_buffer_float32_atomic_add(true)
        .shader_shared_float32_atomics(true)
        .shader_shared_float32_atomic_add(true)
        .shader_image_float32_atomics(true)
        .shader_image_float32_atomic_add(true);
    e.image_int64.shader_image_int64_atomics = 1;
    e.quad_control.shader_quad_control = 1;
    e.uniform_control_flow.shader_subgroup_uniform_control_flow = 1;
    e.relaxed_extended.shader_relaxed_extended_instruction = 1;
    e.ray_pipeline.ray_tracing_pipeline = 1;
    e.ray_pipeline.ray_tracing_pipeline_trace_rays_indirect = 1;
    e.reorder.ray_tracing_invocation_reorder = 1;
    e.lss.linear_swept_spheres = 1;
    e.omm.micromap = 1;
    e.address_commands.device_address_commands = 1;
    e.indirect_copy = e
        .indirect_copy
        .indirect_memory_copy(true)
        .indirect_memory_to_image_copy(true);
    e.unified.unified_image_layouts = 1;
    e.abort.shader_abort = 1;
    e.constant_data.shader_constant_data = 1;
    e.coop.cooperative_vector = 1;
    e.replicated.shader_replicated_composites = 1;
    e.float8.shader_float8 = 1;
    e.decompression.memory_decompression = 1;
    e.frame_boundary.frame_boundary = 1;
    e.occupancy.compute_occupancy_priority = 1;
    e.generated.device_generated_commands = 1;
    e.executable.pipeline_executable_info = 1;
    e.rt_validation.ray_tracing_validation = 1;
    e.swapchain_maintenance.swapchain_maintenance1 = 1;
    e.present_id2.present_id2 = 1;
    e.present_wait2.present_wait2 = 1;
    e.present_metering.present_metering = 1;
    e.present_timing.present_timing = 1;
    e.fault.device_fault = 1;
    e.fault_ext.device_fault = 1;
    e.priority.memory_priority = 1;
    e.pageable.pageable_device_local_memory = 1;
    e.diagnostics_config.flags = vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_DEBUG_INFO
        | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_RESOURCE_TRACKING
        | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_AUTOMATIC_CHECKPOINTS
        | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_ERROR_REPORTING;
    e.link(&opt, true);

    let mut extensions = REQUIRED_EXTENSIONS.to_vec();
    extensions.extend_from_slice(surface_names);
    let mut add = |on: bool, names: &[&'static CStr]| {
        if on {
            extensions.extend_from_slice(names);
        }
    };
    add(
        opt.ray_dispatch,
        &[
            ash::khr::ray_tracing_pipeline::NAME,
            ash::ext::ray_tracing_invocation_reorder::NAME,
        ],
    );
    add(opt.lss, &[ash::nv::ray_tracing_linear_swept_spheres::NAME]);
    // The LSS extension needs the pipeline extension enabled; its feature stays off.
    add(
        opt.lss && !opt.ray_dispatch,
        &[ash::khr::ray_tracing_pipeline::NAME],
    );
    add(opt.omm, &[ash::khr::opacity_micromap::NAME]);
    add(
        opt.device_address_commands,
        &[ash::khr::device_address_commands::NAME],
    );
    add(opt.indirect_copy, &[ash::khr::copy_memory_indirect::NAME]);
    add(
        opt.unified_image_layouts,
        &[ash::khr::unified_image_layouts::NAME],
    );
    add(opt.memory_priority, &[ash::ext::memory_priority::NAME]);
    add(
        opt.pageable_memory,
        &[ash::ext::pageable_device_local_memory::NAME],
    );
    add(opt.device_fault, &[ash::khr::device_fault::NAME]);
    add(opt.device_fault_ext, &[ash::ext::device_fault::NAME]);
    add(
        opt.shader_abort,
        &[
            ash::khr::shader_abort::NAME,
            ash::khr::shader_constant_data::NAME,
        ],
    );
    add(
        opt.checkpoints,
        &[ash::nv::device_diagnostic_checkpoints::NAME],
    );
    add(opt.aftermath, &[ash::nv::device_diagnostics_config::NAME]);
    add(opt.buffer_marker, &[ash::amd::buffer_marker::NAME]);
    add(
        opt.cooperative_vector,
        &[
            ash::nv::cooperative_vector::NAME,
            ash::ext::shader_replicated_composites::NAME,
        ],
    );
    add(opt.shader_float8, &[ash::ext::shader_float8::NAME]);
    add(
        opt.memory_decompression,
        &[ash::ext::memory_decompression::NAME],
    );
    add(opt.host_memory, &[ash::ext::external_memory_host::NAME]);
    add(
        opt.calibrated_timestamps,
        &[ash::khr::calibrated_timestamps::NAME],
    );
    add(opt.frame_boundary, &[ash::ext::frame_boundary::NAME]);
    add(
        opt.occupancy_priority,
        &[ash::nv::compute_occupancy_priority::NAME],
    );
    add(
        opt.generated_commands,
        &[ash::ext::device_generated_commands::NAME],
    );
    add(
        opt.conservative_raster,
        &[ash::ext::conservative_rasterization::NAME],
    );
    add(
        opt.executable_info,
        &[ash::khr::pipeline_executable_properties::NAME],
    );
    add(opt.ray_tracing_validation, &[RAY_TRACING_VALIDATION]);
    add(
        opt.present_wait,
        &[ash::khr::present_id2::NAME, ash::khr::present_wait2::NAME],
    );
    add(opt.low_latency, &[ash::nv::low_latency2::NAME]);
    add(opt.present_metering, &[ash::nv::present_metering::NAME]);
    add(opt.hdr_metadata, &[ash::ext::hdr_metadata::NAME]);
    // Calibrated timestamps are already in the list when present timing is on.
    add(opt.present_timing, &[ash::ext::present_timing::NAME]);
    Ok(Selection {
        extensions,
        offered: list,
        optional: opt,
        enable: e,
    })
}

/// Maps a selection failure on the chosen device to the error `Device::new` returns.
pub(super) fn unsupported(device: &str, missing: String) -> Error {
    Error::Unsupported {
        device: device.to_owned(),
        missing,
    }
}
