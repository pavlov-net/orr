/// What the device offers beyond orr's required set, and its limits.
#[derive(Clone, Debug)]
pub struct Caps {
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// Driver name and version as the driver reports them.
    pub driver: String,
    /// The raw driver version.
    pub driver_version: u32,
    /// Host-visible device-local memory covers VRAM, so [`crate::MemoryKind::Upload`] is
    /// device-local. Without it Upload is host memory.
    pub rebar: bool,
    /// Host image copy into [`crate::TextureUsage::HOST_COPY`] textures (`hostImageCopy`).
    /// Only with `rebar`: host-copy textures take Upload memory.
    pub host_image_copy: bool,
    /// GDeflate 1.0 memory decompression ([`crate::CommandList::decompress`]).
    pub memory_decompression: bool,
    /// [`crate::CommandList::set_occupancy`] sets the occupancy priority of dispatches
    /// (`VK_NV_compute_occupancy_priority`); without it the call does nothing.
    pub occupancy_priority: bool,
    /// Ray-dispatch pipelines (`VK_KHR_ray_tracing_pipeline` with invocation reordering):
    /// [`crate::Device::create_ray_dispatch_pipeline`] and
    /// [`crate::CommandList::dispatch_rays`].
    pub ray_dispatch: bool,
    /// The device reorders invocations at coherence points in hardware. Without it
    /// reordering is a no-op and kernels run their compute entry. False without
    /// `ray_dispatch`.
    pub reorder: bool,
    /// Opacity micromaps (`VK_KHR_opacity_micromap` with device address commands); kernels
    /// that trace OMM geometry load their `spirv_omm` variant.
    pub omm: bool,
    /// Device address commands (`VK_KHR_device_address_commands`).
    pub device_address_commands: bool,
    /// `GENERAL` is guaranteed optimal for every use (`VK_KHR_unified_image_layouts`).
    pub unified_image_layouts: bool,
    /// The memory report carries the OS budget per heap.
    pub memory_budget: bool,
    /// Shader `abort` (`VK_KHR_shader_abort`); enabled in debug builds only. The message of
    /// an abort that loses the device is in [`crate::DiagnosticsReport::abort`].
    pub shader_abort: bool,
    /// Device fault reports (`VK_KHR_device_fault` or `VK_EXT_device_fault`) in device-lost
    /// reports.
    pub device_fault: bool,
    /// Passes write GPU breadcrumbs (`VK_AMD_buffer_marker`), so reports tell which passes
    /// began and ended; with [`crate::Diagnostics::Checkpoints`] or `Aftermath`.
    pub breadcrumbs: bool,
    /// Passes set checkpoints (`VK_NV_device_diagnostic_checkpoints`), which device-lost
    /// reports and crash dumps read; with [`crate::Diagnostics::Checkpoints`] or `Aftermath`.
    pub checkpoints: bool,
    /// Nsight Aftermath writes crash dumps ([`crate::Diagnostics::Aftermath`]).
    pub aftermath: bool,
    /// Pipeline binaries (`VK_KHR_pipeline_binary`).
    pub pipeline_binary: bool,
    /// Cooperative-vector type combinations; empty without cooperative vectors or replicated
    /// composites.
    pub cooperative_vector: Vec<CooperativeVectorCombination>,
    /// 8-bit float shader types (`shaderFloat8`).
    pub shader_float8: bool,
    /// Pass timings carry CPU-clock spans.
    pub calibrated_timestamps: bool,
    /// Submits can carry frame boundaries for capture tools.
    pub frame_boundary: bool,
    /// Pipeline creation logs register, shared-memory and spill counts.
    pub pipeline_executable_info: bool,
    /// Conservative rasterisation (`VK_EXT_conservative_rasterization`) for
    /// [`crate::RasterState::conservative`].
    pub conservative_raster: bool,
    /// `VK_NV_ray_tracing_validation` checks acceleration-structure builds and traces;
    /// debug builds only.
    pub ray_tracing_validation: bool,
    /// Present ids and waits (`VK_KHR_present_id2`, `VK_KHR_present_wait2`):
    /// [`crate::Swapchain::wait_presented`] works. Devices without the Present queue have
    /// none of the present items.
    pub present_wait: bool,
    /// Reflex (`VK_NV_low_latency2`): [`crate::Swapchain::latency`] returns a handle.
    pub low_latency: bool,
    /// Metered present batches (`VK_NV_present_metering`).
    pub present_metering: bool,
    /// Swapchain refresh timing (`VK_EXT_present_timing`): [`crate::Swapchain::refresh`]
    /// can report a variable refresh rate.
    pub present_timing: bool,
    /// HDR mastering metadata (`VK_EXT_hdr_metadata`).
    pub hdr_metadata: bool,
    pub cluster: crate::ClusterCaps,
    pub limits: Limits,
}

/// Device limits orr's callers size against.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub image_descriptor_bytes: u32,
    pub sampler_descriptor_bytes: u32,
    /// Image slots the resource heap can hold.
    pub max_image_slots: u32,
    /// Sampler slots the sampler heap can hold.
    pub max_sampler_slots: u32,
    pub max_push_data_bytes: u32,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period_ns: f32,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub max_workgroup_count: [u32; 3],
    pub max_image_extent_2d: u32,
    pub max_image_extent_3d: u32,
    pub max_image_layers: u32,
    /// Colour targets one mesh pipeline can write.
    pub max_color_targets: u32,
    /// Invocations one [`crate::CommandList::dispatch_rays`] may launch; 0 without
    /// `Caps::ray_dispatch`.
    pub max_ray_dispatch_invocations: u32,
    /// Size of the device-local heap.
    pub device_local_bytes: u64,
    /// Size of the heap behind [`crate::MemoryKind::Upload`].
    pub upload_bytes: u64,
    /// Regions one [`crate::CommandList::decompress`] may list; 0 without
    /// `Caps::memory_decompression`.
    pub max_decompress_regions: u32,
}

/// A component type of a cooperative-vector operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ComponentType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    /// 8-bit float, 4 exponent bits.
    E4m3,
    /// 8-bit float, 5 exponent bits.
    E5m2,
    /// A type this crate does not name.
    Other(i32),
}

/// One supported cooperative-vector matrix-multiply type combination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CooperativeVectorCombination {
    pub input: ComponentType,
    pub input_interpretation: ComponentType,
    pub matrix_interpretation: ComponentType,
    pub bias_interpretation: ComponentType,
    pub result: ComponentType,
    pub transpose: bool,
}
