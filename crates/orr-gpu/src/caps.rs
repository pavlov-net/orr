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
    /// Host-visible device-local memory covers VRAM, so [`crate::MemoryKind::Upload`] is not
    /// limited to the small BAR window.
    pub rebar: bool,
    /// Host image copy (`hostImageCopy`).
    pub host_image_copy: bool,
    /// GDeflate 1.0 memory decompression.
    pub memory_decompression: bool,
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
    /// Shader `abort` (`VK_KHR_shader_abort`); enabled in debug builds only.
    pub shader_abort: bool,
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
