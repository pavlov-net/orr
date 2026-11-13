use core::ffi::CStr;

/// A compiled SPIR-V module with its entry points, and its Metal libraries.
#[derive(Debug)]
pub struct ShaderModule {
    /// Unique within a build: the kernel path under its shader root, without extension.
    pub name: &'static str,
    /// The kernel file the module was compiled from, as a path under its shader root.
    pub kernel: &'static str,
    pub spirv: &'static [u32],
    /// The Metal library of the compute, task and mesh entries; empty unless the module was
    /// built for an Apple target with the MSL check on.
    pub metallib: &'static [u8],
    /// The Metal library of the pixel entries, which Metal needs apart from the others.
    pub metallib_fragment: &'static [u8],
    /// Built with opacity micromaps: its entry points declare the micromap execution mode,
    /// so its pipelines need a device with micromaps.
    pub omm: bool,
    /// The size of the root the kernel holds in the push data (`ORR_ROOT_INLINE`); 0 for a root
    /// in GPU memory (`ORR_ROOT`).
    pub inline_root_bytes: u32,
    /// The absolute path of `spirv`'s separated debug info (`<name>.dbg.spv`); empty when the
    /// debug info is embedded or off.
    pub debug_info: &'static str,
    /// A hash of `spirv`.
    pub hash: u64,
    /// The absolute path of the build manifest that lists the module's inputs.
    pub manifest: &'static str,
    pub entry_points: &'static [EntryPoint],
}

impl ShaderModule {
    /// The entry point named `name`.
    pub fn entry_point(&self, name: &str) -> Option<&EntryPoint> {
        self.entry_points
            .iter()
            .find(|e| e.name.to_bytes() == name.as_bytes())
    }
}

/// One entry point of a [`ShaderModule`].
#[derive(Clone, Copy, Debug)]
pub struct EntryPoint {
    /// The entry's function in the kernel source, which [`ShaderModule::entry_point`] finds it by.
    pub name: &'static CStr,
    /// The entry's name in `spirv`, which pipelines select it by and profilers show for it: the
    /// kernel's module name as dependents import it (`orr_lighting.path.staged.trace`) for
    /// `main`, followed by `.<name>` for any other entry.
    pub spirv_name: &'static CStr,
    /// The entry's function in the module's Metal library: `spirv_name` with `_` for `.`, which
    /// Slang may rename further (with a `_0` suffix); empty without a library and for
    /// ray tracing entries.
    pub metal_name: &'static CStr,
    pub stage: Stage,
    /// The workgroup size; `[1, 1, 1]` for ray tracing and pixel entries.
    pub workgroup: [u32; 3],
    /// The bytes of shader record data a ray tracing entry reads after its group handle: the
    /// extent of the entry's `uniform` parameters. 0 for other entries.
    pub record_bytes: u32,
    /// A hash of the root type's layout, the specialization constants and the entries' record
    /// sizes. Two builds whose hashes match can swap SPIR-V without rebuilding the Rust that
    /// fills the root and the records.
    pub interface: u64,
}

/// The pipeline stage of an entry point.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "build", derive(serde::Serialize, serde::Deserialize))]
pub enum Stage {
    Compute,
    Task,
    Mesh,
    Pixel,
    RayGen,
    ClosestHit,
    AnyHit,
    Intersection,
    Miss,
    Callable,
}

impl Stage {
    /// A stage of a ray tracing pipeline.
    pub const fn ray_tracing(self) -> bool {
        matches!(
            self,
            Self::RayGen
                | Self::ClosestHit
                | Self::AnyHit
                | Self::Intersection
                | Self::Miss
                | Self::Callable
        )
    }
}

/// Which entry point runs of a kernel with a raygen and a compute entry over one body.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PathEntry {
    /// The entry that renders a frame in the least time: the compute entry.
    #[default]
    Fastest,
    Compute,
    /// The raygen entry, which reorders at coherence points. Fails on devices without
    /// ray-dispatch pipelines.
    RayDispatch,
}

/// The Slang type of the environment kernels declare, which reads null data as black.
pub const DEFAULT_ENVIRONMENT: &str = "EnvironmentMap";
/// The Slang type of the medium kernels declare, which reads no data.
pub const DEFAULT_MEDIUM: &str = "NoMedium";

/// Kernels compiled by a consumer's build with its own link-time types: `KERNELS` in the
/// build's `orr_kernels.rs`. Empty when nothing links.
#[derive(Clone, Copy, Debug)]
pub struct KernelSet {
    pub modules: &'static [&'static ShaderModule],
    /// Linked surfaces: their ids are `1..=surfaces`.
    pub surfaces: u32,
    /// The Slang type linked as the environment; [`DEFAULT_ENVIRONMENT`] when none is.
    pub environment: &'static str,
    /// The Slang type linked as the medium; [`DEFAULT_MEDIUM`] when none is.
    pub medium: &'static str,
}

impl Default for KernelSet {
    fn default() -> Self {
        Self::NONE
    }
}

impl KernelSet {
    /// Nothing linked, for `static`s and promoted literals, where `default` can't run.
    pub const NONE: Self = Self {
        modules: &[],
        surfaces: 0,
        environment: DEFAULT_ENVIRONMENT,
        medium: DEFAULT_MEDIUM,
    };

    /// The module to create pipelines from in place of `own`: the set's module compiled from
    /// the same kernel file, else `own`. Fails when that module's entry points differ from
    /// `own`'s in name, stage, workgroup or interface hash.
    pub fn module(
        &self,
        own: &'static ShaderModule,
    ) -> Result<&'static ShaderModule, KernelMismatch> {
        let Some(m) = self.modules.iter().find(|m| m.kernel == own.kernel) else {
            return Ok(own);
        };
        let mismatch = |reason| KernelMismatch {
            kernel: own.kernel,
            reason,
        };
        if m.entry_points.len() != own.entry_points.len() {
            return Err(mismatch(Mismatch::EntryPoints));
        }
        for e in own.entry_points {
            let r = m
                .entry_points
                .iter()
                .find(|r| r.name == e.name)
                .ok_or_else(|| mismatch(Mismatch::EntryPoints))?;
            if r.stage != e.stage || r.workgroup != e.workgroup {
                return Err(mismatch(Mismatch::Stage { entry: e.name }));
            }
            if r.interface != e.interface {
                return Err(mismatch(Mismatch::Interface { entry: e.name }));
            }
        }
        Ok(m)
    }

    /// Checks every module of the set against the compiled-in module of the same kernel
    /// file in `builtin`, as [`module`](Self::module) does, so a renderer can fail before it
    /// creates any pipeline.
    pub fn check(&self, builtin: &[&'static ShaderModule]) -> Result<(), KernelMismatch> {
        for m in self.modules {
            let own = builtin
                .iter()
                .find(|b| b.kernel == m.kernel)
                .ok_or(KernelMismatch {
                    kernel: m.kernel,
                    reason: Mismatch::UnknownKernel,
                })?;
            self.module(own)?;
        }
        Ok(())
    }
}

/// A [`KernelSet`] module that cannot replace the compiled-in module of its kernel: the set
/// was built against other sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelMismatch {
    pub kernel: &'static str,
    pub reason: Mismatch,
}

/// Why a [`KernelSet`] module does not match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mismatch {
    /// No compiled-in module has this kernel.
    UnknownKernel,
    /// The entry point names differ.
    EntryPoints,
    /// An entry point's stage or workgroup size differs.
    Stage { entry: &'static CStr },
    /// An entry point's root layout or specialization constants differ.
    Interface { entry: &'static CStr },
}

impl core::fmt::Display for KernelMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let k = self.kernel;
        match self.reason {
            Mismatch::UnknownKernel => write!(f, "kernel set module {k} matches no kernel"),
            Mismatch::EntryPoints => write!(f, "kernel {k}: entry points differ"),
            Mismatch::Stage { entry } => write!(
                f,
                "kernel {k}: entry point {entry:?} differs in stage or workgroup"
            ),
            Mismatch::Interface { entry } => write!(
                f,
                "kernel {k}: entry point {entry:?} differs in root layout or specialization constants; rebuild against this renderer"
            ),
        }
    }
}

impl std::error::Error for KernelMismatch {}

/// Embeds a SPIR-V file as `&'static [u32]`, aligned to 4 bytes.
#[macro_export]
macro_rules! aligned_spirv {
    ($path:expr) => {{
        #[repr(C, align(4))]
        struct Aligned<B: ?Sized>(B);
        static ALIGNED: &Aligned<[u8]> = &Aligned(*include_bytes!($path));
        // SAFETY: the bytes are 4-aligned, their length is checked to be whole words, and
        // the crate only builds for little-endian targets, the byte order of orr's SPIR-V.
        unsafe {
            assert!(ALIGNED.0.len().is_multiple_of(4));
            ::core::slice::from_raw_parts(ALIGNED.0.as_ptr().cast::<u32>(), ALIGNED.0.len() / 4)
        }
    }};
}
