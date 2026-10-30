use core::ffi::CStr;

/// A compiled SPIR-V module with its entry points.
#[derive(Debug)]
pub struct ShaderModule {
    /// Unique within a build: the kernel path under its shader root, without extension.
    pub name: &'static str,
    /// The kernel file the module was compiled from, as a path under its shader root.
    pub kernel: &'static str,
    pub spirv: &'static [u32],
    /// The same module with the opacity-micromap execution mode on every entry point, for
    /// devices with the micromap feature; empty when the module has no ray query.
    pub spirv_omm: &'static [u32],
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
    pub name: &'static CStr,
    pub stage: Stage,
    /// The workgroup size; `[1, 1, 1]` for ray-generation and pixel entries.
    pub workgroup: [u32; 3],
    /// A hash of the root type's layout and the specialisation constants. Two builds whose
    /// hashes match can swap SPIR-V without rebuilding the Rust that fills the root.
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
}

/// Kernels compiled by a consumer's build with its own link-time types: `KERNELS` in the
/// build's `orr_kernels.rs`. Empty when nothing links.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelSet {
    pub modules: &'static [&'static ShaderModule],
}

impl KernelSet {
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
                .ok_or(mismatch(Mismatch::EntryPoints))?;
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
    /// An entry point's root layout or specialisation constants differ.
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
                "kernel {k}: entry point {entry:?} differs in root layout or specialisation constants; rebuild against this renderer"
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
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: the bytes are 4-aligned, their length is checked to be whole words, and
        // SPIR-V files are little-endian words.
        unsafe {
            assert!(ALIGNED.0.len() % 4 == 0);
            ::core::slice::from_raw_parts(ALIGNED.0.as_ptr().cast::<u32>(), ALIGNED.0.len() / 4)
        }
    }};
}
