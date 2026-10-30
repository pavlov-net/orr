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

/// Kernels compiled by a consumer's build with its own link-time types.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelSet {
    pub modules: &'static [&'static ShaderModule],
}

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
