//! Link points through `orr_shaders::build`. `renderer/` stands in for a renderer crate:
//! its kernels declare every link point. The build links two surfaces (one overriding
//! `coverage`, one omitting it), a displacement, an environment and a medium into them, and
//! `cpu/calls.slang` calls the same linked types on the CPU.

/// The renderer fixture's own build.
pub mod renderer {
    include!(concat!(env!("OUT_DIR"), "/renderer/orr_shaders.rs"));
}

/// Every point linked.
pub mod linked {
    include!(concat!(env!("OUT_DIR"), "/linked/orr_kernels.rs"));
}

/// The same build with nothing linked.
pub mod unlinked {
    include!(concat!(env!("OUT_DIR"), "/unlinked/orr_kernels.rs"));
}

/// Every point linked into a copy of the renderer whose resolve root has one more field.
pub mod edited {
    include!(concat!(env!("OUT_DIR"), "/edited/orr_kernels.rs"));
}

/// The renderer's kernels, as a renderer checks a kernel set against them.
pub static RENDERER: [&orr_shaders::ShaderModule; 4] = [
    &renderer::plain::MODULE,
    &renderer::resolve::MODULE,
    &renderer::sky::MODULE,
    &renderer::tessellate::MODULE,
];

/// What a surface returned through the switch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceCall {
    pub albedo: [f32; 4],
    pub depth_offset: f32,
    pub coverage: f32,
}

/// Calls `surface` and `coverage` of surface `id` through the CPU build's switch.
pub fn surface(id: u8, kind: u32, depth: u32, cone_width: f32, cone_spread: f32) -> SurfaceCall {
    let mut out = [0.0f32; 6];
    // SAFETY: `out` holds the six floats the function writes.
    unsafe {
        ffi::link_surface(
            u32::from(id),
            kind,
            depth,
            cone_width,
            cone_spread,
            out.as_mut_ptr(),
        );
    }
    SurfaceCall {
        albedo: [out[0], out[1], out[2], out[3]],
        depth_offset: out[4],
        coverage: out[5],
    }
}

/// Displaces through the CPU build's switch.
pub fn displace(id: u8, position: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    // SAFETY: the inputs hold three floats each and `out` takes three.
    unsafe {
        ffi::link_displace(
            u32::from(id),
            position.as_ptr(),
            normal.as_ptr(),
            out.as_mut_ptr(),
        );
    }
    out
}

/// The linked environment's radiance toward `dir`.
pub fn environment(dir: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    // SAFETY: `dir` holds three floats and `out` takes three.
    unsafe { ffi::link_environment(dir.as_ptr(), out.as_mut_ptr()) };
    out
}

/// The consumer medium's data, as `consumer/air.slang` reads it through its address.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AirData {
    pub scattering: [f32; 3],
    pub absorption: f32,
    pub anisotropy: f32,
    pub pad: [f32; 3],
}

/// The linked medium sampled with `data`'s address: sigma_s, sigma_t, the phase parameter
/// and the transmittance over `t` metres.
pub fn medium(data: &AirData, t: f32) -> [f32; 10] {
    let mut out = [0.0f32; 10];
    // SAFETY: the address stays valid for the call and `out` takes ten floats.
    unsafe {
        ffi::link_medium(core::ptr::from_ref(data) as u64, t, out.as_mut_ptr());
    }
    out
}

/// `orr.coherence`'s `KEY_BITS` in the CPU build.
pub fn key_bits() -> u32 {
    // SAFETY: no arguments.
    unsafe { ffi::link_key_bits() }
}

mod ffi {
    unsafe extern "C" {
        pub(crate) fn link_surface(
            id: u32,
            kind: u32,
            depth: u32,
            cone_width: f32,
            cone_spread: f32,
            out: *mut f32,
        );
        pub(crate) fn link_displace(
            id: u32,
            position: *const f32,
            normal: *const f32,
            out: *mut f32,
        );
        pub(crate) fn link_environment(dir: *const f32, out: *mut f32);
        pub(crate) fn link_medium(data: u64, t: f32, out: *mut f32);
        pub(crate) fn link_key_bits() -> u32;
    }
}
