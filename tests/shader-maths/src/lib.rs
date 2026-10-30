//! orr's Slang maths libraries compiled for the CPU (`slang/maths.slang`), with safe
//! wrappers for the tests, and the statistics the sampling tests use.

pub mod stats;

/// `gpu/maths.slang` compiled to SPIR-V; building it checks its generated layouts.
pub mod gpu {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

use orr_shaders::AliasEntry;

type V3 = [f32; 3];
type V4 = [f32; 4];
type M3 = [f32; 9];
type M34 = [f32; 12];

mod ffi {
    use orr_shaders::AliasEntry;

    unsafe extern "C" {
        pub(crate) fn orr_orthonormal_basis(n: *const f32, t: *mut f32, b: *mut f32);
        pub(crate) fn orr_quat_mul(a: *const f32, b: *const f32, out: *mut f32);
        pub(crate) fn orr_quat_rotate(q: *const f32, v: *const f32, out: *mut f32);
        pub(crate) fn orr_quat_to_matrix(q: *const f32, out: *mut f32);
        pub(crate) fn orr_affine_mul(a: *const f32, b: *const f32, out: *mut f32);
        pub(crate) fn orr_affine_inverse(m: *const f32, out: *mut f32);
        pub(crate) fn orr_transform_point(m: *const f32, p: *const f32, out: *mut f32);
        pub(crate) fn orr_transform_vector(m: *const f32, v: *const f32, out: *mut f32);
        pub(crate) fn orr_transform_normal(m: *const f32, n: *const f32, out: *mut f32);
        pub(crate) fn orr_offset_ray_origin(p: *const f32, n: *const f32, out: *mut f32);

        pub(crate) fn orr_oct_encode_unorm16(n: *const f32) -> u32;
        pub(crate) fn orr_oct_decode_unorm16(e: u32, out: *mut f32);
        pub(crate) fn orr_oct_encode_snorm16(n: *const f32) -> u32;
        pub(crate) fn orr_oct_decode_snorm16(e: u32, out: *mut f32);
        pub(crate) fn orr_oct_encode_snorm8(n: *const f32) -> u32;
        pub(crate) fn orr_oct_decode_snorm8(e: u32, out: *mut f32);
        pub(crate) fn orr_pack_unorm2x16(x: f32, y: f32) -> u32;
        pub(crate) fn orr_unpack_unorm2x16(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_snorm2x16(x: f32, y: f32) -> u32;
        pub(crate) fn orr_unpack_snorm2x16(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_unorm4x8(v: *const f32) -> u32;
        pub(crate) fn orr_unpack_unorm4x8(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_snorm4x8(v: *const f32) -> u32;
        pub(crate) fn orr_unpack_snorm4x8(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_half2x16(x: f32, y: f32) -> u32;
        pub(crate) fn orr_unpack_half2x16(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_r11g11b10f(c: *const f32) -> u32;
        pub(crate) fn orr_unpack_r11g11b10f(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_rgb9e5(c: *const f32) -> u32;
        pub(crate) fn orr_unpack_rgb9e5(p: u32, out: *mut f32);
        pub(crate) fn orr_grid_step(k: i32) -> f32;
        pub(crate) fn orr_grid_position(
            anchor: *const i32,
            local: *const u32,
            step: f32,
            out: *mut f32,
        );
        pub(crate) fn orr_decode_rotation(r: u32, out: *mut f32);
        pub(crate) fn orr_instance_rotation(r: u32, exact_axis: u32, out: *mut f32);

        pub(crate) fn orr_sobol(index: u32, dim: u32) -> u32;
        pub(crate) fn orr_owen_sobol4(index: u32, seed: u32, out: *mut f32);
        pub(crate) fn orr_sampler_next2(px: u32, py: u32, frame: u32, count: u32, out: *mut f32);

        pub(crate) fn orr_sample_cosine_hemisphere(u0: f32, u1: f32, out: *mut f32);
        pub(crate) fn orr_cosine_hemisphere_pdf(cos_theta: f32) -> f32;
        pub(crate) fn orr_sample_uniform_hemisphere(u0: f32, u1: f32, out: *mut f32);
        pub(crate) fn orr_uniform_hemisphere_pdf() -> f32;
        pub(crate) fn orr_sample_uniform_sphere(u0: f32, u1: f32, out: *mut f32);
        pub(crate) fn orr_uniform_sphere_pdf() -> f32;
        pub(crate) fn orr_sample_cone(u0: f32, u1: f32, cos_max: f32, out: *mut f32);
        pub(crate) fn orr_cone_pdf(cos_max: f32) -> f32;
        pub(crate) fn orr_sample_spherical_rectangle(
            o: *const f32,
            corner: *const f32,
            ex: *const f32,
            ey: *const f32,
            u0: f32,
            u1: f32,
            out: *mut f32,
        ) -> f32;
        pub(crate) fn orr_sample_spherical_triangle(
            o: *const f32,
            v0: *const f32,
            v1: *const f32,
            v2: *const f32,
            u0: f32,
            u1: f32,
            out: *mut f32,
        ) -> f32;
        pub(crate) fn orr_sample_ggx_vndf_bounded(
            v: *const f32,
            ax: f32,
            ay: f32,
            u0: f32,
            u1: f32,
            out: *mut f32,
        );
        pub(crate) fn orr_ggx_vndf_bounded_pdf(
            v: *const f32,
            l: *const f32,
            ax: f32,
            ay: f32,
        ) -> f32;
        pub(crate) fn orr_sample_alias(table: *const AliasEntry, count: u32, u: f32) -> u32;

        pub(crate) fn orr_color_matrix(which: u32, out: *mut f32);
        pub(crate) fn orr_luminance_rec709(c: *const f32) -> f32;
        pub(crate) fn orr_luminance_rec2020(c: *const f32) -> f32;
        pub(crate) fn orr_srgb_encode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_srgb_decode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_pq_encode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_pq_decode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_ictcp_from_rec2020(c: *const f32, out: *mut f32);
        pub(crate) fn orr_rec2020_from_ictcp(c: *const f32, out: *mut f32);
        pub(crate) fn orr_gamut_compress(c: *const f32, out: *mut f32);
    }
}

// Every wrapper passes arrays of the lengths the Slang side reads and writes.
macro_rules! out {
    ($t:ty, |$o:ident| $call:expr) => {{
        let mut $o: $t = Default::default();
        // SAFETY: the Slang function reads its inputs' lengths and writes exactly `$o`.
        unsafe { $call };
        $o
    }};
}

macro_rules! call {
    ($call:expr) => {{
        // SAFETY: the Slang function reads exactly its inputs' lengths.
        unsafe { $call }
    }};
}

pub fn orthonormal_basis(n: V3) -> (V3, V3) {
    let (mut t, mut b) = ([0.0; 3], [0.0; 3]);
    call!(ffi::orr_orthonormal_basis(
        n.as_ptr(),
        t.as_mut_ptr(),
        b.as_mut_ptr()
    ));
    (t, b)
}
pub fn quat_mul(a: V4, b: V4) -> V4 {
    out!(V4, |o| ffi::orr_quat_mul(
        a.as_ptr(),
        b.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn quat_rotate(q: V4, v: V3) -> V3 {
    out!(V3, |o| ffi::orr_quat_rotate(
        q.as_ptr(),
        v.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn quat_to_matrix(q: V4) -> M3 {
    out!(M3, |o| ffi::orr_quat_to_matrix(q.as_ptr(), o.as_mut_ptr()))
}
pub fn affine_mul(a: M34, b: M34) -> M34 {
    out!(M34, |o| ffi::orr_affine_mul(
        a.as_ptr(),
        b.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn affine_inverse(m: M34) -> M34 {
    out!(M34, |o| ffi::orr_affine_inverse(m.as_ptr(), o.as_mut_ptr()))
}
pub fn transform_point(m: M34, p: V3) -> V3 {
    out!(V3, |o| ffi::orr_transform_point(
        m.as_ptr(),
        p.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn transform_vector(m: M34, v: V3) -> V3 {
    out!(V3, |o| ffi::orr_transform_vector(
        m.as_ptr(),
        v.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn transform_normal(m: M34, n: V3) -> V3 {
    out!(V3, |o| ffi::orr_transform_normal(
        m.as_ptr(),
        n.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn offset_ray_origin(p: V3, n: V3) -> V3 {
    out!(V3, |o| ffi::orr_offset_ray_origin(
        p.as_ptr(),
        n.as_ptr(),
        o.as_mut_ptr()
    ))
}

pub fn oct_encode_unorm16(n: V3) -> u32 {
    call!(ffi::orr_oct_encode_unorm16(n.as_ptr()))
}
pub fn oct_decode_unorm16(e: u32) -> V3 {
    out!(V3, |o| ffi::orr_oct_decode_unorm16(e, o.as_mut_ptr()))
}
pub fn oct_encode_snorm16(n: V3) -> u32 {
    call!(ffi::orr_oct_encode_snorm16(n.as_ptr()))
}
pub fn oct_decode_snorm16(e: u32) -> V3 {
    out!(V3, |o| ffi::orr_oct_decode_snorm16(e, o.as_mut_ptr()))
}
pub fn oct_encode_snorm8(n: V3) -> u32 {
    call!(ffi::orr_oct_encode_snorm8(n.as_ptr()))
}
pub fn oct_decode_snorm8(e: u32) -> V3 {
    out!(V3, |o| ffi::orr_oct_decode_snorm8(e, o.as_mut_ptr()))
}
pub fn pack_unorm2x16(v: [f32; 2]) -> u32 {
    call!(ffi::orr_pack_unorm2x16(v[0], v[1]))
}
pub fn unpack_unorm2x16(p: u32) -> [f32; 2] {
    out!([f32; 2], |o| ffi::orr_unpack_unorm2x16(p, o.as_mut_ptr()))
}
pub fn pack_snorm2x16(v: [f32; 2]) -> u32 {
    call!(ffi::orr_pack_snorm2x16(v[0], v[1]))
}
pub fn unpack_snorm2x16(p: u32) -> [f32; 2] {
    out!([f32; 2], |o| ffi::orr_unpack_snorm2x16(p, o.as_mut_ptr()))
}
pub fn pack_unorm4x8(v: V4) -> u32 {
    call!(ffi::orr_pack_unorm4x8(v.as_ptr()))
}
pub fn unpack_unorm4x8(p: u32) -> V4 {
    out!(V4, |o| ffi::orr_unpack_unorm4x8(p, o.as_mut_ptr()))
}
pub fn pack_snorm4x8(v: V4) -> u32 {
    call!(ffi::orr_pack_snorm4x8(v.as_ptr()))
}
pub fn unpack_snorm4x8(p: u32) -> V4 {
    out!(V4, |o| ffi::orr_unpack_snorm4x8(p, o.as_mut_ptr()))
}
pub fn pack_half2x16(v: [f32; 2]) -> u32 {
    call!(ffi::orr_pack_half2x16(v[0], v[1]))
}
pub fn unpack_half2x16(p: u32) -> [f32; 2] {
    out!([f32; 2], |o| ffi::orr_unpack_half2x16(p, o.as_mut_ptr()))
}
pub fn pack_r11g11b10f(c: V3) -> u32 {
    call!(ffi::orr_pack_r11g11b10f(c.as_ptr()))
}
pub fn unpack_r11g11b10f(p: u32) -> V3 {
    out!(V3, |o| ffi::orr_unpack_r11g11b10f(p, o.as_mut_ptr()))
}
pub fn pack_rgb9e5(c: V3) -> u32 {
    call!(ffi::orr_pack_rgb9e5(c.as_ptr()))
}
pub fn unpack_rgb9e5(p: u32) -> V3 {
    out!(V3, |o| ffi::orr_unpack_rgb9e5(p, o.as_mut_ptr()))
}
pub fn grid_step(k: i32) -> f32 {
    call!(ffi::orr_grid_step(k))
}
pub fn grid_position(anchor: [i32; 3], local: [u32; 3], step: f32) -> V3 {
    out!(V3, |o| ffi::orr_grid_position(
        anchor.as_ptr(),
        local.as_ptr(),
        step,
        o.as_mut_ptr()
    ))
}
pub fn decode_rotation(r: u32) -> V4 {
    out!(V4, |o| ffi::orr_decode_rotation(r, o.as_mut_ptr()))
}
pub fn instance_rotation(r: u32, exact_axis: bool) -> M3 {
    out!(M3, |o| ffi::orr_instance_rotation(
        r,
        u32::from(exact_axis),
        o.as_mut_ptr()
    ))
}

pub fn sobol(index: u32, dim: u32) -> u32 {
    call!(ffi::orr_sobol(index, dim))
}
pub fn owen_sobol4(index: u32, seed: u32) -> V4 {
    out!(V4, |o| ffi::orr_owen_sobol4(index, seed, o.as_mut_ptr()))
}
/// `count` consecutive `Sampler::next2` draws for one pixel and frame.
pub fn sampler_next2(pixel: [u32; 2], frame: u32, count: usize) -> Vec<[f32; 2]> {
    let mut out = vec![[0.0f32; 2]; count];
    call!(ffi::orr_sampler_next2(
        pixel[0],
        pixel[1],
        frame,
        count as u32,
        out.as_mut_ptr().cast()
    ));
    out
}

pub fn sample_cosine_hemisphere(u: [f32; 2]) -> V3 {
    out!(V3, |o| ffi::orr_sample_cosine_hemisphere(
        u[0],
        u[1],
        o.as_mut_ptr()
    ))
}
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    call!(ffi::orr_cosine_hemisphere_pdf(cos_theta))
}
pub fn sample_uniform_hemisphere(u: [f32; 2]) -> V3 {
    out!(V3, |o| ffi::orr_sample_uniform_hemisphere(
        u[0],
        u[1],
        o.as_mut_ptr()
    ))
}
pub fn uniform_hemisphere_pdf() -> f32 {
    call!(ffi::orr_uniform_hemisphere_pdf())
}
pub fn sample_uniform_sphere(u: [f32; 2]) -> V3 {
    out!(V3, |o| ffi::orr_sample_uniform_sphere(
        u[0],
        u[1],
        o.as_mut_ptr()
    ))
}
pub fn uniform_sphere_pdf() -> f32 {
    call!(ffi::orr_uniform_sphere_pdf())
}
pub fn sample_cone(u: [f32; 2], cos_max: f32) -> V3 {
    out!(V3, |o| ffi::orr_sample_cone(
        u[0],
        u[1],
        cos_max,
        o.as_mut_ptr()
    ))
}
pub fn cone_pdf(cos_max: f32) -> f32 {
    call!(ffi::orr_cone_pdf(cos_max))
}
/// A point on the rectangle and the solid angle it subtends from `o`.
pub fn sample_spherical_rectangle(o: V3, corner: V3, ex: V3, ey: V3, u: [f32; 2]) -> (V3, f32) {
    let mut p = [0.0; 3];
    let solid_angle = call!(ffi::orr_sample_spherical_rectangle(
        o.as_ptr(),
        corner.as_ptr(),
        ex.as_ptr(),
        ey.as_ptr(),
        u[0],
        u[1],
        p.as_mut_ptr()
    ));
    (p, solid_angle)
}
/// A unit direction through the triangle and the solid angle it subtends from `o`.
pub fn sample_spherical_triangle(o: V3, v: [V3; 3], u: [f32; 2]) -> (V3, f32) {
    let mut d = [0.0; 3];
    let solid_angle = call!(ffi::orr_sample_spherical_triangle(
        o.as_ptr(),
        v[0].as_ptr(),
        v[1].as_ptr(),
        v[2].as_ptr(),
        u[0],
        u[1],
        d.as_mut_ptr()
    ));
    (d, solid_angle)
}
pub fn sample_ggx_vndf_bounded(v: V3, alpha: [f32; 2], u: [f32; 2]) -> V3 {
    out!(V3, |o| ffi::orr_sample_ggx_vndf_bounded(
        v.as_ptr(),
        alpha[0],
        alpha[1],
        u[0],
        u[1],
        o.as_mut_ptr()
    ))
}
pub fn ggx_vndf_bounded_pdf(v: V3, l: V3, alpha: [f32; 2]) -> f32 {
    call!(ffi::orr_ggx_vndf_bounded_pdf(
        v.as_ptr(),
        l.as_ptr(),
        alpha[0],
        alpha[1]
    ))
}
pub fn sample_alias(table: &[AliasEntry], u: f32) -> u32 {
    assert!(!table.is_empty());
    call!(ffi::orr_sample_alias(table.as_ptr(), table.len() as u32, u))
}

/// The colour matrices of `orr.color`, in [`color_matrix`] order.
pub const COLOR_MATRICES: [&str; 10] = [
    "REC709_TO_XYZ",
    "XYZ_TO_REC709",
    "REC2020_TO_XYZ",
    "XYZ_TO_REC2020",
    "P3_TO_XYZ",
    "XYZ_TO_P3",
    "REC709_TO_REC2020",
    "REC2020_TO_REC709",
    "P3_TO_REC2020",
    "REC2020_TO_P3",
];
/// Matrix `which` of [`COLOR_MATRICES`], row-major.
pub fn color_matrix(which: usize) -> M3 {
    out!(M3, |o| ffi::orr_color_matrix(which as u32, o.as_mut_ptr()))
}
pub fn luminance_rec709(c: V3) -> f32 {
    call!(ffi::orr_luminance_rec709(c.as_ptr()))
}
pub fn luminance_rec2020(c: V3) -> f32 {
    call!(ffi::orr_luminance_rec2020(c.as_ptr()))
}
pub fn srgb_encode(c: V3) -> V3 {
    out!(V3, |o| ffi::orr_srgb_encode(c.as_ptr(), o.as_mut_ptr()))
}
pub fn srgb_decode(c: V3) -> V3 {
    out!(V3, |o| ffi::orr_srgb_decode(c.as_ptr(), o.as_mut_ptr()))
}
pub fn pq_encode(nits: V3) -> V3 {
    out!(V3, |o| ffi::orr_pq_encode(nits.as_ptr(), o.as_mut_ptr()))
}
pub fn pq_decode(signal: V3) -> V3 {
    out!(V3, |o| ffi::orr_pq_decode(signal.as_ptr(), o.as_mut_ptr()))
}
pub fn ictcp_from_rec2020(nits: V3) -> V3 {
    out!(V3, |o| ffi::orr_ictcp_from_rec2020(
        nits.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn rec2020_from_ictcp(ictcp: V3) -> V3 {
    out!(V3, |o| ffi::orr_rec2020_from_ictcp(
        ictcp.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn gamut_compress(c: V3) -> V3 {
    out!(V3, |o| ffi::orr_gamut_compress(c.as_ptr(), o.as_mut_ptr()))
}
