//! orr's and orr_color's Slang math compiled for the CPU (`slang/math.slang`,
//! `slang/color.slang`), with safe wrappers for the tests, and the statistics the sampling
//! tests use.

pub mod stats;

/// `gpu/math.slang` compiled to SPIR-V; building it checks its generated layouts.
pub mod gpu {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

type V3 = [f32; 3];
type V4 = [f32; 4];
type M3 = [f32; 9];
type M34 = [f32; 12];

mod ffi {
    unsafe extern "C" {
        pub(crate) fn orr_orthonormal_basis(n: *const f32, t: *mut f32, b: *mut f32);
        pub(crate) fn orr_quat_to_matrix(q: *const f32, out: *mut f32);
        pub(crate) fn orr_affine_mul(a: *const f32, b: *const f32, out: *mut f32);
        pub(crate) fn orr_transform_point(m: *const f32, p: *const f32, out: *mut f32);
        pub(crate) fn orr_transform_vector(m: *const f32, v: *const f32, out: *mut f32);
        pub(crate) fn orr_transform_normal(m: *const f32, n: *const f32, out: *mut f32);
        pub(crate) fn orr_largest_scale(m: *const f32) -> f32;
        pub(crate) fn orr_linear_inverse(m: *const f32, out: *mut f32);

        pub(crate) fn orr_oct_encode_snorm16(n: *const f32) -> u32;
        pub(crate) fn orr_oct_decode_snorm16(e: u32, out: *mut f32);
        pub(crate) fn orr_oct_encode_tangent(t: *const f32, sign: f32) -> u32;
        pub(crate) fn orr_oct_decode_tangent(e: u32, out: *mut f32);
        pub(crate) fn orr_oct_encode_snorm8(n: *const f32) -> u32;
        pub(crate) fn orr_oct_decode_snorm8(e: u32, out: *mut f32);
        pub(crate) fn orr_unpack_unorm2x16(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_snorm2x16(x: f32, y: f32) -> u32;
        pub(crate) fn orr_unpack_snorm2x16(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_unorm4x8(v: *const f32) -> u32;
        pub(crate) fn orr_unpack_unorm4x8(p: u32, out: *mut f32);
        pub(crate) fn orr_unpack_snorm4x8(p: u32, out: *mut f32);
        pub(crate) fn orr_pack_half2x16(x: f32, y: f32) -> u32;
        pub(crate) fn orr_unpack_half2x16(p: u32, out: *mut f32);
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
        pub(crate) fn orr_sample_uniform_sphere(u0: f32, u1: f32, out: *mut f32);
        pub(crate) fn orr_sample_cone(u0: f32, u1: f32, one_minus_cos: f32, out: *mut f32);
        pub(crate) fn orr_cone_pdf(one_minus_cos: f32) -> f32;
        pub(crate) fn orr_mis_balance(a: f32, b: f32) -> f32;
        pub(crate) fn orr_pack_unorm8(v: f32) -> u32;
        pub(crate) fn orr_unpack_unorm8(p: u32) -> f32;
        pub(crate) fn orr_pack_snorm8(v: f32) -> u32;
        pub(crate) fn orr_unpack_snorm8(p: u32) -> f32;
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

        pub(crate) fn orr_color_primaries_matrix(which: u32, out: *mut f32);
        pub(crate) fn orr_color_luminance_rec709(c: *const f32) -> f32;
        pub(crate) fn orr_color_luminance_rec2020(c: *const f32) -> f32;
        pub(crate) fn orr_color_srgb_encode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_srgb_decode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_pq_encode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_pq_decode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_hlg_encode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_hlg_decode(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_ictcp_from_rec2020(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_rec2020_from_ictcp(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_gamut_compress(c: *const f32, out: *mut f32);
        pub(crate) fn orr_color_cct_from_xy(xy: *const f32) -> f32;
        pub(crate) fn orr_color_duv_from_xy(xy: *const f32, cct: f32) -> f32;
        pub(crate) fn orr_color_planckian_xy(cct: f32, out: *mut f32);
        pub(crate) fn orr_color_xy_from_cct_duv(cct: f32, duv: f32, out: *mut f32);
        pub(crate) fn orr_color_uv_1960_from_xy(xy: *const f32, out: *mut f32);
        pub(crate) fn orr_color_xy_from_uv_1960(uv: *const f32, out: *mut f32);
        pub(crate) fn orr_color_cat16_rec2020(
            from_xy: *const f32,
            to_xy: *const f32,
            out: *mut f32,
        );
        pub(crate) fn orr_color_cie_xyz(nm: f32, out: *mut f32);
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
/// `m` row-major.
pub fn largest_scale(m: [f32; 9]) -> f32 {
    call!(ffi::orr_largest_scale(m.as_ptr()))
}
/// `m` and the result row-major.
pub fn linear_inverse(m: [f32; 9]) -> [f32; 9] {
    out!([f32; 9], |o| ffi::orr_linear_inverse(
        m.as_ptr(),
        o.as_mut_ptr()
    ))
}
pub fn oct_encode_snorm16(n: V3) -> u32 {
    call!(ffi::orr_oct_encode_snorm16(n.as_ptr()))
}
pub fn oct_decode_snorm16(e: u32) -> V3 {
    out!(V3, |o| ffi::orr_oct_decode_snorm16(e, o.as_mut_ptr()))
}
pub fn oct_encode_tangent(t: V3, sign: f32) -> u32 {
    call!(ffi::orr_oct_encode_tangent(t.as_ptr(), sign))
}
pub fn oct_decode_tangent(e: u32) -> V4 {
    out!(V4, |o| ffi::orr_oct_decode_tangent(e, o.as_mut_ptr()))
}
pub fn oct_encode_snorm8(n: V3) -> u32 {
    call!(ffi::orr_oct_encode_snorm8(n.as_ptr()))
}
pub fn oct_decode_snorm8(e: u32) -> V3 {
    out!(V3, |o| ffi::orr_oct_decode_snorm8(e, o.as_mut_ptr()))
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
pub fn unpack_snorm4x8(p: u32) -> V4 {
    out!(V4, |o| ffi::orr_unpack_snorm4x8(p, o.as_mut_ptr()))
}
pub fn pack_half2x16(v: [f32; 2]) -> u32 {
    call!(ffi::orr_pack_half2x16(v[0], v[1]))
}
pub fn unpack_half2x16(p: u32) -> [f32; 2] {
    out!([f32; 2], |o| ffi::orr_unpack_half2x16(p, o.as_mut_ptr()))
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
pub fn sample_uniform_sphere(u: [f32; 2]) -> V3 {
    out!(V3, |o| ffi::orr_sample_uniform_sphere(
        u[0],
        u[1],
        o.as_mut_ptr()
    ))
}
pub fn sample_cone(u: [f32; 2], one_minus_cos: f32) -> V3 {
    out!(V3, |o| ffi::orr_sample_cone(
        u[0],
        u[1],
        one_minus_cos,
        o.as_mut_ptr()
    ))
}
pub fn cone_pdf(one_minus_cos: f32) -> f32 {
    call!(ffi::orr_cone_pdf(one_minus_cos))
}
pub fn mis_balance(a: f32, b: f32) -> f32 {
    call!(ffi::orr_mis_balance(a, b))
}
pub fn pack_unorm8(v: f32) -> u32 {
    call!(ffi::orr_pack_unorm8(v))
}
pub fn unpack_unorm8(p: u32) -> f32 {
    call!(ffi::orr_unpack_unorm8(p))
}
pub fn pack_snorm8(v: f32) -> u32 {
    call!(ffi::orr_pack_snorm8(v))
}
pub fn unpack_snorm8(p: u32) -> f32 {
    call!(ffi::orr_unpack_snorm8(p))
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
/// orr_color's Slang modules (`orr_color.*`).
pub mod color {
    use super::{M3, V3, ffi};

    /// The matrices of `orr_color.primaries`, in [`matrix`] order.
    pub const MATRICES: [&str; 10] = [
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

    /// Matrix `which` of [`MATRICES`], row-major.
    pub fn matrix(which: usize) -> M3 {
        out!(M3, |o| ffi::orr_color_primaries_matrix(
            which as u32,
            o.as_mut_ptr()
        ))
    }
    pub fn luminance_rec709(c: V3) -> f32 {
        call!(ffi::orr_color_luminance_rec709(c.as_ptr()))
    }
    pub fn luminance_rec2020(c: V3) -> f32 {
        call!(ffi::orr_color_luminance_rec2020(c.as_ptr()))
    }
    pub fn srgb_encode(c: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_srgb_encode(
            c.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn srgb_decode(c: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_srgb_decode(
            c.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn pq_encode(nits: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_pq_encode(
            nits.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn pq_decode(signal: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_pq_decode(
            signal.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn hlg_encode(c: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_hlg_encode(
            c.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn hlg_decode(signal: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_hlg_decode(
            signal.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn ictcp_from_rec2020(nits: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_ictcp_from_rec2020(
            nits.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn rec2020_from_ictcp(ictcp: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_rec2020_from_ictcp(
            ictcp.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn gamut_compress(c: V3) -> V3 {
        out!(V3, |o| ffi::orr_color_gamut_compress(
            c.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn cct_from_xy(xy: [f32; 2]) -> f32 {
        call!(ffi::orr_color_cct_from_xy(xy.as_ptr()))
    }
    pub fn duv_from_xy(xy: [f32; 2], cct: f32) -> f32 {
        call!(ffi::orr_color_duv_from_xy(xy.as_ptr(), cct))
    }
    pub fn planckian_xy(cct: f32) -> [f32; 2] {
        out!([f32; 2], |o| ffi::orr_color_planckian_xy(
            cct,
            o.as_mut_ptr()
        ))
    }
    pub fn xy_from_cct_duv(cct: f32, duv: f32) -> [f32; 2] {
        out!([f32; 2], |o| ffi::orr_color_xy_from_cct_duv(
            cct,
            duv,
            o.as_mut_ptr()
        ))
    }
    pub fn uv_1960_from_xy(xy: [f32; 2]) -> [f32; 2] {
        out!([f32; 2], |o| ffi::orr_color_uv_1960_from_xy(
            xy.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn xy_from_uv_1960(uv: [f32; 2]) -> [f32; 2] {
        out!([f32; 2], |o| ffi::orr_color_xy_from_uv_1960(
            uv.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn cat16_rec2020(from_xy: [f32; 2], to_xy: [f32; 2]) -> M3 {
        out!(M3, |o| ffi::orr_color_cat16_rec2020(
            from_xy.as_ptr(),
            to_xy.as_ptr(),
            o.as_mut_ptr()
        ))
    }
    pub fn cie_xyz(nm: f32) -> V3 {
        out!(V3, |o| ffi::orr_color_cie_xyz(nm, o.as_mut_ptr()))
    }
}
