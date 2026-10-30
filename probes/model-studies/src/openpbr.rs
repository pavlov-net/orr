//! FFI to the adobe/openpbr-bsdf C++ reference (cpp/shim.cpp).

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    pub base_weight: f32,
    pub base_color: [f32; 3],
    pub diffuse_roughness: f32,
    pub metalness: f32,
    pub sss_weight: f32,
    pub sss_color: [f32; 3],
    pub sss_radius: f32,
    pub sss_radius_scale: [f32; 3],
    pub sss_g: f32,
    pub spec_weight: f32,
    pub spec_color: [f32; 3],
    pub spec_rough: f32,
    pub spec_aniso: f32,
    pub spec_ior: f32,
    pub spec_rot_cs: [f32; 2],
    pub coat_weight: f32,
    pub coat_color: [f32; 3],
    pub coat_rough: f32,
    pub coat_ior: f32,
    pub coat_darkening: f32,
    pub fuzz_weight: f32,
    pub fuzz_color: [f32; 3],
    pub fuzz_rough: f32,
    pub thin_walled: i32,
    pub t: [f32; 3],
    pub b: [f32; 3],
    pub n: [f32; 3],
}

impl Default for Mat {
    /// OpenPBR 1.1.1 defaults, Z-up frame.
    fn default() -> Self {
        Mat {
            base_weight: 1.0,
            base_color: [0.8; 3],
            diffuse_roughness: 0.0,
            metalness: 0.0,
            sss_weight: 0.0,
            sss_color: [0.8; 3],
            sss_radius: 1.0,
            sss_radius_scale: [1.0, 0.5, 0.25],
            sss_g: 0.0,
            spec_weight: 1.0,
            spec_color: [1.0; 3],
            spec_rough: 0.3,
            spec_aniso: 0.0,
            spec_ior: 1.5,
            spec_rot_cs: [1.0, 0.0],
            coat_weight: 0.0,
            coat_color: [1.0; 3],
            coat_rough: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            fuzz_weight: 0.0,
            fuzz_color: [1.0; 3],
            fuzz_rough: 0.5,
            thin_walled: 0,
            t: [1.0, 0.0, 0.0],
            b: [0.0, 1.0, 0.0],
            n: [0.0, 0.0, 1.0],
        }
    }
}

unsafe extern "C" {
    fn orr_prepared_size() -> usize;
    fn orr_prepare(m: *const Mat, view: *const f32, out: *mut u8);
    fn orr_eval(p: *const u8, wi: *const f32, out: *mut f32);
    fn orr_pdf(p: *const u8, wi: *const f32) -> f32;
    fn orr_sample(p: *const u8, rnd: *const f32, wi: *mut f32, w: *mut f32, pdf: *mut f32, lobe: *mut u32) -> i32;
    fn orr_volume(p: *const u8, ext: *mut f32, albedo: *mut f32, g: *mut f32);
}

pub const LOBE_TRANSMISSION: u32 = 2;

pub struct Prepared {
    buf: Vec<u64>,
}

pub struct Sample {
    pub wi: [f32; 3],
    pub weight: [f32; 3],
    pub pdf: f32,
    pub lobe: u32,
}

impl Prepared {
    pub fn new(m: &Mat, view: [f32; 3]) -> Self {
        let n = unsafe { orr_prepared_size() };
        let mut buf = vec![0u64; n.div_ceil(8)];
        unsafe { orr_prepare(m, view.as_ptr(), buf.as_mut_ptr() as *mut u8) };
        Prepared { buf }
    }
    fn ptr(&self) -> *const u8 {
        self.buf.as_ptr() as *const u8
    }
    /// BSDF times |cos| for light direction `wi`.
    pub fn eval(&self, wi: [f32; 3]) -> [f32; 3] {
        let mut o = [0.0; 3];
        unsafe { orr_eval(self.ptr(), wi.as_ptr(), o.as_mut_ptr()) };
        o
    }
    pub fn pdf(&self, wi: [f32; 3]) -> f32 {
        unsafe { orr_pdf(self.ptr(), wi.as_ptr()) }
    }
    pub fn sample(&self, rnd: [f32; 3]) -> Option<Sample> {
        let mut s = Sample { wi: [0.0; 3], weight: [0.0; 3], pdf: 0.0, lobe: 0 };
        let ok = unsafe { orr_sample(self.ptr(), rnd.as_ptr(), s.wi.as_mut_ptr(), s.weight.as_mut_ptr(), &mut s.pdf, &mut s.lobe) };
        (ok != 0).then_some(s)
    }
    /// Interior volume: extinction (1/m), single-scattering albedo, HG g.
    pub fn volume(&self) -> ([f32; 3], [f32; 3], f32) {
        let (mut e, mut a, mut g) = ([0.0; 3], [0.0; 3], 0.0);
        unsafe { orr_volume(self.ptr(), e.as_mut_ptr(), a.as_mut_ptr(), &mut g) };
        (e, a, g)
    }
}
