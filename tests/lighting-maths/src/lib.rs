//! orr-lighting's real-time OpenPBR BSDF (`bsdf`) through Slang's C++ target with the tables
//! compiled in, as an [`openpbr_ref::Bsdf`] the OpenPBR harness can check against
//! adobe/openpbr-bsdf. With feature `gpu`, `RealTimeGpu` runs the same `bsdf` in a Vulkan
//! compute kernel with orr-lighting's `BsdfTables`.
//!
//! Directions are unit vectors in the shading frame: the normal is +z and the tangent +x.

use std::ffi::c_void;

use openpbr_ref::{Bsdf, Eval, Inputs, Lobe, Sample, Volume};
use shader_maths::stats::Rng;

/// `gpu/` compiled to SPIR-V: `bsdf_cases` runs the same calls with heap LUTs.
pub mod gpu {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

#[cfg(feature = "gpu")]
mod real_time_gpu;

#[cfg(feature = "gpu")]
pub use real_time_gpu::{MAX_MATERIALS, RealTimeGpu};

type V3 = [f32; 3];

/// Directional albedo from the view, by kind (`LobeAlbedo`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LobeAlbedo {
    pub diffuse: V3,
    pub specular: V3,
    pub transmission: V3,
    pub subsurface: V3,
}

/// The real-time BSDF, prepared once per material and view.
pub struct RealTime {
    prepared: Vec<u128>,
    collapse: bool,
}

impl RealTime {
    pub fn new() -> Self {
        Self::with_collapse(false)
    }

    /// Prepares every material through `bsdf_collapse` first.
    pub fn collapsed() -> Self {
        Self::with_collapse(true)
    }

    fn with_collapse(collapse: bool) -> Self {
        // SAFETY: the size queries read nothing.
        let (inputs, prepared) =
            unsafe { (ffi::lighting_bsdf_inputs_size(), ffi::lighting_bsdf_size()) };
        assert_eq!(
            inputs as usize,
            size_of::<Inputs>(),
            "bsdf_cpu lays out Inputs differently from the generated struct"
        );
        let mut bsdf = Self {
            prepared: vec![0; (prepared as usize).div_ceil(16)],
            collapse,
        };
        bsdf.prepare(&Inputs::default(), [0.0, 0.0, 1.0]);
        bsdf
    }

    fn ptr(&self) -> *const c_void {
        self.prepared.as_ptr().cast()
    }

    /// `Bsdf.albedo`: the estimates that choose lobes and feed guides.
    pub fn albedo(&self) -> LobeAlbedo {
        let mut out = [0.0f32; 12];
        // SAFETY: `prepared` holds a prepared BSDF; the output is 12 floats.
        unsafe { ffi::lighting_bsdf_albedo(self.ptr(), out.as_mut_ptr()) };
        let part = |i: usize| [out[i], out[i + 1], out[i + 2]];
        LobeAlbedo {
            diffuse: part(0),
            specular: part(3),
            transmission: part(6),
            subsurface: part(9),
        }
    }

    /// `Bsdf.emission`.
    pub fn emission(&self) -> V3 {
        let mut e = [0.0; 3];
        // SAFETY: as in `albedo`; the output is 3 floats.
        unsafe { ffi::lighting_bsdf_emission(self.ptr(), e.as_mut_ptr()) };
        e
    }

    /// `Bsdf.subsurface_entry`.
    pub fn subsurface_entry(&self, u: V3) -> Option<Sample> {
        // SAFETY: as in `albedo`.
        sample_with(|wi, weight, pdf, lobe| unsafe {
            ffi::lighting_bsdf_subsurface_entry(self.ptr(), u.as_ptr(), wi, weight, pdf, lobe);
        })
    }
}

/// `bsdf_subsurface_volume` of the material.
pub fn subsurface_volume(inputs: &Inputs) -> Volume {
    let mut v = Volume::default();
    // SAFETY: the outputs are 3, 3 and 1 floats.
    unsafe {
        ffi::lighting_bsdf_volume(
            inputs,
            v.extinction.as_mut_ptr(),
            v.albedo.as_mut_ptr(),
            &mut v.anisotropy,
        );
    };
    v
}

impl Default for RealTime {
    fn default() -> Self {
        Self::new()
    }
}

fn sample_with(call: impl FnOnce(*mut f32, *mut f32, *mut f32, *mut u32)) -> Option<Sample> {
    let (mut wi, mut weight, mut pdf, mut lobe) = ([0.0; 3], [0.0; 3], 0.0, 0);
    call(wi.as_mut_ptr(), weight.as_mut_ptr(), &mut pdf, &mut lobe);
    (pdf > 0.0).then_some(Sample {
        wi,
        weight,
        pdf,
        lobe: Lobe(lobe),
    })
}

impl Bsdf for RealTime {
    fn prepare(&mut self, inputs: &Inputs, view: V3) {
        // SAFETY: `prepared` has the size the library reports, 16-byte aligned.
        unsafe {
            ffi::lighting_bsdf_prepare(
                inputs,
                view.as_ptr(),
                u32::from(self.collapse),
                self.prepared.as_mut_ptr().cast(),
            );
        };
    }

    fn eval(&self, wi: V3) -> Eval {
        let mut e = Eval::default();
        // SAFETY: `prepared` holds a prepared BSDF; the outputs are 3 floats each.
        unsafe {
            ffi::lighting_bsdf_eval(
                self.ptr(),
                wi.as_ptr(),
                e.diffuse.as_mut_ptr(),
                e.specular.as_mut_ptr(),
            );
        };
        e
    }

    fn pdf(&self, wi: V3) -> f32 {
        // SAFETY: as in `eval`.
        unsafe { ffi::lighting_bsdf_pdf(self.ptr(), wi.as_ptr()) }
    }

    fn sample(&self, u: V3) -> Option<Sample> {
        // SAFETY: as in `eval`.
        sample_with(|wi, weight, pdf, lobe| unsafe {
            ffi::lighting_bsdf_sample(self.ptr(), u.as_ptr(), wi, weight, pdf, lobe);
        })
    }
}

/// Random parameters at their limits (0, 1 or uniform), seen from a random view; case `i`
/// puts every fifth view in the surface plane, every fifth just above and every fifth just
/// below it.
pub fn edge_case(rng: &mut Rng, i: usize) -> (Inputs, V3) {
    fn unit(rng: &mut Rng) -> f32 {
        match rng.next_u32() % 4 {
            0 => 0.0,
            1 => 1.0,
            _ => rng.f32(),
        }
    }
    fn colour(rng: &mut Rng) -> V3 {
        [unit(rng), unit(rng), unit(rng)]
    }
    let angle = rng.range(0.0, std::f32::consts::TAU);
    let inputs = Inputs {
        base_weight: unit(rng),
        base_color: colour(rng),
        base_diffuse_roughness: unit(rng),
        base_metalness: unit(rng),
        subsurface_weight: unit(rng),
        subsurface_color: colour(rng),
        subsurface_scatter_anisotropy: rng.range(-1.0, 1.0),
        specular_weight: unit(rng),
        specular_color: colour(rng),
        specular_roughness: unit(rng),
        specular_roughness_anisotropy: unit(rng),
        specular_ior: [1.0, 1.0001, 1.5, 3.0][(rng.next_u32() % 4) as usize],
        specular_rotation: [angle.cos(), angle.sin()],
        coat_weight: unit(rng),
        coat_color: colour(rng),
        coat_roughness: unit(rng),
        coat_ior: [1.0, 1.6, 3.0][(rng.next_u32() % 3) as usize],
        coat_darkening: unit(rng),
        fuzz_weight: unit(rng),
        fuzz_color: colour(rng),
        fuzz_roughness: unit(rng),
        transmission_weight: unit(rng),
        transmission_color: colour(rng),
        thin_film_weight: unit(rng),
        thin_film_thickness: unit(rng),
        emission_luminance: 10.0,
        geometry_thin_walled: rng.next_u32() % 2,
        exterior_ior: [1.0, 1.33, 2.0][(rng.next_u32() % 3) as usize],
        ..Inputs::default()
    };
    let mut view = rng.unit();
    match i % 5 {
        0 => view[2] = 0.0,
        1 => view[2] = 1e-7,
        2 => view[2] = -1e-6,
        _ => {}
    }
    let len = view.iter().map(|x| x * x).sum::<f32>().sqrt();
    (inputs, view.map(|x| x / len))
}

/// A random light direction, one in four in the surface plane.
pub fn edge_direction(rng: &mut Rng) -> V3 {
    let wi = rng.unit();
    if rng.next_u32().is_multiple_of(4) {
        let len = (wi[0] * wi[0] + wi[1] * wi[1]).sqrt();
        [wi[0] / len, wi[1] / len, 0.0]
    } else {
        wi
    }
}

/// Every component finite and at least 0.
pub fn finite_non_negative(v: V3) -> bool {
    v.iter().all(|x| x.is_finite() && *x >= 0.0)
}

mod ffi {
    use super::Inputs;
    use std::ffi::c_void;

    unsafe extern "C" {
        pub(crate) fn lighting_bsdf_inputs_size() -> u32;
        pub(crate) fn lighting_bsdf_size() -> u32;
        pub(crate) fn lighting_bsdf_prepare(
            inputs: *const Inputs,
            view: *const f32,
            collapse: u32,
            out: *mut c_void,
        );
        pub(crate) fn lighting_bsdf_eval(
            p: *const c_void,
            wi: *const f32,
            diffuse: *mut f32,
            specular: *mut f32,
        );
        pub(crate) fn lighting_bsdf_pdf(p: *const c_void, wi: *const f32) -> f32;
        pub(crate) fn lighting_bsdf_sample(
            p: *const c_void,
            u: *const f32,
            wi: *mut f32,
            weight: *mut f32,
            pdf: *mut f32,
            lobe: *mut u32,
        );
        pub(crate) fn lighting_bsdf_subsurface_entry(
            p: *const c_void,
            u: *const f32,
            wi: *mut f32,
            weight: *mut f32,
            pdf: *mut f32,
            lobe: *mut u32,
        );
        pub(crate) fn lighting_bsdf_albedo(p: *const c_void, out: *mut f32);
        pub(crate) fn lighting_bsdf_emission(p: *const c_void, out: *mut f32);
        pub(crate) fn lighting_bsdf_volume(
            inputs: *const Inputs,
            extinction: *mut f32,
            albedo: *mut f32,
            anisotropy: *mut f32,
        );
    }
}
