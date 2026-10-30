//! The OpenPBR 1.1.1 reference BSDF on the CPU and the harness that checks a BSDF with it.
//!
//! Two builds of adobe/openpbr-bsdf implement [`Bsdf`]: [`Reference`], its own C++ with
//! GLM, and [`SlangCpu`], `orr.openpbr` through Slang's C++ target with the tables compiled
//! in. [`harness`] holds the checks that hold for any correct BSDF (energy, sampling against
//! the density, reciprocity) and the comparison of two BSDFs; [`report`] runs them over
//! [`materials::grid`] and writes the tables. A real-time approximation joins the harness by
//! implementing [`Bsdf`].
//!
//! Directions are unit vectors in the shading frame: the normal is +z and the tangent +x.

pub mod harness;
pub mod materials;
pub mod report;

/// `gpu/` compiled to SPIR-V: `openpbr_cases` runs the same calls with heap LUTs.
pub mod gpu {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

pub use gpu::types::Inputs;

type V3 = [f32; 3];

impl Default for Inputs {
    /// OpenPBR 1.1.1's defaults (`openpbr_make_default_resolved_inputs`) in vacuum.
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            base_color: [0.8; 3],
            base_diffuse_roughness: 0.0,
            base_metalness: 0.0,
            subsurface_weight: 0.0,
            subsurface_color: [0.8; 3],
            subsurface_radius: 1.0,
            subsurface_radius_scale: [1.0, 0.5, 0.25],
            subsurface_scatter_anisotropy: 0.0,
            specular_weight: 1.0,
            specular_color: [1.0; 3],
            specular_roughness: 0.3,
            specular_roughness_anisotropy: 0.0,
            specular_ior: 1.5,
            specular_rotation: [1.0, 0.0],
            coat_weight: 0.0,
            coat_color: [1.0; 3],
            coat_roughness: 0.0,
            coat_roughness_anisotropy: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            coat_rotation: [1.0, 0.0],
            fuzz_weight: 0.0,
            fuzz_color: [1.0; 3],
            fuzz_roughness: 0.5,
            transmission_weight: 0.0,
            transmission_color: [1.0; 3],
            transmission_depth: 0.0,
            transmission_scatter: [0.0; 3],
            transmission_scatter_anisotropy: 0.0,
            transmission_dispersion_scale: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            thin_film_weight: 0.0,
            thin_film_thickness: 0.5,
            thin_film_ior: 1.4,
            emission_luminance: 0.0,
            emission_color: [1.0; 3],
            geometry_opacity: 1.0,
            geometry_thin_walled: 0,
            exterior_ior: 1.0,
        }
    }
}

/// `openpbr_eval`: BSDF times the cosine of the light direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Eval {
    pub diffuse: V3,
    pub specular: V3,
}

impl Eval {
    pub fn total(&self) -> V3 {
        [0, 1, 2].map(|c| self.diffuse[c] + self.specular[c])
    }
}

/// `openpbr_sample` when it succeeds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub wi: V3,
    /// BSDF times cosine over the density, over every lobe.
    pub weight: V3,
    /// Density per solid angle; for an ideal-specular lobe, the discrete probability.
    pub pdf: f32,
    pub lobe: Lobe,
}

/// `OpenPBR_BsdfLobeType` flags of a sampled direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lobe(pub u32);

impl Lobe {
    pub const REFLECTION: u32 = 1;
    pub const TRANSMISSION: u32 = 2;
    pub const DIFFUSE: u32 = 4;
    pub const GLOSSY: u32 = 8;
    pub const SPECULAR: u32 = 16;
    pub const VOLUME: u32 = 32;

    pub fn is(self, flag: u32) -> bool {
        self.0 & flag != 0
    }

    /// For example "reflection glossy" or "transmission specular volume".
    pub fn name(self) -> String {
        let names = [
            (Self::REFLECTION, "reflection"),
            (Self::TRANSMISSION, "transmission"),
            (Self::DIFFUSE, "diffuse"),
            (Self::GLOSSY, "glossy"),
            (Self::SPECULAR, "specular"),
            (Self::VOLUME, "volume"),
        ];
        let parts: Vec<&str> = names
            .iter()
            .filter(|(f, _)| self.is(*f))
            .map(|(_, n)| *n)
            .collect();
        if parts.is_empty() {
            "none".to_owned()
        } else {
            parts.join(" ")
        }
    }
}

/// A BSDF the harness can drive: prepared once per material and view, then evaluated,
/// sampled and asked for densities.
pub trait Bsdf {
    /// Prepares `inputs` seen from `view` (unit, toward the viewer).
    fn prepare(&mut self, inputs: &Inputs, view: V3);
    fn eval(&self, wi: V3) -> Eval;
    fn pdf(&self, wi: V3) -> f32;
    /// `None` when the BSDF produces no direction for `u`.
    fn sample(&self, u: V3) -> Option<Sample>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Volume {
    pub extinction: V3,
    pub albedo: V3,
    pub anisotropy: f32,
}

macro_rules! openpbr_bsdf {
    ($(#[$doc:meta])* $name:ident, $inputs_size:ident, $prepared_size:ident, $prepare:ident,
     $eval:ident, $pdf:ident, $sample:ident, $volume:ident, $emission:ident) => {
        $(#[$doc])*
        pub struct $name {
            prepared: Vec<u128>,
        }

        impl $name {
            pub fn new() -> Self {
                // SAFETY: the size queries read nothing.
                let (inputs, prepared) = unsafe { (ffi::$inputs_size(), ffi::$prepared_size()) };
                assert_eq!(
                    inputs as usize,
                    size_of::<Inputs>(),
                    "{} lays out Inputs differently from the generated struct",
                    stringify!($name)
                );
                let mut bsdf = Self {
                    prepared: vec![0; (prepared as usize).div_ceil(16)],
                };
                bsdf.prepare(&Inputs::default(), [0.0, 0.0, 1.0]);
                bsdf
            }

            /// The homogeneous interior medium of the prepared material.
            pub fn volume(&self) -> Volume {
                let mut v = Volume::default();
                // SAFETY: `prepared` holds a prepared BSDF; the outputs are 3, 3 and 1 floats.
                unsafe {
                    ffi::$volume(
                        self.prepared.as_ptr().cast(),
                        v.extinction.as_mut_ptr(),
                        v.albedo.as_mut_ptr(),
                        &mut v.anisotropy,
                    )
                };
                v
            }

            /// Emission in nits after the coat and fuzz.
            pub fn emission(&self) -> V3 {
                let mut e = [0.0; 3];
                // SAFETY: as in `volume`.
                unsafe { ffi::$emission(self.prepared.as_ptr().cast(), e.as_mut_ptr()) };
                e
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Bsdf for $name {
            fn prepare(&mut self, inputs: &Inputs, view: V3) {
                // SAFETY: `prepared` has the size the library reports, 16-byte aligned.
                unsafe { ffi::$prepare(inputs, view.as_ptr(), self.prepared.as_mut_ptr().cast()) };
            }

            fn eval(&self, wi: V3) -> Eval {
                let mut e = Eval::default();
                // SAFETY: `prepared` holds a prepared BSDF; the outputs are 3 floats each.
                unsafe {
                    ffi::$eval(
                        self.prepared.as_ptr().cast(),
                        wi.as_ptr(),
                        e.diffuse.as_mut_ptr(),
                        e.specular.as_mut_ptr(),
                    )
                };
                e
            }

            fn pdf(&self, wi: V3) -> f32 {
                // SAFETY: as in `eval`.
                unsafe { ffi::$pdf(self.prepared.as_ptr().cast(), wi.as_ptr()) }
            }

            fn sample(&self, u: V3) -> Option<Sample> {
                let (mut wi, mut weight, mut pdf, mut lobe) = ([0.0; 3], [0.0; 3], 0.0, 0);
                // SAFETY: as in `eval`.
                unsafe {
                    ffi::$sample(
                        self.prepared.as_ptr().cast(),
                        u.as_ptr(),
                        wi.as_mut_ptr(),
                        weight.as_mut_ptr(),
                        &mut pdf,
                        &mut lobe,
                    )
                };
                (pdf > 0.0).then_some(Sample {
                    wi,
                    weight,
                    pdf,
                    lobe: Lobe(lobe),
                })
            }
        }
    };
}

openpbr_bsdf!(
    /// adobe/openpbr-bsdf's C++ build with GLM: the reference.
    Reference,
    ref_openpbr_inputs_size,
    ref_openpbr_prepared_size,
    ref_openpbr_prepare,
    ref_openpbr_eval,
    ref_openpbr_pdf,
    ref_openpbr_sample,
    ref_openpbr_volume,
    ref_openpbr_emission
);

openpbr_bsdf!(
    /// `orr.openpbr` compiled by Slang to C++, with the tables as arrays.
    SlangCpu,
    slang_openpbr_inputs_size,
    slang_openpbr_prepared_size,
    slang_openpbr_prepare,
    slang_openpbr_eval,
    slang_openpbr_pdf,
    slang_openpbr_sample,
    slang_openpbr_volume,
    slang_openpbr_emission
);

mod ffi {
    use super::Inputs;
    use std::ffi::c_void;

    macro_rules! entry_points {
        ($($inputs_size:ident $prepared_size:ident $prepare:ident $eval:ident $pdf:ident
           $sample:ident $volume:ident $emission:ident);*) => {
            unsafe extern "C" {
                $(
                    pub(crate) fn $inputs_size() -> u32;
                    pub(crate) fn $prepared_size() -> u32;
                    pub(crate) fn $prepare(inputs: *const Inputs, view: *const f32, out: *mut c_void);
                    pub(crate) fn $eval(p: *const c_void, wi: *const f32, diffuse: *mut f32, specular: *mut f32);
                    pub(crate) fn $pdf(p: *const c_void, wi: *const f32) -> f32;
                    pub(crate) fn $sample(p: *const c_void, u: *const f32, wi: *mut f32, weight: *mut f32,
                        pdf: *mut f32, lobe: *mut u32);
                    pub(crate) fn $volume(p: *const c_void, extinction: *mut f32, albedo: *mut f32,
                        anisotropy: *mut f32);
                    pub(crate) fn $emission(p: *const c_void, out: *mut f32);
                )*
            }
        };
    }

    entry_points!(
        ref_openpbr_inputs_size ref_openpbr_prepared_size ref_openpbr_prepare ref_openpbr_eval
            ref_openpbr_pdf ref_openpbr_sample ref_openpbr_volume ref_openpbr_emission;
        slang_openpbr_inputs_size slang_openpbr_prepared_size slang_openpbr_prepare
            slang_openpbr_eval slang_openpbr_pdf slang_openpbr_sample slang_openpbr_volume
            slang_openpbr_emission
    );
}
