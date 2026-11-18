//! The masks path pipelines and the masked support kernels are made for (`specialize.slang`): what
//! their paths may meet, `path_may` and its bits in the low word and `scene_may.slang`'s
//! `path_may2`, what a scene's surface evaluations and hits may meet, in the high one.

use crate::shaders::consts;

/// The specialization constant id of `path_may`.
pub(crate) const PATH_MAY_ID: u32 = 110;
pub(crate) const MAY_STACK: u64 = consts::MAY_STACK as u64;
pub(crate) const MAY_DISPERSION: u64 = consts::MAY_DISPERSION as u64;
pub(crate) const MAY_CAUSTICS: u64 = consts::MAY_CAUSTICS as u64;
pub(crate) const MAY_VIEW: u64 = consts::MAY_VIEW as u64;
pub(crate) const MAY_BACKGROUND: u64 = consts::MAY_BACKGROUND as u64;
pub(crate) const MAY_UPDATE: u64 = consts::MAY_UPDATE as u64;
pub(crate) const MAY_PHOTON: u64 = consts::MAY_PHOTON as u64;
pub(crate) const MAY_DEBUG: u64 = consts::MAY_DEBUG as u64;
pub(crate) const MAY_REFERENCE: u64 = consts::MAY_REFERENCE as u64;
pub(crate) const MAY_FROST: u64 = consts::MAY_FROST as u64;
pub(crate) const MAY_LENS: u64 = consts::MAY_LENS as u64;
pub(crate) const MAY_WALK: u64 = consts::MAY_WALK as u64;
pub(crate) const MAY_LOCAL: u64 = consts::MAY_LOCAL as u64;
pub(crate) const MAY_TRACED: u64 = consts::MAY_TRACED as u64;
pub(crate) const MAY_EMITTER: u64 = consts::MAY_EMITTER as u64;
pub(crate) const MAY_FROXELS: u64 = consts::MAY_FROXELS as u64;
pub(crate) const MAY_CACHE: u64 = consts::MAY_CACHE as u64;
pub(crate) const MAY_COAT: u64 = consts::MAY_COAT as u64;
pub(crate) const MAY_FUZZ: u64 = consts::MAY_FUZZ as u64;
pub(crate) const MAY_FILM: u64 = consts::MAY_FILM as u64;
pub(crate) const MAY_SUBSURFACE: u64 = consts::MAY_SUBSURFACE as u64;
/// The lobes a plain surface lacks.
pub(crate) const MAY_LOBES: u64 = consts::MAY_LOBES as u64;
pub(crate) const MAY_INTERIORS: u64 = consts::MAY_INTERIORS as u64;
pub(crate) const MAY_LAYERS: u64 = consts::MAY_LAYERS as u64;
pub(crate) const MAY_COVERAGE: u64 = consts::MAY_COVERAGE as u64;
pub(crate) const MAY_PARTICLES: u64 = consts::MAY_PARTICLES as u64;
pub(crate) const MAY_LOCAL_LIGHTS: u64 = consts::MAY_LOCAL_LIGHTS as u64;
pub(crate) const MAY_DIRECTIONAL: u64 = consts::MAY_DIRECTIONAL as u64;
pub(crate) const MAY_EMISSIVE: u64 = consts::MAY_EMISSIVE as u64;
pub(crate) const MAY_PROXIES: u64 = consts::MAY_PROXIES as u64;
pub(crate) const MAY_SMOKE: u64 = consts::MAY_SMOKE as u64;
pub(crate) const MAY_SCATTERING: u64 = consts::MAY_SCATTERING as u64;
pub(crate) const MAY_MEDIUM: u64 = consts::MAY_MEDIUM as u64;
/// Everything: what a pipeline made without a value may meet.
pub(crate) const MAY_ALL: u64 = u64::MAX;
