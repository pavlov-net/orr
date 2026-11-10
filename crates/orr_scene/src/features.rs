//! What a scene may hold that its paths' kernels are compiled for.

use std::collections::HashMap;
use std::ops::{BitOr, BitOrAssign};

use crate::MaterialGpu;
use crate::material::material_flags as f;

/// What a scene may hold in a frame. A bit is set wherever the scene cannot rule its feature
/// out: from the records in the material table, which every scene shares, and from what the
/// scene itself holds. A kernel compiled without a feature runs only while its bit is clear.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SceneFeatures(pub u32);

impl SceneFeatures {
    pub const NONE: Self = Self(0);
    /// Paths through a material may take a random walk: through its subsurface medium or its
    /// dense scattering interior.
    pub const WALKING: Self = Self(1 << 0);
    /// A material's surfaces bound an interior: a dielectric that is not thin-walled. Without
    /// one no point lies in an interior a view does not name.
    pub const INTERIORS: Self = Self(1 << 1);
    /// One of those interiors scatters, read without a surface point: its material's
    /// transmission depth and scatter are above 0.
    pub const SCATTERING_INTERIORS: Self = Self(1 << 2);
    /// A material's transmission has a dispersion scale and an Abbe number above 0.
    pub const DISPERSIVE: Self = Self(1 << 3);
    /// The scene holds a medium proxy.
    pub const MEDIUM_PROXIES: Self = Self(1 << 4);
    /// The scene holds a directional light.
    pub const DIRECTIONAL_LIGHTS: Self = Self(1 << 5);
    /// The scene holds a sphere or rect light.
    pub const LOCAL_LIGHTS: Self = Self(1 << 6);
    /// The scene holds a smoke entry: a view's [`RasterView::smoke_args`](crate::RasterView)
    /// is then set.
    pub const SMOKE: Self = Self(1 << 7);
    /// Every bit a material record sets.
    const MATERIAL: Self = Self(
        Self::WALKING.0 | Self::INTERIORS.0 | Self::SCATTERING_INTERIORS.0 | Self::DISPERSIVE.0,
    );

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// What record `m` may show. A linked surface may set any lobe, so its record counts for
    /// every feature a record can have.
    pub(crate) fn of_material(m: &MaterialGpu) -> Self {
        if m.surface != 0 {
            return Self::MATERIAL;
        }
        let thin = m.flags & f::THIN_WALLED != 0;
        let scatters =
            m.transmission_depth > 0.0 && m.transmission_scatter.iter().any(|&c| c > 0.0);
        let bounds = m.flags & (f::DIELECTRIC | f::THIN_WALLED) == f::DIELECTRIC;
        let walks =
            !thin && (m.subsurface_weight > 0.0 || (m.transmission_weight > 0.0 && scatters));
        let disperses =
            m.transmission_dispersion_scale > 0.0 && m.transmission_dispersion_abbe_number > 0.0;
        let mut out = Self::NONE;
        for (on, bit) in [
            (walks, Self::WALKING),
            (bounds, Self::INTERIORS),
            (bounds && scatters, Self::SCATTERING_INTERIORS),
            (disperses, Self::DISPERSIVE),
        ] {
            if on {
                out |= bit;
            }
        }
        out
    }
}

impl BitOr for SceneFeatures {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOrAssign for SceneFeatures {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// The features of each record of the material table, and how many records hold each.
#[derive(Default)]
pub(crate) struct MaterialFeatures {
    held: HashMap<u32, u32>,
    counts: [u32; 32],
}

impl MaterialFeatures {
    /// Records that material `id` now holds `features`; none frees it.
    pub(crate) fn set(&mut self, id: u32, features: SceneFeatures) {
        let was = if features.0 == 0 {
            self.held.remove(&id)
        } else {
            self.held.insert(id, features.0)
        }
        .unwrap_or(0);
        for (bit, count) in self.counts.iter_mut().enumerate() {
            let mask = 1 << bit;
            *count = *count + u32::from(features.0 & mask != 0) - u32::from(was & mask != 0);
        }
    }

    /// The features any record holds.
    pub(crate) fn any(&self) -> SceneFeatures {
        let mut out = 0;
        for (bit, &count) in self.counts.iter().enumerate() {
            if count > 0 {
                out |= 1 << bit;
            }
        }
        SceneFeatures(out)
    }
}
