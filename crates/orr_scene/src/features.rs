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
    /// A material has a coat, fuzz, a thin film or a subsurface weight above 0.
    pub const COAT: Self = Self(1 << 8);
    pub const FUZZ: Self = Self(1 << 9);
    pub const THIN_FILM: Self = Self(1 << 10);
    pub const SUBSURFACE: Self = Self(1 << 11);
    /// Geometry may have the class of a dielectric or a presence layer: a material that gave
    /// a mesh its class was one, or a producer entry is.
    pub const LAYERS: Self = Self(1 << 12);
    /// Geometry may hold triangles that are not opaque in traversal, particles apart: a
    /// material that a mesh was built with tested alpha, was a presence layer or a thin
    /// dielectric, or a producer entry is not opaque.
    pub const COVERAGE: Self = Self(1 << 13);
    /// The scene holds a particle or smoke entry.
    pub const PARTICLES: Self = Self(1 << 14);
    /// A material emits.
    pub const EMISSIVE: Self = Self(1 << 15);
    /// What a mesh keeps from the material it was built with, whatever the record holds now:
    /// once a record has shown one of these, the table keeps it.
    const BUILT: Self = Self(Self::LAYERS.0 | Self::COVERAGE.0);
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// What record `m` may show. A linked surface may set any lobe, so its record counts for
    /// every lobe, for walks and for emission; what a material's flags and its own medium
    /// decide stays as the record has it.
    pub(crate) fn of_material(m: &MaterialGpu) -> Self {
        let linked = m.surface != 0;
        let thin = m.flags & f::THIN_WALLED != 0;
        let scatters =
            m.transmission_depth > 0.0 && m.transmission_scatter.iter().any(|&c| c > 0.0);
        let bounds = m.flags & (f::DIELECTRIC | f::THIN_WALLED) == f::DIELECTRIC;
        let walks = linked
            || (!thin && (m.subsurface_weight > 0.0 || (m.transmission_weight > 0.0 && scatters)));
        let disperses =
            m.transmission_dispersion_scale > 0.0 && m.transmission_dispersion_abbe_number > 0.0;
        let mut out = Self::NONE;
        for (on, bit) in [
            (walks, Self::WALKING),
            (bounds, Self::INTERIORS),
            (bounds && scatters, Self::SCATTERING_INTERIORS),
            (disperses, Self::DISPERSIVE),
            (linked || m.coat_weight > 0.0, Self::COAT),
            (linked || m.fuzz_weight > 0.0, Self::FUZZ),
            (linked || m.thin_film_weight > 0.0, Self::THIN_FILM),
            (linked || m.subsurface_weight > 0.0, Self::SUBSURFACE),
            (m.flags & (f::DIELECTRIC | f::PRESENCE) != 0, Self::LAYERS),
            (
                m.flags & (f::ALPHA_TESTED | f::PRESENCE) != 0
                    || m.flags & (f::DIELECTRIC | f::THIN_WALLED) == f::DIELECTRIC | f::THIN_WALLED,
                Self::COVERAGE,
            ),
            (linked || m.emission_luminance != 0.0, Self::EMISSIVE),
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
    /// The [`SceneFeatures::BUILT`] features any record has held.
    built: u32,
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
        self.built |= features.0 & SceneFeatures::BUILT.0;
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
        SceneFeatures(out | self.built)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A feature stays while any record holds it, a rewritten or freed record gives its own
    /// back, and what a mesh was built with stays for good.
    #[test]
    fn the_table_holds_what_its_records_hold_and_keeps_what_meshes_were_built_with() {
        let mut t = MaterialFeatures::default();
        let (coat, layers) = (SceneFeatures::COAT, SceneFeatures::LAYERS);
        t.set(3, coat | SceneFeatures::WALKING);
        t.set(7, coat | layers);
        assert_eq!(t.any(), coat | SceneFeatures::WALKING | layers);
        t.set(3, SceneFeatures::NONE);
        assert_eq!(t.any(), coat | layers);
        t.set(7, SceneFeatures::DISPERSIVE);
        assert_eq!(t.any(), SceneFeatures::DISPERSIVE | layers);
        t.set(7, SceneFeatures::NONE);
        assert_eq!(t.any(), layers);
    }
}
