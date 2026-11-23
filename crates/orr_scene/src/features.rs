//! What a scene may hold that its paths' kernels are compiled for.

use std::collections::HashMap;
use std::ops::{BitOr, BitOrAssign};

use crate::MaterialGpu;
use crate::material::{TILED, material_flags as f};

/// What a scene may hold in a frame. A bit is set wherever the scene cannot rule its feature
/// out: from the records in the material table, which every scene shares, and from what the
/// scene itself holds. A kernel compiled without a feature runs only while its bit is clear.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SceneFeatures(pub u64);

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
    /// The scene holds a sphere, point or rect light.
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
    /// dielectric, a producer entry is not opaque, or a model has a chain, whose copies fade
    /// between its steps with every triangle a candidate.
    pub const COVERAGE: Self = Self(1 << 13);
    /// The scene holds a particle or smoke entry.
    pub const PARTICLES: Self = Self(1 << 14);
    /// A material emits.
    pub const EMISSIVE: Self = Self(1 << 15);
    /// A material scales a slot by its meshes' vertex color stream.
    pub const VERTEX_COLORS: Self = Self(1 << 16);
    /// A material's texture slot reads a tile set.
    pub const TILED_TEXTURES: Self = Self(1 << 17);
    /// The scene lists a decal.
    pub const DECALS: Self = Self(1 << 18);
    /// A texture has streamed: one may read no finer than its resident tiles and report the
    /// level it wants.
    pub const STREAMED_TEXTURES: Self = Self(1 << 25);
    /// A neural texture set has stood in for a texture: the material resolve may decode one.
    pub const NEURAL_TEXTURES: Self = Self(1 << 30);
    /// The scene holds a producer entry, and one of strands.
    pub const PRODUCERS: Self = Self(1 << 19);
    pub const STRANDS: Self = Self(1 << 20);
    /// The scene holds a skinned copy or a group hung from a palette's joint.
    pub const SKINNING: Self = Self(1 << 21);
    /// A material emits while one has a coat or fuzz: an emitter's light sample may pass the
    /// layers of the side it shows.
    pub const COATED_EMITTERS: Self = Self(1 << 22);
    /// Geometry of the dielectric class may show a material that bounds no interior: a
    /// thin-walled dielectric, a record rewritten under a mesh built as glass, a consumer's
    /// material set over a section cooked as one, a page's set that puts such a material on
    /// one, or a dielectric producer entry's other material.
    pub const THIN_DIELECTRICS: Self = Self(1 << 23);
    /// The scene may hold a caustic caster: a copy of a model with a caster section, or a
    /// producer entry of clusters that rays trace with a material that casts.
    pub const CASTERS: Self = Self(1 << 26);
    /// A photon may meet a caster at an opaque vertex, where its chain goes on: a copy of a
    /// model with a caster section of the opaque class, or a producer entry that rays trace as
    /// geometry other than a layer while one of its materials casts. Without one a photon
    /// meets at most one opaque vertex, the surface it deposits on.
    pub const OPAQUE_CASTERS: Self = Self(1 << 29);
    /// The scene's environment may radiate: the scene's kernels link one that is not the
    /// default map, or its map is set.
    pub const ENVIRONMENT: Self = Self(1 << 27);
    /// A record has the caster flag, which on a producer entry that rays trace makes
    /// [`Self::CASTERS`].
    const CASTING: Self = Self(1 << 28);
    /// A material is of a linked surface type, which shades its hits.
    pub const LINKED_SURFACES: Self = Self(1 << 33);
    /// A material the standard surface shades emits by its own emission factor.
    pub const EMISSION_FACTORS: Self = Self(1 << 34);
    /// As [`Self::EMISSION_FACTORS`], of a material of a linked type that may leave a hit to
    /// the standard surface: one that does not declare that it shades every hit itself.
    pub const LINKED_EMISSION_FACTORS: Self = Self(1 << 35);
    /// A material the standard surface shades has a coat, fuzz, thin film or subsurface by its
    /// own weights.
    pub const LOBE_FACTORS: Self = Self(1 << 36);
    /// As [`Self::LOBE_FACTORS`], of a material of a linked type that may leave a hit to the
    /// standard surface.
    pub const LINKED_LOBE_FACTORS: Self = Self(1 << 37);
    /// A material of a linked type that may leave a hit to the standard surface transmits by
    /// its own weight.
    pub const LINKED_TRANSMISSION_FACTORS: Self = Self(1 << 38);
    /// The environment's exact radiance is a draw ([`crate::EnvironmentData::DRAWN`]), which
    /// the accumulation reference takes by its paths' rays alone.
    pub const DRAWN_ENVIRONMENT: Self = Self(1 << 39);
    /// The scene holds a light surfaces cast no shadow from.
    pub const UNSHADOWED_LIGHTS: Self = Self(1 << 40);
    /// The scene holds a light whose emission a cone or focus shapes.
    pub const SHAPED_LIGHTS: Self = Self(1 << 41);
    /// What a mesh keeps from the material it was built with, whatever the record holds now:
    /// once a record has shown one of these, the table keeps it.
    const BUILT: Self = Self(Self::LAYERS.0 | Self::COVERAGE.0);

    /// The id of `scene_may.slang`'s `path_may2`.
    pub const SURFACE_MAY_ID: u32 = 111;

    /// The value of `scene_may.slang`'s `path_may2` for a scene that may hold these: each of
    /// its bits clear only where the feature is ruled out, and every bit it does not define
    /// set.
    pub fn surface_may(self) -> u32 {
        let mut ruled_out = 0;
        for (feature, bit) in [
            (Self::VERTEX_COLORS, 1 << 0),
            (Self::TILED_TEXTURES, 1 << 1),
            (Self::STREAMED_TEXTURES, 1 << 2),
            (Self::DECALS, 1 << 3),
            (Self::PRODUCERS, 1 << 5),
            (Self::STRANDS, 1 << 6),
            (Self::SKINNING, 1 << 7),
            (Self::COATED_EMITTERS, 1 << 8),
            (Self::THIN_DIELECTRICS, 1 << 9),
            (Self::NEURAL_TEXTURES, 1 << 10),
            (Self::LINKED_SURFACES, 1 << 11),
            (Self::EMISSION_FACTORS, 1 << 12),
            (Self::LINKED_EMISSION_FACTORS, 1 << 13),
            (Self::LOBE_FACTORS, 1 << 14),
            (Self::LINKED_LOBE_FACTORS, 1 << 15),
            (Self::LINKED_TRANSMISSION_FACTORS, 1 << 16),
            (Self::UNSHADOWED_LIGHTS, 1 << 17),
            (Self::SHAPED_LIGHTS, 1 << 18),
        ] {
            if !self.contains(feature) {
                ruled_out |= bit;
            }
        }
        // Manifold walks take a directional, sphere, point or rect light through the surfaces of
        // the interiors' solid glass.
        let lights = self.contains(Self::DIRECTIONAL_LIGHTS) || self.contains(Self::LOCAL_LIGHTS);
        if !(self.contains(Self::INTERIORS) && lights) {
            ruled_out |= 1 << 19;
        }
        !ruled_out
    }

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// What record `m`, as the material table stores it, may show: the lobes of its record and
    /// of its linked surface type (`MaterialGpu::lobes`), and what its flags and its own medium
    /// decide, with the linked types' `declares` by id from 1.
    pub(crate) fn of_material(m: &MaterialGpu, declares: &[u32]) -> Self {
        use crate::material::material_lobes as l;
        let lobe = |lobe: u32| m.lobes & lobe != 0;
        let thin = m.flags & f::THIN_WALLED != 0;
        let scatters =
            m.transmission_depth > 0.0 && m.transmission_scatter.iter().any(|&c| c > 0.0);
        let bounds = m.flags & (f::DIELECTRIC | f::THIN_WALLED) == f::DIELECTRIC;
        let walks = !thin && (lobe(l::SUBSURFACE) || (lobe(l::TRANSMISSION) && scatters));
        let disperses =
            m.transmission_dispersion_scale > 0.0 && m.transmission_dispersion_abbe_number > 0.0;
        // What the record's own weights show, and whether a linked type shades it.
        let own = crate::material::shown_lobes(m, declares);
        let linked = crate::material::linked_type(declares, m.surface);
        let factors = own & l::RICH != 0;
        let emits = own & l::EMISSION != 0;
        let mut out = Self::NONE;
        for (on, bit) in [
            (walks, Self::WALKING),
            (bounds, Self::INTERIORS),
            (bounds && scatters, Self::SCATTERING_INTERIORS),
            (disperses, Self::DISPERSIVE),
            (lobe(l::COAT), Self::COAT),
            (lobe(l::FUZZ), Self::FUZZ),
            (lobe(l::THIN_FILM), Self::THIN_FILM),
            (lobe(l::SUBSURFACE), Self::SUBSURFACE),
            (m.flags & (f::DIELECTRIC | f::PRESENCE) != 0, Self::LAYERS),
            (
                m.flags & (f::ALPHA_TESTED | f::PRESENCE) != 0
                    || m.flags & (f::DIELECTRIC | f::THIN_WALLED) == f::DIELECTRIC | f::THIN_WALLED,
                Self::COVERAGE,
            ),
            (lobe(l::EMISSION), Self::EMISSIVE),
            (m.flags & f::VERTEX != 0, Self::VERTEX_COLORS),
            (
                m.samplers.iter().any(|s| s.modes & TILED != 0),
                Self::TILED_TEXTURES,
            ),
            (m.flags & f::CASTER != 0, Self::CASTING),
            (
                m.flags & (f::DIELECTRIC | f::THIN_WALLED) == f::DIELECTRIC | f::THIN_WALLED,
                Self::THIN_DIELECTRICS,
            ),
            (m.surface != 0, Self::LINKED_SURFACES),
            (emits && !linked, Self::EMISSION_FACTORS),
            (emits && linked, Self::LINKED_EMISSION_FACTORS),
            (factors && !linked, Self::LOBE_FACTORS),
            (factors && linked, Self::LINKED_LOBE_FACTORS),
            (
                own & l::TRANSMISSION != 0 && linked,
                Self::LINKED_TRANSMISSION_FACTORS,
            ),
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
pub(crate) struct MaterialFeatures {
    held: HashMap<u32, u64>,
    counts: [u32; 64],
    /// The [`SceneFeatures::BUILT`] features any record has held, and
    /// [`SceneFeatures::THIN_DIELECTRICS`] once a record that was a dielectric has been
    /// rewritten as one that is none.
    built: u64,
}

impl Default for MaterialFeatures {
    fn default() -> Self {
        Self {
            held: HashMap::new(),
            counts: [0; 64],
            built: 0,
        }
    }
}

impl MaterialFeatures {
    /// Records that material `id` now holds `features`.
    pub(crate) fn set(&mut self, id: u32, features: SceneFeatures) {
        let was = self.replace(id, features);
        self.built |= features.0 & SceneFeatures::BUILT.0;
        // A mesh built as glass keeps the dielectric class under the new record.
        let glass = SceneFeatures::INTERIORS.0 | SceneFeatures::THIN_DIELECTRICS.0;
        if was & glass != 0 && features.0 & glass == 0 {
            self.built |= SceneFeatures::THIN_DIELECTRICS.0;
        }
    }

    /// Records that material `id` is freed: its record stays as it is until a new material
    /// overwrites it.
    pub(crate) fn free(&mut self, id: u32) {
        self.replace(id, SceneFeatures::NONE);
    }

    /// Counts `features` for `id` in place of what it held, which it returns.
    fn replace(&mut self, id: u32, features: SceneFeatures) -> u64 {
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
        was
    }

    /// Whether material `id`'s record bounds an interior: a dielectric that is not thin-walled.
    pub(crate) fn bounds_interior(&self, id: u32) -> bool {
        self.held
            .get(&id)
            .is_some_and(|&f| f & SceneFeatures::INTERIORS.0 != 0)
    }

    /// The features any record holds.
    pub(crate) fn any(&self) -> SceneFeatures {
        let mut out = 0;
        for (bit, &count) in self.counts.iter().enumerate() {
            if count > 0 {
                out |= 1 << bit;
            }
        }
        let layered = SceneFeatures::COAT.0 | SceneFeatures::FUZZ.0;
        if out & layered != 0 && out & SceneFeatures::EMISSIVE.0 != 0 {
            out |= SceneFeatures::COATED_EMITTERS.0;
        }
        SceneFeatures((out & !SceneFeatures::CASTING.0) | self.built)
    }

    /// Whether any record has the caster flag.
    pub(crate) fn casts(&self) -> bool {
        self.counts[SceneFeatures::CASTING.0.trailing_zeros() as usize] > 0
    }

    /// Whether material `id`'s record has the caster flag.
    pub(crate) fn casting(&self, id: u32) -> bool {
        self.held
            .get(&id)
            .is_some_and(|&f| f & SceneFeatures::CASTING.0 != 0)
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

    fn record(edit: impl FnOnce(&mut MaterialGpu)) -> SceneFeatures {
        let mut m: MaterialGpu = bytemuck::Zeroable::zeroed();
        edit(&mut m);
        m.lobes = crate::material::record_lobes(&m, &[0]);
        SceneFeatures::of_material(&m, &[0])
    }

    /// Each surface feature a record decides is set by the flag or sampler mode the shaders
    /// test for it, and by nothing a plain record holds.
    #[test]
    fn a_record_shows_the_surface_features_its_flags_and_samplers_name() {
        assert_eq!(record(|_| {}), SceneFeatures::NONE);
        let vertex = record(|m| m.flags = f::VERTEX);
        assert_eq!(vertex, SceneFeatures::VERTEX_COLORS);
        let tiled = record(|m| m.samplers[3].modes = TILED);
        assert_eq!(tiled, SceneFeatures::TILED_TEXTURES);
        let thin = record(|m| m.flags = f::DIELECTRIC | f::THIN_WALLED);
        assert!(thin.contains(SceneFeatures::THIN_DIELECTRICS));
        assert!(!thin.contains(SceneFeatures::INTERIORS));
        let solid = record(|m| m.flags = f::DIELECTRIC);
        assert!(!solid.contains(SceneFeatures::THIN_DIELECTRICS));
        // Which surface a record's weights are shaded by, and of what kind.
        let shading = |r: SceneFeatures| {
            let kinds = SceneFeatures::LINKED_SURFACES
                | SceneFeatures::EMISSION_FACTORS
                | SceneFeatures::LINKED_EMISSION_FACTORS
                | SceneFeatures::LOBE_FACTORS
                | SceneFeatures::LINKED_LOBE_FACTORS
                | SceneFeatures::LINKED_TRANSMISSION_FACTORS;
            SceneFeatures(r.0 & kinds.0)
        };
        // An emission factor is the standard surface's emitter, and on a record of a linked type
        // that may leave a hit to it, its type's.
        let emits = record(|m| m.emission_luminance = 2.0);
        assert_eq!(shading(emits), SceneFeatures::EMISSION_FACTORS);
        assert!(emits.contains(SceneFeatures::EMISSIVE));
        let linked = record(|m| {
            m.emission_luminance = 2.0;
            m.surface = 1;
        });
        assert_eq!(
            shading(linked),
            SceneFeatures::LINKED_SURFACES | SceneFeatures::LINKED_EMISSION_FACTORS
        );
        // So is a coat, fuzz, thin film or subsurface the record's weights give it.
        let coated = record(|m| m.coat_weight = 1.0);
        assert_eq!(shading(coated), SceneFeatures::LOBE_FACTORS);
        let skin = record(|m| {
            m.subsurface_weight = 1.0;
            m.surface = 1;
        });
        assert_eq!(
            shading(skin),
            SceneFeatures::LINKED_SURFACES | SceneFeatures::LINKED_LOBE_FACTORS
        );
        // An id nothing links is the standard surface's.
        let unlinked = record(|m| {
            m.subsurface_weight = 1.0;
            m.surface = 2;
        });
        assert_eq!(
            shading(unlinked),
            SceneFeatures::LINKED_SURFACES | SceneFeatures::LOBE_FACTORS
        );
        // A linked record's transmission shows where its type leaves hits to the standard
        // surface, which may be dielectrics.
        let glass = record(|m| {
            m.transmission_weight = 1.0;
            m.surface = 1;
        });
        assert_eq!(
            shading(glass),
            SceneFeatures::LINKED_SURFACES | SceneFeatures::LINKED_TRANSMISSION_FACTORS
        );
    }

    /// An emitter's light passes a coat or fuzz only while a record emits and a record has
    /// one, and geometry built as glass shows a material that bounds no interior once its
    /// record is rewritten as something else; a freed record changes nothing it showed.
    #[test]
    fn the_table_finds_coated_emitters_and_glass_that_stopped_bounding_an_interior() {
        let coated = record(|m| m.coat_weight = 1.0);
        let emits = record(|m| m.emission_luminance = 1.0);
        let fuzzy_light = record(|m| {
            m.fuzz_weight = 1.0;
            m.emission_luminance = 1.0;
        });
        let (emitters, thin) = (
            SceneFeatures::COATED_EMITTERS,
            SceneFeatures::THIN_DIELECTRICS,
        );
        let mut t = MaterialFeatures::default();
        t.set(1, coated);
        assert!(!t.any().contains(emitters));
        t.set(2, emits);
        assert!(t.any().contains(emitters));
        t.free(1);
        assert!(!t.any().contains(emitters));
        t.set(1, fuzzy_light);
        assert!(t.any().contains(emitters));
        // A linked type's own coat passes its emitters' light; a record's coat does not where
        // its type shades every hit itself.
        let emitter_of = |declares: u32, coat_weight: f32| {
            let mut m: MaterialGpu = bytemuck::Zeroable::zeroed();
            m.surface = 1;
            m.emission_luminance = 1.0;
            m.coat_weight = coat_weight;
            m.lobes = crate::material::record_lobes(&m, &[declares]);
            SceneFeatures::of_material(&m, &[declares])
        };
        use orr_shaders::surface_declares as d;
        let mut t = MaterialFeatures::default();
        t.set(
            1,
            emitter_of(d::SURFACE | d::ALWAYS_SHADES | d::EMISSION, 1.0),
        );
        assert!(!t.any().contains(emitters));
        t.set(1, emitter_of(d::COAT, 0.0));
        assert!(t.any().contains(emitters));

        let solid = record(|m| m.flags = f::DIELECTRIC);
        let mut t = MaterialFeatures::default();
        t.set(5, solid);
        t.free(5);
        t.set(5, solid);
        assert!(!t.any().contains(thin));
        t.set(5, record(|m| m.flags = f::DIELECTRIC | f::THIN_WALLED));
        assert!(t.any().contains(thin));
        t.set(5, solid);
        assert!(!t.any().contains(thin));
        t.set(5, coated);
        assert!(t.any().contains(thin));
        t.free(5);
        assert!(t.any().contains(thin));
    }
}
