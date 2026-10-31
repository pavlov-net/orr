//! The global material table and its writer.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use orr_gpu::{Allocation, Device, MemoryKind, SamplerIndex, TextureIndex};
use orr_pack::material::{Material, slot};
use orr_pack::texture::TextureHeader;

use crate::memory::{RangeAllocator, UploadWindow};
use crate::upload::Record;
use crate::{Error, MaterialGpu, Result};

/// An index in the global material table.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable)]
pub struct MaterialId(pub u32);

/// `MaterialGpu::flags` bits.
pub mod material_flags {
    pub const ALPHA_TESTED: u32 = 1 << 0;
    pub const DOUBLE_SIDED: u32 = 1 << 1;
    pub const THIN_WALLED: u32 = 1 << 2;
    pub const DIELECTRIC: u32 = 1 << 3;
    pub const PRESENCE: u32 = 1 << 4;
    pub const EMISSIVE: u32 = 1 << 5;
    pub const UNLIT: u32 = 1 << 6;
    pub const SUBSURFACE: u32 = 1 << 7;
    pub const DISPLACED: u32 = 1 << 8;
    pub const COAT: u32 = 1 << 9;
    pub const FUZZ: u32 = 1 << 10;
    pub const THIN_FILM: u32 = 1 << 11;
    pub const ANISOTROPY: u32 = 1 << 12;
    /// The ray origin policy, 0-2 (default, surface, triangle), in bits 13-14.
    pub const ORIGIN_SHIFT: u32 = 13;
    pub const ORIGIN_MASK: u32 = 3 << ORIGIN_SHIFT;
    /// The cost class, 0-3, in bits 16-17.
    pub const COST_SHIFT: u32 = 16;
    pub const COST_MASK: u32 = 3 << COST_SHIFT;
}

/// A texture bound to a cooked material: its heap index and its
/// [`TextureHeader::primaries`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundTexture {
    pub index: TextureIndex,
    pub primaries: u8,
}

impl BoundTexture {
    pub const NONE: Self = Self {
        index: TextureIndex::NONE,
        primaries: TextureHeader::REC709,
    };
}

/// Linear Rec.709 to linear Rec.2020, row-major.
const REC709_TO_REC2020: [[f32; 3]; 3] = [
    [0.627_403_9, 0.329_283_04, 0.043_313_06],
    [0.069_097_29, 0.919_540_4, 0.011_362_3],
    [0.016_391_44, 0.088_013_3, 0.895_595_3],
];

fn rec2020(c: [f32; 3]) -> [f32; 3] {
    REC709_TO_REC2020.map(|r| r[0] * c[0] + r[1] * c[1] + r[2] * c[2])
}

impl MaterialGpu {
    /// The record of a cooked material whose slots read `textures`, sampled with `sampler`.
    /// Colours convert from Rec.709 to Rec.2020; flags and the cost class follow the
    /// parameters.
    pub fn new(m: &Material, textures: &[BoundTexture; 12], sampler: SamplerIndex) -> Self {
        use material_flags as f;
        let bound = |s: usize| textures[s].index != TextureIndex::NONE;
        let mut flags = 0;
        let mut set = |on: bool, flag: u32| {
            if on {
                flags |= flag;
            }
        };
        let thin = m.thin_walled != 0;
        let dielectric = m.transmission_weight > 0.0;
        let presence = m.alpha_mode == Material::BLEND && m.geometry_opacity < 1.0;
        let subsurface = m.subsurface_weight > 0.0 && !thin && !dielectric;
        let lobes = m.coat_weight > 0.0
            || m.fuzz_weight > 0.0
            || m.thin_film_thickness > 0.0
            || m.specular_anisotropy > 0.0;
        set(m.alpha_mode == Material::MASK, f::ALPHA_TESTED);
        set(m.double_sided != 0, f::DOUBLE_SIDED);
        set(thin, f::THIN_WALLED);
        set(dielectric, f::DIELECTRIC);
        set(presence, f::PRESENCE);
        set(m.emission_luminance > 0.0, f::EMISSIVE);
        set(subsurface, f::SUBSURFACE);
        set(bound(slot::DISPLACEMENT), f::DISPLACED);
        set(m.coat_weight > 0.0, f::COAT);
        set(m.fuzz_weight > 0.0, f::FUZZ);
        set(m.thin_film_thickness > 0.0, f::THIN_FILM);
        set(m.specular_anisotropy > 0.0, f::ANISOTROPY);
        let cost: u32 = if subsurface {
            3
        } else if dielectric || presence {
            2
        } else {
            u32::from(lobes)
        };
        flags |= cost << f::COST_SHIFT;
        let (mut uv_sets, mut channels, mut primaries) = (0u32, 0u64, 0u32);
        for (i, b) in m.textures.iter().enumerate() {
            uv_sets |= u32::from(b.uv_set & 1) << i;
            channels |= u64::from(b.channels & 15) << (4 * i);
            primaries |= u32::from(textures[i].primaries & 3) << (2 * i);
        }
        Self {
            base_color: rec2020(m.base_color),
            base_weight: m.base_weight,
            specular_color: rec2020(m.specular_color),
            specular_weight: m.specular_weight,
            emission_color: rec2020(m.emission_color),
            emission_luminance: m.emission_luminance,
            transmission_color: rec2020(m.transmission_color),
            transmission_depth: m.transmission_depth,
            transmission_scatter: rec2020(m.transmission_scatter),
            transmission_scatter_anisotropy: m.transmission_scatter_anisotropy,
            subsurface_color: rec2020(m.subsurface_color),
            subsurface_weight: m.subsurface_weight,
            subsurface_radius_scale: m.subsurface_radius_scale,
            subsurface_radius: m.subsurface_radius,
            coat_color: rec2020(m.coat_color),
            coat_weight: m.coat_weight,
            fuzz_color: rec2020(m.fuzz_color),
            fuzz_weight: m.fuzz_weight,
            specular_roughness: m.specular_roughness,
            specular_ior: m.specular_ior,
            specular_anisotropy: m.specular_anisotropy,
            specular_rotation: m.specular_rotation,
            metalness: m.metalness,
            diffuse_roughness: m.diffuse_roughness,
            transmission_weight: m.transmission_weight,
            subsurface_scatter_anisotropy: m.subsurface_scatter_anisotropy,
            coat_roughness: m.coat_roughness,
            coat_ior: m.coat_ior,
            fuzz_roughness: m.fuzz_roughness,
            thin_film_thickness: m.thin_film_thickness,
            thin_film_ior: m.thin_film_ior,
            geometry_opacity: m.geometry_opacity,
            alpha_cutoff: m.alpha_cutoff,
            flags,
            displacement_scale: m.displacement_scale,
            displacement_offset: m.displacement_offset,
            displacement_min: m.displacement_min,
            displacement_max: m.displacement_max,
            textures: textures.map(|t| t.index),
            sampler,
            uv_sets,
            channels_lo: channels as u32,
            channels_hi: (channels >> 32) as u32,
            primaries,
            surface: u32::from(m.surface),
            transmission_dispersion_scale: m.transmission_dispersion_scale,
            transmission_dispersion_abbe_number: m.transmission_dispersion_abbe_number,
            data: 0,
        }
    }
}

pub(crate) struct MaterialTable {
    pub table: Allocation<MaterialGpu>,
    pub capacity: u32,
    ids: Mutex<RangeAllocator>,
    /// Materials whose pixels may use the extended G-buffer target.
    extended: Mutex<HashSet<u32>>,
}

/// Whether pixels of `m` may use the extended G-buffer target. A linked surface may set any
/// lobe.
fn needs_extended(m: &MaterialGpu) -> bool {
    use material_flags as f;
    let thin_sss = m.flags & f::THIN_WALLED != 0 && m.subsurface_weight > 0.0;
    thin_sss || m.flags & (f::SUBSURFACE | f::COAT | f::FUZZ | f::ANISOTROPY) != 0 || m.surface != 0
}

impl MaterialTable {
    pub(crate) fn new(device: &Device, capacity: u32) -> Result<Self> {
        Ok(Self {
            table: device.gpu_malloc(MemoryKind::DeviceLocal, u64::from(capacity), "materials")?,
            capacity,
            ids: Mutex::new(RangeAllocator::new(capacity)),
            extended: Mutex::new(HashSet::new()),
        })
    }

    /// Records what material `id` now holds.
    pub(crate) fn note(&self, id: MaterialId, m: &MaterialGpu) {
        let mut set = self.extended.lock().unwrap();
        if needs_extended(m) {
            set.insert(id.0);
        } else {
            set.remove(&id.0);
        }
    }

    /// Whether any material may write the extended G-buffer target.
    pub(crate) fn any_extended(&self) -> bool {
        !self.extended.lock().unwrap().is_empty()
    }

    pub(crate) fn alloc(&self) -> Result<MaterialId> {
        let id = self.ids.lock().unwrap().alloc(1);
        id.map(MaterialId).ok_or(Error::Capacity {
            what: "materials",
            capacity: self.capacity,
        })
    }

    /// Returns `id` for reuse; its record stays until a new material overwrites it.
    pub(crate) fn free(&self, id: MaterialId) {
        self.extended.lock().unwrap().remove(&id.0);
        self.ids.lock().unwrap().free(id.0, 1);
    }
}

/// Writes material records into the frame's upload window. Send: one per thread.
pub struct MaterialWriter<'f> {
    pub(crate) device: &'f Device,
    pub(crate) table: &'f MaterialTable,
    pub(crate) window: &'f UploadWindow,
    pub(crate) seq: &'f AtomicU64,
    pub(crate) records: &'f Mutex<Vec<Record>>,
}

impl MaterialWriter<'_> {
    /// A new material holding `m`.
    pub fn create(&mut self, m: &MaterialGpu) -> Result<MaterialId> {
        let id = self.table.alloc()?;
        self.update(id, m)?;
        Ok(id)
    }

    /// Replaces material `id` from this frame on. Its triangles stay as opaque or
    /// micromapped in traversal as their mount cooked them: coverage that changes at run time
    /// goes through a linked surface's `coverage`.
    pub fn update(&mut self, id: MaterialId, m: &MaterialGpu) -> Result<()> {
        assert!(
            id.0 < self.table.capacity,
            "{id:?} is outside the material table"
        );
        let src = self.window.put(self.device, m)?;
        self.table.note(id, m);
        let dst = self.table.table.gpu.offset(u64::from(id.0));
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.records
            .lock()
            .unwrap()
            .push(Record::values(seq, dst, src, 1));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooked_materials_take_their_flags_cost_class_and_slot_bits() {
        let record = |m: &Material| MaterialGpu::new(m, &[BoundTexture::NONE; 12], SamplerIndex(7));
        let cost = |m: &Material| (record(m).flags & material_flags::COST_MASK) >> 16;
        let base = Material::default();
        assert_eq!(cost(&base), 0);
        let coat = Material {
            coat_weight: 0.5,
            ..base
        };
        assert_eq!(cost(&coat), 1);
        assert_ne!(record(&coat).flags & material_flags::COAT, 0);
        let glass = Material {
            transmission_weight: 1.0,
            subsurface_weight: 1.0,
            ..coat
        };
        assert_eq!(cost(&glass), 2, "transmission wins over subsurface");
        let skin = Material {
            subsurface_weight: 1.0,
            ..base
        };
        assert_eq!(cost(&skin), 3);

        let mut m = base;
        m.textures[slot::DISPLACEMENT].channels = 3;
        m.textures[slot::NORMAL].uv_set = 1;
        let mut textures = [BoundTexture::NONE; 12];
        textures[slot::DISPLACEMENT] = BoundTexture {
            index: TextureIndex(4),
            primaries: TextureHeader::REC2020,
        };
        let g = MaterialGpu::new(&m, &textures, SamplerIndex(7));
        assert_ne!(g.flags & material_flags::DISPLACED, 0);
        assert_eq!(g.channels_hi, 3 << 4, "slot 9 reads from channel 3");
        assert_eq!(g.uv_sets, 1 << slot::NORMAL);
        assert_eq!(g.primaries, 2 << (2 * slot::DISPLACEMENT));
        assert_eq!(g.textures[slot::DISPLACEMENT], TextureIndex(4));
        let white = record(&Material {
            base_color: [1.0; 3],
            ..base
        });
        assert!(white.base_color.iter().all(|c| (c - 1.0).abs() < 1e-6));
    }
}
