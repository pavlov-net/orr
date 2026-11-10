//! The global material table and its writer.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use orr_color::{Primaries, rec709_to_rec2020};
use orr_gpu::{
    AddressMode, Allocation, Device, Filter, GpuPtr, MemoryKind, SamplerDesc, SamplerIndex,
    TextureIndex,
};
use orr_pack::material::{Material, TextureBinding, filter, slot, wrap};
use orr_pack::mesh::SurfaceClass;

use crate::memory::{RangeAllocator, UploadWindow};
use crate::upload::Record;
use crate::{Error, MaterialGpu, Result, TextureSamplerGpu, TextureStreamGpu};

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
    pub const COAT_ON_BASE: u32 = 1 << 18;
    pub const FRESNEL_MIX: u32 = 1 << 19;
    pub const METAL_FLAT: u32 = 1 << 20;
    pub const F0_STATED: u32 = 1 << 21;
    pub const MORE_TEXTURES: u32 = 1 << 22;
    pub const BASE_TINT: u32 = 1 << 23;
    pub const VERTEX: u32 = 1 << 24;
    pub const OBJECT_NORMAL: u32 = 1 << 25;
    pub const OBJECT_COAT_NORMAL: u32 = 1 << 26;
    pub const DISPLACEMENT_WORLD: u32 = 1 << 27;
    /// `MaterialGpu::back` is the material of the surface's back.
    pub const HAS_BACK: u32 = 1 << 28;
    /// Light leaves the surface as a caustic where its roughness falls below
    /// `CHAIN_ROUGHNESS`: [`Material::casts`](orr_pack::material::Material::casts).
    pub const CASTER: u32 = 1 << 29;
    /// Displacement reads `TEXTURE_DISPLACEMENT_U` and `TEXTURE_DISPLACEMENT_V` too.
    pub const DISPLACED_TANGENT: u32 = 1 << 30;
    /// `material_medium` reads textures.
    pub const MEDIUM_TEXTURED: u32 = 1 << 31;
}

/// `TEXTURE_TILED`, beside the filter bits in `TextureSamplerGpu::modes`.
const TILED: u32 = 1 << 10;
/// `TEXTURE_UV_SET`, `TEXTURE_CHANNEL_SHIFT` and `TEXTURE_PRIMARIES_SHIFT`.
const UV_SET: u32 = 1 << 11;
const CHANNEL_SHIFT: u32 = 12;
const PRIMARIES_SHIFT: u32 = 16;
const VERTEX_SHIFT: u32 = 18;
const VERTEX_COMPLEMENT_SHIFT: u32 = 21;

/// A texture bound to a cooked material: its heap index, its
/// [`TextureHeader::primaries`](orr_pack::texture::TextureHeader::primaries) and the sampler of
/// its binding, from [`binding_sampler`]. A tile set has the primaries its tiles share, any index
/// but [`TextureIndex::NONE`], and `tiles`: the table `texture_tile` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundTexture {
    pub index: TextureIndex,
    pub primaries: u8,
    pub sampler: SamplerIndex,
    pub tiles: GpuPtr<u32>,
}

impl BoundTexture {
    pub const NONE: Self = Self {
        index: TextureIndex::NONE,
        primaries: Primaries::Rec709.id(),
        sampler: SamplerIndex(0),
        tiles: GpuPtr::NULL,
    };
}

/// The hardware sampler of binding `b`. Anisotropy stays off under any filter but the default:
/// it can override a nearest filter, and on NVIDIA GPUs it takes a nearest mip filter at level
/// 0.25 to mip 1.
pub fn binding_sampler(b: &TextureBinding) -> SamplerDesc {
    let address = |w: u8| match w {
        wrap::CLAMP | wrap::CONSTANT => AddressMode::ClampToEdge,
        wrap::MIRROR => AddressMode::MirroredRepeat,
        wrap::BORDER => AddressMode::ClampToBorder,
        _ => AddressMode::Repeat,
    };
    let chosen = |bit: u8| {
        if b.filter & bit != 0 {
            Filter::Nearest
        } else {
            Filter::Linear
        }
    };
    SamplerDesc {
        mag: chosen(filter::MAG_NEAREST),
        min: chosen(filter::MIN_NEAREST),
        mip: chosen(filter::MIP_NEAREST),
        address: [address(b.wrap[0]), address(b.wrap[1]), AddressMode::Repeat],
        max_anisotropy: if b.filter == 0 { 16.0 } else { 1.0 },
        ..SamplerDesc::default()
    }
}

/// `TextureSamplerGpu::modes` of binding `b` to `t`. Stray bits of a binding stay out of the fields
/// past theirs, where they could claim a tile table `t` doesn't have.
fn binding_modes(b: &TextureBinding, t: &BoundTexture) -> u32 {
    let tiled = if t.tiles == GpuPtr::NULL { 0 } else { TILED };
    let uv_set = if b.uv_set & 1 != 0 { UV_SET } else { 0 };
    u32::from(b.wrap[0] & 7)
        | u32::from(b.wrap[1] & 7) << 3
        | u32::from(b.filter & 15) << 6
        | tiled
        | uv_set
        | u32::from(b.channels & 15) << CHANNEL_SHIFT
        | u32::from(t.primaries & 3) << PRIMARIES_SHIFT
        | u32::from(b.vertex & 7) << VERTEX_SHIFT
        | u32::from(b.vertex_complement) << VERTEX_COMPLEMENT_SHIFT
}

/// A linear Rec.709 color in linear Rec.2020.
pub(crate) fn rec2020(c: impl Into<orr_math::Vec3>) -> [f32; 3] {
    rec709_to_rec2020(c.into()).as_vec3().to_array()
}

impl MaterialGpu {
    /// The record of a cooked material whose slots read `textures`. Colors convert from Rec.709
    /// to Rec.2020, and a color slot's factor multiplies its texels in the texture's primaries;
    /// flags and the cost class follow the parameters.
    pub fn new(m: &Material, textures: &[BoundTexture; slot::COUNT]) -> Self {
        use material_flags as f;
        let bound = |s: usize| textures[s].index != TextureIndex::NONE;
        let mut flags = 0;
        let mut set = |on: bool, flag: u32| {
            if on {
                flags |= flag;
            }
        };
        let thin = m.thin_walled != 0;
        let class = m.class();
        let dielectric = class == SurfaceClass::Dielectric;
        let presence = class == SurfaceClass::Presence;
        let subsurface = m.subsurface_weight > 0.0 && !thin && !dielectric;
        let lobes = m.coat_weight > 0.0
            || m.fuzz_weight > 0.0
            || m.thin_film_weight > 0.0
            || m.specular_anisotropy > 0.0;
        set(m.alpha_mode == Material::MASK, f::ALPHA_TESTED);
        set(m.double_sided != 0, f::DOUBLE_SIDED);
        set(thin, f::THIN_WALLED);
        set(dielectric, f::DIELECTRIC);
        set(presence, f::PRESENCE);
        set(m.emission_luminance > 0.0, f::EMISSIVE);
        set(m.casts(), f::CASTER);
        set(subsurface, f::SUBSURFACE);
        set(bound(slot::DISPLACEMENT), f::DISPLACED);
        set(
            bound(slot::DISPLACEMENT_U) || bound(slot::DISPLACEMENT_V),
            f::DISPLACED_TANGENT,
        );
        // A thin wall has no medium, but its IOR still disperses.
        let tinted = bound(slot::TRANSMISSION_COLOR) && m.transmission_depth > 0.0;
        let medium = (slot::TRANSMISSION_DEPTH..=slot::TRANSMISSION_SCATTER_ANISOTROPY).any(bound);
        let dispersion = bound(slot::DISPERSION_SCALE) || bound(slot::DISPERSION_ABBE);
        set(
            dispersion || (!thin && (medium || tinted)),
            f::MEDIUM_TEXTURED,
        );
        set(m.coat_weight > 0.0, f::COAT);
        set(m.fuzz_weight > 0.0, f::FUZZ);
        set(m.thin_film_weight > 0.0, f::THIN_FILM);
        set(m.specular_anisotropy > 0.0, f::ANISOTROPY);
        let preview = matches!(
            m.model,
            Material::MODEL_PREVIEW | Material::MODEL_PREVIEW_SPECULAR
        );
        set(preview, f::COAT_ON_BASE | f::METAL_FLAT);
        set(m.model != Material::MODEL_OPENPBR, f::FRESNEL_MIX);
        set(m.model == Material::MODEL_GLTF, f::BASE_TINT);
        set(m.textures.iter().any(|b| b.vertex != 0), f::VERTEX);
        set(m.displacement_world != 0, f::DISPLACEMENT_WORLD);
        set(
            m.object_normals & Material::OBJECT_NORMAL != 0,
            f::OBJECT_NORMAL,
        );
        set(
            m.object_normals & Material::OBJECT_COAT_NORMAL != 0,
            f::OBJECT_COAT_NORMAL,
        );
        set(m.model == Material::MODEL_PREVIEW_SPECULAR, f::F0_STATED);
        set(
            (slot::MORE..slot::DISPLACEMENT_U).any(bound),
            f::MORE_TEXTURES,
        );
        let cost: u32 = if subsurface {
            3
        } else if dielectric || presence {
            2
        } else {
            u32::from(lobes)
        };
        flags |= cost << f::COST_SHIFT;
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
            coat_darkening: m.coat_darkening,
            coat_anisotropy: m.coat_anisotropy,
            coat_rotation: m.coat_rotation,
            fuzz_roughness: m.fuzz_roughness,
            thin_film_weight: m.thin_film_weight,
            thin_film_thickness: m.thin_film_thickness,
            thin_film_ior: m.thin_film_ior,
            geometry_opacity: m.geometry_opacity,
            alpha_cutoff: m.alpha_cutoff,
            flags,
            displacement_scale: m.displacement_scale,
            displacement_offset: m.displacement_offset,
            displacement_min: m.displacement_min,
            displacement_max: m.displacement_max,
            object_frame: m.object_frame.concat().try_into().expect("nine entries"),
            displacement_tangent: [
                m.displacement_tangent[0][0],
                m.displacement_tangent[0][1],
                m.displacement_tangent[1][0],
                m.displacement_tangent[1][1],
            ],
            textures: textures.map(|t| t.index),
            samplers: std::array::from_fn(|i| TextureSamplerGpu {
                sampler: textures[i].sampler,
                modes: binding_modes(&m.textures[i], &textures[i]),
            }),
            surface: u32::from(m.surface),
            tangent_set: u32::from(m.tangent_set()),
            coat_tangent_set: u32::from(m.coat_tangent_set()),
            transmission_dispersion_scale: m.transmission_dispersion_scale,
            transmission_dispersion_abbe_number: m.transmission_dispersion_abbe_number,
            back: 0,
            data: 0,
            texture_streams: GpuPtr::NULL,
            tiles: textures.map(|t| t.tiles),
            uv_transforms: std::array::from_fn(|i| m.uv_transforms[i / 6][i % 6]),
            outside: std::array::from_fn(|i| m.textures[i / 4].outside[i % 4]),
        }
    }
}

pub(crate) struct MaterialTable {
    pub(crate) table: Allocation<MaterialGpu>,
    capacity: u32,
    /// `TextureStreamGpu` addresses by heap index, 0 for textures that do not stream; every
    /// record written points `MaterialGpu::texture_streams` at it.
    pub(crate) streams: Allocation<GpuPtr<TextureStreamGpu>>,
    ids: Mutex<RangeAllocator>,
    /// Materials whose pixels may use the extended G-buffer targets.
    extended: Mutex<HashSet<u32>>,
    /// Materials whose paths may take a random walk.
    walking: Mutex<HashSet<u32>>,
    /// Materials whose surfaces bound an interior.
    interiors: Mutex<HashSet<u32>>,
    /// Those of them whose interior scatters.
    scattering_interiors: Mutex<HashSet<u32>>,
    /// Materials whose transmission disperses.
    dispersive: Mutex<HashSet<u32>>,
    /// Materials whose displacement is a world length.
    world_lengths: Mutex<HashSet<u32>>,
}

/// Whether pixels of `m` may use the extended G-buffer targets. A linked surface may set any
/// lobe.
fn needs_extended(m: &MaterialGpu) -> bool {
    use material_flags as f;
    let thin_sss = m.flags & f::THIN_WALLED != 0 && m.subsurface_weight > 0.0;
    let lobes = f::SUBSURFACE | f::COAT | f::FUZZ | f::THIN_FILM | f::ANISOTROPY;
    thin_sss || m.flags & lobes != 0 || m.surface != 0
}

/// Whether paths through `m` may take a random walk: through its subsurface medium or its dense
/// scattering interior. A linked surface may set any lobe.
fn may_walk(m: &MaterialGpu) -> bool {
    let thin = m.flags & material_flags::THIN_WALLED != 0;
    let interior = m.transmission_weight > 0.0
        && m.transmission_depth > 0.0
        && m.transmission_scatter.iter().any(|&c| c > 0.0);
    m.surface != 0 || (!thin && (m.subsurface_weight > 0.0 || interior))
}

/// Whether a surface of `m` bounds an interior (`crosses_interior`): a dielectric that is not
/// thin-walled.
fn bounds_interior(m: &MaterialGpu) -> bool {
    use material_flags as f;
    m.flags & (f::DIELECTRIC | f::THIN_WALLED) == f::DIELECTRIC
}

/// Whether the interior `m` bounds scatters by the material's own factors, as a list of
/// interiors that holds no surface point reads it (`interior_medium`).
fn interior_scatters(m: &MaterialGpu) -> bool {
    bounds_interior(m)
        && m.transmission_depth > 0.0
        && m.transmission_scatter.iter().any(|&c| c > 0.0)
}

impl MaterialTable {
    pub(crate) fn new(device: &Device, capacity: u32) -> Result<Self> {
        Ok(Self {
            table: device.gpu_malloc(MemoryKind::DeviceLocal, u64::from(capacity), "materials")?,
            capacity,
            streams: device.gpu_malloc(
                MemoryKind::DeviceLocal,
                u64::from(device.image_slots()),
                "texture streams",
            )?,
            ids: Mutex::new(RangeAllocator::new(capacity)),
            extended: Mutex::new(HashSet::new()),
            walking: Mutex::new(HashSet::new()),
            interiors: Mutex::new(HashSet::new()),
            scattering_interiors: Mutex::new(HashSet::new()),
            dispersive: Mutex::new(HashSet::new()),
            world_lengths: Mutex::new(HashSet::new()),
        })
    }

    /// `m` as the table stores it: pointing at the texture stream table.
    pub(crate) fn record(&self, m: &MaterialGpu) -> MaterialGpu {
        MaterialGpu {
            texture_streams: self.streams.gpu(),
            ..*m
        }
    }

    /// Records what material `id` now holds.
    pub(crate) fn note(&self, id: MaterialId, m: &MaterialGpu) {
        let mark = |set: &Mutex<HashSet<u32>>, on: bool| {
            let mut set = set.lock().unwrap();
            if on {
                set.insert(id.0);
            } else {
                set.remove(&id.0);
            }
        };
        mark(&self.extended, needs_extended(m));
        mark(&self.walking, may_walk(m));
        mark(&self.interiors, bounds_interior(m));
        mark(&self.scattering_interiors, interior_scatters(m));
        mark(
            &self.dispersive,
            m.transmission_dispersion_scale > 0.0 && m.transmission_dispersion_abbe_number > 0.0,
        );
        mark(
            &self.world_lengths,
            m.flags & material_flags::DISPLACEMENT_WORLD != 0,
        );
    }

    /// Whether any material may write the extended G-buffer targets.
    pub(crate) fn any_extended(&self) -> bool {
        !self.extended.lock().unwrap().is_empty()
    }

    /// Whether paths through any material may take a random walk.
    pub(crate) fn any_walking(&self) -> bool {
        !self.walking.lock().unwrap().is_empty()
    }

    /// Whether a surface of any material bounds an interior.
    pub(crate) fn any_interior(&self) -> bool {
        !self.interiors.lock().unwrap().is_empty()
    }

    /// Whether the interior any material bounds scatters.
    pub(crate) fn any_scattering_interior(&self) -> bool {
        !self.scattering_interiors.lock().unwrap().is_empty()
    }

    /// Whether any material's transmission disperses.
    pub(crate) fn any_dispersive(&self) -> bool {
        !self.dispersive.lock().unwrap().is_empty()
    }

    /// Whether any of `ids` displaces by a world length.
    pub(crate) fn any_world_length(&self, ids: &[MaterialId]) -> bool {
        let set = self.world_lengths.lock().unwrap();
        ids.iter().any(|id| set.contains(&id.0))
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
        self.walking.lock().unwrap().remove(&id.0);
        self.interiors.lock().unwrap().remove(&id.0);
        self.scattering_interiors.lock().unwrap().remove(&id.0);
        self.dispersive.lock().unwrap().remove(&id.0);
        self.world_lengths.lock().unwrap().remove(&id.0);
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
    /// goes through a linked surface's `coverage`. A subdivision copy keeps last frame's positions
    /// by whether its materials displaced by a world length when it was set.
    pub fn update(&mut self, id: MaterialId, m: &MaterialGpu) -> Result<()> {
        assert!(
            id.0 < self.table.capacity,
            "{id:?} is outside the material table"
        );
        let src = self.window.put(self.device, &self.table.record(m))?;
        self.table.note(id, m);
        let dst = self.table.table.gpu().offset(u64::from(id.0));
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
        let record = |m: &Material| MaterialGpu::new(m, &[BoundTexture::NONE; slot::COUNT]);
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
        let mut veil = Material {
            alpha_mode: Material::BLEND,
            ..base
        };
        veil.textures[slot::OPACITY].vertex = 1;
        assert_eq!(veil.class(), SurfaceClass::Presence);
        assert_ne!(record(&veil).flags & material_flags::PRESENCE, 0);

        let mut m = base;
        m.textures[slot::DISPLACEMENT].channels = 3;
        m.textures[slot::NORMAL].uv_set = 1;
        let mut textures = [BoundTexture::NONE; slot::COUNT];
        textures[slot::DISPLACEMENT] = BoundTexture {
            index: TextureIndex(4),
            primaries: Primaries::Rec2020.id(),
            sampler: SamplerIndex(7),
            ..BoundTexture::NONE
        };
        let g = MaterialGpu::new(&m, &textures);
        assert_ne!(g.flags & material_flags::DISPLACED, 0);
        let modes = g.samplers[slot::DISPLACEMENT].modes;
        assert_eq!(modes >> CHANNEL_SHIFT & 15, 3);
        assert_eq!(
            modes >> PRIMARIES_SHIFT & 3,
            u32::from(Primaries::Rec2020.id())
        );
        assert_ne!(g.samplers[slot::NORMAL].modes & UV_SET, 0);
        assert_eq!(modes & UV_SET, 0);
        assert_eq!(g.textures[slot::DISPLACEMENT], TextureIndex(4));
        m.textures[slot::NORMAL].wrap = [0xff; 2];
        m.textures[slot::NORMAL].filter = 0xff;
        let stray = MaterialGpu::new(&m, &textures);
        assert_eq!(stray.samplers[slot::NORMAL].modes & TILED, 0);
        let white = record(&Material {
            base_color: [1.0; 3],
            ..base
        });
        assert!(white.base_color.iter().all(|c| (c - 1.0).abs() < 1e-6));
    }
}
