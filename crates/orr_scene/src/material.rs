//! The global material table and its writer.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use orr_color::Primaries;
use orr_gpu::{
    AddressMode, Allocation, Device, Filter, GpuPtr, MemoryKind, MemoryPriority, SamplerDesc,
    SamplerIndex, TextureIndex,
};
use orr_pack::material::{Material, TextureBinding, filter, slot, wrap};
use orr_pack::mesh::SurfaceClass;

use crate::features::{MaterialFeatures, SceneFeatures};
use crate::memory::UploadWindow;
use crate::table::Table;
use crate::upload::Record;
use crate::{MaterialGpu, Result, TextureSamplerGpu, TextureStreamGpu};

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
    /// The record is a solid dielectric caster that transmits and that no texture, vertex
    /// color stream, back, displacement, linked surface, coat, fuzz, thin film or anisotropy
    /// varies: a plane stack face where its BSDF is an ideal mirror and window (`shadow.slang`'s
    /// `stack_material`), which only a surface smoother than `CHAIN_ROUGHNESS` is. The material
    /// table sets it on every record it stores, from the record's other fields.
    pub const STACK_FACE: u32 = 1 << 15;
    pub const MORE_TEXTURES: u32 = 1 << 22;
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

/// `MaterialGpu::lobes` bits (`material.slang`'s `MATERIAL_LOBE_*`): what a material's surface may show
/// beyond the plain part. The material table sets them on every record it stores, from the
/// record's weights, which the standard surface shades by, and from what the record's linked
/// surface type declares.
pub mod material_lobes {
    pub const TRANSMISSION: u32 = 1 << 0;
    pub const SUBSURFACE: u32 = 1 << 1;
    pub const COAT: u32 = 1 << 2;
    pub const FUZZ: u32 = 1 << 3;
    pub const THIN_FILM: u32 = 1 << 4;
    pub const EMISSION: u32 = 1 << 5;
    pub const ANISOTROPY: u32 = 1 << 6;
    /// The lobes a path's rich stages hold.
    pub const RICH: u32 = SUBSURFACE | COAT | FUZZ | THIN_FILM;
}

/// `TEXTURE_TILED`, beside the filter bits in `TextureSamplerGpu::modes`.
pub(crate) const TILED: u32 = 1 << 10;
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

impl MaterialGpu {
    /// The lobes a scene's material table gives the record where no surface type links
    /// (`lobes`): those its own weights give it.
    pub fn standard_lobes(&self) -> u32 {
        record_lobes(self, &[0])
    }

    /// The record of a cooked material whose slots read `textures`. A color slot's factor
    /// multiplies its texels in the texture's primaries; the flags follow the parameters.
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
        set(
            (slot::MORE..slot::DISPLACEMENT_U).any(bound),
            f::MORE_TEXTURES,
        );
        Self {
            base_color: m.base_color,
            base_weight: m.base_weight,
            specular_color: m.specular_color,
            specular_weight: m.specular_weight,
            emission_color: m.emission_color,
            emission_luminance: m.emission_luminance,
            transmission_color: m.transmission_color,
            transmission_depth: m.transmission_depth,
            transmission_scatter: m.transmission_scatter,
            transmission_scatter_anisotropy: m.transmission_scatter_anisotropy,
            subsurface_color: m.subsurface_color,
            subsurface_weight: m.subsurface_weight,
            subsurface_radius_scale: m.subsurface_radius_scale,
            subsurface_radius: m.subsurface_radius,
            coat_color: m.coat_color,
            coat_weight: m.coat_weight,
            fuzz_color: m.fuzz_color,
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
            lobes: 0,
            _pad: 0,
            data: 0,
            texture_streams: GpuPtr::NULL,
            tiles: textures.map(|t| t.tiles),
            uv_transforms: std::array::from_fn(|i| m.uv_transforms[i / 6][i % 6]),
            outside: std::array::from_fn(|i| m.textures[i / 4].outside[i % 4]),
        }
    }
}

pub(crate) struct MaterialTable {
    pub(crate) table: Table<MaterialGpu>,
    /// `TextureStreamGpu` addresses by heap index, 0 for textures that do not stream; every
    /// record written points `MaterialGpu::texture_streams` at it.
    pub(crate) streams: Allocation<GpuPtr<TextureStreamGpu>>,
    /// Materials whose pixels may use the extended G-buffer targets.
    extended: Mutex<HashSet<u32>>,
    /// What each record may show a path.
    features: Mutex<MaterialFeatures>,
    /// Materials whose displacement is a world length.
    world_lengths: Mutex<HashSet<u32>>,
    /// What each record gives the hit records of the runs that show it (`hit_key`).
    hit_keys: Mutex<HashMap<u32, [u32; 4]>>,
    /// Counts the records whose `hit_key` changed: a hit table written before then is rewritten
    /// whole.
    hit_epoch: AtomicU64,
    /// What each linked surface type declares (`KernelSet::surface_declares`), by id from 1.
    declares: &'static [u32],
    /// By material, the materials that name it as their back.
    fronts: Mutex<HashMap<u32, HashSet<u32>>>,
}

/// What `orr_scene.runs`' `hit_data` and `sbt.slang`'s pin read of a record: the flags they test,
/// its back, its linked surface id and the lobes that `HIT_LOBED` and `HIT_EMITS` name.
fn hit_key(m: &MaterialGpu) -> [u32; 4] {
    use {material_flags as f, material_lobes as l};
    let flags = f::ALPHA_TESTED | f::THIN_WALLED | f::DIELECTRIC | f::PRESENCE | f::HAS_BACK;
    [
        m.flags & flags,
        m.back,
        m.surface,
        m.lobes & (l::RICH | l::EMISSION),
    ]
}

/// Whether record `m` takes [`material_flags::STACK_FACE`].
fn stack_face(m: &MaterialGpu) -> bool {
    use material_flags as f;
    let varies = f::THIN_WALLED
        | f::VERTEX
        | f::MORE_TEXTURES
        | f::ANISOTROPY
        | f::HAS_BACK
        | f::DISPLACED
        | f::MEDIUM_TEXTURED;
    m.surface == 0
        && m.flags & varies == 0
        && m.flags & (f::DIELECTRIC | f::CASTER) == f::DIELECTRIC | f::CASTER
        && m.coat_weight == 0.0
        && m.fuzz_weight == 0.0
        && m.thin_film_weight == 0.0
        && m.textures.iter().all(|&t| t == TextureIndex::NONE)
        && (1.0 - m.metalness) * m.transmission_weight > 0.0
}

/// Whether pixels of `m` may use the extended G-buffer targets: by its record where the
/// standard surface shades it, and by the lobes `declared` of its linked surface type.
fn needs_extended(m: &MaterialGpu, declared: u32) -> bool {
    use {material_flags as f, material_lobes as l};
    let subsurface = m.subsurface_weight > 0.0 || declared & l::SUBSURFACE != 0;
    let thin_sss = m.flags & f::THIN_WALLED != 0 && subsurface;
    let lobes = f::SUBSURFACE | f::COAT | f::FUZZ | f::THIN_FILM | f::ANISOTROPY;
    thin_sss || m.flags & lobes != 0 || declared & (l::RICH | l::ANISOTROPY) != 0
}

/// What the linked type of surface id `surface` declares among `declares`, the types' by id
/// from 1; nothing for id 0 and an id nothing links, which the standard surface shades.
fn declared(declares: &[u32], surface: u32) -> u32 {
    (surface as usize)
        .checked_sub(1)
        .and_then(|i| declares.get(i))
        .copied()
        .unwrap_or(0)
}

/// The lobes that the linked type of surface id `surface` declares.
fn declared_lobes(declares: &[u32], surface: u32) -> u32 {
    use {material_lobes as l, orr_shaders::surface_declares as d};
    let declared = declared(declares, surface);
    [
        (d::TRANSMISSION, l::TRANSMISSION),
        (d::SUBSURFACE, l::SUBSURFACE),
        (d::COAT, l::COAT),
        (d::FUZZ, l::FUZZ),
        (d::THIN_FILM, l::THIN_FILM),
        (d::EMISSION, l::EMISSION),
        (d::ANISOTROPY, l::ANISOTROPY),
    ]
    .into_iter()
    .filter(|(bit, _)| declared & bit != 0)
    .fold(0, |lobes, (_, lobe)| lobes | lobe)
}

/// `MaterialGpu::lobes` of record `m`: the lobes its weights give it where the standard surface
/// shades it ([`shown_lobes`]), and those its linked type declares.
pub(crate) fn record_lobes(m: &MaterialGpu, declares: &[u32]) -> u32 {
    shown_lobes(m, declares) | declared_lobes(declares, m.surface)
}

/// Whether surface id `surface` names a linked type among `declares`, the types' by id from 1.
pub(crate) fn linked_type(declares: &[u32], surface: u32) -> bool {
    (surface as usize)
        .checked_sub(1)
        .is_some_and(|i| i < declares.len())
}

/// The lobes record `m`'s own weights give it that a hit may show: where the standard surface
/// shades the hit, which a linked type that shades every hit itself (`always_shades`) never
/// leaves it.
pub(crate) fn shown_lobes(m: &MaterialGpu, declares: &[u32]) -> u32 {
    use {material_flags as f, material_lobes as l, orr_shaders::surface_declares as d};
    if declared(declares, m.surface) & d::ALWAYS_SHADES != 0 {
        return 0;
    }
    [
        (m.transmission_weight > 0.0, l::TRANSMISSION),
        (m.subsurface_weight > 0.0, l::SUBSURFACE),
        (m.coat_weight > 0.0, l::COAT),
        (m.fuzz_weight > 0.0, l::FUZZ),
        (m.thin_film_weight > 0.0, l::THIN_FILM),
        (m.emission_luminance != 0.0, l::EMISSION),
        (m.flags & f::ANISOTROPY != 0, l::ANISOTROPY),
    ]
    .into_iter()
    .filter(|(on, _)| *on)
    .fold(0, |lobes, (_, lobe)| lobes | lobe)
}

/// Whether a material of surface id `front` may have one of surface id `back` as its back:
/// hit shaders made for one linked type shade neither side of another's, so the two are one id
/// or a linked type among them declares its other side.
fn sides(declares: &[u32], front: u32, back: u32) -> bool {
    // An id nothing links is the standard surface's, as id 0 is.
    let linked = |id: u32| if id as usize <= declares.len() { id } else { 0 };
    let (front, back) = (linked(front), linked(back));
    let either = declared(declares, front) | declared(declares, back);
    front == back || either & orr_shaders::surface_declares::OTHER_SIDE != 0
}

impl MaterialTable {
    pub(crate) fn new(device: &Device, declares: &'static [u32]) -> Result<Self> {
        Ok(Self {
            declares,
            fronts: Mutex::new(HashMap::new()),
            table: Table::new(device, MemoryPriority::Normal, "materials")?,
            streams: device.gpu_malloc(
                MemoryKind::DeviceLocal,
                u64::from(device.image_slots()),
                "texture streams",
            )?,
            extended: Mutex::new(HashSet::new()),
            features: Mutex::new(MaterialFeatures::default()),
            world_lengths: Mutex::new(HashSet::new()),
            hit_keys: Mutex::new(HashMap::new()),
            hit_epoch: AtomicU64::new(0),
        })
    }

    /// `m` as the table stores it: pointing at the texture stream table, with the stack face
    /// flag as its other fields decide and the lobes its surface may show.
    pub(crate) fn record(&self, m: &MaterialGpu) -> MaterialGpu {
        use material_flags as f;
        let stack = if stack_face(m) { f::STACK_FACE } else { 0 };
        MaterialGpu {
            texture_streams: self.streams.gpu(),
            flags: m.flags & !f::STACK_FACE | stack,
            lobes: record_lobes(m, self.declares),
            ..*m
        }
    }

    /// Fails where `m` as material `id` would share a triangle with a material of another
    /// surface id that neither linked type declares ([`sides`]): as the front of its back, or
    /// as the back of the materials that name it.
    pub(crate) fn check_sides(&self, id: MaterialId, m: &MaterialGpu) -> Result<()> {
        use material_flags as f;
        let keys = self.hit_keys.lock().unwrap();
        if m.flags & f::HAS_BACK != 0 {
            let back = if m.back == id.0 {
                Some(m.surface)
            } else {
                keys.get(&m.back).map(|k| k[2])
            };
            if back.is_some_and(|back| !sides(self.declares, m.surface, back)) {
                return Err(crate::Error::UndeclaredOtherSide {
                    front: id,
                    back: MaterialId(m.back),
                });
            }
        }
        let fronts = self.fronts.lock().unwrap();
        for &front in fronts.get(&id.0).into_iter().flatten() {
            if front != id.0 && !sides(self.declares, keys[&front][2], m.surface) {
                return Err(crate::Error::UndeclaredOtherSide {
                    front: MaterialId(front),
                    back: id,
                });
            }
        }
        Ok(())
    }

    /// Material `id` names `back` as its back in place of `was`.
    fn note_back(&self, id: u32, was: Option<u32>, back: Option<u32>) {
        if was == back {
            return;
        }
        let mut fronts = self.fronts.lock().unwrap();
        if let Some(was) = was
            && let Some(set) = fronts.get_mut(&was)
        {
            set.remove(&id);
            if set.is_empty() {
                fronts.remove(&was);
            }
        }
        if let Some(back) = back {
            fronts.entry(back).or_default().insert(id);
        }
    }

    /// Records what material `id` now holds: `m`, as [`Self::record`] gave it and
    /// [`Self::check_sides`] passed it.
    pub(crate) fn note(&self, id: MaterialId, m: &MaterialGpu) {
        let mark = |set: &Mutex<HashSet<u32>>, on: bool| {
            let mut set = set.lock().unwrap();
            if on {
                set.insert(id.0);
            } else {
                set.remove(&id.0);
            }
        };
        mark(
            &self.extended,
            needs_extended(m, declared_lobes(self.declares, m.surface)),
        );
        self.features
            .lock()
            .unwrap()
            .set(id.0, SceneFeatures::of_material(m, self.declares));
        mark(
            &self.world_lengths,
            m.flags & material_flags::DISPLACEMENT_WORLD != 0,
        );
        let key = hit_key(m);
        let was = self.hit_keys.lock().unwrap().insert(id.0, key);
        let back = |k: [u32; 4]| (k[0] & material_flags::HAS_BACK != 0).then_some(k[1]);
        self.note_back(id.0, was.and_then(back), back(key));
        if was != Some(key) {
            self.hit_epoch.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Counts the records whose hit records changed (`hit_key`).
    pub(crate) fn hit_epoch(&self) -> u64 {
        self.hit_epoch.load(Ordering::Relaxed)
    }

    /// Whether any material may write the extended G-buffer targets.
    pub(crate) fn any_extended(&self) -> bool {
        !self.extended.lock().unwrap().is_empty()
    }

    /// What any material may show a path.
    pub(crate) fn features(&self) -> SceneFeatures {
        self.features.lock().unwrap().any()
    }

    /// Whether material `id`'s record bounds an interior (`MaterialFeatures::bounds_interior`).
    pub(crate) fn bounds_interior(&self, id: MaterialId) -> bool {
        self.features.lock().unwrap().bounds_interior(id.0)
    }

    /// Whether any record has the stack face flag.
    pub(crate) fn stack_faces(&self) -> bool {
        self.features.lock().unwrap().stack_faces()
    }

    /// Whether any of `ids` casts caustics.
    pub(crate) fn any_casts<'a>(&self, ids: impl IntoIterator<Item = &'a MaterialId>) -> bool {
        let features = self.features.lock().unwrap();
        features.casts() && ids.into_iter().any(|id| features.casting(id.0))
    }

    /// Whether any of `ids` has the stack face flag.
    pub(crate) fn any_stack_face<'a>(&self, ids: impl IntoIterator<Item = &'a MaterialId>) -> bool {
        let features = self.features.lock().unwrap();
        features.stack_faces() && ids.into_iter().any(|id| features.stack_face(id.0))
    }

    /// Whether any of `ids` displaces by a world length.
    pub(crate) fn any_world_length(&self, ids: &[MaterialId]) -> bool {
        let set = self.world_lengths.lock().unwrap();
        ids.iter().any(|id| set.contains(&id.0))
    }

    pub(crate) fn alloc(&self, device: &Device) -> Result<MaterialId> {
        self.table.alloc(device, 1).map(MaterialId)
    }

    /// Returns `id` for reuse; its record stays until a new material overwrites it.
    pub(crate) fn free(&self, id: MaterialId) {
        self.extended.lock().unwrap().remove(&id.0);
        self.features.lock().unwrap().free(id.0);
        self.world_lengths.lock().unwrap().remove(&id.0);
        let was = self.hit_keys.lock().unwrap().remove(&id.0);
        let back = |k: [u32; 4]| (k[0] & material_flags::HAS_BACK != 0).then_some(k[1]);
        self.note_back(id.0, was.and_then(back), None);
        self.table.free(id.0, 1);
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
        let id = self.table.alloc(self.device)?;
        if let Err(e) = self.update(id, m) {
            self.table.free(id);
            return Err(e);
        }
        Ok(id)
    }

    /// Replaces material `id` from this frame on. Its triangles stay as opaque or
    /// micromapped in traversal as their mount cooked them: coverage that changes at run time
    /// goes through a linked surface's `coverage`. A subdivision copy keeps last frame's positions
    /// by whether its materials displaced by a world length when it was set. A tessellated
    /// entry's `displacement_reach` must cover what its materials displace as updated.
    pub fn update(&mut self, id: MaterialId, m: &MaterialGpu) -> Result<()> {
        assert!(
            id.0 < self.table.table.capacity(),
            "{id:?} is outside the material table"
        );
        let record = self.table.record(m);
        self.table.check_sides(id, &record)?;
        let src = self.window.put(self.device, &record)?;
        self.table.note(id, &record);
        let dst = self.table.table.entry(id.0);
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

    /// A linked record shows the lobes of its type's lobe methods and of its own weights and
    /// no others, walks only off a thin wall, and stands back to back with another surface id
    /// only where one of the two types declares its other side.
    #[test]
    fn a_linked_record_shows_what_its_type_declares() {
        use orr_shaders::surface_declares as d;
        use {SceneFeatures as F, material_flags as f};
        let declares = [
            0,
            d::SUBSURFACE | d::EMISSION,
            d::OTHER_SIDE | d::ALWAYS_SHADES,
        ];
        let features = |surface: u32, edit: fn(&mut MaterialGpu)| {
            let mut m: MaterialGpu = Zeroable::zeroed();
            m.surface = surface;
            edit(&mut m);
            m.lobes = record_lobes(&m, &declares);
            F::of_material(&m, &declares)
        };
        let linked = F::LINKED_SURFACES;
        assert_eq!(features(1, |_| {}), linked);
        // A type that may leave a hit to the standard surface shows its record's weights there.
        assert_eq!(
            features(1, |m| m.coat_weight = 0.5),
            linked | F::COAT | F::LINKED_LOBE_FACTORS
        );
        // One that shades every hit itself never does.
        assert_eq!(features(3, |m| m.coat_weight = 0.5), linked);
        assert_eq!(features(3, |m| m.emission_luminance = 2.0), linked);
        assert_eq!(
            features(2, |_| {}),
            linked | F::SUBSURFACE | F::EMISSIVE | F::WALKING
        );
        assert_eq!(
            features(2, |m| m.flags = f::THIN_WALLED),
            linked | F::SUBSURFACE | F::EMISSIVE
        );
        assert_eq!(features(4, |_| {}), linked, "an id nothing links");
        assert_eq!(
            features(4, |m| m.coat_weight = 0.5),
            linked | F::COAT | F::LOBE_FACTORS,
            "an id nothing links is the standard's"
        );

        assert!(sides(&declares, 1, 1));
        assert!(!sides(&declares, 1, 0) && !sides(&declares, 0, 2) && !sides(&declares, 1, 2));
        assert!(sides(&declares, 0, 3) && sides(&declares, 3, 1));
        assert!(
            sides(&declares, 0, 7),
            "an id nothing links is the standard's"
        );
    }

    #[test]
    fn cooked_materials_take_their_flags_and_slot_bits() {
        let record = |m: &Material| MaterialGpu::new(m, &[BoundTexture::NONE; slot::COUNT]);
        let base = Material::default();
        let coat = Material {
            coat_weight: 0.5,
            ..base
        };
        assert_ne!(record(&coat).flags & material_flags::COAT, 0);
        let glass = Material {
            transmission_weight: 1.0,
            subsurface_weight: 1.0,
            ..coat
        };
        let flags = record(&glass).flags;
        assert_ne!(flags & material_flags::DIELECTRIC, 0);
        assert_eq!(
            flags & material_flags::SUBSURFACE,
            0,
            "transmission wins over subsurface"
        );
        let skin = Material {
            subsurface_weight: 1.0,
            ..base
        };
        assert_ne!(record(&skin).flags & material_flags::SUBSURFACE, 0);
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
    }
}
