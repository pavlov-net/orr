//! Mesh cook: welding, rigid parts, static displacement, discrete levels or cluster DAGs,
//! grid quantization, spatial clusters and the stored pages.
//!
//! The cook welds the mesh and splits it per (part, section). With `displace_static`,
//! displaced sections are baked first ([`displace`]). In discrete mode level 0 is the
//! source; level `i` simplifies level `i - 1` to an absolute error of `errors[i - 1]`
//! (permissive simplification that keeps uv seams, with part and section borders locked),
//! falling back to sloppy simplification when that keeps more than three quarters of its
//! input. From `merge_level` on, the parts of a section simplify together into the body
//! (part 0, [`Cluster::MERGED`]). Each piece of each level is cut into spatial clusters;
//! pages hold whole clusters of one level, coarsest level first, and the coarsest level's
//! pages are pinned. DAG mode builds the clusters, groups, pages and hierarchy in [`dag`].
//! Alpha-tested clusters put opaque triangles first and carry opacity micromaps over the
//! rest. Clusters of subsurface sections carry their minimum thickness ([`thickness`]). Blob 0
//! ends with the sets of emissive and caster sections for every cut ([`sets`]).

mod clod;
mod dag;
mod displace;
mod lod;
pub(crate) mod maps;
mod omm;
mod sets;
pub(crate) mod tangents;
mod thickness;

use std::ops::Range;
use std::sync::Arc;

use crate::par::prelude::*;
use meshopt::VertexDataAdapter;
use orr_math::{DVec3, Vec2, Vec3};
use orr_pack::instance::MAX_HIDDEN_SECTION;
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{
    Cluster, ClusterSets, FLAT_NORMAL, Level, MAX_SECTIONS, MeshHeader, MeshInfo, OMM_ALIGN,
    PAGE_MAX, PageHeader, PageInfo, Part, PartLevel, Section, SurfaceClass, oct_encode,
};
use orr_pack::{Aabb, AssetId, EncodedBlob, GDeflate, Hash32, Span, Sphere};
use serde::Deserialize;

use crate::assembly::AssemblySettings;
use crate::shading::stream_holds;
use crate::state::Memo;
use crate::texture::Wrap;
use crate::texture::tiles::Cell;
use crate::{Error, Result};
use maps::{Displacement, Emission, Map, Opacity, Raster};
pub use omm::OmmSettings;
use omm::{ClusterOmm, OmmAlpha, OmmRead};
pub(crate) use tangents::TangentCheck;

/// How meshes are simplified, clustered and quantized.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeshSettings {
    pub lod: LodMode,
    /// Cluster vertex limit on solid sections.
    pub max_vertices: u16,
    /// Cluster vertex limit on alpha-tested or double-sided sections, dielectric ones aside.
    pub max_vertices_foliage: u16,
    /// Cluster triangle limit, a multiple of 16 up to 128; clusters aim for at least a quarter.
    pub max_triangles: u16,
    /// Finest grid step, log2 of meters. A mesh too large to span 16 bits per axis at this
    /// step gets the smallest coarser step that does.
    pub position_step_log2: i8,
    /// Meters a skinned vertex may move from its rest position, for every cluster of a
    /// skinned mesh outside animated parts (the bound GPU skinning builds cluster templates
    /// with); ignored without a skin.
    pub deformation_bound: f32,
    /// First discrete level whose parts merge into the body; `None` keeps parts at every
    /// level. Ignored on meshes without parts.
    pub merge_level: Option<u8>,
    /// DAG mode: the error in meters from which the parts of each section continue as one
    /// merged stage; `None` keeps parts at every error. Ignored on meshes without parts.
    pub merge_error: Option<f32>,
    /// DAG mode: dilate the open borders of simplified double-sided sections, dielectric
    /// ones aside, so foliage keeps its coverage.
    pub dilate_double_sided: bool,
    /// DAG mode: node bytes kept in blob 0, resident from mount; a larger hierarchy is
    /// paged.
    pub hierarchy_resident_max: u32,
    /// Opacity micromaps on alpha-tested clusters; `None` for content whose coverage
    /// changes at run time, which never uses them.
    pub omm: Option<OmmSettings>,
    /// Bakes the displacement texture of displaced sections into their geometry.
    pub displace_static: Option<StaticDisplacement>,
    /// Repeated rigid pieces of a static, unskinned, unjoined mesh cook once as unit models
    /// `<model>/unit/<i>`, placed by the instance sources that place the model.
    pub assemblies: Option<AssemblySettings>,
    /// Degrees: a mesh drops `TANGENT` when tangents derived per triangle from positions and
    /// uvs shade its normal maps within this of its tangents, stated or MikkTSpace's, at the
    /// 99th percentile; `None` keeps them.
    pub tangent_error_max: Option<f32>,
}

/// Cook-time displacement: triangles split until no edge is over `max_edge`, vertices moved
/// along their normal by the material's displacement, then simplified to `target_error`.
#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticDisplacement {
    /// Meters.
    pub max_edge: f32,
    /// Meters.
    pub target_error: f32,
}

impl Default for MeshSettings {
    fn default() -> Self {
        Self {
            lod: LodMode::default(),
            max_vertices: 128,
            max_vertices_foliage: 256,
            max_triangles: 128,
            position_step_log2: -11,
            deformation_bound: 0.0,
            merge_level: None,
            merge_error: None,
            dilate_double_sided: true,
            hierarchy_resident_max: 512 << 10,
            omm: Some(OmmSettings::default()),
            displace_static: None,
            assemblies: None,
            tangent_error_max: Some(1.0),
        }
    }
}

/// Level of detail representation.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum LodMode {
    /// Discrete levels: level 0 is the source and level `i` simplifies level `i - 1` to an
    /// absolute error of `errors[i - 1]` meters. Empty gives one level.
    Discrete { errors: Vec<f32> },
    /// Cluster DAGs simplified until they stop reducing: one per (part, section), and with
    /// `merge_error` one merged stage per section.
    Dag,
}

impl Default for LodMode {
    fn default() -> Self {
        Self::Discrete { errors: Vec::new() }
    }
}

/// A mesh built by a library caller ([`crate::Source::Mesh`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshInput {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// xyz and bitangent sign. Without them, a mesh whose materials read tangents gets
    /// MikkTSpace-compatible ones, whose sign takes triangles to wind counter-clockwise seen
    /// from the side of `normals`.
    pub tangents: Option<Vec<orr_math::Vec4>>,
    pub uvs: Option<Vec<Vec2>>,
    /// Uv set 1, which material slots on set 1 read.
    pub uvs1: Option<Vec<Vec2>>,
    /// Per vertex, the values [`TextureBinding::vertex`](orr_pack::material::TextureBinding::vertex)
    /// reads, colors in linear Rec.709.
    pub colors: Option<Vec<orr_math::Vec4>>,
    pub indices: Vec<u32>,
    /// Index ranges, one per material.
    pub sections: Vec<Span>,
    pub parts: Option<MeshParts>,
}

/// Rigid parts of a [`MeshInput`], as `parts_from_attribute` reads them from glTF.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshParts {
    /// Per vertex: the part value.
    pub values: Vec<u16>,
    /// Declared value count; every value is below it.
    pub count: u16,
    /// One part per (value, section): part = value x sections + section.
    pub by_section: bool,
    /// Per vertex: the pivot of its part, alike for every vertex of a part within one grid
    /// step. Without it pivots are the model origin.
    pub pivots: Option<Vec<Vec3>>,
}

/// A mesh ready to cook: one vertex array, index ranges per section.
pub(crate) struct MeshData {
    pub(crate) positions: Vec<[f32; 3]>,
    pub(crate) normals: Vec<[f32; 3]>,
    pub(crate) tangents: Option<Vec<[f32; 4]>>,
    /// Per uv set, one or two, a uv per vertex.
    pub(crate) uvs: Vec<Vec<[f32; 2]>>,
    /// The vertex color stream.
    pub(crate) colors: Option<Vec<[f32; 4]>>,
    /// Per vertex where a displacement is a length of a space the vertices have left: the unit
    /// normal there, then what a displacement of 1 moves the vertex by, which is that normal taken
    /// to the vertices' space as a position is.
    pub(crate) moves: Option<Vec<[[f32; 3]; 2]>>,
    pub(crate) indices: Vec<u32>,
    pub(crate) sections: Vec<SectionData>,
    pub(crate) skin: Option<Skin>,
    pub(crate) parts: Option<PartData>,
    /// The grid step may not be finer than `2^min_exponent` m (models joined crack-free).
    pub(crate) min_exponent: Option<i8>,
}

/// The hash of a mesh with geometry hash `geometry` and these sections.
fn sections_hash(geometry: Hash32, sections: &[SectionData]) -> Hash32 {
    let mut h = blake3::Hasher::new();
    h.update(&geometry.0);
    for s in sections {
        h.update(&(s.indices.start as u64).to_le_bytes());
        h.update(&(s.indices.end as u64).to_le_bytes());
        h.update(&[s.flags, s.subsurface.into()]);
        let f = s.maps.fingerprint();
        h.update(&(f.len() as u64).to_le_bytes());
        h.update(f.as_bytes());
    }
    Hash32(*h.finalize().as_bytes())
}

/// A mesh to cook: in memory, or read when its cook starts.
pub(crate) enum MeshSource {
    Data(Box<MeshData>),
    Lazy(LazyMesh),
}

type LoadMesh = dyn Fn(&Memo) -> Result<MeshData> + Send + Sync;

/// A mesh read when its cook starts.
pub(crate) struct LazyMesh {
    /// Sections of the mesh `load` returns; the plan sets their maps.
    pub(crate) sections: Vec<SectionData>,
    pub(crate) triangles: u64,
    /// Hash of everything `load` reads, besides the sections.
    pub(crate) source: Hash32,
    /// Reads the mesh; its sections are left empty.
    pub(crate) load: Arc<LoadMesh>,
}

impl MeshSource {
    pub(crate) fn sections(&self) -> &[SectionData] {
        match self {
            Self::Data(m) => &m.sections,
            Self::Lazy(l) => &l.sections,
        }
    }

    pub(crate) fn sections_mut(&mut self) -> &mut [SectionData] {
        match self {
            Self::Data(m) => &mut m.sections,
            Self::Lazy(l) => &mut l.sections,
        }
    }

    pub(crate) fn hash(&self) -> Hash32 {
        match self {
            Self::Data(m) => m.hash(),
            Self::Lazy(l) => sections_hash(l.source, &l.sections),
        }
    }

    pub(crate) fn triangles(&self) -> u64 {
        match self {
            Self::Data(m) => m.triangles(),
            Self::Lazy(l) => l.triangles,
        }
    }

    /// The mesh, read now if it is lazy. A lazy mesh whose triangles no longer match its sections
    /// fails as changed.
    pub(crate) fn read(self, asset: &str, memo: &Memo) -> Result<MeshData> {
        let _z = crate::profile::zone!("mesh.load");
        match self {
            Self::Data(m) => Ok(*m),
            Self::Lazy(l) => {
                let mut m = (l.load)(memo)?;
                if m.triangles() != l.triangles {
                    return Err(Error::Mesh {
                        asset: asset.into(),
                        message: format!(
                            "{} triangles read, {} planned: the source changed",
                            m.triangles(),
                            l.triangles
                        ),
                    });
                }
                m.sections = l.sections;
                Ok(m)
            }
        }
    }
}

/// Per vertex: four skeleton joints and their unorm8 weights, summing to 255.
pub(crate) struct Skin {
    pub(crate) joints: Vec<[u8; 4]>,
    pub(crate) weights: Vec<[u8; 4]>,
}

/// Rigid parts from a per-vertex value.
pub(crate) struct PartData {
    pub(crate) values: Vec<u16>,
    pub(crate) count: u16,
    pub(crate) by_section: bool,
    pub(crate) pivots: Pivots,
    /// `Part::joint` per value; the value itself without it.
    pub(crate) joints: Option<Vec<u16>>,
}

pub(crate) enum Pivots {
    PerValue(Vec<[f32; 3]>),
    PerVertex(Vec<[f32; 3]>),
    Origin,
}

#[derive(Clone)]
pub(crate) struct SectionData {
    pub(crate) indices: Range<usize>,
    /// [`Section`] flags.
    pub(crate) flags: u8,
    /// The material scatters below the surface (not thin-walled, no transmission).
    pub(crate) subsurface: bool,
    pub(crate) maps: MapSpec,
}

/// Whether material `m` binds a texture on uv set 1.
fn binds_uv1(m: &Material) -> bool {
    m.textures
        .iter()
        .any(|b| b.texture != AssetId::NONE && b.uv_set & 1 == 1)
}

/// Whether a mesh section drawn with material `m` reads uv set 1, for a texture or for the
/// tangent frame it reads.
pub(crate) fn reads_uv1(m: &Material) -> bool {
    let frame = reads_tangent(m) || m.tangent_map(slot::NORMAL);
    let coat = m.tangent_map(slot::COAT_NORMAL) || m.coat_anisotropy > 0.0;
    binds_uv1(m) || (frame && m.tangent_set() == 1) || (coat && m.coat_tangent_set() == 1)
}

/// Whether material `m` reads the tangent with or without a normal map.
pub(crate) fn reads_tangent(m: &Material) -> bool {
    let coat_map = m.tangent_map(slot::COAT_NORMAL);
    m.specular_anisotropy > 0.0 || m.coat_anisotropy > 0.0 || coat_map || m.surface != 0
}

/// [`Section`] flags of a section drawn with material `m`, whose back is `back` where it has
/// one of its own: the section emits where either side does.
pub(crate) fn section_flags(m: &Material, back: Option<&Material>) -> u8 {
    let mut flags = 0;
    if m.alpha_mode == Material::MASK {
        flags |= Section::ALPHA_TESTED;
    }
    if m.double_sided != 0 {
        flags |= Section::DOUBLE_SIDED;
    }
    if m.emits() || back.is_some_and(Material::emits) {
        flags |= Section::EMISSIVE;
    }
    if m.casts() {
        flags |= Section::CASTER;
    }
    if m.class() == SurfaceClass::Dielectric && m.thin_walled == 0 {
        flags |= Section::SOLID;
    }
    flags | m.class().flags()
}

impl SectionData {
    /// Whether the section's alpha-tested clusters take micromaps. Rays force micromaps to two
    /// states, where a micro-triangle is either skipped or a hit the traversal commits, so none
    /// of them could send a dielectric or presence candidate to the ray's handler. Those classes
    /// run the alpha test there.
    fn takes_micromaps(&self) -> bool {
        SurfaceClass::from_flags(self.flags) == Some(SurfaceClass::Opaque)
    }

    /// A section of `indices` drawn with material `m`, whose back is `back` where it has one
    /// of its own.
    pub(crate) fn new(
        indices: Range<usize>,
        m: &Material,
        back: Option<&Material>,
        maps: MapSpec,
    ) -> Self {
        Self {
            indices,
            flags: section_flags(m, back),
            subsurface: m.subsurface_weight > 0.0
                && m.thin_walled == 0
                && m.transmission_weight == 0.0,
            maps,
        }
    }
}

/// The cooked textures a section's clusters depend on.
#[derive(Clone, Debug, Default)]
pub(crate) struct MapSpec {
    pub(crate) opacity: OpacitySpec,
    pub(crate) displacement: Option<DisplacementSpec>,
    /// Along the surface's u and v, beside `displacement`.
    pub(crate) tangent_displacement: [Option<DisplacementSpec>; 2],
    pub(crate) normal: NormalSpec,
    pub(crate) emission: Option<EmissionSpec>,
    pub(crate) uv1: bool,
}

/// A cooked texture or tile set a material slot binds.
#[derive(Clone, Debug)]
pub(crate) struct MapTexture {
    pub(crate) unit: String,
    pub(crate) key: Hash32,
    /// A set's tiles: each one's column and row, unit and key. Empty for a lone texture.
    pub(crate) tiles: Vec<(Cell, String, Hash32)>,
}

impl MapTexture {
    pub(crate) fn lone(unit: &str, key: Hash32) -> Self {
        Self {
            unit: unit.to_string(),
            key,
            tiles: Vec::new(),
        }
    }

    /// The units whose texels it reads.
    pub(crate) fn units(&self) -> impl Iterator<Item = &str> {
        let lone = self.tiles.is_empty().then_some(self.unit.as_str());
        lone.into_iter()
            .chain(self.tiles.iter().map(|t| t.1.as_str()))
    }
}

/// How a material slot reads a cooked texture: [`maps::Raster`]'s edge modes and uv transform,
/// and per channel what it reads past 0..1.
#[derive(Copy, Clone, Debug)]
pub(crate) struct MapRead {
    pub(crate) wrap: [Wrap; 2],
    pub(crate) uv: [f32; 6],
    pub(crate) outside: [f32; 4],
}

/// The cooked emission texture of a section whose material emits, how it reads and the uv set
/// it reads.
#[derive(Clone, Debug)]
pub(crate) struct EmissionSpec {
    pub(crate) texture: MapTexture,
    pub(crate) read: MapRead,
    pub(crate) uv_set: usize,
}

/// What a section's material reads the vertex tangent for.
#[derive(Clone, Debug, Default)]
pub(crate) enum NormalSpec {
    /// Nothing.
    #[default]
    None,
    /// Only to turn this cooked normal map, read on uv set `uv_set`. Tangents follow
    /// `frame_set`, a [`Material::tangent_set`].
    Texture {
        texture: MapTexture,
        read: MapRead,
        uv_set: usize,
        frame_set: usize,
    },
    /// More than a normal map the cook can read: anisotropy, a linked surface, or a normal
    /// map not cooked here.
    Tangent { frame_set: usize },
}

impl NormalSpec {
    /// What tangents follow, when the material reads them: a [`Material::tangent_set`].
    fn tangent_set(&self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Texture { frame_set, .. } | Self::Tangent { frame_set } => Some(*frame_set),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) enum OpacitySpec {
    /// Not alpha-tested.
    #[default]
    None,
    Constant(bool),
    /// This cooked opacity texture.
    Texture {
        texture: MapTexture,
        cutoff: f32,
        read: MapRead,
        uv_set: usize,
    },
    Unknown,
}

#[derive(Clone, Debug)]
pub(crate) struct DisplacementSpec {
    pub(crate) scale: f32,
    pub(crate) offset: f32,
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) texture: Option<DisplacementTexture>,
}

/// The cooked displacement texture, and how and on which uv set it reads.
#[derive(Clone, Debug)]
pub(crate) struct DisplacementTexture {
    pub(crate) texture: MapTexture,
    pub(crate) read: MapRead,
    pub(crate) uv_set: usize,
}

impl MapSpec {
    /// The texture units the spec reads.
    pub(crate) fn units(&self) -> impl Iterator<Item = &str> {
        let opacity = match &self.opacity {
            OpacitySpec::Texture { texture, .. } => Some(texture),
            _ => None,
        };
        let displacement = self
            .displacement
            .as_ref()
            .and_then(|d| d.texture.as_ref().map(|t| &t.texture));
        let normal = match &self.normal {
            NormalSpec::Texture { texture, .. } => Some(texture),
            _ => None,
        };
        let tangents = self.tangent_displacement.iter().flatten();
        let tangents = tangents.filter_map(|d| d.texture.as_ref().map(|t| &t.texture));
        let emission = self.emission.as_ref().map(|e| &e.texture);
        let textures = opacity
            .into_iter()
            .chain(displacement)
            .chain(tangents)
            .chain(normal)
            .chain(emission);
        textures.flat_map(MapTexture::units)
    }

    /// Everything of the spec that shapes the cooked mesh: values and texture keys.
    fn fingerprint(&self) -> String {
        let opacity = match &self.opacity {
            OpacitySpec::Texture {
                texture,
                cutoff,
                read,
                uv_set,
            } => format!("texture {:?} {cutoff:?} {read:?} {uv_set}", texture.key),
            o => format!("{o:?}"),
        };
        let height = |d: &DisplacementSpec| {
            (
                [d.scale, d.offset, d.min, d.max],
                d.texture
                    .as_ref()
                    .map(|t| (t.texture.key, t.read, t.uv_set)),
            )
        };
        let displacement = self.displacement.as_ref().map(height);
        let tangents = self
            .tangent_displacement
            .each_ref()
            .map(|d| d.as_ref().map(height));
        let normal = match &self.normal {
            NormalSpec::Texture {
                texture,
                read,
                uv_set,
                frame_set,
            } => format!("texture {:?} {read:?} {uv_set} {frame_set}", texture.key),
            n => format!("{n:?}"),
        };
        let emission = self
            .emission
            .as_ref()
            .map(|e| (e.texture.key, e.read, e.uv_set));
        format!(
            "{opacity} {displacement:?} {tangents:?} {normal} {emission:?} {}",
            self.uv1
        )
    }
}

/// What a section's cluster build reads, resolved from its [`MapSpec`].
#[derive(Default)]
pub(crate) struct SectionMaps {
    pub(crate) opacity: Option<Opacity>,
    pub(crate) displacement: Option<Displacement>,
    /// Along the surface's u and v.
    pub(crate) tangent_displacement: [Option<Displacement>; 2],
    /// Mip 0 of the [`NormalSpec::Texture`] map, stored x and y.
    pub(crate) normal: Option<Map<[Raster; 2]>>,
    pub(crate) emission: Option<Emission>,
}

impl SectionMaps {
    /// Static displacement bakes this section into the geometry.
    fn bakes(&self) -> bool {
        self.displacement.as_ref().is_some_and(|d| d.map.is_some())
    }
}

/// How cooked pages are compressed: as the archive writer would, with this zstd level and
/// GDeflate codec.
#[derive(Clone, Copy)]
pub(crate) struct Compress<'a> {
    pub(crate) zstd_level: i32,
    pub(crate) gdeflate: Option<&'a dyn GDeflate>,
}

pub(crate) struct CookedMesh {
    pub(crate) info: MeshInfo,
    /// Page blobs, compressed, in [`MeshInfo::pages`] order; pinned pages are mount blobs.
    pub(crate) pages: Vec<EncodedBlob<'static>>,
    /// Node page blobs, in [`MeshInfo::node_pages`] order; the first is the top page.
    pub(crate) node_pages: Vec<Vec<u8>>,
    pub(crate) warnings: Vec<String>,
    /// The tangent check of a mesh with tangents.
    pub(crate) tangents: Option<TangentCheck>,
}

/// Cooks a [`check`]ed mesh, dropping its vertex streams once the welded mesh holds them and its
/// normal maps once the tangent check has read them.
pub(crate) fn cook(
    asset: &str,
    mut m: MeshData,
    s: &MeshSettings,
    mut maps: Vec<SectionMaps>,
    compress: Compress<'_>,
) -> Result<CookedMesh> {
    let _z = crate::profile::zone!("mesh.cook");
    let fail = |message: String| Error::Mesh {
        asset: asset.into(),
        message,
    };
    let mut warnings = Vec::new();
    // Baked displacement moves the geometry the derived tangents would follow.
    let bakes = s.displace_static.is_some() && maps.iter().any(SectionMaps::bakes);
    let tangent_check = match (&m.tangents, s.tangent_error_max) {
        (Some(t), Some(max)) if !bakes => tangents::check(&m, t, &maps, max),
        _ => TangentCheck::KEPT,
    };
    for sm in &mut maps {
        sm.normal = None;
    }
    let maps = maps.as_slice();
    let stretch_zone = crate::profile::zone!("mesh.stretch");
    let stretch: Vec<[f32; 2]> = m
        .sections
        .iter()
        .map(|sec| {
            let vertices = m.moves.iter().flat_map(|moves| {
                let at = |&v: &u32| Vec3::from(moves[v as usize][1]).length();
                m.indices[sec.indices.clone()].iter().map(at)
            });
            vertices.fold([1.0f32; 2], |[lo, hi], l| [lo.min(l), hi.max(l)])
        })
        .collect();
    drop(stretch_zone);
    let mut w = lod::Welded::new(&m, !tangent_check.derive, bakes);
    let min_exponent = m.min_exponent;
    let source_grid = Grid::new(&w.positions, s.position_step_log2, min_exponent).map_err(fail)?;
    let mut parts = lod::Parts::new(&w, &m, &source_grid).map_err(fail)?;
    let (skinned, stored_tangents) = (m.skin.is_some(), m.tangents.is_some());
    let sections = std::mem::take(&mut m.sections);
    {
        let _z = crate::profile::zone!("mesh.free");
        drop(m);
    }
    for (si, sec) in sections.iter().enumerate() {
        let tangents = maps[si].tangent_displacement.iter().flatten();
        for d in maps[si].displacement.iter().chain(tangents) {
            d.check_range()
                .map_err(|e| fail(format!("section {si}: {e}")))?;
            warnings.extend(lod::displacement_seams(&w, &source_grid, si, d));
        }
        if sec.subsurface {
            let n = lod::border_edges(&w, &source_grid, si);
            if n > 0 {
                warnings.push(format!(
                    "section {si}: subsurface material on {n} border edges; the subsurface walk leaks through them"
                ));
            }
        }
    }
    let mut baked = vec![false; sections.len()];
    let mut grid = match &s.displace_static {
        Some(d) => {
            baked = displace::bake(&mut w, &mut parts, maps, d);
            if baked.iter().any(|&b| b) {
                drop(source_grid);
                let grid =
                    Grid::new(&w.positions, s.position_step_log2, min_exponent).map_err(fail)?;
                displace::simplify(&mut w, &mut parts, &grid, &baked, d.target_error);
                grid
            } else {
                source_grid
            }
        }
        None => source_grid,
    };
    let bounds_zone = crate::profile::zone!("mesh.bounds");
    let mut bounds = Aabb::EMPTY;
    grid.floats.iter().for_each(|&p| bounds.extend(p));
    let sphere = meshopt::compute_sphere_bounds(
        meshopt::PositionDataAdapter {
            data: bytemuck::cast_slice(&grid.floats),
            position_count: grid.floats.len(),
            position_stride: 12,
            position_offset: 0,
        },
        None,
    );
    drop(bounds_zone);
    // Only discrete levels read the float positions past here.
    if s.lod == LodMode::Dag {
        grid.floats = Vec::new();
    }
    let mut omm_alpha = Vec::with_capacity(maps.len());
    for (si, (sm, sec)) in maps.iter().zip(&sections).enumerate() {
        let alpha = match (&s.omm, &sm.opacity) {
            (Some(_), Some(Opacity::Texture { map, cutoff, .. })) if sec.takes_micromaps() => {
                let a = map.try_map(|t| OmmRead::new(t, *cutoff));
                if a.is_none() {
                    let [w, h] = map
                        .each()
                        .fold([0, 0], |[w, h], t| [w.max(t.mip0.w), h.max(t.mip0.h)]);
                    warnings.push(format!(
                        "section {si}: opacity texture {w}x{h} is over {} texels a side; no \
                         micromaps",
                        OmmAlpha::MAX_SIZE
                    ));
                }
                a
            }
            _ => None,
        };
        omm_alpha.push(alpha);
    }

    let attributes = if w.tangents.is_some() {
        MeshHeader::TANGENT
    } else {
        0
    } | if skinned { MeshHeader::SKIN } else { 0 }
        | if w.uvs.len() > 1 { MeshHeader::UV1 } else { 0 }
        | if w.colors.is_some() {
            MeshHeader::COLOR
        } else {
            0
        };
    let cx = Ctx {
        w: &w,
        grid: &grid,
        parts: &parts,
        sections: &sections,
        maps,
        omm_alpha: &omm_alpha,
        baked: &baked,
        stretch: &stretch,
        s,
        compress,
        deformation_bound: if skinned {
            f16_up(s.deformation_bound)
        } else {
            0
        },
        stride: orr_pack::mesh::attr_stride(attributes) as usize,
    };
    let closed: Vec<bool> = (0..sections.len())
        .into_par_iter()
        .map(|i| lod::closed(&w, &grid, i))
        .collect();
    let mut built = match &s.lod {
        LodMode::Discrete { errors } => discrete(&cx, errors)?,
        LodMode::Dag => dag::cook(&cx).map_err(fail)?,
    };

    built.info.header = MeshHeader {
        bounds,
        sphere: Sphere {
            center: sphere.center,
            radius: sphere.radius,
        },
        position_step_log2: s.position_step_log2,
        position_exponent: grid.k,
        clas_truncate_bits: grid.truncate_bits,
        attributes,
        max_vertices: sections
            .iter()
            .map(|sec| cx.max_vertices(sec.flags))
            .max()
            .unwrap_or(0),
        max_triangles: s.max_triangles,
        ..built.info.header
    };
    built.info.sections = sections
        .iter()
        .enumerate()
        .map(|(i, sec)| Section {
            material_slot: i as u16,
            flags: sec.flags | if closed[i] { Section::CLOSED } else { 0 },
            _pad: 0,
        })
        .collect();
    let h = &mut built.info.header;
    for (i, sec) in built.info.sections.iter().enumerate() {
        let bit = 1u32 << i.min(MAX_HIDDEN_SECTION);
        match sec.class().expect("sections take one class") {
            SurfaceClass::Opaque => h.opaque_sections |= bit,
            _ => h.layer_sections |= bit,
        }
    }
    sets::build(&mut built.info, &sections, &built.clusters);
    let _free = crate::profile::zone!("mesh.free");
    drop((built.clusters, w, parts, grid, omm_alpha));
    Ok(CookedMesh {
        info: built.info,
        pages: built.pages,
        node_pages: built.node_pages,
        warnings,
        tangents: stored_tangents.then_some(tangent_check),
    })
}

/// What both LOD modes leave for [`cook`] to finish: the info without its header's mesh
/// facts and sections.
struct Built {
    info: MeshInfo,
    pages: Vec<EncodedBlob<'static>>,
    node_pages: Vec<Vec<u8>>,
    /// What the sets read of every cluster, in mesh cluster order.
    clusters: Vec<sets::SetCluster>,
}

/// What every cluster of a mesh is built from.
struct Ctx<'a> {
    w: &'a lod::Welded,
    grid: &'a Grid,
    parts: &'a lod::Parts,
    sections: &'a [SectionData],
    maps: &'a [SectionMaps],
    omm_alpha: &'a [Option<Map<OmmRead<'a>>>],
    /// Sections whose displacement is in their geometry: no displacement bounds.
    baked: &'a [bool],
    /// Per section, the least and most a height of 1 moves a vertex: by `MeshData::moves`, or by 1
    /// as a run time displacement does.
    stretch: &'a [[f32; 2]],
    s: &'a MeshSettings,
    compress: Compress<'a>,
    deformation_bound: u16,
    stride: usize,
}

impl Ctx<'_> {
    fn max_vertices(&self, flags: u8) -> u16 {
        let dielectric = SurfaceClass::from_flags(flags) == Some(SurfaceClass::Dielectric);
        if flags & (Section::ALPHA_TESTED | Section::DOUBLE_SIDED) != 0 && !dielectric {
            self.s.max_vertices_foliage
        } else {
            self.s.max_vertices
        }
    }

    /// A cluster of section `section` and part `part` over welded vertices `verts` at grid
    /// positions `cells`, with cluster-local corners `tris`. Its errors and spheres are the
    /// caller's to set.
    fn cluster(
        &self,
        section: u16,
        part: u16,
        merged: bool,
        verts: &[u32],
        cells: &[[i32; 3]],
        mut tris: Vec<u8>,
    ) -> ClusterData {
        let _z = crate::profile::zone!("mesh.cluster");
        let w = self.w;
        let n = verts.len();
        let mut order = IDENTITY;
        meshopt::optimize_meshlet_level(&mut order[..n], &mut tris, 3);
        let (mut by_order, mut cells_by_order) =
            ([0; MAX_CLUSTER_VERTICES], [[0; 3]; MAX_CLUSTER_VERTICES]);
        for (k, &i) in order[..n].iter().enumerate() {
            by_order[k] = verts[i as usize];
            cells_by_order[k] = cells[i as usize];
        }
        let (verts, cells) = (&by_order[..n], &cells_by_order[..n]);
        let sec = &self.sections[usize::from(section)];
        let section_maps = &self.maps[usize::from(section)];
        let mut uv = [ClusterUvs::ZERO; 2];
        for (r, u) in uv.iter_mut().zip(&w.uvs) {
            *r = ClusterUvs::new(u, verts);
        }
        let uv = &uv[..w.uvs.len()];
        // The uvs of set `set` as the page stores them, per cluster vertex.
        let stored = |set: usize| {
            let mut out = [[0.0f32; 2]; MAX_CLUSTER_VERTICES];
            for (o, &v) in out.iter_mut().zip(verts) {
                *o = uv[set].stored(w.uvs[set][v as usize]);
            }
            out
        };
        let uvs = stored(section_maps.opacity.as_ref().map_or(0, Opacity::uv_set));
        let opaque = {
            let _z = crate::profile::zone!("mesh.cluster.opaque");
            partition_opaque(&uvs[..n], &mut tris, section_maps.opacity.as_ref())
        };
        let flat = if sec.flags & Section::CASTER != 0 {
            let _z = crate::profile::zone!("mesh.cluster.flat");
            partition_flat(
                w,
                self.grid,
                verts,
                cells,
                &mut tris[..3 * usize::from(opaque)],
            )
        } else {
            0
        };
        stored_corner_order(&mut tris);
        let mut c = build_cluster(w, self.grid, verts, cells, tris, sec.flags, uv);
        if sec.flags & (Section::EMISSIVE | Section::CASTER) != 0 {
            let _z = if section_maps.emission.is_some() {
                crate::profile::zone!("mesh.cluster.emission")
            } else {
                crate::profile::zone!("mesh.cluster.bounds")
            };
            let step = self.grid.step();
            let mut positions = [DVec3::ZERO; MAX_CLUSTER_VERTICES];
            for (p, g) in positions.iter_mut().zip(cells) {
                *p = DVec3::from(g.map(|v| f64::from(v) * step));
            }
            let mut emission_uvs = [[0.0f32; 2]; MAX_CLUSTER_VERTICES];
            let emission = section_maps.emission.as_ref().map(|e| {
                for (o, &v) in emission_uvs.iter_mut().zip(verts) {
                    *o = w.uvs[e.uv_set][v as usize];
                }
                (e, &emission_uvs[..n])
            });
            c.bounds = Some(sets::ClusterBounds::new(
                &positions[..n],
                c.tris.as_chunks::<3>().0,
                emission,
            ));
        }
        c.record.flat_triangles = flat;
        if self.s.omm.is_some() && section_maps.opacity.is_some() && sec.takes_micromaps() {
            c.uvs = uvs[..n].to_vec();
        }
        c.record.section = section;
        c.record.part = part;
        let frame = sec.maps.normal.tangent_set().unwrap_or(0) as u8;
        c.record.flags |= frame << Cluster::TANGENT_SET_SHIFT;
        c.record.alpha_first = opaque;
        if merged {
            c.record.flags |= Cluster::MERGED;
        }
        if self.parts.animated[usize::from(part)] && !merged {
            c.record.flags |= Cluster::ANIMATED_PART;
        } else {
            c.record.deformation_bound = self.deformation_bound;
        }
        if let Some(d) = &section_maps.displacement
            && !self.baked[usize::from(section)]
        {
            // Stored uvs, which the runtime reads: their rounding can cross a tile's edge.
            let bounds = |d: &Displacement| {
                let stored = stored(d.uv_set);
                let at = |i: u8| stored[usize::from(i)];
                d.bounds(c.tris.as_chunks::<3>().0.iter().map(|t| t.map(at)))
            };
            let [lo, hi] = bounds(d);
            // A move across the surface is no longer than these, in any direction.
            let across: f32 = section_maps
                .tangent_displacement
                .iter()
                .flatten()
                .map(|d| {
                    let [lo, hi] = bounds(d);
                    lo.abs().max(hi.abs())
                })
                .sum();
            let [least, most] = self.stretch[usize::from(section)];
            c.record.displacement_min = f16_down((lo * least).min(lo * most) - across);
            c.record.displacement_max = f16_up((hi * least).max(hi * most) + across);
        }
        c
    }

    /// Micromaps of an alpha-tested cluster from its stored uvs, at most `bytes_max`. A
    /// smaller cap than the one that made `c`'s micromaps never takes a higher level, so
    /// measuring starts at their level.
    fn omm(&self, c: &ClusterData, bytes_max: u32) -> Option<ClusterOmm> {
        let o = self.s.omm.as_ref()?;
        let section = usize::from(c.record.section);
        if !self.sections[section].takes_micromaps() {
            return None;
        }
        let opacity = self.maps[section].opacity.as_ref()?;
        let alpha = &c.tris.as_chunks::<3>().0[usize::from(c.record.alpha_first)..];
        let s = OmmSettings {
            cluster_bytes_max: bytes_max,
            max_level: c.omm.as_ref().map_or(o.max_level, |m| m.max_level),
            ..o.clone()
        };
        omm::cluster_omm(opacity, self.omm_alpha[section].as_ref(), &c.uvs, alpha, &s)
    }

    fn omm_all(&self, clusters: Vec<&mut ClusterData>) {
        if let Some(o) = &self.s.omm {
            let _z = crate::profile::zone!("omm");
            clusters.into_par_iter().for_each(|c| {
                c.omm = self.omm(c, o.cluster_bytes_max);
            });
        }
    }
}

/// Discrete levels: pages of whole clusters of one level, coarsest level first.
fn discrete(cx: &Ctx<'_>, errors: &[f32]) -> Result<Built> {
    let levels = lod::levels(cx.w, cx.grid, cx.parts, cx.s, errors);
    let level_count = levels.len();
    let part_count = cx.parts.table.len();
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&cx.grid.floats), 12, 0)
        .expect("positions are 12-byte vertices");
    let mut by_level: Vec<Vec<ClusterData>> = Vec::with_capacity(level_count);
    for (li, level) in levels.iter().enumerate() {
        let parent_error = levels.get(li + 1).map_or(f32::INFINITY, |l| l.error);
        let mut clusters = Vec::new();
        for piece in &level.pieces {
            let sec = &cx.sections[usize::from(piece.section)];
            let maxt = usize::from(cx.s.max_triangles);
            let ml = meshopt::build_meshlets_spatial(
                &piece.indices,
                &adapter,
                usize::from(cx.max_vertices(sec.flags)),
                maxt / 4,
                maxt,
                0.5,
            );
            for r in &ml.meshlets {
                let (vo, vc) = (r.vertex_offset as usize, r.vertex_count as usize);
                let (to, tc) = (r.triangle_offset as usize, r.triangle_count as usize);
                let verts = &ml.vertices[vo..vo + vc];
                let cells: Vec<[i32; 3]> =
                    verts.iter().map(|&v| cx.grid.cells[v as usize]).collect();
                let tris = ml.triangles[to..to + tc * 3].to_vec();
                let mut c =
                    cx.cluster(piece.section, piece.part, piece.merged, verts, &cells, tris);
                c.record.self_error = level.error;
                c.record.parent_error = parent_error;
                clusters.push(c);
            }
        }
        by_level.push(clusters);
    }
    cx.omm_all(by_level.iter_mut().flatten().collect());
    by_level
        .iter_mut()
        .flatten()
        .for_each(|c| c.uvs = Vec::new());
    thickness::assign(
        by_level.iter_mut().flatten().collect(),
        cx.grid,
        cx.sections,
    );

    // Pages, coarsest level first; mesh cluster indices count in page order.
    let mut pages = Vec::new();
    let mut page_info = Vec::new();
    let mut level_spans = vec![Span::default(); level_count];
    let mut part_level_spans = vec![Span::default(); part_count * level_count];
    let mut parts = cx.parts.table.clone();
    let mut set_clusters = Vec::new();
    let mut next = 0u32;
    let mut pinned_pages = 0;
    for li in (0..level_count).rev() {
        let clusters = &by_level[li];
        level_spans[li] = Span {
            first: next,
            count: clusters.len() as u32,
        };
        for p in 0..part_count {
            part_level_spans[p * level_count + li].first = next;
        }
        set_clusters.extend(clusters.iter().map(|c| c.set_cluster(0, -1)));
        for (i, c) in clusters.iter().enumerate() {
            let span = &mut part_level_spans[usize::from(c.record.part) * level_count + li];
            if span.count == 0 {
                span.first = next + i as u32;
            }
            span.count += 1;
            c.add_to(&mut parts, cx.grid);
        }
        let paged = paginate(clusters, cx.stride);
        // The coarsest level's pages are pinned.
        let pinned = li + 1 == level_count;
        let stored = paged
            .par_iter()
            .map(|(bytes, ..)| store_page(bytes, pinned, cx.compress))
            .collect::<Result<Vec<_>>>()?;
        for ((bytes, n, tri_count), stored) in paged.into_iter().zip(stored) {
            page_info.push(PageInfo {
                blob: pages.len() as u32 + 1,
                resident_len: bytes.len() as u32,
                tri_count,
                cluster_count: n as u16,
                _pad: 0,
                deps: Span::default(),
            });
            pages.push(stored);
        }
        if li + 1 == level_count {
            pinned_pages = pages.len() as u32;
        }
        next += clusters.len() as u32;
    }
    let merge_level = levels
        .iter()
        .position(|l| l.pieces.iter().any(|p| p.merged))
        .map_or(MeshHeader::NO_MERGE_LEVEL, |l| l as u8);
    let info = MeshInfo {
        header: MeshHeader {
            merge_level,
            merge_error: f32::INFINITY,
            pinned_pages,
            ..bytemuck::Zeroable::zeroed()
        },
        sections: Vec::new(),
        parts,
        part_levels: part_level_spans
            .into_iter()
            .map(|clusters| PartLevel { clusters })
            .collect(),
        levels: levels
            .iter()
            .zip(level_spans)
            .map(|(l, clusters)| Level {
                error: l.error,
                clusters,
            })
            .collect(),
        nodes: Vec::new(),
        groups: Vec::new(),
        node_pages: Vec::new(),
        roots: Vec::new(),
        pages: page_info,
        page_deps: Vec::new(),
        cut_rows: Vec::new(),
        emissive: ClusterSets::default(),
        casters: ClusterSets::default(),
    };
    Ok(Built {
        info,
        pages,
        node_pages: Vec::new(),
        clusters: set_clusters,
    })
}

/// The stored form of a resident page, checked to decode to the same bytes (micromaps
/// follow the stored corner order), compressed as a blob, in the mount section when
/// `pinned`.
fn store_page(bytes: &[u8], pinned: bool, compress: Compress<'_>) -> Result<EncodedBlob<'static>> {
    let stored = {
        let _z = crate::profile::zone!("page.encode");
        orr_pack::mesh::encode_page(bytes)?
    };
    {
        let _z = crate::profile::zone!("page.verify");
        let mut decoded = vec![0; bytes.len()];
        orr_pack::mesh::decode_page(&stored, &mut decoded)?;
        assert!(decoded == bytes, "a cooked page decodes to itself");
    }
    let _z = crate::profile::zone!("page.compress");
    Ok(EncodedBlob::owned(
        stored,
        pinned,
        compress.zstd_level,
        compress.gdeflate,
    )?)
}

pub(crate) fn check(m: &MeshData, s: &MeshSettings) -> Result<(), String> {
    let n = m.positions.len();
    if s.max_triangles == 0 || s.max_triangles > 128 || !s.max_triangles.is_multiple_of(16) {
        return Err(format!(
            "max_triangles {} is not a multiple of 16 up to 128",
            s.max_triangles
        ));
    }
    for v in [s.max_vertices, s.max_vertices_foliage] {
        if !(3..=256).contains(&v) {
            return Err(format!("cluster vertex limit {v} is outside 3..=256"));
        }
    }
    if !(1..=2).contains(&m.uvs.len()) {
        return Err(format!("{} uv sets; meshes carry one or two", m.uvs.len()));
    }
    if let LodMode::Discrete { errors } = &s.lod {
        if errors.len() > 254 {
            return Err(format!("{} levels; at most 255", errors.len() + 1));
        }
        if errors.iter().any(|e| !(e.is_finite() && *e > 0.0)) || !errors.is_sorted() {
            return Err(format!(
                "level errors {errors:?} are not positive, finite and ascending"
            ));
        }
        if let Some(l) = s.merge_level
            && (l == 0 || usize::from(l) > errors.len())
        {
            return Err(format!(
                "merge_level {l} is not a level after the first of {}",
                errors.len() + 1
            ));
        }
    }
    if let Some(t) = s.tangent_error_max
        && !(t.is_finite() && t >= 0.0)
    {
        return Err(format!(
            "tangent_error_max {t} deg is not finite and non-negative"
        ));
    }
    if let Some(e) = s.merge_error
        && !(e.is_finite() && e > 0.0)
    {
        return Err(format!("merge_error {e} m is not positive and finite"));
    }
    if let Some(d) = &s.displace_static
        && !(d.max_edge.is_finite()
            && d.max_edge > 0.0
            && d.target_error.is_finite()
            && d.target_error >= 0.0)
    {
        return Err(format!(
            "displace_static max_edge {} m and target_error {} m are not a positive edge and an error",
            d.max_edge, d.target_error
        ));
    }
    if m.normals.len() != n
        || m.uvs.iter().any(|u| u.len() != n)
        || m.tangents.as_ref().is_some_and(|t| t.len() != n)
        || m.colors.as_ref().is_some_and(|c| c.len() != n)
        || m.moves.as_ref().is_some_and(|c| c.len() != n)
        || m.skin
            .as_ref()
            .is_some_and(|k| k.joints.len() != n || k.weights.len() != n)
        || m.parts.as_ref().is_some_and(|p| {
            p.values.len() != n || matches!(&p.pivots, Pivots::PerVertex(v) if v.len() != n)
        })
    {
        return Err("vertex attribute counts differ".into());
    }
    if let Some(o) = &s.omm {
        o.check()?;
    }
    if !(0.0..=65504.0).contains(&s.deformation_bound) {
        return Err(format!(
            "deformation_bound {} m is outside 0..=65504",
            s.deformation_bound
        ));
    }
    if m.sections.is_empty() || m.sections.len() > MAX_SECTIONS {
        return Err(format!(
            "{} sections, expected 1 to {MAX_SECTIONS}: a cluster names its section in a u16",
            m.sections.len()
        ));
    }
    for sec in &m.sections {
        let r = &sec.indices;
        if r.start > r.end || r.end > m.indices.len() || !r.len().is_multiple_of(3) {
            return Err("section index range".into());
        }
    }
    if let Some(&i) = m.indices.iter().find(|&&i| i as usize >= n) {
        return Err(format!("index {i} past {n} vertices"));
    }
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    if !m.positions.iter().all(|p| finite(p))
        || !m.normals.iter().all(|p| finite(p))
        || !m.uvs.iter().flatten().all(|p| finite(p))
        || !m.tangents.iter().flatten().all(|p| finite(p))
        || !m.colors.iter().flatten().all(|p| finite(p))
        || !m.moves.iter().flatten().flatten().all(|p| finite(p))
    {
        return Err("non-finite vertex attribute".into());
    }
    if !m
        .colors
        .iter()
        .flatten()
        .flatten()
        .all(|&x| stream_holds(x))
    {
        return Err("a vertex color past what the stream's half floats hold".into());
    }
    Ok(())
}

/// Stable partition of a cluster's triangles over its stored vertex uvs: opaque ones first.
/// Returns their count.
fn partition_opaque(uvs: &[[f32; 2]], tris: &mut [u8], opacity: Option<&Opacity>) -> u8 {
    let n = tris.len() / 3;
    let Some(opacity) = opacity else {
        return n as u8;
    };
    let mut opaque = [false; MAX_CLUSTER_CORNERS / 3];
    for (o, t) in opaque.iter_mut().zip(tris.as_chunks::<3>().0) {
        *o = opacity.opaque(t.map(|i| uvs[usize::from(i)]));
    }
    stable_partition(tris, &opaque[..n])
}

/// Moves the triangles marked in `first` ahead of the rest, each side in order; returns how
/// many are marked.
fn stable_partition(tris: &mut [u8], first: &[bool]) -> u8 {
    let mut out = [0u8; MAX_CLUSTER_CORNERS];
    let mut k = 0;
    for pass in [true, false] {
        for (t, _) in tris
            .as_chunks::<3>()
            .0
            .iter()
            .zip(first)
            .filter(|(_, f)| **f == pass)
        {
            out[k..k + 3].copy_from_slice(t);
            k += 3;
        }
    }
    tris.copy_from_slice(&out[..tris.len()]);
    first.iter().filter(|&&f| f).count() as u8
}

/// Stable partition of a cluster's triangles: those whose vertex normals all lie within
/// [`FLAT_NORMAL`] of the face normal first. Returns their count. A corner at `cells` off its
/// welded grid position, moved by border dilation, has no welded face to compare with.
fn partition_flat(
    w: &lod::Welded,
    grid: &Grid,
    verts: &[u32],
    cells: &[[i32; 3]],
    tris: &mut [u8],
) -> u8 {
    // Per cluster vertex: whether it sits at its welded grid position, its welded position
    // and its unit normal.
    let mut welded = [false; MAX_CLUSTER_VERTICES];
    let (mut p, mut unit) = (
        [DVec3::ZERO; MAX_CLUSTER_VERTICES],
        [DVec3::ZERO; MAX_CLUSTER_VERTICES],
    );
    for (k, &v) in verts.iter().enumerate() {
        let v = v as usize;
        welded[k] = cells[k] == grid.cells[v];
        p[k] = Vec3::from(w.positions[v]).as_dvec3();
        unit[k] = Vec3::from(w.normals[v]).as_dvec3().normalize_or_zero();
    }
    let mut flat = [false; MAX_CLUSTER_CORNERS / 3];
    for (f, t) in flat.iter_mut().zip(tris.as_chunks::<3>().0) {
        let t = t.map(usize::from);
        let face = (p[t[1]] - p[t[0]])
            .cross(p[t[2]] - p[t[0]])
            .normalize_or_zero();
        *f = t.iter().all(|&i| welded[i])
            && t.iter()
                .all(|&i| unit[i].distance(face) <= f64::from(FLAT_NORMAL));
    }
    stable_partition(tris, &flat[..tris.len() / 3])
}

/// Rotates each triangle's corners as the page codec stores them; order and winding stay.
/// meshoptimizer's meshlet encoder (`encodeTriangles`) rotates a triangle to start at the
/// edge it shares with one of the last 6 of the 8 edges it remembers, or else at its largest
/// corner, and remembers its two new edges.
fn stored_corner_order(tris: &mut [u8]) {
    const ROTATIONS: [usize; 5] = [0, 1, 2, 0, 1];
    let mut fifo = [[u16::MAX; 2]; 8];
    let mut next = 0;
    for t in tris.as_chunks_mut::<3>().0 {
        let [a, b, c] = t.map(u16::from);
        let shared = (0..8).find_map(|i| {
            let [e0, e1] = fifo[(next + 7 - i) & 7];
            [(a, b), (b, c), (c, a)]
                .iter()
                .position(|&e| e == (e0, e1))
                .map(|edge| (i, edge))
        });
        let rotation = match shared {
            Some((i, edge)) if i < 6 => edge,
            _ if a > b && a > c => 1,
            _ if b > c => 2,
            _ => 0,
        };
        let r = &ROTATIONS[rotation..rotation + 3];
        *t = [t[r[0]], t[r[1]], t[r[2]]];
        let [a, b, c] = t.map(u16::from);
        fifo[next] = [c, b];
        fifo[(next + 1) & 7] = [a, c];
        next = (next + 2) & 7;
    }
}

/// Most vertices and corners a cluster holds ([`check`] bounds the limits).
const MAX_CLUSTER_VERTICES: usize = 256;
const MAX_CLUSTER_CORNERS: usize = 3 * 128;

/// Cluster-local vertex `i` at index `i`.
const IDENTITY: [u32; MAX_CLUSTER_VERTICES] = {
    let mut a = [0; MAX_CLUSTER_VERTICES];
    let mut i = 0;
    while i < MAX_CLUSTER_VERTICES {
        a[i] = i as u32;
        i += 1;
    }
    a
};

/// A cluster's uvs of one set as the page stores them: unorm16 over the cluster's uv range.
#[derive(Copy, Clone)]
struct ClusterUvs {
    offset: [f32; 2],
    scale: [f32; 2],
}

impl ClusterUvs {
    const ZERO: Self = Self {
        offset: [0.0; 2],
        scale: [0.0; 2],
    };

    fn new(uvs: &[[f32; 2]], verts: &[u32]) -> Self {
        let (lo, hi) = verts
            .iter()
            .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &v| {
                let uv = Vec2::from(uvs[v as usize]);
                (lo.min(uv), hi.max(uv))
            });
        Self {
            offset: lo.to_array(),
            scale: (hi - lo).to_array(),
        }
    }

    fn quantize(&self, uv: [f32; 2]) -> [u32; 2] {
        [0, 1].map(|a| {
            if self.scale[a] > 0.0 {
                ((uv[a] - self.offset[a]) / self.scale[a] * 65535.0)
                    .round()
                    .clamp(0.0, 65535.0) as u32
            } else {
                0
            }
        })
    }

    /// The uv a page reader decodes for `uv`.
    fn stored(&self, uv: [f32; 2]) -> [f32; 2] {
        let q = self.quantize(uv);
        [0, 1].map(|a| self.offset[a] + q[a] as f32 / 65535.0 * self.scale[a])
    }
}

/// f16 bits of the smallest half at or above `v`.
fn f16_up(v: f32) -> u16 {
    let h = half::f16::from_f32(v);
    if h.to_f32() < v {
        f16_next(h, true)
    } else {
        h.to_bits()
    }
}

/// f16 bits of the largest half at or below `v`.
fn f16_down(v: f32) -> u16 {
    let h = half::f16::from_f32(v);
    if h.to_f32() > v {
        f16_next(h, false)
    } else {
        h.to_bits()
    }
}

/// Bits of the neighboring finite half above or below `h`.
fn f16_next(h: half::f16, up: bool) -> u16 {
    let b = h.to_bits();
    if h.to_f32() == 0.0 {
        return if up { 0x0001 } else { 0x8001 };
    }
    if (b & 0x8000 == 0) == up {
        b + 1
    } else {
        b - 1
    }
}

/// The grid exponent `k` of positions: `max(step_log2, ceil(log2(extent / 65535)))`,
/// computed without log2 rounding.
pub(crate) fn grid_exponent(positions: &[[f32; 3]], step_log2: i8) -> i8 {
    bounds_exponent(&position_bounds(positions), step_log2)
}

/// The box of `positions`, found in parallel.
fn position_bounds(positions: &[[f32; 3]]) -> Aabb {
    positions
        .par_chunks(1 << 16)
        .map(|c| {
            let mut b = Aabb::EMPTY;
            c.iter().for_each(|&p| b.extend(p));
            b
        })
        .reduce(
            || Aabb::EMPTY,
            |a, b| Aabb {
                min: [0, 1, 2].map(|i| a.min[i].min(b.min[i])),
                max: [0, 1, 2].map(|i| a.max[i].max(b.max[i])),
            },
        )
}

/// [`grid_exponent`] of positions with box `b`.
fn bounds_exponent(b: &Aabb, step_log2: i8) -> i8 {
    let extent = (0..3)
        .map(|a| f64::from(b.max[a]) - f64::from(b.min[a]))
        .fold(0.0, f64::max);
    let mut k = i32::from(step_log2);
    while extent > 65535.0 * 2f64.powi(k) {
        k += 1;
    }
    k as i8
}

/// The mesh grid: every position as an integer multiple of `2^k` meters.
struct Grid {
    k: i8,
    truncate_bits: u8,
    cells: Vec<[i32; 3]>,
    floats: Vec<[f32; 3]>,
    /// Per vertex: the first vertex at its grid position, the id of that position.
    reps: Vec<u32>,
}

impl Grid {
    fn new(positions: &[[f32; 3]], step_log2: i8, min: Option<i8>) -> Result<Self, String> {
        let _z = crate::profile::zone!("mesh.grid");
        let b = position_bounds(positions);
        let own = bounds_exponent(&b, step_log2);
        let k = i32::from(own.max(min.unwrap_or(i8::MIN)));
        let step = 2f64.powi(k);
        // Dividing by a power of two is multiplying by its exact inverse.
        let inverse = 2f64.powi(-k);
        let cell = |p: &[f32; 3]| p.map(|x| (f64::from(x) * inverse + 0.5).floor());
        let over = |p: &[f32; 3]| cell(p).iter().any(|v| v.abs() >= f64::from(1 << 23));
        if let Some(i) = positions.par_iter().light().position_first(over) {
            let p = positions[i];
            return Err(format!(
                "position {p:?} m needs more than 24 bits on the 2^{k} m grid"
            ));
        }
        let cells: Vec<[i32; 3]> = positions
            .par_iter()
            .light()
            .map(|p| cell(p).map(|v| v as i32))
            .collect();
        let max_abs = cells
            .par_iter()
            .light()
            .map(|g| g.iter().fold(0, |m, v| m.max(v.unsigned_abs())))
            .max()
            .unwrap_or(0);
        let floats: Vec<[f32; 3]> = cells
            .par_iter()
            .light()
            .map(|g| g.map(|v| (f64::from(v) * step) as f32))
            .collect();
        let m = meshopt::compute_position_exponent(b.min, b.max, step_log2.into(), 16);
        let own = i32::from(own);
        assert!(
            m == own || m == own + 1,
            "grid exponent {own}, meshoptimizer {m}"
        );
        // Grid positions are exact floats, so equal floats are equal cells.
        let reps = meshopt::generate_position_remap(
            &VertexDataAdapter::new(bytemuck::cast_slice(&floats), 12, 0)
                .expect("positions are 12-byte vertices"),
        );
        Ok(Self {
            k: k as i8,
            truncate_bits: (24 - (32 - max_abs.leading_zeros())) as u8,
            cells,
            floats,
            reps,
        })
    }

    fn step(&self) -> f64 {
        2f64.powi(i32::from(self.k))
    }
}

struct ClusterData {
    record: Cluster,
    local: Vec<[u16; 3]>,
    attrs: Vec<u32>,
    tris: Vec<u8>,
    /// Vertex uvs as the page stores them, until the micromaps are made.
    uvs: Vec<[f32; 2]>,
    omm: Option<ClusterOmm>,
    /// On sections with a set.
    bounds: Option<sets::ClusterBounds>,
}

impl ClusterData {
    fn slots(&self) -> usize {
        self.local.len().next_multiple_of(2)
    }

    /// What the sets read of the cluster, in DAG `group` and simplified from group `refined`
    /// (-1 for none).
    fn set_cluster(&self, group: u32, refined: i32) -> sets::SetCluster {
        sets::SetCluster {
            self_error: self.record.self_error,
            parent_error: self.record.parent_error,
            flags: self.record.flags,
            section: self.record.section,
            group,
            refined,
            bounds: self.bounds,
        }
    }

    /// Bytes the cluster adds to a resident page, its micromaps included.
    fn cost(&self, stride: usize) -> usize {
        orr_pack::mesh::cluster_resident_len(self.local.len(), self.tris.len() / 3, stride)
            + self.omm.as_ref().map_or(0, ClusterOmm::bytes)
    }

    /// Adds the cluster to its part's record in `parts`: its rest AABB unless it is merged, and
    /// its section when it is of an animated part.
    fn add_to(&self, parts: &mut [Part], grid: &Grid) {
        let r = &self.record;
        let part = &mut parts[usize::from(r.part)];
        if r.flags & Cluster::MERGED == 0 {
            self.extend(&mut part.bounds, grid);
        }
        if r.flags & Cluster::ANIMATED_PART != 0 {
            part.sections |= 1 << usize::from(r.section).min(MAX_HIDDEN_SECTION);
        }
    }

    /// Extends `b` by the cluster's rest AABB.
    fn extend(&self, b: &mut Aabb, grid: &Grid) {
        let corner = |l: [u16; 3]| {
            [0, 1, 2].map(|a| {
                ((f64::from(self.record.anchor[a]) + f64::from(l[a])) * grid.step()) as f32
            })
        };
        b.extend(corner([0; 3]));
        b.extend(corner(self.record.local_max));
    }
}

fn build_cluster(
    w: &lod::Welded,
    grid: &Grid,
    verts: &[u32],
    cells: &[[i32; 3]],
    tris: Vec<u8>,
    flags: u8,
    uv: &[ClusterUvs],
) -> ClusterData {
    let anchor = [0, 1, 2].map(|a| cells.iter().map(|g| g[a]).min().unwrap());
    let local: Vec<[u16; 3]> = cells
        .iter()
        .map(|g| [0, 1, 2].map(|a| u16::try_from(g[a] - anchor[a]).expect("cluster spans 16 bits")))
        .collect();
    let local_max = [0, 1, 2].map(|a| local.iter().map(|l| l[a]).max().unwrap());
    let step = grid.step();
    let mut floats = [[0.0f32; 3]; MAX_CLUSTER_VERTICES];
    for (f, g) in floats.iter_mut().zip(cells) {
        *f = g.map(|v| (f64::from(v) * step) as f32);
    }
    let floats = &floats[..cells.len()];
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(floats), 12, 0)
        .expect("positions are 12-byte vertices");
    let bounds = meshopt::compute_meshlet_bounds(
        meshopt::Meshlet {
            vertices: &IDENTITY[..verts.len()],
            triangles: &tris,
        },
        &adapter,
    );

    let mut attrs = Vec::with_capacity(verts.len() * 3);
    for &v in verts {
        let v = v as usize;
        attrs.push(oct_encode(
            Vec3::from(w.normals[v]).normalize_or(Vec3::Z).to_array(),
        ));
        if let Some(t) = &w.tangents {
            let [x, y, z, s] = t[v];
            let e = oct_encode(Vec3::new(x, y, z).normalize_or(Vec3::X).to_array());
            attrs.push(e & !(1 << 16) | u32::from(s < 0.0) << 16);
        }
        for (u, r) in w.uvs.iter().zip(uv) {
            let q = r.quantize(u[v]);
            attrs.push(q[0] | q[1] << 16);
        }
        if let Some(c) = &w.colors {
            let h = c[v].map(|x| u32::from(half::f16::from_f32(x).to_bits()));
            attrs.push(h[0] | h[1] << 16);
            attrs.push(h[2] | h[3] << 16);
        }
        if let Some(k) = &w.skin {
            attrs.push(u32::from_le_bytes(k.joints[v]));
            attrs.push(u32::from_le_bytes(k.weights[v]));
        }
    }

    let area: f64 = tris
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let p = t.map(|i| Vec3::from(floats[usize::from(i)]).as_dvec3());
            0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
        })
        .sum();
    let tri_count = (tris.len() / 3) as u8;
    let sphere = Sphere {
        center: bounds.center,
        radius: bounds.radius,
    };
    let record = Cluster {
        self_sphere: sphere,
        parent_sphere: sphere,
        cull_sphere: sphere,
        cone: [
            bounds.cone_axis_s8[0],
            bounds.cone_axis_s8[1],
            bounds.cone_axis_s8[2],
            bounds.cone_cutoff_s8,
        ],
        anchor,
        vertex_count: verts.len() as u16,
        tri_count,
        alpha_first: tri_count,
        flags: flags & (Section::CLASS_MASK | Section::SOLID),
        local_max,
        uv_offset: [0, 1].map(|s| uv.get(s).map_or([0.0; 2], |u| u.offset)),
        uv_scale: [0, 1].map(|s| uv.get(s).map_or([0.0; 2], |u| u.scale)),
        omm_first: Cluster::NO_OMM,
        area: area as f32,
        ..bytemuck::Zeroable::zeroed()
    };
    ClusterData {
        record,
        local,
        attrs,
        tris,
        uvs: Vec::new(),
        omm: None,
        bounds: None,
    }
}

/// Packs clusters in order into resident pages of at most [`PAGE_MAX`] bytes; returns each
/// page and its cluster count.
fn paginate(clusters: &[ClusterData], stride: usize) -> Vec<(Vec<u8>, usize, u32)> {
    let mut pages = Vec::new();
    let mut first = 0;
    while first < clusters.len() {
        let mut size = size_of::<PageHeader>();
        let mut end = first;
        let mut aligned = false;
        while end < clusters.len() {
            let c = &clusters[end];
            // Room to align the micromap region, once per page.
            let align = c.omm.is_some() && !aligned;
            let add = c.cost(stride) + if align { OMM_ALIGN as usize - 1 } else { 0 };
            if size + add > PAGE_MAX {
                break;
            }
            size += add;
            aligned |= align;
            end += 1;
        }
        let page: Vec<&ClusterData> = clusters[first..end].iter().collect();
        pages.push((
            write_page(&page, stride),
            end - first,
            page_triangles(&page),
        ));
        first = end;
    }
    pages
}

/// The triangles of a page's clusters.
fn page_triangles(clusters: &[&ClusterData]) -> u32 {
    clusters.iter().map(|c| u32::from(c.record.tri_count)).sum()
}

fn write_page(clusters: &[&ClusterData], stride: usize) -> Vec<u8> {
    let slots: usize = clusters.iter().map(|c| c.slots()).sum();
    let tris_len: usize = clusters
        .iter()
        .map(|c| c.tris.len().next_multiple_of(4))
        .sum();
    let positions = size_of::<PageHeader>() + clusters.len() * size_of::<Cluster>();
    let attrs = positions + slots * 6;
    let tris = attrs + slots * stride;
    let micromaps: Vec<Option<&ClusterOmm>> = clusters.iter().map(|c| c.omm.as_ref()).collect();
    let region = omm::page_region(&micromaps);
    let omm = if region.bytes.is_empty() {
        tris + tris_len
    } else {
        (tris + tris_len).next_multiple_of(OMM_ALIGN as usize)
    };
    let len = omm + region.bytes.len();
    let header = PageHeader {
        cluster_count: clusters.len() as u32,
        attr_stride: stride as u32,
        positions_offset: positions as u32,
        attrs_offset: attrs as u32,
        tris_offset: tris as u32,
        omm_offset: omm as u32,
        len: len as u32,
        omm_usage: region.usage,
        ..bytemuck::Zeroable::zeroed()
    };
    let mut page = vec![0u8; len];
    page[..size_of::<PageHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    page[omm..].copy_from_slice(&region.bytes);
    let (mut vertex_first, mut tri_first) = (0, 0);
    for (i, c) in clusters.iter().enumerate() {
        let mut r = c.record;
        r.vertex_first = vertex_first as u32;
        r.tri_first = tri_first as u32;
        r.omm_first = region.omm_first[i].unwrap_or(Cluster::NO_OMM);
        let at = size_of::<PageHeader>() + i * size_of::<Cluster>();
        page[at..at + size_of::<Cluster>()].copy_from_slice(bytemuck::bytes_of(&r));
        let p = positions + vertex_first * 6;
        page[p..p + c.local.len() * 6].copy_from_slice(bytemuck::cast_slice(&c.local));
        let a = attrs + vertex_first * stride;
        page[a..a + c.attrs.len() * 4].copy_from_slice(bytemuck::cast_slice(&c.attrs));
        let t = tris + tri_first;
        page[t..t + c.tris.len()].copy_from_slice(&c.tris);
        vertex_first += c.slots();
        tri_first += c.tris.len().next_multiple_of(4);
    }
    page
}

impl MeshData {
    /// Gives what a section reads of a uv set the mesh lacks the reading the shaders give it, a
    /// uv of zero. Returns a warning per such section.
    pub(crate) fn pad_uv_sets(&mut self) -> Vec<String> {
        if self.uvs.len() >= 2 {
            return Vec::new();
        }
        let sections = self.sections.iter().enumerate();
        let warnings: Vec<String> = sections
            .filter(|(_, s)| s.maps.uv1)
            .map(|(i, _)| {
                format!("section {i} reads uv set 1, which the mesh lacks; it reads uv 0")
            })
            .collect();
        if !warnings.is_empty() {
            self.uvs.push(vec![[0.0; 2]; self.positions.len()]);
        }
        warnings
    }

    /// Gives a mesh without tangents MikkTSpace-compatible ones when a section reads them,
    /// section by section. A section reading none still gets them, from uv set 0: instances
    /// can bind it a material that reads them. The mesh must pass [`check`].
    pub(crate) fn compute_tangents(&mut self) {
        let reads = |s: &SectionData| s.maps.normal.tangent_set().is_some();
        if self.tangents.is_some() || !self.sections.iter().any(reads) {
            return;
        }
        // One corner holds one tangent, so sections sharing corners get their own.
        let mut taken: Vec<Range<usize>> = Vec::with_capacity(self.sections.len());
        for s in &mut self.sections {
            if taken
                .iter()
                .any(|r| r.start < s.indices.end && s.indices.start < r.end)
            {
                let start = self.indices.len();
                self.indices.extend_from_within(s.indices.clone());
                s.indices = start..self.indices.len();
            }
            taken.push(s.indices.clone());
        }
        let zeros = vec![[0.0; 2]; self.positions.len()];
        let mut corners = vec![tangents::UNREAD; self.indices.len()];
        for s in &self.sections {
            let set = s.maps.normal.tangent_set().unwrap_or(0);
            // A frame on uvs outside the mesh's sets falls back to set 0, as the shaders do
            // without a tangent stream.
            let other = set == usize::from(Material::TANGENT_SET_OTHER);
            let uvs = self.uvs.get(if other { 0 } else { set }).unwrap_or(&zeros);
            let section = &self.indices[s.indices.clone()];
            let at = |i: &u32| *i as usize;
            let positions: Vec<_> = section.iter().map(|i| self.positions[at(i)]).collect();
            let normal = |i| {
                Vec3::from(self.normals[at(i)])
                    .normalize_or_zero()
                    .to_array()
            };
            let normals: Vec<_> = section.iter().map(normal).collect();
            let uvs: Vec<_> = section.iter().map(|i| uvs[at(i)]).collect();
            corners[s.indices.clone()]
                .copy_from_slice(&tangents::generate(None, &positions, &normals, &uvs));
        }
        let (tangents, copies) =
            tangents::per_vertex(self.positions.len(), &mut self.indices, &corners);
        self.tangents = Some(tangents);
        fn copy<T: Copy>(v: &mut Vec<T>, copies: &[u32]) {
            *v = tangents::with_copies(std::mem::take(v), copies);
        }
        copy(&mut self.positions, &copies);
        copy(&mut self.normals, &copies);
        for u in &mut self.uvs {
            copy(u, &copies);
        }
        if let Some(c) = &mut self.colors {
            copy(c, &copies);
        }
        if let Some(c) = &mut self.moves {
            copy(c, &copies);
        }
        if let Some(k) = &mut self.skin {
            copy(&mut k.joints, &copies);
            copy(&mut k.weights, &copies);
        }
        if let Some(p) = &mut self.parts {
            copy(&mut p.values, &copies);
            if let Pivots::PerVertex(v) = &mut p.pivots {
                copy(v, &copies);
            }
        }
    }

    /// Hash of everything the mesh cook reads but its settings and the cooked textures,
    /// which enter through their keys.
    fn hash(&self) -> Hash32 {
        let mut h = blake3::Hasher::new();
        let mut part = |b: &[u8]| {
            h.update(&(b.len() as u64).to_le_bytes());
            h.update(b);
        };
        part(bytemuck::cast_slice(&self.positions));
        part(bytemuck::cast_slice(&self.normals));
        part(self.tangents.as_deref().map_or(&[], bytemuck::cast_slice));
        part(self.colors.as_deref().map_or(&[], bytemuck::cast_slice));
        part(self.moves.as_deref().map_or(&[], bytemuck::cast_slice));
        for u in &self.uvs {
            part(bytemuck::cast_slice(u));
        }
        part(bytemuck::cast_slice(&self.indices));
        if let Some(k) = &self.skin {
            part(bytemuck::cast_slice(&k.joints));
            part(bytemuck::cast_slice(&k.weights));
        }
        if let Some(p) = &self.parts {
            part(bytemuck::cast_slice(&p.values));
            part(&p.count.to_le_bytes());
            part(&[p.by_section.into()]);
            match &p.pivots {
                Pivots::PerValue(v) => {
                    part(b"value");
                    part(bytemuck::cast_slice(v));
                }
                Pivots::PerVertex(v) => {
                    part(b"vertex");
                    part(bytemuck::cast_slice(v));
                }
                Pivots::Origin => part(b"origin"),
            }
            part(p.joints.as_deref().map_or(&[], bytemuck::cast_slice));
        }
        part(format!("{:?}", self.min_exponent).as_bytes());
        sections_hash(Hash32(*h.finalize().as_bytes()), &self.sections)
    }

    pub(crate) fn triangles(&self) -> u64 {
        self.indices.len() as u64 / 3
    }
}

#[cfg(test)]
mod tests {
    use super::stored_corner_order;

    /// The port of the meshlet encoder's rotations matches the codec: random clusters and
    /// strips that reuse edges, after an encode and decode.
    #[test]
    fn stored_corner_order_matches_the_meshlet_codec() {
        let mut state = 0x9e37_79b9_u32;
        let mut random = |n: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % n
        };
        for case in 0..400 {
            let n = 1 + random(128) as usize;
            let verts = 4 + random(252);
            let mut tris = Vec::with_capacity(3 * n);
            for k in 0..n {
                let t = if case % 2 == 0 || k == 0 {
                    let a = random(verts);
                    let b = (a + 1 + random(verts - 1)) % verts;
                    let mut c = random(verts);
                    while c == a || c == b {
                        c = random(verts);
                    }
                    [a, b, c]
                } else {
                    // A neighbor of the previous triangle across one of its edges.
                    let p: [u32; 3] =
                        [tris[3 * k - 3], tris[3 * k - 2], tris[3 * k - 1]].map(u32::from);
                    let e = random(3) as usize;
                    let mut c = random(verts);
                    while p.contains(&c) {
                        c = random(verts);
                    }
                    [p[(e + 1) % 3], p[e], c]
                };
                tris.extend(t.map(|v| v as u8));
            }
            let stored = meshopt::encode_meshlet(meshopt::Meshlet {
                vertices: &[],
                triangles: &tris,
            })
            .unwrap();
            let (_, decoded) = meshopt::decode_meshlet(&stored, 0, n).unwrap();
            stored_corner_order(&mut tris);
            assert_eq!(tris, decoded[..3 * n], "case {case}");
        }
    }
}
