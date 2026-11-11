//! Asset cook: imports source assets and writes orr archives.
//!
//! [`cook`] runs a [`Manifest`]: each [`ArchiveSpec`] becomes one archive, written to a file
//! or returned in memory.
//!
//! A glTF source emits, under the asset name `N`:
//! - `N`: a model binding the mesh to one material per section,
//! - `N/mesh`: every triangle primitive of the default scene, one section per glTF material:
//!   with node transforms applied, or with skins in bind space with joints and weights,
//! - with [`GltfSettings::models`], instead of those two, `N/<node>` and `N/<node>/mesh` per
//!   selected node, in the node's space,
//! - with [`GltfLayout::Meshes`], instead of all three, `N/mesh/<i>` and `N/mesh/<i>/mesh`
//!   per glTF mesh the scene places, in the mesh's space, placed by instance pages
//!   `N/instances/...`,
//! - `N/<model>/baked/<k>` and its mesh for the copies of a model whose transform collapses
//!   an axis, with the transform applied,
//! - `N/<rig>` and `N/<rig>/mesh` for each set of meshes that moves together, in scene space,
//!   named after its first joint and out of the models above: a skin's meshes, and the meshes
//!   under a node an animation targets, the latter as rigid parts, one per joint with part 0 on
//!   a still joint; a scene that is all one such set keeps the names `N` and `N/mesh`. Its mesh
//!   and palettes stand in its skeleton's space, at the skeleton's origin in scene space
//!   (`SkeletonInfo::origin`), which a copy of it takes and its pages hold already,
//! - `N/<model>/skeleton` for a model that moves: the joints of its skins, the nodes animated
//!   above its meshes and the nodes between them,
//! - `N/<model>/clip/<animation>`: each animation that moves the skeleton, resampled at
//!   [`ClipSettings`], running over the whole animation so its clips keep time; an animation
//!   without a unique name that is not all digits is named by its glTF animation index,
//! - `N/material/<i>`: glTF material `i`, or `N/material/default` for primitives without one,
//! - `N/texture/<material>/<role>`: the material's textures (roles `base_color`, `opacity`,
//!   `normal`, `roughness`, `metalness`, `roughness_metalness`, `emission`,
//!   `subsurface_color`, `specular_weight`, `coat`, `transmission`, `displacement`);
//!   materials that would cook the same texture share the first one's entry.
//!
//! A USD source emits the same classes under the same names, with these differences
//! ([`UsdSettings`]): the whole-stage model holds the meshes no other model holds; each
//! `SkelRoot` is a model `N/<root>` of its skinned meshes with skeleton `N/<root>/skeleton`
//! and clips `N/<root>/clip/<animation>`; prims whose xforms are sampled over time move what
//! is under them as glTF's animated nodes do, by a clip `<model>/clip/xform`; every clip and
//! animated page of a stage runs on one timeline, so they keep time; each `PointInstancer` prototype is a model
//! `N/<prototype>` placed by instance pages `N/instances/...`; materials are
//! `N/material/<prim path>` (`<prim path>/double_sided` on double-sided meshes,
//! `<prim path>/primvars/<k>` on meshes that differ in what its primvar readers fetch), their
//! textures `N/texture/<prim path>/<role>`. A MaterialX source emits one material `N` and its
//! textures `N/texture/<role>`.
//!
//! A mesh source emits the model `N`, its mesh `N/mesh` and its materials
//! `N/material/<section>`. A subdivision cage source emits the model `N`, its cage `N/subd`
//! and its materials `N/material/<section>`; with [`Subdivision::Cage`], a USD model holding
//! cages has its cage `<model>/subd` beside its mesh `<model>/mesh`, its material table
//! listing the mesh's sections, then the cage's. A material source emits one material `N`.
//! A texture source emits one texture `N`.
//!
//! With [`MeshSettings::assemblies`], a model `M` whose mesh repeats a rigid piece emits
//! each repeated unit once as model `M/unit/<i>` with mesh `M/unit/<i>/mesh`, and `M/mesh`
//! keeps the rest. Instance sources that place `M` also place every unit copy, so `M` is
//! whole only where an instance source places it.
//!
//! A raster source emits one raster set `N`: a tiled LOD pyramid per layer, built by the
//! cook from a whole raster or imported from producer tiles ([`RasterSet`]).
//!
//! A volume source emits one volume `N`: its scalar grids as sparse bricks with a majorant
//! grid ([`VolumeInput`]).
//!
//! An instances source emits instance pages `N/<x>_<y>_<z>/<i>` per world cell
//! ([`Instances`]); it can place models cooked by any archive of the manifest. A user source
//! emits one entry `N` of a consumer class.
//!
//! An archive with [`ArchiveSpec::gdeflate`] stores its payload blobs as GDeflate pages of
//! 64 KiB, which the GPU decompresses, instead of zstd; mount blobs stay zstd, and a blob
//! that does not compress is stored raw. The codec is NVIDIA's libdeflate fork, built with
//! the `gdeflate` feature (module `gdeflate`).
//!
//! [`CookOptions::quality`] sets how much work the cook spends, on ctt's preset scale. The
//! level is the preset of every BCn encode, sets the zstd level of every blob ([`zstd_level`];
//! [`CookOptions::zstd_level`] overrides it), and decides what happens to textures whose
//! source images hold every mip in a block format:
//!
//! | Quality | zstd | Mips of block-compressed sources with every mip |
//! | --- | --- | --- |
//! | `UltraFast`, `VeryFast`, `Fast` (the default) | 1 | kept |
//! | `Basic` | 6 | kept |
//! | `Slow` | 19 | made from mip 0 |
//! | `VerySlow` | 22 | made from mip 0 |
//!
//! Kept mips are the source's own: its blocks pass through when the texture has the
//! source's format, and are decoded and encoded again when it does not. They get no
//! alpha-coverage scaling, so alpha-tested surfaces can thin out in the distance, and no
//! Toksvig roughness, so distant specular highlights can shimmer; a normal map that keeps its
//! mips adds Toksvig roughness to no texture, so a constant roughness stays a factor.
//! Textures whose sources lack mips or are not block-compressed make their mips the same way
//! at every level. Mesh processing does not depend on the level.
//!
//! The cook is incremental. Every entry carries a cook key: a hash of its class's cook
//! version, the libraries that shape its bytes, its name, its settings, the quality level
//! where it shapes the entry, the content of its inputs and the keys of the entries it
//! depends on. An entry whose key matches an entry of
//! the archive being replaced, or of the cache ([`CookOptions::cache`]), is copied from there
//! without cooking; an archive whose entries all match is left as it is. A file archive
//! keeps `<archive>.cookstate` beside it: the hash of each input file by size and time, and
//! the files and keys of each source, so a source whose files did not change is not even
//! imported. Entries are written in a fixed order: by unit, instance pages last, then the
//! largest planned working set first, then by name, a unit's instance pages in Morton order
//! of their cells. So the same inputs give the same bytes whatever the thread count, the
//! target and the mix of cooked and linked entries, and the payload follows the order jobs
//! start in. The index and the report list entries by id and by name.

mod alloc;
mod assembly;
mod cache;
mod decoded;
#[cfg(feature = "gdeflate")]
pub mod gdeflate;
mod gltf;
mod import;
mod instances;
mod job;
mod key;
mod manifest;
mod materialx;
mod mesh;
mod par;
mod patch;
mod profile;
mod quality;
mod raster;
mod report;
mod run;
mod shading;
mod skin;
mod state;
mod subd;
mod texture;
mod usd;
mod volume;

use std::path::PathBuf;
use std::sync::Arc;

use orr_pack::{BlobIn, Target};
use serde::Deserialize;

pub use alloc::Metered;
pub use assembly::AssemblySettings;
pub use ctt::Quality;
pub use gltf::GltfLayout;
pub use import::{NodeSelect, PartAttribute};
pub use instances::{
    InstanceMotion, Instances, Linear, PageFrame, Placement, PrototypeSpec, SectionSpec,
};
pub use mesh::{LodMode, MeshInput, MeshParts, MeshSettings, OmmSettings, StaticDisplacement};
pub use patch::{MaterialPatch, MeshPatch};
pub use quality::zstd_level;
pub use raster::{
    RasterCompression, RasterFormat, RasterInput, RasterLayerReport, RasterLayerSpec, RasterReport,
    RasterSet, Reduction,
};
pub use report::{
    ArchiveReport, AssemblyReport, AssemblyUnitReport, AssetReport, CookReport, EntryReport,
    LevelReport, MeshReport, NeuralReport, NeuralTextureReport, Outcome, TangentReport,
    TextureReport,
};
pub use skin::ClipSettings;
pub use subd::{
    Boundary, CageColor, CageInput, CageUv, CageValues, Corner, Crease, FaceVarying, SubdSettings,
    UvLinear,
};
pub use texture::material::{InputChannels, TextureOverride, TextureRules};
pub use texture::neural::NeuralSettings;
pub use texture::{DataFormat, MipFilter, Primaries, TextureSettings, Usage, Wrap};
pub use usd::{Subdivision, UsdSettings, VariantSelection};
pub use volume::{
    Block, GridInput, GridVoxels, VolumeFormat, VolumeGridReport, VolumeGridSpec, VolumeGrids,
    VolumeInput, VolumeReport, VolumeSettings,
};

/// Archives to cook. [`Manifest::load`] reads one from TOML.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    #[serde(rename = "archive", default)]
    pub archives: Vec<ArchiveSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveSpec {
    /// In TOML, a file path.
    #[serde(deserialize_with = "manifest::file_target")]
    pub output: Target,
    /// World cells of instance pages have an edge of `2^cell_log2` m; the renderer's cell
    /// size must match. [`Self::DEFAULT_CELL_LOG2`] is 256 m.
    #[serde(default = "manifest::default_cell_log2")]
    pub cell_log2: u8,
    /// Payload blobs as GDeflate pages, decompressed on the GPU, instead of zstd. Needs the
    /// `gdeflate` feature.
    #[serde(default)]
    pub gdeflate: bool,
    #[serde(rename = "asset", default)]
    pub assets: Vec<AssetSpec>,
}

impl ArchiveSpec {
    pub const DEFAULT_CELL_LOG2: u8 = 8;

    pub fn new(output: Target, assets: Vec<AssetSpec>) -> Self {
        Self {
            output,
            cell_log2: Self::DEFAULT_CELL_LOG2,
            gdeflate: false,
            assets,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AssetSpec {
    /// Logical name; asset ids derive from it.
    pub name: String,
    #[serde(flatten)]
    pub source: Source,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// A glTF 2.0 file, `.gltf` or `.glb`.
    Gltf {
        path: PathBuf,
        #[serde(default)]
        settings: GltfSettings,
    },
    /// A USD stage: `.usda`, `.usdc` or `.usdz`.
    Usd {
        path: PathBuf,
        #[serde(default)]
        settings: UsdSettings,
    },
    /// One `surfacematerial` of a MaterialX document (`.mtlx`).
    #[serde(rename = "materialx")]
    MaterialX {
        path: PathBuf,
        /// The `surfacematerial` element's name.
        material: String,
        #[serde(default)]
        textures: TextureRules,
    },
    /// An image file: PNG, JPEG, WebP, Radiance HDR, OpenEXR, KTX2 or DDS. Of an animated WebP
    /// the first frame cooks, with a warning in the report.
    Texture {
        path: PathBuf,
        #[serde(default)]
        settings: TextureSettings,
    },
    /// Placements of models cooked by the same manifest.
    Instances(Instances),
    /// Raster layers tiled into one raster set.
    Raster(RasterSet),
    /// Scalar grids on one lattice.
    Volume {
        input: VolumeInput,
        #[serde(default)]
        settings: VolumeSettings,
    },
    /// A mesh built in memory, one material per section, cooked as `<name>/material/<i>`;
    /// library callers only. Its materials bind texture units of its archive, and a material's
    /// `back` names a material unit of any source or another such mesh's material.
    #[serde(skip)]
    Mesh {
        mesh: MeshInput,
        materials: Vec<orr_pack::material::Material>,
        settings: MeshSettings,
    },
    /// A Catmull-Clark cage built in memory, one material per section; library callers
    /// only. Needs the `subd` feature.
    #[serde(skip)]
    Subd {
        cage: CageInput,
        materials: Vec<orr_pack::material::Material>,
        settings: SubdSettings,
    },
    /// A material: the OpenPBR defaults with these fields set.
    Material(MaterialPatch),
    /// Opaque bytes of a consumer class (at least [`orr_pack::Class::USER_FIRST`]), stored as
    /// one payload blob. `version` changes the cook key only.
    User {
        class: u16,
        schema: u16,
        version: u32,
        bytes: Vec<u8>,
    },
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GltfSettings {
    /// Reads the file as glTF 2 whatever its `asset` says, for a file that puts its own
    /// version there.
    pub ignore_version: bool,
    pub layout: GltfLayout,
    /// Nodes that become models; empty makes the whole scene one model.
    pub models: Vec<NodeSelect>,
    pub mesh: MeshSettings,
    /// Applied in order to the mesh of every model whose node name contains the pattern.
    /// The whole-scene model has an empty name, so only an empty pattern reaches it.
    pub mesh_overrides: Vec<(String, MeshPatch)>,
    pub textures: TextureRules,
    /// Applied in order to every material whose glTF name contains the pattern, before its
    /// textures are cooked.
    pub material_overrides: Vec<(String, MaterialPatch)>,
    /// Rigid parts from a per-vertex attribute.
    pub parts_from_attribute: Option<PartAttribute>,
    /// Skinned meshes cook as rigid parts instead: every vertex follows one joint (one
    /// non-zero weight), its part is that skeleton joint, pivoted at the joint's bind
    /// position, and the mesh keeps no skin attributes. The skeleton still cooks, so its
    /// parents and rest transforms give the parts' hierarchy; skeleton joint 0 is a still joint
    /// of its own, since the renderer does not pose part 0. Without skins it does nothing. Does
    /// not mix with `parts_from_attribute`.
    pub parts_from_skin: bool,
    /// Patterns of selected models that join crack-free: they share one grid step and
    /// their nodes must keep them on it.
    pub joined: Vec<String>,
    pub clips: ClipSettings,
}

/// How [`cook`] runs.
#[derive(Clone, Debug)]
pub struct CookOptions {
    /// Threads of the rayon pool, which the cook builds as the global pool unless the
    /// process already has one; 0 uses one per core. Half as many cook jobs run at once, at
    /// least one.
    pub jobs: usize,
    /// Cook every entry, ignoring the cache, the archive being replaced and the cookstate.
    pub force: bool,
    /// Where to write the [`CookReport`] as JSON.
    pub report: Option<PathBuf>,
    /// Directory of the content-addressed cache; `None` for no cache, e.g. in-process cooks.
    pub cache: Option<PathBuf>,
    /// After the cook, least recently used cache archives are removed down to this size.
    pub cache_cap: Option<u64>,
    /// Bytes the running jobs may hold by their estimates, with the decoded images and maps
    /// they share, whose four caches take up to a quarter of it; a job over it runs alone.
    pub working_set: u64,
    /// Bytes of finished entries that may wait for their turn to be written; past it only
    /// the jobs the next entries wait on start.
    pub reorder_window: u64,
    /// How much work the cook spends; the crate documentation lists what each level does.
    pub quality: Quality,
    /// zstd level of every cooked blob, up to 22; `None` takes the quality's level.
    pub zstd_level: Option<i32>,
}

impl Default for CookOptions {
    fn default() -> Self {
        Self {
            jobs: 0,
            force: false,
            report: None,
            cache: None,
            cache_cap: None,
            working_set: orr_io::physical_memory() / 2,
            reorder_window: 1 << 30,
            quality: Quality::Fast,
            zstd_level: None,
        }
    }
}

/// Result of [`cook`].
#[derive(Debug, Default)]
pub struct Cooked {
    pub report: CookReport,
    /// One archive per [`Target::Memory`] output, in manifest order.
    pub memory: Vec<Arc<[u8]>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Pack(#[from] orr_pack::Error),
    #[error("{path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The manifest cannot be read.
    #[error("{path}: {message}")]
    Manifest { path: PathBuf, message: String },
    /// An input file changed while the cook ran.
    #[error("{path} changed during the cook")]
    Changed { path: PathBuf },
    /// The source cannot be imported.
    #[error("{asset}: {message}")]
    Import { asset: String, message: String },
    /// The mesh does not fit the cooked format or the settings.
    #[error("{asset}: {message}")]
    Mesh { asset: String, message: String },
    /// An image cannot be read or cooked with its settings.
    #[error("{asset}: {message}")]
    Texture { asset: String, message: String },
    /// A prototype names a model the manifest does not cook.
    #[error("{asset}: model {model} is not cooked by the manifest")]
    UnknownModel { asset: String, model: String },
    /// A section override names a material the manifest does not cook.
    #[error("{asset}: material {material} is not cooked by the manifest")]
    UnknownMaterial { asset: String, material: String },
    #[error("{asset}: prototype {prototype} has pose variant {pose}; variants are below 256")]
    PoseVariant {
        asset: String,
        prototype: u16,
        pose: u16,
    },
    #[error("{asset}: prototype {prototype} hides section {section}; only sections 0..15 hide")]
    HiddenSection {
        asset: String,
        prototype: u16,
        section: usize,
    },
    /// Placements that do not fit an instance page.
    #[error("{asset}: {message}")]
    Instances { asset: String, message: String },
    /// A raster set whose inputs or settings do not fit the format.
    #[error("{asset}: {message}")]
    Raster { asset: String, message: String },
    /// A volume whose input or settings do not fit the format.
    #[error("{asset}: {message}")]
    Volume { asset: String, message: String },
    #[error("{asset}: class {class:#06x} is not a user class")]
    UserClass { asset: String, class: u16 },
    /// An archive asks for GDeflate in a cook built without the `gdeflate` feature.
    #[error("archive {archive} asks for GDeflate; orr_cook is built without the gdeflate feature")]
    NoGDeflate { archive: usize },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Cooks every archive of the manifest on the cook's own thread pool.
pub fn cook(manifest: &Manifest, opts: &CookOptions) -> Result<Cooked> {
    run::cook(manifest, opts)
}

fn mount(bytes: &[u8]) -> BlobIn<'_> {
    BlobIn {
        mount: true,
        ..BlobIn::new(bytes)
    }
}
