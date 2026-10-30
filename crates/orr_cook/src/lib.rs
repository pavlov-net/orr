//! Asset cook: imports source assets and writes orr archives.
//!
//! [`cook`] runs a [`Manifest`]: each [`ArchiveSpec`] becomes one archive, written to a file
//! or returned in memory.
//!
//! A glTF source emits, under the asset name `N`:
//! - `N`: a model binding the mesh to one material per section,
//! - `N/mesh`: every triangle primitive of the default scene with node transforms applied,
//!   one section per glTF material,
//! - `N/material/<i>`: glTF material `i`, or `N/material/default` for primitives without one,
//! - `N/texture/<material>/<role>`: the material's textures (roles `base_color`, `opacity`,
//!   `normal`, `roughness`, `metalness`, `roughness_metalness`, `occlusion`, `emission`);
//!   materials that would cook the same texture share the first one's entry.
//!
//! A texture source emits one texture `N`.
//!
//! A raster source emits one raster set `N`: a tiled LOD pyramid per layer, built by the
//! cook from a whole raster or imported from producer tiles ([`RasterSet`]).
//!
//! An instances source emits instance pages `N/<x>_<y>_<z>/<i>` per world cell
//! ([`Instances`]); it can place models cooked by any archive of the manifest. A user source
//! emits one entry `N` of a consumer class.
//!
//! The cook is incremental. Every entry carries a cook key: a hash of its class's cook
//! version, the libraries that shape its bytes, its name, its settings, the content of its
//! inputs and the keys of the entries it depends on. An entry whose key matches an entry of
//! the archive being replaced, or of the cache ([`CookOptions::cache`]), is copied from there
//! without cooking; an archive whose entries all match is left as it is. A file archive
//! keeps `<archive>.cookstate` beside it: the hash of each input file by size and time, and
//! the files and keys of each source, so a source whose files did not change is not even
//! imported. Entries are written in a fixed order (sorted by name, instance pages last in
//! Morton order of their cells), so the same inputs give the same bytes whatever the thread
//! count, the target and the mix of cooked and linked entries.

mod alloc;
mod cache;
mod gltf;
mod instances;
mod key;
mod manifest;
mod mesh;
mod raster;
mod report;
mod run;
mod state;
mod texture;

use std::path::PathBuf;
use std::sync::Arc;

use orr_pack::{BlobIn, Target};
use serde::Deserialize;

pub use instances::{Instances, Placement, PrototypeSpec, SectionSpec};
pub use mesh::MeshSettings;
pub use raster::{
    RasterCompression, RasterFormat, RasterInput, RasterLayerReport, RasterLayerSpec, RasterReport,
    RasterSet, Reduction,
};
pub use report::{
    ArchiveReport, AssetReport, CookReport, EntryReport, MeshReport, Outcome, TextureReport,
};
pub use texture::material::{TextureOverride, TextureRules};
pub use texture::{DataFormat, MipFilter, Primaries, Quality, TextureSettings, Usage, Wrap};

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
    #[serde(rename = "asset", default)]
    pub assets: Vec<AssetSpec>,
}

impl ArchiveSpec {
    pub const DEFAULT_CELL_LOG2: u8 = 8;

    pub fn new(output: Target, assets: Vec<AssetSpec>) -> Self {
        Self {
            output,
            cell_log2: Self::DEFAULT_CELL_LOG2,
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
    /// An image file: PNG, JPEG, Radiance HDR, OpenEXR, KTX2 or DDS.
    Texture {
        path: PathBuf,
        #[serde(default)]
        settings: TextureSettings,
    },
    /// Placements of models cooked by the same manifest.
    Instances(Instances),
    /// Raster layers tiled into one raster set.
    Raster(RasterSet),
    /// Opaque bytes of a consumer class (at least [`Class::USER_FIRST`]), stored as one
    /// payload blob. `version` changes the cook key only.
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
    pub mesh: MeshSettings,
    pub textures: TextureRules,
}

/// How [`cook`] runs.
#[derive(Clone, Debug)]
pub struct CookOptions {
    /// Threads of the cook's own pool; 0 uses one per core. The cook never uses the global
    /// rayon pool.
    pub jobs: usize,
    /// Cook every entry, ignoring the cache, the archive being replaced and the cookstate.
    pub force: bool,
    /// Where to write the [`CookReport`] as JSON.
    pub report: Option<PathBuf>,
    /// Directory of the content-addressed cache; `None` for no cache, e.g. in-process cooks.
    pub cache: Option<PathBuf>,
    /// After the cook, least recently used cache archives are removed down to this size.
    pub cache_cap: Option<u64>,
    /// Bytes the running jobs may hold by their estimates; a job over it runs alone.
    pub working_set: u64,
    /// Bytes of finished entries that may wait for their turn to be written; past it only
    /// the jobs the next entries wait on start.
    pub reorder_window: u64,
}

impl Default for CookOptions {
    fn default() -> Self {
        Self {
            jobs: 0,
            force: false,
            report: None,
            cache: None,
            cache_cap: None,
            working_set: alloc::physical_memory() / 2,
            reorder_window: 1 << 30,
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
    #[error("{asset}: class {class:#06x} is not a user class")]
    UserClass { asset: String, class: u16 },
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

/// The cook version of a class. It changes whenever the cook writes different bytes for the
/// same inputs, and it enters every cook key of the class.
pub fn cook_version(class: u16) -> u32 {
    key::version(class)
}
