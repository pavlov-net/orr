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
//! An instances source emits instance pages `N/<x>_<y>_<z>/<i>` per world cell
//! ([`Instances`]); it can place models cooked by any archive of the manifest, and is cooked
//! after every other source. A user source emits one entry `N` of a consumer class.

mod gltf;
mod instances;
mod mesh;
mod texture;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use orr_pack::material::Material;
use orr_pack::model::{ModelHeader, ModelInfo};
use orr_pack::{ArchiveWriter, AssetId, BlobIn, Class, Hash32, Span, Target};

pub use instances::{Instances, Placement, PrototypeSpec, SectionSpec};
pub use mesh::MeshSettings;
pub use texture::material::{TextureOverride, TextureRules};
pub use texture::{DataFormat, MipFilter, Primaries, Quality, TextureSettings, Usage, Wrap};

/// Archives to cook.
#[derive(Clone, Debug, Default)]
pub struct Manifest {
    pub archives: Vec<ArchiveSpec>,
}

#[derive(Clone, Debug)]
pub struct ArchiveSpec {
    pub output: Target,
    /// World cells of instance pages have an edge of `2^cell_log2` m; the renderer's cell
    /// size must match. [`Self::DEFAULT_CELL_LOG2`] is 256 m.
    pub cell_log2: u8,
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

#[derive(Clone, Debug)]
pub struct AssetSpec {
    /// Logical name; asset ids derive from it.
    pub name: String,
    pub source: Source,
}

#[derive(Clone, Debug)]
pub enum Source {
    /// A glTF 2.0 file, `.gltf` or `.glb`.
    Gltf {
        path: PathBuf,
        settings: GltfSettings,
    },
    /// An image file: PNG, JPEG, Radiance HDR, OpenEXR, KTX2 or DDS.
    Texture {
        path: PathBuf,
        settings: TextureSettings,
    },
    /// Placements of models cooked by the same manifest.
    Instances(Instances),
    /// Opaque bytes of a consumer class (at least [`Class::USER_FIRST`]), stored as one
    /// payload blob. `version` changes the cook key only.
    User {
        class: u16,
        schema: u16,
        version: u32,
        bytes: Vec<u8>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct GltfSettings {
    pub mesh: MeshSettings,
    pub textures: TextureRules,
}

/// Result of [`cook`].
#[derive(Debug, Default)]
pub struct Cooked {
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
    #[error("{asset}: class {class:#06x} is not a user class")]
    UserClass { asset: String, class: u16 },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Changes whenever the cook writes different output for the same inputs and settings.
const COOK_VERSION: u32 = 1;

/// Cooks every archive of the manifest. Work runs on the cook's own thread pool.
pub fn cook(manifest: &Manifest) -> Result<Cooked> {
    let pool = rayon::ThreadPoolBuilder::new()
        .build()
        .map_err(|e| orr_pack::Error::from(std::io::Error::other(e)))?;
    pool.install(|| cook_on_pool(manifest))
}

fn cook_on_pool(manifest: &Manifest) -> Result<Cooked> {
    let mut catalog = instances::Catalog::default();
    let mut writers = Vec::with_capacity(manifest.archives.len());
    for spec in &manifest.archives {
        let mut w = ArchiveWriter::create(spec.output.clone())?;
        for asset in &spec.assets {
            cook_asset(&mut w, asset, &mut catalog)?;
        }
        writers.push(w);
    }
    let mut cooked = Cooked::default();
    for (spec, mut w) in manifest.archives.iter().zip(writers) {
        for asset in &spec.assets {
            if let Source::Instances(src) = &asset.source {
                instances::cook(&mut w, &asset.name, src, spec.cell_log2, &catalog)?;
            }
        }
        if let Some(bytes) = w.finish()? {
            cooked.memory.push(bytes);
        }
    }
    Ok(cooked)
}

/// Cooks every source but instances, which wait for the models of the whole manifest.
fn cook_asset(
    w: &mut ArchiveWriter,
    asset: &AssetSpec,
    catalog: &mut instances::Catalog,
) -> Result<()> {
    let name = &asset.name;
    match &asset.source {
        Source::Gltf { path, settings } => cook_gltf(w, name, path, settings, catalog),
        Source::Texture { path, settings } => {
            let bytes = std::fs::read(path).map_err(|source| Error::Read {
                path: path.clone(),
                source,
            })?;
            let built = texture::cook_file(name, &bytes, settings)?;
            let key = cook_key(Class::Texture, &[&texture::encoder_key(settings), &bytes]);
            texture::add(w, name, key, &built)?;
            Ok(())
        }
        Source::Instances(_) => Ok(()),
        Source::User {
            class,
            schema,
            version,
            bytes,
        } => {
            if *class < Class::USER_FIRST {
                return Err(Error::UserClass {
                    asset: name.clone(),
                    class: *class,
                });
            }
            let key = cook_key(*class, &[&version.to_le_bytes(), bytes]);
            w.add(name, *class, *schema, key, &[BlobIn::new(bytes)])?;
            Ok(())
        }
    }
}

/// Content key of a cooked entry from everything that shapes its output.
fn cook_key(class: impl Into<u16>, parts: &[&[u8]]) -> Hash32 {
    let mut h = blake3::Hasher::new();
    h.update(&COOK_VERSION.to_le_bytes());
    h.update(&class.into().to_le_bytes());
    for p in parts {
        h.update(p);
    }
    Hash32(*h.finalize().as_bytes())
}

fn mount(bytes: &[u8]) -> BlobIn<'_> {
    BlobIn {
        mount: true,
        ..BlobIn::new(bytes)
    }
}

fn cook_gltf(
    w: &mut ArchiveWriter,
    name: &str,
    path: &Path,
    settings: &GltfSettings,
    catalog: &mut instances::Catalog,
) -> Result<()> {
    let src = gltf::import(name, path)?;
    let settings_key = format!("{settings:?}");
    let key = |class: Class| cook_key(class, &[settings_key.as_bytes(), &src.input_hash.0]);

    let cooked = mesh::cook(name, &src.mesh, &settings.mesh)?;
    let info = cooked.info.to_bytes();
    let pinned = cooked.info.header.pinned_pages as usize;
    let mut blobs = vec![mount(&info)];
    blobs.extend(cooked.pages.iter().enumerate().map(|(i, p)| BlobIn {
        mount: i < pinned,
        ..BlobIn::new(p)
    }));
    let mesh = w.add(
        &format!("{name}/mesh"),
        Class::Mesh as u16,
        orr_pack::mesh::SCHEMA,
        key(Class::Mesh),
        &blobs,
    )?;

    let mut packer = texture::material::Packer::new(name, &src.images, &settings.textures);
    let mut materials = Vec::with_capacity(src.materials.len());
    for (suffix, m, textures) in &src.materials {
        let mut m = *m;
        packer.material(w, suffix, &mut m, textures)?;
        let material_name = format!("{name}/material/{suffix}");
        catalog.materials.insert(material_name.clone());
        materials.push(w.add(
            &material_name,
            Class::Material as u16,
            orr_pack::material::SCHEMA,
            key(Class::Material),
            &[mount(bytemuck::bytes_of::<Material>(&m))],
        )?);
    }

    catalog.models.insert(
        name.to_owned(),
        instances::ModelFacts {
            instances: (cooked.info.parts.len() as u32).max(1),
            sections: cooked.info.sections.len() as u32,
            position_exponent: cooked.info.header.position_exponent,
        },
    );
    let model = ModelInfo {
        header: ModelHeader {
            kind: ModelHeader::MODEL,
            _pad: [0; 3],
            bounds: cooked.info.header.bounds,
            mesh,
            skeleton: AssetId::NONE,
            materials: Span::default(),
        },
        materials,
    }
    .to_bytes();
    w.add(
        name,
        Class::Model as u16,
        orr_pack::model::SCHEMA,
        key(Class::Model),
        &[mount(&model)],
    )?;
    Ok(())
}
