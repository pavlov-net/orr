//! Cooked asset archives: the container format, asset identity and the readers and writer.
//!
//! An archive starts with one 4 KiB-aligned index range (header, table of contents and the
//! mount section), followed by 4 KiB-aligned payload blobs. Each asset is an [`Entry`] with
//! one or more [`Blob`]s; blobs are compressed per chunk so a chunk reads and decodes alone.

mod archive;
mod blob;
pub mod clip;
mod codec;
mod format;
pub mod instance;
pub mod material;
pub mod mesh;
pub mod model;
pub mod neural;
pub mod raster;
pub mod skeleton;
pub mod texture;
mod writer;

use std::fmt;
use std::path::PathBuf;

use bytemuck::{Pod, Zeroable};

pub use archive::{Archive, StoredRange};
pub use format::{Blob, Chunk, Entry, FORMAT_VERSION, Header, StrRef};
pub use writer::{ArchiveWriter, BlobIn, EncodedBlob, EntryBlobs, Target};

/// Root of this crate's Slang modules (`orr_pack.*`), for shader include paths.
pub const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");

/// Stable asset identity: the first 16 bytes of blake3 of the logical name.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct AssetId(pub [u8; 16]);

impl AssetId {
    /// No asset.
    pub const NONE: Self = Self([0; 16]);

    pub fn from_name(name: &str) -> Self {
        let mut id = [0; 16];
        id.copy_from_slice(&blake3::hash(name.as_bytes()).as_bytes()[..16]);
        Self(id)
    }
}

impl fmt::Debug for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

/// A blake3 hash.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Hash32(pub [u8; 32]);

impl Hash32 {
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    pub(crate) fn finish(h: &blake3::Hasher) -> Self {
        Self(*h.finalize().as_bytes())
    }
}

impl fmt::Debug for Hash32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

/// An axis-aligned box.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// The box that contains nothing: min above max.
    pub const EMPTY: Self = Self {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    pub fn extend(&mut self, p: [f32; 3]) {
        self.min = [0, 1, 2].map(|a| self.min[a].min(p[a]));
        self.max = [0, 1, 2].map(|a| self.max[a].max(p[a]));
    }
}

/// An affine transform as three rows `[R | t]`, 48 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Affine3 {
    pub rows: [[f32; 4]; 3],
}

impl Affine3 {
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        self.rows
            .map(|r| r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3])
    }

    /// `self x o`: `o` applies first.
    pub fn mul(&self, o: &Self) -> Self {
        let (a, b) = (&self.rows, &o.rows);
        Self {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let t = if j == 3 { a[i][3] } else { 0.0 };
                    a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + t
                })
            }),
        }
    }
}

/// A bounding sphere.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
}

/// A range of records in a table: `first..first + count`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Span {
    pub first: u32,
    pub count: u32,
}

/// Built-in asset classes. Values from [`Class::USER_FIRST`] up are consumer classes.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Class {
    Mesh = 1,
    Texture = 2,
    Material = 3,
    Model = 4,
    InstancePage = 5,
    RasterSet = 6,
    Skeleton = 7,
    Clip = 8,
    NeuralTextureSet = 9,
    Subd = 10,
    Volume = 11,
}

/// Current schema of each built-in class whose format this crate defines.
const SCHEMAS: &[(Class, u16)] = &[
    (Class::Mesh, mesh::SCHEMA),
    (Class::Material, material::SCHEMA),
    (Class::Model, model::SCHEMA),
    (Class::Texture, texture::SCHEMA),
    (Class::InstancePage, instance::SCHEMA),
    (Class::RasterSet, raster::SCHEMA),
    (Class::Skeleton, skeleton::SCHEMA),
    (Class::Clip, clip::SCHEMA),
    (Class::NeuralTextureSet, neural::SCHEMA),
];

impl Class {
    pub const USER_FIRST: u16 = 0x8000;
    const ALL: [Self; 11] = [
        Self::Mesh,
        Self::Texture,
        Self::Material,
        Self::Model,
        Self::InstancePage,
        Self::RasterSet,
        Self::Skeleton,
        Self::Clip,
        Self::NeuralTextureSet,
        Self::Subd,
        Self::Volume,
    ];

    pub fn from_u16(v: u16) -> Option<Self> {
        Self::ALL.into_iter().find(|c| *c as u16 == v)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mesh => "mesh",
            Self::Texture => "texture",
            Self::Material => "material",
            Self::Model => "model",
            Self::InstancePage => "instance-page",
            Self::RasterSet => "raster-set",
            Self::Skeleton => "skeleton",
            Self::Clip => "clip",
            Self::NeuralTextureSet => "neural-texture-set",
            Self::Subd => "subd",
            Self::Volume => "volume",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }

    /// The schema this crate reads and writes for the class, if it defines the class's format.
    pub fn schema(self) -> Option<u16> {
        SCHEMAS.iter().find(|(c, _)| *c == self).map(|(_, s)| *s)
    }
}

impl From<Class> for u16 {
    fn from(c: Class) -> u16 {
        c as u16
    }
}

/// Blob compression.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Codec {
    None = 0,
    Zstd = 1,
    /// Decompressed on the GPU; the CPU decode refuses it.
    GDeflate = 2,
}

/// A reversible byte transform applied to each chunk before compression.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Filter {
    None = 0,
    /// Splits each chunk into four planes of byte 0, 1, 2 and 3 of every 4-byte word; a
    /// trailing partial word stays in place.
    BytePlanes4 = 1,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("not an orr archive")]
    NotArchive,
    #[error("archive format version {found}, this reader reads {expected}")]
    Version { found: u32, expected: u32 },
    #[error("archive truncated: {have} of {need} bytes")]
    Truncated { need: u64, have: u64 },
    #[error("archive table of contents is corrupt")]
    CorruptToc,
    #[error("archive is malformed: {0}")]
    Malformed(&'static str),
    #[error("asset {0} is not in the archive")]
    Missing(AssetId),
    #[error("{name}: class {found}, expected {expected}")]
    Class {
        name: String,
        found: u16,
        expected: u16,
    },
    #[error("{name}: schema {found}, expected {expected}")]
    Schema {
        name: String,
        found: u16,
        expected: u16,
    },
    #[error("{name}: content hash does not match the decoded bytes")]
    ContentHash { name: String },
    #[error("blob does not decode: {0}")]
    Decode(String),
    #[error("GDeflate blobs are decoded on the GPU")]
    GDeflate,
    #[error("buffer holds {have} bytes, needs {need}")]
    BufferSize { need: usize, have: usize },
    #[error("asset {0} was already added")]
    Duplicate(String),
    #[error("invalid blob: {0}")]
    InvalidBlob(&'static str),
    #[error("cannot replace {path}: {source}")]
    Replace {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
