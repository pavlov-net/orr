//! Manifests in TOML.
//!
//! ```toml
//! [[archive]]
//! output = "out/props.orr"      # relative paths are relative to the manifest
//!
//! [[archive.asset]]
//! name = "props/rock"
//! gltf = { path = "rock.glb", settings = { mesh = { position_step_log2 = -10 } } }
//!
//! [[archive.asset]]
//! name = "sky"
//! texture = { path = "sky.exr", settings = { usage = { environment = { max_lights = 4 } } } }
//! ```
//!
//! Each asset names exactly one source: `gltf`, `texture`, `instances` or `user`, with the
//! fields of the matching [`Source`] variant. Settings fields left out take their defaults.

use std::path::{Path, PathBuf};

use orr_pack::Target;
use serde::{Deserialize, Deserializer};

use crate::{ArchiveSpec, Error, Manifest, Result, Source};

impl Manifest {
    /// Reads a TOML manifest; relative paths in it are resolved against its directory.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: path.to_owned(),
            source,
        })?;
        let mut m: Self = toml::from_str(&text).map_err(|e| Error::Manifest {
            path: path.to_owned(),
            message: e.to_string(),
        })?;
        let dir = path.parent().unwrap_or(Path::new(""));
        let resolve = |p: &mut PathBuf| {
            if p.is_relative() {
                *p = dir.join(&*p);
            }
        };
        for a in &mut m.archives {
            if let Target::File(p) = &mut a.output {
                resolve(p);
            }
            for asset in &mut a.assets {
                match &mut asset.source {
                    Source::Gltf { path, .. } | Source::Texture { path, .. } => resolve(path),
                    Source::Instances(_) | Source::User { .. } => {}
                }
            }
        }
        Ok(m)
    }
}

pub(crate) fn file_target<'de, D: Deserializer<'de>>(d: D) -> Result<Target, D::Error> {
    PathBuf::deserialize(d).map(Target::File)
}

pub(crate) fn default_cell_log2() -> u8 {
    ArchiveSpec::DEFAULT_CELL_LOG2
}
