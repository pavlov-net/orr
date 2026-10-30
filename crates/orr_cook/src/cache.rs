//! Content-addressed cache of cooked entries: the entries one cook job wrote, as an archive
//! at `<cache>/<cook key>`. Files are written by rename, so an interrupted cook leaves only
//! whole archives, and the next cook links them.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use orr_pack::{Archive, Hash32};

use crate::key::{from_hex, hex};
use crate::{Error, Result};

pub(crate) struct Cache {
    dir: PathBuf,
}

impl Cache {
    pub(crate) fn new(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).map_err(|source| Error::Write {
            path: dir.to_owned(),
            source,
        })?;
        Ok(Self {
            dir: dir.to_owned(),
        })
    }

    pub(crate) fn path(&self, key: Hash32) -> PathBuf {
        self.dir.join(hex(key))
    }

    /// The cached archive of `key` if it opens and every entry carries the key. Marks it
    /// used now, for eviction by last use.
    pub(crate) fn find(&self, key: Hash32) -> Option<Archive> {
        let path = self.path(key);
        let a = Archive::open(&path).ok()?;
        if a.entries().is_empty() || a.entries().iter().any(|e| e.cook_key != key) {
            return None;
        }
        if let Ok(f) = File::options().write(true).open(&path) {
            let _ = f.set_modified(SystemTime::now());
        }
        Some(a)
    }

    /// Removes the least recently used archives until the cache holds at most `cap` bytes.
    pub(crate) fn evict(&self, cap: u64) -> Result<()> {
        let rd = fs::read_dir(&self.dir).map_err(|source| Error::Read {
            path: self.dir.clone(),
            source,
        })?;
        let mut files: Vec<(SystemTime, u64, PathBuf)> = rd
            .filter_map(|e| {
                let e = e.ok()?;
                from_hex(e.file_name().to_str()?)?;
                let meta = e.metadata().ok()?;
                Some((meta.modified().ok()?, meta.len(), e.path()))
            })
            .collect();
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        files.sort();
        for (_, len, path) in files {
            if total <= cap {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                total -= len;
            }
        }
        Ok(())
    }
}
