//! What a cook remembers between runs, in `<archive>.cookstate`: a stat memo (path, size,
//! modification time -> blake3) so unchanged inputs are not rehashed, and one record per
//! source naming the files it read and the keys of the entries it cooked.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use orr_pack::Hash32;
use serde::{Deserialize, Serialize};

use crate::key::{from_hex, hex};
use crate::{Error, Result};

const STATE_VERSION: u32 = 1;

/// Files modified this recently are not memoised: a write within the clock's resolution
/// could keep size and time and change the bytes.
const SETTLE: Duration = Duration::from_secs(2);

#[derive(Copy, Clone, PartialEq, Eq)]
struct Stat {
    size: u64,
    mtime: u64,
}

impl Stat {
    fn of(meta: &fs::Metadata) -> Option<Self> {
        let mtime = meta
            .modified()
            .ok()?
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?;
        Some(Self {
            size: meta.len(),
            mtime: u64::try_from(mtime.as_nanos()).ok()?,
        })
    }

    fn settled(self) -> bool {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .is_ok_and(|now| u128::from(self.mtime) + SETTLE.as_nanos() < now.as_nanos())
    }
}

/// Content hashes of input files, memoised by size and modification time.
#[derive(Default)]
pub(crate) struct Memo {
    files: Mutex<HashMap<PathBuf, (Stat, Hash32)>>,
}

impl Memo {
    fn insert(&self, path: PathBuf, stat: Stat, hash: Hash32) {
        self.files
            .lock()
            .expect("memo lock")
            .insert(path, (stat, hash));
    }

    /// The file's content hash; reads and hashes it only when its size or time changed.
    pub(crate) fn hash(&self, path: &Path) -> Result<Hash32> {
        let meta = fs::metadata(path).map_err(|source| read_error(path, source))?;
        let stat = Stat::of(&meta);
        if let Some(stat) = stat
            && let Some((s, h)) = self.files.lock().expect("memo lock").get(path)
            && *s == stat
        {
            return Ok(*h);
        }
        let hash = hash_file(path)?;
        if let Some(stat) = stat {
            self.insert(path.to_owned(), stat, hash);
        }
        Ok(hash)
    }

    /// Reads a file for an importer, memoising its hash; the stat comes first, so a write
    /// during the read leaves a stat that no longer matches.
    pub(crate) fn read(&self, path: &Path) -> Result<(Vec<u8>, Hash32)> {
        let stat = fs::metadata(path).ok().as_ref().and_then(Stat::of);
        let bytes = fs::read(path).map_err(|source| read_error(path, source))?;
        let hash = Hash32::of(&bytes);
        if let Some(stat) = stat {
            self.insert(path.to_owned(), stat, hash);
        }
        Ok((bytes, hash))
    }

    fn get(&self, path: &Path) -> Option<(Stat, Hash32)> {
        self.files.lock().expect("memo lock").get(path).copied()
    }
}

/// Hashes a file without holding it whole.
pub(crate) fn hash_file(path: &Path) -> Result<Hash32> {
    let mut h = blake3::Hasher::new();
    let file = fs::File::open(path).map_err(|source| read_error(path, source))?;
    h.update_reader(file)
        .map_err(|source| read_error(path, source))?;
    Ok(Hash32(*h.finalize().as_bytes()))
}

/// Reads a file whose hash a plan recorded; fails if it changed since.
pub(crate) fn read_checked(path: &Path, hash: Hash32) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|source| read_error(path, source))?;
    if Hash32::of(&bytes) != hash {
        return Err(Error::Changed {
            path: path.to_owned(),
        });
    }
    Ok(bytes)
}

fn read_error(path: &Path, source: std::io::Error) -> Error {
    Error::Read {
        path: path.to_owned(),
        source,
    }
}

/// The files a source read and the keys of the entries it cooked.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Record {
    #[serde(with = "hex_hash")]
    pub identity: Hash32,
    pub files: Vec<Input>,
    #[serde(with = "hex_hashes")]
    pub units: Vec<Hash32>,
}

/// A file an importer read and the hash of the bytes it read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Input {
    pub path: PathBuf,
    #[serde(with = "hex_hash")]
    pub hash: Hash32,
}

#[derive(Default, Serialize, Deserialize)]
struct StateFile {
    version: u32,
    files: Vec<FileStat>,
    sources: Vec<Record>,
}

#[derive(Serialize, Deserialize)]
struct FileStat {
    path: PathBuf,
    size: u64,
    mtime: u64,
    #[serde(with = "hex_hash")]
    hash: Hash32,
}

pub(crate) fn state_path(archive: &Path) -> PathBuf {
    let mut s = archive.as_os_str().to_owned();
    s.push(".cookstate");
    s.into()
}

/// Loads a cookstate into the memo and returns its records. A missing or unreadable state
/// is empty: the cook then rehashes and recooks, which is always correct.
pub(crate) fn load(archive: &Path, memo: &Memo) -> Vec<Record> {
    let Ok(text) = fs::read(state_path(archive)) else {
        return Vec::new();
    };
    let Ok(state) = serde_json::from_slice::<StateFile>(&text) else {
        return Vec::new();
    };
    if state.version != STATE_VERSION {
        return Vec::new();
    }
    for f in state.files {
        let stat = Stat {
            size: f.size,
            mtime: f.mtime,
        };
        memo.files
            .lock()
            .expect("memo lock")
            .entry(f.path)
            .or_insert((stat, f.hash));
    }
    state.sources
}

/// Writes the cookstate of an archive: the memo entries of `files` and the records.
pub(crate) fn save(
    archive: &Path,
    memo: &Memo,
    files: &[PathBuf],
    sources: Vec<Record>,
) -> Result<()> {
    let mut stats: Vec<FileStat> = files
        .iter()
        .filter_map(|p| {
            let (stat, hash) = memo.get(p)?;
            stat.settled().then(|| FileStat {
                path: p.clone(),
                size: stat.size,
                mtime: stat.mtime,
                hash,
            })
        })
        .collect();
    stats.sort_by(|a, b| a.path.cmp(&b.path));
    stats.dedup_by(|a, b| a.path == b.path);
    let state = StateFile {
        version: STATE_VERSION,
        files: stats,
        sources,
    };
    let path = state_path(archive);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let write = |p: &Path, b: &[u8]| {
        fs::write(p, b).map_err(|source| Error::Write {
            path: p.to_owned(),
            source,
        })
    };
    write(
        &tmp,
        &serde_json::to_vec(&state).expect("cookstate serialises"),
    )?;
    fs::rename(&tmp, &path).map_err(|source| Error::Write { path, source })
}

mod hex_hash {
    use super::*;

    pub(super) fn serialize<S: serde::Serializer>(h: &Hash32, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex(*h))
    }

    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Hash32, D::Error> {
        let s = String::deserialize(d)?;
        from_hex(&s).ok_or_else(|| serde::de::Error::custom("not a 64-digit hex hash"))
    }
}

mod hex_hashes {
    use super::*;

    pub(super) fn serialize<S: serde::Serializer>(h: &[Hash32], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(h.iter().map(|h| hex(*h)))
    }

    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<Hash32>, D::Error> {
        Vec::<String>::deserialize(d)?
            .iter()
            .map(|s| from_hex(s).ok_or_else(|| serde::de::Error::custom("not a 64-digit hex hash")))
            .collect()
    }
}
