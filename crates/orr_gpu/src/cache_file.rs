//! The pipeline cache file both backends write: magic, format, the device identity (vendor,
//! device, driver version, cache UUID), the backend's cache data, then a BLAKE3 hash of
//! everything before it. A file from another device, driver or format is refused whole.

use std::fs::File;
use std::path::Path;
use std::time::SystemTime;

use crate::Error;

/// What the pipeline cache loaded and how creation has used it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineCacheReport {
    /// The file, or at least one file of the pipeline store, existed and its data was loaded.
    pub loaded: bool,
    /// Pipelines the cache already held: a driver cache hit on Vulkan, a pipeline this session
    /// or the archive held on Metal.
    pub hits: u32,
    /// Pipelines compiled from source.
    pub misses: u32,
    /// Why the file was ignored, when it existed and was.
    pub rejected: Option<String>,
}

const MAGIC: [u8; 8] = *b"orrpipe\0";
const FORMAT: u32 = 2;
/// Magic, format, vendor, device, driver version, then the 16-byte cache UUID.
const HEADER: usize = 24 + 16;
const CHECKSUM: usize = 32;

/// What a cache file must match to be used.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Identity {
    pub(crate) vendor: u32,
    pub(crate) device: u32,
    pub(crate) driver_version: u32,
    pub(crate) cache_uuid: [u8; 16],
}

/// The file: identity, payload, then a BLAKE3 hash of everything before it.
pub(crate) fn encode(id: &Identity, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + payload.len() + CHECKSUM);
    out.extend_from_slice(&MAGIC);
    for v in [FORMAT, id.vendor, id.device, id.driver_version] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&id.cache_uuid);
    out.extend_from_slice(payload);
    let sum = blake3::hash(&out);
    out.extend_from_slice(sum.as_bytes());
    out
}

/// The payload of `file` if it is intact and was written for `expect`; otherwise why not.
pub(crate) fn decode<'a>(file: &'a [u8], expect: &Identity) -> Result<&'a [u8], String> {
    let Some(body_len) = file.len().checked_sub(CHECKSUM) else {
        return Err("truncated".into());
    };
    let (body, sum) = file.split_at(body_len);
    if blake3::hash(body).as_bytes() != sum {
        return Err("checksum mismatch".into());
    }
    let Some((header, payload)) = body.split_at_checked(HEADER) else {
        return Err("truncated".into());
    };
    if header[..8] != MAGIC {
        return Err("not an orr pipeline cache".into());
    }
    let word = |i: usize| u32::from_le_bytes(header[8 + 4 * i..12 + 4 * i].try_into().unwrap());
    let format = word(0);
    if format != FORMAT {
        return Err(format!("format {format}, expected {FORMAT}"));
    }
    let differs = |what: &str| Err(format!("written for another {what}"));
    if (word(1), word(2)) != (expect.vendor, expect.device) {
        return differs("device");
    }
    if word(3) != expect.driver_version {
        return differs("driver version");
    }
    if header[24..] != expect.cache_uuid {
        return differs("pipeline cache UUID");
    }
    Ok(payload)
}

/// Writes `bytes` to `path` whole: to a file of this process beside it, then renamed over it,
/// so processes saving at once each leave a whole file and the last one stays.
pub(crate) fn write(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let io = |e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    };
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(io)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let written = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(io(e));
    }
    Ok(())
}

/// Sets the modification time of `path`, a file a pipeline was just loaded from, to now, so
/// [`trim_pipeline_store`](crate::trim_pipeline_store) sees it as used.
pub(crate) fn mark_used(path: &Path) {
    let touched = File::options()
        .write(true)
        .open(path)
        .and_then(|file| file.set_modified(SystemTime::now()));
    if let Err(e) = touched {
        log::warn!("pipeline file {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_or_damaged_files_are_rejected() {
        let id = Identity {
            vendor: 0x10de,
            device: 0x2b85,
            driver_version: 7,
            cache_uuid: [3; 16],
        };
        let file = encode(&id, b"payload");
        assert_eq!(decode(&file, &id), Ok(&b"payload"[..]));

        let mut other = id.clone();
        other.driver_version += 1;
        assert!(
            decode(&file, &other)
                .unwrap_err()
                .contains("driver version")
        );
        let mut other = id.clone();
        other.device += 1;
        assert!(decode(&file, &other).unwrap_err().contains("device"));
        let mut other = id.clone();
        other.cache_uuid[0] ^= 1;
        assert!(decode(&file, &other).unwrap_err().contains("UUID"));

        let mut flipped = file.clone();
        flipped[30] ^= 1;
        assert!(decode(&flipped, &id).unwrap_err().contains("checksum"));
        assert!(decode(&file[..file.len() - 1], &id).is_err());
        assert!(decode(&[], &id).is_err());
    }
}
