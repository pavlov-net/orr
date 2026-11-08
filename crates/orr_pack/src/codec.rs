use crate::{BcLayout, Codec, DEFAULT_CHUNK, Error, Filter, GDeflate, Result};

impl Filter {
    /// The filter's nibble in [`crate::Blob::codec`].
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::BytePlanes4 => 1,
            Self::BcSplit(l) => 2 + l as u8,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::None),
            1 => Some(Self::BytePlanes4),
            c => BcLayout::ALL
                .get(usize::from(c) - 2)
                .map(|&l| Self::BcSplit(l)),
        }
    }

    /// Writes the filtered form of `src` to `dst`.
    pub(crate) fn apply(self, src: &[u8], dst: &mut Vec<u8>) {
        match self {
            Self::None => {
                dst.clear();
                dst.extend_from_slice(src);
            }
            Self::BytePlanes4 => split_planes(src, dst),
            Self::BcSplit(l) => l.split(src, dst),
        }
    }

    /// Largest filtered form of `raw` bytes.
    fn max_len(self, raw: usize) -> usize {
        match self {
            Self::BcSplit(l) => l.max_split_len(raw),
            _ => raw,
        }
    }

    fn undo(self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        if let Self::BcSplit(l) = self {
            return l.join(src, dst);
        }
        check_len(src.len(), dst.len())?;
        match self {
            Self::BytePlanes4 => join_planes(src, dst),
            _ => dst.copy_from_slice(src),
        }
        Ok(())
    }
}

fn split_planes(src: &[u8], dst: &mut Vec<u8>) {
    let n = src.len() / 4;
    dst.clear();
    for p in 0..4 {
        dst.extend(src[..n * 4].iter().skip(p).step_by(4));
    }
    dst.extend_from_slice(&src[n * 4..]);
}

fn join_planes(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / 4;
    for (p, plane) in src[..n * 4].chunks_exact(n.max(1)).enumerate().take(4) {
        for (i, &b) in plane.iter().enumerate() {
            dst[i * 4 + p] = b;
        }
    }
    dst[n * 4..].copy_from_slice(&src[n * 4..]);
}

/// Decodes one stored chunk into `dst`, whose length is the chunk's decoded length.
/// GDeflate chunks need `gdeflate`.
pub(crate) fn decode_chunk(
    codec: Codec,
    filter: Filter,
    gdeflate: Option<&dyn GDeflate>,
    src: &[u8],
    dst: &mut [u8],
) -> Result<()> {
    match (codec, filter) {
        (Codec::None, _) => filter.undo(src, dst),
        (_, Filter::None) => {
            let n = decompress(codec, gdeflate, src, dst)?;
            check_len(n, dst.len())
        }
        _ => {
            let len = filter.max_len(dst.len());
            let own = dst.len() > DEFAULT_CHUNK as usize;
            let mut undo = move |scratch: &mut [u8]| {
                let n = decompress(codec, gdeflate, src, scratch)?;
                filter.undo(&scratch[..n], dst)
            };
            if own {
                return undo(&mut vec![0; len]);
            }
            FILTERED.with_borrow_mut(|scratch| {
                if scratch.len() < len {
                    scratch.resize(len, 0);
                }
                undo(&mut scratch[..len])
            })
        }
    }
}

thread_local! {
    /// A filtered chunk of up to the default chunk length between its decompression and the
    /// filter's undo, kept between chunks so that each decode neither allocates nor clears
    /// it. Longer ones, such as whole mount blobs, take memory of their own.
    static FILTERED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn check_len(n: usize, want: usize) -> Result<()> {
    if n == want {
        Ok(())
    } else {
        Err(Error::Decode(format!("{n} decoded bytes for {want}")))
    }
}

/// Decompresses `src` into the front of `dst` and returns the decoded length.
fn decompress(
    codec: Codec,
    gdeflate: Option<&dyn GDeflate>,
    src: &[u8],
    dst: &mut [u8],
) -> Result<usize> {
    match codec {
        Codec::None => unreachable!("stored chunks are not decompressed"),
        Codec::Zstd => {
            zstd::bulk::decompress_to_buffer(src, dst).map_err(|e| Error::Decode(e.to_string()))
        }
        Codec::GDeflate => {
            gdeflate.ok_or(Error::GDeflate)?.decompress(src, dst)?;
            Ok(dst.len())
        }
    }
}
