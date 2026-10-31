use crate::{Codec, Error, Filter, GDeflate, Result};

pub(crate) fn split_planes(src: &[u8], dst: &mut Vec<u8>) {
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
    let planes = filter == Filter::BytePlanes4;
    let mut scratch = Vec::new();
    let out: &mut [u8] = if planes {
        scratch.resize(dst.len(), 0);
        &mut scratch
    } else {
        dst
    };
    match codec {
        Codec::None => {
            if src.len() != out.len() {
                return Err(Error::Decode(format!(
                    "{} stored bytes for {}",
                    src.len(),
                    out.len()
                )));
            }
            out.copy_from_slice(src);
        }
        Codec::Zstd => {
            let n = zstd::bulk::decompress_to_buffer(src, &mut *out)
                .map_err(|e| Error::Decode(e.to_string()))?;
            if n != out.len() {
                return Err(Error::Decode(format!(
                    "zstd gave {n} bytes, expected {}",
                    out.len()
                )));
            }
        }
        Codec::GDeflate => gdeflate.ok_or(Error::GDeflate)?.decompress(src, out)?,
    }
    if planes {
        join_planes(&scratch, dst);
    }
    Ok(())
}
