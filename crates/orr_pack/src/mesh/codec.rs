//! Stored form of a mesh page: the resident header and cluster records as they are, then
//! each cluster's vertices through meshoptimizer's vertex codec and its triangles through
//! the meshlet codec, then the micromap region as it is.
//!
//! ```text
//! PageHeader, Cluster[n]            as resident
//! u32 offsets[2n + 1]               offsets[2i]: cluster i's vertex stream,
//!                                   offsets[2i + 1]: its triangle stream,
//!                                   offsets[2n]: the end of the streams (from the page start)
//! streams                           in cluster order
//! micromap region                   omm_offset..len of the resident page
//! ```
//!
//! A vertex stream holds `vertex_count` vertices of `8 + attr_stride` bytes: the three u16
//! grid offsets, a zero u16, then the vertex's attributes. The vertex codec is version 1,
//! encoded at level 2: level 3 searches channel rotations and a third channel mode for a
//! stream zstd then packs no smaller, at a third more encoding time. A triangle stream is
//! the meshlet codec over triangles only.
//!
//! Every region of a page starts 4-aligned. `orr_pack.page_codec` decodes the same
//! form on the GPU and accepts exactly the pages [`decode_page`] accepts.

use meshopt::ffi;

use super::{Cluster, PageHeader, PageLayout, check_cluster, check_page, layout};
use crate::blob::read;
use crate::{Error, Result};

const VERTEX_CODEC_VERSION: i32 = 1;
const VERTEX_CODEC_LEVEL: i32 = 2;
/// First byte of a vertex stream: the codec's 0xA0 and its version. Other versions are
/// rejected.
const VERTEX_CODEC_HEADER: u8 = 0xA0 | VERTEX_CODEC_VERSION as u8;
const MAX_VERTICES: usize = 256;
const MAX_VERTEX_BYTES: usize = 8 + 32;

/// Stored form of a resident page. The meshlet codec may rotate a triangle's corners
/// (order and winding kept), so [`decode_page`] of the result can differ from `page` in
/// corner order; a page that came out of [`decode_page`] round-trips exactly.
pub fn encode_page(page: &[u8]) -> Result<Vec<u8>> {
    let h = check_page(page)?;
    let n = h.cluster_count as usize;
    let stride = 8 + h.attr_stride as usize;
    let records_end = h.positions_offset as usize;
    let mut out = page[..records_end].to_vec();
    let table = out.len();
    out.resize(table + (2 * n + 1) * 4, 0);
    let mut vertices = Vec::with_capacity(MAX_VERTICES * stride);
    for i in 0..n {
        let c = cluster(page, i);
        let (v0, vn) = (c.vertex_first as usize, usize::from(c.vertex_count));
        vertices.clear();
        for v in v0..v0 + vn {
            let p = h.positions_offset as usize + v * 6;
            vertices.extend_from_slice(&page[p..p + 6]);
            vertices.extend_from_slice(&[0; 2]);
            let a = h.attrs_offset as usize + v * h.attr_stride as usize;
            vertices.extend_from_slice(&page[a..a + h.attr_stride as usize]);
        }
        put_offset(&mut out, table, 2 * i);
        // SAFETY: plain bound query.
        let bound = unsafe { ffi::meshopt_encodeVertexBufferBound(vn, stride) };
        out.reserve(bound);
        let at = out.len();
        // SAFETY: `vertices` holds vn vertices of `stride` bytes and `out` has room for
        // `bound` bytes from `at`, which the encoder only writes; it writes the `len` it
        // returns.
        unsafe {
            let len = ffi::meshopt_encodeVertexBufferLevel(
                out.as_mut_ptr().add(at),
                bound,
                vertices.as_ptr().cast(),
                vn,
                stride,
                VERTEX_CODEC_LEVEL,
                VERTEX_CODEC_VERSION,
            );
            assert!(len > 0, "the vertex codec bound holds the stream");
            out.set_len(at + len);
        }

        put_offset(&mut out, table, 2 * i + 1);
        let tn = usize::from(c.tri_count);
        let tris = &page[(h.tris_offset + c.tri_first) as usize..][..tn * 3];
        // SAFETY: plain bound query.
        let bound = unsafe { ffi::meshopt_encodeMeshletBound(0, tn) };
        out.reserve(bound);
        let at = out.len();
        // SAFETY: triangles only, so no vertex references; `tris` holds tn triangles and
        // `out` has room for `bound` bytes from `at`, which the encoder only writes; it
        // writes the `len` it returns.
        unsafe {
            let len = ffi::meshopt_encodeMeshlet(
                out.as_mut_ptr().add(at),
                bound,
                std::ptr::null(),
                0,
                tris.as_ptr(),
                tn,
            );
            assert!(len > 0, "the meshlet codec bound holds the stream");
            out.set_len(at + len);
        }
    }
    put_offset(&mut out, table, 2 * n);
    out.extend_from_slice(&page[h.omm_offset as usize..h.len as usize]);
    Ok(out)
}

/// A stored page's header and cluster records, and each cluster's vertex and triangle
/// streams as `[start, split, end]` byte offsets.
struct Stored {
    h: PageHeader,
    clusters: Vec<Cluster>,
    streams: Vec<[usize; 3]>,
    streams_end: usize,
}

/// Checks everything of a stored page that [`decode_page`] checks before it decodes the
/// streams: header, regions, stream table, every cluster's ranges and stream offsets, and
/// each vertex stream's codec version.
fn check_stored(stored: &[u8]) -> Result<Stored> {
    let bad = |what| Err(Error::Malformed(what));
    let Some(h) = read::<PageHeader>(stored, 0) else {
        return bad("page shorter than its header");
    };
    let (slots, tris_len) = layout(&h)?;
    let n = h.cluster_count as usize;
    let table = h.positions_offset as usize;
    let offset = |i: usize| read::<u32>(stored, table + 4 * i).map(|o| o as usize);
    let Some(streams_end) = offset(2 * n) else {
        return bad("page stream table");
    };
    if offset(0) != Some(table + (2 * n + 1) * 4)
        || streams_end > stored.len()
        || stored.len() - streams_end != (h.len - h.omm_offset) as usize
    {
        return bad("page stream table");
    }
    let stride = 8 + h.attr_stride as usize;
    let mut clusters = Vec::with_capacity(n);
    let mut streams = Vec::with_capacity(n);
    for i in 0..n {
        let c = cluster(stored, i);
        check_cluster(&c, clusters.last(), slots, tris_len)?;
        let [a, b, e] = [2 * i, 2 * i + 1, 2 * i + 2].map(|j| offset(j).unwrap_or(0));
        if !(a <= b && b <= e && e <= streams_end)
            || usize::from(c.vertex_count) > MAX_VERTICES
            || stride > MAX_VERTEX_BYTES
        {
            return bad("cluster streams");
        }
        if a == b || stored[a] != VERTEX_CODEC_HEADER {
            return bad("cluster vertex stream");
        }
        clusters.push(c);
        streams.push([a, b, e]);
    }
    Ok(Stored {
        h,
        clusters,
        streams,
        streams_end,
    })
}

/// Checks a stored page as far as it can without decoding its streams, as a caller of
/// `orr_pack.page_codec` does before handing the page to the GPU: a page that passes
/// can still fail there on a stream's contents. Returns its header and cluster records,
/// which the resident page holds unchanged.
pub fn check_stored_page(stored: &[u8]) -> Result<(PageHeader, Vec<Cluster>)> {
    let s = check_stored(stored)?;
    Ok((s.h, s.clusters))
}

/// Decodes a stored page (from [`crate::Archive::decode`] or [`crate::Archive::read`]) into
/// `dst` in the resident layout. Every cluster's ranges, streams and triangle corners are
/// checked; a page that fails any check is rejected.
pub fn decode_page(stored: &[u8], dst: &mut [u8]) -> Result<PageLayout> {
    let bad = |what| Err(Error::Malformed(what));
    let Stored {
        h,
        clusters,
        streams,
        streams_end,
    } = check_stored(stored)?;
    let len = h.len as usize;
    if dst.len() < len {
        return Err(Error::BufferSize {
            need: len,
            have: dst.len(),
        });
    }
    let records_end = h.positions_offset as usize;
    let dst = &mut dst[..len];
    dst[..records_end].copy_from_slice(&stored[..records_end]);
    dst[records_end..h.omm_offset as usize].fill(0);
    dst[h.omm_offset as usize..].copy_from_slice(&stored[streams_end..]);

    let stride = 8 + h.attr_stride as usize;
    let attr_stride = h.attr_stride as usize;
    let mut vertices = [0u8; MAX_VERTICES * MAX_VERTEX_BYTES];
    let mut tris = [0u8; 256 * 3];
    for (c, [a, b, e]) in clusters.iter().zip(streams) {
        let (v0, vn) = (c.vertex_first as usize, usize::from(c.vertex_count));
        let tn = usize::from(c.tri_count);
        // SAFETY: `vertices` holds MAX_VERTICES vertices of up to MAX_VERTEX_BYTES, and the
        // decoder reads only the `b - a` bytes of the stream; it is safe on any input.
        let rc = unsafe {
            ffi::meshopt_decodeVertexBuffer(
                vertices.as_mut_ptr().cast(),
                vn,
                stride,
                stored[a..b].as_ptr(),
                b - a,
            )
        };
        if rc != 0 {
            return bad("cluster vertex stream");
        }
        for v in 0..vn {
            let src = &vertices[v * stride..(v + 1) * stride];
            let p = h.positions_offset as usize + (v0 + v) * 6;
            dst[p..p + 6].copy_from_slice(&src[..6]);
            let at = h.attrs_offset as usize + (v0 + v) * attr_stride;
            dst[at..at + attr_stride].copy_from_slice(&src[8..]);
        }
        // SAFETY: triangles only (no vertex buffer); `tris` holds 256 triangles, at least
        // the 4-aligned size of tn; the decoder reads only the `e - b` stream bytes.
        let rc = unsafe {
            ffi::meshopt_decodeMeshlet(
                std::ptr::null_mut(),
                0,
                4,
                tris.as_mut_ptr().cast(),
                tn,
                3,
                stored[b..e].as_ptr(),
                e - b,
            )
        };
        if rc != 0 {
            return bad("cluster triangle stream");
        }
        if tris[..tn * 3].iter().any(|&t| usize::from(t) >= vn) {
            return bad("triangle corner outside its cluster");
        }
        let at = (h.tris_offset + c.tri_first) as usize;
        dst[at..at + tn * 3].copy_from_slice(&tris[..tn * 3]);
    }
    Ok(PageLayout::of(&h))
}

fn cluster(page: &[u8], i: usize) -> Cluster {
    read(page, size_of::<PageHeader>() + i * size_of::<Cluster>())
        .expect("the cluster records lie inside the page")
}

fn put_offset(out: &mut [u8], table: usize, i: usize) {
    let at = table + 4 * i;
    let value = out.len() as u32;
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
