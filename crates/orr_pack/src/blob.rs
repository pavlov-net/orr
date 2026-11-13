//! Little-endian POD tables inside class blobs.

use bytemuck::checked::CheckedBitPattern;
use bytemuck::{NoUninit, Pod};

use crate::{Error, Result, Span};

pub(crate) fn read<T: Pod>(b: &[u8], at: usize) -> Option<T> {
    b.get(at..at.checked_add(size_of::<T>())?)
        .map(bytemuck::pod_read_unaligned)
}

pub(crate) fn table<T: CheckedBitPattern>(blob: &[u8], s: Span) -> Result<Vec<T>> {
    let len = s.count as usize * size_of::<T>();
    let at = s.first as usize;
    let bytes = at
        .checked_add(len)
        .and_then(|end| blob.get(at..end))
        .ok_or(Error::Malformed("table outside its blob"))?;
    bytes
        .chunks_exact(size_of::<T>())
        .map(|r| {
            bytemuck::checked::try_pod_read_unaligned(r)
                .map_err(|_| Error::Malformed("table record out of range"))
        })
        .collect()
}

pub(crate) fn put_table<T: NoUninit>(out: &mut Vec<u8>, t: &[T]) -> Span {
    put_table_aligned(out, t, 4)
}

/// [`put_table`] for a table the GPU reads in units wider than a word.
pub(crate) fn put_table_aligned<T: NoUninit>(out: &mut Vec<u8>, t: &[T], align: usize) -> Span {
    out.resize(out.len().next_multiple_of(align), 0);
    let first = out.len() as u32;
    out.extend_from_slice(bytemuck::cast_slice(t));
    Span {
        first,
        count: t.len() as u32,
    }
}
