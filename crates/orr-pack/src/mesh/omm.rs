//! A page's opacity micromaps: the resident bytes from `omm_offset` to the end of the page.
//!
//! ```text
//! MicromapEntry[E]       at omm_offset; E is the sum of PageHeader::omm_usage
//! per cluster with micromaps, at omm_offset + omm_first (4-aligned):
//!     i16 index per triangle from alpha_first to tri_count: an entry of this page,
//!     or a special index (OMM_TRANSPARENT .. OMM_UNKNOWN_OPAQUE)
//! entry data             4-state, 2 bits per micro-triangle in the micromap's own order,
//!                        lowest bits first
//! ```
//!
//! A page with micromaps has `omm_offset` a multiple of [`OMM_ALIGN`], so a page placed at
//! a 256-byte aligned address builds its micromap in place: the entries are the triangle
//! array and `data_offset` counts from the same address.

use bytemuck::{Pod, Zeroable};

use super::{Cluster, PageView};
use crate::blob::read;
use crate::{Error, Result};

/// Alignment of `omm_offset` in a page that has micromaps.
pub const OMM_ALIGN: u32 = 256;

/// Special micromap indices: the whole triangle in one state.
pub const OMM_TRANSPARENT: i16 = -1;
pub const OMM_OPAQUE: i16 = -2;
pub const OMM_UNKNOWN_TRANSPARENT: i16 = -3;
pub const OMM_UNKNOWN_OPAQUE: i16 = -4;

/// Micro-triangle states of the 4-state format.
pub const OMM_STATE_TRANSPARENT: u8 = 0;
pub const OMM_STATE_OPAQUE: u8 = 1;
pub const OMM_STATE_UNKNOWN_TRANSPARENT: u8 = 2;
pub const OMM_STATE_UNKNOWN_OPAQUE: u8 = 3;

/// Highest subdivision level a page may hold.
pub const OMM_MAX_LEVEL: u16 = 12;

/// One micromap of a page, laid out as `VkMicromapTriangleEXT`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MicromapEntry {
    /// Bytes from `omm_offset` to the entry's states.
    pub data_offset: u32,
    pub level: u16,
    /// Always [`Self::FOUR_STATE`].
    pub format: u16,
}

impl MicromapEntry {
    /// `VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT`.
    pub const FOUR_STATE: u16 = 2;
}

/// Bytes of one 4-state micromap at `level`: 2 bits for each of its `4^level` micro-triangles.
pub fn micromap_len(level: u16) -> usize {
    (1usize << (2 * level)).div_ceil(4)
}

/// The checked micromaps of a resident page.
#[derive(Copy, Clone)]
pub struct Micromaps<'a> {
    region: &'a [u8],
    count: u32,
}

impl Micromaps<'_> {
    pub fn entry_count(&self) -> u32 {
        self.count
    }

    pub fn entry(&self, i: u32) -> MicromapEntry {
        assert!(i < self.count);
        read(self.region, i as usize * size_of::<MicromapEntry>()).unwrap()
    }

    /// State of micro-triangle `m` of entry `i`.
    pub fn state(&self, i: u32, m: u32) -> u8 {
        let e = self.entry(i);
        assert!(m < 1 << (2 * e.level));
        let byte = self.region[e.data_offset as usize + m as usize / 4];
        byte >> (2 * (m % 4)) & 3
    }
}

impl<'a> PageView<'a> {
    /// Checks the page's micromap region against its usage counts and every cluster's
    /// indices, then reads it.
    pub fn micromaps(&self) -> Result<Micromaps<'a>> {
        let bad = |what| Err(Error::Malformed(what));
        let h = &self.header;
        let region = &self.page[h.omm_offset as usize..];
        let count: u64 = h.omm_usage.iter().map(|&u| u64::from(u)).sum();
        let clusters = (0..self.cluster_count()).map(|i| self.cluster(i));
        if region.is_empty() {
            if count != 0 || clusters.clone().any(|c| c.omm_first != Cluster::NO_OMM) {
                return bad("micromaps without a micromap region");
            }
            return Ok(Micromaps { region, count: 0 });
        }
        let table = count * size_of::<MicromapEntry>() as u64;
        if !h.omm_offset.is_multiple_of(OMM_ALIGN) || table > region.len() as u64 {
            return bad("micromap region");
        }
        let maps = Micromaps {
            region,
            count: count as u32,
        };
        let mut usage = [0u32; 13];
        for i in 0..maps.count {
            let e = maps.entry(i);
            let fits = (e.data_offset as usize)
                .checked_add(micromap_len(e.level.min(OMM_MAX_LEVEL)))
                .is_some_and(|end| end <= region.len());
            if e.format != MicromapEntry::FOUR_STATE
                || e.level > OMM_MAX_LEVEL
                || u64::from(e.data_offset) < table
                || !fits
            {
                return bad("micromap entry");
            }
            usage[usize::from(e.level)] += 1;
        }
        if usage != h.omm_usage {
            return bad("micromap usage counts");
        }
        for c in clusters.filter(|c| c.omm_first != Cluster::NO_OMM) {
            let n = usize::from(c.tri_count - c.alpha_first);
            let at = c.omm_first as usize;
            if n == 0 || !at.is_multiple_of(4) || at + 2 * n > region.len() {
                return bad("cluster micromap indices");
            }
            for t in 0..n {
                let i: i16 = read(region, at + 2 * t).unwrap();
                if i < OMM_UNKNOWN_OPAQUE || i32::from(i) >= count as i32 {
                    return bad("cluster micromap index");
                }
            }
        }
        Ok(maps)
    }
}
