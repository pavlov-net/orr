//! Model class: one blob holding a [`ModelHeader`] and its material table ([`ModelInfo`]).

use bytemuck::{Pod, Zeroable};

use crate::blob::{put_table, read, table};
use crate::{Aabb, AssetId, Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Spans name tables in the same blob: `first` is a byte offset, `count` a record count.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct ModelHeader {
    /// [`Self::MODEL`].
    pub kind: u8,
    pub _pad: [u8; 3],
    pub bounds: Aabb,
    pub mesh: AssetId,
    /// [`AssetId::NONE`] for a model without a skeleton.
    pub skeleton: AssetId,
    /// Material [`AssetId`] per mesh section.
    pub materials: Span,
}

impl ModelHeader {
    pub const MODEL: u8 = 0;
}

/// A model blob: the header and its material table.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub header: ModelHeader,
    pub materials: Vec<AssetId>,
}

impl ModelInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let header: ModelHeader = read(blob, 0).ok_or(Error::Malformed("model header"))?;
        Ok(Self {
            materials: table(blob, header.materials)?,
            header,
        })
    }

    /// Blob bytes; the header's material span is set from the table.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut out = vec![0; size_of::<ModelHeader>()];
        h.materials = put_table(&mut out, &self.materials);
        out[..size_of::<ModelHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }
}
