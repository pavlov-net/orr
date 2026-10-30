//! Instance page class: placements of cooked models inside one world cell.
//!
//! - Blob 0 (mount): an [`InstancePageHeader`], then its [`Prototype`] table, the
//!   [`SectionOverride`] table the prototypes point into, and the partition table (u16 first
//!   record per partition). Spans are byte offsets in blob 0 and record counts.
//! - Blob 1: [`InstanceRecord`]s in Morton order of [`InstanceRecord::pos`]; the record count
//!   is `raw_len / 16`, at most [`MAX_RECORDS`].
//! - Blob 2, when `user_stride` is not 0: `user_stride` bytes of consumer data per record,
//!   in record order.
//!
//! A record decodes to a transform relative to the page cell (edge `2^cell_log2` m):
//! translation `origin + pos * (extent / 65535)`, rotation from `rot`, and per-axis scale
//! `2^(lerp(log2_scale_min, log2_scale_max, code / 1023))` applied before the rotation.
//! [`decode`] is the reference both for the GPU decode and for consumers' CPU queries.

use bytemuck::{Pod, Zeroable};

use crate::blob::{put_table, read, table};
use crate::{Affine3, AssetId, Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Records per page.
pub const MAX_RECORDS: usize = 16_384;

/// PTLAS instances per partition.
pub const PARTITION_INSTANCES: u32 = 1000;

/// Pose variants a prototype can name.
pub const MAX_POSE: u16 = 256;

/// Sections a prototype can hide: sections `0..MAX_HIDDEN_SECTION`.
pub const MAX_HIDDEN_SECTION: usize = 16;

/// Blob 0 header, 64 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct InstancePageHeader {
    /// World cell of the page.
    pub cell: [i32; 3],
    /// Cell-relative box the positions quantise over.
    pub origin: [f32; 3],
    pub extent: [f32; 3],
    pub log2_scale_min: f32,
    pub log2_scale_max: f32,
    /// [`Prototype`] table; records name an entry by index.
    pub prototypes: Span,
    /// Bytes per record in blob 2; 0 when the page has no user stream.
    pub user_stride: u16,
    pub _pad: u16,
    /// u16 first record of each PTLAS partition, ascending, the first 0. Partition `p`
    /// covers records `[first[p], first[p + 1])`, the last up to the record count.
    pub partitions: Span,
}

/// A model as placed by the page's records, 32 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Prototype {
    pub model: AssetId,
    /// Instance flags, stored for the renderer as given.
    pub flags: u32,
    /// Pose variant, below [`MAX_POSE`].
    pub pose: u16,
    pub _pad: u16,
    /// One [`SectionOverride`] per mesh section of the model; empty uses the model as cooked.
    pub sections: Span,
}

/// 20 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct SectionOverride {
    /// [`AssetId::NONE`] keeps the model's material.
    pub material: AssetId,
    /// 1 hides the section from raster and ray tracing; only sections below
    /// [`MAX_HIDDEN_SECTION`] can be hidden.
    pub hidden: u8,
    pub _pad: [u8; 3],
}

/// One placement, 16 B, uploaded as stored.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct InstanceRecord {
    /// Position on the page's 65535-step grid over `origin..origin + extent`.
    pub pos: [u16; 3],
    pub prototype: u16,
    /// [`encode_rotation`] code, or with [`Self::EXACT_AXIS`] an [`axis_rotation`] index.
    pub rot: u32,
    /// Bits 0-9, 10-19 and 20-29: x, y and z scale codes; bit 30 [`Self::EXACT_AXIS`].
    pub scale: u32,
}

impl InstanceRecord {
    pub const EXACT_AXIS: u32 = 1 << 30;

    pub fn exact_axis(&self) -> bool {
        self.scale & Self::EXACT_AXIS != 0
    }

    pub fn scale_codes(&self) -> [u32; 3] {
        [0, 10, 20].map(|s| (self.scale >> s) & 1023)
    }
}

/// Blob 0 of a page: the header and its tables.
#[derive(Clone, Debug, PartialEq)]
pub struct InstancePageInfo {
    pub header: InstancePageHeader,
    pub prototypes: Vec<Prototype>,
    /// Every prototype's overrides; [`Prototype::sections`] indexes into this table.
    pub sections: Vec<SectionOverride>,
    pub partitions: Vec<u16>,
}

impl InstancePageInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let header: InstancePageHeader =
            read(blob, 0).ok_or(Error::Malformed("instance page header"))?;
        let prototypes: Vec<Prototype> = table(blob, header.prototypes)?;
        let mut sections = Vec::new();
        let mut prototypes_out = Vec::with_capacity(prototypes.len());
        for p in prototypes {
            let first = sections.len() as u32;
            sections.extend(table::<SectionOverride>(blob, p.sections)?);
            prototypes_out.push(Prototype {
                sections: Span {
                    first,
                    count: p.sections.count,
                },
                ..p
            });
        }
        let partitions: Vec<u16> = table(blob, header.partitions)?;
        if partitions.first().is_some_and(|&p| p != 0) || !partitions.is_sorted() {
            return Err(Error::Malformed("instance page partitions"));
        }
        Ok(Self {
            header,
            prototypes: prototypes_out,
            sections,
            partitions,
        })
    }

    /// Blob bytes; header spans and each prototype's section span are set from the tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut out = vec![0; size_of::<InstancePageHeader>()];
        let prototypes: Vec<Prototype> = self
            .prototypes
            .iter()
            .map(|p| Prototype {
                sections: put_table(&mut out, self.sections_of(p)),
                ..*p
            })
            .collect();
        h.prototypes = put_table(&mut out, &prototypes);
        h.partitions = put_table(&mut out, &self.partitions);
        out[..size_of::<InstancePageHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    /// The overrides of prototype `p`.
    pub fn sections_of(&self, p: &Prototype) -> &[SectionOverride] {
        &self.sections[p.sections.first as usize..(p.sections.first + p.sections.count) as usize]
    }
}

/// Decodes a page: its cell and, per record in order, the prototype index and the
/// cell-relative transform.
pub fn decode<'a>(
    header: &[u8],
    records: &'a [u8],
) -> Result<([i32; 3], impl Iterator<Item = (u16, Affine3)> + 'a)> {
    let info = InstancePageInfo::parse(header)?;
    if !records.len().is_multiple_of(size_of::<InstanceRecord>())
        || records.len() / size_of::<InstanceRecord>() > MAX_RECORDS
    {
        return Err(Error::Malformed("instance records"));
    }
    let h = info.header;
    Ok((
        h.cell,
        records
            .chunks_exact(size_of::<InstanceRecord>())
            .map(bytemuck::pod_read_unaligned::<InstanceRecord>)
            .map(move |r| (r.prototype, decode_record(&h, &r))),
    ))
}

/// Cell-relative transform of one record: rows `[R * diag(scale) | translation]`.
pub fn decode_record(h: &InstancePageHeader, r: &InstanceRecord) -> Affine3 {
    let t = decode_position(h, r.pos);
    let s = r.scale_codes().map(|c| decode_scale(h, c));
    let m = instance_rotation(r.rot, r.exact_axis());
    Affine3 {
        rows: [0, 1, 2].map(|i| [m[i][0] * s[0], m[i][1] * s[1], m[i][2] * s[2], t[i]]),
    }
}

pub fn decode_position(h: &InstancePageHeader, pos: [u16; 3]) -> [f32; 3] {
    [0, 1, 2].map(|a| h.origin[a] + f32::from(pos[a]) * (h.extent[a] / 65535.0))
}

pub fn decode_scale(h: &InstancePageHeader, code: u32) -> f32 {
    let (lo, hi) = (h.log2_scale_min, h.log2_scale_max);
    (lo + (hi - lo) * (code as f32 / 1023.0)).exp2()
}

/// 1/sqrt(2): the largest magnitude a smallest-three component can have.
const ROTATION_RANGE: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// The unit quaternion (x, y, z, w) of a smallest-three code. Bits 30-31 name the
/// component of largest magnitude, stored non-negative and dropped; the other three, in
/// x, y, z, w order, sit in bits 0-9, 10-19 and 20-29 as `round((c / (1 / sqrt(2)) x 0.5 +
/// 0.5) x 1023)`.
pub fn decode_rotation(r: u32) -> [f32; 4] {
    let c = [0, 10, 20].map(|s| (((r >> s) & 1023) as f32 / 1023.0 * 2.0 - 1.0) * ROTATION_RANGE);
    let w = (1.0 - c[0] * c[0] - c[1] * c[1] - c[2] * c[2])
        .max(0.0)
        .sqrt();
    match r >> 30 {
        0 => [w, c[0], c[1], c[2]],
        1 => [c[0], w, c[1], c[2]],
        2 => [c[0], c[1], w, c[2]],
        _ => [c[0], c[1], c[2], w],
    }
}

/// Smallest-three code of a rotation (x, y, z, w; any length but zero, either sign): of the
/// codes nearest the stored components, the one whose decode is closest to `q`.
pub fn encode_rotation(q: [f32; 4]) -> u32 {
    let q = q.map(f64::from);
    let len = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    let q = q.map(|c| c / len);
    let largest = (0..4)
        .rev()
        .max_by(|&a, &b| q[a].abs().total_cmp(&q[b].abs()))
        .unwrap_or(3);
    let sign = if q[largest] < 0.0 { -1.0 } else { 1.0 };
    let mut lo = [0u32; 3];
    for (slot, i) in (0..4).filter(|&i| i != largest).enumerate() {
        let v = (q[i] * sign / f64::from(ROTATION_RANGE) * 0.5 + 0.5) * 1023.0;
        lo[slot] = v.floor().clamp(0.0, 1022.0) as u32;
    }
    let mut best = (f64::NEG_INFINITY, 0);
    for corner in 0..8u32 {
        let mut code = (largest as u32) << 30;
        for (slot, base) in lo.iter().enumerate() {
            code |= (base + ((corner >> slot) & 1)) << (10 * slot);
        }
        let d = decode_rotation(code);
        let dot = (0..4).map(|i| q[i] * f64::from(d[i])).sum::<f64>().abs();
        if dot > best.0 {
            best = (dot, code);
        }
    }
    best.1
}

/// Row `i` of exact axis rotation `p * 4 + s` has its non-zero entry in column
/// `AXIS_PERMUTATIONS[p][i]`.
const AXIS_PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];
const AXIS_PARITY: [f32; 6] = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0];

/// The matrix (rows) of exact axis rotation `index`, 0..24: `index = permutation * 4 +
/// signs`, bit 0 of `signs` negating row 0, bit 1 row 1, and row 2's sign making the
/// determinant +1. Index 0 is the identity.
pub fn axis_rotation(index: u32) -> [[f32; 3]; 3] {
    let p = (index >> 2) as usize % 6;
    let s0 = if index & 1 != 0 { -1.0 } else { 1.0 };
    let s1 = if index & 2 != 0 { -1.0 } else { 1.0 };
    let sign = [s0, s1, s0 * s1 * AXIS_PARITY[p]];
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        m[i][AXIS_PERMUTATIONS[p][i]] = sign[i];
    }
    m
}

/// The exact axis rotation whose matrix every entry of `q`'s rotation matrix is within
/// 1e-6 of.
pub fn exact_axis_index(q: [f32; 4]) -> Option<u32> {
    let m = quat_matrix_f64(q.map(f64::from));
    (0..24).find(|&i| {
        let a = axis_rotation(i);
        (0..3).all(|r| (0..3).all(|c| (m[r][c] - f64::from(a[r][c])).abs() <= 1e-6))
    })
}

/// Rotation matrix (rows) of a record's `rot` word.
pub fn instance_rotation(rot: u32, exact_axis: bool) -> [[f32; 3]; 3] {
    if exact_axis {
        axis_rotation(rot)
    } else {
        quat_matrix(decode_rotation(rot))
    }
}

/// Rotation matrix (rows) of a unit quaternion (x, y, z, w), in the operation order the
/// GPU decode uses.
pub fn quat_matrix(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [x, y, z, w] = q;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, yy, zz) = (x * x2, y * y2, z * z2);
    let (xy, xz, yz) = (x * y2, x * z2, y * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    [
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ]
}

fn quat_matrix_f64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let len2: f64 = q.iter().map(|c| c * c).sum();
    let [x, y, z, w] = q.map(|c| c / len2.sqrt());
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Morton code of a quantised position: bit `3 * b + a` holds bit `b` of axis `a`.
pub fn morton(pos: [u16; 3]) -> u64 {
    fn spread(v: u16) -> u64 {
        let mut x = u64::from(v);
        x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
        x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
        x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
        x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
        x = (x | (x << 2)) & 0x1249_2492_4924_9249;
        x
    }
    spread(pos[0]) | spread(pos[1]) << 1 | spread(pos[2]) << 2
}

/// Levels of a Morton code: 16 levels of 3 bits each.
pub const MORTON_LEVELS: u32 = 16;

/// Splits records sorted by Morton `codes` into the largest Morton-prefix cells whose summed
/// `costs` fit in `max`, returning each cell's first record. `None` when the records of one
/// position alone cost more than `max`.
pub fn partitions(codes: &[u64], costs: &[u32], max: u32) -> Option<Vec<u32>> {
    fn split(
        codes: &[u64],
        costs: &[u32],
        first: usize,
        level: u32,
        max: u32,
        out: &mut Vec<u32>,
    ) -> Option<()> {
        let cost: u64 = costs.iter().map(|&c| u64::from(c)).sum();
        if cost <= u64::from(max) {
            out.push(first as u32);
            return Some(());
        }
        if level == MORTON_LEVELS {
            return None;
        }
        let shift = 3 * (MORTON_LEVELS - 1 - level);
        let mut at = 0;
        while at < codes.len() {
            let child = (codes[at] >> shift) & 7;
            let end = at + codes[at..].partition_point(|c| (c >> shift) & 7 == child);
            split(
                &codes[at..end],
                &costs[at..end],
                first + at,
                level + 1,
                max,
                out,
            )?;
            at = end;
        }
        Some(())
    }
    let mut out = Vec::new();
    if !codes.is_empty() {
        split(codes, costs, 0, 0, max, &mut out)?;
    }
    Some(out)
}
