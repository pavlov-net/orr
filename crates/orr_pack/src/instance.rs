//! Instance page class: placements of cooked models inside one world cell.
//!
//! - Blob 0 (mount): an [`InstancePageHeader`], then its [`Prototype`] table, the
//!   [`SectionOverride`] table the prototypes point into, and the partition table (u16 first
//!   record per partition). Spans are byte offsets in blob 0 and record counts.
//! - Blob 1: [`InstancePageHeader::record_count`] [`InstanceRecord`]s in Morton order of
//!   [`InstanceRecord::pos`], at most [`MAX_RECORDS`], then the page's [`LinearRows`] table, so
//!   both stream and stay resident together. The records of a prototype rank
//!   `0..`[`Prototype::records`] in that order ([`InstanceRecord::rank`]).
//! - Blob 2, when `user_stride` is not 0: `user_stride` bytes of consumer data per record,
//!   in record order.
//! - Then on an animated page a blob per window of its [`MotionRow`]s
//!   ([`InstancePageHeader::motion_blob`]), which stream by the page's clock.
//!
//! A record decodes to a transform relative to the page cell (edge `2^cell_log2` m):
//! translation `origin + pos * (extent / 65535)`, and a linear part. A compact record holds a
//! rotation in `rot` and per-axis scale `2^(lerp(log2_scale_min, log2_scale_max, code / 1023))`
//! applied before it, its x negated on a mirrored record ([`InstanceRecord::MIRROR`]). A
//! transform no rotation and scale hold, a sheared one or one scaled far from the page's other
//! records, is a row of the table instead, stored exactly and shared by every record with the same
//! linear part.
//! [`decode`] is the reference both for the GPU decode and for consumers' CPU queries.
//!
//! An animated page moves its records over time: [`InstancePageHeader::motion_times`] starts one
//! interval each, and every interval holds a [`MotionRow`] per record, interval-major, which
//! carries the record from the interval's start until the next. A window is
//! [`InstancePageHeader::motion_window`] intervals in a row, about [`MOTION_WINDOW_BYTES`], so what
//! must be resident at one time does not grow with the length of the animation.
//! [`motion_interval`] finds the interval of a time, and [`motion_transform`] is the reference for
//! the kernel that evaluates a row. The first interval runs back from the second without end, and
//! the last on.
//! An instance needs a transform that inverts, so a scale of 0 on an axis stands at
//! [`MOTION_FLAT`], which draws the same, and one of 0 on all three hides the instance.

use bytemuck::{Pod, Zeroable};
use orr_math::{Affine3, DMat3, DQuat, DVec4, Mat3, Quat, Vec3};

use crate::blob::{put_table, read, table};
use crate::{AffineRows, AssetId, Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Records per page.
pub const MAX_RECORDS: usize = 16_384;

/// PTLAS instances per partition.
pub const PARTITION_INSTANCES: u32 = 1000;

/// Pose variants a prototype can name.
pub const MAX_POSE: u16 = 256;

/// Sections a prototype can hide: sections `0..MAX_HIDDEN_SECTION`.
pub const MAX_HIDDEN_SECTION: usize = 16;

/// The least size of a scale on an axis of a [`MotionRow`].
pub const MOTION_FLAT: f32 = 1.0 / (1u64 << 60) as f32;

/// What a blob of [`MotionRow`]s aims at; one interval of a full page is under it.
pub const MOTION_WINDOW_BYTES: usize = 2 << 20;
const _: () = assert!(MAX_RECORDS * size_of::<MotionRow>() <= MOTION_WINDOW_BYTES);

/// Blob 0 header, 196 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct InstancePageHeader {
    /// World cell of the page.
    pub cell: [i32; 3],
    /// Cell-relative box the positions quantize over.
    pub origin: [f32; 3],
    pub extent: [f32; 3],
    pub log2_scale_min: f32,
    pub log2_scale_max: f32,
    /// [`Prototype`] table; records name an entry by index.
    pub prototypes: Span,
    /// Bytes per record in blob 2; 0 when the page has no user stream.
    pub user_stride: u16,
    /// Records in blob 1, at most [`MAX_RECORDS`]; the rest of the blob is the linear table.
    pub record_count: u16,
    /// u16 first record of each PTLAS partition, ascending, the first 0. Partition `p`
    /// covers records `[first[p], first[p + 1])`, the last up to the record count.
    pub partitions: Span,
    /// The skeleton of the joint the records hang from, [`AssetId::NONE`] for none: they stand in
    /// the bind space of joint `frame_joint`, and its skinning matrix (a palette entry) poses them.
    pub frame_skeleton: AssetId,
    pub frame_joint: u32,
    /// f32 seconds each interval of an animated page starts at; empty on a page that holds still.
    pub motion_times: Span,
    /// A row's transform stands between these, relative to the page's cell:
    /// `motion_left * row * motion_right`.
    pub motion_left: AffineRows,
    pub motion_right: AffineRows,
    /// s of the clock of [`Self::motion_times`] that a page stands at until something sets it.
    pub motion_start: f32,
    /// Intervals in a window of rows, the last window up to the end.
    pub motion_window: u32,
}

impl InstancePageHeader {
    /// Blobs of [`MotionRow`]s.
    pub fn motion_windows(&self) -> u32 {
        match self.motion_times.count {
            0 => 0,
            n => n.div_ceil(self.motion_window),
        }
    }

    /// The blob of window `w`.
    pub fn motion_blob(&self, w: u32) -> u32 {
        2 + u32::from(self.user_stride != 0) + w
    }

    /// The bytes of window `w`.
    pub fn motion_window_bytes(&self, w: u32) -> u32 {
        let left = self.motion_times.count - w * self.motion_window;
        let intervals = left.min(self.motion_window);
        intervals * u32::from(self.record_count) * size_of::<MotionRow>() as u32
    }

    /// Rows of the linear table in a blob 1 of `len` bytes.
    pub fn linear_rows(&self, len: usize) -> Result<usize> {
        let records = usize::from(self.record_count) * size_of::<InstanceRecord>();
        len.checked_sub(records)
            .filter(|table| {
                usize::from(self.record_count) <= MAX_RECORDS
                    && table.is_multiple_of(size_of::<LinearRows>())
            })
            .map(|table| table / size_of::<LinearRows>())
            .ok_or(Error::Malformed("instance records"))
    }
}

/// A record of an animated page over one interval, `dt` seconds after the interval starts, 88 B:
/// position `p + dt (v + a dt / 2)`, the rotation by `spin` degrees a second about its direction,
/// and scale `s + g dt`. Linear interpolation is a velocity, and spherical a spin, so every way an
/// instance can move is this one formula. An instance not there has a rotation of all zeros.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MotionRow {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub acceleration: [f32; 3],
    /// Unit quaternion, xyzw.
    pub rotation: [f32; 4],
    pub spin: [f32; 3],
    pub scale: [f32; 3],
    pub growth: [f32; 3],
}

impl MotionRow {
    /// Scale, rotation and translation `dt` seconds on, `None` where the instance is not there or
    /// has no size.
    pub fn at(&self, dt: f32) -> Option<(Vec3, Quat, Vec3)> {
        if self.rotation == [0.0; 4] {
            return None;
        }
        let (p, v) = (Vec3::from(self.position), Vec3::from(self.velocity));
        let a = Vec3::from(self.acceleration);
        let spin = Vec3::from(self.spin);
        let q = Quat::from_array(self.rotation);
        let turn = (spin.length() * dt).to_radians();
        let turned = match spin.try_normalize() {
            Some(axis) => Quat::from_axis_angle(axis, turn) * q,
            None => q,
        };
        let s = Vec3::from(self.scale) + Vec3::from(self.growth) * dt;
        if s.abs().max_element() < MOTION_FLAT {
            return None;
        }
        let s = s.abs().max(Vec3::splat(MOTION_FLAT)).copysign(s);
        Some((s, turned, p + dt * (v + 0.5 * dt * a)))
    }
}

/// The interval of `times` that `t` falls in, and the seconds since it started.
pub fn motion_interval(times: &[f32], t: f32) -> (usize, f32) {
    let k = times[1..].partition_point(|&x| x <= t);
    (k, t - times[k])
}

/// The cell-relative transform of `row` `dt` seconds into its interval, `None` where the instance
/// is not there.
pub fn motion_transform(h: &InstancePageHeader, row: &MotionRow, dt: f32) -> Option<Affine3> {
    let (s, q, p) = row.at(dt)?;
    let trs = Affine3::from_scale_rotation_translation(s, q, p);
    Some(Affine3::from(h.motion_left) * trs * Affine3::from(h.motion_right))
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
    /// The page's records that name it; the prototypes' add up to the record count.
    pub records: u16,
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

/// One placement, 20 B, uploaded as stored.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct InstanceRecord {
    /// Position on the page's 65535-step grid over `origin..origin + extent`.
    pub pos: [u16; 3],
    pub prototype: u16,
    /// [`encode_rotation`] code, or with [`Self::INDEXED`] an index: below [`AXIS_ROTATIONS`]
    /// an [`axis_rotation`], from there on a row of the page's linear table.
    pub rot: u32,
    /// Bits 0-9, 10-19 and 20-29: x, y and z scale codes; bit 30 [`Self::INDEXED`], bit 31
    /// [`Self::MIRROR`]. A table row replaces the scale and mirror along with the rotation.
    pub scale: u32,
    /// The records before it that name its prototype.
    pub rank: u16,
    pub _pad: u16,
}

impl InstanceRecord {
    /// `rot` is an index, not a rotation code.
    pub const INDEXED: u32 = 1 << 30;
    /// The x scale is negated: the copy is mirrored.
    pub const MIRROR: u32 = 1 << 31;

    pub fn exact_axis(&self) -> bool {
        self.scale & Self::INDEXED != 0 && self.rot < AXIS_ROTATIONS
    }

    /// The row of the page's linear table that holds this record's linear part.
    pub fn linear_row(&self) -> Option<usize> {
        (self.scale & Self::INDEXED != 0)
            .then(|| self.rot.checked_sub(AXIS_ROTATIONS))
            .flatten()
            .map(|row| row as usize)
    }

    fn mirror(&self) -> bool {
        self.scale & Self::MIRROR != 0
    }

    fn scale_codes(&self) -> [u32; 3] {
        [0, 10, 20].map(|s| (self.scale >> s) & 1023)
    }
}

/// A row of a page's linear table: a 3 x 3 linear transform, row-major, 36 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct LinearRows {
    pub rows: [[f32; 3]; 3],
}

impl From<Mat3> for LinearRows {
    fn from(m: Mat3) -> Self {
        Self {
            rows: m.transpose().to_cols_array_2d(),
        }
    }
}

impl From<LinearRows> for Mat3 {
    fn from(l: LinearRows) -> Self {
        Mat3::from_cols_array_2d(&l.rows).transpose()
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
    /// [`InstancePageHeader::motion_times`].
    pub motion_times: Vec<f32>,
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
        let records: u32 = prototypes_out.iter().map(|p| u32::from(p.records)).sum();
        if records != u32::from(header.record_count) {
            return Err(Error::Malformed("instance page prototype records"));
        }
        let partitions: Vec<u16> = table(blob, header.partitions)?;
        if partitions.first().is_some_and(|&p| p != 0) || !partitions.is_sorted() {
            return Err(Error::Malformed("instance page partitions"));
        }
        let times = header.motion_times.count;
        let window = u64::from(header.motion_window.min(times))
            * u64::from(header.record_count)
            * size_of::<MotionRow>() as u64;
        if times != 0 && !(1..=u64::from(u32::MAX)).contains(&window) {
            return Err(Error::Malformed("instance page motion window"));
        }
        Ok(Self {
            motion_times: table(blob, header.motion_times)?,
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
        h.motion_times = put_table(&mut out, &self.motion_times);
        out[..size_of::<InstancePageHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    /// The overrides of prototype `p`.
    pub fn sections_of(&self, p: &Prototype) -> &[SectionOverride] {
        &self.sections[p.sections.first as usize..(p.sections.first + p.sections.count) as usize]
    }
}

/// Blob 1 of a page: its records and its linear table.
pub fn records(
    h: &InstancePageHeader,
    blob: &[u8],
) -> Result<(Vec<InstanceRecord>, Vec<LinearRows>)> {
    h.linear_rows(blob.len())?;
    let (records, table) = blob.split_at(usize::from(h.record_count) * size_of::<InstanceRecord>());
    let records: Vec<InstanceRecord> = bytemuck::pod_collect_to_vec(records);
    let table: Vec<LinearRows> = bytemuck::pod_collect_to_vec(table);
    if records
        .iter()
        .any(|r| r.linear_row().is_some_and(|row| row >= table.len()))
    {
        return Err(Error::Malformed("instance linear row"));
    }
    Ok((records, table))
}

/// Decodes a page: its cell and, per record in order, the prototype index and the
/// cell-relative transform.
pub fn decode(
    header: &[u8],
    blob: &[u8],
) -> Result<([i32; 3], impl Iterator<Item = (u16, Affine3)> + use<>)> {
    let h = InstancePageInfo::parse(header)?.header;
    let (records, table) = records(&h, blob)?;
    Ok((
        h.cell,
        records
            .into_iter()
            .map(move |r| (r.prototype, decode_record(&h, &table, &r))),
    ))
}

/// The rows of a window's blob, per interval a row per record.
pub fn motion_rows(blob: &[u8]) -> Vec<MotionRow> {
    bytemuck::pod_collect_to_vec(blob)
}

/// Cell-relative transform of one record of a page with linear table `table`: its table row,
/// or `R * diag(scale)` with the x scale negative on a mirrored record, and the translation.
pub fn decode_record(h: &InstancePageHeader, table: &[LinearRows], r: &InstanceRecord) -> Affine3 {
    let t = Vec3::from(decode_position(h, r.pos));
    if let Some(row) = r.linear_row() {
        let m = Mat3::from(table[row]);
        return Affine3::from_cols(m.x_axis, m.y_axis, m.z_axis, t);
    }
    let mut s = r.scale_codes().map(|c| decode_scale(h, c));
    if r.mirror() {
        s[0] = -s[0];
    }
    let m = instance_rotation(r.rot, r.exact_axis());
    Affine3::from_cols(m.x_axis * s[0], m.y_axis * s[1], m.z_axis * s[2], t)
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
    let q = DVec4::from(q.map(f64::from));
    let q = (q / q.length()).to_array();
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

/// Exact axis rotations.
pub const AXIS_ROTATIONS: u32 = 24;

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

/// The matrix of exact axis rotation `index`, 0..24: `index = permutation * 4 +
/// signs`, bit 0 of `signs` negating row 0, bit 1 row 1, and row 2's sign making the
/// determinant +1. Index 0 is the identity.
pub fn axis_rotation(index: u32) -> Mat3 {
    let p = (index >> 2) as usize % 6;
    let s0 = if index & 1 != 0 { -1.0 } else { 1.0 };
    let s1 = if index & 2 != 0 { -1.0 } else { 1.0 };
    let sign = [s0, s1, s0 * s1 * AXIS_PARITY[p]];
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        m[i][AXIS_PERMUTATIONS[p][i]] = sign[i];
    }
    Mat3::from_cols_array_2d(&m).transpose()
}

/// The exact axis rotation whose matrix every entry of `q`'s rotation matrix is within
/// 1e-6 of.
pub fn exact_axis_index(q: [f32; 4]) -> Option<u32> {
    let m = DMat3::from_quat(DQuat::from_array(q.map(f64::from)).normalize());
    (0..AXIS_ROTATIONS).find(|&i| m.abs_diff_eq(axis_rotation(i).as_dmat3(), 1e-6))
}

/// Rotation matrix of a record's `rot` word.
pub fn instance_rotation(rot: u32, exact_axis: bool) -> Mat3 {
    if exact_axis {
        axis_rotation(rot)
    } else {
        Mat3::from_quat(Quat::from_array(decode_rotation(rot)))
    }
}

/// Morton code of a quantized position: bit `3 * b + a` holds bit `b` of axis `a`.
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
const MORTON_LEVELS: u32 = 16;

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
