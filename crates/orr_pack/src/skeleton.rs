//! Skeleton class: one mount blob holding a [`SkeletonHeader`] and its tables ([`SkeletonInfo`]).
//!
//! Joints are ordered parents first. A joint's model-space transform is its parent's times
//! its local transform ([`Trs`]); a root joint's parent is [`SkeletonInfo::root`], which unlike
//! a joint may shear or collapse an axis. The skinning matrix of
//! joint `j`, the palette entry that [`crate::mesh::skin`] and GPU skinning read, is
//! `model[j] x inverse_binds[j]`: it moves a mesh vertex from the bind pose into the pose.
//! Mesh joint indices index this skeleton.

use bytemuck::{Pod, Zeroable};
use orr_math::{Affine3, Mat3, Quat, Vec3};

use crate::blob::{put_table, read, table};
use crate::{AffineRows, Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Parent of a root joint.
pub const NO_PARENT: u16 = u16::MAX;

/// Joints per skeleton: parents name joints in a u16, short of [`NO_PARENT`].
pub const MAX_JOINTS: usize = NO_PARENT as usize;

/// Joints of a skeleton that skins a mesh: its vertices name joints in a u8.
pub const MAX_SKINNED_JOINTS: usize = 256;

/// Blob header. Spans name tables in the blob: `first` is a byte offset, `count` a record
/// count, one record per joint.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct SkeletonHeader {
    pub joint_count: u32,
    pub _pad: u32,
    /// u16 parent per joint, [`NO_PARENT`] for roots; a parent comes before its children.
    pub parents: Span,
    /// Rest [`Trs`] per joint, relative to its parent.
    pub rest: Span,
    /// [`AffineRows`] per joint: model space to the joint's bind space.
    pub inverse_binds: Span,
    /// u64 [`name_hash`] per joint.
    pub name_hashes: Span,
    /// [`Span`] per joint: byte offset and length of its UTF-8 name in the blob.
    pub names: Span,
    /// Model space from the space of the root joints.
    pub root: AffineRows,
}

/// A local transform: scale and shear, then rotation, then translation, so any 3x4 matrix has
/// one, and interpolating its fields stays a transform. 52 B, f32 throughout.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Trs {
    /// Unit quaternion, xyzw.
    pub rotation: [f32; 4],
    pub translation: [f32; 3],
    pub scale: [f32; 3],
    /// The upper triangle of the stretch whose diagonal is `scale`: how far unit y moves along x,
    /// unit z along x and unit z along y.
    pub shear: [f32; 3],
}

impl Trs {
    pub const IDENTITY: Self = Self {
        rotation: [0.0, 0.0, 0.0, 1.0],
        translation: [0.0; 3],
        scale: [1.0; 3],
        shear: [0.0; 3],
    };

    pub fn to_affine(&self) -> Affine3 {
        let ([sx, sy, sz], [xy, xz, yz]) = (self.scale, self.shear);
        let stretch = Mat3::from_cols(
            Vec3::new(sx, 0.0, 0.0),
            Vec3::new(xy, sy, 0.0),
            Vec3::new(xz, yz, sz),
        );
        let rotation = Mat3::from_quat(Quat::from_array(self.rotation));
        Affine3::from_mat3_translation(rotation * stretch, Vec3::from(self.translation))
    }

    /// Translation, scale and shear lerped, rotation slerped on the shorter arc.
    pub fn interpolate(&self, b: &Self, u: f32) -> Self {
        let lerp = |a: f32, b: f32| a + (b - a) * u;
        Self {
            rotation: slerp(self.rotation, b.rotation, u),
            translation: std::array::from_fn(|i| lerp(self.translation[i], b.translation[i])),
            scale: std::array::from_fn(|i| lerp(self.scale[i], b.scale[i])),
            shear: std::array::from_fn(|i| lerp(self.shear[i], b.shear[i])),
        }
    }
}

/// Spherical interpolation of unit quaternions on the shorter arc; normalized lerp where
/// they are within about 1.8 degrees, where slerp loses precision.
pub fn slerp(a: [f32; 4], b: [f32; 4], u: f32) -> [f32; 4] {
    let mut d: f32 = (0..4).map(|i| a[i] * b[i]).sum();
    let sign = if d < 0.0 { -1.0 } else { 1.0 };
    d *= sign;
    let (wa, wb) = if d > 0.9995 {
        (1.0 - u, u)
    } else {
        let theta = d.acos();
        let s = theta.sin();
        (((1.0 - u) * theta).sin() / s, (u * theta).sin() / s)
    };
    let q: [f32; 4] = std::array::from_fn(|i| wa * a[i] + wb * sign * b[i]);
    let l = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    q.map(|v| v / l)
}

/// Joint name hash: the first 8 bytes of blake3 of the name, little-endian.
pub fn name_hash(name: &str) -> u64 {
    let h = blake3::hash(name.as_bytes());
    u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
}

/// A skeleton blob: the header and its tables.
#[derive(Clone, Debug, PartialEq)]
pub struct SkeletonInfo {
    /// Model space from the space of the root joints.
    pub root: Affine3,
    pub parents: Vec<u16>,
    pub rest: Vec<Trs>,
    pub inverse_binds: Vec<Affine3>,
    pub names: Vec<String>,
}

impl SkeletonInfo {
    /// Checks every table holds one record per joint, at most [`MAX_JOINTS`], and parents
    /// come first.
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let h: SkeletonHeader = read(blob, 0).ok_or(Error::Malformed("skeleton header"))?;
        let n = h.joint_count;
        if n as usize > MAX_JOINTS
            || [h.parents, h.rest, h.inverse_binds, h.name_hashes, h.names]
                .iter()
                .any(|s| s.count != n)
        {
            return Err(Error::Malformed("skeleton tables"));
        }
        let parents: Vec<u16> = table(blob, h.parents)?;
        if parents
            .iter()
            .enumerate()
            .any(|(j, &p)| p != NO_PARENT && usize::from(p) >= j)
        {
            return Err(Error::Malformed("skeleton parent after its child"));
        }
        let names = table::<Span>(blob, h.names)?
            .into_iter()
            .map(|s| {
                let bytes = (s.first as usize)
                    .checked_add(s.count as usize)
                    .and_then(|end| blob.get(s.first as usize..end))
                    .ok_or(Error::Malformed("joint name outside its blob"))?;
                String::from_utf8(bytes.to_vec()).map_err(|_| Error::Malformed("joint name"))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            root: h.root.into(),
            parents,
            rest: table(blob, h.rest)?,
            inverse_binds: table::<AffineRows>(blob, h.inverse_binds)?
                .into_iter()
                .map(Affine3::from)
                .collect(),
            names,
        })
    }

    /// Blob bytes. Panics unless every table has one record per joint.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.parents.len();
        assert!(
            self.rest.len() == n && self.inverse_binds.len() == n && self.names.len() == n,
            "skeleton tables differ in length"
        );
        let mut out = vec![0; size_of::<SkeletonHeader>()];
        let parents = put_table(&mut out, &self.parents);
        let rest = put_table(&mut out, &self.rest);
        let binds: Vec<AffineRows> = self.inverse_binds.iter().map(|&b| b.into()).collect();
        let inverse_binds = put_table(&mut out, &binds);
        let hashes: Vec<u64> = self.names.iter().map(|s| name_hash(s)).collect();
        let name_hashes = put_table(&mut out, &hashes);
        let names_at = out.len().next_multiple_of(4);
        let mut spans = Vec::with_capacity(n);
        let mut first = (names_at + n * size_of::<Span>()) as u32;
        for s in &self.names {
            spans.push(Span {
                first,
                count: s.len() as u32,
            });
            first += s.len() as u32;
        }
        let names = put_table(&mut out, &spans);
        self.names
            .iter()
            .for_each(|s| out.extend_from_slice(s.as_bytes()));
        let h = SkeletonHeader {
            joint_count: n as u32,
            _pad: 0,
            parents,
            rest,
            inverse_binds,
            name_hashes,
            names,
            root: self.root.into(),
        };
        out[..size_of::<SkeletonHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    pub fn joint_count(&self) -> usize {
        self.parents.len()
    }

    /// The first joint with this name.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Model-space transform of every joint from local transforms (a clip sample or
    /// [`Self::rest`]).
    pub fn model_pose(&self, locals: &[Trs], out: &mut [Affine3]) {
        assert!(locals.len() >= self.joint_count() && out.len() >= self.joint_count());
        for (j, &p) in self.parents.iter().enumerate() {
            let local = locals[j].to_affine();
            out[j] = match p {
                NO_PARENT => self.root * local,
                p => out[usize::from(p)] * local,
            };
        }
    }

    /// Skinning matrices from model-space joint transforms: `model[j] x inverse_binds[j]`.
    pub fn palette(&self, model: &[Affine3], out: &mut [Affine3]) {
        for (j, ib) in self.inverse_binds.iter().enumerate() {
            out[j] = model[j] * *ib;
        }
    }
}
