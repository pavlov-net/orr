//! Decals as the API takes them, their table records and the bin table that lists them by
//! world position.

use orr_math::{DVec3, IVec3, Mat3, Quat, Vec3};

use crate::material::MaterialId;
use crate::world::WorldPos;
use crate::{DecalBinGpu, DecalGpu};

/// A decal of a scene: its entry in the scene's decal table, which holds every entry up to
/// the highest id handed out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DecalId(pub u32);

/// The material parameters a decal writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecalChannels(pub u32);

impl DecalChannels {
    pub const BASE_COLOR: Self = Self(1);
    pub const NORMAL: Self = Self(2);
    pub const ROUGHNESS: Self = Self(4);
    pub const METALNESS: Self = Self(8);
    pub const EMISSION: Self = Self(16);
    pub const ALL: Self = Self(31);
}

impl std::ops::BitOr for DecalChannels {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// A box that projects `material` onto the surfaces inside it, in raster and at every ray
/// hit. The box's local x and y span the material's uv (u along +x, v down +y over [0, 1]);
/// it projects along its local -z onto surfaces facing +z. Its blend mask is `opacity` times
/// the material's opacity: `geometry_opacity` times its opacity texture, else its base color
/// alpha. Where decals overlap, those of higher `order` are written over those of lower, and
/// of equal orders those of higher ids.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decal {
    pub center: DVec3,
    pub rotation: Quat,
    /// m, above 0.
    pub half_extent: Vec3,
    pub material: MaterialId,
    pub channels: DecalChannels,
    pub opacity: f32,
    /// Radians between the surface's geometric normal and the box's +z: full strength up
    /// to the first, none from the second on.
    pub fade_angles: [f32; 2],
    /// The share of the half depth over which the decal fades out toward the box's z faces;
    /// 0 for none.
    pub depth_fade: f32,
    pub order: i32,
}

/// Bins are 2^4 = 16 m cubes, or world cells where those are smaller.
const BIN_LOG2: u32 = 4;

/// A decal touching more bins than this goes on the global list, which every lookup reads.
const MAX_BINS: i32 = 8;

pub(crate) fn bin_log2(cell_log2: u32) -> u32 {
    BIN_LOG2.min(cell_log2)
}

/// The table record of `decal`.
pub(crate) fn decal_gpu(decal: &Decal, cell_log2: u32) -> DecalGpu {
    let w = WorldPos::new(decal.center, cell_log2);
    let r = Mat3::from_quat(decal.rotation.normalize());
    let h = decal.half_extent.max(Vec3::splat(1e-6));
    let [full, none] = decal.fade_angles;
    DecalGpu {
        cell: w.cell.to_array(),
        material: decal.material.0,
        offset: w.offset.to_array(),
        channels: decal.channels.0,
        axis_x: r.x_axis.to_array(),
        inv_half_x: 1.0 / h.x,
        axis_y: r.y_axis.to_array(),
        inv_half_y: 1.0 / h.y,
        axis_z: r.z_axis.to_array(),
        inv_half_z: 1.0 / h.z,
        opacity: decal.opacity,
        cos_full: full.cos(),
        cos_none: none.cos(),
        depth_fade: decal.depth_fade,
    }
}

/// Where a decal is listed, what orders it among the others and its material, by which an
/// unmount finds the decals that name its mount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Listing {
    pub(crate) reach: Reach,
    pub(crate) order: i32,
    pub(crate) material: MaterialId,
}

/// The bins a decal's box touches, or the global list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reach {
    Bins { lo: IVec3, hi: IVec3 },
    Global,
}

pub(crate) fn listing(decal: &Decal, cell_log2: u32) -> Listing {
    Listing {
        reach: reach(decal, cell_log2),
        order: decal.order,
        material: decal.material,
    }
}

fn reach(decal: &Decal, cell_log2: u32) -> Reach {
    let r = Mat3::from_quat(decal.rotation.normalize());
    let h = decal.half_extent.as_dvec3();
    let reach = DVec3::new(
        r.row(0).abs().as_dvec3().dot(h),
        r.row(1).abs().as_dvec3().dot(h),
        r.row(2).abs().as_dvec3().dot(h),
    );
    let edge = f64::from(1u32 << bin_log2(cell_log2));
    let lo = ((decal.center - reach) / edge).floor().as_ivec3();
    let hi = ((decal.center + reach) / edge).floor().as_ivec3();
    let n = (hi - lo + IVec3::ONE).as_i64vec3();
    if n.x * n.y * n.z > i64::from(MAX_BINS) {
        Reach::Global
    } else {
        Reach::Bins { lo, hi }
    }
}

/// The bin table's starting slot for `bin`, as `decal_bin_hash` finds it.
fn hash(bin: IVec3, slots: u32) -> u32 {
    let b = bin.as_uvec3();
    (b.x.wrapping_mul(0x8da6_b343) ^ b.y.wrapping_mul(0xd816_3841) ^ b.z.wrapping_mul(0xcb1a_b31f))
        & (slots - 1)
}

/// The bin table and run list of `decals` (by id). Each list names a decal by its rank in
/// (order, id) order and its id, a pair of words, ascending by rank, which kernels merge the
/// global list and a bin's by: the global decals first, then per bin a count followed by its
/// decals.
pub(crate) struct Bins {
    pub(crate) slots: Vec<DecalBinGpu>,
    pub(crate) refs: Vec<u32>,
    pub(crate) globals: u32,
}

pub(crate) fn build_bins(decals: &[Option<Listing>]) -> Bins {
    let mut ranked: Vec<(i32, u32, Reach)> = decals
        .iter()
        .enumerate()
        .filter_map(|(id, l)| l.map(|l| (l.order, id as u32, l.reach)))
        .collect();
    ranked.sort_unstable_by_key(|r| (r.0, r.1));
    let mut refs: Vec<u32> = Vec::new();
    let mut by_bin: std::collections::BTreeMap<[i32; 3], Vec<u32>> = Default::default();
    for (rank, &(_, id, reach)) in ranked.iter().enumerate() {
        let entry = [rank as u32, id];
        match reach {
            Reach::Global => refs.extend(entry),
            Reach::Bins { lo, hi } => {
                for z in lo.z..=hi.z {
                    for y in lo.y..=hi.y {
                        for x in lo.x..=hi.x {
                            by_bin.entry([x, y, z]).or_default().extend(entry);
                        }
                    }
                }
            }
        }
    }
    let globals = refs.len() as u32 / 2;
    let count = if by_bin.is_empty() {
        0
    } else {
        (2 * by_bin.len() as u32).next_power_of_two()
    };
    let mut slots = vec![
        DecalBinGpu {
            bin: [0; 3],
            first: u32::MAX,
        };
        count as usize
    ];
    for (bin, ids) in by_bin {
        let mut at = hash(IVec3::from(bin), count);
        while slots[at as usize].first != u32::MAX {
            at = (at + 1) & (count - 1);
        }
        slots[at as usize] = DecalBinGpu {
            bin,
            first: refs.len() as u32,
        };
        refs.push(ids.len() as u32 / 2);
        refs.extend(ids);
    }
    Bins {
        slots,
        refs,
        globals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decal(center: DVec3, half: Vec3, rotation: Quat, order: i32) -> Decal {
        Decal {
            center,
            rotation,
            half_extent: half,
            material: MaterialId(0),
            channels: DecalChannels::ALL,
            opacity: 1.0,
            fade_angles: [1.0, 1.5],
            depth_fade: 0.0,
            order,
        }
    }

    /// The decals listed for the bin holding `p`, in the order `surface_decals_each` merges
    /// the global list and the bin's into.
    fn listed(b: &Bins, p: DVec3) -> Vec<u32> {
        let bin = (p / f64::from(1u32 << BIN_LOG2)).floor().as_ivec3();
        let globals = &b.refs[..2 * b.globals as usize];
        let mut run: &[u32] = &[];
        let n = b.slots.len() as u32;
        if n > 0 {
            let mut at = hash(bin, n);
            loop {
                let s = b.slots[at as usize];
                if s.first == u32::MAX {
                    break;
                }
                if s.bin == bin.to_array() {
                    let first = s.first as usize;
                    run = &b.refs[first + 1..first + 1 + 2 * b.refs[first] as usize];
                    break;
                }
                at = (at + 1) & (n - 1);
            }
        }
        let (mut a, mut c) = (globals.chunks(2).peekable(), run.chunks(2).peekable());
        let mut ids = Vec::new();
        loop {
            let ra = a.peek().map_or(u32::MAX, |e| e[0]);
            let rc = c.peek().map_or(u32::MAX, |e| e[0]);
            let next = if ra < rc { a.next() } else { c.next() };
            let Some(e) = next else { break };
            ids.push(e[1]);
        }
        ids
    }

    #[test]
    fn every_point_inside_a_decal_finds_it_in_its_bin_in_order() {
        let mut x = 0x2545_f491u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            f64::from(x >> 8) / f64::from(1u32 << 24)
        };
        let mut decals = Vec::new();
        for i in 0..300 {
            let c = DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 400.0;
            let half = if i % 50 == 0 {
                Vec3::splat(30.0)
            } else {
                Vec3::new(next() as f32, next() as f32, next() as f32) * 6.0 + 0.1
            };
            let q = Quat::from_euler(
                orr_math::EulerRot::XYZ,
                next() as f32 * 6.0,
                next() as f32 * 6.0,
                next() as f32 * 6.0,
            );
            decals.push(decal(c, half, q, i % 7 - 3));
        }
        let listings: Vec<Option<Listing>> = decals.iter().map(|d| Some(listing(d, 8))).collect();
        assert!(listings.iter().flatten().any(|l| l.reach == Reach::Global));
        let bins = build_bins(&listings);
        for (id, d) in decals.iter().enumerate() {
            let r = Mat3::from_quat(d.rotation).as_dmat3();
            for _ in 0..20 {
                let local = DVec3::new(next(), next(), next()) * 2.0 - 1.0;
                let p = d.center + r * (local * d.half_extent.as_dvec3());
                let ids = listed(&bins, p);
                assert!(ids.contains(&(id as u32)), "decal {id} missing at {p}");
                let key = |&i: &u32| (decals[i as usize].order, i);
                assert!(ids.is_sorted_by_key(key), "decals out of order at {p}");
            }
        }
    }
}
