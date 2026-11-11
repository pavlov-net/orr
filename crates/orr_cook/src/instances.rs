//! Instance page cook: world placements grouped by cell, quantized, Morton-ordered and cut
//! into PTLAS partitions.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use bytemuck::Zeroable;
use orr_math::{Affine3, DAffine3, DMat3, DMat4, DQuat, DVec3, Mat3, Quat, Vec3};
use orr_pack::instance::{
    self, InstancePageHeader, InstancePageInfo, InstanceRecord, MAX_HIDDEN_SECTION, MAX_POSE,
    MAX_RECORDS, MotionRow, PARTITION_INSTANCES, POSITION_BITS, Prototype, SectionOverride,
};
use orr_pack::{Aabb, AffineRows, ArchiveWriter, AssetId, BlobIn, Class, Hash32, Span};
use serde::Deserialize;

use crate::import::{collapses, column_lengths, largest};
use crate::{Error, Result, mount};

/// Placements of cooked models. The cook splits them into the archive's cells and writes one
/// instance page per cell, or several when a cell holds more than
/// [`instance::MAX_RECORDS`]; pages are named `N/<x>_<y>_<z>/<i>` after the cell.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Instances {
    pub records: Vec<Placement>,
    pub prototypes: Vec<PrototypeSpec>,
    /// `user_stride` bytes per record, kept with its record through the reordering.
    pub user: Vec<u8>,
    pub user_stride: u16,
    /// The joint the records hang from, in whose bind space they stand.
    pub frame: Option<PageFrame>,
    /// How the records move over time, which an importer works out.
    #[serde(skip)]
    pub motion: Option<Arc<InstanceMotion>>,
    /// Pieces that join without cracks: every placement sits on the coarsest grid step of the
    /// page's meshes, has scale 1 and a rotation that is a multiple of 90 degrees about the
    /// axes, and decodes exactly.
    pub joined: bool,
}

/// Records over time ([`orr_pack::instance`]).
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceMotion {
    /// Seconds each interval starts at.
    pub times: Vec<f32>,
    /// Per interval a row per record, interval-major, each position from its record's anchor.
    pub rows: Vec<MotionRow>,
    /// Per record a point between `left` and `right`, in f64: rows stay small, and a page brings
    /// them to its cell, where an f32 keeps what they move by.
    pub anchors: Vec<[f64; 3]>,
    /// What stands either side of a row: scene space, or `frame`'s bind space, from it.
    pub left: DMat4,
    pub right: DMat4,
    /// Seconds of `times` the stage was read at, where the copies stand until set.
    pub start: f32,
}

/// A joint of a skeleton entry.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageFrame {
    /// The skeleton's entry name.
    pub skeleton: String,
    pub joint: u32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    /// World position, meters.
    pub position: [f64; 3],
    /// Has to invert: a flat transform cannot be an instance. A model cooks once, before anything
    /// places it, so a world length of displacement baked into it scales with this.
    pub linear: Linear,
    /// Index into [`Instances::prototypes`].
    pub prototype: u16,
}

/// The linear part of a placement's transform.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Linear {
    Scaled {
        /// Quaternion, `[x, y, z, w]`.
        rotation: [f32; 4],
        /// Per-axis scale, applied before the rotation. An odd number of negative components
        /// mirrors the copy.
        scale: [f32; 3],
    },
    /// The columns of a matrix, which may shear.
    Matrix([[f32; 3]; 3]),
}

impl Placement {
    pub(crate) fn position(&self) -> DVec3 {
        DVec3::from(self.position)
    }

    pub(crate) fn linear(&self) -> DMat3 {
        match self.linear {
            Linear::Scaled { rotation, scale } => {
                let r = DQuat::from_array(rotation.map(f64::from)).normalize();
                DMat3::from_quat(r) * DMat3::from_diagonal(Vec3::from(scale).as_dvec3())
            }
            Linear::Matrix(columns) => Mat3::from_cols_array_2d(&columns).as_dmat3(),
        }
    }
}

/// The widest log2 scale range of a page's compact records, a factor of 64. One scale past it
/// would coarsen every other record on the page, so it takes a table row.
const SCALE_RANGE: f32 = 6.0;

/// How a record holds the linear transform of a placement.
enum Form {
    /// `R diag(s)`: a rotation and per-axis scale, the x scale negative on a mirrored copy,
    /// which a compact record holds if its decode stays within the copy's [`tolerance`].
    Compact(Quat, Vec3),
    /// The whole transform in a row of the page's exact table.
    Exact,
}

impl Form {
    /// The form of `l`, which [`collapses`] passes: `R diag(s)` with `R` the rotation nearest
    /// `l diag(1 / s)`, unless `s` is past the range of a float. What `R diag(s)` leaves out of
    /// `l`, a shear, counts in the record's error.
    fn of(l: DMat3) -> Self {
        let mut s = column_lengths(l);
        if l.determinant() < 0.0 {
            s.x = -s.x;
        }
        if !s.as_vec3().is_finite() {
            return Self::Exact;
        }
        let unit = l * DMat3::from_diagonal(s.recip());
        // The rotation nearest `unit`, by polar iteration.
        let mut r = unit;
        for _ in 0..16 {
            r = (r + r.inverse().transpose()) * 0.5;
        }
        Self::Compact(DQuat::from_mat3(&r).normalize().as_quat(), s.as_vec3())
    }
}

/// A model as the records place it.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrototypeSpec {
    /// Asset name of a model cooked by the same manifest.
    pub model: String,
    /// Instance flags, stored as given.
    pub flags: u32,
    /// Pose variant, below 256.
    pub pose: u16,
    /// Per mesh section, from section 0; sections past the end keep the model's material and
    /// stay visible. Empty uses the model as cooked.
    pub sections: Vec<SectionSpec>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SectionSpec {
    /// Asset name of a material cooked by the same manifest; `None` keeps the model's.
    pub material: Option<String>,
    /// Only sections 0..15 can be hidden.
    pub hidden: bool,
}

/// What instance pages need to know about the models the manifest cooks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelFacts {
    /// PTLAS instances one copy costs: its body plus one per animated part.
    pub(crate) instances: u32,
    pub(crate) sections: u32,
    /// Grid step 2^k of the model's mesh, the coarsest when it has several; `i8::MIN` for a
    /// subdivision cage.
    pub(crate) position_exponent: i8,
    /// Grid step 2^k of the finest of its meshes; `i8::MIN` with a subdivision cage, whose
    /// points are on no grid.
    pub(crate) finest_exponent: i8,
    /// Rest-pose bounds, model space.
    pub(crate) bounds: Aabb,
}

impl ModelFacts {
    /// The facts of a model holding the geometry of both: one body, the sections of both.
    pub(crate) fn join(self, o: Self) -> Self {
        let mut bounds = self.bounds;
        bounds.extend(o.bounds.min);
        bounds.extend(o.bounds.max);
        Self {
            instances: self.instances.max(o.instances),
            sections: self.sections + o.sections,
            position_exponent: self.position_exponent.max(o.position_exponent),
            finest_exponent: self.finest_exponent.min(o.finest_exponent),
            bounds,
        }
    }
}

/// How far a compact record of a copy of `facts` scaled by `s` may put a point of the model from
/// where its placement does: half the step of the model's finest grid as the copy scales it.
/// The cook snaps each vertex up to half a step from its source, so two copies that meet then
/// meet within one step, the error that snapping alone allows two vertices that coincide.
fn tolerance(facts: &ModelFacts, s: Vec3) -> f64 {
    0.5 * 2f64.powi(i32::from(facts.finest_exponent)) * f64::from(s.abs().min_element())
}

/// Facts of the models an instance source names, by asset name.
pub(crate) type Catalog = HashMap<String, ModelFacts>;

/// Writes the pages of `src`, cells in Morton order, every page under `key`.
pub(crate) fn cook(
    w: &mut ArchiveWriter,
    name: &str,
    src: &Instances,
    cell_log2: u8,
    catalog: &Catalog,
    key: Hash32,
) -> Result<()> {
    let prototypes = resolve_prototypes(name, src, catalog)?;
    check_records(name, src)?;
    let stride = usize::from(src.user_stride);

    let edge = f64::from(cell_log2).exp2();
    let mut cells: BTreeMap<[i32; 3], Vec<usize>> = BTreeMap::new();
    let mut rel = Vec::with_capacity(src.records.len());
    for (i, p) in src.records.iter().enumerate() {
        let c = (p.position() / edge).floor();
        if c.abs().max_element() > f64::from(i32::MAX) {
            return Err(fail(
                name,
                format!("placement {i} is outside the cell range"),
            ));
        }
        rel.push(p.position() - c * edge);
        cells.entry(c.as_ivec3().to_array()).or_default().push(i);
    }

    let mut cells: Vec<_> = cells.into_iter().collect();
    cells.sort_by_key(|(c, _)| morton(*c));
    for (cell, records) in &cells {
        let pages = split_cell(&rel, records, edge).ok_or_else(|| crowded(name, *cell))?;
        for (page, members) in pages.into_iter().enumerate() {
            let built = build_page(name, src, &prototypes, *cell, &rel, &members, edge)?;
            let info = built.info.to_bytes();
            let records = [
                bytemuck::cast_slice(&built.records),
                bytemuck::cast_slice(&built.exact),
            ]
            .concat();
            let mut blobs = vec![mount(&info), BlobIn::new(&records)];
            if stride > 0 {
                blobs.push(BlobIn::new(&built.user));
            }
            let window = built.info.header.motion_window as usize * members.len();
            for rows in built.motion.chunks(window) {
                blobs.push(BlobIn::new(bytemuck::cast_slice(rows)));
            }
            w.add(
                &format!("{name}/{}_{}_{}/{page}", cell[0], cell[1], cell[2]),
                Class::InstancePage as u16,
                instance::SCHEMA,
                key,
                &blobs,
            )?;
        }
    }
    Ok(())
}

fn fail(asset: &str, message: String) -> Error {
    Error::Instances {
        asset: asset.into(),
        message,
    }
}

fn crowded(asset: &str, cell: [i32; 3]) -> Error {
    fail(
        asset,
        format!(
            "copies at one position in cell {cell:?} need more than {PARTITION_INSTANCES} PTLAS instances"
        ),
    )
}

/// A prototype with its model's facts and its overrides resolved to ids.
struct Resolved {
    proto: Prototype,
    sections: Vec<SectionOverride>,
    facts: ModelFacts,
}

fn resolve_prototypes(name: &str, src: &Instances, catalog: &Catalog) -> Result<Vec<Resolved>> {
    src.prototypes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let prototype = i as u16;
            let facts = *catalog.get(&p.model).ok_or_else(|| Error::UnknownModel {
                asset: name.into(),
                model: p.model.clone(),
            })?;
            if p.pose >= MAX_POSE {
                return Err(Error::PoseVariant {
                    asset: name.into(),
                    prototype,
                    pose: p.pose,
                });
            }
            if let Some(section) = p
                .sections
                .iter()
                .enumerate()
                .skip(MAX_HIDDEN_SECTION)
                .find_map(|(s, spec)| spec.hidden.then_some(s))
            {
                return Err(Error::HiddenSection {
                    asset: name.into(),
                    prototype,
                    section,
                });
            }
            if p.sections.len() > facts.sections as usize {
                return Err(fail(
                    name,
                    format!(
                        "prototype {i} overrides {} sections; {} has {}",
                        p.sections.len(),
                        p.model,
                        facts.sections
                    ),
                ));
            }
            let mut sections = Vec::new();
            if !p.sections.is_empty() {
                sections.resize(facts.sections as usize, SectionOverride::zeroed());
                for (s, spec) in p.sections.iter().enumerate() {
                    if let Some(m) = &spec.material {
                        sections[s].material = AssetId::from_name(m);
                    }
                    sections[s].hidden = u8::from(spec.hidden);
                }
            }
            Ok(Resolved {
                proto: Prototype {
                    model: AssetId::from_name(&p.model),
                    flags: p.flags,
                    pose: p.pose,
                    records: 0,
                    sections: Span::default(),
                },
                sections,
                facts,
            })
        })
        .collect()
}

fn check_records(name: &str, src: &Instances) -> Result<()> {
    if let Some(m) = &src.motion
        && m.anchors.len() != src.records.len()
    {
        return Err(fail(
            name,
            format!(
                "motion holds {} anchors for {} records",
                m.anchors.len(),
                src.records.len()
            ),
        ));
    }
    if let Some(m) = &src.motion
        && (m.times.is_empty() || m.rows.len() != m.times.len() * src.records.len())
    {
        return Err(fail(
            name,
            format!(
                "motion holds {} rows, {} intervals of {} records need {}",
                m.rows.len(),
                m.times.len(),
                src.records.len(),
                m.times.len() * src.records.len()
            ),
        ));
    }
    if src.user.len() != src.records.len() * usize::from(src.user_stride) {
        return Err(fail(
            name,
            format!(
                "user stream holds {} bytes, {} records of {} need {}",
                src.user.len(),
                src.records.len(),
                src.user_stride,
                src.records.len() * usize::from(src.user_stride)
            ),
        ));
    }
    for (i, p) in src.records.iter().enumerate() {
        if usize::from(p.prototype) >= src.prototypes.len() {
            return Err(fail(
                name,
                format!("placement {i} names prototype {}", p.prototype),
            ));
        }
        if !p.position().is_finite() {
            return Err(fail(name, format!("placement {i} position is not finite")));
        }
        let l = p.linear();
        if !l.is_finite() || collapses(l) {
            return Err(fail(
                name,
                format!("placement {i} linear transform {l} does not invert"),
            ));
        }
    }
    Ok(())
}

/// Hash of the placements and settings; the models and materials enter keys through their
/// own keys.
pub(crate) fn input_hash(src: &Instances, cell_log2: u8) -> Hash32 {
    let mut h = blake3::Hasher::new();
    h.update(&[cell_log2, u8::from(src.joined)]);
    h.update(format!("{:?} {:?}", src.prototypes, src.frame).as_bytes());
    if let Some(m) = &src.motion {
        h.update(bytemuck::cast_slice(&m.times));
        h.update(bytemuck::cast_slice(&m.rows));
        h.update(bytemuck::cast_slice(&m.anchors));
        h.update(bytemuck::cast_slice(&m.left.to_cols_array()));
        h.update(bytemuck::cast_slice(&m.right.to_cols_array()));
        h.update(&m.start.to_le_bytes());
    }
    for p in &src.records {
        h.update(bytemuck::cast_slice(&p.position));
        h.update(bytemuck::cast_slice(&p.linear().to_cols_array()));
        h.update(&p.prototype.to_le_bytes());
    }
    h.update(&src.user_stride.to_le_bytes());
    h.update(&src.user);
    Hash32(*h.finalize().as_bytes())
}

/// Morton code of a cell, signed coordinates biased to unsigned.
fn morton(cell: [i32; 3]) -> u128 {
    let mut code = 0u128;
    let c = cell.map(|v| (v as u32) ^ 0x8000_0000);
    for bit in 0..32 {
        for (axis, v) in c.iter().enumerate() {
            code |= u128::from((v >> bit) & 1) << (3 * bit + axis);
        }
    }
    code
}

/// Commit order of an instance source's pages: cells in Morton order, then page index.
pub(crate) fn page_order(source: &str, page: &str) -> Option<(u128, u32)> {
    let rest = page.strip_prefix(source)?.strip_prefix('/')?;
    let (cell, index) = rest.split_once('/')?;
    let mut it = cell.split('_').map(str::parse::<i32>);
    let c = [it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?];
    Some((morton(c), index.parse().ok()?))
}

/// The records of one cell in pages of at most [`MAX_RECORDS`]: Morton-prefix cells of the
/// cell's own grid, consecutive ones packed into a page while they fit. `None` when more
/// than a page of records share one grid position.
fn split_cell(rel: &[DVec3], records: &[usize], edge: f64) -> Option<Vec<Vec<usize>>> {
    if records.len() <= MAX_RECORDS {
        return Some(vec![records.to_vec()]);
    }
    let q = |v: f64| (v / edge * 65536.0).floor().clamp(0.0, 65535.0) as u16;
    let mut order: Vec<(u64, usize)> = records
        .iter()
        .map(|&i| (instance::morton(rel[i].to_array().map(q)), i))
        .collect();
    order.sort_unstable();
    let codes: Vec<u64> = order.iter().map(|o| o.0).collect();
    let starts = instance::partitions(&codes, &vec![1; codes.len()], MAX_RECORDS as u32)?;
    let mut pages: Vec<Vec<usize>> = vec![Vec::new()];
    for (k, &s) in starts.iter().enumerate() {
        let end = starts.get(k + 1).map_or(codes.len(), |&e| e as usize);
        let run = &order[s as usize..end];
        if pages
            .last()
            .is_some_and(|p| p.len() + run.len() > MAX_RECORDS)
        {
            pages.push(Vec::new());
        }
        pages
            .last_mut()
            .expect("one page")
            .extend(run.iter().map(|o| o.1));
    }
    Some(pages)
}

struct BuiltPage {
    info: InstancePageInfo,
    records: Vec<InstanceRecord>,
    exact: Vec<AffineRows>,
    motion: Vec<MotionRow>,
    user: Vec<u8>,
}

fn build_page(
    name: &str,
    src: &Instances,
    prototypes: &[Resolved],
    cell: [i32; 3],
    rel: &[DVec3],
    members: &[usize],
    edge: f64,
) -> Result<BuiltPage> {
    let mut used: Vec<u16> = members.iter().map(|&i| src.records[i].prototype).collect();
    used.sort_unstable();
    used.dedup();
    let local: HashMap<u16, u16> = used
        .iter()
        .enumerate()
        .map(|(l, &g)| (g, l as u16))
        .collect();
    let mut sections = Vec::new();
    let mut page_prototypes: Vec<Prototype> = used
        .iter()
        .map(|&g| {
            let r = &prototypes[usize::from(g)];
            let first = sections.len() as u32;
            sections.extend_from_slice(&r.sections);
            Prototype {
                sections: Span {
                    first,
                    count: r.sections.len() as u32,
                },
                ..r.proto
            }
        })
        .collect();

    let mut header = InstancePageHeader {
        cell,
        origin: [0.0; 3],
        extent: [0.0; 3],
        step: [0.0; 3],
        log2_scale_min: 0.0,
        log2_scale_step: 0.0,
        prototypes: Span::default(),
        user_stride: src.user_stride,
        record_count: members.len() as u16,
        partitions: Span::default(),
        frame_skeleton: src
            .frame
            .as_ref()
            .map_or(AssetId::NONE, |f| AssetId::from_name(&f.skeleton)),
        frame_joint: src.frame.as_ref().map_or(0, |f| f.joint),
        motion_times: Span::default(),
        motion_left: AffineRows::from(Affine3::IDENTITY),
        motion_right: AffineRows::from(Affine3::IDENTITY),
        motion_start: 0.0,
        motion_window: 1,
    };
    let mut forms: Vec<Form> = members
        .iter()
        .map(|&i| Form::of(src.records[i].linear()))
        .collect();
    let facts = |n: usize| &prototypes[usize::from(src.records[members[n]].prototype)].facts;
    let targets: Vec<DVec3> = if src.joined {
        let k = used
            .iter()
            .map(|&g| prototypes[usize::from(g)].facts.position_exponent)
            .max()
            .unwrap_or(0);
        joined_targets(name, f64::from(k).exp2(), rel, members)?
    } else {
        members.iter().map(|&i| rel[i]).collect()
    };
    let positions = grid_positions(&mut header, &targets);
    // Where the source puts each copy, against which its record's decode is measured.
    let source: Vec<DAffine3> = members
        .iter()
        .map(|&i| DAffine3::from_mat3_translation(src.records[i].linear(), rel[i]))
        .collect();
    let rotations: Vec<Option<(u32, Option<u32>)>> = forms
        .iter()
        .map(|f| match f {
            Form::Compact(q, _) => {
                let q = q.to_array();
                let exact = instance::exact_axis_index(q);
                Some((exact.unwrap_or_else(|| instance::encode_rotation(q)), exact))
            }
            Form::Exact => None,
        })
        .collect();
    let compact = |n: usize, rotation: (u32, Option<u32>), s: Vec3| {
        let (rot, exact) = rotation;
        let mut scale = if exact.is_some() {
            InstanceRecord::INDEXED
        } else {
            0
        };
        if s.x < 0.0 {
            scale |= InstanceRecord::MIRROR;
        }
        let (pos, fine) = InstanceRecord::split_position(positions[n]);
        InstanceRecord {
            pos,
            prototype: local[&src.records[members[n]].prototype],
            rot,
            scale,
            rank: 0,
            fine,
        }
    };
    let error = |n: usize, placed: Affine3| {
        let placed = DAffine3::from_mat3_translation(
            placed.matrix3.as_dmat3(),
            placed.translation.as_dvec3(),
        );
        instance::displacement(&source[n], &placed, &facts(n).bounds)
    };
    if !src.joined {
        // Rotation and translation alone, at the source's scale: a copy they move too far takes
        // a row, and its scale stays out of the page's range.
        for (n, form) in forms.iter_mut().enumerate() {
            if let (Form::Compact(_, s), Some((rot, axis))) = (&*form, rotations[n]) {
                let m = instance::instance_rotation(rot, axis.is_some());
                let t = instance::decode_position(&header, positions[n]);
                let placed = Affine3::from_cols(
                    m.x_axis * s.x,
                    m.y_axis * s.y,
                    m.z_axis * s.z,
                    Vec3::from(t),
                );
                if error(n, placed) > tolerance(facts(n), *s) {
                    *form = Form::Exact;
                }
            }
        }
        keep_scale_range(&mut forms);
        set_scale_grid(
            &mut header,
            forms.iter().filter_map(|f| match f {
                Form::Compact(_, s) => Some(s.abs()),
                Form::Exact => None,
            }),
        );
    }

    let mut records = Vec::with_capacity(members.len());
    let mut exact = Vec::new();
    let mut rows: HashMap<[u32; 12], u32> = HashMap::new();
    let unjoined = |i: usize| {
        fail(
            name,
            format!(
                "joined placement {i} linear transform {} is not a multiple of 90 degrees at scale 1",
                src.records[i].linear()
            ),
        )
    };
    for (n, &i) in members.iter().enumerate() {
        let p = &src.records[i];
        let record = match (&forms[n], rotations[n]) {
            (Form::Compact(_, s), Some(rotation)) => {
                let mut r = compact(n, rotation, *s);
                if src.joined {
                    // Against the placement itself: a compact form drops a small shear.
                    let turned = rotation.1.map(|e| instance::axis_rotation(e).as_dmat3());
                    if !turned.is_some_and(|m| largest(p.linear() - m) <= 1e-6) {
                        return Err(unjoined(i));
                    }
                    check_joined(name, &header, &r, targets[n], i)?;
                    Some(r)
                } else {
                    for (a, v) in s.abs().to_array().into_iter().enumerate() {
                        r.scale |= scale_code(&header, v) << (10 * a);
                    }
                    let placed = instance::decode_record(&header, &[], &r);
                    (error(n, placed) <= tolerance(facts(n), *s)).then_some(r)
                }
            }
            _ if src.joined => return Err(unjoined(i)),
            _ => None,
        };
        let record = record.unwrap_or_else(|| {
            let placed = Affine3::from_mat3_translation(p.linear().as_mat3(), rel[i].as_vec3());
            let row = AffineRows::from(placed);
            let index = *rows.entry(bytemuck::cast(row)).or_insert_with(|| {
                exact.push(row);
                exact.len() as u32 - 1
            });
            let (pos, fine) = InstanceRecord::split_position(positions[n]);
            InstanceRecord {
                pos,
                prototype: local[&p.prototype],
                rot: instance::AXIS_ROTATIONS + index,
                scale: InstanceRecord::INDEXED,
                rank: 0,
                fine,
            }
        });
        records.push(record);
    }

    // Morton order, ties in source order.
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by_key(|&n| (instance::morton(records[n].pos), n));
    let mut records: Vec<InstanceRecord> = order.iter().map(|&n| records[n]).collect();
    for r in &mut records {
        let p = &mut page_prototypes[usize::from(r.prototype)];
        r.rank = p.records;
        p.records += 1;
    }
    let stride = usize::from(src.user_stride);
    let user: Vec<u8> = order
        .iter()
        .flat_map(|&n| {
            let at = members[n] * stride;
            &src.user[at..at + stride]
        })
        .copied()
        .collect();

    let codes: Vec<u64> = records.iter().map(|r| instance::morton(r.pos)).collect();
    let costs: Vec<u32> = records
        .iter()
        .map(|r| {
            prototypes[usize::from(used[usize::from(r.prototype)])]
                .facts
                .instances
        })
        .collect();
    let partitions = instance::partitions(&codes, &costs, PARTITION_INSTANCES)
        .ok_or_else(|| crowded(name, cell))?
        .into_iter()
        .map(|p| p as u16)
        .collect();

    // Rows follow their records, by interval, from a point of the page.
    let mut motion = Vec::new();
    let mut motion_times = Vec::new();
    if let Some(m) = &src.motion {
        let all = src.records.len();
        let origin = DVec3::from_array(m.anchors[members[0]]);
        for k in 0..m.times.len() {
            motion.extend(order.iter().map(|&n| {
                let i = members[n];
                let mut row = m.rows[k * all + i];
                let from = DVec3::from_array(row.position.map(f64::from));
                let at = from + DVec3::from_array(m.anchors[i]) - origin;
                row.position = at.as_vec3().to_array();
                row
            }));
        }
        motion_times.clone_from(&m.times);
        let corner = DVec3::from_array(cell.map(f64::from)) * edge;
        let left = DMat4::from_translation(-corner) * m.left * DMat4::from_translation(origin);
        header.motion_left = AffineRows::from(Affine3::from_mat4(left.as_mat4()));
        header.motion_right = AffineRows::from(Affine3::from_mat4(m.right.as_mat4()));
        header.motion_start = m.start;
        let interval = members.len() * size_of::<MotionRow>();
        header.motion_window = (instance::MOTION_WINDOW_BYTES / interval) as u32;
    }
    Ok(BuiltPage {
        info: InstancePageInfo {
            header,
            prototypes: page_prototypes,
            sections,
            partitions,
            motion_times,
        },
        records,
        exact,
        motion,
        user,
    })
}

/// Sets the header's box and grid from `targets` and returns each one's grid position, the
/// nearest grid point.
fn grid_positions(h: &mut InstancePageHeader, targets: &[DVec3]) -> Vec<[u32; 3]> {
    let (lo, hi) = targets
        .iter()
        .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), &t| {
            (lo.min(t), hi.max(t))
        });
    let last = (1u32 << POSITION_BITS) - 1;
    for a in 0..3 {
        let (origin, step) = grid(lo[a], hi[a], last);
        let mut e = (hi[a] - f64::from(origin)) as f32;
        while f64::from(origin) + f64::from(e) < hi[a] {
            e = e.next_up();
        }
        h.origin[a] = origin;
        h.step[a] = step;
        h.extent[a] = e;
    }
    targets
        .iter()
        .map(|t| {
            [0, 1, 2].map(|a| {
                let q = (t[a] - f64::from(h.origin[a])) / f64::from(h.step[a]);
                q.round().clamp(0.0, f64::from(last)) as u32
            })
        })
        .collect()
}

/// The first point and step of a grid of `last + 1` points from at or below `lo` to at or above
/// `hi`, with the finest power-of-two step whose points are all exact f32s, a normal float's at
/// the least.
fn grid(lo: f64, hi: f64, last: u32) -> (f32, f32) {
    let mut e = f32::MIN_EXP - 1;
    loop {
        let step = 2f64.powi(e);
        let origin = (lo / step).floor() * step;
        let end = origin + f64::from(last) * step;
        if end >= hi && origin.abs().max(end.abs()) < 2f64.powi(e + f32::MANTISSA_DIGITS as i32) {
            return (origin as f32, step as f32);
        }
        e += 1;
    }
}

/// Of `guess` and its two neighbors, clamped to `0..=max`, the code with the least `err`.
fn nearest_code(guess: f64, max: u32, err: impl Fn(u32) -> f64) -> u32 {
    let g = guess.clamp(0.0, f64::from(max)) as u32;
    (g.saturating_sub(1)..=(g + 1).min(max))
        .min_by(|&a, &b| err(a).total_cmp(&err(b)))
        .unwrap_or(g)
}

/// Gives a table row to every compact form outside the [`SCALE_RANGE`] that holds the most of
/// them, the first such range from the smallest scales up.
fn keep_scale_range(forms: &mut [Form]) {
    // Where a range may start and still hold each form's three scales.
    let starts: Vec<Option<(f32, f32)>> = forms
        .iter()
        .map(|f| match f {
            Form::Compact(_, s) => {
                let l = s.abs().to_array().map(f32::log2);
                let (lo, hi) = (l[0].min(l[1]).min(l[2]), l[0].max(l[1]).max(l[2]));
                Some((hi - SCALE_RANGE, lo)).filter(|(first, last)| first <= last)
            }
            Form::Exact => None,
        })
        .collect();
    let mut edges: Vec<(f32, i32)> = starts
        .iter()
        .flatten()
        .flat_map(|&(first, last)| [(first, 0), (last, 1)])
        .collect();
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let (mut open, mut most, mut start) = (0, 0, f32::NAN);
    for (at, closes) in edges {
        open += 1 - 2 * closes;
        if open > most {
            (most, start) = (open, at);
        }
    }
    for (n, f) in forms.iter_mut().enumerate() {
        let held = starts[n].is_some_and(|(first, last)| first <= start && start <= last);
        if matches!(f, Form::Compact(..)) && !held {
            *f = Form::Exact;
        }
    }
}

/// Sets the header's log2 scale grid to cover `scales`.
fn set_scale_grid(h: &mut InstancePageHeader, scales: impl Iterator<Item = Vec3>) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for s in scales {
        for v in s.to_array() {
            let l = f64::from(v).log2();
            lo = lo.min(l);
            hi = hi.max(l);
        }
    }
    if lo <= hi {
        (h.log2_scale_min, h.log2_scale_step) = grid(lo, hi, 1023);
    }
}

fn scale_code(h: &InstancePageHeader, s: f32) -> u32 {
    let guess = (f64::from(s).log2() - f64::from(h.log2_scale_min)) / f64::from(h.log2_scale_step);
    nearest_code(guess.round(), 1023, |c| {
        (f64::from(instance::decode_scale(h, c)) / f64::from(s) - 1.0).abs()
    })
}

/// The members' positions snapped to the coarsest mesh step of a joined page, which each must
/// already be on.
fn joined_targets(name: &str, step: f64, rel: &[DVec3], members: &[usize]) -> Result<Vec<DVec3>> {
    members
        .iter()
        .map(|&i| {
            let v = rel[i] / step;
            if (v - v.round()).abs().max_element() > 1e-3 {
                return Err(fail(
                    name,
                    format!("joined placement {i} is off the {step} m grid"),
                ));
            }
            Ok(v.round() * step)
        })
        .collect()
}

/// Decodes joined record `r` of placement `i` with the reference decode: its translation is
/// `target` exactly and its linear part has entries of 0 and +-1 alone.
fn check_joined(
    name: &str,
    h: &InstancePageHeader,
    r: &InstanceRecord,
    target: DVec3,
    i: usize,
) -> Result<()> {
    let m = instance::decode_record(h, &[], r);
    let linear = m
        .matrix3
        .to_cols_array()
        .iter()
        .all(|&v| v == 0.0 || v.abs() == 1.0);
    if m.translation.as_dvec3() != target || !linear {
        return Err(fail(
            name,
            format!("joined placement {i} does not decode exactly onto the grid"),
        ));
    }
    Ok(())
}
