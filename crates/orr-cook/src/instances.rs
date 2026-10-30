//! Instance page cook: world placements grouped by cell, quantised, Morton-ordered and cut
//! into PTLAS partitions.

use std::collections::{BTreeMap, HashMap, HashSet};

use bytemuck::Zeroable;
use glam::{DVec3, Quat, Vec3};
use orr_pack::instance::{
    self, InstancePageHeader, InstancePageInfo, InstanceRecord, MAX_HIDDEN_SECTION, MAX_POSE,
    MAX_RECORDS, PARTITION_INSTANCES, Prototype, SectionOverride,
};
use orr_pack::{ArchiveWriter, AssetId, BlobIn, Class, Hash32, Span};
use serde::Deserialize;

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
    /// Pieces that join without cracks: every placement sits on the coarsest grid step of the
    /// page's meshes, has scale 1 and a rotation that is a multiple of 90 degrees about the
    /// axes, and decodes exactly.
    pub joined: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    /// World position, metres.
    pub position: DVec3,
    pub rotation: Quat,
    /// Per-axis scale, applied before the rotation; every component above zero.
    pub scale: Vec3,
    /// Index into [`Instances::prototypes`].
    pub prototype: u16,
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
    pub instances: u32,
    pub sections: u32,
    /// Grid step 2^k of the model's mesh.
    pub position_exponent: i8,
}

#[derive(Default)]
pub(crate) struct Catalog {
    pub models: HashMap<String, ModelFacts>,
    pub materials: HashSet<String>,
}

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
        let c = (p.position / edge).floor();
        if c.abs().max_element() > f64::from(i32::MAX) {
            return Err(fail(
                name,
                format!("placement {i} is outside the cell range"),
            ));
        }
        rel.push(p.position - c * edge);
        cells.entry(c.as_ivec3().to_array()).or_default().push(i);
    }

    let mut cells: Vec<_> = cells.into_iter().collect();
    cells.sort_by_key(|(c, _)| morton(*c));
    for (cell, records) in &cells {
        let pages = split_cell(&rel, records, edge).ok_or_else(|| crowded(name, *cell))?;
        for (page, members) in pages.into_iter().enumerate() {
            let built = build_page(name, src, &prototypes, *cell, &rel, &members)?;
            let info = built.info.to_bytes();
            let mut blobs = vec![
                mount(&info),
                BlobIn::new(bytemuck::cast_slice(&built.records)),
            ];
            if stride > 0 {
                blobs.push(BlobIn::new(&built.user));
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
            let facts = *catalog
                .models
                .get(&p.model)
                .ok_or_else(|| Error::UnknownModel {
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
                        if !catalog.materials.contains(m) {
                            return Err(Error::UnknownMaterial {
                                asset: name.into(),
                                material: m.clone(),
                            });
                        }
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
                    _pad: 0,
                    sections: Span::default(),
                },
                sections,
                facts,
            })
        })
        .collect()
}

fn check_records(name: &str, src: &Instances) -> Result<()> {
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
        if !p.position.is_finite() {
            return Err(fail(name, format!("placement {i} position is not finite")));
        }
        if !(p.scale.is_finite() && p.scale.min_element() > 0.0) {
            return Err(fail(
                name,
                format!("placement {i} scale {} is not positive", p.scale),
            ));
        }
        if !(p.rotation.is_finite() && p.rotation.length_squared() > 1e-12) {
            return Err(fail(
                name,
                format!("placement {i} rotation is not a rotation"),
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
    h.update(format!("{:?}", src.prototypes).as_bytes());
    for p in &src.records {
        h.update(bytemuck::cast_slice(&p.position.to_array()));
        h.update(bytemuck::cast_slice(&p.rotation.to_array()));
        h.update(bytemuck::cast_slice(&p.scale.to_array()));
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
    user: Vec<u8>,
}

fn build_page(
    name: &str,
    src: &Instances,
    prototypes: &[Resolved],
    cell: [i32; 3],
    rel: &[DVec3],
    members: &[usize],
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
    let page_prototypes = used
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
        log2_scale_min: 0.0,
        log2_scale_max: 0.0,
        prototypes: Span::default(),
        user_stride: src.user_stride,
        _pad: 0,
        partitions: Span::default(),
    };
    let positions: Vec<[u16; 3]> = if src.joined {
        let k = used
            .iter()
            .map(|&g| prototypes[usize::from(g)].facts.position_exponent)
            .max()
            .unwrap_or(0);
        joined_positions(name, &mut header, f64::from(k).exp2(), rel, members)?
    } else {
        quantise_positions(&mut header, rel, members)
    };
    if !src.joined {
        set_scale_range(&mut header, members.iter().map(|&i| src.records[i].scale));
    }

    let mut records = Vec::with_capacity(members.len());
    for (n, &i) in members.iter().enumerate() {
        let p = &src.records[i];
        let q = p.rotation.normalize().to_array();
        let exact = instance::exact_axis_index(q);
        let (rot, mut scale) = match exact {
            Some(index) => (index, InstanceRecord::EXACT_AXIS),
            None => (instance::encode_rotation(q), 0),
        };
        if src.joined {
            if exact.is_none() {
                return Err(fail(
                    name,
                    format!("joined placement {i} rotation is not a multiple of 90 degrees"),
                ));
            }
            if (p.scale - Vec3::ONE).abs().max_element() > 1e-6 {
                return Err(fail(
                    name,
                    format!("joined placement {i} has scale {}, not 1", p.scale),
                ));
            }
        } else {
            for (a, s) in p.scale.to_array().into_iter().enumerate() {
                scale |= scale_code(&header, s) << (10 * a);
            }
        }
        records.push(InstanceRecord {
            pos: positions[n],
            prototype: local[&p.prototype],
            rot,
            scale,
        });
    }

    // Morton order, ties in source order.
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by_key(|&n| (instance::morton(records[n].pos), n));
    let records: Vec<InstanceRecord> = order.iter().map(|&n| records[n]).collect();
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

    if src.joined {
        check_joined(name, &header, &records, rel, &order, members)?;
    }
    Ok(BuiltPage {
        info: InstancePageInfo {
            header,
            prototypes: page_prototypes,
            sections,
            partitions,
        },
        records,
        user,
    })
}

fn bounds(rel: &[DVec3], members: &[usize]) -> (DVec3, DVec3) {
    members
        .iter()
        .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), &i| {
            (lo.min(rel[i]), hi.max(rel[i]))
        })
}

/// Sets the header's box from the members' bounds and returns each member's grid position,
/// the nearest of the decoded candidates.
fn quantise_positions(
    h: &mut InstancePageHeader,
    rel: &[DVec3],
    members: &[usize],
) -> Vec<[u16; 3]> {
    let (lo, hi) = bounds(rel, members);
    for a in 0..3 {
        let (lo, hi) = (lo[a], hi[a]);
        let mut o = lo as f32;
        if f64::from(o) > lo {
            o = o.next_down();
        }
        let mut e = (hi - f64::from(o)) as f32;
        while f64::from(o) + f64::from(e) < hi {
            e = e.next_up();
        }
        h.origin[a] = o;
        h.extent[a] = e;
    }
    members
        .iter()
        .map(|&i| {
            let target = rel[i].to_array();
            let mut q = [0u16; 3];
            for a in 0..3 {
                let step = h.extent[a] / 65535.0;
                if step == 0.0 {
                    continue;
                }
                let guess = ((target[a] - f64::from(h.origin[a])) / f64::from(step)).round();
                q[a] = nearest_code(guess, 65535, |c| {
                    let mut pos = [0; 3];
                    pos[a] = c as u16;
                    (f64::from(instance::decode_position(h, pos)[a]) - target[a]).abs()
                }) as u16;
            }
            q
        })
        .collect()
}

/// Of `guess` and its two neighbours, clamped to `0..=max`, the code with the least `err`.
fn nearest_code(guess: f64, max: u32, err: impl Fn(u32) -> f64) -> u32 {
    let g = guess.clamp(0.0, f64::from(max)) as u32;
    (g.saturating_sub(1)..=(g + 1).min(max))
        .min_by(|&a, &b| err(a).total_cmp(&err(b)))
        .unwrap_or(g)
}

fn set_scale_range(h: &mut InstancePageHeader, scales: impl Iterator<Item = Vec3>) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for s in scales {
        for v in s.to_array() {
            let l = v.log2();
            lo = lo.min(l);
            hi = hi.max(l);
        }
    }
    h.log2_scale_min = lo;
    h.log2_scale_max = hi;
}

fn scale_code(h: &InstancePageHeader, s: f32) -> u32 {
    let range = h.log2_scale_max - h.log2_scale_min;
    if range == 0.0 {
        return 0;
    }
    let guess = f64::from((s.log2() - h.log2_scale_min) / range * 1023.0).round();
    nearest_code(guess, 1023, |c| {
        (f64::from(instance::decode_scale(h, c)) / f64::from(s) - 1.0).abs()
    })
}

/// Joined pages quantise on the coarsest mesh step: the box starts on that grid and spans
/// 65535 steps.
fn joined_positions(
    name: &str,
    h: &mut InstancePageHeader,
    step: f64,
    rel: &[DVec3],
    members: &[usize],
) -> Result<Vec<[u16; 3]>> {
    let (lo, hi) = bounds(rel, members);
    for a in 0..3 {
        let (lo, hi) = (lo[a], hi[a]);
        let origin = (lo / step + 1e-3).floor() * step;
        if ((hi - origin) / step).round() > 65535.0 {
            return Err(fail(
                name,
                format!(
                    "joined placements span more than 65535 steps of {step} m in cell {:?}",
                    h.cell
                ),
            ));
        }
        h.origin[a] = origin as f32;
        h.extent[a] = (step * 65535.0) as f32;
        if f64::from(h.origin[a]) != origin {
            return Err(fail(
                name,
                format!("joined page origin {origin} m is not exact in f32"),
            ));
        }
    }
    members
        .iter()
        .map(|&i| {
            let mut q = [0u16; 3];
            for a in 0..3 {
                let v = (rel[i][a] - f64::from(h.origin[a])) / step;
                if (v - v.round()).abs() > 1e-3 {
                    return Err(fail(
                        name,
                        format!("joined placement {i} is off the {step} m grid"),
                    ));
                }
                q[a] = v.round() as u16;
            }
            Ok(q)
        })
        .collect()
}

/// Decodes every joined record with the reference decode: translations on the step grid at
/// the placement, unit scale and a linear part of 0 and +-1 entries.
fn check_joined(
    name: &str,
    h: &InstancePageHeader,
    records: &[InstanceRecord],
    rel: &[DVec3],
    order: &[usize],
    members: &[usize],
) -> Result<()> {
    let step = f64::from(h.extent[0]) / 65535.0;
    for (r, &n) in records.iter().zip(order) {
        let i = members[n];
        let m = instance::decode_record(h, r);
        for (row, want) in m.rows.iter().zip(rel[i].to_array()) {
            let t = f64::from(row[3]);
            let exact = t / step == (t / step).round() && (t - want).abs() <= step * 1e-3;
            let linear = row[..3].iter().all(|&v| v == 0.0 || v.abs() == 1.0);
            if !exact || !linear {
                return Err(fail(
                    name,
                    format!("joined placement {i} does not decode exactly onto the grid"),
                ));
            }
        }
    }
    Ok(())
}
