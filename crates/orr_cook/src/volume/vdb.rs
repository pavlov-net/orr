//! OpenVDB float grids through vdb-rs.
//!
//! vdb-rs reads neither the root background nor root tiles, and reads inactive leaf values
//! of 0 where the file stores the background, so values follow `getValue` for a background
//! of 0.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use vdb_rs::{Grid, Index, Map, Node, VdbReader};

use super::{Block, SourceGrid, SourceVolume, volume_error};
use crate::Result;

const FLOAT_TREE: &str = "Tree_float_5_4_3";

/// Reads the grids `names` (empty: every float grid, sorted by name) of the VDB file bytes.
pub(super) fn read(
    asset: &str,
    path: &Path,
    bytes: &[u8],
    names: &[&str],
) -> Result<SourceVolume<'static>> {
    let fail = |m: String| volume_error(asset, format!("{}: {m}", path.display()));
    let read = catch_unwind(AssertUnwindSafe(|| read_grids(bytes, names)));
    let grids = match read {
        Ok(r) => r.map_err(fail)?,
        Err(p) => {
            let m = p
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("unsupported file");
            return Err(fail(format!("vdb-rs cannot read it: {m}")));
        }
    };
    let mut lattice = None;
    let mut out = Vec::with_capacity(grids.len());
    for g in grids {
        let name = g.descriptor.name.clone();
        let (origin, scale) = match &g.transform {
            Map::UniformScaleMap { scale_values, .. } => ([0.0; 3], scale_values),
            Map::ScaleTranslateMap {
                translation,
                scale_values,
                ..
            } => ([translation.x, translation.y, translation.z], scale_values),
        };
        let this = (
            origin.map(|o| o as f32),
            [scale.x, scale.y, scale.z].map(|s| s as f32),
        );
        if *lattice.get_or_insert(this) != this {
            return Err(fail(format!(
                "grid {name} has another transform than the grids before it"
            )));
        }
        let blocks = blocks(&g).map_err(|m| fail(format!("grid {name}: {m}")))?;
        out.push(SourceGrid {
            name,
            background: 0.0,
            blocks,
        });
    }
    let (origin, voxel_size) = lattice.unwrap_or_default();
    Ok(SourceVolume {
        origin,
        voxel_size,
        grids: out,
    })
}

fn read_grids(bytes: &[u8], names: &[&str]) -> Result<Vec<Grid<f32>>, String> {
    let mut r = VdbReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let is_float = |t: &str| t == FLOAT_TREE || t == format!("{FLOAT_TREE}_HalfFloat");
    let picked: Vec<String> = if names.is_empty() {
        let mut all: Vec<String> = r
            .grid_descriptors
            .values()
            .filter(|d| is_float(&d.grid_type))
            .map(|d| d.name.clone())
            .collect();
        all.sort();
        all
    } else {
        names.iter().map(|n| (*n).to_owned()).collect()
    };
    if picked.is_empty() {
        return Err("no float grid".into());
    }
    let mut grids = Vec::with_capacity(picked.len());
    for name in picked {
        let d = r
            .grid_descriptors
            .get(&name)
            .ok_or_else(|| format!("no grid named {name}"))?;
        if !is_float(&d.grid_type) {
            return Err(format!(
                "grid {name} is a {}, not a float grid",
                d.grid_type
            ));
        }
        if !d.instance_parent.is_empty() {
            return Err(format!(
                "grid {name} is an instance of {}",
                d.instance_parent
            ));
        }
        grids.push(r.read_grid::<f32>(&name).map_err(|e| e.to_string())?);
    }
    Ok(grids)
}

/// The grid's 8^3 blocks with a value other than 0: leaf buffers, and the tiles of
/// internal nodes filled out.
fn blocks(g: &Grid<f32>) -> Result<HashMap<[i32; 3], Cow<'static, Block>>, String> {
    let compact = g.descriptor.file_version < vdb_rs::OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION;
    // Files before OpenVDB's node mask compression store values only for entries without a
    // child.
    let slot = |idx: usize, children_before: usize| {
        if compact { idx - children_before } else { idx }
    };
    let mut out = HashMap::new();
    let mut fill = |min: [i32; 3], blocks_per_axis: i32, v: f32| -> Result<(), String> {
        if !v.is_finite() {
            return Err(format!("tile at {min:?} is not finite"));
        }
        if v == 0.0 {
            return Ok(());
        }
        let b = min.map(|m| m.div_euclid(8));
        for z in 0..blocks_per_axis {
            for y in 0..blocks_per_axis {
                for x in 0..blocks_per_axis {
                    out.insert([b[0] + x, b[1] + y, b[2] + z], Cow::Owned([v; 512]));
                }
            }
        }
        Ok(())
    };
    let coord = |c: vdb_rs::GlobalCoord| [c.0.x, c.0.y, c.0.z];
    let mut leaves = Vec::new();
    for n5 in &g.tree.root_nodes {
        for idx in 0..n5.child_mask.len() {
            if !n5.child_mask[idx] {
                let v = n5.data[slot(idx, n5.child_mask[..idx].count_ones())];
                fill(coord(n5.offset_to_global_coord(Index(idx as u32))), 16, v)?;
                continue;
            }
            let n4 = &n5.nodes[&(idx as u32)];
            for idx in 0..n4.child_mask.len() {
                if n4.child_mask[idx] {
                    leaves.push(&n4.nodes[&(idx as u32)]);
                } else {
                    let v = n4.data[slot(idx, n4.child_mask[..idx].count_ones())];
                    fill(coord(n4.offset_to_global_coord(Index(idx as u32))), 1, v)?;
                }
            }
        }
    }
    for leaf in leaves {
        if leaf.buffer.len() != 512 {
            return Err(format!(
                "leaf at {:?} holds {} values",
                leaf.origin,
                leaf.buffer.len()
            ));
        }
        let mut block = [0f32; 512];
        for (i, v) in block.iter_mut().enumerate() {
            let (x, y, z) = (i % 8, i / 8 % 8, i / 64);
            *v = leaf.buffer[(x << 6) | (y << 3) | z];
        }
        if block.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "leaf at {:?} holds a value that is not finite",
                leaf.origin
            ));
        }
        if block.iter().any(|&v| v != 0.0) {
            let o = [leaf.origin.x, leaf.origin.y, leaf.origin.z];
            out.insert(o.map(|m| m.div_euclid(8)), Cow::Owned(block));
        }
    }
    Ok(out)
}
