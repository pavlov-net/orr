//! What pixel probes read of the streamed units.

use super::Streaming;
use crate::MountId;
use crate::probe::{ProbeTile, UnitStatus};
use crate::residency::Key;
use crate::tiles::Tile;

impl Streaming {
    /// Keeps what each plan does with the requests it holds, while pixel probes wait.
    pub(crate) fn note_outcomes(&mut self, on: bool) {
        self.residency.note_outcomes(on);
    }

    /// Mesh page `page` of mesh `mesh` of `mount`, or with `node` its node page `page`; `None`
    /// when no such unit streams.
    pub(crate) fn page_status(
        &self,
        mount: MountId,
        mesh: u32,
        page: u32,
        node: bool,
    ) -> Option<UnitStatus> {
        let m = self.mounts.get(&mount)?;
        let pages = if node {
            m.node_pages.get(mesh as usize)?
        } else {
            m.mesh_pages.get(mesh as usize)?
        };
        let unit = *pages.get(page as usize)?;
        self.residency.status(Key { mount, unit })
    }

    /// The tile of mip `mip` of the streamed texture of heap index `index` over the mip 0 cell
    /// at `place` (0..1) of layer 0; `None` when the texture does not stream or the mip is in
    /// its tail.
    pub(crate) fn tile_status(&self, index: u32, mip: u32, place: [f32; 2]) -> Option<ProbeTile> {
        let (mount, t) = self.mounts.iter().find_map(|(&mount, m)| {
            m.textures
                .iter()
                .find(|t| t.stream.index.0 == index)
                .map(|t| (mount, t))
        })?;
        let grid = &t.stream.grid;
        if mip >= grid.tail_mip() {
            return None;
        }
        let cells = grid.cells();
        let cell = [0, 1].map(|a| ((place[a] * cells[a] as f32) as u32).min(cells[a] - 1));
        let tile: Tile = grid.covering(0, mip, cell);
        let i = grid.index_of(tile);
        let status = self.residency.status(Key {
            mount,
            unit: t.first_unit + i,
        })?;
        Some(ProbeTile {
            mip,
            xy: tile.xy,
            mapped: t.resident[i as usize],
            heard: t.heard[i as usize],
            status,
        })
    }
}
