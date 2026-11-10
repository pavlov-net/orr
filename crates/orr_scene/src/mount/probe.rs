//! What pixel probes read of the mounts: pack names and the meshes' DAG groups.

use std::sync::Arc;

use orr_pack::mesh::Group;
use orr_pack::{Archive, AssetId};

use super::{Entry, Mounts, Resident};
use crate::MountId;
use crate::scene::ModelId;

/// A model's mount, its mesh in that mount, the names of both in the pack and the mesh's
/// groups.
pub(crate) struct ProbeModel {
    pub(crate) mount: MountId,
    pub(crate) mesh: u32,
    pub(crate) model_name: Option<String>,
    pub(crate) mesh_name: Option<String>,
    pub(crate) groups: Arc<[Group]>,
}

fn name(archive: &Archive, id: AssetId) -> Option<String> {
    archive.find(id).map(|e| archive.name(e).to_owned())
}

impl Mounts {
    fn ready(&self) -> impl Iterator<Item = (MountId, &Resident)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e {
                Some(Entry::Ready(r)) => Some((MountId(i as u32), &**r)),
                _ => None,
            })
    }

    /// What a probe reports of `model`, a model of a ready mount or an edit of one; an edit's
    /// names are its source's.
    pub(crate) fn probe_model(&self, model: ModelId) -> Option<ProbeModel> {
        let &(mount, mesh) = self.owners.get(&model)?;
        let r = self.resident(mount).ok()?;
        let archive = &r.source.archive;
        let source = &r.edit_sources.get(mesh as usize)?.source;
        let model_name = r
            .models
            .iter()
            .find(|&(_, &m)| m == model)
            .and_then(|(&id, _)| name(archive, id));
        let mesh_name = archive
            .entries()
            .get(source.entry as usize)
            .map(|e| archive.name(e).to_owned());
        Some(ProbeModel {
            mount,
            mesh,
            model_name,
            mesh_name,
            groups: Arc::clone(&source.groups),
        })
    }

    /// The pack name of material `material`.
    pub(crate) fn material_name(&self, material: u32) -> Option<String> {
        self.ready().find_map(|(_, r)| {
            let (&id, _) = r.materials.iter().find(|&(_, m)| m.0 == material)?;
            name(&r.source.archive, id)
        })
    }

    /// The pack name of the texture of heap index `index`, a tile of a tile set among them.
    pub(crate) fn texture_name(&self, index: u32) -> Option<String> {
        self.ready().find_map(|(_, r)| {
            let t = r.placed_textures.iter().find(|t| t.slot.index.0 == index)?;
            name(&r.source.archive, t.id)
        })
    }
}
