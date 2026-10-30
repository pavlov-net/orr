//! Producer entries: consumer kernels that emit clusters into orr-owned batches from scene
//! compute.

use orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::material::MaterialId;
use crate::scene::{InstanceFlags, TransformGroupId};
use crate::{ProducerBatch, ProducerClusterGpu, ProducerGpu, Result, TopologyGpu};

/// A producer entry of one scene, below `SceneDesc::producer_capacity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProducerId(pub u32);

/// When an entry's kernel emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProducerUpdate {
    /// Once, in the frame of `set_producer`; the clusters stay.
    Static,
    /// Every frame the scene is updated, from empty counts.
    EveryFrame,
}

/// A cluster shape emitted many times: `vertex_count` vertices and at most 128 triangles
/// of cluster-local corners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterTopology<'a> {
    pub vertex_count: u32,
    pub triangles: &'a [[u8; 3]],
}

/// A producer entry: batch capacities, what its clusters may refer to, and its transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProducerDesc<'a> {
    pub max_clusters: u32,
    pub max_vertices: u32,
    pub update: ProducerUpdate,
    /// Indexed by `emit_template_cluster`'s `topology`; at most 255.
    pub topologies: &'a [ClusterTopology<'a>],
    /// Indexed by each cluster's material; at most 256.
    pub materials: &'a [MaterialId],
    /// Group and index of the transform the batch's positions are relative to.
    pub transform: (TransformGroupId, u32),
    /// The batch has a buffer for last frame's positions.
    pub previous_positions: bool,
    /// The batch has one u32 user word per vertex.
    pub vertex_user: bool,
    pub flags: InstanceFlags,
    /// Never read by orr.
    pub user_tag: u32,
}

/// A set entry's GPU memory: its batch, and its topologies, triangles and materials.
pub(crate) struct Producer {
    pub update: ProducerUpdate,
    /// Held while the entry is set.
    pub _memory: Allocation<u8>,
}

/// What `set_producer` writes: the entry's memory, its records, and the bytes to upload
/// to the start of its memory (topology triangles, topology records, materials), a
/// multiple of 16 long.
pub(crate) struct Built {
    pub producer: Producer,
    pub record: ProducerGpu,
    pub batch: ProducerBatch,
    pub static_bytes: Vec<u8>,
    pub static_at: GpuPtr<u32>,
}

pub(crate) fn build(
    device: &Device,
    desc: &ProducerDesc<'_>,
    counts: GpuPtr<[u32; 2]>,
) -> Result<Built> {
    assert!(
        desc.topologies.len() < 256,
        "{} topologies; at most 255",
        desc.topologies.len()
    );
    assert!(
        desc.materials.len() <= 256,
        "{} materials; at most 256",
        desc.materials.len()
    );
    let mut triangles: Vec<u8> = Vec::new();
    let mut firsts = Vec::with_capacity(desc.topologies.len());
    for t in desc.topologies {
        assert!(
            t.triangles.len() <= 128 && t.vertex_count <= 256,
            "a topology holds at most 128 triangles and 256 vertices"
        );
        firsts.push(triangles.len() as u64);
        triangles.extend(t.triangles.iter().flatten());
    }
    let mut end = 0u64;
    // Each piece starts 16-aligned; take(0) is the end so far.
    let mut take = |bytes: u64| {
        let start = end;
        end = (end + bytes).next_multiple_of(16);
        start
    };
    let triangles_at = take(triangles.len() as u64);
    let topologies_at = take(size_of::<TopologyGpu>() as u64 * desc.topologies.len() as u64);
    let materials_at = take(4 * desc.materials.len() as u64);
    let static_len = take(0) as usize;
    let vertices = u64::from(desc.max_vertices);
    let clusters = take(size_of::<ProducerClusterGpu>() as u64 * u64::from(desc.max_clusters));
    let positions = take(12 * vertices);
    let previous = desc.previous_positions.then(|| take(12 * vertices));
    let attrs = take(8 * vertices);
    let vertex_user = desc.vertex_user.then(|| take(4 * vertices));
    let len = take(0);

    let memory = device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, len.max(16), "producer")?;
    let base = memory.gpu;
    let at = |offset: u64| base.offset(offset);
    let topologies: Vec<TopologyGpu> = desc
        .topologies
        .iter()
        .zip(firsts)
        .map(|(t, first)| TopologyGpu {
            triangles: at(triangles_at + first),
            tri_count: t.triangles.len() as u32,
            vertex_count: t.vertex_count,
        })
        .collect();
    let mut bytes = vec![0u8; static_len];
    let mut put = |offset: u64, data: &[u8]| {
        bytes[offset as usize..][..data.len()].copy_from_slice(data);
    };
    put(triangles_at, &triangles);
    put(topologies_at, bytemuck::cast_slice(&topologies));
    put(materials_at, bytemuck::cast_slice(desc.materials));
    let counts = counts.cast::<u32>();
    Ok(Built {
        record: ProducerGpu {
            materials: at(materials_at).cast(),
            material_count: desc.materials.len() as u32,
            group: desc.transform.0.0,
            group_index: desc.transform.1,
            flags: u32::from(desc.flags.0),
            user_tag: desc.user_tag,
            _pad: 0,
        },
        batch: ProducerBatch {
            clusters: at(clusters).cast(),
            positions: at(positions).cast(),
            previous: previous.map_or(GpuPtr::NULL, |o| at(o).cast()),
            attrs: at(attrs).cast(),
            vertex_user: vertex_user.map_or(GpuPtr::NULL, |o| at(o).cast()),
            cluster_count: counts,
            vertex_count: counts.offset(1),
            topologies: at(topologies_at).cast(),
            max_clusters: desc.max_clusters,
            max_vertices: desc.max_vertices,
        },
        static_bytes: bytes,
        static_at: base.cast(),
        producer: Producer {
            update: desc.update,
            _memory: memory,
        },
    })
}
