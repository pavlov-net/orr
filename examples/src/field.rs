//! Producers the examples share as test-scene stand-ins: a heightfield tile as flat-grid
//! patches displaced in its own kernel from a height raster, procedural grass as template
//! clusters per blade topology, and a displaced-grid dielectric slab. Each is a producer
//! entry and a kernel (`shaders/terrain.slang`, `grass.slang`, `slab.slang`) the consumer
//! records in scene compute.

use orr::orr_gpu::{Allocation, Device, MemoryKind};
use orr::{
    ClusterTopology, MaterialId, Opacity, PipelineId, ProducerDesc, ProducerId, ProducerUpdate,
    SceneCompute, TransformGroupId,
};

use crate::shaders::{grass, slab, terrain};

/// Quads per side of a grid patch: 9 x 9 vertices, 128 triangles.
pub const PATCH_QUADS: u32 = 8;
const PATCH_SIDE: u32 = PATCH_QUADS + 1;
/// Blades per grass cluster.
const BLADES: u32 = 8;

/// The triangles of a grid patch, row-major.
pub fn patch_triangles() -> Vec<[u8; 3]> {
    let mut t = Vec::with_capacity(128);
    for z in 0..PATCH_QUADS {
        for x in 0..PATCH_QUADS {
            let v = |dx: u32, dz: u32| ((z + dz) * PATCH_SIDE + x + dx) as u8;
            t.push([v(0, 0), v(0, 1), v(1, 0)]);
            t.push([v(1, 0), v(0, 1), v(1, 1)]);
        }
    }
    t
}

/// The triangles of a grass cluster: eight tall blades of five vertices, or eight short
/// ones of three.
pub fn blade_triangles(tall: bool) -> Vec<[u8; 3]> {
    let (n, one): (u32, &[[u32; 3]]) = if tall {
        (5, &[[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    } else {
        (3, &[[0, 1, 2]])
    };
    (0..BLADES)
        .flat_map(|b| one.iter().map(move |t| t.map(|k| (b * n + k) as u8)))
        .collect()
}

/// A square height raster on the GPU, and the same heights on the CPU.
pub struct Heights {
    pub side: u32,
    pub spacing: f32,
    pub origin: [f32; 2],
    heights: Vec<f32>,
    samples: Allocation<f32>,
}

impl Heights {
    /// `side` x `side` samples of `f(x, z)`, `spacing` m apart from `origin`.
    pub fn new(
        device: &Device,
        side: u32,
        spacing: f32,
        origin: [f32; 2],
        f: impl Fn(f32, f32) -> f32,
    ) -> orr::Result<Self> {
        let heights: Vec<f32> = (0..side * side)
            .map(|i| {
                let x = origin[0] + (i % side) as f32 * spacing;
                let z = origin[1] + (i / side) as f32 * spacing;
                f(x, z)
            })
            .collect();
        let mut samples = device.gpu_malloc::<f32>(
            MemoryKind::Upload,
            u64::from(side * side),
            "height raster",
        )?;
        samples
            .as_mut_slice()
            .expect("upload is mapped")
            .copy_from_slice(&heights);
        Ok(Self {
            side,
            spacing,
            origin,
            heights,
            samples,
        })
    }

    /// The height at (x, z), bilinear like `field.slang`'s `height_at`.
    pub fn at(&self, x: f32, z: f32) -> f32 {
        let last = (self.side - 1) as f32;
        let px = ((x - self.origin[0]) / self.spacing).clamp(0.0, last);
        let pz = ((z - self.origin[1]) / self.spacing).clamp(0.0, last);
        let (ix, iz) = (
            (px as u32).min(self.side - 2),
            (pz as u32).min(self.side - 2),
        );
        let (fx, fz) = (px - ix as f32, pz - iz as f32);
        let h = |dx: u32, dz: u32| self.heights[((iz + dz) * self.side + ix + dx) as usize];
        let a = h(0, 0) + (h(1, 0) - h(0, 0)) * fx;
        let b = h(0, 1) + (h(1, 1) - h(0, 1)) * fx;
        a + (b - a) * fz
    }
}

/// The terrain's static entry over `transform`: `patches` x `patches` grid patches of
/// `patch_size` m from `corner`.
pub struct Terrain {
    pub patches: u32,
    pub patch_size: f32,
    pub corner: [f32; 2],
}

impl Terrain {
    pub fn desc<'a>(
        &self,
        topology: &'a [ClusterTopology<'a>],
        materials: &'a [MaterialId],
        transform: (TransformGroupId, u32),
    ) -> ProducerDesc<'a> {
        let clusters = self.patches * self.patches;
        ProducerDesc {
            max_clusters: clusters,
            max_vertices: clusters * PATCH_SIDE * PATCH_SIDE,
            update: ProducerUpdate::Static,
            topologies: topology,
            materials,
            transform,
            ..ProducerDesc::default()
        }
    }

    /// Records the terrain's kernel; a static entry emits in the frame it is set.
    pub fn record(
        &self,
        c: &mut SceneCompute<'_, '_>,
        id: ProducerId,
        kernel: PipelineId,
        heights: &Heights,
    ) {
        let root = c
            .put_root(&terrain::Root {
                scene: c.root,
                heights: heights.samples.gpu,
                side: heights.side,
                spacing: heights.spacing,
                origin_x: heights.origin[0],
                origin_z: heights.origin[1],
                producer: id.0,
                patches: self.patches,
                patch_size: self.patch_size,
                corner_x: self.corner[0],
                corner_z: self.corner[1],
                _pad0: 0,
                _pad1: 0,
            })
            .expect("terrain root");
        let groups = (self.patches * self.patches).div_ceil(64);
        c.cmd.dispatch(c.pipeline(kernel), root, [groups, 1, 1]);
    }
}

/// Grass over `cells` x `cells` cells of `cell_size` m from `corner`, one cluster of eight
/// blades per cell within `radius` m of the camera.
pub struct Grass {
    pub cells: u32,
    pub cell_size: f32,
    pub corner: [f32; 2],
    pub radius: f32,
    /// Clusters the entry holds.
    pub capacity: u32,
}

impl Grass {
    /// An every-frame entry with previous positions and vertex words; `topologies` are the
    /// tall and the short cluster.
    pub fn desc<'a>(
        &self,
        topologies: &'a [ClusterTopology<'a>],
        materials: &'a [MaterialId],
        transform: (TransformGroupId, u32),
    ) -> ProducerDesc<'a> {
        ProducerDesc {
            max_clusters: self.capacity,
            max_vertices: self.capacity * BLADES * 5,
            update: ProducerUpdate::EveryFrame,
            topologies,
            materials,
            transform,
            previous_positions: true,
            vertex_user: true,
            ..ProducerDesc::default()
        }
    }

    /// Records the grass kernel for a camera at (`camera[0]`, `camera[1]`) at wind times
    /// `time` and `previous_time`.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        c: &mut SceneCompute<'_, '_>,
        id: ProducerId,
        kernel: PipelineId,
        heights: &Heights,
        camera: [f32; 2],
        time: f32,
        previous_time: f32,
    ) {
        let root = c
            .put_root(&grass::Root {
                scene: c.root,
                heights: heights.samples.gpu,
                side: heights.side,
                spacing: heights.spacing,
                origin_x: heights.origin[0],
                origin_z: heights.origin[1],
                producer: id.0,
                cells: self.cells,
                cell_size: self.cell_size,
                corner_x: self.corner[0],
                corner_z: self.corner[1],
                camera_x: camera[0],
                camera_z: camera[1],
                radius: self.radius,
                time,
                previous_time,
                _pad0: 0,
            })
            .expect("grass root");
        let groups = (self.cells * self.cells).div_ceil(64);
        c.cmd.dispatch(c.pipeline(kernel), root, [groups, 1, 1]);
    }
}

/// A dielectric slab of `patches` x `patches` grid patches of `patch_size` m from `corner`
/// at `level` m, displaced by waves of `amplitude` m.
pub struct Slab {
    pub patches: u32,
    pub patch_size: f32,
    pub corner: [f32; 2],
    pub level: f32,
    pub amplitude: f32,
}

impl Slab {
    pub fn desc<'a>(
        &self,
        topology: &'a [ClusterTopology<'a>],
        materials: &'a [MaterialId],
        transform: (TransformGroupId, u32),
    ) -> ProducerDesc<'a> {
        let clusters = self.patches * self.patches;
        ProducerDesc {
            max_clusters: clusters,
            max_vertices: clusters * PATCH_SIDE * PATCH_SIDE,
            update: ProducerUpdate::EveryFrame,
            topologies: topology,
            materials,
            transform,
            opacity: Opacity::Dielectric,
            ..ProducerDesc::default()
        }
    }

    /// Records the slab's kernel at wave time `time`.
    pub fn record(
        &self,
        c: &mut SceneCompute<'_, '_>,
        id: ProducerId,
        kernel: PipelineId,
        time: f32,
    ) {
        let root = c
            .put_root(&slab::Root {
                scene: c.root,
                producer: id.0,
                patches: self.patches,
                patch_size: self.patch_size,
                corner_x: self.corner[0],
                corner_z: self.corner[1],
                level: self.level,
                amplitude: self.amplitude,
                time,
                _pad0: 0,
            })
            .expect("slab root");
        let groups = (self.patches * self.patches).div_ceil(64);
        c.cmd.dispatch(c.pipeline(kernel), root, [groups, 1, 1]);
    }
}
