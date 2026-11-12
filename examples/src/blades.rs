//! A field of blades with nothing stored per blade (`shaders/blades.slang`): a drawn producer
//! entry whose blades a task and mesh pipeline derives and draws from the view's opaque hook,
//! which the material resolve derives again per pixel through the linked `BladeGeometry`, and
//! whose batch holds only what rays trace: the drawn blades near the eye and fewer, wider
//! one-triangle chords past that. `stored` puts every drawn blade in the batch instead, to
//! compare against.

use orr::orr_gpu::{
    Allocation, CommandList, Device, GpuPtr, Hazards, MemoryKind, MeshPipeline, SpecData, Stages,
};
use orr::orr_pack::material::Material;
use orr::{
    ClusterTopology, InstanceFlags, MaterialId, MeshPipelineId, MeshStages, OpaquePass, PipelineId,
    ProducerDesc, ProducerId, ProducerUpdate, Renderer, SceneCompute, TransformGroupId,
};
use orr_math::{Mat4, Vec2, Vec3};

use crate::field::Heights;
use crate::linked_blades as linked;
use crate::shaders::types::Field;
use crate::shaders::{blade_casters, blade_draw, blade_items};

/// `blades.slang`: the side of a tile, m, its cells, and the candidates of a draw item and of
/// the two kinds of caster item.
const TILE: f32 = 8.0;
const CELLS: u32 = 16;
const CELL_COUNT: u32 = CELLS * CELLS;
const SLICE: u32 = 128;
const CASTER_SLICE_EXACT: u32 = 14;
const CASTER_SLICE_CHORD: u32 = 64;
const CHORD_BLOCK: u32 = 16;
const CHORD_RUN: u32 = CASTER_SLICE_CHORD / CHORD_BLOCK;
const FULL_VERTICES: u32 = 11;
const FULL_TRIANGLES: u32 = 9;
/// How far a tile's box reaches past its square, at most (`blade_reach_max`).
const REACH: f32 = 1.5;
/// `Field.counts`.
pub const FIELD_COUNTS: usize = 4;

/// How a field draws and what of it rays trace.
#[derive(Clone, Copy, Debug)]
pub struct BladeSettings {
    /// Candidate blades per m^2: what is drawn within `full` m of the eye.
    pub density: f32,
    /// Every candidate is drawn nearer the eye than this; past it, the share that keeps their
    /// number per pixel.
    pub full: f32,
    /// No blade is drawn past this.
    pub range: f32,
    /// Tiles nearer draw full blades, farther ones one triangle a blade.
    pub detail: f32,
    /// The blades rays trace as they are drawn, at most: the cells nearest the eye, as far as
    /// this many reach among the tiles that draw full blades.
    pub exact_blades: u32,
    /// No blade past this is traced; 0 traces none.
    pub cast_range: f32,
    /// The chords that stand for the other drawn blades in that range, at most: one for as
    /// many of a cell's blades as keeps their number under it.
    pub chords: u32,
    /// Both ranges count a blade outside the view farther than it is, so most of what is
    /// traced lies where the camera looks: the half angle of the cone around the view's
    /// direction, on the ground plane, within which a blade counts at its distance,
    pub cast_cone: f32,
    /// and how many times its distance one directly behind the eye counts at; 1 counts every
    /// direction alike.
    pub cast_stretch: f32,
    /// Tip sway as a share of height.
    pub wind: f32,
    /// Every drawn blade is in the batch, which the raster draws and rays trace, and nothing
    /// is drawn from the hook.
    pub stored: bool,
    /// Rays trace the field. Without it a drawn field emits nothing and a stored one is only
    /// rastered.
    pub rays: bool,
}

impl Default for BladeSettings {
    fn default() -> Self {
        Self {
            density: 500.0,
            full: 20.0,
            range: 100.0,
            detail: 40.0,
            exact_blades: 16_384,
            cast_range: 50.0,
            chords: 131_072,
            cast_cone: 1.05,
            cast_stretch: 3.0,
            wind: 0.25,
            stored: false,
            rays: true,
        }
    }
}

/// What rays trace of a field, as the kernels read it.
#[derive(Clone, Copy, Debug)]
pub struct Traced {
    /// Cells nearer, by `cast_distance`, trace the blades they draw.
    pub exact: f32,
    /// No blade past this, by `cast_distance`, is traced.
    pub range: f32,
    /// Elsewhere a chord stands for this many of a cell's blades.
    pub stride: u32,
    /// `Field.cast_cone` and `Field.cast_stretch`.
    pub cone: f32,
    pub stretch: f32,
}

/// `blades.slang` `CAST_NEAR`.
const CAST_NEAR: f32 = 2.0;

impl Traced {
    /// `blades.slang` `cast_distance` of a point `d` m from the eye whose direction makes an
    /// angle of cosine `cos` with the view's.
    fn distance(&self, d: f32, cos: f32) -> f32 {
        d + (d - CAST_NEAR).max(0.0) * self.more(cos)
    }

    fn more(&self, cos: f32) -> f32 {
        (self.stretch - 1.0) * (self.cone - cos).max(0.0) / (self.cone + 1.0)
    }

    /// How far from the eye that distance is `reach`, in that direction.
    fn radius(&self, reach: f32, cos: f32) -> f32 {
        let more = self.more(cos);
        reach.min((reach + CAST_NEAR * more) / (1.0 + more))
    }

    /// `blades.slang` `cast_slope`.
    fn slope(&self) -> f32 {
        self.stretch + (self.stretch - 1.0) / (self.cone + 1.0)
    }
}

impl BladeSettings {
    /// What the kernels read. A stored field traces what it draws, everywhere; a drawn one
    /// fits the reach of its exact cells and its chords' stride to their budgets, with every
    /// blade in reach drawn.
    fn traced(&self) -> Traced {
        let mut t = Traced {
            exact: 0.0,
            range: 0.0,
            stride: 1,
            cone: self.cast_cone.cos(),
            stretch: self.cast_stretch.max(1.0),
        };
        if self.stored {
            t.exact = f32::MAX.sqrt();
            t.range = self.range;
            t.stretch = 1.0;
        } else if self.rays {
            t.range = self.cast_range;
            // The reach whose blades fit the budget: their number grows with it.
            let (mut lo, mut hi) = (0.0, self.detail.min(self.cast_range));
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                if self.drawn_within(&t, mid) <= f64::from(self.exact_blades) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            t.exact = lo;
            let rest = self.drawn_within(&t, t.range) - self.drawn_within(&t, t.exact);
            t.stride = (rest / f64::from(self.chords.max(1))).ceil().max(1.0) as u32;
        }
        t
    }

    /// The blades drawn within `reach` m of the eye by `cast_distance`, on flat ground.
    fn drawn_within(&self, t: &Traced, reach: f32) -> f64 {
        const ANGLES: u32 = 256;
        const STEP: f32 = 0.125;
        let mut blades = 0.0;
        for a in 0..ANGLES {
            let angle = (a as f32 + 0.5) / ANGLES as f32 * std::f32::consts::TAU;
            let radius = t.radius(reach, angle.cos());
            let rings = (radius / STEP).ceil() as u32;
            for k in 0..rings {
                let r = (k as f32 + 0.5) * radius / rings as f32;
                let area = r * (radius / rings as f32) * std::f32::consts::TAU / ANGLES as f32;
                blades += f64::from(self.density * self.keep(r) * area);
            }
        }
        blades
    }

    /// `blade_keep`.
    fn keep(&self, distance: f32) -> f32 {
        let near = self.full / distance.max(self.full);
        near * near * ((self.range - distance) / (0.2 * self.range)).clamp(0.0, 1.0)
    }
}

/// What a frame lists at most, with the eye where it lists the most.
#[derive(Clone, Copy, Debug, Default)]
pub struct Capacity {
    pub items: u32,
    pub caster_items: u32,
    pub vertices: u32,
    /// Blades drawn.
    pub blades: u64,
}

pub struct Blades {
    pub settings: BladeSettings,
    corner: [f32; 2],
    tiles: u32,
    /// Candidate blades per cell.
    ranks: u32,
    pub traced: Traced,
    pub capacity: Capacity,
    items: Allocation<u32>,
    item_state: Allocation<u32>,
    caster_items: Allocation<u32>,
    counts: Allocation<u32>,
    args: Allocation<u32>,
    /// One per frame in flight and one more.
    fields: Allocation<Field>,
    readback: Allocation<[u32; FIELD_COUNTS]>,
    clear: PipelineId,
    list: PipelineId,
    size: PipelineId,
    casters: PipelineId,
    /// The draw's pipeline, for [`Self::draw`].
    pub pipeline: MeshPipelineId,
    topologies: Vec<(u32, Vec<[u8; 3]>)>,
}

/// The triangles of a full blade (`full_triangle`).
fn full_triangle(t: u32) -> [u32; 3] {
    let j = t & !1;
    match t {
        8 => [8, 9, 10],
        _ if t & 1 == 0 => [j, j + 1, j + 3],
        _ => [j, j + 3, j + 2],
    }
}

impl Blades {
    /// A field over the square of `side` m from `corner`, in the space of the entry's
    /// transform, for a renderer with `frames_in_flight`.
    pub fn new(
        renderer: &mut Renderer,
        gpu: &Device,
        corner: [f32; 2],
        side: f32,
        settings: BladeSettings,
        frames_in_flight: u32,
    ) -> Self {
        let tiles = (side / TILE).ceil() as u32;
        let cell = TILE / CELLS as f32;
        let ranks = (settings.density * cell * cell).round().max(1.0) as u32;
        assert!(
            tiles * tiles <= 1 << 16 && ranks.div_ceil(CASTER_SLICE_EXACT) * CELL_COUNT <= 1 << 14,
            "an item names its tile in 16 bits and its slice in 14"
        );
        let traced = settings.traced();
        let capacity = capacity(&settings, &traced, tiles, ranks);
        assert!(
            capacity.items <= gpu.caps().limits.max_mesh_draw_groups,
            "one mesh draw covers the field's items"
        );
        let buffer = |n: u32, name| {
            gpu.gpu_malloc::<u32>(MemoryKind::DeviceLocal, u64::from(n.max(1)), name)
                .expect("blade list")
        };
        let slots = u64::from(frames_in_flight) + 1;
        let draw = renderer
            .mesh_pipeline(
                &blade_draw::MODULE,
                &MeshStages {
                    task: Some("task"),
                    pixel: Some("pixel"),
                    ..MeshStages::new("mesh")
                },
            )
            .expect("blade draw pipeline");
        let mut pipeline = |module, entry| {
            renderer
                .compute_pipeline(module, entry, SpecData::default())
                .expect("blade kernel")
        };
        let full = (1..=CASTER_SLICE_EXACT).map(|n| {
            let triangles = (0..n)
                .flat_map(|b| {
                    (0..FULL_TRIANGLES)
                        .map(move |t| full_triangle(t).map(|k| (b * FULL_VERTICES + k) as u8))
                })
                .collect();
            (n * FULL_VERTICES, triangles)
        });
        let chords = (1..=CASTER_SLICE_CHORD).map(|n| {
            let triangles = (0..n)
                .map(|b| [3 * b, 3 * b + 1, 3 * b + 2].map(|k| k as u8))
                .collect();
            (3 * n, triangles)
        });
        Self {
            settings,
            corner,
            tiles,
            ranks,
            traced,
            capacity,
            items: buffer(capacity.items, "blade items"),
            item_state: buffer(capacity.items, "blade item state"),
            caster_items: buffer(capacity.caster_items, "blade caster items"),
            counts: buffer(FIELD_COUNTS as u32, "blade counts"),
            args: buffer(8, "blade args"),
            fields: gpu
                .gpu_malloc(MemoryKind::Upload, slots, "blade fields")
                .expect("blade fields"),
            readback: gpu
                .gpu_malloc(MemoryKind::Readback, slots, "blade counts readback")
                .expect("blade counts readback"),
            clear: pipeline(&blade_items::MODULE, "clear"),
            list: pipeline(&blade_items::MODULE, "tiles"),
            size: pipeline(&blade_items::MODULE, "args"),
            casters: pipeline(&blade_casters::MODULE, "main"),
            pipeline: draw,
            topologies: full.chain(chords).collect(),
        }
    }

    /// The blades' material: thin and translucent, shaded by `BladeSurface`.
    pub fn material() -> Material {
        Material {
            base_color: [0.25, 0.45, 0.15],
            specular_roughness: 0.6,
            subsurface_weight: 0.5,
            subsurface_color: [0.25, 0.45, 0.15],
            double_sided: 1,
            thin_walled: 1,
            surface: linked::surface::BLADES,
            ..Material::default()
        }
    }

    /// The entry's cluster topologies: 1 to 14 full blades, then 1 to 64 chords.
    pub fn topologies(&self) -> Vec<ClusterTopology<'_>> {
        self.topologies
            .iter()
            .map(|(vertex_count, triangles)| ClusterTopology {
                vertex_count: *vertex_count,
                triangles,
            })
            .collect()
    }

    /// The entry: drawn through `BladeGeometry` unless the field is stored, with room for
    /// what rays trace of it.
    pub fn desc<'a>(
        &self,
        topologies: &'a [ClusterTopology<'a>],
        materials: &'a [MaterialId],
        transform: (TransformGroupId, u32),
    ) -> ProducerDesc<'a> {
        let traced = self.settings.rays && self.capacity.caster_items > 0;
        ProducerDesc {
            max_clusters: self.capacity.caster_items,
            max_vertices: self.capacity.vertices,
            update: ProducerUpdate::EveryFrame,
            topologies,
            materials,
            transform,
            previous_positions: true,
            vertex_user: true,
            flags: if traced {
                InstanceFlags::RASTER | InstanceFlags::RAY
            } else {
                InstanceFlags::RASTER
            },
            drawn: if self.settings.stored {
                0
            } else {
                linked::drawn::BLADES
            },
            ..ProducerDesc::default()
        }
    }

    /// Writes frame `frame`'s field for entry `producer` on `heights`, seen from `eye`
    /// along `forward` through `clip_from_entry` at wind times `time` and `previous_time`, and
    /// returns it: the scene's surface data and what [`Self::record`] and [`Self::draw`] take.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &mut self,
        frame: u64,
        heights: &Heights,
        producer: ProducerId,
        eye: Vec3,
        forward: Vec3,
        clip_from_entry: Mat4,
        time: f32,
        previous_time: f32,
    ) -> GpuPtr<Field> {
        let s = self.settings;
        let t = self.traced;
        let ahead = Vec2::new(forward.x, forward.z).normalize_or_zero();
        let m = clip_from_entry;
        // The sides, then the near plane, which reverse Z puts at z = w.
        let planes = [
            m.row(3) + m.row(0),
            m.row(3) - m.row(0),
            m.row(3) + m.row(1),
            m.row(3) - m.row(1),
            m.row(3) - m.row(2),
        ]
        .map(|p| p.to_array());
        let slot = frame % self.fields.len();
        self.fields.as_mut_slice().expect("upload is mapped")[slot as usize] = Field {
            heights: heights.samples(),
            items: self.items.gpu(),
            item_state: self.item_state.gpu(),
            caster_items: self.caster_items.gpu(),
            counts: self.counts.gpu(),
            draw_args: self.args.gpu(),
            planes,
            side: heights.side,
            spacing: heights.spacing,
            origin_x: heights.origin[0],
            origin_z: heights.origin[1],
            corner_x: self.corner[0],
            corner_z: self.corner[1],
            tiles: self.tiles,
            tile_size: TILE,
            ranks: self.ranks,
            item_capacity: self.capacity.items,
            caster_capacity: self.capacity.caster_items,
            full: s.full,
            range: s.range,
            detail: s.detail,
            exact: t.exact,
            cast_range: t.range,
            cast_stride: t.stride,
            forward_x: ahead.x,
            forward_z: ahead.y,
            cast_cone: t.cone,
            cast_stretch: t.stretch,
            wind: s.wind,
            time,
            previous_time,
            eye_x: eye.x,
            eye_y: eye.y,
            eye_z: eye.z,
            producer: producer.0,
        };
        self.fields.gpu().offset(slot)
    }

    /// Records the frame's lists and what rays trace of `field`, in scene compute.
    pub fn record(&self, c: &mut SceneCompute<'_, '_>, field: GpuPtr<Field>) {
        let root = c
            .put_root(&blade_items::Root { field, _pad0: 0 })
            .expect("blade items root");
        let casters = c
            .put_root(&blade_casters::Root {
                scene: c.root,
                field,
            })
            .expect("blade casters root");
        let compute = |list: &mut CommandList<'_>, hazards| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, hazards);
        };
        c.cmd.begin_pass("blade items", false);
        c.cmd.dispatch(c.pipeline(self.clear), root, [1, 1, 1]);
        compute(c.cmd, Hazards::empty());
        let groups = (self.tiles * self.tiles).div_ceil(64);
        c.cmd.dispatch(c.pipeline(self.list), root, [groups, 1, 1]);
        compute(c.cmd, Hazards::empty());
        c.cmd.dispatch(c.pipeline(self.size), root, [1, 1, 1]);
        c.cmd.end_pass();
        compute(c.cmd, Hazards::DRAW_ARGUMENTS);
        if self.capacity.caster_items > 0 {
            c.cmd.begin_pass("blade casters", false);
            c.cmd.dispatch_indirect(
                c.pipeline(self.casters),
                casters,
                self.args.gpu().offset(3).cast(),
            );
            c.cmd.end_pass();
        }
    }

    /// Draws `field` with the frame's `pipeline` of [`Self::pipeline`] in a pass of a view's
    /// raster: the view's opaque hook.
    pub fn draw(
        &self,
        pass: &mut OpaquePass<'_, '_>,
        pipeline: &MeshPipeline,
        field: GpuPtr<Field>,
    ) {
        let root = pass
            .put_root(&blade_draw::Root {
                scene: pass.scene,
                view: pass.view,
                raster: pass.raster,
                field,
                second: u32::from(pass.second),
                _pad0: 0,
                _pad1: 0,
            })
            .expect("blade draw root");
        let args = self.args.gpu();
        pass.cmd
            .draw_mesh_tasks_indirect_count(pipeline, root, args.cast(), args.offset(6), 1);
    }

    /// Copies the frame's counts for [`Self::counts`], after the view rendered.
    pub fn read_counts(&self, list: &mut CommandList<'_>, frame: u64) {
        let slot = frame % self.readback.len();
        list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
        list.copy(
            self.readback.gpu().offset(slot).cast(),
            self.counts.gpu().cast(),
            (FIELD_COUNTS * 4) as u64,
        );
        list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    }

    /// What completed frame `frame` listed: draw items, caster items and blades drawn.
    pub fn counts(&self, frame: u64) -> [u32; FIELD_COUNTS] {
        let slot = frame % self.readback.len();
        self.readback.as_slice().expect("readback is mapped")[slot as usize]
    }

    /// Bytes of the field's own memory: its lists and per-frame blocks.
    pub fn bytes(&self) -> u64 {
        4 * (self.items.len()
            + self.item_state.len()
            + self.caster_items.len()
            + self.counts.len()
            + self.args.len())
            + size_of::<Field>() as u64 * self.fields.len()
    }
}

/// What the lists and the batch hold at most: the kernels' sums over the tiles, from the eye
/// at the field's center, where the most tiles are near.
fn capacity(s: &BladeSettings, t: &Traced, tiles: u32, ranks: u32) -> Capacity {
    let center = tiles as f32 * TILE / 2.0;
    // `cast_distance` of the point at (x, z) from the first tile's corner, looking along x.
    let cast = |x: f32, z: f32| {
        let (dx, dz) = (x - center, z - center);
        let d = dx.hypot(dz);
        t.distance(d, if d > 0.0 { dx / d } else { 1.0 })
    };
    let cell = TILE / CELLS as f32;
    let mut c = Capacity::default();
    for tile in 0..tiles * tiles {
        let lo = [(tile % tiles) as f32 * TILE, (tile / tiles) as f32 * TILE];
        let gap = |lo: f32| (lo - center).max(center - lo - TILE).max(0.0);
        let d = (gap(lo[0]).hypot(gap(lo[1])) - REACH).max(0.0);
        let kept = ((s.keep(d) * ranks as f32).ceil() as u32).min(ranks);
        if kept == 0 {
            continue;
        }
        c.blades += u64::from(kept * CELL_COUNT);
        c.items += kept * CELL_COUNT / SLICE;
        if d >= t.range {
            continue;
        }
        // Squares of `side` cells of the tile whose centers `test` holds for.
        let squares = |side: u32, test: &dyn Fn(f32) -> bool| {
            let count = CELLS / side;
            let at = |lo: f32, k: u32| lo + (k as f32 + 0.5) * side as f32 * cell;
            (0..count * count)
                .filter(|&i| test(cast(at(lo[0], i % count), at(lo[1], i / count))))
                .count() as u32
        };
        let exact_cells = if d < s.detail.min(t.exact) {
            squares(1, &|d| d < t.exact)
        } else {
            0
        };
        // A block's blades stand within half its diagonal of its center.
        let reach = 2.0 * cell * std::f32::consts::SQRT_2 * t.slope();
        let blocks = squares(4, &|d| d - reach < t.range);
        let chords = kept.div_ceil(t.stride);
        c.caster_items +=
            exact_cells * kept.div_ceil(CASTER_SLICE_EXACT) + blocks * chords.div_ceil(CHORD_RUN);
        c.vertices += exact_cells * kept * FULL_VERTICES + blocks * CHORD_BLOCK * chords * 3;
    }
    // The eye off the center of a tile, and turned, shifts which tiles and squares count.
    let grow = |n: u32| n + n / 8 + tiles;
    Capacity {
        items: grow(c.items),
        caster_items: if c.caster_items == 0 {
            0
        } else {
            grow(c.caster_items)
        },
        vertices: grow(c.vertices),
        blades: c.blades,
    }
}
