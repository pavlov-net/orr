//! A field of blades with nothing stored per blade (`blades.slang`): a drawn producer
//! entry whose blades a task and mesh pipeline derives and draws from the view's opaque hook,
//! which the material resolve derives again per pixel through the linked `BladeGeometry`, and
//! whose batch holds only what rays trace: the blades near the eye and fewer, wider
//! one-triangle chords past that, in clusters of blades that stand together. `stored` puts
//! every drawn blade in the batch instead, to compare against.

use orr::orr_gpu::{
    Allocation, CommandList, Device, GpuPtr, Hazards, MemoryKind, MeshPipeline, SpecData, Stages,
};
use orr::orr_pack::material::Material;
use orr::{
    ClusterTopology, InstanceFlags, MaterialId, MeshPipelineId, MeshStages, OpaquePass, PipelineId,
    ProducerDesc, ProducerUpdate, Renderer, SceneCompute, TransformGroupId,
};
use orr_math::{Mat4, Vec2, Vec3};

use crate::field::Heights;
use crate::linked_blades as linked;
use shaders::types::Field;
use shaders::{blade_casters, blade_draw, blade_items};

/// The field's compiled kernels and the Rust types of their Slang structs.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/grass/orr_shaders.rs"));
}

/// `blades.slang`: the side of a tile, m, its cells, the candidates of a draw item and of the
/// two kinds of caster item, and the cells of a chord block.
const TILE: f32 = 8.0;
const CELLS: u32 = 16;
const CELL_COUNT: u32 = CELLS * CELLS;
const SLICE: u32 = 128;
const CASTER_SLICE_EXACT: u32 = 14;
const CASTER_SLICE_CHORD: u32 = 64;
const CHORD_BLOCK: u32 = 16;
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
    /// Past this the blades traced thin out at random, to none at `cast_range`.
    pub cast_fade: f32,
    /// No blade past this is traced; 0 traces none.
    pub cast_range: f32,
    /// The chords that stand for the other blades in that range, at most: one for as many of a
    /// cell's blades as keeps their number under it, fewer with distance.
    pub chords: u32,
    /// Both ranges count a blade outside the view farther than it is, so most of what is
    /// traced lies where the camera looks: the half angle of the cone around the view's
    /// direction, on the ground plane, within which a blade counts at its distance (a half
    /// turn or more counts every direction alike),
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
            cast_fade: 40.0,
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
    /// Cells nearer, by `cast_distance`, trace their blades as drawn.
    pub exact: f32,
    /// Past this, by `cast_distance`, the blades traced thin out,
    pub fade: f32,
    /// to none past this.
    pub range: f32,
    /// The chords' budget term (`chord_keep`).
    pub budget: f32,
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

    /// `blades.slang` `chord_keep`.
    fn chord_keep(&self, cast: f32) -> f32 {
        if cast <= self.exact {
            return 1.0;
        }
        let beyond = cast - self.exact;
        1.0 - (-self.budget / (beyond * beyond)).exp()
    }

    /// `blades.slang` `chord_stride` in a field of `ranks` candidate ranks.
    fn chord_stride(&self, cast: f32, ranks: u32) -> u32 {
        ((1.0 / self.chord_keep(cast).max(1e-6)).round() as u32).clamp(1, ranks)
    }

    /// `blades.slang` `cast_share`.
    fn share(&self, cast: f32) -> f32 {
        if cast <= self.fade {
            return 1.0;
        }
        let t = ((cast - self.fade) / (self.range - self.fade)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
}

/// `blades.slang` `chord_square_cells`.
fn chord_square_cells(chords: u32) -> u32 {
    match chords {
        17.. => 1,
        5.. => 4,
        _ => 16,
    }
}

/// `Field.strata` for `ranks` candidate ranks a cell: the squares a side that a cell's blades
/// stand in, enough that one square's ranks fit an exact caster item.
fn strata(ranks: u32) -> u32 {
    let squares = ranks.div_ceil(CASTER_SLICE_EXACT);
    let side = squares.isqrt();
    if side * side < squares {
        side + 1
    } else {
        side
    }
}

/// `Field.strata_step` for `strata`: coprime to its squares and near their golden share, so
/// consecutive ranks stand in squares far apart.
fn strata_step(strata: u32) -> u32 {
    let count = strata * strata;
    let gcd = |mut a: u32, mut b: u32| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let near = (count as f32 * 0.618).round() as u32;
    (near..)
        .find(|&step| gcd(step, count) == 1)
        .expect("count + 1 is coprime to count")
}

impl BladeSettings {
    /// Candidate blades per cell.
    fn ranks(&self) -> u32 {
        let cell = TILE / CELLS as f32;
        (self.density * cell * cell).round().max(1.0) as u32
    }

    /// What the kernels read. A stored field traces what it draws, everywhere; a drawn one
    /// fits the reach of its exact cells and its chords' budget term to their budgets, with
    /// every blade in reach traced.
    fn traced(&self) -> Traced {
        // A cone of a half turn or more holds every direction, which nothing stretches. Its
        // cosine is -1 already short of a half turn in f32, where `cast_distance` divides by 0.
        let cone = self.cast_cone.cos();
        let every = self.cast_cone >= std::f32::consts::PI || cone <= -1.0;
        let mut t = Traced {
            exact: 0.0,
            fade: 0.0,
            range: 0.0,
            budget: f32::INFINITY,
            cone: if every { 1.0 } else { cone },
            stretch: if every {
                1.0
            } else {
                self.cast_stretch.max(1.0)
            },
        };
        if self.stored {
            t.exact = f32::MAX.sqrt();
            t.range = self.range;
            t.fade = self.range;
            t.stretch = 1.0;
        } else if self.rays {
            t.range = self.cast_range;
            t.fade = self.cast_fade.min(self.cast_range);
            // The reach whose blades fit the budget: their number grows with it.
            let (mut lo, mut hi) = (0.0, self.detail.min(self.cast_range));
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                if self.traced_within(&t, mid, false) <= f64::from(self.exact_blades) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            t.exact = lo;
            // The largest budget term whose chords fit theirs, by halves of its log. Chords
            // within the exact cells' reach stand for one blade each, whatever the term.
            let exact = self.traced_within(&t, t.exact, true);
            let chords = |t: &Traced| self.traced_within(t, t.range, true) - exact;
            if chords(&t) > f64::from(self.chords) {
                let (mut lo, mut hi) = ((1.0f32 / 4096.0).log2(), 1.0e6f32.log2());
                for _ in 0..32 {
                    let mid = 0.5 * (lo + hi);
                    t.budget = mid.exp2();
                    if chords(&t) <= f64::from(self.chords) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                t.budget = lo.exp2();
            }
        }
        t
    }

    /// The blades traced within `reach` m of the eye by `cast_distance`, on flat ground: all
    /// of them to the fade, thinning to none at the cast range (`blade_traced`), or with
    /// `chords` the chords that stand for them (`chord_stride`).
    fn traced_within(&self, t: &Traced, reach: f32, chords: bool) -> f64 {
        const ANGLES: u32 = 256;
        const STEP: f32 = 0.125;
        let ranks = self.ranks();
        let mut blades = 0.0;
        for a in 0..ANGLES {
            let angle = (a as f32 + 0.5) / ANGLES as f32 * std::f32::consts::TAU;
            let cos = angle.cos();
            let radius = t.radius(reach, cos);
            let rings = (radius / STEP).ceil() as u32;
            for k in 0..rings {
                let r = (k as f32 + 0.5) * radius / rings as f32;
                let area = r * (radius / rings as f32) * std::f32::consts::TAU / ANGLES as f32;
                let cast = t.distance(r, cos);
                let mut share = t.share(cast);
                if chords {
                    share /= t.chord_stride(cast, ranks) as f32;
                }
                blades += f64::from(self.density * share * area);
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
    /// `Field.strata` and `Field.strata_step`.
    strata: u32,
    strata_step: u32,
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
        let ranks = settings.ranks();
        let strata = strata(ranks);
        // Every slice is below the exact items': a draw item's of ranks * CELL_COUNT / SLICE and
        // a chord item's of up to 16 runs of 64 chords, with no more ranks than the strata hold.
        assert!(
            tiles * tiles <= 1 << 16 && strata * strata * CELL_COUNT <= 1 << 14,
            "an item names its tile in 16 bits and its slice in 14"
        );
        let traced = settings.traced();
        let capacity = capacity(&settings, &traced, tiles, ranks, strata);
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
            strata,
            strata_step: strata_step(strata),
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
    /// what rays trace of it, and last frame's positions where the raster draws its batch.
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
            previous_positions: self.settings.stored,
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

    /// Writes frame `frame`'s field for producer entry `producer` on `heights`, seen from `eye`
    /// along `forward` through `clip_from_entry` at wind times `time` and `previous_time` in s,
    /// and returns it: the scene's surface data and what [`Self::record`] and [`Self::draw`]
    /// take.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &mut self,
        frame: u64,
        heights: &Heights,
        producer: u32,
        eye: Vec3,
        forward: Vec3,
        clip_from_entry: Mat4,
        time: f64,
        previous_time: f64,
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
        // `blade_sway`'s phases: rad per s of its slow wave and its fast ripple.
        let phase = |rate: f64, t: f64| (rate * t).rem_euclid(std::f64::consts::TAU) as f32;
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
            strata: self.strata,
            strata_step: self.strata_step,
            item_capacity: self.capacity.items,
            caster_capacity: self.capacity.caster_items,
            full: s.full,
            range: s.range,
            detail: s.detail,
            exact: t.exact,
            cast_fade: t.fade,
            cast_range: t.range,
            cast_budget: t.budget,
            stored: u32::from(s.stored),
            forward_x: ahead.x,
            forward_z: ahead.y,
            cast_cone: t.cone,
            cast_stretch: t.stretch,
            wind: s.wind,
            wave: phase(1.7, time),
            ripple: phase(3.1, time),
            wave_before: phase(1.7, previous_time),
            ripple_before: phase(3.1, previous_time),
            eye_x: eye.x,
            eye_y: eye.y,
            eye_z: eye.z,
            producer,
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
fn capacity(s: &BladeSettings, t: &Traced, tiles: u32, ranks: u32, strata: u32) -> Capacity {
    let center = tiles as f32 * TILE / 2.0;
    // `cast_distance` of the point at (x, z) from the first tile's corner, looking along x.
    let cast = |x: f32, z: f32| {
        let (dx, dz) = (x - center, z - center);
        let d = dx.hypot(dz);
        t.distance(d, if d > 0.0 { dx / d } else { 1.0 })
    };
    let cell = TILE / CELLS as f32;
    let (mut items, mut caster_items, mut vertices, mut blades) = (0_u64, 0_u64, 0_u64, 0_u64);
    for tile in 0..tiles * tiles {
        let lo = [(tile % tiles) as f32 * TILE, (tile / tiles) as f32 * TILE];
        let gap = |lo: f32| (lo - center).max(center - lo - TILE).max(0.0);
        let d = (gap(lo[0]).hypot(gap(lo[1])) - REACH).max(0.0);
        let kept = ((s.keep(d) * ranks as f32).ceil() as u32).min(ranks);
        blades += u64::from(kept * CELL_COUNT);
        items += u64::from(kept * CELL_COUNT / SLICE);
        // Rays trace every rank of a drawn field and the kept ones of a stored field.
        let traced = if s.stored { kept } else { ranks };
        if d >= t.range || traced == 0 {
            continue;
        }
        // Per block of 4 x 4 cells, the cells of the tile that trace their blades as drawn.
        let mut exact = [0_u32; (CELL_COUNT / CHORD_BLOCK) as usize];
        if d < s.detail.min(t.exact) {
            for i in 0..CELL_COUNT {
                let (x, z) = (i % CELLS, i / CELLS);
                let at = |lo: f32, k: u32| lo + (k as f32 + 0.5) * cell;
                if cast(at(lo[0], x), at(lo[1], z)) < t.exact {
                    exact[(z / 4 * 4 + x / 4) as usize] += 1;
                }
            }
        }
        let exact_cells: u32 = exact.iter().sum();
        caster_items += u64::from(exact_cells * (strata * strata).min(traced));
        vertices += u64::from(exact_cells * traced * FULL_VERTICES);
        // A block's blades stand within half its diagonal of its center.
        let reach = 2.0 * cell * std::f32::consts::SQRT_2 * t.slope();
        let side = CELLS / 4;
        for b in 0..side * side {
            let at = |lo: f32, k: u32| lo + (k as f32 + 0.5) * 4.0 * cell;
            let d = cast(at(lo[0], b % side), at(lo[1], b / side));
            // A block of exact cells has no chords.
            if exact[b as usize] == CHORD_BLOCK || d - reach >= t.range {
                continue;
            }
            let chords = traced.div_ceil(t.chord_stride(d, ranks));
            let cells = chord_square_cells(chords);
            let run = CASTER_SLICE_CHORD / cells;
            caster_items += u64::from(CHORD_BLOCK / cells * chords.div_ceil(run));
            vertices += u64::from(CHORD_BLOCK * chords * 3);
        }
    }
    // The eye off the center of a tile, and turned, shifts which tiles and squares count.
    let grow = |n: u64, what| {
        u32::try_from(n + n / 8 + u64::from(tiles))
            .unwrap_or_else(|_| panic!("a frame's {what} are counted in 32 bits"))
    };
    Capacity {
        items: grow(items, "draw items"),
        caster_items: if caster_items == 0 {
            0
        } else {
            grow(caster_items, "caster items")
        },
        vertices: grow(vertices, "caster vertices"),
        blades,
    }
}
