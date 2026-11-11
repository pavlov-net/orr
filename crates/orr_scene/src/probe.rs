//! Pixel probes: for one pixel of a view, what the raster drew there and why, read back a few
//! frames later like a screenshot. [`crate::Frame::probe`] records a readback of the pixel's
//! visibility buffer word, the visible cluster, its G-buffer texels, the records of its copy,
//! mesh and cluster and the resolve's texture taps; [`crate::Scenes::probes`] adds what the
//! CPU holds of them (pack names, the DAG groups, the streamed units' states) and returns a
//! [`PixelProbe`], whose `Display` is the report. Nothing is recorded or kept without a probe.

use std::fmt;

use orr_math::{DVec3, Quat, Vec3};
use orr_pack::mesh::Group;

use crate::material::material_flags as mf;
use crate::mount::Mounts;
use crate::shaders::probe::{ProbeGpu, ProbeTexture};
use crate::{DetailStats, MountId, ResidencyStats};

/// `PROBE_*` in `probe.slang`.
const GEOMETRY: u32 = 1;
const PRODUCER: u32 = 2;
const FOCUS: u32 = 4;
const PLACED: u32 = 8;
const EDIT: u32 = 16;
const DAG: u32 = 32;
const FROM_GROUP: u32 = 64;
const FADING: u32 = 128;
const MERGED_STAGE: u32 = 256;
const MERGED_RASTER: u32 = 512;
const LIST_OPAQUE: u32 = 1;
const LIST_ALPHA: u32 = 2;
const LIST_SOFTWARE: u32 = 4;
const LIST_SECOND: u32 = 8;
const PAGE_READ: u32 = 1;
const PAGE_WANTED: u32 = 2;
const PAGE_KEPT: u32 = 4;
const PAGE_STREAMS: u32 = 8;
const TEXTURE_STREAMED: u32 = 1;
const TEXTURE_NEURAL: u32 = 2;
const TEXTURE_OUTSIDE: u32 = 4;
const TEXTURE_TILED: u32 = 8;
const TEXTURE_RECORDED: u32 = 16;
/// `cut.slang`.
const CUT_TRAVERSED: u32 = 2;
const CUT_DISTANT: u32 = 4;
const CUT_OWNER: u32 = 8;
const CUT_PARTS: u32 = 16;
const CUT_FELL_BACK: u32 = 32;
const CUT_STAGE_MERGED: u32 = 64;
const RT_CUT_BUILT: u32 = 255;
const ADMIT_SHIFT: u32 = 17;
/// `tables.slang`.
const COPY_FAR: u32 = 0x8000_0000;
const RT_WRITTEN: u32 = 1;
const RT_IN_RANGE: u32 = 4;
const RT_ANIMATED: u32 = 16;
const INSTANCE_RAY: u32 = 2;
const INSTANCE_SKINNED: u32 = 8;
/// `visibility.slang` `SLOT_*`.
const SLOT_NAMES: [&str; 4] = [
    "culled",
    "drawn by the first pass",
    "occluded in the first pass",
    "revealed by the second pass",
];
/// `gbuffer.slang`.
const GBUFFER_ORIGIN_SHIFT: u32 = 5;
/// `orr_pack.mesh` `CLUSTER_MERGED`.
const CLUSTER_MERGED: u32 = 2;

/// `TEXTURE_*` slots of `material.slang`.
const SLOTS: [&str; 43] = [
    "base color",
    "opacity",
    "normal",
    "metalness",
    "roughness",
    "emission",
    "specular weight",
    "subsurface color",
    "displacement",
    "coat",
    "transmission",
    "specular color",
    "specular IOR",
    "base weight",
    "diffuse roughness",
    "anisotropy",
    "tangent",
    "transmission color",
    "subsurface weight",
    "subsurface radius",
    "subsurface radius scale",
    "subsurface anisotropy",
    "fuzz weight",
    "fuzz color",
    "fuzz roughness",
    "coat color",
    "coat roughness",
    "coat IOR",
    "coat darkening",
    "coat anisotropy",
    "coat normal",
    "coat tangent",
    "thin film weight",
    "thin film thickness",
    "thin film IOR",
    "emission luminance",
    "displacement u",
    "displacement v",
    "transmission depth",
    "transmission scatter",
    "transmission scatter anisotropy",
    "dispersion scale",
    "dispersion Abbe number",
];

/// A streamed unit's residency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitState {
    Absent,
    Loading,
    Resident,
    /// Its load or that of a unit it depends on failed: it never loads.
    Failed,
}

/// What a plan of the loading side did with a request for a unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanOutcome {
    /// Its load started.
    Started,
    /// It was loading from an earlier plan.
    Loading,
    /// A unit it depends on was loading.
    DependencyLoading,
    /// The plan started units it depends on, and it waits for them.
    DependenciesFirst,
    /// The loads in flight were at their limit.
    InFlight,
    /// Its class's budget could not take it: turned down.
    OverBudget,
    /// Taken back after the plan: it waits on a tile's unbind or a dependency taken back.
    HeldWaiting,
    /// Taken back after the plan: its clusters found no run of free resident cluster ids.
    HeldClusterIds,
}

/// A streamed unit as the loading side holds it when the probe is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitStatus {
    pub state: UnitState,
    /// Placed by the mount and never evicted.
    pub pinned: bool,
    /// Cached acceleration structures referencing it.
    pub locks: u32,
    /// The last frame a request or a traversal's read touched it.
    pub touched: u64,
    /// A request waits for the next plan.
    pub requested: bool,
    /// What the latest plan since the probe was recorded did with a request for it, and the
    /// frame of that plan.
    pub outcome: Option<(u64, PlanOutcome)>,
}

/// The tile of a streamed texture a tap wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeTile {
    pub mip: u32,
    pub xy: [u32; 2],
    /// The tile is bound and its texels landed.
    pub mapped: bool,
    /// The last frame whose texture feedback asked for it; `u64::MAX` never.
    pub heard: u64,
    pub status: UnitStatus,
}

/// One texture slot the pixel's material binds.
#[derive(Clone, Debug)]
pub struct ProbeTap {
    pub gpu: ProbeTexture,
    pub name: Option<String>,
    /// The tile of the mip the tap wants over its cell, for a streamed texture below its tail.
    pub tile: Option<ProbeTile>,
}

/// The settings the frame chose geometry with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSettings {
    pub raster_error_pixels: f32,
    pub rt_error_pixels: f32,
    pub culled_error_scale: f32,
    pub rt_merge_pixels: f32,
    pub software_raster_pixels: f32,
    /// Budgets of the mesh page pool and the texture pool, in bytes.
    pub mesh_pool: u64,
    pub texture_pool: u64,
}

/// The view the probed frame rendered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeView {
    pub position: DVec3,
    /// World from view.
    pub rotation: Quat,
    pub fov_y: f32,
    pub render_extent: [u32; 2],
    pub output_extent: [u32; 2],
    pub jitter: [f32; 2],
    pub mip_bias: f32,
}

/// What a probe records with its dispatch.
pub(crate) struct ProbeContext {
    pub(crate) frame: u64,
    pub(crate) pixel: [u32; 2],
    pub(crate) view: ProbeView,
    pub(crate) detail: DetailStats,
    pub(crate) residency: ResidencyStats,
    pub(crate) settings: ProbeSettings,
    pub(crate) focus: Option<(DVec3, f32)>,
}

/// A probe recorded into a frame, read once the frame completes.
pub(crate) struct PendingProbe {
    pub(crate) readback: orr_gpu::Allocation<ProbeGpu>,
    pub(crate) context: ProbeContext,
}

/// One probed pixel. The GPU fields are as the frame drew it; the names, groups and unit
/// states are what the CPU held when the probe was read, frame `read_at`.
#[derive(Clone, Debug)]
pub struct PixelProbe {
    pub frame: u64,
    pub read_at: u64,
    /// In render pixels.
    pub pixel: [u32; 2],
    pub view: ProbeView,
    /// The detail scales in force: `memory_scale`, its `visible_scale` and `hidden_scale` parts
    /// and `rt_scale` the frame's traversal ran at, `mesh_demand`, `mesh_budget` and
    /// `mesh_held` of the last frame completed before it.
    pub detail: DetailStats,
    /// The residency figures of the probed frame's step 1.
    pub residency: ResidencyStats,
    pub settings: ProbeSettings,
    /// The scene focus and the render pixels per radian the traversal took.
    pub focus: Option<(DVec3, f32)>,
    pub gpu: ProbeGpu,
    pub mount: Option<MountId>,
    pub model_name: Option<String>,
    pub mesh_name: Option<String>,
    pub section_material_name: Option<String>,
    pub material_name: Option<String>,
    /// The DAG group holding the cluster and its index, and the group it was simplified from.
    pub group: Option<(u32, Group)>,
    pub from_group: Option<Group>,
    /// The streamed units of the cluster's page and of its from group's page; `None` where
    /// the page is pinned or the mesh does not stream.
    pub page: Option<UnitStatus>,
    pub from_page: Option<UnitStatus>,
    pub textures: Vec<ProbeTap>,
}

impl PixelProbe {
    /// Whether the pixel shows geometry.
    pub fn hit(&self) -> bool {
        self.gpu.flags & GEOMETRY != 0
    }

    /// The DAG depth of the cluster's group.
    pub fn depth(&self) -> Option<u8> {
        self.group.map(|(_, g)| g.depth)
    }

    /// Whether the cluster stands in for a finer group whose page is not resident: its own
    /// error is over the threshold at the focus.
    pub fn stands_in(&self) -> bool {
        let g = &self.gpu;
        g.flags & PLACED != 0
            && g.debug_level < 0
            && g.parent_pixels > g.raster_threshold
            && g.self_pixels > g.raster_threshold
            && g.flags & FROM_GROUP != 0
            && g.from_address == 0
    }
}

/// Adds what the CPU holds of the probed pixel to its readback.
pub(crate) fn resolve(p: &PendingProbe, mounts: &Mounts, read_at: u64) -> PixelProbe {
    let gpu = p.readback.as_slice().expect("readback memory")[0];
    let c = &p.context;
    let mut out = PixelProbe {
        frame: c.frame,
        read_at,
        pixel: c.pixel,
        view: c.view,
        detail: c.detail,
        residency: c.residency,
        settings: c.settings,
        focus: c.focus,
        gpu,
        mount: None,
        model_name: None,
        mesh_name: None,
        section_material_name: None,
        material_name: None,
        group: None,
        from_group: None,
        page: None,
        from_page: None,
        textures: Vec::new(),
    };
    if gpu.flags & GEOMETRY == 0 {
        return out;
    }
    out.section_material_name = mounts.material_name(gpu.section_material);
    out.material_name = mounts.material_name(gpu.material);
    let streaming = &mounts.streaming;
    out.textures = gpu.textures[..gpu.texture_count as usize]
        .iter()
        .map(|t| {
            let streamed = t.flags & TEXTURE_STREAMED != 0 && t.flags & TEXTURE_OUTSIDE == 0;
            let mip = t.wanted.floor().max(0.0) as u32;
            ProbeTap {
                gpu: *t,
                name: mounts.texture_name(t.texture),
                tile: streamed
                    .then(|| streaming.tile_status(t.texture, mip, t.uv))
                    .flatten(),
            }
        })
        .collect();
    if gpu.flags & PRODUCER != 0 {
        return out;
    }
    let Some(m) = mounts.probe_model(crate::scene::ModelId(gpu.model)) else {
        return out;
    };
    out.mount = Some(m.mount);
    out.model_name = m.model_name;
    out.mesh_name = m.mesh_name;
    if gpu.flags & DAG != 0 {
        out.group = m
            .groups
            .iter()
            .position(|g| {
                let first = u32::from(g.first_cluster);
                g.page == gpu.mesh_page
                    && (first..first + u32::from(g.cluster_count)).contains(&gpu.local)
            })
            .map(|i| (i as u32, m.groups[i]));
        if gpu.flags & FROM_GROUP != 0 {
            out.from_group = m.groups.get(gpu.from_group as usize).copied();
        }
    }
    let streams = |page: u32| gpu.pinned_pages != gpu.page_count && page >= gpu.pinned_pages;
    if streams(gpu.mesh_page) {
        out.page = streaming.page_status(m.mount, m.mesh, gpu.mesh_page, false);
    }
    if gpu.flags & FROM_GROUP != 0 && streams(gpu.from_page) {
        out.from_page = streaming.page_status(m.mount, m.mesh, gpu.from_page, false);
    }
    out
}

fn named(id: u32, name: &Option<String>) -> String {
    match name {
        Some(n) => format!("{id} {n}"),
        None => format!("{id} (no pack name)"),
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1 << 20)
}

impl fmt::Display for UnitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match self.state {
            UnitState::Absent => "absent",
            UnitState::Loading => "loading",
            UnitState::Resident => "resident",
            UnitState::Failed => "failed",
        };
        write!(f, "{state}")?;
        if self.pinned {
            write!(f, ", pinned")?;
        }
        if self.locks > 0 {
            write!(f, ", locked by {} cached cuts", self.locks)?;
        }
        write!(f, ", last touched at frame {}", self.touched)?;
        if self.requested {
            write!(f, ", a request waits for the next plan")?;
        }
        match self.outcome {
            Some((frame, o)) => {
                let what = match o {
                    PlanOutcome::Started => "started its load",
                    PlanOutcome::Loading => "found it loading",
                    PlanOutcome::DependencyLoading => "held it for a page it depends on, loading",
                    PlanOutcome::DependenciesFirst => "started the pages it depends on first",
                    PlanOutcome::InFlight => "turned it down: loads in flight at their limit",
                    PlanOutcome::OverBudget => "turned it down: over its class's budget",
                    PlanOutcome::HeldWaiting => "took its load back: waiting on an unbind",
                    PlanOutcome::HeldClusterIds => "took its load back: no free cluster ids",
                };
                write!(f, "; the plan of frame {frame} {what}")
            }
            None => write!(f, "; no plan since the probe held a request for it"),
        }
    }
}

impl fmt::Display for PixelProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = &self.gpu;
        let v = &self.view;
        writeln!(
            f,
            "frame {} (read at {}), render pixel ({}, {}) of {} x {}, output {} x {}",
            self.frame,
            self.read_at,
            self.pixel[0],
            self.pixel[1],
            v.render_extent[0],
            v.render_extent[1],
            v.output_extent[0],
            v.output_extent[1]
        )?;
        let forward = v.rotation * Vec3::NEG_Z;
        writeln!(
            f,
            "view at ({:.3}, {:.3}, {:.3}), yaw {:.2} deg (about +y, 0 down -z), pitch {:.2} deg, vertical fov {:.2} deg, jitter ({:.3}, {:.3}), mip bias {:.3}",
            v.position.x,
            v.position.y,
            v.position.z,
            (-forward.x).atan2(-forward.z).to_degrees(),
            forward.y.asin().to_degrees(),
            v.fov_y.to_degrees(),
            v.jitter[0],
            v.jitter[1],
            v.mip_bias
        )?;
        let s = &self.settings;
        let d = &self.detail;
        match self.focus {
            Some((p, ppr)) => writeln!(
                f,
                "focus ({:.3}, {:.3}, {:.3}) at {ppr:.1} px/rad",
                p.x, p.y, p.z
            )?,
            None => writeln!(f, "no focus: every error above 0 projects to infinity")?,
        }
        writeln!(
            f,
            "detail: memory scale x{:.4} (visible x{:.4}, hidden x{:.4}), gathered cuts x{:.4}; thresholds raster {:.3} px, ray tracing {:.3} px (configured {} and {} px), merge below {} px, culled copies x{}, software raster up to {} px",
            d.memory_scale,
            d.visible_scale,
            d.hidden_scale,
            d.rt_scale,
            g.raster_threshold
                .max(s.raster_error_pixels * d.visible_scale),
            s.rt_error_pixels * d.visible_scale,
            s.raster_error_pixels,
            s.rt_error_pixels,
            s.rt_merge_pixels,
            s.culled_error_scale * d.hidden_scale,
            s.software_raster_pixels
        )?;
        let r = &self.residency;
        let class = |c: crate::ResidencyClass| {
            let c = r.class(c);
            format!(
                "{:.0} of {:.0} MiB resident, {:.0} loading",
                mib(c.resident_bytes),
                mib(c.budget),
                mib(c.loading_bytes)
            )
        };
        writeln!(
            f,
            "memory: mesh pages wanted {:.0} of {:.0} MiB, {:.0} MiB more held by ray cuts; mesh pages {}; textures {}; pools {:.0} and {:.0} MiB",
            mib(d.mesh_demand),
            mib(d.mesh_budget),
            mib(d.mesh_held),
            class(crate::ResidencyClass::MeshPages),
            class(crate::ResidencyClass::Textures),
            mib(s.mesh_pool),
            mib(s.texture_pool)
        )?;
        if !self.hit() {
            return writeln!(f, "sky: the visibility buffer holds no geometry here");
        }
        writeln!(
            f,
            "visibility word {:#018x}: visible cluster {}, triangle {}, depth {:.6}, {:.3} m along the ray",
            g.word, g.visible, g.triangle, g.depth, g.distance
        )?;
        let at = v.position + Vec3::new(g.relative[0], g.relative[1], g.relative[2]).as_dvec3();
        writeln!(f, "hit at world ({:.3}, {:.3}, {:.3})", at.x, at.y, at.z)?;
        let p = &g.placement;
        let origin = v.position + Vec3::new(p[0][3], p[1][3], p[2][3]).as_dvec3();
        writeln!(
            f,
            "placed by rows ({:.6}, {:.6}, {:.6}) ({:.6}, {:.6}, {:.6}) ({:.6}, {:.6}, {:.6}) at world ({:.4}, {:.4}, {:.4})",
            p[0][0],
            p[0][1],
            p[0][2],
            p[1][0],
            p[1][1],
            p[1][2],
            p[2][0],
            p[2][1],
            p[2][2],
            origin.x,
            origin.y,
            origin.z
        )?;
        let lists = [
            (LIST_OPAQUE, "mesh-shader opaque"),
            (LIST_ALPHA, "mesh-shader alpha"),
            (LIST_SOFTWARE, "compute software"),
        ];
        let mut paths = Vec::new();
        for (bit, name) in lists {
            if g.lists & bit != 0 {
                paths.push(format!("{name}, first pass"));
            }
            if g.lists & (bit * LIST_SECOND) != 0 {
                paths.push(format!("{name}, second pass"));
            }
        }
        if g.flags & PRODUCER != 0 {
            writeln!(
                f,
                "drawn: producer entry {}, cluster {}; producer entries have no pack names, DAG levels or cuts",
                g.copy, g.cluster
            )?;
        } else {
            let slot = g.copy & !COPY_FAR;
            writeln!(
                f,
                "drawn: slot {slot}{}, chain step {}{}, {} by the raster",
                if g.copy & COPY_FAR != 0 {
                    " (far step)"
                } else {
                    ""
                },
                g.step,
                if g.flags & FADING != 0 {
                    " while fading"
                } else {
                    ""
                },
                SLOT_NAMES.get(g.slot_visibility as usize).unwrap_or(&"?")
            )?;
            writeln!(
                f,
                "model {}, mesh {}{}",
                named(g.model, &self.model_name),
                named(g.mesh, &self.mesh_name),
                if g.flags & EDIT != 0 {
                    " (an edit: its copies hold what it touches at level 0)"
                } else {
                    ""
                }
            )?;
        }
        writeln!(
            f,
            "raster path: {}",
            if paths.is_empty() {
                "not found in the opaque, alpha or software lists".to_owned()
            } else {
                paths.join("; ")
            }
        )?;
        let mat = g.material_flags;
        let alpha = if mat & mf::ALPHA_TESTED != 0 {
            format!("alpha tested at {:.3}", g.transmission[2])
        } else {
            "opaque".to_owned()
        };
        let mut kind = vec![alpha];
        if mat & mf::DIELECTRIC != 0 {
            kind.push(format!(
                "dielectric (transmission {:.3})",
                g.transmission[0]
            ));
        }
        if mat & mf::PRESENCE != 0 {
            kind.push(format!("presence layer (opacity {:.3})", g.transmission[1]));
        }
        if mat & mf::DOUBLE_SIDED != 0 {
            kind.push("double sided".to_owned());
        }
        if mat & mf::THIN_WALLED != 0 {
            kind.push("thin walled".to_owned());
        }
        writeln!(
            f,
            "section {}: material {}{}; {}",
            g.section,
            named(g.section_material, &self.section_material_name),
            if g.material != g.section_material {
                format!(
                    ", its back material {} shows",
                    named(g.material, &self.material_name)
                )
            } else {
                String::new()
            },
            kind.join(", ")
        )?;
        if g.flags & PRODUCER == 0 {
            self.fmt_cluster(f)?;
            self.fmt_ray_tracing(f)?;
        }
        self.fmt_gbuffer(f)?;
        self.fmt_textures(f)
    }
}

impl PixelProbe {
    fn fmt_cluster(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = &self.gpu;
        writeln!(
            f,
            "cluster: mesh cluster {} ({} of page {}), resident cluster id {}, triangle {} of {}{}, page at {:#x} this frame, flags {:#x}{}",
            g.cluster,
            g.local,
            g.mesh_page,
            g.cluster_id,
            g.triangle,
            g.tri_count,
            if g.alpha_first < g.tri_count {
                format!(", alpha-tested from triangle {}", g.alpha_first)
            } else {
                String::new()
            },
            g.page_address,
            g.cluster_flags,
            if g.cluster_flags & CLUSTER_MERGED != 0 {
                " (merged stage)"
            } else {
                ""
            }
        )?;
        if g.flags & DAG == 0 {
            return writeln!(
                f,
                "lod: a discrete mesh of {} levels; the probe reads no level choice for it",
                g.level_count
            );
        }
        let t = g.raster_threshold;
        let skinned = g.instance.flags & INSTANCE_SKINNED as u8 != 0;
        let depth = self.group.map_or_else(
            || "unknown (no group holds it)".to_owned(),
            |(i, gr)| format!("{} (group {i})", gr.depth),
        );
        writeln!(
            f,
            "lod: DAG depth {depth}; self error {:.6} m, parent error {:.6} m{}",
            g.self_error,
            g.parent_error,
            if g.debug_scale != 1.0 {
                format!(", debug scale x{:.3}", g.debug_scale)
            } else {
                String::new()
            }
        )?;
        if g.flags & PLACED == 0 {
            writeln!(
                f,
                "  the copy has no placement at the focus: no projected errors"
            )?;
        } else {
            let source = self.group.is_some_and(|(_, gr)| gr.depth == 0);
            let finer_absent = g.flags & FROM_GROUP != 0 && g.from_address == 0;
            let merged = g.cluster_flags & CLUSTER_MERGED != 0;
            let why = if skinned {
                "a skinned copy draws the clusters its skinning took, not the traversal's rule"
                    .to_owned()
            } else if g.debug_level >= 0 {
                format!("held at debug level {}", g.debug_level)
            } else if g.flags & FOCUS == 0 {
                "no focus".to_owned()
            } else if merged != (g.flags & MERGED_RASTER != 0) {
                format!(
                    "the stage rule leaves it out: the cluster is of the {} stage and the raster cut takes the other; drawn from another frame's cut",
                    if merged { "merged" } else { "part" }
                )
            } else if g.parent_pixels <= t {
                format!(
                    "the rule leaves it out: parent {:.3} px <= {t:.3}; drawn from another frame's cut",
                    g.parent_pixels
                )
            } else if source {
                format!(
                    "taken: source geometry, parent {:.3} px > {t:.3}",
                    g.parent_pixels
                )
            } else if g.self_pixels <= t {
                format!(
                    "taken because parent {:.3} px > {t:.3} >= self {:.3} px",
                    g.parent_pixels, g.self_pixels
                )
            } else if finer_absent {
                format!(
                    "stands in: self {:.3} px > {t:.3}, finer group's page absent",
                    g.self_pixels
                )
            } else {
                format!(
                    "self {:.3} px > {t:.3} with the finer group's page resident this frame: finer clusters should have drawn",
                    g.self_pixels
                )
            };
            writeln!(
                f,
                "  at the focus (placement scale {:.4}): parent {:.3} px, self {:.3} px, threshold {t:.3} px: {why}",
                g.placement_scale, g.parent_pixels, g.self_pixels
            )?;
        }
        if g.flags & MERGED_STAGE == 0 {
            writeln!(f, "  stage: the mesh has no merged stage")?;
        } else if skinned {
            writeln!(
                f,
                "  stage: merge error {:.6} m; a skinned copy takes no stage: its skinning's cut",
                g.merge_error
            )?;
        } else if g.flags & PLACED == 0 {
            writeln!(f, "  stage: merge error {:.6} m", g.merge_error)?;
        } else if g.flags & FOCUS == 0 {
            writeln!(
                f,
                "  stage: merge error {:.6} m, no focus: the raster cut takes the part stage",
                g.merge_error
            )?;
        } else {
            let taken = if g.flags & MERGED_RASTER != 0 {
                "merged"
            } else {
                "part"
            };
            let held = if g.debug_level >= 0 {
                format!(", held at debug level {}", g.debug_level)
            } else {
                String::new()
            };
            writeln!(
                f,
                "  stage: merge error {:.6} m = {:.3} px at the focus, threshold {t:.3} px{held}: the raster cut takes the {taken} stage",
                g.merge_error, g.merged_pixels
            )?;
        }
        let bits = |b: u32| {
            if b & PAGE_STREAMS == 0 {
                return "pinned".to_owned();
            }
            let mut s = Vec::new();
            if b & PAGE_READ != 0 {
                s.push("read by the traversal");
            }
            if b & PAGE_WANTED != 0 {
                s.push(if b & PAGE_KEPT != 0 {
                    "wanted, kept in the frame's feedback"
                } else {
                    "wanted, lost past the frame's kept requests"
                });
            }
            if s.is_empty() {
                s.push("neither read nor wanted by the traversal");
            }
            s.join(", ")
        };
        writeln!(
            f,
            "  page {}: {}{}",
            g.mesh_page,
            bits(g.page_bits & 15),
            self.page
                .map_or(String::new(), |u| format!("; loading side: {u}"))
        )?;
        if g.flags & FROM_GROUP == 0 {
            return writeln!(f, "  source cluster: simplified from no group");
        }
        writeln!(
            f,
            "  from group {} (depth {}, error {:.6} m = {:.3} px) in page {}: {} this frame; {}{}; wanted error {:.3} px",
            g.from_group,
            self.from_group
                .map_or_else(|| "?".to_owned(), |gr| gr.depth.to_string()),
            g.from_error,
            g.from_pixels,
            g.from_page,
            if g.from_address != 0 {
                format!("resident at {:#x}", g.from_address)
            } else {
                "absent".to_owned()
            },
            bits(g.page_bits >> 4),
            self.from_page
                .map_or(String::new(), |u| format!("; loading side: {u}")),
            g.page_wanted_error
        )?;
        writeln!(
            f,
            "  the frame's feedback kept {} of {} page requests",
            g.feedback_kept, g.feedback_requests
        )
    }

    fn fmt_ray_tracing(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = &self.gpu;
        let c = &g.cut;
        let st = &g.state;
        let inst = &g.instance;
        if inst.flags & INSTANCE_RAY as u8 == 0 || g.rt_cut_count == 0 {
            return writeln!(
                f,
                "ray tracing: the copy is not traced ({})",
                if g.rt_cut_count == 0 {
                    "its mesh has no cached cuts"
                } else {
                    "no INSTANCE_RAY"
                }
            );
        }
        let in_range = c.rt_distance
            <= f32::from(inst.rt_range)
                * if st.rt_flags & RT_IN_RANGE != 0 {
                    1.15
                } else {
                    1.0
                };
        let kind = if !in_range {
            format!(
                "out of its rt range ({:.1} m from the focus, range {:.1} m)",
                c.rt_distance,
                f32::from(inst.rt_range)
            )
        } else if c.rt_cut == RT_CUT_BUILT {
            let parts = if c.flags & CUT_PARTS != 0 {
                ", a BLAS per animated part"
            } else {
                ""
            };
            let stage = if g.flags & MERGED_STAGE == 0 {
                ""
            } else if c.flags & CUT_STAGE_MERGED != 0 {
                " of the merged stage"
            } else {
                " of the part stage"
            };
            let error = format!("largest error {:.6} m", g.built_error);
            if c.flags & CUT_OWNER != 0 {
                format!(
                    "its own gathered cut{stage} of {} clusters, {error}, built this frame (job {}){parts}",
                    c.rt_count, c.rt_job
                )
            } else {
                format!(
                    "a gathered cut{stage} of {} clusters, {error}, shared with slot {}'s BLAS (job {}){parts}",
                    c.rt_count, c.owner, c.rt_job
                )
            }
        } else if c.flags & CUT_DISTANT != 0 {
            format!(
                "its model's merged cut {} (model radius {:.2} px < merge {:.1} px)",
                g.model_cut, g.model_pixels, g.merge_pixels
            )
        } else if c.flags & CUT_FELL_BACK != 0 {
            let why = if c.rt_count == 0 {
                "its gathered cut holds no cluster".to_owned()
            } else if g.nodes_dropped != 0 {
                format!("the traversal dropped {} nodes", g.nodes_dropped)
            } else if c.rt_distance.to_bits() >> ADMIT_SHIFT > g.admitted {
                format!(
                    "past the frame's admission (distance bucket {} > {})",
                    c.rt_distance.to_bits() >> ADMIT_SHIFT,
                    g.admitted
                )
            } else {
                // A cut is admitted by its nearest slot's distance, so no slot of a cut left
                // out lies in a bucket before the one the room ran out in.
                format!(
                    "the frame's admission ran out of room in its distance bucket {}, at \
                     its cut or one before it",
                    g.admitted
                )
            };
            format!("cached cut {} after gathering: fell back, {why}", c.rt_cut)
        } else if c.flags & CUT_TRAVERSED == 0 && c.rt_count > 0 {
            format!("cached cut {}, which equals the cut it gathered", c.rt_cut)
        } else {
            let why = if st.rt_flags & RT_ANIMATED != 0 {
                "it takes its pose"
            } else if g.step > 0 {
                "a chain step past the first"
            } else {
                "its mesh's emissive or caster sections need cached cuts"
            };
            format!("cached cut {} ({why})", c.rt_cut)
        };
        writeln!(f, "ray tracing: traces {kind}")?;
        writeln!(
            f,
            "  thresholds: cached {:.3} px{}, gathered {:.3} px (x{:.4}); {:.1} m from the focus, scale {:.4}",
            g.rt_cached,
            if g.seen == 0 {
                " (no view drew it last frame)"
            } else {
                ""
            },
            g.rt_gathered,
            g.gather_scale,
            c.rt_distance,
            c.rt_scale
        )?;
        if c.rt_cut != RT_CUT_BUILT {
            writeln!(
                f,
                "  cut {}: error {:.6} m = {:.3} px at the focus, BLAS {:#x}{}",
                c.rt_cut.min(g.rt_cut_count - 1),
                g.rt_cut_error,
                g.rt_cut_pixels,
                g.rt_cut_blas,
                if c.rt_cut < g.fine_cuts {
                    " (a finer cut of a streaming mesh)"
                } else {
                    ""
                }
            )?;
        }
        let held = if st.rt_flags & RT_WRITTEN == 0 {
            "nothing (not written)".to_owned()
        } else if u32::from(st.rt_lod) == RT_CUT_BUILT {
            format!("a BLAS built that frame ({:#x})", st.rt_blas)
        } else {
            format!(
                "cut {} (error {:.6} m, BLAS {:#x})",
                st.rt_lod, g.held_cut_error, st.rt_blas
            )
        };
        writeln!(
            f,
            "  its PTLAS instance holds {held}{}",
            if st.rt_age > 0 {
                format!(", its update {} frames late", st.rt_age)
            } else {
                String::new()
            }
        )
    }

    fn fmt_gbuffer(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = &self.gpu;
        let v3 = |v: [f32; 4]| format!("({:.4}, {:.4}, {:.4})", v[0], v[1], v[2]);
        writeln!(
            f,
            "gbuffer: base color {} roughness {:.3} metalness {:.3} specular weight {:.3} color {} IOR {:.3} diffuse roughness {:.3}",
            v3(g.base_color),
            g.base_color[3],
            g.emission[3],
            g.specular_color[3],
            v3(g.specular_color),
            g.normal[3],
            g.geometric_normal[3]
        )?;
        let policy = ["default", "surface", "triangle", "?"]
            [((g.gbuffer_flags >> GBUFFER_ORIGIN_SHIFT) & 3) as usize];
        writeln!(
            f,
            "  normal {} geometric {} emission {} coat {:.3} fuzz {:.3} subsurface {:.3} anisotropy {:.3}; raster error <= {:.6} m, origin policy {policy}, flags {:#x}",
            v3(g.normal),
            v3(g.geometric_normal),
            v3(g.emission),
            g.layers[0],
            g.layers[1],
            g.layers[2],
            g.layers[3],
            g.transmission[3],
            g.gbuffer_flags
        )?;
        writeln!(
            f,
            "  words a {:08x} {:08x} {:08x} {:08x} b {:08x} {:08x} x {:08x} {:08x} {:08x} {:08x} y {:08x} {:08x} {:08x} {:08x}",
            g.a[0],
            g.a[1],
            g.a[2],
            g.a[3],
            g.b[0],
            g.b[1],
            g.x[0],
            g.x[1],
            g.x[2],
            g.x[3],
            g.y[0],
            g.y[1],
            g.y[2],
            g.y[3]
        )
    }

    fn fmt_textures(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = &self.gpu;
        if self.textures.is_empty() {
            return writeln!(f, "textures: the material binds none");
        }
        writeln!(
            f,
            "textures ({} feedback raises so far this frame, records kept from {} for {}):",
            g.texture_raises, g.texture_first, g.texture_capacity
        )?;
        let stamp = (g.frame & 0xff_ffff) << 8;
        for t in &self.textures {
            let x = &t.gpu;
            let slot = SLOTS.get(x.slot as usize).unwrap_or(&"?");
            write!(f, "  {slot}: texture {}", named(x.texture, &t.name))?;
            if x.flags & TEXTURE_OUTSIDE != 0 {
                writeln!(
                    f,
                    " reads no texture here: {}",
                    if x.flags & TEXTURE_TILED != 0 {
                        "its tile set has no tile here"
                    } else {
                        "past a constant edge"
                    }
                )?;
                continue;
            }
            write!(
                f,
                "{} {} x {} at ({:.4}, {:.4}), the tap wants mip {:.2}",
                if x.flags & TEXTURE_TILED != 0 {
                    " (a tile of a tile set)"
                } else {
                    ""
                },
                x.size & 0xffff,
                x.size >> 16,
                x.uv[0],
                x.uv[1],
                x.wanted
            )?;
            if x.flags & TEXTURE_NEURAL != 0 {
                write!(f, ", a neural set stands in")?;
            }
            if x.flags & TEXTURE_STREAMED == 0 {
                writeln!(f, ", whole texture resident")?;
                continue;
            }
            write!(
                f,
                ", finest resident {} (tail from {})",
                x.resident, x.tail_mip
            )?;
            let wanted = x.wanted.floor().clamp(0.0, x.tail_mip as f32) as u32;
            if wanted >= x.tail_mip {
                writeln!(f, ", within the tail")?;
                continue;
            }
            let cell = if x.feedback & !0xff == stamp {
                format!(
                    "the cell {} wants mip {} this frame, {}",
                    x.cell,
                    255 - (x.feedback & 0xff),
                    if x.flags & TEXTURE_RECORDED != 0 {
                        "its raise recorded"
                    } else {
                        "its raise lost to a full feedback"
                    }
                )
            } else {
                format!(
                    "the cell {} has no raise this frame (value {:#x})",
                    x.cell, x.feedback
                )
            };
            write!(f, "; {cell}")?;
            match t.tile {
                Some(tile) => writeln!(
                    f,
                    "; tile mip {} ({}, {}) {}, heard {}: {}",
                    tile.mip,
                    tile.xy[0],
                    tile.xy[1],
                    if tile.mapped { "mapped" } else { "unmapped" },
                    if tile.heard == u64::MAX {
                        "never".to_owned()
                    } else {
                        format!("at frame {}", tile.heard)
                    },
                    tile.status
                )?,
                None => writeln!(f, "; no tile state")?,
            }
        }
        Ok(())
    }
}
