//! Producer entries in raster and ray tracing: a grass field whose cluster count changes
//! every frame against a CPU caster of the same blades, the producer words a linked surface
//! reads at primary and secondary hits, capacity and drops in `FrameStats`, particle
//! entries, drawn only at the transparency point and traced under the particle mask, and
//! tessellated entries: a plane of patches with a uv seam diced at mixed rates against a CPU
//! reference of the pattern table, closed in raster and rays once displaced, with normals
//! of its displaced field, displaced by a material texture, and a rate that steps down
//! near capacity; and subdivision copies: closed cages with creases, a corner, n-gons and
//! extraordinary points diced at mixed rates onto a CPU limit surface, closed bit for bit,
//! rippled along the limit normal, agreeing in raster and rays, and a skinned cage posed
//! before it dices with last frame's pose for motion; and strand entries, whose ribbons
//! resolve where traced swept spheres or tubes are hit and whose swept-sphere shadows match
//! their tubes'.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use glam::{DVec3, Mat3, Quat, UVec2, Vec2, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, CageInput, Corner, Crease, FaceVarying, Manifest, Source, SubdSettings,
    Subdivision, UsdSettings,
};
use orr_gpu::{
    Blend, ClearColor, ColorAttachment, ColorLoad, ColorMask, ColorTarget, ComputePipeline, Device,
    DeviceDesc, Format, Hazards, MemoryKind, MeshPipelineDesc, QueueKind, QueueRequest,
    RasterState, Rendering, SamplerDesc, ShaderStage, SpecData, Stages, StorageImageIndex, Texture,
    TextureDesc, TextureIndex, TextureMemory, TextureRegion, TextureUsage,
};
use orr_pack::AssetId;
use orr_pack::subd::{Page as SubdPage, SubdInfo};
use orr_scene::testing::{
    Image, Pixels, assert_clean, assert_clean_on, device, flip, read_buffer, read_texture, wait,
    wait_on,
};
use orr_scene::{
    Affine3, ClusterTopology, Frame, FrameDesc, FrameTime, GeneratedClusterGpu, GroupWriter, Lens,
    LoadState, MaterialGpu, MaterialId, ModelId, MountId, Opacity, PackSource, ProducerBatch,
    ProducerClusterGpu, ProducerDesc, ProducerId, ProducerStateGpu, ProducerStats, ProducerUpdate,
    Projection, RasterView, RendererDesc, SceneDesc, SceneId, SceneWriter, Scenes, SubdDesc,
    TessellationStats, TextureFilter, TransformGroupDesc, TransformGroupId, TransparencyPass,
    ViewDesc, ViewHistoryId, ViewTarget, VisibleCluster, material_flags, tessellation,
};
use orr_shaders::bytemuck::Zeroable;
use scene_suite::shaders::{
    blades, particle_draw, patches, producer_rays, strand_rays, strands, tess_rays,
};
use scene_suite::{Workers, linked, pipeline};

const EXTENT: [u32; 2] = [160, 96];
const FIELD: ProducerId = ProducerId(1);
const PARTICLES: ProducerId = ProducerId(2);
const FIELD_WORD: u32 = 1000;
const PARTICLE_WORD: u32 = 100;
const MAX_BLADES: u32 = 256;
/// The words surface's id in the suite's linked kernels.
const WORDS_SURFACE: u32 = 4;
/// `orr.trace` masks.
const MASK_OPAQUE: u32 = 1;
const MASK_PARTICLE: u32 = 2;

const BLADE: [[u8; 3]; 3] = [[0, 1, 2], [1, 3, 2], [2, 3, 4]];

fn time(frame: u32) -> FrameTime {
    FrameTime {
        absolute: f64::from(frame) / 60.0,
        delta: 1.0 / 60.0,
    }
}

/// Vertex `k` of blade `i` at `t`, as `blades.slang` places it.
fn blade_vertex(i: u32, k: usize, t: f32, z0: f32) -> Vec3 {
    let x = -2.0 + 0.25 * (i % 16) as f32;
    let z = z0 - 0.25 * (i / 16) as f32;
    let h = 0.3 + 0.05 * (i % 3) as f32;
    let sway = 0.1 * (t + i as f32).sin();
    let w = 0.08;
    let at = [
        Vec2::new(-w, 0.0),
        Vec2::new(w, 0.0),
        Vec2::new(-0.6 * w, 0.5),
        Vec2::new(0.6 * w, 0.5),
        Vec2::new(0.0, 1.0),
    ];
    let p = at[k];
    Vec3::new(x + p.x + sway * p.y * p.y, h * p.y, z)
}

/// A CPU caster's nearest hit: blade, triangle, distance.
fn cast(origin: Vec3, d: Vec3, blades: &[(u32, f32, f32)]) -> Option<(u32, usize, f64)> {
    let (o, d) = (origin.as_dvec3(), d.as_dvec3());
    let mut best: Option<(u32, usize, f64)> = None;
    for &(count, t, z0) in blades {
        for i in 0..count {
            for (j, tri) in BLADE.iter().enumerate() {
                let p = tri.map(|k| blade_vertex(i, k as usize, t, z0).as_dvec3());
                let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
                let q = d.cross(e2);
                let det = e1.dot(q);
                if det.abs() < 1e-12 {
                    continue;
                }
                let s = o - p[0];
                let u = s.dot(q) / det;
                let r = s.cross(e1);
                let v = d.dot(r) / det;
                let dist = e2.dot(r) / det;
                if u < 0.0 || v < 0.0 || u + v > 1.0 || dist <= 0.0 {
                    continue;
                }
                if best.is_none_or(|b| dist < b.2) {
                    best = Some((i, j, dist));
                }
            }
        }
    }
    best
}

fn look(eye: Vec3, target: Vec3) -> Quat {
    let back = (eye - target).normalize();
    let right = Vec3::Y.cross(back).normalize();
    Quat::from_mat3(&Mat3::from_cols(right, back.cross(right), back))
}

/// One emitting entry this frame: blades, time, first-row z and cluster-word base.
#[derive(Clone, Copy)]
struct Emit {
    id: ProducerId,
    count: u32,
    time: f32,
    z: f32,
    base: u32,
}

/// A transparency hook and the radiance it composites over.
type Hook<'a> = (
    StorageImageIndex,
    &'a mut dyn FnMut(&mut TransparencyPass<'_, '_>),
);

struct Field {
    scenes: Scenes,
    scene: SceneId,
    history: ViewHistoryId,
    group: TransformGroupId,
    material: MaterialId,
    triangles: orr_gpu::Allocation<u8>,
    frame: u32,
}

impl Field {
    fn new(software_raster_pixels: f32) -> Self {
        let d = device();
        let mut scenes = Scenes::new(
            d,
            &RendererDesc {
                kernels: &linked::KERNELS,
                software_raster_pixels,
                ..RendererDesc::default()
            },
        )
        .unwrap();
        let scene = scenes
            .create_scene(&SceneDesc {
                producer_capacity: 4,
                ..SceneDesc::default()
            })
            .unwrap();
        let history = scenes.create_view_history().unwrap();
        let mut triangles = d
            .gpu_malloc::<u8>(MemoryKind::Upload, 16, "blade triangles")
            .unwrap();
        triangles.as_mut_slice().unwrap()[..9].copy_from_slice(bytemuck::cast_slice(&BLADE));
        let list = [scene];
        let f = scenes
            .begin_frame(&FrameDesc {
                time: time(0),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let mut m = MaterialGpu::zeroed();
        m.base_color = [0.5; 3];
        m.base_weight = 1.0;
        m.flags = material_flags::DOUBLE_SIDED;
        m.surface = WORDS_SURFACE;
        let material = f.materials().create(&m).unwrap();
        let mut w = f.scene(scene).unwrap();
        let group = w
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        w.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY;
        drop(w);
        wait(f.end(&[]).unwrap());
        Self {
            scenes,
            scene,
            history,
            group,
            material,
            triangles,
            frame: 1,
        }
    }

    fn set(&mut self, id: ProducerId, desc: &ProducerDesc<'_>) {
        let list = [self.scene];
        let f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        f.scene(self.scene)
            .unwrap()
            .set_producer(id, Some(desc))
            .unwrap();
        self.frame += 1;
        wait(f.end(&[]).unwrap());
    }

    /// Runs one frame emitting `emits` and rasters the camera's view; `hook` runs at the
    /// transparency point over `radiance`.
    fn frame(&mut self, emits: &[Emit], hook: Option<Hook<'_>>) -> RasterView {
        let p = pipeline(&blades::MODULE, "main");
        let list = [self.scene];
        let mut f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let triangles = self.triangles.gpu();
        f.update_scenes(&mut |c| {
            for e in emits {
                let r = c
                    .put_root(&blades::Root {
                        scene: c.root,
                        triangles,
                        producer: e.id.0,
                        count: e.count,
                        time: e.time,
                        previous_time: e.time,
                        base: e.base,
                        z: e.z,
                        _pad0: 0,
                        _pad1: 0,
                    })
                    .unwrap();
                c.cmd.dispatch(&p, r, [e.count.div_ceil(64).max(1), 1, 1]);
            }
        })
        .unwrap();
        let eye = Vec3::new(0.0, 0.3, 1.0);
        let view = f
            .raster_view(&ViewDesc {
                scene: self.scene,
                history: self.history,
                position: eye.as_dvec3(),
                rotation: look(eye, Vec3::new(-0.1, 0.1, -2.0)),
                projection: Projection {
                    fov_y: 0.9,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: UVec2::from(EXTENT),
                jitter: Vec2::ZERO,
                reset_history: self.frame <= 2,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
                target: ViewTarget::AfterRr,
            })
            .unwrap();
        if let Some((radiance, hook)) = hook {
            f.transparency(&view, radiance, hook).unwrap();
        }
        self.frame += 1;
        wait(f.end(&[]).unwrap());
        view
    }

    /// The producer stats of the last frame, which the next frame's start retires.
    fn stats(&mut self) -> ProducerStats {
        let f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        self.frame += 1;
        wait(f.end(&[]).unwrap());
        self.scenes.stats().producers
    }
}

/// A blade entry over `group`'s first transform, with vertex words and user tag 77.
fn blade_desc<'a>(
    group: TransformGroupId,
    topologies: &'a [ClusterTopology<'a>],
    materials: &'a [MaterialId],
) -> ProducerDesc<'a> {
    ProducerDesc {
        max_clusters: MAX_BLADES,
        max_vertices: 5 * MAX_BLADES,
        update: ProducerUpdate::EveryFrame,
        topologies,
        materials,
        transform: (group, 0),
        vertex_user: true,
        user_tag: 77,
        ..ProducerDesc::default()
    }
}

/// What `producer_rays` saw at every pixel of `view`.
fn rays(view: &RasterView, mode: u32, mask: u32) -> Vec<producer_rays::Seen> {
    let d = device();
    let n = u64::from(EXTENT[0] * EXTENT[1]);
    let seen = d
        .gpu_malloc::<producer_rays::Seen>(MemoryKind::Readback, n, "seen")
        .unwrap();
    let mut root = d
        .gpu_malloc::<producer_rays::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = producer_rays::Root {
        scene: view.scene,
        view: view.view,
        gbuffer: view.gbuffer,
        seen: seen.gpu(),
        mode,
        mask,
        grid_x: -2.2,
        grid_z: -0.8,
        grid_step: 4.4 / EXTENT[0] as f32,
        height: 3.0,
        _pad0: 0,
        _pad1: 0,
    };
    let module = linked::KERNELS.module(&producer_rays::MODULE).unwrap();
    let p = pipeline(module, "main");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
    list.dispatch(&p, root.gpu(), [n.div_ceil(64) as u32, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    seen.as_slice().unwrap().to_vec()
}

/// The words a hit on triangle `tri` of blade `i` reports: cluster word, vertex words of its
/// corners.
fn words(base: u32, i: u32, tri: usize) -> [f32; 4] {
    let k = BLADE[tri];
    [
        (base + i) as f32,
        (8 * i + u32::from(k[0])) as f32,
        (8 * i + u32::from(k[1])) as f32,
        (8 * i + u32::from(k[2])) as f32,
    ]
}

fn traced_words(s: &producer_rays::Seen) -> [f32; 4] {
    let [a, b, c] = s.traced_words;
    [a, b, c, s.traced_more[0]]
}

fn raster_words(s: &producer_rays::Seen) -> [f32; 4] {
    let [a, b, c] = s.raster_words;
    [a, b, c, s.raster_more[0]]
}

/// Pixels whose `got` disagrees with the CPU caster's `want`, and the pixels the caster
/// covered.
fn disagree(
    want: &[Option<[f32; 4]>],
    got: impl Iterator<Item = Option<[f32; 4]>>,
) -> (usize, usize) {
    let mut bad = 0;
    for (w, g) in want.iter().zip(got) {
        if *w != g {
            bad += 1;
        }
    }
    (bad, want.iter().filter(|w| w.is_some()).count())
}

fn traced(available: bool) -> bool {
    let caps = device().caps().cluster;
    gpu_suite::require(
        !available || (caps.max_tris > 0 && caps.max_partitions > 0),
        "cluster acceleration structures",
    )
}

#[test]
fn grass_changes_cluster_count_every_frame_and_draws_and_traces_only_this_frames_blades() {
    if !traced(true) {
        return;
    }
    for software in [0.0, 32.0] {
        let mut field = Field::new(software);
        let topology = [ClusterTopology {
            vertex_count: 5,
            triangles: &BLADE,
        }];
        let materials = [field.material];
        let desc = blade_desc(field.group, &topology, &materials);
        field.set(FIELD, &desc);
        for (n, &(count, t)) in [
            (37, 0.0),
            (256, 0.7),
            (5, 1.3),
            (120, 2.1),
            (0, 2.5),
            (200, 3.0),
        ]
        .iter()
        .enumerate()
        {
            let emit = Emit {
                id: FIELD,
                count,
                time: t,
                z: -1.0,
                base: FIELD_WORD,
            };
            let view = field.frame(&[emit], None);
            let what = format!("software {software}, frame {n}, {count} blades");
            let seen = rays(&view, 0, MASK_OPAQUE);
            let want: Vec<Option<[f32; 4]>> = seen
                .iter()
                .map(|s| {
                    cast(
                        Vec3::from(s.origin),
                        Vec3::from(s.direction),
                        &[(count, t, -1.0)],
                    )
                    .map(|(i, tri, _)| words(FIELD_WORD, i, tri))
                })
                .collect();
            let entry = FIELD.0 + 1;
            // No pixel shows a blade this frame did not emit.
            for s in &seen {
                for (e, w) in [
                    (s.raster_entry, s.raster_words[0]),
                    (s.traced_entry, s.traced_words[0]),
                ] {
                    if e == entry {
                        assert!(
                            w >= FIELD_WORD as f32 && w < (FIELD_WORD + count) as f32,
                            "{what}: stale blade {w}"
                        );
                    }
                }
            }
            let (bad, covered) = disagree(
                &want,
                seen.iter()
                    .map(|s| (s.raster_entry == entry).then(|| raster_words(s))),
            );
            assert!(
                bad * 50 <= covered.max(50),
                "{what}: raster {bad} of {covered}"
            );
            let (bad, covered) = disagree(
                &want,
                seen.iter().map(|s| {
                    (s.traced_entry == entry && s.traced_valid == 1).then(|| traced_words(s))
                }),
            );
            assert!(
                bad * 50 <= covered.max(50),
                "{what}: traced {bad} of {covered}"
            );
            for s in seen.iter().filter(|s| s.traced_entry == entry) {
                assert_eq!(s.traced_more[1], 77.0, "{what}: user tag");
                assert_eq!(s.traced_more[2], 0.0, "{what}: primary kind");
            }
            if count > 0 {
                assert!(covered > 100, "{what}: the field covers {covered} pixels");
            }

            // Rays from above see the same words as secondary hits.
            let seen = rays(&view, 1, MASK_OPAQUE);
            let want: Vec<Option<[f32; 4]>> = seen
                .iter()
                .map(|s| {
                    cast(
                        Vec3::from(s.origin),
                        Vec3::from(s.direction),
                        &[(count, t, -1.0)],
                    )
                    .map(|(i, tri, _)| words(FIELD_WORD, i, tri))
                })
                .collect();
            let (bad, covered) = disagree(
                &want,
                seen.iter().map(|s| {
                    (s.traced_entry == entry && s.traced_valid == 1).then(|| traced_words(s))
                }),
            );
            assert!(
                bad * 20 <= covered.max(20),
                "{what}: from above {bad} of {covered}"
            );
            for s in seen.iter().filter(|s| s.traced_entry == entry) {
                assert_eq!(s.traced_more[2], 1.0, "{what}: secondary kind");
            }
        }
        let stats = field.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.cluster_capacity, MAX_BLADES);
    }
    assert_clean();
}

#[test]
fn clusters_and_vertices_past_capacity_are_dropped_and_counted() {
    let mut field = Field::new(32.0);
    let topology = [ClusterTopology {
        vertex_count: 5,
        triangles: &BLADE,
    }];
    let materials = [field.material];
    let few_clusters = ProducerDesc {
        max_clusters: 64,
        ..blade_desc(field.group, &topology, &materials)
    };
    let few_vertices = ProducerDesc {
        max_vertices: 200,
        ..blade_desc(field.group, &topology, &materials)
    };
    field.set(ProducerId(0), &few_clusters);
    field.set(ProducerId(3), &few_vertices);
    let emit = |id, z| Emit {
        id,
        count: 100,
        time: 0.0,
        z,
        base: FIELD_WORD,
    };
    let view = field.frame(
        &[emit(ProducerId(0), -1.0), emit(ProducerId(3), -3.0)],
        None,
    );
    let counts: Vec<u32> = read_buffer(view.counts, 1);
    let clusters: Vec<VisibleCluster> = read_buffer(view.targets.clusters, u64::from(counts[0]));
    let stats = field.stats();
    assert_eq!(stats.entries, 2);
    // 100 vertex runs of 5 into 1280 and 200 vertices; 100 and 40 clusters into 64 and 256.
    assert_eq!(stats.vertices, 500 + 200);
    assert_eq!(stats.dropped_vertices, 300);
    assert_eq!(stats.clusters, 64 + 40);
    assert_eq!(stats.dropped_clusters, 36);
    assert_eq!(stats.cluster_capacity, 64 + MAX_BLADES);
    assert_eq!(stats.vertex_capacity, 5 * MAX_BLADES + 200);
    assert_eq!(stats.refused, 0);
    if traced(true) {
        assert_eq!(stats.built, 64 + 40);
    }
    assert!(clusters.iter().all(|c| c.cluster < 64));
    assert_clean();
}

fn half(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let e = i32::from((bits >> 10) & 31);
    let m = f32::from(bits & 1023);
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

#[test]
fn particles_are_drawn_only_at_the_transparency_point_and_traced_only_under_their_mask() {
    if !traced(true) {
        return;
    }
    let d = device();
    let mut field = Field::new(32.0);
    let topology = [ClusterTopology {
        vertex_count: 5,
        triangles: &BLADE,
    }];
    let materials = [field.material];
    let grass = blade_desc(field.group, &topology, &materials);
    field.set(FIELD, &grass);
    let particles = ProducerDesc {
        opacity: Opacity::Particle,
        ..blade_desc(field.group, &topology, &materials)
    };
    field.set(PARTICLES, &particles);

    let radiance = d
        .create_texture(
            &TextureDesc::d2(
                Format::Rgba16Float,
                EXTENT[0],
                EXTENT[1],
                TextureUsage::STORAGE | TextureUsage::COLOR,
            ),
            TextureMemory::Dedicated,
            "radiance",
        )
        .unwrap();
    let slot = d.alloc_image_slot().unwrap();
    let store = d.write_storage(slot, &radiance, &orr_gpu::ViewDesc::default());
    let draw = d
        .create_mesh_pipeline(
            &MeshPipelineDesc {
                task: None,
                mesh: ShaderStage::new(&particle_draw::MODULE, "mesh", SpecData::default())
                    .unwrap(),
                pixel: Some(
                    ShaderStage::new(&particle_draw::MODULE, "pixel", SpecData::default()).unwrap(),
                ),
                colors: &[ColorTarget {
                    format: Format::Rgba16Float,
                    blend: Some(Blend::PREMULTIPLIED),
                    write: ColorMask::ALL,
                }],
                depth: None,
                raster: RasterState::default(),
                samples: 1,
            },
            "particles",
        )
        .unwrap();
    clear(&radiance);

    let emits = [
        Emit {
            id: FIELD,
            count: 64,
            time: 0.4,
            z: -1.5,
            base: FIELD_WORD,
        },
        Emit {
            id: PARTICLES,
            count: 16,
            time: 1.1,
            z: -0.6,
            base: PARTICLE_WORD,
        },
    ];
    let mut hook = |pass: &mut TransparencyPass<'_, '_>| {
        let root = pass
            .put_root(&particle_draw::Root {
                scene: pass.scene,
                view: pass.view,
                raster: pass.raster,
                _pad: 0,
            })
            .unwrap();
        pass.cmd.begin_rendering(&Rendering {
            extent: pass.extent,
            colors: &[ColorAttachment {
                texture: pass.overlay,
                load: ColorLoad::Keep,
                store: true,
            }],
            depth: None,
        });
        pass.cmd.draw_mesh_tasks_indirect_count(
            &draw,
            root,
            pass.particle_args,
            pass.particle_draws,
            1,
        );
        pass.cmd.end_rendering();
    };
    let view = field.frame(&emits, Some((store, &mut hook)));

    let seen = rays(&view, 0, MASK_OPAQUE);
    let image: Vec<[u16; 4]> = bytemuck::pod_collect_to_vec(&read_texture(
        &radiance,
        &TextureRegion::whole_mip(radiance.desc(), 0),
    ));
    let particle_entry = PARTICLES.0 + 1;
    assert!(
        seen.iter().all(|s| s.raster_entry != particle_entry),
        "particles in the raster"
    );
    assert!(
        seen.iter().all(|s| s.traced_entry != particle_entry),
        "particles under the opaque mask"
    );
    let want: Vec<Option<[f32; 4]>> = seen
        .iter()
        .map(|s| {
            cast(
                Vec3::from(s.origin),
                Vec3::from(s.direction),
                &[(16, 1.1, -0.6)],
            )
            .map(|(i, tri, _)| words(PARTICLE_WORD, i, tri))
        })
        .collect();
    // The hook writes the cluster word and the first two vertex words.
    let three: Vec<Option<[f32; 4]>> = want
        .iter()
        .map(|w| w.map(|w| [w[0], w[1], w[2], 0.0]))
        .collect();
    let (bad, covered) = disagree(
        &three,
        image.iter().map(|p| {
            let c = p.map(half);
            (c[0] > 0.0).then_some([c[0], c[1], c[2], 0.0])
        }),
    );
    assert!(covered > 50, "the particles cover {covered} pixels");
    assert!(bad * 20 <= covered, "overlay: {bad} of {covered}");

    // The particle mask sees them.
    let seen = rays(&view, 0, MASK_PARTICLE);
    let (bad, covered) = disagree(
        &want,
        seen.iter().map(|s| {
            (s.traced_entry == particle_entry && s.traced_valid == 1).then(|| traced_words(s))
        }),
    );
    assert!(bad * 20 <= covered, "particle mask: {bad} of {covered}");
    d.free_image_slot_after_submitted(slot);
    assert_clean();
}

/// Clears `t`, a color target, to zero.
fn clear(t: &Texture) {
    let render = device().queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.discard(t);
    list.begin_rendering(&Rendering {
        extent: EXTENT,
        colors: &[ColorAttachment {
            texture: t,
            load: ColorLoad::Clear(ClearColor::Float([0.0; 4])),
            store: true,
        }],
        depth: None,
    });
    list.end_rendering();
    list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
}

/// The ripple displacement's height at `uv` (`surfaces/ripple.slang`).
fn ripple(uv: Vec2) -> f32 {
    0.05 * (9.0 * uv.x).sin() * (7.0 * uv.y).cos()
}

/// Displacement id of `surfaces/ripple.slang` in the suite's linked kernels.
const RIPPLE: u8 = 1;
const PLANE: ProducerId = ProducerId(0);
/// The plane's halves meet at x = 0 with uv this far apart.
const SEAM_UV: Vec2 = Vec2::new(0.37, 0.11);

/// The plane's uv at (x, z) on the side `right` of the seam.
fn plane_uv(x: f32, z: f32, right: bool) -> Vec2 {
    Vec2::new(x, z) * 0.25 + if right { SEAM_UV } else { Vec2::ZERO }
}

/// A plane of patch triangles over x in [-2, 2], z in [-6, 0]: one cluster per half and
/// row of 1 m quads, so the two halves meet at a uv seam.
struct PatchMesh {
    positions: Vec<Vec3>,
    uvs: Vec<Vec2>,
    triangles: Vec<[u8; 3]>,
    /// First vertex, vertex count, first triangle, triangle count.
    clusters: Vec<[u32; 4]>,
}

impl PatchMesh {
    /// The mesh with the right half's uv moved by `seam`.
    fn new(seam: Vec2) -> Self {
        let mut m = Self {
            positions: Vec::new(),
            uvs: Vec::new(),
            triangles: Vec::new(),
            clusters: Vec::new(),
        };
        for row in 0..6 {
            for right in [false, true] {
                let first = m.positions.len() as u32;
                let tri_first = m.triangles.len() as u32;
                let x0 = if right { 0.0 } else { -2.0 };
                let (z0, z1) = (-(row as f32), -(row as f32) - 1.0);
                for z in [z0, z1] {
                    for i in 0..3 {
                        let x = x0 + i as f32;
                        m.positions.push(Vec3::new(x, 0.0, z));
                        let shift = if right { seam } else { Vec2::ZERO };
                        m.uvs.push(plane_uv(x, z, false) + shift);
                    }
                }
                for q in 0..2u8 {
                    m.triangles.push([q, q + 3, q + 1]);
                    m.triangles.push([q + 1, q + 3, q + 4]);
                }
                m.clusters.push([first, 6, tri_first, 4]);
            }
        }
        m
    }
}

/// A tessellated plane entry, its patches emitted once, and a camera above it.
struct Plane {
    scenes: Scenes,
    scene: SceneId,
    history: ViewHistoryId,
    group: TransformGroupId,
    material: MaterialId,
    mesh: PatchMesh,
    /// Positions, attributes, triangles and cluster records of `mesh`, for `patches.slang`.
    gpu: [orr_gpu::Allocation<u8>; 4],
    frame: u32,
}

impl Plane {
    fn new(dice_pixels: f32) -> Self {
        let software = RendererDesc::default().software_raster_pixels;
        Self::build(dice_pixels, software, SEAM_UV)
    }

    /// The plane with its halves' uv `seam` apart.
    fn build(dice_pixels: f32, software_raster_pixels: f32, seam: Vec2) -> Self {
        let d = device();
        let mut scenes = Scenes::new(
            d,
            &RendererDesc {
                kernels: &linked::KERNELS,
                dice_pixels,
                software_raster_pixels,
                ..RendererDesc::default()
            },
        )
        .unwrap();
        let scene = scenes
            .create_scene(&SceneDesc {
                producer_capacity: 2,
                ..SceneDesc::default()
            })
            .unwrap();
        let history = scenes.create_view_history().unwrap();
        let mesh = PatchMesh::new(seam);
        let attrs: Vec<[u32; 2]> = mesh
            .uvs
            .iter()
            .map(|uv| {
                let h = |f: f32| u32::from(half::f16::from_f32(f).to_bits());
                [
                    orr_pack::mesh::oct_encode([0.0, 1.0, 0.0]),
                    h(uv.x) | h(uv.y) << 16,
                ]
            })
            .collect();
        let upload = |bytes: &[u8], name| {
            let len = bytes.len().next_multiple_of(16) as u64;
            let mut a = d.gpu_malloc::<u8>(MemoryKind::Upload, len, name).unwrap();
            a.as_mut_slice().unwrap()[..bytes.len()].copy_from_slice(bytes);
            a
        };
        let positions: Vec<[f32; 3]> = mesh.positions.iter().map(Vec3::to_array).collect();
        let gpu = [
            upload(bytemuck::cast_slice(&positions), "plane positions"),
            upload(bytemuck::cast_slice(&attrs), "plane attributes"),
            upload(bytemuck::cast_slice(&mesh.triangles), "plane triangles"),
            upload(bytemuck::cast_slice(&mesh.clusters), "plane clusters"),
        ];
        let list = [scene];
        let f = scenes
            .begin_frame(&FrameDesc {
                time: time(0),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let mut m = MaterialGpu::zeroed();
        m.base_color = [0.5; 3];
        m.base_weight = 1.0;
        m.flags = material_flags::DOUBLE_SIDED;
        let material = f.materials().create(&m).unwrap();
        let mut w = f.scene(scene).unwrap();
        let group = w
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        w.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY;
        drop(w);
        wait(f.end(&[]).unwrap());
        Self {
            scenes,
            scene,
            history,
            group,
            material,
            mesh,
            gpu,
            frame: 1,
        }
    }

    fn desc<'a>(
        &self,
        materials: &'a [MaterialId],
        displacement: u8,
        max_clusters: u32,
    ) -> ProducerDesc<'a> {
        ProducerDesc {
            max_clusters,
            max_vertices: 78 * max_clusters,
            update: ProducerUpdate::Static,
            tessellated: true,
            max_patch_clusters: self.mesh.clusters.len() as u32,
            max_patch_vertices: self.mesh.positions.len() as u32,
            displacement,
            displacement_reach: 0.05,
            materials,
            transform: (self.group, 0),
            ..ProducerDesc::default()
        }
    }

    /// One frame with the camera at `eye` looking at `target` and the focus at `focus`;
    /// with `desc` the entry is set and its patches emitted this frame.
    fn frame(
        &mut self,
        desc: Option<&ProducerDesc<'_>>,
        eye: Vec3,
        target: Vec3,
        focus: Vec3,
    ) -> RasterView {
        let p = pipeline(&patches::MODULE, "main");
        let list = [self.scene];
        let mut f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let fov_y = 0.9;
        {
            let mut w = f.scene(self.scene).unwrap();
            if let Some(desc) = desc {
                w.set_producer(PLANE, Some(desc)).unwrap();
            }
            w.set_focus(focus.as_dvec3(), EXTENT[1] as f32 / fov_y);
        }
        let emit = desc.is_some();
        let g = self.gpu.each_ref().map(orr_gpu::Allocation::gpu);
        let clusters = self.mesh.clusters.len() as u32;
        f.update_scenes(&mut |c| {
            if !emit {
                return;
            }
            let r = c
                .put_root(&patches::Root {
                    scene: c.root,
                    positions: g[0].cast(),
                    attrs: g[1].cast(),
                    triangles: g[2],
                    clusters: g[3].cast(),
                    producer: PLANE.0,
                    count: clusters,
                })
                .unwrap();
            c.cmd.dispatch(&p, r, [1, 1, 1]);
        })
        .unwrap();
        let view = f
            .raster_view(&ViewDesc {
                scene: self.scene,
                history: self.history,
                position: eye.as_dvec3(),
                rotation: look(eye, target),
                projection: Projection { fov_y, near: 0.05 },
                lens: Lens::default(),
                render_extent: UVec2::from(EXTENT),
                jitter: Vec2::ZERO,
                reset_history: self.frame <= 2,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
                target: ViewTarget::AfterRr,
            })
            .unwrap();
        self.frame += 1;
        wait(f.end(&[]).unwrap());
        view
    }

    /// A material made in a frame of its own.
    fn material(&mut self, m: &MaterialGpu) -> MaterialId {
        let f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        let id = f.materials().create(m).unwrap();
        self.frame += 1;
        wait(f.end(&[]).unwrap());
        id
    }

    /// The tessellation figures of the last frame.
    fn stats(&mut self) -> TessellationStats {
        let f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        self.frame += 1;
        wait(f.end(&[]).unwrap());
        self.scenes.stats().tessellation
    }
}

/// A tessellated entry's generated and patch batches as `view`'s scene holds them after
/// its frame.
struct Generated {
    clusters: Vec<ProducerClusterGpu>,
    records: Vec<GeneratedClusterGpu>,
    positions: Vec<Vec3>,
    /// Empty where the entry keeps no previous positions.
    previous: Vec<Vec3>,
    attrs: Vec<[u32; 2]>,
    patch_clusters: Vec<ProducerClusterGpu>,
    patch_positions: Vec<Vec3>,
    patch_previous: Vec<Vec3>,
    state: ProducerStateGpu,
}

fn generated(view: &RasterView, entry: ProducerId) -> Generated {
    let root = read_buffer(view.scene, 1)[0];
    let i = u64::from(entry.0);
    let b = read_buffer(root.batches.offset(i), 1)[0];
    let p = read_buffer(root.patches.offset(i), 1)[0];
    let count = |b: &ProducerBatch| {
        let [c, v] = read_buffer(b.cluster_count.cast::<[u32; 2]>(), 1)[0];
        (
            u64::from(c.min(b.max_clusters)),
            u64::from(v.min(b.max_vertices)),
        )
    };
    let (clusters, vertices) = count(&b);
    let (patch_clusters, patch_vertices) = count(&p);
    let v3 = |p: Vec<[f32; 3]>| p.into_iter().map(Vec3::from).collect();
    let previous = |b: &ProducerBatch, n| {
        if b.previous.address().0 == 0 {
            Vec::new()
        } else {
            v3(read_buffer(b.previous.cast(), n))
        }
    };
    Generated {
        clusters: read_buffer(b.clusters, clusters),
        records: read_buffer(b.generated, clusters),
        positions: v3(read_buffer(b.positions.cast(), vertices)),
        previous: previous(&b, vertices),
        attrs: read_buffer(b.attrs.cast(), vertices),
        patch_clusters: read_buffer(p.clusters, patch_clusters),
        patch_positions: v3(read_buffer(p.positions.cast(), patch_vertices)),
        patch_previous: previous(&p, patch_vertices),
        state: read_buffer(root.producer_state.offset(i), 1)[0],
    }
}

/// The undisplaced point at weights `w` of corners `p`, as the tessellator interpolates it.
fn interpolate(p: [Vec3; 3], w: [u32; 3]) -> Vec3 {
    let f = w.map(|x| x as f32 * (1.0 / tessellation::BARY_ONE as f32));
    p[0] * f[0] + p[1] * f[1] + p[2] * f[2]
}

/// One generated vertex: its index, its patch triangle (cluster, triangle), its weights in
/// that triangle and the triangle's corners.
struct Weighted {
    vertex: usize,
    patch: u32,
    w: [u32; 3],
    corners: [Vec3; 3],
    /// Last frame's corners, where the batch keeps them.
    previous: Option<[Vec3; 3]>,
}

fn vertex_weights(g: &Generated) -> Vec<Weighted> {
    let mesh = PatchMesh::new(SEAM_UV);
    let mut out = Vec::new();
    for (k, r) in g.clusters.iter().zip(&g.records) {
        let patch = &g.patch_clusters[(r.patch >> 7) as usize];
        let cpu = mesh.clusters[patch.user as usize];
        let tri = mesh.triangles[(cpu[2] + (r.patch & 127)) as usize];
        let at = |c: u8| (patch.first_vertex + u32::from(c)) as usize;
        let corners = tri.map(|c| g.patch_positions[at(c)]);
        let previous = (!g.patch_previous.is_empty()).then(|| tri.map(|c| g.patch_previous[at(c)]));
        let pattern = tessellation::pattern(r.pattern);
        let w = pattern.place(tessellation::unpack_factors(r.factors), r.corners, r.zeros);
        assert_eq!(w.len(), usize::from(k.vertex_count));
        for (i, &w) in w.iter().enumerate() {
            out.push(Weighted {
                vertex: k.first_vertex as usize + i,
                patch: r.patch,
                w,
                corners,
                previous,
            });
        }
    }
    out
}

/// The points of each patch triangle's edge vertices, by edge (its end points' bits,
/// sorted) and patch triangle.
type EdgePoints = HashMap<[[u32; 3]; 2], HashMap<u32, HashSet<[u32; 3]>>>;

fn edge_points(weights: &[Weighted], point: impl Fn(&Weighted) -> Vec3) -> EdgePoints {
    // Zeros of either sign are one point.
    let bits = |p: Vec3| p.to_array().map(|x| (x + 0.0).to_bits());
    let mut edges = EdgePoints::default();
    for v in weights {
        for e in 0..3 {
            if v.w[e] != 0 {
                continue;
            }
            let mut ends = [bits(v.corners[(e + 1) % 3]), bits(v.corners[(e + 2) % 3])];
            ends.sort_unstable();
            edges
                .entry(ends)
                .or_default()
                .entry(v.patch)
                .or_default()
                .insert(bits(point(v)));
        }
    }
    edges
}

#[test]
fn tessellated_patches_dice_at_mixed_rates_with_bit_identical_shared_edges() {
    let mut plane = Plane::new(4.0);
    let materials = [plane.material];
    let desc = ProducerDesc {
        previous_positions: true,
        ..plane.desc(&materials, 0, 4096)
    };
    let (eye, target) = (Vec3::new(0.3, 1.0, 0.5), Vec3::new(0.0, 0.0, -3.0));
    plane.frame(Some(&desc), eye, target, eye);
    let view = plane.frame(None, eye, target, eye);
    let g = generated(&view, PLANE);
    let stats = plane.stats();
    assert_eq!(stats.patches, 48, "{stats:?}");
    assert!(stats.subpatches > 2 * stats.patches, "{stats:?}");
    assert_eq!(
        stats.dice_dropped + stats.split_dropped + stats.weld_dropped,
        0,
        "{stats:?}"
    );
    let patterns: HashSet<u32> = g.records.iter().map(|r| r.pattern).collect();
    assert!(patterns.len() > 10, "{} patterns", patterns.len());

    // Every vertex lies where the CPU interpolates its weights, exactly, and last frame's
    // position interpolates last frame's corners at the same weights.
    let weights = vertex_weights(&g);
    for v in &weights {
        assert_eq!(
            g.positions[v.vertex].to_array(),
            interpolate(v.corners, v.w).to_array(),
            "vertex {}",
            v.vertex
        );
        let previous = v.previous.expect("previous corners");
        assert_eq!(
            g.previous[v.vertex].to_array(),
            interpolate(previous, v.w).to_array(),
            "previous of vertex {}",
            v.vertex
        );
    }
    // The two patch triangles on each side of a shared edge put the same points on it.
    let edges = edge_points(&weights, |v| g.positions[v.vertex]);
    let mut shared = 0;
    for (ends, sides) in &edges {
        if sides.len() == 2 {
            let mut s = sides.values();
            let (a, b) = (s.next().unwrap(), s.next().unwrap());
            assert_eq!(a, b, "edge {ends:?}");
            assert!(a.len() >= 2, "edge {ends:?}");
            shared += 1;
        }
    }
    assert_eq!(shared, 62, "shared edges");
    assert_clean();
}

/// Per pixel of `view`: the distance to the producer surface the raster shows and to the
/// one the camera ray hits, 0 for none.
fn depths(view: &RasterView) -> Vec<tess_rays::Depths> {
    let d = device();
    let n = u64::from(EXTENT[0] * EXTENT[1]);
    let out = d
        .gpu_malloc::<tess_rays::Depths>(MemoryKind::Readback, n, "depths")
        .unwrap();
    let mut root = d
        .gpu_malloc::<tess_rays::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = tess_rays::Root {
        scene: view.scene,
        view: view.view,
        gbuffer: view.gbuffer,
        out: out.gpu(),
    };
    let module = linked::KERNELS.module(&tess_rays::MODULE).unwrap();
    let p = pipeline(module, "main");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
    list.dispatch(&p, root.gpu(), [n.div_ceil(64) as u32, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    out.as_slice().unwrap().to_vec()
}

/// The unit normal of the rippled plane at (x, z) on side `right` of the seam.
fn ripple_normal(x: f32, z: f32, right: bool) -> Vec3 {
    let step = 1e-3;
    let h = |x: f32, z: f32| ripple(plane_uv(x, z, right));
    let dx = (h(x + step, z) - h(x - step, z)) / (2.0 * step);
    let dz = (h(x, z + step) - h(x, z - step)) / (2.0 * step);
    Vec3::new(-dx, 1.0, -dz).normalize()
}

#[test]
fn a_displaced_plane_with_a_uv_seam_closes_in_raster_and_rays_with_normals_of_its_field() {
    if !traced(true) {
        return;
    }
    let mut plane = Plane::new(3.0);
    let materials = [plane.material];
    let desc = plane.desc(&materials, RIPPLE, 8192);
    // The camera looks across the seam from one side; rates fall away from the focus at
    // the eye, and part of the plane is out of view.
    let (eye, target) = (Vec3::new(2.4, 0.8, -1.5), Vec3::new(-1.0, 0.0, -3.0));
    plane.frame(Some(&desc), eye, target, eye);
    for _ in 0..2 {
        plane.frame(None, eye, target, eye);
    }
    let view = plane.frame(None, eye, target, eye);
    let g = generated(&view, PLANE);
    let seen = depths(&view);
    let stats = plane.stats();
    assert!(stats.welded > 0, "{stats:?}");
    assert_eq!(
        stats.dice_dropped + stats.split_dropped + stats.weld_dropped,
        0,
        "{stats:?}"
    );
    let patterns: HashSet<u32> = g.records.iter().map(|r| r.pattern).collect();
    assert!(patterns.len() > 10, "{} patterns", patterns.len());

    // Seam vertices took one displaced position: the two sides of every patch edge put
    // their vertices at the same displaced points.
    let weights = vertex_weights(&g);
    let edges = edge_points(&weights, |v| g.positions[v.vertex]);
    for (ends, sides) in &edges {
        if sides.len() == 2 {
            let mut s = sides.values();
            assert_eq!(s.next(), s.next(), "edge {ends:?}");
        }
    }

    // Bounds hold every displaced vertex.
    let lo = g.state.lo.map(unordered);
    let hi = g.state.hi.map(unordered);
    for p in &g.positions {
        assert!(
            (0..3).all(|k| lo[k] <= p[k] && p[k] <= hi[k]),
            "{p} outside {lo:?}..{hi:?}"
        );
    }
    assert!(
        hi[1] - lo[1] > 0.05,
        "the ripple moves the plane: {lo:?}..{hi:?}"
    );

    // No pixel inside the plane's footprint shows the background, through the raster or a
    // camera ray, and both agree on depth.
    let (mut inside, mut far_apart) = (0, 0);
    for (i, s) in seen.iter().enumerate() {
        if !in_footprint(eye, target, i, 0.05) {
            continue;
        }
        inside += 1;
        assert!(s.raster > 0.0, "pixel {i}: raster shows the background");
        assert!(s.traced > 0.0, "pixel {i}: the camera ray misses");
        if (s.raster - s.traced).abs() > 1e-3 * s.traced {
            far_apart += 1;
        }
    }
    assert!(inside > EXTENT[0] * EXTENT[1] / 4, "{inside} pixels inside");
    assert!(
        far_apart * 1000 <= inside,
        "{far_apart} of {inside} disagree on depth"
    );

    // Written normals follow the displaced field away from the seam.
    let mut worst = 0.0f32;
    for (p, a) in g.positions.iter().zip(&g.attrs) {
        if p.x.abs() < 1e-4 {
            continue;
        }
        let n = Vec3::from(orr_pack::mesh::oct_decode(a[0]));
        let want = ripple_normal(p.x, p.z, p.x > 0.0);
        worst = worst.max(n.dot(want).clamp(-1.0, 1.0).acos());
    }
    assert!(worst < NORMAL_TOLERANCE, "normals within {worst} rad");
    assert_clean();
}

/// The unit direction of the camera ray through pixel `i`'s center.
fn pixel_ray(eye: Vec3, target: Vec3, i: usize) -> Vec3 {
    let fov_y = 0.9f32;
    let aspect = EXTENT[0] as f32 / EXTENT[1] as f32;
    let (px, py) = ((i as u32 % EXTENT[0]) as f32, (i as u32 / EXTENT[0]) as f32);
    let ndc = Vec2::new(
        (px + 0.5) / EXTENT[0] as f32 * 2.0 - 1.0,
        1.0 - (py + 0.5) / EXTENT[1] as f32 * 2.0,
    );
    let t = (fov_y * 0.5).tan();
    look(eye, target) * Vec3::new(ndc.x * t * aspect, ndc.y * t, -1.0).normalize()
}

/// Whether the camera ray of pixel `i` crosses the plane's patches, lifted or lowered by up
/// to `reach`, inside their rectangle and away from its border.
fn in_footprint(eye: Vec3, target: Vec3, i: usize, reach: f32) -> bool {
    let d = pixel_ray(eye, target, i);
    if d.y >= -0.05 {
        return false;
    }
    let within = |y: f32| {
        let hit = eye + d * ((y - eye.y) / d.y);
        hit.x.abs() <= 1.9 && hit.z <= -0.1 && hit.z >= -5.9
    };
    within(reach) && within(-reach)
}

/// Radians between a written normal and the field's, at most (0.0007 measured): central
/// differences and the 16-bit octahedral code.
const NORMAL_TOLERANCE: f32 = 0.003;

/// `producer.slang`'s `unordered`.
fn unordered(u: u32) -> f32 {
    f32::from_bits(if u & 0x8000_0000 != 0 {
        u & 0x7fff_ffff
    } else {
        !u
    })
}

#[test]
fn tessellation_steps_its_rate_down_near_capacity_and_back_up_with_room() {
    let mut plane = Plane::new(2.0);
    let materials = [plane.material];
    let desc = plane.desc(&materials, 0, 512);
    let (eye, target) = (Vec3::new(0.0, 0.6, 0.2), Vec3::new(0.0, 0.0, -3.0));
    plane.frame(Some(&desc), eye, target, eye);
    let mut lowest = 1.0f32;
    for _ in 0..12 {
        plane.frame(None, eye, target, eye);
        lowest = lowest.min(plane.stats().lowest_rate_scale);
    }
    assert!(lowest < 0.9, "rate scale {lowest}");
    // Far away the plane takes little room, and the rate climbs back.
    let far = Vec3::new(0.0, 60.0, 40.0);
    let mut last = 0.0;
    for _ in 0..30 {
        plane.frame(None, eye, target, far);
        last = plane.stats().lowest_rate_scale;
    }
    assert_eq!(last, 1.0, "rate scale after recovering");
    assert_clean();
}

#[test]
fn texture_displacement_moves_patches_along_the_normal_by_the_materials_texture() {
    let d = device();
    let mut plane = Plane::new(4.0);
    // Every texel of a 4 x 4 R16 displacement texture at 0.75.
    let texel = (0.75f32 * 65535.0).round() as u16;
    let desc = TextureDesc::d2(Format::R16Unorm, 4, 4, TextureUsage::SAMPLED);
    let texture = d
        .create_texture(&desc, TextureMemory::Dedicated, "displacement texture")
        .unwrap();
    let mut staging = d
        .gpu_malloc::<u16>(MemoryKind::Upload, 16, "displacement texels")
        .unwrap();
    staging.as_mut_slice().unwrap().fill(texel);
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.discard(&texture);
    list.copy_to_texture(
        &texture,
        &TextureRegion::whole_mip(&desc, 0),
        staging.gpu().cast(),
    );
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    let slot = d.alloc_image_slot().unwrap();
    let mut m = MaterialGpu::zeroed();
    m.base_color = [0.5; 3];
    m.base_weight = 1.0;
    m.flags = material_flags::DOUBLE_SIDED | material_flags::DISPLACED;
    m.textures = [TextureIndex::NONE; 12];
    m.textures[orr_pack::material::slot::DISPLACEMENT] =
        d.write_sampled(slot, &texture, &orr_gpu::ViewDesc::default());
    m.sampler = d.sampler(&SamplerDesc::default()).unwrap();
    m.displacement_scale = 0.2;
    m.displacement_offset = -0.1;
    let material = plane.material(&m);
    let materials = [material];
    let desc = plane.desc(&materials, 0, 4096);
    let (eye, target) = (Vec3::new(0.3, 1.0, 0.5), Vec3::new(0.0, 0.0, -3.0));
    plane.frame(Some(&desc), eye, target, eye);
    let view = plane.frame(None, eye, target, eye);
    let g = generated(&view, PLANE);
    let stats = plane.stats();
    // The same texel on both sides of the seam: nothing for the weld to move.
    assert_eq!(stats.welded, 0, "{stats:?}");
    let want = 0.2 * f32::from(texel) / 65535.0 - 0.1;
    assert!(!g.positions.is_empty());
    for (p, a) in g.positions.iter().zip(&g.attrs) {
        assert!((p.y - want).abs() < 1e-4, "{p} against height {want}");
        let n = Vec3::from(orr_pack::mesh::oct_decode(a[0]));
        assert!(n.dot(Vec3::Y) > 0.9999, "normal {n}");
    }
    d.free_image_slot_after_submitted(slot);
    assert_clean();
}

/// Pixels whose depth may jump in a step: a thin subpatch beside a count change can win a
/// pixel or two through its extended plane, while a pop moves a whole region.
const ISOLATED: usize = 16;

/// The largest change of raster depth, past the [`ISOLATED`] largest, between consecutive
/// frames as the focus rises from the eye by 2 m in `steps`, the camera still over the
/// rippled plane, and the pattern counts of the first and the last frame. The plane has no
/// seam: where a seam's sides displace differently, the weld closes it with a wall whose
/// width follows the pattern beside it, which jumps as counts change.
fn rate_sweep(steps: u32) -> (f32, [HashSet<[u32; 3]>; 2]) {
    let (eye, target) = (Vec3::new(0.2, 1.4, 0.6), Vec3::new(0.0, 0.0, -3.0));
    let focus = |i: u32| eye + Vec3::Y * (2.0 * i as f32 / steps as f32);
    let mut plane = Plane::build(
        8.0,
        RendererDesc::default().software_raster_pixels,
        Vec2::ZERO,
    );
    let materials = [plane.material];
    let desc = plane.desc(&materials, RIPPLE, 8192);
    plane.frame(Some(&desc), eye, target, focus(0));
    plane.frame(None, eye, target, focus(0));
    let counts = |view: &RasterView| -> HashSet<[u32; 3]> {
        generated(view, PLANE)
            .records
            .iter()
            .map(|r| tessellation::pattern(r.pattern).counts)
            .collect()
    };
    let view = plane.frame(None, eye, target, focus(0));
    let first = counts(&view);
    let mut last = depths(&view);
    let mut worst = 0.0f32;
    let mut view = view;
    for i in 1..=steps {
        view = plane.frame(None, eye, target, focus(i));
        let d = depths(&view);
        let mut moved: Vec<f32> = last
            .iter()
            .zip(&d)
            .filter(|(a, b)| a.raster > 0.0 && b.raster > 0.0)
            .map(|(a, b)| (a.raster - b.raster).abs())
            .collect();
        moved.sort_by(|a, b| b.total_cmp(a));
        worst = worst.max(moved[ISOLATED]);
        last = d;
    }
    // Nothing dropped and the rate never stepped, which would move the surface at once.
    let stats = plane.stats();
    assert_eq!(stats.dice_dropped + stats.split_dropped, 0, "{stats:?}");
    assert_eq!(stats.lowest_rate_scale, 1.0, "{stats:?}");
    (worst, [first, counts(&view)])
}

#[test]
fn rates_moving_through_count_changes_move_the_displaced_surface_without_steps() {
    if !traced(true) {
        return;
    }
    // Pattern counts change along the sweep, yet the surface only moves as far as the rates
    // do: four times the steps move it a quarter as far per step, where a pop would stay.
    let (coarse, counts) = rate_sweep(6);
    let (fine, _) = rate_sweep(24);
    assert_ne!(counts[0], counts[1], "the sweep changes pattern counts");
    assert!(coarse > 0.0, "the surface moves");
    assert!(
        fine < 0.4 * coarse,
        "largest step {fine} m in 24 steps against {coarse} m in 6"
    );
    assert_clean();
}

/// Displacement id of `surfaces/cliffs.slang` in the suite's linked kernels.
const CLIFFS: u8 = 2;

#[test]
fn steep_cliffs_draw_stretched_diced_triangles_in_hardware_without_holes() {
    if !traced(true) {
        return;
    }
    let (eye, target) = (Vec3::new(1.8, 1.1, 0.4), Vec3::new(-0.6, 0.0, -3.0));
    // With a low software limit most diced clusters are split by size between the paths;
    // without the software raster every cluster draws in hardware.
    let render = |software_pixels: f32| {
        let mut plane = Plane::build(3.0, software_pixels, SEAM_UV);
        let materials = [plane.material];
        let desc = ProducerDesc {
            displacement_reach: 0.2,
            ..plane.desc(&materials, CLIFFS, 16_384)
        };
        plane.frame(Some(&desc), eye, target, eye);
        for _ in 0..2 {
            plane.frame(None, eye, target, eye);
        }
        let view = plane.frame(None, eye, target, eye);
        let counts: Vec<u32> = read_buffer(view.counts, 8);
        let clusters: Vec<VisibleCluster> =
            read_buffer(view.targets.clusters, u64::from(counts[0]));
        let stats = plane.stats();
        assert_eq!(
            stats.dice_dropped + stats.split_dropped + stats.weld_dropped,
            0,
            "{stats:?}"
        );
        (depths(&view), clusters, counts)
    };
    let (split, clusters, counts) = render(8.0);
    // `gbuffer.slang` VISIBLE_BY_SIZE.
    let by_size = clusters.iter().filter(|c| c.local & 1 != 0).count();
    assert!(by_size > 0, "no cluster split by size");
    assert!(counts[5] > 0 && counts[7] > 0, "draws {counts:?}");
    let (hardware, _, _) = render(0.0);

    let (mut inside, mut differ) = (0, 0);
    for (i, (s, h)) in split.iter().zip(&hardware).enumerate() {
        if !in_footprint(eye, target, i, 0.2) {
            continue;
        }
        inside += 1;
        assert!(s.traced > 0.0, "pixel {i}: the camera ray misses");
        assert!(
            s.raster > 0.0,
            "pixel {i}: the split raster shows the background"
        );
        assert!(
            h.raster > 0.0,
            "pixel {i}: the hardware raster shows the background"
        );
        // Both paths snap and sample alike; pixels on an edge between a wall and the
        // ground behind it may go to either.
        if (s.raster - h.raster).abs() > 1e-3 * h.raster {
            differ += 1;
        }
    }
    assert!(inside > EXTENT[0] * EXTENT[1] / 4, "{inside} pixels inside");
    assert!(
        differ * 100 <= inside,
        "{differ} of {inside} pixels differ from the hardware raster"
    );
    assert_clean();
}

/// A cube with one infinitely sharp edge, an icosahedron (triangles around valence-5
/// points) with a semi-sharp edge and a pentagonal prism with a sharp corner, side by side
/// along x, each closed and wound outward, with planar uvs.
fn closed_cages() -> CageInput {
    let mut c = CageInput::default();
    let mut add = |points: Vec<Vec3>, faces: Vec<Vec<u32>>, at: Vec3| {
        let base = c.positions.len() as u32;
        for f in faces {
            // Wound so the face's normal points away from the shape's center.
            let p: Vec<Vec3> = f.iter().map(|&i| points[i as usize]).collect();
            let mut n = Vec3::ZERO;
            for k in 0..p.len() {
                n += p[k].cross(p[(k + 1) % p.len()]);
            }
            let mut f: Vec<u32> = f.iter().map(|&i| base + i).collect();
            if n.dot(p.iter().sum::<Vec3>()) < 0.0 {
                f.reverse();
            }
            c.face_counts.push(f.len() as u32);
            c.face_indices.extend(f);
        }
        c.positions.extend(points.iter().map(|&p| p + at));
        base
    };
    let cube: Vec<Vec3> = (0..8)
        .map(|i| {
            let b = |k: u32| if i & (1 << k) != 0 { 0.5 } else { -0.5 };
            Vec3::new(b(0), b(1), b(2))
        })
        .collect();
    let cube_faces = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    let first = add(
        cube,
        cube_faces.iter().map(|f| f.to_vec()).collect(),
        Vec3::new(-2.0, 0.0, 0.0),
    );
    c.creases.push(Crease {
        edge: [first, first + 1],
        sharpness: 10.0,
    });
    let phi = (1.0 + 5f32.sqrt()) / 2.0;
    let mut ico = Vec::new();
    for a in [-1.0, 1.0] {
        for b in [-phi, phi] {
            ico.push(Vec3::new(0.0, a, b));
            ico.push(Vec3::new(a, b, 0.0));
            ico.push(Vec3::new(b, 0.0, a));
        }
    }
    let mut ico_faces = Vec::new();
    for i in 0..12 {
        for j in i + 1..12 {
            for k in j + 1..12 {
                let edge = |a: usize, b: usize| (ico[a].distance(ico[b]) - 2.0).abs() < 1e-4;
                if edge(i, j) && edge(j, k) && edge(i, k) {
                    ico_faces.push(vec![i as u32, j as u32, k as u32]);
                }
            }
        }
    }
    assert_eq!(ico_faces.len(), 20);
    let ico: Vec<Vec3> = ico.iter().map(|p| *p * 0.4).collect();
    let [a, b] = [ico_faces[0][0], ico_faces[0][1]];
    let first = add(ico, ico_faces, Vec3::ZERO);
    c.creases.push(Crease {
        edge: [first + a, first + b],
        sharpness: 2.5,
    });
    let mut prism = Vec::new();
    for y in [-0.5, 0.5] {
        for k in 0..5 {
            let a = std::f32::consts::TAU * k as f32 / 5.0;
            prism.push(Vec3::new(0.6 * a.cos(), y, 0.6 * a.sin()));
        }
    }
    let mut prism_faces = vec![(0..5).collect::<Vec<u32>>(), (5..10).collect()];
    for k in 0..5 {
        prism_faces.push(vec![k, (k + 1) % 5, 5 + (k + 1) % 5, 5 + k]);
    }
    let first = add(prism, prism_faces, Vec3::new(2.0, 0.0, 0.0));
    c.corners.push(Corner {
        point: first + 5,
        sharpness: 3.0,
    });
    let values = c
        .positions
        .iter()
        .map(|p| Vec2::new(p.x + 0.4 * p.y, p.z + 0.3 * p.y) * 0.5)
        .collect();
    c.uvs = Some(FaceVarying {
        values,
        indices: c.face_indices.clone(),
    });
    c
}

/// A box arm of three rings along y, skinned to a root joint at y = 0 and a tip joint at
/// y = 1 under a SkelRoot, cooked as a cage.
const ARM: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1
)

def Xform "World"
{
    def SkelRoot "Arm"
    {
        def Skeleton "Rig"
        {
            uniform token[] joints = ["Root", "Root/Tip"]
            uniform matrix4d[] bindTransforms = [((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 0, 1))]
            uniform matrix4d[] restTransforms = [((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 0, 1))]
        }

        def Mesh "Body"
        {
            uniform token subdivisionScheme = "catmullClark"
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7, 4, 5, 9, 8, 5, 6, 10, 9, 6, 7, 11, 10, 7, 4, 8, 11, 3, 2, 1, 0, 8, 9, 10, 11]
            point3f[] points = [(-0.3, 0, -0.3), (-0.3, 0, 0.3), (0.3, 0, 0.3), (0.3, 0, -0.3), (-0.3, 1, -0.3), (-0.3, 1, 0.3), (0.3, 1, 0.3), (0.3, 1, -0.3), (-0.3, 2, -0.3), (-0.3, 2, 0.3), (0.3, 2, 0.3), (0.3, 2, -0.3)]
            int[] primvars:skel:jointIndices = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0] (
                elementSize = 2
                interpolation = "vertex"
            )
            float[] primvars:skel:jointWeights = [1, 0, 1, 0, 1, 0, 1, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0, 1, 0, 1, 0, 1, 0] (
                elementSize = 2
                interpolation = "vertex"
            )
            rel skel:skeleton = </World/Arm/Rig>
            rel material:binding = </World/Looks/Skin>
        }
    }

    def Scope "Looks"
    {
        def Material "Skin"
        {
            token outputs:surface.connect = </World/Looks/Skin/S.outputs:surface>
            def Shader "S"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.6, 0.5, 0.4)
                token outputs:surface
            }
        }
    }
}
"#;

/// The cages: `cages` (cage only) and the skinned `arm/Arm` from USD.
fn cage_pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("subd-arm");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("arm.usda"), ARM).unwrap();
        let material = orr_pack::material::Material {
            double_sided: 1,
            ..Default::default()
        };
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                orr_pack::Target::Memory,
                vec![
                    AssetSpec {
                        name: "cages".into(),
                        source: Source::Subd {
                            cage: closed_cages(),
                            materials: vec![material],
                            settings: SubdSettings::default(),
                        },
                    },
                    AssetSpec {
                        name: "arm".into(),
                        source: Source::Usd {
                            path: dir.join("arm.usda"),
                            settings: UsdSettings {
                                subdivision: Subdivision::Cage,
                                ..UsdSettings::default()
                            },
                        },
                    },
                ],
            )],
        };
        orr_cook::cook(&manifest, &Default::default())
            .unwrap()
            .memory
            .remove(0)
    })
}

/// A cage entry's blob 0 and pages, parsed from the pack.
fn cage_entry(name: &str) -> (SubdInfo, Vec<SubdPage>) {
    let a = orr_pack::Archive::from_memory(Arc::clone(cage_pack())).unwrap();
    let e = a.find(AssetId::from_name(name)).unwrap();
    let read = |b: &orr_pack::Blob| {
        let mut v = vec![0; b.raw_len as usize];
        a.read(b, &mut v).unwrap();
        v
    };
    let blobs = a.blobs(e);
    let info = SubdInfo::parse(&read(&blobs[0])).unwrap();
    let pages = info
        .pages
        .iter()
        .map(|p| SubdPage::parse(&read(&blobs[p.blob as usize])).unwrap())
        .collect();
    (info, pages)
}

/// A cage's limit surface on the CPU, posed by `points`, and the nearest point of it to
/// any point near it.
struct LimitSurface {
    pages: Vec<PosedPage>,
    /// Samples by grid cell.
    cells: HashMap<[i32; 3], Vec<(Vec3, Spot)>>,
    cell: f32,
}

/// A page with its points and uv points.
struct PosedPage {
    page: SubdPage,
    points: Vec<[f32; 3]>,
    uv_points: Vec<[f32; 2]>,
}

/// Face coordinates on face `face` of page `page`.
#[derive(Clone, Copy)]
struct Spot {
    page: usize,
    face: usize,
    u: f64,
    v: f64,
}

/// A point [`LimitSurface`] finds.
#[derive(Clone, Copy)]
struct Nearest {
    position: Vec3,
    /// From differences of positions: Gregory patches' basis derivatives leave out their
    /// rational blend.
    normal: Vec3,
    uv: Vec2,
    distance: f32,
    /// Radians between the normals of the faces that meet at the point.
    spread: f32,
}

impl LimitSurface {
    fn new(info: &SubdInfo, pages: &[SubdPage], points: &[[f32; 3]]) -> Self {
        const GRID: usize = 12;
        let mut s = Self {
            pages: pages
                .iter()
                .map(|p| PosedPage {
                    page: p.clone(),
                    points: p.points(points),
                    uv_points: p.uv_points(&info.uvs),
                })
                .collect(),
            cells: HashMap::new(),
            cell: 0.08,
        };
        for page in 0..s.pages.len() {
            for face in 0..s.pages[page].page.faces.len() {
                for i in 0..=GRID {
                    for j in 0..=GRID {
                        let (u, v) = (i as f64 / GRID as f64, j as f64 / GRID as f64);
                        let spot = Spot { page, face, u, v };
                        let p = Vec3::from(s.at(spot).position);
                        let key = (p / s.cell).floor().as_ivec3().to_array();
                        s.cells.entry(key).or_default().push((p, spot));
                    }
                }
            }
        }
        s
    }

    fn at(&self, s: Spot) -> orr_pack::subd::Limit {
        let p = &self.pages[s.page];
        p.page
            .limit(s.face, s.u as f32, s.v as f32, &p.points)
            .unwrap()
    }

    /// The faces with samples nearest `x`, each at its nearest sample.
    fn starts(&self, x: Vec3) -> Vec<Spot> {
        let c = (x / self.cell).floor().as_ivec3();
        let mut near: Vec<(f32, Spot)> = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let key = (c + glam::IVec3::new(dx, dy, dz)).to_array();
                    for &(p, spot) in self.cells.get(&key).into_iter().flatten() {
                        near.push((p.distance(x), spot));
                    }
                }
            }
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut starts: Vec<Spot> = Vec::new();
        for (_, n) in near {
            if starts.len() < 8 && !starts.iter().any(|s| (s.page, s.face) == (n.page, n.face)) {
                starts.push(n);
            }
        }
        starts
    }

    /// The point of `from`'s face nearest `x`, by Gauss-Newton from `from` with coordinates
    /// clamped to the face.
    fn project(&self, x: Vec3, from: Spot) -> Spot {
        let mut s = from;
        // Derivatives of an n-sided face's patches are twice those in its face coordinates
        // (OpenSubdiv's scale), which halves each step there.
        for _ in 0..48 {
            let l = self.at(s);
            let r = (Vec3::from(l.position) - x).as_dvec3();
            let (a, b) = (Vec3::from(l.du).as_dvec3(), Vec3::from(l.dv).as_dvec3());
            let (aa, ab, bb) = (a.dot(a), a.dot(b), b.dot(b));
            let det = aa * bb - ab * ab;
            if det.abs() < 1e-20 {
                break;
            }
            let (ra, rb) = (a.dot(r), b.dot(r));
            let (du, dv) = ((bb * ra - ab * rb) / det, (aa * rb - ab * ra) / det);
            s.u = (s.u - du).clamp(0.0, 1.0);
            s.v = (s.v - dv).clamp(0.0, 1.0);
            if du.abs().max(dv.abs()) < 1e-10 {
                break;
            }
        }
        s
    }

    /// The surface points nearest `x` on each of the faces with samples nearest it.
    fn candidates(&self, x: Vec3) -> Vec<Nearest> {
        self.starts(x)
            .into_iter()
            .map(|s| self.describe(x, self.project(x, s)))
            .collect()
    }

    /// How far `x` is from the surface displaced along its normal by `height` of the uv,
    /// and that height, on the face that explains it best: on each face, the point whose
    /// displacement lands nearest `x`, found by projecting `x` moved back along the normal.
    fn displaced_distance(&self, x: Vec3, height: impl Fn(Vec2) -> f32) -> (f32, f32) {
        self.starts(x)
            .into_iter()
            .map(|s| {
                let mut s = self.project(x, s);
                for _ in 0..8 {
                    let n = self.describe(x, s);
                    s = self.project(x - n.normal * height(n.uv), s);
                }
                let n = self.describe(x, s);
                let h = height(n.uv);
                ((x - n.position - n.normal * h).length(), h)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap_or_else(|| panic!("no surface near {x}"))
    }

    fn describe(&self, x: Vec3, s: Spot) -> Nearest {
        let position = Vec3::from(self.at(s).position);
        let page = &self.pages[s.page];
        // Differences of positions, one-sided at the face's borders.
        let h = 1e-3;
        let p = |u: f64, v: f64| Vec3::from(self.at(Spot { u, v, ..s }).position).as_dvec3();
        let (u0, u1) = ((s.u - h).max(0.0), (s.u + h).min(1.0));
        let (v0, v1) = ((s.v - h).max(0.0), (s.v + h).min(1.0));
        let normal = (p(u1, s.v) - p(u0, s.v)).cross(p(s.u, v1) - p(s.u, v0));
        let uv = page
            .page
            .limit_uv(s.face, s.u as f32, s.v as f32, &page.uv_points)
            .unwrap_or_default();
        Nearest {
            position,
            normal: normal.normalize().as_vec3(),
            uv: Vec2::from(uv),
            distance: position.distance(x),
            spread: 0.0,
        }
    }

    /// The surface point nearest `x`, with the spread of the normals of the faces that
    /// meet there.
    fn nearest(&self, x: Vec3) -> Nearest {
        let found = self.candidates(x);
        let best = found
            .iter()
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
            .unwrap_or_else(|| panic!("no surface near {x}"));
        let spread = found
            .iter()
            .filter(|c| c.position.distance(best.position) < 1e-5)
            .map(|c| c.normal.dot(best.normal).clamp(-1.0, 1.0).acos())
            .fold(0.0, f32::max);
        Nearest { spread, ..*best }
    }
}

/// Asserts the generated surface is closed and consistently wound: every edge of its
/// triangles is walked the other way, between the same two end points bit for bit, and in
/// each direction by at most one triangle that is not a sliver ([`SLIVER`]). Triangles with
/// two corners at one point are left out. Returns the triangle count.
fn assert_closed(g: &Generated) -> usize {
    let bits = |v: u32| {
        g.positions[v as usize]
            .to_array()
            .map(|x| (x + 0.0).to_bits())
    };
    let mut directed: HashMap<([u32; 3], [u32; 3]), Vec<f64>> = HashMap::new();
    let mut triangles = 0;
    for (k, r) in g.clusters.iter().zip(&g.records) {
        let pattern = tessellation::pattern(r.pattern);
        for t in &pattern.triangles {
            let v = t.map(|c| k.first_vertex + u32::from(c));
            if (0..3).any(|e| bits(v[e]) == bits(v[(e + 1) % 3])) {
                continue;
            }
            let p = v.map(|i| g.positions[i as usize].as_dvec3());
            let longest = (0..3)
                .map(|e| p[e].distance(p[(e + 1) % 3]))
                .fold(0.0, f64::max);
            let height = (p[1] - p[0]).cross(p[2] - p[0]).length() / longest;
            for e in 0..3 {
                directed
                    .entry((bits(v[e]), bits(v[(e + 1) % 3])))
                    .or_default()
                    .push(height);
            }
            triangles += 1;
        }
    }
    let (mut open, mut overlapping) = (0, 0);
    for (&(a, b), heights) in &directed {
        if !directed.contains_key(&(b, a)) {
            open += 1;
        }
        if heights.iter().filter(|&&h| h > SLIVER).count() > 1 {
            overlapping += 1;
        }
    }
    assert_eq!(
        (open, overlapping),
        (0, 0),
        "open and overlapping of {} edges",
        directed.len()
    );
    triangles
}

/// m: triangles lower than this are slivers. Subpatches near a change of count can be
/// thinner than a float's resolution, and their inner vertices sit on their own patch's
/// limit surface, which the neighbor's evaluation differs from by up to the 5.4e-5 m the
/// rippled copy is held to; the fixtures' diced triangles are centimeters high.
const SLIVER: f64 = 1e-3;

/// A mounted cage pack and a scene with room for two copies.
struct CageScene {
    scenes: Scenes,
    _workers: Workers,
    mount: MountId,
    scene: SceneId,
    history: ViewHistoryId,
    frame: u32,
}

impl CageScene {
    fn new(dice_pixels: f32) -> Self {
        let mut scenes = Scenes::new(
            device(),
            &RendererDesc {
                kernels: &linked::KERNELS,
                dice_pixels,
                ..RendererDesc::default()
            },
        )
        .unwrap();
        let workers = Workers::start(&scenes, 2);
        let mount = scenes.mount(PackSource::Memory(Arc::clone(cage_pack())));
        let start = Instant::now();
        while matches!(scenes.mount_state(mount), LoadState::Pending) {
            let f = scenes
                .begin_frame(&FrameDesc {
                    time: time(0),
                    scenes: &[],
                    low_latency: None,
                })
                .unwrap();
            f.end(&[]).unwrap();
            assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
            std::thread::sleep(Duration::from_millis(1));
        }
        if let LoadState::Failed(e) = scenes.mount_state(mount) {
            panic!("mount failed: {e}");
        }
        let scene = scenes
            .create_scene(&SceneDesc {
                producer_capacity: 2,
                ..SceneDesc::default()
            })
            .unwrap();
        let history = scenes.create_view_history().unwrap();
        Self {
            scenes,
            _workers: workers,
            mount,
            scene,
            history,
            frame: 1,
        }
    }

    fn model(&self, name: &str) -> ModelId {
        self.scenes
            .model(self.mount, AssetId::from_name(name))
            .unwrap()
    }

    /// One frame: `write` fills the scene, the focus is at `focus` and the camera at `eye`
    /// looks at `target`.
    fn frame(
        &mut self,
        write: impl FnOnce(&mut SceneWriter<'_>),
        eye: Vec3,
        target: Vec3,
        focus: Vec3,
    ) -> RasterView {
        let list = [self.scene];
        let mut f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let fov_y = 0.9;
        {
            let mut w = f.scene(self.scene).unwrap();
            write(&mut w);
            w.set_focus(focus.as_dvec3(), EXTENT[1] as f32 / fov_y);
        }
        f.update_scenes(&mut |_| {}).unwrap();
        let view = f
            .raster_view(&ViewDesc {
                scene: self.scene,
                history: self.history,
                position: eye.as_dvec3(),
                rotation: look(eye, target),
                projection: Projection { fov_y, near: 0.05 },
                lens: Lens::default(),
                render_extent: UVec2::from(EXTENT),
                jitter: Vec2::ZERO,
                reset_history: self.frame <= 2,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
                target: ViewTarget::AfterRr,
            })
            .unwrap();
        self.frame += 1;
        wait(f.end(&[]).unwrap());
        view
    }

    fn stats(&mut self) -> TessellationStats {
        let f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        self.frame += 1;
        wait(f.end(&[]).unwrap());
        self.scenes.stats().tessellation
    }
}

/// Copies of the closed cages: entry 0 on the limit surface, entry 1 rippled.
const SMOOTH: ProducerId = ProducerId(0);
const RIPPLED: ProducerId = ProducerId(1);
/// m along z from entry 0 to entry 1.
const RIPPLED_Z: f32 = -3.0;

/// m between a generated vertex and the CPU limit surface, at most (1.2e-6 measured): f32
/// evaluation of the patches against the CPU's f64 sums.
const LIMIT_TOLERANCE: f32 = 2e-5;
/// Radians between a generated normal and the CPU limit normal, at most (0.0035 measured,
/// next to the creases): differences of the field and the 16-bit octahedral code.
const LIMIT_NORMAL_TOLERANCE: f32 = 0.006;
/// m between a rippled vertex and the ripple along the limit normal, at most (5.4e-5
/// measured).
const RIPPLE_TOLERANCE: f32 = 2e-4;
/// Radians between the normals of faces meeting at a point beyond which it is on a sharp
/// edge or corner.
const SHARP: f32 = 0.01;
/// m from the prism's semi-sharp corner point within which the surface turns faster than
/// differences resolve, the dice's (0.017 rad) and the CPU's (0.04 rad between the faces
/// at the corner).
const CORNER_REACH: f32 = 0.25;

#[test]
fn subdivision_cages_dice_closed_at_mixed_rates_on_their_limit_surface_in_raster_and_rays() {
    let mut s = CageScene::new(1.5);
    let model = s.model("cages");
    let (info, pages) = cage_entry("cages/subd");
    let limit = LimitSurface::new(&info, &pages, &info.points);
    let desc = |group, index: u32, displacement: u8| SubdDesc {
        model,
        transform: (group, index),
        max_clusters: 16_384,
        max_vertices: 16_384 * 78,
        displacement,
        displacement_reach: if displacement == 0 { 0.0 } else { 0.05 },
        ..SubdDesc::default()
    };
    // The camera looks along the row from beyond the cube, with the focus at the cube:
    // rates fall from the cube to the prism and the rippled copy behind.
    let (eye, target, focus) = (
        Vec3::new(-3.6, 1.4, 1.6),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(-2.0, 0.6, 0.6),
    );
    let mut group = TransformGroupId(0);
    s.frame(
        |w| {
            group = w
                .create_transform_group(&TransformGroupDesc {
                    count: 2,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .unwrap();
            let t = w.transforms(group, 0, 2).unwrap();
            t[0] = Affine3::IDENTITY;
            t[1] = Affine3::IDENTITY;
            t[1].rows[2][3] = RIPPLED_Z;
            w.set_subd(SMOOTH, Some(&desc(group, 0, 0))).unwrap();
            w.set_subd(RIPPLED, Some(&desc(group, 1, RIPPLE))).unwrap();
        },
        eye,
        target,
        focus,
    );
    for _ in 0..2 {
        s.frame(|_| {}, eye, target, focus);
    }
    let view = s.frame(|_| {}, eye, target, focus);
    let smooth = generated(&view, SMOOTH);
    let rippled = generated(&view, RIPPLED);
    let seen = depths(&view);
    let stats = s.stats();
    assert_eq!(stats.entries, 2, "{stats:?}");
    assert_eq!(
        stats.dice_dropped + stats.split_dropped + stats.weld_dropped,
        0,
        "{stats:?}"
    );
    assert!(stats.subpatches > stats.patches, "{stats:?}");
    assert!(stats.welded > 0, "{stats:?}");
    let patterns: HashSet<u32> = smooth.records.iter().map(|r| r.pattern).collect();
    assert!(patterns.len() > 20, "{} patterns", patterns.len());

    // Both copies close: every edge shared by patches, faces, transition edges and creases
    // puts the same points on both sides.
    assert!(assert_closed(&smooth) > 1000);
    assert_closed(&rippled);

    // Every vertex of the smooth copy lies on the CPU limit surface, and its normal is the
    // surface's wherever the faces that meet there nearly agree on one (not on the sharp
    // edge or at the sharp corner) and away from the semi-sharp corner.
    let cage = closed_cages();
    let corner = cage.positions[cage.corners[0].point as usize];
    let (mut worst, mut worst_normal) = (0.0f32, 0.0f32);
    for (p, a) in smooth.positions.iter().zip(&smooth.attrs) {
        let n = limit.nearest(*p);
        worst = worst.max(n.distance);
        if n.spread < SHARP && p.distance(corner) > CORNER_REACH {
            let got = Vec3::from(orr_pack::mesh::oct_decode(a[0]));
            // Faces that meet at the point may disagree by up to their spread.
            let err = got.dot(n.normal).clamp(-1.0, 1.0).acos() - n.spread;
            worst_normal = worst_normal.max(err);
        }
    }
    assert!(worst < LIMIT_TOLERANCE, "{worst} m from the limit surface");
    assert!(
        worst_normal < LIMIT_NORMAL_TOLERANCE,
        "normals within {worst_normal} rad"
    );

    // Each rippled vertex away from the semi-sharp corner lies along the limit normal of a
    // nearby face by the ripple at its limit uv (batch positions are relative to the copy's
    // transform): on the sharp edge and corner the faces displace along different normals
    // and the weld keeps one.
    let (mut worst_ripple, mut moved) = (0.0f32, 0);
    for &x in &rippled.positions {
        if x.distance(corner) < CORNER_REACH {
            continue;
        }
        let (e, h) = limit.displaced_distance(x, ripple);
        worst_ripple = worst_ripple.max(e);
        if h.abs() > 0.02 {
            moved += 1;
        }
    }
    assert!(moved > rippled.positions.len() / 10, "{moved} moved");
    assert!(
        worst_ripple < RIPPLE_TOLERANCE,
        "ripple off by {worst_ripple} m"
    );

    // Raster and camera rays see the same surfaces at the same depths.
    if traced(true) {
        let (mut covered, mut coverage, mut far_apart) = (0, 0, 0);
        for s in &seen {
            if s.raster > 0.0 || s.traced > 0.0 {
                covered += 1;
            }
            if (s.raster > 0.0) != (s.traced > 0.0) {
                coverage += 1;
            } else if s.raster > 0.0 && (s.raster - s.traced).abs() > 1e-3 * s.traced {
                far_apart += 1;
            }
        }
        assert!(
            covered > EXTENT[0] * EXTENT[1] / 8,
            "{covered} pixels covered"
        );
        assert!(
            coverage * 500 <= covered,
            "{coverage} of {covered} disagree on coverage"
        );
        assert!(
            far_apart * 1000 <= covered,
            "{far_apart} of {covered} disagree on depth"
        );
    }
    assert_clean();
}

/// The arm's palette: the root joint fixed, the tip joint bent by `angle` about z at y = 1.
fn arm_palette(angle: f32) -> [Affine3; 2] {
    let r = Mat3::from_rotation_z(angle);
    let pivot = Vec3::new(0.0, 1.0, 0.0);
    let t = pivot - r * pivot;
    let c = r.to_cols_array_2d();
    let mut tip = Affine3::IDENTITY;
    for (i, row) in tip.rows.iter_mut().enumerate() {
        *row = [c[0][i], c[1][i], c[2][i], t[i]];
    }
    [Affine3::IDENTITY, tip]
}

#[test]
fn a_skinned_cage_poses_before_it_dices_and_moves_from_last_frames_pose() {
    let mut s = CageScene::new(2.0);
    let model = s.model("arm/Arm");
    let (info, pages) = cage_entry("arm/Arm/subd");
    assert!(!info.skin.is_empty(), "the arm cooks its skin");
    let (eye, target) = (Vec3::new(1.5, 1.4, 3.0), Vec3::new(0.0, 1.2, 0.0));
    let mut ids = (TransformGroupId(0), TransformGroupId(0));
    let angles = [0.2, 0.5];
    s.frame(
        |w| {
            let make = |w: &mut SceneWriter<'_>, count| {
                w.create_transform_group(&TransformGroupDesc {
                    count,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .unwrap()
            };
            ids = (make(w, 1), make(w, 2));
            w.transforms(ids.0, 0, 1).unwrap()[0] = Affine3::IDENTITY;
            w.transforms(ids.1, 0, 2)
                .unwrap()
                .copy_from_slice(&arm_palette(angles[0]));
            let desc = SubdDesc {
                model,
                transform: (ids.0, 0),
                palette: Some((ids.1, 0)),
                max_clusters: 8192,
                max_vertices: 8192 * 78,
                ..SubdDesc::default()
            };
            w.set_subd(SMOOTH, Some(&desc)).unwrap();
        },
        eye,
        target,
        eye,
    );
    let view = s.frame(
        |w| {
            w.transforms(ids.1, 0, 2)
                .unwrap()
                .copy_from_slice(&arm_palette(angles[1]));
        },
        eye,
        target,
        eye,
    );
    let g = generated(&view, SMOOTH);
    let stats = s.stats();
    assert_eq!(
        stats.dice_dropped + stats.split_dropped + stats.weld_dropped,
        0,
        "{stats:?}"
    );
    assert_closed(&g);
    let posed = |angle: f32| {
        let palette = arm_palette(angle).map(|a| orr_pack::Affine3 { rows: a.rows });
        let points = info.skinned_points(&palette);
        LimitSurface::new(&info, &pages, &points)
    };
    let (now, before) = (posed(angles[1]), posed(angles[0]));
    let mut worst = 0.0f32;
    for (p, q) in g.positions.iter().zip(&g.previous) {
        worst = worst.max(now.nearest(*p).distance);
        worst = worst.max(before.nearest(*q).distance);
    }
    assert!(
        worst < LIMIT_TOLERANCE,
        "{worst} m from the posed limit surfaces"
    );
    // The tip moved between the poses.
    let moved = g
        .positions
        .iter()
        .zip(&g.previous)
        .filter(|(p, q)| p.distance(**q) > 0.1)
        .count();
    assert!(moved > g.positions.len() / 10, "{moved} vertices moved");
    assert_clean();
}

const STRANDS: ProducerId = ProducerId(0);
const STRAND_COUNT: u32 = 128;
const STRAND_SEGMENTS: u32 = 6;
const STRAND_WORD: u32 = 5000;
const STRAND_RADIUS: f32 = 0.02;
const SHADOW_GRID: [u32; 2] = [128, 96];

/// A second device that leaves linear swept spheres off, so strands trace as tubes.
fn tube_device() -> Device {
    let _ = device();
    let d = Device::new(&DeviceDesc {
        name: "orr-test-tubes",
        queues: QueueRequest {
            render_compute: true,
            async_compute: true,
            copy: true,
            present: false,
        },
        skip_extensions: &[c"VK_NV_ray_tracing_linear_swept_spheres"],
        ..DeviceDesc::default()
    })
    .unwrap();
    assert!(!d.caps().lss);
    d
}

/// One scene on `d` holding `strands.slang`'s strand entry, emitted every frame.
struct Strands<'d> {
    d: &'d Device,
    emit: ComputePipeline,
    probe: ComputePipeline,
    scenes: Scenes,
    scene: SceneId,
    history: ViewHistoryId,
    frame: u32,
}

impl<'d> Strands<'d> {
    fn new(d: &'d Device) -> Self {
        let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
        let scene = scenes
            .create_scene(&SceneDesc {
                producer_capacity: 2,
                ..SceneDesc::default()
            })
            .unwrap();
        let history = scenes.create_view_history().unwrap();
        let mut frame = 0;
        let list = [scene];
        let f = begin(&mut scenes, &mut frame, &list);
        let mut m = MaterialGpu::zeroed();
        m.base_color = [0.5; 3];
        m.base_weight = 1.0;
        let material = f.materials().create(&m).unwrap();
        let mut w = f.scene(scene).unwrap();
        let group = w
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        w.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY;
        w.set_producer(
            STRANDS,
            Some(&ProducerDesc {
                max_vertices: STRAND_COUNT * (STRAND_SEGMENTS + 1),
                max_segments: STRAND_COUNT * STRAND_SEGMENTS,
                materials: &[material],
                transform: (group, 0),
                ..ProducerDesc::default()
            }),
        )
        .unwrap();
        drop(w);
        finish(d, f);
        let pipeline = |module, name| {
            let stage = ShaderStage::new(module, "main", SpecData::default()).unwrap();
            d.create_compute_pipeline(&stage, name).unwrap()
        };
        Self {
            d,
            emit: pipeline(&strands::MODULE, "strands"),
            probe: pipeline(&strand_rays::MODULE, "strand rays"),
            scenes,
            scene,
            history,
            frame,
        }
    }

    /// Emits the strands and rasters the camera's view.
    fn frame(&mut self) -> RasterView {
        let p = &self.emit;
        let list = [self.scene];
        let first = self.frame <= 2;
        let scene = self.scene;
        let history = self.history;
        let mut f = begin(&mut self.scenes, &mut self.frame, &list);
        f.update_scenes(&mut |c| {
            let r = c
                .put_root(&strands::Root {
                    scene: c.root,
                    producer: STRANDS.0,
                    count: STRAND_COUNT,
                    segments: STRAND_SEGMENTS,
                    base: STRAND_WORD,
                    radius: STRAND_RADIUS,
                    height: 0.5,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    _pad3: 0,
                })
                .unwrap();
            c.cmd.dispatch(p, r, [STRAND_COUNT.div_ceil(64), 1, 1]);
        })
        .unwrap();
        let eye = Vec3::new(0.0, 0.35, 0.8);
        let view = f
            .raster_view(&ViewDesc {
                scene,
                history,
                position: eye.as_dvec3(),
                rotation: look(eye, Vec3::new(0.0, 0.2, -0.7)),
                projection: Projection {
                    fov_y: 0.9,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: UVec2::from(EXTENT),
                jitter: Vec2::ZERO,
                reset_history: first,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
                target: ViewTarget::AfterRr,
            })
            .unwrap();
        finish(self.d, f);
        view
    }

    /// The producer stats of the last frame, which the next frame's start retires.
    fn stats(&mut self) -> ProducerStats {
        let f = begin(&mut self.scenes, &mut self.frame, &[]);
        finish(self.d, f);
        self.scenes.stats().producers
    }

    /// What `strand_rays` saw in `mode` over `cells` cells.
    fn rays(&self, view: &RasterView, mode: u32, cells: [u32; 2]) -> Vec<strand_rays::Seen> {
        let n = u64::from(cells[0] * cells[1]);
        let seen = self
            .d
            .gpu_malloc::<strand_rays::Seen>(MemoryKind::Readback, n, "seen")
            .unwrap();
        let mut root = self
            .d
            .gpu_malloc::<strand_rays::Root>(MemoryKind::Upload, 1, "root")
            .unwrap();
        let light = Vec3::new(0.4, 0.8, 0.45).normalize();
        root.as_mut_slice().unwrap()[0] = strand_rays::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            seen: seen.gpu(),
            mode,
            entry: STRANDS.0,
            width: cells[0],
            height: cells[1],
            grid_x: -1.6,
            grid_z: 0.0,
            grid_step: 3.2 / cells[0] as f32,
            spread: 0.03,
            light_x: light.x,
            light_y: light.y,
            light_z: light.z,
            _pad: 0,
        };
        let render = self.d.queue(QueueKind::Render).unwrap();
        let mut list = render.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
        list.dispatch(&self.probe, root.gpu(), [n.div_ceil(64) as u32, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        wait_on(self.d, render.submit(vec![list], &[]).unwrap());
        seen.as_slice().unwrap().to_vec()
    }
}

fn begin<'a>(scenes: &'a mut Scenes, frame: &mut u32, list: &'a [SceneId]) -> Frame<'a> {
    let t = time(*frame);
    *frame += 1;
    scenes
        .begin_frame(&FrameDesc {
            time: t,
            scenes: list,
            low_latency: None,
        })
        .unwrap()
}

fn finish(d: &Device, f: Frame<'_>) {
    wait_on(d, f.end(&[]).unwrap());
}

#[test]
fn strand_raster_depth_matches_swept_sphere_and_tube_traces() {
    if !traced(true) {
        return;
    }
    let tubes = tube_device();
    for d in [device(), &tubes] {
        let swept = d.caps().lss;
        let what = if swept { "swept spheres" } else { "tubes" };
        let mut s = Strands::new(d);
        let view = s.frame();
        let seen = s.rays(&view, 0, EXTENT);
        let raster = seen.iter().filter(|o| o.raster_t > 0.0).count();
        let traced = seen.iter().filter(|o| o.traced_t > 0.0).count();
        let both: Vec<_> = seen
            .iter()
            .filter(|o| o.raster_t > 0.0 && o.traced_t > 0.0)
            .collect();
        assert!(raster > 500, "{what}: the raster covers {raster} pixels");
        // Tubes' facets move their silhouettes by up to a tenth of the radius.
        let share = if swept { 0.95 } else { 0.85 };
        assert!(
            both.len() as f32 >= share * raster.max(traced) as f32,
            "{what}: raster {raster}, traced {traced}, both {}",
            both.len()
        );
        assert!(
            both.iter()
                .all(|o| o.valid == 1 && o.swept == u32::from(swept)),
            "{what}: unresolved or misclassified hits"
        );
        let same: Vec<_> = both
            .iter()
            .filter(|o| o.raster_word == o.traced_word)
            .collect();
        // Where strands cross, a tube's facets can put the other strand in front.
        let named = if swept { 0.98 } else { 0.95 };
        assert!(
            same.len() as f32 >= named * both.len() as f32,
            "{what}: {} of {} pixels name the same strand",
            same.len(),
            both.len()
        );
        assert!(
            same.iter()
                .all(|o| { (STRAND_WORD..STRAND_WORD + STRAND_COUNT).contains(&o.raster_word) })
        );
        // Swept spheres meet the raster's resolve exactly. A tube's facets lie within a
        // tenth of the radius of the swept surface, which grows along rays that graze it.
        let slack = if swept { 0.0 } else { 0.12 * STRAND_RADIUS };
        let compared: Vec<_> = same.iter().filter(|o| swept || o.facing > 0.5).collect();
        let off = compared
            .iter()
            .filter(|o| (o.raster_t - o.traced_t).abs() > 1e-3 * o.traced_t + slack)
            .count();
        assert!(
            off * 200 <= compared.len(),
            "{what}: depth off on {off} of {} pixels",
            compared.len()
        );
        let turned = same.iter().filter(|o| o.normal_dot < 0.95).count();
        assert!(
            turned * 50 <= same.len(),
            "{what}: normals disagree on {turned} of {} pixels",
            same.len()
        );
        let stats = s.stats();
        assert_eq!(stats.segments, STRAND_COUNT * STRAND_SEGMENTS, "{what}");
        assert_eq!(stats.dropped_segments, 0, "{what}");
        assert_eq!(stats.refused, 0, "{what}");
        assert_eq!(stats.built, STRAND_COUNT * STRAND_SEGMENTS, "{what}");
    }
    assert_clean();
    assert_clean_on(&tubes);
}

/// The lit fraction of each shadow-grid cell as a gray display-encoded image.
fn shadow_image(seen: &[strand_rays::Seen]) -> Image {
    let pixels = seen
        .iter()
        .flat_map(|o| {
            let g = (orr_color::srgb_encode(f64::from(o.lit)) * 255.0).round() as u8;
            [g, g, g, 255]
        })
        .collect();
    Image {
        width: SHADOW_GRID[0],
        height: SHADOW_GRID[1],
        pixels: Pixels::Rgba8(pixels),
    }
}

#[test]
fn swept_sphere_shadows_match_tube_shadows() {
    if !traced(true) || !gpu_suite::require(device().caps().lss, "linear swept spheres") {
        return;
    }
    let tubes = tube_device();
    let [swept, tube] = [device(), &tubes].map(|d| {
        let mut s = Strands::new(d);
        let view = s.frame();
        s.rays(&view, 1, SHADOW_GRID)
    });
    let shaded = swept.iter().filter(|o| o.lit < 0.5).count();
    assert!(shaded > 300, "{shaded} cells in shadow");
    let e = flip(&shadow_image(&tube), &shadow_image(&swept));
    assert!(e.mean <= 0.05, "FLIP {e:?}");
    assert_clean();
    assert_clean_on(&tubes);
}
