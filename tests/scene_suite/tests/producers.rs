//! Producer entries in raster and ray tracing: a grass field whose cluster count changes
//! every frame against a CPU caster of the same blades, the producer words a linked surface
//! reads at primary and secondary hits, capacity and drops in `FrameStats`, and particle
//! entries, drawn only at the transparency point and traced under the particle mask.

use glam::{DVec3, Mat3, Quat, UVec2, Vec2, Vec3};
use orr_gpu::{
    Blend, ClearColor, ColorAttachment, ColorLoad, ColorMask, ColorTarget, Format, Hazards,
    MemoryKind, MeshPipelineDesc, QueueKind, RasterState, Rendering, ShaderStage, SpecData, Stages,
    StorageImageIndex, Texture, TextureDesc, TextureMemory, TextureRegion, TextureUsage,
};
use orr_scene::testing::{assert_clean, device, read_buffer, read_texture, wait};
use orr_scene::{
    Affine3, ClusterTopology, FrameDesc, FrameTime, GroupWriter, Lens, MaterialGpu, MaterialId,
    Opacity, ProducerDesc, ProducerId, ProducerStats, ProducerUpdate, Projection, RasterView,
    RendererDesc, SceneDesc, SceneId, Scenes, TextureFilter, TransformGroupDesc, TransformGroupId,
    TransparencyPass, ViewDesc, ViewHistoryId, ViewTarget, VisibleCluster, material_flags,
};
use orr_shaders::bytemuck::Zeroable;
use scene_suite::shaders::{blades, particle_draw, producer_rays};
use scene_suite::{linked, pipeline};

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
        let triangles = self.triangles.gpu;
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
        seen: seen.gpu,
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
    list.dispatch(&p, root.gpu, [n.div_ceil(64) as u32, 1, 1]);
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
