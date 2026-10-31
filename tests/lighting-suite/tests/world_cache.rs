//! The world cache: keys across render origins, probing and eviction against a CPU model,
//! the sample guards, the reset, the global light rescale and the responsive class, then
//! cache-terminated paths on the bed against the reference and through a fall of the light.

use bytemuck::Zeroable;
use lighting_suite::bed::{
    Accumulated, Analytic, Bed, CORNELL, EMITTER, FLOOR, Item, Paths, Spec, WHITE_SPHERE,
};
use lighting_suite::shaders::cache_probe;
use lighting_suite::{TestScene, assert_clean, finish, pipeline, readback, render, upload};
use orr_color::luminance;
use orr_gpu::{ComputePipeline, GpuPtr, Hazards, KernelSet, Stages};
use orr_lighting::{
    DirectionalLight, DiscProfile, Light, LightId, LightingDebugView, LightingStats, PathEntry,
    ReferenceSettings, SceneLighting, SceneLightingSettings, ViewLighting, WorldCacheSettings,
};
use orr_math::{Mat3, Vec3};
use orr_scene::{SceneDesc, SceneWriter};

/// `world_cache`'s `CACHE_PROBE` and `CACHE_STORE_SCALE`.
const PROBE: usize = 8;
const STORE_SCALE: f32 = 1.0 / 256.0;

fn rgb9e5(p: u32) -> [f32; 3] {
    let scale = 2f32.powi((p >> 27) as i32 - 24);
    [p, p >> 9, p >> 18].map(|c| (c & 0x1FF) as f32 * scale / STORE_SCALE)
}

/// A slot as the resolve left it.
#[derive(Clone, Copy, Debug)]
struct Cell {
    check: u32,
    radiance: [f32; 3],
    life: u32,
    history: u32,
}

impl Cell {
    fn new(c: [u32; 4]) -> Self {
        Self {
            check: c[0],
            radiance: rgb9e5(c[1]),
            life: c[2] & 0xFF,
            history: (c[2] >> 8) & 0xFF,
        }
    }
}

/// The key of render-space `p` under the render origin at `origin` (cells of 2^`cell_log2`
/// m), from the world position in f64: `world_cache`'s packing.
fn expected_key(world: [f64; 3], k: i32, level: u32, bucket: u64) -> u64 {
    let mask = (1u64 << 19) - 1;
    let c = world.map(|w| ((w / 2f64.powi(k)).floor() as i64 as u64) & mask);
    c[0] | (c[1] << 19) | (c[2] << 38) | (u64::from(level) << 57) | (bucket << 61)
}

/// One scene with lighting whose cache is driven sample by sample.
struct Cache {
    scene: TestScene,
    lighting: SceneLighting,
    keys: ComputePipeline,
    add: ComputePipeline,
    cells: ComputePipeline,
    capacity: u32,
}

impl Cache {
    fn new(world_cache: WorldCacheSettings) -> Self {
        let mut scene = TestScene::new(SceneDesc {
            light_capacity: 4,
            ..SceneDesc::default()
        });
        scene.write(|_, _| {});
        let mut lighting = scene.lighting(&KernelSet::default());
        lighting
            .set_settings(&SceneLightingSettings {
                world_cache,
                ..SceneLightingSettings::default()
            })
            .unwrap();
        Self {
            scene,
            lighting,
            keys: pipeline(&cache_probe::MODULE, "keys"),
            add: pipeline(&cache_probe::MODULE, "add"),
            cells: pipeline(&cache_probe::MODULE, "cells"),
            capacity: 1 << world_cache.capacity_log2,
        }
    }

    fn root(&self, cache: GpuPtr<orr_lighting::WorldCacheGpu>) -> cache_probe::Root {
        cache_probe::Root {
            cache,
            cases: GpuPtr::NULL,
            keys: GpuPtr::NULL,
            samples: GpuPtr::NULL,
            cells: GpuPtr::NULL,
            count: 0,
            capacity: self.capacity,
        }
    }

    fn keys(&self, cases: &[cache_probe::KeyCase]) -> Vec<cache_probe::KeyOut> {
        let input = upload(cases, "key cases");
        let out = readback::<cache_probe::KeyOut>(cases.len() as u64, "keys");
        let root = upload(
            &[cache_probe::Root {
                cases: input.gpu,
                keys: out.gpu,
                count: cases.len() as u32,
                ..self.root(GpuPtr::NULL)
            }],
            "key root",
        );
        let mut list = render().command_list().unwrap();
        list.dispatch(
            &self.keys,
            root.gpu,
            [cases.len().div_ceil(64) as u32, 1, 1],
        );
        finish(list);
        out.as_slice().unwrap().to_vec()
    }

    /// Runs one frame: `write` changes the scene, the scene pass resolves the samples of the
    /// frame before, then `samples` are added in order. Returns the cells after the resolve.
    fn frame(
        &mut self,
        write: impl FnOnce(&mut SceneWriter<'_>),
        samples: &[(u64, [f32; 3])],
    ) -> Vec<Cell> {
        self.scene.write(|w, _| write(w));
        let samples: Vec<_> = samples
            .iter()
            .map(|&(key, v)| cache_probe::CacheSample {
                key: [key as u32, (key >> 32) as u32],
                _pad0: 0,
                _pad1: 0,
                value: [v[0], v[1], v[2], 0.0],
            })
            .collect();
        let zero = [cache_probe::CacheSample::zeroed()];
        let input = upload(
            if samples.is_empty() {
                &zero
            } else {
                &samples[..]
            },
            "cache samples",
        );
        let cells = readback::<[u32; 4]>(u64::from(self.capacity), "cells");
        let frame = self.scene.frame;
        let root = upload(
            &[cache_probe::Root {
                samples: input.gpu,
                cells: cells.gpu,
                count: samples.len() as u32,
                ..self.root(self.lighting.world_cache(frame))
            }],
            "cache root",
        );
        let mut list = render().command_list().unwrap();
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.lighting.record(&mut list, &self.scene.pass([0.0; 3]));
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.cells, root.gpu, [self.capacity.div_ceil(64), 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        if !samples.is_empty() {
            list.dispatch(&self.add, root.gpu, [1, 1, 1]);
        }
        finish(list);
        cells
            .as_slice()
            .unwrap()
            .iter()
            .map(|&c| Cell::new(c))
            .collect()
    }

    fn stats(&self) -> LightingStats {
        self.lighting.stats(self.scene.frame).unwrap()
    }
}

fn settings(capacity_log2: u32, cell_lifetime: u32) -> WorldCacheSettings {
    WorldCacheSettings {
        capacity_log2,
        cell_lifetime,
        ..WorldCacheSettings::default()
    }
}

/// A cell's key at a distinct point per index, base level, facing up.
fn spread_keys(cache: &Cache, count: u32) -> Vec<cache_probe::KeyOut> {
    let cases: Vec<_> = (0..count)
        .map(|i| cache_probe::KeyCase {
            origin_cell: [0, 0, 0, 8],
            position: [i as f32 * 0.5 + 0.1, 0.1, (i % 7) as f32 * 0.5 + 0.1],
            k: -3,
            normal: [0.0, 1.0, 0.0],
            level: 0,
        })
        .collect();
    cache.keys(&cases)
}

fn key64(k: &cache_probe::KeyOut) -> u64 {
    u64::from(k.key[0]) | (u64::from(k.key[1]) << 32)
}

#[test]
fn keys_hold_across_render_origins() {
    let cache = Cache::new(settings(4, 10));
    let origins = [[0, 0, 0], [2, -4, 6], [-118, 40, 2], [64, 116, -90]];
    let mut seed = 0x2545_f491u32;
    let mut next = |range: i64| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        i64::from(seed) % (2 * range) - range
    };
    let normals = [
        ([0.0f32, 1.0, 0.0], 3u64),
        ([0.0, -1.0, 0.0], 4),
        ([1.0, 0.0, 0.0], 1),
        ([0.0, 0.3, -0.9], 6),
    ];
    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for k in -3..=8 {
        let edge = 2f64.powi(k);
        for i in 0..96 {
            // Dyadic world points within 30 km: on a voxel boundary, a 64th of a metre below
            // one, and anywhere.
            let base =
                [next(30_000), next(500), next(30_000)].map(|v| (v as f64 / edge).floor() * edge);
            let world = match i % 3 {
                0 => base,
                1 => base.map(|v| v - 1.0 / 64.0),
                _ => base.map(|v| v + (next(1 << 20).unsigned_abs() as f64 / 64.0) % edge),
            };
            let (normal, bucket) = normals[i % normals.len()];
            for o in origins {
                let p = [0, 1, 2].map(|a| world[a] - f64::from(o[a]) * 256.0);
                if p.iter().any(|v| f64::from(*v as f32) != *v) {
                    continue;
                }
                cases.push(cache_probe::KeyCase {
                    origin_cell: [o[0], o[1], o[2], 8],
                    position: p.map(|v| v as f32),
                    k,
                    normal,
                    level: (k + 3) as u32,
                });
                expected.push(expected_key(world, k, (k + 3) as u32, bucket));
            }
        }
    }
    assert!(cases.len() > 3000, "{} cases", cases.len());
    for ((c, k), e) in cases.iter().zip(cache.keys(&cases)).zip(expected) {
        assert_eq!(key64(&k), e, "{c:?}");
    }
    assert_clean();
}

/// The cache's slots as the CPU expects them: keys by index into `keys`.
struct Model {
    slots: Vec<Option<(usize, u32)>>,
    lifetime: u32,
}

impl Model {
    /// Adds one sample of key `i` the way `cache_insert` claims slots; false on overflow.
    fn add(&mut self, keys: &[cache_probe::KeyOut], i: usize, seen: &mut Vec<usize>) -> bool {
        let n = self.slots.len();
        let home = keys[i].hash as usize & (n - 1);
        let window = (0..PROBE).map(|j| (home + j) & (n - 1));
        let slot = window
            .clone()
            .find(|&s| self.slots[s].is_some_and(|(k, _)| keys[k].check == keys[i].check))
            .or_else(|| window.clone().find(|&s| self.slots[s].is_none()));
        if let Some(s) = slot {
            self.slots[s].get_or_insert((i, 0));
            seen.push(s);
        }
        slot.is_some()
    }

    fn resolve(&mut self, seen: &[usize]) {
        for (s, slot) in self.slots.iter_mut().enumerate() {
            if let Some((k, life)) = *slot {
                let life = if seen.contains(&s) { 0 } else { life + 1 };
                *slot = (life <= self.lifetime).then_some((k, life));
            }
        }
    }
}

#[test]
fn probing_and_eviction_match_a_cpu_model() {
    let lifetime = 2;
    let mut cache = Cache::new(settings(4, lifetime));
    let keys = spread_keys(&cache, 40);
    let mut checks: Vec<_> = keys.iter().map(|k| k.check).collect();
    checks.sort_unstable();
    checks.dedup();
    assert_eq!(checks.len(), keys.len(), "check collision in the test keys");
    let value = |i: usize| [1.0 + i as f32, 2.0, 0.5];
    let mut model = Model {
        slots: vec![None; 16],
        lifetime,
    };
    let mut seen = Vec::new();
    let mut overflows = 0;
    let mut added = 0;
    let (mut overflowed, mut evicted) = (0, 0);
    for frame in 0..10 {
        let batch: Vec<usize> = match frame {
            0 => Vec::new(),
            1 => (0..25).collect(),
            5 => (0..5).chain(25..40).collect(),
            _ => (0..5).collect(),
        };
        let cells = cache.frame(
            |_| {},
            &batch
                .iter()
                .map(|&i| (key64(&keys[i]), value(i)))
                .collect::<Vec<_>>(),
        );
        model.resolve(&seen);
        if frame > 0 {
            let s = cache.stats().world_cache;
            assert_eq!(
                (s.updates, s.overflows),
                (added, overflows),
                "frame {frame}"
            );
            let occupied = model.slots.iter().filter(|s| s.is_some()).count() as u32;
            assert_eq!(s.occupied, occupied, "frame {frame}");
            overflowed += s.overflows;
            evicted += s.evicted;
        }
        for (s, (cell, want)) in cells.iter().zip(&model.slots).enumerate() {
            match want {
                None => assert_eq!(cell.check, 0, "frame {frame} slot {s}: {cell:?}"),
                Some((k, life)) => {
                    assert_eq!(cell.check, keys[*k].check, "frame {frame} slot {s}");
                    assert_eq!(cell.life, *life, "frame {frame} slot {s}");
                    for c in 0..3 {
                        let v = value(*k)[c];
                        assert!(
                            (cell.radiance[c] - v).abs() <= v * 2e-3 + 0.01,
                            "frame {frame} slot {s}: {cell:?}, want {:?}",
                            value(*k)
                        );
                    }
                }
            }
        }
        seen.clear();
        overflows = 0;
        added = batch.len() as u32;
        for &i in &batch {
            if !model.add(&keys, i, &mut seen) {
                overflows += 1;
            }
        }
    }
    assert!(
        overflowed > 0 && evicted > 0,
        "{overflowed} overflows, {evicted} evictions"
    );
    assert_clean();
}

#[test]
fn samples_that_are_not_finite_are_dropped() {
    let mut cache = Cache::new(settings(6, 10));
    let key = key64(&spread_keys(&cache, 1)[0]);
    let samples = [
        (key, [f32::NAN, 1.0, 1.0]),
        (key, [f32::INFINITY, 1.0, 1.0]),
        (key, [-1.0, 2.0, 3.0]),
        (key, [1.0, 2.0, 3.0]),
    ];
    cache.frame(|_| {}, &samples);
    let cells = cache.frame(|_| {}, &[]);
    let cell = cells.iter().find(|c| c.check != 0).unwrap();
    assert_eq!(cell.history, 2, "{cell:?}");
    for (got, want) in cell.radiance.iter().zip([0.5, 2.0, 3.0]) {
        assert!((got - want).abs() < 0.01, "{cell:?}");
    }
    assert_eq!(cache.stats().world_cache.updates, 2);
    assert_clean();
}

#[test]
fn a_lighting_reset_clears_the_cache() {
    let mut cache = Cache::new(settings(6, 10));
    let keys = spread_keys(&cache, 20);
    let samples: Vec<_> = keys.iter().map(|k| (key64(k), [1.0; 3])).collect();
    cache.frame(|_| {}, &samples);
    let cells = cache.frame(|_| {}, &[]);
    assert!(cells.iter().filter(|c| c.check != 0).count() >= 19);
    cache.lighting.reset_history();
    let cells = cache.frame(|_| {}, &[]);
    assert!(cells.iter().all(|c| c.check == 0 && c.history == 0));
    assert_eq!(cache.stats().world_cache.occupied, 0);
    assert_clean();
}

fn sun(illuminance: f32) -> Light {
    Light::Directional(DirectionalLight {
        direction_to_light: Vec3::Y,
        illuminance: Vec3::splat(illuminance),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    })
}

#[test]
fn cells_scale_down_with_the_global_light_and_not_up() {
    let mut cache = Cache::new(settings(6, 30));
    let key = key64(&spread_keys(&cache, 1)[0]);
    let set =
        |l: f32| move |w: &mut SceneWriter<'_>| w.set_light(LightId(0), Some(&sun(l))).unwrap();
    cache.frame(set(100_000.0), &[(key, [8.0, 4.0, 2.0])]);
    let radiance = |cells: &[Cell]| cells.iter().find(|c| c.check != 0).unwrap().radiance;
    let before = radiance(&cache.frame(set(100_000.0), &[]));
    assert_eq!(cache.stats().world_cache.rescale, 1.0);
    let fallen = radiance(&cache.frame(set(50_000.0), &[]));
    let s = cache.stats().world_cache;
    assert!((s.rescale - 0.5).abs() < 1e-6, "{s:?}");
    assert!((s.light_estimate - 50_000.0).abs() < 1.0, "{s:?}");
    let risen = radiance(&cache.frame(set(100_000.0), &[]));
    assert_eq!(cache.stats().world_cache.rescale, 1.0);
    for c in 0..3 {
        assert!(
            (fallen[c] - before[c] * 0.5).abs() <= before[c] * 4e-3,
            "{before:?} {fallen:?}"
        );
        assert_eq!(risen[c], fallen[c]);
    }
    assert_clean();
}

#[test]
fn a_cell_whose_light_doubles_catches_up_within_a_second() {
    let max_temporal_samples = 32.0;
    let mut cache = Cache::new(WorldCacheSettings {
        max_temporal_samples,
        ..settings(6, 10)
    });
    let key = key64(&spread_keys(&cache, 1)[0]);
    let mut cell = None;
    for _ in 0..40 {
        cell = cache
            .frame(|_| {}, &[(key, [1.0; 3])])
            .into_iter()
            .find(|c| c.check != 0);
    }
    let steady = cell.unwrap();
    assert_eq!(steady.history, max_temporal_samples as u32, "{steady:?}");
    for _ in 0..60 {
        cell = cache
            .frame(|_| {}, &[(key, [2.0; 3])])
            .into_iter()
            .find(|c| c.check != 0);
    }
    let r = cell.unwrap().radiance;
    assert!(r[1] > 1.9, "{r:?} after 60 frames of twice the light");
    assert_clean();
}

fn realtime() -> ViewLighting {
    ViewLighting::default()
}

fn reference(max_bounces: u32) -> ViewLighting {
    ViewLighting {
        max_bounces,
        reference: Some(ReferenceSettings::default()),
        ..ViewLighting::default()
    }
}

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), PathEntry::Fastest, settings)
}

fn render_paths(bed: &mut Bed, paths: &Paths, first: u32, frames: u32) -> Accumulated {
    bed.accumulate_from(first, frames, 1, &mut |list, f| paths.record(list, f))
}

/// Mean RGB over surface pixels of each `tile` x `tile` block, and the block's surface
/// pixel count.
fn tile_means(a: &Accumulated, tile: u32) -> Vec<([f64; 3], u32)> {
    let [w, h] = lighting_suite::bed::EXTENT;
    let mut out = Vec::new();
    for ty in 0..h / tile {
        for tx in 0..w / tile {
            let mut sum = [0.0; 3];
            let mut n = 0;
            for y in ty * tile..(ty + 1) * tile {
                for x in tx * tile..(tx + 1) * tile {
                    let p = a.mean[(y * w + x) as usize];
                    if p[3] > 0.5 {
                        for c in 0..3 {
                            sum[c] += f64::from(p[c]);
                        }
                        n += 1;
                    }
                }
            }
            out.push((sum.map(|s| s / f64::from(n.max(1))), n));
        }
    }
    out
}

fn surface_mean(a: &Accumulated) -> [f64; 3] {
    let tiles = tile_means(a, 1);
    let mut sum = [0.0; 3];
    let mut n = 0.0;
    for (m, count) in tiles.iter().filter(|(_, c)| *c > 0) {
        for c in 0..3 {
            sum[c] += m[c] * f64::from(*count);
        }
        n += f64::from(*count);
    }
    sum.map(|s| s / n)
}

/// The realtime path, ending in the cache, against the reference on diffuse scenes: the
/// whole image and each 32 x 32 block within 5%, once the cache is warm.
#[test]
fn cache_terminated_paths_match_the_reference_in_diffuse_regions() {
    for analytic in [Analytic::CornellBox, Analytic::SunAndGround] {
        let mut bed = Bed::new(analytic);
        let warm = 256;
        let frames = 512;
        let realtime_paths = paths(realtime());
        render_paths(&mut bed, &realtime_paths, 0, warm);
        let cached = render_paths(&mut bed, &realtime_paths, warm, frames);
        let s = bed.lighting.stats(bed.last_frame()).unwrap().world_cache;
        assert!(
            s.occupied > 0 && s.hits > 0 && s.updates > 0,
            "{analytic:?}: {s:?}"
        );
        assert!(
            f64::from(s.overflows) < 0.01 * f64::from(s.updates),
            "{analytic:?}: {s:?}"
        );
        let truth = render_paths(&mut bed, &paths(reference(12)), 0, frames);
        let (a, b) = (
            luminance(surface_mean(&cached)),
            luminance(surface_mean(&truth)),
        );
        assert!(
            (a / b - 1.0).abs() < 0.05,
            "{analytic:?}: image {a} against {b}"
        );
        let ratios: Vec<f64> = tile_means(&cached, 32)
            .iter()
            .zip(tile_means(&truth, 32))
            .filter(|((_, n), _)| *n >= 512)
            .map(|((c, _), (t, _))| luminance(*c) / luminance(t))
            .collect();
        assert!(
            ratios.iter().all(|r| (r - 1.0).abs() < 0.05),
            "{analytic:?}: blocks against the reference {ratios:.3?}"
        );
    }
    assert_clean();
}

/// The cache's radiance at every first vertex, one frame.
fn cache_view(bed: &mut Bed, view: &Paths, index: u32) -> [f64; 3] {
    surface_mean(&render_paths(bed, view, index, 1))
}

fn directional(direction: Vec3, illuminance: f32) -> Light {
    Light::Directional(DirectionalLight {
        direction_to_light: direction.normalize(),
        illuminance: Vec3::splat(illuminance),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    })
}

const SUN_DIRECTION: Vec3 = Vec3::new(0.3, 0.8, -0.5);
const MOON_DIRECTION: Vec3 = Vec3::new(-0.6, 0.5, 0.4);
const MOON: f32 = 0.3;

/// A sun of 100 klx setting over one second at 60 frames a second, to 0.1 lx and then out,
/// while a 0.3 lx moon rises; `each` sees every frame's index after the light changed.
fn sunset(bed: &mut Bed, run: &Paths, start: u32, each: &mut dyn FnMut(&mut Bed, u32)) -> u32 {
    let steps = 60;
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        let sun = 100_000.0 * 1e-6f32.powf(t);
        bed.set_light(
            LightId(0),
            (i < steps).then(|| directional(SUN_DIRECTION, sun)),
        );
        bed.set_light(LightId(1), Some(directional(MOON_DIRECTION, MOON * t)));
        render_paths(bed, run, start + i, 1);
        each(bed, start + i);
    }
    start + steps
}

fn day_spec(copies: Vec<Item>, eye: Vec3, target: Vec3) -> Spec {
    Spec {
        copies,
        lights: vec![
            directional(SUN_DIRECTION, 100_000.0),
            directional(MOON_DIRECTION, 0.0),
        ],
        environment: None,
        eye,
        target,
        fov_y: 0.8,
        focus: None,
    }
}

fn at(model: &'static str, p: Vec3) -> Item {
    Item {
        model,
        linear: Mat3::IDENTITY,
        at: p,
    }
}

/// Two seconds after a sunset, the cache holds the moonlit scene, not daylight.
#[test]
fn a_sunset_leaves_no_daylight_in_the_cache() {
    let copies = vec![
        at(FLOOR, Vec3::ZERO),
        at(WHITE_SPHERE, Vec3::new(0.0, 1.0, 0.0)),
    ];
    let eye = Vec3::new(0.0, 2.0, 5.0);
    let target = Vec3::new(0.0, 0.7, 0.0);
    let run = paths(realtime());
    let view = paths(ViewLighting {
        debug_view: LightingDebugView::WorldCache,
        ..realtime()
    });

    let mut night = Bed::with(Spec {
        lights: vec![directional(MOON_DIRECTION, MOON)],
        ..day_spec(copies.clone(), eye, target)
    });
    render_paths(&mut night, &run, 0, 180);
    let moonlit = luminance(cache_view(&mut night, &view, 180));

    let mut bed = Bed::with(day_spec(copies, eye, target));
    render_paths(&mut bed, &run, 0, 120);
    let daylit = luminance(cache_view(&mut bed, &view, 120));
    let end = sunset(&mut bed, &run, 120, &mut |_, _| {});
    render_paths(&mut bed, &run, end + 1, 120);
    let after = luminance(cache_view(&mut bed, &view, end + 121));
    assert!(daylit > 1e4 * moonlit, "day {daylit}, night {moonlit}");
    assert!(
        (after / moonlit - 1.0).abs() < 0.1,
        "two seconds after sunset {after}, moonlit {moonlit}, daylit {daylit}"
    );
    assert_clean();
}

/// A closed room lit only by its emitter, its opening turned away from the sun: the sunset
/// dims its cells, which return within 5% of a room whose sun stayed up within one second.
#[test]
fn a_room_lit_by_its_own_light_recovers_after_a_sunset() {
    let run = paths(realtime());
    let view = paths(ViewLighting {
        debug_view: LightingDebugView::WorldCache,
        ..realtime()
    });
    let room = |set: bool| {
        let copies = vec![
            at(CORNELL, Vec3::ZERO),
            at(EMITTER, Vec3::new(0.0, 1.98, 0.0)),
        ];
        let mut bed = Bed::with(day_spec(
            copies,
            Vec3::new(0.0, 1.0, 0.9),
            Vec3::new(0.0, 1.0, -1.0),
        ));
        render_paths(&mut bed, &run, 0, 240);
        let mut lowest = f64::INFINITY;
        let mut each = |bed: &mut Bed, i| lowest = lowest.min(luminance(cache_view(bed, &view, i)));
        let end = if set {
            sunset(&mut bed, &run, 240, &mut each)
        } else {
            for i in 1..=60 {
                render_paths(&mut bed, &run, 240 + i, 1);
                each(&mut bed, 240 + i);
            }
            300
        };
        render_paths(&mut bed, &run, end + 1, 60);
        (lowest, luminance(cache_view(&mut bed, &view, end + 61)))
    };
    let (_, held) = room(false);
    let (lowest, after) = room(true);
    assert!(
        lowest < 0.9 * held,
        "the sunset never dimmed the room: {lowest} against {held}"
    );
    assert!(
        (after / held - 1.0).abs() < 0.05,
        "one second after sunset {after}, with the sun held {held}"
    );
    assert_clean();
}

#[test]
fn cache_terminated_paths_stay_finite_on_every_analytic_scene() {
    let run = paths(realtime());
    let lod = paths(ViewLighting {
        debug_view: LightingDebugView::WorldCacheLod,
        ..realtime()
    });
    for analytic in Analytic::ALL {
        let mut bed = Bed::new(analytic);
        render_paths(&mut bed, &run, 0, 8);
        let image = render_paths(&mut bed, &run, 8, 1);
        assert!(
            image.mean.iter().flatten().all(|v| v.is_finite()),
            "{analytic:?}"
        );
        let levels = render_paths(&mut bed, &lod, 9, 1);
        let found = levels
            .mean
            .iter()
            .filter(|p| p[3] > 0.5 && p[1] > 0.5)
            .count();
        let surface = levels.mean.iter().filter(|p| p[3] > 0.5).count();
        assert!(
            found * 2 > surface,
            "{analytic:?}: {found} of {surface} first vertices in a resolved cell"
        );
    }
    assert_clean();
}
