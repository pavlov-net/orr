//! A view's raster half on cooked meshes and the scene's ray tracing beside it: G-buffer
//! packing against a CPU decoder, reference images of each G-buffer channel, the resolve as
//! device-generated commands against its recorded dispatches, motion vectors that reproject
//! the previous frame, discrete levels chosen by distance, a linked surface whose coverage
//! and depth offset give the depth and motion of the shape it stands for,
//! the software raster against the mesh-shader raster, camera rays through the PTLAS against
//! the visibility buffer, resolved hits against a CPU decode, hit contexts, `rt_range`,
//! far-from-origin and rebase invariance, the PTLAS budget, posed rigid parts in raster
//! and ray tracing with their motion, hiding, section masks, material sets and color factors
//! multiplied in their texture's primaries, the DAG traversal's cuts, two-pass occlusion and
//! `debug_lod`, the stochastic texture filter against CPU bicubic and bilinear filters, a
//! fractured model's pieces, partial edits, posed and skinned edits and the motion of vertices
//! kernels move, cooked instance pages against the same copies
//! placed at run time, and a streaming mesh's finer cached cuts.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use orr_color::RgbPrimaries;
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, InstanceMotion, Instances, Linear, LodMode, Manifest,
    MeshInput, MeshParts, MeshSettings, Placement, Primaries, PrototypeSpec, SectionSpec, Source,
    TextureSettings, Usage, UsdSettings,
};
use orr_gpu::{
    ComputePipeline, GpuPtr, Hazards, KernelSet, MemoryKind, QueueKind, ShaderModule, Stages,
    Texture, TextureRegion,
};
use orr_math::{Affine3, DMat4, DQuat, DVec2, DVec3, DVec4, Mat3, Quat, UVec2, Vec2, Vec3, Vec4};
use orr_pack::clip::Clip;
use orr_pack::instance::{InstanceRecord, MOTION_WINDOW_BYTES, MotionRow};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{Cluster, Group, MeshInfo, PageView, decode_page};
use orr_pack::model::ModelInfo;
use orr_pack::skeleton::{SkeletonInfo, Trs};
use orr_pack::{Archive, AssetId, Entry, Span, Target};
use orr_scene::testing::{
    Image, Pixels, Tolerance, assert_clean, check, device, flip, read_buffer, read_texture,
    visible_clusters, wait,
};
use orr_scene::{
    AffineRows, BoundTexture, ClusterWork, CopyVariant, CutGpu, DebugLod, Decal, DecalChannels,
    DecalGpu, DecalId, EditDesc, EditId, EditRef, FrameDesc, FrameTime, GroupWriter, InstanceFlags,
    InstanceRange, InstancesDesc, Lens, LoadState, MaterialGpu, MaterialId, MeshGpu, ModelGpu,
    ModelId, MountId, PackSource, Pose, Projection, RasterGpu, RasterView, RendererDesc,
    SceneCompute, SceneDesc, SceneId, SceneRoot, SceneWriter, Scenes, SkinStateGpu, SkinnedDesc,
    SkinnedGpu, SkinnedPageRef, SlotCutGpu, SlotStateGpu, TextureFilter, TransformGroupDesc,
    TransformGroupId, ViewDesc, ViewHistoryId,
};
use orr_scene::{Budgets, InstancePageRef, PageId, PixelProbe, PoolSize, RequestSource};
use scene_suite::shaders::{
    codec, fracture, grid, pose, skin_pose, surface_at, surface_params, trace, triangle_at,
    triangle_point,
};
use scene_suite::{Workers, linked, pipeline};

mod paged;

const EXTENT: [u32; 2] = [160, 96];

fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn reference(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("references/gbuffer")
        .join(name)
}

// --- CPU decoder of the G-buffer words -------------------------------------------------------

fn srgb_decode(v: f32) -> f32 {
    orr_color::srgb_decode(f64::from(v)) as f32
}

fn srgb_encode(v: f32) -> f32 {
    orr_color::srgb_encode(f64::from(v)) as f32
}

fn byte(w: u32, at: u32) -> f32 {
    ((w >> at) & 255) as f32 / 255.0
}

fn rgb8(w: u32) -> Vec3 {
    Vec3::new(byte(w, 0), byte(w, 8), byte(w, 16)).map(srgb_decode)
}

fn oct_unwrap(x: f32, y: f32) -> Vec3 {
    let z = 1.0 - x.abs() - y.abs();
    let (x, y) = if z < 0.0 {
        (
            (1.0 - y.abs()) * if x >= 0.0 { 1.0 } else { -1.0 },
            (1.0 - x.abs()) * if y >= 0.0 { 1.0 } else { -1.0 },
        )
    } else {
        (x, y)
    };
    Vec3::new(x, y, z).normalize()
}

fn oct16(w: u32) -> Vec3 {
    let x = f32::from(w as u16 as i16) / 32767.0;
    let y = f32::from((w >> 16) as u16 as i16) / 32767.0;
    oct_unwrap(x.max(-1.0), y.max(-1.0))
}

fn oct8(w: u32) -> Vec3 {
    let x = f32::from(w as u8 as i8) / 127.0;
    let y = f32::from((w >> 8) as u8 as i8) / 127.0;
    oct_unwrap(x.max(-1.0), y.max(-1.0))
}

fn rgb9e5(w: u32) -> Vec3 {
    let scale = 2f32.powi((w >> 27) as i32 - 24);
    Vec3::new(
        (w & 511) as f32,
        ((w >> 9) & 511) as f32,
        ((w >> 18) & 511) as f32,
    ) * scale
}

fn ior(code: u32) -> f32 {
    let c = (code & 255) as f32;
    let t = match c {
        ..16.0 => c / 16.0 - 1.0,
        ..201.0 => (c - 16.0) / 370.0,
        _ => 0.5 + (c - 201.0) / 108.0,
    };
    (1.0 + t) / (1.0 - t).max(1e-4)
}

/// The largest error of `ior`'s code around `t = (n - 1) / (n + 1)`.
fn ior_step(t: f32) -> f32 {
    0.5 * match t {
        ..0.0 => 1.0 / 16.0,
        ..0.5 => 1.0 / 370.0,
        _ => 1.0 / 108.0,
    } + 1e-6
}

/// The core words as the G-buffer layout defines them: what `decode_gsurface` must return.
#[derive(Debug)]
struct Core {
    base_color: Vec3,
    roughness: f32,
    normal: Vec3,
    emission: Vec3,
    metalness: f32,
    specular_weight: f32,
    ior: f32,
    flags: u32,
    specular_color: Vec3,
    raster_error: f32,
    geometric_normal: Vec3,
    diffuse_roughness: f32,
}

fn decode_core(a: [u32; 4], b: [u32; 2]) -> Core {
    let e = (b[0] >> 21) & 31;
    Core {
        base_color: rgb8(a[0]),
        roughness: byte(a[0], 24),
        normal: oct16(a[1]),
        emission: rgb9e5(a[2]),
        metalness: byte(a[3], 0),
        specular_weight: byte(a[3], 8) * (1u32 << ((b[0] >> 26) & 31)) as f32,
        ior: ior(a[3] >> 16),
        flags: a[3] >> 24,
        specular_color: Vec3::new(
            (b[0] & 127) as f32,
            ((b[0] >> 7) & 127) as f32,
            ((b[0] >> 14) & 127) as f32,
        )
        .map(|v| srgb_decode(v / 127.0)),
        raster_error: if e == 0 {
            0.0
        } else {
            2f32.powi(e as i32 - 21)
        },
        geometric_normal: oct8(b[1]),
        diffuse_roughness: byte(b[1], 16),
    }
}

const THIN_TWO_SIDED: u32 = 1;
const EXTENDED: u32 = 4;
const SUBSURFACE: u32 = 8;
const LAYERS: u32 = 16;
const BSDF_THIN_WALLED: u32 = 1;
const BSDF_COAT_FRAME: u32 = 64;
const BSDF_FRESNEL_MIX: u32 = 128;
const BSDF_METAL_FLAT: u32 = 256;

// --- Packing ------------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn rgb(&mut self) -> Vec3 {
        Vec3::new(self.next(), self.next(), self.next())
    }

    fn unit(&mut self) -> Vec3 {
        loop {
            let v = self.rgb() * 2.0 - 1.0;
            if (0.01..1.0).contains(&v.length_squared()) {
                return v.normalize();
            }
        }
    }
}

fn arr(v: Vec3) -> [f32; 3] {
    v.to_array()
}

fn angle(a: [f32; 3], b: [f32; 3]) -> f64 {
    let (a, b) = (Vec3::from(a).as_dvec3(), Vec3::from(b).as_dvec3());
    a.cross(b).length().atan2(a.dot(b))
}

#[test]
fn gbuffer_words_decode_on_the_cpu_and_round_trip_within_their_precision() {
    let _serial = serial();
    let d = device();
    const N: usize = 4096;
    let mut rng = Lcg(7);
    let cases: Vec<codec::Case> = (0..N)
        .map(|i| {
            let kind = i % 4; // plain, subsurface, thin subsurface, coat and fuzz
            let mut c: codec::Case = bytemuck::Zeroable::zeroed();
            c.base_color = arr(rng.rgb());
            c.roughness = rng.next();
            c.normal = arr(rng.unit());
            c.metalness = rng.next();
            c.emission = arr(rng.rgb() * 4000.0 * rng.next());
            // Weights past 1, up to 1 / F0, and IORs over all of 0 to infinity.
            c.specular_weight = rng.next() * if i % 3 == 0 { 2000.0 } else { 1.0 };
            c.specular_color = arr(rng.rgb());
            c.ior = match i % 5 {
                0 => rng.next(),
                1 => 3.0 + 100.0 * rng.next(),
                2 => 1.5,
                _ => 1.0 + 2.0 * rng.next(),
            };
            c.geometric_normal = arr(rng.unit());
            c.diffuse_roughness = rng.next();
            c.raster_error = if i % 7 == 0 {
                0.0
            } else {
                1e-5 * 1e7f32.powf(rng.next())
            };
            c.flags = ((i as u32 % 3) << 5) | ((i as u32 % 2) * 2);
            if i % 6 == 5 {
                c.specular_color = arr(rng.rgb() * rng.next());
                c.specular_weight = rng.next();
                c.bsdf_flags |= BSDF_FRESNEL_MIX;
            }
            if i % 4 == 3 {
                c.bsdf_flags |= BSDF_METAL_FLAT;
            }
            if kind == 1 || kind == 2 {
                c.subsurface_weight = 0.05 + 0.95 * rng.next();
                c.subsurface_color = arr(rng.rgb());
                c.subsurface_radius = arr(rng.rgb() * 0.05);
                c.subsurface_anisotropy = rng.next() * 1.8 - 0.9;
            }
            if kind == 2 {
                c.bsdf_flags |= BSDF_THIN_WALLED;
            }
            if kind == 3 {
                c.coat_weight = rng.next();
                c.coat_color = arr(rng.rgb());
                c.fuzz_weight = rng.next();
                c.fuzz_color = arr(rng.rgb());
                c.coat_anisotropy = rng.next();
                c.coat_tangent_angle = rng.next() * std::f32::consts::PI;
                c.coat_normal = arr(rng.unit());
                c.bsdf_flags |= BSDF_COAT_FRAME;
            }
            if kind == 3 || i % 16 == 1 {
                c.thin_film_weight = 0.05 + 0.95 * rng.next();
                c.thin_film_thickness = 1500.0 * rng.next();
                c.thin_film_ior = 1.0 + 2.0 * rng.next();
                c.coat_darkening = rng.next();
            }
            if kind != 0 {
                c.coat_roughness = rng.next();
                c.coat_ior = 0.5 + 6.0 * rng.next();
                c.fuzz_roughness = rng.next();
                c.anisotropy = rng.next();
                c.tangent_angle = rng.next() * std::f32::consts::PI;
            }
            c
        })
        .collect();
    let upload = |bytes: &[u8]| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "codec input")
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let input = upload(bytemuck::cast_slice(&cases));
    let words = d
        .gpu_malloc::<codec::Words>(MemoryKind::Readback, N as u64, "words")
        .unwrap();
    let decoded = d
        .gpu_malloc::<codec::Case>(MemoryKind::Readback, N as u64, "decoded")
        .unwrap();
    let root = upload(bytemuck::bytes_of(&codec::Root {
        cases: input.gpu().cast(),
        words: words.gpu(),
        decoded: decoded.gpu(),
        count: N as u32,
        pad0: 0,
    }));
    let p = pipeline(&codec::MODULE, "round_trip");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.dispatch(
        &p,
        root.gpu().cast::<codec::Root>(),
        [N.div_ceil(64) as u32, 1, 1],
    );
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());

    let step8 = 0.5 / 255.0 + 1e-6;
    let close = |a: f32, b: f32, tol: f32, what: &str, i: usize| {
        assert!((a - b).abs() <= tol, "case {i} {what}: {a} vs {b}");
    };
    let close3 = |a: [f32; 3], b: [f32; 3], tol: f32, what: &str, i: usize| {
        for k in 0..3 {
            close(a[k], b[k], tol, what, i);
        }
    };
    let srgb3 = |v: [f32; 3]| v.map(|c| srgb_encode(c.clamp(0.0, 1.0)));
    for (i, ((c, w), g)) in cases
        .iter()
        .zip(words.as_slice().unwrap())
        .zip(decoded.as_slice().unwrap())
        .enumerate()
    {
        // The GPU decode is the contract's layout.
        let cpu = decode_core(w.a, w.b);
        close3(
            arr(cpu.base_color),
            g.base_color,
            1e-6,
            "base color layout",
            i,
        );
        close(cpu.roughness, g.roughness, 1e-6, "roughness layout", i);
        close3(arr(cpu.normal), g.normal, 1e-6, "normal layout", i);
        close3(
            arr(cpu.emission),
            g.emission,
            1e-6 * cpu.emission.max_element(),
            "emission layout",
            i,
        );
        close(cpu.metalness, g.metalness, 1e-6, "metalness layout", i);
        close(
            cpu.specular_weight,
            g.specular_weight,
            1e-6 * cpu.specular_weight.max(1.0),
            "specular weight layout",
            i,
        );
        close(cpu.ior, g.ior, 1e-5 * cpu.ior.max(1.0), "IOR layout", i);
        let mix = c.bsdf_flags & BSDF_FRESNEL_MIX != 0;
        assert_eq!(g.bsdf_flags & BSDF_FRESNEL_MIX != 0, mix, "case {i} mix");
        let flat = c.bsdf_flags & BSDF_METAL_FLAT;
        assert_eq!(g.bsdf_flags & BSDF_METAL_FLAT, flat, "case {i} flat metal");
        assert_eq!(cpu.flags, g.flags, "case {i} flags layout");
        let root = byte(w.b[1], 24);
        let scale = if mix { root * root } else { 1.0 };
        close3(
            arr(cpu.specular_color * scale),
            g.specular_color,
            1e-6,
            "specular color layout",
            i,
        );
        close(
            cpu.raster_error,
            g.raster_error,
            0.0,
            "raster error layout",
            i,
        );
        close3(
            arr(cpu.geometric_normal),
            g.geometric_normal,
            1e-6,
            "geometric normal layout",
            i,
        );
        close(
            cpu.diffuse_roughness,
            g.diffuse_roughness,
            1e-6,
            "diffuse roughness layout",
            i,
        );

        // Each field within its encoding's precision.
        close3(
            srgb3(g.base_color),
            srgb3(c.base_color),
            step8,
            "base color",
            i,
        );
        close(g.roughness, c.roughness, step8, "roughness", i);
        assert!(angle(g.normal, c.normal) < 7e-5, "case {i} normal");
        assert!(
            angle(g.geometric_normal, c.geometric_normal) < 0.02,
            "case {i} geometric normal"
        );
        // A largest channel past 511.5 steps takes the next exponent, whose half step is m / 511.5.
        let m = Vec3::from(c.emission).max_element();
        close3(g.emission, c.emission, m / 511.0 + 1e-30, "emission", i);
        close(g.metalness, c.metalness, step8, "metalness", i);
        close(
            g.specular_weight,
            c.specular_weight,
            step8 * (2.0 * c.specular_weight).max(1.0),
            "specular weight",
            i,
        );
        let r = |n: f32| (n - 1.0) / (n + 1.0);
        let step = if c.ior == 1.5 {
            1e-6
        } else {
            ior_step(r(c.ior))
        };
        close(r(g.ior), r(c.ior), step, "IOR", i);
        if mix {
            // The size of F0 to a byte of its root, its hue to seven sRGB bits.
            let peak = Vec3::from(c.specular_color).max_element();
            let got = Vec3::from(g.specular_color).max_element();
            close(got.sqrt(), peak.sqrt(), step8, "F0 size", i);
            let tint = |v: [f32; 3], p: f32| v.map(|x| x / p.max(1e-9));
            close3(
                srgb3(tint(g.specular_color, got)),
                srgb3(tint(c.specular_color, peak)),
                0.5 / 127.0 + 1e-6,
                "F0 hue",
                i,
            );
        } else {
            close3(
                srgb3(g.specular_color),
                srgb3(c.specular_color),
                0.5 / 127.0 + 1e-6,
                "specular color",
                i,
            );
        }
        if c.raster_error == 0.0 {
            assert_eq!(g.raster_error, 0.0);
        } else {
            assert!(
                g.raster_error >= c.raster_error && g.raster_error < 2.0 * c.raster_error,
                "case {i} raster error {} for {}",
                g.raster_error,
                c.raster_error
            );
        }
        assert_eq!(g.flags & 0x62, c.flags, "case {i} unlit and origin policy");

        let thin = c.bsdf_flags & BSDF_THIN_WALLED != 0;
        assert_eq!(g.flags & THIN_TWO_SIDED != 0, thin, "case {i} thin");
        assert_eq!(
            g.bsdf_flags & BSDF_THIN_WALLED != 0,
            thin,
            "case {i} thin BSDF"
        );
        let sss = c.subsurface_weight > 0.0;
        let lobes = c.coat_weight > 0.0 || c.fuzz_weight > 0.0 || c.thin_film_weight > 0.0;
        assert_eq!(
            g.flags & EXTENDED != 0,
            sss || lobes || c.anisotropy > 0.0,
            "case {i} extended"
        );
        assert_eq!(
            g.flags & SUBSURFACE != 0,
            sss && !thin,
            "case {i} subsurface"
        );
        assert_eq!(g.flags & LAYERS != 0, lobes, "case {i} layers");
        if g.flags & EXTENDED == 0 {
            continue;
        }
        close(g.coat_weight, c.coat_weight, step8, "coat weight", i);
        close(
            g.coat_roughness,
            c.coat_roughness,
            step8,
            "coat roughness",
            i,
        );
        close(
            r(g.coat_ior),
            r(c.coat_ior),
            ior_step(r(c.coat_ior)),
            "coat IOR",
            i,
        );
        close(g.fuzz_weight, c.fuzz_weight, step8, "fuzz weight", i);
        close(
            g.fuzz_roughness,
            c.fuzz_roughness,
            step8,
            "fuzz roughness",
            i,
        );
        close(g.anisotropy, c.anisotropy, step8, "anisotropy", i);
        let da = (g.tangent_angle - c.tangent_angle).rem_euclid(std::f32::consts::PI);
        assert!(
            da.min(std::f32::consts::PI - da) <= 0.5 / 256.0 * std::f32::consts::PI + 1e-5,
            "case {i} tangent angle"
        );
        close(
            g.subsurface_weight,
            c.subsurface_weight,
            step8,
            "subsurface weight",
            i,
        );
        if lobes {
            close3(
                srgb3(g.coat_color),
                srgb3(c.coat_color),
                step8,
                "coat color",
                i,
            );
            close3(
                srgb3(g.fuzz_color),
                srgb3(c.fuzz_color),
                step8,
                "fuzz color",
                i,
            );
            close(
                g.thin_film_weight,
                c.thin_film_weight,
                step8,
                "film weight",
                i,
            );
            let (t, want) = (g.thin_film_thickness, c.thin_film_thickness);
            close(t, want, want / 256.0 + 1e-6, "film thickness", i);
            if c.bsdf_flags & BSDF_COAT_FRAME != 0 {
                close(
                    g.coat_anisotropy,
                    c.coat_anisotropy,
                    step8,
                    "coat anisotropy",
                    i,
                );
                assert!(
                    angle(g.coat_normal, c.coat_normal) < 2e-3,
                    "case {i} coat normal"
                );
                assert_ne!(g.bsdf_flags & BSDF_COAT_FRAME, 0, "case {i} coat frame");
            }
            close(
                g.coat_darkening,
                c.coat_darkening,
                step8,
                "coat darkening",
                i,
            );
            assert_eq!(
                g.bsdf_flags & 8 != 0,
                c.thin_film_weight > 0.0,
                "case {i} film flag"
            );
            let step = ior_step(r(c.thin_film_ior));
            close(r(g.thin_film_ior), r(c.thin_film_ior), step, "film IOR", i);
        }
        if sss {
            close(
                g.subsurface_anisotropy,
                c.subsurface_anisotropy,
                0.5 / 127.0 + 1e-6,
                "subsurface anisotropy",
                i,
            );
            let mr = Vec3::from(c.subsurface_radius).max_element();
            close3(
                g.subsurface_radius,
                c.subsurface_radius,
                mr / 512.0 + 1e-30,
                "mean free path",
                i,
            );
            // The tint is 8 bits of sss x color (x (g + 1) / 2 thin-walled); dividing by the
            // 8-bit weight widens its error.
            let share = if thin {
                c.subsurface_weight * (c.subsurface_anisotropy + 1.0) * 0.5
            } else {
                c.subsurface_weight
            };
            let tint = Vec3::from(c.subsurface_color) * share;
            let got = Vec3::from(g.subsurface_color) * share;
            for k in 0..3 {
                let e = (srgb_encode(got[k].min(1.0)) - srgb_encode(tint[k])).abs();
                assert!(e < 8.0 / 255.0, "case {i} subsurface color {got} vs {tint}");
            }
        }
    }
    assert_clean();
}

// --- Cooked scenes -----------------------------------------------------------------------

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr_scene_gbuffer_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn png(
    name: &str,
    w: u32,
    h: u32,
    color: png::ColorType,
    texel: impl Fn(u32, u32) -> Vec<u8>,
) -> PathBuf {
    let path = temp(name);
    let data: Vec<u8> = (0..w * h).flat_map(|i| texel(i % w, i / w)).collect();
    let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), w, h);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&data).unwrap();
    path
}

/// Triangles `tris` over `positions`, each wound so its front faces `outward(centroid)`.
fn mesh(
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    tris: Vec<[u32; 3]>,
    sections: &[u32],
) -> MeshInput {
    let mut m = MeshInput {
        positions,
        normals,
        uvs: Some(uvs),
        ..MeshInput::default()
    };
    let mut first = 0;
    for &count in sections {
        m.sections.push(Span {
            first: first * 3,
            count: count * 3,
        });
        first += count;
    }
    for t in tris {
        let [a, b, c] = t.map(|i| m.positions[i as usize]);
        let n = m.normals[t[0] as usize] + m.normals[t[1] as usize] + m.normals[t[2] as usize];
        let t = if (b - a).cross(c - a).dot(n) >= 0.0 {
            t
        } else {
            [t[0], t[2], t[1]]
        };
        m.indices.extend(t);
    }
    m
}

/// An `n` x `n` grid over `x, z` in [-size / 2, size / 2], heights from `height`, uv over
/// [0, uv].
fn grid(n: u32, size: f32, uv: f32, height: impl Fn(f32, f32) -> f32) -> MeshInput {
    let (mut p, mut nrm, mut uvs, mut tris) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let w = n + 1;
    let h = |x: f32, z: f32| height(x, z);
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            let (x, z) = ((u - 0.5) * size, (v - 0.5) * size);
            let e = 1e-3;
            let dx = (h(x + e, z) - h(x - e, z)) / (2.0 * e);
            let dz = (h(x, z + e) - h(x, z - e)) / (2.0 * e);
            p.push(Vec3::new(x, h(x, z), z));
            nrm.push(Vec3::new(-dx, 1.0, -dz).normalize());
            uvs.push(Vec2::new(u, v) * uv);
        }
    }
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            tris.push([c, c + w, c + 1]);
            tris.push([c + 1, c + w, c + w + 1]);
        }
    }
    let count = tris.len() as u32;
    mesh(p, nrm, uvs, tris, &[count])
}

/// A sphere of `radius` in `rings` latitude bands: the top half section 0, the bottom half
/// section 1.
fn sphere(rings: u32, radius: f32) -> MeshInput {
    let segments = 2 * rings;
    let (mut p, mut nrm, mut uvs) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..=rings {
        for s in 0..=segments {
            let (v, u) = (r as f32 / rings as f32, s as f32 / segments as f32);
            let (theta, phi) = (v * std::f32::consts::PI, u * std::f32::consts::TAU);
            let n = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );
            p.push(n * radius);
            nrm.push(n);
            uvs.push(Vec2::new(u * 4.0, v * 2.0));
        }
    }
    let (mut top, mut bottom) = (Vec::new(), Vec::new());
    let w = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let c = r * w + s;
            let half = if r < rings / 2 { &mut top } else { &mut bottom };
            if r > 0 {
                half.push([c, c + 1, c + w]);
            }
            if r < rings - 1 {
                half.push([c + 1, c + w + 1, c + w]);
            }
        }
    }
    let sections = [top.len() as u32, bottom.len() as u32];
    mesh(p, nrm, uvs, [top, bottom].concat(), &sections)
}

/// A quad over `corners` (counter-clockwise seen from its front), uv over [0, 1].
fn quad(corners: [Vec3; 4]) -> MeshInput {
    let n = (corners[1] - corners[0])
        .cross(corners[3] - corners[0])
        .normalize();
    mesh(
        corners.to_vec(),
        vec![n; 4],
        vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
        &[2],
    )
}

/// The 256-part rig: a 2 m plate (part 0, section 0) under 255 boxes 0.08 m wide and 0.15 m
/// tall (parts 1 to 255, section 1), box p standing at grid cell (p % 16, p / 16) and pivoted
/// at its base, where `pose.slang` turns it.
fn rig() -> MeshInput {
    let (mut p, mut nrm, mut uvs, mut part, mut pivot) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut plate, mut boxes) = (Vec::new(), Vec::new());
    let mut face =
        |corners: [Vec3; 4], n: Vec3, value: u16, base: Vec3, tris: &mut Vec<[u32; 3]>| {
            let first = p.len() as u32;
            for (c, uv) in corners
                .iter()
                .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
            {
                p.push(*c);
                nrm.push(n);
                uvs.push(Vec2::from(uv));
                part.push(value);
                pivot.push(base);
            }
            tris.push([first, first + 1, first + 2]);
            tris.push([first, first + 2, first + 3]);
        };
    let square = |a: f32, b: f32, at: f32, axis: usize| {
        let corner = |u: f32, v: f32| {
            let mut c = [0.0; 3];
            c[axis] = at;
            c[(axis + 1) % 3] = u;
            c[(axis + 2) % 3] = v;
            Vec3::from(c)
        };
        [corner(-a, -b), corner(a, -b), corner(a, b), corner(-a, b)]
    };
    face(square(1.0, 1.0, 0.0, 1), Vec3::Y, 0, Vec3::ZERO, &mut plate);
    for b in 1..RIG_PARTS as u16 {
        let base = Vec3::new(
            (f32::from(b % 16) - 7.5) * 0.125,
            0.0,
            (f32::from(b / 16) - 7.5) * 0.125,
        );
        let (h, w) = (0.15, 0.04);
        let center = base + Vec3::Y * (h * 0.5);
        for axis in 0..3 {
            let half = if axis == 1 { h * 0.5 } else { w };
            for sign in [-1.0, 1.0] {
                let mut n = [0.0; 3];
                n[axis] = sign;
                let n = Vec3::from(n);
                let (u, v) = match axis {
                    0 => (h * 0.5, w),
                    1 => (w, w),
                    _ => (w, h * 0.5),
                };
                let corners = square(u, v, sign * half, axis).map(|c| c + center);
                face(corners, n, b, base, &mut boxes);
            }
        }
    }
    let sections = [plate.len() as u32, boxes.len() as u32];
    let mut m = mesh(p, nrm, uvs, [plate, boxes].concat(), &sections);
    m.parts = Some(MeshParts {
        values: part,
        count: RIG_PARTS as u16,
        by_section: false,
        pivots: Some(pivot),
    });
    m
}

/// The parallax card's virtual quad, as `surfaces/parallax.slang` places it.
fn slab_corners() -> [Vec3; 4] {
    let z = |x: f32| -0.2 - 0.3 * x;
    [
        Vec3::new(-0.4, -0.4, z(-0.4)),
        Vec3::new(0.4, -0.4, z(0.4)),
        Vec3::new(0.4, 0.4, z(0.4)),
        Vec3::new(-0.4, 0.4, z(-0.4)),
    ]
}

/// A saturated red base color factor on a saturated green texture (sRGB bytes), whose models
/// tag the texture with each primaries.
const TINT_FACTOR: [f32; 3] = [0.9, 0.05, 0.05];
const TINT_TEXEL: [u8; 3] = [40, 230, 60];
const TINTED: [(&str, Primaries); 3] = [
    ("tinted rec709", Primaries::Rec709),
    ("tinted p3", Primaries::DisplayP3),
    ("tinted rec2020", Primaries::Rec2020),
];

/// The halves of the texture the second-uv quad reads on uv set 1 (sRGB bytes); its uv set
/// 0 sits in the red half.
const HALF_RED: [u8; 3] = [230, 20, 20];
const HALF_GREEN: [u8; 3] = [20, 230, 20];

/// The decal texture's opaque texels (sRGB bytes, alpha) and the roughness of its material
/// and of the matte decal under it.
const STICKER_TEXEL: [u8; 4] = [40, 90, 230, 255];
const STICKER_ROUGHNESS: f32 = 0.15;
/// The tangent-space normal of the tilted models' texture.
const TILT: [f32; 3] = [0.3, 0.5, 0.812];
type SkewCorner = ([f32; 3], [f32; 3], [f32; 3], f32);

/// Position, normal, tangent and sign of a triangle's corners, whose frames spread far enough
/// that between them the normal and tangent are neither unit nor perpendicular.
const SKEW: [SkewCorner; 3] = [
    ([-1.0, -1.0, 0.0], [-0.8, -0.2, 1.0], [1.0, 0.2, 0.8], -1.0),
    ([1.0, -1.0, 0.0], [0.8, -0.3, 1.0], [1.0, -0.5, -0.8], -1.0),
    ([0.0, 1.0, 0.0], [0.0, 0.9, 1.0], [1.0, 0.0, 0.0], -1.0),
];
const MATTE_ROUGHNESS: f32 = 0.6;

fn cook() -> Arc<[u8]> {
    let albedo = png("albedo.png", 256, 256, png::ColorType::Rgba, |x, y| {
        let check = ((x / 32) ^ (y / 32)) & 1 == 1;
        let base = if check {
            [200, 170, 120]
        } else {
            [90, 120, 70]
        };
        vec![base[0], base[1], (u32::from(base[2]) + x / 4) as u8, 255]
    });
    let bumps = png("normal.png", 128, 128, png::ColorType::Rgb, |x, y| {
        let a = (x as f32 / 16.0 * std::f32::consts::TAU).sin() * 0.5;
        let b = (y as f32 / 16.0 * std::f32::consts::TAU).cos() * 0.5;
        let n = Vec3::new(a, b, 1.0).normalize() * 0.5 + 0.5;
        vec![
            (n.x * 255.0) as u8,
            (n.y * 255.0) as u8,
            (n.z * 255.0) as u8,
        ]
    });
    let turned_quad = || MeshInput {
        // Its second uvs turn a quarter turn: their u runs down -y and their up runs +x.
        uvs1: Some(vec![
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
        ]),
        ..quad([
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(-0.5, 0.5, 0.0),
        ])
    };
    let tilt = png("tilt.png", 16, 16, png::ColorType::Rgb, |_, _| {
        TILT.map(|c| ((c * 0.5 + 0.5) * 255.0).round() as u8)
            .to_vec()
    });
    let holes = png("holes.png", 128, 128, png::ColorType::Grayscale, |x, y| {
        let (u, v) = (x as f32 / 128.0 - 0.5, y as f32 / 128.0 - 0.5);
        let ring = ((u * u + v * v).sqrt() * 10.0).fract();
        vec![if ring < 0.6 { 255 } else { 0 }]
    });
    let texture = |name: &str, path: PathBuf, usage| AssetSpec {
        name: name.into(),
        source: Source::Texture {
            path,
            settings: TextureSettings {
                usage,
                ..TextureSettings::default()
            },
        },
    };
    let id = AssetId::from_name;
    let mut ground = Material {
        specular_roughness: 0.7,
        ..Material::default()
    };
    ground.textures[slot::BASE_COLOR].texture = id("tex/albedo");
    ground.textures[slot::NORMAL].texture = id("tex/normal");
    let mut tilted = Material::default();
    tilted.textures[slot::NORMAL].texture = id("tex/tilt");
    let mut tilted_object = Material {
        object_normals: Material::OBJECT_NORMAL,
        ..Material::default()
    };
    tilted_object.textures[slot::NORMAL].texture = id("tex/tilt object");
    let mut tilted_uv1 = tilted;
    tilted_uv1.textures[slot::NORMAL].uv_set = 1;
    let mut tilted_keeping_uv1 = tilted;
    tilted_keeping_uv1.textures[slot::BASE_COLOR].texture = id("tex/albedo");
    tilted_keeping_uv1.textures[slot::BASE_COLOR].uv_set = 1;
    let coat = Material {
        base_color: [0.1, 0.25, 0.8],
        specular_roughness: 0.4,
        coat_weight: 0.8,
        coat_roughness: 0.1,
        ..Material::default()
    };
    let lamp = Material {
        base_color: [0.9, 0.5, 0.2],
        emission_color: [1.0, 0.6, 0.3],
        emission_luminance: 40.0,
        ..Material::default()
    };
    let mut leaf = Material {
        base_color: [0.2, 0.6, 0.15],
        alpha_mode: Material::MASK,
        alpha_cutoff: 0.5,
        double_sided: 1,
        ..Material::default()
    };
    leaf.textures[slot::OPACITY].texture = id("tex/holes");
    let metal = Material {
        base_color: [0.9, 0.8, 0.6],
        metalness: 1.0,
        specular_roughness: 0.25,
        ..Material::default()
    };
    let card = Material {
        base_color: [0.8, 0.3, 0.3],
        alpha_mode: Material::MASK,
        geometry_opacity: 0.0,
        alpha_cutoff: 0.5,
        surface: 1,
        ..Material::default()
    };
    let plain = Material {
        base_color: [0.8, 0.3, 0.3],
        ..Material::default()
    };
    let probe = Material {
        surface: linked::surface::PROBE,
        ..Material::default()
    };
    let steady = Material {
        surface: linked::surface::STEADY,
        ..Material::default()
    };
    let model = |name: &str, mesh: MeshInput, materials: Vec<Material>, lod: LodMode| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings {
                lod,
                ..MeshSettings::default()
            },
        },
    };
    let one = LodMode::default;
    let unit = [
        Vec3::new(-0.5, -0.5, 0.0),
        Vec3::new(0.5, -0.5, 0.0),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(-0.5, 0.5, 0.0),
    ];
    let sticker = png("sticker.png", 64, 64, png::ColorType::Rgba, |x, y| {
        let (u, v) = (x as f32 / 32.0 - 1.0, y as f32 / 32.0 - 1.0);
        let r = (u * u + v * v).sqrt();
        let shape = (0.55..0.85).contains(&r) || (r < 0.85 && (u.abs() < 0.12 || v > 0.3));
        if shape {
            STICKER_TEXEL.to_vec()
        } else {
            vec![0, 0, 0, 0]
        }
    });
    let mut sticker_material = Material {
        specular_roughness: STICKER_ROUGHNESS,
        ..Material::default()
    };
    sticker_material.textures[slot::BASE_COLOR].texture = id("tex/sticker");
    let matte = Material {
        specular_roughness: MATTE_ROUGHNESS,
        ..Material::default()
    };
    let green = png("green.png", 4, 4, png::ColorType::Rgb, |_, _| {
        TINT_TEXEL.to_vec()
    });
    let mut tinted = Vec::new();
    for (name, primaries) in TINTED {
        let tex = format!("tex/{name}");
        tinted.push(AssetSpec {
            name: tex.clone(),
            source: Source::Texture {
                path: green.clone(),
                settings: TextureSettings {
                    usage: Usage::Color,
                    primaries: Some(primaries),
                    ..TextureSettings::default()
                },
            },
        });
        let mut m = Material {
            base_color: TINT_FACTOR,
            ..Material::default()
        };
        m.textures[slot::BASE_COLOR].texture = id(&tex);
        tinted.push(model(name, quad(unit), vec![m], one()));
    }
    let halves = png("halves.png", 64, 64, png::ColorType::Rgb, |x, _| {
        if x < 32 { HALF_RED } else { HALF_GREEN }.to_vec()
    });
    tinted.push(texture("tex/halves", halves, Usage::Color));
    let mut second = Material::default();
    second.textures[slot::BASE_COLOR].texture = id("tex/halves");
    second.textures[slot::BASE_COLOR].uv_set = 1;
    let mut second_quad = quad(unit);
    second_quad.uvs1 = second_quad.uvs.take();
    second_quad.uvs = Some(vec![Vec2::new(0.25, 0.5); 4]);
    tinted.push(model("second uv", second_quad, vec![second], one()));
    // Two pages of placed balls, a world cell apart, the second prototype without its lamp, every
    // third ball sheared its own way.
    let placements = |x: f64| -> Vec<Placement> {
        (0..12)
            .map(|i| Placement {
                position: DVec3::new(
                    x + 2.0 + 3.0 * f64::from(i % 4),
                    1.0,
                    2.0 + 3.0 * f64::from(i / 4),
                )
                .to_array(),
                linear: if i % 3 == 2 {
                    let shear = i as f32 * 0.1;
                    Linear::Matrix([[1.0, 0.0, 0.0], [shear, 1.0, 0.0], [0.0, 0.5, 1.0]])
                } else {
                    Linear::Scaled {
                        rotation: Quat::from_rotation_y(i as f32 * 0.7).to_array(),
                        scale: [1.0; 3],
                    }
                },
                prototype: (i % 2) as u16,
            })
            .collect()
    };
    tinted.push(AssetSpec {
        name: "rows".into(),
        source: Source::Instances(Instances {
            records: [placements(0.0), placements(4096.0)].concat(),
            prototypes: vec![
                PrototypeSpec {
                    model: "ball".into(),
                    ..PrototypeSpec::default()
                },
                PrototypeSpec {
                    model: "ball".into(),
                    sections: vec![
                        SectionSpec::default(),
                        SectionSpec {
                            material: None,
                            hidden: true,
                        },
                    ],
                    ..PrototypeSpec::default()
                },
            ],
            ..Instances::default()
        }),
    });
    // A representation chain: a DAG ball, then three coarser balls, each its own color.
    for (step, (rings, color)) in CHAIN_SHAPES.into_iter().enumerate() {
        let m = Material {
            base_color: color,
            ..Material::default()
        };
        let lod = if step == 0 { LodMode::Dag } else { one() };
        tinted.push(model(CHAIN[step], sphere(rings, 0.9), vec![m, m], lod));
    }
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                texture("tex/albedo", albedo, Usage::Color),
                texture("tex/normal", bumps, Usage::Normal),
                texture("tex/holes", holes, Usage::Linear1),
                texture("tex/sticker", sticker, Usage::Color),
                texture("tex/tilt object", tilt.clone(), Usage::Linear4),
                texture("tex/tilt", tilt, Usage::Normal),
                model("tilted", quad(unit), vec![tilted], one()),
                model("tilted object", quad(unit), vec![tilted_object], one()),
                AssetSpec {
                    name: "tilted skew".into(),
                    source: Source::Mesh {
                        mesh: MeshInput {
                            tangents: Some(SKEW.map(|c| Vec3::from(c.2).extend(c.3)).to_vec()),
                            ..mesh(
                                SKEW.map(|c| c.0.into()).to_vec(),
                                SKEW.map(|c| Vec3::from(c.1).normalize()).to_vec(),
                                vec![Vec2::ZERO, Vec2::X, Vec2::Y],
                                vec![[0, 1, 2]],
                                &[1],
                            )
                        },
                        materials: vec![tilted],
                        settings: MeshSettings {
                            tangent_error_max: None,
                            ..MeshSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: "tilted sets".into(),
                    source: Source::Mesh {
                        mesh: turned_quad(),
                        materials: vec![tilted_keeping_uv1],
                        settings: MeshSettings {
                            tangent_error_max: None,
                            ..MeshSettings::default()
                        },
                    },
                },
                model("tilted on uv1", turned_quad(), vec![tilted_uv1], one()),
                AssetSpec {
                    name: "tilted computed".into(),
                    source: Source::Mesh {
                        mesh: quad(unit),
                        materials: vec![tilted],
                        settings: MeshSettings {
                            tangent_error_max: None,
                            ..MeshSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: "tilted stored".into(),
                    source: Source::Mesh {
                        mesh: MeshInput {
                            tangents: Some(vec![Vec4::new(1.0, 0.0, 0.0, 1.0); 4]),
                            ..quad(unit)
                        },
                        materials: vec![tilted],
                        settings: MeshSettings {
                            tangent_error_max: None,
                            ..MeshSettings::default()
                        },
                    },
                },
                model("sticker", quad(unit), vec![sticker_material], one()),
                model("matte", quad(unit), vec![matte], one()),
                model(
                    "ground",
                    grid(24, 8.0, 3.0, |_, _| 0.0),
                    vec![ground],
                    one(),
                ),
                model("ball", sphere(24, 0.9), vec![coat, lamp], LodMode::Dag),
                model("leaf", quad(unit.map(|c| c * 1.6)), vec![leaf], one()),
                model(
                    "bumps",
                    grid(64, 2.0, 1.0, |x, z| {
                        0.08 * (x * 7.0).sin() * (z * 5.0).cos()
                    }),
                    vec![metal],
                    LodMode::Discrete {
                        errors: vec![0.004, 0.03, 0.12],
                    },
                ),
                model("card", quad(unit), vec![card], one()),
                model("slab", quad(slab_corners()), vec![plain], one()),
                model("plain", quad(unit), vec![plain], one()),
                model("probe", quad(unit.map(|c| c * 12.0)), vec![probe], one()),
                model("steady ball", sphere(16, 0.9), vec![steady, steady], one()),
                model("rig", rig(), vec![plain, metal], one()),
                model("dag rig", rig(), vec![plain, metal], LodMode::Dag),
                model(
                    "steady ground",
                    grid(12, 6.0, 2.0, |x, z| 0.1 * (x * 0.9).sin() * z.cos()),
                    vec![steady],
                    one(),
                ),
                AssetSpec {
                    name: "worm".into(),
                    source: Source::Gltf {
                        path: worm_glb(),
                        settings: GltfSettings {
                            mesh: MeshSettings {
                                deformation_bound: 1.0,
                                ..MeshSettings::default()
                            },
                            ..GltfSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: "door".into(),
                    source: Source::Usd {
                        path: door_stage(),
                        settings: UsdSettings::default(),
                    },
                },
                AssetSpec {
                    name: "hung".into(),
                    source: Source::Usd {
                        path: hung_stage(),
                        settings: UsdSettings::default(),
                    },
                },
                AssetSpec {
                    name: "paced".into(),
                    source: Source::Usd {
                        path: paced_stage("paced.usda", MOVER),
                        settings: UsdSettings::default(),
                    },
                },
                AssetSpec {
                    name: "paced alone".into(),
                    source: Source::Usd {
                        path: paced_stage("paced-alone.usda", ""),
                        settings: UsdSettings::default(),
                    },
                },
                AssetSpec {
                    name: "swaying door".into(),
                    source: Source::Usd {
                        path: swaying_door_stage(),
                        settings: UsdSettings::default(),
                    },
                },
                AssetSpec {
                    name: "leaning worm".into(),
                    source: Source::Gltf {
                        path: worm_glb_of("lean.glb", &[], true),
                        settings: GltfSettings::default(),
                    },
                },
                AssetSpec {
                    name: "crowd".into(),
                    source: Source::Gltf {
                        path: worm_glb_of("crowd.glb", &CROWD, false),
                        settings: GltfSettings {
                            mesh: MeshSettings {
                                deformation_bound: 1.0,
                                ..MeshSettings::default()
                            },
                            ..GltfSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: "long walk".into(),
                    source: Source::Instances(long_walk()),
                },
                AssetSpec {
                    name: "paced crowd".into(),
                    source: Source::Instances(paced_crowd()),
                },
                AssetSpec {
                    name: "unbounded worm".into(),
                    source: Source::Gltf {
                        path: worm_glb(),
                        settings: GltfSettings::default(),
                    },
                },
                model(
                    "dag ground",
                    grid(96, 8.0, 2.0, |x, z| {
                        0.15 * (x * 1.3).sin() * (z * 0.9).cos()
                    }),
                    vec![metal],
                    LodMode::Dag,
                ),
                model(
                    "discrete ground",
                    grid(96, 8.0, 2.0, |x, z| {
                        0.15 * (x * 1.3).sin() * (z * 0.9).cos()
                    }),
                    vec![metal],
                    LodMode::Discrete { errors: vec![0.05] },
                ),
            ]
            .into_iter()
            .chain(tinted)
            .collect(),
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0)
}

fn pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(cook)
}

struct World {
    scenes: Scenes,
    mount: MountId,
    _workers: Workers,
    texture_filter: TextureFilter,
}

fn time(frame: u32) -> FrameTime {
    FrameTime {
        absolute: f64::from(frame) / 60.0,
        delta: 1.0 / 60.0,
    }
}

impl World {
    fn new(kernels: &'static KernelSet) -> Self {
        Self::with(&RendererDesc {
            kernels,
            ..RendererDesc::default()
        })
    }

    fn with(desc: &RendererDesc) -> Self {
        Self::with_pack(desc, pack())
    }

    /// A world that mounts `bytes` in place of [`pack`].
    fn with_pack(desc: &RendererDesc, bytes: &Arc<[u8]>) -> Self {
        let mut scenes = Scenes::new(device(), desc).unwrap();
        let workers = Workers::start(&scenes, 2);
        let mount = scenes.mount(PackSource::Memory(Arc::clone(bytes)));
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
        Self {
            scenes,
            mount,
            texture_filter: TextureFilter::Hardware,
            _workers: workers,
        }
    }

    fn model(&self, name: &str) -> ModelId {
        self.scenes
            .model(self.mount, AssetId::from_name(name))
            .unwrap()
    }
}

/// World from view for a camera at `eye` looking at `target`, y up.
fn look(eye: Vec3, target: Vec3) -> Quat {
    orr_math::camera::rh::view::look_at_quat(eye, target, Vec3::Y).inverse()
}

/// One placed copy: model name, rotation and translation in world space.
#[derive(Clone, Copy)]
struct Item {
    model: &'static str,
    linear: Mat3,
    at: Vec3,
}

/// A scene of copies in one transform group at the world origin.
struct Placed {
    scene: SceneId,
    ranges: Vec<InstanceRange>,
    group: TransformGroupId,
    history: ViewHistoryId,
    count: u32,
}

fn place(w: &mut World, copies: &[Item]) -> Placed {
    place_at(w, copies, DVec3::ZERO, 100.0)
}

/// [`place`] with the group anchored at `anchor` and the copies traced within `rt_range`.
fn place_at(w: &mut World, copies: &[Item], anchor: DVec3, rt_range: f32) -> Placed {
    place_with(w, copies, anchor, rt_range, None, None)
}

/// [`place_at`] with one-section copies drawn in `material`, in place of the section's, and
/// held at `lod` from their first frame.
fn place_with(
    w: &mut World,
    copies: &[Item],
    anchor: DVec3,
    rt_range: f32,
    material: Option<MaterialId>,
    lod: Option<DebugLod>,
) -> Placed {
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let models: Vec<ModelId> = copies.iter().map(|c| w.model(c.model)).collect();
    let mut ranges = Vec::new();
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    if let Some(m) = material {
        s.material_sets(0, 1).unwrap()[0] = m;
    }
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: copies.len() as u32,
            writer: GroupWriter::Cpu,
            anchor,
        })
        .unwrap();
    for (i, (c, model)) in copies.iter().zip(models).enumerate() {
        s.transforms(group, i as u32, 1).unwrap()[0] =
            Affine3::from_mat3_translation(c.linear, c.at).into();
        let range = s
            .add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: group,
                first: i as u32,
                pose: None,
                material_set: material.map(|_| 0),
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range,
                animation_range: 0.0,
                user_tag: i as u32,
            })
            .unwrap();
        if lod.is_some() {
            s.debug_lod(range, lod);
        }
        ranges.push(range);
    }
    drop(s);
    wait(f.end(&[]).unwrap());
    Placed {
        scene,
        ranges,
        group,
        history,
        count: copies.len() as u32,
    }
}

/// What one rendered view reads back.
struct Frame {
    view: RasterView,
    a: Vec<[u32; 4]>,
    b: Vec<[u32; 2]>,
    x: Option<Vec<[u32; 4]>>,
    y: Option<Vec<[u32; 4]>>,
    depth: Vec<f32>,
    motion: Vec<[f32; 2]>,
    /// Instance slot + 1 per pixel, 0 where nothing was drawn.
    ids: Vec<u32>,
    /// Mesh cluster + 1 per pixel, 0 where nothing was drawn.
    clusters: Vec<u32>,
    counts: Vec<u32>,
}

fn texels<T: bytemuck::Pod>(t: &Texture) -> Vec<T> {
    let bytes = read_texture(t, &TextureRegion::whole_mip(t.desc(), 0));
    bytemuck::pod_collect_to_vec(&bytes)
}

fn read_back(view: &RasterView) -> Frame {
    let pixels = u64::from(EXTENT[0] * EXTENT[1]);
    let vis: Vec<u64> = read_buffer(view.targets.visibility, pixels);
    let counts = read_buffer(view.counts, 16);
    let clusters = visible_clusters(view);
    let ids = vis
        .iter()
        .map(|&w| {
            if w == 0 {
                0
            } else {
                clusters[(w as u32 >> 7) as usize].instance + 1
            }
        })
        .collect();
    let mesh_clusters = vis
        .iter()
        .map(|&w| {
            if w == 0 {
                0
            } else {
                clusters[(w as u32 >> 7) as usize].cluster + 1
            }
        })
        .collect();
    let motion: Vec<[f32; 2]> = texels::<[u16; 2]>(&view.motion)
        .iter()
        .map(|m| m.map(|bits| half::f16::from_bits(bits).to_f32()))
        .collect();
    assert!(motion.iter().flatten().all(|v| !v.is_nan()), "NaN motion");
    Frame {
        view: view.clone(),
        a: texels(&view.gbuf_a),
        b: texels(&view.gbuf_b),
        x: view.gbuf_x.as_ref().map(|x| texels(x)),
        y: view.gbuf_y.as_ref().map(|y| texels(y)),
        depth: texels(&view.depth),
        motion,
        ids,
        clusters: mesh_clusters,
        counts,
    }
}

/// The views' pixels per radian at the image center: half the rows over tan(0.45), half
/// the 0.9 rad vertical field.
const VIEW_PIXELS_PER_RADIAN: f32 = 0.5 * EXTENT[1] as f32 / 0.483_055_07;

/// Renders `placed` from a camera at `eye` looking at `target`, the scene focus at the eye,
/// with copy transforms `moved` written this frame, and reads the view back.
fn render(
    w: &mut World,
    placed: &Placed,
    eye: Vec3,
    target: Vec3,
    moved: &[Item],
    reset: bool,
    frame: u32,
) -> Frame {
    let list = [placed.scene];
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(frame),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(placed.scene).unwrap();
        s.set_focus(eye.as_dvec3());
        f.request_detail(placed.scene, VIEW_PIXELS_PER_RADIAN)
            .unwrap();
        for (i, c) in moved.iter().enumerate() {
            s.transforms(placed.group, i as u32, 1).unwrap()[0] =
                Affine3::from_mat3_translation(c.linear, c.at).into();
        }
        assert!(moved.is_empty() || moved.len() as u32 == placed.count);
    }
    let view = f
        .raster_view(&ViewDesc {
            scene: placed.scene,
            history: placed.history,
            position: eye.as_dvec3(),
            rotation: look(eye, target),
            projection: Projection {
                fov_y: 0.9,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: Vec2::new(0.25, -0.125),
            camera_cut: reset,
            mip_bias: 0.0,
            texture_filter: w.texture_filter,
        })
        .unwrap();
    wait(f.end(&[]).unwrap());
    read_back(&view)
}

fn showcase() -> Vec<Item> {
    let tilt = Mat3::from_rotation_y(0.6) * Mat3::from_rotation_x(-0.3);
    vec![
        Item {
            model: "ground",
            linear: Mat3::IDENTITY,
            at: Vec3::ZERO,
        },
        Item {
            model: "ball",
            linear: Mat3::IDENTITY,
            at: Vec3::new(-1.6, 0.95, 0.0),
        },
        Item {
            model: "leaf",
            linear: tilt,
            at: Vec3::new(1.4, 1.0, 0.6),
        },
        Item {
            model: "bumps",
            linear: Mat3::from_rotation_y(0.4),
            at: Vec3::new(0.4, 0.15, -1.8),
        },
    ]
}

/// One G-buffer channel as RGB in [0, 1] per pixel.
type Channel<'a> = Box<dyn Fn(usize) -> [f32; 3] + 'a>;

/// An 8-bit image from per-pixel RGB in [0, 1].
fn image(f: impl Fn(usize) -> [f32; 3]) -> Image {
    let n = (EXTENT[0] * EXTENT[1]) as usize;
    let pixels = (0..n)
        .flat_map(|i| {
            let c = f(i).map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
            [c[0], c[1], c[2], 255]
        })
        .collect();
    Image {
        width: EXTENT[0],
        height: EXTENT[1],
        pixels: Pixels::Rgba8(pixels),
    }
}

#[test]
fn each_gbuffer_channel_matches_its_reference() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let copies = showcase();
    let placed = place(&mut w, &copies);
    let eye = Vec3::new(0.3, 2.6, 5.5);
    let target = Vec3::new(0.0, 0.6, 0.0);
    render(
        &mut w,
        &placed,
        eye + Vec3::new(-0.15, 0.05, 0.1),
        target,
        &[],
        true,
        1,
    );
    let g = render(&mut w, &placed, eye, target, &[], false, 2);
    assert_eq!(g.counts[1..3], [0, 0], "nothing dropped");
    assert!(g.x.is_some(), "the coat needs the extended target");
    let x = g.x.as_ref().unwrap();
    let flip = Tolerance::Flip {
        mean: 0.01,
        p99: 0.1,
    };
    let core = |i: usize| decode_core(g.a[i], g.b[i]);
    let n01 = |n: Vec3| arr(n * 0.5 + 0.5);
    let channels: [(&str, Channel<'_>); 9] = [
        (
            "base_color",
            Box::new(|i| [byte(g.a[i][0], 0), byte(g.a[i][0], 8), byte(g.a[i][0], 16)]),
        ),
        (
            "normal",
            Box::new(|i| {
                if g.ids[i] == 0 {
                    [0.0; 3]
                } else {
                    n01(core(i).normal)
                }
            }),
        ),
        (
            "geometric_normal",
            Box::new(|i| {
                if g.ids[i] == 0 {
                    [0.0; 3]
                } else {
                    n01(core(i).geometric_normal)
                }
            }),
        ),
        (
            "roughness_metalness_specular",
            Box::new(|i| {
                let c = core(i);
                [c.roughness, c.metalness, c.specular_weight]
            }),
        ),
        (
            "emission",
            Box::new(|i| arr(core(i).emission.map(|e| e / (1.0 + e)))),
        ),
        ("depth", Box::new(|i| [g.depth[i] * 20.0; 3])),
        (
            "motion",
            Box::new(|i| {
                let m = g.motion[i];
                [0.5 + m[0] * 20.0, 0.5 + m[1] * 20.0, 0.5]
            }),
        ),
        (
            "raster_error_flags_diffuse",
            Box::new(|i| {
                [
                    ((g.b[i][0] >> 21) & 31) as f32 / 31.0,
                    (g.a[i][3] >> 24) as f32 / 127.0,
                    core(i).diffuse_roughness,
                ]
            }),
        ),
        (
            "extended",
            // The target holds data only where the pixel's flags say so.
            Box::new(|i| {
                if (g.a[i][3] >> 24) & EXTENDED == 0 {
                    [0.0; 3]
                } else {
                    [byte(x[i][2], 0), byte(x[i][2], 8), byte(x[i][3], 8)]
                }
            }),
        ),
    ];
    for (name, f) in &channels {
        check(&reference(name), &image(f), flip);
    }
    assert_clean();
}

#[test]
fn the_generated_resolve_writes_the_recorded_resolves_gbuffer_and_runs_only_bins_with_tiles() {
    let _serial = serial();
    if !gpu_suite::require(
        device().caps().generated_commands,
        "device-generated commands",
    ) {
        return;
    }
    let mut copies = showcase();
    copies.push(Item {
        model: "card",
        linear: Mat3::from_rotation_y(0.35) * Mat3::from_rotation_x(-0.2) * 1.3,
        at: Vec3::new(0.2, 0.9, 1.4),
    });
    let eye = Vec3::new(0.3, 2.6, 5.5);
    let target = Vec3::new(0.0, 0.6, 0.0);
    // The G-buffer of a second frame, and the sequences the generated resolve wrote.
    let frames = |generated_resolve| {
        let mut w = World::with(&RendererDesc {
            kernels: &linked::KERNELS,
            generated_resolve,
            ..RendererDesc::default()
        });
        let placed = place(&mut w, &copies);
        render(&mut w, &placed, eye + Vec3::X * 0.1, target, &[], true, 1);
        let g = render(&mut w, &placed, eye, target, &[], false, 2);
        let raster: RasterGpu = read_buffer(g.view.raster, 1)[0];
        let sequences = read_buffer(raster.bin_args.offset(3 * u64::from(raster.bins)), 1)[0];
        (g, raster.bins, sequences)
    };
    let (generated, bins, sequences) = frames(true);
    let (recorded, _, none) = frames(false);
    assert_eq!(bins, linked::KERNELS.surfaces + 1);
    assert_eq!(
        (sequences, none),
        (2, 0),
        "sequences for the standard and the card's bins only"
    );
    let bits = |f: &Frame| {
        let depth: Vec<u32> = f.depth.iter().map(|d| d.to_bits()).collect();
        let motion: Vec<[u32; 2]> = f.motion.iter().map(|m| m.map(f32::to_bits)).collect();
        // The extended targets hold words only where a pixel's flags say so.
        let held = |t: &Option<Vec<[u32; 4]>>, flag: u32| {
            t.as_ref().map(|t| {
                (t.iter().zip(&f.a))
                    .map(|(&w, a)| if (a[3] >> 24) & flag != 0 { w } else { [0; 4] })
                    .collect::<Vec<_>>()
            })
        };
        (
            f.a.clone(),
            f.b.clone(),
            held(&f.x, EXTENDED),
            held(&f.y, LAYERS),
            depth,
            motion,
        )
    };
    assert!(bits(&generated) == bits(&recorded), "the G-buffers differ");
    assert_clean();
}

#[test]
fn a_stochastic_view_changes_magnified_texels_by_frame_and_averages_near_hardware_filtering() {
    const FRAMES: u32 = 32;
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let placed = place(&mut w, &showcase());
    // Close over the ground's checker corner at the origin: about a quarter of an albedo
    // texel per pixel.
    let eye = Vec3::new(0.05, 0.25, 0.05);
    let target = Vec3::ZERO;
    let base = |g: &Frame, i: usize| [0, 8, 16].map(|s| byte(g.a[i][0], s));
    let hardware = render(&mut w, &placed, eye, target, &[], true, 1);
    w.texture_filter = TextureFilter::Stochastic;
    let pixels = hardware.ids.len();
    let mut sum = vec![[0.0f32; 3]; pixels];
    let mut changed = 0;
    let mut last: Option<Frame> = None;
    for frame in 2..2 + FRAMES {
        let g = render(&mut w, &placed, eye, target, &[], false, frame);
        assert_eq!(
            g.ids, hardware.ids,
            "frame {frame}: the filter leaves visibility alone"
        );
        for (i, s) in sum.iter_mut().enumerate() {
            let c = base(&g, i);
            for k in 0..3 {
                s[k] += c[k] / FRAMES as f32;
            }
        }
        if let Some(p) = &last {
            changed += (0..pixels).filter(|&i| base(p, i) != base(&g, i)).count();
        }
        last = Some(g);
    }
    assert!(
        changed > pixels / 100,
        "{changed} texels changed between frames over {pixels} pixels"
    );
    let difference = (0..pixels)
        .map(|i| {
            let h = base(&hardware, i);
            (0..3).map(|k| (sum[i][k] - h[k]).abs()).sum::<f32>() / 3.0
        })
        .sum::<f32>()
        / pixels as f32;
    assert!(
        difference < 0.01,
        "mean difference from hardware {difference}"
    );
    assert_clean();
}

/// Whether any of the four pixels around uv `p` holds `id`.
fn footprint_has(ids: &[u32], p: Vec2, id: u32) -> Option<bool> {
    let [w, h] = EXTENT.map(|e| e as i32);
    let q = p * Vec2::new(w as f32, h as f32) - 0.5;
    let (x0, y0) = (q.x.floor() as i32, q.y.floor() as i32);
    if x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h {
        return None;
    }
    Some(
        [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .any(|&(dx, dy)| ids[((y0 + dy) * w + x0 + dx) as usize] == id),
    )
}

fn uv(i: usize) -> Vec2 {
    let w = EXTENT[0] as usize;
    (Vec2::new((i % w) as f32, (i / w) as f32) + 0.5)
        / Vec2::new(EXTENT[0] as f32, EXTENT[1] as f32)
}

#[test]
fn motion_vectors_reproject_the_previous_frame() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    // Separate objects over the sky, so the only disocclusion is sky they uncover.
    let at = |t: f32| {
        vec![
            Item {
                model: "ball",
                linear: Mat3::from_rotation_y(t),
                at: Vec3::new(-1.7 + 0.3 * t, 0.2 + 0.1 * t, -0.2),
            },
            Item {
                model: "bumps",
                linear: Mat3::from_rotation_x(0.9) * Mat3::from_rotation_y(0.3 * t),
                at: Vec3::new(1.3, 0.9 - 0.2 * t, -1.0),
            },
            Item {
                model: "leaf",
                linear: Mat3::from_rotation_y(0.4 + 0.2 * t) * 0.8,
                at: Vec3::new(1.2, -1.0 + 0.1 * t, 0.3),
            },
            Item {
                model: "plain",
                linear: Mat3::from_rotation_z(0.3 * t),
                at: Vec3::new(-0.6, -1.2, 0.5 - 0.2 * t),
            },
        ]
    };
    let placed = place(&mut w, &at(0.0));
    let target = Vec3::new(0.0, 0.0, -0.5);
    let before = render(
        &mut w,
        &placed,
        Vec3::new(0.0, 0.3, 5.0),
        target,
        &at(0.0),
        true,
        1,
    );
    let eye = Vec3::new(0.25, 0.4, 4.8);
    let after = render(
        &mut w,
        &placed,
        eye,
        target + Vec3::new(0.05, 0.0, 0.0),
        &at(1.0),
        false,
        2,
    );
    // Pixels whose previous position holds another surface, out of those it can be checked
    // for; with `scale` 0 the motion is ignored.
    let mismatch = |scale: f32| {
        let (mut counted, mut wrong) = (0, 0);
        for (i, &id) in after.ids.iter().enumerate() {
            let m = Vec2::from(after.motion[i]) * scale;
            let Some(found) = footprint_has(&before.ids, uv(i) + m, id) else {
                continue;
            };
            if id == 0 && !found {
                continue; // sky uncovered by a moving object
            }
            counted += 1;
            if !found {
                wrong += 1;
            }
        }
        (wrong, counted)
    };
    let (wrong, counted) = mismatch(1.0);
    let (still, _) = mismatch(0.0);
    assert!(
        still * 100 > counted,
        "the scene moves too little to test: {still} of {counted}"
    );
    let covered = after.ids.iter().filter(|&&id| id != 0).count();
    assert!(covered > 2000, "{covered} pixels of geometry");
    assert!(
        wrong * 1000 < counted,
        "{wrong} of {counted} pixels reproject onto another surface"
    );
    assert_clean();
}

#[test]
fn a_change_of_aspect_adds_no_motion() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let placed = place(&mut w, &showcase());
    let eye = Vec3::new(0.3, 2.6, 5.5);
    let target = Vec3::new(0.0, 0.6, 0.0);
    let mut frame = 0;
    let mut motion_at = |extent: [u32; 2], eye: Vec3, camera_cut: bool| {
        frame += 1;
        let list = [placed.scene];
        let mut f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        f.scene(placed.scene).unwrap().set_focus(eye.as_dvec3());
        let view = f
            .raster_view(&ViewDesc {
                scene: placed.scene,
                history: placed.history,
                position: eye.as_dvec3(),
                rotation: look(eye, target),
                projection: Projection {
                    fov_y: 0.9,
                    near: 0.1,
                },
                lens: Lens::default(),
                render_extent: UVec2::from(extent),
                output_extent: UVec2::from(extent),
                jitter: Vec2::new(0.25, -0.125),
                camera_cut,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
            })
            .unwrap();
        wait(f.end(&[]).unwrap());
        texels::<[u16; 2]>(&view.motion)
            .iter()
            .map(|m| Vec2::from(m.map(|bits| half::f16::from_bits(bits).to_f32())))
            .collect::<Vec<_>>()
    };
    const WIDE: [u32; 2] = [320, 180];
    const NARROW: [u32; 2] = [240, 180];
    fn largest(motion: impl IntoIterator<Item = Vec2>) -> f32 {
        motion.into_iter().fold(0.0, |m, v| {
            assert!(v.is_finite(), "motion {v}");
            m.max(v.abs().max_element())
        })
    }

    motion_at(WIDE, eye, true);
    let resized = largest(motion_at(NARROW, eye, false));
    assert!(
        resized < 1e-5,
        "a still camera moves {resized} of the image"
    );

    // A moving camera reports the same motion whether or not the extent changed with it.
    let moved = eye + Vec3::new(0.4, 0.0, 0.0);
    motion_at(NARROW, eye, true);
    let steady = motion_at(NARROW, moved, false);
    motion_at(WIDE, eye, true);
    let across = motion_at(NARROW, moved, false);
    assert!(
        largest(steady.iter().copied()) > 0.01,
        "the camera moves too little to test"
    );
    let apart = largest(steady.iter().zip(&across).map(|(a, b)| *a - *b));
    assert!(
        apart < 1e-3,
        "the resize moves motion by {apart} of the image"
    );
    assert_clean();
}

/// A camera that turns past 90 degrees between frames sees points that were behind it, whose
/// motion runs off the previous image on their own side, not mirrored through its center.
#[test]
fn a_point_behind_the_previous_camera_moves_off_its_own_side() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let placed = place(&mut w, &showcase());
    let eye = DVec3::new(0.0, 50.0, 0.0);
    let shot = |frame, yaw: f64| Shot {
        focus: Some(DVec3::ZERO),
        jitter: Vec2::ZERO,
        ..Shot::at(eye, eye + DVec3::new(-yaw.sin(), 0.0, -yaw.cos()), frame)
    };
    shoot(&mut w, &placed, &shot(1, 0.0));
    // The center now looks along (-sin yaw, 0, -cos yaw) in the previous view: behind it, and
    // to its left.
    let motion = shoot(&mut w, &placed, &shot(2, 120f64.to_radians())).motion;
    assert!(
        motion.iter().flatten().all(|m| m.is_finite()),
        "non-finite motion"
    );
    let center = motion[(EXTENT[1] / 2 * EXTENT[0] + EXTENT[0] / 2) as usize];
    assert!(center[0] < -1.0, "the center moves {center:?}");
    assert_clean();
}

/// The raster error code most pixels of copy `id` carry.
fn error_code(frame: &Frame, id: u32) -> u32 {
    let mut codes: Vec<u32> = frame
        .ids
        .iter()
        .zip(&frame.b)
        .filter(|(i, _)| **i == id)
        .map(|(_, b)| (b[0] >> 21) & 31)
        .collect();
    assert!(codes.len() > 10, "copy {id} covers {} pixels", codes.len());
    codes.sort_unstable();
    codes[codes.len() / 2]
}

#[test]
fn levels_coarsen_with_distance() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let copies = [
        Item {
            model: "bumps",
            linear: Mat3::from_rotation_x(1.2),
            at: Vec3::ZERO,
        },
        Item {
            model: "ball",
            linear: Mat3::IDENTITY,
            at: Vec3::new(0.0, 0.0, -400.0),
        },
    ];
    let placed = place(&mut w, &copies);
    let mut bumps = Vec::new();
    for (frame, d) in [1.6f32, 3.0, 8.0, 30.0].into_iter().enumerate() {
        let eye = Vec3::new(0.0, 0.0, d);
        let g = render(
            &mut w,
            &placed,
            eye,
            Vec3::ZERO,
            &[],
            true,
            frame as u32 + 1,
        );
        bumps.push(error_code(&g, 1));
    }
    assert!(
        bumps.windows(2).all(|p| p[0] < p[1]),
        "discrete levels by distance: codes {bumps:?}"
    );
    assert_eq!(bumps[0], 0, "the finest level has no error");
    let mut ball = Vec::new();
    for (frame, d) in [2.0f32, 5.0, 12.0, 25.0].into_iter().enumerate() {
        let eye = Vec3::new(0.0, 0.0, d - 400.0);
        let g = render(
            &mut w,
            &placed,
            eye,
            Vec3::new(0.0, 0.0, -400.0),
            &[],
            true,
            frame as u32 + 10,
        );
        ball.push(error_code(&g, 2));
    }
    assert!(
        ball.windows(2).all(|p| p[0] <= p[1]) && ball[0] < ball[3],
        "DAG cut by distance: codes {ball:?}"
    );
    assert_clean();
}

#[test]
fn a_linked_surface_offsets_depth_and_motion_like_the_geometry_it_stands_for() {
    let _serial = serial();
    let mut w = World::new(&linked::KERNELS);
    let item = |model: &'static str, t: f32| {
        vec![Item {
            model,
            linear: Mat3::from_rotation_y(0.35 + 0.1 * t) * Mat3::from_rotation_x(-0.2) * 1.3,
            at: Vec3::new(0.1 * t, 0.05 * t, -0.1 * t),
        }]
    };
    let card = place(&mut w, &item("card", 0.0));
    let slab = place(&mut w, &item("slab", 0.0));
    let target = Vec3::new(0.0, 0.0, -0.3);
    let eyes = [Vec3::new(0.2, 0.3, 1.8), Vec3::new(0.3, 0.25, 1.7)];
    let frames = |w: &mut World, placed: &Placed, model| {
        render(w, placed, eyes[0], target, &item(model, 0.0), true, 1);
        render(w, placed, eyes[1], target, &item(model, 1.0), false, 2)
    };
    let a = frames(&mut w, &card, "card");
    let b = frames(&mut w, &slab, "slab");
    let [width, height] = EXTENT.map(|e| e as usize);
    let interior = |f: &Frame, x: usize, y: usize| {
        (y.saturating_sub(1)..=(y + 1).min(height - 1)).all(|j| {
            (x.saturating_sub(1)..=(x + 1).min(width - 1)).all(|i| f.ids[j * width + i] != 0)
        })
    };
    let mut compared = 0usize;
    for y in 0..height {
        for x in 0..width {
            if !interior(&a, x, y) || !interior(&b, x, y) {
                continue;
            }
            let i = y * width + x;
            compared += 1;
            let (da, db) = (a.depth[i], b.depth[i]);
            assert!(
                ((da - db) / db).abs() < 1e-3,
                "pixel {x},{y}: depth {da} for the card, {db} for the slab"
            );
            for (k, &e) in EXTENT.iter().enumerate() {
                let px = (a.motion[i][k] - b.motion[i][k]).abs() * e as f32;
                assert!(px < 0.05, "pixel {x},{y}: motion differs by {px} px");
            }
        }
    }
    let slab_pixels = b.ids.iter().filter(|&&id| id != 0).count();
    assert!(
        compared * 10 > slab_pixels * 8,
        "{compared} of the slab's {slab_pixels} pixels compared"
    );
    assert_clean();
}

#[test]
fn hzb_mips_hold_the_farthest_depth_they_cover() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let placed = place(&mut w, &showcase());
    let g = render(
        &mut w,
        &placed,
        Vec3::new(0.3, 2.6, 5.5),
        Vec3::new(0.0, 0.6, 0.0),
        &[],
        true,
        1,
    );
    let hzb = &g.view.hzb;
    let mips = hzb.desc().mips;
    assert_eq!(mips, 8, "160 x 96 down to 1 x 1");
    let level = |m: u32| -> Vec<f32> {
        bytemuck::pod_collect_to_vec(&read_texture(hzb, &TextureRegion::whole_mip(hzb.desc(), m)))
    };
    let vis: Vec<u64> = read_buffer(g.view.targets.visibility, u64::from(EXTENT[0] * EXTENT[1]));
    let finest: Vec<f32> = vis
        .iter()
        .map(|&v| f32::from_bits((v >> 32) as u32))
        .collect();
    assert_eq!(level(0), finest, "mip 0 is the visibility buffer's depth");
    let mut fine = finest;
    for m in 1..mips {
        let [fw, fh] = EXTENT.map(|e| (e >> (m - 1)).max(1) as usize);
        let [w, h] = EXTENT.map(|e| (e >> m).max(1) as usize);
        let mut want = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                // Three wide on the last row or column of an odd-sized mip.
                let x1 = (2 * x + 1 + usize::from(fw % 2 == 1 && x == w - 1)).min(fw - 1);
                let y1 = (2 * y + 1 + usize::from(fh % 2 == 1 && y == h - 1)).min(fh - 1);
                want[y * w + x] = (2 * y..=y1)
                    .flat_map(|j| (2 * x..=x1).map(move |i| (i, j)))
                    .map(|(i, j)| fine[j * fw + i])
                    .fold(1.0, f32::min);
            }
        }
        assert_eq!(level(m), want, "mip {m}");
        fine = want;
    }
    assert_clean();
}

#[test]
fn load_gsurface_reads_the_visible_surface() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let placed = place(&mut w, &showcase());
    let g = render(
        &mut w,
        &placed,
        Vec3::new(0.3, 2.6, 5.5),
        Vec3::new(0.0, 0.6, 0.0),
        &[],
        true,
        1,
    );
    let d = device();
    let n = u64::from(EXTENT[0] * EXTENT[1]);
    let out = d
        .gpu_malloc::<surface_at::At>(MemoryKind::Readback, n, "surfaces")
        .unwrap();
    let mut root = d
        .gpu_malloc::<surface_at::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    // The view's records stay in the frame's upload window until a later frame reuses it.
    root.as_mut_slice().unwrap()[0] = surface_at::Root {
        view: g.view.view,
        gbuffer: g.view.gbuffer,
        out: out.gpu(),
        pad0: 0,
        pad1: 0,
    };
    let p = pipeline(&surface_at::MODULE, "main");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.dispatch(&p, root.gpu(), [n.div_ceil(64) as u32, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    let vis: Vec<u64> = read_buffer(g.view.targets.visibility, n);
    let clusters = visible_clusters(&g.view);
    for (i, (a, &v)) in out.as_slice().unwrap().iter().zip(&vis).enumerate() {
        if v == 0 {
            assert_eq!(a.material, u32::MAX, "pixel {i}: sky has no material");
            continue;
        }
        let c = clusters[(v as u32 >> 7) as usize];
        assert_eq!(
            (a.instance, a.material),
            (c.instance, c.material),
            "pixel {i}"
        );
        assert_eq!(a.primitive, v as u32 & 127, "pixel {i}");
        assert!(a.nv >= 0.0099, "pixel {i}: N.V {} under the lean", a.nv);
    }
    assert_clean();
}

#[test]
fn masked_sections_draw_nothing() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let ball = [Item {
        model: "ball",
        linear: Mat3::IDENTITY,
        at: Vec3::ZERO,
    }];
    let placed = place(&mut w, &ball);
    let eye = Vec3::new(0.0, 0.0, 3.0);
    let emitting = |g: &Frame| {
        (0..g.a.len())
            .filter(|&i| decode_core(g.a[i], g.b[i]).emission.max_element() > 0.0)
            .count()
    };
    let whole = render(&mut w, &placed, eye, Vec3::ZERO, &[], true, 1);
    assert!(emitting(&whole) > 500, "the lower half emits");
    let list = [placed.scene];
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(2),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    // Section 1 is the emitting lower half.
    let lower = CopyVariant {
        pose: 0,
        material_set: 0,
        section_mask: 1 << 1,
    };
    f.scene(placed.scene)
        .unwrap()
        .copy_variants(placed.ranges[0], 0, &[lower])
        .unwrap();
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    let masked = render(&mut w, &placed, eye, Vec3::ZERO, &[], true, 3);
    assert_eq!(emitting(&masked), 0, "the masked lower half is gone");
    let drawn = |g: &Frame| g.ids.iter().filter(|&&id| id != 0).count();
    assert!(
        drawn(&masked) * 3 > drawn(&whole) && drawn(&masked) * 3 < 2 * drawn(&whole),
        "about half the sphere stays: {} of {} pixels",
        drawn(&masked),
        drawn(&whole)
    );
    assert_clean();
}

/// Per pixel: the visible (instance, mesh cluster, triangle) and its depth.
type Surfaces = Vec<Option<((u32, u32, u32), f32)>>;

/// The view's visibility buffer decoded, and its counts.
fn surfaces(view: &RasterView) -> (Surfaces, Vec<u32>) {
    let vis: Vec<u64> = read_buffer(view.targets.visibility, u64::from(EXTENT[0] * EXTENT[1]));
    let counts: Vec<u32> = read_buffer(view.counts, 8);
    let clusters = visible_clusters(view);
    let decoded = vis
        .iter()
        .map(|&w| {
            (w != 0).then(|| {
                let c = clusters[(w as u32 >> 7) as usize];
                let depth = f32::from_bits((w >> 32) as u32);
                ((c.instance, c.cluster, w as u32 & 127), depth)
            })
        })
        .collect();
    (decoded, counts)
}

#[test]
fn software_and_hardware_raster_draw_the_same_visibility_buffer() {
    let _serial = serial();
    let shots = [
        (Vec3::new(0.3, 2.6, 5.5), Vec3::new(0.0, 0.6, 0.0)),
        (Vec3::new(-1.0, 1.2, 2.2), Vec3::new(1.2, 0.9, 0.4)),
        (Vec3::new(1.5, 5.0, 14.0), Vec3::new(0.0, 0.6, 0.0)),
    ];
    let run = |software_raster_pixels: f32| {
        let mut w = World::with(&RendererDesc {
            software_raster_pixels,
            ..RendererDesc::default()
        });
        let placed = place(&mut w, &showcase());
        shots
            .iter()
            .enumerate()
            .map(|(i, &(eye, target))| {
                let list = [placed.scene];
                let mut f = w
                    .scenes
                    .begin_frame(&FrameDesc {
                        time: time(i as u32),
                        scenes: &list,
                        low_latency: None,
                    })
                    .unwrap();
                {
                    let mut w = f.scene(placed.scene).unwrap();
                    w.set_focus(eye.as_dvec3());
                    f.request_detail(placed.scene, VIEW_PIXELS_PER_RADIAN)
                        .unwrap();
                }
                let view = f
                    .raster_view(&ViewDesc {
                        scene: placed.scene,
                        history: placed.history,
                        position: eye.as_dvec3(),
                        rotation: look(eye, target),
                        projection: Projection {
                            fov_y: 0.9,
                            near: 0.1,
                        },
                        lens: Lens::default(),
                        render_extent: UVec2::from(EXTENT),
                        output_extent: UVec2::from(EXTENT),
                        jitter: Vec2::new(0.25, -0.125),
                        camera_cut: true,
                        mip_bias: 0.0,
                        texture_filter: TextureFilter::Hardware,
                    })
                    .unwrap();
                wait(f.end(&[]).unwrap());
                surfaces(&view)
            })
            .collect::<Vec<_>>()
    };
    let hardware = run(0.0);
    let software = run(1e9);
    let [w, h] = EXTENT.map(|v| v as usize);
    for (shot, ((hw, hw_counts), (sw, sw_counts))) in hardware.iter().zip(&software).enumerate() {
        assert_eq!(
            hw_counts[7], 0,
            "shot {shot}: no software clusters when it is off"
        );
        assert_eq!(
            hw_counts[0], sw_counts[0],
            "shot {shot}: the same clusters are visible"
        );
        assert!(
            sw_counts[7] > 0,
            "shot {shot}: some clusters raster in software"
        );
        if shot == 2 {
            assert_eq!(
                sw_counts[7], sw_counts[0],
                "far away, every cluster is software"
            );
        }
        let id = |s: &Surfaces, x: usize, y: usize| s[y * w + x].map(|(id, _)| id);
        let mut covered = 0;
        let mut differ = 0;
        let mut depth_error = 0f32;
        for y in 0..h {
            for x in 0..w {
                let (a, b) = (hw[y * w + x], sw[y * w + x]);
                covered += usize::from(a.is_some() || b.is_some());
                match (a, b) {
                    (Some((ia, da)), Some((ib, db))) if ia == ib => {
                        depth_error = depth_error.max((da - db).abs() / da);
                        continue;
                    }
                    (None, None) => continue,
                    _ => differ += 1,
                }
                // A tie on an edge: the other path drew a neighbor's surface here.
                let near = |s: &Surfaces, want: Option<(u32, u32, u32)>| {
                    [(0, 1), (2, 1), (1, 0), (1, 2)].iter().any(|&(dx, dy)| {
                        let (nx, ny) = ((x + dx).wrapping_sub(1), (y + dy).wrapping_sub(1));
                        nx < w && ny < h && id(s, nx, ny) == want
                    })
                };
                assert!(
                    near(hw, id(sw, x, y)) && near(sw, id(hw, x, y)),
                    "shot {shot}: pixel ({x}, {y}) differs away from an edge: {a:?} against {b:?}"
                );
            }
        }
        assert!(
            covered > w * h / 100,
            "shot {shot}: {covered} pixels covered"
        );
        assert!(
            differ * 200 < covered,
            "shot {shot}: {differ} of {covered} pixels differ"
        );
        // Both snap corners to 1/256 pixel; the hardware's depth plane differs from the
        // software's by a few thousandths of a pixel's depth slope on sub-pixel triangles.
        assert!(
            depth_error < 1e-4,
            "shot {shot}: depth differs by {depth_error:e}"
        );
    }
    assert_clean();
}

/// Frames whose clusters split between the two rasters cover the pixels the hardware raster
/// alone covers: both paths draw a corner at one grid point, so no pixel opens along an edge a
/// software cluster shares with a hardware one. The views are fine enough for clusters on both
/// sides of the software limit, and among them are ones where the hardware's own snapping of
/// the corners left such pixels open.
#[test]
fn software_and_hardware_clusters_share_edges_without_open_pixels() {
    let _serial = serial();
    const E: [u32; 2] = [1920, 1080];
    const JITTERS: u32 = 10;
    let detail = 4.0 * 0.5 * E[1] as f32 / 0.483_055_07;
    let target = Vec3::new(0.0, -1.0, -12.0);
    let eyes = [Vec3::new(2.0, 6.0, 20.0), Vec3::new(-4.0, 10.0, 40.0)];
    // Per view: a bit per pixel it covered, and its counts.
    let run = |software_raster_pixels: f32| -> Vec<(Vec<u64>, Vec<u32>)> {
        let mut w = World::with(&RendererDesc {
            software_raster_pixels,
            ..RendererDesc::default()
        });
        let placed = place(&mut w, &dag_items());
        let mut out = Vec::new();
        let mut frame = 0;
        for eye in eyes {
            for j in 0..JITTERS {
                let jitter = Vec2::new(
                    (j as f32 * 0.754_877_7).fract() - 0.5,
                    (j as f32 * 0.569_840_3).fract() - 0.5,
                );
                // The detail settles over the eye's first frames; the second frame of a jitter
                // tests the first's depth.
                let mut view = None;
                for _ in 0..if j == 0 { 4 } else { 2 } {
                    frame += 1;
                    let list = [placed.scene];
                    let mut f = w
                        .scenes
                        .begin_frame(&FrameDesc {
                            time: time(frame),
                            scenes: &list,
                            low_latency: None,
                        })
                        .unwrap();
                    f.scene(placed.scene).unwrap().set_focus(eye.as_dvec3());
                    f.request_detail(placed.scene, detail).unwrap();
                    view = Some(
                        f.raster_view(&ViewDesc {
                            scene: placed.scene,
                            history: placed.history,
                            position: eye.as_dvec3(),
                            rotation: look(eye, target),
                            projection: Projection {
                                fov_y: 0.9,
                                near: 0.1,
                            },
                            lens: Lens::default(),
                            render_extent: UVec2::from(E),
                            output_extent: UVec2::from(E),
                            jitter,
                            camera_cut: true,
                            mip_bias: 0.0,
                            texture_filter: TextureFilter::Hardware,
                        })
                        .unwrap(),
                    );
                    wait(f.end(&[]).unwrap());
                }
                let view = view.unwrap();
                let words: Vec<u64> = read_buffer(view.targets.visibility, u64::from(E[0] * E[1]));
                let mut covered = vec![0u64; words.len().div_ceil(64)];
                for (p, _) in words.iter().enumerate().filter(|(_, w)| **w != 0) {
                    covered[p / 64] |= 1 << (p % 64);
                }
                out.push((covered, read_buffer(view.counts, 8)));
            }
        }
        out
    };
    let hardware = run(0.0);
    let mixed = run(64.0);
    let (mut software, mut visible) = (0, 0);
    for (i, ((hw, hw_counts), (mx, counts))) in hardware.iter().zip(&mixed).enumerate() {
        assert_eq!(
            hw_counts[..3],
            counts[..3],
            "view {i}: the same clusters are visible"
        );
        assert_eq!((counts[1], counts[2]), (0, 0), "view {i}: nothing dropped");
        software += counts[7];
        visible += counts[0];
        let open: u32 = hw.iter().zip(mx).map(|(h, m)| (h & !m).count_ones()).sum();
        let more: u32 = hw.iter().zip(mx).map(|(h, m)| (m & !h).count_ones()).sum();
        assert_eq!(
            (open, more),
            (0, 0),
            "view {i}: pixels only the hardware frame covers, and only the mixed one"
        );
    }
    assert!(
        8 * software > visible && 8 * software < 7 * visible,
        "{software} of {visible} clusters raster in software"
    );
    assert_clean();
}

// --- Ray tracing ------------------------------------------------------------------------------

/// The focus's pixels per radian: render height over the vertical field of view.
const PIXELS_PER_RADIAN: f32 = EXTENT[1] as f32 / 0.9;

/// One frame of a placed scene: the camera, the scene focus and the copies moved.
struct Shot {
    eye: DVec3,
    target: DVec3,
    focus: Option<DVec3>,
    moved: Vec<(u32, Affine3)>,
    reset: bool,
    frame: u32,
    jitter: Vec2,
    mip_bias: f32,
}

impl Shot {
    fn at(eye: DVec3, target: DVec3, frame: u32) -> Self {
        Self {
            eye,
            target,
            focus: Some(eye),
            moved: Vec::new(),
            reset: frame <= 1,
            frame,
            jitter: Vec2::new(0.25, -0.125),
            mip_bias: 0.0,
        }
    }
}

fn shoot(w: &mut World, placed: &Placed, shot: &Shot) -> Frame {
    shoot_with(w, placed, shot, |_| {})
}

/// [`shoot`] with `write` run on the scene's writer after the shot's moves.
fn shoot_with(
    w: &mut World,
    placed: &Placed,
    shot: &Shot,
    write: impl FnOnce(&mut SceneWriter<'_>),
) -> Frame {
    shoot_computed(w, placed, shot, write, |_| {})
}

/// [`shoot_with`] with `compute` run in scene compute.
fn shoot_computed(
    w: &mut World,
    placed: &Placed,
    shot: &Shot,
    write: impl FnOnce(&mut SceneWriter<'_>),
    compute: impl FnMut(&mut SceneCompute<'_, '_>),
) -> Frame {
    shoot_probed(w, placed, shot, write, compute, &[]).0
}

/// [`shoot_computed`] with pixel probes of `pixels`, read once the frame completes.
fn shoot_probed(
    w: &mut World,
    placed: &Placed,
    shot: &Shot,
    write: impl FnOnce(&mut SceneWriter<'_>),
    mut compute: impl FnMut(&mut SceneCompute<'_, '_>),
    pixels: &[[u32; 2]],
) -> (Frame, Vec<PixelProbe>) {
    let list = [placed.scene];
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(shot.frame),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(placed.scene).unwrap();
        if let Some(focus) = shot.focus {
            s.set_focus(focus);
            f.request_detail(placed.scene, PIXELS_PER_RADIAN).unwrap();
        }
        for &(i, t) in &shot.moved {
            s.transforms(placed.group, i, 1).unwrap()[0] = t.into();
        }
        write(&mut s);
    }
    f.update_scenes(&mut compute).unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: placed.scene,
            history: placed.history,
            position: shot.eye,
            rotation: look(Vec3::ZERO, (shot.target - shot.eye).as_vec3()),
            projection: Projection {
                fov_y: 0.9,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: shot.jitter,
            camera_cut: shot.reset,
            mip_bias: shot.mip_bias,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
    for &pixel in pixels {
        f.probe(&view, pixel).unwrap();
    }
    let id = f.id();
    wait(f.end(&[]).unwrap());
    let probes = w.scenes.probes(placed.history, id);
    (read_back(&view), probes)
}

/// What the `trace` kernels wrote for one view.
struct Rays {
    traced: Vec<trace::Traced>,
    seen: Vec<trace::Seen>,
    cones: Vec<trace::Cone>,
}

/// Runs `entry` of `module` (a `trace` build) over every pixel of `view`.
fn rays(view: &RasterView, module: &'static ShaderModule, entry: &str, mode: u32) -> Rays {
    let later = rays_later(view, module, entry, mode);
    wait(later.point);
    later.read()
}

/// [`rays`] submitted after the work submitted so far, read once `point` has completed.
struct RaysLater {
    traced: orr_gpu::Allocation<trace::Traced>,
    seen: orr_gpu::Allocation<trace::Seen>,
    cones: orr_gpu::Allocation<trace::Cone>,
    _root: orr_gpu::Allocation<trace::Root>,
    point: orr_gpu::TimelinePoint,
    /// The root of the view's scene.
    scene: GpuPtr<SceneRoot>,
}

impl RaysLater {
    fn read(&self) -> Rays {
        Rays {
            traced: self.traced.as_slice().unwrap().to_vec(),
            seen: self.seen.as_slice().unwrap().to_vec(),
            cones: self.cones.as_slice().unwrap().to_vec(),
        }
    }
}

/// [`rays`] without waiting for them.
fn rays_later(
    view: &RasterView,
    module: &'static ShaderModule,
    entry: &str,
    mode: u32,
) -> RaysLater {
    let d = device();
    let n = u64::from(EXTENT[0] * EXTENT[1]);
    let traced = d
        .gpu_malloc::<trace::Traced>(MemoryKind::Readback, n, "traced")
        .unwrap();
    let seen = d
        .gpu_malloc::<trace::Seen>(MemoryKind::Readback, n, "seen")
        .unwrap();
    let cones = d
        .gpu_malloc::<trace::Cone>(MemoryKind::Readback, n, "cones")
        .unwrap();
    let mut root = d
        .gpu_malloc::<trace::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = trace::Root {
        scene: view.scene,
        view: view.view,
        gbuffer: view.gbuffer,
        traced: traced.gpu(),
        seen: seen.gpu(),
        cones: cones.gpu(),
        mode,
        pad0: 0,
        pad1: 0,
    };
    let p = pipeline(module, entry);
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
    list.dispatch(&p, root.gpu(), [n.div_ceil(64) as u32, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = render.submit(vec![list], &[]).unwrap();
    RaysLater {
        traced,
        seen,
        cones,
        _root: root,
        point,
        scene: view.scene,
    }
}

/// Pixels where the camera ray and the raster agree on the instance and, where something
/// was hit, on the depth within `depth` relative; and the pixels the raster covered.
fn parity(frame: &Frame, rays: &Rays, depth: f32) -> (usize, usize) {
    let mut matched = 0;
    for (i, t) in rays.traced.iter().enumerate() {
        let id = frame.ids[i];
        let z = frame.depth[i];
        if t.instance == id && (id == 0 || (t.depth - z).abs() <= depth * z) {
            matched += 1;
        }
    }
    (matched, frame.ids.iter().filter(|&&id| id != 0).count())
}

/// A second instance set for parity: scaled balls, turned leaves and a tilted ground.
fn crowd() -> Vec<Item> {
    let mut items = vec![
        Item {
            model: "ground",
            linear: Mat3::from_rotation_y(0.3),
            at: Vec3::new(0.5, -0.2, 0.0),
        },
        Item {
            model: "bumps",
            linear: Mat3::from_rotation_y(-0.7) * Mat3::from_rotation_z(0.2),
            at: Vec3::new(-1.0, 0.3, -1.5),
        },
    ];
    for i in 0..5 {
        let f = i as f32;
        items.push(Item {
            model: "ball",
            linear: Mat3::from_diagonal(Vec3::splat(0.35 + 0.1 * f)),
            at: Vec3::new(-2.0 + f, 0.6, -0.5 - 0.4 * f),
        });
    }
    for i in 0..3 {
        let f = i as f32;
        items.push(Item {
            model: "leaf",
            linear: Mat3::from_rotation_y(0.5 * f - 0.4) * Mat3::from_rotation_x(-0.4),
            at: Vec3::new(-1.2 + 1.3 * f, 1.3, 0.9),
        });
    }
    items
}

/// A world whose raster and ray tracing both draw every mesh at its finest geometry.
fn finest_world() -> World {
    World::with(&RendererDesc {
        raster_error_pixels: 1e-3,
        rt_error_pixels: 1e-3,
        ..RendererDesc::default()
    })
}

#[test]
fn camera_rays_hit_what_the_raster_drew() {
    let _serial = serial();
    let mut w = finest_world();
    let views = [
        (
            showcase(),
            DVec3::new(0.3, 2.6, 5.5),
            DVec3::new(0.0, 0.6, 0.0),
        ),
        (
            crowd(),
            DVec3::new(-0.5, 2.2, 4.5),
            DVec3::new(0.0, 0.5, -0.5),
        ),
    ];
    for (items, eye, target) in &views {
        let placed = place(&mut w, items);
        let g = shoot(&mut w, &placed, &Shot::at(*eye, *target, 1));
        let r = rays(&g.view, &trace::MODULE, "pixels", 0);
        let (matched, covered) = parity(&g, &r, 1e-4);
        let total = r.traced.len();
        assert!(covered > total / 3, "{covered} of {total} pixels covered");
        assert!(
            matched as f64 >= 0.999 * total as f64,
            "{matched} of {total} pixels agree"
        );
    }
    assert_clean();
}

/// A mesh's resident pages decoded on the CPU from the pack.
struct CpuMesh {
    header: orr_pack::mesh::MeshHeader,
    pages: Vec<Vec<u8>>,
    firsts: Vec<u32>,
    groups: Vec<Group>,
    /// Per page, the pages it depends on.
    deps: Vec<Vec<u32>>,
    /// Its node pages; one for a resident hierarchy.
    node_pages: usize,
}

fn blob(a: &Archive, e: &Entry, i: usize) -> Vec<u8> {
    let b = a.blobs(e)[i];
    let mut out = vec![0; b.raw_len as usize];
    a.read(&b, &mut out).unwrap();
    out
}

impl CpuMesh {
    fn of(model: &str) -> Self {
        Self::of_pack(pack(), model)
    }

    fn of_pack(bytes: &Arc<[u8]>, model: &str) -> Self {
        let a = Archive::from_memory(Arc::clone(bytes)).unwrap();
        let entry = a.find(AssetId::from_name(model)).unwrap();
        let info = ModelInfo::parse(&blob(&a, entry, 0)).unwrap();
        let me = a.find(info.header.mesh).unwrap();
        let mi = MeshInfo::parse(&blob(&a, me, 0)).unwrap();
        let mut first = 0;
        let mut firsts = Vec::new();
        let pages = mi
            .pages
            .iter()
            .map(|p| {
                firsts.push(first);
                first += u32::from(p.cluster_count);
                let mut page = vec![0; p.resident_len as usize];
                decode_page(&blob(&a, me, p.blob as usize), &mut page).unwrap();
                page
            })
            .collect();
        let deps = mi
            .pages
            .iter()
            .map(|p| mi.page_deps[p.deps.first as usize..][..p.deps.count as usize].to_vec())
            .collect();
        Self {
            header: mi.header,
            pages,
            firsts,
            groups: mi.groups,
            deps,
            node_pages: mi.node_pages.len().max(1),
        }
    }

    /// Positions, normals and uvs of triangle `t` of mesh cluster `cluster`, and whether it
    /// is alpha-tested.
    fn triangle(&self, cluster: u32, t: u32) -> ([Vec3; 3], [Vec3; 3], [Vec2; 3], bool) {
        let p = self.firsts.partition_point(|&f| f <= cluster) - 1;
        let view = PageView::new(&self.pages[p], &self.header).unwrap();
        let c = view.cluster(cluster - self.firsts[p]);
        let k = view.triangle(&c, t).map(u32::from);
        (
            k.map(|v| Vec3::from(view.position(&c, v))),
            k.map(|v| Vec3::from(view.normal(&c, v))),
            k.map(|v| Vec2::from(view.uv(&c, v, 0))),
            t >= u32::from(c.alpha_first),
        )
    }
}

#[test]
fn resolved_hits_interpolate_their_triangle_like_the_cpu() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let items = showcase();
    let placed = place(&mut w, &items);
    let eye = DVec3::new(0.3, 2.6, 5.5);
    let mut shot = Shot::at(eye, DVec3::new(0.0, 0.6, 0.0), 1);
    shot.focus = None;
    let g = shoot(&mut w, &placed, &shot);
    let r = rays(&g.view, &trace::MODULE, "pixels", 1);
    let meshes: Vec<CpuMesh> = items.iter().map(|i| CpuMesh::of(i.model)).collect();
    let (mut alpha, mut opaque) = (0, 0);
    for (i, t) in r.traced.iter().enumerate() {
        if t.instance == 0 {
            continue;
        }
        let k = placed
            .ranges
            .iter()
            .position(|r| r.first == t.instance - 1)
            .unwrap();
        let item = &items[k];
        let (p, n, uv, alpha_tested) = meshes[k].triangle(t.cluster, t.primitive);
        let b = Vec2::from(t.barycentrics);
        let wt = [1.0 - b.x - b.y, b.x, b.y];
        let lerp3 = |v: [Vec3; 3]| v[0] * wt[0] + v[1] * wt[1] + v[2] * wt[2];
        let position = item.linear * lerp3(p) + item.at;
        let mut ng = (item.linear * (p[1] - p[0]))
            .cross(item.linear * (p[2] - p[0]))
            .normalize();
        let mut normal = (item.linear.inverse().transpose() * lerp3(n).normalize()).normalize();
        let d = (position - eye.as_vec3()).normalize();
        if item.model == "leaf" && ng.dot(d) > 0.0 {
            ng = -ng;
            normal = -normal;
        }
        let uv = uv[0] * wt[0] + uv[1] * wt[1] + uv[2] * wt[2];
        assert_eq!(t.valid, 1, "pixel {i}");
        let got = Vec3::from(t.position);
        assert!(
            got.distance(position) < 1e-4,
            "pixel {i} ({}): position {got} want {position}",
            item.model
        );
        // The ground's shading normal comes from its normal map, not the vertex normals.
        assert!(
            item.model == "ground" || Vec3::from(t.normal).dot(normal) > 1.0 - 1e-5,
            "pixel {i} ({}): normal {:?} want {normal}",
            item.model,
            t.normal
        );
        assert!(
            Vec3::from(t.geometric_normal).dot(ng) > 1.0 - 1e-5,
            "pixel {i}: geometric normal"
        );
        assert!(
            Vec2::from(t.uv).distance(uv) < 1e-4,
            "pixel {i}: uv {:?} want {uv}",
            t.uv
        );
        if alpha_tested {
            alpha += 1;
        } else {
            opaque += 1;
        }
    }
    assert!(
        alpha > 50 && opaque > 2000,
        "{alpha} alpha-tested, {opaque} opaque hits"
    );
    assert_clean();
}

#[test]
fn a_surface_that_ignores_its_context_sees_one_point_the_same_from_every_ray() {
    let _serial = serial();
    let mut w = World::new(&linked::KERNELS);
    let items = [
        Item {
            model: "steady ground",
            linear: Mat3::IDENTITY,
            at: Vec3::ZERO,
        },
        Item {
            model: "steady ball",
            linear: Mat3::from_rotation_y(0.7),
            at: Vec3::new(0.2, 1.0, 0.0),
        },
    ];
    let placed = place(&mut w, &items);
    let g = shoot(
        &mut w,
        &placed,
        &Shot::at(DVec3::new(0.0, 2.5, 4.5), DVec3::new(0.0, 0.5, 0.0), 1),
    );
    let module = linked::KERNELS.module(&trace::MODULE).unwrap();
    let r = rays(&g.view, module, "same_point", 0);
    let wrap = |a: f32, b: f32| {
        let d = (a - b).abs();
        d.min(1.0 - d)
    };
    let mut compared = 0usize;
    for (i, s) in r.seen.iter().enumerate() {
        // Rays through a shared edge may take either triangle.
        let same = |k: usize| s.material[k] == s.material[0] && s.triangle[k] == s.triangle[0];
        if s.hit != 7 || !same(1) || !same(2) {
            continue;
        }
        let p0 = Vec3::from(s.position[0]);
        compared += 1;
        for k in 1..3 {
            let p = Vec3::from(s.position[k]);
            assert!(p.distance(p0) < 2e-4, "pixel {i} ray {k}: {p} want {p0}");
            let n = Vec3::from(s.normal[k]);
            assert!(
                n.dot(Vec3::from(s.normal[0])) > 1.0 - 1e-5,
                "pixel {i} ray {k}: normal"
            );
            let ng = Vec3::from(s.geometric_normal[k]);
            assert!(
                ng.dot(Vec3::from(s.geometric_normal[0])) > 1.0 - 1e-5,
                "pixel {i} ray {k}: geometric normal"
            );
            for c in 0..3 {
                let (a, b) = (s.base_color[k][c], s.base_color[0][c]);
                assert!(wrap(a, b) < 1e-3, "pixel {i} ray {k}: color {a} want {b}");
            }
            assert!(
                (s.roughness[k] - s.roughness[0]).abs() < 1e-4,
                "pixel {i} ray {k}"
            );
        }
    }
    let covered = g.ids.iter().filter(|&&id| id != 0).count();
    assert!(
        compared as f64 > 0.9 * covered as f64,
        "{compared} of {covered} covered pixels seen by all three rays"
    );
    assert_clean();
}

/// The view's mip bias reaches the texture footprint of the material resolve and of a camera
/// ray's first hit alike, and leaves other rays' hits alone.
#[test]
fn mip_bias_scales_the_footprints_of_primary_hits_alone() {
    let _serial = serial();
    let mut w = World::new(&linked::KERNELS);
    let items = [Item {
        model: "steady ground",
        linear: Mat3::IDENTITY,
        at: Vec3::ZERO,
    }];
    let placed = place(&mut w, &items);
    let module = linked::KERNELS.module(&trace::MODULE).unwrap();
    let mut seen = |frame, mip_bias| {
        let shot = Shot {
            mip_bias,
            ..Shot::at(DVec3::new(0.0, 2.5, 4.5), DVec3::new(0.0, 0.5, 0.0), frame)
        };
        let g = shoot(&mut w, &placed, &shot);
        rays(&g.view, module, "same_point", 0).seen
    };
    let plain = seen(1, 0.0);
    let bias = -2.0f32;
    let biased = seen(2, bias);
    let primary = 4.0f32.powf(bias);
    let mut compared = 0usize;
    for (a, b) in plain.iter().zip(&biased) {
        if a.hit != 7 || b.hit != 7 || a.triangle != b.triangle || a.footprint.contains(&0.0) {
            continue;
        }
        compared += 1;
        for (k, want) in [primary, primary, 1.0].into_iter().enumerate() {
            let ratio = b.footprint[k] / a.footprint[k];
            assert!(
                (ratio / want - 1.0).abs() < 0.01,
                "ray {k}: footprint scaled by {ratio}, want {want}"
            );
        }
    }
    assert!(compared > 1000, "{compared} pixels compared");
    assert_clean();
}

#[test]
fn hit_contexts_report_the_kind_depth_and_cone_of_each_hit() {
    let _serial = serial();
    let mut w = World::new(&linked::KERNELS);
    let wall_z = -4.0f32;
    let items = [
        Item {
            model: "probe",
            linear: Mat3::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            at: Vec3::ZERO,
        },
        Item {
            model: "probe",
            linear: Mat3::IDENTITY,
            at: Vec3::new(0.0, 0.0, wall_z),
        },
    ];
    let placed = place(&mut w, &items);
    let eye = Vec3::new(0.0, 1.5, 3.0);
    let g = shoot(
        &mut w,
        &placed,
        &Shot::at(eye.as_dvec3(), DVec3::new(0.0, 0.0, -1.0), 1),
    );
    let module = linked::KERNELS.module(&trace::MODULE).unwrap();
    let r = rays(&g.view, module, "cones", 0);
    let spread = 2.0 * 0.45f32.tan() / EXTENT[1] as f32;
    let close = |a: f32, b: f32, rel: f32| (a - b).abs() <= rel * b.abs().max(1e-6);
    let mut checked = 0;
    for (i, c) in r.cones.iter().enumerate() {
        // The CPU's own rays: the floor, then the mirror ray to the wall.
        let d = Vec3::from(c.direction);
        if d.y >= -1e-3 {
            continue;
        }
        let t1 = eye.y / -d.y;
        let p1 = eye + d * t1;
        if p1.z <= wall_z + 0.05 || p1.x.abs() > 5.9 || p1.z > 5.9 {
            continue;
        }
        let o = p1 + Vec3::Y * 1e-3;
        let r2 = Vec3::new(d.x, -d.y, d.z);
        let t2 = (wall_z - o.z) / r2.z;
        let p2 = o + r2 * t2;
        if r2.z >= 0.0 || p2.y > 5.9 || p2.x.abs() > 5.9 {
            continue;
        }
        assert_eq!(c.hits, 2, "pixel {i}: floor then wall");
        assert!(close(c.t[0], t1, 1e-4), "pixel {i}: t {} want {t1}", c.t[0]);
        assert!(close(c.t[1], t2, 1e-3), "pixel {i}: t {} want {t2}", c.t[1]);
        let want = [
            [0.0, 0.0, spread * c.t[0], spread],
            [1.0, 1.0, spread * (c.t[0] + c.t[1]), spread],
        ];
        for (k, w) in want.iter().enumerate() {
            let got = c.context[k];
            assert_eq!(
                [got[0], got[1]],
                [w[0], w[1]],
                "pixel {i} hit {k}: kind, depth"
            );
            assert!(
                close(got[2], w[2], 1e-4),
                "pixel {i} hit {k}: width {}",
                got[2]
            );
            assert!(
                close(got[3], w[3], 1e-5),
                "pixel {i} hit {k}: spread {}",
                got[3]
            );
        }
        checked += 1;
    }
    assert!(checked > 2000, "{checked} pixels checked");
    assert_clean();
}

/// Writes the decal test's two decals: a matte decal larger than a bin under a sticker.
fn set_decals(s: &mut SceneWriter<'_>, matte: MaterialId, sticker: MaterialId, wall_z: f64) {
    let decal = |center: DVec3, half_extent: Vec3, rotation: Quat, material, channels| Decal {
        center,
        rotation,
        half_extent,
        material,
        channels,
        opacity: 1.0,
        fade_angles: [1.0, 1.4],
        depth_fade: 0.0,
    };
    let under = decal(
        DVec3::new(0.0, 2.0, wall_z),
        Vec3::new(20.0, 20.0, 0.5),
        Quat::IDENTITY,
        matte,
        DecalChannels::ROUGHNESS,
    );
    let over = decal(
        DVec3::new(0.4, 1.6, wall_z),
        Vec3::new(1.2, 1.2, 0.25),
        Quat::from_rotation_z(0.3),
        sticker,
        DecalChannels::BASE_COLOR | DecalChannels::ROUGHNESS,
    );
    s.set_decal(DecalId(0), Some(&under)).unwrap();
    s.set_decal(DecalId(1), Some(&over)).unwrap();
}

/// MikkTSpace's pixel transform, which bakers invert: each corner's normal and tangent placed
/// and normalized, then interpolated and used as they stand. A mirror negates the sign.
#[test]
fn normal_maps_shade_through_the_interpolated_frame_they_were_baked_in() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let cpu = CpuMesh::of("tilted skew");
    for linear in [
        Mat3::IDENTITY,
        Mat3::from_diagonal(Vec3::new(-1.5, 1.0, 2.0)),
    ] {
        let item = Item {
            model: "tilted skew",
            linear,
            at: Vec3::ZERO,
        };
        let placed = place(&mut w, &[item]);
        let mut shot = Shot::at(DVec3::new(0.0, 0.0, 3.0), DVec3::ZERO, 1);
        shot.focus = None;
        let g = shoot(&mut w, &placed, &shot);
        let r = rays(&g.view, &trace::MODULE, "pixels", 1);
        let mirror = linear.determinant().signum();
        let normals = linear.inverse().transpose();
        let mut seen = 0;
        for t in r.traced.iter().filter(|t| t.instance != 0) {
            let (p, ..) = cpu.triangle(t.cluster, t.primitive);
            let corner = p.map(|p| {
                let near = |c: &&(_, _, _, _)| Vec3::from(c.0).distance(p) < 1e-3;
                SKEW.iter().find(near).unwrap()
            });
            let b = Vec2::from(t.barycentrics);
            let wt = [1.0 - b.x - b.y, b.x, b.y];
            let sum = |f: &dyn Fn(usize) -> Vec3| (0..3).map(|k| f(k) * wt[k]).sum::<Vec3>();
            let n = sum(&|k| (normals * Vec3::from(corner[k].1).normalize()).normalize());
            let tangent = sum(&|k| (linear * Vec3::from(corner[k].2)).normalize());
            let bitangent = n.cross(tangent) * corner[0].3 * mirror;
            let [x, y, z] = TILT;
            let want = (tangent * x + bitangent * y + n * z).normalize();
            let got = Vec3::from(t.normal);
            assert!(got.dot(want) > 0.9999, "{got} want {want}");
            seen += 1;
        }
        assert!(seen > 1000, "{seen} pixels");
    }
}

/// Asserts that every drawn pixel of `g` whose normal `pick` keeps is `want`, and counts them.
fn tilted_pixels(g: &Frame, want: Vec3, pick: impl Fn(Vec3) -> bool) -> usize {
    let mut seen = 0;
    for i in 0..g.ids.len() {
        let n = decode_core(g.a[i], g.b[i]).normal;
        if g.ids[i] != 0 && pick(n) {
            assert!(
                n.dot(want.normalize()) > 0.9995,
                "pixel {i}: {n} want {want}"
            );
            seen += 1;
        }
    }
    seen
}

/// A copy's material can read a normal map on the uv set the mesh's tangents don't follow, which
/// turns the frame with that set's uvs.
#[test]
fn a_replaced_material_turns_its_normal_map_with_its_own_uv_set() {
    let _serial = serial();
    let mut w = World::new(&linked::KERNELS);
    let material = w
        .scenes
        .material(w.mount, AssetId::from_name("tilted on uv1/material/0"))
        .unwrap();
    let [x, y, z] = TILT;
    let item = Item {
        model: "tilted sets",
        linear: Mat3::IDENTITY,
        at: Vec3::ZERO,
    };
    let eye = DVec3::new(0.0, 0.0, 2.0);
    for (over, want) in [
        (None, Vec3::new(x, y, z)),
        (Some(material), Vec3::new(y, -x, z)),
    ] {
        let placed = place_with(&mut w, &[item], DVec3::ZERO, 100.0, over, None);
        let mut shot = Shot::at(eye, DVec3::ZERO, 1);
        shot.jitter = Vec2::ZERO;
        let g = shoot(&mut w, &placed, &shot);
        assert!(tilted_pixels(&g, want, |_| true) > 1000);
    }
}

/// A quad seen from its front, u along +x and uv's v down -y: x of the normal map turns along
/// its u axis and y up the image, with tangents from the triangle, MikkTSpace or the caller,
/// under a mirror or a turn, and on a decal.
#[test]
fn normal_maps_turn_toward_the_image_up_on_meshes_and_decals() {
    let _serial = serial();
    let mut w = World::new(&linked::KERNELS);
    let [x, y, z] = TILT;
    let eye = DVec3::new(0.0, 0.0, 2.0);
    for model in [
        "tilted",
        "tilted computed",
        "tilted stored",
        "tilted object",
    ] {
        let cases = [
            (Mat3::IDENTITY, Vec3::new(x, y, z)),
            (
                Mat3::from_diagonal(Vec3::new(-1.0, 1.0, 1.0)),
                Vec3::new(-x, y, z),
            ),
            (
                Mat3::from_diagonal(Vec3::new(1.0, -1.0, 1.0)),
                Vec3::new(x, -y, z),
            ),
            (
                Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2),
                Vec3::new(-y, x, z),
            ),
        ];
        for (linear, want) in cases {
            let item = Item {
                model,
                linear,
                at: Vec3::ZERO,
            };
            let placed = place(&mut w, &[item]);
            let mut shot = Shot::at(eye, DVec3::ZERO, 1);
            shot.jitter = Vec2::ZERO;
            let g = shoot(&mut w, &placed, &shot);
            let seen = tilted_pixels(&g, want, |_| true);
            assert!(seen > 1000, "{model} {linear}: {seen} pixels");
        }
    }

    let tilted = w
        .scenes
        .material(w.mount, AssetId::from_name("tilted/material/0"))
        .unwrap();
    let object = w
        .scenes
        .material(w.mount, AssetId::from_name("tilted object/material/0"))
        .unwrap();
    let wall = Item {
        model: "plain",
        linear: Mat3::from_diagonal(Vec3::splat(8.0)),
        at: Vec3::ZERO,
    };
    let past = Quat::from_rotation_x(2.0);
    let cases = [
        (tilted, Quat::IDENTITY, Vec3::new(x, y, z)),
        (
            tilted,
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            Vec3::new(-y, x, z),
        ),
        // Past 90 degrees from the decal's +z: its y axis runs down the wall.
        (tilted, past, Vec3::new(x, -y, z)),
        // Mesh space directions are the box's, which no surface under it mirrors.
        (object, past, past * Vec3::new(x, y, z)),
    ];
    for (material, rotation, want) in cases {
        let placed = place(&mut w, &[wall]);
        let mut shot = Shot::at(eye, DVec3::ZERO, 1);
        shot.jitter = Vec2::ZERO;
        let g = shoot_with(&mut w, &placed, &shot, |s| {
            let decal = Decal {
                center: DVec3::ZERO,
                rotation,
                half_extent: Vec3::new(0.5, 0.5, 0.25),
                material,
                channels: DecalChannels::NORMAL,
                opacity: 1.0,
                fade_angles: [2.5, 3.0],
                depth_fade: 0.0,
            };
            s.set_decal(DecalId(0), Some(&decal)).unwrap();
        });
        assert!(tilted_pixels(&g, want, |n| n.z < 0.95) > 1000, "decal");
    }
}

#[test]
fn a_decal_on_a_wall_matches_between_the_gbuffer_and_a_mirror_reflection() {
    let _serial = serial();
    let mut w = World::new(&linked::KERNELS);
    let material = |w: &World, name: &str| {
        w.scenes
            .material(w.mount, AssetId::from_name(&format!("{name}/material/0")))
            .unwrap()
    };
    let (matte, sticker) = (material(&w, "matte"), material(&w, "sticker"));
    let wall_z = -4.0f32;
    let wall = Item {
        model: "plain",
        linear: Mat3::from_diagonal(Vec3::splat(8.0)),
        at: Vec3::new(0.0, 2.0, wall_z),
    };
    let floor = Item {
        model: "plain",
        linear: Mat3::from_rotation_x(-std::f32::consts::FRAC_PI_2) * 12.0,
        at: Vec3::ZERO,
    };
    let target = DVec3::new(0.0, 0.0, -1.0);
    let decals = |s: &mut SceneWriter<'_>| set_decals(s, matte, sticker, f64::from(wall_z));

    // The camera sees the wall in the floor, which `cones` treats as a mirror; a camera
    // mirrored in the floor sees the same wall directly, its image flipped upside down.
    let eye = Vec3::new(0.0, 1.5, 3.0);
    let mirrored = place(&mut w, &[floor, wall]);
    let mut shot = Shot::at(eye.as_dvec3(), target, 1);
    shot.jitter = Vec2::ZERO;
    let g = shoot_with(&mut w, &mirrored, &shot, decals);
    assert!(g.view.features.contains(orr_scene::SceneFeatures::DECALS));
    let module = linked::KERNELS.module(&trace::MODULE).unwrap();
    let r = rays(&g.view, module, "cones", 0);
    let direct = place(&mut w, &[wall]);
    let mut shot = Shot::at(DVec3::new(0.0, -1.5, 3.0), target, 1);
    shot.jitter = Vec2::ZERO;
    let d = shoot_with(&mut w, &direct, &shot, decals);

    let (width, height) = (EXTENT[0] as usize, EXTENT[1] as usize);
    let mut reflected = vec![[0.0; 3]; width * height];
    let mut seen = vec![[0.0; 3]; width * height];
    let mut reflected_roughness = vec![[0.0; 3]; width * height];
    let mut seen_roughness = vec![[0.0; 3]; width * height];
    let (mut compared, mut stickered, mut matte_pixels) = (0, [0, 0], 0);
    for (i, c) in r.cones.iter().enumerate() {
        // Pixels whose camera ray meets the floor in front of the wall and whose mirror ray
        // meets the wall away from its edges.
        let dir = Vec3::from(c.direction);
        if dir.y >= -1e-3 {
            continue;
        }
        let p1 = eye + dir * (eye.y / -dir.y);
        if p1.z <= wall_z + 0.05 || p1.x.abs() > 5.9 || p1.z > 5.9 {
            continue;
        }
        let r2 = Vec3::new(dir.x, -dir.y, dir.z);
        let p2 = p1 + r2 * ((wall_z - p1.z) / r2.z);
        if r2.z >= 0.0 || p2.y > 5.8 || p2.x.abs() > 3.8 {
            continue;
        }
        assert_eq!(c.hits, 2, "pixel {i}: floor then wall");
        let (x, y) = (i % width, i / width);
        let j = (height - 1 - y) * width + x;
        assert_eq!(d.ids[j], 1, "pixel {j}: the mirrored camera sees the wall");
        let core = decode_core(d.a[j], d.b[j]);
        let got = Vec3::new(c.context[1][0], c.context[1][1], c.context[1][2]);
        reflected[i] = arr(got.map(srgb_encode));
        seen[i] = [byte(d.a[j][0], 0), byte(d.a[j][0], 8), byte(d.a[j][0], 16)];
        reflected_roughness[i] = [c.context[1][3]; 3];
        seen_roughness[i] = [core.roughness; 3];
        for (k, rgb) in [reflected[i], seen[i]].into_iter().enumerate() {
            if rgb[2] > rgb[0] + 0.3 {
                stickered[k] += 1;
            }
        }
        if (core.roughness - MATTE_ROUGHNESS).abs() < 0.01 {
            matte_pixels += 1;
        }
        compared += 1;
    }
    assert!(compared > 3000, "{compared} pixels compared");
    assert!(
        stickered.iter().all(|&n| n > 150),
        "sticker pixels reflected and seen: {stickered:?}"
    );
    assert!(matte_pixels > 1000, "{matte_pixels} matte pixels");
    for (name, a, b) in [
        ("base color", &seen, &reflected),
        ("roughness", &seen_roughness, &reflected_roughness),
    ] {
        let e = flip(&image(|i| a[i]), &image(|i| b[i]));
        assert!(e.mean <= 0.01 && e.p99 <= 0.1, "{name}: FLIP {e:?}");
    }
    assert_clean();
}

#[test]
fn copies_leaving_their_rt_range_stop_tracing_and_trace_again_on_return() {
    let _serial = serial();
    let mut w = finest_world();
    let items = [
        Item {
            model: "ball",
            linear: Mat3::IDENTITY,
            at: Vec3::new(-1.0, 1.0, -2.0),
        },
        Item {
            model: "probe",
            linear: Mat3::IDENTITY,
            at: Vec3::new(0.0, 0.0, -40.0),
        },
    ];
    let placed = place_at(&mut w, &items, DVec3::ZERO, 30.0);
    let eye = DVec3::new(0.0, 2.0, 6.0);
    let target = DVec3::new(0.0, 1.0, -20.0);
    let near = DVec3::new(0.0, 2.0, -20.0);
    let far = DVec3::new(0.0, 2.0, 80.0);
    let probe = placed.ranges[1].first + 1;
    for (frame, focus, traced) in [(1, near, true), (2, far, false), (3, near, true)] {
        let mut shot = Shot::at(eye, target, frame);
        shot.focus = Some(focus);
        let g = shoot(&mut w, &placed, &shot);
        let r = rays(&g.view, &trace::MODULE, "pixels", 0);
        let drawn = g.ids.iter().filter(|&&id| id == probe).count();
        let hit = r.traced.iter().filter(|t| t.instance == probe).count();
        let agree = (0..g.ids.len())
            .filter(|&i| g.ids[i] == probe && r.traced[i].instance == probe)
            .count();
        assert!(
            drawn > 200,
            "frame {frame}: the probe covers {drawn} pixels"
        );
        if traced {
            assert!(
                agree as f64 >= 0.99 * drawn as f64 && hit <= drawn + drawn / 100,
                "frame {frame}: {agree} of {drawn} probe pixels traced, {hit} hits"
            );
        } else {
            assert_eq!(hit, 0, "frame {frame}: traced outside its rt_range");
        }
    }
    assert_clean();
}

/// The chain's models, nearest first.
const CHAIN: [&str; 4] = ["chain 0", "chain 1", "chain 2", "chain 3"];
const CHAIN_SHAPES: [(u32, [f32; 3]); 4] = [
    (24, [0.8, 0.2, 0.2]),
    (12, [0.2, 0.8, 0.2]),
    (8, [0.2, 0.2, 0.8]),
    (5, [0.8, 0.8, 0.2]),
];
/// Where steps 1, 2 and 3 start, in m from the focus.
const CHAIN_DISTANCES: [f32; 3] = [20.0, 40.0, 80.0];
const CHAIN_HYSTERESIS: f32 = 1.15;
const CHAIN_FADE_FRAMES: u8 = 16;
const CHAIN_SEEN: u8 = 0x80;
const COPY_FAR: u32 = 0x8000_0000;
const RT_WRITTEN: u32 = 1;
const RT_FADING: u32 = 64;
const RT_STEP_SHIFT: u32 = 8;
const RT_FAR_CUT_SHIFT: u32 = 16;

/// Whether a traced or drawn id, a copy + 1, is a copy's far step.
fn far_copy(id: u32) -> bool {
    id != 0 && (id - 1) & COPY_FAR != 0
}

/// The reference for a copy's chain step: farther past a switch distance x 1.15,
/// nearer inside it, at once for a copy seen first, else cross-fading over 16 frames; a
/// change back to the step it fades from reverses the fade, and another change keeps the
/// step that holds the larger share.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ChainState {
    to: u8,
    from: u8,
    fade: u8,
    seen: bool,
}

impl ChainState {
    fn next(self, distance: f32) -> Self {
        let n = CHAIN_DISTANCES.len() as u8;
        if !self.seen {
            let to = CHAIN_DISTANCES.iter().filter(|&&s| distance >= s).count() as u8;
            return Self {
                to,
                from: to,
                fade: 0,
                seen: true,
            };
        }
        let mut s = self;
        if s.fade != 0 {
            s.fade = if s.fade + 1 >= CHAIN_FADE_FRAMES {
                0
            } else {
                s.fade + 1
            };
        }
        let mut k = s.to;
        while k < n && distance > CHAIN_DISTANCES[k as usize] * CHAIN_HYSTERESIS {
            k += 1;
        }
        while k > 0 && distance < CHAIN_DISTANCES[k as usize - 1] {
            k -= 1;
        }
        if k != s.to {
            if s.fade != 0 && k == s.from {
                s.from = s.to;
                s.fade = CHAIN_FADE_FRAMES - s.fade;
            } else {
                if s.fade == 0 || s.fade >= CHAIN_FADE_FRAMES / 2 {
                    s.from = s.to;
                }
                s.fade = u8::from(s.from != k);
            }
            s.to = k;
        }
        if s.fade == 0 {
            s.from = s.to;
        }
        s
    }

    /// `SlotStateGpu::chain_step`.
    fn word(self) -> u8 {
        let from = if self.fade != 0 { self.from << 2 } else { 0 };
        self.to | from | CHAIN_SEEN
    }

    fn near(self) -> u8 {
        self.to.min(self.from)
    }
}

/// A world and the chain's four models.
fn chain_world() -> (World, [ModelId; 4]) {
    let w = finest_world();
    let models = CHAIN.map(|name| w.model(name));
    (w, models)
}

/// Sets the chain on the first of `models`.
fn set_chain(w: &mut World, models: &[ModelId; 4]) {
    let chain: Vec<(ModelId, f32)> = models[1..].iter().copied().zip(CHAIN_DISTANCES).collect();
    w.scenes.set_chain(models[0], &chain).unwrap();
}

/// The materials of each of `models`' two sections.
fn model_materials(root: &SceneRoot, models: &[ModelId]) -> Vec<[u32; 2]> {
    models
        .iter()
        .map(|m| {
            let gpu = read_buffer(root.models.offset(u64::from(m.0)), 1)[0];
            let ids = read_buffer(root.model_materials.offset(u64::from(gpu.materials)), 2);
            [ids[0], ids[1]]
        })
        .collect()
}

/// The material of each pixel's visible cluster; `u32::MAX` where nothing was drawn.
fn pixel_materials(view: &RasterView) -> Vec<u32> {
    let pixels = u64::from(EXTENT[0] * EXTENT[1]);
    let vis: Vec<u64> = read_buffer(view.targets.visibility, pixels);
    let clusters = visible_clusters(view);
    vis.iter()
        .map(|&w| {
            if w == 0 {
                u32::MAX
            } else {
                clusters[(w as u32 >> 7) as usize].material
            }
        })
        .collect()
}

/// Copies of the chain's model in a row, traced at any distance.
fn chain_copies() -> Vec<Item> {
    [-2.7, -0.9, 0.9, 2.7]
        .into_iter()
        .map(|x| Item {
            model: CHAIN[0],
            linear: Mat3::IDENTITY,
            at: Vec3::new(x, 1.0, 0.0),
        })
        .collect()
}

#[test]
fn chain_steps_and_fades_follow_the_reference_over_a_distance_sweep() {
    let _serial = serial();
    let (mut w, models) = chain_world();
    let items = chain_copies();
    let placed = place_at(&mut w, &items, DVec3::ZERO, 1e4);
    let slots: Vec<u32> = placed.ranges.iter().map(|r| r.first).collect();
    let eye = DVec3::new(0.0, 2.0, 6.0);
    let target = DVec3::new(0.0, 1.0, 0.0);
    let g = shoot(&mut w, &placed, &Shot::at(eye, target, 1));
    let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
    let m = read_buffer(root.models.offset(u64::from(models[0].0)), 1)[0];
    set_chain(&mut w, &models);
    // Out past every step and back, a pause, then back and forth across the first switch
    // faster than a fade.
    let mut path: Vec<f32> = (0..45).map(|i| 3.0 + 2.7 * i as f32).collect();
    path.extend([path[44]; 20]);
    path.extend((0..45).rev().map(|i| 3.0 + 2.7 * i as f32));
    path.extend([3.0; 20]);
    path.extend((0..40).map(|i| 22.0 + 7.0 * (i as f32 * 0.6).sin()));
    let mut state = vec![ChainState::default(); items.len()];
    let mut fades = 0;
    for (i, &z) in path.iter().enumerate() {
        let frame = i as u32 + 2;
        let mut shot = Shot::at(eye, target, frame);
        let mut focus = DVec3::new(0.0, 1.0, f64::from(z));
        let center = (Vec3::from(m.bounds_min) + Vec3::from(m.bounds_max)) * 0.5;
        let radius = 0.5 * (Vec3::from(m.bounds_max) - Vec3::from(m.bounds_min)).length();
        // Keeps every copy's distance clear of the switch distances, where f32 rounding on
        // the GPU could fall either side.
        let distances = |f: DVec3| -> Vec<f32> {
            items
                .iter()
                .map(|c| ((c.at + center).as_dvec3() - f).length() as f32 - radius)
                .map(|d| d.max(0.0))
                .collect()
        };
        let near_switch = |d: f32| {
            CHAIN_DISTANCES
                .iter()
                .any(|&s| (d - s).abs() < 0.01 || (d - s * CHAIN_HYSTERESIS).abs() < 0.01)
        };
        while distances(focus).into_iter().any(near_switch) {
            focus.z += 0.05;
        }
        shot.focus = Some(focus);
        shoot(&mut w, &placed, &shot);
        let got = read_buffer(root.slot_state, u64::from(placed.count));
        for (k, d) in distances(focus).into_iter().enumerate() {
            state[k] = state[k].next(d);
            let s = got[slots[k] as usize];
            let want = state[k];
            assert_eq!(
                (s.chain_step, s.fade),
                (want.word(), want.fade),
                "frame {frame}, copy {k} at {d} m: {want:?}"
            );
            // Ray tracing holds the step the raster draws, and a fading copy's far step.
            let fading = want.fade != 0;
            assert_eq!(
                (s.rt_flags >> RT_STEP_SHIFT) & 3,
                u32::from(want.near()),
                "frame {frame}, copy {k}: traced step"
            );
            assert_eq!(
                s.rt_flags & RT_FADING != 0,
                fading,
                "frame {frame}, copy {k}"
            );
            assert_eq!(
                s.far_part != 0,
                fading,
                "frame {frame}, copy {k}: far instance"
            );
            fades += usize::from(fading);
        }
    }
    assert!(fades > 100, "{fades} fading copy frames");
    assert_clean();
}

#[test]
fn each_chain_step_traces_what_the_raster_draws_and_a_fade_shows_both() {
    let _serial = serial();
    let (mut w, models) = chain_world();
    set_chain(&mut w, &models);
    let placed = place_at(&mut w, &chain_copies(), DVec3::ZERO, 1e4);
    let eye = DVec3::new(0.0, 2.0, 6.0);
    let target = DVec3::new(0.0, 1.0, 0.0);
    let mut frame = 0;
    let mut shoot_at = |w: &mut World, z: f64| {
        frame += 1;
        let mut shot = Shot::at(eye, target, frame);
        shot.focus = Some(DVec3::new(0.0, 1.0, z));
        shoot(w, &placed, &shot)
    };
    // Focus distances inside steps 0 to 3.
    for (step, z) in [3.0, 30.0, 60.0, 120.0].into_iter().enumerate() {
        let mut g = shoot_at(&mut w, z);
        for _ in 0..CHAIN_FADE_FRAMES {
            g = shoot_at(&mut w, z);
        }
        let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
        let shown = model_materials(&root, &models[step..=step])[0];
        let r = rays(&g.view, &trace::MODULE, "pixels", 1);
        let (matched, covered) = parity(&g, &r, 1e-4);
        let total = r.traced.len();
        assert!(
            covered > total / 20,
            "step {step}: {covered} pixels covered"
        );
        assert!(
            matched as f64 >= 0.999 * total as f64,
            "step {step}: {matched} of {total} pixels agree"
        );
        // Outside a fade every pixel shows the step's model and no far step.
        let materials = pixel_materials(&g.view);
        for (i, t) in r.traced.iter().enumerate() {
            assert!(!far_copy(g.ids[i]) && !far_copy(t.instance));
            assert!(materials[i] == u32::MAX || shown.contains(&materials[i]));
            assert!(t.instance == 0 || shown.contains(&t.material));
        }
    }
    // Back to step 0: four frames into the fade both steps show, in raster and traced.
    let mut g = shoot_at(&mut w, 3.0);
    for _ in 0..3 {
        g = shoot_at(&mut w, 3.0);
    }
    let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
    let both = model_materials(&root, &[models[0], models[3]]);
    let r = rays(&g.view, &trace::MODULE, "pixels", 1);
    let materials = pixel_materials(&g.view);
    let drawn_far = g.ids.iter().filter(|&&id| far_copy(id)).count();
    let traced_far = r.traced.iter().filter(|t| far_copy(t.instance)).count();
    let covered = g.ids.iter().filter(|&&id| id != 0).count();
    for (lo, hi, what) in [
        (drawn_far, covered, "drawn"),
        (traced_far, covered, "traced"),
    ] {
        assert!(
            lo > hi / 20 && lo < hi * 19 / 20,
            "{what}: {lo} of {hi} pixels on the far step"
        );
    }
    for (i, t) in r.traced.iter().enumerate() {
        let on = |m: u32| both.iter().any(|b| b.contains(&m));
        assert!(materials[i] == u32::MAX || on(materials[i]));
        assert!(t.instance == 0 || on(t.material));
    }
    assert_clean();
}

/// The chain's copies settled at step 2 in the cell next to the focus's, in a world whose
/// PTLAS budget is 0, and a decoy copy in the focus's cell. Moved, the decoy's write is the
/// nearest work past the mandatory and takes the frame's one admission the budget leaves.
struct Starved {
    w: World,
    models: [ModelId; 4],
    placed: Placed,
    decoy: TransformGroupId,
    slots: Vec<u32>,
    frame: u32,
}

impl Starved {
    fn new() -> Self {
        let mut w = World::with(&RendererDesc {
            raster_error_pixels: 1e-3,
            rt_error_pixels: 1e-3,
            budgets: Budgets {
                ptlas_rebuilt: 0,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        });
        let models = CHAIN.map(|name| w.model(name));
        set_chain(&mut w, &models);
        let placed = place_at(&mut w, &chain_copies(), DVec3::new(300.0, 0.0, 0.0), 1e4);
        let ball = w.model("ball");
        let decoy = {
            let list = [placed.scene];
            let f = w
                .scenes
                .begin_frame(&FrameDesc {
                    time: time(0),
                    scenes: &list,
                    low_latency: None,
                })
                .unwrap();
            let mut s = f.scene(placed.scene).unwrap();
            let group = s
                .create_transform_group(&TransformGroupDesc {
                    count: 1,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::new(240.0, 0.0, 0.0),
                })
                .unwrap();
            s.transforms(group, 0, 1).unwrap()[0] = Affine3::from_translation(Vec3::Y).into();
            s.add_instances(&InstancesDesc {
                model: ball,
                copies: 1,
                transforms: group,
                first: 0,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 1e4,
                animation_range: 0.0,
                user_tag: 0,
            })
            .unwrap();
            drop(s);
            wait(f.end(&[]).unwrap());
            group
        };
        let slots = placed.ranges.iter().map(|r| r.first).collect();
        let mut starved = Self {
            w,
            models,
            placed,
            decoy,
            slots,
            frame: 0,
        };
        // Placed without a focus, the copies fade from step 0 to step 2.
        let mut g = starved.shoot(Vec::new(), None);
        for _ in 0..CHAIN_FADE_FRAMES + 3 {
            g = starved.shoot(Vec::new(), None);
        }
        let state = starved.state(&g);
        for &slot in &starved.slots {
            let s = state[slot as usize];
            assert_eq!((s.chain_step, s.fade), (2 | CHAIN_SEEN, 0), "slot {slot}");
            assert_eq!(
                s.rt_flags & (RT_WRITTEN | 3 << RT_STEP_SHIFT),
                RT_WRITTEN | 2 << RT_STEP_SHIFT,
                "slot {slot}"
            );
        }
        starved
    }

    /// A frame with the focus 57 m from the copies and the camera before them; with a decoy
    /// position, the decoy moves there.
    fn shoot(&mut self, moved: Vec<(u32, Affine3)>, decoy: Option<Vec3>) -> Frame {
        self.frame += 1;
        let mut shot = Shot::at(
            DVec3::new(300.0, 2.0, 6.0),
            DVec3::new(300.0, 1.0, 0.0),
            self.frame,
        );
        shot.focus = Some(DVec3::new(250.0, 1.0, 30.0));
        shot.moved = moved;
        let group = self.decoy;
        shoot_with(&mut self.w, &self.placed, &shot, |s| {
            if let Some(at) = decoy {
                s.transforms(group, 0, 1).unwrap()[0] = Affine3::from_translation(at).into();
            }
        })
    }

    /// The slot state of the copies and the decoy.
    fn state(&self, g: &Frame) -> Vec<SlotStateGpu> {
        let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
        read_buffer(root.slot_state, u64::from(self.placed.count) + 1)
    }
}

#[test]
fn a_chain_copy_hidden_past_the_ptlas_budget_stops_tracing_its_step_at_once() {
    let _serial = serial();
    let mut c = Starved::new();
    // Hiding a copy takes it to step 0 at once. Its null write is not the nearest work, but
    // its instance holds step 2, whose hits would resolve through step 0.
    let hidden = 1;
    let g = c.shoot(
        vec![(hidden, Affine3::ZERO)],
        Some(Vec3::new(0.5, 1.0, 0.0)),
    );
    let state = c.state(&g);
    for &slot in &c.slots {
        let s = state[slot as usize];
        let near = u32::from(s.chain_step) & 3;
        assert!(
            s.rt_flags & RT_WRITTEN == 0 || (s.rt_flags >> RT_STEP_SHIFT) & 3 == near,
            "slot {slot} traces step {} of step {near}",
            (s.rt_flags >> RT_STEP_SHIFT) & 3
        );
    }
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let hits = |slot: u32| r.traced.iter().filter(|t| t.instance == slot + 1).count();
    assert_eq!(
        hits(c.slots[hidden as usize]),
        0,
        "the hidden copy is traced"
    );
    for (k, &slot) in c.slots.iter().enumerate() {
        assert!(
            k == hidden as usize || hits(slot) > 20,
            "copy {k} is not traced"
        );
    }
    assert_clean();
}

#[test]
fn a_chain_replacing_a_shown_step_model_past_the_ptlas_budget_traces_it_at_once() {
    let _serial = serial();
    let mut c = Starved::new();
    // Step 2's model becomes step 3's. The copies' rewrite is not the nearest work, but their
    // instances hold the old model, whose hits would resolve through the new one.
    let m = c.models;
    let chain = [
        (m[1], CHAIN_DISTANCES[0]),
        (m[3], CHAIN_DISTANCES[1]),
        (m[2], CHAIN_DISTANCES[2]),
    ];
    c.w.scenes.set_chain(m[0], &chain).unwrap();
    let g = c.shoot(Vec::new(), Some(Vec3::new(0.5, 1.0, 0.0)));
    let state = c.state(&g);
    for &slot in &c.slots {
        let s = state[slot as usize];
        assert_eq!((s.chain_step, s.fade), (2 | CHAIN_SEEN, 0), "slot {slot}");
        assert_eq!(s.rt_model, m[3].0, "slot {slot}: traced model");
    }
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, covered) = parity(&g, &r, 1e-4);
    let total = r.traced.len();
    assert!(covered > total / 20, "{covered} pixels covered");
    assert!(
        matched as f64 >= 0.999 * total as f64,
        "{matched} of {total} pixels agree"
    );
    assert_clean();
}

#[test]
fn a_fade_from_a_step_its_chain_loses_ends_at_once() {
    let _serial = serial();
    let (mut w, models) = chain_world();
    let placed = place_at(&mut w, &chain_copies(), DVec3::ZERO, 1e4);
    let slots: Vec<u32> = placed.ranges.iter().map(|r| r.first).collect();
    let eye = DVec3::new(0.0, 2.0, 6.0);
    let target = DVec3::new(0.0, 1.0, 0.0);
    let mut frame = 0;
    let mut shoot_at = |w: &mut World, z: f64| {
        frame += 1;
        let mut shot = Shot::at(eye, target, frame);
        shot.focus = Some(DVec3::new(0.0, 1.0, z));
        let g = shoot(w, &placed, &shot);
        let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
        let state = read_buffer(root.slot_state, u64::from(placed.count));
        slots.iter().map(|&s| state[s as usize]).collect::<Vec<_>>()
    };
    // Far out the copies take step 3 once their chain is set, and in one frame to about 28 m
    // they fade from step 3 to step 1.
    shoot_at(&mut w, 120.0);
    set_chain(&mut w, &models);
    for (k, s) in shoot_at(&mut w, 120.0).into_iter().enumerate() {
        assert_eq!((s.chain_step, s.fade), (3 | CHAIN_SEEN, 0), "copy {k}");
    }
    for (k, s) in shoot_at(&mut w, 30.0).into_iter().enumerate() {
        assert_eq!(
            (s.chain_step, s.fade),
            (3 << 2 | 1 | CHAIN_SEEN, 1),
            "copy {k}"
        );
    }
    // The chain loses step 3 while the copies are about 42 m out, past step 2's start but
    // inside its hysteresis: the fade ends on step 1, and no far instance stays.
    let chain = [
        (models[1], CHAIN_DISTANCES[0]),
        (models[2], CHAIN_DISTANCES[1]),
    ];
    w.scenes.set_chain(models[0], &chain).unwrap();
    for (k, s) in shoot_at(&mut w, 44.0).into_iter().enumerate() {
        assert_eq!((s.chain_step, s.fade), (1 | CHAIN_SEEN, 0), "copy {k}");
        assert_eq!(
            s.rt_flags & (RT_WRITTEN | RT_FADING | 3 << RT_STEP_SHIFT),
            RT_WRITTEN | 1 << RT_STEP_SHIFT,
            "copy {k}"
        );
        assert_eq!(s.far_part, 0, "copy {k}: far instance");
    }
    assert_clean();
}

#[test]
fn a_fade_that_turns_to_another_far_step_traces_that_step() {
    let _serial = serial();
    let (mut w, models) = chain_world();
    set_chain(&mut w, &models);
    let placed = place_at(&mut w, &chain_copies(), DVec3::ZERO, 1e4);
    let slots: Vec<u32> = placed.ranges.iter().map(|r| r.first).collect();
    let eye = DVec3::new(0.0, 2.0, 6.0);
    let target = DVec3::new(0.0, 1.0, 0.0);
    let mut frame = 0;
    let mut shoot_at = |w: &mut World, z: f64| {
        frame += 1;
        let mut shot = Shot::at(eye, target, frame);
        shot.focus = Some(DVec3::new(0.0, 1.0, z));
        shoot(w, &placed, &shot)
    };
    let state = |g: &Frame| {
        let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
        read_buffer(root.slot_state, u64::from(placed.count))
    };
    // Step 0, then three frames into a fade to step 1, which the far instances trace.
    for _ in 0..=CHAIN_FADE_FRAMES {
        shoot_at(&mut w, 3.0);
    }
    let mut g = shoot_at(&mut w, 30.0);
    for _ in 0..2 {
        g = shoot_at(&mut w, 30.0);
    }
    let before = state(&g);
    // Farther, the fade turns to step 2 and restarts while step 0 stays the near one; the
    // far instances keep the cut index they held.
    for _ in 0..3 {
        g = shoot_at(&mut w, 60.0);
    }
    let after = state(&g);
    for &slot in &slots {
        let (a, b) = (before[slot as usize], after[slot as usize]);
        assert_eq!((a.chain_step, a.fade), (1 | CHAIN_SEEN, 3), "slot {slot}");
        assert_eq!((b.chain_step, b.fade), (2 | CHAIN_SEEN, 3), "slot {slot}");
        // The far instance is held across the turn, not taken again, so only its model
        // differs from what it holds.
        assert!(a.far_part != 0 && b.far_part == a.far_part, "slot {slot}");
        let held = |s: SlotStateGpu| (s.far_model, (s.rt_flags >> RT_FAR_CUT_SHIFT) & 0xff);
        let (model, cut) = held(a);
        assert_eq!(model, models[1].0, "slot {slot}: far model");
        assert_eq!(
            held(b),
            (models[2].0, cut),
            "slot {slot}: far model and cut"
        );
    }
    let turned = rays(&g.view, &trace::MODULE, "pixels", 0);
    // Settled at step 2, the copies trace it through their own instances.
    for _ in 0..CHAIN_FADE_FRAMES {
        g = shoot_at(&mut w, 60.0);
    }
    let settled = rays(&g.view, &trace::MODULE, "pixels", 0);
    let mut checked = 0;
    let mut matched = 0;
    for (t, s) in turned.traced.iter().zip(&settled.traced) {
        if !far_copy(t.instance) {
            continue;
        }
        checked += 1;
        let same = (t.instance - 1) & !COPY_FAR == s.instance.wrapping_sub(1)
            && t.cluster == s.cluster
            && t.primitive == s.primitive
            && (t.t - s.t).abs() <= 1e-5 * s.t;
        matched += usize::from(same);
    }
    let covered = settled.traced.iter().filter(|t| t.instance != 0).count();
    assert!(
        checked > covered / 20,
        "{checked} of {covered} pixels on the far step"
    );
    assert!(
        matched as f64 >= 0.99 * checked as f64,
        "{matched} of {checked} far pixels trace step 2"
    );
    assert_clean();
}

#[test]
fn a_chain_replacing_the_far_step_model_of_a_fading_copy_traces_the_new_model() {
    let _serial = serial();
    let (mut w, models) = chain_world();
    set_chain(&mut w, &models);
    let placed = place_at(&mut w, &chain_copies(), DVec3::ZERO, 1e4);
    let slots: Vec<u32> = placed.ranges.iter().map(|r| r.first).collect();
    let eye = DVec3::new(0.0, 2.0, 6.0);
    let target = DVec3::new(0.0, 1.0, 0.0);
    let mut frame = 0;
    let mut shoot_at = |w: &mut World, z: f64| {
        frame += 1;
        let mut shot = Shot::at(eye, target, frame);
        shot.focus = Some(DVec3::new(0.0, 1.0, z));
        shoot(w, &placed, &shot)
    };
    let state = |g: &Frame| {
        let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
        read_buffer(root.slot_state, u64::from(placed.count))
    };
    // Settled at step 1, then two frames into a fade to step 2, which the far instances trace.
    for _ in 0..=CHAIN_FADE_FRAMES {
        shoot_at(&mut w, 30.0);
    }
    shoot_at(&mut w, 60.0);
    let mut g = shoot_at(&mut w, 60.0);
    let before = state(&g);
    // Step 2's model becomes step 3's while the fade goes on; the far instances keep their
    // step and the cut index they held.
    let chain = [
        (models[1], CHAIN_DISTANCES[0]),
        (models[3], CHAIN_DISTANCES[1]),
        (models[2], CHAIN_DISTANCES[2]),
    ];
    w.scenes.set_chain(models[0], &chain).unwrap();
    g = shoot_at(&mut w, 60.0);
    let after = state(&g);
    for &slot in &slots {
        let (a, b) = (before[slot as usize], after[slot as usize]);
        let fading = 2 | 1 << 2 | CHAIN_SEEN;
        assert_eq!((a.chain_step, a.fade), (fading, 2), "slot {slot}");
        assert_eq!((b.chain_step, b.fade), (fading, 3), "slot {slot}");
        assert!(a.far_part != 0 && b.far_part == a.far_part, "slot {slot}");
        let held = |s: SlotStateGpu| (s.far_model, (s.rt_flags >> RT_FAR_CUT_SHIFT) & 0xff);
        let (model, cut) = held(a);
        assert_eq!(model, models[2].0, "slot {slot}: far model");
        assert_eq!(
            held(b),
            (models[3].0, cut),
            "slot {slot}: far model and cut"
        );
    }
    let turned = rays(&g.view, &trace::MODULE, "pixels", 0);
    // Settled at step 2, the copies trace its new model through their own instances.
    for _ in 0..CHAIN_FADE_FRAMES {
        g = shoot_at(&mut w, 60.0);
    }
    let settled = rays(&g.view, &trace::MODULE, "pixels", 0);
    let mut checked = 0;
    let mut matched = 0;
    for (t, s) in turned.traced.iter().zip(&settled.traced) {
        if !far_copy(t.instance) {
            continue;
        }
        checked += 1;
        let same = (t.instance - 1) & !COPY_FAR == s.instance.wrapping_sub(1)
            && t.cluster == s.cluster
            && t.primitive == s.primitive
            && (t.t - s.t).abs() <= 1e-5 * s.t;
        matched += usize::from(same);
    }
    let covered = settled.traced.iter().filter(|t| t.instance != 0).count();
    assert!(
        checked > covered / 20,
        "{checked} of {covered} pixels on the far step"
    );
    assert!(
        matched as f64 >= 0.99 * checked as f64,
        "{matched} of {checked} far pixels trace the new model"
    );
    assert_clean();
}

/// Traced normals as an image, for FLIP.
fn traced_image(r: &Rays) -> Image {
    image(|i| {
        let t = &r.traced[i];
        if t.instance == 0 {
            return [0.0; 3];
        }
        Vec3::from(t.normal).map(|v| v * 0.5 + 0.5).to_array()
    })
}

#[test]
fn moving_the_scene_and_camera_far_from_the_origin_changes_nothing() {
    let _serial = serial();
    let mut w = finest_world();
    let eye = DVec3::new(0.3, 2.6, 5.5);
    let target = DVec3::new(0.0, 0.6, 0.0);
    let mut reference: Option<(Frame, Image)> = None;
    // 0, 8, 16 and 31 km, on the render origin's grid.
    for shift in [0.0, 8192.0, 16384.0, 31744.0] {
        let at = DVec3::new(shift, 0.0, -0.5 * shift);
        let placed = place_at(&mut w, &showcase(), at, 100.0);
        // The first frame moves the render origin to the focus, which defers PTLAS writes.
        shoot(&mut w, &placed, &Shot::at(at + eye, at + target, 1));
        let g = shoot(&mut w, &placed, &Shot::at(at + eye, at + target, 2));
        let r = rays(&g.view, &trace::MODULE, "pixels", 1);
        let (matched, _) = parity(&g, &r, 1e-4);
        assert!(
            matched as f64 >= 0.999 * r.traced.len() as f64,
            "{shift} m: {matched} pixels agree"
        );
        let traced = traced_image(&r);
        match &reference {
            None => reference = Some((g, traced)),
            Some((f0, i0)) => {
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert!(bits(&g.depth) == bits(&f0.depth), "{shift} m: depth");
                let normals = |f: &Frame| {
                    f.a.iter()
                        .zip(&f.b)
                        .map(|(a, b)| (a[1], b[1]))
                        .collect::<Vec<_>>()
                };
                assert!(normals(&g) == normals(f0), "{shift} m: normals");
                let e = flip(i0, &traced).mean;
                assert!(e < 0.01, "{shift} m: traced FLIP {e}");
            }
        }
    }
    assert_clean();
}

#[test]
fn motion_vectors_do_not_change_when_the_render_origin_moves() {
    let _serial = serial();
    let mut w = finest_world();
    let items = [
        Item {
            model: "ground",
            linear: Mat3::IDENTITY,
            at: Vec3::ZERO,
        },
        Item {
            model: "ball",
            linear: Mat3::IDENTITY,
            at: Vec3::ZERO,
        },
    ];
    let path = |i: u32| {
        let x = 40.0 * f64::from(i);
        let wobble = (f64::from(i) * 0.7).sin();
        let eye = DVec3::new(x, 2.0 + 0.1 * wobble, 4.0);
        let moved = vec![
            (0, Affine3::from_translation(Vec3::new(x as f32, 0.0, 0.0))),
            (
                1,
                Affine3::from_mat3_translation(
                    Mat3::from_rotation_y(0.1 * i as f32),
                    Vec3::new(x as f32 - 1.0 + 0.2 * wobble as f32, 0.9, -0.5),
                ),
            ),
        ];
        (eye, DVec3::new(x, 0.5, 0.0), moved)
    };
    let mut runs = Vec::new();
    for rebase in [true, false] {
        let placed = place(&mut w, &items);
        let mut motion = Vec::new();
        let mut origins = Vec::new();
        for i in 1..30 {
            let (eye, target, moved) = path(i);
            let shot = Shot {
                eye,
                target,
                focus: rebase.then_some(eye),
                moved,
                reset: i == 1,
                frame: i,
                jitter: Vec2::new(0.25, -0.125),
                mip_bias: 0.0,
            };
            let g = shoot(&mut w, &placed, &shot);
            let root: Vec<SceneRoot> = read_buffer(g.view.scene, 1);
            origins.push(root[0].origin_cell[0]);
            motion.push(g.motion);
        }
        origins.dedup();
        runs.push((motion, origins));
    }
    let (with, origins) = &runs[0];
    assert_eq!(
        origins.len(),
        3,
        "two rebases: render origin cells {origins:?}"
    );
    assert_eq!(runs[1].1.len(), 1, "no rebase without a focus");
    // Motion is stored as f16: two runs may round one value to neighboring steps.
    let step = |v: f32| {
        let e = v.abs().max(6.1e-5).log2().floor();
        2f32.powf(e - 10.0)
    };
    let mut worst = 0f32;
    for (a, b) in with.iter().zip(&runs[1].0).skip(1) {
        for (m, n) in a.iter().zip(b) {
            for c in 0..2 {
                let px = (m[c] - n[c]).abs() * EXTENT[c] as f32;
                let allowed = (step(m[c]) * EXTENT[c] as f32).max(1e-3);
                worst = worst.max(px / allowed);
            }
        }
    }
    assert!(
        worst <= 1.0,
        "motion differs by {worst} times the larger of 1e-3 px and one f16 step"
    );
    assert_clean();
}

#[test]
fn ptlas_work_past_the_budget_waits_and_lands_nearest_first() {
    let _serial = serial();
    let mut w = World::with(&RendererDesc {
        budgets: Budgets {
            ptlas_rebuilt: 50,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    });
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let model = w.model("plain");
    let list = [scene];
    let mut seen = Vec::new();
    for frame in 1..6u32 {
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        if frame == 1 {
            let mut s = f.scene(scene).unwrap();
            s.set_focus(DVec3::new(5.0, 1.0, 5.0));
            f.request_detail(scene, PIXELS_PER_RADIAN).unwrap();
            // Three cells of 40 copies, 0, 2 and 5 cells from the focus.
            for cell in [0.0, 2.0, 5.0] {
                let g = s
                    .create_transform_group(&TransformGroupDesc {
                        count: 40,
                        writer: GroupWriter::Cpu,
                        anchor: DVec3::new(cell * 256.0 + 10.0, 0.0, 10.0),
                    })
                    .unwrap();
                for (i, t) in s.transforms(g, 0, 40).unwrap().iter_mut().enumerate() {
                    *t = Affine3::from_translation(Vec3::new(i as f32 * 2.0, 0.0, 0.0)).into();
                }
                s.add_instances(&InstancesDesc {
                    model,
                    copies: 40,
                    transforms: g,
                    first: 0,
                    pose: None,
                    material_set: None,
                    flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                    rt_range: 10_000.0,
                    animation_range: 0.0,
                    user_tag: 0,
                })
                .unwrap();
            }
        }
        wait(f.end(&[]).unwrap());
        // An empty frame takes in the completed one's figures.
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(frame),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        wait(f.end(&[]).unwrap());
        let rt = w.scenes.stats().rt;
        seen.push((rt.writes, rt.deferred, rt.max_age));
    }
    assert_eq!(
        seen,
        [(40, 80, 1), (40, 40, 2), (40, 0, 0), (0, 0, 0), (0, 0, 0)],
        "writes, deferred and the oldest wait per frame"
    );
    assert_eq!(w.scenes.stats().rt.partitions, 3, "one partition per cell");
    assert_clean();
}

#[test]
fn later_mounts_and_unmounts_leave_traced_scenes_intact() {
    let _serial = serial();
    let mut w = finest_world();
    let placed = place(&mut w, &showcase());
    let eye = DVec3::new(0.3, 2.6, 5.5);
    let target = DVec3::new(0.0, 0.6, 0.0);
    let check = |w: &mut World, frame| {
        let g = shoot(w, &placed, &Shot::at(eye, target, frame));
        let r = rays(&g.view, &trace::MODULE, "pixels", 0);
        let (matched, _) = parity(&g, &r, 1e-4);
        assert!(
            matched as f64 >= 0.999 * r.traced.len() as f64,
            "frame {frame}: {matched} pixels agree"
        );
    };
    check(&mut w, 1);
    for round in 0..3 {
        let m = w.scenes.mount(PackSource::Memory(Arc::clone(pack())));
        let start = Instant::now();
        while matches!(w.scenes.mount_state(m), LoadState::Pending) {
            let f = w
                .scenes
                .begin_frame(&FrameDesc {
                    time: time(2),
                    scenes: &[],
                    low_latency: None,
                })
                .unwrap();
            f.end(&[]).unwrap();
            assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
        }
        assert!(matches!(w.scenes.mount_state(m), LoadState::Ready));
        w.scenes.unmount(m);
        check(&mut w, 3 + round);
    }
    assert_clean();
}

// --- Rigid parts ------------------------------------------------------------------------------

/// Parts of the rig: the plate and 255 boxes.
const RIG_PARTS: u32 = 256;
/// Pose variants of every rig copy.
const RIG_VARIANTS: u32 = 4;

/// Copies of the rig on a grid, posed from a GPU-written group by `pose.slang`.
struct Rigs {
    placed: Placed,
    range: InstanceRange,
    pose: TransformGroupId,
    blocks: u32,
    centers: Vec<Vec3>,
    animation_range: f32,
}

/// `count` rig copies in rows of `row`, `spacing` m apart around the origin, pose variant
/// `copy % 4`, their pose blocks shared (`stride` 0) or one per copy (`stride` 4), placed in
/// a frame focused on the origin.
fn place_rigs(w: &mut World, count: u32, row: u32, spacing: f32, stride: u32, range: f32) -> Rigs {
    let model = w.model("rig");
    place_rigs_of(w, model, count, row, spacing, stride, range)
}

fn place_rigs_of(
    w: &mut World,
    model: ModelId,
    count: u32,
    row: u32,
    spacing: f32,
    stride: u32,
    range: f32,
) -> Rigs {
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let rows = count.div_ceil(row);
    let centers: Vec<Vec3> = (0..count)
        .map(|i| {
            let (x, z) = ((i % row) as f32, (i / row) as f32);
            let half = Vec2::new(row as f32 - 1.0, rows as f32 - 1.0) * 0.5;
            Vec3::new((x - half.x) * spacing, 0.0, (z - half.y) * spacing)
        })
        .collect();
    let blocks = if stride == 0 {
        RIG_VARIANTS
    } else {
        (count - 1) * stride + RIG_VARIANTS
    };
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    s.set_focus(DVec3::ZERO);
    f.request_detail(scene, PIXELS_PER_RADIAN).unwrap();
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let pose = s
        .create_transform_group(&TransformGroupDesc {
            count: blocks * RIG_PARTS,
            writer: GroupWriter::Gpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    for (t, &c) in s
        .transforms(group, 0, count)
        .unwrap()
        .iter_mut()
        .zip(&centers)
    {
        *t = Affine3::from_translation(c).into();
    }
    let r = s
        .add_instances(&InstancesDesc {
            model,
            copies: count,
            transforms: group,
            first: 0,
            pose: Some(Pose {
                group: pose,
                index: 0,
                variants: RIG_VARIANTS,
                stride,
            }),
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 10_000.0,
            animation_range: range,
            user_tag: 0,
        })
        .unwrap();
    let variants: Vec<CopyVariant> = (0..count)
        .map(|i| CopyVariant {
            pose: (i % RIG_VARIANTS) as u8,
            material_set: 0,
            section_mask: 0,
        })
        .collect();
    s.copy_variants(r, 0, &variants).unwrap();
    drop(s);
    wait(f.end(&[]).unwrap());
    Rigs {
        placed: Placed {
            scene,
            ranges: vec![r],
            group,
            history,
            count,
        },
        range: r,
        pose,
        blocks,
        centers,
        animation_range: range,
    }
}

/// One frame of the rigs with their poses written by `entry` of `pose.slang` at `t`.
fn shoot_rigs(
    w: &mut World,
    rigs: &Rigs,
    shot: &Shot,
    entry: &str,
    t: f32,
    amplitude: f32,
) -> Frame {
    shoot_rigs_edited(w, rigs, shot, entry, t, amplitude, &[])
}

/// [`shoot_rigs`] in which `edits` (edit, stripe of `pieces`, lift) also change.
fn shoot_rigs_edited(
    w: &mut World,
    rigs: &Rigs,
    shot: &Shot,
    entry: &str,
    t: f32,
    amplitude: f32,
    edits: &[(EditId, u32, u32, f32)],
) -> Frame {
    let refs: Vec<EditRef> = edits.iter().map(|e| w.scenes.edit(e.0).unwrap()).collect();
    let kernel = pipeline(&pose::MODULE, entry);
    let list = [rigs.placed.scene];
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(shot.frame),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(rigs.placed.scene).unwrap();
        if let Some(focus) = shot.focus {
            s.set_focus(focus);
            f.request_detail(rigs.placed.scene, PIXELS_PER_RADIAN)
                .unwrap();
        }
        for &(i, m) in &shot.moved {
            s.transforms(rigs.placed.group, i, 1).unwrap()[0] = m.into();
        }
    }
    for e in edits {
        assert!(
            f.edit_dirty(e.0),
            "the frame's rebuild budget takes the edit"
        );
    }
    f.update_scenes(&mut |c| {
        dispatch_fractures(c, edits, &refs);
        let root = pose::Root {
            scene: c.root,
            group: rigs.pose.0,
            blocks: rigs.blocks,
            time: t,
            amplitude,
            _pad0: 0,
            _pad1: 0,
        };
        let root = c.put_root(&root).unwrap();
        c.cmd.dispatch(
            &kernel,
            root,
            [(rigs.blocks * RIG_PARTS).div_ceil(64), 1, 1],
        );
    })
    .unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: rigs.placed.scene,
            history: rigs.placed.history,
            position: shot.eye,
            rotation: look(Vec3::ZERO, (shot.target - shot.eye).as_vec3()),
            projection: Projection {
                fov_y: 0.9,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: shot.jitter,
            camera_cut: shot.reset,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
    wait(f.end(&[]).unwrap());
    read_back(&view)
}

/// The figures of the last frame, taken in by an empty one.
fn last_rt(w: &mut World, frame: u32) -> orr_scene::RtStats {
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(frame),
            scenes: &[],
            low_latency: None,
        })
        .unwrap();
    wait(f.end(&[]).unwrap());
    w.scenes.stats().rt
}

/// The rig's rest bounds: the plate and the boxes' tops.
const RIG_CENTER: Vec3 = Vec3::new(0.0, 0.075, 0.0);
fn rig_radius() -> f32 {
    0.5 * Vec3::new(2.0, 0.15, 2.0).length()
}

/// Which copies take their pose with the focus at `focus`, given which did before, with the
/// animation range's 15% hysteresis. Fails when a copy lies within 1 cm of a boundary.
fn animated(rigs: &Rigs, focus: Vec3, before: &[bool]) -> Vec<bool> {
    rigs.centers
        .iter()
        .zip(before)
        .map(|(&c, &was)| {
            let d = ((c + RIG_CENTER - focus).length() - rig_radius()).max(0.0);
            let range = rigs.animation_range * if was { 1.15 } else { 1.0 };
            assert!(
                (d - range).abs() > 0.01,
                "a copy {d} m away sits on the range"
            );
            d <= range
        })
        .collect()
}

#[test]
fn posed_rigid_parts_trace_where_the_raster_draws_them() {
    let _serial = serial();
    let mut w = finest_world();
    let eye = DVec3::new(0.0, 7.0, 12.0);
    let target = DVec3::new(0.0, 0.0, 1.0);
    let mut frame = 1;
    for stride in [0, RIG_VARIANTS] {
        let rigs = place_rigs(&mut w, 1000, 40, 2.2, stride, 6.09);
        let mut held = animated(&rigs, Vec3::ZERO, &[false; 1000]);
        // The focus crosses the field, leaves it (within the render origin's grid step, so
        // no frame rebases) and comes back.
        let foci = [
            Vec3::new(0.3, 0.0, 2.1),
            Vec3::new(6.2, 0.0, 1.7),
            Vec3::new(300.0, 0.0, 0.0),
            Vec3::new(-3.1, 0.0, 4.3),
        ];
        for focus in foci {
            held = animated(&rigs, focus, &held);
            let posed = held.iter().filter(|&&h| h).count() as u32;
            let mut previous: Option<Vec<f32>> = None;
            for step in 0..3 {
                let mut shot = Shot::at(eye, target, frame);
                shot.focus = Some(focus.as_dvec3());
                let t = frame as f32 * 0.21;
                let g = shoot_rigs(&mut w, &rigs, &shot, "sway", t, 0.35);
                let r = rays(&g.view, &trace::MODULE, "pixels", 0);
                // Thousands of sub-pixel boxes put many samples on edges, where the raster
                // and the ray may take the neighboring surface.
                let (matched, covered) = parity(&g, &r, 1e-4);
                let (same, _) = parity(&g, &r, f32::INFINITY);
                let total = r.traced.len();
                let context = format!("stride {stride}, focus {focus}, step {step}");
                assert!(covered > total / 3, "{covered} of {total} pixels covered");
                assert!(
                    same as f64 >= 0.998 * total as f64 && matched as f64 >= 0.995 * total as f64,
                    "{context}: {same} of {total} pixels agree on the copy, {matched} on the depth"
                );
                // Only posed copies move from one step to the next.
                if let Some(before) = &previous {
                    let moved = (0..total).filter(|&i| g.depth[i] != before[i]).count();
                    if posed == 0 {
                        assert_eq!(moved, 0, "{context}: copies at rest moved");
                    } else {
                        assert!(moved > 100, "{context}: only {moved} pixels moved");
                    }
                }
                previous = Some(g.depth);
                let rt = last_rt(&mut w, frame);
                assert_eq!(
                    (rt.parts, rt.parts_refused),
                    (posed * (RIG_PARTS - 1), 0),
                    "stride {stride}, focus {focus}: part instances of {posed} posed copies"
                );
                frame += 1;
            }
        }
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(frame),
                scenes: &[rigs.placed.scene],
                low_latency: None,
            })
            .unwrap();
        f.scene(rigs.placed.scene).unwrap().remove(rigs.range);
        wait(f.end(&[]).unwrap());
        assert_eq!(
            last_rt(&mut w, frame).parts,
            0,
            "removed copies release their runs"
        );
        w.scenes.destroy_scene(rigs.placed.scene);
    }
    assert_clean();
}

/// A stage of a prim that moves, with an instancer under it placing a 0.4 m square at (0.5, 0, 0).
fn hung_stage() -> PathBuf {
    let path = temp("hung.usda");
    std::fs::write(
        &path,
        r#"#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "Mover"
{
    double3 xformOp:translate.timeSamples = { 0: (0, 0, 0), 24: (1, 0, 0) }
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def PointInstancer "Crowd"
    {
        rel prototypes = [</Mover/Crowd/Leaf>]
        int[] protoIndices = [0]
        point3f[] positions = [(0.5, 0, 0)]
        def Mesh "Leaf"
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-0.2, -0.2, 0), (0.2, -0.2, 0), (0.2, 0.2, 0), (-0.2, 0.2, 0)]
        }
    }
}
"#,
    )
    .unwrap();
    path
}

/// What [`paced_stage`]'s instancer hangs from.
const MOVER: &str = r#"double3 xformOp:translate.timeSamples = { 0: (0, 0, 0), 24: (1, 0, 0) }
    uniform token[] xformOpOrder = ["xformOp:translate"]"#;

/// A stage of an instancer under a prim of `mover`, whose one instance goes from (0.5, 0, 0) to
/// (0.9, 0, 0) in the first second and is gone after the second.
fn paced_stage(name: &str, mover: &str) -> PathBuf {
    let path = temp(name);
    let stage = format!(
        r#"#usda 1.0
(
    metersPerUnit = 1
    timeCodesPerSecond = 10
    upAxis = "Y"
)
def Xform "Mover"
{{
    {mover}
    def PointInstancer "Crowd"
    {{
        rel prototypes = [</Mover/Crowd/Leaf>]
        int[] protoIndices.timeSamples = {{ 0: [0], 20: [] }}
        point3f[] positions.timeSamples = {{ 0: [(0.5, 0, 0)], 10: [(0.9, 0, 0)], 20: [] }}
        def Mesh "Leaf"
        {{
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-0.2, -0.2, 0), (0.2, -0.2, 0), (0.2, 0.2, 0), (-0.2, 0.2, 0)]
        }}
    }}
}}
"#
    );
    std::fs::write(&path, stage).unwrap();
    path
}

/// [`door_stage`] with a door whose shear grows over its clip, as glTF cannot animate one.
fn swaying_door_stage() -> PathBuf {
    let path = temp("swaying-door.usda");
    std::fs::write(
        &path,
        r#"#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "Door"
{
    matrix4d xformOp:transform.timeSamples = {
        0: ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        24: ((1, 0, 0, 0), (1.5, 1, 0, 0), (0.5, 0.25, 1, 0), (0, 0, 0, 1)),
    }
    uniform token[] xformOpOrder = ["xformOp:transform"]
    def Mesh "Leaf"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-0.2, -0.2, 0), (0.2, -0.2, 0), (0.2, 0.2, 0), (-0.2, 0.2, 0)]
    }
}
"#,
    )
    .unwrap();
    path
}

/// A stage of one prim that moves, holding a 0.4 m square facing +z: all of it on one rigid part,
/// and part 0, which takes no pose, empty.
fn door_stage() -> PathBuf {
    let path = temp("door.usda");
    std::fs::write(
        &path,
        r#"#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "Door"
{
    double3 xformOp:translate.timeSamples = { 0: (0, 0, 0), 24: (1, 0, 0) }
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "Leaf"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-0.2, -0.2, 0), (0.2, -0.2, 0), (0.2, 0.2, 0), (-0.2, 0.2, 0)]
    }
}
"#,
    )
    .unwrap();
    path
}

/// The door moved `x` m along x by the pose of its part.
fn place_door(w: &mut World, x: f32) -> Placed {
    let model = w.model("door");
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    s.set_focus(DVec3::ZERO);
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let pose = s
        .create_transform_group(&TransformGroupDesc {
            count: 2,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    s.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
    let parts = s.transforms(pose, 0, 2).unwrap();
    parts[0] = Affine3::IDENTITY.into();
    parts[1] = Affine3::from_translation(Vec3::X * x).into();
    let range = s
        .add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first: 0,
            pose: Some(Pose {
                group: pose,
                index: 0,
                variants: 1,
                stride: 0,
            }),
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 10_000.0,
            animation_range: 100.0,
            user_tag: 0,
        })
        .unwrap();
    drop(s);
    wait(f.end(&[]).unwrap());
    Placed {
        scene,
        ranges: vec![range],
        group,
        history,
        count: 1,
    }
}

/// `hung`'s page, hung from a palette whose joint moves it `x` m along x.
fn place_hung(w: &mut World, x: f32) -> Placed {
    place_page(w, "hung", x)
}

/// The page of `asset`, hung from a palette whose joint moves it `x` m along x if it names one.
fn place_page(w: &mut World, asset: &str, x: f32) -> Placed {
    let a = Archive::from_memory(Arc::clone(pack())).unwrap();
    let prefix = format!("{asset}/");
    let is_page = |e: &&Entry| {
        e.class == orr_pack::Class::InstancePage as u16 && a.name(e).starts_with(&prefix)
    };
    let entry = a.entries().iter().find(is_page).expect("the page");
    let header = orr_pack::instance::InstancePageInfo::parse(&blob(&a, entry, 0))
        .unwrap()
        .header;
    let hung = header.frame_skeleton != AssetId::NONE;
    if hung {
        assert_eq!(
            header.frame_skeleton,
            AssetId::from_name(&format!("{asset}/skeleton")),
            "the page names its skeleton"
        );
    }
    let page = w.scenes.instance_page(w.mount, entry.id).unwrap();
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    s.set_focus(DVec3::ZERO);
    let palette = s
        .create_transform_group(&TransformGroupDesc {
            count: 2,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let joints = s.transforms(palette, 0, 2).unwrap();
    joints[0] = Affine3::IDENTITY.into();
    joints[1] = Affine3::from_translation(Vec3::X * x).into();
    let range = s
        .add_instance_page(&InstancePageRef {
            page,
            frame: hung.then_some((palette, header.frame_joint)),
            poses: &[],
            material_sets: &[],
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    drop(s);
    wait(f.end(&[]).unwrap());
    Placed {
        scene,
        ranges: vec![range],
        group: palette,
        history,
        count: 1,
    }
}

/// A page hung from a joint draws and traces where the joint's palette entry poses its records,
/// without a rebuilt page: the joint moves it, the record says where it stands in bind space.
#[test]
fn a_page_hung_from_a_joint_follows_its_palette_entry() {
    let _serial = serial();
    let mut w = finest_world();
    let (eye, target) = (DVec3::new(0.5, 0.0, 2.5), DVec3::new(0.5, 0.0, 0.0));
    let middle = |g: &Frame| {
        let columns = g.ids.iter().enumerate().filter(|(_, id)| **id != 0);
        let (n, sum) = columns.fold((0u32, 0.0f64), |(n, sum), (i, _)| {
            (n + 1, sum + f64::from(i as u32 % EXTENT[0]))
        });
        assert!(n > 100, "{n} pixels of the copy");
        sum / f64::from(n)
    };
    let landed = |w: &mut World, placed: &Placed, from: u32| {
        let start = Instant::now();
        let mut frame = from;
        loop {
            frame += 1;
            let g = shoot(w, placed, &Shot::at(eye, target, frame));
            if frame > from + 3 && g.ids.iter().any(|&id| id != 0) {
                return g;
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "the page never landed"
            );
        }
    };
    let still = place_hung(&mut w, 0.0);
    let at_rest = landed(&mut w, &still, 0);
    let moved = place_hung(&mut w, 0.4);
    let g = landed(&mut w, &moved, 100);
    let f = 1.0 / (0.45f64).tan();
    let aspect = f64::from(EXTENT[0]) / f64::from(EXTENT[1]);
    let want = 0.5 * 0.4 / 2.5 * f / aspect * f64::from(EXTENT[0]);
    let got = middle(&g) - middle(&at_rest);
    assert!((got - want).abs() < 0.05 * want, "{got} px, not {want}");
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, _) = parity(&g, &r, 1e-4);
    assert!(
        matched as f64 >= 0.999 * r.traced.len() as f64,
        "{matched} of {} pixels agree",
        r.traced.len()
    );
    assert_clean();
}

/// A copy placed by hand goes with its pack too: the next pack to take its model's id holds
/// another model, which draws nothing of the old placement.
#[test]
fn a_copy_of_an_unmounted_model_stays_hidden_when_its_id_is_reused() {
    let _serial = serial();
    let mut w = finest_world();
    let (eye, target) = (DVec3::new(0.0, 0.0, 2.5), DVec3::ZERO);
    let old = w.model("plain");
    let placed = place(
        &mut w,
        &[Item {
            model: "plain",
            linear: Mat3::IDENTITY,
            at: Vec3::ZERO,
        }],
    );
    let mut frame = 1;
    let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
    assert!(g.ids.iter().any(|&id| id != 0), "the copy never drew");
    w.scenes.unmount(w.mount);
    for _ in 0..6 {
        frame += 1;
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        assert!(g.ids.iter().all(|&id| id == 0), "drawn with its pack gone");
    }
    let again = w.scenes.mount(PackSource::Memory(Arc::clone(pack())));
    let start = Instant::now();
    while matches!(w.scenes.mount_state(again), LoadState::Pending) {
        frame += 1;
        shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the mount never landed"
        );
    }
    assert!(matches!(w.scenes.mount_state(again), LoadState::Ready));
    let new = w
        .scenes
        .model(again, AssetId::from_name("plain"))
        .expect("the same pack again");
    assert_eq!(old, new, "the id goes to the next mount");
    for _ in 0..12 {
        frame += 1;
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        assert!(
            g.ids.iter().all(|&id| id == 0),
            "another mount's model drew the old copy"
        );
        let r = rays(&g.view, &trace::MODULE, "pixels", 0);
        assert!(
            r.traced.iter().all(|t| t.instance == 0),
            "another mount's model traced the old copy"
        );
    }
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(frame + 1),
            scenes: &[placed.scene],
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(placed.scene).unwrap();
        for &r in &placed.ranges {
            s.remove(r);
        }
    }
    wait(f.end(&[]).unwrap());
    assert_clean();
}

/// A copy goes with its pack. The next pack to take its page's slot holds another page, whose
/// records are no copy's of the old placement, however alike the slot.
#[test]
fn a_copy_of_an_unmounted_page_stays_hidden_when_its_slot_is_reused() {
    let _serial = serial();
    let mut w = finest_world();
    let (eye, target) = (DVec3::new(0.5, 0.0, 2.5), DVec3::new(0.5, 0.0, 0.0));
    let placed = place_hung(&mut w, 0.0);
    let start = Instant::now();
    let mut frame = 0;
    loop {
        frame += 1;
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        if frame > 3 && g.ids.iter().any(|&id| id != 0) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the page never landed"
        );
    }
    w.scenes.unmount(w.mount);
    for _ in 0..6 {
        frame += 1;
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        assert!(g.ids.iter().all(|&id| id == 0), "drawn with its pack gone");
    }
    let again = w.scenes.mount(PackSource::Memory(Arc::clone(pack())));
    while matches!(w.scenes.mount_state(again), LoadState::Pending) {
        frame += 1;
        shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the mount never landed"
        );
    }
    assert!(matches!(w.scenes.mount_state(again), LoadState::Ready));
    for _ in 0..12 {
        frame += 1;
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        assert!(
            g.ids.iter().all(|&id| id == 0),
            "another mount's page drew the old copy"
        );
    }
    assert_clean();
}

/// An animated page draws and traces where its rows stand at the page's clock, frame after frame
/// of one placement: the rows move a copy on, a joint above carries it further, and past its last
/// sample it is gone, to come back when the clock does.
#[test]
fn an_animated_page_follows_its_clock_and_its_joint() {
    let _serial = serial();
    let mut w = finest_world();
    let (eye, target) = (DVec3::new(0.7, 0.0, 2.5), DVec3::new(0.7, 0.0, 0.0));
    let middle = |g: &Frame| {
        let columns = g.ids.iter().enumerate().filter(|(_, id)| **id != 0);
        let (n, sum) = columns.fold((0u32, 0.0f64), |(n, sum), (i, _)| {
            (n + 1, sum + f64::from(i as u32 % EXTENT[0]))
        });
        assert!(n > 100, "{n} pixels of the copy");
        sum / f64::from(n)
    };
    let f = 1.0 / (0.45f64).tan();
    let aspect = f64::from(EXTENT[0]) / f64::from(EXTENT[1]);
    let px = |m: f64| 0.5 * m / 2.5 * f / aspect * f64::from(EXTENT[0]);
    for (asset, base) in [("paced", 0), ("paced alone", 100)] {
        let placed = place_page(&mut w, asset, 0.0);
        let range = placed.ranges[0];
        let mut frame = base;
        let mut from = |w: &mut World, before: f32, t: f32, joint: Option<f32>| {
            frame += 4;
            let mut g = None;
            for n in 0..4 {
                let mut shot = Shot::at(eye, target, frame + n);
                shot.reset = false;
                if let Some(x) = joint {
                    shot.moved = vec![(1, Affine3::from_translation(Vec3::X * x))];
                }
                let t = if n < 3 { before } else { t };
                g = Some(shoot_with(w, &placed, &shot, |s| s.set_page_time(range, t)));
            }
            g.unwrap()
        };
        let start = Instant::now();
        let at_rest = loop {
            let g = from(&mut w, 0.0, 0.0, None);
            if g.ids.iter().any(|&id| id != 0) {
                break g;
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "the page never landed"
            );
        };
        let rest = middle(&at_rest);
        let g = from(&mut w, 0.25, 0.5, None);
        let got = middle(&g) - rest;
        assert!((got - px(0.2)).abs() < 0.05 * px(0.2), "{asset}: {got} px");
        // A frame's motion is what its clock moved the copy since the last.
        for (i, _) in g.ids.iter().enumerate().filter(|(_, id)| **id != 0) {
            let m = f64::from(g.motion[i][0]) * f64::from(EXTENT[0]);
            assert!((m + px(0.1)).abs() < 0.05, "{asset}: {m} px of motion");
        }
        let g = from(&mut w, 1.5, 1.5, None);
        let got = middle(&g) - rest;
        assert!((got - px(0.4)).abs() < 0.05 * px(0.4), "{asset}: {got} px");
        let r = rays(&g.view, &trace::MODULE, "pixels", 0);
        let (matched, _) = parity(&g, &r, 1e-4);
        assert!(
            matched as f64 >= 0.999 * r.traced.len() as f64,
            "{asset}: {matched} of {} pixels agree",
            r.traced.len()
        );
        if asset == "paced" {
            let g = from(&mut w, 1.5, 1.5, Some(-0.3));
            let got = middle(&g) - rest;
            let want = px(0.1);
            assert!((got - want).abs() < 0.05 * want, "{got} px, not {want}");
        }
        let g = from(&mut w, 2.5, 2.5, None);
        assert!(g.ids.iter().all(|&id| id == 0), "{asset}: the copy is gone");
        let g = from(&mut w, 0.0, 0.0, Some(0.0));
        let got = middle(&g) - rest;
        assert!(got.abs() < 1.0, "{asset}: {got} px after it came back");
    }
    assert_clean();
}

/// Rows far past the instance page pool stream by the page's clock: its copy reaches every time
/// along the animation and back to its start, never hidden on the way, while the pool stays in
/// its budget.
#[test]
fn an_animated_page_past_its_pool_streams_by_its_clock() {
    let _serial = serial();
    let pool = 6u64 << 20;
    let rows = WALK_RECORDS * WALK_INTERVALS * size_of::<MotionRow>();
    assert!(rows as u64 > 2 * pool, "{rows} B of rows");
    let mut w = walk_world(pool);
    let placed = place_page(&mut w, "long walk", 0.0);
    let mut frame = 0;
    let rest = walk_rest(&mut w, &placed, &mut frame);
    let times = (1..40).map(|i| i as f32 * 0.5).chain([0.0]);
    for t in times {
        let want = walk_px(t);
        let start = Instant::now();
        loop {
            let g = walk_shot(&mut w, &placed, &mut frame, t);
            let x = walk_middle(&g).unwrap_or_else(|| panic!("hidden on the way to {t} s"));
            let stats = *w.scenes.residency();
            let used = stats
                .class(orr_scene::ResidencyClass::InstancePages)
                .resident_bytes;
            assert!(used <= pool, "{used} B resident at {t} s");
            assert_eq!(stats.pages_over_budget, 0, "at {t} s");
            if (x - rest - want).abs() < 1.0 {
                break;
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "stuck {} px short of {t} s",
                x - rest - want
            );
        }
    }
    assert_clean();
}

/// A pool with room for one window of rows gives it to the clock, not to the window copies hold
/// in: the copy may hide on the way but gets there.
#[test]
fn an_animated_page_with_room_for_one_window_follows_its_clock() {
    let _serial = serial();
    let window = MOTION_WINDOW_BYTES as u64;
    let mut w = walk_world(window + (window >> 1));
    let placed = place_page(&mut w, "long walk", 0.0);
    let mut frame = 0;
    let rest = walk_rest(&mut w, &placed, &mut frame);
    let t = 5.0;
    let want = walk_px(t);
    let start = Instant::now();
    loop {
        let g = walk_shot(&mut w, &placed, &mut frame, t);
        if walk_middle(&g).is_some_and(|x| (x - rest - want).abs() < 1.0) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "never reached {t} s"
        );
    }
    assert_clean();
}

/// A prim that moves cooks to a model with nothing on part 0, which the renderer does not pose:
/// the pose of its part moves all of it, drawn and traced alike.
#[test]
fn a_rigid_model_with_an_empty_part_0_draws_and_traces_where_its_pose_says() {
    let _serial = serial();
    let mut w = finest_world();
    let (eye, target) = (DVec3::new(0.0, 0.0, 2.5), DVec3::ZERO);
    let middle = |g: &Frame| {
        let columns = g.ids.iter().enumerate().filter(|(_, id)| **id != 0);
        let (n, sum) = columns.fold((0u32, 0.0f64), |(n, sum), (i, _)| {
            (n + 1, sum + f64::from(i as u32 % EXTENT[0]))
        });
        assert!(n > 100, "{n} pixels of the door");
        sum / f64::from(n)
    };
    let still = place_door(&mut w, 0.0);
    let at_rest = shoot(&mut w, &still, &Shot::at(eye, target, 1));
    let moved = place_door(&mut w, 0.4);
    // The part's BLAS follows its cut, a frame or two after the copy first takes a pose.
    for frame in 2..6 {
        shoot(&mut w, &moved, &Shot::at(eye, target, frame));
    }
    let g = shoot(&mut w, &moved, &Shot::at(eye, target, 6));
    let f = 1.0 / (0.45f64).tan();
    let aspect = f64::from(EXTENT[0]) / f64::from(EXTENT[1]);
    let want = 0.5 * 0.4 / 2.5 * f / aspect * f64::from(EXTENT[0]);
    let got = middle(&g) - middle(&at_rest);
    assert!((got - want).abs() < 0.05 * want, "{got} px, not {want}");
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, _) = parity(&g, &r, 1e-4);
    assert!(
        matched as f64 >= 0.999 * r.traced.len() as f64,
        "{matched} of {} pixels agree",
        r.traced.len()
    );
    assert_clean();
}

#[test]
fn a_sliding_part_moves_by_its_analytic_motion() {
    let _serial = serial();
    let mut w = finest_world();
    let rigs = place_rigs(&mut w, 1, 1, 0.0, 0, 100.0);
    // Level with the boxes and just under the plate, whose back faces away.
    let eye = DVec3::new(0.0, -0.01, 2.5);
    let target = DVec3::new(0.0, -0.01, 0.0);
    let amplitude = 0.05;
    let (t0, t1) = (0.0f32, 0.5f32);
    shoot_rigs(
        &mut w,
        &rigs,
        &Shot::at(eye, target, 1),
        "slide",
        t0,
        amplitude,
    );
    let g = shoot_rigs(
        &mut w,
        &rigs,
        &Shot::at(eye, target, 2),
        "slide",
        t1,
        amplitude,
    );
    // Every box moved along the camera's x by d: a point at view depth z moves on screen by
    // -d f / (aspect z) in NDC, half that in uv.
    let d = amplitude * (t1.sin() - t0.sin());
    let f = 1.0 / (0.45f32).tan();
    let aspect = EXTENT[0] as f32 / EXTENT[1] as f32;
    let (mut n, mut worst) = (0, 0.0f32);
    for (i, &id) in g.ids.iter().enumerate() {
        if id == 0 {
            continue;
        }
        let z = 0.1 / g.depth[i];
        let expected = -0.5 * d * f / (aspect * z) * EXTENT[0] as f32;
        let m = Vec2::from(g.motion[i]) * Vec2::new(EXTENT[0] as f32, EXTENT[1] as f32);
        worst = worst.max((m.x - expected).abs()).max(m.y.abs());
        n += 1;
    }
    assert!(n > 1000, "{n} box pixels");
    assert!(worst <= 0.05, "motion off its analytic value by {worst} px");
    assert_clean();
}

/// A frame of `rigs` posed by `sway` at a fixed time, with `write` run on the scene's writer,
/// whose camera rays are traced after it, neither waited for. Returns the scene's root too.
fn shoot_rigs_later(
    w: &mut World,
    rigs: &Rigs,
    shot: &Shot,
    write: impl FnOnce(&mut SceneWriter<'_>),
) -> (RaysLater, GpuPtr<SceneRoot>) {
    let kernel = pipeline(&pose::MODULE, "sway");
    let list = [rigs.placed.scene];
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(shot.frame),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(rigs.placed.scene).unwrap();
        s.set_focus(shot.eye);
        write(&mut s);
    }
    f.request_detail(rigs.placed.scene, PIXELS_PER_RADIAN)
        .unwrap();
    let mut scene = GpuPtr::NULL;
    f.update_scenes(&mut |c| {
        scene = c.root;
        let root = pose::Root {
            scene: c.root,
            group: rigs.pose.0,
            blocks: rigs.blocks,
            time: 0.3,
            amplitude: 0.3,
            _pad0: 0,
            _pad1: 0,
        };
        let root = c.put_root(&root).unwrap();
        c.cmd.dispatch(
            &kernel,
            root,
            [(rigs.blocks * RIG_PARTS).div_ceil(64), 1, 1],
        );
    })
    .unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: rigs.placed.scene,
            history: rigs.placed.history,
            position: shot.eye,
            rotation: look(Vec3::ZERO, (shot.target - shot.eye).as_vec3()),
            projection: Projection {
                fov_y: 0.9,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: shot.jitter,
            camera_cut: shot.reset,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
    f.end(&[]).unwrap();
    (rays_later(&view, &trace::MODULE, "pixels", 0), scene)
}

/// The PTLAS of a scene grows under frames in flight. Two posed rigs hold part runs, and runs
/// of two more rigs, placed and removed, wait on the pool's free list. Then 64 more posed rigs
/// behind the camera, in their animation range, take more slots, cell partitions and part
/// instances than the PTLAS holds: it is made again in a larger layout, and the part pool
/// carries over. Every frame meanwhile traces what the frames before did, the first two rigs'
/// parts posed as before, and every copy holds its run.
#[test]
fn the_ptlas_grows_past_its_layout_under_frames_in_flight_and_its_part_runs_hold() {
    let _serial = serial();
    let mut w = finest_world();
    let rigs = place_rigs(&mut w, 2, 2, 2.4, 0, 100.0);
    let model = w.model("rig");
    let (eye, target) = (DVec3::new(0.0, 2.5, 3.5), DVec3::ZERO);
    let behind = |s: &mut SceneWriter<'_>, count: u32| {
        let group = s
            .create_transform_group(&TransformGroupDesc {
                count,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        for (i, t) in s
            .transforms(group, 0, count)
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *t = Affine3::from_translation(Vec3::new(
                2.4 * (i % 8) as f32,
                0.0,
                20.0 + 2.4 * (i / 8) as f32,
            ))
            .into();
        }
        s.add_instances(&InstancesDesc {
            model,
            copies: count,
            transforms: group,
            first: 0,
            pose: Some(Pose {
                group: rigs.pose,
                index: 0,
                variants: RIG_VARIANTS,
                stride: 0,
            }),
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 10_000.0,
            animation_range: 100.0,
            user_tag: 0,
        })
        .unwrap()
    };
    let mut frame = 0;
    let mut next = |w: &mut World, write: &mut dyn FnMut(&mut SceneWriter<'_>)| {
        frame += 1;
        shoot_rigs_later(w, &rigs, &Shot::at(eye, target, frame), write)
    };
    // Frames, each read while the next is in flight, until two trace the same.
    let start = Instant::now();
    let (mut flight, scene) = next(&mut w, &mut |_| {});
    let mut last: Option<Vec<u8>> = None;
    let settled = loop {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the rigs never settled"
        );
        let following = next(&mut w, &mut |_| {}).0;
        wait(flight.point);
        let now: Vec<u8> = bytemuck::cast_slice(&flight.read().traced).to_vec();
        flight = following;
        if last.as_ref() == Some(&now) {
            break now;
        }
        last = Some(now);
    };
    let mut later = vec![flight];
    let mut gone = None;
    later.push(next(&mut w, &mut |s| gone = Some(behind(s, 2))).0);
    later.push(next(&mut w, &mut |s| s.remove(gone.unwrap())).0);
    for _ in 0..2 {
        later.push(next(&mut w, &mut |_| {}).0);
    }
    wait(later.last().unwrap().point);
    let held: SceneRoot = read_buffer(scene, 1)[0];
    let rt = last_rt(&mut w, 1000);
    assert_eq!((rt.parts, rt.parts_refused), (2 * (RIG_PARTS - 1), 0));
    // Room for the slots handed out and a run of 2^8 part instances for each of the four rigs
    // posed at once.
    assert_eq!((held.part_base, held.part_capacity), (4, 4 * RIG_PARTS));

    later.push(
        next(&mut w, &mut |s| {
            behind(s, 64);
        })
        .0,
    );
    for _ in 0..3 {
        later.push(next(&mut w, &mut |_| {}).0);
    }
    wait(later.last().unwrap().point);
    let grown: SceneRoot = read_buffer(scene, 1)[0];
    assert!(
        grown.part_base >= 66,
        "{} slots for 66 rigs",
        grown.part_base
    );
    assert_eq!(grown.part_capacity, 66 * RIG_PARTS);
    for (i, l) in later.iter().enumerate() {
        let traced: Vec<u8> = bytemuck::cast_slice(&l.read().traced).to_vec();
        assert!(
            traced == settled,
            "frame {} of {} traced other hits at {} pixels",
            i + 1,
            later.len(),
            differing(&traced, &settled),
        );
    }
    let rt = last_rt(&mut w, 1001);
    assert_eq!(
        (rt.parts, rt.parts_refused, rt.unplaced),
        (66 * (RIG_PARTS - 1), 0, 0)
    );
    assert_clean();
}

#[test]
fn a_zero_transform_or_the_null_group_hides_a_posed_copy_with_its_parts() {
    let _serial = serial();
    let mut w = finest_world();
    let rigs = place_rigs(&mut w, 2, 2, 2.4, 0, 100.0);
    let eye = DVec3::new(0.0, 2.5, 3.5);
    let target = DVec3::ZERO;
    let count = |g: &Frame, r: &Rays, slot: u32| {
        let drawn = g.ids.iter().filter(|&&id| id == slot + 1).count();
        let traced = r.traced.iter().filter(|t| t.instance == slot + 1).count();
        (drawn, traced)
    };
    let g = shoot_rigs(&mut w, &rigs, &Shot::at(eye, target, 1), "sway", 0.3, 0.3);
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let first = rigs.range.first;
    for slot in [first, first + 1] {
        let (drawn, traced) = count(&g, &r, slot);
        assert!(
            drawn > 500 && traced > 500,
            "slot {slot}: {drawn} drawn, {traced} traced"
        );
    }
    assert_eq!(last_rt(&mut w, 2).parts, 2 * (RIG_PARTS - 1));
    // Copy 0 takes an all-zero transform; copy 1 is placed again in the null group.
    let model = w.model("rig");
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(3),
            scenes: &[rigs.placed.scene],
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(rigs.placed.scene).unwrap();
        s.transforms(rigs.placed.group, 0, 1).unwrap()[0] = Affine3::ZERO.into();
        s.remove(InstanceRange {
            first: first + 1,
            count: 1,
        });
    }
    wait(f.end(&[]).unwrap());
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(4),
            scenes: &[rigs.placed.scene],
            low_latency: None,
        })
        .unwrap();
    let hidden = f
        .scene(rigs.placed.scene)
        .unwrap()
        .add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: TransformGroupId(0),
            first: 0,
            pose: Some(Pose {
                group: rigs.pose,
                index: 0,
                variants: RIG_VARIANTS,
                stride: 0,
            }),
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 10_000.0,
            animation_range: 100.0,
            user_tag: 0,
        })
        .unwrap();
    wait(f.end(&[]).unwrap());
    let g = shoot_rigs(&mut w, &rigs, &Shot::at(eye, target, 5), "sway", 0.4, 0.3);
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    for slot in [first, first + 1, hidden.first] {
        assert_eq!(count(&g, &r, slot), (0, 0), "slot {slot} is hidden");
    }
    assert_eq!(
        last_rt(&mut w, 6).parts,
        0,
        "hidden copies hold no part runs"
    );
    assert_clean();
}

#[test]
fn a_section_mask_hides_its_sections_in_raster_and_ray_tracing_and_nothing_else() {
    let _serial = serial();
    let mut w = finest_world();
    let rigs = place_rigs(&mut w, 2, 2, 2.4, 0, 100.0);
    let eye = DVec3::new(0.0, 2.5, 3.5);
    let target = DVec3::ZERO;
    let first = rigs.range.first;
    let shot = |frame| Shot::at(eye, target, frame);
    let whole = shoot_rigs(&mut w, &rigs, &shot(1), "sway", 0.3, 0.3);
    // Copy 1 hides section 1, every box: what stays is its plate.
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(2),
            scenes: &[rigs.placed.scene],
            low_latency: None,
        })
        .unwrap();
    let boxes = CopyVariant {
        pose: 0,
        material_set: 0,
        section_mask: 1 << 1,
    };
    f.scene(rigs.placed.scene)
        .unwrap()
        .copy_variants(rigs.range, 1, &[boxes])
        .unwrap();
    wait(f.end(&[]).unwrap());
    let masked = shoot_rigs(&mut w, &rigs, &shot(3), "sway", 0.3, 0.3);
    let r = rays(&masked.view, &trace::MODULE, "pixels", 1);
    let (matched, _) = parity(&masked, &r, 1e-4);
    assert!(
        matched as f64 >= 0.999 * r.traced.len() as f64,
        "{matched} of {} pixels agree",
        r.traced.len()
    );
    // The boxes are metal and pale (green above 0.6 in Rec.2020); the plate is neither.
    let metal = |i: usize, g: &Frame| decode_core(g.a[i], g.b[i]).metalness > 0.5;
    let pixels = |g: &Frame, slot: u32, boxes: bool| {
        (0..g.ids.len())
            .filter(|&i| g.ids[i] == slot + 1 && metal(i, g) == boxes)
            .count()
    };
    let traced_boxes = |slot: u32| {
        (0..r.traced.len())
            .filter(|&i| r.traced[i].instance == slot + 1 && r.traced[i].base_color[1] > 0.6)
            .count()
    };
    assert_eq!(
        pixels(&masked, first, true),
        pixels(&whole, first, true),
        "copy 0 keeps its boxes"
    );
    assert!(
        pixels(&whole, first + 1, true) > 100,
        "copy 1 shows boxes first"
    );
    assert_eq!(pixels(&masked, first + 1, true), 0, "copy 1 draws no box");
    assert_eq!(traced_boxes(first + 1), 0, "rays hit no box of copy 1");
    assert!(traced_boxes(first) > 100, "rays hit copy 0's boxes");
    assert!(
        pixels(&masked, first + 1, false) > pixels(&whole, first + 1, false),
        "copy 1 keeps its plate"
    );
    assert_eq!(
        last_rt(&mut w, 4).parts,
        RIG_PARTS - 1,
        "the masked copy's boxes leave the PTLAS; copy 0 keeps its run"
    );
    assert_clean();
}

#[test]
fn rewriting_a_material_set_changes_only_the_copies_that_name_it() {
    let _serial = serial();
    let mut w = finest_world();
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let plain = w.model("plain");
    let list = [scene];
    let mut materials = Vec::new();
    let mut records = Vec::new();
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    {
        let mut mats = f.materials();
        for base_color in [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] {
            let m = Material {
                base_color,
                ..Material::default()
            };
            let m = MaterialGpu::new(&m, &[BoundTexture::NONE; slot::COUNT]);
            materials.push(mats.create(&m).unwrap());
            records.push(Vec3::from(m.base_color));
        }
    }
    let mut s = f.scene(scene).unwrap();
    s.material_sets(0, 2)
        .unwrap()
        .copy_from_slice(&[materials[0], materials[1]]);
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: 3,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let mut ranges = Vec::new();
    for (i, set) in [Some(0), Some(1), None].into_iter().enumerate() {
        s.transforms(group, i as u32, 1).unwrap()[0] =
            Affine3::from_translation(Vec3::new(i as f32 * 1.2 - 1.2, 0.0, 0.0)).into();
        ranges.push(
            s.add_instances(&InstancesDesc {
                model: plain,
                copies: 1,
                transforms: group,
                first: i as u32,
                pose: None,
                material_set: set,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 100.0,
                animation_range: 0.0,
                user_tag: 0,
            })
            .unwrap(),
        );
    }
    drop(s);
    wait(f.end(&[]).unwrap());
    let placed = Placed {
        scene,
        ranges,
        group,
        history,
        count: 3,
    };
    let eye = DVec3::new(0.0, 0.0, 3.0);
    let colors = |w: &mut World, frame| {
        let g = shoot(w, &placed, &Shot::at(eye, DVec3::ZERO, frame));
        let r = rays(&g.view, &trace::MODULE, "pixels", 1);
        placed
            .ranges
            .iter()
            .map(|range| {
                let id = range.first + 1;
                let i = g.ids.iter().position(|&p| p == id).expect("drawn");
                let t = r.traced.iter().find(|t| t.instance == id).expect("traced");
                (
                    decode_core(g.a[i], g.b[i]).base_color,
                    Vec3::from(t.base_color),
                )
            })
            .collect::<Vec<_>>()
    };
    let before = colors(&mut w, 1);
    // Set 0 now names the blue material.
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(2),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    f.scene(scene).unwrap().material_sets(0, 1).unwrap()[0] = materials[2];
    wait(f.end(&[]).unwrap());
    let after = colors(&mut w, 3);
    let close = |a: Vec3, b: Vec3| (a - b).abs().max_element() < 0.02;
    let blue = records[2];
    assert!(
        close(after[0].0, blue) && close(after[0].1, blue),
        "copy 0: {:?}",
        after[0]
    );
    for i in 1..3 {
        assert!(
            close(after[i].0, before[i].0) && close(after[i].1, before[i].1),
            "copy {i}: {:?} then {:?}",
            before[i],
            after[i]
        );
    }
    assert!(
        close(before[0].0, records[0]),
        "copy 0 starts red: {:?}",
        before[0]
    );
    assert!(
        close(before[1].0, records[1]),
        "copy 1 is green: {:?}",
        before[1]
    );
    assert_clean();
}

#[test]
fn a_color_factor_multiplies_its_texel_in_the_texture_primaries() {
    let _serial = serial();
    let mut w = finest_world();
    let items: Vec<Item> = TINTED
        .iter()
        .enumerate()
        .map(|(i, &(model, ..))| Item {
            model,
            linear: Mat3::IDENTITY,
            at: Vec3::new(i as f32 * 1.2 - 1.2, 0.0, 0.0),
        })
        .collect();
    let placed = place(&mut w, &items);
    let g = shoot(
        &mut w,
        &placed,
        &Shot::at(DVec3::new(0.0, 0.0, 3.0), DVec3::ZERO, 1),
    );
    let r = rays(&g.view, &trace::MODULE, "pixels", 1);
    let texel = TINT_TEXEL.map(|c| f64::from(srgb_decode(f32::from(c) / 255.0)));
    for (range, &(model, primaries)) in placed.ranges.iter().zip(&TINTED) {
        let p = primaries.rgb();
        let factor = RgbPrimaries::REC709.matrix_to(p) * DVec3::from(TINT_FACTOR.map(f64::from));
        let want = (p.matrix_to(RgbPrimaries::REC2020) * (factor * DVec3::from(texel))).as_vec3();
        let id = range.first + 1;
        let i = g.ids.iter().position(|&p| p == id).expect("drawn");
        let raster = decode_core(g.a[i], g.b[i]).base_color;
        let t = r.traced.iter().find(|t| t.instance == id).expect("traced");
        let traced = Vec3::from(t.base_color);
        assert!(
            (raster - want).abs().max_element() < 0.01
                && (traced - want).abs().max_element() < 0.003,
            "{model}: raster {raster}, traced {traced}, want {want}"
        );
    }
    assert_clean();
}

#[test]
fn a_texture_on_uv_set_1_reads_that_set_in_the_raster_and_at_ray_hits() {
    let _serial = serial();
    let mut w = finest_world();
    let item = Item {
        model: "second uv",
        linear: Mat3::IDENTITY,
        at: Vec3::ZERO,
    };
    let placed = place(&mut w, &[item]);
    let g = shoot(
        &mut w,
        &placed,
        &Shot::at(DVec3::new(0.0, 0.0, 1.5), DVec3::ZERO, 1),
    );
    let r = rays(&g.view, &trace::MODULE, "pixels", 1);
    let id = placed.ranges[0].first + 1;
    // Set 1 spans both halves; set 0 would read red everywhere.
    let halves = |colors: Vec<Vec3>| {
        let red = colors.iter().filter(|c| c.x > 2.0 * c.y).count();
        let green = colors.iter().filter(|c| c.y > 2.0 * c.x).count();
        (red, green, colors.len())
    };
    let raster = (0..g.ids.len())
        .filter(|&i| g.ids[i] == id)
        .map(|i| decode_core(g.a[i], g.b[i]).base_color)
        .collect();
    let traced = r
        .traced
        .iter()
        .filter(|t| t.instance == id)
        .map(|t| Vec3::from(t.base_color))
        .collect();
    for (path, (red, green, n)) in [("raster", halves(raster)), ("traced", halves(traced))] {
        assert!(
            n > 0 && red * 3 > n && green * 3 > n,
            "{path}: {red} red and {green} green of {n} pixels"
        );
    }
    assert_clean();
}

// --- The DAG traversal, two-pass occlusion and debug_lod --------------------------------------

/// What a cut needs of one mesh cluster: its record and, in DAG mode, its group, the group's
/// depth and error, and the group it was simplified from.
struct CutCluster {
    cluster: Cluster,
    depth: u8,
    group_error: f32,
    group: u32,
    finer: Option<u32>,
}

/// Which pages and node pages of a mesh are resident.
struct Resident {
    pages: Vec<bool>,
    nodes: Vec<bool>,
}

impl CpuMesh {
    /// Every cluster by mesh cluster index.
    fn clusters(&self) -> Vec<CutCluster> {
        let mut out: Vec<CutCluster> = self
            .pages
            .iter()
            .flat_map(|page| {
                let view = PageView::new(page, &self.header).unwrap();
                (0..view.cluster_count()).map(move |i| {
                    let cluster = view.cluster(i);
                    CutCluster {
                        cluster,
                        depth: 0,
                        group_error: 0.0,
                        group: 0,
                        finer: (cluster.from_group != Cluster::SOURCE)
                            .then_some(cluster.from_group),
                    }
                })
            })
            .collect();
        for (gi, g) in (0u32..).zip(&self.groups) {
            for i in 0..u32::from(g.cluster_count) {
                let c = &mut out
                    [(self.firsts[g.page as usize] + u32::from(g.first_cluster) + i) as usize];
                c.depth = g.depth;
                c.group_error = g.error;
                c.group = gi;
            }
        }
        out
    }

    fn all_resident(&self) -> Resident {
        Resident {
            pages: vec![true; self.pages.len()],
            nodes: vec![true; self.node_pages],
        }
    }

    /// Whether group `g` can be drawn while `r` is resident: its page and the node page that
    /// holds its leaf are.
    fn drawable(&self, g: u32, r: &Resident) -> bool {
        let g = &self.groups[g as usize];
        r.pages[g.page as usize] && r.nodes[g.node_page as usize]
    }

    /// Whether `c` stands in for the clusters of the group it was simplified from: that group
    /// cannot be drawn while `r` is resident.
    fn stands_in(&self, c: &CutCluster, r: &Resident) -> bool {
        c.finer.is_some_and(|f| !self.drawable(f, r))
    }
}

/// A copy's mesh-space bounds as the traversal projects them from `eye`.
struct Projector {
    linear: Mat3,
    at: Vec3,
    eye: Vec3,
    scale: f32,
}

impl Projector {
    fn new(item: &Item, eye: Vec3) -> Self {
        let scale = (0..3)
            .map(|i| item.linear.col(i).length())
            .fold(0.0, f32::max);
        Self {
            linear: item.linear,
            at: item.at,
            eye,
            scale,
        }
    }

    /// `error` over `sphere` in pixels at the eye.
    fn error(&self, sphere: &orr_pack::Sphere, error: f32) -> f32 {
        let c = self.linear * Vec3::from(sphere.center) + self.at - self.eye;
        let d = (c.length() - sphere.radius * self.scale).max(0.01);
        error * self.scale / d * PIXELS_PER_RADIAN
    }
}

/// The mesh clusters the cluster rule takes at `threshold` pixels with errors scaled by
/// `scale` while `resident` is: of groups that can be drawn, with a parent error over the
/// threshold, and an own error at or under it, a source cluster, or standing in for the
/// clusters of a group that cannot be drawn. Also those whose own or parent error lies within
/// 1e-4 of the threshold, where f32 rounding may decide either way.
fn rule_cut(
    m: &CpuMesh,
    clusters: &[CutCluster],
    p: &Projector,
    threshold: f32,
    scale: f32,
    resident: &Resident,
) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let (mut cut, mut near) = (BTreeSet::new(), BTreeSet::new());
    for (i, c) in clusters.iter().enumerate() {
        let k = &c.cluster;
        let own = p.error(&k.self_sphere, k.self_error * scale);
        let parent = p.error(&k.parent_sphere, k.parent_error * scale);
        let fine = own <= threshold || c.depth == 0 || m.stands_in(c, resident);
        if m.drawable(c.group, resident) && fine && parent > threshold {
            cut.insert(i as u32);
        }
        let close = |e: f32| e.is_finite() && (e - threshold).abs() <= 1e-4 * threshold;
        if close(own) || close(parent) {
            near.insert(i as u32);
        }
    }
    (cut, near)
}

/// The mesh clusters debug level `level` holds while `resident` is: of groups that can be
/// drawn, those at its depth, those of shallower terminal groups, and those of deeper groups
/// standing in for the clusters of a group that cannot be drawn.
fn level_cut(
    m: &CpuMesh,
    clusters: &[CutCluster],
    level: u8,
    resident: &Resident,
) -> BTreeSet<u32> {
    (0..clusters.len() as u32)
        .filter(|&i| {
            let c = &clusters[i as usize];
            let held = c.depth == level || (c.depth < level && c.group_error.is_infinite());
            let stands = c.depth > level && m.stands_in(c, resident);
            m.drawable(c.group, resident) && (held || stands)
        })
        .collect()
}

/// Panics unless `drawn` covers the mesh's surface once. The clusters simplified from a group
/// each are drawn or lie in a group whose own place coarser drawn clusters take, all of them
/// or none of them, and so take that group's place or leave it to its own clusters; every
/// source cluster is drawn or has its place taken.
fn assert_whole(m: &CpuMesh, clusters: &[CutCluster], drawn: &BTreeSet<u32>, what: &str) {
    let mut from: Vec<Vec<u32>> = vec![Vec::new(); m.groups.len()];
    for (i, c) in (0u32..).zip(clusters) {
        if let Some(f) = c.finer {
            from[f as usize].push(i);
        }
    }
    let mut order: Vec<usize> = (0..m.groups.len()).collect();
    order.sort_by_key(|&g| std::cmp::Reverse(m.groups[g].depth));
    let mut taken = vec![false; m.groups.len()];
    let covers = |c: u32, taken: &[bool]| {
        u32::from(drawn.contains(&c)) + u32::from(taken[clusters[c as usize].group as usize])
    };
    for g in order {
        if m.groups[g].error.is_infinite() {
            continue;
        }
        let counts: BTreeSet<u32> = from[g].iter().map(|&c| covers(c, &taken)).collect();
        assert!(
            counts.len() == 1 && counts.iter().all(|&n| n <= 1),
            "{what}: the clusters simplified from group {g} are covered {counts:?} times"
        );
        taken[g] = counts.contains(&1);
    }
    for (i, c) in (0u32..).zip(clusters) {
        if c.depth == 0 {
            assert_eq!(covers(i, &taken), 1, "{what}: source cluster {i}");
        }
    }
}

/// The scene's cut tables after a frame.
struct CutRead {
    counters: Vec<u32>,
    candidates: Vec<ClusterWork>,
    slots: Vec<SlotCutGpu>,
    /// The clusters of the ray-tracing cuts built this frame: the slot that built one,
    /// cluster id.
    entries: Vec<[u32; 2]>,
}

/// `SlotCutGpu::rt_cut` of a slot tracing a cut built this frame.
const RT_CUT_BUILT: u32 = 255;

fn read_cut(view: &RasterView, slots: u32) -> CutRead {
    let root: SceneRoot = read_buffer(view.scene, 1)[0];
    let cut: CutGpu = read_buffer(root.cut, 1)[0];
    let counters: Vec<u32> = read_buffer(cut.counters, 32);
    let n = counters[2].min(cut.candidate_capacity);
    let e = counters[5];
    CutRead {
        candidates: read_buffer(cut.candidates, u64::from(n.max(1)))[..n as usize].to_vec(),
        slots: read_buffer(cut.slots, u64::from(slots)),
        entries: read_buffer(cut.entries, u64::from(e.max(1)))[..e as usize].to_vec(),
        counters,
    }
}

impl CutRead {
    /// The mesh clusters of `slot`'s raster candidates.
    fn raster(&self, m: &CpuMesh, slot: u32) -> BTreeSet<u32> {
        self.candidates
            .iter()
            .filter(|w| w.instance == slot)
            .flat_map(|w| {
                let first = m.firsts[w.page as usize] + w.first;
                (0..32)
                    .filter(move |b| w.mask >> b & 1 != 0)
                    .map(move |b| first + b)
            })
            .collect()
    }

    /// The mesh clusters of `slot`'s gathered ray-tracing cut, by the cluster ids of the
    /// mesh's resident `pages` from their `bases`: the entries of the slot that built it, or
    /// for a cut equal to a cached one, which builds nothing, `like`, which it must equal.
    fn ray(
        &self,
        m: &CpuMesh,
        slot: u32,
        pages: &[u64],
        bases: &[u32],
        like: &BTreeSet<u32>,
    ) -> BTreeSet<u32> {
        let s = self.slots[slot as usize];
        if s.rt_cut != RT_CUT_BUILT {
            assert_eq!(
                (s.rt_count, s.rt_hash),
                (like.len() as u32, cut_hash(like)),
                "slot {slot}'s gathered cut, which built nothing"
            );
            return like.clone();
        }
        let count = |p: usize| {
            PageView::new(&m.pages[p], &m.header)
                .unwrap()
                .cluster_count()
        };
        self.entries
            .iter()
            .filter(|e| e[0] == s.owner)
            .map(|e| {
                let p = (0..pages.len())
                    .find(|&p| pages[p] != 0 && (bases[p]..bases[p] + count(p)).contains(&e[1]))
                    .expect("a resident cluster id");
                m.firsts[p] + e[1] - bases[p]
            })
            .collect()
    }
}

/// The traversal's order-free hash of a set of mesh clusters.
fn cut_hash(clusters: &BTreeSet<u32>) -> [u32; 2] {
    let clusters: Vec<u32> = clusters.iter().copied().collect();
    orr_scene::testing::cut_hash(&clusters)
}

/// A world that traces the cut it rasters, every copy's own, at the same detail wherever the
/// views saw it.
fn traversed_world() -> World {
    World::with(&RendererDesc {
        rt_merge_pixels: 0.0,
        culled_error_scale: 1.0,
        ..RendererDesc::default()
    })
}

/// A view's render as the traversal tests what it saw: its constants, the camera, and its HZB's
/// mips.
struct SeenRender {
    v: orr_scene::ViewConstants,
    eye: DVec3,
    hzb: Vec<Vec<f32>>,
}

impl SeenRender {
    fn of(view: &RasterView, eye: DVec3) -> Self {
        let mips = view.hzb.desc().mips;
        Self {
            v: read_buffer(view.view, 1)[0],
            eye,
            hzb: (0..mips)
                .map(|m| {
                    let bytes =
                        read_texture(&view.hzb, &TextureRegion::whole_mip(view.hzb.desc(), m));
                    bytemuck::pod_collect_to_vec(&bytes)
                })
                .collect(),
        }
    }

    /// Whether `traverse.slang`'s hidden test finds the mesh-space sphere `s` of `item`, placed
    /// with its radius times `grow`, unseen: outside the frustum or with its box behind the
    /// HZB (`visibility.slang` `sphere_unseen`), in f64.
    fn unseen(&self, item: &Item, s: &orr_pack::Sphere, grow: f64) -> bool {
        let rows = |r: [[f32; 4]; 4]| r.map(|row| DVec4::from_array(row.map(f64::from)));
        let m = rows(self.v.clip_from_view_jittered);
        let r = self
            .v
            .view_from_world_rotation
            .map(|row| DVec3::new(row[0].into(), row[1].into(), row[2].into()));
        let l = item.linear.as_dmat3();
        let a = l.transpose() * l;
        let scale = (0..3)
            .map(|i| a.row(i).abs().element_sum())
            .fold(0.0, f64::max)
            .sqrt();
        let c = l * DVec3::from(s.center.map(f64::from)) + item.at.as_dvec3() - self.eye;
        let radius = f64::from(s.radius) * scale * grow;
        let view = |x: DVec3| DVec3::new(r[0].dot(x), r[1].dot(x), r[2].dot(x));
        let p = view(c).extend(1.0);
        for plane in [
            m[3] + m[0],
            m[3] - m[0],
            m[3] + m[1],
            m[3] - m[1],
            m[3] - m[2],
        ] {
            if plane.dot(p) < -radius * plane.truncate().length() {
                return true;
            }
        }
        let extent = DVec2::new(f64::from(EXTENT[0]), f64::from(EXTENT[1]));
        let (mut lo, mut hi, mut nearest) = (DVec2::splat(1e30), DVec2::splat(-1e30), 0.0f64);
        for i in 0..8 {
            let corner = c + DVec3::new(
                if i & 1 != 0 { radius } else { -radius },
                if i & 2 != 0 { radius } else { -radius },
                if i & 4 != 0 { radius } else { -radius },
            );
            let v = view(corner).extend(1.0);
            let clip = DVec4::new(m[0].dot(v), m[1].dot(v), m[2].dot(v), m[3].dot(v));
            if !(clip.w > 0.0 && clip.z < clip.w) {
                return false;
            }
            let xy = (DVec2::new(clip.x, clip.y) / clip.w * 0.5 + 0.5) * extent;
            lo = lo.min(xy);
            hi = hi.max(xy);
            nearest = nearest.max(clip.z / clip.w);
        }
        let lo = (lo - 1.0).max(DVec2::ZERO);
        let hi = (hi + 1.0).min(extent - 1.0);
        if lo.x > hi.x || lo.y > hi.y {
            return false;
        }
        let size = hi - lo;
        let mips = self.hzb.len() as u32;
        let mip = ((size.x.max(size.y).max(1.0).log2().ceil() - 1.0).max(0.0) as u32).min(mips - 1);
        let [w, h] = EXTENT.map(|e| (e >> mip).max(1));
        let (a, b) = (
            [
                (lo.x as u32 >> mip).min(w - 1),
                (lo.y as u32 >> mip).min(h - 1),
            ],
            [
                (hi.x as u32 >> mip).min(w - 1),
                (hi.y as u32 >> mip).min(h - 1),
            ],
        );
        let texels = &self.hzb[mip as usize];
        let farthest = (a[1]..=b[1])
            .flat_map(|y| (a[0]..=b[0]).map(move |x| texels[(y * w + x) as usize]))
            .fold(1.0f32, f32::min);
        nearest < f64::from(farthest)
    }

    /// [`Self::unseen`], or None where a sphere a little smaller or larger is decided
    /// otherwise, so the GPU's f32 may decide either way.
    fn hidden(&self, item: &Item, s: &orr_pack::Sphere) -> Option<bool> {
        let (small, large) = (
            self.unseen(item, s, 1.0 - 1e-4),
            self.unseen(item, s, 1.0 + 1e-4),
        );
        (small == large).then_some(small)
    }
}

/// [`rule_cut`] where the traversal saw `seen`: an error over the threshold is over it at
/// `culled` times the threshold over a group sphere `seen` hides. Also returns the clusters the
/// hidden threshold took out or put in, and the near clusters take those whose decision hangs
/// on a sphere whose test is near a tie, or on an error near either threshold.
fn seen_rule_cut(
    m: &CpuMesh,
    clusters: &[CutCluster],
    item: &Item,
    threshold: f32,
    culled: f32,
    seen: &SeenRender,
    resident: &Resident,
) -> (BTreeSet<u32>, BTreeSet<u32>, BTreeSet<u32>) {
    let p = Projector::new(item, seen.eye.as_vec3());
    let (mut cut, mut near, mut decided) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    let mut tests = std::collections::HashMap::new();
    let mut hidden = |s: &orr_pack::Sphere| {
        let key = (s.center.map(f32::to_bits), s.radius.to_bits());
        *tests.entry(key).or_insert_with(|| seen.hidden(item, s))
    };
    let hidden_threshold = threshold * culled;
    for (i, c) in clusters.iter().enumerate() {
        let k = &c.cluster;
        let own = p.error(&k.self_sphere, k.self_error);
        let parent = p.error(&k.parent_sphere, k.parent_error);
        // Over the threshold its sphere takes; None where that hangs on a near tie.
        let mut over = |e: f32, s: &orr_pack::Sphere| -> Option<bool> {
            if e <= threshold {
                Some(false)
            } else if e > hidden_threshold {
                Some(true)
            } else {
                hidden(s).map(|h| !h)
            }
        };
        let parent_over = over(parent, &k.parent_sphere);
        let own_over = if c.depth == 0 || m.stands_in(c, resident) {
            Some(false)
        } else {
            over(own, &k.self_sphere)
        };
        let close = |e: f32| {
            e.is_finite()
                && [threshold, hidden_threshold]
                    .iter()
                    .any(|&t| (e - t).abs() <= 1e-4 * t)
        };
        if parent_over.is_none() || own_over.is_none() || close(own) || close(parent) {
            near.insert(i as u32);
        }
        let fine = own_over == Some(false) || c.depth == 0 || m.stands_in(c, resident);
        let taken = m.drawable(c.group, resident) && fine && parent_over == Some(true);
        if taken {
            cut.insert(i as u32);
        }
        let uniform = m.drawable(c.group, resident)
            && (own <= threshold || c.depth == 0 || m.stands_in(c, resident))
            && parent > threshold;
        if taken != uniform {
            decided.insert(i as u32);
        }
    }
    (cut, near, decided)
}

/// Panics unless `got` is `want` outside `near`.
fn assert_cut(got: &BTreeSet<u32>, want: &BTreeSet<u32>, near: &BTreeSet<u32>, what: &str) {
    let differ: Vec<u32> = want.symmetric_difference(got).copied().collect();
    assert!(
        differ.iter().all(|c| near.contains(c)),
        "{what}: the cut differs from the reference at {differ:?} ({} against {})",
        got.len(),
        want.len()
    );
}

/// The DAG ground tilted away from the camera, so its cut spans several levels, and balls
/// at three distances.
fn dag_items() -> Vec<Item> {
    let mut items = vec![Item {
        model: "dag ground",
        linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
        at: Vec3::new(0.0, -1.0, -12.0),
    }];
    for (i, z) in [-2.0f32, -9.0, -30.0].into_iter().enumerate() {
        items.push(Item {
            model: "ball",
            linear: Mat3::from_rotation_y(i as f32),
            at: Vec3::new(1.5 - i as f32, 0.6, z),
        });
    }
    items
}

#[test]
fn the_traversal_takes_the_cluster_rule_over_every_cluster() {
    let _serial = serial();
    let mut w = traversed_world();
    let items = dag_items();
    let placed = place(&mut w, &items);
    let eye = Vec3::new(0.3, 1.2, 4.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    // The second frame traces at the RT threshold itself: every copy was drawn in the first.
    shoot(
        &mut w,
        &placed,
        &Shot::at(eye.as_dvec3(), target.as_dvec3(), 1),
    );
    let g = shoot(
        &mut w,
        &placed,
        &Shot::at(eye.as_dvec3(), target.as_dvec3(), 2),
    );
    let cut = read_cut(&g.view, placed.count);
    let mut levels = BTreeSet::new();
    for (slot, item) in items.iter().enumerate() {
        let slot = placed.ranges[slot].first;
        let m = CpuMesh::of(item.model);
        let clusters = m.clusters();
        let p = Projector::new(item, eye);
        let (want, near) = rule_cut(&m, &clusters, &p, 1.0, 1.0, &m.all_resident());
        let got = cut.raster(&m, slot);
        let differ: Vec<u32> = want.symmetric_difference(&got).copied().collect();
        assert!(
            differ.iter().all(|c| near.contains(c)),
            "{}: raster cut differs from the rule at {differ:?} ({} against {})",
            item.model,
            got.len(),
            want.len()
        );
        let s = cut.slots[slot as usize];
        assert_eq!(
            s.rt_count as usize,
            got.len(),
            "{}: ray-tracing cut size",
            item.model
        );
        if differ.is_empty() {
            assert_eq!(
                s.rt_hash,
                cut_hash(&want),
                "{}: ray-tracing cut",
                item.model
            );
        }
        levels.extend(want.iter().map(|&c| clusters[c as usize].depth));
    }
    assert!(levels.len() >= 3, "the cut spans levels {levels:?}");
    assert_eq!(cut.counters[8], 0, "no node dropped");
    assert_clean();
}

#[test]
fn traced_dag_cuts_hit_what_the_raster_drew_and_close_without_cracks() {
    let _serial = serial();
    let mut w = traversed_world();
    let items = dag_items();
    let placed = place(&mut w, &items);
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    shoot(&mut w, &placed, &Shot::at(eye, target, 1));
    let g = shoot(&mut w, &placed, &Shot::at(eye, target, 2));
    let cuts = last_stats(&mut w, 3).cuts;
    assert!(cuts.rt_built >= 1, "built BLAS: {cuts:?}");
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, covered) = parity(&g, &r, 1e-4);
    let total = r.traced.len();
    assert!(covered > total / 3, "{covered} of {total} pixels covered");
    assert!(
        matched as f64 >= 0.999 * total as f64,
        "{matched} of {total} pixels agree"
    );
    // A crack shows as background between ground pixels across a row and a column.
    let ground = placed.ranges[0].first + 1;
    let [wd, ht] = EXTENT.map(|e| e as usize);
    let id = |x: usize, y: usize| g.ids[y * wd + x];
    let mut cracks = 0;
    for y in 1..ht - 1 {
        for x in 1..wd - 1 {
            let closed = |a: u32, b: u32| a == ground && b == ground;
            if id(x, y) == 0
                && closed(id(x - 1, y), id(x + 1, y))
                && closed(id(x, y - 1), id(x, y + 1))
            {
                cracks += 1;
            }
        }
    }
    assert_eq!(cracks, 0, "background pixels inside the ground");
    assert_clean();
}

/// The first path vertex on the ground, a copy tracing a cut it gathered and built, offsets its
/// rays by that cut's largest cluster error times the copy's scale, as the pixel probes read
/// it: the cut's clusters read back from the frame's built entries, far finer than the
/// coarsest cached cut.
#[test]
fn a_copy_tracing_its_built_cut_offsets_its_rays_by_that_cuts_largest_error() {
    use orr::orr_lighting::OriginProbes;
    let _serial = serial();
    let mut w = traversed_world();
    let items = dag_items();
    let placed = place(&mut w, &items);
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    shoot(&mut w, &placed, &Shot::at(eye, target, 1));
    let model = w.model("dag ground");
    let m = CpuMesh::of("dag ground");
    let slot = placed.ranges[0].first;
    let pixel = [EXTENT[0] / 2, EXTENT[1] * 3 / 4];
    let mut origins = OriginProbes::default();
    let list = [placed.scene];
    let shot = Shot::at(eye, target, 2);
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(shot.frame),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    f.scene(placed.scene).unwrap().set_focus(eye);
    f.request_detail(placed.scene, PIXELS_PER_RADIAN).unwrap();
    f.update_scenes(&mut |_| {}).unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: placed.scene,
            history: placed.history,
            position: eye,
            rotation: look(Vec3::ZERO, (target - eye).as_vec3()),
            projection: Projection {
                fov_y: 0.9,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: Vec2::ZERO,
            camera_cut: false,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
    f.probe(&view, pixel).unwrap();
    let id = f.id();
    origins
        .record(
            device(),
            &KernelSet::NONE,
            f.list().unwrap(),
            id,
            view.scene,
            view.view,
            view.gbuffer,
            pixel,
        )
        .unwrap();
    wait(f.end(&[]).unwrap());
    let probe = w.scenes.probes(placed.history, id).remove(0);
    let origin = origins.take(id).remove(0).gpu;
    assert_eq!(probe.gpu.copy, slot, "the probe shows the ground:\n{probe}");
    let cut = read_cut(&view, placed.count);
    let s = cut.slots[slot as usize];
    assert_eq!(
        s.rt_cut, RT_CUT_BUILT,
        "the ground traces its built cut:\n{probe}"
    );
    let (pages, bases, _) = mesh_pages(&w, model);
    let clusters = m.clusters();
    let raster = cut.raster(&m, slot);
    let largest = cut
        .ray(&m, slot, &pages, &bases, &raster)
        .iter()
        .map(|&c| clusters[c as usize].cluster.self_error)
        .fold(0.0, f32::max);
    let scale = Projector::new(&items[0], Vec3::ZERO).scale;
    let want = largest * scale;
    let coarsest = probe.gpu.rt_cut_count - 1;
    let (_, meshes, _) = orr_scene::testing::global_tables(&w.scenes);
    let mesh: MeshGpu = read_buffer(meshes.offset(u64::from(probe.gpu.mesh)), 1)[0];
    let cached: orr_scene::RtCutGpu = read_buffer(mesh.rt_cuts.offset(u64::from(coarsest)), 1)[0];
    assert!(
        want > 0.0 && want < cached.error * scale,
        "{want} m against {}",
        cached.error
    );
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * b;
    assert!(close(probe.gpu.built_error, largest), "{probe}");
    assert!(
        close(origin.traced_error, want),
        "traced {} m, want {want} m",
        origin.traced_error
    );
    assert!(
        close(origin.origin_error, origin.raster_error.max(want)),
        "origin error {} m, want {want} m",
        origin.origin_error
    );
    // The pixel's traced camera ray meets the copy it drew, within that error of the drawn point.
    let traced = origin.traced;
    assert!(traced.hit != 0 && traced.copy == slot, "traced {traced:?}");
    let at = |v: [f32; 4]| Vec3::new(v[0], v[1], v[2]);
    let met = at(origin.camera_origin) + traced.t * at(origin.camera_direction);
    let apart = met.distance(at(origin.position));
    assert!(
        apart <= origin.origin_error,
        "the traced hit lies {apart} m from the drawn one"
    );
    assert_clean();
}

/// `CutGpu.counters` of the hidden test: spheres tested, clusters it decided.
const CUT_HIDDEN_TESTS: usize = 29;
const CUT_HIDDEN_RULES: usize = 23;

/// A still view of the streaming ground that a wall and the frustum hide in part, at a
/// threshold fine enough that its cut spans many levels. The first frame, with no view yet,
/// takes the cluster rule; each later one takes the culled threshold over the group spheres
/// the frame before hid, as a CPU reference of the hidden test over that frame's camera and HZB
/// says, both cuts whole, while the pages it wants land. Some clusters are the hidden
/// threshold's. Once the loads settle the cut holds still, and every cluster drawn to a pixel
/// is the rule's.
#[test]
fn hidden_groups_take_the_culled_threshold_and_a_still_view_draws_the_rule() {
    let _serial = serial();
    const ERROR_PIXELS: f32 = 0.004;
    let bytes = fine_ground_pack();
    // A band between the thresholds wide enough to take groups of several levels.
    let desc = RendererDesc {
        raster_error_pixels: ERROR_PIXELS,
        rt_error_pixels: ERROR_PIXELS,
        rt_merge_pixels: 0.0,
        culled_error_scale: 4.0,
        ..RendererDesc::default()
    };
    let mut w = World::with_pack(&desc, bytes);
    let items = walled_fine_ground();
    let placed = place(&mut w, &items);
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    let m = CpuMesh::of_pack(bytes, "fine ground");
    let clusters = m.clusters();
    let p = Projector::new(&items[0], eye.as_vec3());
    let slot = placed.ranges[0].first;
    let model = w.model("fine ground");
    let culled = desc.culled_error_scale;
    let (uniform, uniform_near) = rule_cut(&m, &clusters, &p, ERROR_PIXELS, 1.0, &m.all_resident());
    let mut seen: Option<SeenRender> = None;
    let (mut still, mut idle, mut taken) = (None, 0, 0);
    let start = Instant::now();
    for frame in 1.. {
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the loads never settled"
        );
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        let cut = read_cut(&g.view, placed.count);
        let (pages, bases, nodes) = mesh_pages(&w, model);
        let resident = Resident {
            pages: pages.iter().map(|&a| a != 0).collect(),
            nodes: nodes.iter().map(|&a| a != 0).collect(),
        };
        let raster = cut.raster(&m, slot);
        let ray = cut.ray(&m, slot, &pages, &bases, &raster);
        let what = format!("frame {frame}");
        assert_whole(&m, &clusters, &raster, &format!("{what}: raster"));
        assert_whole(&m, &clusters, &ray, &format!("{what}: ray tracing"));
        match &seen {
            None => {
                let (want, near) = rule_cut(&m, &clusters, &p, ERROR_PIXELS, 1.0, &resident);
                assert_cut(&raster, &want, &near, &what);
            }
            Some(s) => {
                let (want, near, decided) =
                    seen_rule_cut(&m, &clusters, &items[0], ERROR_PIXELS, culled, s, &resident);
                assert_cut(&raster, &want, &near, &format!("{what}: raster"));
                // The copy was drawn the frame before, so it traces at the raster's thresholds.
                assert_cut(&ray, &want, &near, &format!("{what}: ray tracing"));
                let by_hidden = decided.difference(&near).count();
                taken = taken.max(by_hidden);
                assert!(
                    by_hidden == 0
                        || cut.counters[CUT_HIDDEN_TESTS] > 0 && cut.counters[CUT_HIDDEN_RULES] > 0,
                    "{what}: counters {:?}",
                    cut.counters
                );
            }
        }
        let r = w.scenes.residency();
        let loading =
            r.loads > 0 || r.class(orr_scene::ResidencyClass::MeshPages).loading_bytes > 0;
        idle = if loading { 0 } else { idle + 1 };
        if idle >= 8 {
            let ground = slot + 1;
            let drawn: BTreeSet<u32> = (0..g.ids.len())
                .filter(|&i| g.ids[i] == ground)
                .map(|i| g.clusters[i] - 1)
                .collect();
            assert!(!drawn.is_empty(), "{what}: the ground shows");
            let off: Vec<u32> = drawn
                .iter()
                .filter(|c| !uniform.contains(c) && !uniform_near.contains(c))
                .copied()
                .collect();
            assert!(
                off.is_empty(),
                "{what}: drawn clusters off the rule: {off:?}"
            );
            let s = still.get_or_insert_with(|| raster.clone());
            assert_eq!(&raster, s, "{what}: the cut moved");
        }
        seen = Some(SeenRender::of(&g.view, eye));
        if idle >= 12 {
            break;
        }
    }
    assert!(taken > 0, "no cluster is the hidden threshold's");
    assert_eq!(w.scenes.residency().failed, 0);
    assert_clean();
}

/// The figures of the last completed frame, taken in by an empty one.
fn last_stats(w: &mut World, frame: u32) -> orr_scene::FrameStats {
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(frame),
            scenes: &[],
            low_latency: None,
        })
        .unwrap();
    wait(f.end(&[]).unwrap());
    w.scenes.stats().clone()
}

/// Frames by which the work lists, from one entry, hold what the DAG scene wants: each frame's
/// readback grows them for the next to half again what it wanted, and what a list drops hides
/// only what the lists after it want.
const GROWTH_FRAMES: u32 = 4;

/// Two mesh draws of two rows each over three rows and a few groups of list entries: each draw
/// keeps to its whole rows, and every entry is drawn once.
#[test]
fn list_draws_cover_every_entry_once_past_a_row_and_a_draw() {
    let _serial = serial();
    const GRID_X: u32 = 65535;
    let d = device();
    let up = MemoryKind::Upload;
    let (draws, draw_groups, count) = (2, 2 * GRID_X, 3 * GRID_X + 7);
    let args = d
        .gpu_malloc::<u32>(up, 3 * u64::from(draws), "draw args")
        .unwrap();
    let mut raster: RasterGpu = bytemuck::Zeroable::zeroed();
    raster.draw_args = args.gpu();
    raster.draws = draws;
    raster.draw_groups = draw_groups;
    let mut record = d.gpu_malloc::<RasterGpu>(up, 1, "raster").unwrap();
    record.as_mut_slice().unwrap()[0] = raster;
    let mut hits = d
        .gpu_malloc::<u32>(up, u64::from(count + 1), "hits")
        .unwrap();
    hits.as_mut_slice().unwrap().fill(0);
    let root = |first: u32| {
        let mut r = d.gpu_malloc::<grid::Root>(up, 1, "grid root").unwrap();
        r.as_mut_slice().unwrap()[0] = grid::Root {
            raster: record.gpu(),
            hits: hits.gpu(),
            count,
            first,
            pad0: 0,
            pad1: 0,
        };
        r
    };
    let sized = root(0);
    let roots: Vec<_> = (0..draws).map(|i| root(i * draw_groups)).collect();
    let (size, mark) = (
        pipeline(&grid::MODULE, "size"),
        pipeline(&grid::MODULE, "mark"),
    );
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.dispatch(&size, sized.gpu(), [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
    for (i, r) in (0u64..).zip(&roots) {
        list.dispatch_indirect(&mark, r.gpu(), args.gpu().offset(3 * i).cast());
    }
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    let grids: Vec<u32> = read_buffer(args.gpu(), 3 * u64::from(draws));
    assert_eq!(grids, [GRID_X, 2, 1, GRID_X, 2, 1]);
    let hits: Vec<u32> = read_buffer(hits.gpu(), u64::from(count + 1));
    let missed: Vec<usize> = (0..count as usize).filter(|&i| hits[i] != 1).collect();
    assert!(missed.is_empty(), "{} entries not drawn once", missed.len());
    assert_eq!(
        hits[count as usize],
        2 * draw_groups - count,
        "the last row's groups past it"
    );
    assert_clean();
}

/// `CutGpu.counters` (`cut.slang`) of the gathered cuts that trace cached ones: past the
/// admission, of a traversal that dropped nodes, holding no cluster.
const CUT_FALLBACK: usize = 14;
const CUT_LOST: usize = 26;
const CUT_EMPTY: usize = 27;
/// `SlotCutGpu::flags` of a slot that gathered its cut and traces its cached cut instead.
const CUT_FELL_BACK: u32 = 32;

#[test]
fn work_lists_far_too_small_grow_until_nothing_drops_and_the_cut_takes_the_cluster_rule() {
    let _serial = serial();
    let mut w = World::with(&RendererDesc {
        rt_merge_pixels: 0.0,
        culled_error_scale: 1.0,
        budgets: Budgets {
            initial_traversal_nodes: 1,
            initial_candidate_items: 1,
            initial_visible_clusters: 1,
            initial_rt_cut_clusters: 1,
            initial_rt_cut_blas: 1,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    });
    let items = dag_items();
    let placed = place(&mut w, &items);
    let eye = Vec3::new(0.3, 1.2, 4.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    // Per frame: nodes, candidate items, visible clusters and second-pass work dropped, and
    // gathered cuts tracing cached ones.
    let mut drops = Vec::new();
    let mut last = None;
    for frame in 1..=GROWTH_FRAMES + 3 {
        let g = shoot(
            &mut w,
            &placed,
            &Shot::at(eye.as_dvec3(), target.as_dvec3(), frame),
        );
        let c = read_cut(&g.view, placed.count).counters;
        drops.push([
            c[8],
            c[9],
            g.counts[1],
            g.counts[2],
            c[CUT_FALLBACK],
            c[CUT_LOST],
            c[CUT_EMPTY],
        ]);
        last = Some(g);
    }
    let clean = drops.iter().position(|d| *d == [0; 7]);
    assert!(
        drops[0][..3].iter().all(|&d| d > 0)
            && clean
                .is_some_and(|f| f < GROWTH_FRAMES as usize && drops[..f].iter().any(|d| d[4] > 0)),
        "drops by frame: {drops:?}"
    );
    assert!(
        drops[clean.unwrap()..].iter().all(|d| *d == [0; 7]),
        "drops by frame: {drops:?}"
    );
    let cut = read_cut(&last.unwrap().view, placed.count);
    let mut built = 0;
    for (slot, item) in items.iter().enumerate() {
        let slot = placed.ranges[slot].first;
        let m = CpuMesh::of(item.model);
        let clusters = m.clusters();
        let p = Projector::new(item, eye);
        let (want, near) = rule_cut(&m, &clusters, &p, 1.0, 1.0, &m.all_resident());
        let got = cut.raster(&m, slot);
        let differ: Vec<u32> = want.symmetric_difference(&got).copied().collect();
        assert!(
            differ.iter().all(|c| near.contains(c)),
            "{}: raster cut differs from the rule at {differ:?} ({} against {})",
            item.model,
            got.len(),
            want.len()
        );
        // Ray tracing takes the raster's threshold, so the copy traces what it draws.
        let (pages, bases, _) = mesh_pages(&w, w.model(item.model));
        let ray = cut.ray(&m, slot, &pages, &bases, &got);
        assert_eq!(ray, got, "{}: traced cut", item.model);
        built += u32::from(cut.slots[slot as usize].rt_cut == RT_CUT_BUILT);
    }
    assert!(built > 0, "no copy built its cut");
    assert_clean();
}

/// Tilted grounds at growing distances, each cut over several levels, under tables of the
/// built cuts that hold two BLAS and may not grow: the same copies build their cuts every
/// frame of a still view, the nearest, and the farther ones trace cached cuts.
#[test]
fn past_tables_that_cannot_grow_the_farthest_copies_trace_cached_cuts_every_frame() {
    let _serial = serial();
    let mut w = World::with(&RendererDesc {
        rt_merge_pixels: 0.0,
        // Fine enough that each copy's cut mixes the levels several groups make up: a cut that
        // is one whole level may equal a cached cut, which builds nothing.
        rt_error_pixels: 1.0 / 32.0,
        budgets: Budgets {
            initial_rt_cut_blas: 2,
            rt_cut_memory: PoolSize::Bytes(0),
            ..Budgets::default()
        },
        ..RendererDesc::default()
    });
    // Each copy's bounds lie farther from the focus than the last's: copies whose bounds hold
    // the focus are equally near, and build by their slots, not by how far they are. The
    // farthest takes the first slot, so the order of the slots is not that of the distances.
    let items: Vec<Item> = [-18.2f32, -19.8, -21.5, -24.4, -30.1]
        .into_iter()
        .enumerate()
        .map(|(i, z)| Item {
            model: "dag ground",
            linear: Mat3::from_rotation_y(0.4 * i as f32)
                * Mat3::from_rotation_x(0.25)
                * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
            at: Vec3::new(3.0 * i as f32 - 6.0, -1.0, z),
        })
        .rev()
        .collect();
    let placed = place(&mut w, &items);
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    let mut sets = Vec::new();
    for frame in 1..=8 {
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        if frame < 3 {
            continue;
        }
        let cut = read_cut(&g.view, placed.count);
        // Per copy that wanted a BLAS of its own: its distance and whether it built one.
        let mut wanted: Vec<(f32, bool)> = placed
            .ranges
            .iter()
            .map(|r| cut.slots[r.first as usize])
            .filter(|s| s.rt_cut == RT_CUT_BUILT || s.flags & CUT_FELL_BACK != 0)
            .map(|s| (s.rt_distance, s.rt_cut == RT_CUT_BUILT))
            .collect();
        wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let built = wanted.iter().filter(|c| c.1).count();
        assert!(
            built == 2 && wanted.len() > built,
            "frame {frame}: copies by distance and whether they built: {wanted:?}"
        );
        assert!(
            wanted[..built].iter().all(|c| c.1),
            "frame {frame}: a farther copy built before a nearer one: {wanted:?}"
        );
        assert_eq!(cut.counters[CUT_FALLBACK] as usize, wanted.len() - built);
        sets.push(wanted);
    }
    assert!(
        sets.windows(2).all(|s| s[0] == s[1]),
        "the copies that build change between frames: {sets:?}"
    );
    assert_clean();
}

/// `CutGpu.counters` (`cut.slang`): the CLAS handed to the frame's BLAS builds, and the builds.
const CUT_CLAS: usize = 5;
const CUT_JOBS: usize = 6;

/// Tilted grounds whose bounds all hold the focus, so their distance tells none apart, under
/// tables of the built cuts that may not grow: they hold two BLAS, or fewer CLAS than the cuts
/// take until those have coarsened to fit. Every frame the cuts build in the order of their
/// first slots up to the first that finds no room, under two BLAS the same two every frame,
/// and every copy has a BLAS to trace. The last world puts enough slots before the grounds'
/// that the admission's keys take a digit more.
#[test]
fn equally_near_cuts_fill_tables_that_cannot_grow_in_slot_order() {
    let _serial = serial();
    for (blas, clas, before) in [(2u32, 1u32 << 18, 0), (1024, 300, 0), (2, 1 << 18, 2100)] {
        let mut w = World::with(&RendererDesc {
            rt_merge_pixels: 0.0,
            budgets: Budgets {
                initial_rt_cut_clusters: clas,
                initial_rt_cut_blas: blas,
                rt_cut_memory: PoolSize::Bytes(0),
                ..Budgets::default()
            },
            ..RendererDesc::default()
        });
        let what = format!("{blas} BLAS, {clas} CLAS, {before} slots before");
        // Out of ray-tracing range behind the camera.
        let far = Item {
            model: "plain",
            linear: Mat3::IDENTITY,
            at: Vec3::new(0.0, 0.0, 500.0),
        };
        let grounds = [-4.0f32, -6.0, -8.0, -10.0, -12.0]
            .into_iter()
            .enumerate()
            .map(|(i, z)| Item {
                model: "dag ground",
                linear: Mat3::from_rotation_y(0.4 * i as f32)
                    * Mat3::from_rotation_x(0.25)
                    * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
                at: Vec3::new(3.0 * i as f32 - 6.0, -1.0, z),
            });
        let items: Vec<Item> = std::iter::repeat_n(far, before).chain(grounds).collect();
        let placed = place(&mut w, &items);
        let (models, meshes, _) = orr_scene::testing::global_tables(&w.scenes);
        let model = u64::from(w.model("dag ground").0);
        let model: ModelGpu = read_buffer(models.offset(model), 1)[0];
        let mesh: MeshGpu = read_buffer(meshes.offset(u64::from(model.mesh)), 1)[0];
        let cached: Vec<orr_scene::RtCutGpu> =
            read_buffer(mesh.rt_cuts, u64::from(mesh.rt_cut_count));
        let eye = DVec3::new(0.3, 1.2, 4.0);
        let target = DVec3::new(0.0, 0.0, -10.0);
        // Per frame: which copies built, and whether the CLAS ran out before the BLAS.
        let mut sets = Vec::new();
        let mut clas_short = false;
        for frame in 1..=8 {
            let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
            let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
            let tables: CutGpu = read_buffer(root.cut, 1)[0];
            let cut = read_cut(&g.view, placed.count);
            assert_eq!(
                (cut.counters[CUT_LOST], cut.counters[CUT_EMPTY]),
                (0, 0),
                "{what}, frame {frame}"
            );
            let slots: Vec<(u32, SlotCutGpu)> = placed.ranges[before..]
                .iter()
                .map(|r| (r.first, cut.slots[r.first as usize]))
                .collect();
            let wants = |s: &SlotCutGpu| s.rt_cut == RT_CUT_BUILT || s.flags & CUT_FELL_BACK != 0;
            // The cuts that want a BLAS, by their clusters: nearest distance, then first slot.
            let mut cuts = std::collections::BTreeMap::new();
            for (slot, s) in slots.iter().filter(|(_, s)| wants(s)) {
                let c = cuts
                    .entry((s.rt_hash, s.rt_count))
                    .or_insert((u32::MAX, u32::MAX));
                *c = (c.0.min(s.rt_distance.to_bits()), c.1.min(*slot));
            }
            let mut order: Vec<_> = cuts.iter().map(|(cut, key)| (*key, *cut)).collect();
            order.sort_unstable();
            let (mut used, mut fit) = (0, BTreeSet::new());
            for (n, &(_, cut)) in order.iter().enumerate() {
                let no_clas = used + cut.1 > tables.clas_capacity;
                if n as u32 == tables.job_capacity || no_clas {
                    clas_short |= no_clas && n > 0;
                    break;
                }
                used += cut.1;
                fit.insert(cut);
            }
            let jobs: Vec<u64> = read_buffer(tables.job_blas, u64::from(tables.job_capacity));
            for (slot, s) in &slots {
                let built = s.rt_cut == RT_CUT_BUILT;
                if wants(s) {
                    assert_eq!(
                        built,
                        fit.contains(&(s.rt_hash, s.rt_count)),
                        "{what}, frame {frame}: slot {slot} of cuts in order {order:?}"
                    );
                }
                let blas = if built {
                    jobs[s.rt_job as usize]
                } else {
                    cached[s.rt_cut as usize].blas
                };
                assert!(blas != 0, "{what}, frame {frame}: slot {slot} has no BLAS");
            }
            assert_eq!(
                (cut.counters[CUT_JOBS] as usize, cut.counters[CUT_CLAS]),
                (fit.len(), used),
                "{what}, frame {frame}: the builds and their CLAS"
            );
            if frame < 3 {
                continue;
            }
            if blas == 2 {
                assert!(
                    order.len() > 2 && order.iter().all(|c| c.0.0 == 0) && fit.len() == 2,
                    "frame {frame}: {} of the cuts {order:?} fit",
                    fit.len()
                );
            }
            sets.push(slots.iter().map(|(_, s)| s.rt_cut).collect::<Vec<_>>());
            if frame == 8 {
                // Camera rays reach every copy the raster drew, built or not.
                let r = rays(&g.view, &trace::MODULE, "pixels", 0);
                let hit: BTreeSet<u32> = r.traced.iter().map(|t| t.instance).collect();
                let drawn: BTreeSet<u32> = g.ids.iter().copied().filter(|&id| id != 0).collect();
                assert!(
                    drawn.len() > 2 && drawn.is_subset(&hit),
                    "{what}: drawn {drawn:?}, traced {hit:?}"
                );
            }
        }
        if blas == 2 {
            assert!(
                sets.windows(2).all(|s| s[0] == s[1]),
                "the cuts the copies trace change between frames: {sets:?}"
            );
        } else {
            assert!(clas_short, "the CLAS never ran out before the BLAS");
        }
    }
    assert_clean();
}

#[test]
fn occluded_copies_wait_for_the_second_pass_and_the_image_holds() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let items = [
        Item {
            model: "plain",
            linear: Mat3::from_diagonal(Vec3::new(5.0, 4.0, 1.0)),
            at: Vec3::new(0.0, 1.0, 0.0),
        },
        Item {
            model: "ball",
            linear: Mat3::IDENTITY,
            at: Vec3::new(0.0, 1.0, -3.0),
        },
        Item {
            model: "ball",
            linear: Mat3::from_diagonal(Vec3::splat(0.3)),
            at: Vec3::new(0.6, 0.9, 1.5),
        },
    ];
    let placed = place(&mut w, &items);
    let hidden = placed.ranges[1].first + 1;
    let target = DVec3::new(0.0, 1.0, -3.0);
    let front = DVec3::new(0.0, 1.0, 5.0);
    let same = |a: &Frame, b: &Frame, what: &str| {
        assert!(a.ids == b.ids, "{what}: visible copies differ");
        assert!(a.depth == b.depth, "{what}: depth differs");
        assert!(a.a == b.a && a.b == b.b, "{what}: G-buffer differs");
    };
    let first = shoot(&mut w, &placed, &Shot::at(front, target, 1));
    assert!(!first.ids.contains(&hidden), "the wall hides the far ball");
    let mut shot = Shot::at(front, target, 2);
    shot.reset = false;
    let second = shoot(&mut w, &placed, &shot);
    assert!(
        second.counts[9] >= 1,
        "occluded copies: {:?}",
        second.counts
    );
    assert!(
        second.counts[0] < first.counts[0],
        "visible clusters: {} then {}",
        first.counts[0],
        second.counts[0]
    );
    same(&first, &second, "occlusion");

    // Stepped aside, the far ball shows past the wall's edge: the first pass still finds it
    // behind last frame's wall, the second draws it.
    let side = DVec3::new(7.5, 1.0, 5.0);
    let mut shot = Shot::at(side, target, 3);
    shot.reset = false;
    let revealed = shoot(&mut w, &placed, &shot);
    assert!(revealed.ids.contains(&hidden), "the far ball shows");
    assert!(
        revealed.counts[11] > 0,
        "drawn by the second pass: {:?}",
        revealed.counts
    );
    let fresh = shoot(&mut w, &placed, &Shot::at(side, target, 4));
    same(&fresh, &revealed, "revealed");
    assert_clean();
}

#[test]
fn debug_lod_holds_a_dag_level_or_scales_the_error_in_raster_and_ray_tracing() {
    let _serial = serial();
    let mut w = traversed_world();
    let items = [Item {
        model: "dag ground",
        linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
        at: Vec3::new(0.0, -1.0, -12.0),
    }];
    let placed = place(&mut w, &items);
    let eye = Vec3::new(0.3, 1.2, 4.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    let m = CpuMesh::of("dag ground");
    let clusters = m.clusters();
    let p = Projector::new(&items[0], eye);
    let slot = placed.ranges[0].first;
    let mut frame = 1;
    let mut run = |w: &mut World, lod: Option<DebugLod>| {
        let list = [placed.scene];
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        f.scene(placed.scene)
            .unwrap()
            .debug_lod(placed.ranges[0], lod);
        wait(f.end(&[]).unwrap());
        // Twice: the copy was drawn the frame before, so it traces at the RT threshold.
        shoot(
            w,
            &placed,
            &Shot::at(eye.as_dvec3(), target.as_dvec3(), frame + 1),
        );
        let g = shoot(
            w,
            &placed,
            &Shot::at(eye.as_dvec3(), target.as_dvec3(), frame + 2),
        );
        frame += 3;
        let cut = read_cut(&g.view, placed.count);
        (cut.raster(&m, slot), cut.slots[slot as usize])
    };
    let deepest = clusters.iter().map(|c| c.depth).max().unwrap();
    assert!(deepest >= 2, "the DAG has {deepest} levels");
    for level in [1, deepest] {
        let want = level_cut(&m, &clusters, level, &m.all_resident());
        let (got, s) = run(&mut w, Some(DebugLod::Level(level)));
        assert_eq!(got, want, "raster cut at level {level}");
        assert_eq!(
            (s.rt_count, s.rt_hash),
            (want.len() as u32, cut_hash(&want)),
            "ray-tracing cut at level {level}"
        );
    }
    let (plain, _) = run(&mut w, None);
    let (got, s) = run(&mut w, Some(DebugLod::ErrorScale(0.125)));
    let (want, near) = rule_cut(&m, &clusters, &p, 0.125, 1.0, &m.all_resident());
    let differ: Vec<u32> = want.symmetric_difference(&got).copied().collect();
    assert!(
        differ.iter().all(|c| near.contains(c)),
        "scaled cut differs at {differ:?}"
    );
    assert!(
        got.len() > plain.len(),
        "finer: {} against {}",
        got.len(),
        plain.len()
    );
    assert_eq!(
        s.rt_count as usize,
        got.len(),
        "ray tracing follows the scale"
    );
    let (back, _) = run(&mut w, None);
    assert_eq!(back, plain, "clearing the hold restores the cut");
    drop(w);
    debug_level_of_a_streaming_mesh(&items, eye, target, 1);
    assert_clean();
}

/// Frames of the streaming ground at debug level `level` until its loads settle: the cut is
/// the level's with every page resident, and no page holding only groups finer than the
/// level, which no cluster of the level comes from, is wanted, read or resident.
fn debug_level_of_a_streaming_mesh(items: &[Item], eye: Vec3, target: Vec3, level: u8) {
    let mut w = World::with(&RendererDesc {
        rt_merge_pixels: 0.0,
        budgets: Budgets {
            mesh_resident_max: 0,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    });
    let lod = Some(DebugLod::Level(level));
    let placed = place_with(&mut w, items, DVec3::ZERO, 100.0, None, lod);
    let model = w.model("dag ground");
    let m = CpuMesh::of("dag ground");
    let clusters = m.clusters();
    let slot = placed.ranges[0].first;
    let finer: Vec<usize> = (0..m.pages.len())
        .filter(|&p| {
            m.groups
                .iter()
                .filter(|g| g.page as usize == p)
                .all(|g| g.depth < level && g.error.is_finite())
        })
        .collect();
    assert!(!finer.is_empty(), "pages finer than level {level}");
    let start = Instant::now();
    let mut idle = 0;
    for frame in 1.. {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the level's pages never settled"
        );
        let shot = Shot::at(eye.as_dvec3(), target.as_dvec3(), frame);
        let g = shoot_with(&mut w, &placed, &shot, |s| {
            s.debug_lod(placed.ranges[0], lod);
        });
        let (pages, _, nodes) = mesh_pages(&w, model);
        let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
        let cut: CutGpu = read_buffer(root.cut, 1)[0];
        let words = u64::from(cut.feedback_words);
        let (wanted, touched): (Vec<u32>, Vec<u32>) = (
            read_buffer(cut.wanted, words),
            read_buffer(cut.touched, words),
        );
        let (models, meshes, _) = orr_scene::testing::global_tables(&w.scenes);
        let record: ModelGpu = read_buffer(models.offset(u64::from(model.0)), 1)[0];
        let mesh: MeshGpu = read_buffer(meshes.offset(u64::from(record.mesh)), 1)[0];
        for &p in &finer {
            let e = mesh.feedback_pages as usize + p;
            let bit = |words: &[u32]| words[e / 32] >> (e % 32) & 1 != 0;
            assert!(
                !bit(&wanted) && !bit(&touched) && pages[p] == 0,
                "frame {frame}: page {p}, finer than level {level}, wanted {}, read {}, resident {}",
                bit(&wanted),
                bit(&touched),
                pages[p] != 0
            );
        }
        let r = *w.scenes.residency();
        idle = if r.loads == 0 && r.class(orr_scene::ResidencyClass::MeshPages).loading_bytes == 0 {
            idle + 1
        } else {
            0
        };
        if idle < 8 {
            continue;
        }
        let resident = Resident {
            pages: pages.iter().map(|&a| a != 0).collect(),
            nodes: nodes.iter().map(|&a| a != 0).collect(),
        };
        let raster = read_cut(&g.view, placed.count).raster(&m, slot);
        assert_eq!(raster, level_cut(&m, &clusters, level, &resident));
        assert_eq!(
            raster,
            level_cut(&m, &clusters, level, &m.all_resident()),
            "the level's pages and what they depend on are resident"
        );
        break;
    }
    assert_eq!(w.scenes.residency().failed, 0);
}

/// The page table, page bases and node page table of `model`'s mesh as the last frame left
/// them.
fn mesh_pages(w: &World, model: ModelId) -> (Vec<u64>, Vec<u32>, Vec<u64>) {
    let (models, meshes, _) = orr_scene::testing::global_tables(&w.scenes);
    let model: ModelGpu = read_buffer(models.offset(u64::from(model.0)), 1)[0];
    let mesh: MeshGpu = read_buffer(meshes.offset(u64::from(model.mesh)), 1)[0];
    let n = u64::from(mesh.page_count);
    let nodes = match mesh.node_page_count {
        0 => vec![1],
        k => read_buffer(mesh.node_pages, u64::from(k)),
    };
    (
        read_buffer(mesh.pages, n),
        read_buffer(mesh.page_bases, n),
        nodes,
    )
}

/// [`pack`] with the DAG ground's hierarchy in a top page and one streamed node page.
fn paged_pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let a = Archive::from_memory(Arc::clone(pack())).unwrap();
        let model = a.find(AssetId::from_name("dag ground")).unwrap();
        let mesh = ModelInfo::parse(&blob(&a, model, 0)).unwrap().header.mesh;
        paged::split_hierarchy(pack(), mesh)
    })
}

/// Frames of a streaming ground while its pages land a request a frame, from none of its
/// streamed pages to all of them, every other frame at debug level 0, with its hierarchy
/// resident and with it paged: the raster and ray-tracing cuts follow the cluster rule with
/// clusters standing in for those of groups that cannot be drawn, whose page or whose leaf's
/// node page is not resident, cover the ground once, and with every page resident are the
/// plain rule's. A page is resident only with the pages it depends on and the node pages
/// over its groups' leaves.
#[test]
fn clusters_stand_in_for_pages_not_resident_and_the_cut_covers_the_mesh_once() {
    let _serial = serial();
    for (bytes, paged) in [(pack(), false), (paged_pack(), true)] {
        stand_in(bytes, paged);
    }
    assert_clean();
}

fn stand_in(bytes: &Arc<[u8]>, paged: bool) {
    // Copies not yet drawn trace at the raster threshold too, so the first frame compares.
    let mut w = World::with_pack(
        &RendererDesc {
            rt_merge_pixels: 0.0,
            culled_error_scale: 1.0,
            budgets: Budgets {
                mesh_resident_max: 0,
                traversal_requests: 1,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        },
        bytes,
    );
    let items = [Item {
        model: "dag ground",
        linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
        at: Vec3::new(0.0, -1.0, -12.0),
    }];
    let placed = place(&mut w, &items);
    let model = w.model("dag ground");
    let eye = Vec3::new(0.3, 1.2, 4.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    let m = CpuMesh::of_pack(bytes, "dag ground");
    assert_eq!(m.node_pages, if paged { 2 } else { 1 });
    let clusters = m.clusters();
    let p = Projector::new(&items[0], eye);
    let slot = placed.ranges[0].first;
    let pinned = m.header.pinned_pages as usize;
    assert!(pinned < m.pages.len(), "the ground's pages stream");
    // Per frame: whether at level 0, its resident streamed pages, the clusters standing in
    // and whether its node pages are all resident.
    let mut frames: Vec<(bool, usize, usize, bool)> = Vec::new();
    let start = Instant::now();
    let done = |frames: &[(bool, usize, usize, bool)]| {
        let streamed = m.pages.len() - pinned;
        let some = |f: &&(bool, usize, usize, bool)| f.1 > 0 && f.1 < streamed && f.2 > 0;
        let error = frames.iter().filter(|f| !f.0);
        error.clone().any(|f| f.1 == 0)
            && error.clone().any(|f| some(&f))
            && error.clone().any(|f| f.1 == streamed)
            && frames.iter().filter(|f| f.0).any(|f| some(&f))
            && frames.iter().any(|f| !f.3) == paged
    };
    for frame in 1.. {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "frames (level, streamed pages, clusters standing in, node pages): {frames:?}"
        );
        let level = frame % 2 == 0;
        let shot = Shot::at(eye.as_dvec3(), target.as_dvec3(), frame);
        let g = shoot_with(&mut w, &placed, &shot, |s| {
            s.debug_lod(placed.ranges[0], level.then_some(DebugLod::Level(0)));
        });
        let (pages, bases, nodes) = mesh_pages(&w, model);
        let resident = Resident {
            pages: pages.iter().map(|&a| a != 0).collect(),
            nodes: nodes.iter().map(|&a| a != 0).collect(),
        };
        for (q, deps) in m.deps.iter().enumerate() {
            assert!(
                !resident.pages[q] || deps.iter().all(|&d| resident.pages[d as usize]),
                "frame {frame}: page {q} is resident without the pages it depends on"
            );
        }
        for g in &m.groups {
            assert!(
                !resident.pages[g.page as usize] || resident.nodes[g.node_page as usize],
                "frame {frame}: page {} is resident without node page {}",
                g.page,
                g.node_page
            );
        }
        let streamed = resident.pages[pinned..].iter().filter(|&&r| r).count();
        let hierarchy = resident.nodes.iter().all(|&r| r);
        let cut = read_cut(&g.view, placed.count);
        let raster = cut.raster(&m, slot);
        // Ray tracing takes the raster's threshold.
        let ray = cut.ray(&m, slot, &pages, &bases, &raster);
        let what = format!(
            "paged {paged}, frame {frame}, {streamed} streamed pages and all node pages {hierarchy}, level {level}"
        );
        assert_whole(&m, &clusters, &raster, &format!("{what}: raster"));
        assert_whole(&m, &clusters, &ray, &format!("{what}: ray tracing"));
        let standing = |want: &BTreeSet<u32>| {
            let fine = if level {
                level_cut(&m, &clusters, 0, &m.all_resident())
            } else {
                rule_cut(&m, &clusters, &p, 1.0, 1.0, &m.all_resident()).0
            };
            want.difference(&fine).count()
        };
        if level {
            let want = level_cut(&m, &clusters, 0, &resident);
            assert_eq!(raster, want, "{what}: raster");
            assert_eq!(ray, want, "{what}: ray tracing");
            frames.push((true, streamed, standing(&want), hierarchy));
        } else {
            let (want, near) = rule_cut(&m, &clusters, &p, 1.0, 1.0, &resident);
            for (got, kind) in [(&raster, "raster"), (&ray, "ray tracing")] {
                let differ: Vec<u32> = want.symmetric_difference(got).copied().collect();
                assert!(
                    differ.iter().all(|c| near.contains(c)),
                    "{what}: {kind} cut differs from the rule at {differ:?}"
                );
            }
            frames.push((false, streamed, standing(&want), hierarchy));
        }
        if done(&frames) {
            break;
        }
        if frames.iter().filter(|f| f.0).any(|f| f.1 > 0 && f.2 > 0) {
            w.scenes.prefetch(model, 0);
        }
    }
    assert_eq!(w.scenes.residency().failed, 0);
}

/// Pixel probes over the streaming ground while its pages land a request a frame: each names
/// the copy, its model and the visible cluster the visibility buffer holds, gives the
/// cluster's group, DAG depth and projected errors as the CPU rule computes them at the focus,
/// and says it stands in exactly when its own error is over the threshold and the group it was
/// simplified from cannot be drawn. Early frames show stand-ins, and once every page is
/// resident none.
#[test]
fn pixel_probes_name_the_drawn_cluster_its_depth_and_whether_it_stands_in() {
    let _serial = serial();
    let mut w = World::with(&RendererDesc {
        rt_merge_pixels: 0.0,
        budgets: Budgets {
            mesh_resident_max: 0,
            traversal_requests: 1,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    });
    let items = [Item {
        model: "dag ground",
        linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
        at: Vec3::new(0.0, -1.0, -12.0),
    }];
    let placed = place(&mut w, &items);
    let model = w.model("dag ground");
    let eye = Vec3::new(0.3, 1.2, 4.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    let m = CpuMesh::of("dag ground");
    let clusters = m.clusters();
    let p = Projector::new(&items[0], eye);
    let slot = placed.ranges[0].first;
    let archive = Archive::from_memory(Arc::clone(pack())).unwrap();
    let ground = archive.find(AssetId::from_name("dag ground")).unwrap();
    let mesh_id = ModelInfo::parse(&blob(&archive, ground, 0))
        .unwrap()
        .header
        .mesh;
    let mesh_name = archive.name(archive.find(mesh_id).unwrap()).to_owned();
    let pixels: Vec<[u32; 2]> = [0.3, 0.5, 0.7]
        .into_iter()
        .flat_map(|x| {
            [0.55, 0.7, 0.85]
                .map(|y| [(x * EXTENT[0] as f32) as u32, (y * EXTENT[1] as f32) as u32])
        })
        .collect();
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * b.abs().max(1e-3);
    let (mut standing, mut whole) = (0, false);
    let start = Instant::now();
    for frame in 1.. {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the ground's pages never all landed; {standing} probes stood in"
        );
        let shot = Shot::at(eye.as_dvec3(), target.as_dvec3(), frame);
        let (g, probes) = shoot_probed(&mut w, &placed, &shot, |_| {}, |_| {}, &pixels);
        let (pages, _, nodes) = mesh_pages(&w, model);
        let resident = Resident {
            pages: pages.iter().map(|&a| a != 0).collect(),
            nodes: nodes.iter().map(|&a| a != 0).collect(),
        };
        assert_eq!(probes.len(), pixels.len());
        for (probe, pixel) in probes.iter().zip(&pixels) {
            let at = (pixel[1] * EXTENT[0] + pixel[0]) as usize;
            let what = format!("frame {frame}, pixel {pixel:?}:\n{probe}");
            assert_eq!(probe.hit(), g.ids[at] != 0, "{what}");
            if !probe.hit() {
                continue;
            }
            let gpu = &probe.gpu;
            assert_eq!(gpu.copy + 1, g.ids[at], "{what}");
            assert_eq!(gpu.copy, slot, "{what}");
            assert_eq!(gpu.cluster + 1, g.clusters[at], "{what}");
            assert_eq!(gpu.model, model.0, "{what}");
            assert_eq!(probe.model_name.as_deref(), Some("dag ground"), "{what}");
            assert_eq!(
                probe.mesh_name.as_deref(),
                Some(mesh_name.as_str()),
                "{what}"
            );
            assert_eq!(gpu.raster_threshold, 1.0, "{what}");
            let c = &clusters[gpu.cluster as usize];
            assert_eq!(probe.depth(), Some(c.depth), "{what}");
            assert_eq!(probe.group.map(|(i, _)| i), Some(c.group), "{what}");
            let own = p.error(&c.cluster.self_sphere, c.cluster.self_error);
            let parent = p.error(&c.cluster.parent_sphere, c.cluster.parent_error);
            assert!(
                close(gpu.self_pixels, own),
                "{what}: self {own} px on the CPU"
            );
            assert!(
                parent.is_infinite() == gpu.parent_pixels.is_infinite()
                    && (parent.is_infinite() || close(gpu.parent_pixels, parent)),
                "{what}: parent {parent} px on the CPU"
            );
            // f32 rounding decides either way within a hair of the threshold.
            if close(own, 1.0) || close(parent, 1.0) {
                continue;
            }
            let stands = own > 1.0 && c.depth > 0 && m.stands_in(c, &resident);
            assert_eq!(probe.stands_in(), stands, "{what}");
            standing += usize::from(stands);
        }
        if resident.pages.iter().all(|&r| r) && resident.nodes.iter().all(|&r| r) {
            whole = true;
            break;
        }
        // The view wants only some pages; once stand-ins showed, every page loads.
        if standing > 0 {
            w.scenes.prefetch(model, 0);
        }
    }
    assert!(whole && standing > 0, "{standing} probes stood in");
    assert_clean();
}

/// The streaming ground's view under an ample mesh page pool, then under one a third the
/// size of the streamed pages the view wants: the first keeps the configured detail, the
/// second settles at a coarser one whose raster and ray-tracing cuts are the cluster rule's
/// at the scaled threshold with every page resident, over the whole ground, while what the
/// view wants stays under the pool's budget and no request of the traversal is turned down.
#[test]
fn a_mesh_page_pool_short_of_what_the_view_wants_coarsens_the_whole_cut_evenly() {
    let _serial = serial();
    let (scale, demand, pinned) = settled_detail(Budgets::default().mesh_page_pool);
    assert_eq!(scale, 1.0, "the configured detail fits an ample pool");
    assert!(demand > pinned, "the view wants streamed pages");
    let pool = pinned + (demand - pinned) * 2 / 5;
    let (scale, _, _) = settled_detail(PoolSize::Bytes(pool));
    assert!(scale > 1.0, "a pool of {pool} B for {demand} B coarsens");
    assert_clean();
}

/// A pack of one DAG ground over a fine grid, whose many pages stream in steps small beside
/// a pool of a few times their size, a second ground of pages of its own, and a wall.
fn fine_ground_pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(|| {
        let unit = [
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(-0.5, 0.5, 0.0),
        ];
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::Memory,
                vec![
                    AssetSpec {
                        name: "fine ground".into(),
                        source: Source::Mesh {
                            mesh: grid(320, 8.0, 2.0, |x, z| {
                                0.15 * (x * 1.3).sin() * (z * 0.9).cos()
                            }),
                            materials: vec![Material::default()],
                            settings: MeshSettings {
                                lod: LodMode::Dag,
                                ..MeshSettings::default()
                            },
                        },
                    },
                    AssetSpec {
                        name: "hidden ground".into(),
                        source: Source::Mesh {
                            mesh: grid(320, 8.0, 2.0, |x, z| {
                                0.15 * (x * 0.7).cos() * (z * 1.1).sin()
                            }),
                            materials: vec![Material::default()],
                            settings: MeshSettings {
                                lod: LodMode::Dag,
                                ..MeshSettings::default()
                            },
                        },
                    },
                    AssetSpec {
                        name: "wall".into(),
                        source: Source::Mesh {
                            mesh: quad(unit),
                            materials: vec![Material::default()],
                            settings: MeshSettings::default(),
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

/// Frames of the streaming ground's view under a mesh page pool of `pool` until its loads
/// and detail scale settle, then a run of frames that keep the scale, turn no traversal
/// request down and draw the rule's cut at the scaled threshold with every page resident;
/// returns the scale, what the mesh page pool was asked for, and what the mount pinned.
fn settled_detail(pool: PoolSize) -> (f32, u64, u64) {
    // Fine enough that the view wants most of the ground's pages, each small beside the pool.
    const ERROR_PIXELS: f32 = 0.25;
    let bytes = fine_ground_pack();
    let mut w = World::with_pack(
        &RendererDesc {
            raster_error_pixels: ERROR_PIXELS,
            rt_error_pixels: ERROR_PIXELS,
            rt_merge_pixels: 0.0,
            culled_error_scale: 1.0,
            hidden_detail_lead: 1.0,
            budgets: Budgets {
                mesh_page_pool: pool,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        },
        bytes,
    );
    let mesh_pool = |w: &World| {
        *w.scenes
            .residency()
            .class(orr_scene::ResidencyClass::MeshPages)
    };
    let pinned = mesh_pool(&w).resident_bytes;
    let items = [Item {
        model: "fine ground",
        linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
        at: Vec3::new(0.0, -1.0, -12.0),
    }];
    let placed = place(&mut w, &items);
    let model = w.model("fine ground");
    let eye = Vec3::new(0.3, 1.2, 4.0);
    let target = Vec3::new(0.0, 0.0, -10.0);
    let m = CpuMesh::of_pack(bytes, "fine ground");
    let clusters = m.clusters();
    let p = Projector::new(&items[0], eye);
    let slot = placed.ranges[0].first;
    let mut frame = 1;
    let mut shot = |w: &mut World| {
        frame += 1;
        let g = shoot(
            w,
            &placed,
            &Shot::at(eye.as_dvec3(), target.as_dvec3(), frame),
        );
        (g, w.scenes.stats().detail)
    };
    let start = Instant::now();
    let (mut still, mut scale) = (0, 0.0);
    while still < 24 {
        let (_, d) = shot(&mut w);
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "pool {pool:?}: the detail never settled, at {d:?}"
        );
        let idle = w.scenes.residency().loads == 0 && mesh_pool(&w).loading_bytes == 0;
        still = if idle && d.memory_scale == scale {
            still + 1
        } else {
            0
        };
        scale = d.memory_scale;
    }
    let mut demand = 0;
    for _ in 0..30 {
        let (g, d) = shot(&mut w);
        let what = format!("pool {pool:?} at {d:?}");
        assert_eq!(d.memory_scale, scale, "{what}: the scale moved");
        assert!(
            d.mesh_demand as f64 <= 0.92 * d.mesh_budget as f64,
            "{what}: over the band"
        );
        let r = w.scenes.residency();
        assert_eq!(r.dropped_from(RequestSource::Traversal), 0, "{what}");
        demand = d.mesh_demand;
        let (pages, bases, _) = mesh_pages(&w, model);
        let cut = read_cut(&g.view, placed.count);
        let raster = cut.raster(&m, slot);
        assert_whole(&m, &clusters, &raster, &format!("{what}: raster"));
        let ray = cut.ray(&m, slot, &pages, &bases, &raster);
        let (want, near) = rule_cut(
            &m,
            &clusters,
            &p,
            ERROR_PIXELS * scale,
            1.0,
            &m.all_resident(),
        );
        for (got, kind) in [(&raster, "raster"), (&ray, "ray tracing")] {
            let differ: Vec<u32> = want.symmetric_difference(got).copied().collect();
            assert!(
                differ.iter().all(|c| near.contains(c)),
                "{what}: the {kind} cut differs from the rule at {differ:?}"
            );
        }
    }
    assert_eq!(w.scenes.residency().failed, 0);
    (scale, demand, pinned)
}

/// Every page of both grounds loads, mixed over the stream pool's blocks, under a view that
/// draws some of the fine ground's. The mesh page budget then falls to what the view wants: the
/// other pages go, and the pool empties the blocks they leave partly used by moving the pages
/// the view draws. Every frame meanwhile draws the pixels the frames before drew, and once the
/// pool has settled its blocks hold little more than its pages, some of which moved, each
/// cluster of a moved page naming the page's new slot with the rest of its entry as it was.
#[test]
fn mesh_pages_move_out_of_draining_blocks_under_the_frames_that_draw_them() {
    let _serial = serial();
    const ERROR_PIXELS: f32 = 2.0;
    const BLOCK: u64 = 1 << 20;
    let mesh_pages_class = orr_scene::ResidencyClass::MeshPages;
    let bytes = fine_ground_pack();
    let mut w = World::with_pack(
        &RendererDesc {
            raster_error_pixels: ERROR_PIXELS,
            rt_error_pixels: ERROR_PIXELS,
            rt_merge_pixels: 0.0,
            culled_error_scale: 1.0,
            hidden_detail_lead: 1.0,
            budgets: Budgets {
                mesh_page_pool: PoolSize::Bytes(128 << 20),
                texture_pool: PoolSize::Bytes(16 << 20),
                instance_page_pool: 16 << 20,
                in_flight_bytes: 16 << 20,
                stream_block: BLOCK,
                ..Budgets::default()
            },
            ..RendererDesc::default()
        },
        bytes,
    );
    let items = [Item {
        model: "fine ground",
        linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
        at: Vec3::new(0.0, -1.0, -12.0),
    }];
    let placed = place(&mut w, &items);
    let models = ["fine ground", "hidden ground"].map(|name| w.model(name));
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    let mut frame = 1;
    let mut shot = |w: &mut World| {
        frame += 1;
        shoot(w, &placed, &Shot::at(eye, target, frame))
    };
    let stream_pool = |w: &World| {
        let memory = w.scenes.memory();
        *memory
            .pools
            .iter()
            .find(|p| p.name == "stream pool")
            .expect("the stream pool")
    };
    let refs = |w: &World, g: &Frame| {
        let (models_at, meshes, _) = orr_scene::testing::global_tables(&w.scenes);
        let model: ModelGpu = read_buffer(models_at.offset(u64::from(models[0].0)), 1)[0];
        let mesh: MeshGpu = read_buffer(meshes.offset(u64::from(model.mesh)), 1)[0];
        let firsts: Vec<u32> = read_buffer(mesh.page_clusters, u64::from(mesh.page_count) + 1);
        let root: SceneRoot = read_buffer(g.view.scene, 1)[0];
        let (pages, bases, _) = mesh_pages(w, models[0]);
        let pinned = mesh.pinned_pages as usize;
        let of_page: Vec<Vec<orr_scene::ClusterRef>> = (0..pages.len())
            .map(|p| {
                if p < pinned || pages[p] == 0 {
                    return Vec::new();
                }
                let count = u64::from(firsts[p + 1] - firsts[p]);
                read_buffer(root.clusters.offset(u64::from(bases[p])), count)
            })
            .collect();
        (pages, of_page)
    };

    let start = Instant::now();
    loop {
        assert!(start.elapsed() < Duration::from_secs(120), "pages pending");
        for &m in &models {
            w.scenes.prefetch(m, 0);
        }
        shot(&mut w);
        if models.iter().all(|&m| !mesh_pages(&w, m).0.contains(&0)) {
            break;
        }
    }
    let full = stream_pool(&w);
    assert!(full.blocks >= 8, "pages over many blocks: {full:?}");
    // The prefetched pages leave their hold while the view's frames repeat.
    let mut before = shot(&mut w);
    for _ in 0..4 {
        let g = shot(&mut w);
        assert!(
            g.clusters == before.clusters && g.a == before.a,
            "a still view"
        );
        before = g;
    }
    let detail = w.scenes.stats().detail;
    assert_eq!(detail.memory_scale, 1.0, "{detail:?}");
    let (pages_before, refs_before) = refs(&w, &before);

    let budget = detail.mesh_demand * 100 / 85;
    w.scenes.lower_budget(mesh_pages_class, budget);
    let start = Instant::now();
    let (mut settled, mut drained) = (0, 0);
    let mut last = before;
    while settled < 8 {
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the pool never settled: {:?}",
            stream_pool(&w)
        );
        let g = shot(&mut w);
        assert!(
            g.clusters == last.clusters && g.depth == last.depth && g.a == last.a && g.b == last.b,
            "frame {frame} drew other pixels"
        );
        assert_eq!(w.scenes.stats().detail.memory_scale, 1.0);
        last = g;
        let fill = stream_pool(&w);
        drained = drained.max(fill.draining);
        let used = w.scenes.residency().class(mesh_pages_class).resident_bytes;
        let idle = used <= budget && fill.draining == 0 && fill.retired == 0;
        settled = if idle { settled + 1 } else { 0 };
    }
    let fill = stream_pool(&w);
    assert!(drained > 0, "no block drained: {full:?} to {fill:?}");
    assert!(
        fill.used < full.used / 2 && fill.bytes <= fill.used + 3 * BLOCK,
        "{full:?} to {fill:?}"
    );
    let (pages, refs) = refs(&w, &last);
    let cpu = CpuMesh::of_pack(bytes, "fine ground");
    let mut moved = 0;
    for p in 0..pages.len() {
        if pages[p] == 0 || refs[p].is_empty() || pages[p] == pages_before[p] {
            continue;
        }
        moved += 1;
        assert_eq!(refs[p].len(), refs_before[p].len(), "page {p}");
        for (now, was) in refs[p].iter().zip(&refs_before[p]) {
            assert_eq!(now.page.address().0, pages[p], "page {p}");
            assert_eq!(
                (now.local, now.mesh_cluster, now.clas),
                (was.local, was.mesh_cluster, was.clas),
                "page {p}"
            );
        }
        // SAFETY: the page's resident slot.
        let at = unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(pages[p])) };
        let resident: Vec<u8> = read_buffer(at, cpu.pages[p].len() as u64);
        assert!(resident == cpu.pages[p], "page {p} in its new slot");
    }
    assert!(moved > 0, "no page of the view moved: {full:?} to {fill:?}");
    assert_eq!(w.scenes.residency().failed, 0);
    assert_clean();
}

/// Bytes of a block of the streamed pools in [`ClasMoves`]: two CLAS pool pages of 64 KiB, so
/// a page left alone leaves its block half used.
const CLAS_BLOCK: u64 = 128 << 10;

/// Both grounds loaded whole under a view that traces some of the fine ground, so the CLAS of
/// the two lie mixed over the pages of the CLAS pool and the pages' memory over its blocks. The
/// memory is planned as under pressure, so that it empties every block it can.
struct ClasMoves {
    w: World,
    placed: Placed,
    frame: u32,
    /// What the view's camera rays hit with every page loaded and none moved.
    traced: Vec<u8>,
}

impl ClasMoves {
    const EYE: DVec3 = DVec3::new(0.3, 1.2, 4.0);
    const TARGET: DVec3 = DVec3::new(0.0, 0.0, -10.0);

    fn new() -> Self {
        let mut w = World::with_pack(
            &RendererDesc {
                raster_error_pixels: 2.0,
                rt_error_pixels: 2.0,
                rt_merge_pixels: 0.0,
                culled_error_scale: 1.0,
                hidden_detail_lead: 1.0,
                // Small enough that the blocks stay at `CLAS_BLOCK`.
                budgets: Budgets {
                    mesh_page_pool: PoolSize::Bytes(96 << 20),
                    texture_pool: PoolSize::Bytes(4 << 20),
                    instance_page_pool: 4 << 20,
                    in_flight_bytes: 8 << 20,
                    stream_block: CLAS_BLOCK,
                    ..Budgets::default()
                },
                ..RendererDesc::default()
            },
            fine_ground_pack(),
        );
        let items = [Item {
            model: "fine ground",
            linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
            at: Vec3::new(0.0, -1.0, -12.0),
        }];
        let placed = place(&mut w, &items);
        let models = ["fine ground", "hidden ground"].map(|name| w.model(name));
        w.scenes.plan_clas_memory_tight(true);
        let mut c = Self {
            w,
            placed,
            frame: 1,
            traced: Vec::new(),
        };
        let start = Instant::now();
        loop {
            assert!(start.elapsed() < Duration::from_secs(120), "pages pending");
            c.prefetch(&models);
            c.shot();
            if models.iter().all(|&m| !mesh_pages(&c.w, m).0.contains(&0)) {
                break;
            }
        }
        let (fill, _) = c.pool();
        assert!(fill.blocks >= 8, "CLAS pages over many blocks: {fill:?}");
        let g = c.shot();
        let hits = rays(&g.view, &trace::MODULE, "pixels", 0).traced;
        assert!(
            hits.iter().filter(|t| t.instance != 0).count() > hits.len() / 4,
            "the view traces the ground"
        );
        c.traced = bytemuck::cast_slice(&hits).to_vec();
        for _ in 0..4 {
            c.exact();
        }
        c
    }

    fn prefetch(&mut self, models: &[ModelId]) {
        for &m in models {
            self.w.scenes.prefetch(m, 0);
        }
    }

    fn shot(&mut self) -> Frame {
        self.frame += 1;
        let shot = Shot::at(Self::EYE, Self::TARGET, self.frame);
        shoot(&mut self.w, &self.placed, &shot)
    }

    /// A frame whose camera rays hit what they hit before anything moved.
    fn exact(&mut self) {
        let g = self.shot();
        assert!(
            traced_hits(&g) == self.traced,
            "frame {} traced other hits",
            self.frame
        );
        assert_eq!(self.w.scenes.stats().detail.memory_scale, 1.0);
    }

    /// The pool of the CLAS pool's page memory and the CLAS pool's pages.
    fn pool(&self) -> (orr_scene::PoolFill, orr_scene::ClasPages) {
        let memory = self.w.scenes.memory();
        let fill = *memory
            .pools
            .iter()
            .find(|p| p.name == "CLAS pool pages")
            .expect("the CLAS pool's page memory");
        (fill, memory.clas.expect("a CLAS pool"))
    }

    /// Lowers the mesh page budget to `bytes`.
    fn budget(&mut self, bytes: u64) {
        let class = orr_scene::ResidencyClass::MeshPages;
        self.w.scenes.lower_budget(class, bytes);
    }

    /// The mesh page budget that keeps what the view wants.
    fn view_budget(&self) -> u64 {
        self.w.scenes.stats().detail.mesh_demand * 100 / 85
    }

    /// Exact frames, each after one begun and dropped when `dropping`, until the pool's page
    /// memory has rested for eight of them, no page wholly free or held, nothing draining or
    /// retired and every mesh page within `budget`.
    fn settle(&mut self, budget: u64, dropping: bool) {
        let class = orr_scene::ResidencyClass::MeshPages;
        let start = Instant::now();
        let mut settled = 0;
        while settled < 8 {
            assert!(
                start.elapsed() < Duration::from_secs(120),
                "the CLAS pool never settled: {:?}",
                self.pool()
            );
            if dropping {
                let desc = FrameDesc {
                    time: time(self.frame),
                    scenes: &[],
                    low_latency: None,
                };
                drop(self.w.scenes.begin_frame(&desc).unwrap());
            }
            self.exact();
            let (fill, pages) = self.pool();
            let used = self.w.scenes.residency().class(class).resident_bytes;
            let rest = fill.draining == 0 && fill.retired == 0;
            let idle = used <= budget && rest && pages.held == 0 && pages.wholly_free == 0;
            settled = if idle { settled + 1 } else { 0 };
        }
    }

    /// CLAS pool pages moved so far.
    fn moved(&self) -> u64 {
        self.w.scenes.residency().clas_pages_moved
    }

    /// The pool's page memory holds what its bound pages take in the fewest blocks, and no load
    /// failed.
    fn check_settled(&self) {
        let (fill, pages) = self.pool();
        assert_eq!(
            fill.used,
            u64::from(pages.bound) * pages.page_bytes,
            "{fill:?}"
        );
        assert_eq!(
            fill.bytes,
            fill.used.next_multiple_of(CLAS_BLOCK),
            "{fill:?} for {pages:?}"
        );
        assert_eq!(self.w.scenes.residency().failed, 0);
    }
}

/// What each pixel's camera ray hit, as bytes.
fn traced_hits(g: &Frame) -> Vec<u8> {
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    bytemuck::cast_slice(&r.traced).to_vec()
}

/// A frame of `placed` from `shot` whose camera rays are traced after it, neither waited for.
fn shoot_later(w: &mut World, placed: &Placed, shot: &Shot) -> RaysLater {
    let list = [placed.scene];
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(shot.frame),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    f.scene(placed.scene).unwrap().set_focus(shot.eye);
    f.request_detail(placed.scene, PIXELS_PER_RADIAN).unwrap();
    f.update_scenes(&mut |_| {}).unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: placed.scene,
            history: placed.history,
            position: shot.eye,
            rotation: look(Vec3::ZERO, (shot.target - shot.eye).as_vec3()),
            projection: Projection {
                fov_y: 0.9,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: shot.jitter,
            camera_cut: shot.reset,
            mip_bias: shot.mip_bias,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
    f.end(&[]).unwrap();
    rays_later(&view, &trace::MODULE, "pixels", 0)
}

/// Frames of `placed`, each read while the next is in flight, until two in a row trace the
/// same; returns what their camera rays hit and the frame still in flight.
fn settled_hits(
    w: &mut World,
    placed: &Placed,
    (eye, target): (DVec3, DVec3),
    frame: &mut u32,
) -> (Vec<u8>, RaysLater) {
    let start = Instant::now();
    let mut last = None;
    *frame += 1;
    let mut flight = shoot_later(w, placed, &Shot::at(eye, target, *frame));
    loop {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the view never settled"
        );
        *frame += 1;
        let next = shoot_later(w, placed, &Shot::at(eye, target, *frame));
        wait(flight.point);
        let now = bytemuck::cast_slice(&flight.read().traced).to_vec();
        flight = next;
        if last.as_ref() == Some(&now) {
            return (now, flight);
        }
        last = Some(now);
    }
}

/// How many pixels of `a` and `b` differ.
fn differing(a: &[u8], b: &[u8]) -> usize {
    let n = size_of::<trace::Traced>();
    a.chunks(n).zip(b.chunks(n)).filter(|(x, y)| x != y).count()
}

/// The global material, model, mesh and section-material tables grow while frames are in
/// flight: two more mounts of the pack land between frames nothing waits for, and the tables
/// move. Every frame meanwhile traces what the frames before did, and copies of the last mount's
/// models then show the same surfaces through its own entries of the moved tables.
#[test]
fn global_tables_grow_under_frames_in_flight_and_every_frame_stays_exact() {
    let _serial = serial();
    let mut w = World::new(&KernelSet::NONE);
    let items = showcase();
    let placed = place(&mut w, &items);
    let (eye, target) = (DVec3::new(0.5, 2.5, 6.5), DVec3::new(0.0, 0.5, -0.5));
    let mut frame = 0;
    let (settled, flight) = settled_hits(&mut w, &placed, (eye, target), &mut frame);
    wait(flight.point);
    // The PTLAS holds the copies' slots, and no part instance: no copy is posed.
    let root: SceneRoot = read_buffer(flight.scene, 1)[0];
    assert_eq!(
        (root.part_base, root.part_capacity),
        (items.len() as u32, 0)
    );
    let tables = orr_scene::testing::global_tables(&w.scenes);
    let lists = orr_scene::testing::model_materials(&w.scenes);
    let ball = w.model("ball");

    let more = [0, 1].map(|_| w.scenes.mount(PackSource::Memory(Arc::clone(pack()))));
    let mut later = vec![flight];
    let mut after = 0;
    let start = Instant::now();
    while after < 3 {
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the mounts never landed"
        );
        frame += 1;
        later.push(shoot_later(&mut w, &placed, &Shot::at(eye, target, frame)));
        let ready = more
            .iter()
            .all(|&m| matches!(w.scenes.mount_state(m), LoadState::Ready));
        after += u32::from(ready);
    }
    wait(later.last().unwrap().point);
    let moved = orr_scene::testing::global_tables(&w.scenes);
    assert!(
        moved.0 != tables.0 && moved.1 != tables.1 && moved.2 != tables.2,
        "the model, mesh and material tables never grew: {tables:?} to {moved:?}"
    );
    assert_ne!(
        orr_scene::testing::model_materials(&w.scenes),
        lists,
        "the section-material table never grew"
    );
    for (i, l) in later.iter().enumerate() {
        let traced: Vec<u8> = bytemuck::cast_slice(&l.read().traced).to_vec();
        assert!(
            traced == settled,
            "frame {} of the {} under the mounts hit other surfaces at {} pixels",
            i + 1,
            later.len(),
            differing(&traced, &settled),
        );
    }

    // Copies of the last mount's models in place of the first's.
    let last = World {
        mount: more[1],
        ..w
    };
    let mut w = last;
    let models: Vec<ModelId> = items.iter().map(|c| w.model(c.model)).collect();
    // The last mount's ball names its own materials, whose records match the first's.
    let (models_now, _, materials) = orr_scene::testing::global_tables(&w.scenes);
    let lists = orr_scene::testing::model_materials(&w.scenes);
    let sections = |model: ModelId| {
        let m: ModelGpu = read_buffer(models_now.offset(u64::from(model.0)), 1)[0];
        read_buffer(
            lists.offset(u64::from(m.materials)),
            u64::from(m.section_count),
        )
    };
    let record = |id: u32| -> MaterialGpu { read_buffer(materials.offset(u64::from(id)), 1)[0] };
    let (first, last) = (sections(ball), sections(w.model("ball")));
    assert_eq!(first.len(), last.len());
    for (&a, &b) in first.iter().zip(&last) {
        assert_ne!(a, b, "the last mount's ball names the first's material");
        assert!(bytemuck::bytes_of(&record(a)) == bytemuck::bytes_of(&record(b)));
    }
    let list = [placed.scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(frame + 1),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(placed.scene).unwrap();
    let mut ranges = Vec::new();
    for (i, (&range, model)) in placed.ranges.iter().zip(models).enumerate() {
        s.remove(range);
        ranges.push(
            s.add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: placed.group,
                first: i as u32,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 100.0,
                animation_range: 0.0,
                user_tag: i as u32,
            })
            .unwrap(),
        );
    }
    drop(s);
    wait(f.end(&[]).unwrap());
    frame += 1;
    let placed = Placed { ranges, ..placed };
    let (traced, flight) = settled_hits(&mut w, &placed, (eye, target), &mut frame);
    wait(flight.point);
    let old: &[trace::Traced] = bytemuck::cast_slice(&settled);
    let new: &[trace::Traced] = bytemuck::cast_slice(&traced);
    let mut hits = 0;
    for (a, b) in old.iter().zip(new) {
        assert_eq!(a.valid, b.valid);
        if a.instance == 0 {
            assert_eq!(b.instance, 0);
            continue;
        }
        hits += 1;
        let same = trace::Traced {
            instance: a.instance,
            material: a.material,
            ..*b
        };
        assert!(
            bytemuck::bytes_of(&same) == bytemuck::bytes_of(a),
            "{a:?} became {b:?}"
        );
    }
    assert!(hits > old.len() / 4, "the view traces the copies");
    assert_clean();
}

/// The mesh page budget falls to what the view wants: the hidden ground's pages go, the CLAS
/// pool's pages they leave wholly free unbind, and its page memory empties the blocks they
/// leave partly used by binding the pages the view traces to other memory, their CLAS in place.
/// Every frame meanwhile traces the hits of the frames before, and once settled the memory
/// holds the fewest blocks the pages bound need.
#[test]
fn clas_pool_pages_move_to_other_memory_under_the_frames_that_trace_them() {
    let _serial = serial();
    let mut c = ClasMoves::new();
    let (full, _) = c.pool();
    let moved = c.moved();
    let budget = c.view_budget();
    c.budget(budget);
    c.settle(budget, false);
    assert!(
        c.moved() > moved,
        "no page moved: {full:?} to {:?}",
        c.pool()
    );
    assert!(
        c.pool().0.bytes < full.bytes / 2,
        "{full:?} to {:?}",
        c.pool()
    );
    c.check_settled();
    assert_clean();
}

/// The binds of pages whose memory moves wait for many frames, as a loading side holding the
/// pool makes them, while the CLAS pool's other wholly free pages unbind: the held pages keep
/// their CLAS where frames trace them, and once the binds go the pages move and every frame
/// traces the same hits.
#[test]
fn clas_pool_pages_held_for_their_bind_trace_exact() {
    let _serial = serial();
    let mut c = ClasMoves::new();
    let moved = c.moved();
    c.w.scenes.hold_clas_binds(true);
    let budget = c.view_budget();
    c.budget(budget);
    let mut held_frames = 0;
    let start = Instant::now();
    while held_frames < 16 {
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "no page held: {:?}",
            c.pool()
        );
        c.exact();
        assert_eq!(c.moved(), moved, "a held bind went");
        held_frames += u32::from(c.pool().1.held > 0);
    }
    c.w.scenes.hold_clas_binds(false);
    c.settle(budget, false);
    assert!(c.moved() > moved, "no page moved: {:?}", c.pool());
    c.check_settled();
    assert_clean();
}

/// A frame begun and never submitted before each frame while pages move: their binds go on the
/// render queue at the step and their copies with the batches, not with the frames' lists, and
/// every frame that is submitted traces the same hits.
#[test]
fn frames_dropped_while_clas_pool_pages_move_leave_them_exact() {
    let _serial = serial();
    let mut c = ClasMoves::new();
    let moved = c.moved();
    let budget = c.view_budget();
    c.budget(budget);
    c.settle(budget, true);
    assert!(c.moved() > moved, "no page moved: {:?}", c.pool());
    c.check_settled();
    assert_clean();
}

/// The streaming ground settles at the configured detail under an ample mesh page pool, its
/// finer cuts built over the pages they hold. The pool's budget then falls to the pinned pages
/// and two fifths of the streamed ones the view wants: the detail scale rises to one that
/// fits and stays there, never above where it settles, while the pages the finer cuts held
/// leave with the cuts the coarser frames stop using.
#[test]
fn a_mesh_page_budget_cut_moves_the_detail_scale_to_what_fits_and_no_further() {
    let _serial = serial();
    let mesh_pages_class = orr_scene::ResidencyClass::MeshPages;
    // Every page streams, every copy traces a finer cut, and no cut merges.
    let mut w = World::with(&RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            ..Budgets::default()
        },
        rt_merge_pixels: 1e9,
        raster_error_pixels: 1e-3,
        rt_error_pixels: 1e-3,
        ..RendererDesc::default()
    });
    let pinned = w.scenes.residency().class(mesh_pages_class).resident_bytes;
    let items = dag_items();
    let placed = place(&mut w, &items);
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    let mut frame = 1;
    // Frames until the loads are idle and the scale has held for `still` of them; returns
    // every frame's detail.
    let mut settle = |w: &mut World, still: u32| {
        let start = Instant::now();
        let mut details: Vec<orr_scene::DetailStats> = Vec::new();
        let mut held = 0;
        while held < still {
            frame += 1;
            shoot(w, &placed, &Shot::at(eye, target, frame));
            let d = w.scenes.stats().detail;
            assert!(
                start.elapsed() < Duration::from_secs(120),
                "the detail never settled, at {d:?}"
            );
            let r = w.scenes.residency();
            let idle = r.loads == 0 && r.class(mesh_pages_class).loading_bytes == 0;
            let same = details
                .last()
                .is_some_and(|l| l.memory_scale == d.memory_scale);
            held = if idle && same { held + 1 } else { 0 };
            details.push(d);
        }
        details
    };
    let ample = *settle(&mut w, 48).last().unwrap();
    assert_eq!(ample.memory_scale, 1.0, "{ample:?}");
    let budget = pinned + (ample.mesh_demand - pinned) * 2 / 5;
    w.scenes.lower_budget(mesh_pages_class, budget);
    let cut = settle(&mut w, 96);
    let last = *cut.last().unwrap();
    assert!(last.memory_scale > 1.0, "{last:?}");
    assert!(
        last.mesh_demand as f64 <= 0.92 * last.mesh_budget as f64 && last.mesh_shortfall == 0,
        "{last:?}"
    );
    let scales: Vec<f32> = cut.iter().map(|d| d.memory_scale).collect();
    let most = scales.iter().copied().fold(0.0, f32::max);
    assert_eq!(
        most, last.memory_scale,
        "the scale passed where it settles: {scales:?}"
    );
    assert_eq!(w.scenes.residency().failed, 0);
    assert_clean();
}

/// The streaming ground, a wall before most of it, which the camera sees from in front, and a
/// second ground standing upright wholly behind the wall.
fn walled_fine_ground() -> [Item; 3] {
    [
        Item {
            model: "fine ground",
            linear: Mat3::from_rotation_x(0.25) * Mat3::from_diagonal(Vec3::new(2.0, 1.0, 4.0)),
            at: Vec3::new(0.0, -1.0, -12.0),
        },
        Item {
            model: "wall",
            linear: Mat3::from_diagonal(Vec3::new(10.0, 8.0, 1.0)),
            at: Vec3::new(0.0, 2.5, -6.0),
        },
        Item {
            model: "hidden ground",
            linear: Mat3::from_rotation_x(std::f32::consts::FRAC_PI_2)
                * Mat3::from_diagonal(Vec3::splat(0.8)),
            at: Vec3::new(0.3, 2.5, -8.0),
        },
    ]
}

/// Shoots frames of `placed` from the walled ground's camera, `moved` written each frame,
/// until the loads are idle and the detail scales have held for 24 frames; returns the last
/// frame and its detail.
fn settle_walled(
    w: &mut World,
    placed: &Placed,
    frame: &mut u32,
    moved: &[(u32, Affine3)],
) -> (Frame, orr_scene::DetailStats) {
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    let start = Instant::now();
    let (mut still, mut scale) = (0, -1.0);
    loop {
        *frame += 1;
        let mut shot = Shot::at(eye, target, *frame);
        shot.moved = moved.to_vec();
        let g = shoot(w, placed, &shot);
        let d = w.scenes.stats().detail;
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the detail never settled, at {d:?}"
        );
        let pool = *w
            .scenes
            .residency()
            .class(orr_scene::ResidencyClass::MeshPages);
        let idle = w.scenes.residency().loads == 0 && pool.loading_bytes == 0;
        still = if idle && d.memory_scale == scale {
            still + 1
        } else {
            0
        };
        scale = d.memory_scale;
        if still >= 24 {
            return (g, d);
        }
    }
}

/// The streaming grounds behind and before a wall. Under an ample mesh page pool the view
/// wants less with hidden thresholds coarser than at one threshold over everything
/// (`culled_error_scale` 1). At one threshold, a pool short of what that wants coarsens hidden
/// geometry alone, the visible scale holding at 1. Under a pool shorter still the wall goes:
/// every frame's cut is whole while the pages the revealed ground wants land, some clusters
/// standing in for them, and once the loads and the scales settle it takes the reference of
/// the hidden test over the last frame at the scales the frame ran at.
#[test]
fn hidden_geometry_sheds_mesh_pages_first_and_a_revealed_ground_stays_whole() {
    let _serial = serial();
    // Fine enough that the view wants the streamed pages of both grounds.
    const ERROR_PIXELS: f32 = 0.001;
    let bytes = fine_ground_pack();
    let world = |culled: f32, pool: PoolSize| {
        World::with_pack(
            &RendererDesc {
                raster_error_pixels: ERROR_PIXELS,
                rt_error_pixels: ERROR_PIXELS,
                rt_merge_pixels: 0.0,
                culled_error_scale: culled,
                budgets: Budgets {
                    mesh_page_pool: pool,
                    ..Budgets::default()
                },
                ..RendererDesc::default()
            },
            bytes,
        )
    };
    let items = walled_fine_ground();
    let ample = Budgets::default().mesh_page_pool;
    // Hidden thresholds the lead times coarser, as the short pool below takes them.
    let culled = RendererDesc::default().hidden_detail_lead;

    let mut w = world(1.0, ample);
    let placed = place(&mut w, &items);
    let (_, even) = settle_walled(&mut w, &placed, &mut 0, &[]);
    drop(w);

    let mut w = world(culled, ample);
    let placed = place(&mut w, &items);
    let mut frame = 0;
    let (_, seen) = settle_walled(&mut w, &placed, &mut frame, &[]);
    assert_eq!(seen.memory_scale, 1.0, "{seen:?}");
    assert!(
        seen.mesh_demand < even.mesh_demand,
        "the hidden test sheds no pages: {seen:?} against {even:?}"
    );

    drop(w);

    // At one threshold over everything, a pool that holds what the view wants with hidden
    // thresholds the lead times coarser, and not what it wants at one threshold: the memory
    // scale goes to the hidden thresholds.
    let pool = seen.mesh_demand * 5 / 4;
    assert!(
        even.mesh_demand as f64 > 0.92 * pool as f64,
        "the wall hides too little: {seen:?} against {even:?}"
    );
    let mut w = world(1.0, PoolSize::Bytes(pool));
    let placed = place(&mut w, &items);
    let mut frame = 0;
    let (_, short) = settle_walled(&mut w, &placed, &mut frame, &[]);
    assert!(
        short.memory_scale > 1.0,
        "a pool of {pool} B coarsens: {short:?}"
    );
    assert_eq!(
        short.visible_scale, 1.0,
        "visible detail holds while hidden detail coarsens: {short:?}"
    );
    assert!(
        short.mesh_demand as f64 <= 0.92 * short.mesh_budget as f64,
        "{short:?}"
    );
    drop(w);

    // Under a pool short enough that the pages the ground behind the wall would want are not
    // resident, the wall goes.
    let mut w = world(1.0, PoolSize::Bytes(seen.mesh_demand * 3 / 4));
    let placed = place(&mut w, &items);
    let mut frame = 0;
    settle_walled(&mut w, &placed, &mut frame, &[]);
    let model = w.model("fine ground");
    let m = CpuMesh::of_pack(bytes, "fine ground");
    let clusters = m.clusters();
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let slot = placed.ranges[0].first;
    let gone = [(1, Affine3::from_mat3_translation(Mat3::ZERO, Vec3::ZERO))];
    let target = DVec3::new(0.0, 0.0, -10.0);
    let start = Instant::now();
    let (mut standing, mut still, mut scale) = (0, 0, -1.0);
    let mut last: Option<SeenRender> = None;
    loop {
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the reveal never settled"
        );
        frame += 1;
        let mut shot = Shot::at(eye, target, frame);
        shot.moved = gone.to_vec();
        let g = shoot(&mut w, &placed, &shot);
        let (pages, bases, nodes) = mesh_pages(&w, model);
        let resident = Resident {
            pages: pages.iter().map(|&a| a != 0).collect(),
            nodes: nodes.iter().map(|&a| a != 0).collect(),
        };
        let cut = read_cut(&g.view, placed.count);
        let raster = cut.raster(&m, slot);
        let ray = cut.ray(&m, slot, &pages, &bases, &raster);
        let what = format!("frame {frame} after the wall");
        assert_whole(&m, &clusters, &raster, &format!("{what}: raster"));
        assert_whole(&m, &clusters, &ray, &format!("{what}: ray tracing"));
        standing = standing.max(
            raster
                .iter()
                .filter(|&&c| m.stands_in(&clusters[c as usize], &resident))
                .count(),
        );
        let d = w.scenes.stats().detail;
        let r = w.scenes.residency();
        let idle = r.loads == 0 && r.class(orr_scene::ResidencyClass::MeshPages).loading_bytes == 0;
        still = if idle && d.memory_scale == scale {
            still + 1
        } else {
            0
        };
        scale = d.memory_scale;
        if still >= 24 {
            let s = last.as_ref().expect("a frame before");
            let (want, near, _) = seen_rule_cut(
                &m,
                &clusters,
                &items[0],
                ERROR_PIXELS * d.visible_scale,
                d.hidden_scale,
                s,
                &resident,
            );
            assert_cut(&raster, &want, &near, &format!("{what} at {d:?}: raster"));
            break;
        }
        last = Some(SeenRender::of(&g.view, eye));
    }
    assert!(
        standing > 0,
        "no cluster stood in for the revealed ground's pages"
    );
    assert_eq!(w.scenes.residency().failed, 0);
    assert_clean();
}

// --- Skinned copies ------------------------------------------------------------------------

/// Heights of the worm's joints, root first, in bind and rest pose alike.
const WORM_JOINTS: [f32; 3] = [0.0, 0.55, 1.1];
const WORM_HEIGHT: f32 = 1.6;

/// A skinned GLB: a closed tube 1.6 m tall over a three-joint chain up its axis, weights
/// blending from joint to joint with height, and a `bend` clip that sways the upper two
/// joints about z over 2 s.
fn worm_glb() -> PathBuf {
    worm_glb_of("worm.glb", &[], false)
}

/// [`worm_glb`] at `name`, with `EXT_mesh_gpu_instancing` placing it at `copies`, and with `lean`
/// a shear in the rest pose of its middle joint.
fn worm_glb_of(name: &str, copies: &[[f32; 3]], lean: bool) -> PathBuf {
    let (rings, sides, radius) = (24u32, 16u32, 0.2f32);
    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut joints: Vec<u8> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut vertex = |p: Vec3, n: Vec3| {
        positions.extend(p.to_array());
        normals.extend(n.to_array());
        let y = p.y.clamp(0.0, WORM_HEIGHT);
        let (a, b) = (WORM_JOINTS[1], WORM_JOINTS[2]);
        let w = if y < a {
            [1.0 - y / a, y / a, 0.0]
        } else if y < b {
            [0.0, 1.0 - (y - a) / (b - a), (y - a) / (b - a)]
        } else {
            [0.0, 0.0, 1.0]
        };
        joints.extend([0, 1, 2, 0]);
        weights.extend([w[0], w[1], w[2], 0.0]);
    };
    for r in 0..=rings {
        let y = WORM_HEIGHT * r as f32 / rings as f32;
        for k in 0..sides {
            let a = std::f32::consts::TAU * k as f32 / sides as f32;
            let n = Vec3::new(a.cos(), 0.0, -a.sin());
            vertex(Vec3::new(0.0, y, 0.0) + n * radius, n);
        }
    }
    vertex(Vec3::ZERO, -Vec3::Y);
    vertex(Vec3::new(0.0, WORM_HEIGHT, 0.0), Vec3::Y);
    let bottom = (rings + 1) * sides;
    let top = bottom + 1;
    let at = |r: u32, k: u32| r * sides + k % sides;
    let mut indices: Vec<u32> = Vec::new();
    for r in 0..rings {
        for k in 0..sides {
            let (a, b, c, d) = (at(r, k), at(r, k + 1), at(r + 1, k + 1), at(r + 1, k));
            indices.extend([a, b, c, a, c, d]);
        }
    }
    for k in 0..sides {
        indices.extend([bottom, at(0, k + 1), at(0, k)]);
        indices.extend([top, at(rings, k), at(rings, k + 1)]);
    }
    let vertices = positions.len() / 3;
    let inverse_binds: Vec<f32> = WORM_JOINTS
        .iter()
        .flat_map(|&y| {
            [
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., -y, 0., 1.,
            ]
        })
        .collect();
    let times = [0.0f32, 0.5, 1.0, 1.5, 2.0];
    let angles = [0.0f32, 0.5, 0.0, -0.5, 0.0];
    let rotations: Vec<f32> = angles
        .iter()
        .flat_map(|a| [0.0, 0.0, (a / 2.0).sin(), (a / 2.0).cos()])
        .collect();

    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut add = |bytes: &[u8]| {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        views.len() - 1
    };
    let v_pos = add(bytemuck::cast_slice(&positions));
    let v_nrm = add(bytemuck::cast_slice(&normals));
    let v_jnt = add(&joints);
    let v_wgt = add(bytemuck::cast_slice(&weights));
    let v_idx = add(bytemuck::cast_slice(&indices));
    let v_ibm = add(bytemuck::cast_slice(&inverse_binds));
    let v_time = add(bytemuck::cast_slice(&times));
    let v_rot = add(bytemuck::cast_slice(&rotations));
    let (body, used, placed) = if copies.is_empty() {
        (String::new(), String::new(), String::new())
    } else {
        let v_copies = add(bytemuck::cast_slice(copies));
        (
            r#","extensions":{"EXT_mesh_gpu_instancing":{"attributes":{"TRANSLATION":8}}}"#.into(),
            r#""extensionsUsed":["EXT_mesh_gpu_instancing"],"#.into(),
            format!(
                r#",{{"bufferView":{v_copies},"componentType":5126,"count":{},"type":"VEC3"}}"#,
                copies.len()
            ),
        )
    };
    let m = WORM_JOINTS[1];
    let middle = if lean {
        format!(r#""matrix":[1,0,0,0, 0.3,1,0,0, 0,0,1,0, 0,{m},0,1]"#)
    } else {
        format!(r#""translation":[0,{m},0]"#)
    };
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},{used}"scene":0,"scenes":[{{"nodes":[0,3]}}],
"nodes":[{{"name":"root","children":[1]}},
{{"name":"mid",{middle},"children":[2]}},
{{"name":"top","translation":[0,{t},0]}},
{{"name":"body","mesh":0,"skin":0{body}}}],
"skins":[{{"joints":[0,1,2],"inverseBindMatrices":5}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3}},"indices":4}}]}}],
"animations":[{{"name":"bend","samplers":[{{"input":6,"output":7,"interpolation":"LINEAR"}}],
"channels":[{{"sampler":0,"target":{{"node":1,"path":"rotation"}}}},{{"sampler":0,"target":{{"node":2,"path":"rotation"}}}}]}}],
"accessors":[
{{"bufferView":{v_pos},"componentType":5126,"count":{vertices},"type":"VEC3","min":[{lo},0,{lo}],"max":[{hi},{h},{hi}]}},
{{"bufferView":{v_nrm},"componentType":5126,"count":{vertices},"type":"VEC3"}},
{{"bufferView":{v_jnt},"componentType":5121,"count":{vertices},"type":"VEC4"}},
{{"bufferView":{v_wgt},"componentType":5126,"count":{vertices},"type":"VEC4"}},
{{"bufferView":{v_idx},"componentType":5125,"count":{n_idx},"type":"SCALAR"}},
{{"bufferView":{v_ibm},"componentType":5126,"count":3,"type":"MAT4"}},
{{"bufferView":{v_time},"componentType":5126,"count":5,"type":"SCALAR","min":[0],"max":[2]}},
{{"bufferView":{v_rot},"componentType":5126,"count":5,"type":"VEC4"}}{placed}],
"bufferViews":[{views}],
"buffers":[{{"byteLength":{len}}}]}}"#,
        t = WORM_JOINTS[2] - WORM_JOINTS[1],
        lo = -radius,
        hi = radius,
        h = WORM_HEIGHT,
        n_idx = indices.len(),
        views = views.join(","),
        len = bin.len(),
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend(b"glTF");
    glb.extend(2u32.to_le_bytes());
    glb.extend(((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend((json.len() as u32).to_le_bytes());
    glb.extend(b"JSON");
    glb.extend(&json);
    glb.extend((bin.len() as u32).to_le_bytes());
    glb.extend(b"BIN\0");
    glb.extend(&bin);
    let path = temp(name);
    std::fs::write(&path, glb).unwrap();
    path
}

/// The worm's skeleton and clip, decoded from the pack.
fn worm_rig() -> (SkeletonInfo, Clip) {
    worm_rig_of("worm")
}

/// The skeleton and clip of worm `name`.
fn worm_rig_of(name: &str) -> (SkeletonInfo, Clip) {
    rig_of(name, "bend")
}

/// The skeleton of asset `name` and its clip `clip`.
fn rig_of(name: &str, clip: &str) -> (SkeletonInfo, Clip) {
    let a = Archive::from_memory(Arc::clone(pack())).unwrap();
    let skeleton = a
        .find(AssetId::from_name(&format!("{name}/skeleton")))
        .unwrap();
    let clip = a
        .find(AssetId::from_name(&format!("{name}/clip/{clip}")))
        .unwrap();
    (
        SkeletonInfo::parse(&blob(&a, skeleton, 0)).unwrap(),
        Clip::parse(&blob(&a, clip, 0), &blob(&a, clip, 1)).unwrap(),
    )
}

/// The worm's skinning matrices `t` seconds into its clip, as orr_pack's CPU reference
/// samples and chains them.
fn worm_palette(t: f32) -> Vec<Affine3> {
    worm_palette_of("worm", t)
}

/// [`worm_palette`] of worm `name`.
fn worm_palette_of(name: &str, t: f32) -> Vec<Affine3> {
    palette_of(name, "bend", t)
}

/// The skinning matrices of asset `name`, `t` seconds into its clip `clip`.
fn palette_of(name: &str, clip: &str, t: f32) -> Vec<Affine3> {
    let (skeleton, clip) = rig_of(name, clip);
    let n = skeleton.joint_count();
    let mut locals = vec![Trs::IDENTITY; n];
    clip.sample(t, &mut locals);
    let mut model = vec![Affine3::IDENTITY; n];
    skeleton.model_pose(&locals, &mut model);
    let mut palette = vec![Affine3::IDENTITY; n];
    skeleton.palette(&model, &mut palette);
    palette
}

/// Skinned worms in one transform group, their palette in a second group.
struct Worms {
    placed: Placed,
    palette: TransformGroupId,
    joints: u32,
    /// The asset whose skeleton and clip pose them.
    rig: String,
    clip: String,
}

/// Where `crowd`'s instances place its worms, in one world cell and in record order.
const CROWD: [[f32; 3]; 2] = [[0.2, 0.0, 0.1], [1.0, 0.0, 0.1]];

/// m/s of [`paced_crowd`].
const PACE: Vec3 = Vec3::new(0.3, 0.0, 0.0);

/// Records of [`long_walk`].
const WALK_RECORDS: usize = 1024;
/// Its intervals, a tenth of a second each.
const WALK_INTERVALS: usize = 200;
/// How far along x it steps at each.
const WALK_STEP: f32 = 0.004;

/// A page of copies far longer in rows than in records: the first starts at (0.5, 0, 0) in view,
/// the rest stand behind the eye, and all step along x at every interval.
fn long_walk() -> Instances {
    let rest = |i: usize| match i {
        0 => [0.5, 0.0, 0.0],
        _ => [i as f32 * 0.01, 0.0, 20.0],
    };
    let row = |k: usize, i: usize| {
        let [x, y, z] = rest(i);
        MotionRow {
            position: [x + WALK_STEP * k as f32, y, z],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            ..MotionRow::default()
        }
    };
    let spot = |i: usize| Placement {
        position: rest(i).map(f64::from),
        linear: Linear::Scaled {
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        },
        prototype: 0,
    };
    let rows = (0..WALK_INTERVALS).flat_map(|k| (0..WALK_RECORDS).map(move |i| row(k, i)));
    Instances {
        records: (0..WALK_RECORDS).map(spot).collect(),
        prototypes: vec![PrototypeSpec {
            model: "plain".into(),
            ..PrototypeSpec::default()
        }],
        motion: Some(Arc::new(InstanceMotion {
            times: (0..WALK_INTERVALS).map(|k| k as f32 * 0.1).collect(),
            rows: rows.collect(),
            anchors: vec![[0.0; 3]; WALK_RECORDS],
            left: DMat4::IDENTITY,
            right: DMat4::IDENTITY,
            start: 0.0,
        })),
        ..Instances::default()
    }
}

/// A world of `pool` bytes of instance pages, to watch [`long_walk`] in.
fn walk_world(pool: u64) -> World {
    World::with(&RendererDesc {
        raster_error_pixels: 1e-3,
        rt_error_pixels: 1e-3,
        budgets: Budgets {
            instance_page_pool: pool,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    })
}

/// A shot of [`long_walk`] at `t` s of its clock.
fn walk_shot(w: &mut World, placed: &Placed, frame: &mut u32, t: f32) -> Frame {
    *frame += 1;
    let (eye, target) = (DVec3::new(0.9, 0.0, 2.5), DVec3::new(0.9, 0.0, 0.0));
    let mut shot = Shot::at(eye, target, *frame);
    shot.reset = *frame == 1;
    let range = placed.ranges[0];
    shoot_with(w, placed, &shot, |s| s.set_page_time(range, t))
}

/// The middle column of what a shot of [`long_walk`] sees.
fn walk_middle(g: &Frame) -> Option<f64> {
    let columns = g.ids.iter().enumerate().filter(|(_, id)| **id != 0);
    let (n, sum) = columns.fold((0u32, 0.0f64), |(n, sum), (i, _)| {
        (n + 1, sum + f64::from(i as u32 % EXTENT[0]))
    });
    (n > 0).then(|| sum / f64::from(n))
}

/// Pixels the copy of [`long_walk`] in view has moved by `t` s.
fn walk_px(t: f32) -> f64 {
    let interval = f64::from((t / 0.1 + 1e-3).floor());
    let f = 1.0 / (0.45f64).tan();
    let aspect = f64::from(EXTENT[0]) / f64::from(EXTENT[1]);
    0.5 * f64::from(WALK_STEP) * interval / 2.5 * f / aspect * f64::from(EXTENT[0])
}

/// Where [`long_walk`]'s copy stands at its start, once the page lands.
fn walk_rest(w: &mut World, placed: &Placed, frame: &mut u32) -> f64 {
    let start = Instant::now();
    loop {
        if let Some(x) = walk_middle(&walk_shot(w, placed, frame, 0.0)) {
            return x;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the page never landed"
        );
    }
}

/// Worms of [`CROWD`] walking at [`PACE`], on a page of the model all of `worm`'s copies share.
fn paced_crowd() -> Instances {
    let row = |p: [f32; 3]| MotionRow {
        position: p,
        velocity: PACE.to_array(),
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        ..MotionRow::default()
    };
    let spot = |p: [f32; 3]| Placement {
        position: p.map(f64::from),
        linear: Linear::Scaled {
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        },
        prototype: 0,
    };
    Instances {
        records: CROWD.map(spot).to_vec(),
        prototypes: vec![PrototypeSpec {
            model: "worm".into(),
            ..PrototypeSpec::default()
        }],
        motion: Some(Arc::new(InstanceMotion {
            times: vec![0.0],
            rows: CROWD.map(row).to_vec(),
            anchors: vec![[0.0; 3]; CROWD.len()],
            left: DMat4::IDENTITY,
            right: DMat4::IDENTITY,
            start: 0.0,
        })),
        ..Instances::default()
    }
}

/// The copies of `crowd`'s page, with a palette group written on the CPU.
fn place_crowd(w: &mut World) -> Worms {
    place_crowd_of(w, "crowd")
}

/// The copies of `asset`'s page, with a palette group written on the CPU.
fn place_crowd_of(w: &mut World, asset: &str) -> Worms {
    let a = Archive::from_memory(Arc::clone(pack())).unwrap();
    let prefix = format!("{asset}/");
    let is_crowd = |e: &&Entry| {
        e.class == orr_pack::Class::InstancePage as u16 && a.name(e).starts_with(&prefix)
    };
    let entry = a.entries().iter().find(is_crowd).expect("the crowd's page");
    let page = w.scenes.instance_page(w.mount, entry.id).unwrap();
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let joints = worm_rig().0.joint_count() as u32;
    let n = CROWD.len() as u32;
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    let palette = s
        .create_transform_group(&TransformGroupDesc {
            count: n * joints,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let refused = s.add_instance_page(&InstancePageRef {
        frame: None,
        page,
        poses: &[],
        material_sets: &[],
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 100.0,
        animation_range: 0.0,
        user_tag: 0,
    });
    assert!(
        refused.is_err(),
        "a page of skinned copies would draw at the bind pose"
    );
    let range = s
        .add_skinned_page(&SkinnedPageRef {
            frame: None,
            page,
            palette,
            palette_first: 0,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 100.0,
            user_tag: 0,
        })
        .unwrap();
    assert_eq!(range.count, n);
    drop(s);
    wait(f.end(&[]).unwrap());
    Worms {
        placed: Placed {
            scene,
            ranges: vec![range],
            group: palette,
            history,
            count: n,
        },
        palette,
        joints,
        rig: "worm".to_string(),
        clip: "bend".to_string(),
    }
}

/// Worm copies placed by `copies`, with a palette group written on the CPU or, with
/// `gpu`, by a scene compute kernel.
fn place_worms(w: &mut World, name: &str, copies: &[(Mat3, Vec3)], gpu: bool) -> Worms {
    let model = w.model(name);
    Worms {
        rig: name.to_string(),
        clip: "bend".to_string(),
        ..place_worms_of(w, model, copies, gpu)
    }
}

fn place_worms_of(w: &mut World, model: ModelId, copies: &[(Mat3, Vec3)], gpu: bool) -> Worms {
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let joints = worm_rig().0.joint_count() as u32;
    let n = copies.len() as u32;
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: n,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let palette = s
        .create_transform_group(&TransformGroupDesc {
            count: n * joints,
            writer: if gpu {
                GroupWriter::Gpu
            } else {
                GroupWriter::Cpu
            },
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let out = s.transforms(group, 0, n).unwrap();
    for (o, &(m, t)) in out.iter_mut().zip(copies) {
        *o = Affine3::from_mat3_translation(m, t).into();
    }
    let range = s
        .add_skinned(&SkinnedDesc {
            model,
            copies: n,
            transforms: group,
            first: 0,
            palette,
            palette_first: 0,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 100.0,
            user_tag: 0,
        })
        .unwrap();
    drop(s);
    wait(f.end(&[]).unwrap());
    Worms {
        placed: Placed {
            scene,
            ranges: vec![range],
            group,
            history,
            count: n,
        },
        palette,
        joints,
        rig: "worm".to_string(),
        clip: "bend".to_string(),
    }
}

/// Writes every worm's palette, copy `i` posed `t + i x phase` seconds into the clip.
fn write_palette(s: &mut SceneWriter<'_>, worms: &Worms, t: f32, phase: f32) {
    let n = worms.placed.count;
    let out = s.transforms(worms.palette, 0, n * worms.joints).unwrap();
    for (i, chunk) in out.chunks_mut(worms.joints as usize).enumerate() {
        for (o, p) in chunk.iter_mut().zip(worm_palette(t + i as f32 * phase)) {
            *o = p.into();
        }
    }
}

/// One frame of the worms posed at `t`.
fn shoot_worms(w: &mut World, worms: &Worms, shot: &Shot, t: f32, phase: f32) -> Frame {
    shoot_with(w, &worms.placed, shot, |s| {
        write_palette(s, worms, t, phase);
    })
}

fn worm_shot(frame: u32) -> Shot {
    Shot::at(DVec3::new(0.2, 0.9, 3.2), DVec3::new(0.0, 0.8, 0.0), frame)
}

/// Skinned copy `entry`'s record and state, and the clusters it skinned this frame.
fn skinned_entry(view: &RasterView, entry: u32) -> (SkinnedGpu, SkinStateGpu, Vec<u32>) {
    let root = read_buffer(view.scene, 1)[0];
    let k = read_buffer(root.skinned.offset(u64::from(entry)), 1)[0];
    let st = read_buffer(root.skin_state.offset(u64::from(entry)), 1)[0];
    let work: Vec<[u32; 2]> = read_buffer(
        root.skin_work
            .cast::<[u32; 2]>()
            .offset(u64::from(st.first)),
        u64::from(st.clusters.max(1)),
    );
    let clusters = work[..st.clusters as usize].iter().map(|w| w[1]).collect();
    (k, st, clusters)
}

/// The largest distance between `positions` (a copy's vertex array) and the CPU skin of the
/// mesh's vertices in `clusters` by `palette`.
fn skin_error(mesh: &CpuMesh, positions: &[Vec3], clusters: &[u32], palette: &[Affine3]) -> f32 {
    let mut first = Vec::new();
    let mut total = 0u32;
    for page in &mesh.pages {
        let view = PageView::new(page, &mesh.header).unwrap();
        for i in 0..view.cluster_count() {
            first.push(total);
            total += u32::from(view.cluster(i).vertex_count);
        }
    }
    let mut worst = 0.0f32;
    for &c in clusters {
        let p = mesh.firsts.partition_point(|&f| f <= c) - 1;
        let view = PageView::new(&mesh.pages[p], &mesh.header).unwrap();
        let cl = view.cluster(c - mesh.firsts[p]);
        let mut want = vec![[0.0f32; 3]; usize::from(cl.vertex_count)];
        orr_pack::mesh::skin(&view, &cl, palette, &mut want);
        for (i, w) in want.iter().enumerate() {
            let got = positions[first[c as usize] as usize + i];
            worst = worst.max(got.distance(Vec3::from(*w)));
        }
    }
    worst
}

#[test]
fn skinned_vertices_match_the_cpu_skin_by_this_and_the_last_palette() {
    let _serial = serial();
    let mut w = finest_world();
    let worms = place_worms(&mut w, "worm", &[(Mat3::IDENTITY, Vec3::ZERO)], false);
    let mesh = CpuMesh::of("worm");
    shoot_worms(&mut w, &worms, &worm_shot(1), 0.3, 0.0);
    let g = shoot_worms(&mut w, &worms, &worm_shot(2), 0.7, 0.0);
    let (k, st, clusters) = skinned_entry(&g.view, 0);
    assert!(!clusters.is_empty(), "the copy skinned nothing");
    assert_eq!(st.clusters as usize, clusters.len());
    let vertices: u32 = mesh
        .pages
        .iter()
        .map(|p| {
            let v = PageView::new(p, &mesh.header).unwrap();
            (0..v.cluster_count())
                .map(|i| u32::from(v.cluster(i).vertex_count))
                .sum::<u32>()
        })
        .sum();
    let read = |p: GpuPtr<[f32; 3]>| -> Vec<Vec3> {
        read_buffer(p, u64::from(vertices))
            .into_iter()
            .map(Vec3::from)
            .collect()
    };
    let now = skin_error(
        &mesh,
        &read(k.positions.cast()),
        &clusters,
        &worm_palette(0.7),
    );
    let before = skin_error(
        &mesh,
        &read(k.previous.cast()),
        &clusters,
        &worm_palette(0.3),
    );
    assert!(now < 1e-5, "positions {now} m from the CPU skin");
    assert!(
        before < 1e-5,
        "previous positions {before} m from the CPU skin"
    );
    assert_clean();
}

/// Unmounting the pack of a skinned page's model takes the page from the scene whole: its copies,
/// their skinned entries, and the groups of its records and its motion.
#[test]
fn an_unmount_takes_a_skinned_page_and_its_motion_from_the_scene() {
    let _serial = serial();
    let manifest = Manifest {
        archives: vec![
            ArchiveSpec::new(
                Target::Memory,
                vec![AssetSpec {
                    name: "worm".into(),
                    source: Source::Gltf {
                        path: worm_glb(),
                        settings: GltfSettings {
                            mesh: MeshSettings {
                                deformation_bound: 1.0,
                                ..MeshSettings::default()
                            },
                            ..GltfSettings::default()
                        },
                    },
                }],
            ),
            ArchiveSpec::new(
                Target::Memory,
                vec![AssetSpec {
                    name: "paced crowd".into(),
                    source: Source::Instances(paced_crowd()),
                }],
            ),
        ],
    };
    let mut packs = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory;
    let (pages, models) = (packs.remove(1), packs.remove(0));
    let archive = Archive::from_memory(Arc::clone(&pages)).unwrap();
    let is_page = |e: &&Entry| e.class == orr_pack::Class::InstancePage as u16;
    let entry = archive.entries().iter().find(is_page).expect("the page");

    let mut scenes = Scenes::new(device(), &RendererDesc::default()).unwrap();
    let _workers = Workers::start(&scenes, 2);
    let scene = scenes.create_scene(&SceneDesc::default()).unwrap();
    let list = [scene];
    let desc = FrameDesc {
        time: time(0),
        scenes: &list,
        low_latency: None,
    };
    let mounted = |scenes: &mut Scenes, pack| {
        let m = scenes.mount(PackSource::Memory(pack));
        while matches!(scenes.mount_state(m), LoadState::Pending) {
            scenes.begin_frame(&desc).unwrap().end(&[]).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(matches!(scenes.mount_state(m), LoadState::Ready));
        m
    };
    let models = mounted(&mut scenes, models);
    let pack = mounted(&mut scenes, pages);
    let page = scenes.instance_page(pack, entry.id).unwrap();
    let joints = worm_rig().0.joint_count() as u32;

    let frame = |scenes: &mut Scenes, write: &mut dyn FnMut(&mut SceneWriter<'_>)| {
        let mut f = scenes.begin_frame(&desc).unwrap();
        write(&mut f.scene(scene).unwrap());
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| root = c.root).unwrap();
        wait(f.end(&[]).unwrap());
        root
    };
    let root: SceneRoot = read_buffer(
        frame(&mut scenes, &mut |s| {
            let palette = s
                .create_transform_group(&TransformGroupDesc {
                    count: CROWD.len() as u32 * joints,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .unwrap();
            s.add_skinned_page(&SkinnedPageRef {
                frame: None,
                page,
                palette,
                palette_first: 0,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 100.0,
                user_tag: 0,
            })
            .unwrap();
        }),
        1,
    )[0];
    // Live copies, skinned entries and groups.
    let state = || {
        let copies = read_buffer(root.instances, u64::from(root.part_base));
        let skins = read_buffer(root.skinned, u64::from(root.skinned_capacity));
        let groups = read_buffer(root.groups, u64::from(SceneDesc::default().group_capacity));
        [
            copies.iter().filter(|c| c.flags != 0).count(),
            skins.iter().filter(|k| k.positions != GpuPtr::NULL).count(),
            groups.iter().filter(|g| g.count != 0).count(),
        ]
    };
    assert_eq!(state(), [CROWD.len(), CROWD.len(), 3]);
    scenes.unmount(models);
    for _ in 0..3 {
        frame(&mut scenes, &mut |_| {});
    }
    assert_eq!(state(), [0, 0, 1], "only the palette stays");
    assert_clean();
}

/// A skinned copy of an animated page stands where its rows say, like one placed there by hand:
/// the rows give the transform and the palette the pose.
#[test]
fn skinned_copies_on_an_animated_page_follow_its_clock() {
    let _serial = serial();
    let mut w = finest_world();
    let by_hand = CROWD.map(|p| (Mat3::IDENTITY, Vec3::from(p) + PACE));
    let hand = place_worms(&mut w, "worm", &by_hand, false);
    let want = shoot_worms(&mut w, &hand, &worm_shot(1), 0.7, 0.45);
    let crowd = place_crowd_of(&mut w, "paced crowd");
    let range = crowd.placed.ranges[0];
    let shoot_at = |w: &mut World, frame: u32| {
        shoot_with(w, &crowd.placed, &worm_shot(frame), |s| {
            write_palette(s, &crowd, 0.7, 0.45);
            s.set_page_time(range, 1.0);
        })
    };
    let start = Instant::now();
    let mut frame = 2;
    let g = loop {
        frame += 1;
        let g = shoot_at(&mut w, frame);
        if frame > 4 && g.ids.iter().any(|&id| id != 0) {
            break shoot_at(&mut w, frame + 1);
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the page never landed"
        );
    };
    let n = g.depth.len();
    let same = (0..n)
        .filter(|&i| (want.depth[i] - g.depth[i]).abs() <= 2e-3 * want.depth[i].max(1e-3))
        .count();
    let drawn = g.ids.iter().filter(|&&id| id != 0).count();
    assert!(drawn > n / 50, "{drawn} of {n} pixels drawn");
    assert!(
        same as f64 >= 0.995 * n as f64,
        "{same} of {n} depths agree"
    );
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, _) = parity(&g, &r, 1e-4);
    assert!(
        matched as f64 >= 0.999 * r.traced.len() as f64,
        "{matched} of {} pixels agree",
        r.traced.len()
    );
    assert_clean();
}

/// A page of skinned copies draws and traces like the same copies placed by hand, each on its
/// own pose: the page gives the transform and the palette the pose.
#[test]
fn skinned_copies_on_a_page_pose_like_copies_placed_by_hand() {
    let _serial = serial();
    let mut w = finest_world();
    let by_hand = CROWD.map(|p| (Mat3::IDENTITY, Vec3::from(p)));
    let hand = place_worms(&mut w, "worm", &by_hand, false);
    let want = shoot_worms(&mut w, &hand, &worm_shot(1), 0.7, 0.45);
    let crowd = place_crowd(&mut w);
    let mesh = CpuMesh::of("crowd/body");
    let start = Instant::now();
    let mut frame = 2;
    let g = loop {
        frame += 1;
        let g = shoot_worms(&mut w, &crowd, &worm_shot(frame), 0.7, 0.45);
        if frame > 4 && g.ids.iter().any(|&id| id != 0) {
            break shoot_worms(&mut w, &crowd, &worm_shot(frame + 1), 0.7, 0.45);
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the page never landed"
        );
    };
    let n = g.depth.len();
    let same = (0..n)
        .filter(|&i| (want.depth[i] - g.depth[i]).abs() <= 2e-3 * want.depth[i].max(1e-3))
        .count();
    let drawn = g.ids.iter().filter(|&&id| id != 0).count();
    assert!(drawn > n / 50, "{drawn} of {n} pixels drawn");
    assert!(
        same as f64 >= 0.995 * n as f64,
        "{same} of {n} depths agree"
    );
    let vertices: u32 = mesh
        .pages
        .iter()
        .map(|p| {
            let v = PageView::new(p, &mesh.header).unwrap();
            (0..v.cluster_count())
                .map(|i| u32::from(v.cluster(i).vertex_count))
                .sum::<u32>()
        })
        .sum();
    for copy in 0..CROWD.len() as u32 {
        let (k, _, clusters) = skinned_entry(&g.view, copy);
        assert!(!clusters.is_empty(), "copy {copy} skinned nothing");
        let at: Vec<Vec3> = read_buffer::<[f32; 3]>(k.positions.cast(), u64::from(vertices))
            .into_iter()
            .map(Vec3::from)
            .collect();
        let pose = worm_palette(0.7 + 0.45 * copy as f32);
        let error = skin_error(&mesh, &at, &clusters, &pose);
        assert!(error < 1e-5, "copy {copy} is {error} m from its own pose");
    }
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, _) = parity(&g, &r, 1e-4);
    assert!(
        matched as f64 >= 0.999 * r.traced.len() as f64,
        "{matched} of {} pixels agree",
        r.traced.len()
    );
    assert_clean();
}

/// One frame of the worms posed at `t` by the test pose kernel.
fn shoot_gpu_worms(
    w: &mut World,
    worms: &Worms,
    kernel: &ComputePipeline,
    shot: &Shot,
    t: f32,
    phase: f32,
) -> Frame {
    let skeleton = w
        .scenes
        .skeleton(
            w.mount,
            AssetId::from_name(&format!("{}/skeleton", worms.rig)),
        )
        .unwrap();
    let clip = w
        .scenes
        .clip(
            w.mount,
            AssetId::from_name(&format!("{}/clip/{}", worms.rig, worms.clip)),
        )
        .unwrap();
    assert_eq!(skeleton.joint_count, worms.joints);
    assert_eq!(clip.joint_count, worms.joints);
    shoot_computed(
        w,
        &worms.placed,
        shot,
        |_| {},
        |c| {
            let root = c
                .put_root(&skin_pose::Root {
                    scene: c.root,
                    skeleton: skeleton.blob,
                    clip_header: clip.header,
                    clip_samples: clip.samples,
                    group: worms.palette.0,
                    copies: worms.placed.count,
                    time: t,
                    phase,
                })
                .unwrap();
            c.cmd.dispatch(kernel, root, [1, 1, 1]);
        },
    )
}

#[test]
fn a_pose_kernel_samples_clips_on_the_gpu_like_the_cpu() {
    let _serial = serial();
    let mut w = finest_world();
    let copies = [(Mat3::IDENTITY, Vec3::ZERO), (Mat3::IDENTITY, Vec3::X)];
    assert_ne!(worm_rig_of("leaning worm").0.rest[1].shear, [0.0; 3]);
    let kernel = pipeline(&skin_pose::MODULE, "palette");
    for (frame, name) in (1..).zip(["worm", "leaning worm"]) {
        let worms = place_worms(&mut w, name, &copies, true);
        let (t, phase) = (0.37, 0.81);
        let shot = worm_shot(frame);
        let frame = shoot_gpu_worms(&mut w, &worms, &kernel, &shot, t, phase);
        let view = &frame.view;
        let root = read_buffer(view.scene, 1)[0];
        let group = read_buffer(root.groups.offset(u64::from(worms.palette.0)), 1)[0];
        let got: Vec<[[f32; 4]; 3]> = read_buffer(
            group.current.cast::<[[f32; 4]; 3]>(),
            u64::from(2 * worms.joints),
        );
        let mut worst = 0.0f32;
        for copy in 0..2 {
            let want = worm_palette_of(name, t + copy as f32 * phase);
            for (j, p) in want.iter().enumerate() {
                let g = got[copy * worms.joints as usize + j];
                for (a, b) in g
                    .iter()
                    .flatten()
                    .zip(AffineRows::from(*p).rows.iter().flatten())
                {
                    worst = worst.max((a - b).abs());
                }
            }
        }
        assert!(
            worst < 1e-5,
            "{name}: palette entries {worst} from the CPU's"
        );
        assert!(
            frame.ids.iter().any(|&id| id != 0),
            "{name}: the GPU-posed copies draw"
        );
    }

    // A shear that varies over the clip, read between frames, is the kernel's shear lerp. The
    // worms stand in for what the palette poses: only its entries are read.
    let (name, clip) = ("swaying door", "xform");
    let (skeleton, sway) = rig_of(name, clip);
    let n = skeleton.joint_count();
    let frames = sway.header.frame_count as usize;
    let at = |f: usize| sway.samples[f * n + 1].shear;
    assert_ne!(at(0), at(frames - 1), "the shear holds still");
    let times = [0.37f32, 0.81];
    for t in times {
        let x = t * sway.header.sample_rate;
        assert!((x - x.round()).abs() > 0.1, "{t} s lies on a frame");
    }
    let worms = place_worms(&mut w, "worm", &copies, true);
    let worms = Worms {
        rig: name.to_string(),
        clip: clip.to_string(),
        joints: n as u32,
        ..worms
    };
    let (t, phase) = (times[0], times[1] - times[0]);
    let frame = shoot_gpu_worms(&mut w, &worms, &kernel, &worm_shot(3), t, phase);
    let root = read_buffer(frame.view.scene, 1)[0];
    let group = read_buffer(root.groups.offset(u64::from(worms.palette.0)), 1)[0];
    let got: Vec<[[f32; 4]; 3]> = read_buffer(group.current.cast::<[[f32; 4]; 3]>(), 2 * n as u64);
    let mut worst = 0.0f32;
    for copy in 0..2 {
        let want = palette_of(name, clip, t + copy as f32 * phase);
        for (j, p) in want.iter().enumerate() {
            let rows = AffineRows::from(*p).rows;
            for (a, b) in got[copy * n + j]
                .iter()
                .flatten()
                .zip(rows.iter().flatten())
            {
                worst = worst.max((a - b).abs());
            }
        }
    }
    assert!(
        worst < 1e-5,
        "{name}: palette entries {worst} from the CPU's"
    );
    assert_clean();
}

#[test]
fn camera_rays_hit_skinned_copies_where_the_raster_drew_them() {
    let _serial = serial();
    let mut w = finest_world();
    let copies = [
        (Mat3::IDENTITY, Vec3::new(-0.8, 0.0, 0.0)),
        (Mat3::from_rotation_y(0.7), Vec3::new(0.0, 0.0, -0.4)),
        (Mat3::from_rotation_y(-1.2) * 0.8, Vec3::new(0.8, 0.1, 0.2)),
    ];
    // With a deformation bound the instances keep one AABB over every pose; without one
    // they are rewritten with each frame's bounds.
    for name in ["worm", "unbounded worm"] {
        let worms = place_worms(&mut w, name, &copies, false);
        for (frame, t) in [(1, 0.4), (2, 1.3)] {
            let g = shoot_worms(&mut w, &worms, &worm_shot(frame), t, 0.45);
            assert!(g.view.features.contains(orr_scene::SceneFeatures::SKINNING));
            assert!(!g.view.features.contains(orr_scene::SceneFeatures::DECALS));
            let r = rays(&g.view, &trace::MODULE, "pixels", 0);
            let (matched, covered) = parity(&g, &r, 1e-4);
            let total = r.traced.len();
            assert!(
                covered > total / 10,
                "{name}: {covered} of {total} pixels covered"
            );
            assert!(
                matched as f64 >= 0.999 * total as f64,
                "{name}: {matched} of {total} pixels agree at {t} s"
            );
        }
    }
    assert_clean();
}

#[test]
fn motion_vectors_of_skinned_copies_carry_their_deformation() {
    let _serial = serial();
    let mut w = finest_world();
    let worms = place_worms(&mut w, "worm", &[(Mat3::IDENTITY, Vec3::ZERO)], false);
    // The copy and the camera stay put; only the palette moves, so only the deformation
    // takes a pixel's cluster elsewhere.
    let before = shoot_worms(&mut w, &worms, &worm_shot(1), 0.1, 0.0);
    let after = shoot_worms(&mut w, &worms, &worm_shot(2), 0.45, 0.0);
    let mismatch = |scale: f32| {
        let (mut counted, mut wrong) = (0, 0);
        for (i, &c) in after.clusters.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let m = Vec2::from(after.motion[i]) * scale;
            let Some(found) = footprint_has(&before.clusters, uv(i) + m, c) else {
                continue;
            };
            counted += 1;
            if !found {
                wrong += 1;
            }
        }
        (wrong, counted)
    };
    let (wrong, counted) = mismatch(1.0);
    let (still, _) = mismatch(0.0);
    assert!(
        still * 10 > counted,
        "the copy bends too little to test: {still} of {counted}"
    );
    assert!(
        wrong * 50 < counted,
        "{wrong} of {counted} pixels reproject onto another cluster ({still} without motion)"
    );
    assert_clean();
}

/// A pose that jumps moves a skinned copy as far as it really went, from a CPU or a kernel
/// palette; written without motion it stays put.
#[test]
fn a_pose_jump_keeps_valid_previous_positions() {
    let _serial = serial();
    let mut w = finest_world();
    let one = [(Mat3::IDENTITY, Vec3::ZERO)];
    let (from, to) = (0.1, 1.3);

    let cpu = place_worms(&mut w, "worm", &one, false);
    shoot_worms(&mut w, &cpu, &worm_shot(1), from, 0.0);
    let jumped = shoot_worms(&mut w, &cpu, &worm_shot(2), to, 0.0);
    let reach = largest_motion(&jumped);
    assert!(reach > 2.0, "the jump moves {reach} pixels");
    let declared = shoot_with(&mut w, &cpu.placed, &worm_shot(3), |s| {
        let out = s
            .transforms_without_motion(cpu.palette, 0, cpu.joints)
            .unwrap();
        for (o, p) in out.iter_mut().zip(worm_palette(from)) {
            *o = p.into();
        }
    });
    let reach = largest_motion(&declared);
    assert!(reach < 0.01, "a jump without motion moves {reach} pixels");

    let kernel = pipeline(&skin_pose::MODULE, "palette");
    let gpu = place_worms(&mut w, "worm", &one, true);
    shoot_gpu_worms(&mut w, &gpu, &kernel, &worm_shot(1), from, 0.0);
    let posed = shoot_gpu_worms(&mut w, &gpu, &kernel, &worm_shot(2), to, 0.0);
    let (apart, compared) = motion_difference(&jumped, &posed);
    assert!(compared > 100, "{compared} pixels compared");
    assert!(
        apart < 0.05,
        "kernel and CPU palettes differ by {apart} pixels"
    );
    assert_clean();
}

// --- Stochastic texture filtering ------------------------------------------------------------

#[test]
fn stochastic_texture_taps_average_to_the_bicubic_filter_under_magnification() {
    use orr_gpu::{
        AddressMode, Filter, Format, SamplerDesc, TextureDesc, TextureMemory, TextureUsage,
    };
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};

    const SIZE: u32 = 8;
    const TAPS: u32 = 4096;
    let _guard = serial();
    let d = device();
    let texels: Vec<[u8; 4]> = (0..SIZE * SIZE)
        .map(|i| {
            let h = (i.wrapping_mul(2_654_435_761) >> 7) as u8;
            [h, h.wrapping_mul(7), h ^ 0x5a, 255]
        })
        .collect();
    let texel = |x: i32, y: i32| -> [f32; 4] {
        let n = SIZE as i32;
        texels[(y.rem_euclid(n) * n + x.rem_euclid(n)) as usize].map(|c| f32::from(c) / 255.0)
    };
    // The repeating texture filtered with `weights(t)` over taps `-r + 1..=r` around the
    // texel below each sample.
    let filter = |uv: [f32; 2], r: i32, weights: &dyn Fn(f32, i32) -> f32| -> [f32; 4] {
        let x = uv[0] * SIZE as f32 - 0.5;
        let y = uv[1] * SIZE as f32 - 0.5;
        let (bx, by) = (x.floor(), y.floor());
        let (tx, ty) = (x - bx, y - by);
        let mut sum = [0.0; 4];
        for j in -r + 1..=r {
            for i in -r + 1..=r {
                let w = weights(tx, i) * weights(ty, j);
                let t = texel(bx as i32 + i, by as i32 + j);
                for c in 0..4 {
                    sum[c] += w * t[c];
                }
            }
        }
        sum
    };
    let bspline = |t: f32, k: i32| -> f32 {
        let s = 1.0 - t;
        match k {
            -1 => s * s * s / 6.0,
            0 => (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0,
            1 => (1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t * t * t) / 6.0,
            _ => t * t * t / 6.0,
        }
    };
    let linear = |t: f32, k: i32| -> f32 { if k == 0 { 1.0 - t } else { t } };
    let cubic = |uv| filter(uv, 2, &bspline);
    let bilinear = |uv| filter(uv, 1, &linear);

    let uvs = [[0.31, 0.47], [0.02, 0.97], [0.99, 0.01], [0.5626, 0.1874]];
    let footprint = |uv: [f32; 2], texels: f32, hardware: bool| Footprint {
        uv,
        dx: [texels / SIZE as f32, 0.0],
        dy: [0.0, texels / SIZE as f32],
        hardware: u32::from(hardware),
        material: 0,
        inward: [0.0; 2],
    };
    let mut cases = Vec::new();
    for &uv in &uvs {
        let mix: [f32; 4] = std::array::from_fn(|c| 0.5 * cubic(uv)[c] + 0.5 * bilinear(uv)[c]);
        cases.extend([
            (footprint(uv, 0.1, false), cubic(uv), "magnified"),
            (footprint(uv, 0.75, false), mix, "three quarters of a texel"),
            (footprint(uv, 2.0, false), bilinear(uv), "minified"),
            (footprint(uv, 0.1, true), bilinear(uv), "hardware"),
        ]);
    }
    assert!(
        uvs.iter()
            .any(|&uv| (0..3).any(|c| (cubic(uv)[c] - bilinear(uv)[c]).abs() > 0.05)),
        "the texture tells bicubic from bilinear"
    );

    let desc = TextureDesc::d2(Format::Rgba8Unorm, SIZE, SIZE, TextureUsage::SAMPLED);
    let texture = d
        .create_texture(&desc, TextureMemory::Dedicated, "filtered texture")
        .unwrap();
    let upload = |bytes: &[u8], name| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, name)
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let staging = upload(bytemuck::cast_slice(&texels), "texels");
    let footprints: Vec<Footprint> = cases.iter().map(|c| c.0).collect();
    let input = upload(bytemuck::cast_slice(&footprints), "footprints");
    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, cases.len() as u64, "means")
        .unwrap();
    let slot = d.alloc_image_slot().unwrap();
    let sampler = d
        .sampler(&SamplerDesc {
            mag: Filter::Linear,
            min: Filter::Linear,
            mip: Filter::Nearest,
            address: [AddressMode::Repeat; 3],
            ..SamplerDesc::default()
        })
        .unwrap();
    let root = upload(
        bytemuck::bytes_of(&Root {
            footprints: input.gpu().cast(),
            out: out.gpu(),
            texture: d.write_sampled(slot, &texture, &orr_gpu::ViewDesc::default()),
            sampler,
            count: cases.len() as u32,
            taps: TAPS,
            streams: GpuPtr::NULL,
            modes: 0,
            pad: 0,
            materials: GpuPtr::NULL,
            pad2: 0,
        }),
        "texture filter root",
    );
    let p = pipeline(&texture_filter::MODULE, "mean");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.discard(&texture);
    list.copy_to_texture(&texture, &TextureRegion::whole_mip(&desc, 0), staging.gpu());
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&p, root.gpu().cast::<Root>(), [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());

    for ((footprint, want, what), mean) in cases.iter().zip(out.as_slice().unwrap()) {
        for c in 0..4 {
            assert!(
                (mean[c] - want[c]).abs() < 4e-3,
                "{what} at {:?}: channel {c} is {} against {}",
                footprint.uv,
                mean[c],
                want[c]
            );
        }
    }
    d.free_image_slot_after_submitted(slot);
    assert_clean();
}

/// Every wrap and filter a binding can state, read through `material_texture` against a CPU
/// reference, inside the unit square and past it. Neighboring lanes read different materials,
/// so each wave's taps run on several samplers at once.
#[test]
fn material_slots_read_their_wrap_and_filter_like_a_cpu_reference() {
    use orr_gpu::{Format, TextureDesc, TextureMemory, TextureUsage};
    use orr_pack::material::{Material, TextureBinding, filter, slot, wrap};
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};

    const SIZE: i32 = 8;
    const TAPS: u32 = 4096;
    const OUTSIDE: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    let _guard = serial();
    let d = device();
    let hashed = |n: i32, salt: u32| -> Vec<[u8; 4]> {
        (0..(n * n) as u32)
            .map(|i| {
                let h = ((i + salt).wrapping_mul(2_654_435_761) >> 7) as u8;
                [h, h.wrapping_mul(7), h ^ 0x5a, h.wrapping_mul(13) | 1]
            })
            .collect()
    };
    let mips = [hashed(SIZE, 0), hashed(SIZE / 2, 99)];

    let fold = |i: i32, n: i32, mode: u8| -> Option<i32> {
        match mode {
            wrap::REPEAT => Some(i.rem_euclid(n)),
            wrap::MIRROR => {
                let m = i.rem_euclid(2 * n);
                Some(if m < n { m } else { 2 * n - 1 - m })
            }
            wrap::BORDER if !(0..n).contains(&i) => None,
            _ => Some(i.clamp(0, n - 1)),
        }
    };
    let texel = |modes: [u8; 2], mip: usize, x: i32, y: i32| -> [f32; 4] {
        let n = SIZE >> mip;
        match (fold(x, n, modes[0]), fold(y, n, modes[1])) {
            (Some(x), Some(y)) => mips[mip][(y * n + x) as usize].map(|c| f32::from(c) / 255.0),
            _ => [0.0; 4],
        }
    };
    let filtered =
        |modes: [u8; 2], mip: usize, uv: [f32; 2], r: i32, w: &dyn Fn(f32, i32) -> f32| {
            let n = (SIZE >> mip) as f32;
            let (x, y) = (uv[0] * n - 0.5, uv[1] * n - 0.5);
            let (bx, by) = (x.floor(), y.floor());
            let mut sum = [0.0; 4];
            for j in -r + 1..=r {
                for i in -r + 1..=r {
                    let weight = w(x - bx, i) * w(y - by, j);
                    let t = texel(modes, mip, bx as i32 + i, by as i32 + j);
                    for c in 0..4 {
                        sum[c] += weight * t[c];
                    }
                }
            }
            sum
        };
    let linear = |t: f32, k: i32| if k == 0 { 1.0 - t } else { t };
    let bspline = |t: f32, k: i32| {
        let s = 1.0 - t;
        match k {
            -1 => s * s * s / 6.0,
            0 => (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0,
            1 => (1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t * t * t) / 6.0,
            _ => t * t * t / 6.0,
        }
    };
    let nearest = |modes: [u8; 2], mip: usize, uv: [f32; 2]| {
        let n = (SIZE >> mip) as f32;
        let at = uv.map(|c| (c * n).floor() as i32);
        texel(modes, mip, at[0], at[1])
    };

    #[derive(Copy, Clone)]
    enum Read {
        Bilinear(usize),
        Bicubic,
        Nearest(usize),
    }
    use Read::*;
    // Filter bits, footprint in texels, hardware filtering alone, and what that reads.
    let filters: [(u8, f32, bool, Read); 8] = [
        (0, 0.1, true, Bilinear(0)),
        (0, 0.1, false, Bicubic),
        (0, 2.0, true, Bilinear(1)),
        (filter::MIP_NEAREST, 1.1, true, Bilinear(0)),
        (filter::MAG_NEAREST, 0.1, false, Nearest(0)),
        (
            filter::MIP_NEAREST | filter::ONE_MIP,
            2.0,
            true,
            Bilinear(0),
        ),
        (
            filter::MIN_NEAREST | filter::MIP_NEAREST | filter::ONE_MIP,
            2.0,
            true,
            Nearest(0),
        ),
        (
            filter::MIN_NEAREST | filter::MIP_NEAREST,
            2.0,
            true,
            Nearest(1),
        ),
    ];
    let wraps = [
        [wrap::REPEAT, wrap::REPEAT],
        [wrap::CLAMP, wrap::CLAMP],
        [wrap::MIRROR, wrap::MIRROR],
        [wrap::BORDER, wrap::BORDER],
        [wrap::CONSTANT, wrap::CONSTANT],
        [wrap::CLAMP, wrap::MIRROR],
        [wrap::BORDER, wrap::REPEAT],
        [wrap::REPEAT, wrap::CONSTANT],
    ];
    let uvs = [
        [0.31, 0.53],
        [-0.03, 0.53],
        [1.02, 0.97],
        [1.6, -0.4],
        [-1.3, 2.2],
        [0.01, 0.99],
    ];

    let desc = TextureDesc {
        mips: 2,
        ..TextureDesc::d2(
            Format::Rgba8Unorm,
            SIZE as u32,
            SIZE as u32,
            TextureUsage::SAMPLED,
        )
    };
    let texture = d
        .create_texture(&desc, TextureMemory::Dedicated, "wrapped texture")
        .unwrap();
    let slot_index = d.alloc_image_slot().unwrap();
    let index = d.write_sampled(slot_index, &texture, &orr_gpu::ViewDesc::default());
    let mut records = Vec::new();
    for (wi, &modes) in wraps.iter().enumerate() {
        for (fi, f) in filters.iter().enumerate() {
            let binding = TextureBinding {
                wrap: modes,
                filter: f.0,
                outside: OUTSIDE,
                ..bytemuck::Zeroable::zeroed()
            };
            let mut m = Material::default();
            m.textures[slot::BASE_COLOR] = binding;
            let mut bound = [BoundTexture::NONE; slot::COUNT];
            bound[slot::BASE_COLOR] = BoundTexture {
                index,
                sampler: d.sampler(&orr_scene::binding_sampler(&binding)).unwrap(),
                ..BoundTexture::NONE
            };
            records.push(MaterialGpu::new(&m, &bound));
            assert_eq!(records.len(), wi * filters.len() + fi + 1);
        }
    }
    let mut cases = Vec::new();
    for &uv in &uvs {
        for (fi, &(_, texels, hardware, read)) in filters.iter().enumerate() {
            for (wi, &modes) in wraps.iter().enumerate() {
                let past = |a: usize| modes[a] == wrap::CONSTANT && !(0.0..=1.0).contains(&uv[a]);
                let want = if past(0) || past(1) {
                    OUTSIDE
                } else {
                    match read {
                        Bilinear(mip) => filtered(modes, mip, uv, 1, &linear),
                        Bicubic => filtered(modes, 0, uv, 2, &bspline),
                        Nearest(mip) => nearest(modes, mip, uv),
                    }
                };
                let footprint = Footprint {
                    uv,
                    dx: [texels / SIZE as f32, 0.0],
                    dy: [0.0, texels / SIZE as f32],
                    hardware: u32::from(hardware),
                    material: (wi * filters.len() + fi) as u32,
                    inward: [0.0; 2],
                };
                cases.push((footprint, want, modes, fi));
            }
        }
    }

    let upload = |bytes: &[u8], name| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, name)
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let levels = mips
        .each_ref()
        .map(|m| upload(bytemuck::cast_slice(m), "texels"));
    let footprints: Vec<Footprint> = cases.iter().map(|c| c.0).collect();
    let input = upload(bytemuck::cast_slice(&footprints), "footprints");
    let materials = upload(bytemuck::cast_slice(&records), "materials");
    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, cases.len() as u64, "means")
        .unwrap();
    let root = upload(
        bytemuck::bytes_of(&Root {
            footprints: input.gpu().cast(),
            out: out.gpu(),
            texture: index,
            sampler: orr_gpu::SamplerIndex(0),
            count: cases.len() as u32,
            taps: TAPS,
            streams: GpuPtr::NULL,
            modes: 0,
            pad: 0,
            materials: materials.gpu().cast(),
            pad2: 0,
        }),
        "texture filter root",
    );
    let p = pipeline(&texture_filter::MODULE, "slot_mean");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.discard(&texture);
    for (mip, level) in levels.iter().enumerate() {
        list.copy_to_texture(
            &texture,
            &TextureRegion::whole_mip(&desc, mip as u32),
            level.gpu(),
        );
    }
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(
        &p,
        root.gpu().cast::<Root>(),
        [cases.len().div_ceil(64) as u32, 1, 1],
    );
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());

    for ((footprint, want, modes, fi), mean) in cases.iter().zip(out.as_slice().unwrap()) {
        for c in 0..4 {
            assert!(
                (mean[c] - want[c]).abs() < 6e-3,
                "wrap {modes:?}, filter row {fi} at {:?}: channel {c} is {} against {}",
                footprint.uv,
                mean[c],
                want[c]
            );
        }
    }
    d.free_image_slot_after_submitted(slot_index);
    assert_clean();
}

/// A tile set read through `material_texture` against a CPU reference: every unit square reads its
/// own tile, of its own size and mips, at the point's place in the square, and filtering stops at
/// the tile's edges as the binding says. Squares without a tile read `outside`, under any wrap,
/// those whose keys land in a bucket that holds others among them, and a tile far from the rest
/// costs nothing. Neighboring lanes read different tiles and materials.
#[test]
fn material_tile_sets_read_each_square_like_a_cpu_reference() {
    use orr_gpu::{Format, TextureDesc, TextureMemory, TextureUsage};
    use orr_pack::material::{Material, TextureBinding, filter, slot, wrap};
    use orr_pack::tiles::{BUCKET, TextureTiles};
    use scene_suite::shaders::texture_filter::{self, Footprint, Root};

    const TAPS: u32 = 4096;
    const OUTSIDE: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    // The size of each texture, which the tiles share out.
    const SIZES: [i32; 3] = [8, 4, 16];
    let _guard = serial();
    let d = device();
    let hashed = |n: i32, salt: u32| -> Vec<[u8; 4]> {
        (0..(n * n) as u32)
            .map(|i| {
                let h = ((i + salt).wrapping_mul(2_654_435_761) >> 7) as u8;
                [h, h.wrapping_mul(7), h ^ 0x5a, h.wrapping_mul(13) | 1]
            })
            .collect()
    };
    let mips: Vec<[Vec<[u8; 4]>; 2]> = SIZES
        .iter()
        .enumerate()
        .map(|(t, &n)| {
            [
                hashed(n, 1000 * t as u32),
                hashed(n / 2, 1000 * t as u32 + 99),
            ]
        })
        .collect();
    // Column, row and texture of each tile: enough of them for buckets of several keys, and one
    // whose name lies far off.
    let mut tiles: Vec<(u16, u16, usize)> =
        vec![(0, 0, 0), (1, 0, 1), (2, 1, 2), (40_000, 39_999, 0)];
    tiles.extend((0..40).map(|i| (3 + i % 8, 2 + i / 8, usize::from(i) % 3)));

    let fold = |i: i32, n: i32, mode: u8| match mode {
        wrap::REPEAT => i.rem_euclid(n),
        _ => i.clamp(0, n - 1),
    };
    let texel = |tile: usize, mode: u8, mip: usize, x: i32, y: i32| -> [f32; 4] {
        let n = SIZES[tile] >> mip;
        let at = fold(y, n, mode) * n + fold(x, n, mode);
        mips[tile][mip][at as usize].map(|c| f32::from(c) / 255.0)
    };
    let filtered =
        |tile: usize, mode: u8, mip: usize, uv: [f32; 2], r: i32, w: &dyn Fn(f32, i32) -> f32| {
            let n = (SIZES[tile] >> mip) as f32;
            let (x, y) = (uv[0] * n - 0.5, uv[1] * n - 0.5);
            let (bx, by) = (x.floor(), y.floor());
            let mut sum = [0.0; 4];
            for j in -r + 1..=r {
                for i in -r + 1..=r {
                    let weight = w(x - bx, i) * w(y - by, j);
                    let t = texel(tile, mode, mip, bx as i32 + i, by as i32 + j);
                    for c in 0..4 {
                        sum[c] += weight * t[c];
                    }
                }
            }
            sum
        };
    let linear = |t: f32, k: i32| if k == 0 { 1.0 - t } else { t };
    let bspline = |t: f32, k: i32| {
        let s = 1.0 - t;
        match k {
            -1 => s * s * s / 6.0,
            0 => (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0,
            1 => (1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t * t * t) / 6.0,
            _ => t * t * t / 6.0,
        }
    };

    #[derive(Copy, Clone)]
    enum Read {
        Bilinear(usize),
        Bicubic,
        Nearest,
    }
    use Read::*;
    // Filter bits, footprint in texels of the tile, hardware filtering alone, and what that reads.
    let filters: [(u8, f32, bool, Read); 4] = [
        (0, 0.1, true, Bilinear(0)),
        (0, 0.1, false, Bicubic),
        (0, 2.0, true, Bilinear(1)),
        (filter::MAG_NEAREST, 0.1, false, Nearest),
    ];
    let wraps = [wrap::CLAMP, wrap::REPEAT, wrap::CONSTANT];

    let mut textures = Vec::new();
    let mut indices = Vec::new();
    for &n in &SIZES {
        let desc = TextureDesc {
            mips: 2,
            ..TextureDesc::d2(
                Format::Rgba8Unorm,
                n as u32,
                n as u32,
                TextureUsage::SAMPLED,
            )
        };
        let texture = d
            .create_texture(&desc, TextureMemory::Dedicated, "tile")
            .unwrap();
        let slot_index = d.alloc_image_slot().unwrap();
        indices.push((
            slot_index,
            d.write_sampled(slot_index, &texture, &Default::default()),
        ));
        textures.push((texture, desc));
    }
    let upload = |bytes: &[u8], name| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, name)
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let texture_id = |t: usize| AssetId::from_name(&t.to_string());
    let cells: Vec<_> = tiles
        .iter()
        .map(|&(u, v, t)| (u, v, texture_id(t)))
        .collect();
    let set = TextureTiles::new(&cells);
    let mut held = vec![0; set.buckets as usize];
    for t in &set.tiles {
        held[(t.entry / BUCKET) as usize] += 1;
    }
    assert!(
        held.iter().any(|&n| n >= 3),
        "a bucket of several keys: {held:?}"
    );
    let words = set
        .table(|id| {
            (0..SIZES.len())
                .find(|&t| texture_id(t) == id)
                .map(|t| indices[t].1.0)
        })
        .unwrap();
    let table = upload(bytemuck::cast_slice(&words), "tile table");

    let mut records = Vec::new();
    for &mode in &wraps {
        for f in &filters {
            let binding = TextureBinding {
                wrap: [mode; 2],
                filter: f.0,
                outside: OUTSIDE,
                ..bytemuck::Zeroable::zeroed()
            };
            let mut m = Material::default();
            m.textures[slot::BASE_COLOR] = binding;
            let mut bound = [BoundTexture::NONE; slot::COUNT];
            bound[slot::BASE_COLOR] = BoundTexture {
                index: indices[0].1,
                sampler: d.sampler(&orr_scene::binding_sampler(&binding)).unwrap(),
                tiles: table.gpu().cast(),
                ..BoundTexture::NONE
            };
            records.push(MaterialGpu::new(&m, &bound));
        }
    }
    // Places in a square: its middle, and beside each edge that filtering reaches past.
    let places = [[0.31, 0.53], [0.01, 0.99], [0.99, 0.02]];
    // Uvs with no tile: past every tile on each side, past what a key holds, and squares among
    // the tiles, of which those last share a bucket with some.
    let mut holes = vec![
        [-0.5, 0.5],
        [0.5, 1.5],
        [65_536.5, 0.5],
        [0.5, -65_535.5],
        [2.5, 0.5],
        [0.5, -0.5],
        [1.5, -0.5],
        [40_001.5, -39_998.5],
    ];
    let taken = |u: u16, v: u16| tiles.iter().any(|t| (t.0, t.1) == (u, v));
    let crowded = (0..200u16)
        .flat_map(|v| (0..200u16).map(move |u| (u, v)))
        .filter(|&(u, v)| !taken(u, v) && held[set.bucket(u, v) as usize] >= 3)
        .take(16);
    holes.extend(crowded.map(|(u, v)| [f32::from(u) + 0.5, 0.5 - f32::from(v)]));
    let mut cases = Vec::new();
    for (fi, &(_, texels, hardware, read)) in filters.iter().enumerate() {
        for (wi, &mode) in wraps.iter().enumerate() {
            let material = (wi * filters.len() + fi) as u32;
            let want = |t: usize, place: [f32; 2]| match read {
                Bilinear(mip) => filtered(t, mode, mip, place, 1, &linear),
                Bicubic => filtered(t, mode, 0, place, 2, &bspline),
                Nearest => {
                    let at = place.map(|c| (c * SIZES[t] as f32).floor() as i32);
                    texel(t, mode, 0, at[0], at[1])
                }
            };
            for &(column, row, t) in &tiles {
                let n = SIZES[t];
                for &place in &places {
                    // A far tile's uv holds fewer bits of its place.
                    let uv = [f32::from(column) + place[0], place[1] - f32::from(row)];
                    let place = uv.map(|c| c - c.floor());
                    let size = texels / n as f32;
                    cases.push((uv, [0.0; 2], size, hardware, material, want(t, place)));
                }
            }
            // Past the last tile of a row and below the first row, a point on the edge still lies
            // on the tile before it, at its far edge.
            for (uv, t, place) in [
                ([2.0, 0.5], 1, [1.0, 0.5]),
                ([0.5, 1.0], 0, [0.5, 1.0]),
                ([2.0, 1.0], 1, [1.0, 1.0]),
            ] {
                let size = texels / SIZES[t] as f32;
                cases.push((uv, [0.0; 2], size, hardware, material, want(t, place)));
            }
            // On the edge of two tiles, or rounded past it, a point on a surface reads the tile the
            // surface lies on.
            let past = 1.0 + f32::EPSILON;
            for (uv, inward, t, place) in [
                ([1.0, 0.5], [-0.25, 0.0], 0, [1.0, 0.5]),
                ([1.0, 0.5], [0.25, 0.0], 1, [0.0, 0.5]),
                ([past, 0.5], [-0.25, 0.0], 0, [past, 0.5]),
            ] {
                let size = texels / SIZES[t] as f32;
                cases.push((uv, inward, size, hardware, material, want(t, place)));
            }
            for &uv in &holes {
                cases.push((uv, [0.0; 2], texels / 8.0, hardware, material, OUTSIDE));
            }
        }
    }
    let footprints: Vec<Footprint> = cases
        .iter()
        .map(|&(uv, inward, size, hardware, material, _)| Footprint {
            uv,
            dx: [size, 0.0],
            dy: [0.0, size],
            hardware: u32::from(hardware),
            material,
            inward,
        })
        .collect();
    let levels: Vec<_> = mips
        .iter()
        .map(|m| {
            m.each_ref()
                .map(|m| upload(bytemuck::cast_slice(m), "texels"))
        })
        .collect();
    let input = upload(bytemuck::cast_slice(&footprints), "footprints");
    let materials = upload(bytemuck::cast_slice(&records), "materials");
    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, cases.len() as u64, "means")
        .unwrap();
    let root = upload(
        bytemuck::bytes_of(&Root {
            footprints: input.gpu().cast(),
            out: out.gpu(),
            texture: indices[0].1,
            sampler: orr_gpu::SamplerIndex(0),
            count: cases.len() as u32,
            taps: TAPS,
            streams: GpuPtr::NULL,
            modes: 0,
            pad: 0,
            materials: materials.gpu().cast(),
            pad2: 0,
        }),
        "texture filter root",
    );
    let p = pipeline(&texture_filter::MODULE, "slot_mean");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    for ((texture, desc), levels) in textures.iter().zip(&levels) {
        list.discard(texture);
        for (mip, level) in levels.iter().enumerate() {
            list.copy_to_texture(
                texture,
                &TextureRegion::whole_mip(desc, mip as u32),
                level.gpu(),
            );
        }
    }
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(
        &p,
        root.gpu().cast::<Root>(),
        [cases.len().div_ceil(64) as u32, 1, 1],
    );
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());

    for (case, mean) in cases.iter().zip(out.as_slice().unwrap()) {
        let (uv, _, _, _, material, want) = case;
        for c in 0..4 {
            assert!(
                (mean[c] - want[c]).abs() < 6e-3,
                "material {material} at {uv:?}: channel {c} is {} against {}",
                mean[c],
                want[c]
            );
        }
    }
    for (slot_index, _) in indices {
        d.free_image_slot_after_submitted(slot_index);
    }
    assert_clean();
}

// --- Runtime edits ----------------------------------------------------------------------------

/// Renders a frame in which `edits` (edit, stripe of `pieces`, lift) change at the scene
/// compute point.
fn shoot_edits(
    w: &mut World,
    placed: &Placed,
    shot: &Shot,
    edits: &[(EditId, u32, u32, f32)],
) -> Frame {
    shoot_edits_with(w, placed, shot, edits, true, |_| {})
}

/// The fracture kernel over each of `edits` (edit, stripe of `pieces`, lift).
fn dispatch_fractures(
    c: &mut SceneCompute<'_, '_>,
    edits: &[(EditId, u32, u32, f32)],
    refs: &[EditRef],
) {
    let p = pipeline(&fracture::MODULE, "main");
    for (e, r) in edits.iter().zip(refs) {
        let root = c
            .put_root(&fracture::Root {
                scene: c.root,
                groups: r.groups,
                mesh: r.mesh,
                piece: e.1,
                pieces: e.2,
                min_x: -4.0,
                size_x: 8.0,
                lift: e.3,
                _pad: 0,
            })
            .unwrap();
        c.cmd.dispatch(&p, root, [r.group_count, 1, 1]);
        c.cmd
            .barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    }
}

/// [`shoot_edits`] with `write` run on the scene's writer; without `motion`, the edits have no
/// motion this frame.
fn shoot_edits_with(
    w: &mut World,
    placed: &Placed,
    shot: &Shot,
    edits: &[(EditId, u32, u32, f32)],
    motion: bool,
    write: impl FnOnce(&mut SceneWriter<'_>),
) -> Frame {
    let list = [placed.scene];
    let refs: Vec<EditRef> = edits.iter().map(|e| w.scenes.edit(e.0).unwrap()).collect();
    let mut f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(shot.frame),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    {
        let mut s = f.scene(placed.scene).unwrap();
        s.set_focus(shot.focus.unwrap_or(shot.eye));
        f.request_detail(placed.scene, PIXELS_PER_RADIAN).unwrap();
        for &(i, t) in &shot.moved {
            s.transforms(placed.group, i, 1).unwrap()[0] = t.into();
        }
        write(&mut s);
    }
    for e in edits {
        let admitted = if motion {
            f.edit_dirty(e.0)
        } else {
            f.edit_dirty_without_motion(e.0)
        };
        assert!(admitted, "the frame's rebuild budget takes the edit");
    }
    f.update_scenes(&mut |c| dispatch_fractures(c, edits, &refs))
        .unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: placed.scene,
            history: placed.history,
            position: shot.eye,
            rotation: look(Vec3::ZERO, (shot.target - shot.eye).as_vec3()),
            projection: Projection {
                fov_y: 0.9,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: shot.jitter,
            camera_cut: shot.reset,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
    wait(f.end(&[]).unwrap());
    read_back(&view)
}

/// Runs frames until every edit of `edits` is ready.
fn settle_edits(w: &mut World, edits: &[EditId]) {
    let start = Instant::now();
    loop {
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(0),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        wait(f.end(&[]).unwrap());
        if edits
            .iter()
            .all(|&e| matches!(w.scenes.edit_state(e), LoadState::Ready))
        {
            return;
        }
        for &e in edits {
            if let LoadState::Failed(x) = w.scenes.edit_state(e) {
                panic!("edit failed: {x}");
            }
        }
        assert!(start.elapsed() < Duration::from_secs(60), "edit pending");
    }
}

/// A scene of the source model then `models`, in one group whose transforms start hidden.
fn place_models(w: &mut World, models: &[ModelId]) -> Placed {
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: models.len() as u32,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let ranges = models
        .iter()
        .enumerate()
        .map(|(i, &model)| {
            s.add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: group,
                first: i as u32,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 1000.0,
                animation_range: 0.0,
                user_tag: i as u32,
            })
            .unwrap()
        })
        .collect();
    drop(s);
    wait(f.end(&[]).unwrap());
    Placed {
        scene,
        ranges,
        group,
        history,
        count: models.len() as u32,
    }
}

#[test]
fn a_fractured_dag_model_draws_and_traces_alike_every_frame() {
    fracture_model("dag ground");
}

#[test]
fn a_fractured_discrete_model_draws_and_traces_alike_every_frame() {
    fracture_model("discrete ground");
}

/// Splits `model`, 8 m wide in x, into four edits that move apart and are lifted, checking
/// raster and rays against each other every frame.
fn fracture_model(model: &str) {
    const PIECES: u32 = 4;
    let _serial = serial();
    let mut w = finest_world();
    let source = w.model(model);
    let edits: Vec<EditId> = (0..PIECES)
        .map(|_| {
            w.scenes
                .create_edit(&EditDesc {
                    model: source,
                    touched: None,
                    bounds: None,
                })
                .unwrap()
        })
        .collect();
    settle_edits(&mut w, &edits);
    let mut models = vec![source];
    models.extend(edits.iter().map(|&e| w.scenes.edit(e).unwrap().model));
    let placed = place_models(&mut w, &models);
    let linear = Mat3::from_rotation_x(0.9);
    let at = Vec3::new(0.0, 0.0, -6.0);
    let shown = Affine3::from_mat3_translation(linear, at);
    let hidden = Affine3::ZERO;
    let eye = DVec3::new(0.0, 0.5, 2.0);
    let target = DVec3::new(0.0, 0.0, -6.0);
    let shot = |frame: u32, moved: Vec<(u32, Affine3)>| {
        let mut s = Shot::at(eye, target, frame);
        s.moved = moved;
        s
    };
    let slot = |i: usize| placed.ranges[i].first + 1;
    let check_parity = |g: &Frame, what: &str| {
        let r = rays(&g.view, &trace::MODULE, "pixels", 0);
        let (matched, covered) = parity(g, &r, 1e-3);
        let total = r.traced.len();
        assert!(covered > total / 8, "{what}: {covered} of {total} covered");
        assert!(
            matched as f64 >= 0.995 * total as f64,
            "{what}: {matched} of {total} pixels agree"
        );
    };

    // The model itself, then one unwritten edit: the same pixels at the same depth.
    let mut moved: Vec<(u32, Affine3)> = (0..=PIECES).map(|i| (i, hidden)).collect();
    moved[0].1 = shown;
    let base = shoot(&mut w, &placed, &shot(1, moved.clone()));
    check_parity(&base, "the model");
    moved[0].1 = hidden;
    moved[1].1 = shown;
    let whole = shoot(&mut w, &placed, &shot(2, moved.clone()));
    check_parity(&whole, "an unwritten edit");
    let same = (0..base.depth.len())
        .filter(|&i| (base.depth[i] - whole.depth[i]).abs() <= 1e-4 * base.depth[i].max(1e-6))
        .count();
    assert!(
        same as f64 >= 0.995 * base.depth.len() as f64,
        "{same} of {} depths equal the model's",
        base.depth.len()
    );

    // The split, in place: the pieces together cover what the model covered, each its stripe.
    let split: Vec<(EditId, u32, u32, f32)> = edits
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i as u32, PIECES, 0.0))
        .collect();
    let mut moved: Vec<(u32, Affine3)> = (1..=PIECES).map(|i| (i, shown)).collect();
    moved.push((0, hidden));
    let g = shoot_edits(&mut w, &placed, &shot(3, moved), &split);
    check_parity(&g, "the split");
    let same = (0..base.depth.len())
        .filter(|&i| (base.depth[i] - g.depth[i]).abs() <= 1e-4 * base.depth[i].max(1e-6))
        .count();
    assert!(
        same as f64 >= 0.99 * base.depth.len() as f64,
        "{same} of {} depths equal the model's after the split",
        base.depth.len()
    );
    for i in 1..=PIECES as usize {
        let n = g.ids.iter().filter(|&&id| id == slot(i)).count();
        assert!(n > 100, "piece {i} covers {n} pixels");
    }

    // The pieces move apart over frames: raster and rays agree on every one.
    let mut last = 0;
    for step in 1..=6u32 {
        let spread = 0.15 * step as f32;
        let moved: Vec<(u32, Affine3)> = (1..=PIECES)
            .map(|i| {
                let side = i as f32 - 0.5 - 0.5 * PIECES as f32;
                let t = at + Vec3::new(side * spread, 0.0, 0.0);
                (i, Affine3::from_mat3_translation(linear, t))
            })
            .collect();
        let g = shoot(&mut w, &placed, &shot(3 + step, moved));
        check_parity(&g, &format!("step {step}"));
        let gaps = (0..g.ids.len())
            .filter(|&i| base.ids[i] != 0 && g.ids[i] == 0)
            .count();
        assert!(gaps >= last, "step {step}: {gaps} gap pixels after {last}");
        last = gaps;
        for i in 1..=PIECES as usize {
            let n = g.ids.iter().filter(|&&id| id == slot(i)).count();
            assert!(n > 100, "step {step}: piece {i} covers {n} pixels");
        }
    }
    assert!(last > 200, "the pieces opened {last} gap pixels");

    // A piece lifted by its kernel: culling data and CLAS follow the new positions.
    let moved: Vec<(u32, Affine3)> = (1..=PIECES).map(|i| (i, shown)).collect();
    let flat = shoot(&mut w, &placed, &shot(10, moved.clone()));
    let g = shoot_edits(
        &mut w,
        &placed,
        &shot(11, moved),
        &[(edits[1], 1, PIECES, 0.8)],
    );
    check_parity(&g, "the lift");
    let changed = (0..g.ids.len())
        .filter(|&i| flat.ids[i] != g.ids[i])
        .count();
    assert!(changed > 100, "the lift changed {changed} pixels");
    let edit_stats = w.scenes.stats().cuts;
    assert_eq!(
        edit_stats.rt_fallback, 0,
        "every piece fits the BLAS budget"
    );

    for e in edits {
        w.scenes.destroy_edit(e);
    }
    assert_clean();
}

/// The largest difference in pixels between two frames' motion, over the pixels where both show
/// a surface at one depth, and how many those are.
fn motion_difference(a: &Frame, b: &Frame) -> (f32, usize) {
    let (mut n, mut worst) = (0, 0.0f32);
    let size = Vec2::new(EXTENT[0] as f32, EXTENT[1] as f32);
    for i in 0..a.ids.len() {
        let same = (a.depth[i] - b.depth[i]).abs() <= 1e-4 * a.depth[i];
        if a.ids[i] != 0 && b.ids[i] != 0 && same {
            let d = (Vec2::from(a.motion[i]) - Vec2::from(b.motion[i])) * size;
            worst = worst.max(d.abs().max_element());
            n += 1;
        }
    }
    (worst, n)
}

/// The largest motion in pixels over the frame's covered pixels.
fn largest_motion(g: &Frame) -> f32 {
    let size = Vec2::new(EXTENT[0] as f32, EXTENT[1] as f32);
    (0..g.ids.len())
        .filter(|&i| g.ids[i] != 0)
        .map(|i| (Vec2::from(g.motion[i]) * size).abs().max_element())
        .fold(0.0, f32::max)
}

/// Every ray hit of `g` lies `by` from where it was last frame.
fn assert_hits_moved(g: &Frame, by: Vec3, name: &str) {
    let now = rays(&g.view, &trace::MODULE, "pixels", 1);
    let before = rays(&g.view, &trace::MODULE, "pixels", 2);
    let (mut hits, mut off) = (0, 0.0f32);
    for (a, b) in now.traced.iter().zip(&before.traced) {
        if a.valid != 0 && b.valid != 0 {
            let moved = Vec3::from(a.position) - Vec3::from(b.position);
            off = off.max((moved - by).abs().max_element());
            hits += 1;
        }
    }
    assert!(hits > 1000, "{name}: {hits} hits");
    assert!(
        off < 1e-3,
        "{name}: a ray hit's last position is off by {off} m"
    );
}

#[test]
fn vertices_a_kernel_moves_have_motion_for_the_frame_they_move() {
    let _serial = serial();
    let mut w = finest_world();
    for name in ["dag ground", "discrete ground"] {
        let source = w.model(name);
        let edit = w
            .scenes
            .create_edit(&EditDesc {
                model: source,
                touched: None,
                bounds: None,
            })
            .unwrap();
        settle_edits(&mut w, &[edit]);
        let model = w.scenes.edit(edit).unwrap().model;
        let placed = place_models(&mut w, &[source, model]);
        let linear = Mat3::from_rotation_x(0.9);
        let at = Vec3::new(0.0, 0.0, -6.0);
        let hidden = Affine3::ZERO;
        let lift = 0.25;
        let up = linear * Vec3::Y * lift;
        let eye = DVec3::new(0.0, 0.5, 2.0);
        let target = DVec3::new(0.0, 0.0, -6.0);
        let shot = |frame: u32, moved: Vec<(u32, Affine3)>| {
            let mut s = Shot::at(eye, target, frame);
            s.moved = moved;
            s
        };

        // The reference: the model itself moved by the same lift through its transform.
        shoot(
            &mut w,
            &placed,
            &shot(
                1,
                vec![(0, Affine3::from_mat3_translation(linear, at)), (1, hidden)],
            ),
        );
        let rigid = shoot(
            &mut w,
            &placed,
            &shot(
                2,
                vec![(0, Affine3::from_mat3_translation(linear, at + up))],
            ),
        );
        assert!(largest_motion(&rigid) > 1.0, "{name}: the lift shows");

        shoot(
            &mut w,
            &placed,
            &shot(
                3,
                vec![(0, hidden), (1, Affine3::from_mat3_translation(linear, at))],
            ),
        );
        let quiet = shoot(&mut w, &placed, &shot(4, Vec::new()));
        assert!(
            largest_motion(&quiet) < 1e-3,
            "{name}: the unwritten edit is still"
        );
        let lifted = shoot_edits(&mut w, &placed, &shot(5, Vec::new()), &[(edit, 0, 1, lift)]);
        let (worst, n) = motion_difference(&rigid, &lifted);
        assert!(n > 1000, "{name}: {n} pixels to compare");
        assert!(
            worst <= 0.05,
            "{name}: motion off the rigid lift's by {worst} px"
        );

        let r = rays(&lifted.view, &trace::MODULE, "pixels", 0);
        let (matched, _) = parity(&lifted, &r, 1e-3);
        assert!(
            matched as f64 >= 0.995 * r.traced.len() as f64,
            "{name}: lifted parity"
        );
        assert_hits_moved(&lifted, up, name);

        // The frame after, nothing moves.
        let still = shoot(&mut w, &placed, &shot(6, Vec::new()));
        assert!(
            largest_motion(&still) < 1e-3,
            "{name}: motion after the lift"
        );
        assert!(
            still.depth == lifted.depth,
            "{name}: the lift holds once kernels stop"
        );

        // A jump marked as having no motion leaves none, in its frame and after.
        let jumped = shoot_edits_with(
            &mut w,
            &placed,
            &shot(7, Vec::new()),
            &[(edit, 0, 1, lift)],
            false,
            |_| {},
        );
        assert!(
            largest_motion(&jumped) < 1e-3,
            "{name}: a jump without motion has motion"
        );
        let changed = (0..jumped.depth.len())
            .filter(|&i| jumped.depth[i] != still.depth[i])
            .count();
        assert!(changed > 1000, "{name}: the jump changed {changed} depths");
        let after = shoot(&mut w, &placed, &shot(8, Vec::new()));
        assert!(
            largest_motion(&after) < 1e-3,
            "{name}: motion after the jump"
        );

        // A drop past the clusters' anchors, which the kernel lowers, undoes both lifts exactly.
        let dropped = shoot_edits(
            &mut w,
            &placed,
            &shot(9, Vec::new()),
            &[(edit, 0, 1, -2.0 * lift)],
        );
        assert_hits_moved(&dropped, -2.0 * up, name);
        assert!(dropped.depth == quiet.depth, "{name}: the drop clamped");

        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(10),
                scenes: &[placed.scene],
                low_latency: None,
            })
            .unwrap();
        {
            let mut s = f.scene(placed.scene).unwrap();
            for &r in &placed.ranges {
                s.remove(r);
            }
        }
        wait(f.end(&[]).unwrap());
        w.scenes.destroy_edit(edit);
        w.scenes.destroy_scene(placed.scene);
    }
    assert_clean();
}

#[test]
fn a_posed_edit_draws_and_traces_like_its_posed_source() {
    let _serial = serial();
    let mut w = finest_world();
    let around_one_corner = [[-0.5f32, 0.0, -0.5, 0.3]];
    for (name, touched) in [
        ("rig", None),
        ("dag rig", None),
        ("dag rig", Some(&around_one_corner[..])),
    ] {
        posed_edit(&mut w, name, touched);
    }
    assert_clean();
}

/// Two posed copies of an edit of `name` against two of the model: alike unwritten, in their
/// poses' motion, under a kernel's lift and under section masks.
fn posed_edit(w: &mut World, name: &str, touched: Option<&[[f32; 4]]>) {
    const COPIES: u32 = 2;
    let source = w.model(name);
    let edit = w
        .scenes
        .create_edit(&EditDesc {
            model: source,
            touched,
            bounds: Some(orr_pack::Aabb {
                min: [-2.0, -1.0, -2.0],
                max: [2.0, 2.0, 2.0],
            }),
        })
        .unwrap();
    settle_edits(w, &[edit]);
    let model = w.scenes.edit(edit).unwrap().model;
    let rigs = place_rigs_of(w, source, COPIES, COPIES, 2.2, 0, 100.0);
    let edited = place_rigs_of(w, model, COPIES, COPIES, 2.2, 0, 100.0);
    let eye = DVec3::new(0.0, 2.5, 4.0);
    let target = DVec3::ZERO;
    let check = |w: &mut World, g: &Frame, frame: u32, parts: u32, what: &str| {
        let r = rays(&g.view, &trace::MODULE, "pixels", 0);
        let (matched, covered) = parity(g, &r, 1e-3);
        let total = r.traced.len();
        assert!(
            covered > total / 20,
            "{name}, {what}: {covered} of {total} covered"
        );
        assert!(
            matched as f64 >= 0.995 * total as f64,
            "{name}, {what}: {matched} of {total} pixels agree"
        );
        let rt = last_rt(w, frame);
        assert_eq!(
            (rt.parts, rt.parts_refused),
            (parts, 0),
            "{name}, {what}: part instances"
        );
    };
    let all_parts = COPIES * (RIG_PARTS - 1);

    // Unwritten, it stands where its source does, every part posed.
    let mut frame = 1;
    for t in [0.4, 1.1, 2.3] {
        let shot = |frame| Shot::at(eye, target, frame);
        let a = shoot_rigs(w, &rigs, &shot(frame), "sway", t, 0.35);
        let b = shoot_rigs(w, &edited, &shot(frame + 1), "sway", t, 0.35);
        check(w, &b, frame + 1, all_parts, "unwritten");
        let same = (0..a.depth.len())
            .filter(|&i| (a.depth[i] - b.depth[i]).abs() <= 1e-3 * a.depth[i].max(1e-6))
            .count();
        assert!(
            same as f64 >= 0.995 * a.depth.len() as f64,
            "{name}: {same} of {} depths equal the source's at {t} s",
            a.depth.len()
        );
        frame += 2;
    }

    // A part's own motion comes through its pose, as on the source.
    let slide = |w: &mut World, rigs: &Rigs, frame: u32, t: f32| {
        shoot_rigs(w, rigs, &Shot::at(eye, target, frame), "slide", t, 0.05)
    };
    slide(w, &rigs, frame, 0.0);
    let a = slide(w, &rigs, frame + 1, 0.5);
    slide(w, &edited, frame + 2, 0.0);
    let b = slide(w, &edited, frame + 3, 0.5);
    let (worst, n) = motion_difference(&a, &b);
    assert!(n > 1000, "{name}: {n} pixels to compare");
    assert!(
        worst <= 0.05,
        "{name}: the pose's motion differs by {worst} px"
    );
    frame += 4;

    // The kernel's lift moves the posed parts too, and rays follow.
    let lifted = shoot_rigs_edited(
        w,
        &edited,
        &Shot::at(eye, target, frame),
        "slide",
        0.5,
        0.05,
        &[(edit, 0, 1, 0.25)],
    );
    check(w, &lifted, frame, all_parts, "the lift");
    assert!(largest_motion(&lifted) > 1.0, "{name}: the lift has motion");
    let moved = (0..b.depth.len())
        .filter(|&i| lifted.depth[i] != b.depth[i])
        .count();
    assert!(moved > 100, "{name}: the lift changed {moved} depths");
    frame += 1;

    // A mask leaving only the parts traces no body, one leaving only the body no parts; the
    // other copy keeps every section and so a cut of its own.
    for (mask, parts) in [(1u16, all_parts), (2, RIG_PARTS - 1), (0, all_parts)] {
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(frame),
                scenes: &[edited.placed.scene],
                low_latency: None,
            })
            .unwrap();
        {
            let mut s = f.scene(edited.placed.scene).unwrap();
            let variants = [mask, 0].map(|section_mask| CopyVariant {
                pose: 0,
                material_set: 0,
                section_mask,
            });
            s.copy_variants(edited.range, 0, &variants).unwrap();
        }
        wait(f.end(&[]).unwrap());
        frame += 1;
        for _ in 0..2 {
            let g = shoot_rigs(
                w,
                &edited,
                &Shot::at(eye, target, frame),
                "sway",
                frame as f32 * 0.21,
                0.35,
            );
            check(w, &g, frame, parts, &format!("mask {mask}"));
            frame += 1;
        }
    }
}

#[test]
fn a_skinned_edit_follows_its_palette_and_its_kernel() {
    let _serial = serial();
    let mut w = finest_world();
    for name in ["worm", "unbounded worm"] {
        let source = w.model(name);
        let edit = w
            .scenes
            .create_edit(&EditDesc {
                model: source,
                touched: None,
                bounds: None,
            })
            .unwrap();
        settle_edits(&mut w, &[edit]);
        let model = w.scenes.edit(edit).unwrap().model;
        let worms = place_worms(&mut w, name, &[(Mat3::IDENTITY, Vec3::ZERO)], false);
        let edited = place_worms_of(&mut w, model, &[(Mat3::IDENTITY, Vec3::ZERO)], false);
        let check = |g: &Frame, what: &str| {
            let r = rays(&g.view, &trace::MODULE, "pixels", 0);
            let (matched, covered) = parity(g, &r, 1e-3);
            let total = r.traced.len();
            assert!(
                covered > total / 100,
                "{name}, {what}: {covered} of {total} covered"
            );
            assert!(
                matched as f64 >= 0.995 * total as f64,
                "{name}, {what}: {matched} of {total} pixels agree"
            );
        };

        // Unwritten, it deforms like its source's finest cut.
        let t = 0.7;
        let eye = worm_shot(0);
        let (e, target) = (eye.eye, eye.target);
        let shot = |frame| {
            let mut s = Shot::at(e, target, frame);
            s.focus = None;
            s
        };
        let a = shoot_worms(&mut w, &worms, &shot(1), t, 0.0);
        let b = shoot_worms(&mut w, &edited, &shot(2), t, 0.0);
        check(&b, "unwritten");
        let same = (0..a.depth.len())
            .filter(|&i| (a.depth[i] - b.depth[i]).abs() <= 1e-4 * a.depth[i].max(1e-6))
            .count();
        assert!(
            same as f64 >= 0.995 * a.depth.len() as f64,
            "{name}: {same} of {} depths equal the source's",
            a.depth.len()
        );

        // Half of it goes: its templates rebuild over the triangles left.
        let palette = |s: &mut SceneWriter<'_>| write_palette(s, &edited, t, 0.0);
        let split = shoot_edits_with(
            &mut w,
            &edited.placed,
            &shot(3),
            &[(edit, 0, 2, 0.0)],
            true,
            palette,
        );
        check(&split, "split");
        let gone = (0..b.ids.len())
            .filter(|&i| b.ids[i] != 0 && split.ids[i] == 0)
            .count();
        assert!(gone > 100, "{name}: the split removed {gone} pixels");

        // A lift of the rest pose: this frame's positions move, last frame's stay what they were.
        let before = skinned_positions(&split);
        let lifted = shoot_edits_with(
            &mut w,
            &edited.placed,
            &shot(4),
            &[(edit, 0, 2, 0.25)],
            true,
            palette,
        );
        check(&lifted, "lifted");
        let (now, previous) = (skinned_positions(&lifted), skinned_previous(&lifted));
        assert_eq!(
            previous, before,
            "{name}: previous positions off last frame's"
        );
        assert_ne!(now, before, "{name}: the lift moved nothing");
        assert!(largest_motion(&lifted) > 1.0, "{name}: the lift has motion");

        // The frame after, last frame's positions are the lifted ones.
        let still = shoot_worms(&mut w, &edited, &shot(5), t, 0.0);
        check(&still, "still");
        assert_eq!(
            skinned_previous(&still),
            now,
            "{name}: motion after the lift"
        );
        assert!(
            largest_motion(&still) < 1e-3,
            "{name}: motion after the lift"
        );
    }
    assert_clean();
}

/// Skinned copy 0's positions of the clusters it skinned this frame.
fn skinned_positions(g: &Frame) -> Vec<[f32; 3]> {
    let (k, _, _) = skinned_entry(&g.view, 0);
    skinned_array(g, k.positions)
}

/// Skinned copy 0's previous positions of the clusters it skinned this frame.
fn skinned_previous(g: &Frame) -> Vec<[f32; 3]> {
    let (k, _, _) = skinned_entry(&g.view, 0);
    skinned_array(g, k.previous)
}

fn skinned_array(g: &Frame, array: GpuPtr<[f32; 3]>) -> Vec<[f32; 3]> {
    let root = read_buffer(g.view.scene, 1)[0];
    let (_, _, clusters) = skinned_entry(&g.view, 0);
    let k: SkinnedGpu = read_buffer(root.skinned, 1)[0];
    let inst = read_buffer(root.instances.offset(u64::from(k.slot)), 1)[0];
    let model = read_buffer(root.models.offset(u64::from(inst.model)), 1)[0];
    let mesh = read_buffer(root.meshes.offset(u64::from(model.mesh)), 1)[0];
    let skin = read_buffer(mesh.skin, 1)[0];
    let first: Vec<u32> = read_buffer(skin.vertex_first, u64::from(mesh.cluster_count) + 1);
    let all: Vec<[f32; 3]> = read_buffer(array, u64::from(*first.last().unwrap()));
    clusters
        .iter()
        .flat_map(|&c| all[first[c as usize] as usize..first[c as usize + 1] as usize].to_vec())
        .collect()
}

/// A copy of an edit reads the edit's memory, so it goes with the edit, whether the edit ends by
/// itself or with its pack.
#[test]
fn a_copy_of_an_edit_ends_with_the_edit() {
    let _serial = serial();
    let ends: [fn(&mut World, EditId); 2] = [
        |w, edit| w.scenes.destroy_edit(edit),
        |w, _| w.scenes.unmount(w.mount),
    ];
    for end in ends {
        let mut w = World::with(&RendererDesc {
            raster_error_pixels: 4.0,
            rt_error_pixels: 4.0,
            ..RendererDesc::default()
        });
        let source = w.model("dag ground");
        let edit = w
            .scenes
            .create_edit(&EditDesc {
                model: source,
                touched: Some(&[[-2.0, 0.0, -2.0, 1.0]]),
                bounds: None,
            })
            .unwrap();
        settle_edits(&mut w, &[edit]);
        let model = w.scenes.edit(edit).unwrap().model;
        let placed = place_models(&mut w, &[model]);
        let (eye, target) = (DVec3::new(0.0, 0.5, 0.0), DVec3::new(0.0, 0.0, -14.0));
        let mut sh = Shot::at(eye, target, 1);
        let at = Vec3::new(0.0, 0.0, -14.0);
        let linear = Mat3::from_rotation_x(1.1);
        sh.moved = vec![(0, Affine3::from_mat3_translation(linear, at))];
        let g = shoot(&mut w, &placed, &sh);
        assert!(g.ids.iter().any(|&id| id != 0), "the copy never drew");
        end(&mut w, edit);
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, 2));
        assert!(
            g.ids.iter().all(|&id| id == 0),
            "the copy outlived its edit"
        );
        assert_clean();
    }
}

#[test]
fn a_partial_edit_holds_what_it_touches_at_level_0_and_the_rest_keeps_its_detail() {
    let _serial = serial();
    let mut w = World::with(&RendererDesc {
        raster_error_pixels: 4.0,
        rt_error_pixels: 4.0,
        ..RendererDesc::default()
    });
    let source = w.model("dag ground");
    let touched = [[-2.0f32, 0.0, -2.0, 1.0]];
    let edit = w
        .scenes
        .create_edit(&EditDesc {
            model: source,
            touched: Some(&touched),
            bounds: None,
        })
        .unwrap();
    settle_edits(&mut w, &[edit]);
    let model = w.scenes.edit(edit).unwrap().model;
    let placed = place_models(&mut w, &[source, model]);
    let linear = Mat3::from_rotation_x(1.1);
    let at = Vec3::new(0.0, 0.0, -14.0);
    let hidden = Affine3::ZERO;
    let eye = DVec3::new(0.0, 0.5, 0.0);
    let target = DVec3::new(0.0, 0.0, -14.0);
    let mut sh = Shot::at(eye, target, 1);
    sh.moved = vec![(0, Affine3::from_mat3_translation(linear, at)), (1, hidden)];
    let base = shoot(&mut w, &placed, &sh);
    let mut sh = Shot::at(eye, target, 2);
    sh.moved = vec![(0, hidden), (1, Affine3::from_mat3_translation(linear, at))];
    let g = shoot(&mut w, &placed, &sh);
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, covered) = parity(&g, &r, 2e-2);
    assert!(covered > r.traced.len() / 8, "{covered} pixels covered");
    assert!(
        matched as f64 >= 0.98 * r.traced.len() as f64,
        "{matched} of {} pixels agree",
        r.traced.len()
    );
    let base_covered = base.ids.iter().filter(|&&id| id != 0).count();
    assert!(
        covered.abs_diff(base_covered) * 50 < base_covered,
        "cracks: {covered} pixels covered against the source's {base_covered}"
    );
    let cpu = CpuMesh::of("dag ground");
    let t = Vec3::new(touched[0][0], touched[0][1], touched[0][2]);
    let (mut inside, mut outside_coarse) = (0, 0);
    for &c in g.clusters.iter().filter(|&&c| c != 0) {
        let cluster = c - 1;
        let p = cpu.firsts.partition_point(|&f| f <= cluster) - 1;
        let view = PageView::new(&cpu.pages[p], &cpu.header).unwrap();
        let k = view.cluster(cluster - cpu.firsts[p]);
        let center = Vec3::from(k.cull_sphere.center);
        let reach = k.cull_sphere.radius;
        let distance = center.distance(t);
        if distance < touched[0][3] {
            inside += 1;
            assert_eq!(k.self_error, 0.0, "a cluster inside the sphere is coarser");
        } else if distance > touched[0][3] + reach && k.self_error > 0.0 {
            outside_coarse += 1;
        }
    }
    assert!(inside > 50, "{inside} pixels inside the sphere");
    assert!(
        outside_coarse > 500,
        "{outside_coarse} coarser pixels outside the sphere"
    );
    w.scenes.destroy_edit(edit);
    assert_clean();
}

// --- Cooked instance pages ----------------------------------------------------------------------

/// The mounted pages of the pack, by the cell they sit in.
fn row_pages(
    w: &World,
) -> Vec<(
    PageId,
    orr_pack::instance::InstancePageInfo,
    Vec<InstanceRecord>,
    Vec<AffineRows>,
)> {
    let a = Archive::from_memory(Arc::clone(pack())).unwrap();
    let mut out = Vec::new();
    for e in a.entries() {
        if e.class != orr_pack::Class::InstancePage as u16 || !a.name(e).starts_with("rows/") {
            continue;
        }
        let id = w.scenes.instance_page(w.mount, e.id).unwrap();
        let info = orr_pack::instance::InstancePageInfo::parse(&blob(&a, e, 0)).unwrap();
        let (records, table) = orr_pack::instance::records(&info.header, &blob(&a, e, 1)).unwrap();
        out.push((id, info, records, table));
    }
    out.sort_by_key(|p| p.1.header.cell[0]);
    out
}

fn add_page(s: &mut SceneWriter<'_>, page: PageId) -> InstanceRange {
    s.add_instance_page(&InstancePageRef {
        frame: None,
        page,
        poses: &[],
        material_sets: &[],
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 1000.0,
        animation_range: 0.0,
        user_tag: 7,
    })
    .unwrap()
}

/// A scene without copies yet.
fn empty_placed(w: &mut World, write: impl FnOnce(&mut SceneWriter<'_>)) -> Placed {
    let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = w.scenes.create_view_history().unwrap();
    let list = [scene];
    let f = w
        .scenes
        .begin_frame(&FrameDesc {
            time: time(0),
            scenes: &list,
            low_latency: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    write(&mut s);
    drop(s);
    wait(f.end(&[]).unwrap());
    Placed {
        scene,
        ranges: Vec::new(),
        group,
        history,
        count: 0,
    }
}

#[test]
fn a_cooked_page_draws_and_traces_like_its_placements() {
    let _serial = serial();
    let mut w = finest_world();
    let pages = row_pages(&w);
    let (near, near_info, near_records, near_table) = &pages[0];
    // The sheared balls and those turned off the axes hold rows; the unturned one is compact.
    assert!(near_table.len() >= 4);
    assert!(near_records.iter().any(|r| r.exact_row().is_none()));
    let far = pages.last().unwrap().0;
    assert_ne!(*near, far, "the pack has a page per cell");
    let eye = DVec3::new(9.0, 6.0, 26.0);
    let target = DVec3::new(9.0, 1.0, 8.0);

    // The same copies placed at run time, section masks as the page's prototypes hide them.
    let runtime = {
        let ball = w.model("ball");
        let scene = w.scenes.create_scene(&SceneDesc::default()).unwrap();
        let history = w.scenes.create_view_history().unwrap();
        let list = [scene];
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(0),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let mut s = f.scene(scene).unwrap();
        let group = s
            .create_transform_group(&TransformGroupDesc {
                count: near_records.len() as u32,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        let mut ranges = Vec::new();
        for (i, r) in near_records.iter().enumerate() {
            let m = orr_pack::instance::decode_record(&near_info.header, near_table, r);
            s.transforms(group, i as u32, 1).unwrap()[0] = m.into();
            let range = s
                .add_instances(&InstancesDesc {
                    model: ball,
                    copies: 1,
                    transforms: group,
                    first: i as u32,
                    pose: None,
                    material_set: None,
                    flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                    rt_range: 1000.0,
                    animation_range: 0.0,
                    user_tag: 7,
                })
                .unwrap();
            let hidden = near_info
                .sections_of(&near_info.prototypes[r.prototype as usize])
                .iter()
                .enumerate()
                .fold(0u16, |m, (k, o)| m | (u16::from(o.hidden != 0) << k));
            s.copy_variants(
                range,
                0,
                &[CopyVariant {
                    pose: 0,
                    material_set: 0,
                    section_mask: hidden,
                }],
            )
            .unwrap();
            ranges.push(range);
        }
        drop(s);
        wait(f.end(&[]).unwrap());
        Placed {
            scene,
            ranges,
            group,
            history,
            count: near_records.len() as u32,
        }
    };
    let base = shoot(&mut w, &runtime, &Shot::at(eye, target, 1));
    let base = {
        // The section-mask BLAS exist from the second frame.
        let _ = base;
        shoot(&mut w, &runtime, &Shot::at(eye, target, 2))
    };

    let mut range = None;
    let placed = empty_placed(&mut w, |s| range = Some(add_page(s, *near)));
    let range = range.unwrap();
    assert_eq!(range.count as usize, near_records.len());
    let start = Instant::now();
    let mut frame = 2;
    let g = loop {
        frame += 1;
        let g = shoot(&mut w, &placed, &Shot::at(eye, target, frame));
        if frame > 4 && g.ids.iter().any(|&id| id != 0) {
            break shoot(&mut w, &placed, &Shot::at(eye, target, frame + 1));
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the page never landed"
        );
    };
    let n = g.depth.len();
    let same = (0..n)
        .filter(|&i| (base.depth[i] - g.depth[i]).abs() <= 1e-4 * base.depth[i].max(1e-6))
        .count();
    assert!(
        same as f64 >= 0.995 * n as f64,
        "{same} of {n} depths equal the runtime copies'"
    );
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, covered) = parity(&g, &r, 1e-3);
    assert!(covered > 200, "{covered} pixels covered");
    assert!(
        matched as f64 >= 0.995 * n as f64,
        "{matched} of {n} pixels agree"
    );
    assert_clean();
}

#[test]
fn instance_pages_come_nearest_first_under_their_budget_and_their_ranges_stay_valid() {
    let _serial = serial();
    let bytes: Vec<u64> = {
        let a = Archive::from_memory(Arc::clone(pack())).unwrap();
        a.entries()
            .iter()
            .filter(|e| e.class == orr_pack::Class::InstancePage as u16)
            .filter(|e| a.name(e).starts_with("rows/"))
            .map(|e| u64::from(a.blobs(e)[1].raw_len).next_multiple_of(256))
            .collect()
    };
    // Room for one page of records.
    let mut w = World::with(&RendererDesc {
        raster_error_pixels: 1e-3,
        rt_error_pixels: 1e-3,
        budgets: Budgets {
            instance_page_pool: bytes.iter().copied().max().unwrap(),
            ..Budgets::default()
        },
        ..RendererDesc::default()
    });
    let pages = row_pages(&w);
    let near = pages[0].0;
    let far = pages.last().unwrap().0;
    let mut ranges = (None, None);
    let placed = empty_placed(&mut w, |s| {
        ranges = (Some(add_page(s, near)), Some(add_page(s, far)));
    });
    let (near_range, far_range) = (ranges.0.unwrap(), ranges.1.unwrap());
    let in_range = |g: &Frame, r: InstanceRange| {
        g.ids
            .iter()
            .filter(|&&id| id > r.first && id <= r.first + r.count)
            .count()
    };
    let near_eye = DVec3::new(9.0, 6.0, 26.0);
    let near_at = DVec3::new(9.0, 1.0, 8.0);
    let far_x = 4096.0;
    let far_eye = DVec3::new(far_x + 9.0, 6.0, 26.0);
    let far_at = DVec3::new(far_x + 9.0, 1.0, 8.0);
    let mut frame = 0;
    let mut shot_until = |w: &mut World, eye, at, r: InstanceRange, other: InstanceRange| {
        let start = Instant::now();
        loop {
            frame += 1;
            let g = shoot(w, &placed, &Shot::at(eye, at, frame));
            let pool = *w
                .scenes
                .residency()
                .class(orr_scene::ResidencyClass::InstancePages);
            assert!(pool.resident_bytes + pool.loading_bytes <= pool.budget);
            if in_range(&g, r) > 100 {
                assert_eq!(in_range(&g, other), 0, "the other page's range draws");
                return;
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "the page never landed"
            );
        }
    };
    shot_until(&mut w, near_eye, near_at, near_range, far_range);
    assert!(
        w.scenes.page(far).unwrap().records.address().0 == 0,
        "budget holds one page"
    );
    // Moving away, the near page leaves after its hold and the far one takes its place.
    shot_until(&mut w, far_eye, far_at, far_range, near_range);
    assert!(
        w.scenes.page(near).unwrap().records.address().0 == 0,
        "the near page left"
    );
    // Back again: the same range draws without being placed anew.
    shot_until(&mut w, near_eye, near_at, near_range, far_range);
    assert_clean();
}

#[test]
fn prefetch_regions_load_the_pages_they_meet() {
    let _serial = serial();
    let mut w = finest_world();
    let pages = row_pages(&w);
    let near = pages[0].0;
    let far = pages.last().unwrap().0;
    let start = Instant::now();
    while w.scenes.page(near).unwrap().records.address().0 == 0 {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the region never landed"
        );
        w.scenes.prefetch_region(DVec3::new(9.0, 1.0, 8.0), 10.0);
        let f = w
            .scenes
            .begin_frame(&FrameDesc {
                time: time(0),
                scenes: &[],
                low_latency: None,
            })
            .unwrap();
        wait(f.end(&[]).unwrap());
    }
    assert!(
        w.scenes.page(far).unwrap().records.address().0 == 0,
        "the region stays away from the far page"
    );
    assert_clean();
}

#[test]
fn a_streaming_mesh_traces_its_finer_cached_cuts_where_the_raster_draws() {
    let _serial = serial();
    let mut w = World::with(&RendererDesc {
        budgets: Budgets {
            mesh_resident_max: 0,
            ..Budgets::default()
        },
        rt_merge_pixels: 1e9,
        raster_error_pixels: 1e-3,
        rt_error_pixels: 1e-3,
        ..RendererDesc::default()
    });
    let items = dag_items();
    let placed = place(&mut w, &items);
    let eye = DVec3::new(0.3, 1.2, 4.0);
    let target = DVec3::new(0.0, 0.0, -10.0);
    let mut last = None;
    for frame in 1..=240 {
        last = Some(shoot(&mut w, &placed, &Shot::at(eye, target, frame)));
    }
    let g = last.unwrap();
    let r = rays(&g.view, &trace::MODULE, "pixels", 0);
    let (matched, _) = parity(&g, &r, 3e-2);
    let total = r.traced.len();
    assert!(
        matched as f64 >= 0.97 * total as f64,
        "{matched} of {total} pixels agree"
    );
    assert_eq!(w.scenes.residency().failed, 0);
    assert_clean();
}

/// Every record of `materials` through `StandardSurface` at a flat, facing point.
fn surface_params(
    d: &orr_gpu::Device,
    records: &[MaterialGpu],
    list_extra: impl FnOnce(&mut orr_gpu::CommandList<'_>),
    vertex_color: [f32; 4],
) -> Vec<surface_params::Params> {
    surface_params_under(d, records, None, list_extra, vertex_color)
}

/// [`surface_params`] with `decal`, if any, over every point.
fn surface_params_under(
    d: &orr_gpu::Device,
    records: &[MaterialGpu],
    decal: Option<DecalGpu>,
    list_extra: impl FnOnce(&mut orr_gpu::CommandList<'_>),
    vertex_color: [f32; 4],
) -> Vec<surface_params::Params> {
    let upload = |bytes: &[u8], name| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, name)
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let materials = upload(bytemuck::cast_slice(records), "materials");
    let decals = upload(
        bytemuck::cast_slice(&[decal.unwrap_or_else(bytemuck::Zeroable::zeroed)]),
        "decals",
    );
    let refs = upload(bytemuck::bytes_of(&0u32), "decal refs");
    let scene = upload(
        bytemuck::bytes_of(&SceneRoot {
            materials: materials.gpu().cast(),
            decals: decals.gpu().cast(),
            decal_refs: refs.gpu().cast(),
            decal_globals: u32::from(decal.is_some()),
            ..bytemuck::Zeroable::zeroed()
        }),
        "scene",
    );
    let out = d
        .gpu_malloc::<surface_params::Params>(MemoryKind::Readback, records.len() as u64, "params")
        .unwrap();
    let root = upload(
        bytemuck::bytes_of(&surface_params::Root {
            scene: scene.gpu().cast(),
            out: out.gpu(),
            vertex_color,
            count: records.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }),
        "root",
    );
    let p = pipeline(&surface_params::MODULE, "main");
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list_extra(&mut list);
    list.dispatch(
        &p,
        root.gpu().cast::<surface_params::Root>(),
        [records.len().div_ceil(64) as u32, 1, 1],
    );
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![list], &[]).unwrap());
    out.as_slice().unwrap().to_vec()
}

/// Each slot past the first ones reaches the one parameter it states, and no other: a value of a
/// texture times its factor, or what its encoding says.
#[test]
fn every_slot_reaches_its_parameter() {
    use orr_gpu::{Format, TextureDesc, TextureMemory, TextureUsage};
    use orr_pack::material::{Material, TextureBinding, slot};
    type P = surface_params::Params;

    let _guard = serial();
    let d = device();
    const TEXEL: [u8; 4] = [128, 64, 192, 255];
    let v = TEXEL.map(|c| f32::from(c) / 255.0);
    let desc = TextureDesc::d2(Format::Rgba8Unorm, 4, 4, TextureUsage::SAMPLED);
    let texture = d
        .create_texture(&desc, TextureMemory::Dedicated, "one texel")
        .unwrap();
    let slot_index = d.alloc_image_slot().unwrap();
    let index = d.write_sampled(slot_index, &texture, &orr_gpu::ViewDesc::default());
    let binding = TextureBinding {
        ..bytemuck::Zeroable::zeroed()
    };
    let bound_texture = BoundTexture {
        index,
        primaries: Primaries::Rec2020.id(),
        sampler: d.sampler(&orr_scene::binding_sampler(&binding)).unwrap(),
        ..BoundTexture::NONE
    };
    let base = Material {
        coat_weight: 0.5,
        coat_anisotropy: 0.5,
        fuzz_weight: 0.5,
        thin_film_weight: 0.5,
        specular_anisotropy: 0.5,
        subsurface_weight: 0.5,
        subsurface_radius: 1.0,
        emission_luminance: 2.0,
        ..Material::default()
    };
    let with = |s: usize| {
        let mut bound = [BoundTexture::NONE; slot::COUNT];
        bound[s] = bound_texture;
        MaterialGpu::new(&base, &bound)
    };
    // Per slot, the parameter, its factor and what it reads with the texture.
    let scalar = |f: f32| f * v[0];
    let turned = {
        let [x, y] = [2.0 * v[0] - 1.0, 2.0 * v[1] - 1.0];
        y.atan2(x)
    };
    type Read = fn(&P) -> f32;
    let rows: Vec<(usize, Read, f32)> = vec![
        (
            slot::SPECULAR_IOR,
            |p| p.specular_ior,
            scalar(base.specular_ior),
        ),
        (slot::DIFFUSE_ROUGHNESS, |p| p.diffuse_roughness, 0.0),
        (slot::ANISOTROPY, |p| p.specular_anisotropy, scalar(0.5)),
        (
            slot::SUBSURFACE_WEIGHT,
            |p| p.subsurface_weight,
            scalar(0.5),
        ),
        (
            slot::SUBSURFACE_ANISOTROPY,
            |p| p.subsurface_anisotropy,
            2.0 * v[0] - 1.0,
        ),
        (slot::FUZZ_WEIGHT, |p| p.fuzz_weight, scalar(0.5)),
        (
            slot::FUZZ_ROUGHNESS,
            |p| p.fuzz_roughness,
            scalar(base.fuzz_roughness),
        ),
        (slot::COAT_IOR, |p| p.coat_ior, scalar(base.coat_ior)),
        (slot::COAT_DARKENING, |p| p.coat_darkening, scalar(1.0)),
        (slot::COAT_ANISOTROPY, |p| p.coat_anisotropy, scalar(0.5)),
        (slot::THIN_FILM_WEIGHT, |p| p.thin_film_weight, scalar(0.5)),
        (
            slot::THIN_FILM_THICKNESS,
            |p| p.thin_film_thickness,
            scalar(base.thin_film_thickness),
        ),
        (
            slot::THIN_FILM_IOR,
            |p| p.thin_film_ior,
            scalar(base.thin_film_ior),
        ),
    ];
    type Color = fn(&P) -> [f32; 3];
    let colors: Vec<(usize, Color, [f32; 3])> = vec![
        (
            slot::SPECULAR_COLOR,
            |p| p.specular_color,
            [v[0], v[1], v[2]],
        ),
        (
            slot::TRANSMISSION_COLOR,
            |p| p.transmission_color,
            [v[0], v[1], v[2]],
        ),
        (slot::FUZZ_COLOR, |p| p.fuzz_color, [v[0], v[1], v[2]]),
        (slot::COAT_COLOR, |p| p.coat_color, [v[0], v[1], v[2]]),
    ];
    let words = |p: &P| -> Vec<u32> { bytemuck::cast_slice(bytemuck::bytes_of(p)).to_vec() };
    let mut records = vec![MaterialGpu::new(&base, &[BoundTexture::NONE; slot::COUNT])];
    let slots: Vec<usize> = rows
        .iter()
        .map(|r| r.0)
        .chain(colors.iter().map(|c| c.0))
        .chain([
            slot::BASE_WEIGHT,
            slot::SUBSURFACE_RADIUS,
            slot::SUBSURFACE_RADIUS_SCALE,
            slot::TANGENT,
            slot::COAT_TANGENT,
            slot::COAT_ROUGHNESS,
            slot::COAT_NORMAL,
            slot::EMISSION_LUMINANCE,
        ])
        .collect();
    records.extend(slots.iter().map(|&s| with(s)));
    // OpenPBR's medium, at a point of each slot that describes it.
    let medium_base = Material {
        transmission_color: [0.5, 0.6, 0.7],
        transmission_depth: 2.0,
        transmission_scatter: [0.25, 0.5, 0.75],
        transmission_scatter_anisotropy: 0.25,
        transmission_dispersion_scale: 0.5,
        transmission_dispersion_abbe_number: 40.0,
        ..base
    };
    let medium_slots = [
        slot::TRANSMISSION_COLOR,
        slot::TRANSMISSION_DEPTH,
        slot::TRANSMISSION_SCATTER,
        slot::TRANSMISSION_SCATTER_ANISOTROPY,
        slot::DISPERSION_SCALE,
        slot::DISPERSION_ABBE,
    ];
    let first_medium = records.len();
    records.push(MaterialGpu::new(
        &medium_base,
        &[BoundTexture::NONE; slot::COUNT],
    ));
    records.extend(medium_slots.map(|s| {
        let mut bound = [BoundTexture::NONE; slot::COUNT];
        bound[s] = bound_texture;
        MaterialGpu::new(&medium_base, &bound)
    }));
    let thin_dispersion = records.len();
    let mut bound = [BoundTexture::NONE; slot::COUNT];
    bound[slot::DISPERSION_SCALE] = bound_texture;
    records.push(MaterialGpu::new(
        &Material {
            thin_walled: 1,
            ..medium_base
        },
        &bound,
    ));
    let mut object = with(slot::COAT_NORMAL);
    object.flags |= orr_scene::material_flags::OBJECT_COAT_NORMAL;
    records.push(object);
    // A baked quarter turn about x, from the identity: not symmetric, so rows read as rows.
    object.object_frame = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0];
    records.push(object);

    let texels = {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, 64, "texels")
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(&TEXEL.repeat(16));
        a
    };
    let out = surface_params(
        d,
        &records,
        |list| {
            list.discard(&texture);
            list.copy_to_texture(&texture, &TextureRegion::whole_mip(&desc, 0), texels.gpu());
            list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        },
        [1.0; 4],
    );
    let plain = &out[0];
    let at = |s: usize| &out[1 + slots.iter().position(|&x| x == s).unwrap()];
    let close = |a: f32, b: f32, s: usize| assert!((a - b).abs() < 1e-4, "slot {s}: {a} vs {b}");
    // The parameters a slot moves, in words of `Params`.
    let moved = |s: usize| {
        let (a, b) = (words(plain), words(at(s)));
        (0..a.len() - 2).filter(|&i| a[i] != b[i]).count()
    };
    for &(s, read, want) in &rows {
        // A factor of 0 has nothing to scale.
        if s != slot::DIFFUSE_ROUGHNESS {
            close(read(at(s)), want, s);
            assert_eq!(moved(s), 1, "slot {s}");
        }
    }
    for &(s, read, want) in &colors {
        let was = read(plain);
        for c in 0..3 {
            close(read(at(s))[c], was[c] * want[c], s);
        }
    }
    close(
        at(slot::BASE_WEIGHT).base_color[0],
        plain.base_color[0] * v[0],
        slot::BASE_WEIGHT,
    );
    close(
        at(slot::COAT_ROUGHNESS).coat_roughness,
        0.0,
        slot::COAT_ROUGHNESS,
    );
    for c in 0..3 {
        close(
            at(slot::SUBSURFACE_RADIUS).subsurface_radius[c],
            plain.subsurface_radius[c] * v[0],
            slot::SUBSURFACE_RADIUS,
        );
        close(
            at(slot::SUBSURFACE_RADIUS_SCALE).subsurface_radius[c],
            plain.subsurface_radius[c] * v[c],
            slot::SUBSURFACE_RADIUS_SCALE,
        );
        close(
            at(slot::EMISSION_LUMINANCE).emission[c],
            plain.emission[c] * v[0],
            slot::EMISSION_LUMINANCE,
        );
    }
    close(at(slot::TANGENT).tangent_angle, turned, slot::TANGENT);
    close(
        at(slot::COAT_TANGENT).coat_tangent_angle,
        turned,
        slot::COAT_TANGENT,
    );
    let n = at(slot::COAT_NORMAL).coat_normal;
    assert!(
        n[2] < 0.99 && (Vec3::from(n).length() - 1.0).abs() < 1e-4,
        "coat normal {n:?}"
    );
    assert_eq!(plain.coat_normal, [0.0, 0.0, 1.0]);
    let plain_medium = &out[first_medium];
    let medium = |k: usize| &out[first_medium + 1 + k];
    for (c, &texel) in v.iter().enumerate().take(3) {
        close(
            medium(0).medium_color[c],
            plain_medium.medium_color[c] * texel,
            slot::TRANSMISSION_COLOR,
        );
        close(
            medium(2).medium_scatter[c],
            plain_medium.medium_scatter[c] * texel,
            slot::TRANSMISSION_SCATTER,
        );
    }
    close(medium(1).medium_depth, 2.0 * v[0], slot::TRANSMISSION_DEPTH);
    close(
        medium(3).medium_anisotropy,
        2.0 * v[0] - 1.0,
        slot::TRANSMISSION_SCATTER_ANISOTROPY,
    );
    close(
        medium(4).medium_dispersion_scale,
        0.5 * v[0],
        slot::DISPERSION_SCALE,
    );
    close(
        medium(5).medium_dispersion_abbe,
        40.0 * v[0],
        slot::DISPERSION_ABBE,
    );
    // A thin wall has no medium, but its IOR disperses by the same texture.
    close(
        out[thin_dispersion].medium_dispersion_scale,
        0.5 * v[0],
        slot::DISPERSION_SCALE,
    );
    // A mesh space direction turns with the placement, a quarter turn about z.
    let [x, y, z] = [v[0], v[1], v[2]].map(|c| 2.0 * c - 1.0);
    let want = Vec3::new(-y, x, z).normalize();
    let n = Vec3::from(out[out.len() - 2].coat_normal);
    assert!(
        n.abs_diff_eq(want, 1e-4),
        "object normal {n} against {want}"
    );
    // The baked turn comes first, to (x, -z, y), then the placement's.
    let want = Vec3::new(z, x, y).normalize();
    let n = Vec3::from(out[out.len() - 1].coat_normal);
    assert!(
        n.abs_diff_eq(want, 1e-4),
        "baked object normal {n} against {want}"
    );
    d.free_image_slot_after_submitted(slot_index);
    assert_clean();
}

/// What each model makes of the same fields, at one point.
#[test]
fn models_state_their_specular_and_tint() {
    use orr_pack::material::{Material, slot};
    let _guard = serial();
    let d = device();
    let specular = Material {
        specular_color: [2.0, 1.0, 0.5],
        specular_ior: 1.5,
        specular_weight: 0.5,
        base_color: [0.5, 0.25, 0.125],
        transmission_weight: 1.0,
        thin_walled: 1,
        ..Material::default()
    };
    let record = |m: &Material| MaterialGpu::new(m, &[BoundTexture::NONE; slot::COUNT]);
    let models = [
        Material::MODEL_OPENPBR,
        Material::MODEL_GLTF,
        Material::MODEL_PREVIEW_SPECULAR,
    ];
    let records: Vec<MaterialGpu> = models
        .iter()
        .map(|&model| record(&Material { model, ..specular }))
        .collect();
    let out = surface_params(d, &records, |_| {}, [1.0; 4]);
    let rec2020 = |c: [f32; 3]| {
        orr_color::rec709_to_rec2020(Vec3::from(c))
            .as_vec3()
            .to_array()
    };
    let near = |a: [f32; 3], b: [f32; 3], what: &str| {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-5, "{what}: {a:?} vs {b:?}");
        }
    };
    near(
        out[0].specular_color,
        rec2020(specular.specular_color),
        "OpenPBR keeps the tint",
    );
    near(
        out[1].specular_color,
        rec2020(specular.specular_color.map(|c| 0.04 * c)),
        "glTF scales the IOR's F0",
    );
    near(
        out[2].specular_color,
        rec2020(specular.specular_color.map(|c| c.min(1.0))),
        "the specular workflow states F0, a reflectance",
    );
    assert_eq!(out[0].flags & BSDF_FRESNEL_MIX, 0);
    assert_ne!(out[1].flags & BSDF_FRESNEL_MIX, 0);
    near(out[0].transmission_color, rec2020([1.0; 3]), "no tint");
    near(
        out[1].transmission_color,
        rec2020(specular.base_color),
        "glTF's base color tints",
    );
    near(out[2].transmission_color, rec2020([1.0; 3]), "no tint");
    assert_clean();
}

/// A hit's vertex color weighs each corner by its barycentric, and its normals stay on the side
/// a mirrored placement puts them, at sizes whose determinant is outside a float's range and on
/// triangles whose area is. A ray from behind meets the whole frame reversed toward it, the
/// tangent and bitangent with the normals, on a single-sided material as on a double-sided one,
/// but for a dielectric's; it meets a double-sided material's back.
#[test]
fn triangle_geometry_interpolates_colors_and_keeps_a_mirror_at_any_size() {
    use orr_pack::material::slot;
    let _guard = serial();
    let d = device();
    let upload = |bytes: &[u8], name| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, name)
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let two_sided = Material {
        double_sided: 1,
        ..Material::default()
    };
    let mut front = MaterialGpu::new(&two_sided, &[BoundTexture::NONE; slot::COUNT]);
    front.flags |= orr_scene::material_flags::HAS_BACK;
    front.back = 1;
    let mut back = front;
    back.flags ^= orr_scene::material_flags::HAS_BACK;
    let glass = |mut m: MaterialGpu, first: u32| {
        m.flags |= orr_scene::material_flags::DIELECTRIC;
        m.back += first;
        m
    };
    let thin = |mut m: MaterialGpu| {
        m.flags |= orr_scene::material_flags::THIN_WALLED;
        m
    };
    let single = MaterialGpu::new(&Material::default(), &[BoundTexture::NONE; slot::COUNT]);
    let mut single_glass = single;
    single_glass.flags |= orr_scene::material_flags::DIELECTRIC;
    // The section material's walls decide the interior: a thin back on a solid front leaves it,
    // and a solid back on a thin front has none.
    let records = [
        front,
        back,
        glass(front, 2),
        thin(glass(back, 2)),
        thin(glass(front, 4)),
        glass(back, 4),
        single,
        single_glass,
    ];
    let materials = upload(bytemuck::cast_slice(&records), "materials");
    let scene = upload(
        bytemuck::bytes_of(&SceneRoot {
            materials: materials.gpu().cast(),
            ..bytemuck::Zeroable::zeroed()
        }),
        "scene",
    );
    let corners = [
        [1.0, 0.0, 0.0, 0.25],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 1.0],
    ];
    let p = pipeline(&triangle_at::MODULE, "main");
    let render = d.queue(QueueKind::Render).unwrap();
    for (size, edge, side, first) in [
        (1.0, 1.0, 1.0, 0),
        (1e-16, 1.0, 1.0, 0),
        (1e-25, 1.0, 1.0, 0),
        (1e25, 1.0, 1.0, 0),
        (1e12, 1e-12, 1.0, 0),
        (1.0, 1.0, -1.0, 0),
        (1e-16, 1.0, -1.0, 0),
        (1.0, 1.0, 1.0, 2),
        (1.0, 1.0, -1.0, 2),
        (1.0, 1.0, 1.0, 4),
        (1.0, 1.0, -1.0, 4),
        (1.0, 1.0, 1.0, 6),
        (1.0, 1.0, -1.0, 6),
        (1e-16, 1.0, -1.0, 6),
        (1.0, 1.0, 1.0, 7),
        (1.0, 1.0, -1.0, 7),
    ] {
        let out = d
            .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 6, "out")
            .unwrap();
        let root = upload(
            bytemuck::bytes_of(&triangle_at::Root {
                scene: scene.gpu().cast(),
                view: [0.0, 0.0, -side, first as f32],
                out: out.gpu().cast(),
                color0: corners[0],
                color1: corners[1],
                color2: corners[2],
                point: [0.25, 0.5, size, edge],
            }),
            "root",
        );
        let mut list = render.command_list().unwrap();
        list.dispatch(&p, root.gpu().cast::<triangle_at::Root>(), [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        wait(render.submit(vec![list], &[]).unwrap());
        let [color, geometric, normal, material, tangent, coat] = out.as_slice().unwrap()[..]
        else {
            panic!("six rows");
        };
        assert_eq!(color, [0.25, 0.25, 0.5, 0.6875], "{size}");
        let dielectric = matches!(first, 2 | 4 | 7);
        let turned = if dielectric { 1.0 } else { side };
        let up = |n: [f32; 4]| Vec4::from(n).abs_diff_eq(Vec4::Z * turned, 1e-5);
        assert!(up(geometric), "{size} {first}: {geometric:?}");
        assert!(up(normal), "{size} {first}: {normal:?}");
        // The triangle's u grows along -x and its v shrinks along -y.
        for t in [tangent, coat] {
            let along = Vec3::from_slice(&t[..3]);
            let across = Vec3::from_slice(&normal[..3]).cross(along) * t[3];
            assert!(
                along.abs_diff_eq(Vec3::NEG_X * turned, 1e-5),
                "{size} {first}: tangent {t:?}"
            );
            assert!(
                across.abs_diff_eq(Vec3::NEG_Y * turned, 1e-5),
                "{size} {first}: bitangent {across}"
            );
        }
        let shown = if first >= 6 {
            first
        } else {
            first + u32::from(side < 0.0)
        };
        assert_eq!(material[0], shown as f32, "{size} {first}");
        assert_eq!(
            material[1],
            f32::from(u8::from(first == 2 || first == 7)),
            "{size} {first}"
        );
    }
    assert_clean();
}

/// The distance from `p` to the line from the origin along unit `d`.
fn off_line(p: DVec3, d: DVec3) -> f64 {
    (p - d * p.dot(d)).length()
}

/// A pixel ray that passes a needle the raster drew it for shows the needle's point nearest it,
/// at weights inside the triangle where the plane's lie far outside it, so no attribute leaves
/// its corners' range. A ray along a triangle's plane shows a point of the triangle where the
/// plane gives none, and a ray that meets a triangle keeps the plane's point.
#[test]
fn a_pixel_ray_passing_its_triangle_shows_the_triangles_nearest_point() {
    let _guard = serial();
    let d = device();
    let p = pipeline(&triangle_point::MODULE, "main");
    let render = d.queue(QueueKind::Render).unwrap();
    let run = |toward: DVec3, corner: DVec3, edges: [DVec3; 2]| {
        let out = d
            .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 3, "out")
            .unwrap();
        let row = |v: DVec3| v.as_vec3().extend(0.0).to_array();
        let record = triangle_point::Root {
            out: out.gpu().cast(),
            pad0: 0,
            pad1: 0,
            direction: row(toward.normalize()),
            corner: row(corner),
            edge1: row(edges[0]),
            edge2: row(edges[1]),
        };
        let bytes = bytemuck::bytes_of(&record);
        let mut root = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "root")
            .unwrap();
        root.as_mut_slice().unwrap().copy_from_slice(bytes);
        let mut list = render.command_list().unwrap();
        list.dispatch(&p, root.gpu().cast::<triangle_point::Root>(), [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        wait(render.submit(vec![list], &[]).unwrap());
        let rows = out.as_slice().unwrap();
        let v = |i: usize| Vec3::from_slice(&rows[i][..3]).as_dvec3();
        (v(0), v(1), v(2))
    };
    // What the kernel reads, as it reads it.
    let single = |v: DVec3| v.as_vec3().as_dvec3();
    // The least distance from the line along `toward` to the triangle's edges.
    let nearest = |toward: DVec3, corner: DVec3, edges: [DVec3; 2]| {
        let d = single(toward.normalize());
        let (q, [e1, e2]) = (single(corner), edges.map(single));
        let sides = [(q, e1), (q, e2), (q + e1, e2 - e1)];
        let steps = 200_000;
        let along = (0..=steps).map(|i| f64::from(i) / f64::from(steps));
        along
            .flat_map(|u| sides.map(|(a, e)| off_line(a + e * u, d)))
            .fold(f64::INFINITY, f64::min)
    };
    let turn = DQuat::from_axis_angle(DVec3::new(0.3, 0.5, 0.8).normalize(), 0.7);

    // A needle 2 cm long and 0.3 um tall, 3 m away, and a ray 10 um under it.
    let corner = turn * DVec3::new(-0.01, 0.0, -3.0);
    let edges = [DVec3::new(0.02, 0.0, 0.0), DVec3::new(0.01, 3e-7, 0.0)].map(|e| turn * e);
    let toward = turn * DVec3::new(0.002, -1e-5, -3.0);
    let (b, at, plane) = run(toward, corner, edges);
    assert!(plane.y < -10.0, "the plane's weights {plane:?}");
    let weights = [1.0 - b.x - b.y, b.x, b.y];
    assert!(
        weights.iter().all(|w| (-1e-6..=1.0 + 1e-6).contains(w)),
        "weights {weights:?}"
    );
    let [e1, e2] = edges.map(single);
    let on_triangle = single(corner) + e1 * b.x + e2 * b.y;
    assert!(at.distance(on_triangle) < 2e-6, "{at:?} for {b:?}");
    let ray = single(toward.normalize());
    let (off, least) = (off_line(at, ray), nearest(toward, corner, edges));
    assert!(
        off < least + 2e-6,
        "{off:e} m off the ray, the nearest {least:e}"
    );
    assert!((b.z - at.dot(ray)).abs() < 1e-5, "distance {} m", b.z);

    // A triangle seen along its plane, which holds the eye.
    let corner = DVec3::new(-0.01, 0.0, -3.0);
    let edges = [DVec3::new(0.02, 0.0, 0.0), DVec3::new(0.01, 0.0, -0.5)];
    let (b, at, _) = run(DVec3::new(0.002, 0.0, -3.0), corner, edges);
    assert!(b.truncate().distance(DVec2::new(0.6, 0.0)) < 1e-4, "{b:?}");
    assert!(at.distance(DVec3::new(0.002, 0.0, -3.0)) < 1e-5, "{at:?}");
    assert!((b.z - at.length()).abs() < 1e-5, "distance {} m", b.z);

    // A ray that meets a triangle.
    let corner = turn * DVec3::new(-1.0, -1.0, -3.0);
    let edges = [DVec3::new(2.0, 0.0, 0.0), DVec3::new(0.0, 2.0, 0.5)].map(|e| turn * e);
    let toward = turn * DVec3::new(-0.5, -0.2, -2.8);
    let (b, at, plane) = run(toward, corner, edges);
    assert_eq!(b, plane);
    assert_eq!(at.as_vec3(), toward.normalize().as_vec3() * plane.z as f32);
    assert!(b.truncate().distance(DVec2::new(0.25, 0.4)) < 1e-5, "{b:?}");
    assert_clean();
}

/// OpenPBR `geometry_tangent` and `geometry_coat_tangent` are a tangent of the base and of the
/// coat, and MaterialX `tangent`'s `index` picks the coordinates of each: a frame of the coat's
/// own follows its uv set where the base's follows another.
#[test]
fn a_coat_frame_follows_a_uv_set_of_its_own() {
    use orr_pack::material::slot;
    let _guard = serial();
    let d = device();
    let upload = |bytes: &[u8], name| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, name)
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let p = pipeline(&triangle_at::MODULE, "main");
    let render = d.queue(QueueKind::Render).unwrap();
    for (frame, coat_frame, base, coat) in [
        (Material::FRAME_UV0, Material::FRAME_UV1, Vec3::X, Vec3::Y),
        (Material::FRAME_UV1, Material::FRAME_UV0, Vec3::Y, Vec3::X),
        (
            Material::FRAME_UV1,
            Material::FRAME_NORMAL_MAP,
            Vec3::Y,
            Vec3::Y,
        ),
    ] {
        let m = Material {
            frame_uv: frame,
            coat_frame_uv: coat_frame,
            ..Material::default()
        };
        let record = MaterialGpu::new(&m, &[BoundTexture::NONE; slot::COUNT]);
        let materials = upload(bytemuck::bytes_of(&record), "materials");
        let scene = upload(
            bytemuck::bytes_of(&SceneRoot {
                materials: materials.gpu().cast(),
                ..bytemuck::Zeroable::zeroed()
            }),
            "scene",
        );
        let out = d
            .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 6, "out")
            .unwrap();
        let root = upload(
            bytemuck::bytes_of(&triangle_at::Root {
                scene: scene.gpu().cast(),
                view: [0.0, 0.0, -1.0, 0.0],
                out: out.gpu().cast(),
                color0: [1.0; 4],
                color1: [1.0; 4],
                color2: [1.0; 4],
                point: [0.25, 0.5, 1.0, 1.0],
            }),
            "root",
        );
        let mut list = render.command_list().unwrap();
        list.dispatch(&p, root.gpu().cast::<triangle_at::Root>(), [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        wait(render.submit(vec![list], &[]).unwrap());
        let rows = out.as_slice().unwrap();
        // The placement mirrors x, and a tangent's sign doesn't tell a frame's set.
        let along =
            |row: [f32; 4], axis: Vec3| Vec3::from_slice(&row).abs().abs_diff_eq(axis, 1e-5);
        assert!(along(rows[4], base), "{:?}", rows[4]);
        assert!(along(rows[5], coat), "{:?}", rows[5]);
    }
    assert_clean();
}

/// `Material::MODEL_GLTF`: "The base color tints light through the surface", and a decal's base
/// color is the surface's base color where it lies.
#[test]
fn a_decals_base_color_tints_gltf_transmission() {
    use orr_pack::material::Material;
    let _guard = serial();
    let d = device();
    let glass = Material {
        base_color: [0.5; 3],
        transmission_weight: 1.0,
        model: Material::MODEL_GLTF,
        ..Material::default()
    };
    let paint = Material {
        base_color: [1.0, 0.0, 0.0],
        ..Material::default()
    };
    let surface = Material {
        model: Material::MODEL_OPENPBR,
        ..glass
    };
    let record = |m: &Material| MaterialGpu::new(m, &[BoundTexture::NONE; slot::COUNT]);
    let decal = DecalGpu {
        material: 1,
        channels: DecalChannels::BASE_COLOR.0,
        axis_x: [1.0, 0.0, 0.0],
        inv_half_x: 1e-3,
        axis_y: [0.0, 1.0, 0.0],
        inv_half_y: 1e-3,
        axis_z: [0.0, 0.0, 1.0],
        inv_half_z: 1e-3,
        opacity: 1.0,
        cos_full: 0.5,
        ..bytemuck::Zeroable::zeroed()
    };
    let records = [record(&glass), record(&paint), record(&surface)];
    let out = surface_params_under(d, &records, Some(decal), |_| {}, [1.0; 4]);
    let want = orr_color::rec709_to_rec2020(Vec3::X).as_vec3();
    let near = |got: [f32; 3], want: Vec3| (Vec3::from(got) - want).abs().max_element() < 1e-5;
    assert!(near(out[0].base_color, want), "{:?}", out[0].base_color);
    assert!(
        near(out[0].transmission_color, want),
        "{:?}",
        out[0].transmission_color
    );
    assert_eq!(
        out[2].transmission_color, [1.0; 3],
        "OpenPBR's tint is its own"
    );
    assert_clean();
}

/// The vertex color stream multiplies the slots that name a channel of it, a color from that
/// channel on, bound to a texture or not, in linear Rec.709.
#[test]
fn vertex_colors_multiply_the_slots_that_read_them() {
    use orr_pack::material::{Material, slot};
    let _guard = serial();
    let d = device();
    let mut m = Material {
        base_color: [0.5, 0.5, 0.25],
        metalness: 0.5,
        ..Material::default()
    };
    m.textures[slot::BASE_COLOR].vertex = 1;
    m.textures[slot::METALNESS].vertex = 4;
    let plain = Material {
        textures: Material::default().textures,
        ..m
    };
    let record = |m: &Material| MaterialGpu::new(m, &[BoundTexture::NONE; slot::COUNT]);
    let vertex = [0.5, 0.25, 1.0, 0.75];
    let out = surface_params(d, &[record(&m), record(&plain)], |_| {}, vertex);
    let rec2020 = |c: [f32; 3]| {
        orr_color::rec709_to_rec2020(Vec3::from(c))
            .as_vec3()
            .to_array()
    };
    let want = rec2020([0.25, 0.125, 0.25]);
    let near = |(a, b): (&f32, &f32)| (a - b).abs() < 1e-5;
    assert!(
        out[0].base_color.iter().zip(&want).all(near),
        "{:?}",
        out[0].base_color
    );
    assert!((out[0].metalness - 0.375).abs() < 1e-6);
    assert_eq!(out[1].metalness, 0.5, "a slot that names none reads none");
    assert_clean();
}

/// PS `opacity`: 1 is "fully opaque" and 0 "fully transparent", as a slot's complement of the
/// stream reads it. A mesh without the stream reads 1, so it is opaque.
#[test]
fn a_complemented_slot_reads_the_stream_as_opacity() {
    use orr_pack::material::{Material, slot};
    let _guard = serial();
    let d = device();
    let mut m = Material {
        transmission_weight: 1.0,
        thin_walled: 1,
        ..Material::default()
    };
    m.textures[slot::TRANSMISSION].vertex = 3;
    m.textures[slot::TRANSMISSION].vertex_complement = 1;
    let record = MaterialGpu::new(&m, &[BoundTexture::NONE; slot::COUNT]);
    for (opacity, weight) in [
        (0.25, 0.75),
        (1.0, 0.0),
        (0.0, 1.0),
        (4.0, 0.0),
        (-1.0, 1.0),
    ] {
        let out = surface_params(d, &[record], |_| {}, [1.0, 1.0, opacity, 1.0]);
        assert_eq!(out[0].transmission_weight, weight, "opacity {opacity}");
    }
    // Over 2^3 in the stream, as one past half floats holds it.
    m.textures[slot::TRANSMISSION].vertex_complement = 4;
    let record = MaterialGpu::new(&m, &[BoundTexture::NONE; slot::COUNT]);
    let out = surface_params(d, &[record], |_| {}, [1.0, 1.0, 0.25 / 8.0, 1.0]);
    assert_eq!(out[0].transmission_weight, 0.75);
    assert_clean();
}
