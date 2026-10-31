//! A view's raster half on cooked meshes: G-buffer packing against a CPU decoder, reference
//! images of each G-buffer channel, motion vectors that reproject the previous frame,
//! discrete levels chosen by distance, and a linked surface whose coverage and depth offset
//! give the depth and motion of the shape it stands for.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use glam::{DVec3, Mat3, Quat, UVec2, Vec2, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshSettings, Quality, Source,
    TextureSettings, Usage,
};
use orr_gpu::{KernelSet, MemoryKind, Texture, TextureRegion};
use orr_pack::material::{Material, slot};
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::{
    Image, Pixels, Tolerance, assert_clean, check, device, read_buffer, read_texture, wait,
};
use orr_scene::{
    Affine3, CopyVariant, FrameDesc, FrameTime, GroupWriter, InstanceFlags, InstanceRange,
    InstancesDesc, Lens, LoadState, Loading, MaterialGpu, ModelId, MountId, PackSource, Projection,
    RasterView, RendererDesc, SceneDesc, SceneId, Scenes, TransformGroupDesc, TransformGroupId,
    ViewDesc, ViewHistoryId, ViewTarget, VisibleCluster, material_flags,
};
use scene_suite::shaders::{codec, surface_at};
use scene_suite::{linked, pipeline};

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
    let r = (code & 255) as f32 / 255.0 * 0.5;
    (1.0 + r) / (1.0 - r)
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
    occlusion: f32,
}

fn decode_core(a: [u32; 4], b: [u32; 2]) -> Core {
    let e = (b[0] >> 21) & 31;
    Core {
        base_color: rgb8(a[0]),
        roughness: byte(a[0], 24),
        normal: oct16(a[1]),
        emission: rgb9e5(a[2]),
        metalness: byte(a[3], 0),
        specular_weight: byte(a[3], 8),
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
        occlusion: byte(b[1], 24),
    }
}

const THIN_TWO_SIDED: u32 = 1;
const EXTENDED: u32 = 4;
const SUBSURFACE: u32 = 8;
const LOBE_COLORS: u32 = 16;
const BSDF_THIN_WALLED: u32 = 1;

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
            c.specular_weight = rng.next();
            c.specular_color = arr(rng.rgb());
            c.ior = 1.0 + 2.0 * rng.next();
            c.geometric_normal = arr(rng.unit());
            c.diffuse_roughness = rng.next();
            c.occlusion = rng.next();
            c.raster_error = if i % 7 == 0 {
                0.0
            } else {
                1e-5 * 1e7f32.powf(rng.next())
            };
            c.flags = ((i as u32 % 3) << 5) | ((i as u32 % 2) * 2);
            if kind == 1 || kind == 2 {
                c.subsurface_weight = 0.05 + 0.95 * rng.next();
                c.subsurface_color = arr(rng.rgb());
                c.subsurface_radius = arr(rng.rgb() * 0.05);
                c.subsurface_anisotropy = rng.next() * 1.8 - 0.9;
            }
            if kind == 2 {
                c.bsdf_flags = BSDF_THIN_WALLED;
            }
            if kind == 3 {
                c.coat_weight = rng.next();
                c.coat_color = arr(rng.rgb());
                c.fuzz_weight = rng.next();
                c.fuzz_color = arr(rng.rgb());
            }
            if kind != 0 {
                c.coat_roughness = rng.next();
                c.coat_ior = 1.0 + 2.0 * rng.next();
                c.fuzz_roughness = rng.next();
                c.anisotropy = rng.next();
                c.tangent_angle = rng.next() * std::f32::consts::PI;
            }
            c
        })
        .collect();
    let material = MaterialGpu {
        coat_color: [0.25, 0.5, 0.75],
        fuzz_color: [0.75, 0.5, 0.25],
        thin_film_thickness: 400.0,
        thin_film_ior: 1.4,
        flags: material_flags::THIN_FILM,
        ..bytemuck::Zeroable::zeroed()
    };
    let upload = |bytes: &[u8]| {
        let mut a = d
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "codec input")
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        a
    };
    let input = upload(bytemuck::cast_slice(&cases));
    let mat = upload(bytemuck::bytes_of(&material));
    let words = d
        .gpu_malloc::<codec::Words>(MemoryKind::Readback, N as u64, "words")
        .unwrap();
    let decoded = d
        .gpu_malloc::<codec::Case>(MemoryKind::Readback, N as u64, "decoded")
        .unwrap();
    let root = upload(bytemuck::bytes_of(&codec::Root {
        cases: input.gpu.cast(),
        words: words.gpu,
        decoded: decoded.gpu,
        material: mat.gpu.cast(),
        count: N as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }));
    let p = pipeline(&codec::MODULE, "round_trip");
    let render = d.queue(orr_gpu::QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.dispatch(
        &p,
        root.gpu.cast::<codec::Root>(),
        [N.div_ceil(64) as u32, 1, 1],
    );
    list.barrier(
        orr_gpu::Stages::COMPUTE,
        orr_gpu::Stages::HOST,
        orr_gpu::Hazards::empty(),
    );
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
            1e-6,
            "specular weight layout",
            i,
        );
        close(cpu.ior, g.ior, 1e-5, "IOR layout", i);
        assert_eq!(cpu.flags, g.flags, "case {i} flags layout");
        close3(
            arr(cpu.specular_color),
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
        close(cpu.occlusion, g.occlusion, 1e-6, "occlusion layout", i);

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
        let m = Vec3::from(c.emission).max_element();
        close3(g.emission, c.emission, m / 512.0 + 1e-30, "emission", i);
        close(g.metalness, c.metalness, step8, "metalness", i);
        close(
            g.specular_weight,
            c.specular_weight,
            step8,
            "specular weight",
            i,
        );
        let r = |n: f32| (n - 1.0) / (n + 1.0);
        close(r(g.ior), r(c.ior), 0.5 / 510.0 + 1e-6, "IOR", i);
        close3(
            srgb3(g.specular_color),
            srgb3(c.specular_color),
            0.5 / 127.0 + 1e-6,
            "specular color",
            i,
        );
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
        close(g.occlusion, c.occlusion, step8, "occlusion", i);
        assert_eq!(g.flags & 0x62, c.flags, "case {i} unlit and origin policy");

        let thin = c.bsdf_flags & BSDF_THIN_WALLED != 0;
        assert_eq!(g.flags & THIN_TWO_SIDED != 0, thin, "case {i} thin");
        assert_eq!(
            g.bsdf_flags & BSDF_THIN_WALLED != 0,
            thin,
            "case {i} thin BSDF"
        );
        let sss = c.subsurface_weight > 0.0;
        let lobes = c.coat_weight > 0.0 || c.fuzz_weight > 0.0;
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
        assert_eq!(
            g.flags & LOBE_COLORS != 0,
            !sss && lobes,
            "case {i} lobe colors"
        );
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
            0.5 / 510.0 + 1e-6,
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
        if lobes && !sss {
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
        } else {
            assert_eq!(
                g.coat_color, material.coat_color,
                "case {i} coat color from the material"
            );
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
        assert!(
            g.bsdf_flags & 8 != 0,
            "case {i} thin film from the material"
        );
    }
    assert_clean();
}

// --- Cooked scenes -----------------------------------------------------------------------

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-scene-gbuffer-{}", std::process::id()));
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
                quality: Quality::Fast,
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
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                texture("tex/albedo", albedo, Usage::Color),
                texture("tex/normal", bumps, Usage::Normal),
                texture("tex/holes", holes, Usage::Linear1),
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
            ],
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

/// Threads lent to the loading side until dropped.
struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.loading.stop_workers();
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}

struct World {
    scenes: Scenes,
    mount: MountId,
    _workers: Workers,
}

fn time(frame: u32) -> FrameTime {
    FrameTime {
        absolute: f64::from(frame) / 60.0,
        delta: 1.0 / 60.0,
    }
}

impl World {
    fn new(kernels: &'static KernelSet) -> Self {
        let mut scenes = Scenes::new(
            device(),
            &RendererDesc {
                kernels,
                ..RendererDesc::default()
            },
        )
        .unwrap();
        let loading = scenes.loading();
        let threads = (0..2)
            .map(|_| {
                let l = Arc::clone(&loading);
                std::thread::spawn(move || l.run_worker())
            })
            .collect();
        let workers = Workers { loading, threads };
        let mount = scenes.mount(PackSource::Memory(Arc::clone(pack())));
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
            _workers: workers,
        }
    }

    fn model(&self, name: &str) -> ModelId {
        self.scenes
            .model(self.mount, AssetId::from_name(name))
            .unwrap()
    }
}

fn affine(m: Mat3, t: Vec3) -> Affine3 {
    let c = m.to_cols_array_2d();
    Affine3 {
        rows: [
            [c[0][0], c[1][0], c[2][0], t.x],
            [c[0][1], c[1][1], c[2][1], t.y],
            [c[0][2], c[1][2], c[2][2], t.z],
        ],
    }
}

/// World from view for a camera at `eye` looking at `target`, y up.
fn look(eye: Vec3, target: Vec3) -> Quat {
    let back = (eye - target).normalize();
    let right = Vec3::Y.cross(back).normalize();
    Quat::from_mat3(&Mat3::from_cols(right, back.cross(right), back))
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
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: copies.len() as u32,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    for (i, (c, model)) in copies.iter().zip(models).enumerate() {
        s.transforms(group, i as u32, 1).unwrap()[0] = affine(c.linear, c.at);
        let range = s
            .add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: group,
                first: i as u32,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 100.0,
                animation_range: 0.0,
                user_tag: i as u32,
            })
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
        count: copies.len() as u32,
    }
}

/// What one rendered view reads back.
struct Frame {
    view: RasterView,
    a: Vec<[u32; 4]>,
    b: Vec<[u32; 2]>,
    x: Option<Vec<[u32; 4]>>,
    depth: Vec<f32>,
    motion: Vec<[f32; 2]>,
    /// Instance slot + 1 per pixel, 0 where nothing was drawn.
    ids: Vec<u32>,
    counts: Vec<u32>,
}

fn f16(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let e = i32::from((bits >> 10) & 31);
    let m = f32::from(bits & 1023);
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

fn texels<T: bytemuck::Pod>(t: &Texture) -> Vec<T> {
    let bytes = read_texture(t, &TextureRegion::whole_mip(t.desc(), 0));
    bytemuck::pod_collect_to_vec(&bytes)
}

fn read_back(view: &RasterView) -> Frame {
    let pixels = u64::from(EXTENT[0] * EXTENT[1]);
    let vis: Vec<u64> = read_buffer(view.targets.visibility, pixels);
    let counts = read_buffer(view.counts, 3);
    let clusters: Vec<VisibleCluster> =
        read_buffer(view.targets.clusters, u64::from(counts[0].max(1)));
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
    let motion: Vec<[u16; 2]> = texels(&view.motion);
    Frame {
        view: view.clone(),
        a: texels(&view.gbuf_a),
        b: texels(&view.gbuf_b),
        x: view.gbuf_x.as_ref().map(|x| texels(x)),
        depth: texels(&view.depth),
        motion: motion.iter().map(|m| m.map(f16)).collect(),
        ids,
        counts,
    }
}

/// Renders `placed` from a camera at `eye` looking at `target`, with copy transforms
/// `moved` written this frame, and reads the view back.
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
    if !moved.is_empty() {
        let mut s = f.scene(placed.scene).unwrap();
        for (i, c) in moved.iter().enumerate() {
            s.transforms(placed.group, i as u32, 1).unwrap()[0] = affine(c.linear, c.at);
        }
        assert_eq!(moved.len() as u32, placed.count);
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
            jitter: Vec2::new(0.25, -0.125),
            reset_history: reset,
            mip_bias: 0.0,
            target: ViewTarget::AfterRr,
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
    let mut w = World::new(&KernelSet {
        modules: &[],
        surfaces: 0,
    });
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
    assert_eq!(g.counts[1..], [0, 0], "nothing dropped");
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
            Box::new(|i| [byte(x[i][2], 0), byte(x[i][2], 8), byte(x[i][3], 8)]),
        ),
    ];
    for (name, f) in &channels {
        check(&reference(name), &image(f), flip);
    }
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
    let mut w = World::new(&KernelSet {
        modules: &[],
        surfaces: 0,
    });
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
    let mut w = World::new(&KernelSet {
        modules: &[],
        surfaces: 0,
    });
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
    let mut compared = 0;
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
    let mut w = World::new(&KernelSet {
        modules: &[],
        surfaces: 0,
    });
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
    let mut w = World::new(&KernelSet {
        modules: &[],
        surfaces: 0,
    });
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
        scene: g.view.scene,
        view: g.view.view,
        gbuffer: g.view.gbuffer,
        out: out.gpu,
    };
    let p = pipeline(&surface_at::MODULE, "main");
    let render = d.queue(orr_gpu::QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.dispatch(&p, root.gpu, [n.div_ceil(64) as u32, 1, 1]);
    list.barrier(
        orr_gpu::Stages::COMPUTE,
        orr_gpu::Stages::HOST,
        orr_gpu::Hazards::empty(),
    );
    wait(render.submit(vec![list], &[]).unwrap());
    let vis: Vec<u64> = read_buffer(g.view.targets.visibility, n);
    let clusters: Vec<VisibleCluster> =
        read_buffer(g.view.targets.clusters, u64::from(g.counts[0]));
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
    let mut w = World::new(&KernelSet {
        modules: &[],
        surfaces: 0,
    });
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
    f.scene(placed.scene)
        .unwrap()
        .copy_variants(placed.ranges[0], 0, 1)
        .unwrap()[0] = CopyVariant {
        pose: 0,
        material_set: 0,
        section_mask: 1 << 1,
    };
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
