//! Mixed cuts of cooked cluster LOD DAGs drawn and traced on the GPU from their pages: a
//! wavy two-section grid and a statically displaced plane with uv seams. The mesh-shader
//! pass covers every pixel of the surface once with one winding, and every ray over it hits
//! a selected cluster's CLAS, built with the mesh's truncation, at the positions the page
//! stores.

use glam::{Vec2, Vec3};
use gpu_suite::clusters::{self, Builder, data_flags};
use gpu_suite::{assert_clean, device, readback, render, run, upload};
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MeshInput, MeshSettings, Source,
    StaticDisplacement,
};
use orr_gpu::rtas::consts::GEOMETRY_OPAQUE;
use orr_gpu::{
    ClassicInstance, GpuPtr, Hazards, InstanceFlags, MemoryKind, MeshPipeline, MeshPipelineDesc,
    Rendering, ShaderStage, SpecData, Stages,
};
use orr_pack::material::Material;
use orr_pack::mesh::{self, Cluster, MeshInfo, PageView};
use orr_pack::{Archive, AssetId, Span, Sphere, Target};
use pack_suite::read;
use pack_suite::shaders::crack;

/// Raster pixels per side.
const W: u32 = 1024;
/// Page offsets in the shared buffer.
const PAGE_ALIGN: usize = 256;

/// A cooked DAG mesh with its pages decoded.
struct Mesh {
    info: MeshInfo,
    pages: Vec<Vec<u8>>,
}

impl Mesh {
    fn load(a: &Archive, name: &str) -> Self {
        let e = a.find(AssetId::from_name(name)).unwrap();
        let blobs = a.blobs(e);
        let info = MeshInfo::parse(&read(a, &blobs[0])).unwrap();
        let pages = info
            .pages
            .iter()
            .map(|p| {
                let mut page = vec![0; p.resident_len as usize];
                mesh::decode_page(&read(a, &blobs[p.blob as usize]), &mut page).unwrap();
                page
            })
            .collect();
        Self { info, pages }
    }

    fn view(&self, page: usize) -> PageView<'_> {
        PageView::new(&self.pages[page], &self.info.header).unwrap()
    }

    /// (page, cluster) of every cluster `cut` selects.
    fn select(&self, cut: Cut) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        for p in 0..self.pages.len() {
            let v = self.view(p);
            for c in 0..v.cluster_count() {
                if cut.selects(&v.cluster(c)) {
                    out.push((p, c));
                }
            }
        }
        out
    }

    /// Whether a cut's clusters come from more than one level.
    fn mixed(&self, selected: &[(usize, u32)]) -> bool {
        let error = |&(p, c): &(usize, u32)| self.view(p).cluster(c).self_error.to_bits();
        selected.iter().any(|s| error(s) != error(&selected[0]))
    }
}

#[derive(Copy, Clone, Debug)]
enum Cut {
    Uniform(f32),
    View(Vec3, f32),
}

impl Cut {
    fn over(self, sphere: &Sphere, error: f32) -> bool {
        match self {
            Cut::Uniform(t) => error > t,
            Cut::View(eye, t) => {
                let d = (Vec3::from(sphere.center) - eye).length() - sphere.radius;
                error == f32::INFINITY || error / d.max(0.01) > t
            }
        }
    }

    fn selects(self, c: &Cluster) -> bool {
        !self.over(&c.self_sphere, c.self_error) && self.over(&c.parent_sphere, c.parent_error)
    }
}

/// The square of xz the checks cover: corner and side, and the inset from its edges that
/// simplification may pull the rim back by.
#[derive(Copy, Clone)]
struct Square {
    lo: Vec2,
    size: f32,
    inset: f32,
}

impl Square {
    fn inside(&self, p: Vec2) -> bool {
        let q = p - self.lo;
        q.min_element() >= self.inset && q.max_element() <= self.size - self.inset
    }
}

/// Every page of a mesh in one upload buffer.
struct Pages {
    buf: orr_gpu::Allocation<u8>,
    offsets: Vec<usize>,
}

impl Pages {
    fn new(m: &Mesh) -> Self {
        let mut bytes = Vec::new();
        let mut offsets = Vec::new();
        for p in &m.pages {
            offsets.push(bytes.len());
            bytes.extend_from_slice(p);
            bytes.resize(bytes.len().next_multiple_of(PAGE_ALIGN), 0);
        }
        Self {
            buf: upload(&bytes, "pages"),
            offsets,
        }
    }

    fn page(&self, p: usize) -> GpuPtr<u8> {
        self.buf.gpu.offset(self.offsets[p] as u64)
    }
}

/// Signed coverage per pixel of the selected clusters drawn from their pages.
fn raster(
    m: &Mesh,
    p: &MeshPipeline,
    pages: &Pages,
    selected: &[(usize, u32)],
    sq: Square,
) -> Vec<i32> {
    let d = device();
    let draws: Vec<crack::Draw> = selected
        .iter()
        .map(|&(page, cluster)| crack::Draw {
            page: pages.page(page),
            cluster,
            pad: 0,
        })
        .collect();
    let draws = upload(&draws, "draws");
    let px = u64::from(W * W);
    let zeros = upload(&vec![0i32; px as usize], "zeros");
    let counts = d
        .gpu_malloc::<i32>(MemoryKind::DeviceLocal, px, "counts")
        .unwrap();
    let out = readback::<i32>(px, "counts out");
    let root = upload(
        &[crack::Root {
            draws: draws.gpu,
            counts: counts.gpu,
            lo_x: sq.lo.x,
            lo_z: sq.lo.y,
            scale: 2.0 / sq.size,
            exponent: i32::from(m.info.header.position_exponent),
            width: W,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }],
        "crack root",
    );
    let mut list = render().command_list().unwrap();
    list.copy(counts.gpu.cast(), zeros.gpu.cast(), px * 4);
    list.barrier(Stages::TRANSFER, Stages::PIXEL, Hazards::empty());
    list.begin_rendering(&Rendering {
        extent: [W, W],
        colors: &[],
        depth: None,
    });
    list.draw_mesh_tasks(p, root.gpu, [selected.len() as u32, 1, 1]);
    list.end_rendering();
    list.barrier(Stages::PIXEL, Stages::TRANSFER, Hazards::empty());
    list.copy(out.gpu.cast(), counts.gpu.cast(), px * 4);
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run(list);
    out.as_slice().unwrap().to_vec()
}

/// Pixels inside the square not covered exactly once with the surface's winding, taken
/// from the first pixel inside.
fn raster_wrong(counts: &[i32], sq: Square) -> (usize, usize) {
    let inside: Vec<i32> = (0..W * W)
        .filter(|&i| {
            let px = Vec2::new((i % W) as f32, (i / W) as f32);
            sq.inside(sq.lo + (px + 0.5) / W as f32 * sq.size)
        })
        .map(|i| counts[i as usize])
        .collect();
    let sign = inside.first().map_or(0, |c| c.signum());
    let wrong = inside.iter().filter(|&&c| c != sign || sign == 0).count();
    (inside.len(), wrong)
}

/// Traces the selected clusters' CLAS over the square; returns the rays inside it that miss
/// or hit other than a page triangle's positions.
fn trace_wrong(m: &Mesh, selected: &[(usize, u32)], sq: Square) -> (usize, usize) {
    let truncate = u32::from(m.info.header.clas_truncate_bits);
    let mut tris: Vec<Vec<[[f32; 3]; 3]>> = Vec::new();
    let input: Vec<clusters::Cluster> = selected
        .iter()
        .enumerate()
        .map(|(id, &(p, ci))| {
            let v = m.view(p);
            let c = v.cluster(ci);
            let verts: Vec<[f32; 3]> = (0..u32::from(c.vertex_count))
                .map(|i| v.position(&c, i))
                .collect();
            let t: Vec<[u32; 3]> = (0..u32::from(c.tri_count))
                .map(|t| v.triangle(&c, t).map(u32::from))
                .collect();
            tris.push(t.iter().map(|t| t.map(|i| verts[i as usize])).collect());
            clusters::Cluster {
                id: id as u32,
                verts,
                tris: t,
                index_bytes: 1,
                truncate,
                base_geom: 0,
                geom_flags: GEOMETRY_OPAQUE,
                tri_gflags: None,
                omm_index: None,
            }
        })
        .collect();
    // Mesh xz onto the unit square the rays cover, y along the rays.
    let s = 1.0 / sq.size;
    let transform = [
        [s, 0.0, 0.0, -sq.lo.x * s],
        [0.0, 0.0, s, -sq.lo.y * s],
        [0.0, 1.0, 0.0, 0.0],
    ];
    let mut b = Builder::new();
    let clas = b.build_clas(&input, data_flags(), 0, None, None);
    let all: Vec<u32> = (0..input.len() as u32).collect();
    let blas = b.build_blas(clas.addrs.gpu, &[all], None);
    let instance = ClassicInstance::new(
        transform,
        0,
        0xff,
        InstanceFlags::CULL_DISABLE,
        orr_gpu::GpuAddress(0),
    );
    let tlas = b.build_tlas(blas.addrs.gpu, &[instance]);
    let out = b.trace(tlas, 0);
    b.flush();
    let out = out.as_slice().unwrap();
    let (mut inside, mut wrong) = (0, 0);
    for y in 0..clusters::H {
        for x in 0..clusters::W {
            let uv = (Vec2::new(x as f32, y as f32) + 0.5)
                / Vec2::new(clusters::W as f32, clusters::H as f32);
            if !sq.inside(sq.lo + uv * sq.size) {
                continue;
            }
            inside += 1;
            let r = &out[(y * clusters::W + x) as usize * clusters::REC..][..clusters::REC];
            assert_eq!(r[19], 0x600d);
            let hit = r[0] == 1
                && tris
                    .get(r[3] as usize)
                    .and_then(|t| t.get(r[4] as usize))
                    .is_some_and(|t| (0..9).all(|k| r[10 + k] == t[k / 3][k % 3].to_bits()));
            if !hit {
                wrong += 1;
            }
        }
    }
    (inside, wrong)
}

/// Draws and traces every cut; at least one mixes levels.
fn check_cuts(m: &Mesh, cuts: &[Cut], sq: Square) {
    let stage = |entry| ShaderStage::new(&crack::MODULE, entry, SpecData::default()).unwrap();
    let pipeline = device()
        .create_mesh_pipeline(
            &MeshPipelineDesc {
                pixel: Some(stage("pixel_main")),
                ..MeshPipelineDesc::new(stage("mesh_main"))
            },
            "crack",
        )
        .unwrap();
    let pages = Pages::new(m);
    let mut mixed = false;
    for &cut in cuts {
        let selected = m.select(cut);
        assert!(!selected.is_empty(), "{cut:?}");
        mixed |= m.mixed(&selected);
        let counts = raster(m, &pipeline, &pages, &selected, sq);
        let (inside, wrong) = raster_wrong(&counts, sq);
        assert!(inside > 0);
        assert_eq!(wrong, 0, "raster {cut:?}: {wrong} of {inside} pixels");
        let (inside, wrong) = trace_wrong(m, &selected, sq);
        assert!(inside > 0);
        assert_eq!(wrong, 0, "trace {cut:?}: {wrong} of {inside} rays");
    }
    assert!(mixed, "a cut mixes levels");
    assert_clean();
}

fn cook(name: &str, source: Source) -> Archive {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: name.into(),
                source,
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    Archive::from_memory(bytes).unwrap()
}

fn dag() -> MeshSettings {
    MeshSettings {
        lod: LodMode::Dag,
        ..MeshSettings::default()
    }
}

/// A wavy grid of `n` x `n` quads over 4 m in xz, in two sections split at z = 0.
fn wavy(n: u32) -> MeshInput {
    let mut m = MeshInput::default();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            let (x, z) = ((u - 0.5) * 4.0, (v - 0.5) * 4.0);
            m.positions
                .push(Vec3::new(x, 0.15 * (x * 2.3).sin() * (z * 1.7).cos(), z));
            m.normals.push(Vec3::Y);
        }
    }
    m.uvs = Some(
        (0..=n)
            .flat_map(|j| (0..=n).map(move |i| Vec2::new(i as f32, j as f32) / n as f32))
            .collect(),
    );
    let w = n + 1;
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            let half = if j >= n / 2 { &mut upper } else { &mut lower };
            half.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
        }
    }
    m.sections = vec![
        Span {
            first: 0,
            count: lower.len() as u32,
        },
        Span {
            first: lower.len() as u32,
            count: upper.len() as u32,
        },
    ];
    m.indices = [lower, upper].concat();
    m
}

#[test]
fn mixed_cuts_of_a_two_section_grid_have_no_cracks() {
    let a = cook(
        "m",
        Source::Mesh {
            mesh: wavy(96),
            materials: vec![Material::default(); 2],
            settings: dag(),
        },
    );
    let m = Mesh::load(&a, "m/mesh");
    let mut cuts = vec![Cut::Uniform(0.0), Cut::Uniform(0.01)];
    for eye in [
        Vec3::new(-2.5, 0.6, -2.5),
        Vec3::new(0.0, 0.4, 0.0),
        Vec3::new(3.0, 1.5, 1.0),
    ] {
        for t in [0.002, 0.01] {
            cuts.push(Cut::View(eye, t));
        }
    }
    let sq = Square {
        lo: Vec2::splat(-2.0),
        size: 4.0,
        inset: 0.02,
    };
    check_cuts(&m, &cuts, sq);
}

/// A GLB of a plane in two uv islands that meet at x = 0.5 with different uvs there,
/// displaced by `png`.
fn displaced_plane(png: &[u8]) -> Vec<u8> {
    let (mut pos, mut uv, mut idx) = (Vec::<[f32; 3]>::new(), Vec::<[f32; 2]>::new(), Vec::new());
    for island in 0..2u32 {
        let base = pos.len() as u32;
        let n = 6;
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                pos.push([0.5 * (island as f32 + u), 0.0, v]);
                uv.push([0.45 * u + 0.55 * island as f32, v]);
            }
        }
        let w = n + 1;
        for j in 0..n {
            for i in 0..n {
                let c = base + j * w + i;
                idx.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
            }
        }
    }
    let normals = vec![[0.0f32, 1.0, 0.0]; pos.len()];
    let mut bin = Vec::new();
    let mut views = Vec::new();
    for bytes in [
        bytemuck::cast_slice::<_, u8>(&pos),
        bytemuck::cast_slice(&normals),
        bytemuck::cast_slice(&uv),
        bytemuck::cast_slice(&idx),
        png,
    ] {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
    }
    let (v, i) = (pos.len(), idx.len());
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":0}}]}}],
"extensionsUsed":["KHR_materials_displacement"],
"materials":[{{"extensions":{{"KHR_materials_displacement":{{"displacementGeometryFactor":0.04,"displacementGeometryOffset":-0.01,"displacementGeometryTexture":{{"index":0}}}}}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":4,"mimeType":"image/png"}}],
"buffers":[{{"byteLength":{}}}],"bufferViews":[{}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":{v},"type":"VEC3","min":[0,0,0],"max":[1,0,1]}},
{{"bufferView":1,"componentType":5126,"count":{v},"type":"VEC3"}},
{{"bufferView":2,"componentType":5126,"count":{v},"type":"VEC2"}},
{{"bufferView":3,"componentType":5125,"count":{i},"type":"SCALAR"}}]}}"#,
        bin.len(),
        views.join(",")
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    glb
}

/// A 64 x 64 RGBA PNG of a smooth height pattern.
fn height_png() -> Vec<u8> {
    let (w, h) = (64, 64);
    let mut rgba = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = (128.0 + 100.0 * ((x as f32 * 0.15).sin() * (y as f32 * 0.1).cos())) as u8;
            rgba.extend([v, v, v, 255]);
        }
    }
    let mut out = Vec::new();
    let mut e = png::Encoder::new(&mut out, w, h);
    e.set_color(png::ColorType::Rgba);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&rgba).unwrap();
    out
}

#[test]
fn mixed_cuts_of_a_displaced_plane_with_uv_seams_have_no_cracks() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("crack-displaced");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("plane.glb");
    std::fs::write(&path, displaced_plane(&height_png())).unwrap();
    let settings = GltfSettings {
        mesh: MeshSettings {
            displace_static: Some(StaticDisplacement {
                max_edge: 0.01,
                target_error: 0.0005,
            }),
            ..dag()
        },
        ..Default::default()
    };
    let a = cook("g", Source::Gltf { path, settings });
    let m = Mesh::load(&a, "g/mesh");
    let mut cuts = vec![Cut::Uniform(0.0), Cut::Uniform(0.002)];
    for eye in [Vec3::new(0.0, 0.3, 0.0), Vec3::new(0.5, 0.2, 0.5)] {
        cuts.push(Cut::View(eye, 0.002));
    }
    let sq = Square {
        lo: Vec2::ZERO,
        size: 1.0,
        inset: 0.005,
    };
    check_cuts(&m, &cuts, sq);
}
