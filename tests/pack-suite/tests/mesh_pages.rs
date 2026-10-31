//! `orr_pack.mesh` on the GPU reads what `orr_pack::mesh::PageView` reads from the same
//! `decode_page` output, and `orr_pack.page_codec` on the loading queue writes what
//! `decode_page` writes and rejects what it rejects: synthetic pages at every attribute
//! layout with micromap indices, every page of a cooked mesh, damaged pages, and with
//! `ORR_COOKED_SET` every page of a set of cooked archives.

use std::path::PathBuf;

use glam::{Vec2, Vec3, Vec4};
use orr_cook::{ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshParts, MeshSettings};
use orr_gpu::MemoryKind;
use orr_pack::material::Material;
use orr_pack::mesh::{self, Cluster, MeshHeader, MeshInfo, PageHeader, PageView};
use orr_pack::{Archive, AssetId, Span, Target};
use orr_scene::testing::{assert_clean, device};
use pack_suite::shaders::page::{self, TriangleOut, VertexOut};
use pack_suite::{UNWRITTEN, cooked_set_pages, decode_stored, dispatch, read, stored_pages};

/// Slots per cluster in the kernel's outputs: the most vertices and triangles a cluster has.
const SLOTS: usize = 256;

/// Largest differences between the GPU and CPU float decodes. Integer values (positions,
/// grid offsets, joints, corners, micromap indices, records) must match exactly; unit
/// vectors, uvs and weights go through divisions and a square root, which the GPU rounds
/// differently. Uv differences are relative to the cluster's `|uv_offset| + |uv_scale|`.
#[derive(Debug, Default)]
struct Spread {
    unit: f32,
    uv: f32,
    weight: f32,
}

impl Spread {
    /// Vulkan allows 2.5 ulp on a division and inherits the square root from
    /// `inversesqrt`; on 621.01 unit vectors differ by up to 2.5 ulp at 1.0, uvs by 1 ulp and
    /// weights by half an ulp.
    fn assert_within(&self) {
        let bounds = Spread {
            unit: 2f32.powi(-21),
            uv: 2f32.powi(-22),
            weight: 2f32.powi(-23),
        };
        assert!(
            self.unit <= bounds.unit && self.uv <= bounds.uv && self.weight <= bounds.weight,
            "{self:?} against {bounds:?}"
        );
    }

    fn add(max: &mut f32, gpu: &[f32], cpu: &[f32], scale: f32) {
        for (g, c) in gpu.iter().zip(cpu) {
            *max = max.max((g - c).abs() / scale);
        }
    }
}

/// Runs the loaders over every cluster of `page`, checks the exact values against the CPU
/// and adds the float differences to `spread`.
fn check(page: &[u8], mesh: &MeshHeader, spread: &mut Spread) {
    let d = device();
    let view = PageView::new(page, mesh).unwrap();
    let n = view.cluster_count() as usize;
    let mut buf = d
        .gpu_malloc::<u8>(MemoryKind::Upload, page.len() as u64, "page")
        .unwrap();
    buf.as_mut_slice().unwrap().copy_from_slice(page);
    let header = d
        .gpu_malloc::<PageHeader>(MemoryKind::Readback, 1, "header")
        .unwrap();
    let clusters = d
        .gpu_malloc::<Cluster>(MemoryKind::Readback, n as u64, "clusters")
        .unwrap();
    let vertices = d
        .gpu_malloc::<VertexOut>(MemoryKind::Readback, (n * SLOTS) as u64, "vertices")
        .unwrap();
    let triangles = d
        .gpu_malloc::<TriangleOut>(MemoryKind::Readback, (n * SLOTS) as u64, "triangles")
        .unwrap();
    let mut root = d
        .gpu_malloc::<page::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = page::Root {
        page: buf.gpu,
        header: header.gpu,
        clusters: clusters.gpu,
        vertices: vertices.gpu,
        triangles: triangles.gpu,
        attributes: u32::from(mesh.attributes),
        exponent: i32::from(mesh.position_exponent),
    };
    dispatch(&page::MODULE, root.gpu, n as u32);

    assert_eq!(
        bytemuck::bytes_of(&header.as_slice().unwrap()[0]),
        &page[..size_of::<PageHeader>()]
    );
    let (clusters, vertices, triangles) = (
        clusters.as_slice().unwrap(),
        vertices.as_slice().unwrap(),
        triangles.as_slice().unwrap(),
    );
    for (ci, got) in clusters.iter().enumerate() {
        let c = view.cluster(ci as u32);
        assert_eq!(
            bytemuck::bytes_of(got),
            bytemuck::bytes_of(&c),
            "cluster {ci}"
        );
        let uv_scale = (0..2)
            .map(|a| c.uv_offset[a].abs() + c.uv_scale[a].abs())
            .fold(f32::MIN_POSITIVE, f32::max);
        for v in 0..u32::from(c.vertex_count) {
            let g = &vertices[ci * SLOTS + v as usize];
            let at = format!("cluster {ci} vertex {v}");
            let position = view.position(&c, v);
            assert_eq!(
                [0, 1, 2].map(|a| g.position[a].to_bits()),
                position.map(f32::to_bits),
                "{at}"
            );
            let grid = view.grid(&c, v);
            let local = [0, 1, 2].map(|a| (grid[a] - c.anchor[a]) as u32);
            assert_eq!(g.local[..3], local, "{at}");
            Spread::add(&mut spread.unit, &g.normal[..3], &view.normal(&c, v), 1.0);
            let t = view.tangent(&c, v).unwrap_or([1.0, 0.0, 0.0, 1.0]);
            Spread::add(&mut spread.unit, &g.tangent[..3], &t[..3], 1.0);
            assert_eq!(g.tangent[3], t[3], "{at}: bitangent sign");
            Spread::add(&mut spread.uv, &g.uv[..2], &view.uv(&c, v), uv_scale);
            let joints = view.joints(&c, v).unwrap_or_default().map(u32::from);
            assert_eq!(g.joints, joints, "{at}");
            let w = view.weights(&c, v).unwrap_or_default();
            Spread::add(&mut spread.weight, &g.weights, &w, 1.0);
        }
        for t in 0..u32::from(c.tri_count) {
            let g = triangles[ci * SLOTS + t as usize].corners;
            let corners = view.triangle(&c, t).map(u32::from);
            let omm = view.omm_index(&c, t).map_or(0, |i| i32::from(i) as u32);
            assert_eq!(
                g,
                [corners[0], corners[1], corners[2], omm],
                "cluster {ci} triangle {t}"
            );
        }
    }
    assert_clean();
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }

    fn float(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() as f32 / u32::MAX as f32)
    }
}

/// A resident page of `clusters` clusters with random contents: anchors over the whole
/// 24-bit grid, any attribute bits, and on some clusters with alpha-tested triangles
/// micromap indices, special ones included, into a micromap region of `entries` entries.
fn synthetic(rng: &mut Rng, clusters: usize, attr_stride: usize, entries: u32) -> Vec<u8> {
    let counts: Vec<(usize, usize)> = (0..clusters)
        .map(|_| (3 + rng.below(SLOTS - 2), 1 + rng.below(255)))
        .collect();
    let slots: usize = counts.iter().map(|c| c.0.next_multiple_of(2)).sum();
    let tris: usize = counts.iter().map(|c| (c.1 * 3).next_multiple_of(4)).sum();
    let positions = size_of::<PageHeader>() + clusters * size_of::<Cluster>();
    let attrs = positions + slots * 6;
    let tris_offset = attrs + slots * attr_stride;
    let omm_offset = (tris_offset + tris).next_multiple_of(256);
    let mut p = vec![0u8; omm_offset];
    // The entry table, then each cluster's indices, then the entries' states.
    let mut omm = vec![0u8; entries as usize * 8];
    let (mut v0, mut t0) = (0, 0);
    for (i, &(vn, tn)) in counts.iter().enumerate() {
        let alpha_first = rng.below(tn + 1);
        let omm_first = if entries > 0 && alpha_first < tn && rng.below(3) != 0 {
            let first = omm.len() as u32;
            for _ in alpha_first..tn {
                let index = rng.below(entries as usize + 4) as i16 - 4;
                omm.extend(index.to_le_bytes());
            }
            omm.resize(omm.len().next_multiple_of(4), 0);
            first
        } else {
            Cluster::NO_OMM
        };
        let reach = (1 << 23) - 65536;
        let c = Cluster {
            vertex_first: v0 as u32,
            tri_first: t0 as u32,
            vertex_count: vn as u16,
            tri_count: tn as u8,
            alpha_first: alpha_first as u8,
            anchor: [0; 3].map(|_| rng.below(2 * reach) as i32 - reach as i32),
            uv_offset: [rng.float(-4.0, 4.0), rng.float(-4.0, 4.0)],
            uv_scale: [rng.float(0.01, 8.0), rng.float(0.01, 8.0)],
            omm_first,
            section: i as u8,
            area: rng.float(0.0, 10.0),
            ..bytemuck::Zeroable::zeroed()
        };
        let at = size_of::<PageHeader>() + i * size_of::<Cluster>();
        p[at..at + size_of::<Cluster>()].copy_from_slice(bytemuck::bytes_of(&c));
        for v in 0..vn {
            let at = positions + (v0 + v) * 6;
            for b in &mut p[at..at + 6] {
                *b = rng.next() as u8;
            }
            let at = attrs + (v0 + v) * attr_stride;
            for b in &mut p[at..at + attr_stride] {
                *b = rng.next() as u8;
            }
        }
        for t in 0..tn * 3 {
            p[tris_offset + t0 + t] = rng.below(vn) as u8;
        }
        v0 += vn.next_multiple_of(2);
        t0 += (tn * 3).next_multiple_of(4);
    }
    let mut omm_usage = [0; 13];
    for e in 0..entries as usize {
        let level = rng.below(4) as u16;
        omm_usage[usize::from(level)] += 1;
        // data_offset u32, level u16, format u16 (4-state), then 2 bits per micro-triangle.
        let entry = [
            (omm.len() as u32).to_le_bytes(),
            (u32::from(level) | 2 << 16).to_le_bytes(),
        ];
        omm[e * 8..e * 8 + 8].copy_from_slice(entry.as_flattened());
        let states = (1usize << (2 * level)).div_ceil(4);
        omm.extend((0..states).map(|_| rng.next() as u8));
    }
    p.extend(omm);
    let h = PageHeader {
        cluster_count: clusters as u32,
        attr_stride: attr_stride as u32,
        positions_offset: positions as u32,
        attrs_offset: attrs as u32,
        tris_offset: tris_offset as u32,
        omm_offset: omm_offset as u32,
        len: p.len() as u32,
        omm_usage,
        ..bytemuck::Zeroable::zeroed()
    };
    p[..size_of::<PageHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
    p
}

#[test]
fn synthetic_pages_load_alike_at_every_attribute_layout() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut spread = Spread::default();
    for attributes in [
        0,
        MeshHeader::TANGENT,
        MeshHeader::SKIN,
        MeshHeader::TANGENT | MeshHeader::SKIN,
    ] {
        for exponent in [-11, 3] {
            let mut mesh: MeshHeader = bytemuck::Zeroable::zeroed();
            mesh.attributes = attributes;
            mesh.position_exponent = exponent;
            let entries = if exponent < 0 { 37 } else { 0 };
            let p = synthetic(&mut rng, 24, mesh.attr_stride() as usize, entries);
            let mut resident = vec![0; p.len()];
            mesh::decode_page(&mesh::encode_page(&p).unwrap(), &mut resident).unwrap();
            check(&resident, &mesh, &mut spread);
        }
    }
    spread.assert_within();
}

/// A wavy 4 m grid with tangents, two sections and rigid parts by column band.
fn wavy(n: u32) -> MeshInput {
    let mut m = MeshInput::default();
    let mut values = Vec::new();
    let mut tangents = Vec::new();
    let mut uvs = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            let (x, z) = ((u - 0.5) * 4.0, (v - 0.5) * 4.0);
            m.positions
                .push(Vec3::new(x, 0.15 * (x * 2.3).sin() * (z * 1.7).cos(), z));
            m.normals
                .push(Vec3::new(-0.3 * (x * 2.3).cos(), 1.0, 0.2 * z).normalize());
            tangents.push(Vec4::new(
                1.0,
                0.1 * x,
                0.0,
                if i % 2 == 0 { 1.0 } else { -1.0 },
            ));
            uvs.push(Vec2::new(u * 3.0, v * 2.0 - 1.0));
            values.push((i * 4 / (n + 1)) as u16);
        }
    }
    m.tangents = Some(
        tangents
            .into_iter()
            .map(|t| t.truncate().normalize().extend(t.w))
            .collect(),
    );
    m.uvs = Some(uvs);
    let w = n + 1;
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            let quad = [c, c + w, c + 1, c + 1, c + w, c + w + 1];
            if j < n / 2 { &mut lower } else { &mut upper }.extend(quad);
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
    m.parts = Some(MeshParts {
        values,
        count: 4,
        by_section: false,
        pivots: None,
    });
    m
}

/// The wavy grid cooked at two discrete levels and a merged one, as mesh "m".
fn cooked_wavy() -> Archive {
    let input = wavy(64);
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "m".into(),
                source: orr_cook::Source::Mesh {
                    materials: vec![Material::default(); input.sections.len()],
                    mesh: input,
                    settings: MeshSettings {
                        lod: LodMode::Discrete {
                            errors: vec![0.002, 0.02],
                        },
                        merge_level: Some(2),
                        ..Default::default()
                    },
                },
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    Archive::from_memory(bytes).unwrap()
}

#[test]
fn cooked_mesh_pages_load_alike() {
    let a = cooked_wavy();
    let e = a.find(AssetId::from_name("m/mesh")).unwrap();
    let blobs = a.blobs(e);
    let info = MeshInfo::parse(&read(&a, &blobs[0])).unwrap();
    assert_eq!(info.header.attributes, MeshHeader::TANGENT);
    assert!(info.pages.len() >= 3);
    let mut spread = Spread::default();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        mesh::decode_page(&read(&a, &blobs[p.blob as usize]), &mut page).unwrap();
        check(&page, &info.header, &mut spread);
    }
    spread.assert_within();
}

/// Decodes `stored` on the GPU and the CPU: a page the CPU rejects sets status bits, and an
/// accepted one comes out byte for byte alike with nothing written past its length. Returns
/// how many pages the CPU accepted.
fn check_decode(stored: &[Vec<u8>]) -> usize {
    let gpu = decode_stored(stored);
    let mut accepted = 0;
    for (i, ((p, status), got)) in stored
        .iter()
        .zip(&gpu.status)
        .zip(&gpu.resident)
        .enumerate()
    {
        let mut cpu = vec![UNWRITTEN; mesh::PAGE_MAX];
        match mesh::decode_page(p, &mut cpu) {
            Ok(layout) => {
                assert_eq!(*status, 0, "page {i}: the CPU accepts it");
                let len = layout.len as usize;
                let wrong = (0..len).filter(|&b| got[b] != cpu[b]).count();
                assert_eq!(wrong, 0, "page {i}: {wrong} of {len} bytes differ");
                assert!(
                    got[len..].iter().all(|&b| b == UNWRITTEN),
                    "page {i}: written past its length"
                );
                accepted += 1;
            }
            Err(e) => assert_ne!(*status, 0, "page {i}: the CPU rejects it ({e})"),
        }
    }
    assert_clean();
    accepted
}

#[test]
fn synthetic_pages_decode_alike_on_the_gpu() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut stored = Vec::new();
    for attributes in [
        0,
        MeshHeader::TANGENT,
        MeshHeader::SKIN,
        MeshHeader::TANGENT | MeshHeader::SKIN,
    ] {
        for entries in [0, 23] {
            let stride = mesh::attr_stride(attributes) as usize;
            let p = synthetic(&mut rng, 24, stride, entries);
            stored.push(mesh::encode_page(&p).unwrap());
        }
    }
    assert_eq!(check_decode(&stored), stored.len());
}

#[test]
fn cooked_mesh_pages_decode_alike_on_the_gpu() {
    let a = cooked_wavy();
    let stored = stored_pages(&a, a.find(AssetId::from_name("m/mesh")).unwrap());
    assert_eq!(check_decode(&stored), stored.len());
}

/// Single-bit changes anywhere in stored pages, most in the stream table and the streams,
/// some with the page cut short: the GPU rejects what the CPU rejects and decodes the rest
/// alike.
#[test]
fn damaged_pages_are_rejected_alike_on_the_gpu() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut stored = Vec::new();
    for attributes in [0, MeshHeader::TANGENT | MeshHeader::SKIN] {
        let stride = mesh::attr_stride(attributes) as usize;
        let p = synthetic(&mut rng, 6, stride, 5);
        let good = mesh::encode_page(&p).unwrap();
        let h: PageHeader = bytemuck::pod_read_unaligned(&good[..size_of::<PageHeader>()]);
        let streams = h.positions_offset as usize;
        for k in 0..192 {
            let mut bad = good.clone();
            let at = if k % 8 == 0 {
                rng.below(streams)
            } else {
                streams + rng.below(good.len() - streams)
            };
            bad[at] ^= 1 << rng.below(8);
            if k % 16 == 1 {
                bad.truncate(good.len() - 1 - rng.below(8));
            }
            stored.push(bad);
        }
    }
    let accepted = check_decode(&stored);
    assert!(
        accepted > 0 && accepted < stored.len(),
        "{accepted} of {} accepted",
        stored.len()
    );
}

/// `ORR_COOKED_SET=<dir>`: every mesh page of every `.orr` archive under `dir` decodes on
/// the GPU byte for byte as on the CPU.
#[test]
#[ignore = "needs ORR_COOKED_SET"]
fn cooked_set_pages_decode_alike_on_the_gpu() {
    let dir = PathBuf::from(std::env::var("ORR_COOKED_SET").expect("ORR_COOKED_SET"));
    let set = cooked_set_pages(&dir);
    assert!(
        set.iter().any(|(_, pages)| !pages.is_empty()),
        "no mesh pages under {}",
        dir.display()
    );
    for (path, pages) in &set {
        for chunk in pages.chunks(64) {
            assert_eq!(check_decode(chunk), chunk.len(), "{}", path.display());
        }
    }
}
