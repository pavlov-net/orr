use orr_pack::mesh::{self, Cluster, PageHeader};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// A resident page of `clusters` random clusters with `attr_stride`-byte attributes, high
/// halves when `highs`, and an OMM region of `omm` bytes.
fn page(rng: &mut Rng, clusters: usize, attr_stride: usize, highs: bool, omm: usize) -> Vec<u8> {
    let counts: Vec<(usize, usize)> = (0..clusters)
        .map(|_| (3 + rng.next() as usize % 254, 1 + rng.next() as usize % 128))
        .collect();
    let slots: usize = counts.iter().map(|c| c.0.next_multiple_of(2)).sum();
    let tris: usize = counts.iter().map(|c| (c.1 * 3).next_multiple_of(4)).sum();
    let positions = size_of::<PageHeader>() + clusters * size_of::<Cluster>();
    let highs_offset = positions + slots * 6;
    let attrs = highs_offset + if highs { slots * 6 } else { 0 };
    let tris_offset = attrs + slots * attr_stride;
    let omm_offset = tris_offset + tris;
    let len = omm_offset + omm;
    let mut p = vec![0u8; len];
    let h = PageHeader {
        cluster_count: clusters as u32,
        attr_stride: attr_stride as u32,
        positions_offset: positions as u32,
        highs_offset: highs_offset as u32,
        attrs_offset: attrs as u32,
        tris_offset: tris_offset as u32,
        omm_offset: omm_offset as u32,
        len: len as u32,
        ..bytemuck::Zeroable::zeroed()
    };
    p[..size_of::<PageHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
    let (mut v0, mut t0) = (0, 0);
    for (i, &(vn, tn)) in counts.iter().enumerate() {
        let c = Cluster {
            vertex_first: v0 as u32,
            tri_first: t0 as u32,
            vertex_count: vn as u16,
            tri_count: tn as u8,
            alpha_first: (rng.next() as usize % (tn + 1)) as u8,
            anchor: [rng.next() as i32 % 1000, -7, 3],
            area: rng.next() as f32,
            ..bytemuck::Zeroable::zeroed()
        };
        let at = size_of::<PageHeader>() + i * size_of::<Cluster>();
        p[at..at + size_of::<Cluster>()].copy_from_slice(bytemuck::bytes_of(&c));
        // Smooth positions, as real clusters have, and noisy attributes.
        for v in 0..vn {
            let at = positions + (v0 + v) * 6;
            for a in 0..3 {
                let x = (v * (a + 3) * 97 + rng.next() as usize % 64) as u16;
                p[at + 2 * a..at + 2 * a + 2].copy_from_slice(&x.to_le_bytes());
            }
            if highs {
                let at = highs_offset + (v0 + v) * 6;
                for b in (0..6).step_by(2) {
                    p[at + b] = rng.next() as u8;
                }
            }
            let at = attrs + (v0 + v) * attr_stride;
            for b in &mut p[at..at + attr_stride] {
                *b = rng.next() as u8;
            }
        }
        for t in 0..tn * 3 {
            p[tris_offset + t0 + t] = (rng.next() as usize % vn) as u8;
        }
        v0 += vn.next_multiple_of(2);
        t0 += (tn * 3).next_multiple_of(4);
    }
    for b in &mut p[omm_offset..] {
        *b = rng.next() as u8;
    }
    p
}

fn rotations_equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.chunks(3)
            .zip(b.chunks(3))
            .all(|(x, y)| (0..3).any(|r| (0..3).all(|k| x[k] == y[(k + r) % 3])))
}

#[test]
fn stored_pages_decode_to_the_resident_page() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for (clusters, stride, highs, omm) in [
        (1, 8, false, 0),
        (5, 12, true, 40),
        (9, 16, false, 0),
        (12, 20, true, 7),
        (30, 8, true, 0),
    ] {
        let p = page(&mut rng, clusters, stride, highs, omm);
        let stored = mesh::encode_page(&p).unwrap();
        let mut out = vec![0xAB; p.len() + 3];
        let layout = mesh::decode_page(&stored, &mut out).unwrap();
        assert_eq!(layout.len as usize, p.len());
        let tris = layout.tris_offset as usize..layout.omm_offset as usize;
        assert_eq!(out[..tris.start], p[..tris.start]);
        assert_eq!(out[tris.end..p.len()], p[tris.end..]);
        // Triangles keep their order and winding; the codec may rotate corners.
        let h: PageHeader = bytemuck::pod_read_unaligned(&p[..size_of::<PageHeader>()]);
        for i in 0..clusters {
            let at = size_of::<PageHeader>() + i * size_of::<Cluster>();
            let c: Cluster = bytemuck::pod_read_unaligned(&p[at..at + size_of::<Cluster>()]);
            let r = (h.tris_offset + c.tri_first) as usize..;
            let n = usize::from(c.tri_count) * 3;
            assert!(rotations_equal(&out[r.clone()][..n], &p[r.clone()][..n]));
            let pad = n.next_multiple_of(4) - n;
            assert!(out[r.start + n..][..pad].iter().all(|&b| b == 0));
        }
        // A page decoded once encodes and decodes to itself.
        let again = mesh::encode_page(&out[..p.len()]).unwrap();
        let mut twice = vec![0; p.len()];
        mesh::decode_page(&again, &mut twice).unwrap();
        assert_eq!(twice, out[..p.len()]);
    }
}

/// A widened page reads as the page did, every region past the positions moved.
#[test]
fn widened_pages_read_as_before() {
    let mut rng = Rng(5);
    let p = page(&mut rng, 7, 16, false, 40);
    let wide = mesh::widen_page(&p).unwrap();
    let h: PageHeader = bytemuck::pod_read_unaligned(&wide[..size_of::<PageHeader>()]);
    assert!(h.has_highs());
    assert_eq!(h.omm_offset % mesh::OMM_ALIGN, 0);
    let header = mesh::MeshHeader {
        attributes: mesh::MeshHeader::TANGENT | mesh::MeshHeader::UV1,
        ..bytemuck::Zeroable::zeroed()
    };
    let (a, b) = (
        mesh::PageView::new(&p, &header).unwrap(),
        mesh::PageView::new(&wide, &header).unwrap(),
    );
    for i in 0..a.cluster_count() {
        let c = a.cluster(i);
        assert_eq!(bytemuck::bytes_of(&c), bytemuck::bytes_of(&b.cluster(i)));
        for v in 0..u32::from(c.vertex_count) {
            assert_eq!(a.grid(&c, v), b.grid(&c, v));
            let words = |view: &mesh::PageView<'_>| {
                let t = view.tangent(&c, v).unwrap().map(f32::to_bits);
                (view.normal(&c, v).map(f32::to_bits), t)
            };
            assert_eq!(words(&a), words(&b));
        }
        for t in 0..u32::from(c.tri_count) {
            assert_eq!(a.triangle(&c, t), b.triangle(&c, t));
        }
    }
    let omm = |page: &[u8], h: &PageHeader| page[h.omm_offset as usize..].to_vec();
    let ph: PageHeader = bytemuck::pod_read_unaligned(&p[..size_of::<PageHeader>()]);
    assert_eq!(omm(&p, &ph), omm(&wide, &h));
}

#[test]
fn damaged_stored_pages_are_rejected() {
    let mut rng = Rng(7);
    let p = page(&mut rng, 6, 12, true, 16);
    let stored = mesh::encode_page(&p).unwrap();
    let mut out = vec![0; p.len()];
    assert!(mesh::decode_page(&stored[..stored.len() - 1], &mut out).is_err());
    assert!(mesh::decode_page(&stored, &mut out[..p.len() - 1]).is_err());
    let table = size_of::<PageHeader>() + 6 * size_of::<Cluster>();
    // A stream offset past the end of the streams.
    let mut bad = stored.clone();
    bad[table + 12..table + 16].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(mesh::decode_page(&bad, &mut out).is_err());
    // A truncated vertex stream.
    let mut bad = stored.clone();
    let second: u32 = bytemuck::pod_read_unaligned(&stored[table + 4..table + 8]);
    bad[table + 4..table + 8].copy_from_slice(&(second - 4).to_le_bytes());
    assert!(mesh::decode_page(&bad, &mut out).is_err());
    // A cluster whose vertices run past the vertex arrays.
    let mut bad = stored.clone();
    let at = size_of::<PageHeader>() + std::mem::offset_of!(Cluster, vertex_first);
    bad[at..at + 4].copy_from_slice(&1_000_000u32.to_le_bytes());
    assert!(mesh::decode_page(&bad, &mut out).is_err());
    // Garbage streams decode to garbage or fail, but never to corners outside a cluster.
    for seed in 0..64 {
        let mut bad = stored.clone();
        let mut r = Rng(seed + 11);
        let streams = table + (2 * 6 + 1) * 4;
        for _ in 0..8 {
            let i = streams + r.next() as usize % (stored.len() - 16 - streams);
            bad[i] ^= 1 << (r.next() % 8);
        }
        if mesh::decode_page(&bad, &mut out).is_ok() {
            let h: PageHeader = bytemuck::pod_read_unaligned(&out[..size_of::<PageHeader>()]);
            for i in 0..6 {
                let at = size_of::<PageHeader>() + i * size_of::<Cluster>();
                let c: Cluster = bytemuck::pod_read_unaligned(&out[at..at + size_of::<Cluster>()]);
                let t =
                    &out[(h.tris_offset + c.tri_first) as usize..][..usize::from(c.tri_count) * 3];
                assert!(t.iter().all(|&v| u16::from(v) < c.vertex_count));
            }
        }
    }
}

#[test]
fn clas_truncation_keeps_every_cell_of_a_box() {
    let mut h: mesh::MeshHeader = bytemuck::Zeroable::zeroed();
    h.position_exponent = -11;
    let step = f64::from(h.step());
    // Past a 1 m mesh at the origin, to an odd cell at the far corner.
    let b = orr_pack::Aabb {
        min: [-1.0, -0.25, 0.0],
        max: [16.0004, 0.5, 1.0],
    };
    let t = u32::from(h.clas_truncate_bits_within(&b));
    let kept = |cell: i64, bits: u32| {
        let x = (cell as f64 * step) as f32;
        x.to_bits() >> bits << bits == x.to_bits()
    };
    for &x in b.min.iter().chain(&b.max) {
        let far = (f64::from(x) / step).round() as i64;
        let inward = -far.signum();
        for cell in (0..512).map(|i| far + inward * i) {
            assert!(kept(cell, t), "cell {cell} with {t} bits truncated");
        }
    }
    let far = (f64::from(b.max[0]) / step).round() as i64;
    assert!(!kept(far, t + 1), "{t} bits is not the most the box allows");
}

#[test]
fn axes_round_trip_exactly_through_the_normal_and_tangent_codes() {
    for a in 0..6 {
        let mut n = [0.0f32; 3];
        n[a / 2] = if a % 2 == 0 { 1.0 } else { -1.0 };
        assert_eq!(mesh::oct_decode(mesh::oct_encode(n)), n, "normal {n:?}");
        for sign in [1.0, -1.0] {
            let [x, y, z, s] = mesh::oct_decode_tangent(mesh::oct_encode_tangent(n, sign));
            assert_eq!(([x, y, z], s), (n, sign), "tangent {n:?} sign {sign}");
        }
    }
}
