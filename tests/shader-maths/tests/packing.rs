use bytemuck::Zeroable;
use orr_pack::mesh::{self, Cluster, MeshHeader, PageHeader, PageView};
use shader_maths::stats::Rng;
use shader_maths::*;

/// Unit vectors that sit on the octahedral map's folds and edges, then random ones.
fn normals(count: usize) -> Vec<[f32; 3]> {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let mut out = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [-0.0, -0.0, 1.0],
        [-0.0, 0.0, -1.0],
        [s, s, 0.0],
        [-s, s, 0.0],
        [s, 0.0, -s],
        [0.0, -s, -s],
    ];
    let mut rng = Rng::new(1);
    out.extend((0..count).map(|_| rng.unit()));
    out
}

#[test]
fn oct_unorm16_matches_orr_pack_bit_for_bit() {
    for n in normals(200_000) {
        assert_eq!(oct_encode_unorm16(n), mesh::oct_encode(n), "{n:?}");
    }
    let mut rng = Rng::new(2);
    let corners = [
        0u32,
        0xFFFF,
        0xFFFF_0000,
        0xFFFF_FFFF,
        0x7FFF_7FFF,
        0x8000_8000,
        0x0000_7FFF,
    ];
    for e in corners
        .into_iter()
        .chain((0..200_000).map(|_| rng.next_u32()))
    {
        let (a, b) = (oct_decode_unorm16(e), mesh::oct_decode(e));
        assert_eq!(
            a.map(f32::to_bits),
            b.map(f32::to_bits),
            "{e:#010x}: {a:?} vs {b:?}"
        );
    }
}

/// A cluster's anchor and its vertices' grid offsets and normal codes.
type TestCluster = ([i32; 3], Vec<([u16; 3], u32)>);

/// One page holding `clusters`.
fn page(clusters: &[TestCluster]) -> Vec<u8> {
    let n = clusters.len();
    let slots: usize = clusters
        .iter()
        .map(|(_, v)| v.len().next_multiple_of(2))
        .sum();
    let positions = size_of::<PageHeader>() + n * size_of::<Cluster>();
    let attrs = positions + slots * 6;
    let len = attrs + slots * 8;
    let header = PageHeader {
        cluster_count: n as u32,
        attr_stride: 8,
        positions_offset: positions as u32,
        attrs_offset: attrs as u32,
        tris_offset: len as u32,
        omm_offset: len as u32,
        len: len as u32,
        ..PageHeader::zeroed()
    };
    let mut out = vec![0u8; len];
    out[..size_of::<PageHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    let mut first = 0usize;
    for (i, (anchor, vertices)) in clusters.iter().enumerate() {
        let c = Cluster {
            anchor: *anchor,
            vertex_first: first as u32,
            vertex_count: vertices.len() as u16,
            omm_first: Cluster::NO_OMM,
            ..Cluster::zeroed()
        };
        let at = size_of::<PageHeader>() + i * size_of::<Cluster>();
        out[at..at + size_of::<Cluster>()].copy_from_slice(bytemuck::bytes_of(&c));
        for (v, (local, normal)) in vertices.iter().enumerate() {
            let p = positions + (first + v) * 6;
            out[p..p + 6].copy_from_slice(bytemuck::cast_slice(local));
            let a = attrs + (first + v) * 8;
            out[a..a + 4].copy_from_slice(&normal.to_le_bytes());
        }
        first += vertices.len().next_multiple_of(2);
    }
    out
}

#[test]
fn mesh_page_decode_matches_orr_pack_bit_for_bit() {
    let mut rng = Rng::new(3);
    let limit = (1 << 23) - 65536;
    let clusters: Vec<_> = (0..64)
        .map(|_| {
            let anchor = [0; 3].map(|_: i32| (rng.next_u32() % (2 * limit as u32)) as i32 - limit);
            let count = 1 + rng.next_u32() as usize % 255;
            let vertices = (0..count)
                .map(|_| ([0; 3].map(|_: u16| rng.next_u32() as u16), rng.next_u32()))
                .collect();
            (anchor, vertices)
        })
        .collect();
    let bytes = page(&clusters);
    for k in [-24i8, -13, -7, -1, 0, 3, 8] {
        let mesh_header = MeshHeader {
            position_exponent: k,
            ..MeshHeader::zeroed()
        };
        let view = PageView::new(&bytes, &mesh_header).unwrap();
        let step = grid_step(i32::from(k));
        assert_eq!(step.to_bits(), mesh_header.step().to_bits(), "step 2^{k}");
        for (i, (anchor, vertices)) in clusters.iter().enumerate() {
            let c = view.cluster(i as u32);
            for (v, (local, _)) in vertices.iter().enumerate() {
                let ours = grid_position(*anchor, local.map(u32::from), step);
                let theirs = view.position(&c, v as u32);
                assert_eq!(
                    ours.map(f32::to_bits),
                    theirs.map(f32::to_bits),
                    "cluster {i} vertex {v} k {k}"
                );
                let n = oct_decode_unorm16(vertices[v].1);
                assert_eq!(
                    n.map(f32::to_bits),
                    view.normal(&c, v as u32).map(f32::to_bits)
                );
            }
        }
    }
}

/// The octahedral square point of `n`, in f64.
fn oct_wrap(n: [f32; 3]) -> [f64; 2] {
    let n = n.map(f64::from);
    let sign = |v: f64| if v >= 0.0 { 1.0 } else { -1.0 };
    let l = n[0].abs() + n[1].abs() + n[2].abs();
    let (x, y) = (n[0] / l, n[1] / l);
    if n[2] < 0.0 {
        [(1.0 - y.abs()) * sign(x), (1.0 - x.abs()) * sign(y)]
    } else {
        [x, y]
    }
}

/// The unit vector of an octahedral square point, in f64.
fn oct_unwrap(e: [f64; 2]) -> [f64; 3] {
    let sign = |v: f64| if v >= 0.0 { 1.0 } else { -1.0 };
    let (mut x, mut y) = (e[0], e[1]);
    let z = 1.0 - x.abs() - y.abs();
    if z < 0.0 {
        (x, y) = ((1.0 - y.abs()) * sign(x), (1.0 - x.abs()) * sign(y));
    }
    let l = (x * x + y * y + z * z).sqrt();
    [x / l, y / l, z / l]
}

#[test]
fn oct_forms_round_trip_within_their_step() {
    let unorm16 = |c: u32| {
        [
            f64::from(c & 0xFFFF) / 65535.0 * 2.0 - 1.0,
            f64::from(c >> 16) / 65535.0 * 2.0 - 1.0,
        ]
    };
    let snorm = |bits: u32, max: f64| {
        move |c: u32| {
            let lane = |v: u32| {
                let q = ((v << (32 - bits)) as i32) >> (32 - bits);
                (f64::from(q) / max).max(-1.0)
            };
            [lane(c), lane(c >> bits)]
        }
    };
    let snorm16 = snorm(16, 32767.0);
    let snorm8 = snorm(8, 127.0);
    type Form<'a> = (
        &'a str,
        fn([f32; 3]) -> u32,
        fn(u32) -> [f32; 3],
        &'a dyn Fn(u32) -> [f64; 2],
        f64,
    );
    let forms: [Form; 3] = [
        (
            "unorm16",
            oct_encode_unorm16,
            oct_decode_unorm16,
            &unorm16,
            1.0 / 65535.0,
        ),
        (
            "snorm16",
            oct_encode_snorm16,
            oct_decode_snorm16,
            &snorm16,
            1.0 / 32767.0,
        ),
        (
            "snorm8",
            oct_encode_snorm8,
            oct_decode_snorm8,
            &snorm8,
            1.0 / 127.0,
        ),
    ];
    for (name, encode, decode, square, half_step) in forms {
        for n in normals(100_000) {
            let code = encode(n);
            let (want, got) = (oct_wrap(n), square(code));
            for a in 0..2 {
                // f32 evaluation adds up to a few ulps of the scaled value.
                assert!(
                    (want[a] - got[a]).abs() <= half_step + 3e-7,
                    "{name} {n:?}: {want:?} coded as {got:?}"
                );
            }
            let d = decode(code);
            let r = oct_unwrap(got);
            for a in 0..3 {
                assert!(
                    (f64::from(d[a]) - r[a]).abs() < 4e-7,
                    "{name} {code:#x}: {d:?} vs {r:?}"
                );
            }
        }
    }
}

#[test]
fn scalar_words_round_trip_within_their_step() {
    for c in 0..=0xFFFFu32 {
        let p = c | (0xFFFF - c) << 16;
        assert_eq!(pack_unorm2x16(unpack_unorm2x16(p)), p);
        let s = unpack_snorm2x16(p);
        // -32768 decodes to -1 like -32767 and encodes back as -32767.
        let canonical = |h: u32| if h == 0x8000 { 0x8001 } else { h };
        assert_eq!(
            pack_snorm2x16(s),
            canonical(p & 0xFFFF) | canonical(p >> 16) << 16
        );
    }
    for c in 0..=0xFFu32 {
        let p = c | (0xFF - c) << 8 | (c ^ 0x5A) << 16 | (c.wrapping_mul(7) & 0xFF) << 24;
        assert_eq!(pack_unorm4x8(unpack_unorm4x8(p)), p);
        let canonical = |b: u32| if b == 0x80 { 0x81 } else { b };
        let want = (0..4)
            .map(|i| canonical(p >> (8 * i) & 0xFF) << (8 * i))
            .sum::<u32>();
        assert_eq!(pack_snorm4x8(unpack_snorm4x8(p)), want);
    }
    for c in 0..=0xFFFFu32 {
        let exp = c >> 10 & 0x1F;
        if exp != 0x1F {
            let p = c | c << 16;
            assert_eq!(pack_half2x16(unpack_half2x16(p)), p, "half {c:#06x}");
        }
    }
    let mut rng = Rng::new(4);
    for _ in 0..100_000 {
        let (u, s) = (
            [rng.f32(), rng.f32()],
            [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)],
        );
        let (du, ds) = (
            unpack_unorm2x16(pack_unorm2x16(u)),
            unpack_snorm2x16(pack_snorm2x16(s)),
        );
        let u4 = [rng.f32(), rng.f32(), rng.f32(), rng.f32()];
        let s4 = u4.map(|v| v * 2.0 - 1.0);
        let (du4, ds4) = (
            unpack_unorm4x8(pack_unorm4x8(u4)),
            unpack_snorm4x8(pack_snorm4x8(s4)),
        );
        for i in 0..2 {
            assert!((du[i] - u[i]).abs() <= 0.5 / 65535.0 + 1e-7);
            assert!((ds[i] - s[i]).abs() <= 0.5 / 32767.0 + 1e-7);
        }
        for i in 0..4 {
            assert!((du4[i] - u4[i]).abs() <= 0.5 / 255.0 + 1e-7);
            assert!((ds4[i] - s4[i]).abs() <= 0.5 / 127.0 + 1e-7);
        }
        // Normal halves: relative error at most half an ulp, 2^-11.
        let h = [rng.range(6.2e-5, 65000.0), -rng.range(6.2e-5, 1.0)];
        let dh = unpack_half2x16(pack_half2x16(h));
        for i in 0..2 {
            assert!(
                (dh[i] - h[i]).abs() <= h[i].abs() * 2f32.powi(-11),
                "{} -> {}",
                h[i],
                dh[i]
            );
        }
    }
    assert_eq!(
        unpack_half2x16(pack_half2x16([70000.0, -1e6])),
        [f32::INFINITY, f32::NEG_INFINITY]
    );
}

/// The value of an unsigned float with a 5-bit exponent and `m` mantissa bits.
fn small_float(v: u32, m: u32) -> f64 {
    let (e, f) = (v >> m, v & ((1 << m) - 1));
    if e == 0 {
        f64::from(f) * 2f64.powi(-14 - m as i32)
    } else {
        (1.0 + f64::from(f) / f64::from(1u32 << m)) * 2f64.powi(e as i32 - 15)
    }
}

#[test]
fn packed_float_colour_round_trips() {
    // Every finite code decodes to its value and encodes back to itself.
    for c in 0..(31u32 << 6) {
        let p = c | c << 11 | (c >> 1) << 22;
        let d = unpack_r11g11b10f(p);
        assert_eq!(f64::from(d[0]), small_float(c, 6));
        assert_eq!(f64::from(d[2]), small_float(c >> 1, 5));
        assert_eq!(pack_r11g11b10f(d), p, "{c:#x}");
    }
    let mut rng = Rng::new(5);
    for _ in 0..200_000 {
        let c = [0; 3].map(|_: u8| 2f32.powf(rng.range(-20.0, 15.9)));
        let d = unpack_r11g11b10f(pack_r11g11b10f(c));
        for (i, m) in [6, 6, 5].into_iter().enumerate() {
            // Half an ulp at the value's exponent, or of the denormal step below 2^-14.
            let e = f64::from(c[i]).log2().floor().max(-14.0);
            let half_ulp = 2f64.powf(e - f64::from(m) - 1.0);
            assert!(
                (f64::from(d[i]) - f64::from(c[i])).abs() <= half_ulp * (1.0 + 1e-6),
                "{c:?} -> {d:?}"
            );
        }

        let c = [rng.f32(), rng.f32(), rng.f32()].map(|v| v * 2f32.powf(rng.range(-16.0, 15.0)));
        let d = unpack_rgb9e5(pack_rgb9e5(c));
        let max = c.iter().copied().fold(0.0f32, f32::max);
        // The shared step is 2^(floor(log2 max) - 8), doubled when max rounds up to 512 steps.
        let mut step = 2f64.powf(f64::from(max).log2().floor().max(-16.0) - 8.0);
        if (f64::from(max) / step + 0.5).floor() >= 512.0 {
            step *= 2.0;
        }
        let half_step = step / 2.0;
        for i in 0..3 {
            assert!(
                (f64::from(d[i]) - f64::from(c[i])).abs() <= half_step * (1.0 + 1e-6),
                "{c:?} -> {d:?}"
            );
        }
    }
    let largest11 = small_float((31 << 6) - 1, 6) as f32;
    let largest10 = small_float((31 << 5) - 1, 5) as f32;
    assert_eq!(
        unpack_r11g11b10f(pack_r11g11b10f([1e9, f32::INFINITY, 1e30])),
        [largest11, largest11, largest10]
    );
    assert_eq!(
        unpack_r11g11b10f(pack_r11g11b10f([-1.0, f32::NAN, -0.0])),
        [0.0; 3]
    );
    assert_eq!(
        unpack_rgb9e5(pack_rgb9e5([1e9, -1.0, f32::NAN])),
        [65408.0, 0.0, 0.0]
    );
}

const RANGE: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Smallest-three encoding of a unit quaternion (x, y, z, w), as `orr.packing` decodes it.
fn encode_rotation(q: [f64; 4]) -> u32 {
    let largest = (0..4)
        .max_by(|&a, &b| q[a].abs().total_cmp(&q[b].abs()))
        .unwrap();
    let sign = if q[largest] < 0.0 { -1.0 } else { 1.0 };
    let mut code = (largest as u32) << 30;
    for (slot, i) in (0..4).filter(|&i| i != largest).enumerate() {
        let v = ((q[i] * sign / RANGE * 0.5 + 0.5).clamp(0.0, 1.0) * 1023.0).round() as u32;
        code |= v << (10 * slot);
    }
    code
}

#[test]
fn rotations_round_trip_within_their_step() {
    let mut rng = Rng::new(6);
    let half_step = RANGE / 1023.0;
    for _ in 0..200_000 {
        let q = loop {
            let v = [0; 4].map(|_: u8| f64::from(rng.range(-1.0, 1.0)));
            let l = v.iter().map(|c| c * c).sum::<f64>().sqrt();
            if l > 1e-3 && l <= 1.0 {
                break v.map(|c| c / l);
            }
        };
        let code = encode_rotation(q);
        let d = decode_rotation(code).map(f64::from);
        let largest = (code >> 30) as usize;
        let sign = if q[largest] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..4 {
            let err = (d[i] - q[i] * sign).abs();
            // Stored components within half a step. The dropped one is at least 1/2, so
            // their errors move it by at most sqrt(3) half steps over 1/2.
            let bound = if i == largest {
                3f64.sqrt() * half_step / 0.5
            } else {
                half_step
            };
            assert!(err <= bound * (1.0 + 1e-4) + 1e-6, "{q:?} -> {d:?}");
        }
        let m = instance_rotation(code, false);
        let r = glam::DMat3::from_quat(glam::DQuat::from_array(d));
        for i in 0..3 {
            for (j, w) in r.row(i).to_array().into_iter().enumerate() {
                assert!((f64::from(m[3 * i + j]) - w).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn exact_axis_rotations_are_the_24_proper_rotations() {
    let mut seen = Vec::new();
    for index in 0..24 {
        let m = instance_rotation(index, true);
        assert!(
            m.iter().all(|&v| v == 0.0 || v == 1.0 || v == -1.0),
            "{index}: {m:?}"
        );
        let r = glam::Mat3::from_cols_array(&m).transpose();
        assert_eq!(r.determinant(), 1.0, "{index}");
        assert_eq!(r * r.transpose(), glam::Mat3::IDENTITY, "{index}");
        assert!(!seen.contains(&m), "{index} repeats");
        seen.push(m);
    }
    assert_eq!(
        instance_rotation(0, true),
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}
