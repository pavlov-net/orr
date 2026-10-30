use orr_pack::instance::{
    self, InstancePageHeader, InstanceRecord, Prototype, SectionOverride, axis_rotation,
    decode_rotation, encode_rotation, exact_axis_index, instance_rotation,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Largest rotation angle, in degrees, between a quaternion and its decoded code. Ten bits
/// per component bound it near 0.16 degrees, where all four components are near 1/2.
const ROTATION_BOUND_DEG: f64 = 0.17;

fn angle_deg(q: [f64; 4], d: [f32; 4]) -> f64 {
    let len = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    let dot: f64 = (0..4).map(|i| q[i] / len * f64::from(d[i])).sum();
    2.0 * dot.abs().min(1.0).acos().to_degrees()
}

#[test]
fn layouts_match_the_gpu_structs() {
    assert_eq!(size_of::<InstancePageHeader>(), 64);
    assert_eq!(size_of::<Prototype>(), 32);
    assert_eq!(size_of::<SectionOverride>(), 20);
    assert_eq!(size_of::<InstanceRecord>(), 16);
    assert_eq!(std::mem::offset_of!(InstancePageHeader, user_stride), 52);
    assert_eq!(std::mem::offset_of!(InstancePageHeader, partitions), 56);
    assert_eq!(std::mem::offset_of!(InstanceRecord, rot), 8);
}

#[test]
fn rotations_decode_within_the_bound() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut cases: Vec<[f64; 4]> = Vec::new();
    for _ in 0..1_000_000 {
        let q = loop {
            let v = [0; 4].map(|_: u8| rng.unit() * 2.0 - 1.0);
            let l = v.iter().map(|c| c * c).sum::<f64>().sqrt();
            if l > 1e-3 && l <= 1.0 {
                break v.map(|c| c / l);
            }
        };
        cases.push(q);
        // Both signs of a quaternion are one rotation.
        cases.push(q.map(|c| -c));
    }
    let h = std::f64::consts::FRAC_1_SQRT_2;
    // Two or four components tie for the largest, components at +-1/sqrt(2), zeros (not on
    // the grid), single axes and turns about one axis.
    cases.extend([
        [h, 0.0, 0.0, h],
        [-h, 0.0, 0.0, h],
        [0.5, 0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1e-9, 0.0, -1e-9, 1.0],
    ]);
    for i in 0..3600 {
        let a = f64::from(i).to_radians() * 0.5;
        cases.push([0.0, a.sin(), 0.0, a.cos()]);
        cases.push([a.sin(), 0.0, 0.0, a.cos()]);
    }
    let mut worst = 0f64;
    for q in &cases {
        let code = encode_rotation(q.map(|c| c as f32));
        let d = decode_rotation(code);
        let len: f32 = d.iter().map(|c| c * c).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-5, "{q:?}: decoded length {len}");
        let e = angle_deg(*q, d);
        assert!(e <= ROTATION_BOUND_DEG, "{q:?} -> {d:?}: {e} deg");
        worst = worst.max(e);
    }
    // The matrix decode is the quaternion's rotation.
    for q in cases.iter().step_by(997) {
        let code = encode_rotation(q.map(|c| c as f32));
        let [x, y, z, w] = decode_rotation(code).map(f64::from);
        let m = instance_rotation(code, false);
        let r = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!((f64::from(m[i][j]) - r[i][j]).abs() < 1e-6);
            }
        }
    }
    assert!(
        worst > 0.1,
        "worst {worst} deg: the cases miss the coarse corners of the grid"
    );
}

/// The quaternion of a proper rotation matrix (rows).
fn matrix_quat(m: [[f32; 3]; 3]) -> [f32; 4] {
    let m = m.map(|r| r.map(f64::from));
    let t = m[0][0] + m[1][1] + m[2][2];
    let q = if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            s / 4.0,
        ]
    } else if m[0][0] >= m[1][1] && m[0][0] >= m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            s / 4.0,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] >= m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][1] + m[1][0]) / s,
            s / 4.0,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            s / 4.0,
            (m[1][0] - m[0][1]) / s,
        ]
    };
    q.map(|c| c as f32)
}

#[test]
fn exact_axis_rotations_are_the_24_signed_permutations() {
    let mut seen = Vec::new();
    for index in 0..24 {
        let m = axis_rotation(index);
        for row in m {
            assert_eq!(row.iter().filter(|v| v.abs() == 1.0).count(), 1, "{index}");
            assert_eq!(row.iter().filter(|&&v| v == 0.0).count(), 2, "{index}");
        }
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        assert_eq!(det, 1.0, "{index}");
        assert!(!seen.contains(&m), "{index} repeats");
        seen.push(m);
        assert_eq!(instance_rotation(index, true), m);
        // The encoder finds each table entry from its quaternion, either sign.
        let q = matrix_quat(m);
        assert_eq!(exact_axis_index(q), Some(index));
        assert_eq!(exact_axis_index(q.map(|c| -c)), Some(index));
    }
    assert_eq!(
        axis_rotation(0),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );
    let a = 0.5f32.to_radians() * 0.5;
    assert_eq!(exact_axis_index([0.0, a.sin(), 0.0, a.cos()]), None);
}

#[test]
fn morton_interleaves_bits() {
    let mut rng = Rng(77);
    for _ in 0..10_000 {
        let p = [0; 3].map(|_: u8| rng.next() as u16);
        let mut want = 0u64;
        for b in 0..16 {
            for (a, v) in p.iter().enumerate() {
                want |= u64::from(v >> b & 1) << (3 * b + a);
            }
        }
        assert_eq!(instance::morton(p), want);
    }
}

/// Checks that partitions cover the records in order, each within `max`, each a whole
/// Morton-prefix cell and the largest such cell that fits.
fn check_partitions(codes: &[u64], costs: &[u32], starts: &[u32], max: u32) {
    assert_eq!(starts.first(), Some(&0));
    let prefix = |c: u64, level: u32| c >> (3 * (16 - level));
    let cost = |r: std::ops::Range<usize>| costs[r].iter().map(|&c| u64::from(c)).sum::<u64>();
    // Records whose code shares `level` levels with record `i`.
    let cell = |i: usize, level: u32| {
        let p = prefix(codes[i], level);
        let a = codes.partition_point(|&c| prefix(c, level) < p);
        let b = codes.partition_point(|&c| prefix(c, level) <= p);
        a..b
    };
    for (k, &s) in starts.iter().enumerate() {
        let (a, b) = (
            s as usize,
            starts.get(k + 1).map_or(codes.len(), |&e| e as usize),
        );
        assert!(a < b);
        assert!(cost(a..b) <= u64::from(max), "partition {k} over {max}");
        let level = (0..=16)
            .find(|&l| cell(a, l) == (a..b))
            .unwrap_or_else(|| panic!("partition {k} is not a Morton-prefix cell"));
        if level > 0 {
            assert!(
                cost(cell(a, level - 1)) > u64::from(max),
                "partition {k}: its parent cell fits too"
            );
        }
    }
}

#[test]
fn partitions_are_the_largest_fitting_morton_cells() {
    let mut rng = Rng(12345);
    for round in 0..40 {
        let n = 1 + (rng.next() % 16_384) as usize;
        // Clustered positions so cells split at many levels.
        let centres: Vec<[u16; 3]> = (0..1 + rng.next() % 8)
            .map(|_| [0; 3].map(|_: u8| rng.next() as u16))
            .collect();
        let mut codes: Vec<u64> = (0..n)
            .map(|_| {
                let c = centres[(rng.next() % centres.len() as u64) as usize];
                let spread = 1u64 << (10 + rng.next() % 7);
                instance::morton(c.map(|v| v.wrapping_add((rng.next() % spread) as u16)))
            })
            .collect();
        codes.sort_unstable();
        let costs: Vec<u32> = (0..n)
            .map(|_| match round % 3 {
                0 => 1,
                1 => 1 + (rng.next() % 33) as u32,
                _ => [1, 1, 1, 241][(rng.next() % 4) as usize],
            })
            .collect();
        let starts = instance::partitions(&codes, &costs, 1000).unwrap();
        check_partitions(&codes, &costs, &starts, 1000);
    }
    // Copies at one position over the limit cannot be cut into cells.
    assert_eq!(instance::partitions(&[5; 3], &[400; 3], 1000), None);
    assert_eq!(instance::partitions(&[], &[], 1000), Some(vec![]));
}
