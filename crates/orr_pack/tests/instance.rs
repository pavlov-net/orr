use orr_math::{DVec4, Mat3, Quat, Vec3, Vec4};
use orr_pack::instance::{
    self, AXIS_ROTATIONS, InstancePageHeader, InstancePageInfo, InstanceRecord, LinearRows,
    MotionRow, Prototype, SectionOverride, axis_rotation, decode_rotation, encode_rotation,
    exact_axis_index, instance_rotation,
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
    let dot = DVec4::from(q).normalize().dot(Vec4::from(d).as_dvec4());
    2.0 * dot.abs().min(1.0).acos().to_degrees()
}

#[test]
fn layouts_match_the_gpu_structs() {
    assert_eq!(size_of::<InstancePageHeader>(), 196);
    assert_eq!(size_of::<MotionRow>(), 88);
    assert_eq!(size_of::<Prototype>(), 32);
    assert_eq!(size_of::<SectionOverride>(), 20);
    assert_eq!(size_of::<InstanceRecord>(), 16);
    assert_eq!(size_of::<LinearRows>(), 36);
    assert_eq!(std::mem::offset_of!(InstancePageHeader, user_stride), 52);
    assert_eq!(std::mem::offset_of!(InstancePageHeader, partitions), 56);
    assert_eq!(std::mem::offset_of!(InstanceRecord, rot), 8);
}

/// A page of a compact record and two table rows, one of which holds a mirrored shear.
fn sheared_page() -> (InstancePageInfo, Vec<u8>) {
    let shear = Mat3::from_cols(
        Vec3::new(1.0, 0.25, -0.5),
        Vec3::new(0.75, 2.0, 0.125),
        Vec3::new(-0.375, 0.0, 3.0),
    );
    let mirrored = shear * Mat3::from_diagonal(Vec3::new(-1.0, 1.0, 1.0));
    let record = |pos, rot, scale| InstanceRecord {
        pos,
        prototype: 0,
        rot,
        scale,
    };
    let records = [
        record([0, 0, 0], 0, InstanceRecord::INDEXED),
        record([65535, 0, 0], AXIS_ROTATIONS, InstanceRecord::INDEXED),
        record([0, 65535, 0], AXIS_ROTATIONS + 1, InstanceRecord::INDEXED),
    ];
    let table = [LinearRows::from(shear), LinearRows::from(mirrored)];
    let info = InstancePageInfo {
        header: InstancePageHeader {
            extent: [8.0; 3],
            record_count: records.len() as u16,
            ..bytemuck::Zeroable::zeroed()
        },
        prototypes: vec![bytemuck::Zeroable::zeroed()],
        sections: Vec::new(),
        partitions: vec![0],
        motion_times: Vec::new(),
    };
    let blob = [
        bytemuck::cast_slice::<_, u8>(&records),
        bytemuck::cast_slice(&table),
    ]
    .concat();
    (info, blob)
}

#[test]
fn a_table_that_does_not_match_its_records_is_malformed() {
    let (info, blob) = sheared_page();
    let head = info.to_bytes();
    assert!(instance::decode(&head, &blob).is_ok());
    // A row short, and one of the two rows gone with a record still naming it.
    assert!(instance::decode(&head, &blob[..blob.len() - 4]).is_err());
    assert!(instance::decode(&head, &blob[..blob.len() - 36]).is_err());
    // Fewer bytes than the header's records.
    assert!(instance::decode(&head, &blob[..16]).is_err());
}

#[test]
fn rotations_decode_within_the_bound() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut cases: Vec<[f64; 4]> = Vec::new();
    for _ in 0..1_000_000 {
        let q = loop {
            let v = [0; 4].map(|_: u8| rng.unit() * 2.0 - 1.0);
            let v = DVec4::from(v);
            let l = v.length();
            if l > 1e-3 && l <= 1.0 {
                break (v / l).to_array();
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
        let len = Vec4::from(d).length();
        assert!((len - 1.0).abs() < 1e-5, "{q:?}: decoded length {len}");
        let e = angle_deg(*q, d);
        assert!(e <= ROTATION_BOUND_DEG, "{q:?} -> {d:?}: {e} deg");
        worst = worst.max(e);
    }
    assert!(
        worst > 0.1,
        "worst {worst} deg: the cases miss the coarse corners of the grid"
    );
}

#[test]
fn exact_axis_rotations_are_the_24_signed_permutations() {
    let mut seen = Vec::new();
    for index in 0..24 {
        let m = axis_rotation(index);
        for row in [m.row(0), m.row(1), m.row(2)].map(|r| r.to_array()) {
            assert_eq!(row.iter().filter(|v| v.abs() == 1.0).count(), 1, "{index}");
            assert_eq!(row.iter().filter(|&&v| v == 0.0).count(), 2, "{index}");
        }
        assert_eq!(m.determinant(), 1.0, "{index}");
        assert!(!seen.contains(&m), "{index} repeats");
        seen.push(m);
        assert_eq!(instance_rotation(index, true), m);
        // The encoder finds each table entry from its quaternion, either sign.
        let q = Quat::from_mat3(&m).to_array();
        assert_eq!(exact_axis_index(q), Some(index));
        assert_eq!(exact_axis_index(q.map(|c| -c)), Some(index));
    }
    assert_eq!(axis_rotation(0), Mat3::IDENTITY);
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
        let centers: Vec<[u16; 3]> = (0..1 + rng.next() % 8)
            .map(|_| [0; 3].map(|_: u8| rng.next() as u16))
            .collect();
        let mut codes: Vec<u64> = (0..n)
            .map(|_| {
                let c = centers[(rng.next() % centers.len() as u64) as usize];
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
