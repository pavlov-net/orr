//! `orr_pack.instance`'s `motion_transform`, the kernel that poses animated pages, on the GPU
//! against `orr_pack::instance::motion_transform`, which keeps the clock and the turn in f64.

use orr_gpu::MemoryKind;
use orr_math::{Affine3, DMat3, Mat3, Quat, Vec3};
use orr_pack::AffineRows;
use orr_pack::instance::{self, InstancePageHeader, MotionRow};
use orr_scene::testing::{assert_clean, device};
use pack_suite::dispatch;
use pack_suite::shaders::motion;

/// The GPU transform of each `(row, dt)` between `left` and `right`; `None` where it hides.
fn motion_on_gpu(
    rows: &[(MotionRow, f64)],
    left: AffineRows,
    right: AffineRows,
) -> Vec<Option<AffineRows>> {
    let d = device();
    let n = rows.len() as u64;
    let mut r = d
        .gpu_malloc::<MotionRow>(MemoryKind::Upload, n, "rows")
        .unwrap();
    let mut dts = d
        .gpu_malloc::<[f32; 2]>(MemoryKind::Upload, n, "dts")
        .unwrap();
    for (i, (row, dt)) in rows.iter().enumerate() {
        r.as_mut_slice().unwrap()[i] = *row;
        dts.as_mut_slice().unwrap()[i] = instance::motion_dt(*dt);
    }
    let mut sides = d
        .gpu_malloc::<AffineRows>(MemoryKind::Upload, 2, "sides")
        .unwrap();
    sides
        .as_mut_slice()
        .unwrap()
        .copy_from_slice(&[left, right]);
    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 3 * n, "out")
        .unwrap();
    let mut root = d
        .gpu_malloc::<motion::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = motion::Root {
        rows: r.gpu().cast(),
        dts: dts.gpu().cast(),
        left: sides.gpu().cast(),
        right: sides.gpu().offset(1).cast(),
        out: out.gpu(),
        count: n as u32,
        _pad0: 0,
    };
    dispatch(&motion::MODULE, root.gpu(), n.div_ceil(64) as u32);
    assert_clean();
    let hidden = |m: &AffineRows| m.rows == [[0.0; 4]; 3];
    out.as_slice()
        .unwrap()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|m| Some(AffineRows { rows: *m }).filter(|m| !hidden(m)))
        .collect()
}

/// The angle in degrees between the linear parts of two transforms whose linear parts turn alone.
fn angle(a: &AffineRows, b: &AffineRows) -> f64 {
    let linear = |m: &AffineRows| {
        let rows = m.rows.map(|r| [r[0], r[1], r[2]].map(f64::from));
        DMat3::from_cols_array_2d(&rows).transpose()
    };
    let d = linear(a).transpose() * linear(b);
    // From the turn's axis as well as its trace: the trace alone loses a small angle to the
    // rounding of the matrices.
    let cos = (d.x_axis.x + d.y_axis.y + d.z_axis.z - 1.0) / 2.0;
    let axis = orr_math::DVec3::new(
        d.y_axis.z - d.z_axis.y,
        d.z_axis.x - d.x_axis.z,
        d.x_axis.y - d.y_axis.x,
    );
    (axis.length() / 2.0).atan2(cos).to_degrees()
}

#[test]
fn the_kernel_poses_rows_as_the_reference() {
    let turn = Quat::from_axis_angle(Vec3::new(0.3, -1.0, 0.5).normalize(), 0.7).to_array();
    let still = MotionRow {
        position: [12.5, -3.25, 7.0],
        rotation: turn,
        scale: [1.0, 2.0, 0.5],
        ..MotionRow::default()
    };
    let moving = MotionRow {
        velocity: [0.75, 0.0, -2.0],
        acceleration: [0.0, -9.8, 0.0],
        growth: [0.25, 0.0, -0.125],
        spin: [0.0, 30.0, 10.0],
        ..still
    };
    // Turning 7200 degrees a second about an axis whose rate an f32 does not hold.
    let fast = MotionRow {
        spin: (Vec3::new(1.0, 2.0, 3.0).normalize() * 7200.0).to_array(),
        position: [0.0; 3],
        scale: [1.0; 3],
        ..still
    };
    let gone = MotionRow {
        rotation: [0.0; 4],
        ..still
    };
    // An hour into an interval that runs on without end, and the first, which runs back.
    let late = [3_600.123_456_789, 3_599.987_654_321, -1_234.567_891_23];
    let mut rows: Vec<(MotionRow, f64)> = Vec::new();
    for dt in [0.0, 0.25, 1.5, 17.0] {
        rows.extend([(still, dt), (moving, dt), (gone, dt)]);
    }
    for dt in [0.0, 0.25, 1.5].into_iter().chain(late) {
        rows.extend([(still, dt), (fast, dt)]);
    }
    let identity = AffineRows::from(Affine3::IDENTITY);
    let sides = [
        (identity, identity),
        (
            AffineRows::from(Affine3::from_mat3_translation(
                Mat3::from_quat(Quat::from_rotation_y(0.4)) * 1.5,
                Vec3::new(100.0, -20.0, 3.0),
            )),
            AffineRows::from(Affine3::from_translation(Vec3::new(-0.5, 0.0, 0.25))),
        ),
    ];
    for (left, right) in sides {
        let gpu = motion_on_gpu(&rows, left, right);
        let h = InstancePageHeader {
            motion_left: left,
            motion_right: right,
            ..bytemuck::Zeroable::zeroed()
        };
        for ((row, dt), got) in rows.iter().zip(&gpu) {
            let want = instance::motion_transform(&h, row, *dt).map(AffineRows::from);
            let (Some(got), Some(want)) = (got, want) else {
                assert!(got.is_none() && want.is_none(), "{row:?} at {dt} s");
                continue;
            };
            // A row that holds still stands the same at every time, as in the reference.
            if row.velocity == [0.0; 3] && row.spin == [0.0; 3] {
                let first = rows.iter().zip(&gpu).find(|(r, _)| r.0 == *row).unwrap();
                assert_eq!(first.1.unwrap().rows, got.rows, "{dt} s");
                let at_rest = instance::motion_transform(&h, row, 0.0).map(AffineRows::from);
                assert_eq!(at_rest.unwrap().rows, want.rows, "{dt} s");
            }
            let size = (0..3)
                .flat_map(|r| (0..4).map(move |c| want.rows[r][c].abs()))
                .fold(1.0f32, f32::max);
            for (g, w) in got.rows.iter().zip(&want.rows) {
                for c in 0..4 {
                    let off = (g[c] - w[c]).abs() / size;
                    assert!(off <= 2e-6, "{row:?} at {dt} s: {got:?} not {want:?}");
                }
            }
            if *row == fast && left == identity {
                let apart = angle(got, &want);
                assert!(apart <= 1e-4, "{apart} degrees apart at {dt} s");
            }
        }
    }
}
