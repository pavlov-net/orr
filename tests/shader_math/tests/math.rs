use orr_math::{Affine3, DMat3, DVec3, Mat3, Quat, Vec3};
use orr_pack::AffineRows;
use shader_math::stats::Rng;
use shader_math::*;

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

fn rows(m: &Affine3) -> [f32; 12] {
    bytemuck::cast(AffineRows::from(*m).rows)
}

fn random_affine(rng: &mut Rng) -> Affine3 {
    let q = Quat::from_array([
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ])
    .normalize();
    let mut scale = Vec3::new(
        rng.range(0.2, 5.0),
        rng.range(0.2, 5.0),
        rng.range(0.2, 5.0),
    );
    if rng.f32() < 0.25 {
        scale.y = -scale.y;
    }
    let t = Vec3::new(
        rng.range(-100.0, 100.0),
        rng.range(-100.0, 100.0),
        rng.range(-100.0, 100.0),
    );
    Affine3::from_scale_rotation_translation(scale, q, t)
}

#[test]
fn orthonormal_basis_is_right_handed_everywhere() {
    let mut rng = Rng::new(10);
    let mut normals: Vec<[f32; 3]> = (0..100_000).map(|_| rng.unit()).collect();
    for z in [-1.0f32, -0.999_999_9, -0.99999, -0.0, 0.0, 1.0] {
        let r = (1.0 - z * z).max(0.0).sqrt();
        normals.push([r, 0.0, z]);
        normals.push([0.0, -r, z]);
    }
    for n in normals {
        let (t, b) = orthonormal_basis(n);
        let (t, b, n) = (Vec3::from(t), Vec3::from(b), Vec3::from(n));
        assert!(
            (t.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5,
            "{n}: {t} {b}"
        );
        assert!(
            t.dot(b).abs() < 1e-5 && t.dot(n).abs() < 1e-5 && b.dot(n).abs() < 1e-5,
            "{n}: {t} {b}"
        );
        assert!(t.cross(b).distance(n) < 1e-5, "{n}: {t} {b}");
    }
}

#[test]
fn quaternions_and_affines_match_glam() {
    let mut rng = Rng::new(11);
    for _ in 0..20_000 {
        let qa = Quat::from_array([
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ])
        .normalize();
        let v = Vec3::new(
            rng.range(-10.0, 10.0),
            rng.range(-10.0, 10.0),
            rng.range(-10.0, 10.0),
        );
        assert!(close(
            &quat_to_matrix(qa.to_array()),
            &Mat3::from_quat(qa).transpose().to_cols_array(),
            1e-6
        ));

        let (a, b) = (random_affine(&mut rng), random_affine(&mut rng));
        let (ra, rb) = (rows(&a), rows(&b));
        assert!(close(&affine_mul(ra, rb), &rows(&(a * b)), 1e-5));
        assert!(close(
            &transform_point(ra, v.into()),
            &a.transform_point3(v).to_array(),
            1e-5
        ));
        assert!(close(
            &transform_vector(ra, v.into()),
            &a.transform_vector3(v).to_array(),
            1e-5
        ));
        let n = Vec3::from(rng.unit());
        let want = (a.matrix3.inverse().transpose() * n).normalize();
        assert!(
            close(&transform_normal(ra, n.into()), &want.to_array(), 1e-4),
            "{a:?} {n}"
        );
    }
}

/// The largest singular value of `m`, by power iteration on `m^T m`.
fn largest_singular_value(m: Mat3) -> f64 {
    let m = m.as_dmat3();
    let a: DMat3 = m.transpose() * m;
    let mut v = DVec3::new(0.6, 0.5, 0.62).normalize();
    for _ in 0..1000 {
        v = (a * v).normalize();
    }
    (m * v).length()
}

fn row_major(m: Mat3) -> [f32; 9] {
    m.transpose().to_cols_array()
}

#[test]
fn largest_scale_bounds_the_stretch_of_a_matrix() {
    let mut rng = Rng::new(0x51ea);
    let shear =
        |x: f32, y: f32| Mat3::from_cols(Vec3::X, Vec3::new(x, 1.0, 0.0), Vec3::new(0.0, y, 1.0));
    let mut cases = vec![
        shear(1.0, 0.0),
        shear(4.0, -3.0),
        Mat3::IDENTITY * 1e-25,
        shear(1.0, 1.0) * 1e-25,
        shear(1.0, 1.0) * 1e20,
        Mat3::IDENTITY * 3e38,
    ];
    for _ in 0..1000 {
        let a = random_affine(&mut rng).matrix3;
        cases.push(a * shear(rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)));
    }
    for m in cases {
        let got = f64::from(largest_scale(row_major(m)));
        let want = largest_singular_value(m);
        assert!(
            got >= want * (1.0 - 1e-5) && got <= want * 1.32,
            "{m}: {got} against {want}"
        );
    }
    assert_eq!(largest_scale([0.0; 9]), 0.0);
    let subnormal = shear(1.0, 1.0) * 1e-40;
    assert!(f64::from(largest_scale(row_major(subnormal))) >= largest_singular_value(subnormal));
    // The columns of a copy that does not shear are orthogonal: the bound is the longest.
    for _ in 0..1000 {
        let a = random_affine(&mut rng).matrix3;
        let want = a
            .x_axis
            .length()
            .max(a.y_axis.length())
            .max(a.z_axis.length());
        let got = largest_scale(row_major(a));
        assert!((got / want - 1.0).abs() < 1e-5, "{a}: {got} against {want}");
    }
}

#[test]
fn linear_inverse_holds_at_any_uniform_size() {
    let m = Mat3::from_cols(
        Vec3::new(1.0, 0.25, 0.0),
        Vec3::new(0.5, 1.0, 0.25),
        Vec3::new(0.0, -0.5, 2.0),
    );
    for size in [1e-30, 1e-20, 1e-13, 1.0, 1e13, 1e20, 1.5e38] {
        let s = m * size;
        let inverse = Mat3::from_cols_array(&linear_inverse(row_major(s))).transpose();
        let one = inverse * s;
        assert!(one.abs_diff_eq(Mat3::IDENTITY, 1e-5), "{size}: {one}");
    }
}
