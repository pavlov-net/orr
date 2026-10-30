use glam::{Affine3A, Mat3, Quat, Vec3};
use shader_maths::stats::Rng;
use shader_maths::*;

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

fn rows(m: &Affine3A) -> [f32; 12] {
    let r = Mat3::from(m.matrix3);
    let t = m.translation;
    [
        r.row(0).x,
        r.row(0).y,
        r.row(0).z,
        t.x,
        r.row(1).x,
        r.row(1).y,
        r.row(1).z,
        t.y,
        r.row(2).x,
        r.row(2).y,
        r.row(2).z,
        t.z,
    ]
}

fn random_affine(rng: &mut Rng) -> Affine3A {
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
    Affine3A::from_scale_rotation_translation(scale, q, t)
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
        let qb = Quat::from_array([
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
            &quat_mul(qa.to_array(), qb.to_array()),
            &(qa * qb).to_array(),
            1e-6
        ));
        assert!(close(
            &quat_rotate(qa.to_array(), v.into()),
            &(qa * v).to_array(),
            1e-5
        ));
        assert!(close(
            &quat_to_matrix(qa.to_array()),
            &Mat3::from_quat(qa).transpose().to_cols_array(),
            1e-6
        ));

        let (a, b) = (random_affine(&mut rng), random_affine(&mut rng));
        let (ra, rb) = (rows(&a), rows(&b));
        assert!(close(&affine_mul(ra, rb), &rows(&(a * b)), 1e-5));
        assert!(close(&affine_inverse(ra), &rows(&a.inverse()), 1e-4));
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
        let want = (Mat3::from(a.matrix3).inverse().transpose() * n).normalize();
        assert!(
            close(&transform_normal(ra, n.into()), &want.to_array(), 1e-4),
            "{a:?} {n}"
        );
    }
}

#[test]
fn ray_origin_offset_leaves_the_surface_by_a_relative_step() {
    let mut rng = Rng::new(12);
    for _ in 0..100_000 {
        let scale = 10f32.powf(rng.range(-4.0, 5.0));
        let p = Vec3::from(rng.unit()) * scale * rng.f32();
        let n = Vec3::from(rng.unit());
        let o = Vec3::from(offset_ray_origin(p.into(), n.into()));
        let moved = o - p;
        assert!(moved.dot(n) > 0.0, "p {p} n {n} -> {o}");
        // At most 256 ulps per axis away from the origin, a fixed step near it.
        let bound = p.abs().max(Vec3::splat(1.0 / 32.0)) * (256.0 * f32::EPSILON)
            + Vec3::splat(1.0 / 65536.0);
        assert!(moved.abs().cmple(bound).all(), "p {p} n {n} moved {moved}");
    }
}
