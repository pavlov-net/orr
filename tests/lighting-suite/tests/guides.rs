//! Ray reconstruction guides on the bed's scenes: albedo, normal and roughness against the
//! materials, sky defaults, mirror chains, specular motion under camera motion, and the
//! sources of the history reset.

use lighting_suite::assert_clean;
use lighting_suite::bed::{
    Analytic, Bed, EXTENT, FLOOR, GuideImages, Item, MIRROR, Paths, SKY, SUN, Spec, look,
};
use orr_color::RgbPrimaries;
use orr_gpu::KernelSet;
use orr_lighting::{GuideKind, Light, LightingDebugView, PathEntry, ViewLighting};
use orr_math::{Mat3, Quat, Vec2, Vec3};

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), PathEntry::Fastest, settings)
}

/// One frame of `bed` through the opaque dispatch and the guide resolve; its guides.
fn guides(bed: &mut Bed) -> GuideImages {
    let p = paths(ViewLighting::default());
    bed.accumulate(1, 0, &mut |list, f| p.record(list, f));
    bed.guides()
}

fn rgb(p: [f32; 4]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// A material color as the G-buffer holds it: Rec.709 converted to Rec.2020.
fn rec2020(c: [f32; 3]) -> Vec3 {
    let m = RgbPrimaries::REC709.matrix_to(RgbPrimaries::REC2020);
    let c = c.map(f64::from);
    Vec3::from_array(m.map(|r| (r[0] * c[0] + r[1] * c[1] + r[2] * c[2]) as f32))
}

/// Within one 8-bit step and `slack` of `want` in every channel.
fn near(got: Vec3, want: Vec3, slack: f32) -> bool {
    (got - want).abs().max_element() <= 1.0 / 255.0 + slack
}

const UP_NORMAL: f32 = 1e-3;
const FLOOR_ROUGHNESS: f32 = 0.8;

#[test]
fn guides_hold_the_material_albedo_and_sky_defaults() {
    let mut bed = Bed::new(Analytic::SunAndGround);
    let g = guides(&mut bed);
    let (mut sky, mut floor, mut sphere) = (0, 0, 0);
    for i in 0..g.depth.len() {
        let (diffuse, specular) = (rgb(g.diffuse_albedo[i]), rgb(g.specular_albedo[i]));
        let n = g.normal_roughness[i];
        assert_eq!(g.specular_motion[i], g.motion[i], "pixel {i}");
        assert_eq!(
            g.motion[i], [0.0; 4],
            "a cut frame has no motion, pixel {i}"
        );
        if g.depth[i][0] == 0.0 {
            sky += 1;
            assert!(near(diffuse, Vec3::splat(0.5), 0.0), "sky {i}: {diffuse}");
            assert_eq!(specular, Vec3::ZERO, "sky {i}");
            assert_eq!(n, [0.0; 4], "sky {i}");
            continue;
        }
        let normal = rgb(n);
        assert!((normal.length() - 1.0).abs() < 2e-3, "pixel {i}: {normal}");
        if (normal - Vec3::Y).length() < UP_NORMAL {
            floor += 1;
            // The resolve widens roughness where neighbors' normals differ.
            assert!(n[3] >= FLOOR_ROUGHNESS - 1e-3, "floor {i}: {n:?}");
            assert!(specular.min_element() > 0.0, "floor {i}");
        } else {
            sphere += 1;
            // Diffuse albedo 1, no specular.
            assert!(near(diffuse, Vec3::ONE, 0.02), "sphere {i}: {diffuse}");
            assert_eq!(specular, Vec3::ZERO, "sphere {i}");
        }
    }
    assert!(
        sky > 100 && floor > 100 && sphere > 100,
        "{sky} {floor} {sphere}"
    );

    // The debug view shows the guide the target holds.
    let p = paths(ViewLighting {
        debug_view: LightingDebugView::Guide(GuideKind::DiffuseAlbedo),
        ..ViewLighting::default()
    });
    let shown = bed.accumulate(1, 0, &mut |list, f| p.record(list, f));
    for (i, s) in shown.mean.iter().enumerate() {
        let target = rgb(g.diffuse_albedo[i]);
        assert!(near(rgb(*s), target, 1e-3), "pixel {i}: {s:?} {target}");
        assert_eq!(s[3], if g.depth[i][0] > 0.0 { 1.0 } else { 0.0 });
    }
    assert_clean();
}

#[test]
fn cornell_box_walls_take_their_base_colors() {
    let mut bed = Bed::new(Analytic::CornellBox);
    let g = guides(&mut bed);
    let colors = [
        rec2020([0.73; 3]),
        rec2020([0.63, 0.065, 0.05]),
        rec2020([0.14, 0.45, 0.091]),
        Vec3::ZERO, // the emitter
    ];
    let mut counts = [0; 4];
    for (i, d) in g.diffuse_albedo.iter().enumerate() {
        if g.depth[i][0] == 0.0 {
            continue;
        }
        let d = rgb(*d);
        let k = colors
            .iter()
            .position(|&c| near(d, c, 0.02))
            .unwrap_or_else(|| panic!("pixel {i}: diffuse albedo {d} is no wall's"));
        counts[k] += 1;
    }
    assert!(counts[..3].iter().all(|&n| n > 100), "{counts:?}");
    assert_clean();
}

#[test]
fn subsurface_pixels_take_the_subsurface_color_share() {
    let mut bed = Bed::new(Analytic::SubsurfaceSlab);
    let g = guides(&mut bed);
    // The skin slab: subsurface weight 1 in this color; no diffuse lobe of its own.
    let color = rec2020([0.9, 0.6, 0.5]);
    let mut top = 0;
    for i in 0..g.depth.len() {
        let n = g.normal_roughness[i];
        let flat = (rgb(n) - Vec3::Y).length() < UP_NORMAL;
        // The skin's roughness is 0.3; the floor's 0.8 or more.
        if g.depth[i][0] == 0.0 || !flat || n[3] > 0.6 {
            continue;
        }
        top += 1;
        // What the specular layer does not reflect enters the subsurface lobe.
        let share = rgb(g.diffuse_albedo[i]) / color + rgb(g.specular_albedo[i]);
        assert!(near(share, Vec3::ONE, 0.04), "pixel {i}: {share}");
    }
    assert!(top > 100, "{top}");
    assert_clean();
}

/// A floor and a mirror tilted slightly down about x, under the sun and sky.
fn mirror_scene() -> (Spec, Mirror) {
    let mirror = Mirror {
        center: Vec3::new(0.0, 1.1, -1.0),
        linear: Mat3::from_rotation_x(0.05),
    };
    let spec = Spec {
        copies: vec![
            Item {
                model: FLOOR,
                linear: Mat3::IDENTITY,
                at: Vec3::ZERO,
            },
            Item {
                model: MIRROR,
                linear: mirror.linear,
                at: mirror.center,
            },
        ],
        lights: vec![Light::Directional(SUN)],
        environment: Some(SKY),
        eye: Vec3::new(0.0, 1.3, 2.5),
        target: Vec3::new(0.0, 1.0, -1.0),
        fov_y: 0.8,
        focus: None,
    };
    (spec, mirror)
}

/// The bed's mirror model placed: a 2 m square facing local +z.
struct Mirror {
    center: Vec3,
    linear: Mat3,
}

/// The floor model: a 10 m square at y = 0.
const FLOOR_HALF: f32 = 5.0;

/// What a camera ray reaches, analytically.
enum Ray {
    Floor,
    /// The mirror at `x1`, its reflection reaching the floor `t` beyond.
    MirrorToFloor {
        x1: Vec3,
        t: f32,
    },
    MirrorToSky,
    /// Too near an edge to tell from one sample.
    Edge,
    Sky,
}

impl Mirror {
    fn normal(&self) -> Vec3 {
        self.linear * Vec3::Z
    }

    /// Distance along `dir` from `o` to the mirror, if it lands inside its square shrunk by
    /// `margin`, and whether it lands within the square at all.
    fn hit(&self, o: Vec3, dir: Vec3, margin: f32) -> Option<(f32, bool)> {
        let n = self.normal();
        let t = (self.center - o).dot(n) / dir.dot(n);
        if t.is_nan() || t <= 0.0 {
            return None;
        }
        let local = self.linear.transpose() * (o + dir * t - self.center);
        let reach = local.x.abs().max(local.y.abs());
        (reach <= 1.0 + margin).then_some((t, reach <= 1.0 - margin))
    }
}

fn floor_hit(o: Vec3, dir: Vec3) -> Option<(f32, f32)> {
    let t = -o.y / dir.y;
    let p = o + dir * t;
    (t > 0.0).then_some((t, p.x.abs().max(p.z.abs())))
}

const MARGIN: f32 = 0.05;

fn trace(m: &Mirror, o: Vec3, dir: Vec3) -> Ray {
    let floor = floor_hit(o, dir);
    match m.hit(o, dir, MARGIN) {
        Some((t, inside)) if floor.is_none_or(|(tf, _)| t < tf) => {
            if !inside {
                return Ray::Edge;
            }
            let x1 = o + dir * t;
            let n = m.normal();
            let reflected = dir - 2.0 * dir.dot(n) * n;
            match floor_hit(x1, reflected) {
                Some((t, reach)) if reach < FLOOR_HALF - MARGIN => Ray::MirrorToFloor { x1, t },
                Some((_, reach)) if reach < FLOOR_HALF + MARGIN => Ray::Edge,
                _ => Ray::MirrorToSky,
            }
        }
        Some(_) => Ray::Edge,
        None => match floor {
            Some((_, reach)) if reach < FLOOR_HALF - MARGIN => Ray::Floor,
            Some((_, reach)) if reach < FLOOR_HALF + MARGIN => Ray::Edge,
            _ => Ray::Sky,
        },
    }
}

/// A bed camera: its eye, world-from-view rotation and projection.
struct Camera {
    eye: Vec3,
    rotation: Quat,
    f: f32,
    aspect: f32,
}

impl Camera {
    fn new(spec: &Spec) -> Self {
        Self {
            eye: spec.eye,
            rotation: look(spec.eye, spec.target),
            f: 1.0 / (0.5 * spec.fov_y).tan(),
            aspect: EXTENT[0] as f32 / EXTENT[1] as f32,
        }
    }

    /// The unit world direction through the centre of pixel `i`, and its ndc.
    fn pixel(&self, i: usize) -> (Vec3, Vec2) {
        let (x, y) = (i as u32 % EXTENT[0], i as u32 / EXTENT[0]);
        let ndc = Vec2::new(
            (x as f32 + 0.5) / EXTENT[0] as f32 * 2.0 - 1.0,
            (y as f32 + 0.5) / EXTENT[1] as f32 * 2.0 - 1.0,
        );
        let view = Vec3::new(ndc.x * self.aspect / self.f, -ndc.y / self.f, -1.0);
        ((self.rotation * view).normalize(), ndc)
    }

    /// The ndc of world point `p`, or of direction `p` at infinity with `point` false.
    fn ndc(&self, p: Vec3, point: bool) -> Vec2 {
        let v = self.rotation.inverse() * if point { p - self.eye } else { p };
        Vec2::new(self.f / self.aspect * v.x, -self.f * v.y) / -v.z
    }
}

#[test]
fn near_mirrors_take_the_guides_of_what_they_reflect() {
    let (spec, mirror) = mirror_scene();
    let camera = Camera::new(&spec);
    let mut bed = Bed::with(spec);
    let g = guides(&mut bed);
    let n = mirror.normal();
    let mirrored_up = Vec3::Y - 2.0 * Vec3::Y.dot(n) * n;
    let (mut floor_diffuse, mut floors) = (Vec3::ZERO, 0);
    let mut reflected = Vec::new();
    let (mut to_floor, mut to_sky) = (0, 0);
    for i in 0..g.depth.len() {
        let (dir, _) = camera.pixel(i);
        let diffuse = rgb(g.diffuse_albedo[i]);
        let nr = g.normal_roughness[i];
        match trace(&mirror, camera.eye, dir) {
            Ray::Floor => {
                floor_diffuse += diffuse;
                floors += 1;
            }
            Ray::MirrorToFloor { .. } => {
                to_floor += 1;
                assert!((rgb(nr) - mirrored_up).length() < 1e-2, "pixel {i}: {nr:?}");
                assert!((nr[3] - FLOOR_ROUGHNESS).abs() < 1e-3, "pixel {i}: {nr:?}");
                reflected.push((i, diffuse));
            }
            Ray::MirrorToSky => {
                to_sky += 1;
                // The sky's defaults through a white metal mirror.
                assert!(
                    near(diffuse, Vec3::splat(0.5), 0.02),
                    "pixel {i}: {diffuse}"
                );
                assert_eq!(rgb(g.specular_albedo[i]), Vec3::ZERO, "pixel {i}");
                assert_eq!(nr, [0.0; 4], "pixel {i}");
            }
            Ray::Edge | Ray::Sky => {}
        }
    }
    assert!(
        floors > 100 && to_floor > 100 && to_sky > 100,
        "{floors} {to_floor} {to_sky}"
    );
    let floor_diffuse = floor_diffuse / floors as f32;
    for (i, d) in reflected {
        assert!(
            near(d, floor_diffuse, 0.05),
            "pixel {i}: {d} against {floor_diffuse}"
        );
    }
    assert_clean();
}

#[test]
fn specular_motion_follows_the_mirror_image_under_camera_motion() {
    let (spec, mirror) = mirror_scene();
    let previous = Camera::new(&spec);
    let target = spec.target;
    let eye = spec.eye + Vec3::new(0.35, 0.15, -0.2);
    let mut bed = Bed::with(spec);
    let p = paths(ViewLighting::default());
    bed.accumulate(1, 0, &mut |list, f| p.record(list, f));
    bed.follow(eye, target, 1, &mut |list, f| p.record(list, f));
    let current = Camera::new(&bed.spec);
    let g = bed.guides();
    let (mut floors, mut to_floor, mut to_sky) = (0, 0, 0);
    let mut apart = 0;
    for i in 0..g.depth.len() {
        let (dir, ndc) = current.pixel(i);
        let motion = |m: [f32; 4]| Vec2::new(m[0], m[1]);
        let expected =
            |virtual_point: Vec3, point: bool| (previous.ndc(virtual_point, point) - ndc) * 0.5;
        let got = motion(g.specular_motion[i]);
        let want = match trace(&mirror, current.eye, dir) {
            Ray::Floor => {
                floors += 1;
                // Rough: the surface's own motion.
                assert_eq!(g.specular_motion[i], g.motion[i], "pixel {i}");
                continue;
            }
            // The virtual point lies along the camera ray, as far behind the mirror as the
            // reflection travels.
            Ray::MirrorToFloor { x1, t } => {
                to_floor += 1;
                expected(x1 + dir * t, true)
            }
            // A reflection of the sky moves as the sky does along the camera ray.
            Ray::MirrorToSky => {
                to_sky += 1;
                expected(dir, false)
            }
            Ray::Edge | Ray::Sky => continue,
        };
        assert!(
            (got - want).length() < 1e-3,
            "pixel {i}: {got} against {want}"
        );
        if (motion(g.motion[i]) - want).length() > 1e-2 {
            apart += 1;
        }
    }
    assert!(
        floors > 100 && to_floor > 100 && to_sky > 100,
        "{floors} {to_floor} {to_sky}"
    );
    // The mirror's own surface moves differently from its reflection.
    assert!(apart * 10 > (to_floor + to_sky) * 9, "{apart}");
    assert_clean();
}

/// One frame of `bed` from `eye` with no camera cut; whether the view's history resets.
fn history_frame(bed: &mut Bed, eye: Vec3, exposure_cut: bool, lighting: ViewLighting) -> bool {
    let target = bed.spec.target;
    let mut reset = None;
    bed.follow(eye, target, 0, &mut |_, f| {
        reset = Some(
            f.target
                .begin(f.frame, f.lighting, &f.desc, exposure_cut, &lighting),
        );
    });
    reset.unwrap()
}

#[test]
fn history_resets_on_each_of_its_sources() {
    let mut bed = Bed::new(Analytic::SunAndGround);
    let lighting = ViewLighting::default();
    let frame = history_frame;
    let eye = bed.spec.eye;
    assert!(
        frame(&mut bed, eye, false, lighting),
        "the view's first frame"
    );
    assert!(!frame(&mut bed, eye, false, lighting));
    assert!(frame(&mut bed, eye, true, lighting), "an exposure cut");
    let debug = ViewLighting {
        debug_view: LightingDebugView::Guide(GuideKind::Normal),
        ..lighting
    };
    assert!(
        !frame(&mut bed, eye + Vec3::X, false, debug),
        "a step under the jump distance, with another debug view"
    );
    let eye = eye + Vec3::new(3.5, 0.0, 0.0);
    assert!(frame(&mut bed, eye, false, lighting), "a jump");
    assert!(!frame(&mut bed, eye, false, lighting));
    let target = bed.spec.target;
    bed.spec.target = eye + Mat3::from_rotation_y(0.7) * (target - eye);
    assert!(
        frame(&mut bed, eye, false, lighting),
        "a turn past 30 degrees"
    );
    assert!(!frame(&mut bed, eye, false, lighting));
    bed.lighting.reset_history();
    assert!(frame(&mut bed, eye, false, lighting), "a lighting reset");
    assert!(
        !frame(&mut bed, eye, false, lighting),
        "a lighting reset lasts one frame"
    );
    // `accumulate` cuts the camera on its first frame.
    let mut reset = false;
    bed.accumulate(1, 0, &mut |_, f| {
        reset = f
            .target
            .begin(f.frame, f.lighting, &f.desc, false, &lighting);
    });
    assert!(reset, "a camera cut");
    assert_clean();
}
