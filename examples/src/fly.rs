//! A camera flown with the keyboard and the mouse, and written down as a `--view` value.

use orr_math::{Quat, Vec3};

use crate::window::{KeyCode, Window};

/// How many times faster shift moves.
const RUN_SCALE: f32 = 8.0;
/// Radians per device unit of mouse motion.
const LOOK_SCALE: f32 = 0.0025;

/// A named camera.
#[derive(Clone, Copy, PartialEq)]
pub struct Camera {
    pub name: &'static str,
    pub position: Vec3,
    /// Yaw about +Y and pitch, in radians; yaw 0 looks down -Z.
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn looking(name: &'static str, eye: Vec3, at: Vec3, fov_y_degrees: f32) -> Self {
        let d = (at - eye).normalize();
        Camera {
            name,
            position: eye,
            yaw: (-d.x).atan2(-d.z),
            pitch: d.y.asin(),
            fov_y: fov_y_degrees.to_radians(),
        }
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    /// `--view`'s value: `x,y,z,yaw,pitch,fov`, the angles in degrees.
    pub fn parse(view: &str) -> Self {
        let v: Vec<f32> = view
            .split(',')
            .map(|x| x.trim().parse().expect("a --view number"))
            .collect();
        let [x, y, z, yaw, pitch, fov] = v[..] else {
            panic!("--view takes x,y,z,yaw,pitch,fov");
        };
        Camera {
            name: "view",
            position: Vec3::new(x, y, z),
            yaw: yaw.to_radians(),
            pitch: pitch.to_radians(),
            fov_y: fov.to_radians(),
        }
    }

    /// This view as a `--view` value.
    pub fn view(&self) -> String {
        let p = self.position;
        format!(
            "{:.3},{:.3},{:.3},{:.2},{:.2},{:.2}",
            p.x,
            p.y,
            p.z,
            self.yaw.to_degrees(),
            self.pitch.to_degrees(),
            self.fov_y.to_degrees()
        )
    }

    /// Moves by the held keys over `dt` seconds at `speed` m/s and turns by the mouse: WASD
    /// moves, Q and E sink and rise, shift moves faster and dragging with the right mouse button
    /// looks around.
    pub fn fly(&mut self, window: &mut Window, speed: f32, dt: f32) {
        let [x, y] = window.take_look();
        self.yaw -= x as f32 * LOOK_SCALE;
        self.pitch = (self.pitch - y as f32 * LOOK_SCALE)
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        let r = self.rotation();
        let walk = r * Vec3::NEG_Z * window.axis(KeyCode::KeyW, KeyCode::KeyS)
            + r * Vec3::X * window.axis(KeyCode::KeyD, KeyCode::KeyA)
            + Vec3::Y * window.axis(KeyCode::KeyE, KeyCode::KeyQ);
        let run = window.held(KeyCode::ShiftLeft) || window.held(KeyCode::ShiftRight);
        let speed = if run { speed * RUN_SCALE } else { speed };
        self.position += walk.normalize_or_zero() * speed * dt;
    }
}

/// Logs `camera` as a `--view` value and appends it to `{example}-views.txt` beside the
/// executable.
pub fn note_view(example: &str, camera: &Camera) {
    use std::io::Write as _;
    let view = camera.view();
    log::info!("--view {view}");
    let path = std::env::current_exe()
        .expect("executable path")
        .with_file_name(format!("{example}-views.txt"));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path);
    match file.and_then(|mut f| writeln!(f, "{view}")) {
        Ok(()) => {}
        Err(e) => log::warn!("{}: {e}", path.display()),
    }
}
