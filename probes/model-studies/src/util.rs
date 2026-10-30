//! Small vector maths, RNG and image output shared by the studies.

pub type V3 = [f32; 3];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn mul(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn hadamard(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}
pub fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: V3) -> V3 {
    mul(a, 1.0 / dot(a, a).sqrt())
}

/// Orthonormal basis (t, b) around unit n (Duff et al. 2017).
pub fn basis(n: V3) -> (V3, V3) {
    let s = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n[2]);
    let b = n[0] * n[1] * a;
    ([1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]], [b, s + n[1] * n[1] * a, -n[1]])
}

/// PCG32.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(stream | 1));
        r.next_u32();
        r
    }
    pub fn next_u32(&mut self) -> u32 {
        let old = self.0;
        self.0 = old.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xs.rotate_right(rot)
    }
    /// Uniform in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16777216.0)
    }
    pub fn f64(&mut self) -> f64 {
        let hi = (self.next_u32() >> 5) as u64;
        let lo = (self.next_u32() >> 6) as u64;
        ((hi << 26) | lo) as f64 * (1.0 / 9007199254740992.0)
    }
}

pub fn hash(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

/// Roberts R3 low-discrepancy point i with a Cranley-Patterson shift.
pub fn r3(i: u32, shift: [f32; 3]) -> [f32; 3] {
    const G: f64 = 1.2207440846057596;
    let a = [1.0 / G, 1.0 / (G * G), 1.0 / (G * G * G)];
    let mut o = [0.0f32; 3];
    for k in 0..3 {
        let v = (0.5 + a[k] * i as f64 + shift[k] as f64).fract();
        o[k] = (v as f32).min(0.99999994);
    }
    o
}

/// Writes `w h` as u32 then RGB f32 rows.
pub fn write_image(path: &std::path::Path, w: usize, h: usize, rgb: &[V3]) {
    let mut out = Vec::with_capacity(8 + rgb.len() * 12);
    out.extend_from_slice(&(w as u32).to_le_bytes());
    out.extend_from_slice(&(h as u32).to_le_bytes());
    for p in rgb {
        for c in p {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    std::fs::write(path, out).unwrap();
}

pub fn out_dir(name: &str) -> std::path::PathBuf {
    let base = std::env::var("PROBE_OUT").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from("out"));
    let d = base.join(name);
    std::fs::create_dir_all(&d).unwrap();
    d
}
