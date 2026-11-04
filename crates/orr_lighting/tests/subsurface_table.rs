//! Regenerates `src/subsurface_table.rs` from walks in a semi-infinite slab: for walks still
//! inside after `cap` steps, the throughput that would yet leave through the boundary per unit
//! throughput at the cap. A boundary with an IOR above the outside's reflects what meets it by
//! Fresnel, each reflection one step, as a dielectric interior's walk does. Indexed by IOR, cap,
//! anisotropy and single-scattering albedo, albedo fastest; the axes are spaced where the values
//! bend.

use std::f64::consts::TAU;
use std::fmt::Write as _;
use std::thread;

const ALBEDOS: usize = 32;
const ANISOTROPIES: usize = 8;
const CAPS: [u32; 9] = [4, 8, 16, 32, 64, 128, 256, 512, 1024];
/// The anisotropy axis spans -`G_MAX` to `G_MAX`, evenly in atanh(g).
const G_MAX: f64 = 0.95;
const IORS: usize = 5;
/// The IOR axis spans 1 - 1/ior^2 evenly from 0 to `IOR_SHARE_MAX`.
const IOR_SHARE_MAX: f64 = 0.8;

const WALKS: u32 = 16384;
const MAX_STEPS: u32 = 4096;

/// Albedo index `i` holds 1 - s^4 for s = i / (ALBEDOS - 1).
fn albedo(i: usize) -> f64 {
    let s = i as f64 / (ALBEDOS - 1) as f64;
    1.0 - s.powi(4)
}

/// Anisotropy index `j` holds tanh(x) for x from -atanh(`G_MAX`) to atanh(`G_MAX`).
fn anisotropy(j: usize) -> f64 {
    let x = G_MAX.atanh();
    (-x + 2.0 * x * j as f64 / (ANISOTROPIES - 1) as f64).tanh()
}

/// IOR index `e`, against the outside's.
fn ior(e: usize) -> f64 {
    (1.0 - IOR_SHARE_MAX * e as f64 / (IORS - 1) as f64)
        .sqrt()
        .recip()
}

/// Fresnel reflectance from inside a medium of `ior` at cosine `c` to the boundary.
fn reflectance(ior: f64, c: f64) -> f64 {
    let s2 = (1.0 - c * c) * ior * ior;
    if s2 >= 1.0 {
        return 1.0;
    }
    let t = (1.0 - s2).sqrt();
    let rs = (ior * c - t) / (ior * c + t);
    let rp = (ior * t - c) / (ior * t + c);
    0.5 * (rs * rs + rp * rp)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The cosine to the slab's normal of a direction scattered by Henyey-Greenstein `g` from
/// one at cosine `mu`.
fn scatter(mu: f64, g: f64, rng: &mut Rng) -> f64 {
    let u = rng.next();
    let c = if g.abs() < 1e-3 {
        1.0 - 2.0 * u
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        ((1.0 + g * g - s * s) / (2.0 * g)).clamp(-1.0, 1.0)
    };
    let sine = (1.0 - c * c).max(0.0).sqrt();
    let phi = TAU * rng.next();
    (mu * c + (1.0 - mu * mu).max(0.0).sqrt() * sine * phi.cos()).clamp(-1.0, 1.0)
}

/// The values at every cap for albedo `a`, anisotropy `g` and boundary `ior`. Each flight adds
/// the share that would leave through the boundary and carries on with the rest, reflected
/// at the boundary or with its length drawn short of it.
fn cell(a: f64, g: f64, ior: f64, seed: u64) -> [f32; CAPS.len()] {
    let mut rng = Rng(seed);
    // Throughput leaving within each cap, and throughput still inside after it.
    let mut left = [0.0f64; CAPS.len()];
    let mut inside = [0.0f64; CAPS.len()];
    let mut total = 0.0;
    for _ in 0..WALKS {
        // A refracting surface is entered from every angle of view, a sphere's as seen from afar.
        let entry = if ior > 1.0 {
            -(1.0 - rng.next() / (ior * ior)).sqrt()
        } else {
            -1.0
        };
        let (mut depth, mut mu, mut throughput) = (0.0f64, entry, 1.0f64);
        let mut out = 0.0;
        let mut next = 0;
        for step in 1..=MAX_STEPS {
            let mut t = -(1.0 - rng.next()).ln();
            let mut reflected = false;
            if mu > 0.0 {
                let reach = (-depth / mu).exp();
                let leave = reach * (1.0 - reflectance(ior, mu));
                out += throughput * leave;
                throughput *= 1.0 - leave;
                reflected = rng.next() * (1.0 - leave) < reach - leave;
                t = -(1.0 - rng.next() * (1.0 - reach)).ln();
            }
            if reflected {
                depth = 0.0;
                mu = -mu;
            } else {
                depth -= mu * t;
                throughput *= a;
                mu = scatter(mu, g, &mut rng);
            }
            if next < CAPS.len() && step == CAPS[next] {
                left[next] += out;
                inside[next] += throughput;
                next += 1;
            }
            if throughput < 1e-4 {
                break;
            }
        }
        // What is still inside at the last step leaves after as many steps again.
        total += out + throughput * a.powi(MAX_STEPS as i32);
        for l in &mut left[next..] {
            *l += out;
        }
    }
    std::array::from_fn(|k| {
        if inside[k] > 0.0 {
            ((total - left[k]) / inside[k]) as f32
        } else {
            0.0
        }
    })
}

/// The values, `IORS` x `CAPS.len()` x `ANISOTROPIES` x `ALBEDOS`.
fn table() -> Vec<f32> {
    let cells: Vec<(usize, usize, usize)> = (0..IORS)
        .flat_map(|e| (0..ANISOTROPIES).flat_map(move |j| (0..ALBEDOS).map(move |i| (e, j, i))))
        .collect();
    let threads = thread::available_parallelism().map_or(4, usize::from);
    let per = cells.len().div_ceil(threads);
    let cell_values: Vec<[f32; CAPS.len()]> = thread::scope(|scope| {
        let handles: Vec<_> = cells
            .chunks(per)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&(e, j, i)| {
                            let seed = (e * ANISOTROPIES + j) * ALBEDOS + i;
                            cell(albedo(i), anisotropy(j), ior(e), seed as u64)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("simulation thread"))
            .collect()
    });
    let per_ior = ANISOTROPIES * ALBEDOS;
    let mut out = vec![0.0; CAPS.len() * cells.len()];
    for (c, values) in cell_values.iter().enumerate() {
        let (e, cell) = (c / per_ior, c % per_ior);
        for (k, &v) in values.iter().enumerate() {
            out[(e * CAPS.len() + k) * per_ior + cell] = v;
        }
    }
    out
}

/// `v` to 5 significant digits, far below the simulation's noise.
fn rounded(v: f32) -> f32 {
    format!("{v:.4e}").parse().unwrap()
}

/// The text of `src/subsurface_table.rs`.
fn source() -> String {
    let caps = CAPS.map(|c| c.to_string()).join(", ");
    let mut s = format!(
        "//! What a subsurface walk stopped at its step cap would yet carry out through the surface, per
//! unit throughput at the cap, in a semi-infinite medium. Indexed by surface IOR, cap, anisotropy
//! and single-scattering albedo, one row per IOR, cap and anisotropy; `subsurface.slang` gives
//! the axes.
//!
//! Made by simulation: {WALKS} seeded walks per entry, rounded to 5 significant digits.
//! `tests/subsurface_table.rs` holds the simulator, checks this file against it and rewrites it
//! with `ORR_BLESS=1`.

pub(crate) const CAPS: [u32; {}] = [{caps}];
pub(crate) const ALBEDOS: u32 = {ALBEDOS};
pub(crate) const ANISOTROPIES: u32 = {ANISOTROPIES};
pub(crate) const G_MAX: f32 = {:?};
pub(crate) const IORS: u32 = {IORS};
pub(crate) const IOR_SHARE_MAX: f32 = {:?};

#[rustfmt::skip]
#[allow(clippy::approx_constant)]
pub(crate) static VALUES: [f32; {}] = [
",
        CAPS.len(),
        G_MAX as f32,
        IOR_SHARE_MAX as f32,
        IORS * CAPS.len() * ANISOTROPIES * ALBEDOS,
    );
    for row in table().chunks(ALBEDOS) {
        s.push_str("   ");
        for &v in row {
            write!(s, " {:?},", rounded(v)).unwrap();
        }
        s.push('\n');
    }
    s.push_str("];\n");
    s
}

#[test]
#[ignore = "runs the simulation, about a minute; ORR_BLESS=1 rewrites src/subsurface_table.rs"]
fn the_committed_table_matches_the_simulation() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/subsurface_table.rs");
    let generated = source();
    if std::env::var_os("ORR_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(path, generated).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(path).unwrap();
    let different = generated
        .lines()
        .zip(committed.lines())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| generated.lines().count().min(committed.lines().count()));
    assert!(
        generated == committed,
        "{path} differs from the simulation (first at line {different}); ORR_BLESS=1 rewrites it"
    );
}
