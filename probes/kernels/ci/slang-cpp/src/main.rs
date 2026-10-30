//! Calls the Slang-emitted C++ (OpenPBR wrapper and octahedral packing) and checks them against
//! the OpenPBR C++ reference and a Rust packing reference. Exits non-zero on a mismatch.

unsafe extern "C" {
    fn orr_test_openpbr_eval(out3: *mut f32, r: f32, g: f32, b: f32, rough: f32, ox: f32, oy: f32, oz: f32, ix: f32, iy: f32, iz: f32);
    fn ref_openpbr_eval(out3: *mut f32, r: f32, g: f32, b: f32, rough: f32, ox: f32, oy: f32, oz: f32, ix: f32, iy: f32, iz: f32);
    fn orr_test_pack_oct(x: f32, y: f32, z: f32) -> u32;
}

fn pack_oct(n: [f32; 3]) -> u32 {
    let s = n[0].abs() + n[1].abs() + n[2].abs();
    let (x, y, z) = (n[0] / s, n[1] / s, n[2] / s);
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let e = if z >= 0.0 { [x, y] } else { [(1.0 - y.abs()) * sign(x), (1.0 - x.abs()) * sign(y)] };
    let q = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0).round() as u32;
    q(e[0]) | (q(e[1]) << 16)
}

fn main() {
    let mut state = 12345u32;
    let mut rnd = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 8) as f32 / 16_777_216.0
    };
    let (mut max_rel, mut evals) = (0.0f32, 0);
    for _ in 0..1000 {
        let base = [rnd(), rnd(), rnd()];
        let rough = rnd();
        let wo = [rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() + 0.05];
        let wi = [rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() + 0.05];
        let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
        unsafe {
            orr_test_openpbr_eval(a.as_mut_ptr(), base[0], base[1], base[2], rough, wo[0], wo[1], wo[2], wi[0], wi[1], wi[2]);
            ref_openpbr_eval(b.as_mut_ptr(), base[0], base[1], base[2], rough, wo[0], wo[1], wo[2], wi[0], wi[1], wi[2]);
        }
        for c in 0..3 {
            max_rel = max_rel.max((a[c] - b[c]).abs() / b[c].abs().max(1e-3));
        }
        evals += 1;
    }
    let mut pack_bad = 0;
    for _ in 0..10000 {
        let n = [rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0];
        if unsafe { orr_test_pack_oct(n[0], n[1], n[2]) } != pack_oct(n) {
            pack_bad += 1;
        }
    }
    println!("openpbr eval: {evals} cases, Slang C++ vs C++ reference max_rel={max_rel:.2e}");
    println!("pack_oct: 10000 normals, mismatches vs Rust = {pack_bad}");
    let ok = max_rel < 1e-4 && pack_bad == 0;
    println!("RESULT {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
