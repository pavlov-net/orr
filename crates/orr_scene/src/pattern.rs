//! The tessellation pattern table, which the build script writes out for the GPU.
//!
//! A pattern dices a triangle with odd segment counts `[a, b, c]` on its edges v0-v1, v1-v2
//! and v2-v0, each from 1 to `MAX_DICE`. Each edge has a fractional factor `t` (fixed point,
//! `FACTOR_ONE` per segment) whose count `n` is the smallest odd number at or above it: the
//! edge has `n - 2` segments of length `1 / t` and two shorter ones beside the middle
//! segment, which shrink to nothing as `t` falls to `n - 2`, where the pattern turns into the
//! one with `n - 2` segments. Points move continuously as factors change, and an edge's
//! points depend only on its own factor. Inside lie rings spaced the same way at the largest
//! factor: ring `r` has `n - 2r` segments per side and the innermost is one triangle. The
//! outer edges stitch to ring 1 as if they had ring 1's count plus two, with the extra
//! points collapsed into the edge's middle segment.
//!
//! The table holds topology only: per vertex a key (`KEY_*`) naming its ring, side and
//! index, from which `weights` places it at given factors. Weights are barycentrics in units
//! of `1 / BARY_ONE`, computed in integers, so the GPU and the CPU agree bit for bit, and
//! edge points are quantized from whichever end is nearer, so the patch on the other side
//! of an edge, walking it the other way, gives its two corners the same weights. Subpatches
//! of splits nest, and near a change of count they are thin, so weights have 30 bits.

use std::collections::HashMap;

/// The largest edge count: the pattern with 9 segments on every edge has 121 triangles.
const MAX_DICE: u32 = 9;
/// Barycentric 1.
pub const BARY_ONE: u32 = 1 << 30;
/// A pattern's own weights are multiples of this.
const PATTERN_STEP: u32 = 1 << 6;
/// Edge factor 1 in fixed point.
const FACTOR_ONE: u32 = 4096;
/// Odd counts per edge.
const COUNTS: u32 = MAX_DICE.div_ceil(2);
/// Patterns in the table, one per triple of odd counts.
pub(crate) const PATTERNS: usize = (COUNTS * COUNTS * COUNTS) as usize;

/// Vertex keys: kind in bits 0-1, corner, edge or side in bits 2-3, ring in bits 4-7 and
/// the index along the edge or side from bit 8.
const KEY_CORNER: u32 = 0;
const KEY_EDGE: u32 = 1;
const KEY_RING: u32 = 2;

/// One pattern of the table: its counts, a key per vertex and triangles of cluster-local
/// corners. A generated cluster's vertex `i` lies at pattern vertex `i` of its subpatch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    pub counts: [u32; 3],
    pub keys: Vec<u32>,
    pub triangles: Vec<[u8; 3]>,
}

impl Pattern {
    /// Each vertex's barycentrics at fixed-point edge `factors`, whose counts are this
    /// pattern's.
    pub fn weights(&self, factors: [u32; 3]) -> Vec<[u32; 3]> {
        assert_eq!(
            factors.map(factor_count),
            self.counts,
            "factors {factors:?}"
        );
        self.keys.iter().map(|&k| weights(k, factors)).collect()
    }

    /// Each vertex's barycentrics in the patch triangle when the pattern dices a subpatch
    /// with `corners` (`w1, w2` each) and corner `zeros` (the tessellator's `zeros`
    /// words), as the tessellator places them.
    pub fn place(&self, factors: [u32; 3], corners: [u32; 6], zeros: u32) -> Vec<[u32; 3]> {
        let w = self.weights(factors);
        let corners = [0, 1, 2].map(|i| {
            let (w1, w2) = (corners[2 * i], corners[2 * i + 1]);
            [BARY_ONE - w1 - w2, w1, w2]
        });
        self.keys
            .iter()
            .zip(w)
            .map(|(&k, w)| place(w, corners, vertex_zeros(k, zeros)))
            .collect()
    }
}

/// The zero weights of the pattern vertex with `key` from its subpatch's corner `zeros`:
/// a corner's own, those shared by both ends of the subpatch edge it lies on, else none.
fn vertex_zeros(key: u32, zeros: u32) -> u32 {
    let e = (key >> 2) & 3;
    let corner = |c: u32| (zeros >> (3 * c)) & 7;
    match key & 3 {
        KEY_CORNER => corner(e),
        KEY_EDGE => corner(e) & corner((e + 1) % 3),
        _ => 0,
    }
}

/// `s / BARY_ONE` rounded so that r(s) + r(BARY_ONE^2 - s) == BARY_ONE.
fn round_symmetric(s: u64) -> u32 {
    const ONE: u64 = BARY_ONE as u64;
    if s <= ONE / 2 * ONE {
        ((s + ONE / 2) / ONE) as u32
    } else {
        (ONE - (ONE * ONE - s + ONE / 2) / ONE) as u32
    }
}

/// The point at subpatch weights `w` in the patch triangle's weights: composed with the
/// subpatch's `corners`, then each weight that is not zero by construction (`zeros`) but
/// rounds to zero takes a unit from the largest, so only points on a patch edge lie on it.
fn place(w: [u32; 3], corners: [[u32; 3]; 3], zeros: u32) -> [u32; 3] {
    let s = |k: usize| {
        (0..3)
            .map(|i| u64::from(w[i]) * u64::from(corners[i][k]))
            .sum()
    };
    let (w1, w2) = (round_symmetric(s(1)), round_symmetric(s(2)));
    let mut out = [BARY_ONE - w1 - w2, w1, w2];
    for j in 0..3 {
        if out[j] == 0 && zeros & 1 << j == 0 {
            let most = (0..3).fold(0, |m, k| if out[k] > out[m] { k } else { m });
            out[most] -= 1;
            out[j] = 1;
        }
    }
    out
}

/// The odd segment count of fixed-point factor `q`, at least [`FACTOR_ONE`].
fn factor_count(q: u32) -> u32 {
    2 * ((q + FACTOR_ONE - 1) / (2 * FACTOR_ONE)) + 1
}

/// The counts of table entry `i`.
pub(crate) fn counts(i: usize) -> [u32; 3] {
    let i = i as u32;
    [i / (COUNTS * COUNTS), i / COUNTS % COUNTS, i % COUNTS].map(|x| 2 * x + 1)
}

/// Segment lengths at factor `q` of count `n`: full, then short.
fn lengths(n: u32, q: u32) -> (u64, u64) {
    let short = if n >= 3 { q - (n - 2) * FACTOR_ONE } else { 0 };
    (2 * u64::from(FACTOR_ONE), u64::from(short))
}

/// Where point `k` of `c` segments lies, as a numerator and the total length.
fn along(c: u32, k: u32, (full, short): (u64, u64)) -> (u64, u64) {
    if c < 3 {
        return (u64::from(k) * full, full);
    }
    let shorts = u64::from(k > (c - 3) / 2) + u64::from(k > c.div_ceil(2));
    (
        (u64::from(k) - shorts) * full + shorts * short,
        u64::from(c - 2) * full + 2 * short,
    )
}

/// `num / den` in multiples of `PATTERN_STEP`, rounded half up.
fn scaled(num: u64, den: u64) -> u32 {
    let one = u64::from(BARY_ONE / PATTERN_STEP);
    ((2 * one * num + den) / (2 * den)) as u32 * PATTERN_STEP
}

/// The weight point `s` of an edge with factor `q` gives the edge's far corner, quantized
/// from whichever end is nearer.
fn edge_weight(s: u32, q: u32) -> u32 {
    let n = factor_count(q);
    let l = lengths(n, q);
    if 2 * s < n {
        let (num, total) = along(n, s, l);
        scaled(num, total)
    } else {
        let (num, total) = along(n, n - s, l);
        BARY_ONE - scaled(num, total)
    }
}

/// The barycentrics of the vertex with `key` at fixed-point edge `factors`.
fn weights(key: u32, factors: [u32; 3]) -> [u32; 3] {
    let e = ((key >> 2) & 3) as usize;
    let r = (key >> 4) & 15;
    let k = key >> 8;
    let mut w = [0u32; 3];
    match key & 3 {
        KEY_CORNER => w[e] = BARY_ONE,
        KEY_EDGE => {
            let x = edge_weight(k, factors[e]);
            w[e] = BARY_ONE - x;
            w[(e + 1) % 3] = x;
        }
        _ => {
            let q = factors.into_iter().max().expect("three factors");
            let n = factor_count(q);
            let l = lengths(n, q);
            // Ring r is the triangle scaled about its center by 1 - 2 depth / total.
            let (depth, total) = along(n, r, l);
            let (a, d) = (2 * depth, 3 * total);
            let (m, side) = along(n - 2 * r, k, l);
            let mut x = [0u64; 3];
            x[e] = (d - 2 * a) * (side - m) + a * m;
            x[(e + 1) % 3] = a * (side - m) + (d - 2 * a) * m;
            x[(e + 2) % 3] = a * side;
            let den = d * side;
            let step = PATTERN_STEP;
            let w1 = scaled(x[1], den).clamp(step, BARY_ONE - 2 * step);
            let w2 = scaled(x[2], den).clamp(step, BARY_ONE - step - w1);
            w = [BARY_ONE - w1 - w2, w1, w2];
        }
    }
    w
}

fn corner_key(c: usize) -> u32 {
    KEY_CORNER | (c as u32) << 2
}

/// The pattern of odd edge counts `n`.
pub(crate) fn generate(n: [u32; 3]) -> Pattern {
    assert!(
        n.iter().all(|&c| c % 2 == 1 && c <= MAX_DICE),
        "counts {n:?}"
    );
    let inside = n.into_iter().max().expect("three counts");
    let rings = (inside - 1) / 2;
    // Point s of edge e, from corner e.
    let edge = |e: usize, s: u32| {
        if s == 0 {
            corner_key(e)
        } else if s == n[e] {
            corner_key((e + 1) % 3)
        } else {
            KEY_EDGE | (e as u32) << 2 | s << 8
        }
    };
    // Point k of side e of ring r, from the ring's corner e.
    let ring = |r: u32, e: usize, k: u32| {
        let (e, k) = if k == inside - 2 * r {
            ((e + 1) % 3, 0)
        } else {
            (e, k)
        };
        KEY_RING | (e as u32) << 2 | r << 4 | k << 8
    };

    let mut keys: Vec<u32> = Vec::from_fn(3, corner_key);
    let mut at: HashMap<u32, u8> = keys
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i as u8))
        .collect();
    let mut triangles = Vec::new();
    let mut push = |t: [u32; 3]| {
        let t = t.map(|key| {
            *at.entry(key).or_insert_with(|| {
                keys.push(key);
                (keys.len() - 1) as u8
            })
        });
        if t[0] != t[1] && t[1] != t[2] && t[2] != t[0] {
            triangles.push(t);
        }
    };
    if inside == 1 {
        push([0, 1, 2].map(corner_key));
    }
    // The strip between a side of `v` segments and the next ring's side of `v - 2`: corner
    // triangles at both ends and quads between, their diagonals mirrored about the middle
    // quad, so that collapsing the two quads beside it leaves the strip of `v - 2`.
    let mut strip = |v: u32, outer: &dyn Fn(u32) -> u32, inner: &dyn Fn(u32) -> u32| {
        push([outer(0), outer(1), inner(0)]);
        for j in 0..v - 2 {
            let (a, b, c, d) = (outer(j + 1), outer(j + 2), inner(j + 1), inner(j));
            if j <= (v - 3) / 2 {
                push([a, b, c]);
                push([a, c, d]);
            } else {
                push([a, b, d]);
                push([b, c, d]);
            }
        }
        push([outer(v - 1), outer(v), inner(v - 2)]);
    };
    for (e, &count) in n.iter().enumerate() {
        if inside > 1 {
            // An outer edge of `count` segments as one of `inside` with the extra points
            // collapsed into its middle segment.
            let h = (count - 1) / 2;
            let mid = (inside - 1) / 2;
            let virtual_point = |k: u32| {
                if k <= mid {
                    edge(e, k.min(h))
                } else {
                    edge(e, count - (inside - k).min(h))
                }
            };
            strip(inside, &virtual_point, &|j| ring(1, e, j));
        }
        for r in 1..rings {
            strip(inside - 2 * r, &|k| ring(r, e, k), &|j| ring(r + 1, e, j));
        }
    }
    if rings > 0 {
        push([0, 1, 2].map(|e| ring(rings, e, 0)));
    }
    Pattern {
        counts: n,
        keys,
        triangles,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// The table index of odd edge counts `n`.
    fn index(n: [u32; 3]) -> usize {
        (((n[0] / 2) * COUNTS + n[1] / 2) * COUNTS + n[2] / 2) as usize
    }

    fn table() -> Vec<Pattern> {
        Vec::from_fn(PATTERNS, |i| generate(counts(i)))
    }

    /// Twice the signed area of triangle `t` in barycentric units.
    fn area(w: &[[u32; 3]], t: [u8; 3]) -> i64 {
        let v = t.map(|c| w[c as usize].map(i64::from));
        (v[1][1] - v[0][1]) * (v[2][2] - v[0][2]) - (v[2][1] - v[0][1]) * (v[1][2] - v[0][2])
    }

    fn uniform(n: [u32; 3]) -> [u32; 3] {
        n.map(|c| c * FACTOR_ONE)
    }

    /// Factors of counts `n`, each with some fraction.
    fn fractional(n: [u32; 3], seed: u32) -> [u32; 3] {
        let mut s = seed.wrapping_mul(0x9e37_79b9) | 1;
        n.map(|c| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if c == 1 {
                FACTOR_ONE
            } else {
                (c - 2) * FACTOR_ONE + 1 + s % (2 * FACTOR_ONE)
            }
        })
    }

    #[test]
    fn every_pattern_triangulates_its_triangle_at_any_factors() {
        let whole = i64::from(BARY_ONE) * i64::from(BARY_ONE);
        for (i, p) in table().iter().enumerate() {
            let n = counts(i);
            assert_eq!(index(n), i);
            assert!(p.triangles.len() <= 128 && p.keys.len() <= 256, "{n:?}");
            for seed in 0..8 {
                let f = if seed == 0 {
                    uniform(n)
                } else {
                    fractional(n, seed)
                };
                let w = p.weights(f);
                let mut sum = 0;
                let mut edges: HashMap<(u8, u8), i32> = HashMap::new();
                for &t in &p.triangles {
                    let a = area(&w, t);
                    // Triangles beside a short segment may round to nothing, never fold.
                    assert!(
                        if seed == 0 { a > 0 } else { a >= 0 },
                        "{n:?} at {f:?}: triangle {t:?} has area {a}"
                    );
                    sum += a;
                    for e in 0..3 {
                        let (x, y) = (t[e], t[(e + 1) % 3]);
                        *edges.entry((x.min(y), x.max(y))).or_default() +=
                            if x < y { 1 } else { -1 };
                    }
                }
                assert_eq!(sum, whole, "{n:?}");
                // Inner edges are walked once each way; only the boundary is walked once.
                let boundary = edges.values().filter(|&&k| k != 0).count() as u32;
                assert_eq!(boundary, n.iter().sum::<u32>(), "{n:?}");
                assert!(edges.values().all(|k| k.abs() <= 1), "{n:?}");
            }
        }
    }

    #[test]
    fn edge_points_depend_only_on_their_edge_and_agree_from_either_end() {
        for (i, p) in table().iter().enumerate() {
            let n = counts(i);
            for seed in 0..4 {
                let f = fractional(n, seed);
                let mut on = [Vec::new(), Vec::new(), Vec::new()];
                for w in p.weights(f) {
                    for e in 0..3 {
                        if w[(e + 2) % 3] == 0 {
                            on[e].push(w[(e + 1) % 3]);
                        }
                    }
                }
                for (e, mut got) in on.into_iter().enumerate() {
                    got.sort_unstable();
                    let want: Vec<u32> = (0..=n[e]).map(|s| edge_weight(s, f[e])).collect();
                    assert_eq!(got, want, "{n:?} edge {e}");
                    for s in 0..=n[e] {
                        assert_eq!(edge_weight(s, f[e]) + edge_weight(n[e] - s, f[e]), BARY_ONE);
                    }
                }
            }
        }
    }

    /// The triangles of `p` at `f` with more than a sliver of area, as their corners'
    /// weights.
    fn solid(p: &Pattern, f: [u32; 3]) -> Vec<[[u32; 3]; 3]> {
        let w = p.weights(f);
        p.triangles
            .iter()
            .filter(|&&t| area(&w, t) > 1 << 48)
            .map(|t| t.map(|c| w[c as usize]))
            .collect()
    }

    fn near(a: [u32; 3], b: [u32; 3]) -> bool {
        (0..3).all(|k| a[k].abs_diff(b[k]) <= 1 << 18)
    }

    /// Whether `set` holds a triangle with the corners of `t` in the same cyclic order.
    fn holds(set: &[[[u32; 3]; 3]], t: [[u32; 3]; 3]) -> bool {
        set.iter()
            .any(|s| (0..3).any(|r| (0..3).all(|k| near(s[(k + r) % 3], t[k]))))
    }

    #[test]
    fn factors_falling_to_the_next_count_down_turn_into_its_pattern() {
        for i in 0..PATTERNS {
            let n = counts(i);
            // Every set of edges with more than one segment falls together.
            for set in 1..8u32 {
                if (0..3).any(|e| set & 1 << e != 0 && n[e] < 3) {
                    continue;
                }
                let mut coarse = n;
                let mut f = uniform(n);
                for e in (0..3).filter(|e| set & 1 << e != 0) {
                    coarse[e] -= 2;
                    f[e] = coarse[e] * FACTOR_ONE + 1;
                }
                let fine = solid(&generate(n), f);
                let want = solid(&generate(coarse), uniform(coarse));
                assert_eq!(fine.len(), want.len(), "{n:?} to {coarse:?}");
                for &t in &fine {
                    assert!(holds(&want, t), "{n:?} to {coarse:?}: {t:?}");
                }
            }
        }
    }
}
