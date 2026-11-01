//! The tessellation pattern table, which the build script writes out for the GPU.
//!
//! A pattern dices a triangle with integer factors `[a, b, c]` on its edges v0-v1, v1-v2 and
//! v2-v0, each from 1 to `MAX_DICE`. It is the uniform grid at the largest factor with each
//! edge's grid points moved to the nearest of that edge's own points, and the triangles that
//! collapse dropped. Vertices are barycentrics `[w1, w2]` (weights of v1 and v2, v0's
//! implied) in units of `1 / BARY_ONE`. An edge point's weights depend only on the edge's
//! factor and its distance along the edge, quantized symmetrically about the edge's middle,
//! so the patch on the other side of an edge, walking it the other way, gives its two
//! corners the same weights.

/// The largest edge factor: the uniform grid at 11 has 121 triangles.
pub const MAX_DICE: u32 = 11;
/// Barycentric 1.
pub const BARY_ONE: u32 = 65534;
/// Patterns in the table, one per factor triple.
pub const PATTERNS: usize = (MAX_DICE * MAX_DICE * MAX_DICE) as usize;

/// One pattern of the table: barycentrics `[w1, w2]` in units of `1 / BARY_ONE` (v0's
/// weight implied) and triangles of cluster-local corners. A generated cluster's vertex `i`
/// lies at pattern vertex `i` of its subpatch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    pub vertices: Vec<[u16; 2]>,
    pub triangles: Vec<[u8; 3]>,
}

/// The factors of table entry `i`.
pub(crate) fn factors(i: usize) -> [u32; 3] {
    let i = i as u32;
    [
        i / (MAX_DICE * MAX_DICE) + 1,
        i / MAX_DICE % MAX_DICE + 1,
        i % MAX_DICE + 1,
    ]
}

/// `n * BARY_ONE / d`, rounded half up.
fn scaled(n: u32, d: u32) -> u32 {
    (2 * n * BARY_ONE + d) / (2 * d)
}

/// The weight point `s` of an edge of factor `f` gives the edge's far corner: `s / f`
/// quantized from whichever end is nearer, so `weight(s, f) + weight(f - s, f) == BARY_ONE`.
pub(crate) fn edge_weight(s: u32, f: u32) -> u32 {
    if 2 * s <= f {
        scaled(s, f)
    } else {
        BARY_ONE - scaled(f - s, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Corner(u8),
    /// Edge, point along it from its first corner.
    Edge(u8, u32),
    Interior(u32, u32),
}

/// The pattern of edge factors `f`.
pub(crate) fn generate(f: [u32; 3]) -> Pattern {
    let m = f.iter().copied().max().expect("three factors");
    // Grid point (i, j, k), i + j + k = m, weights i / m, j / m, k / m on v0, v1, v2.
    let key = |i: u32, j: u32, k: u32| -> Key {
        // Edge e from corner e to corner e + 1, at grid parameter `t` of `m`.
        let on_edge = |e: u8, t: u32| {
            let n = f[e as usize];
            let s = (2 * t * n + m) / (2 * m);
            if s == 0 {
                Key::Corner(e)
            } else if s == n {
                Key::Corner((e + 1) % 3)
            } else {
                Key::Edge(e, s)
            }
        };
        if k == 0 {
            on_edge(0, j)
        } else if i == 0 {
            on_edge(1, k)
        } else if j == 0 {
            on_edge(2, i)
        } else {
            Key::Interior(j, k)
        }
    };
    let weights = |key: Key| -> [u32; 2] {
        match key {
            Key::Corner(0) => [0, 0],
            Key::Corner(1) => [BARY_ONE, 0],
            Key::Corner(_) => [0, BARY_ONE],
            Key::Edge(0, s) => [edge_weight(s, f[0]), 0],
            Key::Edge(1, s) => {
                let w = edge_weight(s, f[1]);
                [BARY_ONE - w, w]
            }
            Key::Edge(_, s) => [0, BARY_ONE - edge_weight(s, f[2])],
            Key::Interior(j, k) => [scaled(j, m), scaled(k, m)],
        }
    };

    let mut keys: Vec<Key> = Vec::new();
    let corner = |key: Key, keys: &mut Vec<Key>| -> u8 {
        match keys.iter().position(|&k| k == key) {
            Some(p) => p as u8,
            None => {
                keys.push(key);
                (keys.len() - 1) as u8
            }
        }
    };
    for c in 0..3 {
        corner(Key::Corner(c), &mut keys);
    }
    let mut triangles = Vec::new();
    let mut push = |a: [u32; 3], b: [u32; 3], c: [u32; 3], keys: &mut Vec<Key>| {
        let t = [a, b, c].map(|p| corner(key(p[0], p[1], p[2]), keys));
        if t[0] != t[1] && t[1] != t[2] && t[2] != t[0] {
            triangles.push(t);
        }
    };
    for i in 0..m {
        for j in 0..m - i {
            let k = m - 1 - i - j;
            push([i + 1, j, k], [i, j + 1, k], [i, j, k + 1], &mut keys);
            if k > 0 {
                push(
                    [i, j + 1, k],
                    [i + 1, j, k],
                    [i + 1, j + 1, k - 1],
                    &mut keys,
                );
            }
        }
    }
    let vertices = keys.iter().map(|&k| weights(k).map(|w| w as u16)).collect();
    Pattern {
        vertices,
        triangles,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn table() -> Vec<Pattern> {
        (0..PATTERNS).map(|i| generate(factors(i))).collect()
    }

    /// Twice the signed area of triangle `t` in barycentric units.
    fn area(p: &Pattern, t: [u8; 3]) -> i64 {
        let v = t.map(|c| p.vertices[c as usize].map(i64::from));
        (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1])
    }

    #[test]
    fn every_pattern_triangulates_its_triangle() {
        let whole = i64::from(BARY_ONE) * i64::from(BARY_ONE);
        for (i, p) in table().iter().enumerate() {
            let f = factors(i);
            assert!(p.triangles.len() <= 128 && p.vertices.len() <= 256, "{f:?}");
            let mut sum = 0;
            let mut edges: HashMap<(u8, u8), i32> = HashMap::new();
            for &t in &p.triangles {
                let a = area(p, t);
                assert!(a > 0, "{f:?}: triangle {t:?} has area {a}");
                sum += a;
                for e in 0..3 {
                    let (x, y) = (t[e], t[(e + 1) % 3]);
                    *edges.entry((x.min(y), x.max(y))).or_default() += if x < y { 1 } else { -1 };
                }
            }
            assert_eq!(sum, whole, "{f:?}");
            // Inner edges are walked once each way; only the boundary is walked once.
            let boundary = edges.values().filter(|&&n| n != 0).count() as u32;
            assert_eq!(boundary, f.iter().sum::<u32>(), "{f:?}");
            assert!(edges.values().all(|n| n.abs() <= 1), "{f:?}");
        }
    }

    #[test]
    fn edge_points_are_the_factor_points() {
        for (i, p) in table().iter().enumerate() {
            let f = factors(i);
            let mut on = [Vec::new(), Vec::new(), Vec::new()];
            for v in &p.vertices {
                let [w1, w2] = v.map(u32::from);
                if w2 == 0 {
                    on[0].push(w1);
                }
                if w1 + w2 == BARY_ONE {
                    on[1].push(w2);
                }
                if w1 == 0 {
                    on[2].push(BARY_ONE - w2);
                }
            }
            for (e, mut got) in on.into_iter().enumerate() {
                got.sort_unstable();
                let want: Vec<u32> = (0..=f[e]).map(|s| edge_weight(s, f[e])).collect();
                assert_eq!(got, want, "{f:?} edge {e}");
            }
        }
    }

    #[test]
    fn edge_weights_agree_from_either_end() {
        for n in 1..=MAX_DICE {
            for s in 0..=n {
                assert_eq!(edge_weight(s, n) + edge_weight(n - s, n), BARY_ONE);
            }
        }
    }
}
