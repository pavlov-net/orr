use bytemuck::{Pod, Zeroable};

/// One bin of an alias table, `orr.sampling`'s `AliasEntry`: bin i keeps itself with
/// `probability`, else yields `alias`. `sample_alias` draws from a table in GPU memory.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, Default, Pod, Zeroable)]
pub struct AliasEntry {
    pub probability: f32,
    pub alias: u32,
}

impl AliasEntry {
    /// The alias table that draws index i with probability `weights[i] / sum`. Weights must
    /// be finite and non-negative; when every weight is 0 the draw is uniform.
    pub fn table(weights: &[f32]) -> Vec<Self> {
        let n = weights.len();
        let sum: f64 = weights.iter().map(|&w| f64::from(w)).sum();
        let mut scaled: Vec<f64> = if sum > 0.0 {
            weights
                .iter()
                .map(|&w| f64::from(w) * n as f64 / sum)
                .collect()
        } else {
            vec![1.0; n]
        };
        let mut table: Vec<Self> = (0..n as u32)
            .map(|i| Self {
                probability: 1.0,
                alias: i,
            })
            .collect();
        let (mut small, mut large): (Vec<usize>, Vec<usize>) =
            (0..n).partition(|&i| scaled[i] < 1.0);
        while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
            small.pop();
            table[s] = Self {
                probability: scaled[s] as f32,
                alias: l as u32,
            };
            scaled[l] -= 1.0 - scaled[s];
            if scaled[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        // Rounding leaves bins whose scaled weight is 1 up to error; they keep themselves.
        table
    }
}
