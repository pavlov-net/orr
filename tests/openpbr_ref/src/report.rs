//! Runs every harness check over [`grid`] and [`VIEW_COSINES`] for a candidate BSDF and a
//! reference, applies the gates and writes the results as Markdown.
//!
//! Gates:
//! - energy: every albedo estimate (total and per lobe) at most max(1, the reference's)
//!   within 3 combined standard errors plus [`TABLE_TOLERANCE`]. The reference itself gains
//!   energy in places (rough anisotropic metal, a coat over a rough base at grazing view),
//!   so a candidate may match that gain but not exceed it;
//! - chi-square p above 0.01 where the material has no ideal-specular lobe;
//! - reciprocity within 1e-5 relative on the reciprocal materials (reported for the rest);
//! - when the candidate is the same model as the reference, eval and pdf within 1e-3
//!   relative. A different model (a real-time approximation) gets the albedo tables and
//!   error histograms without that gate.

use std::fmt::Write as _;

use crate::harness::{self, Albedo, Comparison, Estimate, HISTOGRAM_BINS};
use crate::materials::{Material, VIEW_COSINES, grid, view};
use crate::{Bsdf, Lobe, in_parallel};

/// Sample counts. Seeds are fixed, so a run is deterministic.
#[derive(Clone, Copy, Debug)]
pub struct Counts {
    pub albedo: usize,
    pub chi_square: usize,
    pub reciprocity: usize,
    pub compare: usize,
}

impl Default for Counts {
    fn default() -> Self {
        Self {
            albedo: 200_000,
            chi_square: 200_000,
            reciprocity: 2_000,
            compare: 4_000,
        }
    }
}

pub const CHI_SQUARE_P: f64 = 0.01;
/// Energy the 16-bit compensation tables and their interpolation add: white rough metal
/// reaches 1.001 in the reference.
pub const TABLE_TOLERANCE: f64 = 1e-3;
pub const RECIPROCITY: f64 = 1e-5;
pub const SAME_MODEL: f64 = 1e-3;

/// One material's results; per-view vectors follow [`VIEW_COSINES`].
#[derive(Clone, Debug)]
pub struct MaterialResults {
    pub material: Material,
    pub albedo: Vec<Albedo>,
    pub reference_albedo: Vec<Albedo>,
    /// Empty for materials with an ideal-specular lobe.
    pub chi_square: Vec<f64>,
    /// For materials without an ideal-specular lobe; gated only where reciprocal.
    pub reciprocity: Option<f64>,
    pub comparison: Comparison,
}

#[derive(Clone, Debug)]
pub struct Results {
    pub candidate: String,
    /// The candidate implements the reference's model, so eval and pdf must agree.
    pub same_model: bool,
    pub counts: Counts,
    pub materials: Vec<MaterialResults>,
}

/// Runs the checks, one thread per material in turn up to the machine's parallelism. Each
/// thread makes its own BSDFs.
pub fn evaluate<C: Bsdf, R: Bsdf>(
    candidate_name: &str,
    candidate: impl Fn() -> C + Sync,
    reference: impl Fn() -> R + Sync,
    same_model: bool,
    counts: Counts,
) -> Results {
    let materials = grid();
    let results = in_parallel(
        materials.len(),
        &|| (candidate(), reference()),
        |(c, r), i| run_material(c, r, &materials[i], i as u64, counts),
    );
    Results {
        candidate: candidate_name.to_owned(),
        same_model,
        counts,
        materials: results,
    }
}

fn run_material(
    c: &mut dyn Bsdf,
    r: &mut dyn Bsdf,
    m: &Material,
    index: u64,
    counts: Counts,
) -> MaterialResults {
    let seed = |view: usize, check: u64| (index << 16) | ((view as u64) << 8) | check;
    let mut out = MaterialResults {
        material: *m,
        albedo: Vec::new(),
        reference_albedo: Vec::new(),
        chi_square: Vec::new(),
        reciprocity: None,
        comparison: Comparison::default(),
    };
    for (v, &mu) in VIEW_COSINES.iter().enumerate() {
        let dir = view(mu);
        out.albedo.push(harness::albedo(
            c,
            &m.inputs,
            dir,
            counts.albedo,
            seed(v, 1),
        ));
        out.reference_albedo.push(harness::albedo(
            r,
            &m.inputs,
            dir,
            counts.albedo,
            seed(v, 1),
        ));
        if !m.delta {
            out.chi_square.push(harness::chi_square(
                c,
                &m.inputs,
                dir,
                counts.chi_square,
                seed(v, 2),
            ));
        }
        let cmp = harness::compare(c, r, &m.inputs, dir, counts.compare, seed(v, 3));
        out.comparison.merge(&cmp);
    }
    if !m.delta {
        out.reciprocity = Some(harness::reciprocity(
            c,
            &m.inputs,
            counts.reciprocity,
            seed(0, 4),
        ));
    }
    out
}

/// How far `candidate` exceeds max(1, `reference`) beyond 3 combined standard errors and
/// [`TABLE_TOLERANCE`], over the total and every lobe and channel; at most 0 when it passes.
pub fn energy_excess(candidate: &Albedo, reference: &Albedo) -> f64 {
    let pairs = candidate
        .lobes
        .iter()
        .map(|(lobe, e)| (e, reference.lobes.get(lobe).copied().unwrap_or_default()));
    pairs
        .chain([(&candidate.total, reference.total)])
        .flat_map(|(c, r)| {
            (0..3).map(move |i| {
                c.mean[i] - r.mean[i].max(1.0) - 3.0 * combined_se(c, &r, i) - TABLE_TOLERANCE
            })
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

fn combined_se(a: &Estimate, b: &Estimate, channel: usize) -> f64 {
    a.se[channel].hypot(b.se[channel])
}

impl Results {
    /// One line per failed gate.
    pub fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        for m in &self.materials {
            let name = m.material.name;
            for ((a, r), mu) in m.albedo.iter().zip(&m.reference_albedo).zip(VIEW_COSINES) {
                let excess = energy_excess(a, r);
                if excess > 0.0 {
                    out.push(format!(
                        "{name}, mu {mu}: albedo above max(1, reference) by {excess:.4} beyond 3 SE and the table tolerance"
                    ));
                }
            }
            for (p, mu) in m.chi_square.iter().zip(VIEW_COSINES) {
                if *p <= CHI_SQUARE_P {
                    out.push(format!("{name}, mu {mu}: chi-square p = {p:.3e}"));
                }
            }
            if let Some(e) = m.reciprocity
                && m.material.reciprocal
                && e > RECIPROCITY
            {
                out.push(format!("{name}: reciprocity error {e:.3e}"));
            }
            if self.same_model {
                let c = &m.comparison;
                if c.eval.max > SAME_MODEL {
                    out.push(format!("{name}: eval error {:.3e}", c.eval.max));
                }
                if c.pdf.max > SAME_MODEL {
                    out.push(format!("{name}: pdf error {:.3e}", c.pdf.max));
                }
            }
        }
        out
    }

    pub fn markdown(&self) -> String {
        let mut s = String::new();
        let w = &mut s;
        let failures = self.failures();
        let _ = writeln!(w, "# OpenPBR harness: {}\n", self.candidate);
        let _ = writeln!(
            w,
            "Reference: adobe/openpbr-bsdf C++ (GLM). Samples per view: albedo {}, chi-square {}; \
             {} eval/pdf cases per view; {} reciprocity pairs per material. View cosines {:?}.\n",
            self.counts.albedo,
            self.counts.chi_square,
            self.counts.compare,
            self.counts.reciprocity,
            VIEW_COSINES
        );
        if failures.is_empty() {
            let _ = writeln!(w, "All gates pass.\n");
        } else {
            let _ = writeln!(w, "Failed gates:\n");
            for f in &failures {
                let _ = writeln!(w, "- {f}");
            }
            let _ = writeln!(w);
        }

        let mut all = Comparison::default();
        for m in &self.materials {
            all.merge(&m.comparison);
        }
        let _ = writeln!(w, "## Eval and pdf against the reference\n");
        let _ = writeln!(
            w,
            "Relative error |a - b| / max(|b|, 1e-3).\n\n| Material | eval max | eval RMS | pdf max | pdf RMS | sample mismatches |\n| --- | --- | --- | --- | --- | --- |"
        );
        for (name, c) in self
            .materials
            .iter()
            .map(|m| (m.material.name, &m.comparison))
            .chain([("all", &all)])
        {
            let _ = writeln!(
                w,
                "| {name} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {} / {} |",
                c.eval.max,
                c.eval.rms(),
                c.pdf.max,
                c.pdf.rms(),
                c.sample_mismatches,
                c.samples
            );
        }
        let _ = writeln!(
            w,
            "\nError histogram over all materials (counts per decade):\n"
        );
        let labels = [
            "< 1e-7", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", ">= 1e-1",
        ];
        let _ = writeln!(w, "| | {} |", labels.join(" | "));
        let _ = writeln!(w, "|{}", " --- |".repeat(HISTOGRAM_BINS + 1));
        for (name, e) in [("eval", &all.eval), ("pdf", &all.pdf)] {
            let cells: Vec<String> = e.histogram.iter().map(ToString::to_string).collect();
            let _ = writeln!(w, "| {name} | {} |", cells.join(" | "));
        }

        let _ = writeln!(w, "\n## Chi-square and reciprocity\n");
        let mus: Vec<String> = VIEW_COSINES
            .iter()
            .map(|m| format!("p at mu {m}"))
            .collect();
        let _ = writeln!(w, "| Material | {} | reciprocity |", mus.join(" | "));
        let _ = writeln!(w, "|{}", " --- |".repeat(VIEW_COSINES.len() + 2));
        for m in &self.materials {
            let ps: Vec<String> = if m.chi_square.is_empty() {
                vec!["delta".to_owned(); VIEW_COSINES.len()]
            } else {
                m.chi_square.iter().map(|p| format!("{p:.3}")).collect()
            };
            let rec = match m.reciprocity {
                Some(e) if m.material.reciprocal => format!("{e:.2e}"),
                Some(e) => format!("{e:.2e} (not reciprocal)"),
                None => "-".to_owned(),
            };
            let _ = writeln!(w, "| {} | {} | {rec} |", m.material.name, ps.join(" | "));
        }

        let _ = writeln!(
            w,
            "\n## Directional albedo per lobe\n\nMean R G B over sampled directions; candidate, with the reference below it when they differ by more than 3 standard errors.\n"
        );
        for m in &self.materials {
            let _ = writeln!(w, "### {}\n", m.material.name);
            let heads: Vec<String> = VIEW_COSINES.iter().map(|m| format!("mu {m}")).collect();
            let _ = writeln!(w, "| Lobe | {} |", heads.join(" | "));
            let _ = writeln!(w, "|{}", " --- |".repeat(VIEW_COSINES.len() + 1));
            let mut lobes: Vec<Option<Lobe>> = m
                .albedo
                .iter()
                .chain(&m.reference_albedo)
                .flat_map(|a| a.lobes.keys().copied())
                .map(Some)
                .collect();
            lobes.sort();
            lobes.dedup();
            lobes.push(None);
            for lobe in lobes {
                let pick = |a: &Albedo| match lobe {
                    Some(l) => a.lobes.get(&l).copied().unwrap_or_default(),
                    None => a.total,
                };
                let label = lobe.map_or("total".to_owned(), Lobe::name);
                let cells: Vec<String> = m
                    .albedo
                    .iter()
                    .zip(&m.reference_albedo)
                    .map(|(a, r)| {
                        let (a, r) = (pick(a), pick(r));
                        let differs = (0..3)
                            .any(|c| (a.mean[c] - r.mean[c]).abs() > 3.0 * combined_se(&a, &r, c));
                        let rgb = |e: &Estimate| {
                            format!("{:.3} {:.3} {:.3}", e.mean[0], e.mean[1], e.mean[2])
                        };
                        if differs {
                            format!("{} (ref {})", rgb(&a), rgb(&r))
                        } else {
                            rgb(&a)
                        }
                    })
                    .collect();
                let _ = writeln!(w, "| {label} | {} |", cells.join(" | "));
            }
            let _ = writeln!(w);
        }
        s
    }
}
