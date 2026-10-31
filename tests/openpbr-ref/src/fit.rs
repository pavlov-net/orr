//! Fits a BSDF's coefficients to a reference by damped Gauss-Newton (Levenberg-Marquardt).
//!
//! The candidate gives eval and pdf with their derivatives along each coefficient (Slang's
//! forward-mode autodiff on the CPU target). Residuals are the harness's relative errors,
//! |a - b| / max(|b|, 1e-3), of eval per channel and of pdf, at the directions
//! [`harness::compare`](crate::harness::compare) uses for each material of [`grid`] and view
//! of [`VIEW_COSINES`], so the errors before and after equal the report's eval and pdf
//! columns. Sums run in a fixed order: the same seed gives the same coefficients.

use std::fmt::Write as _;

use shader_maths::stats::Rng;

use crate::materials::{Material, VIEW_COSINES, grid, view};
use crate::{Bsdf, Inputs, V3, in_parallel};

/// A BSDF with coefficients to fit, prepared once per material and view.
pub trait Fit {
    /// The number of coefficients.
    fn coefficients(&self) -> usize;
    fn prepare(&mut self, inputs: &Inputs, view: V3);
    /// Eval (BSDF x cos, summed over lobes) and pdf toward `wi` at `coefficients`, with their
    /// derivatives along each coefficient in `d_eval` and `d_pdf`.
    fn eval(&self, coefficients: &[f32], wi: V3, d_eval: &mut [V3], d_pdf: &mut [f32])
    -> (V3, f32);
}

/// Inputs of a fit. Seeds follow the harness: the directions for material `i` at view `v`
/// come from seed `(i << 16 | v << 8) ^ seed`.
#[derive(Clone, Debug)]
pub struct Settings {
    pub materials: Vec<Material>,
    /// Directions per material and view.
    pub directions: usize,
    /// Weight of the pdf residuals against the eval residuals.
    pub pdf_weight: f64,
    pub max_iterations: usize,
    pub seed: u64,
}

impl Default for Settings {
    /// The report's comparison: [`grid`], 4,000 directions per view and its seed, eval only.
    fn default() -> Self {
        Self {
            materials: grid(),
            directions: 4_000,
            pdf_weight: 0.0,
            max_iterations: 100,
            seed: 3,
        }
    }
}

/// RMS relative errors at one material and view.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PointErrors {
    pub eval: f64,
    pub pdf: f64,
}

#[derive(Clone, Debug)]
pub struct Fitted {
    pub start: Vec<f32>,
    pub coefficients: Vec<f32>,
    pub iterations: usize,
    /// Weighted sum of squared residuals at the start and at the fit.
    pub start_cost: f64,
    pub cost: f64,
    /// Per material of [`Settings::materials`], per view of [`VIEW_COSINES`].
    pub start_errors: Vec<[PointErrors; VIEW_COSINES.len()]>,
    pub errors: Vec<[PointErrors; VIEW_COSINES.len()]>,
    pub materials: Vec<Material>,
}

struct Case {
    material: usize,
    view: usize,
    inputs: Inputs,
    wi: Vec<V3>,
    eval: Vec<V3>,
    pdf: Vec<f32>,
}

/// Sums over a set of cases at one coefficient vector.
#[derive(Clone, Default)]
struct Sums {
    cost: f64,
    /// J^T J and J^T r, row-major n x n and n.
    jtj: Vec<f64>,
    jtr: Vec<f64>,
    eval_sq: f64,
    pdf_sq: f64,
    count: usize,
}

impl Sums {
    fn new(n: usize) -> Self {
        Self {
            jtj: vec![0.0; n * n],
            jtr: vec![0.0; n],
            ..Self::default()
        }
    }

    fn add(&mut self, other: &Sums) {
        self.cost += other.cost;
        for (a, b) in self.jtj.iter_mut().zip(&other.jtj) {
            *a += b;
        }
        for (a, b) in self.jtr.iter_mut().zip(&other.jtr) {
            *a += b;
        }
        self.eval_sq += other.eval_sq;
        self.pdf_sq += other.pdf_sq;
        self.count += other.count;
    }

    /// Adds the relative residual of `a` against `b` with weight `weight`; `jacobian` holds
    /// the derivatives of `a` and is scaled in place. Returns the unweighted squared residual.
    fn residual(&mut self, a: f32, b: f32, jacobian: &mut [f64], weight: f64) -> f64 {
        let scale = f64::from(b.abs().max(1e-3));
        let r = (f64::from(a) - f64::from(b)) / scale;
        let n = jacobian.len();
        for j in jacobian.iter_mut() {
            *j /= scale;
        }
        if weight > 0.0 {
            self.cost += weight * r * r;
            for row in 0..n {
                self.jtr[row] += weight * jacobian[row] * r;
                for col in 0..n {
                    self.jtj[row * n + col] += weight * jacobian[row] * jacobian[col];
                }
            }
        }
        r * r
    }

    fn errors(&self) -> PointErrors {
        let n = self.count.max(1) as f64;
        PointErrors {
            eval: (self.eval_sq / (3.0 * n)).sqrt(),
            pdf: (self.pdf_sq / n).sqrt(),
        }
    }
}

/// Fits the coefficients of the BSDFs `candidate` makes to those `reference` makes, from
/// `start`. Each thread makes its own BSDFs.
pub fn fit<C: Fit, R: Bsdf>(
    candidate: impl Fn() -> C + Sync,
    reference: impl Fn() -> R + Sync,
    start: &[f32],
    settings: &Settings,
) -> Fitted {
    let n = start.len();
    assert_eq!(candidate().coefficients(), n, "start has the wrong length");
    let cases = reference_cases(&reference, settings);
    let evaluate = |c: &[f32]| sums(&candidate, &cases, c, settings.pdf_weight);

    let mut c = start.to_vec();
    let first = evaluate(&c);
    let mut current = total(&first, n);
    let mut lambda = 1e-3;
    let mut iterations = 0;
    while iterations < settings.max_iterations {
        iterations += 1;
        let mut accepted = false;
        while lambda < 1e12 {
            let Some(step) = damped_step(&current, n, lambda) else {
                break;
            };
            let next_c: Vec<f32> = c.iter().zip(&step).map(|(a, d)| a + *d as f32).collect();
            if next_c == c {
                break;
            }
            let next = total(&evaluate(&next_c), n);
            if next.cost < current.cost {
                let gain = (current.cost - next.cost) / current.cost.max(f64::MIN_POSITIVE);
                c = next_c;
                current = next;
                lambda = (lambda * 0.3).max(1e-12);
                accepted = gain > 1e-12;
                break;
            }
            lambda *= 4.0;
        }
        if !accepted {
            break;
        }
    }
    let last = evaluate(&c);
    let per_point = |s: &[Sums]| {
        let mut out = vec![[PointErrors::default(); 4]; settings.materials.len()];
        for (case, s) in cases.iter().zip(s) {
            out[case.material][case.view] = s.errors();
        }
        out
    };
    Fitted {
        start: start.to_vec(),
        coefficients: c,
        iterations,
        start_cost: total(&first, n).cost,
        cost: total(&last, n).cost,
        start_errors: per_point(&first),
        errors: per_point(&last),
        materials: settings.materials.clone(),
    }
}

fn reference_cases<R: Bsdf>(reference: &(impl Fn() -> R + Sync), settings: &Settings) -> Vec<Case> {
    let points: Vec<(usize, usize)> = (0..settings.materials.len())
        .flat_map(|m| (0..VIEW_COSINES.len()).map(move |v| (m, v)))
        .collect();
    in_parallel(points.len(), reference, |r, i| {
        let (material, v) = points[i];
        let mut rng = Rng::new(((material as u64) << 16 | (v as u64) << 8) ^ settings.seed);
        let wi: Vec<V3> = (0..settings.directions).map(|_| rng.unit()).collect();
        r.prepare(&settings.materials[material].inputs, view(VIEW_COSINES[v]));
        Case {
            material,
            view: v,
            inputs: settings.materials[material].inputs,
            eval: r.eval_all(&wi).iter().map(crate::Eval::total).collect(),
            pdf: r.pdf_all(&wi),
            wi,
        }
    })
}

/// The sums of every case at `c`, in case order.
fn sums<C: Fit>(
    candidate: &(impl Fn() -> C + Sync),
    cases: &[Case],
    c: &[f32],
    pdf_weight: f64,
) -> Vec<Sums> {
    let n = c.len();
    in_parallel(cases.len(), candidate, |bsdf, i| {
        let case = &cases[i];
        let mut s = Sums::new(n);
        let (mut d_eval, mut d_pdf) = (vec![[0.0; 3]; n], vec![0.0; n]);
        let mut jacobian = vec![0.0; n];
        bsdf.prepare(&case.inputs, view(VIEW_COSINES[case.view]));
        for ((&wi, reference), &reference_pdf) in case.wi.iter().zip(&case.eval).zip(&case.pdf) {
            let (eval, pdf) = bsdf.eval(c, wi, &mut d_eval, &mut d_pdf);
            assert!(
                eval.iter().all(|x| x.is_finite()) && pdf.is_finite(),
                "non-finite eval {eval:?} or pdf {pdf} toward {wi:?}"
            );
            for ch in 0..3 {
                for (j, d) in jacobian.iter_mut().zip(&d_eval) {
                    *j = f64::from(d[ch]);
                }
                s.eval_sq += s.residual(eval[ch], reference[ch], &mut jacobian, 1.0);
            }
            jacobian
                .iter_mut()
                .zip(&d_pdf)
                .for_each(|(j, d)| *j = f64::from(*d));
            s.pdf_sq += s.residual(pdf, reference_pdf, &mut jacobian, pdf_weight);
            s.count += 1;
        }
        s
    })
}

fn total(sums: &[Sums], n: usize) -> Sums {
    let mut t = Sums::new(n);
    for s in sums {
        t.add(s);
    }
    t
}

/// Solves (J^T J + lambda diag(J^T J)) step = -J^T r; coefficients nothing depends on keep
/// a zero step. `None` when the system is singular.
fn damped_step(s: &Sums, n: usize, lambda: f64) -> Option<Vec<f64>> {
    let free: Vec<usize> = (0..n).filter(|&k| s.jtj[k * n + k] > 0.0).collect();
    let m = free.len();
    if m == 0 {
        return None;
    }
    let mut a = vec![0.0; m * (m + 1)];
    for (i, &r) in free.iter().enumerate() {
        for (j, &c) in free.iter().enumerate() {
            a[i * (m + 1) + j] = s.jtj[r * n + c];
        }
        a[i * (m + 1) + i] *= 1.0 + lambda;
        a[i * (m + 1) + m] = -s.jtr[r];
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..m {
        let pivot = (col..m).max_by(|&x, &y| {
            a[x * (m + 1) + col]
                .abs()
                .total_cmp(&a[y * (m + 1) + col].abs())
        })?;
        if a[pivot * (m + 1) + col] == 0.0 {
            return None;
        }
        for k in 0..=m {
            a.swap(col * (m + 1) + k, pivot * (m + 1) + k);
        }
        for row in col + 1..m {
            let f = a[row * (m + 1) + col] / a[col * (m + 1) + col];
            for k in col..=m {
                a[row * (m + 1) + k] -= f * a[col * (m + 1) + k];
            }
        }
    }
    let mut x = vec![0.0; m];
    for row in (0..m).rev() {
        let mut v = a[row * (m + 1) + m];
        for k in row + 1..m {
            v -= a[row * (m + 1) + k] * x[k];
        }
        x[row] = v / a[row * (m + 1) + row];
    }
    let mut step = vec![0.0; n];
    for (i, &k) in free.iter().enumerate() {
        step[k] = x[i];
    }
    Some(step)
}

impl Fitted {
    /// Points where the fitted eval RMS error is above the starting one.
    pub fn worse(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (m, (start, fitted)) in self.start_errors.iter().zip(&self.errors).enumerate() {
            for (v, (a, b)) in start.iter().zip(fitted).enumerate() {
                if b.eval > a.eval {
                    out.push(format!(
                        "{}, mu {}: eval RMS {:.3e} from {:.3e}",
                        self.materials[m].name, VIEW_COSINES[v], b.eval, a.eval
                    ));
                }
            }
        }
        out
    }

    /// The coefficients and the errors before and after, as Markdown.
    pub fn markdown(&self, title: &str, names: &[&str]) -> String {
        let mut s = String::new();
        let w = &mut s;
        let _ = writeln!(w, "# Fit: {title}\n");
        let _ = writeln!(
            w,
            "{} iterations; weighted squared error {:.6e} from {:.6e}.\n",
            self.iterations, self.cost, self.start_cost
        );
        let _ = writeln!(w, "| Coefficient | start | fitted |\n| --- | --- | --- |");
        for (k, (a, b)) in self.start.iter().zip(&self.coefficients).enumerate() {
            let name = names.get(k).copied().unwrap_or("?");
            let _ = writeln!(w, "| {name} | {a} | {b} |");
        }
        let _ = writeln!(
            w,
            "\nRMS relative error against the reference, start -> fitted.\n"
        );
        let heads: Vec<String> = VIEW_COSINES.iter().map(|m| format!("mu {m}")).collect();
        let _ = writeln!(w, "| Material | | {} |", heads.join(" | "));
        let _ = writeln!(w, "|{}", " --- |".repeat(VIEW_COSINES.len() + 2));
        for (m, (start, fitted)) in self.start_errors.iter().zip(&self.errors).enumerate() {
            for kind in ["eval", "pdf"] {
                let pick = |e: &PointErrors| if kind == "eval" { e.eval } else { e.pdf };
                let cells: Vec<String> = start
                    .iter()
                    .zip(fitted)
                    .map(|(a, b)| format!("{:.2e} -> {:.2e}", pick(a), pick(b)))
                    .collect();
                let _ = writeln!(
                    w,
                    "| {} | {kind} | {} |",
                    self.materials[m].name,
                    cells.join(" | ")
                );
            }
        }
        s
    }
}
