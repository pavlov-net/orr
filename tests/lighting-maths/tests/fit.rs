//! Fitting the real-time BSDF's coefficients with derivatives from Slang's autodiff.

use lighting_maths::{COEFFICIENTS, Fitting, RealTime, START, fitted_constants};
use openpbr_ref::Reference;
use openpbr_ref::fit::{Settings, fit};

fn relative(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs()
}

/// A target made by the same BSDF at known coefficients, inside the range where every one
/// moves the eval on the grid.
#[test]
fn recovers_known_coefficients() {
    let known = [0.3, 1.5, 0.6];
    let fitted = fit(
        Fitting::collapsed,
        || RealTime::collapsed_with(known),
        &START,
        &Settings::default(),
    );
    for k in 0..COEFFICIENTS {
        assert!(
            relative(fitted.coefficients[k], known[k]) <= 1e-3,
            "{:?} against {known:?} after {} iterations",
            fitted.coefficients,
            fitted.iterations
        );
    }
}

/// Against the reference: no material and view gets worse than the start, the same seeds
/// give the same coefficients, and they are the ones in `bsdf.slang`.
#[test]
fn the_source_holds_the_fit_to_the_reference() {
    let settings = Settings::default();
    let fitted = fit(Fitting::collapsed, Reference::new, &START, &settings);
    let worse = fitted.worse();
    assert!(worse.is_empty(), "{}", worse.join("\n"));
    assert!(fitted.cost < fitted.start_cost);
    let again = fit(Fitting::collapsed, Reference::new, &START, &settings);
    assert_eq!(fitted.coefficients, again.coefficients);
    let source = fitted_constants();
    assert!(
        (0..COEFFICIENTS).all(|k| relative(source[k], fitted.coefficients[k]) <= 1e-3),
        "bsdf.slang holds {source:?}, the fit gives {:?}; run bsdf-fit",
        fitted.coefficients
    );
}
