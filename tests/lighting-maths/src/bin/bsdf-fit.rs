//! Fits the coefficients of orr-lighting's real-time BSDF (`FOLD_*` in `bsdf.slang`) to
//! adobe/openpbr-bsdf's eval over the harness grid from [`START`], and writes the
//! coefficients and the error report to the path given, or to stdout. Exits 1 if the fit is
//! worse than the start at any material and view.

use lighting_maths::{COEFFICIENT_NAMES, Fitting, START, fitted_constants};
use openpbr_ref::Reference;
use openpbr_ref::fit::{Settings, fit};

fn main() {
    let fitted = fit(
        Fitting::collapsed,
        Reference::new,
        &START,
        &Settings::default(),
    );
    let mut markdown = fitted.markdown(
        "orr-lighting bsdf, collapsed, against adobe/openpbr-bsdf",
        &COEFFICIENT_NAMES,
    );
    markdown.push_str("\nFor `bsdf.slang`:\n\n```slang\n");
    for (name, value) in COEFFICIENT_NAMES.iter().zip(&fitted.coefficients) {
        markdown.push_str(&format!("public static const float {name} = {value:?};\n"));
    }
    markdown.push_str(&format!(
        "```\n\nThe source holds {:?}.\n",
        fitted_constants()
    ));
    match std::env::args_os().nth(1) {
        Some(path) => std::fs::write(&path, markdown)
            .unwrap_or_else(|e| panic!("{}: {e}", path.to_string_lossy())),
        #[allow(clippy::print_stdout)]
        None => print!("{markdown}"),
    }
    let worse = fitted.worse();
    #[allow(clippy::print_stderr)]
    for w in &worse {
        eprintln!("worse than the start: {w}");
    }
    std::process::exit(i32::from(!worse.is_empty()));
}
