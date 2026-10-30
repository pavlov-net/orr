//! Runs the OpenPBR harness on orr-lighting's real-time BSDF (Slang, CPU) against
//! adobe/openpbr-bsdf's C++ and writes the Markdown report to the path given, or to stdout.
//! Exits 1 if a gate fails.

use lighting_maths::RealTime;
use openpbr_ref::Reference;
use openpbr_ref::report::{Counts, evaluate};

fn main() {
    let results = evaluate(
        "orr-lighting bsdf (Slang C++ target, array LUTs)",
        RealTime::new,
        Reference::new,
        false,
        Counts::default(),
    );
    let markdown = results.markdown();
    match std::env::args_os().nth(1) {
        Some(path) => std::fs::write(&path, markdown)
            .unwrap_or_else(|e| panic!("{}: {e}", path.to_string_lossy())),
        #[allow(clippy::print_stdout)]
        None => print!("{markdown}"),
    }
    std::process::exit(i32::from(!results.failures().is_empty()));
}
