//! Runs the OpenPBR harness on `orr.openpbr` against adobe/openpbr-bsdf's C++ and writes the
//! Markdown report to the path given, or to stdout. Exits 1 if a gate fails.
//!
//! `openpbr-report [--gpu] [out.md]`: the candidate is the Slang C++ build (array LUTs), or
//! with `--gpu` (feature `gpu`) the Vulkan kernel (heap LUTs).

use openpbr_ref::Reference;
use openpbr_ref::report::{Counts, Results, evaluate};

fn main() {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let gpu = args.iter().position(|a| a == "--gpu");
    let results = if let Some(i) = gpu {
        args.remove(i);
        on_gpu()
    } else {
        on_cpu()
    };
    let markdown = results.markdown();
    match args.first() {
        Some(path) => std::fs::write(path, markdown)
            .unwrap_or_else(|e| panic!("{}: {e}", path.to_string_lossy())),
        #[allow(clippy::print_stdout)]
        None => print!("{markdown}"),
    }
    std::process::exit(i32::from(!results.failures().is_empty()));
}

fn on_cpu() -> Results {
    evaluate(
        "orr.openpbr (Slang C++ target, array LUTs)",
        openpbr_ref::SlangCpu::new,
        Reference::new,
        true,
        Counts::default(),
    )
}

#[cfg(feature = "gpu")]
fn on_gpu() -> Results {
    let results = evaluate(
        "orr.openpbr (Vulkan compute, heap LUTs)",
        openpbr_ref::SlangGpu::new,
        Reference::new,
        true,
        Counts::default(),
    );
    let messages = openpbr_ref::SlangGpu::device().validation_messages();
    assert_eq!(messages, 0, "validation messages (see the log)");
    results
}

#[cfg(not(feature = "gpu"))]
fn on_gpu() -> Results {
    panic!("--gpu needs feature gpu");
}
