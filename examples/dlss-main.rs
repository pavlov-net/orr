//! The dlss example's entry point: `dlss.rs`, which a build without DLSS can't compile.

#[cfg(dlss)]
#[path = "dlss.rs"]
mod dlss;

#[cfg(dlss)]
fn main() {
    dlss::main();
}

#[cfg(not(dlss))]
#[allow(clippy::print_stderr)]
fn main() {
    eprintln!("this build has no DLSS");
    std::process::exit(1);
}
