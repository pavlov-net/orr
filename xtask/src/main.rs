//! Repository tasks, run as `cargo run -p xtask -- <task>`.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool reports on its standard streams"
)]

mod test;

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("test") => test::run(args.collect()),
        _ => {
            eprintln!("usage: cargo run -p xtask -- test [--help]");
            ExitCode::from(2)
        }
    }
}
