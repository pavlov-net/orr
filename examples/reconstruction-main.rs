//! The dlss example's entry point: `dlss.rs`, which a build without DLSS can't compile.

cfg_select! {
    dlss => {
        #[path = "dlss.rs"]
        mod dlss;

        fn main() {
            dlss::main();
        }
    }
    _ => {
        #[allow(clippy::print_stderr)]
        fn main() {
            eprintln!("this build has no DLSS");
            std::process::exit(1);
        }
    }
}
