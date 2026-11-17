//! The reconstruction example's entry point: `reconstruction.rs`, which a build without DLSS or
//! MetalFX can't compile.

cfg_select! {
    reconstruction => {
        #[path = "reconstruction.rs"]
        mod reconstruction;

        fn main() {
            reconstruction::main();
        }
    }
    _ => {
        #[allow(clippy::print_stderr)]
        fn main() {
            eprintln!("this build has neither DLSS nor MetalFX");
            std::process::exit(1);
        }
    }
}
