//! The dense vegetation test scene in a window, or headless: a forest of trees with rigid
//! parts swaying inside their animation range, grass, a pond and emissive particles over a
//! 1 km heightfield tile, path traced under a low sun and a moon. See `examples::vegetation`
//! for what it holds and the switches it takes.
//!
//! `dense-vegetation [--synthetic] [--script walk|turn|overview|flythrough] [--frames N]
//! [--headless] [--out file.png] [scene switches...]`. Without `--synthetic` the trees come
//! from `$ORR_CONTENT/cooked`. Headless runs render 480 x 270 for 120 frames unless told
//! otherwise and write the last frame as a PNG.

use examples::vegetation::{Settings, run};
use examples::{Capture, Options, assert_clean};

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;

fn main() {
    let options = Options::from_args();
    if let Some(c) = capture(&options) {
        let path = options.out_path("dense-vegetation");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Runs the example; headless runs return their last frame.
pub(crate) fn capture(options: &Options) -> Option<Capture> {
    let mut settings = Settings::parse(&options.rest);
    let mut options = options.clone();
    if options.headless {
        if !options.rest.iter().any(|a| a == "--extent") {
            settings.extent = HEADLESS_EXTENT;
        }
        options.frames = options.frames.or(Some(HEADLESS_FRAMES));
    }
    run(&options, &settings).capture
}
