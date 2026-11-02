//! What each [`Quality`] level does to a cook.

use ctt::Quality;

/// zstd level of the blobs a cook at `q` writes when [`crate::CookOptions::zstd_level`] is
/// `None`.
pub fn zstd_level(q: Quality) -> i32 {
    match q {
        Quality::UltraFast | Quality::VeryFast | Quality::Fast => 1,
        Quality::Basic => 6,
        Quality::Slow => 19,
        Quality::VerySlow => 22,
    }
}

/// Whether textures make their mips from mip 0 (alpha coverage, Toksvig roughness) even
/// when their sources hold every mip in a block format, instead of keeping the source's.
pub(crate) fn remakes_mips(q: Quality) -> bool {
    matches!(q, Quality::Slow | Quality::VerySlow)
}
