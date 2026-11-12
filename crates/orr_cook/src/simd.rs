//! The vector instruction level of the cook's data-parallel loops.
//!
//! A loop runs as a function generic over [`Simd`] and annotated [`simd`], which compiles once
//! per level, called through [`dispatch`] at [`level`]. Every level cooks the same bytes: the
//! loops vectorize lane-wise IEEE operations in the order the scalar code states them, with no
//! fused multiply-add, reordered sums, estimates or lane-defined min and max. Target features
//! reach only code inlined into the annotated function, so a parallel loop dispatches inside
//! each piece of work, never around the loop.

use std::sync::OnceLock;

use fearless_simd::Level;
pub(crate) use fearless_simd::{Simd, dispatch};
pub(crate) use fearless_simd_macros::simd;

use crate::par::prelude::*;

static LEVEL: OnceLock<Level> = OnceLock::new();

/// Elements per piece of [`zip`], [`mapped`] and [`update`]: a light loop's piece, as each
/// element takes a few nanoseconds.
pub(crate) const PIECE: usize = crate::par::LIGHT_MIN;

/// The level of this CPU, or the one [`force_level`] set.
pub(crate) fn level() -> Level {
    *LEVEL.get_or_init(Level::new)
}

/// `f(&mut dst[i], &src[i])` over the shorter of the two, in parallel.
pub(crate) fn zip<T: Send, U: Sync>(dst: &mut [T], src: &[U], f: impl Fn(&mut T, &U) + Sync) {
    dst.par_chunks_mut(PIECE)
        .zip(src.par_chunks(PIECE))
        .for_each(|(d, s)| dispatch!(level(), simd => zip_piece(simd, d, s, &f)));
}

/// `f` over `src` into a new vector, in parallel.
pub(crate) fn mapped<T: Sync, U: Send + Copy + Default>(
    src: &[T],
    f: impl Fn(&T) -> U + Sync,
) -> Vec<U> {
    let mut dst = vec![U::default(); src.len()];
    zip(&mut dst, src, |d, s| *d = f(s));
    dst
}

/// `f` on every element of `v`, in parallel.
pub(crate) fn update<T: Send>(v: &mut [T], f: impl Fn(&mut T) + Sync) {
    v.par_chunks_mut(PIECE)
        .for_each(|p| dispatch!(level(), simd => update_piece(simd, p, &f)));
}

#[simd]
fn zip_piece<S: Simd, T, U>(_: S, dst: &mut [T], src: &[U], f: &impl Fn(&mut T, &U)) {
    for (d, s) in dst.iter_mut().zip(src) {
        f(d, s);
    }
}

#[simd]
fn update_piece<S: Simd, T>(_: S, v: &mut [T], f: &impl Fn(&mut T)) {
    v.iter_mut().for_each(f);
}

/// Runs the cook's vector loops at the level named `name` (`sse2`, `sse4.2`, `avx2`, `avx512`
/// or `neon`, as the CPU has them, or `fallback` with the `simd-fallback` feature) in place of
/// the best one, for the rest of the process. Fails if the CPU lacks the level or a loop
/// already ran.
pub fn force_level(name: &str) -> Result<(), String> {
    let detected = Level::new();
    let forced = match name {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "sse2" => detected.as_sse2().map(Simd::level),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "sse4.2" => detected.as_sse4_2().map(Simd::level),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "avx2" => detected.as_avx2().map(Simd::level),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "avx512" => detected.as_avx512().map(Simd::level),
        #[cfg(target_arch = "aarch64")]
        "neon" => detected.as_neon().map(Simd::level),
        #[cfg(feature = "simd-fallback")]
        "fallback" => Some(Level::fallback()),
        _ => return Err(format!("unknown SIMD level {name}")),
    }
    .ok_or_else(|| format!("this CPU has no {name}"))?;
    LEVEL
        .set(forced)
        .map_err(|_| "the SIMD level is already in use".to_string())
}
