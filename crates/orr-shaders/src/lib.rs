//! Slang shaders for orr.
//!
//! The `build` feature is for `build.rs`: it locates and version-checks the pinned Slang
//! compiler and holds the one compiler option set.

#[cfg(feature = "build")]
pub mod build;
