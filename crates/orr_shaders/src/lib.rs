//! Slang shaders for orr.
//!
//! The default feature holds the types shared by Rust and Slang ([`abi`]) and the compiled
//! module types that generated code fills in. The `build` feature is for `build.rs`: it
//! compiles a crate's Slang kernels in process with the pinned Slang release, checks their
//! layout and ABI rules, and generates Rust for their roots, records and modules. The
//! `hot-reload` feature adds [`Watcher`], which recompiles compiled-in modules from their
//! sources at run time.

// orr's SPIR-V is little-endian words, and code using it casts them without swapping.
const _: () = assert!(cfg!(target_endian = "little"));

pub mod abi;
#[cfg(feature = "build")]
pub mod build;
mod module;

#[cfg(feature = "hot-reload")]
pub use build::{Reload, Watcher};
pub use module::{EntryPoint, KernelMismatch, KernelSet, Mismatch, PathEntry, ShaderModule, Stage};

#[doc(hidden)]
pub use bytemuck;
