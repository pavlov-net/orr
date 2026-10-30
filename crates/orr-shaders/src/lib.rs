//! Slang shaders for orr.
//!
//! The default feature holds the types shared by Rust and Slang ([`abi`]) and the compiled
//! module types that generated code fills in. The `build` feature is for `build.rs`: it
//! compiles a crate's Slang kernels in process with the pinned Slang release, checks their
//! layout and ABI rules, and generates Rust for their roots, records and modules.

pub mod abi;
mod alias;
#[cfg(feature = "build")]
pub mod build;
mod module;

pub use alias::AliasEntry;
pub use module::{EntryPoint, KernelSet, ShaderModule, Stage};

/// The directory of orr's shared Slang libraries, imported as `orr.<name>`.
pub const LIB_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");

#[doc(hidden)]
pub use bytemuck;
