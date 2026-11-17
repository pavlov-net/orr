//! Vector, matrix and quaternion types for orr's CPU-side code: the [`glam`] prelude,
//! re-exported, and the sample [`jitter`] of temporal reconstruction.
//!
//! orr crates import math from here so the glam version and features are chosen in one
//! place. GPU-facing and on-disk structs use plain arrays such as `[f32; 3]`, not these
//! types; convert with `From`/`to_array`.
//!
//! Features `serde`, `bytemuck` and `mint` enable the matching glam integrations.

pub mod jitter;

pub use glam::prelude::*;

/// Projection and view matrices per handedness and graphics API.
pub use glam::camera;
