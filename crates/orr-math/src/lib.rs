//! Vector, matrix and quaternion types for orr's CPU-side code, re-exported from [`glam`].
//!
//! orr crates import math from here so the glam version and features are chosen in one
//! place. GPU-facing and on-disk structs use plain arrays such as `[f32; 3]`, not these
//! types; convert with `From`/`to_array`.
//!
//! Features `serde`, `bytemuck` and `mint` enable the matching glam integrations.

pub use glam;
pub use glam::{
    Affine3A, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4, IVec2, IVec3, IVec4, Mat3, Mat4, Quat,
    UVec2, UVec3, UVec4, Vec2, Vec3, Vec4,
};
