//! Clip class: a skeleton's local joint transforms sampled at a fixed rate.
//!
//! - Blob 0 (mount): a [`ClipHeader`].
//! - Blob 1: `frame_count x joint_count` [`Trs`] records, frame-major: joint `j` at frame `f`
//!   is record `f x joint_count + j`, ten f32 (rotation xyzw, translation, scale) that a
//!   consumer kernel reads in place. Joints follow the skeleton's order; joints the source
//!   does not animate hold their rest transform.
//!
//! Frame `f` is the pose `f / sample_rate` seconds after the clip starts, and the last frame
//! is its end, so a looping clip wraps from `duration` to 0. Consecutive frames of a joint
//! hold rotations with a non-negative dot product. [`Clip::sample`] is the reference
//! sampler, for GPU samplers too.

use bytemuck::{Pod, Zeroable};

use crate::blob::read;
use crate::skeleton::Trs;
use crate::{AssetId, Error, Result};

pub const SCHEMA: u16 = 1;

/// Blob 0, 32 B.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct ClipHeader {
    /// The skeleton the clip poses.
    pub skeleton: AssetId,
    pub joint_count: u32,
    /// At least 1; a one-frame clip is a fixed pose.
    pub frame_count: u32,
    /// Frames per second.
    pub sample_rate: f32,
    /// Seconds: `(frame_count - 1) / sample_rate`.
    pub duration: f32,
}

/// A decoded clip: the header and its samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    pub header: ClipHeader,
    pub samples: Vec<Trs>,
}

impl Clip {
    /// From blob 0 and blob 1.
    pub fn parse(head: &[u8], samples: &[u8]) -> Result<Self> {
        let header: ClipHeader = read(head, 0).ok_or(Error::Malformed("clip header"))?;
        let n = header.frame_count as usize * header.joint_count as usize;
        if header.frame_count == 0
            || !header.sample_rate.is_finite()
            || header.sample_rate <= 0.0
            || samples.len() != n * size_of::<Trs>()
        {
            return Err(Error::Malformed("clip samples"));
        }
        Ok(Self {
            header,
            samples: samples
                .chunks_exact(size_of::<Trs>())
                .map(bytemuck::pod_read_unaligned)
                .collect(),
        })
    }

    /// Blob 0 and blob 1.
    pub fn to_bytes(&self) -> (Vec<u8>, Vec<u8>) {
        (
            bytemuck::bytes_of(&self.header).to_vec(),
            bytemuck::cast_slice(&self.samples).to_vec(),
        )
    }

    /// The joints' local transforms at frame `f`.
    pub fn frame(&self, f: u32) -> &[Trs] {
        let n = self.header.joint_count as usize;
        &self.samples[f as usize * n..][..n]
    }

    /// Local transforms of every joint at `t` seconds, clamped to the clip: the two
    /// neighbouring frames interpolated by [`Trs::interpolate`].
    pub fn sample(&self, t: f32, out: &mut [Trs]) {
        let h = &self.header;
        assert!(out.len() >= h.joint_count as usize, "one output per joint");
        if h.frame_count == 1 {
            out[..h.joint_count as usize].copy_from_slice(self.frame(0));
            return;
        }
        let x = (t * h.sample_rate).clamp(0.0, (h.frame_count - 1) as f32);
        let f = (x as u32).min(h.frame_count - 2);
        let u = x - f as f32;
        let (a, b) = (self.frame(f), self.frame(f + 1));
        for (o, (a, b)) in out.iter_mut().zip(a.iter().zip(b)) {
            *o = a.interpolate(b, u);
        }
    }
}
