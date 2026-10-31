//! Neural texture set class: textures of one material compressed together by RTXNTC's
//! LibNTC into latent images and a decoder network. A set stands in for BC textures that are
//! cooked and shipped beside it.
//!
//! - Blob 0 holds a [`NeuralHeader`], its texture table, its weight table and the inference
//!   constants ([`NeuralInfo`]). Then one blob per weight table row.
//! - The latent images are the texture entry [`NeuralHeader::latents`]:
//!   [`vk::A4R4G4B4_UNORM_PACK16`](crate::texture::vk::A4R4G4B4_UNORM_PACK16), four features
//!   per layer, one mip per latent mip, laid out and streamed like any texture.
//! - Weights and constants are LibNTC's own layouts for one weight type (`NtcTextureSetConstants`
//!   for the constants), read by the inference shaders of the LibNTC version in
//!   [`NeuralHeader::library`]. A set from another version is not readable by them.

use bytemuck::{Pod, Zeroable};

use crate::blob::{put_table, read, table};
use crate::{AssetId, Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Bytes of LibNTC's `NtcTextureSetConstants`.
pub const CONSTANTS_BYTES: usize = 336;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct NeuralHeader {
    /// LibNTC version that wrote the set: major, minor, point.
    pub library: [u16; 3],
    pub _pad: u16,
    /// LibNTC's `InterfaceVersion` and NTC file schema version.
    pub interface: u32,
    pub file_schema: u32,
    /// Mip 0 texels per axis; every texture in the set has this size.
    pub width: u32,
    pub height: u32,
    pub mip_count: u8,
    /// Channels the network outputs, 1..16.
    pub channel_count: u8,
    /// Features per latent texel: 4, 8, 12 or 16; the latent texture has `features / 4`
    /// layers.
    pub latent_features: u8,
    pub _pad1: u8,
    /// Latent and weight bits per mip 0 texel.
    pub bits_per_pixel: f32,
    /// Texture entry holding the latent images.
    pub latents: AssetId,
    /// [`NeuralTexture`] table.
    pub textures: Span,
    /// [`NeuralWeights`] table.
    pub weights: Span,
}

/// A texture of the set: channels of the network output that stand in for a BC texture
/// entry. Textures whose quality fell under the cook's floor are left out of the table and
/// sampled from their BC entry.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct NeuralTexture {
    /// The BC texture entry this one stands in for; its channel `c` is output channel
    /// `first_channel + c`.
    pub texture: AssetId,
    pub first_channel: u8,
    pub channel_count: u8,
    /// How output values map to the BC texture's: [`Self::LINEAR`], [`Self::SRGB`] (color
    /// channels as stored by a BC7 sRGB texture) or [`Self::NORMAL`] (x and y as
    /// `0.5 v + 0.5`, z reconstructed, as BC5 normal maps store them).
    pub stored: u8,
    pub _pad: u8,
    /// Lowest PSNR over the texture's channels at mip 0 against the cook's input, dB.
    pub psnr: f32,
}

impl NeuralTexture {
    pub const LINEAR: u8 = 0;
    pub const SRGB: u8 = 1;
    pub const NORMAL: u8 = 2;
}

/// Network weights of one LibNTC weight type.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct NeuralWeights {
    /// LibNTC's `InferenceWeightType`: [`Self::GENERIC_INT8`], [`Self::GENERIC_FP8`] or
    /// [`Self::COOP_VEC_FP8`].
    pub kind: u32,
    /// Blob holding the weights.
    pub blob: u32,
    /// For cooperative-vector weights, bytes after the device converts the blob to its
    /// optimal matrix layout; 0 otherwise.
    pub converted_bytes: u64,
    /// The `NtcTextureSetConstants` for these weights: a byte range of blob 0
    /// ([`CONSTANTS_BYTES`] long).
    pub constants: Span,
}

impl NeuralWeights {
    pub const GENERIC_INT8: u32 = 1;
    pub const GENERIC_FP8: u32 = 2;
    pub const COOP_VEC_FP8: u32 = 3;
}

/// Blob 0 of a neural texture set.
#[derive(Clone, Debug, PartialEq)]
pub struct NeuralInfo {
    pub header: NeuralHeader,
    pub textures: Vec<NeuralTexture>,
    /// With each row, its constants.
    pub weights: Vec<(NeuralWeights, Vec<u8>)>,
}

impl NeuralInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let header: NeuralHeader = read(blob, 0).ok_or(Error::Malformed("neural set header"))?;
        let h = &header;
        if h.width == 0
            || h.height == 0
            || h.mip_count == 0
            || !(1..=16).contains(&h.channel_count)
            || !matches!(h.latent_features, 4 | 8 | 12 | 16)
        {
            return Err(Error::Malformed("neural set header"));
        }
        let textures: Vec<NeuralTexture> = table(blob, h.textures)?;
        if textures.iter().any(|t| {
            t.channel_count == 0
                || t.stored > NeuralTexture::NORMAL
                || u32::from(t.first_channel) + u32::from(t.channel_count)
                    > u32::from(h.channel_count)
        }) {
            return Err(Error::Malformed("neural set texture"));
        }
        let weights = table::<NeuralWeights>(blob, h.weights)?
            .into_iter()
            .map(|w| {
                if w.constants.count as usize != CONSTANTS_BYTES {
                    return Err(Error::Malformed("neural set constants"));
                }
                Ok((w, table::<u8>(blob, w.constants)?))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            header,
            textures,
            weights,
        })
    }

    /// Blob bytes; the header's spans and each row's constants span are set from the
    /// tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut out = vec![0; size_of::<NeuralHeader>()];
        h.textures = put_table(&mut out, &self.textures);
        let rows: Vec<NeuralWeights> = self
            .weights
            .iter()
            .map(|(w, c)| NeuralWeights {
                constants: put_table(&mut out, c),
                ..*w
            })
            .collect();
        h.weights = put_table(&mut out, &rows);
        out[..size_of::<NeuralHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    /// The row of weight type `kind`.
    pub fn weights(&self, kind: u32) -> Option<&(NeuralWeights, Vec<u8>)> {
        self.weights.iter().find(|(w, _)| w.kind == kind)
    }
}

const _: () = {
    assert!(size_of::<NeuralHeader>() == 64);
    assert!(size_of::<NeuralTexture>() == 24);
    assert!(size_of::<NeuralWeights>() == 24);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_round_trips() {
        let info = NeuralInfo {
            header: NeuralHeader {
                library: [0, 10, 0],
                _pad: 0,
                interface: 0x2512_1500,
                file_schema: 3,
                width: 512,
                height: 256,
                mip_count: 10,
                channel_count: 9,
                latent_features: 12,
                _pad1: 0,
                bits_per_pixel: 4.5,
                latents: AssetId::from_name("m/neural/0/latents"),
                textures: Span::default(),
                weights: Span::default(),
            },
            textures: vec![NeuralTexture {
                texture: AssetId::from_name("m/texture/0/normal"),
                first_channel: 3,
                channel_count: 2,
                stored: NeuralTexture::NORMAL,
                _pad: 0,
                psnr: 41.5,
            }],
            weights: vec![(
                NeuralWeights {
                    kind: NeuralWeights::GENERIC_FP8,
                    blob: 1,
                    converted_bytes: 0,
                    constants: Span::default(),
                },
                (0..CONSTANTS_BYTES).map(|i| i as u8).collect(),
            )],
        };
        let back = NeuralInfo::parse(&info.to_bytes()).unwrap();
        assert_eq!(back.textures, info.textures);
        assert_eq!(back.weights[0].1, info.weights[0].1);
        assert_eq!(back.weights[0].0.blob, 1);
    }
}
