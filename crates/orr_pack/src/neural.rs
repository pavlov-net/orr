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

pub const SCHEMA: u16 = 2;

/// Bytes of LibNTC's `NtcTextureSetConstants`.
const CONSTANTS_BYTES: usize = 336;

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
    /// channels as stored by a BC7 sRGB texture) or [`Self::SIGNED`] (values `v` of a snorm
    /// texture, which the network holds as `0.5 v + 0.5`).
    pub stored: u8,
    pub _pad: u8,
    /// Lowest PSNR over the texture's channels at mip 0 against the cook's input, dB.
    pub psnr: f32,
}

impl NeuralTexture {
    pub const LINEAR: u8 = 0;
    pub const SRGB: u8 = 1;
    pub const SIGNED: u8 = 2;
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
    /// (336 bytes long).
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
                || t.stored > NeuralTexture::SIGNED
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

/// The LibNTC major and minor version whose network the constants below describe; sets of
/// other versions hold another network.
pub const LIBRARY: [u16; 2] = [0, 10];

/// The decoder network's layers as (outputs, inputs): three hidden layers with the clamped
/// HGELU activation, then the output layer. Smaller networks in a set are padded with zero
/// weights to these sizes.
pub const LAYERS: [(usize, usize); 4] = [(64, 48), (48, 64), (32, 48), (16, 32)];

/// The network's inputs: the features of two latent mips, 16 each (zero past the set's
/// `latent_features`), then 3 four-value position encodings, then the color mip's position
/// level twice, then zeros.
pub const INPUTS: usize = 48;
/// Output channels of the network.
pub const OUTPUTS: usize = 16;

/// Color mips per set, at most.
pub const MAX_MIPS: usize = 16;

/// Where color mip `m`'s texels read the latents: latent mips `latent_mip` and
/// `latent_mip + 1`, and the position encoding of `texel * position_scale` with level
/// `position_lod`. A negative `latent_mip` marks a mip the set does not hold.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct ColorMip {
    pub latent_mip: i32,
    pub position_lod: f32,
    pub position_scale: f32,
    pub _pad: i32,
}

/// LibNTC's `NtcTextureSetConstants`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Constants {
    pub color_mips: [ColorMip; MAX_MIPS],
    /// Byte offsets of each layer's weights, biases and scales in the row's weight blob, in
    /// the layout of the row's weight type.
    pub weight_offsets: [i32; 4],
    pub bias_offsets: [i32; 4],
    pub scale_offsets: [i32; 4],
    pub width: i32,
    pub height: i32,
    pub mips: i32,
    pub _pad0: i32,
    /// Bit per output channel holding texture data.
    pub valid_channels: u32,
    /// Two bits per output channel: 0 linear, 1 sRGB, 2 hybrid log-gamma.
    pub color_spaces: u32,
    pub _pad1: [i32; 2],
}

impl Constants {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        read(bytes, 0).ok_or(Error::Malformed("neural set constants"))
    }
}

/// The 8-bit float with 4 exponent and 3 mantissa bits (E4M3, no infinities; `0x7f` and
/// `0xff` are NaN) that FP8 weights are stored as.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from(b >> 3 & 15);
    let m = f32::from(b & 7);
    if e == 15 && b & 7 == 7 {
        return f32::NAN;
    }
    if e == 0 {
        return sign * m / 8.0 * 2f32.powi(-6);
    }
    sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
}

/// The weights of a [`NeuralWeights::GENERIC_FP8`] or [`NeuralWeights::COOP_VEC_FP8`] row:
/// row-major E4M3 matrices with FP16 biases for the hidden layers, then a row-major Int8
/// matrix whose rows scale by FP32 `scale` after adding Int32 `bias` for the output layer.
/// The blob holds the four matrices, then the hidden biases, then the scales, then the
/// output biases.
#[derive(Clone, Debug, PartialEq)]
pub struct Fp8Network {
    pub hidden: [(Vec<u8>, Vec<half::f16>); 3],
    pub output: Vec<i8>,
    pub scale: Vec<f32>,
    pub bias: Vec<i32>,
}

impl Fp8Network {
    /// Blob bytes.
    pub const BYTES: usize = {
        let mut n = 0;
        let mut i = 0;
        while i < 4 {
            n += LAYERS[i].0 * LAYERS[i].1;
            i += 1;
        }
        n + 2 * (LAYERS[0].0 + LAYERS[1].0 + LAYERS[2].0) + 8 * OUTPUTS
    };

    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() != Self::BYTES {
            return Err(Error::Malformed("neural set weights"));
        }
        let mut at = 0;
        let mut take = |n: usize| {
            let b = &blob[at..at + n];
            at += n;
            b
        };
        let matrices: [&[u8]; 4] = std::array::from_fn(|l| take(LAYERS[l].0 * LAYERS[l].1));
        let hidden = std::array::from_fn(|l| {
            let biases = bytemuck::pod_collect_to_vec(take(2 * LAYERS[l].0));
            (matrices[l].to_vec(), biases)
        });
        Ok(Self {
            hidden,
            output: bytemuck::cast_slice(matrices[3]).to_vec(),
            scale: bytemuck::pod_collect_to_vec(take(4 * OUTPUTS)),
            bias: bytemuck::pod_collect_to_vec(take(4 * OUTPUTS)),
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::BYTES);
        for (m, _) in &self.hidden {
            out.extend_from_slice(m);
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.output));
        for (_, b) in &self.hidden {
            out.extend_from_slice(bytemuck::cast_slice(b));
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.scale));
        out.extend_from_slice(bytemuck::cast_slice(&self.bias));
        out
    }
}

const _: () = {
    assert!(size_of::<Constants>() == CONSTANTS_BYTES);
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
                stored: NeuralTexture::SIGNED,
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
                Vec::from_fn(CONSTANTS_BYTES, |i| i as u8),
            )],
        };
        let back = NeuralInfo::parse(&info.to_bytes()).unwrap();
        assert_eq!(back.textures, info.textures);
        assert_eq!(back.weights[0].1, info.weights[0].1);
        assert_eq!(back.weights[0].0.blob, 1);
    }

    #[test]
    fn e4m3_decodes_normals_subnormals_and_nan() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xc0), -2.0);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert!(e4m3_to_f32(0x7f).is_nan());
    }

    #[test]
    fn fp8_weights_round_trip() {
        let n = Fp8Network {
            hidden: std::array::from_fn(|l| {
                let (rows, cols) = LAYERS[l];
                (
                    Vec::from_fn(rows * cols, |i| (i * 7 + l) as u8),
                    Vec::from_fn(rows, |i| half::f16::from_f32(i as f32)),
                )
            }),
            output: Vec::from_fn(OUTPUTS * LAYERS[3].1, |i| i as i8),
            scale: Vec::from_fn(OUTPUTS, |i| i as f32 * 0.5),
            bias: (0..OUTPUTS as i32).map(|i| -i).collect(),
        };
        let bytes = n.to_bytes();
        assert_eq!(bytes.len(), Fp8Network::BYTES);
        assert_eq!(Fp8Network::parse(&bytes).unwrap(), n);
    }
}
