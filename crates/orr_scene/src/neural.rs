//! Neural texture sets: which sets a device decodes, at which precision, and their decoder
//! weights in the device's matrix layout.

use std::sync::Arc;

use half::f16;
use orr_gpu::{
    ComponentType, Device, GpuPtr, MATRIX_ALIGN, MatrixConversion, MatrixFormat, MatrixLayout,
};
use orr_pack::AssetId;
use orr_pack::neural::{
    self, Constants, Fp8Network, LAYERS, NeuralInfo, NeuralTexture, NeuralWeights, e4m3_to_f32,
};

use crate::tiles::TexStream;
use crate::{NeuralMipGpu, NeuralSetGpu, NeuralTextureGpu, Result, TextureStreamGpu};

/// Which textures decode from neural texture sets ([`crate::RendererDesc::neural_textures`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NeuralTextures {
    /// Every texture samples its BC entry.
    Off,
    /// Textures a set stands in for decode from it where the device and the shaders can:
    /// with FP8 hidden layers on devices with `shaderFloat8`, FP16 elsewhere.
    #[default]
    Fastest,
    /// As `Fastest`, with FP16 hidden layers on every device.
    Fp16,
}

/// The precision of a set's hidden layers on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Precision {
    Fp8,
    Fp16,
}

/// The precision `device` decodes sets at under `mode`, or none when textures sample their
/// BC entries: the shaders were built without neural decoding (`ORR_SHADER_NEURAL=0`) or
/// the device lacks the cooperative-vector combinations the decoder multiplies with.
pub(crate) fn precision(device: &Device, mode: NeuralTextures) -> Option<Precision> {
    if !cfg!(orr_neural) || mode == NeuralTextures::Off {
        return None;
    }
    let caps = device.caps();
    let has = |m: ComponentType| {
        caps.cooperative_vector.iter().any(|c| {
            c.input == ComponentType::F16
                && c.input_interpretation == m
                && c.matrix_interpretation == m
                && c.bias_interpretation == ComponentType::F16
                && c.result == ComponentType::F16
        })
    };
    if !has(ComponentType::F16) {
        return None;
    }
    let fp8 = mode == NeuralTextures::Fastest && caps.shader_float8 && has(ComponentType::E4m3);
    Some(if fp8 { Precision::Fp8 } else { Precision::Fp16 })
}

/// A set the decoder reads: its FP8 weight row and constants. None for sets of another
/// LibNTC version or without FP8 weights, whose textures sample their BC entries.
pub(crate) fn readable(info: &NeuralInfo) -> Option<(NeuralWeights, Constants)> {
    let h = &info.header;
    if h.library[..2] != neural::LIBRARY || usize::from(h.mip_count) > neural::MAX_MIPS {
        return None;
    }
    let (row, bytes) = info
        .weights(NeuralWeights::GENERIC_FP8)
        .or_else(|| info.weights(NeuralWeights::COOP_VEC_FP8))?;
    let c = Constants::parse(bytes).ok()?;
    let mips = usize::from(h.mip_count);
    let sized = c.width == h.width as i32 && c.height == h.height as i32;
    let mapped = c.color_mips[..mips].iter().all(|m| m.latent_mip >= 0);
    (sized && c.mips == i32::from(h.mip_count) && mapped).then_some((*row, c))
}

/// Bit per output channel that `info`'s textures store sRGB-encoded.
fn srgb_channels(info: &NeuralInfo) -> u32 {
    info.textures
        .iter()
        .filter(|t| t.stored == NeuralTexture::SRGB)
        .map(|t| ((1u32 << t.channel_count.min(3)) - 1) << t.first_channel)
        .fold(0, |a, b| a | b)
}

/// A set's decoder weights as the device multiplies them, and the set's record without its
/// weights' address, latents and sampler.
pub(crate) struct DeviceWeights {
    bytes: Vec<MatrixBlock>,
    len: usize,
    pub(crate) record: NeuralSetGpu,
}

impl DeviceWeights {
    /// The weights, a multiple of 16 bytes.
    pub(crate) fn bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.bytes)[..self.len]
    }
}

/// Weights memory comes in these, so a matrix at a multiple of [`MATRIX_ALIGN`] from its start
/// is at an aligned host address, as converting a matrix into it needs.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(64))]
struct MatrixBlock([u8; 64]);

const _: () = assert!(align_of::<MatrixBlock>() as u64 == MATRIX_ALIGN);

/// Every layer is padded with zero weights and biases to this width, so that kernels run
/// the four layers as one multiply in a loop (`NEURAL_WIDTH`).
const WIDTH: usize = 64;

/// One layer's FP16 biases. Matrices end at a multiple of [`MATRIX_ALIGN`], so biases fall on
/// the multiples of 16 kernels read them at.
const BIAS_BYTES: usize = 2 * WIDTH;

/// `rows` x `cols` row-major `elements` of `bytes` each, padded to [`WIDTH`] square.
fn padded(m: &[u8], rows: usize, cols: usize, bytes: usize) -> Vec<u8> {
    let mut out = vec![0; WIDTH * WIDTH * bytes];
    for r in 0..rows {
        let src = &m[r * cols * bytes..(r + 1) * cols * bytes];
        out[r * WIDTH * bytes..][..cols * bytes].copy_from_slice(src);
    }
    out
}

/// Converts `net` into the device's inference layout, each layer padded to [`WIDTH`]
/// square: the hidden layers' matrices E4M3 or FP16 by `p`, the output layer's FP16 with
/// each row's scale folded into its weights and bias, then the FP16 biases. Offsets in the
/// record are from the start of `bytes`, which must land at a multiple of [`MATRIX_ALIGN`].
pub(crate) fn device_weights(
    device: &Device,
    info: &NeuralInfo,
    constants: &Constants,
    net: &Fp8Network,
    p: Precision,
) -> Result<DeviceWeights> {
    let (rows3, cols3) = LAYERS[3];
    let output: Vec<f16> = (0..rows3 * cols3)
        .map(|i| f16::from_f32(f32::from(net.output[i]) * net.scale[i / cols3]))
        .collect();
    let mut sources: Vec<(ComponentType, Vec<u8>)> = net
        .hidden
        .iter()
        .enumerate()
        .map(|(l, (m, _))| {
            let (rows, cols) = LAYERS[l];
            match p {
                Precision::Fp8 => (ComponentType::E4m3, padded(m, rows, cols, 1)),
                Precision::Fp16 => {
                    let h: Vec<f16> = m.iter().map(|&b| f16::from_f32(e4m3_to_f32(b))).collect();
                    let h = bytemuck::cast_slice(&h);
                    (ComponentType::F16, padded(h, rows, cols, 2))
                }
            }
        })
        .collect();
    sources.push((
        ComponentType::F16,
        padded(bytemuck::cast_slice(&output), rows3, cols3, 2),
    ));
    let n = WIDTH as u32;
    let mut matrices = [0u32; 4];
    let mut at = 0u64;
    for (l, (component, _)) in sources.iter().enumerate() {
        let format = MatrixFormat::new(*component, MatrixLayout::InferencingOptimal);
        matrices[l] = at as u32;
        at = (at + device.matrix_bytes(n, n, format)?).next_multiple_of(MATRIX_ALIGN);
    }
    let biases: [u32; 4] = std::array::from_fn(|l| (at as usize + l * BIAS_BYTES) as u32);
    let len = at as usize + biases.len() * BIAS_BYTES;
    // Zeros fill the padding of biases narrower than WIDTH, which kernels read in full.
    let mut bytes = vec![MatrixBlock([0; 64]); len.div_ceil(size_of::<MatrixBlock>())];
    let out: &mut [u8] = bytemuck::cast_slice_mut(&mut bytes);
    for (l, (component, src)) in sources.iter().enumerate() {
        let c = MatrixConversion {
            rows: n,
            columns: n,
            src_format: MatrixFormat::new(*component, MatrixLayout::RowMajor),
            dst_format: MatrixFormat::new(*component, MatrixLayout::InferencingOptimal),
        };
        device.convert_matrix(&c, src, &mut out[matrices[l] as usize..])?;
    }
    let output_bias: Vec<f16> = (0..rows3)
        .map(|r| f16::from_f32(net.bias[r] as f32 * net.scale[r]))
        .collect();
    for (l, b) in net
        .hidden
        .iter()
        .map(|(_, b)| b.as_slice())
        .chain([&output_bias[..]])
        .enumerate()
    {
        let b: &[u8] = bytemuck::cast_slice(b);
        out[biases[l] as usize..][..b.len()].copy_from_slice(b);
    }
    let h = &info.header;
    let mut color_mips = [<NeuralMipGpu as bytemuck::Zeroable>::zeroed(); neural::MAX_MIPS];
    for (g, c) in color_mips
        .iter_mut()
        .zip(&constants.color_mips[..usize::from(h.mip_count)])
    {
        *g = NeuralMipGpu {
            latent_mip: c.latent_mip,
            position_lod: c.position_lod,
            position_scale: c.position_scale,
            _pad: 0,
        };
    }
    Ok(DeviceWeights {
        bytes,
        len,
        record: NeuralSetGpu {
            weights: 0,
            matrices,
            biases,
            fp8: u32::from(p == Precision::Fp8),
            srgb: srgb_channels(info),
            latents: orr_gpu::TextureIndex::NONE,
            sampler: orr_gpu::SamplerIndex(0),
            size: [h.width, h.height],
            mips: u32::from(h.mip_count),
            latent_layers: u32::from(h.latent_features / 4),
            color_mips,
        },
    })
}

/// A texture's link to the neural set that stands in for it.
pub(crate) struct NeuralLink {
    pub(crate) set: AssetId,
    pub(crate) record: GpuPtr<NeuralTextureGpu>,
    /// The texture's own stream record when it does not stream, else null.
    pub(crate) own: GpuPtr<TextureStreamGpu>,
    /// The record's contents; `set` is filled in once the set is placed.
    pub(crate) gpu: NeuralTextureGpu,
    /// The set's latents when they stream, and the latent mip of each color mip.
    pub(crate) latents: Option<Arc<TexStream>>,
    pub(crate) latent_mips: Vec<u32>,
}
