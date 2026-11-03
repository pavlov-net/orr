// Neural texture sets (orr_pack.neural) as kernels decode them: one texel of one color mip
// runs the set's decoder network over its latents with cooperative vectors.
module neural_texture;

import orr_shaders.abi;
import orr_color.transfer;
import orr_shaders.coopvec;
import orr_pack.neural;

// Where a color mip's texels read the latents (orr_pack::neural::ColorMip).
public struct NeuralMipGpu
{
    public int latent_mip;
    public float position_lod;
    public float position_scale;
    public uint _pad;
}

// The width every layer is padded to with zero weights and biases, so that one multiply in
// a loop runs all four layers.
public static const int NEURAL_WIDTH = 64;

// A neural texture set as kernels decode it.
public struct NeuralSetGpu
{
    public uint64_t weights;                // the layers' matrices and FP16 biases
    public uint matrices[4];                // NEURAL_WIDTH square, from `weights`, in the device's inference layout; multiples of 64
    public uint biases[4];                  // NEURAL_WIDTH halves, from `weights`; multiples of 16
    public uint fp8;                        // 1: the hidden layers' matrices are E4M3, else FP16; the output layer's are FP16
    public uint srgb;                       // bit per output channel stored sRGB-encoded
    public TextureIndex latents;            // A4R4G4B4 array: four features per layer, one mip per latent mip
    public SamplerIndex sampler;            // linear, repeating
    public uint2 size;                      // color mip 0 texels
    public uint mips;                       // color mips
    public uint latent_layers;
    public NeuralMipGpu color_mips[16];
}

// The last texel decoded by this invocation, which the texture slots of one material
// sharing a set and footprint read again.
static uint64_t cached_set = 0;
static uint3 cached_texel = uint3(0);
static float cached[NEURAL_OUTPUTS] = {};

Vector<NEURAL_WIDTH> neural_activation(Vector<NEURAL_WIDTH> x)
{
    typealias V = Vector<NEURAL_WIDTH>;
    return min(x, V(half(NEURAL_ACTIVATION_MAX))) * clamp(x * V(half(1.0 / 3.0)) + V(half(0.5)), V(half(0.0)), V(half(1.0)));
}

// The network's outputs at `texel` (wrapped into the mip) of color mip `mip` of `n`, sRGB
// channels decoded to linear, into the invocation's cache.
void neural_decode(NeuralSetGpu* n, int2 texel, uint mip)
{
    NeuralMipGpu c = n.color_mips[mip];
    int2 size = int2(max(n.size >> mip, uint2(1)));
    texel = ((texel % size) + size) % size;
    float2 uv = (float2(texel) + 0.5) / float2(size);
    Texture2DArray<float4> latents = n.latents.get<Texture2DArray<float4>>();
    SamplerState s = n.sampler.get();
    Vector<NEURAL_WIDTH> x = Vector<NEURAL_WIDTH>(half(0.0));
    [unroll]
    for (int level = 0; level < 2; level++)
    {
        [unroll]
        for (int layer = 0; layer < NEURAL_FEATURES / 4; layer++)
        {
            if (layer == 0 || layer < int(n.latent_layers))
            {
                // A4R4G4B4 holds features 0..3 in blue, green, red, alpha.
                float4 v = latents.SampleLevel(s, float3(uv, float(layer)), float(c.latent_mip + level)).bgra * 2.0 - 1.0;
                int at = level * NEURAL_FEATURES + layer * 4;
                x[at] = half(v.x);
                x[at + 1] = half(v.y);
                x[at + 2] = half(v.z);
                x[at + 3] = half(v.w);
            }
        }
    }
    float2 p = float2(texel) * c.position_scale;
    [unroll]
    for (int wave = 0; wave < NEURAL_POSITION_WAVES; wave++)
    {
        float4 e = neural_position_wave(p);
        int at = 2 * NEURAL_FEATURES + 4 * wave;
        x[at] = half(e.x);
        x[at + 1] = half(e.y);
        x[at + 2] = half(e.z);
        x[at + 3] = half(e.w);
        p *= 2.0;
    }
    int lod_at = 2 * NEURAL_FEATURES + 4 * NEURAL_POSITION_WAVES;
    x[lod_at] = half(c.position_lod);
    x[lod_at + 1] = half(c.position_lod);
    for (uint l = 0; l < 4; l++)
    {
        if (n.fp8 != 0 && l < 3)
            x = matmul_add<NEURAL_WIDTH, NEURAL_WIDTH>(x, n.weights, n.matrices[l], Weights.E4M3, n.weights, n.biases[l]);
        else
            x = matmul_add<NEURAL_WIDTH, NEURAL_WIDTH>(x, n.weights, n.matrices[l], Weights.F16, n.weights, n.biases[l]);
        if (l < 3)
            x = neural_activation(x);
        if (l == 2)
        {
            [unroll]
            for (int i = 0; i < NEURAL_HIDDEN2; i++)
                x[i] = half(clamp(round(float(x[i]) * NEURAL_QUANT_STEP + NEURAL_QUANT_BIAS), -128.0, 127.0));
        }
    }
    [unroll]
    for (int ch = 0; ch < NEURAL_OUTPUTS; ch++)
    {
        float v = float(x[ch]);
        cached[ch] = (n.srgb >> ch & 1) != 0 ? srgb_decode(float3(saturate(v))).x : v;
    }
    cached_set = (uint64_t)n;
    cached_texel = uint3(uint2(texel), mip);
}

// Channels `first` to `first + count - 1` (count 1 to 4) of `n`'s texel `texel` of color mip
// `mip`, wrapped; channels past `count` read 0, and 1 in w, as a texture with fewer channels
// reads.
public float4 neural_texel(NeuralSetGpu* n, int2 texel, uint mip, uint first, uint count)
{
    int2 size = int2(max(n.size >> mip, uint2(1)));
    uint3 key = uint3(uint2(((texel % size) + size) % size), mip);
    if (cached_set != (uint64_t)n || any(cached_texel != key))
        neural_decode(n, texel, mip);
    float4 v = float4(0.0, 0.0, 0.0, 1.0);
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        if (c < count)
            v[c] = cached[min(first + c, uint(NEURAL_OUTPUTS - 1))];
    }
    return v;
}
