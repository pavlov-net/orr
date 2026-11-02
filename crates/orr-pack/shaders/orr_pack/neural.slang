// The decoder network of orr_pack::neural sets: layer sizes, the clamped HGELU activation,
// the output layer's input quantization and the position encoding.
module neural;

// Layer outputs and inputs: three hidden layers, then the output layer.
public static const int NEURAL_INPUTS = 48;
public static const int NEURAL_HIDDEN0 = 64;
public static const int NEURAL_HIDDEN1 = 48;
public static const int NEURAL_HIDDEN2 = 32;
public static const int NEURAL_OUTPUTS = 16;

// Features per latent mip in the inputs; latent texels hold four per layer.
public static const int NEURAL_FEATURES = 16;
public static const int NEURAL_POSITION_WAVES = 3;

// The activation is `min(x, NEURAL_ACTIVATION_MAX) * saturate(x / 3 + 0.5)`; the last hidden
// layer's outputs reach the output layer as `round(NEURAL_QUANT_STEP * y + NEURAL_QUANT_BIAS)`.
public static const float NEURAL_ACTIVATION_MAX = 3.0;
public static const float NEURAL_QUANT_STEP = 80.0;
public static const float NEURAL_QUANT_BIAS = -113.0;

// One wave of the position encoding of `p` (texel times the mip's position scale); each
// next wave doubles `p`.
public float4 neural_position_wave(float2 p)
{
    return float4(frac(p), frac(p + 0.25)) * 2.0 - 1.0;
}
