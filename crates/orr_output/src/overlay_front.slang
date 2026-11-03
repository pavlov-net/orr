// Lighting's front layer over each pixel's dominant surface: reads blended from the dominant
// surface's value toward the layer's by the layer's share of the pixel (R16F `weight`; NONE
// without a layer).
module overlay_front;

import orr_shaders.abi;

public float weight_at(TextureIndex weight, int2 pixel)
{
    if (weight.index == IMAGE_NONE)
        return 0.0;
    return weight.get<Texture2D<float>>().Load(int3(pixel, 0));
}

// The reverse-Z at render pixel `pixel`.
public float blended_depth(TextureIndex dominant, TextureIndex front, TextureIndex weight, int2 pixel)
{
    float z = dominant.get<Texture2D<float>>().Load(int3(pixel, 0));
    float w = weight_at(weight, pixel);
    return w > 0.0 ? lerp(z, front.get<Texture2D<float>>().Load(int3(pixel, 0)), w) : z;
}

// The previous minus current UV at render pixel `pixel`.
public float2 blended_motion(TextureIndex dominant, TextureIndex front, TextureIndex weight, int2 pixel)
{
    float2 m = dominant.get<Texture2D<float2>>().Load(int3(pixel, 0));
    float w = weight_at(weight, pixel);
    return w > 0.0 ? lerp(m, front.get<Texture2D<float2>>().Load(int3(pixel, 0)), w) : m;
}
