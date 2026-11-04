// Tangent-space normal maps as orr_pack::texture::normal stores them: x and y at
// `0.5 v + 0.5`, z rebuilt.
module normal_map;

// The tangent-space normal of stored x and y.
public float3 normal_decode(float2 stored)
{
    float2 xy = stored * 2.0 - 1.0;
    return float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
}
