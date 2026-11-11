// Normal maps as orr_pack::texture stores them. Tangent-space maps (`normal`) hold x and y as
// they are, z rebuilt; maps of mesh space directions (`object_normal`) hold each component at
// `(127 v + 128) / 255`.
module normal_map;

// The tangent-space normal of stored x and y.
public float3 normal_decode(float2 xy)
{
    return float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
}

// The mesh space direction of a stored texel, not yet unit.
public float3 object_normal_decode(float3 stored)
{
    return (stored * 255.0 - 128.0) / 127.0;
}