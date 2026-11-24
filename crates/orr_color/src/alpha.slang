module alpha;

// Premultiply straight RGBA after applying opacity to alpha.
public float4 premultiply_alpha(float4 color, float opacity = 1.0)
{
    float alpha = color.a * opacity;
    return float4(color.rgb * alpha, alpha);
}

// Source-over compositing of premultiplied RGBA in the same color space.
public float4 over_premultiplied(float4 source, float4 destination)
{
    return source + (1.0 - source.a) * destination;
}
