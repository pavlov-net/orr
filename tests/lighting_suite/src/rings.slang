// Linked surfaces with their own coverage, shaded as standard surfaces: `Rings` is the bed's
// rings texture worked out from the uv, `Veil` a quarter of every triangle.
module rings;

import surface;

// The texel count a side of the bed's rings texture.
static const float RINGS_TEXELS = 128.0;

// 1 on the inner 60% of each of 8 rings out from the uv center, as the rings texture's
// texels hold it, else 0.
public float rings_coverage(float2 uv)
{
    float2 d = uv - 0.5 - 0.5 / RINGS_TEXELS;
    return frac(length(d) * 8.0) < 0.6 ? 1.0 : 0.0;
}

public static const float VEIL_COVERAGE = 0.25;

public struct Rings : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        return StandardSurface.surface(i, depth_offset);
    }

    public override static float coverage(SurfaceInput i)
    {
        return rings_coverage(i.uv.uv);
    }
}

public struct Veil : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        return StandardSurface.surface(i, depth_offset);
    }

    public override static float coverage(SurfaceInput i)
    {
        return VEIL_COVERAGE;
    }
}
