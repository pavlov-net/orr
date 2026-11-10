// Linked surfaces with their own coverage, shaded as standard surfaces: `Rings` is the alpha
// test of the bed's rings texture worked out from the uv, `Veil` a quarter of every triangle.
module rings;

import orr_scene.surface;

// The texel count a side of the bed's rings texture, a power of two.
static const int RINGS_TEXELS = 128;

// Texels across one ring of the rings texture, and across its opaque inner 60%.
static const float RING_TEXELS = 16.0;
static const float RING_OPAQUE_TEXELS = 9.6;

// Texel `t` of the bed's rings texture: 1 on the opaque part of each ring out from the middle
// texel, else 0. Squared distances compare exactly, where a root's rounding could put a texel
// that starts a ring on either side of it.
float rings_texel(int2 t)
{
    int2 d = t - RINGS_TEXELS / 2;
    float squared = float(dot(d, d));
    // The texture's corners are the farthest texels, under 6 rings out.
    for (uint ring = 0; ring < 6; ring++)
    {
        float inner = RING_TEXELS * float(ring);
        float outer = inner + RING_OPAQUE_TEXELS;
        if (squared >= inner * inner && squared < outer * outer)
            return 1.0;
    }
    return 0.0;
}

// The alpha test the bed's leaf material makes of its rings texture at `uv`
// (`material_coverage`): the texture's mip 0 read bilinearly, repeating, against a cutoff of
// a half.
public float rings_coverage(float2 uv)
{
    float2 x = uv * float(RINGS_TEXELS) - 0.5;
    float2 below = floor(x);
    float2 w = x - below;
    float opacity = 0.0;
    for (uint k = 0; k < 4; k++)
    {
        int2 o = int2(k & 1, k >> 1);
        float2 b = select(o == 1, w, 1.0 - w);
        opacity += b.x * b.y * rings_texel((int2(below) + o) & (RINGS_TEXELS - 1));
    }
    return opacity >= 0.5 ? 1.0 : 0.0;
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
        return rings_coverage(i.uv[0].uv);
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
