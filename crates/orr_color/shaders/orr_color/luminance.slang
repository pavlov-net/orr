// Luminance (CIE Y) of linear color.
module luminance;

import orr_color.primaries;

public float luminance_rec709(float3 c)
{
    return dot(REC709_TO_XYZ[1], c);
}

public float luminance_rec2020(float3 c)
{
    return dot(REC2020_TO_XYZ[1], c);
}

// The luminance ray reconstruction's depth-of-field guide takes.
public float guide_luminance(float3 c)
{
    return dot(c, float3(0.25, 0.5, 0.25));
}
