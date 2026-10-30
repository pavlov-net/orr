// Display encodings: display-linear Rec.709 (1.0 = paper white) to the signal a swapchain
// image of each `OutputFormat` holds.
module display;

import orr.color;

public static const uint ENCODING_SDR = 0;
public static const uint ENCODING_PQ = 1;
public static const uint ENCODING_SCRGB = 2;

// scRGB's 1.0 in nits.
static const float SCRGB_NITS = 80.0;

public float3 encode_display(float3 linear, uint encoding, float paper_white_nits)
{
    if (encoding == ENCODING_PQ)
        return pq_encode(mul(REC709_TO_REC2020, linear) * paper_white_nits);
    if (encoding == ENCODING_SCRGB)
        return linear * (paper_white_nits / SCRGB_NITS);
    return srgb_encode(saturate(linear));
}
