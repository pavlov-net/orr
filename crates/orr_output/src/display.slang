// Display encodings: display-linear Rec.709 (1.0 = paper white) to the signal a swapchain
// image of each `OutputFormat` holds, and back.
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

public float3 decode_display(float3 signal, uint encoding, float paper_white_nits)
{
    if (encoding == ENCODING_PQ)
        return mul(REC2020_TO_REC709, pq_decode(signal)) / paper_white_nits;
    if (encoding == ENCODING_SCRGB)
        return signal * (SCRGB_NITS / paper_white_nits);
    return srgb_decode(signal);
}
