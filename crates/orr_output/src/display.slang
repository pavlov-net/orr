// Display encodings: display-linear color (1.0 = paper white) to the signal a swapchain image
// of each `OutputFormat` holds, and back. Display-linear color is in the encoding's primaries:
// Rec.2020 for PQ, Rec.709 for SDR and scRGB. `signal_white_nits` is paper white's level in the
// encoding's nits.
module display;

import orr_color.primaries;
import orr_color.transfer;

public static const uint ENCODING_SDR = 0;
public static const uint ENCODING_PQ = 1;
public static const uint ENCODING_SCRGB = 2;

// Linear Rec.2020 in the primaries of `encoding`'s display-linear color.
public float3 display_from_rec2020(float3 rec2020, uint encoding)
{
    return encoding == ENCODING_PQ ? rec2020 : mul(REC2020_TO_REC709, rec2020);
}

// Linear color in primaries `primaries` (a `PRIMARIES_*` id) in the primaries of `encoding`'s
// display-linear color.
public float3 display_from(float3 c, uint primaries, uint encoding)
{
    if (primaries == (encoding == ENCODING_PQ ? PRIMARIES_REC2020 : PRIMARIES_REC709))
        return c;
    if (primaries == PRIMARIES_REC709)
        c = mul(REC709_TO_REC2020, c);
    else if (primaries == PRIMARIES_DISPLAY_P3)
        c = mul(P3_TO_REC2020, c);
    return display_from_rec2020(c, encoding);
}

public float3 encode_display(float3 linear, uint encoding, float signal_white_nits)
{
    if (encoding == ENCODING_PQ)
        return pq_encode(linear * signal_white_nits);
    if (encoding == ENCODING_SCRGB)
        return linear * (signal_white_nits / SCRGB_WHITE_NITS);
    return srgb_encode(saturate(linear));
}

public float3 decode_display(float3 signal, uint encoding, float signal_white_nits)
{
    if (encoding == ENCODING_PQ)
        return pq_decode(signal) / signal_white_nits;
    if (encoding == ENCODING_SCRGB)
        return signal * (SCRGB_WHITE_NITS / signal_white_nits);
    return srgb_decode(signal);
}
