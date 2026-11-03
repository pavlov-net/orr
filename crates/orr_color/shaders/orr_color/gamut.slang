// SDR gamut compression: brings linear Rec.709 color from outside the Rec.709 gamut (some
// channels below 0) inside it and leaves colors near the achromatic axis as they are.
// ACES 1.3 Reference Gamut Compression applied in Rec.709 with the ACES thresholds: per
// channel, the distance d = (max - c) / max beyond the threshold is compressed so that the
// limit maps to 1. The limits are the largest distances any Rec.2020 color reaches in
// Rec.709, so every Rec.2020 color lands inside Rec.709 and distinct inputs stay distinct
// and ordered. Channels within their threshold are returned unchanged.
module gamut;

static const float3 GAMUT_THRESHOLD = float3(0.815, 0.803, 0.880);
static const float GAMUT_POWER = 1.2;
// (limit - threshold) / (((1 - threshold) / (limit - threshold))^-power - 1)^(1 / power),
// limits (1.5938696, 1.0874998, 1.1165758).
static const float3 GAMUT_SCALE = float3(0.217865883, 0.465177228, 0.195374901);

public float3 gamut_compress(float3 rgb)
{
    float ach = max(rgb.r, max(rgb.g, rgb.b));
    if (ach == 0.0)
        return rgb;
    float3 dist = (ach - rgb) / abs(ach);
    float3 over = max(dist - GAMUT_THRESHOLD, 0.0) / GAMUT_SCALE;
    float3 compressed = GAMUT_THRESHOLD + GAMUT_SCALE * over / pow(1.0 + pow(over, GAMUT_POWER), 1.0 / GAMUT_POWER);
    return select(dist < GAMUT_THRESHOLD, rgb, ach - compressed * abs(ach));
}
