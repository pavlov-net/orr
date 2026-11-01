// Exposure units and adaptation. Scene luminance L is in cd/m2; EV100 is the reflected-light
// meter value at ISO 100 with K = 12.5. Exposure s scales luminance to exposed values
// (1.0 = paper white); auto exposure aims the metered average at MID_GRAY.
module exposure;

public static const float MID_GRAY = 0.18;

// The meter's histogram: per bin, the pixel weight, then sums of weight times each pixel's
// offset inside the bin in 1/256 of a bin, its CIE x and its CIE y, both in 1/65536, each
// sum as two words, low first. The chromaticity sums are 0 unless the meter is asked for
// them.
public static const uint HISTOGRAM_BINS = 128;
public static const uint HISTOGRAM_WORDS = 7;
public static const float BIN_OFFSET_STEPS = 256.0;
public static const float CHROMA_STEPS = 65536.0;

// Entries of the compensation curve's table, spread evenly over the histogram's range.
public static const uint COMPENSATION_ENTRIES = 256;

public float ev100_from_luminance(float luminance)
{
    return log2(luminance) + 3.0;
}

public float luminance_from_ev100(float ev100)
{
    return exp2(ev100 - 3.0);
}

// Manual exposure at `ev100`.
public float exposure_from_ev100(float ev100)
{
    return 1.0 / (1.2 * exp2(ev100));
}

public float ev100_from_exposure(float exposure)
{
    return -log2(1.2 * exposure);
}

// The exposure EV100 that shows a scene metered at `metered_ev100` at MID_GRAY * 2^stops.
public float keyed_ev100(float metered_ev100, float stops)
{
    return ev100_from_exposure(MID_GRAY * exp2(stops) / luminance_from_ev100(metered_ev100));
}

// One step of `dt` seconds from `current` toward `target`, both EV100: `speed_up` stops per
// second while the target is above, `speed_down` while below, linear while more than
// `exponential_ev` away and exponential inside it, never past the target.
public float adapt_ev100(float current, float target, float speed_up, float speed_down,
                         float exponential_ev, float dt)
{
    float d = target - current;
    if (d == 0.0)
        return current;
    float speed = d > 0.0 ? speed_up : speed_down;
    if (abs(d) > exponential_ev)
        return current + sign(d) * min(speed * dt, abs(d));
    return current + d * (1.0 - exp(-speed * dt / exponential_ev));
}

// Local exposure's bilateral grid over the view: cells of LOCAL_CELL x LOCAL_CELL pixels, and
// per cell LOCAL_BINS bins of scene EV100 spread evenly over the histogram's range. The
// meter sums into each bin the pixel count and the pixels' EV100 above the range's bottom,
// in 1/LOCAL_EV_STEPS EV, as two words; blurred, each bin holds float2(that EV sum in EV,
// the count).
public static const uint LOCAL_CELL = 32;
public static const uint LOCAL_BINS = 16;
public static const float LOCAL_EV_STEPS = 4096.0;

