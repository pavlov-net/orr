// Exposure units. Scene luminance L is in cd/m2; EV100 is the reflected-light meter value
// at ISO 100 with K = 12.5. Exposure s scales luminance to exposed values (1.0 = paper
// white); auto exposure aims the metered average at MID_GREY.
module exposure;

public static const float MID_GREY = 0.18;

public float ev100_from_luminance(float luminance)
{
    return log2(luminance) + 3.0;
}

public float luminance_from_ev100(float ev100)
{
    return exp2(ev100 - 3.0);
}

// The exposure that shows a scene metered at `ev100` with manual exposure.
public float exposure_from_ev100(float ev100)
{
    return 1.0 / (1.2 * exp2(ev100));
}

public float ev100_from_exposure(float exposure)
{
    return -log2(1.2 * exposure);
}
