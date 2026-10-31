// The lens both glare methods model: its F-number from the adapted exposure, and the
// exposure tonemap applies to the radiance they read.
module glare_lens;
import view;
import exposure;

public static const float GLARE_F_MIN = 1.4;
public static const float GLARE_F_MAX = 22.0;
// The largest exposed value glare reads; NaN and negatives read as 0.
public static const float GLARE_STORAGE_MAX = 65504.0;
// The F-number of a lens metering the scene at a 1/SHUTTER_RATE s shutter:
// N^2 = 2^EV100 / SHUTTER_RATE.
static const float SHUTTER_RATE = 100.0;

// The F-number at the reference pixel pitch for the view's adapted exposure: brighter
// scenes stop down, which widens the PSF. `hdr_shift` is the stops the HDR rendition moves
// it down by, `pitch_ratio` the reference pitch over the view's.
public float glare_f_number(ViewConstants* view, float hdr_shift, float pitch_ratio)
{
    ExposureState* next = view.exposure + (view.exposure_slot ^ 1);
    float n = sqrt(exp2(next.adapted_ev100 - hdr_shift) / SHUTTER_RATE) * pitch_ratio;
    return clamp(n, GLARE_F_MIN, GLARE_F_MAX);
}

// The exposure tonemap applies: the adapted exposure over the pre-exposure.
public float glare_exposure_scale(ViewConstants* view)
{
    ExposureState* cur = view.exposure + view.exposure_slot;
    ExposureState* next = view.exposure + (view.exposure_slot ^ 1);
    return exposure_from_ev100(next.adapted_ev100) / cur.pre_exposure;
}

// `radiance` exposed by `scale`, clamped to [0, GLARE_STORAGE_MAX].
public float3 glare_exposed(float3 radiance, float scale)
{
    float3 c = radiance * scale;
    return select(c > 0.0, min(c, GLARE_STORAGE_MAX), 0.0);
}
