// The lens both glare methods model: its F-number from the adapted exposure, the exposure
// tonemap applies to the radiance they read, and the part of it they spread.
module glare_lens;
import view;
import exposure;

public static const float GLARE_F_MIN = 4.0;
public static const float GLARE_F_MAX = 22.0;
// The largest exposed value glare reads; NaN and negatives read as 0.
public static const float GLARE_STORAGE_MAX = 65504.0;
// Glare spreads only what lies past this exposed value (1.0 is paper white), about where SDR's
// curve reaches its peak, on SDR and HDR alike. The knee is quadratic over KNEE_WIDTH to each
// side.
static const float KNEE = 1.5;
static const float KNEE_WIDTH = 0.5;
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

// The part of `radiance`, exposed by `scale`, that glare spreads: its color, scaled so its
// largest channel keeps what lies past the knee. Tonemap takes the part the bands spread back
// out of the pixel, which keeps the energy whatever the knee does.
public float3 glare_extract(float3 radiance, float scale)
{
    float3 exposed = glare_exposed(radiance, scale);
    float m = max(exposed.r, max(exposed.g, exposed.b));
    if (m <= KNEE - KNEE_WIDTH)
        return 0.0;
    float over = m - KNEE;
    if (over < KNEE_WIDTH)
    {
        float t = over + KNEE_WIDTH;
        over = t * t / (4.0 * KNEE_WIDTH);
    }
    return exposed * (over / m);
}

// `glare_extract` on `source`'s pixels, filtered bilinearly at `uv`. Extraction is not linear, so
// it runs per pixel before the filter: on filtered colors a pixel's neighbors would change what
// it gives up and the energy would drop.
public float3 glare_extract_at(Texture2D<float4> source, float2 uv, float scale)
{
    uint width, height;
    source.GetDimensions(width, height);
    int2 last = int2(width, height) - 1;
    float2 t = uv * float2(width, height) - 0.5;
    float2 lo = floor(t);
    float2 f = t - lo;
    int2 a = int2(lo);
    float3 c[4];
    for (int i = 0; i < 4; i++)
    {
        int2 at = clamp(a + int2(i & 1, i >> 1), 0, last);
        c[i] = glare_extract(source.Load(int3(at, 0)).rgb, scale);
    }
    return lerp(lerp(c[0], c[1], f.x), lerp(c[2], c[3], f.x), f.y);
}

// The exposure tonemap applies: the adapted exposure over the pre-exposure.
public float glare_exposure_scale(ViewConstants* view)
{
    ExposureState* cur = view.exposure + view.exposure_slot;
    ExposureState* next = view.exposure + (view.exposure_slot ^ 1);
    return exposure_from_ev100(next.adapted_ev100) / cur.pre_exposure;
}

// `radiance` exposed by `scale`, clamped to [0, GLARE_STORAGE_MAX].
float3 glare_exposed(float3 radiance, float scale)
{
    float3 c = radiance * scale;
    return select(c > 0.0, min(c, GLARE_STORAGE_MAX), 0.0);
}
