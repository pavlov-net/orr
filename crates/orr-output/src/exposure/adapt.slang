// Turns the meter's histogram and the scene's sky statistics into this render's exposure and
// white point: reads the view's current exposure slot, writes the next one, the white
// balance matrix and the telemetry, and clears the histogram for the next frame.
#include "orr/root.h"
import orr.abi;
import orr.color;
import view;
import exposure;

static const uint MODE_MANUAL = 0;
static const uint MODE_AUTO = 1;

// ExposureState::initialized: 0 before the first metered frame, 1 after a snap, 2 adapting.
// After a snap the next metered frame, lit with the snapped pre-exposure, snaps again; the
// view adapts once two snaps agree, so a snap metered through clipping or RGBA16F underflow
// is corrected.
static const float SNAPS_AGREE_EV = 0.5;
// A frame with nothing to meter may hide light that underflowed RGBA16F. Pre-exposure does
// not show on a black frame, so it rises by this many stops a frame to find it.
static const float SEARCH_STOPS = 8.0;
static const uint STATE_UNSET = 0;
static const uint STATE_SNAPPED = 1;
static const uint STATE_ADAPTING = 2;
// The white point's range: correlated colour temperature and distance from the locus.
static const float CCT_MIN = 2500.0;
static const float CCT_MAX = 7000.0;
static const float DUV_LIMIT = 0.05;

public struct AdaptParams
{
    public uint mode;
    public float manual_ev100;
    public uint cut;
    public uint physiological;
    // Histogram range: EV100 at the bottom of bin 0 and EV per bin.
    public float ev_min;
    public float bin_ev;
    // Percentiles of the metered weight the mean is taken over.
    public float filter_low;
    public float filter_high;
    // Exposure EV100 window.
    public float clamp_min;
    public float clamp_max;
    public float speed_brighten;
    public float speed_darken;
    public float exponential_ev;
    // Reference weights: histogram 1 and the others 0 meter the histogram alone.
    public float weight_histogram;
    public float weight_dome;
    public float weight_directional;
    // The long-term stage, with `physiological`.
    public float long_speed_brighten;
    public float long_speed_darken;
    public float bound_brighten;
    public float bound_darken;
    // Auto white balance, with `white_balance`: its speed (per second), the D65 light added
    // to the mix (cd/m2) and the reference weights.
    public uint white_balance;
    public float white_speed;
    public float virtual_light_nits;
    public float white_histogram;
    public float white_dome;
    public float white_directional;
    public uint _pad0;
    public uint _pad1;
}

// What adapt read and wrote for one render, read back by the CPU.
public struct ExposureTelemetryGpu
{
    public float adapted_ev100;
    public float target_ev100;
    public float long_term_ev100;
    // The histogram mean, scene EV100; valid when `metered` is 1.
    public float metered_ev100;
    public float2 white_point_xy;
    public float2 white_target_xy;
    // The next render's pre-exposure.
    public float pre_exposure;
    public uint metered;
    public SkyStats sky;
}

struct Root
{
    AdaptParams* params;
    // COMPENSATION_ENTRIES stops spread over the histogram's range.
    float* compensation;
    uint* histogram;
    ViewConstants* view;
    // Null without a scene.
    SkyStats* sky;
    ExposureTelemetryGpu* telemetry;
    // Rows of the white balance matrix on linear Rec.2020, which tonemap applies.
    float4* white_balance;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

static const float PI = 3.14159265358979;

// The 64-bit sum whose low word is at `low`.
float wide(uint* low)
{
    return float(*(low + 1)) * 4294967296.0 + float(*low);
}

float compensation(AdaptParams* p, float ev100)
{
    float range = p.bin_ev * float(HISTOGRAM_BINS);
    float t = saturate((ev100 - p.ev_min) / range) * float(COMPENSATION_ENTRIES - 1);
    uint i = min(uint(t), COMPENSATION_ENTRIES - 2);
    float* lut = root.compensation + i;
    return lerp(*lut, *(lut + 1), t - float(i));
}

// The exposure EV100 that keys a scene metered at `metered_ev100`, inside the window.
float keyed(AdaptParams* p, float metered_ev100)
{
    float ev = keyed_ev100(metered_ev100, compensation(p, metered_ev100));
    return clamp(ev, p.clamp_min, p.clamp_max);
}

// The histogram between the filter percentiles.
struct Metered
{
    // The mean EV100.
    float ev100;
    // The mean luminance, and the luminance-weighted mean chromaticity when the meter summed
    // it.
    float luminance;
    float2 xy;
}

// False when the histogram is empty.
bool histogram_mean(AdaptParams* p, out Metered m)
{
    m.ev100 = 0.0;
    m.luminance = 0.0;
    m.xy = D65_XY;
    float total = 0.0;
    for (uint b = 0; b < HISTOGRAM_BINS; b++)
        total += float(*(root.histogram + HISTOGRAM_WORDS * b));
    if (total == 0.0)
        return false;
    float low = p.filter_low * total;
    float high = p.filter_high * total;
    float below = 0.0;
    float sum = 0.0;
    float weight = 0.0;
    float3 yxy = 0.0;
    for (uint b = 0; b < HISTOGRAM_BINS; b++)
    {
        uint* bin = root.histogram + HISTOGRAM_WORDS * b;
        float count = float(*bin);
        if (count == 0.0)
            continue;
        float w = min(below + count, high) - max(below, low);
        below += count;
        if (w <= 0.0)
            continue;
        float ev = p.ev_min + (float(b) + (wide(bin + 1) / count + 0.5) / BIN_OFFSET_STEPS) * p.bin_ev;
        sum += w * ev;
        weight += w;
        float2 xy = float2(wide(bin + 3), wide(bin + 5)) / (count * CHROMA_STEPS);
        yxy += w * luminance_from_ev100(ev) * float3(xy, 1.0);
    }
    if (weight <= 0.0)
        return false;
    m.ev100 = sum / weight;
    m.luminance = yxy.z / weight;
    m.xy = yxy.xy / yxy.z;
    return true;
}

// The histogram mean mixed with the sky references, as EV100.
float reference_mix(AdaptParams* p, float metered, SkyStats sky)
{
    float sum = p.weight_histogram * metered;
    float weight = p.weight_histogram;
    if (p.weight_dome > 0.0 && sky.dome_luminance > 0.0)
    {
        sum += p.weight_dome * ev100_from_luminance(sky.dome_luminance);
        weight += p.weight_dome;
    }
    float illuminance = luminance_rec2020(sky.directional.rgb);
    if (p.weight_directional > 0.0 && illuminance > 0.0)
    {
        sum += p.weight_directional * ev100_from_luminance(MID_GREY * illuminance / PI);
        weight += p.weight_directional;
    }
    return weight > 0.0 ? sum / weight : metered;
}

// `xy` moved onto the nearest point inside the white point's range, by CCT and Duv.
float2 white_in_range(float2 xy)
{
    float cct = cct_from_xy(xy);
    float duv = duv_from_xy(xy, cct);
    float c = clamp(cct, CCT_MIN, CCT_MAX);
    float d = clamp(duv, -DUV_LIMIT, DUV_LIMIT);
    return c == cct && d == duv ? xy : xy_from_cct_duv(c, d);
}

// The scene's white point: the histogram and the sky references averaged in Yxy, weighted by
// luminance times their weights, with the virtual D65 light at its own luminance.
float2 white_target(AdaptParams* p, Metered m, SkyStats sky)
{
    float3 sum = p.virtual_light_nits * float3(D65_XY, 1.0);
    sum += p.white_histogram * m.luminance * float3(m.xy, 1.0);
    if (sky.dome_luminance > 0.0)
        sum += p.white_dome * sky.dome_luminance * float3(sky.dome_xy, 1.0);
    float3 e = mul(REC2020_TO_XYZ, sky.directional.rgb);
    if (e.y > 0.0)
        sum += p.white_directional * MID_GREY * e.y / PI * float3(xy_from_xyz(e), 1.0);
    if (!(sum.z > 0.0))
        return D65_XY;
    return white_in_range(sum.xy / sum.z);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    AdaptParams* p = root.params;
    ViewConstants* view = root.view;
    ExposureState cur = *(view.exposure + view.exposure_slot);
    float dt = view.delta_time;
    SkyStats sky;
    sky.dome_luminance = 0.0;
    sky.dome_xy = D65_XY;
    sky._pad = 0.0;
    sky.directional = float4(0.0);
    if (root.sky != nullptr)
        sky = *root.sky;
    bool cut = p.cut != 0;

    bool white = p.white_balance != 0;
    Metered m = { 0.0, 0.0, D65_XY };
    bool histogram = false;
    if (p.mode == MODE_AUTO || white)
        histogram = histogram_mean(p, m);
    bool metered = p.mode == MODE_AUTO && histogram;
    float metered_ev100 = m.ev100;

    ExposureState next = cur;
    next.prev_pre_exposure = cur.pre_exposure;
    bool snap = cut || cur.initialized != STATE_ADAPTING;
    bool hold = false;
    if (p.mode == MODE_MANUAL)
    {
        next.adapted_ev100 = p.manual_ev100;
        next.target_ev100 = p.manual_ev100;
        next.long_term_ev100 = p.manual_ev100;
        snap = cut || cur.initialized == STATE_UNSET;
        next.initialized = STATE_ADAPTING;
    }
    else if (metered)
    {
        float long_target = keyed(p, reference_mix(p, metered_ev100, sky));
        float long_term = long_target;
        float target = long_target;
        if (p.physiological != 0)
        {
            if (!snap)
                long_term = adapt_ev100(cur.long_term_ev100, long_target, p.long_speed_brighten,
                                        p.long_speed_darken, p.exponential_ev, dt);
            target = clamp(keyed(p, metered_ev100), long_term - p.bound_darken,
                           long_term + p.bound_brighten);
        }
        float adapted = target;
        if (!snap)
            adapted = adapt_ev100(cur.adapted_ev100, target, p.speed_brighten, p.speed_darken,
                                  p.exponential_ev, dt);
        next.adapted_ev100 = clamp(adapted, p.clamp_min, p.clamp_max);
        next.target_ev100 = clamp(target, p.clamp_min, p.clamp_max);
        next.long_term_ev100 = long_term;
        bool agree = !cut && cur.initialized == STATE_SNAPPED
            && abs(next.adapted_ev100 - cur.adapted_ev100) < SNAPS_AGREE_EV;
        next.initialized = snap && !agree ? STATE_SNAPPED : STATE_ADAPTING;
    }
    else
    {
        // Nothing to meter: hold the exposure, snap at the next metered frame after a cut,
        // and search upward with the pre-exposure, no further than the exposure that keys
        // the bottom of the histogram.
        hold = true;
        if (cut)
            next.initialized = STATE_UNSET;
        float highest = exposure_from_ev100(keyed(p, p.ev_min));
        next.pre_exposure = max(cur.pre_exposure, min(cur.pre_exposure * exp2(SEARCH_STOPS), highest));
    }

    if (!hold)
    {
        float s = exposure_from_ev100(next.adapted_ev100);
        float stops = log2(s / cur.pre_exposure);
        if (snap || !(abs(stops) < 64.0))
            next.pre_exposure = s;
        else if (abs(stops) > 1.0)
            next.pre_exposure = cur.pre_exposure * exp2(round(stops));
    }

    // White point: (0, 0) is unset and snaps at the next frame with light.
    bool white_set = cur.white_point_xy.y > 0.0;
    float2 white_target_xy = white && white_set ? cur.white_point_xy : D65_XY;
    if (!white)
        next.white_point_xy = D65_XY;
    else if (histogram)
    {
        white_target_xy = white_target(p, m, sky);
        next.white_point_xy = snap || !white_set
            ? white_target_xy
            : xy_from_uv(lerp(uv_from_xy(cur.white_point_xy), uv_from_xy(white_target_xy),
                              1.0 - exp(-p.white_speed * dt)));
    }
    else if (cut)
        next.white_point_xy = float2(0.0);
    bool balanced = next.white_point_xy.y > 0.0;
    float3x3 balance = balanced ? cat16_rec2020(next.white_point_xy, D65_XY)
                                : float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    for (uint r = 0; r < 3; r++)
        *(root.white_balance + r) = float4(balance[r], 0.0);
    *(view.exposure + (view.exposure_slot ^ 1)) = next;

    ExposureTelemetryGpu t;
    t.adapted_ev100 = next.adapted_ev100;
    t.target_ev100 = next.target_ev100;
    t.long_term_ev100 = next.long_term_ev100;
    t.metered_ev100 = metered_ev100;
    t.white_point_xy = balanced ? next.white_point_xy : D65_XY;
    t.white_target_xy = white_target_xy;
    t.pre_exposure = next.pre_exposure;
    t.metered = metered ? 1 : 0;
    t.sky = sky;
    *root.telemetry = t;
    for (uint w = 0; w < HISTOGRAM_BINS * HISTOGRAM_WORDS; w++)
        *(root.histogram + w) = 0;
}
