// Pre-exposed linear Rec.2020 radiance to display-linear Rec.709: the view's adapted
// exposure, its white balance, the grade, then GT7, or one of the debug views.
#include "orr/root.h"
import orr.abi;
import orr.color;
import view;
import exposure;
import gt7;

static const uint DEBUG_NONE = 0;
static const uint DEBUG_LINEAR = 1;
static const uint DEBUG_PATTERN = 2;
static const uint DEBUG_EXPOSURE_HEATMAP = 3;

static const uint NO_IMAGE = 0xffffffff;

public struct TonemapConstants
{
    public Gt7Curve curve;
    // Rows of the grade matrix on linear Rec.2020; w unused.
    public float4 grade_r;
    public float4 grade_g;
    public float4 grade_b;
    // 2^grade exposure.
    public float grade_scale;
    public uint sdr;
    public uint debug;
    // Calibration pattern levels, display-linear.
    public float pattern_patch;
    public float pattern_surround;
    // 1 applies the white balance matrix adapt wrote.
    public uint white_balance;
    public uint _pad1;
    public uint _pad2;
}

struct Root
{
    TonemapConstants* constants;
    ViewConstants* view;
    // Rows of adapt's white balance matrix.
    float4* white_balance;
    TextureIndex source;
    StorageImageIndex target;
    // The `display_linear` hook's source, written with the same result; NONE without one.
    StorageImageIndex hook_source;
    uint width;
    uint height;
    uint _pad;
}
ORR_ROOT(Root);

// Stops from MID_GREY as colour: blue below, green at mid grey, red above, saturating at
// six stops, with each whole stop a step darker at its top.
float3 heatmap(float exposed_luminance)
{
    float stops = log2(max(exposed_luminance, 1e-20) / MID_GREY);
    float t = clamp(stops / 6.0, -1.0, 1.0);
    float3 c = t < 0.0 ? lerp(float3(0.0, 1.0, 0.0), float3(0.0, 0.2, 1.0), -t)
                       : lerp(float3(0.0, 1.0, 0.0), float3(1.0, 0.1, 0.0), t);
    return c * (1.0 - 0.3 * frac(stops));
}

// The calibration pattern's patch: a centred square half the image height across.
bool in_patch(uint2 p)
{
    float2 d = abs(float2(p) + 0.5 - 0.5 * float2(root.width, root.height));
    return all(d < 0.25 * float(root.height));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    TonemapConstants* k = root.constants;
    float3 c;
    if (k.debug == DEBUG_PATTERN)
    {
        c = float3(in_patch(id.xy) ? k.pattern_patch : k.pattern_surround);
        if (k.sdr != 0)
            c = saturate(c);
    }
    else
    {
        ViewConstants* view = root.view;
        ExposureState* cur = view.exposure + view.exposure_slot;
        ExposureState* next = view.exposure + (view.exposure_slot ^ 1);
        float scale = exposure_from_ev100(next.adapted_ev100) / cur.pre_exposure;
        float3 radiance = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb;
        float3 exposed = radiance * scale;
        float3 balanced = exposed;
        if (k.white_balance != 0)
        {
            float4* m = root.white_balance;
            balanced = float3(dot(m[0].xyz, exposed), dot(m[1].xyz, exposed), dot(m[2].xyz, exposed));
        }
        float3 graded = k.grade_scale * float3(dot(k.grade_r.xyz, balanced),
                                               dot(k.grade_g.xyz, balanced),
                                               dot(k.grade_b.xyz, balanced));
        if (k.debug == DEBUG_EXPOSURE_HEATMAP)
            c = heatmap(luminance_rec2020(exposed));
        else if (k.debug == DEBUG_LINEAR)
        {
            c = mul(REC2020_TO_REC709, graded);
            if (k.sdr != 0)
                c = saturate(c);
        }
        else
            c = gt7_display_linear(graded, k.curve, k.sdr != 0);
    }
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
    if (root.hook_source.index != NO_IMAGE)
        root.hook_source.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
}
