// Pre-exposed linear Rec.2020 radiance to display-linear Rec.709: the lens's distortion and
// chromatic aberration as the gather's sample positions, the view's adapted exposure, its
// glare, its white balance, the grade, then GT7, or one of the debug views. With lens
// distortion on it also composes the lens's map into frame generation's distortion field.
#include "orr/root.h"
import orr.abi;
import orr.color;
import view;
import exposure;
import gt7;
import tent;

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
    // 1 gathers through the lens's distortion.
    public uint distortion;
    // Brown-Conrady radial terms on the radius in half-diagonals.
    public float k1;
    public float k2;
    // Red's radius is green's times 1 + this, blue's times 1 - this.
    public float chromatic_aberration;
    // Scales green's radius so every tap lands inside the source.
    public float distortion_scale;
    public uint _pad;
}

struct Root
{
    TonemapConstants* constants;
    ViewConstants* view;
    // Rows of adapt's white balance matrix.
    float4* white_balance;
    // The glare pass's weights: entry 0's xyz is the identity weight.
    float4* glare_weights;
    TextureIndex source;
    StorageImageIndex target;
    // The `display_linear` hook's source, written with the same result; NONE without one.
    // With lens distortion `copy_hook_source` writes it after `main`, since it is the image
    // `main` gathers from.
    StorageImageIndex hook_source;
    // The glare chain, whose mip 0 holds the weighted band sum; NONE without glare.
    TextureIndex glare;
    // Clamps to the edge.
    SamplerIndex linear;
    // Frame generation's distortion field at half the extent, rounded up, which `main`
    // writes; NONE unless lens distortion is on and the view has the field.
    StorageImageIndex field;
    // The field as earlier stages left it this frame; NONE when none wrote it.
    TextureIndex field_source;
    uint width;
    uint height;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

// Where the gather at output position `p` (pixels) takes green from: the radius from the
// center `c`, in half-diagonals `hd`, scaled by scale * (1 + k1 r^2 + k2 r^4).
float2 lens_forward(TonemapConstants* k, float2 p, float2 c, float hd)
{
    float2 d = p - c;
    float u = dot(d, d) / (hd * hd);
    return c + d * (k.distortion_scale * (1.0 + u * (k.k1 + u * k.k2)));
}

// The output position whose green tap lands on `x`: three Newton steps on the radius,
// started from the forward map's local inverse.
float2 lens_inverse(TonemapConstants* k, float2 x, float2 c, float hd)
{
    float2 d = x - c;
    float rho = length(d) / hd;
    if (rho == 0.0)
        return c;
    float m = k.distortion_scale;
    float r = rho / m;
    float u = r * r;
    r /= 1.0 + u * (k.k1 + u * k.k2);
    for (uint i = 0; i < 3; i++)
    {
        u = r * r;
        float h = m * r * (1.0 + u * (k.k1 + u * k.k2)) - rho;
        r -= h / (m * (1.0 + u * (3.0 * k.k1 + 5.0 * u * k.k2)));
    }
    return c + d * (r / rho);
}

// Composes the lens into field texel `t`: RG' = f + RG(uv + f) with f the forward offset,
// BA' = BA + g(uv + BA) with g the inverse offset, all in UV.
void write_field(TonemapConstants* k, uint2 t, float2 size)
{
    float2 c = 0.5 * size;
    float hd = 0.5 * length(size);
    float2 uv = (float2(t) + 0.5) / float2((uint2(size) + 1) / 2);
    float2 p = uv * size;
    float2 f = (lens_forward(k, p, c, hd) - p) / size;
    float4 at_f = 0.0;
    float4 here = 0.0;
    if (root.field_source.index != NO_IMAGE)
    {
        Texture2D<float4> previous = root.field_source.get<Texture2D<float4>>();
        at_f = previous.SampleLevel(root.linear.get(), uv + f, 0.0);
        here = previous.Load(int3(t, 0));
    }
    float2 x = (uv + here.ba) * size;
    float2 g = (lens_inverse(k, x, c, hd) - x) / size;
    root.field.get<RWTexture2D<float4>>()[t] = float4(f + at_f.rg, here.ba + g);
}

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
        float2 size = float2(root.width, root.height);
        float2 texel = 1.0 / size;
        float2 uv_g = (float2(id.xy) + 0.5) * texel;
        float2 uv_r = uv_g;
        float2 uv_b = uv_g;
        Texture2D<float4> source = root.source.get<Texture2D<float4>>();
        float3 radiance;
        if (k.distortion != 0)
        {
            float2 c = 0.5 * size;
            float2 g = lens_forward(k, float2(id.xy) + 0.5, c, 0.5 * length(size));
            uv_g = g * texel;
            uv_r = (c + (g - c) * (1.0 + k.chromatic_aberration)) * texel;
            uv_b = (c + (g - c) * (1.0 - k.chromatic_aberration)) * texel;
            SamplerState linear = root.linear.get();
            radiance = float3(source.SampleLevel(linear, uv_r, 0.0).r,
                              source.SampleLevel(linear, uv_g, 0.0).g,
                              source.SampleLevel(linear, uv_b, 0.0).b);
        }
        else
            radiance = source.Load(int3(id.xy, 0)).rgb;
        float3 exposed = radiance * scale;
        float3 glared = exposed;
        if (root.glare.index != NO_IMAGE)
        {
            Texture2D<float4> chain = root.glare.get<Texture2D<float4>>();
            SamplerState linear = root.linear.get();
            float3 bands = tent(chain, linear, uv_g, texel, 0.0);
            if (k.chromatic_aberration != 0.0)
                bands = float3(tent(chain, linear, uv_r, texel, 0.0).r, bands.g,
                               tent(chain, linear, uv_b, texel, 0.0).b);
            glared = root.glare_weights[0].xyz * exposed + bands;
        }
        float3 balanced = glared;
        if (k.white_balance != 0)
        {
            float4* m = root.white_balance;
            balanced = float3(dot(m[0].xyz, glared), dot(m[1].xyz, glared), dot(m[2].xyz, glared));
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
    if (k.distortion == 0 && root.hook_source.index != NO_IMAGE)
        root.hook_source.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
    if (root.field.index != NO_IMAGE && all((id.xy & 1) == 0))
        write_field(k, id.xy >> 1, float2(root.width, root.height));
}

// The `display_linear` hook's source as a copy of `main`'s target, for a lens-distorted view
// whose gather read the hook source's image.
[shader("compute")]
[numthreads(8, 8, 1)]
void copy_hook_source(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float4 c = root.target.get<RWTexture2D<float4>>()[id.xy];
    root.hook_source.get<RWTexture2D<float4>>()[id.xy] = c;
}
