// Per-view constants, exposure state, sky statistics and position reconstruction.
//
// View space has no translation: x right, y up, the camera looking down -z. Clip space is
// Vulkan's (y down) with reverse Z and an infinite far plane: depth 1 at the near plane, 0
// at infinity.
module view;

import orr_shaders.math;

public struct ExposureState
{
    public float pre_exposure;
    public float prev_pre_exposure;
    public float adapted_ev100;
    public float target_ev100;
    public float long_term_ev100;
    public float2 white_point_xy;
    public uint initialized;
}

public struct SkyStats
{
    public float dome_luminance;
    public float2 dome_xy;
    public float _pad;
    public float4 directional;   // rgb: sum of directional illuminance at the focus
}

public struct ViewConstants
{
    public float4x4 clip_from_view;
    public float4x4 clip_from_view_jittered;
    public float4x4 view_from_clip;           // inverse of clip_from_view_jittered
    public float4x4 prev_clip_from_view;      // last frame's projection at this frame's aspect
    public float3x4 view_from_world_rotation;  // translation column 0
    public float3x4 prev_view_from_world_rotation;
    public float4 eye_render;                 // camera in render space, w 0
    public float4 eye_motion;                 // this frame's camera minus last frame's, w 0
    public int4 eye_cell;                     // w = cell_log2
    public float4 eye_offset;
    public float2 jitter;                     // render pixels
    public uint2 extent;                      // render pixels
    public ExposureState* exposure;           // two slots
    public uint exposure_slot;                // this render's slot; the other is the next render's
    public uint frame_index;
    public float delta_time;
    public float mip_bias;                    // texture level-of-detail bias of the upscaler
    public uint texture_filter;               // TEXTURE_FILTER_*
    public uint _pad;
}

// `ViewConstants.texture_filter`.
public static const uint TEXTURE_FILTER_HARDWARE = 0;
public static const uint TEXTURE_FILTER_STOCHASTIC = 1;

public static const float HDR_CEILING = 16384.0;

// Scales `c` so its largest channel is at most HDR_CEILING; NaN and infinite colors give 0. A
// channel over it lands on it exactly: the ceiling is a half float's power of two, so a rounded
// product a step under it stores a whole half-float step lower.
public float3 apply_hdr_ceiling(float3 c)
{
    if (any(isnan(c)) || any(isinf(c)))
        return float3(0);
    float m = max_component(c);
    return m > HDR_CEILING ? select(c == m, float3(HDR_CEILING), c * (HDR_CEILING / m)) : c;
}

// How far a position may land from the center of the previous image, in NDC: far off screen
// and well inside the range of the RG16F motion targets.
static const float PREVIOUS_NDC_LIMIT = 1024.0;

// Previous uv minus current uv, from a position through `prev_clip_from_view` and through
// `clip_from_view`. A point at or behind the previous camera's plane was on no part of that
// image, so it lands far off screen on its own side, where a point just in front of the plane
// does, not mirrored through the center.
public float2 motion_between(float4 previous, float4 current)
{
    float2 ndc = previous.xy / max(previous.w, 1e-6);
    float squared = dot(ndc, ndc);
    if (squared > PREVIOUS_NDC_LIMIT * PREVIOUS_NDC_LIMIT)
        ndc *= PREVIOUS_NDC_LIMIT / sqrt(squared);
    return (ndc - current.xy / current.w) * 0.5;
}

// The view-space position at `uv` (0..1, y down, of the render extent) and reverse-Z
// `depth`, as the jittered projection placed it. Depth 0 (infinity) gives a direction at
// infinite distance; use `view_direction` there.
public float3 view_position(ViewConstants* v, float2 uv, float depth)
{
    float4 clip = float4(uv.x * 2 - 1, uv.y * 2 - 1, depth, 1);
    float4 p = mul(v.view_from_clip, clip);
    return p.xyz / p.w;
}

// The unit view-space direction through `uv`.
public float3 view_direction(ViewConstants* v, float2 uv)
{
    float4 p = mul(v.view_from_clip, float4(uv.x * 2 - 1, uv.y * 2 - 1, 1, 1));
    return normalize(p.xyz / p.w);
}

// A view-space point in render space.
public float3 render_from_view(ViewConstants* v, float3 p)
{
    float3x4 r = v.view_from_world_rotation;
    return mul(transpose(float3x3(r[0].xyz, r[1].xyz, r[2].xyz)), p) + v.eye_render.xyz;
}
