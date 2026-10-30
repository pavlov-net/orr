// Per-view constants, exposure state, sky statistics and position reconstruction.
//
// View space has no translation: x right, y up, the camera looking down -z. Clip space is
// Vulkan's (y down) with reverse Z and an infinite far plane: depth 1 at the near plane, 0
// at infinity.
module view;

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
    public float4x4 prev_clip_from_view;
    public float3x4 view_from_world_rotation;  // translation column 0
    public float3x4 prev_view_from_world_rotation;
    public float4 eye_render;                 // camera in render space, w 0
    public float4 prev_eye_render;            // last frame's camera, current render origin
    public int4 eye_cell;                     // w = cell_log2
    public float4 eye_offset;
    public float2 jitter;                     // render pixels
    public uint2 extent;                      // render pixels
    public ExposureState* exposure;           // two slots
    public uint exposure_slot;                // this render's slot; the other is the next render's
    public uint frame_index;
    public float delta_time;
    public uint _pad;
}

public static const float HDR_CEILING = 16384.0;

// Scales `c` so its largest channel is at most HDR_CEILING; NaN and infinite colours give 0.
public float3 apply_hdr_ceiling(float3 c)
{
    if (any(isnan(c)) || any(isinf(c)))
        return float3(0);
    float m = max(c.x, max(c.y, c.z));
    return m > HDR_CEILING ? c * (HDR_CEILING / m) : c;
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
