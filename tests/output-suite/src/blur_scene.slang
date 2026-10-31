// Moving test scenes for motion blur, point-sampled at pixel centers: a spinning wheel over a
// static checkerboard, and a camera pan past a near box with parallax, its textures fine or
// coarse. `frame` writes the radiance, reverse-Z depth and motion (previous minus current UV,
// one frame back) at `time`;
// `reference` writes the radiance averaged over `subframes` times spread across the open
// shutter, centered on time 0.
#include "orr/root.h"
import orr.abi;

static const uint WHEEL = 0;
static const uint PAN = 1;
// The pan with its textures seven times coarser: the box stripes repeat beyond two steps,
// so its motion is unambiguous, and they stay well above the pixel pitch through a lens.
static const uint COARSE_PAN = 2;

static const float FAR = 0.001;
static const float NEAR = 0.05;

struct Root
{
    float* reference;
    StorageImageIndex radiance;
    StorageImageIndex depth;
    StorageImageIndex motion;
    uint width;
    uint height;
    uint scene;
    uint subframes;
    float shutter;
    // Scales the motion written, not the scene.
    float motion_scale;
    // Frames, for `frame`.
    float time;
}
ORR_ROOT(Root);

static const float PI = 3.14159265;
// Radians per frame.
static const float WHEEL_SPEED = 0.45;
// Pixels per frame.
static const float BACKGROUND_SPEED = 8.0;
static const float BOX_SPEED = 36.0;

struct Sample
{
    float3 color;
    float depth;
    // Screen position one frame earlier.
    float2 previous;
}

float3 checker(float2 x)
{
    int2 c = int2(floor(x / 16.0));
    return ((c.x + c.y) & 1) != 0 ? float3(0.7, 0.68, 0.62) : float3(0.25, 0.27, 0.3);
}

Sample wheel(float2 x, float t)
{
    float2 c = 0.5 * float2(root.width, root.height);
    float r = 0.4 * float(root.height);
    float2 q = x - c;
    Sample s;
    if (length(q) >= r)
    {
        s.color = checker(x);
        s.depth = FAR;
        s.previous = x;
        return s;
    }
    float a = atan2(q.y, q.x) - WHEEL_SPEED * t;
    float spoke = frac(a * 8.0 / (2.0 * PI));
    s.color = spoke < 0.25 ? float3(0.95, 0.6, 0.15) : float3(0.08, 0.12, 0.3);
    if (length(q) < 0.15 * r)
        s.color = float3(0.5);
    s.depth = NEAR;
    float sn = sin(-WHEEL_SPEED);
    float cs = cos(-WHEEL_SPEED);
    s.previous = c + float2(cs * q.x - sn * q.y, sn * q.x + cs * q.y);
    return s;
}

Sample pan(float2 x, float t, float coarse)
{
    float w = float(root.width);
    float h = float(root.height);
    Sample s;
    float box = x.x - 0.5 * w + BOX_SPEED * t;
    if (abs(box) < 0.2 * w && abs(x.y - 0.5 * h) < 0.25 * h)
    {
        int stripe = int(floor(box / (6.0 * coarse)));
        s.color = (stripe & 1) != 0 ? float3(0.9, 0.2, 0.15) : float3(0.15, 0.8, 0.3);
        s.depth = NEAR;
        s.previous = x + float2(BOX_SPEED, 0.0);
        return s;
    }
    s.color = checker((x + float2(BACKGROUND_SPEED * t, 0.0)) / coarse);
    s.depth = FAR;
    s.previous = x + float2(BACKGROUND_SPEED, 0.0);
    return s;
}

Sample scene(float2 x, float t)
{
    if (root.scene == WHEEL)
        return wheel(x, t);
    return pan(x, t, root.scene == COARSE_PAN ? 7.0 : 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void frame(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float2 x = float2(id.xy) + 0.5;
    Sample s = scene(x, root.time);
    float2 extent = float2(root.width, root.height);
    root.radiance.get<RWTexture2D<float4>>()[id.xy] = float4(s.color, 1.0);
    root.depth.get<RWTexture2D<float>>()[id.xy] = s.depth;
    root.motion.get<RWTexture2D<float2>>()[id.xy] = root.motion_scale * (s.previous - x) / extent;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reference(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float2 x = float2(id.xy) + 0.5;
    float3 sum = 0.0;
    for (uint i = 0; i < root.subframes; i++)
    {
        float t = root.shutter * ((float(i) + 0.5) / float(root.subframes) - 0.5);
        sum += scene(x, t).color;
    }
    float3 c = sum / float(root.subframes);
    float* out = root.reference + 4 * (id.y * root.width + id.x);
    out[0] = c.x;
    out[1] = c.y;
    out[2] = c.z;
    out[3] = 1.0;
}
