// Layered test scenes for depth of field, point-sampled at pixel centers through a pinhole
// camera looking down -z: cards facing the camera at fixed distances and, in the water
// scene, a horizontal water plane. `frame` writes the pinhole radiance (translucent layers
// composited) and the reverse-Z depth of the nearest opaque surface; `reference` averages
// `lens_samples` rays from points of the aperture through each pixel's point on the focal
// plane.
#include "orr/root.h"
import orr.abi;

static const uint FOREGROUND = 0;
static const uint BACKGROUND = 1;
static const uint HIGHLIGHTS = 2;
static const uint WATER = 3;

static const float PI = 3.14159265;

struct Root
{
    float* reference;
    StorageImageIndex radiance;
    StorageImageIndex depth;
    StorageImageIndex motion;
    uint width;
    uint height;
    uint scene;
    uint lens_samples;
    // tan of half the field of view.
    float tan_x;
    float tan_y;
    float near;
    // Meters.
    float focus;
    // Aperture radius on the lens plane, m.
    float lens_radius;
    uint blades;
    // Diaphragm curvature: 1 round, 0 a straight-edged polygon.
    float roundness;
}
ORR_ROOT(Root);

struct Layer
{
    float3 color;
    float alpha;
}

Layer layer(float3 c, float a)
{
    Layer l;
    l.color = c;
    l.alpha = a;
    return l;
}

float3 checker(float2 x, float size, float3 a, float3 b)
{
    int2 c = int2(floor(x / size));
    return ((c.x + c.y) & 1) != 0 ? a : b;
}

// Surfaces per scene: cards at `distance(i)` meters, and the water plane.
uint surfaces()
{
    return root.scene == WATER ? 3 : 2;
}

bool is_plane(uint i)
{
    return root.scene == WATER && i == 1;
}

float distance(uint i)
{
    switch (root.scene)
    {
    case FOREGROUND:
        return i == 0 ? 0.35 : 1.0;
    case BACKGROUND:
        return i == 0 ? 1.0 : 6.0;
    case HIGHLIGHTS:
        return i == 0 ? 5.0 : 40.0;
    default:
        return i == 0 ? 0.5 : 8.0;
    }
}

static const float WATER_LEVEL = -0.25;

// Surface `i` at point `x` (view space), transparent where it has a hole.
Layer surface(uint i, float3 x)
{
    float2 p = x.xy;
    switch (root.scene)
    {
    case FOREGROUND:
        if (i == 0)
        {
            bool bar = abs(p.x + 0.04) < 0.022;
            bool disc = length(p - float2(0.07, 0.035)) < 0.03;
            return layer(bar ? float3(0.85, 0.35, 0.1) : float3(0.2, 0.55, 0.9), bar || disc ? 1.0 : 0.0);
        }
        return layer(checker(p, 0.04, float3(0.8, 0.78, 0.7), float3(0.15, 0.2, 0.3))
                         + 0.15 * float3(frac(p.x * 7.0), 0.0, frac(p.y * 5.0)),
                     1.0);
    case BACKGROUND:
        if (i == 0)
        {
            float r = length(p);
            float a = atan2(p.y, p.x);
            bool star = r < 0.11 * (0.65 + 0.35 * cos(5.0 * a));
            return layer(checker(p, 0.02, float3(0.9, 0.8, 0.2), float3(0.6, 0.15, 0.1)), star ? 1.0 : 0.0);
        }
        return layer(checker(p, 0.3, float3(0.7, 0.75, 0.8), float3(0.1, 0.25, 0.12))
                         * (0.6 + 0.4 * sin(p.y * 9.0)),
                     1.0);
    case HIGHLIGHTS:
    {
        // Patterns in view angle, so both layers are sized on screen.
        float2 a = p / distance(i);
        if (i == 0)
        {
            // Leaves: a lattice of discs, open between them.
            float2 cell = frac(a / 0.1) - 0.5;
            float2 cell2 = frac(a / 0.1 + float2(0.5, 0.31)) - 0.5;
            bool leaf = length(cell) < 0.3 || length(cell2 * float2(1.3, 0.8)) < 0.22;
            return layer(float3(0.05, 0.18, 0.04) * (1.0 + 0.5 * cell.x), leaf ? 1.0 : 0.0);
        }
        float2 cell = frac(a / 0.07 + 0.37 * floor(a.yx / 0.07)) - 0.5;
        float bright = length(cell) < 0.08 ? 40.0 : 0.0;
        return layer(float3(0.02, 0.025, 0.04) + bright * float3(1.0, 0.85, 0.6), 1.0);
    }
    default:
        if (i == 0)
        {
            // Particles: soft translucent discs.
            float2 cell = frac(p / 0.06) - 0.5;
            float a = 0.5 * saturate(1.0 - length(cell) / 0.3);
            return layer(float3(0.9, 0.85, 0.7), a);
        }
        if (i == 1)
            return layer(float3(0.04, 0.12, 0.16) + 0.04 * sin(x.z * 20.0), 0.7);
        return layer(checker(p, 0.4, float3(0.75, 0.6, 0.45), float3(0.2, 0.3, 0.2)), 1.0);
    }
}

// Ray parameter where the ray from `o` along `d` meets surface `i`, or -1; `d.z` is -1 at
// the focal distance.
float hit(uint i, float3 o, float3 d)
{
    if (is_plane(i))
    {
        if (d.y >= 0.0)
            return -1.0;
        float t = (WATER_LEVEL - o.y) / d.y;
        return -(o.z + t * d.z) > 0.0 ? t : -1.0;
    }
    return (distance(i) + o.z) / -d.z;
}

float3 sky(float3 d)
{
    float up = saturate(normalize(d).y * 0.5 + 0.5);
    return lerp(float3(0.5, 0.55, 0.6), float3(0.2, 0.35, 0.7), up);
}

struct Trace
{
    float3 color;
    // View distance of the nearest opaque surface; 0 for sky.
    float depth;
}

// Surfaces composited front to back along the ray.
Trace trace(float3 o, float3 d)
{
    Trace r;
    r.color = 0.0;
    r.depth = 0.0;
    float transmit = 1.0;
    float last = 0.0;
    for (uint step = 0; step < surfaces(); step++)
    {
        float best = 1e30;
        uint which = 0;
        for (uint i = 0; i < surfaces(); i++)
        {
            float t = hit(i, o, d);
            if (t > last && t < best)
            {
                best = t;
                which = i;
            }
        }
        if (best >= 1e30)
            break;
        last = best;
        float3 x = o + best * d;
        Layer l = surface(which, x);
        r.color += transmit * l.alpha * l.color;
        transmit *= 1.0 - l.alpha;
        if (l.alpha >= 1.0)
        {
            r.depth = -x.z;
            return r;
        }
    }
    r.color += transmit * sky(d);
    return r;
}

// The focal-plane point of pixel center `id`.
float3 focal_point(uint2 id)
{
    float2 uv = (float2(id) + 0.5) / float2(root.width, root.height);
    return root.focus * float3(root.tan_x * (2.0 * uv.x - 1.0), -root.tan_y * (2.0 * uv.y - 1.0), -1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void frame(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    Trace t = trace(0.0, focal_point(id.xy) / root.focus);
    root.radiance.get<RWTexture2D<float4>>()[id.xy] = float4(t.color, 1.0);
    root.depth.get<RWTexture2D<float>>()[id.xy] = t.depth > 0.0 ? root.near / t.depth : 0.0;
    root.motion.get<RWTexture2D<float2>>()[id.xy] = 0.0;
}

float aperture(float angle)
{
    if (root.blades < 3)
        return 1.0;
    float sector = 2.0 * PI / float(root.blades);
    float a = frac((angle - 0.5 * PI) / sector) - 0.5;
    return lerp(cos(0.5 * sector) / cos(a * sector), 1.0, root.roundness);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reference(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float3 f = focal_point(id.xy);
    float3 sum = 0.0;
    uint taken = 0;
    for (uint i = 0; taken < root.lens_samples && i < 8 * root.lens_samples; i++)
    {
        // R2 points over the disc, kept inside the diaphragm.
        float2 u = frac(0.5 + float(i) * float2(0.7548776662, 0.5698402910));
        float r = sqrt(u.x);
        float angle = 2.0 * PI * u.y;
        if (r > aperture(angle))
            continue;
        float3 o = float3(root.lens_radius * r * float2(cos(angle), sin(angle)), 0.0);
        sum += trace(o, (f - o) / root.focus).color;
        taken++;
    }
    float3 c = sum / float(taken);
    float* out = root.reference + 4 * (id.y * root.width + id.x);
    out[0] = c.x;
    out[1] = c.y;
    out[2] = c.z;
    out[3] = 1.0;
}
