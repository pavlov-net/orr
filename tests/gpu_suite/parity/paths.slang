// A path tracer through alpha-tested foliage with one body and two entries: the raygen entry
// reorders by hit material after each traversal, the compute entry does not. Each pixel
// writes its radiance and a hash of every segment's hit. Built with `-fp-mode precise`.
#include "orr/root.h"
import orr.abi;
import orr.coherence;
import orr.math;
import orr.random;
import orr.sampling;
import orr.trace;

static const uint GROUND = 0;
static const uint BARK = 1;
static const uint LEAF = 2;
static const uint HEAVY = 3;

struct Root
{
    float4 cam_pos;   // xyz, tan of half the vertical field of view
    float4 cam_fwd;   // xyz, aspect
    float4 cam_right;
    float4 cam_up;
    float4 sun;       // direction xyz, intensity
    float4* out_rad;
    uint* out_hash;
    float3x4* transforms; // per instance, object to world
    Tlas scene;
    uint width;
    uint height;
    uint frame;
    uint bounces;
}
ORR_ROOT(Root);

// Leaf quads are two triangles; the leaf is an ellipse inside the quad.
struct LeafAlpha : IAnyHit
{
    bool accept(Candidate c)
    {
        return c.custom_index != LEAF || leaf_covers(c.primitive_index, c.barycentrics);
    }
}

bool leaf_covers(uint prim, float2 b)
{
    float2 c1 = (prim & 1) == 0 ? float2(1, 0) : float2(1, 1);
    float2 c2 = (prim & 1) == 0 ? float2(1, 1) : float2(0, 1);
    float2 uv = c1 * b.x + c2 * b.y - 0.5;
    return uv.x * uv.x * 4.0 + uv.y * uv.y * 9.0 < 1.0;
}

// The material class of a hit; one leaf instance in four is the costly kind.
uint material(Hit h)
{
    if (h.custom_index == LEAF && (pcg(h.instance_index * 2654435761u) & 3) == 0)
        return HEAVY;
    return h.custom_index;
}

float noise(float3 p)
{
    float3 fl = floor(p);
    float3 f = p - fl;
    int3 i = int3(fl);
    f = f * f * (3.0 - 2.0 * f);
    float c[8];
    for (uint k = 0; k < 8; ++k)
    {
        int3 o = i + int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        c[k] = float(pcg(uint(o.x) * 73856093u ^ uint(o.y) * 19349663u ^ uint(o.z) * 83492791u) & 0xffff) / 65535.0;
    }
    float x0 = lerp(lerp(c[0], c[1], f.x), lerp(c[2], c[3], f.x), f.y);
    float x1 = lerp(lerp(c[4], c[5], f.x), lerp(c[6], c[7], f.x), f.y);
    return lerp(x0, x1, f.z);
}

float fbm(float3 p, uint octaves)
{
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (uint o = 0; o < octaves; ++o)
    {
        sum += amp * noise(p);
        norm += amp;
        p = p * 2.03 + float3(1.7, 9.2, 4.1);
        p -= 4096.0 * floor(p * (1.0 / 4096.0));
        amp *= 0.55;
    }
    return sum / max(norm, 1e-6);
}

// Shading whose cost depends on the material.
float3 albedo(uint m, float3 p)
{
    switch (m)
    {
    case GROUND:
        return lerp(float3(0.22, 0.16, 0.09), float3(0.16, 0.28, 0.07), fbm(p * 0.35, 4));
    case BARK:
        return float3(0.30, 0.22, 0.15) * (0.45 + 0.55 * fbm(p * float3(6.0, 0.8, 6.0), 6));
    case LEAF:
        return float3(0.12, 0.30, 0.06) * (0.8 + 0.4 * fbm(p * 3.0, 2));
    default:
        float f = fbm(p * float3(5.0, 1.3, 5.0), 40);
        float g = fbm(p.zxy * 2.0, 8);
        return lerp(float3(0.35, 0.30, 0.26), float3(0.12, 0.20, 0.10), f * g * 1.6);
    }
}

float3 sky(float3 d)
{
    return lerp(float3(0.9, 0.95, 1.0), float3(0.35, 0.55, 1.0), saturate(d.y * 0.5 + 0.5)) * 1.2;
}

uint hash_mix(uint h, uint v)
{
    return (h ^ v) * 16777619u;
}

RayDesc ray(float3 o, float3 d)
{
    RayDesc r;
    r.Origin = o;
    r.Direction = d;
    r.TMin = 0.0;
    r.TMax = 1e4;
    return r;
}

void body<C : ICoherence>(uint2 px)
{
    uint pix = px.y * root.width + px.x;
    Sampler rng = Sampler(px, root.frame);
    float2 ndc = (float2(px) + rng.next2()) / float2(root.width, root.height) * 2.0 - 1.0;
    float th = root.cam_pos.w;
    float3 o = root.cam_pos.xyz;
    float3 d = normalize(root.cam_fwd.xyz + root.cam_right.xyz * (ndc.x * th * root.cam_fwd.w) - root.cam_up.xyz * (ndc.y * th));
    float3 thr = 1.0;
    float3 rad = 0.0;
    uint hash = 2166136261u;
    bool alive = true;

    // A uniform trip count keeps the coherence point in uniform control flow.
    for (uint b = 0; b <= root.bounces; ++b)
    {
        Hit h;
        h.hit = false;
        float3 corners[3] = { float3(0), float3(0), float3(0) };
        if (alive)
            h = trace_closest(root.scene, ray(o, d), 0xff, RAY_FLAG_NONE, LeafAlpha(), corners);
        uint m = h.hit ? material(h) : 0;
        C.point(h.hit ? 1 | (m << 1) : 0, 3);
        if (alive)
        {
            if (!h.hit)
            {
                rad += thr * sky(d);
                hash = hash_mix(hash, 0xfffffffeu);
                alive = false;
            }
            else
            {
                hash = hash_mix(hash_mix(hash, h.instance_index), h.primitive_index);
                float3x4 xf = root.transforms[h.instance_index];
                float3 n = normalize(transform_vector(xf, cross(corners[1] - corners[0], corners[2] - corners[0])));
                if (dot(n, d) > 0.0)
                    n = -n;
                float3 p = o + d * h.t;
                float3 a = albedo(m, p);
                float3 po = offset_ray_origin(p, n);
                float ndl = dot(n, root.sun.xyz);
                if (ndl > 0.0 && !trace_any(root.scene, ray(po, root.sun.xyz), 0xff, RAY_FLAG_NONE, LeafAlpha()))
                    rad += thr * a * (root.sun.w * ndl);
                thr *= a;
                o = po;
                float3 t, bt;
                orthonormal_basis(n, t, bt);
                d = normalize(from_local(sample_cosine_hemisphere(rng.next2()), t, bt, n));
            }
        }
    }
    root.out_rad[pix] = float4(rad, 1.0);
    root.out_hash[pix] = hash;
}

[shader("raygeneration")]
void raygen_main()
{
    body<Reorder>(DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    if (all(id.xy < uint2(root.width, root.height)))
        body<NoReorder>(id.xy);
}
