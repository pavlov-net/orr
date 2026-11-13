// Ray tracing pipeline shaders for the suite. `closest` and `traced` trace orthographic +z rays
// over the unit square, as `cluster_trace` does, and write the same 20-word ray record from the
// closest-hit, any-hit and miss shaders; beside it SEEN words of what the shaders saw:
//   0 the record data the closest-hit or miss shader read (custom index and geometry index
//     it was written for, or the miss value), 1 which shader ran, 2 the ray type its record
//     was written for, 3 a callable's result, 4 any-hit records written for another hit,
//   5 the hit object's geometry index, 6 the record index it names, 7 the custom index its
//   record data names, read through the record's address before the invoke. With
//   MODE_REDIRECT `traced` then runs the hit's record of the other ray type.
// `visibility` checks that stores reach the shaders after a reorder, an invoke and a call.
// `standin` runs the closest-hit's body as a compute kernel over the inputs it dumped.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.pipeline;

static const uint REC = 20;
static const uint SEEN = 8;
static const uint NONE = 0xffffffff;
static const uint MODE_FORCE_NON_OPAQUE = 1;
static const uint MODE_ALPHA = 2;
static const uint MODE_BOX = 4;
// `traced` runs a hit's record of the other ray type.
static const uint MODE_REDIRECT = 8;

static const uint SHADER_HIT_A = 1;
static const uint SHADER_HIT_B = 2;
static const uint SHADER_MISS = 3;
static const uint SHADER_BOX = 4;

struct Root
{
    uint* out;
    uint* seen;
    Tlas scene;
    Dumped* dumped;
    uint width;
    uint height;
    uint mode;
    uint ray_type;
    uint ray_types;
    uint callable;
    uint pad0;
    uint pad1;
}
ORR_ROOT_INLINE(Root);

// Changes the closest-hit shaders' code, so a pipeline made with a new value compiles them anew.
[vk::constant_id(0)] const uint SALT = 0;

// A hit record's data: the instance and geometry it was written for, and in `tag` its ray type
// in the high word.
struct HitData
{
    uint custom;
    uint geometry;
    uint64_t tag;
}

struct MissData
{
    uint value;
    uint pad;
}

struct CallData
{
    uint add;
    uint scale;
}

// What the closest-hit saw, for its stand-in.
struct Dumped
{
    HitInputs hit;
    HitData record;
    uint shader;
    uint pad0;
    uint pad1;
    uint pad2;
}

struct BoxAttributes
{
    float entry;
}

uint index(uint3 launch)
{
    return launch.y * root.width + launch.x;
}

uint* ray_record(uint3 launch)
{
    return root.out + index(launch) * REC;
}

uint* seen(uint3 launch)
{
    return root.seen + index(launch) * SEEN;
}

// The closest-hit's body: the ray record of the hit, and in the payload what the record held.
void closest_body<H : IHit>(H h, HitData rec, uint shader, inout Payload p, uint* o)
{
    o[0] = 1;
    o[1] = h.custom_index();
    o[2] = h.instance_index();
    o[3] = uint(h.cluster_id());
    o[4] = h.primitive_index();
    o[5] = h.geometry_index();
    o[6] = asuint(h.t());
    for (uint v = 0; v < 3; v++)
    {
        float3 c = h.corner(v);
        o[10 + v * 3] = asuint(c.x);
        o[11 + v * 3] = asuint(c.y);
        o[12 + v * 3] = asuint(c.z);
    }
    p.words.x = (rec.custom << 16) | rec.geometry;
    p.words.y = shader;
    p.words.z = uint(rec.tag >> 32);
    p.words.w = SALT;
}

void closest_hit(HitData rec, uint shader, inout Payload p, TriangleAttributes a)
{
    TriangleHit h = { a };
    uint3 launch = h.launch();
    closest_body(h, rec, shader, p, ray_record(launch));
    if (root.dumped != nullptr)
    {
        Dumped* d = root.dumped + index(launch);
        d.hit.launch = launch;
        d.hit.instance_index = h.instance_index();
        d.hit.custom_index = h.custom_index();
        d.hit.geometry_index = h.geometry_index();
        d.hit.primitive_index = h.primitive_index();
        d.hit.cluster_id = h.cluster_id();
        d.hit.barycentrics = h.barycentrics();
        d.hit.t = h.t();
        d.hit.front_face = uint(h.front_face());
        d.hit.world_origin = h.world_origin();
        d.hit.world_direction = h.world_direction();
        d.hit.object_origin = h.object_origin();
        d.hit.object_direction = h.object_direction();
        for (uint v = 0; v < 3; v++)
            d.hit.corners[v] = h.corner(v);
        d.record = rec;
        d.shader = shader;
    }
}

[shader("closesthit")]
void hit_a(uniform HitData rec, inout Payload p, TriangleAttributes a)
{
    closest_hit(rec, SHADER_HIT_A, p, a);
}

[shader("closesthit")]
void hit_b(uniform HitData rec, inout Payload p, TriangleAttributes a)
{
    closest_hit(rec, SHADER_HIT_B, p, a);
}

// Counts candidates and keeps the nearest, as `cluster_trace` does; with MODE_ALPHA ignores
// those where `(cluster + primitive) % 3 == 0`. Counts records written for another hit.
[shader("anyhit")]
void cover(uniform HitData rec, inout Payload p, TriangleAttributes a)
{
    uint3 launch = DispatchRaysIndex();
    uint* o = ray_record(launch);
    o[7] += 1;
    float t = RayTCurrent();
    int cluster = GetClusterID();
    if (t < asfloat(o[19]))
    {
        o[19] = asuint(t);
        o[8] = uint(cluster);
        o[9] = PrimitiveIndex();
    }
    if (rec.custom != InstanceID() || rec.geometry != GeometryIndex() || uint(rec.tag >> 32) != root.ray_type)
        seen(launch)[4] += 1;
    if ((root.mode & MODE_ALPHA) != 0 && (uint(cluster) + PrimitiveIndex()) % 3 == 0)
        IgnoreHit();
}

[shader("miss")]
void miss(uniform MissData rec, inout Payload p)
{
    uint* o = ray_record(DispatchRaysIndex());
    o[0] = 0;
    for (uint i = 1; i < 7; i++)
        o[i] = NONE;
    p.words.x = rec.value;
    p.words.y = SHADER_MISS;
    p.words.z = NONE;
}

[shader("callable")]
void scale(uniform CallData rec, inout Payload p)
{
    p.words.y = p.words.x * rec.scale + rec.add;
}

// A box AABB's entry distance along +z rays: the procedural geometry's one primitive spans
// z from `entry` (in the record) on.
[shader("intersection")]
void box(uniform MissData rec)
{
    BoxAttributes a;
    a.entry = asfloat(rec.value);
    ReportHit(a.entry - ObjectRayOrigin().z, 0, a);
}

[shader("closesthit")]
void box_hit(inout Payload p, BoxAttributes a)
{
    uint* o = ray_record(DispatchRaysIndex());
    o[0] = 1;
    o[1] = InstanceID();
    o[2] = InstanceIndex();
    o[3] = NONE;
    o[4] = PrimitiveIndex();
    o[5] = GeometryIndex();
    o[6] = asuint(RayTCurrent());
    p.words.x = asuint(a.entry);
    p.words.y = SHADER_BOX;
}

RayDesc ray_at(uint3 launch)
{
    RayDesc ray;
    ray.Origin = float3((float2(launch.xy) + 0.5) / float2(root.width, root.height), -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    return ray;
}

uint* begin(uint3 launch)
{
    uint* o = ray_record(launch);
    o[7] = 0;
    o[8] = NONE;
    o[9] = NONE;
    o[19] = asuint(1e30f);
    uint* s = seen(launch);
    for (uint i = 0; i < SEEN; i++)
        s[i] = 0;
    return o;
}

uint flags()
{
    return (root.mode & MODE_FORCE_NON_OPAQUE) != 0 ? RAY_FLAG_FORCE_NON_OPAQUE : RAY_FLAG_NONE;
}

void end(uint3 launch, Payload p, uint* o)
{
    uint* s = seen(launch);
    s[0] = p.words.x;
    s[1] = p.words.y;
    s[2] = p.words.z;
    if (root.callable != NONE)
    {
        Payload c;
        c.words = uint4(p.words.x, 0, 0, 0);
        call(root.callable, c);
        s[3] = c.words.y;
    }
    o[19] = 0x600d;
}

[shader("raygeneration")]
void closest()
{
    uint3 launch = DispatchRaysIndex();
    uint* o = begin(launch);
    Payload p;
    p.words = 0;
    trace(root.scene, flags(), 0xff, root.ray_type, root.ray_types, root.ray_type, ray_at(launch), p);
    end(launch, p, o);
}

[shader("raygeneration")]
void traced()
{
    uint3 launch = DispatchRaysIndex();
    uint* o = begin(launch);
    Payload p;
    p.words = 0;
    Traced t = trace_hit(root.scene, flags(), 0xff, root.ray_type, root.ray_types, root.ray_type, ray_at(launch), p);
    uint* s = seen(launch);
    s[5] = t.hit() ? t.geometry_index() : NONE;
    s[6] = t.hit() || t.miss() ? t.record_index() : NONE;
    s[7] = t.hit() && (root.mode & MODE_BOX) == 0 ? t.record<HitData>().custom : t.miss() ? t.record<MissData>().value : NONE;
    if (t.hit() && (root.mode & MODE_REDIRECT) != 0)
        t.set_record_index(t.record_index() + 1 - 2 * root.ray_type);
    t.reorder(launch.x * 7 + launch.y, 3);
    t.invoke(p);
    end(launch, p, o);
}

// Words of `visibility` per thread: 0 the raygen's before its reorder, 1 the hit or miss
// shader's, 2 the callable's, 3 the flags of what each read. With MODE_SHADER_CALL every
// access to them is coherent at shader call scope, with MODE_DEVICE at device scope; with
// MODE_READ_AFTER_CALL the raygen reads the callable's store right after the call, else after
// a reorder that follows it.
static const uint VIS = 4;
static const uint MODE_SHADER_CALL = 16;
static const uint MODE_READ_AFTER_CALL = 32;
static const uint MODE_DEVICE = 64;
static const uint VIS_REORDERED = 1;
static const uint VIS_INVOKED = 2;
static const uint VIS_RETURNED = 4;
static const uint VIS_CALLED = 8;
static const uint VIS_CALL_RETURNED = 16;

uint vis_value(uint i, uint word)
{
    return i * 0x9e3779b1u + word * 0x85ebca6bu + 1;
}

uint* vis(uint3 launch)
{
    return root.seen + index(launch) * VIS;
}

uint load_shader_call(uint* p)
{
    return spirv_asm {
        result : $$uint = OpLoad $p Aligned|MakePointerVisible|NonPrivatePointer !4 $(6u);
    };
}

void store_shader_call(uint* p, uint v)
{
    spirv_asm {
        OpStore $p $v Aligned|MakePointerAvailable|NonPrivatePointer !4 $(6u);
    };
}

uint vis_load(uint* p)
{
    if ((root.mode & MODE_DEVICE) != 0)
        return load_coherent(p);
    return (root.mode & MODE_SHADER_CALL) != 0 ? load_shader_call(p) : *p;
}

void vis_store(uint* p, uint v)
{
    if ((root.mode & MODE_DEVICE) != 0)
        store_coherent(p, v);
    else if ((root.mode & MODE_SHADER_CALL) != 0)
        store_shader_call(p, v);
    else
        *p = v;
}

// The hit or miss shader of `visibility`: reads the raygen's store, stores its own.
void vis_shade(inout Payload p)
{
    uint3 launch = DispatchRaysIndex();
    uint i = index(launch);
    uint* w = vis(launch);
    p.words.y = vis_load(w) == vis_value(i, 0) ? VIS_INVOKED : 0;
    vis_store(w + 1, vis_value(i, 1));
}

[shader("closesthit")]
void vis_hit(inout Payload p, TriangleAttributes a)
{
    vis_shade(p);
}

[shader("miss")]
void vis_miss(inout Payload p)
{
    vis_shade(p);
}

[shader("callable")]
void vis_call(inout Payload p)
{
    uint3 launch = DispatchRaysIndex();
    uint i = index(launch);
    uint* w = vis(launch);
    p.words.z = vis_load(w + 1) == vis_value(i, 1) ? VIS_CALLED : 0;
    vis_store(w + 2, vis_value(i, 2));
}

// Every thread stores, traces, reorders, invokes, calls and reorders again; each shader reads
// what the one before it stored. Counts the threads with an atomic after the last reorder.
[shader("raygeneration")]
void visibility()
{
    uint3 launch = DispatchRaysIndex();
    uint i = index(launch);
    uint* w = vis(launch);
    vis_store(w, vis_value(i, 0));
    Payload p;
    p.words = 0;
    // `ray_type` names the miss record here.
    Traced t = trace_hit(root.scene, RAY_FLAG_NONE, 0xff, 0, 1, root.ray_type, ray_at(launch), p);
    uint hint = (i * 0x2545f491u) >> 29;
    t.reorder(hint, 3);
    t.invoke(p);
    uint flags = p.words.y;
    if (vis_load(w + 1) == vis_value(i, 1))
        flags |= VIS_RETURNED;
    call(root.callable, p);
    flags |= p.words.z;
    bool after_call = (root.mode & MODE_READ_AFTER_CALL) != 0;
    if (after_call && vis_load(w + 2) == vis_value(i, 2))
        flags |= VIS_CALL_RETURNED;
    reorder(hint ^ 5, 3);
    if (!after_call && vis_load(w + 2) == vis_value(i, 2))
        flags |= VIS_CALL_RETURNED;
    if (vis_load(w) == vis_value(i, 0))
        flags |= VIS_REORDERED;
    w[3] = flags;
    InterlockedAdd(*root.out, 1);
}

// The closest-hit's body as a compute kernel over the hits `closest_hit` dumped: writes the
// ray record and payload words where the closest-hit wrote them, in `out` and `seen`.
[shader("compute")]
[numthreads(64, 1, 1)]
void standin(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width * root.height)
        return;
    Dumped* d = root.dumped + id.x;
    if (d.shader == 0)
        return;
    StoredHit h = { &d.hit };
    Payload p;
    p.words = 0;
    closest_body(h, d.record, d.shader, p, root.out + id.x * REC);
    uint* s = root.seen + id.x * SEEN;
    s[0] = p.words.x;
    s[1] = p.words.y;
    s[2] = p.words.z;
}
