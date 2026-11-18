// Plane stacks in the path tracer: a directional job's ray that ends at solid glass may reach its
// light through a stack past it, as `path_light` walks one. The raygen traces a probe along the
// light's direction up to past the glass, then the legs face to face along the bent directions,
// each a stack ray to its face and a shadow ray up to it; a dielectric's hit answers whether it is
// on a stack face (the stack question), and this callable takes each step between, from the
// path's stack words (`STACK_*_WORD`) and its data: the light's bend at the vertex, and per leg the
// medium along it, the crossing of the face it ends at and its Fresnel.
module stack;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.interiors;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.state;
import orr_lighting.path.vertex;

// What the stack's callable takes and gives: the step and its answer (`words`), the face the
// stack ray met as its hit answered the stack question (`face`), and the leg, from its origin
// along its direction: the one the step ends, and then the next.
public struct StackStep
{
    public uint4 words;
    public uint4 face;
    public uint4 leg;
}

// The step its data's first word asks for, in its low four bits: the bend at the vertex from
// the probe's hit, or the end of a leg at its stack ray's hit, `t` m along it in the second word.
public static const uint STEP_START = 0;
public static const uint STEP_LEG = 1;
// What the stack ray met, from bit 4: nothing, a dielectric whose hit answered the stack question,
// or anything else. From bit 8 whether the job's ray starts in an interior
// (STACK_JOB_LIGHT), and from bit 12 the faces the legs crossed so far.
public static const uint MET_NONE = 0;
public static const uint MET_FACE = 1;
public static const uint MET_OTHER = 2;
public static const uint STEP_INSIDE = 1 << 8;
public static const uint STEP_CROSSED_SHIFT = 12;
// What the step answers in the first word: no light comes through; the next leg, in the data's
// leg, inside a medium of the IOR in the second word, which the thin walls it crosses lie in; or
// the light reached the vertex, as the stack words' through holds what the legs' media and faces
// let through. With STEP_PROBED the probe's hit is the first leg's,
// along the light's own direction.
public static const uint STEP_DARK = 0;
public static const uint STEP_ON = 1;
public static const uint STEP_LIT = 2;
public static const uint STEP_PROBED = 4;

// How far a ray from `origin` along unit `dir` reaches toward a hit `t` m on: short of it by the
// hit's rounding.
public float short_of(float3 origin, float3 dir, float t)
{
    float3 at = origin + dir * t;
    float magnitude = max(abs(at.x), max(abs(at.y), abs(at.z)));
    return max(0.0, t - max(1e-4 * t, magnitude * F32_PRECISION));
}

// The stack callable's body for `slot`'s path, with its data `c`.
//
// The bend: the face the probe met gives the stack's normal. A probe that leaves the face's
// material from inside finds the vertex inside it, which its list need not hold. The light's
// direction at the vertex follows from the IOR there alone, and comes with the ratio of the faces'
// cosines outside and at the vertex. A vertex that took its light as unbent kept nothing to weigh
// it again, and its light through a bend is 0: one whose job's ray starts in no interior.
//
// A leg: its segment takes the innermost interior's medium from the point it starts at, and out of
// every interior the linked medium's, which a job whose ray starts in an interior leaves out. A
// leg that meets nothing reaches the light where it lies outside every interior. One that ends at
// a face parallel to the first, of fewer than STACK_CROSSINGS, crosses it into the next medium,
// whose IOR bends the light anew, by the face's Fresnel on its exterior side.
public void path_stack_step<M : IMedium>(PathRoot r, uint slot, inout StackStep c)
{
    uint4* at = path_stack(r, slot);
    uint4* hot = path_hot(r, slot);
    uint4 words = c.words;
    uint step = words.x & 0xf;
    uint met = (words.x >> 4) & 0xf;
    bool inside = (words.x & STEP_INSIDE) != 0;
    uint crossed = words.x >> STEP_CROSSED_SHIFT;
    float t = asfloat(words.y);
    c.words = uint4(STEP_DARK, 0, 0, 0);
    // The light's direction, the job's target.
    float3 light = asfloat(load_coherent(hot + 5).xyz);
    uint4 face = c.face;
    bool on_face = met == MET_FACE && face.z != MATERIAL_NONE;
    float3 normal = oct_decode_snorm16(face.x);
    float3 ng = oct_decode_snorm16(face.y);
    uint m = face.z;
    // A face's material reads no texture, so its medium reads no point.
    float4 point = NO_MEDIUM_POINT;
    if (step == STEP_START)
    {
        if (!on_face)
            return;
        Interior list = Interior();
        if (inside)
        {
            uint cold = load_coherent(at + STACK_FROM_WORD).w;
            list = load_interior(cold != 0 ? path_cold(r, slot) : at + STACK_LIST_WORD);
            if (cold == STACK_COLD_OUT)
                list.cross(load_coherent(path_cold(r, slot) + COLD_WALK_WORD).w, false);
        }
        if (dot(ng, light) > 0.0 && !list.holds(m))
            list.nest(m, list, point);
        float ior = interior_ior(r.scene, list.innermost());
        float3 d;
        if (!stack_direction(light, normal, ior, d))
            return;
        bool bends = any(d != light);
        if (bends && !inside)
            return;
        float jacobian = abs(dot(light, normal)) / abs(dot(d, normal));
        float3 origin = asfloat(load_coherent(hot + 4).xyz);
        store_list(at + STACK_LIST_WORD, list);
        c.leg = uint4(asuint(origin), oct_encode_snorm16(d));
        // Past the bend the segment start's last lane keeps the stack's normal.
        float3 from = inside ? asfloat(load_coherent(at + STACK_FROM_WORD).xyz) : origin;
        store_coherent(at + STACK_FROM_WORD, uint4(asuint(from), oct_encode_snorm16(normal)));
        store_coherent(at + STACK_THROUGH_WORD, asuint(float4(1.0, 1.0, 1.0, jacobian)));
        store_coherent(at + STACK_BEND_WORD, uint4(asuint(d), asuint(jacobian / (ior * ior))));
        c.words = uint4(STEP_ON | (bends ? 0 : STEP_PROBED), asuint(ior), 0, 0);
        return;
    }
    uint4 leg = c.leg;
    float3 o = asfloat(leg.xyz);
    float3 d = oct_decode_snorm16(leg.w);
    uint4 start = load_coherent(at + STACK_FROM_WORD);
    float3 from = asfloat(start.xyz);
    float4 kept = asfloat(load_coherent(at + STACK_THROUGH_WORD));
    float3 through = kept.xyz;
    Interior list = load_interior(at + STACK_LIST_WORD);
    float reach = met == MET_NONE ? positive_infinity() : short_of(o, d, t);
    uint inner = list.innermost();
    if (inner != MATERIAL_NONE)
        through *= beer(interior_medium(r.scene, inner, list.point(inner)).sigma_t, segment_length(from, o, d, reach));
    else if (inside)
        through *= medium_transmittance<M>(r.scene.medium, o, d, reach);
    if (met == MET_NONE)
    {
        if (interior_ior(r.scene, inner) != AMBIENT_IOR || !any(through > 0.0))
            return;
        store_coherent(at + STACK_THROUGH_WORD, asuint(float4(through, kept.w)));
        c.words.x = STEP_LIT;
        return;
    }
    float3 n = oct_decode_snorm16(start.w);
    if (!on_face || crossed == STACK_CROSSINGS || !(abs(dot(normal, n)) >= STACK_PARALLEL))
        return;
    bool front = dot(ng, d) < 0.0;
    uint exterior = list.outside(m, m);
    Interior next = list;
    next.cross(m, front, point);
    float3 after;
    if (!stack_direction(light, n, interior_ior(r.scene, next.innermost()), after))
        return;
    BsdfParams p;
    stack_material(r.scene, m, false, p);
    // Fresnel takes the direction on the exterior side.
    through *= straight_transmission(straight_surface(p, ng, front ? d : after), interior_ior(r.scene, exterior));
    float3 position = o + d * t;
    store_list(at + STACK_LIST_WORD, next);
    c.leg = uint4(asuint(ray_origin(position, position, ng, after, ORIGIN_DEFAULT, asfloat(face.w), 0.0)), oct_encode_snorm16(after));
    store_coherent(at + STACK_FROM_WORD, uint4(asuint(position), start.w));
    store_coherent(at + STACK_THROUGH_WORD, asuint(float4(through, kept.w)));
    c.words = uint4(STEP_ON, asuint(interior_ior(r.scene, next.innermost())), 0, 0);
}
