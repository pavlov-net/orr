// The boundary step: a raygen traces a PATH ray and reads what it met from the hit object and the
// hit record's data, running no shader. A caller that must evaluate the surface invokes the hit
// object itself, which it keeps in a local: a copied struct that holds one loses it. Interior and
// subsurface walks, the crossings before a path's start and a shadow ray's crossings take their
// steps here.
module boundary;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_scene.material;
import orr_scene.runs;
import orr_scene.tables;
import orr_scene.candidates;
import orr_lighting.interiors;
import orr_lighting.specialize;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.slot;

// What a boundary step met. On a miss `t` is the ray's reach and the rest is empty.
public struct Boundary
{
    public bool hit;
    public bool triangle;           // a triangle's hit, which has corners; a strand's has none
    public float t;
    public uint instance;           // the hit's instance index
    public uint material;           // the record's section material, MATERIAL_NONE for none
    public uint back;               // the record's back material, MATERIAL_NONE for none
    public uint flags;              // the record's HIT_* and class
}

// Traces `ray` with ray flags `flags` and instance mask `mask` under the cover in `p`'s words
// (`path_payload`), which the hit's any-hit shaders read and its shader would take if invoked.
[ForceInline]
public Traced boundary_trace(SceneRoot* s, RayDesc ray, uint flags, uint mask, inout Payload p)
{
    return trace_hit(s.tlas, flags, mask, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
}

// What `boundary_trace`'s `h` met, from the hit object and its record's data.
public Boundary boundary_of(Traced h)
{
    Boundary b;
    b.hit = h.hit();
    b.t = h.t();
    b.triangle = false;
    b.instance = 0;
    b.material = MATERIAL_NONE;
    b.back = MATERIAL_NONE;
    b.flags = 0;
    if (!b.hit)
        return b;
    uint kind = h.hit_kind();
    b.triangle = kind == HIT_KIND_TRIANGLE_FRONT_FACE || kind == HIT_KIND_TRIANGLE_BACK_FACE;
    b.instance = h.instance_index();
    HitData d = h.record<HitData>();
    b.material = d.material;
    b.back = d.back;
    b.flags = d.flags;
    return b;
}

// The render-space geometric normal of triangle hit `b` (`h`) along unit `dir`, as `hit_geometry`
// gives it from the corners traversal fetched: the triangle's front face, turned toward the ray on
// a back face whose side's material is no dielectric's, so a ray knows the way it crosses a
// dielectric. From the corners rather than the hit kind, whose side a mirroring placement's facing
// flags may turn.
public float3 boundary_normal(SceneRoot* s, Traced h, Boundary b, float3 dir)
{
    float3 corners[3] = h.corners();
    float3x3 unit = unit_linear(h.object_to_world());
    float3 e1 = mul(unit, corners[1] - corners[0]);
    float3 e2 = mul(unit, corners[2] - corners[0]);
    float size = unit_factor(max(abs(e1), abs(e2)));
    float3 n = unit_direction(cross(e1 * size, e2 * size)) * linear_mirror(unit);
    if (!(dot(n, dir) > 0.0))
        return n;
    uint side = b.back != MATERIAL_NONE ? b.back : b.material;
    return (s.materials[side].flags & MATERIAL_DIELECTRIC) != 0 ? n : -n;
}

// The crossings before a path's start, which its start callable asks for with STEP_CROSS
// ---------------------------------------------------------------------------------------------

// Leaves in `slot`'s shadow ray words, which no shadow ray holds before the path's first vertex,
// the ray from `origin` along unit `dir` to `t_max` whose crossings a start asks for.
public void store_crossing_ray(PathRoot r, uint slot, float3 origin, float3 dir, float t_max)
{
    uint4* at = hot_words(r, slot);
    ShadowOrigin o = { origin, t_max };
    ShadowDir d = { dir, 0 };
    store_word(at + SHADOW_ORIGIN_AT, o);
    store_word(at + SHADOW_DIR_AT, d);
}

// The cross fold callable's body: flips the interiors `slot`'s path started in at each of the
// `crossings` crossings the raygen traced for its start, in their order, and stores them with
// what the start made of them: whether the path lies in one, whether its camera ray takes an
// interior's segment, and whether the pixel of a first vertex inside one runs. `step` is the
// start's, which it hands back as the path goes on with it.
public uint launch_cross_fold(PathRoot r, uint slot, uint step, uint crossings)
{
    uint4* at = hot_words(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    LoopThroughput t = load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT);
    PathLoop p = empty_loop(r, pixel);
    p.flags = t.flags;
    ShadeState c = launch_state(LayerRay());
    // A path that started in no interior stored none: its ray starts at its origin.
    c.ray.start = load_word<RayOrigin>(at + RAY_ORIGIN_AT).origin;
    load_interiors(r, slot, p, c);
    uint2* crossed = (uint2*)scratch_words(r, slot);
    for (uint k = 0; k < crossings; k++)
    {
        uint2 x = load_coherent(crossed + k);
        c.interior.cross(x.x, x.y != 0);
    }
    store_interiors(r, slot, p, c);
    set_flag(p.flags, LOOP_CARRIED, c.interior.count > 0 && !has_froxels(r));
    t.flags = p.flags;
    store_word(at + LOOP_THROUGHPUT_AT, t);
    if ((step & STEP_RAY) != 0)
    {
        RayDir d = load_word<RayDir>(at + RAY_DIR_AT);
        d.word = (d.word & ~(LAUNCH_INSIDE | LAUNCH_SEGMENTED)) | (in_interior(r, p, c.interior) ? LAUNCH_INSIDE | LAUNCH_SEGMENTED : 0);
        store_word(at + RAY_DIR_AT, d);
    }
    if ((step & STEP_VERTEX) != 0)
        step = (step & ~STEP_UNSHADED) | (c.interior.count == 0 && !runs_path(r, pixel) ? STEP_UNSHADED : 0);
    return step & ~STEP_CROSS;
}

// The surfaces of an interior that the crossing ray in `slot`'s shadow ray words crosses, as
// `surface_crossings` takes them: each step meets the nearest surface past the last but presence
// layers, up to SURFACE_CROSSINGS steps, and the walk stops at the ray's reach or a hit with no
// geometry or material. Each crossing's material and whether the ray meets its front go to the
// scratch: their count.
uint trace_crossings(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    ShadowOrigin o = load_word<ShadowOrigin>(at + SHADOW_ORIGIN_AT);
    float3 dir = load_word<ShadowDir>(at + SHADOW_DIR_AT).dir;
    uint2* out = (uint2*)scratch_words(r, slot);
    uint mask = MASK_OPAQUE | MASK_DIELECTRIC;
    PathCover cover = new_path_cover(mask, COVER_LAYERS, PRESENCE_NONE, 0, hit_context(HIT_PRIMARY, 0, camera_cone(r.view), 0.0));
    uint crossed = 0;
    float along = 0.0;
    for (uint k = 0; k < SURFACE_CROSSINGS && along < o.t_max; k++)
    {
        Payload p;
        p.words = path_payload(cover);
        Traced h = boundary_trace(r.scene, path_ray(o.origin + dir * along, dir, o.t_max - along), RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, p);
        Boundary b = boundary_of(h);
        if (!b.triangle || b.material == MATERIAL_NONE)
            break;
        if ((b.flags & HIT_CROSSES_INTERIOR) != 0)
        {
            bool front = dot(boundary_normal(r.scene, h, b, dir), dir) < 0.0;
            store_coherent(out + crossed, uint2(b.material, front ? 1u : 0u));
            crossed++;
        }
        along += b.t + 1e-4 * (1.0 + b.t);
    }
    return crossed;
}

// Runs callable `index`, a start or a finish, with `word` in its data's second word, then traces
// the crossings it asks for (STEP_CROSS), which the cross fold takes: the step the path goes on
// with.
uint crossed_call(PathRoot r, uint slot, uint index, uint word)
{
    uint step = step_call(index, word);
    // A path that ends where it starts crosses nothing.
    if (!may(MAY_INTERIORS) || (step & STEP_CROSS) == 0 || (step & (STEP_RAY | STEP_VERTEX)) == 0)
        return step & ~STEP_CROSS;
    uint crossings = trace_crossings(r, slot);
    return data_call(CALL_CROSS_FOLD, uint4(step, crossings, 0, 0)).x;
}

// The start of `slot`'s path by its source's callable: a view's debug view of its first vertex
// alone starts from its own.
public uint start_path(PathRoot r, uint slot)
{
    if (from_view(r))
        return crossed_call(r, slot, may(MAY_DEBUG) && first_vertex_mode(r) ? CALL_START_FIRST_VERTEX : CALL_START_VIEW, 0);
    if (from_background(r))
        return step_call(CALL_START_BACKGROUND);
    if (from_update(r))
        return crossed_call(r, slot, CALL_START_UPDATE, 0);
    return step_call(CALL_START_PHOTON);
}

// The finish of `slot`'s path in pass `pass` by its source's callable: the step of a pass that
// follows, or STEP_DONE.
public uint finish_path(PathRoot r, uint slot, uint pass)
{
    if (from_view(r))
        return crossed_call(r, slot, CALL_FINISH_VIEW, pass);
    if (from_background(r))
        return step_call(CALL_FINISH_BACKGROUND);
    return step_call(CALL_FINISH_RECORD);
}
