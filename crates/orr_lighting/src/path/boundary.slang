// What a walk's ray met, read from its hit object and the hit record's data with no shader run: the
// boundary steps of subsurface and interior walks. A caller that must evaluate the surface invokes
// the hit object itself, which it keeps in a local: a copied struct that holds one loses it.
module boundary;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.pipeline;
import orr_scene.tables;
import orr_scene.material;
import orr_scene.runs;

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

// What a path ray's `h` met, from the hit object and its record's data.
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
