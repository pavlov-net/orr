// What the shaders of manifold next-event estimation (`orr_lighting.path.manifold`) share: the
// state of a walk in a path's manifold words, the step a surface's hit hands the face step, what
// the steps answer, and the microfacet normals a walk refracts about.
module facet;

import orr_shaders.abi;
import orr_shaders.pipeline;
import orr_shaders.math;
import orr_shaders.sampling;
import orr_lighting.interiors;
import orr_lighting.path.state;
import orr_shaders.packing;
__exported import orr_scene.gbuffer;

// Surfaces a walk crosses at most, and the walks a solve takes at most, whether its steps are
// Newton's or a surface's model, which gives the next start within rounding where the chain
// crosses that surface's triangle: a chain across one surface whose root lies past the triangle
// the walk met takes Newton's steps too.
public static const uint MANIFOLD_CROSSINGS = 4;
public static const uint MANIFOLD_WALKS = 8;
// The traces a solve takes at most: the probes, then each walk's face rays.
public static const uint MANIFOLD_TRACES = 4 + MANIFOLD_WALKS * (MANIFOLD_CROSSINGS + 1);
// A chain whose last direction lies within this many radians of the light's has converged.
public static const float MANIFOLD_TOLERANCE = 1e-4;
// The largest move of a Newton step, radians, and the one past which the next walk is not
// likely to converge.
public static const float MANIFOLD_STEP = 0.25;
public static const float MANIFOLD_SETTLED = 3e-3;
// A ray's chain is an estimate's where the estimate's start direction from the ray's origin lies
// within this many radians of the ray's: a converged chain's last direction lies within
// MANIFOLD_TOLERANCE of the light's, so its start direction within that over the chain's
// Jacobian, which a caustic's focus makes small; two chains to one light lie closer only near the
// fold where they merge.
public static const float MANIFOLD_OWNS = 1e-2;
// A walk whose next start direction lies within this many radians of a ray's or a photon's chain
// converges to that chain, the start an exact surface model gives lying within rounding of it.
public static const float MANIFOLD_PREDICT = 1e-3;
// The Newton steps a face step takes on its surface's model, and the gap within which a chain on
// the model has converged.
public static const uint MODEL_STEPS = 8;
public static const float MODEL_TOLERANCE = 1e-5;

// The walk's words, from MANIFOLD_WALK_WORD: its start (the ray origin of its point) with its state
// bits; the light's direction, with the IOR of the medium the walk's current leg lies in, which its
// thin walls take; a slope per surface (`unit_slope`), as two halves each; the current leg's start
// point and direction; the chain's differentials there with respect to the start direction, the
// point's along the basis of the start direction and then the direction's, three floats each; and
// the interior list the current leg lies in; and where the chain across the walk's first surface
// alone meets that surface's plane as its model gives it (`model_aim`), with 1 in its last lane
// where it gave one within the triangle the model is exact across, MODEL_PAST where past it, or,
// right after a walk starts, 2 where its first ray aims at that point; and
// what a textured medium reads at the surface a face step is at, which its hit leaves there, or
// after a step that answers STEP_SETTLED what the end step would weigh its chain by.
public static const uint WALK_START = MANIFOLD_WALK_WORD;
public static const uint WALK_TARGET = MANIFOLD_WALK_WORD + 1;
public static const uint WALK_SLOPES = MANIFOLD_WALK_WORD + 2;
public static const uint WALK_POINT = MANIFOLD_WALK_WORD + 3;
public static const uint WALK_DIR = MANIFOLD_WALK_WORD + 4;
public static const uint WALK_DIFF = MANIFOLD_WALK_WORD + 5;
public static const uint WALK_LIST = MANIFOLD_WALK_WORD + 8;
public static const uint WALK_MODEL = MANIFOLD_WALK_WORD + 13;
public static const uint WALK_MEDIUM = MANIFOLD_WALK_WORD + 14;

// The state bits of a walk, in its start word's last lane: what it is (the probe, a walk toward
// the solution, or one that weighs its chain), whether it estimates, which weighs the chain once it
// converges, whether its first probe went along the light's direction bent by the vertex's normal
// and whether its second went along the light's own (`path_manifold_end`), the walks begun, and the
// surfaces it crossed. Before the first walk the direction word holds the vertex's unit normal
// with 1 in its last lane, or zeros where the vertex has none.
public static const uint WALK_PROBE = 0;
public static const uint WALK_SOLVE = 1;
public static const uint WALK_WEIGH = 2;
public static const uint WALK_PHASE = 3;
public static const uint WALK_ESTIMATES = 1 << 2;
public static const uint WALK_PROBED = 1 << 3;
public static const uint WALK_PROBED_AGAIN = 1 << 11;
// The walk's target is a point, which its target word holds, not a direction.
public static const uint WALK_POINT_TARGET = 1 << 12;
// A froxel's walk (`froxel_walks`), which weighs its chain without the linked medium, which the
// froxels take straight toward the light; and one of a shadowed light's job, which weighs it
// without the medium of its first leg, whose length its point word's last lane keeps.
public static const uint WALK_FROXEL = 1 << 15;
public static const uint WALK_OWN_LEG = 1 << 16;
// A walk of the solve started where a surface's model put the chain past the triangle the model
// is exact across: the next walk met another triangle, whose model can send it back, so the solve
// takes no more such starts (`path_manifold_end`).
public static const uint WALK_MODEL_PAST = 1 << 17;
public static const uint MODEL_PAST = 3;
// An estimate's probe leaves toward the point its model word holds (`start_probe`), not along the
// light bent across the start's plane.
public static const uint WALK_AIMED = 1 << 18;
// The solve accepted a walk whose chain met nothing past its last surface, whose start direction
// the bend word's last lane holds as an octahedral code (`path_manifold_end`).
public static const uint WALK_ACCEPTED = 1 << 19;
// The bits every walk of an estimate or a check keeps from its start.
public static const uint WALK_KEPT = WALK_ESTIMATES | WALK_POINT_TARGET | WALK_FROXEL | WALK_OWN_LEG | WALK_MODEL_PAST | WALK_ACCEPTED;
// The probes sent toward the start a surface's model gave past the triangle it is exact across,
// MANIFOLD_PROBES at most with the first.
public static const uint WALK_PROBES_SHIFT = 13;
public static const uint MANIFOLD_PROBES = 3;
public static const uint WALK_BEGUN_SHIFT = 4;
public static const uint WALK_CROSSED_SHIFT = 8;

// The track of the surfaces a path's ray or a photon crossed by transmission, in its manifold words
// from MANIFOLD_TRACK_WORD: for a ray, the origin of the vertex's ray that scattered before them
// with the density it drew the ray at, and for a photon, the direction toward its light; the
// vertex's ray direction and shading normal as octahedral codes, for a ray where the vertex is a
// surface that leaves its jobs toward lights past it to the checks of the ray that reached it, its
// geometric normal on its view's side as one, else TRACK_NO_FACE, and in the last lane its
// probability of running next-event estimation as a half in the low bits and the crossings, their
// sides, whether it lay in a medium and its depth above them (TRACK_*); for a photon the first
// surface's point with what the track is, and for a ray the origin and normal such a surface
// leaves for the track that may start past it (`manifold_track`), no origin from the path's start;
// the slopes it crossed at (`face_slope`), as two halves each; and the surfaces' section
// materials.
public static const uint TRACK_NONE = 0;
public static const uint TRACK_PHOTON = 2;
public static const uint TRACK_NO_FACE = 0x80008000u;
public static const uint TRACK_CROSSED_SHIFT = 16;
public static const uint TRACK_FRONTS_SHIFT = 19;
public static const uint TRACK_MEDIUM = 1 << 23;
public static const uint TRACK_DEPTH_SHIFT = 24;
// The track's start drew its directional lights with no normal (HOT_ANY_SIDE).
public static const uint TRACK_ANY_SIDE = 1u << 31;
// A walk's direction word's last lane before its probe, where the word holds the normal of its
// start: the probe bends the light across planes facing the normal on the light's side (1), or
// mirrors a light behind the normal to its side first (WALK_MIRRORS), as a vertex that draws its
// directional lights with no normal does (`start_probe`).
public static const float WALK_MIRRORS = 2.0;

// Lane `i` of `v`, and `v` with lane `i` set to `x`, by selects: a lane a run-time index picks
// would put the vector in each thread's local memory.
public uint lane(uint4 v, uint i)
{
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

public uint4 with_lane(uint4 v, uint i, uint x)
{
    return select(uint4(0, 1, 2, 3) == i, uint4(x), v);
}

// The surfaces a track crossed, from its second word.
public uint track_crossed(uint4 k)
{
    return (k.w >> TRACK_CROSSED_SHIFT) & 0x7;
}

// The interior list where the ray track in manifold words `at` started, from `past`, the list past
// its last crossing: back across its surfaces.
public Interior track_start(uint4* at, Interior past)
{
    Interior list = past;
    uint4* track = at + MANIFOLD_TRACK_WORD;
    uint4 k = load_coherent(track + 1);
    uint4 materials = load_coherent(track + 4);
    for (uint i = track_crossed(k); i > 0; i--)
        list.cross(lane(materials, i - 1), (k.w & (1u << (TRACK_FRONTS_SHIFT + i - 1))) == 0);
    return list;
}

// What a face ray met: nothing, a dielectric whose hit stepped the walk, or anything else; and what
// a walk past its probe hands the end step where the face step at a dielectric found no way on.
public static const uint MET_NONE = 0;
public static const uint MET_FACE = 1;
public static const uint MET_OTHER = 2;
public static const uint MET_FAILED = 3;

// What a step answers in its first word: no chain reaches the light; the next leg, from the origin
// in the other three along the walk's direction word; the next walk from the start along it; the
// weighed walk reached the light; a walk that weighs nothing converged. With STEP_WEIGHS the next
// leg is the weighed walk's, whose shadow ray the raygen's loop traces, and with STEP_OUT it leaves
// every interior. With STEP_INNER the step weighed in place the chain its probe solved (`face_step`),
// whose first leg, from the walk's start to the point the step leaves, no face ray traced.
public static const uint STEP_DARK = 0;
public static const uint STEP_ON = 1;
public static const uint STEP_RESTART = 2;
public static const uint STEP_LIT = 3;
public static const uint STEP_FOUND = 4;
// What a check's walk answers where its next start direction is the chain it checks
// (MANIFOLD_PREDICT).
public static const uint STEP_MATCH = 5;
public static const uint STEP_WEIGHS = 8;
public static const uint STEP_OUT = 16;
public static const uint STEP_INNER = 32;
// With STEP_SETTLED the chain weighed in place has converged too: where its leg out of every
// interior meets nothing more, it reached the light, at what the walk's medium word holds, the
// linked medium's transmittance past the surface and the density (`face_step`).
public static const uint STEP_SETTLED = 64;
public static const uint STEP_STATUS = 7;

// What a surface's hit hands the face step: its shading normal on the material's front side at the
// hit as its model holds it (`SurfaceChange`), with the alpha of its base specular lobe; its
// geometric normal on that side, with the error a ray origin there clears; the changes of the
// normal and the alpha along the basis (`orthonormal_basis`) of the geometric normal; the hit
// point, with its section material; the hit's barycentrics and their changes; and its BSDF's
// parameters. The step answers in the first word.
public struct ManifoldStep
{
    public uint4 answer;
    public float4 normal;
    public float4 geometric;
    public float4 gradient_t;
    public float4 gradient_b;
    public float4 at;
    public float2 bary;
    public float4 bary_change;
    public BsdfParams p;
}

// A face step as the hit's call hands it on (`face_call`), small so the call keeps little of its
// stage's stack: the model's normal with the section material; the hit point with its rounding
// error; the geometric normal as an octahedral code, the barycentrics, the alpha's changes and the
// barycentrics' as halves; the normal's changes as halves; and of the BSDF's parameters what
// refraction through the surface reads, all but the base's roughness, the subsurface lobe's and
// the emission, which add nothing toward a refracted direction past solid glass: IORs and a thin
// film's thickness in full, the rest as halves and the coat's normal as an octahedral code. What a
// textured medium reads at the hit goes in the walk's medium word. The step's answer replaces the
// first word.
public struct FaceCall
{
    public uint4 normal;
    public uint4 at;
    public uint4 frame;
    public uint4 change;
    public uint4 lobes;
    public uint4 coat;
    public uint4 colors;
    public uint4 more;
    public uint4 film;
    public uint extra;
}

public FaceCall face_call(ManifoldStep c)
{
    BsdfParams p = c.p;
    FaceCall f;
    f.normal = uint4(asuint(c.normal.xyz), asuint(c.at.w));
    f.at = uint4(asuint(c.at.xyz), asuint(c.geometric.w));
    f.frame = uint4(oct_encode_snorm16(c.geometric.xyz), as_halves(c.bary), as_halves(float2(c.gradient_t.w, c.gradient_b.w)), as_halves(c.bary_change.xy));
    f.change = uint4(as_halves(c.gradient_t.xy), as_halves(float2(c.gradient_t.z, c.gradient_b.x)), as_halves(c.gradient_b.yz), as_halves(c.bary_change.zw));
    f.lobes = uint4(as_halves(float2(p.metalness, p.specular_weight)), as_halves(float2(p.specular_roughness, p.specular_anisotropy)), asuint(p.specular_ior), p.flags);
    f.coat = uint4(as_halves(float2(p.tangent_angle, p.transmission_weight)), as_halves(float2(p.coat_weight, p.coat_roughness)), asuint(p.coat_ior), as_halves(float2(p.coat_darkening, p.coat_anisotropy)));
    f.colors = uint4(as_halves(p.specular_color.xy), as_halves(float2(p.specular_color.z, p.transmission_color.x)), as_halves(p.transmission_color.yz), as_halves(p.coat_color.xy));
    f.more = uint4(as_halves(float2(p.coat_color.z, p.fuzz_color.x)), as_halves(p.fuzz_color.yz), as_halves(p.base_color.xy), as_halves(float2(p.base_color.z, p.fuzz_weight)));
    f.film = uint4(as_halves(float2(p.coat_tangent_angle, p.fuzz_roughness)), oct_encode_snorm16(p.coat_normal), as_halves(float2(p.thin_film_weight, p.thin_film_ior)), asuint(p.thin_film_thickness));
    f.extra = 0;
    return f;
}

public ManifoldStep face_step_of(FaceCall f)
{
    ManifoldStep c;
    c.answer = uint4(0);
    float2 alpha = halves(f.frame.z);
    float2 g0 = halves(f.change.x);
    float2 g1 = halves(f.change.y);
    float2 g2 = halves(f.change.z);
    c.normal = float4(asfloat(f.normal.xyz), 0.0);
    c.at = float4(asfloat(f.at.xyz), asfloat(f.normal.w));
    c.geometric = float4(oct_decode_snorm16(f.frame.x), asfloat(f.at.w));
    c.gradient_t = float4(g0, g1.x, alpha.x);
    c.gradient_b = float4(g1.y, g2, alpha.y);
    c.bary = halves(f.frame.y);
    c.bary_change = float4(halves(f.frame.w), halves(f.change.w));
    BsdfParams p = BsdfParams();
    float2 a = halves(f.lobes.x);
    float2 b = halves(f.lobes.y);
    p.metalness = a.x;
    p.specular_weight = a.y;
    p.specular_roughness = b.x;
    p.specular_anisotropy = b.y;
    p.specular_ior = asfloat(f.lobes.z);
    p.flags = f.lobes.w;
    float2 e = halves(f.coat.x);
    float2 k = halves(f.coat.y);
    float2 d = halves(f.coat.w);
    p.tangent_angle = e.x;
    p.transmission_weight = e.y;
    p.coat_weight = k.x;
    p.coat_roughness = k.y;
    p.coat_ior = asfloat(f.coat.z);
    p.coat_darkening = d.x;
    p.coat_anisotropy = d.y;
    float2 c0 = halves(f.colors.x);
    float2 c1 = halves(f.colors.y);
    float2 c2 = halves(f.colors.z);
    float2 c3 = halves(f.colors.w);
    float2 m0 = halves(f.more.x);
    float2 m1 = halves(f.more.y);
    float2 m2 = halves(f.more.z);
    float2 m3 = halves(f.more.w);
    p.specular_color = float3(c0, c1.x);
    p.transmission_color = float3(c1.y, c2);
    p.coat_color = float3(c3, m0.x);
    p.fuzz_color = float3(m0.y, m1);
    p.base_color = float3(m2, m3.x);
    p.fuzz_weight = m3.y;
    float2 t0 = halves(f.film.x);
    float2 t1 = halves(f.film.z);
    p.coat_tangent_angle = t0.x;
    p.fuzz_roughness = t0.y;
    p.coat_normal = oct_decode_snorm16(f.film.y);
    p.thin_film_weight = t1.x;
    p.thin_film_ior = t1.y;
    p.thin_film_thickness = asfloat(f.film.w);
    c.p = p;
    return c;
}

// The slope of unit microfacet normal `m` about unit normal `n`, which `facet` turns back into `m`.
public float2 facet_slope(float3 n, float3 m)
{
    float3 t, b;
    facet_basis(n, t, b);
    float c = dot(m, n);
    return float2(-dot(m, t), -dot(m, b)) / c;
}

// The density per solid angle of the microfacet normal of slope `a` x `slope` that `unit_slope`
// draws at alpha `a`: GGX's D(m) cos(m), from the slope, whose cosine near 1 loses the digits a
// small alpha needs.
public float facet_density(float a, float2 slope)
{
    float s2 = dot(slope, slope);
    float t2 = a * a * s2;
    return (1.0 + t2) * sqrt(1.0 + t2) / (PI * a * a * (1.0 + s2) * (1.0 + s2));
}
