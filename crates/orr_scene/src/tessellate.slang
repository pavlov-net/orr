// The tessellator: dices the patch triangles a tessellated entry's kernel emits into
// `SceneRoot.patches` into generated clusters in the entry's batch, once per frame for
// raster and ray tracing alike.
//
// `begin` resets the entry's counts and steps its rate; `patches` classifies each patch
// triangle; `args` sizes the next pass; each of SPLITS `split` passes cuts the subpatches
// whose edges need more than MAX_DICE segments by the pattern of their split factors;
// `dice` writes one generated cluster per subpatch, its vertices on the patch triangle or a
// subdivision copy's limit surface, displaced through the linked `IDisplacement` with
// normals from finite differences of the displaced field, and a copy's uv sets and vertex color
// stream at each;
// `weld_clear`, `weld` and `weld_apply` give vertices on patch triangle edges that share an
// undisplaced point one displaced position, which closes uv seams; `finish` keeps the
// figures. An edge's factor depends only on its endpoints, this frame's and last frame's, so
// patches on both sides of it cut it alike: its undisplaced world length in pixels at the
// scene focus over the dice rate, times the entry's rate scale, over 15 where last frame's
// first view of the scene saw the edge, where it was then, neither in its frustum nor in front
// of its HZB.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.atomic;
import orr_shaders.math;
import orr_shaders.packing;
import orr_scene.producer;
import orr_gpu.rtas;
import orr_scene.tables;
import orr_scene.tessellation;
import orr_scene.visibility;

// Its taps drive texture streaming.
export static const bool TEXTURE_FEEDBACK = true;
__exported import orr_scene.displacement;

extern struct Displace : IDisplacement = TextureDisplacement;

// Split passes: patch triangles take up to MAX_DICE^(SPLITS + 1) segments per edge.
static const uint SPLITS = 3;
static const uint MAX_FACTOR = MAX_DICE * MAX_DICE * MAX_DICE * MAX_DICE * FACTOR_ONE;
// Edges no view saw last frame take this share of the rate, never less than one segment.
static const float UNSEEN_RATE = 1.0 / 15.0;
// The rate steps by this factor above STEP_DOWN and below STEP_UP of capacity asked for.
static const float RATE_STEP = 0.9;
static const float STEP_DOWN = 0.85;
static const float STEP_UP = 0.6;
static const float FOCUS_NEAR = 0.01;
static const uint WELD_PROBES = 32;

// `TessRoot.flags`.
static const uint TESS_FOCUS = 1;   // the scene has a focus

public struct TessRoot
{
    public SceneRoot* scene;
    public TessGpu* tess;
    public TessStateGpu* state;      // the entry's
    public PatternTable* patterns;
    public ViewConstants* view;      // last frame's first view of the scene; null: none
    public RasterGpu* raster;        // its HZB; null: frustum only
    public int4 focus_cell;
    public float4 focus_offset;      // w: pixels per radian
    public uint entry;
    public uint level;               // the split pass
    public float dice_pixels;
    public uint flags;               // TESS_*
}
ORR_ROOT(TessRoot);

ProducerGpu entry()
{
    return root.scene.producers[root.entry];
}

ProducerBatch patches()
{
    return root.scene.patches[root.entry];
}

ProducerBatch drawn()
{
    return root.scene.batches[root.entry];
}

// An item's fixed-point factors, each from one segment to MAX_DICE.
uint3 item_factors(PatchItem item)
{
    return clamp(unpack_factors(item.factors), FACTOR_ONE, MAX_DICE * FACTOR_ONE);
}

// One patch triangle: entry-space corners, attributes and words, and for a subdivision
// copy the base triangle it covers.
struct Patch
{
    ProducerClusterGpu cluster;
    ProducerTriangle current;
    ProducerTriangle previous;
    bool has_previous;
    SubdTriangleGpu base;
}

Patch load_patch(uint code)
{
    ProducerBatch b = patches();
    Patch p;
    p.cluster = b.clusters[code >> 7];
    p.current = producer_triangle(b, p.cluster, code & 127, false);
    p.has_previous = b.previous != nullptr;
    p.previous = producer_triangle(b, p.cluster, code & 127, p.has_previous);
    p.base = (SubdTriangleGpu)0;
    SubdCopyGpu* c = root.tess.subd;
    if (c != nullptr)
        p.base = c.cage.triangles[c.cage.clusters[code >> 7].z + (code & 127)];
    return p;
}

// What edge factors read: the entry placed at the focus, and as last frame placed it at last
// frame's view.
struct Rates
{
    Placement focus;
    Placement view;
    bool seen_test;
    float scale;
    float reach;
}

Rates rates()
{
    Rates r;
    r.scale = root.state.rate_scale;
    r.reach = root.tess.reach;
    ProducerGpu p = entry();
    SceneRoot* s = root.scene;
    if (!producer_placement(s, p, false, root.focus_cell.xyz, root.focus_offset.xyz, uint(s.origin_cell.w), r.focus))
        r.focus = (Placement)0;
    r.seen_test = false;
    if (root.view != nullptr)
    {
        // Where its group hid it last frame, that view's depth holds nothing of it, and it is
        // tested where it is now.
        ViewConstants* v = root.view;
        int3 cell = v.eye_cell.xyz;
        uint cell_log2 = uint(v.eye_cell.w);
        r.seen_test = producer_placement(s, p, true, cell, v.eye_offset.xyz, cell_log2, r.view)
            || producer_placement(s, p, false, cell, v.eye_offset.xyz, cell_log2, r.view);
    }
    return r;
}

// Whether last frame's view saw the sphere of `radius` m (entry space) around `center`, both
// where they were last frame.
bool seen(Rates r, float3 center, float radius)
{
    if (!r.seen_test)
        return true;
    return !sphere_unseen(root.view, root.raster, false, place(r.view, center), radius * r.view.scale);
}

// Segments edge a-b asks for, which ran from `a0` to `b0` last frame: the same from either
// end, bit for bit.
float edge_factor(Rates r, float3 a, float3 b, float3 a0, float3 b0)
{
    if ((root.flags & TESS_FOCUS) == 0)
        return 1.0;
    precise float len = length(mul(r.focus.linear, a - b));
    precise float d = max(min(length(place(r.focus, a)), length(place(r.focus, b))), FOCUS_NEAR);
    precise float t = len / d * root.focus_offset.w / root.dice_pixels * r.scale;
    float3 mid = 0.5 * (a0 + b0);
    if (!seen(r, mid, 0.5 * length(a0 - b0) + r.reach))
    {
        InterlockedAdd(root.tess.counters[TESS_UNSEEN], 1);
        t *= UNSEEN_RATE;
    }
    return t;
}

void push_dice(PatchItem item)
{
    TessGpu* t = root.tess;
    uint at;
    InterlockedAdd(t.counters[TESS_DICE], 1, at);
    if (at < t.dice_capacity)
        t.dice[at] = item;
    else
        InterlockedAdd(t.counters[TESS_DICE_DROPPED], 1);
}

// Sends the subpatch with corners at `p`, at `p0` last frame, to the split queue of pass
// `level` or the dice list; `item.rate` is set at level 0. A split cuts each edge by a
// fraction of its factor, so subpatch corners move continuously with the factors as dice
// points do.
void classify(Rates r, inout PatchItem item, float3 p[3], float3 p0[3], uint level)
{
    float3 t = float3(edge_factor(r, p[0], p[1], p0[0], p0[1]), edge_factor(r, p[1], p[2], p0[1], p0[2]),
                      edge_factor(r, p[2], p[0], p0[2], p0[0]));
    // Fixed point, at least one segment (NaN included) and at most what the splits reach.
    float3 f = round(t * float(FACTOR_ONE));
    uint3 q = select(f >= float(FACTOR_ONE), uint3(min(f, float(MAX_FACTOR))), uint3(FACTOR_ONE));
    uint most = max_component(q);
    if (level == 0)
        item.rate = float(most) / float(FACTOR_ONE);
    TessGpu* g = root.tess;
    if (most > MAX_DICE * FACTOR_ONE && level < SPLITS)
    {
        uint3 split = clamp((q + MAX_DICE - 1) / MAX_DICE, FACTOR_ONE, MAX_DICE * FACTOR_ONE);
        uint at;
        InterlockedAdd(g.counters[TESS_QUEUE + (level & 1)], 1, at);
        if (at < g.queue_capacity)
        {
            pack_factors(split, item.factors);
            g.queue[level & 1][at] = item;
            return;
        }
        InterlockedAdd(g.counters[TESS_SPLIT_DROPPED], 1);
    }
    pack_factors(min(q, MAX_DICE * FACTOR_ONE), item.factors);
    push_dice(item);
}

// The point of the patch triangle at `corner`, last frame's with `previous`.
float3 corner_point(Patch pt, uint3 corner, bool previous)
{
    ProducerTriangle tri = previous ? pt.previous : pt.current;
    float3 c[3] = tri.corners.position;
    return interpolate(c, bary_weights(corner));
}

// The corners of triangle `i` of patch cluster `k`, last frame's with `previous` where the
// batch keeps them, as `load_patch` reads them.
void patch_corners(ProducerBatch b, ProducerClusterGpu k, uint i, bool previous, out float3 p[3])
{
    uint3 c = cluster_corners(k, i) + k.first_vertex;
    float3* positions = previous && b.previous != nullptr ? b.previous : b.positions;
    for (uint j = 0; j < 3; j++)
        p[j] = positions[c[j]];
}

// Steps the rate from last frame's capacity asked for and resets this frame's counts.
[shader("compute")]
[numthreads(1, 1, 1)]
void begin(uint3 id: SV_DispatchThreadID)
{
    TessGpu* t = root.tess;
    TessStateGpu* st = root.state;
    float scale = st.rate_scale > 0.0 ? st.rate_scale : 1.0;
    if (st.usage > STEP_DOWN)
        scale *= RATE_STEP;
    else if (st.usage < STEP_UP)
        scale = min(scale / RATE_STEP, 1.0);
    st.rate_scale = scale;
    for (uint i = 0; i < TESS_COUNTERS; i++)
        t.counters[i] = 0;
    ProducerBatch d = drawn();
    *d.cluster_count = 0;
    *d.vertex_count = 0;
    write_grid(t.args + TESS_ARGS_PATCHES, emitted_clusters(patches()));
}

// One workgroup per patch cluster, one thread per triangle.
[shader("compute")]
[numthreads(128, 1, 1)]
void patches(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    ProducerBatch b = patches();
    uint c = grid_index(group);
    if (c >= emitted_clusters(b))
        return;
    ProducerClusterGpu k = b.clusters[c];
    if (!producer_cluster_valid(b, k) || thread >= k.tri_count)
        return;
    InterlockedAdd(root.tess.counters[TESS_PATCHES], 1);
    float3 p[3], p0[3];
    patch_corners(b, k, thread, false, p);
    patch_corners(b, k, thread, true, p0);
    PatchItem item;
    item.patch = (c << 7) | thread;
    set_corner(item.corners, 0, uint3(BARY_ONE, 0, 0));
    set_corner(item.corners, 1, uint3(0, BARY_ONE, 0));
    set_corner(item.corners, 2, uint3(0, 0, BARY_ONE));
    item.rate = 1.0;
    item.zeros = PATCH_ZEROS;
    classify(rates(), item, p, p0, 0);
}

// Sizes the next split pass and empties the queue it pushes to; with `level` == SPLITS,
// sizes the dice pass.
[shader("compute")]
[numthreads(1, 1, 1)]
void args(uint3 id: SV_DispatchThreadID)
{
    TessGpu* t = root.tess;
    uint level = root.level;
    if (level < SPLITS)
    {
        uint n = min(t.counters[TESS_QUEUE + (level & 1)], t.queue_capacity);
        write_grid(t.args + TESS_ARGS_SPLIT, (n + 63) / 64);
        t.counters[TESS_QUEUE + ((level + 1) & 1)] = 0;
        return;
    }
    write_grid(t.args + TESS_ARGS_DICE, min(t.counters[TESS_DICE], t.dice_capacity));
}

// One thread per subpatch of split pass `level`: its pattern's triangles become the
// subpatches of the next level.
[shader("compute")]
[numthreads(64, 1, 1)]
void split(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    TessGpu* t = root.tess;
    uint level = root.level;
    uint i = grid_index(group) * 64 + thread;
    if (i >= min(t.counters[TESS_QUEUE + (level & 1)], t.queue_capacity))
        return;
    PatchItem item = t.queue[level & 1][i];
    Patch pt = load_patch(item.patch);
    Rates r = rates();
    PatternTable* table = root.patterns;
    uint3 q = item_factors(item);
    PatternGpu pattern = table.patterns[pattern_index(factor_counts(q))];
    for (uint k = 0; k < pattern.tri_count; k++)
    {
        uint8_t* tri = table.triangles + 3 * (pattern.tri_first + k);
        PatchItem child = item;
        child.zeros = 0;
        float3 p[3], p0[3];
        for (uint j = 0; j < 3; j++)
        {
            uint key = table.vertices[pattern.vertex_first + uint(tri[j])];
            uint zeros = vertex_zeros(key, item.zeros);
            child.zeros |= zeros << (3 * j);
            uint3 c = place(pattern_weights(key, q), item.corners, zeros);
            set_corner(child.corners, j, c);
            p[j] = corner_point(pt, c, false);
            p0[j] = corner_point(pt, c, true);
        }
        classify(r, child, p, p0, level + 1);
    }
}

// A point of the undisplaced surface.
struct Surface
{
    float3 position;
    float3 normal;
    float2 uv[2];
}

// Where patch weights `w` of a subdivision copy's patch triangle lie on its patch.
float2 patch_st(Patch pt, float3 w)
{
    float2 corners[3] = { triangle_corner(pt.base, 0), triangle_corner(pt.base, 1), triangle_corner(pt.base, 2) };
    return interpolate(corners, w);
}

// The undisplaced surface at patch weights `w`, last frame's with `previous`: the patch
// triangle interpolated, or a subdivision copy's limit surface.
Surface undisplaced(Patch pt, float3 w, bool previous)
{
    ProducerTriangle tri = previous ? pt.previous : pt.current;
    float3 positions[3] = tri.corners.position;
    float3 normals[3] = { tri.corners.attributes[0].normal, tri.corners.attributes[1].normal, tri.corners.attributes[2].normal };
    Surface o;
    o.normal = safe_normalize(interpolate(normals, w), float3(0.0, 1.0, 0.0));
    SubdCopyGpu* c = root.tess.subd;
    if (c == nullptr)
    {
        o.position = interpolate(positions, w);
        for (uint s = 0; s < 2; s++)
        {
            float2 uvs[3] = { tri.corners.attributes[0].uv[s], tri.corners.attributes[1].uv[s], tri.corners.attributes[2].uv[s] };
            o.uv[s] = interpolate(uvs, w);
        }
        return o;
    }
    float2 st = patch_st(pt, w);
    float3 du, dv;
    o.position = subd_limit(c.cage, previous ? c.previous : c.points, pt.base.patch, st, du, dv);
    o.normal = subd_normal(du, dv, o.normal);
    o.uv[0] = subd_limit_uv(c.cage, pt.base.patch, st, 0);
    o.uv[1] = drawn().uv1 != nullptr ? subd_limit_uv(c.cage, pt.base.patch, st, 1) : o.uv[0];
    return o;
}

// What places the entry: last frame's with `previous`, this frame's where its group hid it then.
float3x3 entry_linear(bool previous)
{
    SceneRoot* s = root.scene;
    ProducerGpu p = entry();
    Placement placed;
    if (previous && producer_placement(s, p, true, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), placed))
        return placed.linear;
    if (producer_placement(s, p, false, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), placed))
        return placed.linear;
    return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

// The displaced point at patch weights `w`, last frame's with `previous`, under `placing`.
float3 displaced(Patch pt, PatchItem item, uint material, float3 w, bool previous, float3x3 placing, out Surface base)
{
    ProducerTriangle tri = previous ? pt.previous : pt.current;
    MaterialGpu* m = root.scene.materials + material;
    base = undisplaced(pt, w, previous);
    DisplacementInput i = (DisplacementInput)0;
    i.position = base.position;
    i.displacement = entry().displacement;
    i.normal = base.normal;
    i.material = material;
    i.linear = placing;
    float3 e1 = tri.corners.position[1] - tri.corners.position[0];
    float3 e2 = tri.corners.position[2] - tri.corners.position[0];
    uint set = material_uv_set(m, TEXTURE_DISPLACEMENT);
    float2 first = tri.corners.attributes[0].uv[set];
    float3 along, across;
    uv_directions(e1, e2, tri.corners.attributes[1].uv[set] - first, tri.corners.attributes[2].uv[set] - first, along, across);
    i.tangent = unit_direction(along - base.normal * dot(base.normal, along), float3(0.0));
    i.bitangent = unit_direction(across - base.normal * dot(base.normal, across), float3(0.0));
    for (uint s = 0; s < 2; s++)
    {
        float2 uvs[3] = { tri.corners.attributes[0].uv[s], tri.corners.attributes[1].uv[s], tri.corners.attributes[2].uv[s] };
        float2 inward = (uvs[0] + uvs[1] + uvs[2]) / 3.0 - base.uv[s];
        UvFootprint f = { base.uv[s], (uvs[1] - uvs[0]) / item.rate, (uvs[2] - uvs[0]) / item.rate, float2(-1.0), inward };
        i.uv[s] = f;
    }
    i.user = pt.cluster.user;
    i.user_tag = entry().user_tag;
    i.previous = previous ? 1 : 0;
    i.scene = root.scene;
    i.data = m.data != 0 ? m.data : root.scene.surface_data;
    return base.position + Displace.displace(i);
}

// Whether patch weights `w` lie on a subdivision copy's patch, where its limit surface is
// defined; always for other entries.
bool on_patch(Patch pt, float3 w)
{
    if (root.tess.subd == nullptr)
        return true;
    float2 st = patch_st(pt, w);
    return all(st >= -1e-6) && all(st <= 1.0 + 1e-6);
}

// Twice a step `e` times the displaced field's derivative along it at `w`, where it is `x`:
// central differences, or second-order one-sided ones toward the patch where a step would
// leave it.
float3 difference(Patch pt, PatchItem item, uint material, float3 w, float3 e, float3 x, float3x3 placing)
{
    Surface unused;
    bool plus = on_patch(pt, w + e);
    bool minus = on_patch(pt, w - e);
    if (plus == minus)
        return displaced(pt, item, material, w + e, false, placing, unused) - displaced(pt, item, material, w - e, false, placing, unused);
    float s = plus ? 1.0 : -1.0;
    float3 a = displaced(pt, item, material, w + s * e, false, placing, unused);
    float3 b = displaced(pt, item, material, w + 2.0 * s * e, false, placing, unused);
    return s * (4.0 * a - b - 3.0 * x);
}

groupshared uint g_first;
groupshared uint g_pattern;

// One workgroup per subpatch of the dice list, one thread per pattern vertex.
[shader("compute")]
[numthreads(128, 1, 1)]
void dice(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    TessGpu* t = root.tess;
    uint at = grid_index(group);
    if (at >= min(t.counters[TESS_DICE], t.dice_capacity))
        return;
    PatchItem item = t.dice[at];
    PatternTable* table = root.patterns;
    ProducerBatch d = drawn();
    Patch pt = load_patch(item.patch);
    uint3 q = item_factors(item);
    if (thread == 0)
    {
        g_pattern = pattern_index(factor_counts(q));
        PatternGpu pattern = table.patterns[g_pattern];
        g_first = alloc_vertices(d, pattern.vertex_count);
        if (g_first != NO_ROOM)
        {
            uint c = emit_template_cluster(d, g_pattern, g_first, pt.cluster.material, pt.cluster.user);
            if (c != NO_ROOM)
            {
                GeneratedClusterGpu g;
                g.patch = item.patch;
                g.corners = item.corners;
                pack_factors(q, g.factors);
                g.zeros = item.zeros;
                d.generated[c] = g;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    PatternGpu pattern = table.patterns[g_pattern];
    if (g_first == NO_ROOM || thread >= pattern.vertex_count)
        return;
    uint material = producer_material(entry(), pt.cluster);
    if (material == MATERIAL_NONE)
        material = 0;
    uint key = table.vertices[pattern.vertex_first + thread];
    uint3 wi = place(pattern_weights(key, q), item.corners, vertex_zeros(key, item.zeros));
    float3 w = bary_weights(wi);
    Surface base;
    float3x3 placing = entry_linear(false);
    float3 x = displaced(pt, item, material, w, false, placing, base);
    // Differences along two directions, over a quarter of a generated step and at most a
    // quarter of the finest step one pattern dices: the patch triangle's edges, or a
    // subdivision patch's s and t, which always have a side on the patch.
    float h = 0.25 / max(item.rate, float(MAX_DICE));
    float3 e1 = float3(-h, h, 0.0);
    float3 e2 = float3(-h, 0.0, h);
    if (root.tess.subd != nullptr)
    {
        float2 a = triangle_corner(pt.base, 1) - triangle_corner(pt.base, 0);
        float2 b = triangle_corner(pt.base, 2) - triangle_corner(pt.base, 0);
        float k = 0.5 * h / (a.x * b.y - a.y * b.x);
        float2 s = float2(b.y, -a.y) * k;
        float2 t = float2(-b.x, a.x) * k;
        e1 = float3(-s.x - s.y, s.x, s.y);
        e2 = float3(-t.x - t.y, t.x, t.y);
    }
    float3 du = difference(pt, item, material, w, e1, x, placing);
    float3 dv = difference(pt, item, material, w, e2, x, placing);
    float3 n = base.normal;
    float3 nd = safe_normalize(cross(du, dv), n);
    if (dot(nd, n) < 0.0)
        nd = -nd;
    uint v = g_first + thread;
    d.positions[v] = x;
    d.attrs[v] = producer_attributes(nd, base.uv[0]);
    if (d.uv1 != nullptr)
        d.uv1[v] = base.uv[1];
    if (d.colors != nullptr)
        d.colors[v] = producer_color_word(subd_limit_color(root.tess.subd.cage, pt.base.patch, patch_st(pt, w)));
    if (d.previous != nullptr)
    {
        Surface before;
        d.previous[v] = pt.has_previous ? displaced(pt, item, material, w, true, entry_linear(true), before) : x;
    }
    if (d.vertex_user != nullptr)
    {
        uint corner = wi.x >= wi.y && wi.x >= wi.z ? 0 : wi.y >= wi.z ? 1 : 2;
        d.vertex_user[v] = pt.current.vertex_user[corner];
    }
    // The weld matches points of the patch triangle interpolated, which both sides of a
    // shared edge agree on where a subdivision copy's limit surface comes from two patches.
    bool edge = wi.x == 0 || wi.y == 0 || wi.z == 0;
    float3 corners[3] = pt.current.corners.position;
    t.undisplaced[v] = float4(interpolate(corners, w), asfloat(edge ? weld_tag() : 0));
}

// Empties the weld table; one thread per slot.
[shader("compute")]
[numthreads(64, 1, 1)]
void weld_clear(uint3 id: SV_DispatchThreadID)
{
    TessGpu* t = root.tess;
    if (id.x <= t.weld_mask)
    {
        t.weld[id.x] = 0;
        t.weld_owner[id.x] = 0;
    }
}

// `TessGpu.undisplaced` w of this frame's vertices on a patch triangle's edge: vertices
// left from an earlier frame past a failed allocation never weld.
uint weld_tag()
{
    return (root.scene.frame_index << 1) | 1;
}

uint weld_hash(float3 p)
{
    uint3 b = asuint(p);
    uint h = b.x * 0x8da6b343u ^ b.y * 0xd8163841u ^ b.z * 0xcb1ab31fu;
    return h ^ (h >> 15);
}

uint generated_vertices()
{
    ProducerBatch d = drawn();
    return min(*d.vertex_count, d.max_vertices);
}

// The slot of the undisplaced point of vertex `v`, which `claim` takes when it is free;
// false past WELD_PROBES.
bool weld_slot(uint v, float3 u, bool claim, out uint slot)
{
    TessGpu* t = root.tess;
    slot = weld_hash(u) & t.weld_mask;
    for (uint probe = 0; probe < WELD_PROBES; probe++)
    {
        uint before = t.weld[slot];
        if (before == 0 && claim)
            InterlockedCompareExchange(t.weld[slot], 0, v + 1, before);
        if (before == 0)
            return claim;
        if (all(t.undisplaced[before - 1].xyz == u))
            return true;
        slot = (slot + 1) & t.weld_mask;
    }
    return false;
}

// One thread per generated vertex on a patch triangle's edge: its slot keeps, of the
// vertices at its undisplaced point, the one with the lowest hash of its uv and then the lowest
// index, whose displaced position they all take. Indices follow the order the dice allocated
// vertices in, so where hashes tie the owner can change from frame to frame.
[shader("compute")]
[numthreads(64, 1, 1)]
void weld(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    TessGpu* t = root.tess;
    uint v = grid_index(group) * 64 + thread;
    if (v >= generated_vertices())
        return;
    float4 u = t.undisplaced[v];
    if (asuint(u.w) != weld_tag())
        return;
    uint slot;
    if (!weld_slot(v, u.xyz, true, slot))
    {
        InterlockedAdd(t.counters[TESS_WELD_DROPPED], 1);
        return;
    }
    uint uv = weld_hash(float3(producer_uv(drawn().attrs[v]), 0.0));
    atomic_max_u64(t.weld_owner + slot, (uint64_t(~uv) << 32) | uint64_t(~v));
}

// One thread per generated vertex on a patch triangle's edge: it takes the displaced
// position of its slot's owner.
[shader("compute")]
[numthreads(64, 1, 1)]
void weld_apply(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    TessGpu* t = root.tess;
    uint v = grid_index(group) * 64 + thread;
    if (v >= generated_vertices())
        return;
    float4 u = t.undisplaced[v];
    if (asuint(u.w) != weld_tag())
        return;
    uint slot;
    if (!weld_slot(v, u.xyz, false, slot))
        return;
    uint owner = ~uint(t.weld_owner[slot]);
    if (owner == v)
        return;
    ProducerBatch d = drawn();
    float3 x = d.positions[owner];
    if (any(x != d.positions[v]))
        InterlockedAdd(t.counters[TESS_WELDED], 1);
    d.positions[v] = x;
    if (d.previous != nullptr)
        d.previous[v] = d.previous[owner];
}

// Sizes the weld to the generated vertices.
[shader("compute")]
[numthreads(1, 1, 1)]
void weld_args(uint3 id: SV_DispatchThreadID)
{
    write_grid(root.tess.args + TESS_ARGS_WELD, (generated_vertices() + 63) / 64);
}

// Keeps the frame's figures and the capacity it asked for.
[shader("compute")]
[numthreads(1, 1, 1)]
void finish(uint3 id: SV_DispatchThreadID)
{
    TessGpu* t = root.tess;
    TessStateGpu* st = root.state;
    ProducerBatch d = drawn();
    float clusters = float(*d.cluster_count) / float(max(d.max_clusters, 1));
    float vertices = float(*d.vertex_count) / float(max(d.max_vertices, 1));
    float dice = float(t.counters[TESS_DICE]) / float(max(t.dice_capacity, 1));
    st.usage = max(clusters, max(vertices, dice));
    st.patches = t.counters[TESS_PATCHES];
    st.subpatches = min(t.counters[TESS_DICE], t.dice_capacity);
    st.split_dropped = t.counters[TESS_SPLIT_DROPPED];
    st.dice_dropped = t.counters[TESS_DICE_DROPPED];
    st.weld_dropped = t.counters[TESS_WELD_DROPPED];
    st.unseen = t.counters[TESS_UNSEEN];
    st.welded = t.counters[TESS_WELDED];
}

