// The tessellator: dices the patch triangles a tessellated entry's kernel emits into
// `SceneRoot.patches` into generated clusters in the entry's batch, once per frame for
// raster and ray tracing alike.
//
// `begin` resets the entry's counts and steps its rate; `patches` classifies each patch
// triangle; `args` sizes the next pass; each of SPLITS `split` passes cuts the subpatches
// whose edges need more than MAX_DICE segments by the pattern of their split factors;
// `dice` writes one generated cluster per subpatch, its vertices displaced through the
// linked `IDisplacement` with normals from finite differences of the displaced field;
// `weld_clear`, `weld` and `weld_apply` give vertices on patch triangle edges that share an
// undisplaced point one displaced position, which closes uv seams; `finish` keeps the
// figures. An edge's factor depends only on its endpoints, so patches on both sides of it
// cut it alike: its undisplaced world length in pixels at the scene focus over the dice rate,
// times the entry's rate scale, over 15 where last frame's first view of the scene saw
// neither the edge in its frustum nor in front of its HZB.
#include "orr/root.h"
import orr.abi;
import orr.math;
import orr.packing;
import producer;
import rtas;
import tables;
import tessellation;
import visibility;
__exported import displacement;

extern struct Displace : IDisplacement = TextureDisplacement;

// Split passes: patch triangles take up to MAX_DICE^(SPLITS + 1) segments per edge.
static const uint SPLITS = 3;
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

uint3 load_factors(uint packed)
{
    uint3 f = uint3(packed & 0xff, (packed >> 8) & 0xff, (packed >> 16) & 0xff);
    return clamp(f, 1, MAX_DICE);
}

uint pack_factors(uint3 f)
{
    return f.x | (f.y << 8) | (f.z << 16);
}

// One patch triangle: entry-space corners, attributes and words.
struct Patch
{
    ProducerClusterGpu cluster;
    ProducerTriangle current;
    ProducerTriangle previous;
    bool has_previous;
}

Patch load_patch(uint code)
{
    ProducerBatch b = patches();
    Patch p;
    p.cluster = b.clusters[code >> 7];
    p.current = producer_triangle(b, p.cluster, code & 127, false);
    p.has_previous = b.previous != nullptr;
    p.previous = producer_triangle(b, p.cluster, code & 127, p.has_previous);
    return p;
}

// What edge factors read: the entry placed at the focus and at last frame's view.
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
        ViewConstants* v = root.view;
        r.seen_test = producer_placement(s, p, false, v.eye_cell.xyz, v.eye_offset.xyz, uint(v.eye_cell.w), r.view);
    }
    return r;
}

// Whether last frame's view saw the sphere of `radius` m (entry space) around `center`.
bool seen(Rates r, float3 center, float radius)
{
    if (!r.seen_test)
        return true;
    ViewConstants* v = root.view;
    float3 c = place(r.view, center);
    float rad = radius * r.view.scale;
    if (!sphere_visible(v, c, rad))
        return false;
    if (root.raster == nullptr)
        return true;
    Placement here = (Placement)0;
    here.linear = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    here.scale = 1.0;
    return !box_occluded(v, root.raster, here, c - rad, c + rad, false);
}

// Segments edge a-b asks for, before rounding: the same from either end.
float edge_factor(Rates r, float3 a, float3 b)
{
    if ((root.flags & TESS_FOCUS) == 0)
        return 1.0;
    float len = length(mul(r.focus.linear, a - b));
    float d = max(min(length(place(r.focus, a)), length(place(r.focus, b))), FOCUS_NEAR);
    float t = len / d * root.focus_offset.w / root.dice_pixels * r.scale;
    float3 mid = 0.5 * (a + b);
    if (!seen(r, mid, 0.5 * length(a - b) + r.reach))
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

// Sends the subpatch with corners at `p` to the split queue of pass `level` or the dice
// list; `item.rate` is set at level 0.
void classify(Rates r, inout PatchItem item, float3 p[3], uint level)
{
    float3 t = float3(edge_factor(r, p[0], p[1]), edge_factor(r, p[1], p[2]), edge_factor(r, p[2], p[0]));
    // At least one segment, NaN included.
    float3 c = min(ceil(t), 65535.0);
    float3 n = select(c >= 1.0, c, 1.0);
    if (level == 0)
        item.rate = max(n.x, max(n.y, n.z));
    TessGpu* g = root.tess;
    if (max(n.x, max(n.y, n.z)) > float(MAX_DICE) && level < SPLITS)
    {
        uint3 split = uint3(min(ceil(n / float(MAX_DICE)), float(MAX_DICE)));
        uint at;
        InterlockedAdd(g.counters[TESS_QUEUE + (level & 1)], 1, at);
        if (at < g.queue_capacity)
        {
            item.factors = pack_factors(split);
            g.queue[level & 1][at] = item;
            return;
        }
        InterlockedAdd(g.counters[TESS_SPLIT_DROPPED], 1);
    }
    item.factors = pack_factors(uint3(min(n, float(MAX_DICE))));
    push_dice(item);
}

float3 corner_point(Patch pt, uint corner)
{
    float3 c[3] = pt.current.corners.position;
    return interpolate(c, bary_weights(unpack_bary(corner)));
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
    float3 p[3];
    cluster_triangle(b, k, thread, p[0], p[1], p[2]);
    PatchItem item;
    item.patch = (c << 7) | thread;
    item.corners[0] = pack_bary(uint3(BARY_ONE, 0, 0));
    item.corners[1] = pack_bary(uint3(0, BARY_ONE, 0));
    item.corners[2] = pack_bary(uint3(0, 0, BARY_ONE));
    item.rate = 1.0;
    classify(rates(), item, p, 0);
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
    PatternGpu pattern = table.patterns[pattern_index(load_factors(item.factors))];
    for (uint k = 0; k < pattern.tri_count; k++)
    {
        uint8_t* tri = table.triangles + 3 * (pattern.tri_first + k);
        PatchItem child = item;
        float3 p[3];
        for (uint j = 0; j < 3; j++)
        {
            uint3 w = unpack_bary(table.vertices[pattern.vertex_first + uint(tri[j])]);
            child.corners[j] = pack_bary(compose(w, item.corners));
            p[j] = corner_point(pt, child.corners[j]);
        }
        classify(r, child, p, level + 1);
    }
}

// The displaced point at patch weights `w`, last frame's with `previous`.
float3 displaced(Patch pt, PatchItem item, uint material, float3 w, bool previous, out float3 base)
{
    ProducerTriangle tri = previous ? pt.previous : pt.current;
    float3 positions[3] = tri.corners.position;
    float3 normals[3] = { tri.corners.attributes[0].normal, tri.corners.attributes[1].normal, tri.corners.attributes[2].normal };
    float2 uvs[3] = { tri.corners.attributes[0].uv, tri.corners.attributes[1].uv, tri.corners.attributes[2].uv };
    base = interpolate(positions, w);
    DisplacementInput i = (DisplacementInput)0;
    i.position = base;
    i.displacement = entry().displacement;
    i.normal = safe_normalize(interpolate(normals, w), float3(0.0, 1.0, 0.0));
    i.material = material;
    i.uv.uv = interpolate(uvs, w);
    i.uv.dx = (uvs[1] - uvs[0]) / item.rate;
    i.uv.dy = (uvs[2] - uvs[0]) / item.rate;
    i.uv.tap = float2(-1.0);
    i.user = pt.cluster.user;
    i.user_tag = entry().user_tag;
    i.previous = previous ? 1 : 0;
    i.scene = root.scene;
    MaterialGpu* m = root.scene.materials + material;
    i.data = m.data != 0 ? m.data : root.scene.surface_data;
    return base + Displace.displace(i);
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
    if (thread == 0)
    {
        g_pattern = pattern_index(load_factors(item.factors));
        PatternGpu pattern = table.patterns[g_pattern];
        g_first = alloc_vertices(d, pattern.vertex_count);
        if (g_first != NO_ROOM)
        {
            uint c = emit_template_cluster(d, g_pattern, g_first, pt.cluster.material, pt.cluster.user);
            if (c != NO_ROOM)
            {
                GeneratedClusterGpu g;
                g.patch = item.patch;
                g.pattern = g_pattern;
                g.corners = item.corners;
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
    uint3 wi = compose(unpack_bary(table.vertices[pattern.vertex_first + thread]), item.corners);
    float3 w = bary_weights(wi);
    float3 base;
    float3 x = displaced(pt, item, material, w, false, base);
    // Central differences along the two patch directions, over a quarter of a generated
    // step and at most a quarter of the finest step one pattern dices.
    float h = 0.25 / max(item.rate, float(MAX_DICE));
    float3 e1 = float3(-h, h, 0.0);
    float3 e2 = float3(-h, 0.0, h);
    float3 unused;
    float3 du = displaced(pt, item, material, w + e1, false, unused) - displaced(pt, item, material, w - e1, false, unused);
    float3 dv = displaced(pt, item, material, w + e2, false, unused) - displaced(pt, item, material, w - e2, false, unused);
    float3 normals[3] = { pt.current.corners.attributes[0].normal, pt.current.corners.attributes[1].normal, pt.current.corners.attributes[2].normal };
    float3 n = safe_normalize(interpolate(normals, w), float3(0.0, 1.0, 0.0));
    float3 nd = safe_normalize(cross(du, dv), n);
    if (dot(nd, n) < 0.0)
        nd = -nd;
    float2 uvs[3] = { pt.current.corners.attributes[0].uv, pt.current.corners.attributes[1].uv, pt.current.corners.attributes[2].uv };
    uint v = g_first + thread;
    d.positions[v] = x;
    d.attrs[v] = producer_attributes(nd, interpolate(uvs, w));
    if (d.previous != nullptr)
        d.previous[v] = pt.has_previous ? displaced(pt, item, material, w, true, unused) : x;
    if (d.vertex_user != nullptr)
    {
        uint corner = wi.x >= wi.y && wi.x >= wi.z ? 0 : wi.y >= wi.z ? 1 : 2;
        d.vertex_user[v] = pt.current.vertex_user[corner];
    }
    bool edge = wi.x == 0 || wi.y == 0 || wi.z == 0;
    t.undisplaced[v] = float4(base, asfloat(edge ? weld_tag() : 0));
}

// Empties the weld table; one thread per slot.
[shader("compute")]
[numthreads(64, 1, 1)]
void weld_clear(uint3 id: SV_DispatchThreadID)
{
    TessGpu* t = root.tess;
    if (id.x <= t.weld_mask)
        t.weld[id.x] = 0;
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

// One thread per generated vertex on a patch triangle's edge: its slot keeps the lowest
// vertex at its undisplaced point.
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
    uint slot = weld_hash(u.xyz) & t.weld_mask;
    for (uint probe = 0; probe < WELD_PROBES; probe++)
    {
        uint before;
        InterlockedCompareExchange(t.weld[slot], 0, v + 1, before);
        if (before == 0)
            return;
        if (all(t.undisplaced[before - 1].xyz == u.xyz))
        {
            InterlockedMin(t.weld[slot], v + 1);
            return;
        }
        slot = (slot + 1) & t.weld_mask;
    }
    InterlockedAdd(t.counters[TESS_WELD_DROPPED], 1);
}

// One thread per generated vertex on a patch triangle's edge: it takes the displaced
// position of its slot's vertex.
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
    uint slot = weld_hash(u.xyz) & t.weld_mask;
    for (uint probe = 0; probe < WELD_PROBES; probe++)
    {
        uint owner = t.weld[slot];
        if (owner == 0)
            return;
        if (all(t.undisplaced[owner - 1].xyz == u.xyz))
        {
            if (owner - 1 != v)
            {
                ProducerBatch d = drawn();
                float3 x = d.positions[owner - 1];
                if (any(x != d.positions[v]))
                    InterlockedAdd(t.counters[TESS_WELDED], 1);
                d.positions[v] = x;
                if (d.previous != nullptr)
                    d.previous[v] = d.previous[owner - 1];
            }
            return;
        }
        slot = (slot + 1) & t.weld_mask;
    }
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

