// The raster into the visibility buffer, of cooked and producer clusters alike. The
// hardware path draws the opaque and alpha lists through mesh pipelines; `software` rasters the clusters small on screen in compute
// with the same snapping, sample points and coverage. Alpha-tested triangles keep a pixel by
// each surface's coverage. Pixel stages write only the visibility buffer. `depth` copies the
// resolved depth into the depth target.
#include "orr/root.h"
import orr.atomic;
import orr.random;
import orr_pack.mesh;
import producer;
import surface;
import visibility;

extern struct Surface : ISurface = StandardSurface;

ORR_ROOT(RasterRoot);

struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint visible : VISIBLE;
}

struct Primitive
{
    uint triangle : SV_PrimitiveID;
    bool cull : SV_CullPrimitive;
}

groupshared float3 corners[256];

// One cluster's draw: its vertices and triangles `first..first + count`, from a resident
// page or a producer batch.
struct ClusterDraw
{
    VisibleCluster vc;
    bool producer;
    PageHeader h;
    Cluster c;
    float step;          // the mesh's position step
    ProducerBatch b;
    ProducerClusterGpu k;
    Placement p;
    uint vertex_count;
    uint tri_count;
    uint first;
    uint count;
    uint split;          // first alpha-tested triangle
    bool single_sided;
}

// The draw of visible cluster `visible`: its opaque triangles, or with `alpha` its
// alpha-tested ones.
ClusterDraw cluster_draw(uint visible, bool alpha)
{
    SceneRoot* s = root.scene;
    ClusterDraw d = (ClusterDraw)0;
    d.vc = root.raster.gbuffer.clusters[visible];
    MaterialGpu* m = s.materials + d.vc.material;
    uint entry;
    if (visible_producer(s, d.vc, entry))
    {
        d.producer = true;
        d.b = s.batches[entry];
        d.k = d.b.clusters[d.vc.cluster];
        d.vertex_count = d.k.vertex_count;
        d.tri_count = d.k.tri_count;
        bool tested = s.producers[entry].opacity == PRODUCER_ALPHA_TESTED && (m.flags & MATERIAL_ALPHA_TESTED) != 0;
        d.split = tested ? 0 : d.tri_count;
    }
    else
    {
        d.h = load_page_header(d.vc.page);
        d.c = load_cluster(d.vc.page, d.vc.local);
        d.step = s.meshes[s.models[s.instances[d.vc.instance].model].mesh].position_step;
        d.vertex_count = d.c.vertex_count;
        d.tri_count = d.c.tri_count;
        d.split = alpha_start(m, d.c);
    }
    d.first = alpha ? d.split : 0;
    d.count = alpha ? d.tri_count - d.split : d.split;
    d.single_sided = (m.flags & MATERIAL_DOUBLE_SIDED) == 0;
    visible_placement(s, root.view, d.vc, false, d.p);
    return d;
}

// The corners of triangle `t` of the draw.
uint3 draw_triangle(ClusterDraw d, uint t)
{
    return d.producer ? cluster_corners(d.k, t) : load_triangle(d.vc.page, d.h, d.c, t);
}

// Vertex `i` of the draw; keeps its eye-relative position for the back-face test.
Vertex cluster_vertex(ClusterDraw d, uint visible, uint i)
{
    ViewConstants* v = root.view;
    float3 x = d.producer ? d.b.positions[d.k.first_vertex + i] : load_position(d.vc.page, d.h, d.c, i, d.step);
    float3 q = place(d.p, x);
    corners[i] = q;
    Vertex o;
    o.position = mul(v.clip_from_view_jittered, float4(view_from_relative(v, q), 1.0));
    o.visible = visible;
    return o;
}

// Whether a single-sided material shows the back of the triangle with corners `k`.
bool back_facing(ClusterDraw d, uint3 k)
{
    float3 q0 = corners[k.x];
    float3 n = cross(corners[k.y] - q0, corners[k.z] - q0) * sign(determinant(d.p.linear));
    return d.single_sided && dot(n, q0) > 0.0;
}

// Triangle `first + t` of the draw, culled when it shows its back.
Primitive cluster_primitive(ClusterDraw d, uint t, out uint3 corners_out)
{
    uint triangle = d.first + t;
    uint3 k = draw_triangle(d, triangle);
    Primitive o;
    o.triangle = triangle;
    o.cull = back_facing(d, k);
    corners_out = k;
    return o;
}

// One workgroup per entry of the opaque list, or with root value 1 the alpha list, past the
// entries an earlier pass drew, drawing the cluster's opaque or alpha-tested triangles.
[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void cluster_mesh(
    uint3 group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[128],
    out vertices Vertex vertices[256],
    out primitives Primitive primitives[128])
{
    RasterGpu* r = root.raster;
    bool alpha = root.value == 1;
    uint index = grid_index(group) + r.counts[alpha ? COUNT_ALPHA_BASE : COUNT_OPAQUE_BASE];
    if (index >= r.counts[alpha ? COUNT_ALPHA : COUNT_OPAQUE])
    {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint visible = alpha ? r.alpha[index] : r.opaque[index];
    ClusterDraw d = cluster_draw(visible, alpha);
    SetMeshOutputCounts(d.vertex_count, d.count);
    for (uint i = thread; i < d.vertex_count; i += 128)
        vertices[i] = cluster_vertex(d, visible, i);
    GroupMemoryBarrierWithGroupSync();
    if (thread < d.count)
    {
        uint3 k;
        primitives[thread] = cluster_primitive(d, thread, k);
        triangles[thread] = k;
    }
}

void write_pixel(uint2 pixel, float depth, uint visible, uint triangle)
{
    RasterGpu* r = root.raster;
    atomic_max_u64(r.gbuffer.visibility + pixel.y * r.gbuffer.extent.x + pixel.x, visibility_word(depth, visible, triangle));
}

void write_visibility(float4 position, uint visible, uint triangle)
{
    write_pixel(uint2(position.xy), position.z, visible, triangle);
}

[shader("pixel")]
void opaque_pixel(Vertex v, uint triangle: SV_PrimitiveID)
{
    write_visibility(v.position, v.visible, triangle);
}

// Whether an alpha-tested triangle keeps the pixel at `pixel_center`: with probability
// coverage x (1 - fade), a copy's fade from its slot state.
bool alpha_keeps(VisibleCluster vc, uint triangle, float2 pixel_center)
{
    SceneRoot* s = root.scene;
    PrimaryHit hit = primary_hit(s, root.view, vc, triangle, pixel_center);
    uint entry;
    float fade = visible_producer(s, vc, entry) ? 0.0 : float(s.slot_state[vc.instance].fade) / 255.0;
    float keep = Surface.coverage(hit.input) * (1.0 - fade);
    float noise = unit_float(pcg3d(uint3(uint2(pixel_center), s.frame_index)).x);
    return keep > noise;
}

[shader("pixel")]
void alpha_pixel(Vertex v, uint triangle: SV_PrimitiveID)
{
    if (alpha_keeps(root.raster.gbuffer.clusters[v.visible], triangle, v.position.xy))
        write_visibility(v.position, v.visible, triangle);
}

// The software raster snaps corners to 1/256 pixel and samples pixel centers, as the
// hardware does; ties on an edge go to top and left edges.
static const int SUBPIXEL_BITS = 8;
static const int SUBPIXEL = 1 << SUBPIXEL_BITS;
// Triangles whose bounding box spans at most this many pixels in x run one per lane over
// the whole box; wider ones hand their rows out across the wave.
static const int MICROPOLY_SPAN = 4;

// A triangle set up for the software raster. Corners are in 1/256 pixel from `origin`,
// wound so that the edge functions are positive inside.
struct SoftTriangle
{
    int2 v[3];
    int2 origin;
    int bias[3];     // per edge opposite each corner: 0 on top and left edges, else -1
    float3 z;        // reverse-Z depth per corner
    float inv_area;
    int4 rect;       // pixels covered by the box: x0, y0, x1, y1, inclusive
    uint triangle;
    bool alpha;
}

groupshared int2 soft_xy[256];
groupshared float soft_z[256];
groupshared SoftTriangle soft[128];

// Twice the signed area of (a, b, p); positive with p left of a -> b in y-down pixels.
int edge(int2 a, int2 b, int2 p)
{
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

bool top_left(int2 a, int2 b)
{
    return b.y < a.y || (b.y == a.y && b.x > a.x);
}

bool micropoly(SoftTriangle t)
{
    return t.rect.z - t.rect.x < MICROPOLY_SPAN;
}

// Sets up triangle `triangle` of the draw; false when it covers no pixel center or shows
// its back.
bool soft_setup(ClusterDraw d, uint triangle, uint2 extent, out SoftTriangle t)
{
    t = (SoftTriangle)0;
    uint3 k = draw_triangle(d, triangle);
    if (back_facing(d, k))
        return false;
    int2 v0 = soft_xy[k.x];
    int2 v1 = soft_xy[k.y];
    int2 v2 = soft_xy[k.z];
    float3 z = float3(soft_z[k.x], soft_z[k.y], soft_z[k.z]);
    int2 lo = min(v0, min(v1, v2));
    int2 hi = max(v0, max(v1, v2));
    v0 -= lo;
    v1 -= lo;
    v2 -= lo;
    int area = edge(v0, v1, v2);
    if (area == 0)
        return false;
    if (area < 0)
    {
        int2 w = v1;
        v1 = v2;
        v2 = w;
        z = z.xzy;
        area = -area;
    }
    int2 first = max((lo + SUBPIXEL / 2 - 1) >> SUBPIXEL_BITS, int2(0));
    int2 last = min((hi - SUBPIXEL / 2) >> SUBPIXEL_BITS, int2(extent) - 1);
    t.rect = int4(first, last);
    if (any(t.rect.xy > t.rect.zw))
        return false;
    t.v[0] = v0;
    t.v[1] = v1;
    t.v[2] = v2;
    t.origin = lo;
    t.bias[0] = top_left(v1, v2) ? 0 : -1;
    t.bias[1] = top_left(v2, v0) ? 0 : -1;
    t.bias[2] = top_left(v0, v1) ? 0 : -1;
    t.z = z;
    t.inv_area = 1.0 / float(area);
    t.triangle = triangle;
    t.alpha = triangle >= d.split;
    return true;
}

// Writes pixel (x, y) when its center lies inside `t` and, for alpha-tested triangles, the
// surface's coverage keeps it.
void soft_pixel(SoftTriangle t, VisibleCluster vc, uint visible, int x, int y)
{
    int2 p = int2(x, y) * SUBPIXEL + SUBPIXEL / 2 - t.origin;
    int e0 = edge(t.v[1], t.v[2], p);
    int e1 = edge(t.v[2], t.v[0], p);
    int e2 = edge(t.v[0], t.v[1], p);
    if (e0 + t.bias[0] < 0 || e1 + t.bias[1] < 0 || e2 + t.bias[2] < 0)
        return;
    float depth = dot(float3(e0, e1, e2), t.z) * t.inv_area;
    if (t.alpha && !alpha_keeps(vc, t.triangle, float2(x, y) + 0.5))
        return;
    write_pixel(uint2(x, y), depth, visible, t.triangle);
}

// The pixels of row `y` whose centers can lie inside `t`, from its edge functions.
int2 soft_span(SoftTriangle t, int y)
{
    int2 span = t.rect.xz;
    int2 p = int2(t.rect.x, y) * SUBPIXEL + SUBPIXEL / 2 - t.origin;
    for (uint i = 0; i < 3; i++)
    {
        int2 a = t.v[(i + 1) % 3];
        int2 b = t.v[(i + 2) % 3];
        int base = edge(a, b, p) + t.bias[i];
        int step = (a.y - b.y) * SUBPIXEL;
        if (step == 0)
        {
            if (base < 0)
                span = int2(1, 0);
        }
        else if (step > 0)
        {
            if (base < 0)
                span.x = max(span.x, t.rect.x + (step - 1 - base) / step);
        }
        else
        {
            span.y = base < 0 ? span.x - 1 : min(span.y, t.rect.x + base / -step);
        }
    }
    return span;
}

// One workgroup per software cluster: corners to snapped pixels, one triangle set up per
// thread, then each wave rasters its triangles with the rows of wide ones spread over its
// lanes.
[shader("compute")]
[numthreads(128, 1, 1)]
void software(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    RasterGpu* r = root.raster;
    ViewConstants* v = root.view;
    uint index = grid_index(group) + r.counts[COUNT_SOFTWARE_BASE];
    if (index >= r.counts[COUNT_SOFTWARE])
        return;
    uint visible = r.software[index];
    ClusterDraw d = cluster_draw(visible, false);
    for (uint i = thread; i < d.vertex_count; i += 128)
    {
        float4 clip = cluster_vertex(d, visible, i).position;
        float2 xy = pixel_from_clip(v, clip);
        soft_xy[i] = int2(floor(xy * float(SUBPIXEL) + 0.5));
        soft_z[i] = clip.z / clip.w;
    }
    GroupMemoryBarrierWithGroupSync();

    SoftTriangle t = (SoftTriangle)0;
    uint units = 0;
    if (thread < d.tri_count && soft_setup(d, thread, v.extent, t))
        units = micropoly(t) ? 1 : uint(t.rect.w - t.rect.y + 1);
    soft[thread] = t;
    GroupMemoryBarrierWithGroupSync();

    uint lanes = WaveGetLaneCount();
    uint start = WavePrefixSum(units);
    uint total = WaveActiveSum(units);
    VisibleCluster vc = d.vc;
    for (uint base = 0; base < total; base += lanes)
    {
        uint u = base + WaveGetLaneIndex();
        // The first lane whose units reach past u.
        uint lane = 0;
        for (uint step = lanes >> 1; step > 0; step >>= 1)
        {
            uint end = WaveReadLaneAt(start + units, lane + step - 1);
            if (end <= u)
                lane += step;
        }
        uint source = WaveReadLaneAt(thread, lane);
        uint first = WaveReadLaneAt(start, lane);
        if (u >= total)
            continue;
        SoftTriangle s = soft[source];
        if (micropoly(s))
        {
            for (int y = s.rect.y; y <= s.rect.w; y++)
                for (int x = s.rect.x; x <= s.rect.z; x++)
                    soft_pixel(s, vc, visible, x, y);
        }
        else
        {
            int y = s.rect.y + int(u - first);
            int2 span = soft_span(s, y);
            for (int x = span.x; x <= span.y; x++)
                soft_pixel(s, vc, visible, x, y);
        }
    }
}

struct DepthVertex
{
    float4 position : SV_Position;
}

// One triangle over the whole target.
[shader("mesh")]
[numthreads(3, 1, 1)]
[outputtopology("triangle")]
void fullscreen(uint thread: SV_GroupThreadID, out indices uint3 triangles[1], out vertices DepthVertex vertices[3])
{
    SetMeshOutputCounts(3, 1);
    float2 uv = float2((thread << 1) & 2, thread & 2);
    DepthVertex o;
    o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    vertices[thread] = o;
    if (thread == 0)
        triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
float depth(DepthVertex v) : SV_Depth
{
    RasterGpu* r = root.raster;
    uint2 pixel = uint2(v.position.xy);
    return r.depth[pixel.y * r.gbuffer.extent.x + pixel.x];
}
