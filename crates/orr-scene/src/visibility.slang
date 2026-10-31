// One view's raster state, shared by the culling, raster, resolve and HZB kernels, and the
// geometry helpers they share.
module visibility;

import orr.abi;
import orr.math;
import orr.packing;
import orr_pack.mesh;
import producer;
__exported import hit;

// `RasterGpu.counts` entries.
public static const uint COUNT_VISIBLE = 0;          // visible clusters written
public static const uint COUNT_VISIBLE_DROPPED = 1;  // visible clusters past capacity
public static const uint COUNT_WORK_DROPPED = 2;     // second-pass work past capacity
public static const uint COUNT_DRAWS = 3;            // 1: the draw count of the indirect draws
public static const uint COUNT_WORK = 4;             // second-pass work items asked for
public static const uint COUNT_OPAQUE = 5;           // hardware opaque draws
public static const uint COUNT_ALPHA = 6;            // hardware alpha draws
public static const uint COUNT_SOFTWARE = 7;         // clusters the software raster draws
public static const uint COUNT_PARTICLES = 8;        // visible particle clusters
public static const uint COUNT_OCCLUDED_INSTANCES = 9; // instances the first pass found occluded
public static const uint COUNT_OCCLUDED_CLUSTERS = 10; // clusters the first pass found occluded
public static const uint COUNT_SECOND_PASS = 11;     // clusters drawn by the second pass
public static const uint COUNT_HIDDEN = 12;          // clusters both passes found occluded
public static const uint COUNT_OPAQUE_BASE = 13;     // list entries the first pass drew
public static const uint COUNT_ALPHA_BASE = 14;
public static const uint COUNT_SOFTWARE_BASE = 15;
public static const uint COUNTS = 16;

// `RasterGpu.flags`.
public static const uint RASTER_OCCLUSION = 1; // the first pass tests the previous frame's HZB

// `RasterGpu.slot_visibility`, per instance slot.
public static const uint SLOT_CULLED = 0;
public static const uint SLOT_VISIBLE = 1;    // the first pass draws its clusters
public static const uint SLOT_OCCLUDED = 2;   // the first pass found it occluded
public static const uint SLOT_REVEALED = 3;   // the second pass draws its clusters

// The software raster's largest screen rect, in pixels: its edge functions stay in 32 bits
// up to here.
public static const float SOFTWARE_MAX_PIXELS = 128.0;

public struct RasterGpu
{
    public GBufferGpu gbuffer;       // what later passes read
    public ClusterWork* work;        // what the second pass retests
    public uint* args;               // workgroups x, y, z: work, opaque, alpha, software, particles
    public uint* opaque;             // visible clusters with opaque triangles, hardware path
    public uint* alpha;              // visible clusters with alpha-tested triangles, hardware path
    public uint* software;           // visible clusters of the software raster
    public uint* particles;          // visible clusters of particle entries, for the transparency point
    public uint* counts;             // COUNT_*
    public uint* slot_visibility;    // per instance slot: SLOT_*
    public uint* bin_args;           // per surface bin: x, y, z
    public uint* bin_tiles;          // per bin: tile_capacity tile indices
    public float* depth;             // per pixel: resolved reverse-Z depth
    public StorageImageIndex a;
    public StorageImageIndex b;
    public StorageImageIndex x;      // index TEXTURE_NONE without the extended target
    public StorageImageIndex motion;
    public StorageImageIndex hzb[16]; // per mip
    public uint tiles_x;             // 8 x 8 tiles per row
    public uint tile_capacity;
    public uint work_capacity;
    public uint visible_capacity;
    public uint bins;                // surface bins: linked surfaces + 1
    public float software_pixels;    // longest screen-rect side the software raster takes; 0 off
    public uint flags;               // RASTER_*
    public uint slots;               // instance slots to walk
    public uint hzb_mips;            // HZB mips the occlusion tests read
    public uint _pad;
}

// `RasterGpu.args` offsets.
public static const uint ARGS_WORK = 0;
public static const uint ARGS_OPAQUE = 3;
public static const uint ARGS_ALPHA = 6;
public static const uint ARGS_SOFTWARE = 9;
public static const uint ARGS_PARTICLES = 12;
public static const uint ARGS = 15;

// Workgroup grids cover `n` groups in rows of GRID_X, the smallest maximum Vulkan allows.
public static const uint GRID_X = 65535;

public void write_grid(uint* args, uint n)
{
    args[0] = min(n, GRID_X);
    args[1] = (n + GRID_X - 1) / GRID_X;
    args[2] = 1;
}

public uint grid_index(uint3 group)
{
    return group.y * GRID_X + group.x;
}

// What every raster dispatch and draw takes.
public struct RasterRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public RasterGpu* raster;
    public uint value;               // the dispatch's bin or HZB mip
    public uint _pad;
}

// The transform of `index` in `group` relative to the eye, from `transforms` (the group's
// current or previous buffer). False when the group hides the instance.
public bool placement(GroupGpu* g, uint8_t* transforms, uint index, ViewConstants* v, out Placement p)
{
    return placement(g, transforms, index, v.eye_cell.xyz, v.eye_offset.xyz, uint(v.eye_cell.w), p);
}

// The instance's current placement.
public bool instance_placement(SceneRoot* s, InstanceGpu inst, ViewConstants* v, out Placement p)
{
    GroupGpu* g = s.groups + inst.group;
    return placement(g, g.current, inst.group_index, v, p);
}

// A view-space point from an eye-relative render-space one.
public float3 view_from_relative(ViewConstants* v, float3 x)
{
    return mul(v.view_from_world_rotation, float4(x, 0.0));
}

// Render pixels per radian at the image center.
public float pixels_per_radian(ViewConstants* v)
{
    return 0.5 * float(v.extent.y) * abs(v.clip_from_view[1][1]);
}

// `error` (m) at the nearest point of the sphere at eye-relative `center` of `radius` m,
// in render pixels.
public float projected_error(ViewConstants* v, float3 center, float radius, float error)
{
    float near = v.clip_from_view[2][3];
    float d = max(length(center) - radius, near);
    return error / d * pixels_per_radian(v);
}

// False when the sphere at eye-relative `center` lies wholly outside the view frustum.
public bool sphere_visible(ViewConstants* v, float3 center, float radius)
{
    float3 c = view_from_relative(v, center);
    float4x4 m = v.clip_from_view_jittered;
    float4 p = float4(c, 1.0);
    float4 planes[5] = { m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] - m[2] };
    for (uint i = 0; i < 5; i++)
    {
        float l = length(planes[i].xyz);
        if (dot(planes[i], p) < -radius * l)
            return false;
    }
    return true;
}

// Pixel coordinates of clip-space `clip`, as the viewport places them.
public float2 pixel_from_clip(ViewConstants* v, float4 clip)
{
    return (clip.xy / clip.w * 0.5 + 0.5) * float2(v.extent);
}

// True when the box `lo..hi`, placed by `p`, lies in front of the near plane and its screen
// rect's longer side is at most `pixels`.
public bool small_on_screen(ViewConstants* v, Placement p, float3 lo, float3 hi, float pixels)
{
    float2 rect_min = float2(1e30);
    float2 rect_max = float2(-1e30);
    for (uint i = 0; i < 8; i++)
    {
        float3 corner = float3((i & 1) != 0 ? hi.x : lo.x, (i & 2) != 0 ? hi.y : lo.y, (i & 4) != 0 ? hi.z : lo.z);
        float4 clip = mul(v.clip_from_view_jittered, float4(view_from_relative(v, place(p, corner)), 1.0));
        if (!(clip.w > 0.0 && clip.z < clip.w))
            return false;
        float2 xy = pixel_from_clip(v, clip);
        rect_min = min(rect_min, xy);
        rect_max = max(rect_max, xy);
    }
    float2 size = rect_max - rect_min;
    return max(size.x, size.y) <= min(pixels, SOFTWARE_MAX_PIXELS);
}

// Whether the box `lo..hi`, placed by `p` relative to this frame's eye, lies behind the
// HZB's farthest depth over its screen rect: through this view, or with `previous` through
// the previous frame's camera (the HZB still holding the previous frame). False when the box
// reaches the near plane.
public bool box_occluded(ViewConstants* v, RasterGpu* r, Placement p, float3 lo, float3 hi, bool previous)
{
    float2 rect_min = float2(1e30);
    float2 rect_max = float2(-1e30);
    float nearest = 0.0;
    for (uint i = 0; i < 8; i++)
    {
        float3 corner = float3((i & 1) != 0 ? hi.x : lo.x, (i & 2) != 0 ? hi.y : lo.y, (i & 4) != 0 ? hi.z : lo.z);
        float3 x = place(p, corner);
        float3 view = previous ? mul(v.prev_view_from_world_rotation, float4(x + v.eye_motion.xyz, 0.0)) : view_from_relative(v, x);
        float4 clip = mul(previous ? v.prev_clip_from_view : v.clip_from_view_jittered, float4(view, 1.0));
        if (!(clip.w > 0.0 && clip.z < clip.w))
            return false;
        float2 xy = pixel_from_clip(v, clip);
        rect_min = min(rect_min, xy);
        rect_max = max(rect_max, xy);
        nearest = max(nearest, clip.z / clip.w);
    }
    // A pixel more on each side covers the jitter between the frames.
    float2 extent = float2(r.gbuffer.extent);
    rect_min = max(rect_min - 1.0, 0.0);
    rect_max = min(rect_max + 1.0, extent - 1.0);
    if (any(rect_min > rect_max))
        return false;
    // A mip whose texels are half the rect's longer side: at most 3 x 3 texels cover it.
    float2 size = rect_max - rect_min;
    uint mip = min(uint(max(ceil(log2(max(max(size.x, size.y), 1.0))) - 1.0, 0.0)), r.hzb_mips - 1);
    uint2 top = max(r.gbuffer.extent >> mip, 1) - 1;
    uint2 a = min(uint2(rect_min) >> mip, top);
    uint2 b = min(uint2(rect_max) >> mip, top);
    RWTexture2D<float> hzb = r.hzb[mip].get<RWTexture2D<float>>();
    float farthest = 1.0;
    for (uint y = a.y; y <= b.y; y++)
        for (uint x = a.x; x <= b.x; x++)
            farthest = min(farthest, hzb[uint2(x, y)]);
    return nearest < farthest;
}

// The triangles of a cluster the alpha bin draws: from the returned index to `tri_count`.
// Linked surfaces with an alpha test run their coverage on every triangle.
public uint alpha_start(MaterialGpu* m, Cluster c)
{
    if ((m.flags & MATERIAL_ALPHA_TESTED) == 0)
        return c.tri_count;
    return m.surface != 0 ? 0 : c.alpha_first;
}

// Pixel `t` of an 8 x 8 tile whose 64 threads run in 2 x 2 quads: threads 4q to 4q + 3 are
// one quad, so a thread's x and y neighbors are threads t ^ 1 and t ^ 2.
public uint2 tile_pixel(uint tile, uint tiles_x, uint t)
{
    uint q = t >> 2;
    uint2 in_tile = uint2((q & 3) * 2 + (t & 1), (q >> 2) * 2 + ((t >> 1) & 1));
    return uint2(tile % tiles_x, tile / tiles_x) * 8 + in_tile;
}

// A render-space direction from a view-space one.
public float3 render_direction(ViewConstants* v, float3 d)
{
    float3x4 r = v.view_from_world_rotation;
    return mul(transpose(float3x3(r[0].xyz, r[1].xyz, r[2].xyz)), d);
}

// Where the ray along `d` from the eye meets the plane of the triangle with eye-relative
// corners `q`: the weights of corners 1 and 2 in xy, the distance along `d` in z.
float3 ray_barycentrics(float3 d, float3 q0, float3 e1, float3 e2)
{
    float3 pv = cross(d, e2);
    float det = dot(e1, pv);
    float inv = abs(det) > 1e-30 ? 1.0 / det : 0.0;
    float3 tv = -q0;
    float3 qv = cross(tv, e1);
    return float3(dot(tv, pv), dot(d, qv), dot(e2, qv)) * inv;
}

// The pixel ray's hit on a visible triangle, and the surface input at it.
public struct PrimaryHit
{
    public SurfaceInput input;
    public float3 relative;    // eye-relative render-space position
    public float3 direction;   // unit pixel ray, render space
    public float distance;     // m along the ray
    public Placement placement;
    public float3 object;      // the hit in mesh or entry space
    public float error;        // m: the cluster's LOD error times the instance scale; 0 on producer clusters
    public uint origin;        // a producer entry's origin policy, else ORIGIN_DEFAULT
}

// The producer entry a visible cluster names; false for cooked clusters.
public bool visible_producer(SceneRoot* s, VisibleCluster vc, out uint entry)
{
    entry = 0;
    return vc.page == nullptr && hit_producer(s, vc.instance, entry);
}

// The view-relative placement of visible cluster `vc`, current or previous; false when its
// group or pose hides it.
public bool visible_placement(SceneRoot* s, ViewConstants* v, VisibleCluster vc, bool previous, out Placement p)
{
    uint entry;
    if (visible_producer(s, vc, entry))
        return producer_placement(s, s.producers[entry], previous, v.eye_cell.xyz, v.eye_offset.xyz, uint(v.eye_cell.w), p);
    InstanceGpu inst = s.instances[vc.instance];
    GroupGpu* g = s.groups + inst.group;
    Placement copy;
    p = (Placement)0;
    if (!placement(g, previous ? g.previous : g.current, inst.group_index, v, copy))
        return false;
    return cluster_placement(s, vc.instance, inst, load_cluster(vc.page, vc.local), copy, previous, p);
}

// The surface input where the ray through `pixel_center` (render pixels) meets triangle
// `triangle` of `vc`: barycentrics and their screen gradients from the ray and its
// one-pixel neighbors, uv footprint with the view's mip bias, normals and tangent frame in
// render space (back faces of double-sided materials turned to the viewer), and the world
// cell. Without mesh tangents the tangent follows the triangle's uv.
public PrimaryHit primary_hit(SceneRoot* s, ViewConstants* v, VisibleCluster vc, uint triangle, float2 pixel_center)
{
    MaterialGpu* m = s.materials + vc.material;
    PrimaryHit hit = (PrimaryHit)0;
    SurfaceInput i = (SurfaceInput)0;
    Corners t;
    uint attributes = 0;
    uint entry;
    visible_placement(s, v, vc, false, hit.placement);
    if (visible_producer(s, vc, entry))
    {
        ProducerBatch b = s.batches[entry];
        ProducerTriangle pt = producer_triangle(b, b.clusters[vc.cluster], triangle, false);
        t = pt.corners;
        i.user_tag = s.producers[entry].user_tag;
        i.cluster_user = pt.cluster_user;
        i.vertex_user = pt.vertex_user;
        hit.origin = s.producers[entry].origin;
    }
    else
    {
        InstanceGpu inst = s.instances[vc.instance];
        MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
        Cluster c = load_cluster(vc.page, vc.local);
        t = load_corners(vc.page, c, triangle, mesh);
        SkinnedGpu k;
        if (copy_skin(s, inst, k))
            skinned_corners(k, mesh, vc.page, c, vc.cluster, triangle, false, t);
        attributes = mesh.attributes;
        hit.error = c.self_error * hit.placement.scale;
        i.part = c.part;
        i.user_tag = inst.user_tag;
    }
    Placement pl = hit.placement;
    float3 q0 = place(pl, t.position[0]);
    float3 e1 = mul(pl.linear, t.position[1] - t.position[0]);
    float3 e2 = mul(pl.linear, t.position[2] - t.position[0]);

    float2 extent = float2(v.extent);
    float3 d = render_direction(v, view_direction(v, pixel_center / extent));
    float3 dx = render_direction(v, view_direction(v, (pixel_center + float2(1, 0)) / extent));
    float3 dy = render_direction(v, view_direction(v, (pixel_center + float2(0, 1)) / extent));
    float3 b = ray_barycentrics(d, q0, e1, e2);
    float2 bx = ray_barycentrics(dx, q0, e1, e2).xy - b.xy;
    float2 by = ray_barycentrics(dy, q0, e1, e2).xy - b.xy;

    triangle_geometry(i, t, attributes, m, pl, b.xy, bx, by, d, exp2(v.mip_bias));
    float spread = 1.0 / pixels_per_radian(v);
    hit.relative = b.z * d;
    hit.direction = d;
    hit.distance = b.z;
    hit.object = (1.0 - b.x - b.y) * t.position[0] + b.x * t.position[1] + b.y * t.position[2];
    i.context.kind = HIT_PRIMARY;
    i.context.depth = 0;
    i.context.cone_width = b.z * spread;
    i.context.cone_spread = spread;
    i.instance = vc.instance;
    i.cluster = vc.cluster;
    i.primitive = triangle;
    i.barycentrics = b.xy;
    i.material = vc.material;
    i.surface = m.surface;
    i.position = hit.relative + v.eye_render.xyz;
    world_cell(v.eye_cell.xyz, v.eye_offset.xyz, hit.relative, uint(v.eye_cell.w), i.cell, i.offset);
    i.view_direction = d;
    i.scene = s;
    i.view = v;
    i.data = m.data != 0 ? m.data : s.surface_data;
    hit.input = i;
    return hit;
}
