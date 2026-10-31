// One view's raster state, shared by the culling, raster, resolve and HZB kernels, and the
// geometry helpers they share.
module visibility;

import orr.abi;
import orr.math;
import orr_pack.mesh;
__exported import hit;

// `RasterGpu.counts` entries.
public static const uint COUNT_VISIBLE = 0;          // visible clusters written
public static const uint COUNT_VISIBLE_DROPPED = 1;  // visible clusters past capacity
public static const uint COUNT_WORK_DROPPED = 2;     // cluster work past capacity
public static const uint COUNT_DRAWS = 3;            // 1: the draw count of the indirect draws
public static const uint COUNTS = 4;

// `TaskWork.flags`.
public static const uint WORK_DAG = 1;      // select clusters by the DAG cluster rule
public static const uint WORK_MERGED = 2;   // DAG: the merged stage, else the part stage

// Up to 32 clusters of one page for one task workgroup.
public struct TaskWork
{
    public uint instance;
    public uint page;          // mesh page index
    public uint first;         // first cluster in the page
    public uint count_flags;   // count in the low byte, WORK_* above it
}

public struct RasterGpu
{
    public GBufferGpu gbuffer;       // what later passes read
    public TaskWork* work;
    public uint* task_args;          // task workgroups: x, y, z
    public uint* alpha_args;         // alpha-bin mesh workgroups: x, y, z
    public uint* alpha;              // visible clusters with alpha-tested triangles
    public uint* counts;             // COUNT_*
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
    public float error_pixels;       // raster LOD error threshold
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
    public float error;        // m: the cluster's LOD error times the instance scale
}

// The surface input where the ray through `pixel_center` (render pixels) meets triangle
// `triangle` of `vc`: barycentrics and their screen gradients from the ray and its
// one-pixel neighbors, uv footprint with the view's mip bias, normals and tangent frame in
// render space (back faces of double-sided materials turned to the viewer), and the world
// cell. Without mesh tangents the tangent follows the triangle's uv.
public PrimaryHit primary_hit(SceneRoot* s, ViewConstants* v, VisibleCluster vc, uint triangle, float2 pixel_center)
{
    InstanceGpu inst = s.instances[vc.instance];
    MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
    MaterialGpu* m = s.materials + vc.material;
    PrimaryHit hit;
    instance_placement(s, inst, v, hit.placement);
    Placement pl = hit.placement;
    Cluster c = load_cluster(vc.page, vc.local);
    Corners t = load_corners(vc.page, c, triangle, mesh);
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

    SurfaceInput i;
    triangle_geometry(i, t, mesh, m, pl, b.xy, bx, by, d, exp2(v.mip_bias));
    float spread = 1.0 / pixels_per_radian(v);
    hit.relative = b.z * d;
    hit.direction = d;
    hit.distance = b.z;
    hit.error = c.self_error * pl.scale;
    i.context.kind = HIT_PRIMARY;
    i.context.depth = 0;
    i.context.cone_width = b.z * spread;
    i.context.cone_spread = spread;
    i.instance = vc.instance;
    i.cluster = vc.cluster;
    i.part = c.part;
    i.primitive = triangle;
    i.barycentrics = b.xy;
    i.user_tag = inst.user_tag;
    i.cluster_user = 0;
    i.vertex_user = uint3(0);
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
