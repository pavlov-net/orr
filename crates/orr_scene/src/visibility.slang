// One view's raster state, shared by the culling, raster, resolve and HZB kernels, and the
// geometry helpers they share.
module visibility;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_pack.mesh;
import orr_scene.producer;
__exported import orr_scene.drawn;
__exported import orr_scene.hit;

// `RasterGpu.counts` entries.
public static const uint COUNT_VISIBLE = 0;          // visible clusters asked for, past capacity too
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
public static const uint COUNT_COVERAGE = 16;        // visible dielectric and presence clusters
public static const uint COUNT_SMOKE = 17;          // visible smoke clusters
public static const uint COUNTS = 18;

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
    public uint* args;               // workgroups x, y, z: work, software, smoke
    public uint* draw_args;          // per DRAW_* list, each of its `draws` mesh draws' workgroups x, y, z
    public uint* opaque;             // visible clusters with opaque triangles, hardware path
    public uint* alpha;              // visible clusters with alpha-tested triangles, hardware path
    public uint* software;           // visible clusters of the software raster
    public uint* particles;          // visible clusters of particle entries, for the transparency point
    public uint* coverage;           // visible dielectric and presence clusters, for the layer tiles
    public uint* smoke;              // visible clusters of smoke entries, for the froxels
    public uint* counts;             // COUNT_*
    public uint* slot_visibility;    // per instance slot: SLOT_*
    public uint* bin_args;           // per surface bin: x, y, z; then the resolve's sequence count
    public uint* bin_tiles;          // per bin: tile_capacity tile indices
    public uint64_t sequences;       // the resolve's device-generated sequences, one per bin at most
    public uint64_t bin_roots;       // per bin: its resolve root, BIN_ROOT_STRIDE bytes apart
    public float* depth;             // per pixel: resolved reverse-Z depth
    public StorageImageIndex a;
    public StorageImageIndex b;
    public StorageImageIndex x;      // index IMAGE_NONE without the extended targets
    public StorageImageIndex y;      // with `x`
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
    public uint draws;               // mesh draws over each DRAW_* list
    public uint draw_groups;         // list entries one of them covers at most, whole rows of GRID_X
    public uint draw_first;          // in the record a mesh draw takes: the first list entry it covers
    public uint _pad;
}

// `RasterGpu.args` offsets.
public static const uint ARGS_WORK = 0;
public static const uint ARGS_SOFTWARE = 3;
public static const uint ARGS_SMOKE = 6;
public static const uint ARGS = 9;

// The lists mesh draws take, by their place in `RasterGpu.draw_args`. A mesh draw launches a
// bounded number of workgroups, so each list takes `RasterGpu.draws` draws of up to
// `draw_groups` entries.
public static const uint DRAW_OPAQUE = 0;
public static const uint DRAW_ALPHA = 1;
public static const uint DRAW_COVERAGE = 2;
public static const uint DRAW_PARTICLES = 3;
public static const uint DRAW_LISTS = 4;

// Workgroup grids cover `n` groups in rows of GRID_X, the smallest maximum Vulkan allows. Every
// row is full, so a workgroup checks `grid_index` against the count of what it reads.
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

// Sizes the mesh draws over list `list`'s `n` entries: draw i takes them from i * draw_groups.
public void write_draws(RasterGpu* r, uint list, uint n)
{
    for (uint i = 0; i < r.draws; i++)
    {
        uint first = min(n, i * r.draw_groups);
        write_grid(r.draw_args + 3 * (list * r.draws + i), min(n - first, r.draw_groups));
    }
}

// The index of a new visible cluster, or false past capacity.
public bool add_visible(RasterGpu* r, VisibleCluster vc, out uint visible)
{
    InterlockedAdd(r.counts[COUNT_VISIBLE], 1, visible);
    if (visible >= r.visible_capacity)
    {
        InterlockedAdd(r.counts[COUNT_VISIBLE_DROPPED], 1);
        return false;
    }
    r.gbuffer.clusters[visible] = vc;
    return true;
}

// Bytes between the bins' roots: the largest uniform buffer offset alignment Vulkan allows.
public static const uint BIN_ROOT_STRIDE = 256;

// What every raster dispatch and draw takes.
public struct RasterRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public RasterGpu* raster;
    public uint value;               // the dispatch's bin, HZB mip or producer entry
    public uint _pad;
}

// The current or `previous` transform of `index` in `group` relative to the eye. False when the
// group hides the instance.
public bool placement(GroupGpu* g, bool previous, uint index, ViewConstants* v, out Placement p)
{
    return placement(g, previous, index, v.eye_cell.xyz, v.eye_offset.xyz, uint(v.eye_cell.w), p);
}

// The instance's current placement.
public bool instance_placement(SceneRoot* s, InstanceGpu inst, ViewConstants* v, out Placement p)
{
    GroupGpu* g = s.groups + inst.group;
    return placement(g, false, inst.group_index, v, p);
}

// A view-space point from an eye-relative render-space one.
public float3 view_from_relative(ViewConstants* v, float3 x)
{
    return mul(v.view_from_world_rotation, float4(x, 0.0));
}

// Reverse-Z device depth of render-space point `p`, as the jittered projection places it.
public float device_depth(ViewConstants* v, float3 p)
{
    float4 clip = mul(v.clip_from_view_jittered, float4(view_from_relative(v, p - v.eye_render.xyz), 1.0));
    return clip.z / clip.w;
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

// The `exact_` functions do arithmetic that two kernels, or two sites of one, must round
// alike: each operation is a scalar instruction in a `precise` variable's own initializer,
// which marks it as one a driver may not contract with another. The mark reaches neither the
// arguments of a call nor `mul`, `dot` and `length`, which are one instruction each.

// `place(p, x)`.
public float3 exact_place(Placement p, float3 x)
{
    float3x3 l = p.linear;
    float3 t = p.translation;
    precise float a = l[0].x * x.x + l[0].y * x.y + l[0].z * x.z + t.x;
    precise float b = l[1].x * x.x + l[1].y * x.y + l[1].z * x.z + t.y;
    precise float c = l[2].x * x.x + l[2].y * x.y + l[2].z * x.z + t.z;
    return float3(a, b, c);
}

// `mul(m, float4(x, 0.0))`: `view_from_relative` by `m`.
public float3 exact_rotate(float3x4 m, float3 x)
{
    precise float a = m[0].x * x.x + m[0].y * x.y + m[0].z * x.z;
    precise float b = m[1].x * x.x + m[1].y * x.y + m[1].z * x.z;
    precise float c = m[2].x * x.x + m[2].y * x.y + m[2].z * x.z;
    return float3(a, b, c);
}

// `placed_after(p, m)`.
public Placement exact_placed_after(Placement p, float3x4 m)
{
    float3x4 l = affine(p.linear, float3(0.0));
    float3 a = exact_rotate(l, float3(m[0].x, m[1].x, m[2].x));
    float3 b = exact_rotate(l, float3(m[0].y, m[1].y, m[2].y));
    float3 c = exact_rotate(l, float3(m[0].z, m[1].z, m[2].z));
    Placement o;
    o.linear = float3x3(float3(a.x, b.x, c.x), float3(a.y, b.y, c.y), float3(a.z, b.z, c.z));
    o.translation = exact_place(p, affine_translation(m));
    o.scale = largest_scale(o.linear);
    return o;
}

// `mul(m, float4(x, 1.0))`.
public float4 exact_project(float4x4 m, float3 x)
{
    precise float a = m[0].x * x.x + m[0].y * x.y + m[0].z * x.z + m[0].w;
    precise float b = m[1].x * x.x + m[1].y * x.y + m[1].z * x.z + m[1].w;
    precise float c = m[2].x * x.x + m[2].y * x.y + m[2].z * x.z + m[2].w;
    precise float d = m[3].x * x.x + m[3].y * x.y + m[3].z * x.z + m[3].w;
    return float4(a, b, c, d);
}

// `length(x)`.
public float exact_length(float3 x)
{
    precise float q = x.x * x.x + x.y * x.y + x.z * x.z;
    return sqrt(q);
}

// Distances from the focus clamp here, so a sphere around the focus takes its finest
// clusters.
public static const float FOCUS_NEAR = 0.01;

// The sphere around the model's bounds, in mesh space.
public float4 model_sphere(ModelGpu* model)
{
    return float4(0.5 * (model.bounds_min + model.bounds_max), 0.5 * length(model.bounds_max - model.bounds_min));
}

// Whether the DAG mesh has a merged stage: roots past the part stage's, one per part and
// section.
public bool mesh_has_merged_stage(MeshGpu* mesh)
{
    return mesh.root_count > mesh.part_count * mesh.section_count;
}

// `error` (m, mesh space) over `sphere` placed by `p` relative to the focus, in pixels at
// `pixels_per_radian`: the projection the traversal tests its thresholds with.
public float focus_pixels(Placement p, float4 sphere, float error, float pixels_per_radian)
{
    float3 c = exact_place(p, sphere.xyz);
    precise float gap = exact_length(c) - sphere.w * p.scale;
    float d = max(gap, FOCUS_NEAR);
    precise float e = error * p.scale / d * pixels_per_radian;
    return e;
}

// `pixel_from_clip(v, clip)`.
public float2 exact_pixel(ViewConstants* v, float4 clip)
{
    precise float2 ndc = clip.xy / clip.w;
    precise float2 unit = ndc * 0.5 + 0.5;
    precise float2 xy = unit * float2(v.extent);
    return xy;
}

// False when the sphere at eye-relative `center` lies wholly outside the view frustum.
public bool sphere_visible(ViewConstants* v, float3 center, float radius)
{
    float3 c = exact_rotate(v.view_from_world_rotation, center);
    float4x4 m = v.clip_from_view_jittered;
    precise float4 left = m[3] + m[0];
    precise float4 right = m[3] - m[0];
    precise float4 bottom = m[3] + m[1];
    precise float4 top = m[3] - m[1];
    precise float4 far = m[3] - m[2];
    float4 planes[5] = { left, right, bottom, top, far };
    for (uint i = 0; i < 5; i++)
    {
        float4 q = planes[i];
        precise float d = q.x * c.x + q.y * c.y + q.z * c.z + q.w;
        precise float reach = -radius * exact_length(q.xyz);
        if (d < reach)
            return false;
    }
    return true;
}

// Whether the sphere at eye-relative `center` of `radius` reaches the space between the eye
// and the near plane, which the frustum test leaves out.
public bool near_eye(ViewConstants* v, float3 center, float radius)
{
    float4x4 m = v.clip_from_view;
    float near = m[2][3];
    precise float corner2 = 1.0 + 1.0 / (m[0][0] * m[0][0]) + 1.0 / (m[1][1] * m[1][1]);
    precise float limit = 1.01 * near * sqrt(corner2);
    precise float gap = exact_length(center) - radius;
    return gap < limit;
}

// A view's last render of a scene, for kernels outside its raster that test what it saw: its
// constants and, while its history's HZB still holds that render, a raster record holding the
// HZB (`ViewTargets::hzb_record`), else nullptr.
public struct SeenView
{
    public ViewConstants* view;
    public RasterGpu* raster;
}

// Whether view `v`'s render saw none of the sphere at eye-relative `center` of `radius` m: it
// lies outside the frustum or, with `r` holding the render's HZB, its box lies behind the depth
// the render drew. A sphere holding dielectric or presence clusters (`layer`) is seen wherever
// it reaches between the eye and the near plane, as the culling's frustum test takes it. The
// tests grow with the sphere, so a sphere inside another is unseen wherever the other is, and
// their arithmetic is exact (`exact_place`), so every site that tests a sphere finds the same.
public bool sphere_unseen(ViewConstants* v, RasterGpu* r, bool layer, float3 center, float radius)
{
    if (layer && near_eye(v, center, radius))
        return false;
    if (!sphere_visible(v, center, radius))
        return true;
    if (r == nullptr)
        return false;
    Placement here = (Placement)0;
    here.linear = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    here.scale = 1.0;
    precise float3 lo = center - radius;
    precise float3 hi = center + radius;
    return box_occluded(v, r, here, lo, hi, false);
}

// Pixel coordinates of clip-space `clip`, as the viewport places them.
public float2 pixel_from_clip(ViewConstants* v, float4 clip)
{
    return (clip.xy / clip.w * 0.5 + 0.5) * float2(v.extent);
}

// The longer side in pixels of the screen rect of the box `lo..hi`, placed by `p`; false
// unless the box lies in front of the near plane.
public bool screen_size(ViewConstants* v, Placement p, float3 lo, float3 hi, out float size)
{
    size = 0.0;
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
    float2 sides = rect_max - rect_min;
    size = max(sides.x, sides.y);
    return true;
}

// The longest screen-rect side the software raster takes, from `RasterGpu.software_pixels`.
public float software_limit(RasterGpu* r)
{
    return min(r.software_pixels, SOFTWARE_MAX_PIXELS);
}

// True when the software raster takes the box `lo..hi`, placed by `p`: it lies in front of
// the near plane and its screen rect's longer side is at most `software_limit`.
public bool small_on_screen(ViewConstants* v, Placement p, float3 lo, float3 hi, RasterGpu* r)
{
    float size;
    return r.software_pixels > 0.0 && screen_size(v, p, lo, hi, size) && size <= software_limit(r);
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
        float3 x = exact_place(p, corner);
        precise float3 moved = x + v.eye_motion.xyz;
        float3 view = exact_rotate(previous ? v.prev_view_from_world_rotation : v.view_from_world_rotation, previous ? moved : x);
        float4 clip = exact_project(previous ? v.prev_clip_from_view : v.clip_from_view_jittered, view);
        if (!(clip.w > 0.0 && clip.z < clip.w))
            return false;
        float2 xy = exact_pixel(v, clip);
        rect_min = min(rect_min, xy);
        rect_max = max(rect_max, xy);
        precise float depth = clip.z / clip.w;
        nearest = max(nearest, depth);
    }
    // A pixel more on each side covers the jitter between the frames.
    precise float2 before = rect_min - 1.0;
    precise float2 after = rect_max + 1.0;
    precise float2 last = float2(r.gbuffer.extent) - 1.0;
    rect_min = max(before, 0.0);
    rect_max = min(after, last);
    if (any(rect_min > rect_max))
        return false;
    // A mip whose texels are half the rect's longer side: at most 3 x 3 texels cover it.
    precise float2 size = rect_max - rect_min;
    precise float level = ceil(log2(max(max(size.x, size.y), 1.0))) - 1.0;
    uint mip = min(uint(max(level, 0.0)), r.hzb_mips - 1);
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

// The triangles of visible cluster `vc` the alpha bin draws: from the returned index to
// `tri_count`. Linked surfaces with an alpha test run their coverage on every triangle, and
// a fading copy goes through the alpha bin whole.
public uint alpha_start(SceneRoot* s, VisibleCluster vc, MaterialGpu* m, Cluster c)
{
    if (copy_fading(s, vc.instance))
        return 0;
    if ((m.flags & MATERIAL_ALPHA_TESTED) == 0)
        return c.tri_count;
    return m.surface != 0 ? 0 : c.alpha_first;
}

// Pixel `t` of an 8 x 8 tile.
public uint2 tile_pixel(uint tile, uint tiles_x, uint t)
{
    return uint2(tile % tiles_x, tile / tiles_x) * 8 + uint2(t & 7, t >> 3);
}

// A render-space direction from a view-space one.
public float3 render_direction(ViewConstants* v, float3 d)
{
    float3x4 r = v.view_from_world_rotation;
    return mul(transpose(float3x3(r[0].xyz, r[1].xyz, r[2].xyz)), d);
}

// Where the ray along `d` from the eye meets the plane of the triangle with eye-relative
// corner `q0` and edges `e1` and `e2`: the weights of corners 1 and 2 in xy, the distance along
// `d` in z. Zeros, with `crosses` false, where the ray runs along the plane.
float3 ray_barycentrics(float3 d, float3 q0, float3 e1, float3 e2, out bool crosses)
{
    // At unit size, since the determinant multiplies two edge lengths.
    float size = unit_factor(max(abs(e1), abs(e2)));
    e1 *= size;
    e2 *= size;
    float3 pv = cross(d, e2);
    float det = dot(e1, pv);
    crosses = abs(det) > 1e-30;
    float inv = crosses ? 1.0 / det : 0.0;
    float3 tv = -q0;
    float3 qv = cross(tv, e1);
    return float3(dot(tv, pv) * size, dot(d, qv) * size, dot(e2, qv)) * inv;
}

// The weights of corners 1 and 2 at the point of that triangle nearest the line from the eye
// along unit `d`, a line that passes outside it: a point of one of its edges.
float2 nearest_barycentrics(float3 d, float3 q0, float3 e1, float3 e2)
{
    // At unit size, since an edge's squared length divides.
    float size = unit_factor(max(abs(e1), abs(e2)));
    float3 from[3] = { q0, q0, q0 + e1 };
    float3 along[3] = { e1 * size, e2 * size, (e2 - e1) * size };
    float nearest = 0.0;
    float2 b = float2(0.0);
    for (uint k = 0; k < 3; k++)
    {
        // Across the line: a vector less its part along `d`.
        float3 p = from[k] - d * dot(from[k], d);
        float3 s = along[k] - d * dot(along[k], d);
        float l = dot(s, s);
        float u = l > 0.0 ? saturate(-dot(p, s) / l * size) : 0.0;
        float3 across = p + s * (u / size);
        float distance = dot(across, across);
        if (k == 0 || distance < nearest)
        {
            nearest = distance;
            b = k == 0 ? float2(u, 0.0) : k == 1 ? float2(0.0, u) : float2(1.0 - u, u);
        }
    }
    return b;
}

// The point of the triangle with eye-relative corner `q0` and edges `e1` and `e2` that the ray
// from the eye along unit `d` shows: the weights of corners 1 and 2 in xy and its distance along
// `d` in z, the point in `at`, and in `plane` what `ray_barycentrics` gives. The raster draws
// the pixels whose centers its snapped corners hold, so the ray through one can pass outside
// the triangle, far outside in weights where the triangle is thin; it shows the triangle's
// point nearest it.
public float3 triangle_point(float3 d, float3 q0, float3 e1, float3 e2, out float3 at, out float3 plane)
{
    bool crosses;
    plane = ray_barycentrics(d, q0, e1, e2, crosses);
    at = plane.z * d;
    if (crosses && all(float3(1.0 - plane.x - plane.y, plane.xy) >= 0.0))
        return plane;
    float2 b = nearest_barycentrics(d, q0, e1, e2);
    at = q0 + b.x * e1 + b.y * e2;
    return float3(b, dot(at, d));
}

// The pixel ray's hit on a visible triangle, and the surface input at it.
public struct PrimaryHit
{
    public SurfaceInput input;
    public float3 relative;    // eye-relative render-space position; beside the pixel ray where that misses the triangle
    public float3 direction;   // unit pixel ray, render space
    public float distance;     // m along the ray
    public Placement placement;
    public float3 object;      // the hit in mesh or entry space
    public float error;        // m: the cluster's LOD error, or a drawn triangle's, times the instance scale; 0 on producer clusters
    public float3 previous[3]; // a drawn triangle's corners last frame, in the entry's space
    public uint origin;        // a producer entry's origin policy, else ORIGIN_DEFAULT
}

// The producer entry a visible cluster names, by a cluster of its batch or a drawn record;
// false for cooked clusters.
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
    InstanceGpu inst = copy_record(s, vc.instance);
    GroupGpu* g = s.groups + inst.group;
    Placement copy;
    p = (Placement)0;
    if (!placement(g, previous, inst.group_index, v, copy))
        return false;
    return cluster_placement(s, vc.instance, inst, load_cluster(vc.page, vc.local), copy, previous, p);
}

// Triangle `triangle` of drawn record `vc` of entry `entry`, as the linked `D` derives it.
DrawnTriangle drawn_triangle<D : IDrawn>(SceneRoot* s, ViewConstants* v, VisibleCluster vc, uint entry, uint triangle)
{
    ProducerGpu p = s.producers[entry];
    MaterialGpu* m = s.materials + vc.material;
    DrawnInput i;
    i.drawn = p.drawn;
    i.entry = entry;
    i.user_tag = p.user_tag;
    i.word = vc.cluster;
    i.primitive = triangle;
    i.scene = s;
    i.view = v;
    i.data = m.data != 0 ? m.data : s.surface_data;
    return D.triangle(i);
}

// The corners of a drawn triangle as a producer cluster's: one uv set, no tangents or colors.
Corners drawn_corners(DrawnTriangle t)
{
    Corners c;
    c.tangent_set = 0;
    for (uint j = 0; j < 3; j++)
    {
        c.position[j] = t.position[j];
        VertexAttributes a = (VertexAttributes)0;
        a.normal = t.normal[j];
        a.uv[0] = t.uv[j];
        a.uv[1] = t.uv[j];
        a.tangent = float4(1.0, 0.0, 0.0, 1.0);
        a.color = float4(1.0);
        c.attributes[j] = a;
    }
    return c;
}

// The segment of a strand entry's raster cluster `cluster` that ribbon triangle `triangle`
// draws.
public uint ribbon_segment(uint cluster, uint triangle)
{
    return cluster * STRAND_GROUP + triangle / 2;
}

// Where the unit ray from the eye along `d` first meets the strand around segment `segment`
// of `b` placed by `pl`: the swept spheres of the segment and of the segments joined to its
// ends, so a ribbon reaching past its segment resolves on the strand's surface. Gives `t`
// and the segment hit, or where all three miss, the point nearest the segment's axis.
float strand_ray(ProducerBatch b, Placement pl, float3 d, inout uint segment)
{
    Segment g;
    bool valid = load_segment(b, segment, false, g);
    float best = 1e30;
    uint nearest = segment;
    for (int k = -1; k <= 1; k++)
    {
        Segment n = g;
        uint s = segment + uint(k);
        if (k == 0 ? !valid : (k < 0 && segment == 0) || !load_segment(b, s, false, n) || n.vertex != g.vertex + uint(k))
            continue;
        float t;
        float3 normal;
        if (swept_intersect(float3(0.0), d, place(pl, n.a), place(pl, n.b), n.ra * pl.scale, n.rb * pl.scale, t, normal) && t < best)
        {
            best = t;
            nearest = s;
        }
    }
    segment = nearest;
    return best < 1e30 ? best : ray_axis_closest(float3(0.0), d, place(pl, g.a), place(pl, g.b));
}

// The surface input where the ray through `pixel_center` (render pixels) meets triangle
// `triangle` of `vc`: barycentrics and their screen gradients from the ray and its
// one-pixel neighbors, uv footprint with the view's mip bias and texture tap, normals and
// tangent frame in render space (back faces turned to the viewer, as `triangle_geometry`
// turns them), and the world cell. Without mesh tangents the tangent follows the triangle's uv.
// On a strand entry's ribbon the ray meets the strand's swept spheres instead
// (`strand_ray`); the input is a strand hit's (`strand_geometry`). A drawn record's triangle is
// the one `D` derives, and the hit's error and previous corners that triangle's.
public PrimaryHit primary_hit<D : IDrawn>(SceneRoot* s, ViewConstants* v, VisibleCluster vc, uint triangle, float2 pixel_center)
{
    PrimaryHit hit = (PrimaryHit)0;
    SurfaceInput i = (SurfaceInput)0;
    visible_placement(s, v, vc, false, hit.placement);
    Placement pl = hit.placement;
    float2 extent = float2(v.extent);
    float3 d = render_direction(v, view_direction(v, pixel_center / extent));
    float spread = 1.0 / pixels_per_radian(v);
    float bias = exp2(v.mip_bias);
    uint entry;
    bool producer = visible_producer(s, vc, entry);
    if (producer && s.producers[entry].strands != 0)
    {
        ProducerBatch b = s.batches[entry];
        uint segment = ribbon_segment(vc.cluster, triangle);
        float t = strand_ray(b, pl, d, segment);
        hit.distance = t;
        hit.relative = t * d;
        hit.object = mul(linear_inverse(pl.linear), t * d - pl.translation);
        strand_geometry(i, b, segment, hit.object, pl, false, t * spread * bias);
        i.user_tag = s.producers[entry].user_tag;
        hit.origin = s.producers[entry].origin;
    }
    else
    {
        Corners c;
        uint attributes = 0;
        bool drawn = producer && visible_drawn(vc);
        if (drawn)
        {
            DrawnTriangle t = drawn_triangle<D>(s, v, vc, entry, triangle);
            c = drawn_corners(t);
            hit.previous = t.previous;
            i.user_tag = s.producers[entry].user_tag;
            i.cluster_user = t.cluster_user;
            i.vertex_user = t.vertex_user;
            hit.origin = s.producers[entry].origin;
            hit.error = t.error * pl.scale;
        }
        else if (producer)
        {
            ProducerBatch b = s.batches[entry];
            ProducerTriangle pt = producer_triangle(b, b.clusters[vc.cluster], triangle, false);
            c = pt.corners;
            i.user_tag = s.producers[entry].user_tag;
            i.cluster_user = pt.cluster_user;
            i.vertex_user = pt.vertex_user;
            hit.origin = s.producers[entry].origin;
        }
        else
        {
            InstanceGpu inst = copy_record(s, vc.instance);
            MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
            Cluster k = load_cluster(vc.page, vc.local);
            c = load_corners(vc.page, k, triangle, mesh);
            SkinnedGpu sk;
            if (copy_skin(s, inst, sk))
                skinned_corners(sk, mesh, vc.page, k, vc.cluster, triangle, false, c);
            attributes = mesh.attributes;
            hit.error = k.self_error * pl.scale;
            i.part = k.part;
            i.user_tag = inst.user_tag;
        }
        float3 q0 = place(pl, c.position[0]);
        float3 e1 = mul(pl.linear, c.position[1] - c.position[0]);
        float3 e2 = mul(pl.linear, c.position[2] - c.position[0]);
        float3 dx = render_direction(v, view_direction(v, (pixel_center + float2(1, 0)) / extent));
        float3 dy = render_direction(v, view_direction(v, (pixel_center + float2(0, 1)) / extent));
        float3 plane;
        float3 b = triangle_point(d, q0, e1, e2, hit.relative, plane);
        bool crosses;
        float2 bx = ray_barycentrics(dx, q0, e1, e2, crosses).xy - plane.xy;
        float2 by = ray_barycentrics(dy, q0, e1, e2, crosses).xy - plane.xy;
        triangle_geometry(i, c, attributes, s, vc.material, pl, b.xy, bx, by, d, bias);
        hit.distance = b.z;
        hit.object = (1.0 - b.x - b.y) * c.position[0] + b.x * c.position[1] + b.y * c.position[2];
        i.cluster = drawn ? NO_CLUSTER : vc.cluster;
        i.primitive = triangle;
        i.barycentrics = b.xy;
    }
    hit.direction = d;
    i.context.kind = HIT_PRIMARY;
    i.context.depth = 0;
    i.context.cone_width = hit.distance * spread;
    i.context.cone_spread = spread;
    // A 2x2 quad shares its tap, so neighbors under magnification read the same texel
    // offset and the resolve's normal variance across the quad sees little filter noise.
    i.context.tap = texture_tap(v, uint2(pixel_center) / 2, v.frame_index, 0, 0);
    i.uv[0].tap = i.uv[1].tap = i.context.tap;
    i.instance = vc.instance;
    i.material = side_material(s, vc.material, i);
    i.swapped = i.material != vc.material;
    MaterialGpu* m = s.materials + i.material;
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

// `primary_hit` of a cluster no drawn entry's record names.
public PrimaryHit primary_hit(SceneRoot* s, ViewConstants* v, VisibleCluster vc, uint triangle, float2 pixel_center)
{
    return primary_hit<NoDrawn>(s, v, vc, triangle, pixel_center);
}

// Previous uv minus current uv, unjittered, of the point at eye-relative `relative` near `hit`
// on triangle `triangle` of visible cluster `vc`: its previous position comes from the previous
// transform of the same mesh-space point. A producer batch with previous positions, a skinned
// copy skinned by last frame's palette, or an edit whose vertices a kernel moved, moves the
// point by its vertices' motion too, as does a drawn triangle, by the corners `hit` holds of
// last frame's draw.
public float2 visible_motion(SceneRoot* s, ViewConstants* v, VisibleCluster vc, PrimaryHit hit, uint triangle, float3 relative)
{
    float4 current = mul(v.clip_from_view, float4(view_from_relative(v, relative), 1.0));
    Placement p = hit.placement;
    float3 object = mul(linear_inverse(p.linear), relative - p.translation);
    Placement prev;
    if (!visible_placement(s, v, vc, true, prev))
        prev = p;
    uint entry;
    bool deformed = false;
    Corners c = (Corners)0;
    SkinnedGpu k;
    bool producer = visible_producer(s, vc, entry);
    if (producer && s.producers[entry].strands != 0)
    {
        object += strand_motion(s.batches[entry], hit.input.cluster, hit.input.barycentrics.x);
    }
    else if (producer && visible_drawn(vc))
    {
        deformed = true;
        c.position = hit.previous;
    }
    else if (producer)
    {
        ProducerBatch pb = s.batches[entry];
        deformed = pb.previous != nullptr;
        if (deformed)
            c = producer_triangle(pb, pb.clusters[vc.cluster], triangle, true).corners;
    }
    else
    {
        InstanceGpu inst = copy_record(s, vc.instance);
        MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
        deformed = copy_skin(s, inst, k);
        if (deformed)
            skinned_corners(k, mesh, vc.page, load_cluster(vc.page, vc.local), vc.cluster, triangle, true, c);
        else
            deformed = edited_corners(mesh, vc.page, vc.local, triangle, c);
    }
    if (deformed)
    {
        float2 bc = hit.input.barycentrics;
        object += (1.0 - bc.x - bc.y) * c.position[0] + bc.x * c.position[1] + bc.y * c.position[2] - hit.object;
    }
    float3 prev_relative = place(prev, object) + v.eye_motion.xyz;
    float4 previous = mul(v.prev_clip_from_view, float4(mul(v.prev_view_from_world_rotation, float4(prev_relative, 0.0)), 1.0));
    return motion_between(previous, current);
}
