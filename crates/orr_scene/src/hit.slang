// Hit resolution for ray tracing: `resolve_hit` and `resolve_alpha` turn a traced hit on a
// scene's PTLAS into the surface the G-buffer would show there, through the kernel's linked
// surface. Also the instance placement and per-triangle surface input the material resolve
// shares, and ray-cone helpers for the `HitContext` callers pass.
module hit;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_pack.mesh;
import orr_scene.producer;
__exported import orr_scene.surface;

// An instance's transform as hits and the raster use it: mesh space to a frame centered
// on an eye or on the render origin.
public struct Placement
{
    public float3x3 linear;
    public float3 translation;
    public float scale;              // bounds the stretch of a mesh-space length
}

// The current or `previous` transform of `index` in `group`, relative to the point at `cell` +
// `offset`. False when the group hides it.
public bool placement(GroupGpu* g, bool previous, uint index, int3 cell, float3 offset, uint cell_log2, out Placement p)
{
    p = (Placement)0;
    if ((previous ? g.previous : g.current) == nullptr || index >= g.count)
        return false;
    float3x4 m = group_entry(g, previous, index);
    if (all(m[0] == 0.0) && all(m[1] == 0.0) && all(m[2] == 0.0))
        return false;
    p.linear = affine_linear(m);
    p.translation = view_relative(g.anchor_cell, g.anchor_offset, cell, offset, cell_log2) + affine_translation(m);
    p.scale = largest_scale(p.linear);
    return true;
}

public float3 place(Placement p, float3 x)
{
    return mul(p.linear, x) + p.translation;
}

// `p` applied after the mesh-space transform `m`.
public Placement placed_after(Placement p, float3x4 m)
{
    Placement o;
    o.linear = mul(p.linear, affine_linear(m));
    o.translation = mul(p.linear, affine_translation(m)) + p.translation;
    o.scale = largest_scale(o.linear);
    return o;
}

// Transform `part` of the copy's pose block, from the pose group's current or previous
// buffer: `group[(pose_index + group_index * stride + variant) * parts + part]`, the pose
// variant from the copy's variant word, below `pose_variants`. False when the entry is
// outside the group or all zero: the pose hides the part.
public bool part_pose(SceneRoot* s, InstanceGpu inst, MeshGpu* mesh, uint part, bool previous, out float3x4 pose)
{
    pose = float3x4(0.0);
    if (inst.pose_group == 0)
        return false;
    GroupGpu* g = s.groups + inst.pose_group;
    uint8_t* transforms = previous ? g.previous : g.current;
    uint variant = min(inst.variant & 0xff, max(uint(inst.pose_variants), 1u) - 1);
    uint block = inst.pose_index + inst.group_index * uint(inst.pose_stride) + variant;
    uint index = block * mesh.part_count + part;
    if (transforms == nullptr || index >= g.count)
        return false;
    pose = ((float3x4*)transforms)[index];
    return any(pose[0] != 0.0) || any(pose[1] != 0.0) || any(pose[2] != 0.0);
}

// Whether the raster poses cluster `c` of slot copy `copy` this frame, or with `previous` in
// the previous one: an animated part's cluster of a copy inside its animation range. A far
// chain step is never posed.
public bool cluster_posed(SceneRoot* s, uint copy, Cluster c, bool previous)
{
    if ((c.flags & CLUSTER_ANIMATED_PART) == 0 || (copy & COPY_FAR) != 0)
        return false;
    return (s.slot_state[copy].rt_flags & (previous ? RT_WAS_ANIMATED : RT_ANIMATED)) != 0;
}

// The raster's placement of cluster `c` of slot copy `copy` (record `inst`), from the copy's
// placement: posed by its part's transform when `cluster_posed`. False when the pose hides
// the part.
public bool cluster_placement(SceneRoot* s, uint copy, InstanceGpu inst, Cluster c, Placement placed, bool previous, out Placement p)
{
    p = placed;
    if (!cluster_posed(s, copy, c, previous))
        return true;
    float3x4 pose;
    if (!part_pose(s, inst, s.meshes + s.models[inst.model].mesh, c.part, previous, pose))
        return false;
    p = placed_after(placed, pose);
    return true;
}

// The producer entry of PTLAS instance or visible-cluster index `index`; false for slots
// and part instances.
public bool hit_producer(SceneRoot* s, uint index, out uint entry)
{
    entry = index - s.producer_base;
    return index >= s.producer_base && entry < s.producer_capacity;
}

// The transform of producer entry `p`, current or previous, relative to the point at
// `cell` + `offset`. False when its group hides it.
public bool producer_placement(SceneRoot* s, ProducerGpu p, bool previous, int3 cell, float3 offset, uint cell_log2, out Placement pl)
{
    GroupGpu* g = s.groups + p.group;
    return placement(g, previous, p.group_index, cell, offset, cell_log2, pl);
}

// The material of an emitted cluster, MATERIAL_NONE outside the entry's materials.
public uint producer_material(ProducerGpu p, ProducerClusterGpu k)
{
    return uint(k.material) < p.material_count ? p.materials[k.material] : MATERIAL_NONE;
}

// The material of a strand entry's segments: its first.
public uint strand_material(ProducerGpu p)
{
    return p.material_count > 0 ? p.materials[0] : MATERIAL_NONE;
}

// `Hit.cluster_id` of a swept-sphere hit, as a uint.
static const uint SWEPT_CLUSTER = 0xffffffffu;

// The segment of strand entry hit `cluster_id`, `primitive`: a swept-sphere hit names it, a
// tube hit names a triangle of a tube cluster.
public uint strand_segment(uint cluster_id, uint primitive)
{
    return cluster_id == SWEPT_CLUSTER ? primitive : cluster_id * TUBE_SEGMENTS + primitive / TUBE_SEGMENT_TRIANGLES;
}

// The geometry of `i` at the entry-relative point `object` on segment `segment` of strand
// batch `b` placed by `pl`: the swept sphere's normal there, the tangent along the strand,
// the uv along the segment and its steps across a footprint `width` m wide (along the
// segment only), the segment's and its vertices' words, and in `barycentrics.x` where along
// the segment the point lies. With `previous` the point moves with the segment's vertices
// to last frame's, where the batch keeps them. False for a segment that is not drawn or
// traced.
public bool strand_geometry(inout SurfaceInput i, ProducerBatch b, uint segment, float3 object, Placement pl, bool previous, float width)
{
    Segment g;
    if (!load_segment(b, segment, false, g))
        return false;
    float u;
    float3 n_obj = swept_normal(g.a, g.b, g.ra, g.rb, object, u);
    if (previous)
        object += strand_motion(b, segment, u);
    float3 n = unit_direction(mul(normal_matrix(pl.linear), n_obj));
    float size = unit_factor(pl.linear);
    float3 axis = mul(pl.linear * size, g.b - g.a);
    float3 t = unit_direction(axis, perpendicular(n));
    float2 uv0 = producer_uv(b.attrs[g.vertex]);
    float2 uv1 = producer_uv(b.attrs[g.vertex + 1]);
    i.uv[0].uv = lerp(uv0, uv1, u);
    i.uv[0].dx = (uv1 - uv0) * (width / max(unit_length(axis), 1e-30)) * size;
    i.uv[0].dy = float2(0.0);
    i.uv[0].inward = float2(0.0);
    i.uv[1] = i.uv[0];
    i.geometric_normal = n;
    i.normal = n;
    i.frame_normal = n;
    i.tangent = float4(t, 1.0);
    i.coat_frame_normal = n;
    i.coat_tangent = i.tangent;
    i.object_position = object;
    i.object_normal = n_obj;
    i.object_to_render = pl.linear;
    i.barycentrics = float2(u, 0.0);
    i.cluster = segment;
    i.primitive = 0;
    i.cluster_user = b.segment_user[segment];
    i.vertex_user = b.vertex_user != nullptr ? uint3(b.vertex_user[g.vertex], b.vertex_user[g.vertex + 1], 0) : uint3(0);
    i.vertex_color = float4(1.0);
    return true;
}

// How far the point `u` along segment `segment` of strand batch `b` moved since last frame,
// in the entry's space: zero where the batch keeps no previous positions.
public float3 strand_motion(ProducerBatch b, uint segment, float u)
{
    Segment g, last;
    if (b.previous == nullptr || !load_segment(b, segment, false, g))
        return float3(0.0);
    load_segment(b, segment, true, last);
    return lerp(last.a - g.a, last.b - g.b, u);
}

// The origin policy of a hit on material `m`: a producer entry's own unless it is the
// default, else the material's.
public uint hit_origin(MaterialGpu* m, uint producer_origin)
{
    uint material = (m.flags & MATERIAL_ORIGIN_MASK) >> MATERIAL_ORIGIN_SHIFT;
    return producer_origin != ORIGIN_DEFAULT ? producer_origin : material;
}

// One triangle of an emitted cluster: entry-relative corners, last frame's with `previous`
// where the batch keeps them, attributes, and the cluster's and vertices' user words.
public struct ProducerTriangle
{
    public Corners corners;
    public uint cluster_user;
    public uint3 vertex_user;
}

public ProducerTriangle producer_triangle(ProducerBatch b, ProducerClusterGpu k, uint triangle, bool previous)
{
    uint3 c = cluster_corners(k, triangle) + k.first_vertex;
    float3* positions = previous && b.previous != nullptr ? b.previous : b.positions;
    ProducerTriangle t;
    t.corners.tangent_set = 0;
    for (uint j = 0; j < 3; j++)
    {
        uint v = c[j];
        t.corners.position[j] = positions[v];
        VertexAttributes a = (VertexAttributes)0;
        a.normal = producer_normal(b.attrs[v]);
        a.uv[0] = producer_uv(b.attrs[v]);
        a.uv[1] = producer_uv1(b, v, a.uv[0]);
        a.tangent = float4(1.0, 0.0, 0.0, 1.0);
        a.color = b.colors != nullptr ? producer_color(b.colors[v]) : float4(1.0);
        t.corners.attributes[j] = a;
        t.vertex_user[j] = b.vertex_user != nullptr ? b.vertex_user[v] : 0;
    }
    t.cluster_user = k.user;
    return t;
}

// The material of a cluster's section for one copy.
public uint section_material(SceneRoot* s, InstanceGpu inst, ModelGpu* model, uint section)
{
    if (section >= model.section_count)
        return MATERIAL_NONE;
    if (inst.material_base == NO_MATERIAL_SET)
        return model.materials[section];
    uint set = (inst.variant >> 8) & 0xff;
    return s.material_sets[inst.material_base + set * model.section_count + section];
}

// Separates the chain fade's draws from a pixel's or ray's other draws.
static const uint CHAIN_FADE_STREAM = 0x3c6ef372u;

// The chain fade's uniform draw at `pixel` in frame `frame`, shared by both steps of slot
// `slot`.
public float fade_draw_pixel(uint2 pixel, uint frame, uint slot)
{
    return unit_float(pcg3d(uint3(pixel.x ^ CHAIN_FADE_STREAM, pixel.y ^ (slot * 0x9e3779b9u), frame)).x);
}

// The chain fade's uniform draw for a ray along `d` in frame `frame`, shared by both steps
// of slot `slot`.
public float fade_draw_ray(float3 d, uint frame, uint slot)
{
    return unit_float(pcg3d(uint3(asuint(d.x) ^ (slot * 0x9e3779b9u), asuint(d.y) ^ CHAIN_FADE_STREAM, asuint(d.z) ^ frame)).x);
}

// Whether the copy's section mask hides `section`.
public bool section_hidden(InstanceGpu inst, uint section)
{
    return section < 16 && ((inst.variant >> (16 + section)) & 1) != 0;
}

// The corners of one triangle of a resident cluster: mesh-space positions and attributes.
public struct Corners
{
    public float3 position[3];
    public VertexAttributes attributes[3];
    public uint tangent_set;       // what the mesh's tangents here follow
}

public Corners load_corners(uint8_t* page, Cluster c, uint triangle, MeshGpu* mesh)
{
    PageHeader h = load_page_header(page);
    uint3 k = load_triangle(page, h, c, triangle);
    Corners t;
    t.position[0] = load_position(page, h, c, k.x, mesh.position_step);
    t.position[1] = load_position(page, h, c, k.y, mesh.position_step);
    t.position[2] = load_position(page, h, c, k.z, mesh.position_step);
    t.attributes[0] = load_attributes(page, h, c, k.x, mesh.attributes);
    t.attributes[1] = load_attributes(page, h, c, k.y, mesh.attributes);
    t.attributes[2] = load_attributes(page, h, c, k.z, mesh.attributes);
    t.tangent_set = (uint(c.flags) >> CLUSTER_TANGENT_SET_SHIFT) & 3;
    return t;
}

// Where u grows and where v shrinks, up the image, on a triangle of edges `e1` and `e2` along which
// the uv changes by `uv1` and `uv2`, at no set size: 0 where its uvs span no surface.
public void uv_directions(float3 e1, float3 e2, float2 uv1, float2 uv2, out float3 along, out float3 across)
{
    float r = uv1.x * uv2.y - uv2.x * uv1.y;
    along = r != 0.0 ? (e1 * uv2.y - e2 * uv1.y) / r : float3(0.0);
    across = r != 0.0 ? (e1 * uv2.x - e2 * uv1.x) / r : float3(0.0);
}

// The tangent that follows `set` on triangle `t`, at weights `w`: the mesh's where its tangents
// do, else that of the triangle's uv in the set, whose `frame` normal is then `n`. `e1` and `e2`
// are its edges, and `unit` and `mirror` what the placement gives directions.
float4 triangle_tangent(Corners t, uint attributes, uint set, float3 w, float3x3 unit, float mirror, float3 e1, float3 e2, float3 n, inout float3 frame)
{
    VertexAttributes a0 = t.attributes[0];
    VertexAttributes a1 = t.attributes[1];
    VertexAttributes a2 = t.attributes[2];
    if ((attributes & MESH_TANGENT) != 0 && (set == t.tangent_set || set == TANGENT_SET_OTHER))
    {
        float3 tt = float3(0.0);
        for (uint c = 0; c < 3; c++)
            tt += w[c] * unit_direction(mul(unit, t.attributes[c].tangent.xyz), float3(1, 0, 0));
        return float4(tt, a0.tangent.w * mirror);
    }
    bool set1 = set == 1;
    float2 uv1 = (set1 ? a1.uv[1] : a1.uv[0]) - (set1 ? a0.uv[1] : a0.uv[0]);
    float2 uv2 = (set1 ? a2.uv[1] : a2.uv[0]) - (set1 ? a0.uv[1] : a0.uv[0]);
    // Uv v runs down the image and a normal map's y up it.
    float3 tt, bt;
    uv_directions(e1, e2, uv1, uv2, tt, bt);
    if (dot(tt, tt) == 0.0)
    {
        float3 tb;
        orthonormal_basis(n, tt, tb);
        bt = tb;
    }
    tt = safe_normalize(tt - n * dot(n, tt), float3(1, 0, 0));
    frame = n;
    return float4(tt, dot(cross(n, tt), bt) < 0.0 ? -1.0 : 1.0);
}

// The geometry of `i` at barycentrics `b` of triangle `t` placed by `pl`, with `attributes`
// the mesh's MESH_* flags: both uv sets and their footprints from the barycentric steps `bx` and `by` of one footprint width, normals and
// tangent frame in render space (back faces of double-sided materials turned toward the
// ray `d`, but for a dielectric's, which stay on its front so a ray knows the way it crosses),
// and the object-space position and normal. `material` is the section's, and the frame follows
// the material of the side `d` meets. Without mesh tangents on the uvs that frame follows, as
// where a copy's material replaces the section's, the tangent follows the triangle's uv in
// that set, as a coat's frame of its own does when the mesh's tangents follow the base's.
public void triangle_geometry(inout SurfaceInput i, Corners t, uint attributes, SceneRoot* scene, uint material, Placement pl, float2 b, float2 bx, float2 by, float3 d, float bias)
{
    VertexAttributes a0 = t.attributes[0];
    VertexAttributes a1 = t.attributes[1];
    VertexAttributes a2 = t.attributes[2];
    float3 w = float3(1.0 - b.x - b.y, b.x, b.y);
    // Only the directions of what the placement gives are read, so its size stays out of them.
    float3x3 unit = unit_linear(pl.linear);
    float3 e1 = mul(unit, t.position[1] - t.position[0]);
    float3 e2 = mul(unit, t.position[2] - t.position[0]);
    float size = unit_factor(max(abs(e1), abs(e2)));
    e1 *= size;
    e2 *= size;

    for (uint s = 0; s < 2; s++)
    {
        float2 along1 = a1.uv[s] - a0.uv[s];
        float2 along2 = a2.uv[s] - a0.uv[s];
        i.uv[s].uv = a0.uv[s] + b.x * along1 + b.y * along2;
        i.uv[s].inward = a0.uv[s] + (along1 + along2) / 3.0 - i.uv[s].uv;
        i.uv[s].dx = (bx.x * along1 + bx.y * along2) * bias;
        i.uv[s].dy = (by.x * along1 + by.y * along2) * bias;
    }

    float3x3 normals = normal_matrix(pl.linear);
    float mirror = linear_mirror(pl.linear);
    float3 ng = unit_direction(cross(e1, e2)) * mirror;
    float3 face = cross(t.position[1] - t.position[0], t.position[2] - t.position[0]);
    float3 n_obj = safe_normalize(w.x * a0.normal + w.y * a1.normal + w.z * a2.normal, safe_normalize(face));
    // Placed per vertex and interpolated as they stand, the frame normal maps are baked in.
    float3 frame = float3(0.0);
    for (uint c = 0; c < 3; c++)
        frame += w[c] * unit_direction(mul(normals, t.attributes[c].normal), ng);
    float3 n = safe_normalize(frame, ng);
    i.back_face = dot(ng, d) > 0.0 && (scene.materials[material].flags & MATERIAL_DOUBLE_SIDED) != 0;
    MaterialGpu* side = scene.materials + side_material(scene, material, i);
    float3 coat_frame = frame;
    float4 tangent = triangle_tangent(t, attributes, side.tangent_set, w, unit, mirror, e1, e2, n, frame);
    float4 coat_tangent = tangent;
    if (side.coat_tangent_set != side.tangent_set)
        coat_tangent = triangle_tangent(t, attributes, side.coat_tangent_set, w, unit, mirror, e1, e2, n, coat_frame);
    else
        coat_frame = frame;
    if (i.back_face && (side.flags & MATERIAL_DIELECTRIC) == 0)
    {
        ng = -ng;
        n = -n;
        frame = -frame;
        coat_frame = -coat_frame;
        tangent.w = -tangent.w;
        coat_tangent.w = -coat_tangent.w;
    }
    i.geometric_normal = ng;
    i.normal = n;
    i.frame_normal = frame;
    i.tangent = tangent;
    i.coat_frame_normal = coat_frame;
    i.coat_tangent = coat_tangent;
    i.object_position = w.x * t.position[0] + w.y * t.position[1] + w.z * t.position[2];
    i.object_normal = n_obj;
    i.object_to_render = pl.linear;
    i.vertex_color = w.x * a0.color + w.y * a1.color + w.z * a2.color;
}

// The material shading the side of `material` that `i` shows: its back's on a back face.
public uint side_material(SceneRoot* s, uint material, SurfaceInput i)
{
    MaterialGpu* m = s.materials + material;
    return i.back_face && (m.flags & MATERIAL_HAS_BACK) != 0 ? m.back : material;
}

// The material whose interior a surface of `material` bounds: one surface has one interior, so
// that of the material it backs.
public uint interior_material(SceneRoot* s, uint material)
{
    MaterialGpu* m = s.materials + material;
    return material != MATERIAL_NONE && (m.flags & MATERIAL_BACK_SIDE) != 0 ? m.back : material;
}

// Whether a surface of `material` bounds an interior, which only a solid dielectric has. The
// material that owns the interior decides, so a thin back on a solid front still leaves it.
public bool crosses_interior(SceneRoot* s, uint material)
{
    uint flags = s.materials[interior_material(s, material)].flags;
    return (flags & (MATERIAL_DIELECTRIC | MATERIAL_THIN_WALLED)) == MATERIAL_DIELECTRIC;
}

// The skinned copy `inst` names; false for copies not from `add_skinned`.
public bool copy_skin(SceneRoot* s, InstanceGpu inst, out SkinnedGpu k)
{
    k = (SkinnedGpu)0;
    if ((inst.flags & INSTANCE_SKINNED) == 0 || inst.pose_index >= s.skinned_capacity)
        return false;
    k = s.skinned[inst.pose_index];
    return k.positions != nullptr;
}

// The copy's mesh-space bounds this frame: those of its skinned clusters, else its model's
// rest bounds. False for a skinned copy that skinned nothing this frame.
public bool copy_bounds(SceneRoot* s, InstanceGpu inst, out float3 lo, out float3 hi)
{
    ModelGpu* model = s.models + inst.model;
    lo = model.bounds_min;
    hi = model.bounds_max;
    if ((inst.flags & INSTANCE_SKINNED) == 0)
        return true;
    if (inst.pose_index >= s.skinned_capacity)
        return false;
    SkinStateGpu st = s.skin_state[inst.pose_index];
    lo = float3(unordered(st.lo.x), unordered(st.lo.y), unordered(st.lo.z));
    hi = float3(unordered(st.hi.x), unordered(st.hi.y), unordered(st.hi.z));
    return st.clusters > 0 && all(lo <= hi);
}

// The mesh-space box a skinned copy's vertices stay in whatever its pose: its model's rest
// bounds grown by its mesh's deformation bound. False when the mesh is unbounded, so only
// this frame's skinned bounds hold.
public bool skinned_reach(SceneRoot* s, InstanceGpu inst, out float3 lo, out float3 hi)
{
    ModelGpu* model = s.models + inst.model;
    float bound = s.meshes[model.mesh].skin.bound;
    lo = model.bounds_min - bound;
    hi = model.bounds_max + bound;
    return bound > 0.0;
}

// The mesh-space box of mesh cluster `mesh_cluster`'s skinned vertices this frame.
public void skinned_box(SkinnedGpu k, uint mesh_cluster, out float3 lo, out float3 hi)
{
    float* b = k.boxes + 6 * mesh_cluster;
    lo = float3(b[0], b[1], b[2]);
    hi = float3(b[3], b[4], b[5]);
}

// Gives the corners of triangle `triangle` of cluster `c` (mesh cluster `mesh_cluster`) of
// `page` the copy's skinned positions, last frame's with `previous`, and its skinned normals
// and tangents.
public void skinned_corners(SkinnedGpu k, MeshGpu* mesh, uint8_t* page, Cluster c, uint mesh_cluster, uint triangle, bool previous, inout Corners t)
{
    uint3 v = load_triangle(page, load_page_header(page), c, triangle) + mesh.skin.vertex_first[mesh_cluster];
    float3* p = previous ? k.previous : k.positions;
    for (uint j = 0; j < 3; j++)
    {
        t.position[j] = p[v[j]];
        t.attributes[j].normal = oct_decode_unorm16(k.normals[v[j]]);
        if ((mesh.attributes & MESH_TANGENT) != 0)
        {
            uint w = k.tangents[v[j]];
            t.attributes[j].tangent = float4(oct_decode_unorm16(w), (w & (1u << 16)) != 0 ? -1.0 : 1.0);
        }
    }
}

// Gives the corners of triangle `triangle` of cluster `local` of `page` their positions before
// this frame's kernels, when `page` is one of `mesh`'s edited pages; false for any other page.
public bool edited_corners(MeshGpu* mesh, uint8_t* page, uint local, uint triangle, inout Corners t)
{
    uint8_t* before = previous_page(mesh, page);
    if (before == page)
        return false;
    PageHeader h = load_page_header(before);
    Cluster was = load_cluster(before, local);
    uint3 v = load_triangle(page, load_page_header(page), load_cluster(page, local), triangle);
    for (uint j = 0; j < 3; j++)
        t.position[j] = load_position(before, h, was, v[j], mesh.position_step);
    return true;
}

// The world cell and the offset in it of the point `relative` from `cell` + `offset`.
public void world_cell(int3 cell, float3 offset, float3 relative, uint cell_log2, out int3 out_cell, out float3 out_offset)
{
    float size = float(1u << cell_log2);
    float3 o = offset + relative;
    float3 cells = floor(o / size);
    out_cell = cell + int3(cells);
    out_offset = o - cells * size;
}

// A ray cone: its width at the ray origin (m) and its spread angle (rad).
public struct RayCone
{
    public float width;
    public float spread;
}

// The camera's cone through a pixel: width 0 at the eye, one pixel of spread.
public RayCone camera_cone(ViewConstants* v)
{
    RayCone c;
    c.width = 0.0;
    c.spread = 2.0 / (float(v.extent.y) * abs(v.clip_from_view[1][1]));
    return c;
}

// The cone's width `t` m along its ray.
public float cone_width(RayCone c, float t)
{
    return c.width + c.spread * t;
}

// The cone of a ray leaving the hit `t` m along `c`, its spread widened by `spread_change`
// rad (0 at a flat mirror).
public RayCone cone_after_hit(RayCone c, float t, float spread_change)
{
    RayCone o;
    o.width = cone_width(c, t);
    o.spread = c.spread + spread_change;
    return o;
}

// The context of a hit `t` m along cone `c` whose stochastic texture filter draws `tap`
// (`texture_tap`; negative x: hardware filtering).
public HitContext hit_context(uint kind, uint depth, RayCone c, float t, float2 tap = float2(-1.0))
{
    HitContext h;
    h.kind = kind;
    h.depth = depth;
    h.cone_width = cone_width(c, t);
    h.cone_spread = c.spread;
    h.tap = tap;
    return h;
}

// Separates the stochastic texture filter's samples from a pixel's other sample streams.
static const uint TEXTURE_TAP_STREAM = 0x7e47a9u;

// The stochastic texture filter's uniforms for the hit at `depth` on the path through
// `pixel` in sample `frame` of stream `seed`, from their own sample stream; negative where
// `v` takes hardware filtering.
public float2 texture_tap(ViewConstants* v, uint2 pixel, uint frame, uint seed, uint depth)
{
    if (v.texture_filter != TEXTURE_FILTER_STOCHASTIC)
        return float2(-1.0);
    Sampler rng = Sampler(pixel, frame, seed ^ TEXTURE_TAP_STREAM);
    rng.dimension = depth;
    return rng.next2();
}

// A resolved ray hit.
public struct SurfaceHit
{
    public GSurface surface;      // position includes the surface's depth offset
    public UvFootprint uv[2];     // per uv set
    public uint user_tag;
    public uint part;
    public uint cluster_user;     // producer words, 0 on cooked clusters
    public uint3 vertex_user;     // per corner
    public float geometric_error; // m: the hit cluster's LOD error times the instance scale
    public float depth_offset;    // m, along the ray
    public bool back_face;        // on a double-sided material's back
    public bool valid;            // false for hits on entries this module does not resolve
}

// The slot copy of PTLAS instance `index`: slot instances name themselves, part instances
// their copy's slot, and the instance tracing a fading copy's far step its slot with
// COPY_FAR. False for producer entries (`hit_producer`).
public bool hit_copy(SceneRoot* s, uint index, out uint copy)
{
    copy = index;
    if (index < s.part_base)
        return true;
    if (index < s.producer_base)
    {
        copy = s.part_slots[index - s.part_base];
        return true;
    }
    return false;
}

// The material of the hit on cluster `cluster_id` of PTLAS instance `instance_index`, or
// MATERIAL_NONE where it names none.
public uint traced_material(SceneRoot* s, uint instance_index, uint cluster_id)
{
    uint entry;
    if (hit_producer(s, instance_index, entry))
    {
        ProducerGpu p = s.producers[entry];
        if (p.strands != 0)
            return strand_material(p);
        ProducerBatch b = s.batches[entry];
        if (cluster_id >= emitted_clusters(b))
            return MATERIAL_NONE;
        return producer_material(p, b.clusters[cluster_id]);
    }
    uint copy;
    if (!hit_copy(s, instance_index, copy))
        return MATERIAL_NONE;
    InstanceGpu inst = copy_record(s, copy);
    ClusterRef r = s.clusters[cluster_id];
    return section_material(s, inst, s.models + inst.model, load_cluster(r.page, r.local).section);
}

// What `resolve_hit` and `resolve_alpha` share: the surface input of a hit and its cluster.
struct HitInput
{
    SurfaceInput input;
    Placement placement;
    float error;       // m, mesh space: the cluster's LOD error; 0 for producer clusters
    uint origin;       // a producer entry's origin policy, else ORIGIN_DEFAULT
}

// With `sided` the input is that of the side `d` meets, else of the section's material.
bool hit_input(SceneRoot* s, ViewConstants* v, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, Corners t, bool fetched, bool previous, bool sided, HitContext context, out HitInput h)
{
    h = (HitInput)0;
    uint cell_log2 = uint(s.origin_cell.w);
    float bias = footprint_bias(context.kind, v);
    SurfaceInput i = (SurfaceInput)0;
    Corners corners = (Corners)0;
    uint material;
    uint attributes;
    Placement pl;
    uint entry;
    bool deformed = false;
    bool producer = hit_producer(s, instance_index, entry);
    bool strand = producer && s.producers[entry].strands != 0;
    if (strand)
    {
        // Strands are opaque, so only closest hits with their hit point reach here.
        ProducerGpu p = s.producers[entry];
        ProducerBatch b = s.batches[entry];
        if (!fetched || !producer_placement(s, p, previous, s.origin_cell.xyz, float3(0.0), cell_log2, pl))
            return false;
        uint segment = strand_segment(cluster_id, primitive);
        float3 object = cluster_id == SWEPT_CLUSTER ? t.position[0]
            : (1.0 - bary.x - bary.y) * t.position[0] + bary.x * t.position[1] + bary.y * t.position[2];
        if (!strand_geometry(i, b, segment, object, pl, previous, context.cone_width * bias))
            return false;
        material = strand_material(p);
        attributes = 0;
        i.instance = instance_index;
        i.user_tag = p.user_tag;
        h.origin = p.origin;
    }
    else if (producer)
    {
        ProducerGpu p = s.producers[entry];
        ProducerBatch b = s.batches[entry];
        if (cluster_id >= emitted_clusters(b))
            return false;
        ProducerClusterGpu k = b.clusters[cluster_id];
        if (!producer_placement(s, p, previous, s.origin_cell.xyz, float3(0.0), cell_log2, pl))
            return false;
        ProducerTriangle pt = producer_triangle(b, k, primitive, previous);
        corners = pt.corners;
        material = producer_material(p, k);
        attributes = 0;
        i.instance = instance_index;
        i.cluster = cluster_id;
        i.part = 0;
        i.user_tag = p.user_tag;
        i.cluster_user = pt.cluster_user;
        i.vertex_user = pt.vertex_user;
        h.origin = p.origin;
    }
    else
    {
        uint copy;
        if (!hit_copy(s, instance_index, copy))
            return false;
        InstanceGpu inst = copy_record(s, copy);
        ModelGpu* model = s.models + inst.model;
        MeshGpu* mesh = s.meshes + model.mesh;
        ClusterRef r = s.clusters[cluster_id];
        Cluster c = load_cluster(r.page, r.local);
        corners = load_corners(r.page, c, primitive, mesh);
        SkinnedGpu k;
        deformed = copy_skin(s, inst, k);
        if (deformed)
            skinned_corners(k, mesh, r.page, c, r.mesh_cluster, primitive, previous, corners);
        else if (previous)
            deformed = edited_corners(mesh, r.page, r.local, primitive, corners);
        material = section_material(s, inst, model, c.section);
        attributes = mesh.attributes;
        GroupGpu* g = s.groups + inst.group;
        if (!placement(g, previous, inst.group_index, s.origin_cell.xyz, float3(0.0), cell_log2, pl))
            return false;
        // A part instance traces its part posed.
        if (instance_index >= s.part_base && (copy & COPY_FAR) == 0)
        {
            float3x4 pose;
            if (!part_pose(s, inst, mesh, c.part, previous, pose))
                return false;
            pl = placed_after(pl, pose);
        }
        i.instance = copy;
        i.cluster = r.mesh_cluster;
        i.part = c.part;
        i.user_tag = inst.user_tag;
        h.error = c.self_error;
        h.origin = ORIGIN_DEFAULT;
    }
    if (material == MATERIAL_NONE)
        return false;
    if (!strand)
        hit_triangle(i, d, corners, t, fetched, deformed, previous, attributes, s, material, pl, primitive, bary, context, bias);
    if (sided)
        material = side_material(s, material, i);
    MaterialGpu* m = s.materials + material;
    i.uv[0].tap = i.uv[1].tap = context.tap;
    i.context = context;
    i.material = material;
    i.surface = m.surface;
    i.position = place(pl, i.object_position);
    world_cell(s.origin_cell.xyz, float3(0.0), i.position, cell_log2, i.cell, i.offset);
    i.view_direction = d;
    i.scene = s;
    i.view = v;
    i.data = m.data != 0 ? m.data : s.surface_data;
    h.input = i;
    h.placement = pl;
    return true;
}

// The triangle geometry of a hit at barycentrics `bary` of triangle `primitive` with
// `corners`, the fetched positions `t` replacing theirs where they are this frame's.
void hit_triangle(inout SurfaceInput i, float3 d, Corners corners, Corners t, bool fetched, bool deformed, bool previous, uint attributes, SceneRoot* s, uint material, Placement pl, uint primitive, float2 bary, HitContext context, float bias)
{
    // A deformed copy's fetched positions are this frame's.
    if (fetched && !(deformed && previous))
        corners.position = t.position;

    // Barycentric steps of the cone's footprint: two directions across the ray, carried
    // along it onto the triangle's plane.
    float3 e1 = mul(pl.linear, corners.position[1] - corners.position[0]);
    float3 e2 = mul(pl.linear, corners.position[2] - corners.position[0]);
    // At unit size, since the steps multiply four edge lengths.
    float size = unit_factor(max(abs(e1), abs(e2)));
    e1 *= size;
    e2 *= size;
    float3 n = cross(e1, e2);
    float nn = max(dot(n, n), 1e-30);
    float3 across = cross(d, n);
    float3 x = dot(across, across) > 1e-12 * nn ? normalize(across) : float3(0.0);
    if (all(x == 0.0))
    {
        float3 y0;
        orthonormal_basis(d, x, y0);
    }
    float3 y = cross(x, d);
    float dn = dot(d, n);
    dn = abs(dn) < 1e-6 * sqrt(nn) ? (dn < 0.0 ? -1e-6 : 1e-6) * sqrt(nn) : dn;
    float3 yp = y - d * (dot(y, n) / dn);
    float w = context.cone_width;
    float2 bx = float2(dot(cross(x * w, e2), n), dot(cross(e1, x * w), n)) / nn * size;
    float2 by = float2(dot(cross(yp * w, e2), n), dot(cross(e1, yp * w), n)) / nn * size;

    triangle_geometry(i, corners, attributes, s, material, pl, bary, bx, by, d, bias);
    i.primitive = primitive;
    i.barycentrics = bary;
}

// The surface at a ray hit through linked surface `S` (the kernel's `Surface`): the
// committed hit's instance index, cluster id, primitive, barycentrics and fetched
// object-space positions (`orr_shaders.trace`'s: on a swept-sphere hit the first is the hit point),
// the unit render-space ray direction `d`, and the caller's context. `previous` places the
// hit with the previous frame's transforms. `v` is passed on to the surface and may be null.
// On a strand entry the surface's `cluster` is the segment and its `primitive` 0.
public SurfaceHit resolve_hit<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, float3 object_positions[3], bool previous, HitContext context)
{
    SurfaceHit hit = (SurfaceHit)0;
    Corners t;
    t.position = object_positions;
    HitInput h;
    if (!hit_input(s, v, d, instance_index, cluster_id, primitive, bary, t, true, previous, true, context, h))
        return hit;
    float offset;
    GSurface g = S.surface(h.input, offset);
    offset = max(offset, 0.0);
    MaterialGpu* m = s.materials + h.input.material;
    g.position = h.input.position + d * offset;
    g.raster_error = 0.0;
    g.flags = (g.flags & ~GBUFFER_ORIGIN_MASK) | (hit_origin(m, h.origin) << GBUFFER_ORIGIN_SHIFT);
    g.instance = h.input.instance;
    g.cluster = h.input.cluster;
    g.primitive = h.input.primitive;
    g.material = h.input.material;
    g.medium_point = float4(h.input.uv[0].uv, h.input.uv[1].uv);
    hit.surface = g;
    hit.uv = h.input.uv;
    hit.user_tag = h.input.user_tag;
    hit.part = h.input.part;
    hit.geometric_error = h.error * h.placement.scale;
    hit.cluster_user = h.input.cluster_user;
    hit.vertex_user = h.input.vertex_user;
    hit.depth_offset = offset;
    hit.back_face = h.input.back_face;
    hit.valid = true;
    return hit;
}

// Whether a fading copy's step is present for the ray along `d`; true for every other hit.
bool fade_present(SceneRoot* s, float3 d, uint instance_index)
{
    uint copy;
    if (!hit_copy(s, instance_index, copy))
        return true;
    return copy_present(s, copy, fade_draw_ray(d, s.frame_index, copy_slot(copy)));
}

// Whether a non-opaque triangle candidate is present: on a fading copy's step, by its share
// of the fade against a draw both steps share; then for alpha-tested materials with
// probability the surface's coverage (the material's alpha test unless a linked surface has
// its own), against a draw hashed from the candidate and the frame. Coverage is the section
// material's on both sides, as the raster's alpha test and the opacity micromaps are.
public bool resolve_alpha<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, HitContext context)
{
    if (!fade_present(s, d, instance_index))
        return false;
    Corners t = (Corners)0;
    HitInput h;
    if (!hit_input(s, v, d, instance_index, cluster_id, primitive, bary, t, false, false, false, context, h))
        return false;
    // A fading copy's triangles of materials without an alpha test are candidates only for
    // the fade.
    if ((s.materials[h.input.material].flags & MATERIAL_ALPHA_TESTED) == 0)
        return true;
    uint3 key = uint3(asuint(bary.x) ^ (instance_index * 0x9E3779B9u), asuint(bary.y) ^ (primitive * 0x85EBCA6Bu), s.frame_index ^ cluster_id);
    return S.coverage(h.input) > unit_float(pcg3d(key).x);
}

// How much of a non-opaque triangle candidate is present, 0 to 1: a presence material's
// opacity, else the linked surface's coverage or the material's alpha test. Presence layers
// pass rays through by it.
public float resolve_coverage<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, HitContext context)
{
    if (!fade_present(s, d, instance_index))
        return 0.0;
    Corners t = (Corners)0;
    HitInput h;
    if (!hit_input(s, v, d, instance_index, cluster_id, primitive, bary, t, false, false, false, context, h))
        return 0.0;
    if ((s.materials[h.input.material].flags & MATERIAL_PRESENCE) != 0)
        return material_opacity(h.input);
    return S.coverage(h.input);
}

// What a triangle candidate of a particle producer entry shows rays that take particles by
// their emission alone: its material, its unit render-space front-face normal, the emission
// there in nits, unexposed (the emission texture where bound), and its opacity
// (`material_opacity`), both at the hit's uv from the entry's batch. False for any other
// candidate.
public bool resolve_particle(SceneRoot* s, uint instance_index, uint cluster_id, uint primitive, float2 bary, out uint material, out float3 normal, out float3 emission, out float opacity)
{
    material = MATERIAL_NONE;
    normal = float3(0.0);
    emission = float3(0.0);
    opacity = 0.0;
    uint entry;
    SurfaceInput i;
    if (!hit_producer(s, instance_index, entry) || !particle_candidate(s, entry, cluster_id, primitive, bary, material, normal, i))
        return false;
    MaterialGpu* m = s.materials + material;
    emission = material_emission_at(m, i.uv);
    opacity = material_opacity(i);
    return true;
}

// The material, unit render-space front-face normal and textures' input of a triangle candidate
// of producer entry `entry`, from its batch. False unless the entry holds particles or smoke.
bool particle_candidate(SceneRoot* s, uint entry, uint cluster_id, uint primitive, float2 bary, out uint material, out float3 normal, out SurfaceInput i)
{
    material = MATERIAL_NONE;
    normal = float3(0.0);
    i = (SurfaceInput)0;
    ProducerGpu p = s.producers[entry];
    ProducerBatch b = s.batches[entry];
    if ((p.opacity != PRODUCER_PARTICLE && p.opacity != PRODUCER_SMOKE) || cluster_id >= emitted_clusters(b))
        return false;
    ProducerClusterGpu k = b.clusters[cluster_id];
    material = producer_material(p, k);
    if (material == MATERIAL_NONE)
        return false;
    Placement pl;
    if (!producer_placement(s, p, false, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return false;
    Corners t = producer_triangle(b, k, primitive, false).corners;
    float3x3 unit = unit_linear(pl.linear);
    float3 e1 = mul(unit, t.position[1] - t.position[0]);
    float3 e2 = mul(unit, t.position[2] - t.position[0]);
    float size = unit_factor(max(abs(e1), abs(e2)));
    normal = unit_direction(cross(e1 * size, e2 * size)) * linear_mirror(pl.linear);
    float2 uv = (1.0 - bary.x - bary.y) * t.attributes[0].uv[0] + bary.x * t.attributes[1].uv[0] + bary.y * t.attributes[2].uv[0];
    i = particle_input(s, material, uv);
    i.uv[0].inward = (t.attributes[0].uv[0] + t.attributes[1].uv[0] + t.attributes[2].uv[0]) / 3.0 - uv;
    i.uv[1].inward = i.uv[0].inward;
    return true;
}

// What a particle material's textures read at `uv`, with no filter footprint.
public SurfaceInput particle_input(SceneRoot* s, uint material, float2 uv)
{
    SurfaceInput i = (SurfaceInput)0;
    i.scene = s;
    i.material = material;
    i.vertex_color = float4(1.0);
    i.uv[0].uv = uv;
    i.uv[0].tap = float2(-1.0);
    i.uv[1] = i.uv[0];
    return i;
}

// A smoke particle's base color, emission in nits and opacity at input `i`.
public void smoke_look(SurfaceInput i, out float3 albedo, out float3 emission, out float opacity)
{
    MaterialGpu* m = i.scene.materials + i.material;
    albedo = m.base_weight * material_color(m, TEXTURE_BASE_COLOR, i.uv, m.base_color);
    emission = material_emission_at(m, i.uv);
    opacity = saturate(material_opacity(i));
}

// The look of a triangle candidate of a smoke entry at its hit; false for any other candidate.
public bool resolve_smoke(SceneRoot* s, uint instance_index, uint cluster_id, uint primitive, float2 bary, out float3 albedo, out float3 emission, out float opacity)
{
    albedo = float3(0.0);
    emission = float3(0.0);
    opacity = 0.0;
    uint entry;
    if (!hit_producer(s, instance_index, entry) || s.producers[entry].opacity != PRODUCER_SMOKE)
        return false;
    uint material;
    float3 normal;
    SurfaceInput i;
    if (!particle_candidate(s, entry, cluster_id, primitive, bary, material, normal, i))
        return false;
    smoke_look(i, albedo, emission, opacity);
    return true;
}

// The render-space position last frame of the point a ray hit: the previous transform for
// rigid content, the previous positions of producer batches that keep them, of skinned copies
// and of the vertices kernels move in an edit. False where the hit does not resolve.
public bool hit_previous_position(SceneRoot* s, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, out float3 position)
{
    position = float3(0.0);
    Corners t = (Corners)0;
    HitInput h;
    HitContext context = (HitContext)0;
    if (!hit_input(s, nullptr, d, instance_index, cluster_id, primitive, bary, t, false, true, false, context, h))
        return false;
    position = h.input.position;
    return true;
}

// The geometric normal (the triangle's front face, turned toward the ray where
// `triangle_geometry` turns it) and the material whose interior a ray hit bounds, without
// shading it. False where the hit does not resolve.
public bool hit_geometry(SceneRoot* s, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, float3 object_positions[3], out float3 normal, out uint material)
{
    normal = float3(0.0);
    material = MATERIAL_NONE;
    Corners t = (Corners)0;
    t.position = object_positions;
    HitInput h;
    HitContext context = (HitContext)0;
    if (!hit_input(s, nullptr, d, instance_index, cluster_id, primitive, bary, t, true, false, true, context, h))
        return false;
    normal = h.input.geometric_normal;
    material = interior_material(s, h.input.material);
    return true;
}
