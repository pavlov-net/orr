// Hit resolution for ray tracing: `resolve_hit` and `resolve_alpha` turn a traced hit on a
// scene's PTLAS into the surface the G-buffer would show there, through the kernel's linked
// surface. Also the instance placement and per-triangle surface input the material resolve
// shares, and ray-cone helpers for the `HitContext` callers pass.
module hit;

import orr.abi;
import orr.math;
import orr_pack.mesh;
import producer;
__exported import surface;

// An instance's transform as hits and the raster use it: mesh space to a frame centered
// on an eye or on the render origin.
public struct Placement
{
    public float3x3 linear;
    public float3 translation;
    public float scale;              // largest axis scale
}

// The transform of `index` in `group` from `transforms` (the group's current or previous
// buffer), relative to the point at `cell` + `offset`. False when the group hides it.
public bool placement(GroupGpu* g, uint8_t* transforms, uint index, int3 cell, float3 offset, uint cell_log2, out Placement p)
{
    p = (Placement)0;
    if (transforms == nullptr || index >= g.count)
        return false;
    float3x4 m = ((float3x4*)transforms)[index];
    if (all(m[0] == 0.0) && all(m[1] == 0.0) && all(m[2] == 0.0))
        return false;
    p.linear = affine_linear(m);
    p.translation = view_relative(g.anchor_cell, g.anchor_offset, cell, offset, cell_log2) + affine_translation(m);
    p.scale = largest_scale(p.linear);
    return true;
}

// The length of the longest column of `r`: its largest axis scale.
float largest_scale(float3x3 r)
{
    float3 c0 = float3(r[0].x, r[1].x, r[2].x);
    float3 c1 = float3(r[0].y, r[1].y, r[2].y);
    float3 c2 = float3(r[0].z, r[1].z, r[2].z);
    return sqrt(max(dot(c0, c0), max(dot(c1, c1), dot(c2, c2))));
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

// Whether the raster poses cluster `c` of the copy in `slot` this frame, or with
// `previous` in the previous one: an animated part's cluster of a copy inside its
// animation range.
public bool cluster_posed(SceneRoot* s, uint slot, Cluster c, bool previous)
{
    if ((c.flags & CLUSTER_ANIMATED_PART) == 0)
        return false;
    return (s.slot_state[slot].rt_flags & (previous ? RT_WAS_ANIMATED : RT_ANIMATED)) != 0;
}

// The raster's placement of cluster `c` of the copy in `slot`, from the copy's placement:
// posed by its part's transform when `cluster_posed`. False when the pose hides the part.
public bool cluster_placement(SceneRoot* s, uint slot, InstanceGpu inst, Cluster c, Placement copy, bool previous, out Placement p)
{
    p = copy;
    if (!cluster_posed(s, slot, c, previous))
        return true;
    float3x4 pose;
    if (!part_pose(s, inst, s.meshes + s.models[inst.model].mesh, c.part, previous, pose))
        return false;
    p = placed_after(copy, pose);
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
    return placement(g, previous ? g.previous : g.current, p.group_index, cell, offset, cell_log2, pl);
}

// The material of an emitted cluster, MATERIAL_NONE outside the entry's materials.
public uint producer_material(ProducerGpu p, ProducerClusterGpu k)
{
    return uint(k.material) < p.material_count ? p.materials[k.material] : MATERIAL_NONE;
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
    for (uint j = 0; j < 3; j++)
    {
        uint v = c[j];
        t.corners.position[j] = positions[v];
        VertexAttributes a = (VertexAttributes)0;
        a.normal = producer_normal(b.attrs[v]);
        a.uv = producer_uv(b.attrs[v]);
        a.tangent = float4(1.0, 0.0, 0.0, 1.0);
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
    return t;
}

// The geometry of `i` at barycentrics `b` of triangle `t` placed by `pl`, with `attributes`
// the mesh's MESH_* flags: uv and its
// footprint from the barycentric steps `bx` and `by` of one footprint width, normals and
// tangent frame in render space (back faces of double-sided materials turned toward the
// ray `d`), and the object-space position and normal. Without mesh tangents the tangent
// follows the triangle's uv.
public void triangle_geometry(inout SurfaceInput i, Corners t, uint attributes, MaterialGpu* m, Placement pl, float2 b, float2 bx, float2 by, float3 d, float bias)
{
    VertexAttributes a0 = t.attributes[0];
    VertexAttributes a1 = t.attributes[1];
    VertexAttributes a2 = t.attributes[2];
    float3 w = float3(1.0 - b.x - b.y, b.x, b.y);
    float3 e1 = mul(pl.linear, t.position[1] - t.position[0]);
    float3 e2 = mul(pl.linear, t.position[2] - t.position[0]);

    float2 uv1 = a1.uv - a0.uv;
    float2 uv2 = a2.uv - a0.uv;
    i.uv.uv = a0.uv + b.x * uv1 + b.y * uv2;
    i.uv.dx = (bx.x * uv1 + bx.y * uv2) * bias;
    i.uv.dy = (by.x * uv1 + by.y * uv2) * bias;

    float det = determinant(pl.linear);
    float mirror = det < 0.0 ? -1.0 : 1.0;
    float3 ng = safe_normalize(cross(e1, e2)) * mirror;
    float3x3 normal_matrix = cofactors(pl.linear);
    float3 n_obj = safe_normalize(w.x * a0.normal + w.y * a1.normal + w.z * a2.normal, ng);
    float3 n = safe_normalize(mul(normal_matrix, n_obj) * mirror, ng);
    float4 tangent;
    if ((attributes & MESH_TANGENT) != 0)
    {
        float3 tt = w.x * a0.tangent.xyz + w.y * a1.tangent.xyz + w.z * a2.tangent.xyz;
        tangent = float4(safe_normalize(mul(pl.linear, tt), float3(1, 0, 0)), a0.tangent.w * mirror);
    }
    else
    {
        float r = uv1.x * uv2.y - uv2.x * uv1.y;
        float3 tt = r != 0.0 ? (e1 * uv2.y - e2 * uv1.y) / r : float3(0.0);
        float3 bt = r != 0.0 ? (e2 * uv1.x - e1 * uv2.x) / r : float3(0.0);
        if (dot(tt, tt) == 0.0)
        {
            float3 tb;
            orthonormal_basis(n, tt, tb);
            bt = tb;
        }
        tt = safe_normalize(tt - n * dot(n, tt), float3(1, 0, 0));
        tangent = float4(tt, dot(cross(n, tt), bt) < 0.0 ? -1.0 : 1.0);
    }
    if (dot(ng, d) > 0.0 && (m.flags & MATERIAL_DOUBLE_SIDED) != 0)
    {
        ng = -ng;
        n = -n;
        tangent.w = -tangent.w;
    }
    i.geometric_normal = ng;
    i.normal = n;
    i.tangent = tangent;
    i.object_position = w.x * t.position[0] + w.y * t.position[1] + w.z * t.position[2];
    i.object_normal = n_obj;
    i.object_to_render = pl.linear;
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

// The context of a hit `t` m along cone `c`.
public HitContext hit_context(uint kind, uint depth, RayCone c, float t)
{
    HitContext h;
    h.kind = kind;
    h.depth = depth;
    h.cone_width = cone_width(c, t);
    h.cone_spread = c.spread;
    return h;
}

// A resolved ray hit.
public struct SurfaceHit
{
    public GSurface surface;      // position includes the surface's depth offset
    public UvFootprint uv;
    public uint user_tag;
    public uint part;
    public uint cluster_user;     // producer words, 0 on cooked clusters
    public uint3 vertex_user;     // per corner
    public float geometric_error; // m: the hit cluster's LOD error times the instance scale
    public float depth_offset;    // m, along the ray
    public bool valid;            // false for hits on entries this module does not resolve
}

// The instance slot of PTLAS instance `index`: slot instances name themselves, part
// instances their copy's slot. False for producer entries (`hit_producer`).
public bool hit_slot(SceneRoot* s, uint index, out uint slot)
{
    slot = index;
    if (index < s.part_base)
        return true;
    if (index < s.producer_base)
    {
        slot = s.part_slots[index - s.part_base];
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
        ProducerBatch b = s.batches[entry];
        if (cluster_id >= emitted_clusters(b))
            return MATERIAL_NONE;
        return producer_material(s.producers[entry], b.clusters[cluster_id]);
    }
    uint slot;
    if (!hit_slot(s, instance_index, slot))
        return MATERIAL_NONE;
    InstanceGpu inst = s.instances[slot];
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

bool hit_input(SceneRoot* s, ViewConstants* v, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, Corners t, bool fetched, bool previous, HitContext context, out HitInput h)
{
    h = (HitInput)0;
    uint cell_log2 = uint(s.origin_cell.w);
    SurfaceInput i = (SurfaceInput)0;
    Corners corners;
    uint material;
    uint attributes;
    Placement pl;
    uint entry;
    if (hit_producer(s, instance_index, entry))
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
        uint slot;
        if (!hit_slot(s, instance_index, slot))
            return false;
        InstanceGpu inst = s.instances[slot];
        ModelGpu* model = s.models + inst.model;
        MeshGpu* mesh = s.meshes + model.mesh;
        ClusterRef r = s.clusters[cluster_id];
        Cluster c = load_cluster(r.page, r.local);
        corners = load_corners(r.page, c, primitive, mesh);
        material = section_material(s, inst, model, c.section);
        attributes = mesh.attributes;
        GroupGpu* g = s.groups + inst.group;
        if (!placement(g, previous ? g.previous : g.current, inst.group_index, s.origin_cell.xyz, float3(0.0), cell_log2, pl))
            return false;
        // A part instance traces its part posed.
        if (instance_index >= s.part_base)
        {
            float3x4 pose;
            if (!part_pose(s, inst, mesh, c.part, previous, pose))
                return false;
            pl = placed_after(pl, pose);
        }
        i.instance = slot;
        i.cluster = r.mesh_cluster;
        i.part = c.part;
        i.user_tag = inst.user_tag;
        h.error = c.self_error;
        h.origin = ORIGIN_DEFAULT;
    }
    if (material == MATERIAL_NONE)
        return false;
    if (fetched)
        corners.position = t.position;
    MaterialGpu* m = s.materials + material;

    // Barycentric steps of the cone's footprint: two directions across the ray, carried
    // along it onto the triangle's plane.
    float3 e1 = mul(pl.linear, corners.position[1] - corners.position[0]);
    float3 e2 = mul(pl.linear, corners.position[2] - corners.position[0]);
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
    float2 bx = float2(dot(cross(x * w, e2), n), dot(cross(e1, x * w), n)) / nn;
    float2 by = float2(dot(cross(yp * w, e2), n), dot(cross(e1, yp * w), n)) / nn;

    triangle_geometry(i, corners, attributes, m, pl, bary, bx, by, d, 1.0);
    i.context = context;
    i.primitive = primitive;
    i.barycentrics = bary;
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

// The surface at a ray hit through linked surface `S` (the kernel's `Surface`): the
// committed hit's instance index, cluster id, primitive, barycentrics and fetched
// object-space positions, the unit render-space ray direction `d`, and the caller's
// context. `previous` places the hit with the previous frame's transforms. `v` is passed
// on to the surface and may be null.
public SurfaceHit resolve_hit<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, float3 object_positions[3], bool previous, HitContext context)
{
    SurfaceHit hit = (SurfaceHit)0;
    Corners t;
    t.position = object_positions;
    HitInput h;
    if (!hit_input(s, v, d, instance_index, cluster_id, primitive, bary, t, true, previous, context, h))
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
    g.primitive = primitive;
    g.material = h.input.material;
    hit.surface = g;
    hit.uv = h.input.uv;
    hit.user_tag = h.input.user_tag;
    hit.part = h.input.part;
    hit.geometric_error = h.error * h.placement.scale;
    hit.cluster_user = h.input.cluster_user;
    hit.vertex_user = h.input.vertex_user;
    hit.depth_offset = offset;
    hit.valid = true;
    return hit;
}

// Whether a non-opaque triangle candidate is present: the linked surface's coverage, else
// the material's alpha test, at least one half.
public bool resolve_alpha<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 d, uint instance_index, uint cluster_id, uint primitive, float2 bary, HitContext context)
{
    Corners t = (Corners)0;
    HitInput h;
    if (!hit_input(s, v, d, instance_index, cluster_id, primitive, bary, t, false, false, context, h))
        return false;
    return S.coverage(h.input) >= 0.5;
}
