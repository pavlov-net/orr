// What the frame's caustic photons start from, after the light set's selection and the
// emissive entries' pass: `aim` and `aim_sweep` fold the last frame's photons into the aim cells
// and evict the cells no caster list holds any more, `pool` places each caster entry's list in
// the pool, which holds every cluster the scene's caster entries can hold (a traced producer's
// emitted clusters, an instance's caster set at the row its copy traces or the list its
// traversal gathered), and the lists' clusters in blocks, `lists_weigh`, `lists_place` and
// `lists_finish` write those lists' CDFs (`cluster_lists`) over the clusters whose material
// casts, each by its area, its falloff from the focus and the share of its aim cell's photons
// that deposited where the view reads, `weigh` weighs every entry by its list's weight, builds
// the entry CDF and sizes the frame's photons, and `interiors` walks each light out to find the
// interiors that hold it.
//
// The weights only aim the photons: each carries its flux over the density its caster was drawn
// at, and every caster's weight stays above 0 by the share the weights give area and falloff
// alone, so the caustics they converge to do not depend on them.
#include "orr_root.h"
import orr_shaders.math;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.producer_blocks;
import orr_scene.visibility;
import orr_scene.tables;
import orr_lighting.caustics;
import orr_lighting.cluster_lists;
import orr_lighting.interiors;
import orr_lighting.light_sampling;
import orr_lighting.media;
import orr_lighting.world_cache;

extern struct Surface : ISurfaces = StandardSurface;

struct Root
{
    SceneRoot* scene;
    LightSetGpu* set;
    PhotonSourceGpu* source;
    EmissiveSelection* entries;
    float* cdf;
    // Per entry: its caster list's weight, its first position in the pool, and its casters' area
    // by focus falloff with that by their aim cells' yields.
    float* weights;
    uint* firsts;
    float2* masses;
    // What each block of the caster lists' clusters adds up to, and its `masses`.
    ListBlock* blocks;
    float2* block_masses;
    // The caster lists' CDFs, and each position's aim cell.
    uint* pool;
    uint* pool_cells;
    // The caster lists' clusters in blocks (`producer_blocks`), over the emissive list's
    // entries.
    uint* block_layout;
    EyeInterior* interiors;
    // The aim cells' checks (`slot_find`), and the cells (`PhotonSourceGpu.aim`).
    uint* aim_checks;
    uint4* aim;
    // The focus in render space, and in w the m over which a caster's weight falls with the
    // square of its distance from the focus.
    float4 focus;
    uint clear;                     // 1: forget the aim cells, the photons' yield and the grid's level
    int base_cell_log2;             // the grid's level 0, as for `CausticsGpu`
    uint photons;                   // the most photons a frame starts
    float cell_photons;             // the photons a frame aims to deposit in each cell the view reads
    // The pool's positions for the producers' lists, which hold every cluster they can emit,
    // and after them those for the copies' caster lists, which the frames before measured.
    uint producer_room;
    uint list_room;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

// What a caster cluster's weight adds to its aim cell's yield, over the source's `aimed`: about
// AIM_EXPLORE / (1 + AIM_EXPLORE) of the photons go by area and focus falloff alone, whatever
// their cells' yields, which finds a caster newly lit or newly seen and keeps every caster's
// density above 0.
static const float AIM_EXPLORE = 0.1;

// Aim cells: a cluster's center at distance d m from the focus takes the voxel of edge
// 2^(AIM_BASE_LOG2 + level) m, level = floor(log2(d)), from 1/16 to 1/8 of d, at least 1/8 m.
// A plane through the focus then takes about 600 cells per doubling of distance, so AIM_CELLS
// hold casters at every distance the levels reach; a cluster that finds no room weighs as a new
// cell does and counts no photons.
static const int AIM_BASE_LOG2 = -3;

// The share of a voxel's edge aim cells lie offset by: 2 - the golden ratio.
static const float AIM_OFFSET = 0.381966;

// Frames a cell outlives the last caster list that held it.
static const uint AIM_LIFETIME = 32;

// An empty cell: no photons counted and no yield measured, which a negative yield (f32 bits)
// marks. A caster no photon has measured weighs by the source's `aimed`: a cluster's center can
// cross into a voxel it never held as the caster moves, as a wave does, and an optimistic yield
// there would draw photons from where they were measured to show nothing.
static const uint4 AIM_EMPTY = uint4(0, 0, 0xbf800000, 0);

static const uint AIM_GROUP = 256;

// The least share of its photons a frame starts: a caustic coming into view converges as soon
// under a 64th or a quarter, but a 64th finds its first deposits frames later, and a quarter
// starts more photons than the caustics in view ask for.
static const float PHOTON_FLOOR = 1.0 / 16.0;

static const uint GROUP = 1024;

groupshared float scan_w[GROUP];
groupshared uint scan_c[GROUP];
groupshared float2 block_sums[BLOCK_CLUSTERS];

uint scan_uint(uint lane, uint v, out uint total)
{
    scan_c[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < GROUP; off <<= 1)
    {
        uint add = lane >= off ? scan_c[lane - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        scan_c[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = scan_c[lane];
    total = scan_c[GROUP - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float scan_float(uint lane, float v, out float total)
{
    scan_w[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < GROUP; off <<= 1)
    {
        float add = lane >= off ? scan_w[lane - off] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        scan_w[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    float r = scan_w[lane];
    total = scan_w[GROUP - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// The sum of `v` over a workgroup of BLOCK_CLUSTERS lanes, in every lane.
float2 block_total(uint lane, float2 v)
{
    block_sums[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = BLOCK_CLUSTERS / 2; width > 0; width >>= 1)
    {
        if (lane < width)
            block_sums[lane] += block_sums[lane + width];
        GroupMemoryBarrierWithGroupSync();
    }
    float2 r = block_sums[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// One thread per aim cell: the last frame's photons folded into its yield, its counts started
// over, and its eviction once no caster list has held it for AIM_LIFETIME frames, which leaves a
// tombstone. A clear empties every cell.
[shader("compute")]
[numthreads(AIM_GROUP, 1, 1)]
void aim(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(aim);
    uint i = id.x;
    if (root.clear != 0)
    {
        root.aim_checks[i] = 0;
        root.aim[i] = AIM_EMPTY;
        return;
    }
    uint check = root.aim_checks[i];
    if (check == 0 || check == CACHE_TOMBSTONE)
        return;
    uint4 c = root.aim[i];
    float y = asfloat(c.z);
    if (c.x > 0)
    {
        float frame = float(c.y) / float(c.x);
        y = y < 0.0 ? frame : lerp(frame, y, SOURCE_YIELD_HISTORY);
    }
    if (c.w >= AIM_LIFETIME)
    {
        root.aim_checks[i] = CACHE_TOMBSTONE;
        root.aim[i] = AIM_EMPTY;
    }
    else
        root.aim[i] = uint4(0, 0, asuint(y), c.w + 1);
}

// One thread per aim cell, after `aim`: empties the tombstones no key lies past.
[shader("compute")]
[numthreads(AIM_GROUP, 1, 1)]
void aim_sweep(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(aim_sweep);
    uint i = id.x;
    if (root.aim_checks[i] == CACHE_TOMBSTONE && sweeps(root.aim_checks, AIM_CELLS - 1, i))
        root.aim_checks[i] = 0;
}

// The key of the aim cell of a caster cluster centered at render-space `p`. Producers and cooks
// lay clusters on power-of-two grids as the voxels lie, which puts many a center on a voxel's
// face, where the least motion, a wave's, takes it across to a cell that has measured nothing;
// the voxels lie offset by AIM_OFFSET of their edge, which no such grid takes.
uint2 aim_key(SceneRoot* s, float3 p)
{
    float d = length(p - root.focus.xyz);
    float top = float(clamp(s.origin_cell.w - AIM_BASE_LOG2, 0, int(CACHE_MAX_LEVEL)));
    uint level = uint(clamp(floor(log2(d)), 0.0, top));
    int e = AIM_BASE_LOG2 + int(level);
    return cache_key(s.origin_cell, p + AIM_OFFSET * exp2(float(e)), e, level, 0);
}

// Whether producer entry `producer` is traced this frame: only what rays meet can cast.
bool producer_traced(SceneRoot* s, uint producer)
{
    ProducerGpu p = s.producers[producer];
    return p.strands == 0 && (p.flags & INSTANCE_RAY) != 0
        && (s.producer_state[producer].rt_flags & PRODUCER_RT_WRITTEN) != 0;
}

// The caster list of instance entry `e`, its copy's mesh `mesh`: the cooked set of the row its
// copy traces, or the list its traversal gathered.
EmissiveSelection instance_casters(SceneRoot* s, uint e, MeshGpu* mesh)
{
    EmissiveGpu g = s.emissive[e];
    EmissiveSelection sel = EmissiveSelection();
    sel.count = g.cluster_count;
    if (g.set == GATHERED_SET)
    {
        uint* list = gathered_list(s.emissive_lists, e);
        sel.clusters = list;
        sel.cdf = list + GATHERED_CLUSTERS;
        return sel;
    }
    SetTablesGpu t = mesh.casters;
    ClusterSet k = ((ClusterSet*)t.sets)[g.set];
    sel.cdf = t.cdf + k.first;
    sel.clusters = t.clusters + k.first;
    sel.index = t.index + k.first;
    return sel;
}

// The copy and mesh of instance entry `i` where it is a caster section's that casts this frame.
bool instance_caster(SceneRoot* s, uint i, out InstanceGpu inst, out MeshGpu* mesh)
{
    EmissiveGpu g = s.emissive[i];
    inst = InstanceGpu();
    mesh = nullptr;
    if (g.kind != EMISSIVE_INSTANCE || g.set == NO_SET || g.cluster_count == 0 || !(g.area > 0.0))
        return false;
    inst = traced_record(s, g.source);
    mesh = s.meshes + s.models[inst.model].mesh;
    return g.rank >= mesh.emissive.count;
}

// The caster list of entry `i` this frame, its CDF aside: a traced producer's emitted clusters,
// a caster section's cooked set or gathered list; empty for any other entry.
EmissiveSelection caster_list(SceneRoot* s, uint i)
{
    EmissiveGpu g = s.emissive[i];
    EmissiveSelection sel = EmissiveSelection();
    InstanceGpu inst;
    MeshGpu* mesh;
    if (g.kind == EMISSIVE_PRODUCER && producer_traced(s, g.source))
        sel.count = emitted_clusters(s.batches[g.source]);
    else if (instance_caster(s, i, inst, mesh))
        sel = instance_casters(s, i, mesh);
    return sel;
}

// The caster entries' places in the pool and their clusters' blocks, one workgroup: the
// producers' in theirs, the copies' after them in the order of the emissive list while they
// fit. A copy's list that does not fit keeps its cooked CDF, over its clusters' areas, and the
// pool none of it (AIM_NO_LIST); the positions all of them ask for go to the source, which
// the next frames' pools grow to.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void pool(uint lane: SV_GroupIndex)
{
    ORR_OMM(pool);
    SceneRoot* s = root.scene;
    uint n = s.emissive_capacity;
    uint carry_producers = 0;
    uint carry_copies = 0;
    uint carry_blocks = 0;
    for (uint base = 0; base < n; base += GROUP)
    {
        uint i = base + lane;
        EmissiveSelection e = EmissiveSelection();
        bool producer = false;
        if (i < n)
        {
            e = caster_list(s, i);
            producer = s.emissive[i].kind == EMISSIVE_PRODUCER;
        }
        uint of_producer = producer ? e.count : 0;
        uint of_copy = producer ? 0 : e.count;
        uint chunk;
        uint first_producer = carry_producers + scan_uint(lane, of_producer, chunk) - of_producer;
        carry_producers += chunk;
        uint first_copy = carry_copies + scan_uint(lane, of_copy, chunk) - of_copy;
        carry_copies += chunk;
        bool pooled = producer || first_copy + of_copy <= root.list_room;
        uint blocks = pooled ? cluster_blocks(e.count) : 0;
        uint first_block = carry_blocks + scan_uint(lane, blocks, chunk) - blocks;
        carry_blocks += chunk;
        if (i < n)
        {
            set_first_block(root.block_layout, i, first_block);
            uint first = producer ? first_producer : root.producer_room + first_copy;
            if (pooled)
                e.cdf = root.pool + first;
            root.entries[i] = e;
            root.firsts[i] = pooled ? first : AIM_NO_LIST;
        }
    }
    if (lane == 0)
    {
        close_layout(root.block_layout, n, carry_blocks);
        root.source.list_wanted = carry_copies;
    }
}

// Render-space area of cluster `c` of producer entry `producer`'s batch where its material
// casts, and in `center` its render-space centroid; 0 elsewhere.
float producer_caster_area(SceneRoot* s, uint producer, uint c, out float3 center)
{
    center = float3(0.0);
    ProducerGpu p = s.producers[producer];
    ProducerBatch b = s.batches[producer];
    ProducerClusterGpu k = b.clusters[c];
    uint material = producer_material(p, k);
    Placement pl;
    if (material == MATERIAL_NONE || (s.materials[material].flags & MATERIAL_CASTER) == 0
        || !producer_placement(s, p, false, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return 0.0;
    float3 centroid;
    float area = cluster_area(b, k, pl.linear, centroid);
    center = place(pl, centroid);
    return area;
}

// Render-space area of position `c` of caster entry `i`'s cooked set or gathered list, whose
// CDF runs over its clusters' areas, and in `center` the center of its cluster's bounds as its
// copy places them; 0 for a position of no area.
float instance_caster_area(SceneRoot* s, uint i, uint c, out float3 center)
{
    center = float3(0.0);
    InstanceGpu inst;
    MeshGpu* mesh;
    Placement pl;
    if (!instance_caster(s, i, inst, mesh)
        || !placement(s.groups + inst.group, false, inst.group_index, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return 0.0;
    EmissiveSelection e = instance_casters(s, i, mesh);
    uint instance_index, cluster_id, triangles, alpha_first;
    if (!list_cluster(s, e, i, c, instance_index, cluster_id, triangles, alpha_first))
        return 0.0;
    ClusterRef r = s.clusters[cluster_id];
    Cluster k = load_cluster(r.page, r.local);
    // A part instance's clusters lie where its pose puts them, as traced_triangle places them.
    uint copy;
    float3x4 pose;
    if (instance_index >= s.part_base && hit_copy(s, instance_index, copy) && (copy & COPY_FAR) == 0)
    {
        if (!part_pose(s, inst, mesh, k.part, false, pose))
            return 0.0;
        pl = placed_after(pl, pose);
    }
    center = place(pl, k.cull_sphere.xyz);
    return list_step(e, c) * s.emissive[i].area;
}

// The weight of caster entry `i`, a copy's whose list the pool had no room for, and in `mass`
// its casters' area by focus falloff and that by `aimed`: its cooked CDF draws its clusters by
// area, so the entry weighs as one caster at its model's center whose yield is `aimed`.
float cooked_weight(SceneRoot* s, uint i, float aimed, out float2 mass)
{
    mass = float2(0.0);
    InstanceGpu inst;
    MeshGpu* mesh;
    Placement pl;
    if (!instance_caster(s, i, inst, mesh)
        || !placement(s.groups + inst.group, false, inst.group_index, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return 0.0;
    ModelGpu* model = s.models + inst.model;
    float d = length(place(pl, 0.5 * (model.bounds_min + model.bounds_max)) - root.focus.xyz) / root.focus.w;
    mass.x = s.emissive[i].area / (1.0 + d * d);
    mass.y = mass.x * aimed;
    return mass.x * aimed * (1.0 + AIM_EXPLORE);
}

// The caster lists, with the weight of the lane's own position, which `lists_weigh` gives
// before `weigh_blocks` asks for it.
struct CasterLists : IClusterLists
{
    float lane_weight;

    uint* list(uint i, out uint count)
    {
        EmissiveSelection e = root.entries[i];
        count = e.count;
        return e.cdf;
    }

    float weight(uint i, uint c)
    {
        return lane_weight;
    }
}

// One workgroup per block of the caster lists' clusters: each cluster's aim cell, and its weight,
// its caster area by focus falloff times its aim cell's yield plus AIM_EXPLORE of the source's
// `aimed`, summed within the block, with what the block's clusters add to `masses`. A cluster
// whose cell has measured nothing, or that finds no cell, takes `aimed` as its yield.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void lists_weigh(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    ORR_OMM(lists_weigh);
    SceneRoot* s = root.scene;
    uint b = grid_index(group);
    uint i, j;
    if (!block_entry(root.block_layout, s.emissive_capacity, b, i, j))
        return;
    uint count = root.entries[i].count;
    uint c = j * BLOCK_CLUSTERS + lane;
    float aimed = root.clear != 0 ? 1.0 : root.source.aimed;
    float2 mass = float2(0.0);
    CasterLists lists = { 0.0 };
    if (c < count)
    {
        EmissiveGpu g = s.emissive[i];
        float3 center;
        float area = g.kind == EMISSIVE_PRODUCER ? producer_caster_area(s, g.source, c, center) : instance_caster_area(s, i, c, center);
        uint cell = AIM_NONE;
        if (area > 0.0)
        {
            float d = length(center - root.focus.xyz) / root.focus.w;
            float y = -1.0;
            if (slot_insert(root.aim_checks, AIM_CELLS - 1, aim_key(s, center), cell))
            {
                y = asfloat(root.aim[cell].z);
                root.aim[cell].w = 0;
            }
            else
                cell = AIM_NONE;
            if (y < 0.0)
                y = aimed;
            mass.x = area / (1.0 + d * d);
            mass.y = mass.x * y;
            lists.lane_weight = mass.x * (y + AIM_EXPLORE * aimed);
        }
        root.pool_cells[root.firsts[i] + c] = cell;
    }
    weigh_blocks(lists, root.block_layout, root.blocks, s.emissive_capacity, b, lane);
    float2 total = block_total(lane, mass);
    if (lane == 0)
        root.block_masses[b] = total;
}

// One workgroup per entry: its blocks' places in its list, its list's weight and its `masses`.
// A list of no weight is left empty.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void lists_place(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    ORR_OMM(lists_place);
    uint i = grid_index(group);
    if (i >= root.scene.emissive_capacity || root.entries[i].count == 0 || root.firsts[i] == AIM_NO_LIST)
        return;
    float total = place_blocks(root.block_layout, root.blocks, i, lane);
    uint first, end;
    entry_blocks(root.block_layout, i, first, end);
    float2 mass = float2(0.0);
    for (uint b = first + lane; b < end; b += BLOCK_CLUSTERS)
        mass += root.block_masses[b];
    mass = block_total(lane, mass);
    if (lane == 0)
    {
        root.weights[i] = total;
        root.masses[i] = mass;
        if (!(total > 0.0))
            root.entries[i].count = 0;
    }
}

// One workgroup per block of the caster lists' clusters: their fixed-point CDF.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void lists_finish(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    ORR_OMM(lists_finish);
    CasterLists lists = { 0.0 };
    finish_blocks(lists, root.block_layout, root.blocks, root.scene.emissive_capacity, grid_index(group), lane);
}

// The entry CDF over the entries' caster lists' weights, the frame's photon count, the casters'
// next `aimed` and the photon source record, one workgroup.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void weigh(uint lane: SV_GroupIndex)
{
    ORR_OMM(weigh);
    SceneRoot* s = root.scene;
    uint n = s.emissive_capacity;
    float carry = 0.0;
    float2 mass_sum = float2(0.0);
    float aimed_before = root.clear != 0 ? 1.0 : root.source.aimed;
    for (uint base = 0; base < n; base += GROUP)
    {
        uint i = base + lane;
        float w = 0.0;
        float2 mass = float2(0.0);
        if (i < n && root.entries[i].count > 0)
        {
            if (root.firsts[i] == AIM_NO_LIST)
                w = cooked_weight(s, i, aimed_before, mass);
            else
            {
                w = root.weights[i];
                mass = root.masses[i];
            }
        }
        float chunk;
        float cdf = carry + scan_float(lane, w, chunk);
        if (i < n)
            root.cdf[i] = cdf;
        carry += chunk;
        float sum;
        scan_float(lane, mass.x, sum);
        mass_sum.x += sum;
        scan_float(lane, mass.y, sum);
        mass_sum.y += sum;
    }
    if (lane != 0)
        return;
    PhotonSourceGpu* p = root.source;
    p.set = root.set;
    p.entries = root.entries;
    p.cdf = root.cdf;
    p.firsts = root.firsts;
    p.cells = root.pool_cells;
    p.interiors = root.interiors;
    p.aim = root.aim;
    p.entry_count = n;
    p.total = carry;

    // As many photons as `cell_photons` in each cell the view read take, by the share of the
    // frames before's that deposited where it read, both of which the grid's `finish` keeps; a
    // clear starts them over. At least PHOTON_FLOOR of the photons find caustics coming into
    // view, and at least one. The yield halves each frame no photon deposits where the view
    // reads, down to 0, while a view that reads no cell wants none.
    // A cell a read first found fewer frames ago than a read keeps it read has had too few
    // photons counted where the view reads to tell what it needs: it asks for its share of
    // every photon, as all do until the grid has resolved that many frames since it emptied.
    if (root.clear != 0)
    {
        p.yield = 1.0;
        p.read = 0;
        p.fresh = 0;
        p.measured = 0;
    }
    uint measured = p.read - p.fresh;
    float wanted = p.read == 0 ? 0.0
        : ceil((measured > 0 ? root.cell_photons * float(measured) / p.yield : 0.0) + float(root.photons) * float(p.fresh) / float(p.read));
    p.count = p.measured < CAUSTIC_READ_FRAMES
        ? root.photons
        : uint(clamp(wanted, ceil(float(root.photons) * PHOTON_FLOOR), float(root.photons)));
    // With no yield above 0 anywhere every caster's photons go by area and falloff.
    float aimed = mass_sum.y / mass_sum.x;
    p.aimed = aimed > 0.0 ? aimed : 1.0;

    // The grid's finest level: the one the last resolve's `finish` chose, which starts over with
    // the grid, within the levels the origin's cell allows.
    uint top = caustic_max_level(s, root.base_cell_log2);
    if (root.clear != 0)
    {
        p.next_finest = 0;
        p.held = 0;
        p.absent = 0.0;
        p.empty = 0.0;
    }
    p.finest = min(p.next_finest, top);
    p.top = top;
}

// A point on light `id` (a feedback id) to walk out from, and the way out: a sphere, point or
// rect light's center, an emissive entry's first triangle's centroid on the side it emits, nothing
// for directional lights and the environment, which lie outside every interior.
bool light_point(SceneRoot* s, LightSetGpu* set, uint id, out float3 p, out float3 dir)
{
    p = float3(0.0);
    dir = float3(0.0, 1.0, 0.0);
    if (id < s.light_capacity)
    {
        LightGpu l = s.lights[id];
        p = l.position;
        return l.kind == LIGHT_SPHERE || l.kind == LIGHT_POINT || l.kind == LIGHT_RECT;
    }
    uint entry = id - s.light_capacity;
    if (entry >= s.emissive_capacity)
        return false;
    EmissiveSelection e = set.emissive[entry];
    uint instance_index, cluster_id, triangles, alpha_first;
    TracedTriangle t;
    if (e.count == 0 || !list_cluster(s, e, entry, 0, instance_index, cluster_id, triangles, alpha_first) || triangles == 0
        || !traced_triangle(s, instance_index, cluster_id, 0, false, t))
        return false;
    float3 n;
    if (!(traced_area(t, n) > 0.0))
        return false;
    dir = n;
    p = traced_point(t, float2(1.0 / 3.0));
    return true;
}

// One thread per feedback id: the interiors that hold the light, from the surfaces a ray from it
// crosses on its way out, every back face of an interior the walk has not entered showing one.
[shader("compute")]
[numthreads(64, 1, 1)]
void interiors(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(interiors);
    SceneRoot* s = root.scene;
    uint n = s.light_capacity + s.emissive_capacity + 1;
    if (id.x >= n)
        return;
    float3 origin;
    float3 dir;
    Interior inside = Interior();
    if (light_point(s, root.set, id.x, origin, dir))
    {
        Interior walked = Interior();
        RayCone cone = { 0.0, 0.0 };
        HitContext context = hit_context(HIT_SECONDARY, 0, cone, 0.0);
        LayerAccept<Surface> a = { s, nullptr, dir, context, 0, PRESENCE_NONE };
        float at = 0.0;
        for (uint k = 0; k < SURFACE_CROSSINGS; k++)
        {
            RayDesc ray;
            ray.Origin = origin + dir * at;
            ray.Direction = dir;
            ray.TMin = 1e-4 * (1.0 + at);
            ray.TMax = positive_infinity();
            float3 positions[3];
            Hit h = trace_closest(scene_traced(s), ray, MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
            float3 normal;
            uint m;
            if (!h.hit || h.cluster_id < 0 || !hit_geometry(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, normal, m))
                break;
            at += h.t;
            if (h.class_id != CLASS_DIELECTRIC || !crosses_interior(s, m))
                continue;
            bool front = dot(normal, dir) < 0.0;
            if (front || walked.holds(m))
                walked.cross(m, front);
            else
                inside.nest(m, walked, NO_MEDIUM_POINT);
        }
    }
    root.interiors[id.x] = EyeInterior(inside);
}
