// The scene tables every pass reads through one `SceneRoot` pointer.
module tables;

import orr.abi;
__exported import cut;
import material;
import lights;
import producer;
import view;

// `InstanceGpu.flags`.
public static const uint INSTANCE_RASTER = 1;
public static const uint INSTANCE_RAY = 2;
public static const uint INSTANCE_CAST_ONLY = 4;
// Set by orr on copies from `add_skinned`: `pose_group` is the palette group and
// `pose_index` the copy's entry in `SceneRoot.skinned`.
public static const uint INSTANCE_SKINNED = 8;

// `InstanceGpu.material_base` when the copy uses its model's section materials.
public static const uint NO_MATERIAL_SET = 0xffffffff;
// `InstanceGpu.first_part` when the copy holds no part run.
public static const uint NO_PART_RUN = 0xffffffff;

// `GroupGpu.format`.
public static const uint GROUP_AFFINE3X4_F32 = 0;
public static const uint GROUP_QUANTISED_PAGE = 1;

// One copy of a model in an instance slot. A slot whose flags are 0 is empty.
public struct InstanceGpu
{
    public uint group;         // transform group; 0 is the null group
    public uint group_index;   // the copy's transform in its group
    public uint model;
    public uint material_base; // in the scene's material-set table, or NO_MATERIAL_SET
    public uint pose_group;    // 0: no animated parts
    public uint pose_index;
    public uint first_part;    // the copy's part run, or NO_PART_RUN
    public half rt_range;      // m from the scene focus
    public half animation_range;
    public uint8_t flags;      // INSTANCE_*
    public uint8_t candidate_class; // orr.trace CLASS_*
    public uint16_t pose_variants;
    public uint16_t pose_stride;
    public uint16_t debug_lod; // 0 none; DEBUG_LOD_LEVEL | level; else the f16 bits of an error threshold scale
    public uint user_tag;      // never read by orr
    public uint variant;       // pose u8, material set u8, section mask u16 (bit set = hidden)
}

// `InstanceGpu.debug_lod` holding the copy at one DAG level, in its low byte.
public static const uint DEBUG_LOD_LEVEL = 0x8000;

// `SlotStateGpu.rt_flags`.
public static const uint RT_WRITTEN = 1;    // its PTLAS instance holds a BLAS
public static const uint RT_GLOBAL = 2;     // written into the global partition
public static const uint RT_IN_RANGE = 4;   // inside rt_range at the last decision
public static const uint RT_SEEN = 8;       // the PTLAS kernel has run over the slot
public static const uint RT_ANIMATED = 16;  // inside animation_range: raster takes the pose
public static const uint RT_WAS_ANIMATED = 32; // RT_ANIMATED of the previous frame

// Per-slot state written only by the GPU. Handing out a slot zeroes every field before
// `part_run`; a part run outlives its slot until the PTLAS kernels release it. The `rt_`
// fields describe what the slot's PTLAS instance holds.
public struct SlotStateGpu
{
    public uint8_t rt_lod;          // the mesh cut it was written or updated with, RT_CUT_BUILT for a BLAS built that frame
    public uint8_t stable_frames;   // frames since its transform last moved, saturating
    public uint8_t chain_step;
    public uint8_t fade;
    public uint last_visible_frame; // the last frame a view drew one of its clusters
    public uint64_t rt_blas;        // 0 while null-written
    public uint rt_partition;       // cell partition, or the partition count for the global one
    public uint rt_flags;           // RT_*
    public uint2 rt_key;            // hash of the transform, model and class it was written with
    public uint rt_age;             // frames its pending PTLAS work has waited
    public uint _pad;
    public uint part_run;           // first part instance of the run it holds
    public uint part_count;         // part instances in the run; 0: none
}

// A transform group: `count` transforms relative to the group's anchor. A null `current`
// or an all-zero transform hides the instances that use it.
public struct GroupGpu
{
    public uint8_t* current;
    public uint8_t* previous;  // last frame's transforms; equals `current` when unchanged
    public uint count;
    public uint format;        // GROUP_*
    public int3 anchor_cell;
    public float3 anchor_offset;
    public float3 origin;      // GROUP_QUANTISED_PAGE: cell-relative page box
    public float3 extent;
    public float log2_scale_min;
    public float log2_scale_max;
}

// A discrete level of a mesh: its error and its range of mesh cluster indices.
public struct MeshLevel
{
    public float error;            // m, mesh space
    public uint first;
    public uint count;
}

// A resident cluster, named by the cluster id of its CLAS.
public struct ClusterRef
{
    public uint8_t* page;
    public uint local;             // index in the page
    public uint mesh_cluster;      // mesh cluster index
    public uint64_t clas;          // 0 where the device builds none
}

// One cut of a mesh that ray tracing can use: the cluster BLAS over its clusters, the
// largest error among them, and for a cut with animated part clusters the BLAS of its body
// (the clusters without CLUSTER_ANIMATED_PART) and of each animated part.
public struct RtCutGpu
{
    public uint64_t blas;          // every cluster at rest pose
    public float error;            // m, mesh space
    public uint clusters;
    public uint64_t body;          // `blas` when the cut has no animated part clusters; 0 when its body is empty
    public uint64_t* parts;        // per animated part, 0 where it has no clusters; null without animated part clusters
    public uint2 hash;             // cut_hash over its clusters
}

// An animated part: its rest bounds in mesh space and the sections its clusters use.
public struct PartGpu
{
    public float3 bounds_min;
    public uint sections;          // bit per section below 32
    public float3 bounds_max;
    public uint _pad;
}

// The BLAS of a mesh with some sections hidden: `blas[cut]` over every remaining cluster,
// `blas[rt_cut_count + cut]` over the remaining body clusters; 0 where none remain.
public struct MaskGpu
{
    public uint mask;              // section bits, below the mesh's section count and 16
    public uint _pad;
    public uint64_t* blas;
}

// A mesh made resident from a mounted pack. Page and node page addresses are 0 until
// their data is resident.
public struct MeshGpu
{
    public uint8_t* info;          // blob 0: MeshHeader and its tables (orr_pack.mesh)
    public uint64_t* pages;        // resident page per MeshHeader page
    public uint64_t* node_pages;   // node page per MeshHeader node page
    public uint* page_clusters;    // first mesh cluster index per page, then the cluster count
    public MeshLevel* levels;      // finest first; none in DAG mode
    public uint page_count;
    public uint node_page_count;
    public uint cluster_count;
    public uint level_count;       // 0: DAG mode
    public float position_step;    // m: the position grid step
    public uint attributes;        // orr_pack.mesh MESH_*
    public float merge_error;      // DAG mode: the error from which the merged stage serves; infinity without one
    public uint cluster_base;      // cluster id of mesh cluster 0
    public uint rt_cut_count;      // 0: not traced
    public uint part_count;        // transforms per pose block: the mesh's parts, at least 1
    public RtCutGpu* rt_cuts;      // finest first
    public PartGpu* parts;         // per animated part
    public MaskGpu* masks;         // section-mask BLAS, `mask_count` entries
    public uint first_animated;    // the first animated part; animated parts follow it
    public uint animated_parts;    // 0: no part is animated
    public uint mask_count;
    public uint animated_sections; // the sections of every animated part, bit per section below 32
    public uint8_t* nodes;         // DAG mode: the Node table of a resident hierarchy, else null
    public uint8_t* node_table;    // DAG mode: the NodePage table of a paged hierarchy, else null
    public uint8_t* groups;        // DAG mode: the Group table
    public uint* roots;            // DAG mode: MeshHeader.roots
    public uint* node_stamps;      // per node page: the last frame the traversal reported it
    public uint root_count;
    public uint _pad;
    public SkinMeshGpu* skin;      // MESH_SKIN meshes: what skinned copies build from, else null
}

// What skinned copies of a MESH_SKIN mesh read: each cluster's place in a copy's vertex
// arrays, its cluster template, and the cuts a copy takes, finest first.
public struct SkinMeshGpu
{
    public uint* vertex_first;     // per mesh cluster: its first vertex in a copy's arrays, then the total
    public uint64_t* templates;    // per mesh cluster: its cluster template; 0 without cluster AS
    public uint* cut_clusters;     // the mesh clusters of each cut, ascending, cut after cut
    public uint* cut_first;        // per cut: its first entry in `cut_clusters`, then the total
    public float* cut_error;       // per cut: the largest error of its clusters, m, mesh space
    public uint cut_count;
    public uint vertex_count;
    public float bound;            // the largest deformation bound of its clusters, m; 0: unbounded
    public uint _pad;
}

// A skinned copy (`add_skinned`): its vertex arrays, rewritten every frame its scene is
// updated, and where its joint palette lives. An unused entry is all zero.
public struct SkinnedGpu
{
    public float3* positions;      // per vertex: skinned by this frame's palette, mesh space
    public float3* previous;       // per vertex: skinned by last frame's palette
    public uint* normals;          // per vertex: unit normal, oct 2x16 unorm
    public uint* tangents;         // per vertex: unit tangent, oct 2x16 unorm with the bitangent sign in bit 16
    public float* boxes;           // per mesh cluster: min xyz, max xyz of its skinned vertices
    public uint slot;
    public uint palette_group;
    public uint palette_first;     // the palette entry of joint 0
    public uint joint_count;
}

// `SkinStateGpu.cut` before a copy has taken one.
public static const uint SKIN_NO_CUT = 0xffffffff;

// What the frame's skinning made of a skinned copy, written by the GPU.
public struct SkinStateGpu
{
    public uint3 lo;               // bounds of its skinned clusters, mesh space, as `ordered` floats
    public uint cut;               // the cut it takes, or SKIN_NO_CUT
    public uint3 hi;
    public uint clusters;          // clusters skinned this frame; 0: neither drawn nor traced
    public uint64_t blas;          // its cluster BLAS this frame; 0: not traced
    public uint first;             // its first skinning work item
    public uint job;               // its BLAS build record
}

// A model made resident from a mounted pack: its mesh and the material of each section.
public struct ModelGpu
{
    public uint mesh;            // in SceneRoot.meshes
    public uint section_count;
    public uint* materials;      // MaterialId per mesh section
    public float3 bounds_min;
    public float3 bounds_max;
}

public struct SceneRoot
{
    public InstanceGpu* instances;
    public SlotStateGpu* slot_state;
    public GroupGpu* groups;
    public ModelGpu* models;         // global, by ModelId
    public MeshGpu* meshes;          // global
    public MaterialGpu* materials;   // global
    public ClusterRef* clusters;     // global, by cluster id
    public uint* material_sets;      // MaterialId per model section per set
    public LightGpu* lights;
    public EmissiveGpu* emissive;    // producer_capacity + emissive_capacity entries
    public ProducerGpu* producers;   // by ProducerId
    public ProducerBatch* batches;   // by ProducerId
    public ProducerStateGpu* producer_state; // by ProducerId, written by the GPU
    public SkyStats* sky_stats;
    public uint64_t environment;     // consumer data for the linked environment
    public uint64_t medium;          // consumer data for the linked medium
    public uint64_t surface_data;    // consumer data for linked surfaces whose material has none
    public Tlas tlas;                // the scene's PTLAS, or the device's empty TLAS
    public uint* part_slots;         // per part instance: the slot of the copy holding it
    public int4 origin_cell;         // render origin cell, w = cell_log2
    public float delta_time;
    public uint frame_index;
    public uint instance_capacity;
    public uint group_capacity;
    public uint light_capacity;
    public uint material_set_capacity;
    public uint producer_capacity;
    public uint emissive_capacity;   // instance span entries, after the producer entries
    public float near_range;         // m
    public uint part_base;           // PTLAS index of part instance 0
    public uint part_capacity;
    public uint producer_base;       // PTLAS index of producer entry 0; also how `SurfaceInput`, `GSurface` and `VisibleCluster` name producer entries
    public CutGpu* cut;              // the frame's cuts
    public SkinnedGpu* skinned;      // skinned copies, `skinned_capacity` entries
    public SkinStateGpu* skin_state; // per skinned copy, written by the GPU
    public uint2* skin_work;         // this frame's skinning work items: skinned copy, mesh cluster
    public uint skinned_capacity;
    public uint _pad;
}

// The page of `mesh` holding mesh cluster `cluster`.
public uint cluster_page(MeshGpu* mesh, uint cluster)
{
    uint lo = 0;
    uint hi = mesh.page_count;
    while (hi - lo > 1)
    {
        uint mid = (lo + hi) / 2;
        if (mesh.page_clusters[mid] <= cluster)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// Transform `index` of `group` as a row-major [R | t] relative to the group's anchor.
public float3x4 group_transform(GroupGpu* group, uint index)
{
    return ((float3x4*)group.current)[index];
}

public float3x4 previous_group_transform(GroupGpu* group, uint index)
{
    return ((float3x4*)group.previous)[index];
}

// An order-free hash of a set of mesh clusters: the sum of two hashes of each.
public uint2 cluster_hash(uint mesh_cluster)
{
    uint a = mesh_cluster * 0x9e3779b1u + 0x7f4a7c15u;
    a ^= a >> 16;
    a *= 0x85ebca6bu;
    a ^= a >> 13;
    a *= 0xc2b2ae35u;
    a ^= a >> 16;
    uint b = (mesh_cluster ^ 0x5bd1e995u) * 0x27d4eb2fu;
    b ^= b >> 15;
    b *= 0x165667b1u;
    b ^= b >> 13;
    return uint2(a, b);
}

// The translation from `eye` to the point at `cell` + `offset`, exact in its cell part for
// every cell delta below 2^(24 - cell_log2) cells.
public float3 view_relative(int3 cell, float3 offset, int3 eye_cell, float3 eye_offset, uint cell_log2)
{
    return float3((cell - eye_cell) << cell_log2) + (offset - eye_offset);
}
