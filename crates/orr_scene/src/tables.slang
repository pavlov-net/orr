// The scene tables every pass reads through one `SceneRoot` pointer.
module tables;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.trace;
import orr_pack.instance;
__exported import orr_scene.cut;
__exported import orr_scene.decal;
__exported import orr_scene.medium_proxy;
import orr_scene.material;
import orr_scene.lights;
import orr_scene.producer;
import orr_scene.scene_may;
import orr_scene.view;

// `InstanceGpu.flags`.
public static const uint INSTANCE_RASTER = 1;
public static const uint INSTANCE_RAY = 2;
public static const uint INSTANCE_CAST_ONLY = 4;
// Set by orr on copies from `add_skinned`: `pose_group` is the palette group and
// `pose_index` the copy's entry in `SceneRoot.skinned`.
public static const uint INSTANCE_SKINNED = 8;

// Whether candidate class `c` is a dielectric or presence layer's: its clusters raster into
// the layer tiles instead of the visibility buffer.
public bool layer_class(uint c)
{
    return c == CLASS_DIELECTRIC || c == CLASS_PRESENCE;
}

// Whether a dielectric or presence copy with instance flags `flags` may meet camera rays,
// which trace every one the TLAS holds.
public bool layer_traced(uint flags)
{
    return (flags & (INSTANCE_RASTER | INSTANCE_RAY)) != 0;
}

// Whether the opaque clusters of a copy with instance flags `flags` raster.
public bool opaque_rastered(uint flags)
{
    return (flags & INSTANCE_RASTER) != 0 && (flags & INSTANCE_CAST_ONLY) == 0;
}

// Whether a cluster of class `c` of a copy with instance flags `flags` raster.
public bool class_rastered(uint c, uint flags)
{
    return layer_class(c) ? layer_traced(flags) : opaque_rastered(flags);
}

// `InstanceGpu.material_base` when the copy uses its model's section materials.
public static const uint NO_MATERIAL_SET = 0xffffffff;
// `InstanceGpu.first_part` when the copy holds no part run.
public static const uint NO_PART_RUN = 0xffffffff;
// `InstanceGpu.first_emissive` of a copy with no emissive list entries.
public static const uint NO_EMISSIVE = 0xffffffff;

// `GroupGpu.format`.
public static const uint GROUP_AFFINE3X4_F32 = 0;
public static const uint GROUP_QUANTIZED_PAGE = 1;

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
    public uint8_t _pad;
    public uint16_t pose_variants;
    public uint16_t pose_stride;
    public uint16_t debug_lod; // 0 none; DEBUG_LOD_LEVEL | level; else the f16 bits of an error threshold scale
    public uint user_tag;      // never read by orr
    public uint variant;       // pose u8, material set u8, section mask u16 (bit set = hidden)
    public uint first_emissive; // in SceneRoot.emissive, or NO_EMISSIVE
    // Entries from it: one per emissive section rank of its model, then one per caster rank.
    public uint emissive_count;
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
public static const uint RT_FADING = 64;    // its PTLAS instance was written for a chain fade
public static const uint RT_STEP_SHIFT = 8; // 2 bits: the chain step its PTLAS instance holds
public static const uint RT_FAR_CUT_SHIFT = 16; // 8 bits: the cut its far instance holds

// `SlotStateGpu.chain_step`: the step a copy is at, or fades to, in bits 0-1; the step it
// fades from in bits 2-3 while `fade` is not 0.
public static const uint CHAIN_STEP_MASK = 3;
public static const uint CHAIN_FROM_SHIFT = 2;
public static const uint CHAIN_SEEN = 0x80; // a step has been decided for the copy

// A model and up to 3 farther models.
public static const uint CHAIN_STEPS = 4;
// Frames a step change cross-fades over.
public static const uint CHAIN_FADE_FRAMES = 16;
public static const float CHAIN_HYSTERESIS = 1.15;

// An instance slot with this bit names the slot's farther chain step while it fades between
// two; without it, its nearer step, its only one outside a fade. Visible clusters, surface
// inputs and resolved hits name copies this way.
public static const uint COPY_FAR = 0x80000000;

// Per-slot state written only by the GPU. Handing out a slot zeroes every field before
// `part_run`; part runs outlive their slot until the PTLAS kernels release them. The `rt_`
// fields describe what the slot's PTLAS instance holds.
public struct SlotStateGpu
{
    public uint8_t rt_lod;          // the mesh cut it was written or updated with, RT_CUT_BUILT for a BLAS built that frame
    public uint8_t stable_frames;   // frames since its transform last moved, saturating
    public uint8_t chain_step;      // CHAIN_*
    public uint8_t fade;            // frames into a chain fade, 1 to CHAIN_FADE_FRAMES - 1; 0: none
    public uint last_visible_frame; // the last frame a view drew one of its clusters
    public uint64_t rt_blas;        // 0 while null-written
    public uint rt_partition;       // cell partition, or the partition count for the global one
    public uint rt_flags;           // RT_*
    public uint2 rt_key;            // hash of the transform, model and shown sections it was written with
    public uint rt_age;             // frames its pending PTLAS work has waited
    public uint rt_model;           // the model it holds while RT_WRITTEN
    public uint part_run;           // first part instance of the run it holds
    public uint part_count;         // part instances in the run; 0: none
    public uint far_part;           // 1 + the part instance tracing its far chain step; 0: none
    public uint far_model;          // the model that part instance holds
}

// A transform group: `count` transforms relative to the group's anchor. A null `current`
// or an all-zero transform hides the instances that use it.
public struct GroupGpu
{
    public uint8_t* current;
    public uint8_t* previous;  // last frame's transforms; equals `current` when unchanged,
                               // though `frame` can still move them
    public uint count;
    public uint format;        // GROUP_*
    public int3 anchor_cell;
    public float3 anchor_offset;
    public float3 origin;      // GROUP_QUANTIZED_PAGE: cell-relative position grid
    public float3 step;
    public float log2_scale_min;
    public float log2_scale_step;
    // A group hung from a joint: the palette group holding the joint and its entry there, which
    // takes every entry from its bind space to its pose. Null for none.
    public uint8_t* frame;  // a GroupGpu*, which Metal cannot emit inside GroupGpu itself
    public uint frame_index;
    public float3 frame_origin;    // the group's anchor in the palette's space
}

bool palette_unset(float3x4 m)
{
    return all(m[0] == 0.0) && all(m[1] == 0.0) && all(m[2] == 0.0);
}

// Whether `count` skinning palette entries from `first` are all set: an all-zero entry hides
// the copy, as an all-zero transform hides a rigid one.
public bool palette_set(GroupGpu* palette, uint first, uint count)
{
    float3x4* p = (float3x4*)palette.current + first;
    for (uint j = 0; j < count; j++)
    {
        if (palette_unset(p[j]))
            return false;
    }
    return true;
}

// The weighted sum of the palette entries of `joints` (clamped to the `count` from `first`),
// this frame's or last frame's; last frame's entry is this frame's where it was unset.
public float3x4 palette_blend(GroupGpu* palette, uint first, uint count, uint4 joints, float4 weights, bool previous)
{
    float3x4 m = float3x4(0.0);
    for (uint i = 0; i < 4; i++)
    {
        if (weights[i] <= 0.0)
            continue;
        uint at = first + min(joints[i], count - 1);
        float3x4 e = ((float3x4*)palette.current)[at];
        if (previous)
        {
            float3x4 before = ((float3x4*)palette.previous)[at];
            e = palette_unset(before) ? e : before;
        }
        m += weights[i] * e;
    }
    return m;
}

// A discrete level of a mesh: its error and its range of mesh cluster indices.
public struct MeshLevel
{
    public float error;            // m, mesh space
    public uint first;
    public uint count;
}

// A resident cluster, named by its cluster id: the id its CLAS carries, its page's base in
// `MeshGpu.page_bases` plus its index in the page.
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
    public uint row;               // its row of the mesh's cooked sets
    public uint _pad;
}

// An animated part: its rest bounds in mesh space and the sections its clusters use.
public struct PartGpu
{
    public float3 bounds_min;
    public uint sections;          // bit per section a mask can hide, below 16; bit 16 for those it can't
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

// A DAG group an edit holds at level 0, for the consumer kernel that writes it: clusters
// `first_cluster..first_cluster + cluster_count` of mesh page `page`.
public struct EditGroupGpu
{
    public uint page;
    public uint first_cluster;
    public uint cluster_count;
    public uint _pad;
}

// What an edit's mesh pins to level 0: the whole mesh, or what meets its spheres (mesh space).
public struct EditGpu
{
    public float4* spheres;
    public EditGroupGpu* groups;   // the groups held at level 0
    public uint8_t* pages;         // the pages it holds a copy of, in one block
    public uint64_t pages_bytes;   // their copy of last frame's state follows them at this offset
    public uint sphere_count;
    public uint group_count;
    public uint whole;             // not 0: every cluster
    public uint _pad;
}

// Whether `page` is one of the copies an edit of `mesh` holds.
public bool edit_copy(MeshGpu* mesh, uint8_t* page)
{
    EditGpu* e = mesh.edit;
    return e != nullptr && (uint64_t)page - (uint64_t)e.pages < e.pages_bytes;
}

// A mesh page as it stood before this frame's kernels: an edited page
// keeps a copy of last frame's state, and no kernel writes any other page.
public uint8_t* previous_page(MeshGpu* mesh, uint8_t* page)
{
    return edit_copy(mesh, page) ? (uint8_t*)((uint64_t)page + mesh.edit.pages_bytes) : page;
}

// The cooked sets of one purpose, in blob 0 (orr_pack::mesh::ClusterSets).
public struct SetTablesGpu
{
    public uint8_t* sets;          // ClusterSet per row * count + rank
    public uint* ranks;            // per section: its rank, or NO_RANK
    public uint* clusters;         // mesh clusters in tree leaf order
    public uint* cdf;              // inclusive fixed point, the last of a set 0xffffffff
    public uint2* index;           // (mesh cluster, position in the set), ascending by cluster
    public uint8_t* nodes;         // SetNode table
    public uint count;             // ranks
    // 1 when copies of the mesh trace cached cuts only: a cut can hold more clusters of one
    // section than a gathered list keeps.
    public uint cached;
}

// A mesh made resident from a mounted pack, or an edit of one. Page and node page addresses
// are 0 while their data is not resident.
public struct MeshGpu
{
    public uint64_t* pages;        // resident page per MeshHeader page
    public uint64_t* node_pages;   // node page per MeshHeader node page
    public uint* page_clusters;    // first mesh cluster index per page, then the cluster count
    public uint* page_bases;       // per page while it is resident: the cluster id of its first cluster
    public MeshLevel* levels;      // finest first; none in DAG mode
    public uint page_count;
    public uint node_page_count;
    public uint cluster_count;
    public uint level_count;       // 0: DAG mode
    public float position_step;    // m: the position grid step
    public uint attributes;        // orr_pack.mesh MESH_*
    public float merge_error;      // DAG mode: the error from which the merged stage serves; infinity without one
    public uint _pad;
    public uint rt_cut_count;      // 0: not traced
    public uint part_count;        // transforms per pose block: the mesh's parts, at least 1
    public RtCutGpu* rt_cuts;      // finest first
    public PartGpu* parts;         // per animated part
    public MaskGpu* masks;         // section-mask BLAS, `mask_count` entries
    public uint first_animated;    // the first animated part; animated parts follow it
    public uint animated_parts;    // 0: no part is animated
    public uint mask_count;
    public uint animated_sections; // the sections of every animated part, as `PartGpu.sections`
    public uint8_t* nodes;         // DAG mode: the Node table of a resident hierarchy, else null
    public uint8_t* node_table;    // DAG mode: the NodePage table of a paged hierarchy, else null
    public uint8_t* groups;        // DAG mode: the Group table
    public uint* roots;            // DAG mode: MeshHeader.roots
    public uint root_count;
    public uint pinned_pages;      // pages resident from mount, all of them unless pages stream
    public SkinMeshGpu* skin;      // MESH_SKIN meshes: what skinned copies build from, else null
    public EditGpu* edit;          // an edit's mesh: what stays at level 0; its copies always gather their cut
    public uint feedback_pages;    // the mesh page table entry the traversal's feedback reports page 0 at
    public uint feedback_nodes;    // ...node page 0 at, the finer cuts following the node pages
    public uint fine_cuts;         // the first `fine_cuts` cached cuts need streamed pages: built once wanted, BLAS 0 until then
    public uint section_count;     // sections of the mesh, which a model's cage sections follow
    public uint opaque_sections;   // the sections of opaque class, as `PartGpu.sections`
    public uint layer_sections;    // the sections of dielectric or presence class, likewise
    public SetTablesGpu emissive;  // the sections that emit
    public SetTablesGpu casters;   // the sections light leaves as a caustic
}

public bool mesh_has_layers(MeshGpu* mesh)
{
    return mesh.layer_sections != 0;
}

public bool mesh_has_opaque(MeshGpu* mesh)
{
    return mesh.opaque_sections != 0;
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
    public uint* cut_row;          // per cut: its row of the mesh's cooked sets
    public uint cut_count;
    public float bound;            // the largest deformation bound of its clusters, m; 0: unbounded
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

// A model made resident from a mounted pack: its mesh, the material of each section and its
// representation chain.
public struct ModelGpu
{
    public uint mesh;            // in SceneRoot.meshes
    public uint section_count;
    public uint* materials;      // MaterialId per mesh section
    public float3 bounds_min;
    public float3 bounds_max;
    public uint chain_count;     // farther models, below CHAIN_STEPS
    public uint chain[3];        // ModelId of steps 1 to chain_count
    public float chain_distance[3]; // m from the focus where each of those steps starts, ascending
    public uint _pad;
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
    public ProducerBatch* batches;   // by ProducerId: what is drawn and traced; a tessellated entry's generated clusters
    public ProducerBatch* patches;   // by ProducerId: the patches a tessellated entry's kernel emits; zero for others
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
    public uint skinned_capacity;
    public uint light_capacity;
    public uint producer_capacity;
    public uint emissive_capacity;   // instance entries, after the producer entries
    public float near_range;         // m
    public uint part_base;           // PTLAS index of part instance 0; the instance slots come first, so also their count
    public uint part_capacity;
    public uint producer_base;       // PTLAS index of producer entry 0; also how `SurfaceInput`, `GSurface` and `VisibleCluster` name producer entries
    public CutGpu* cut;              // the frame's cuts
    public SkinnedGpu* skinned;      // skinned copies, `skinned_capacity` entries
    public SkinStateGpu* skin_state; // per skinned copy, written by the GPU
    public uint2* skin_work;         // this frame's skinning work items: skinned copy, mesh cluster
    public DecalGpu* decals;         // by DecalId
    public DecalBinGpu* decal_bins;  // `decal_bin_slots` entries
    public uint* decal_refs;         // the global decals' ids, then each bin's run
    public uint decal_globals;       // decals too large for bins, listed first in `decal_refs`
    public uint decal_bin_slots;     // a power of two; 0 without binned decals
    public uint decal_bin_log2;      // bins are 2^decal_bin_log2 m cubes on the world cell grid
    public uint proxy_base;          // PTLAS index of medium proxy 0
    public MediumProxyGpu* proxies;  // by MediumProxyId
    public uint proxy_capacity;
    public uint light_base;          // PTLAS index of the proxy of LightId 0
    public uint* emissive_lists;     // per instance entry: its gathered list (`gathered_list`)
}

// The medium proxy a PTLAS instance index of class CLASS_MEDIUM names.
public MediumProxyGpu scene_proxy(SceneRoot* s, uint instance_index)
{
    return s.proxies[instance_index - s.proxy_base];
}

// Render-space center of proxy `p`'s box.
public float3 proxy_render_center(SceneRoot* s, MediumProxyGpu p)
{
    return float3((p.cell - s.origin_cell.xyz) << uint(s.origin_cell.w)) + p.offset;
}

// The slot a copy names.
public uint copy_slot(uint copy)
{
    return copy & ~COPY_FAR;
}

// A slot's `chain_step` in bits 0-7 and `fade` in bits 8-15.
uint chain_word(SlotStateGpu st)
{
    return uint(st.chain_step) | (uint(st.fade) << 8);
}

// The same word read with one load.
uint slot_chain_word(SceneRoot* s, uint slot)
{
    return ((uint*)(s.slot_state + slot))[0] >> 16;
}

// The step a copy with chain word `w` is at or fades to, and the one it fades from.
uint word_to(uint w)
{
    return w & CHAIN_STEP_MASK;
}

uint word_from(uint w)
{
    return (w >> 8) != 0 ? (w >> CHAIN_FROM_SHIFT) & CHAIN_STEP_MASK : word_to(w);
}

// The step of a copy with chain word `w`: its nearer, or with `far` its farther.
uint word_step(uint w, bool far)
{
    return far ? max(word_to(w), word_from(w)) : min(word_to(w), word_from(w));
}

// The nearer and the farther of the steps a copy shows; the same outside a fade.
public uint chain_near(SlotStateGpu st)
{
    return word_step(chain_word(st), false);
}

public uint chain_far(SlotStateGpu st)
{
    return word_step(chain_word(st), true);
}

// What the emissive entries of slot `slot` sample: the cooked row of the cut its PTLAS
// instance holds of `mesh`, its record `inst`'s mesh at its nearer step, or with `gathered`
// the list its traversal gathered for the BLAS built this frame. A skinned copy's skinning cut
// names its row, and an edit's copies gather level 0, which cut 0 stands for. False when the
// instance holds another step or model: a chain can name its own model, whose record past
// step 0 shows other sections and materials.
public bool traced_row(SceneRoot* s, uint slot, InstanceGpu inst, MeshGpu* mesh, out uint row, out bool gathered)
{
    SlotStateGpu st = s.slot_state[slot];
    row = 0;
    gathered = false;
    if ((st.rt_flags & RT_WRITTEN) == 0 || ((st.rt_flags >> RT_STEP_SHIFT) & CHAIN_STEP_MASK) != chain_near(st)
        || st.rt_model != inst.model)
        return false;
    if (may2(MAY2_SKINNING) && (inst.flags & INSTANCE_SKINNED) != 0)
    {
        if (inst.pose_index >= s.skinned_capacity || mesh.skin == nullptr)
            return false;
        SkinStateGpu k = s.skin_state[inst.pose_index];
        if (st.rt_lod != RT_CUT_BUILT || k.blas == 0 || k.cut >= mesh.skin.cut_count)
            return false;
        row = mesh.skin.cut_row[k.cut];
        return true;
    }
    if (mesh.edit != nullptr)
    {
        row = mesh.rt_cuts[0].row;
        return mesh.rt_cut_count > 0;
    }
    if (st.rt_lod == RT_CUT_BUILT)
    {
        gathered = true;
        return true;
    }
    row = mesh.rt_cuts[st.rt_lod].row;
    return true;
}

// The step slot copy `copy` draws and traces.
public uint copy_step(SceneRoot* s, uint copy)
{
    return word_step(slot_chain_word(s, copy_slot(copy)), (copy & COPY_FAR) != 0);
}

public bool copy_fading(SceneRoot* s, uint copy)
{
    return (slot_chain_word(s, copy_slot(copy)) >> 8) != 0;
}

// Turns copy record `inst` into its record at chain step `step`: past step 0 the chain's
// model, rigid, with that model's own materials and every section shown.
public void to_step(SceneRoot* s, inout InstanceGpu inst, uint step)
{
    if (step == 0)
        return;
    inst.model = s.models[inst.model].chain[step - 1];
    inst.material_base = NO_MATERIAL_SET;
    inst.pose_group = 0;
    inst.first_part = NO_PART_RUN;
    inst.variant = 0;
}

public InstanceGpu step_record(SceneRoot* s, InstanceGpu inst, uint step)
{
    InstanceGpu r = inst;
    to_step(s, r, step);
    return r;
}

// The record slot copy `copy` draws and traces.
public InstanceGpu copy_record(SceneRoot* s, uint copy)
{
    InstanceGpu inst = s.instances[copy_slot(copy)];
    to_step(s, inst, copy_step(s, copy));
    return inst;
}

// How much of chain step `step` a copy shows: 1 at its step outside a fade; while fading,
// its share of the fade, `fade / CHAIN_FADE_FRAMES` for the step it fades to.
public float step_presence(SlotStateGpu st, uint step)
{
    uint w = chain_word(st);
    float share = float(st.fade) / float(CHAIN_FADE_FRAMES);
    if (st.fade == 0)
        return step == word_to(w) ? 1.0 : 0.0;
    return step == word_to(w) ? share : step == word_from(w) ? 1.0 - share : 0.0;
}

// Whether slot copy `copy` is present for the uniform draw `u`: always outside a fade; while
// fading, the step it fades to below its share and the step it fades from above it, so one
// draw shared by both steps sees exactly one.
public bool copy_present(SceneRoot* s, uint copy, float u)
{
    uint w = slot_chain_word(s, copy_slot(copy));
    uint fade = w >> 8;
    if (fade == 0)
        return true;
    bool to = word_step(w, (copy & COPY_FAR) != 0) == word_to(w);
    return to == (u < float(fade) / float(CHAIN_FADE_FRAMES));
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

// The cluster id of mesh cluster `cluster` of `mesh`, whose page is resident. A mounted mesh's
// pinned pages take one run of ids from page 0's; an edit's own pages take ids of the edit's.
public uint mesh_cluster_id(MeshGpu* mesh, uint cluster)
{
    if (mesh.edit == nullptr && cluster < mesh.page_clusters[mesh.pinned_pages])
        return mesh.page_bases[0] + cluster;
    uint page = cluster_page(mesh, cluster);
    return mesh.page_bases[page] + cluster - mesh.page_clusters[page];
}

// Entry `index` of the transforms in `transforms` as a row-major [L | t] relative to the group's
// anchor. A page's exact table follows its records.
float3x4 stored_entry(GroupGpu* group, uint8_t* transforms, uint index)
{
    if (group.format == GROUP_QUANTIZED_PAGE)
    {
        InstanceRecord* records = (InstanceRecord*)transforms;
        ExactRow* table = (ExactRow*)(records + group.count);
        return decode_record(group.origin, group.step, group.log2_scale_min, group.log2_scale_step, table, records[index]);
    }
    return ((float3x4*)transforms)[index];
}

// Entry `index` of the group's current or previous transforms, as `stored_entry`, posed by the
// joint it hangs from. A joint without a pose hides it, as an all-zero entry does.
public float3x4 group_entry(GroupGpu* group, bool previous, uint index)
{
    float3x4 m = stored_entry(group, previous ? group.previous : group.current, index);
    GroupGpu* palette = (GroupGpu*)group.frame;
    if (!may2(MAY2_SKINNING) || palette == nullptr || palette_unset(m))
        return m;
    uint8_t* joints = previous ? palette.previous : palette.current;
    float3x4 zero = float3x4(float4(0.0), float4(0.0), float4(0.0));
    if (joints == nullptr || group.frame_index >= palette.count)
        return zero;
    float3x4 joint = ((float3x4*)joints)[group.frame_index];
    if (palette_unset(joint))
        return zero;
    // The joint poses the palette's space, of which the group's anchor is a point.
    float3 o = group.frame_origin;
    float3 moved = mul(affine_linear(joint), o) + affine_translation(joint) - o;
    return affine_mul(affine(affine_linear(joint), moved), m);
}

public float3x4 group_transform(GroupGpu* group, uint index)
{
    return group_entry(group, false, index);
}

public float3x4 previous_group_transform(GroupGpu* group, uint index)
{
    return group_entry(group, true, index);
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
