// The pixel probe: what one pixel of a view drew and why, read from the buffers the frame's
// raster, resolve and traversal wrote. `probe` runs one thread after the material resolve and
// writes a `ProbeGpu`; `lists` then finds the visible cluster in the raster's lists, which say
// which path drew it. The projected errors are the traversal's at the scene focus, with the
// thresholds the frame's traversal ran with.
#include "orr_root.h"
import orr_shaders.abi;
import orr_pack.mesh;
import orr_scene.producer;
import orr_scene.gbuffer;
import orr_scene.material;
import orr_scene.tables;
import orr_scene.visibility;

// `ProbeGpu.flags`.
public static const uint PROBE_GEOMETRY = 1;   // the pixel shows geometry
public static const uint PROBE_PRODUCER = 2;   // a producer entry's cluster
public static const uint PROBE_FOCUS = 4;      // the scene has a focus
public static const uint PROBE_PLACED = 8;     // the copy has a placement at the focus
public static const uint PROBE_EDIT = 16;      // the copy's mesh is an edit
public static const uint PROBE_DAG = 32;       // the mesh is a cluster DAG, not discrete levels
public static const uint PROBE_FROM_GROUP = 64; // the cluster was simplified from a group
public static const uint PROBE_FADING = 128;   // the copy fades between chain steps
public static const uint PROBE_MERGED_STAGE = 256;  // the mesh has a merged stage
public static const uint PROBE_MERGED_RASTER = 512; // the raster cut takes it: `merged_cuts`
public static const uint PROBE_RT_MERGED_STAGE = 1024; // the mesh of the slot's nearer step has a merged stage
public static const uint PROBE_RT_LEVELS = 2048;  // the mesh of the slot's nearer step has discrete levels

// `ProbeGpu.lists`: the raster lists holding the visible cluster, first pass in the low bits
// and second pass shifted by PROBE_LIST_SECOND.
public static const uint PROBE_LIST_OPAQUE = 1;
public static const uint PROBE_LIST_ALPHA = 2;
public static const uint PROBE_LIST_SOFTWARE = 4;
public static const uint PROBE_LIST_SECOND = 8;

// `ProbeGpu.page_bits` for the cluster's page (low nibble) and its from group's (shifted by 4):
// the traversal read it, wanted it, and its request is among those the frame's feedback kept.
public static const uint PROBE_PAGE_READ = 1;
public static const uint PROBE_PAGE_WANTED = 2;
public static const uint PROBE_PAGE_KEPT = 4;
public static const uint PROBE_PAGE_STREAMS = 8;

// `ProbeTexture.flags`.
public static const uint PROBE_TEXTURE_STREAMED = 1;  // it streams: `resident` and the cell hold
public static const uint PROBE_TEXTURE_NEURAL = 2;    // a neural set stands in for it
public static const uint PROBE_TEXTURE_OUTSIDE = 4;   // the slot reads no texture there
public static const uint PROBE_TEXTURE_TILED = 8;     // the slot reads a tile of a tile set
public static const uint PROBE_TEXTURE_RECORDED = 16; // the cell's raise this frame left a record

// One texture slot the pixel's material binds, as the resolve's tap of it saw it.
public struct ProbeTexture
{
    public uint slot;
    public uint texture;        // heap index read; a tile set's tile
    public uint flags;          // PROBE_TEXTURE_*
    public uint size;           // mip 0: width | height << 16
    public float2 uv;           // where it reads, in 0..1
    public float wanted;        // the level the tap wants, the view's mip bias in it
    public float resident;      // the finest mip resident over the cell
    public uint tail_mip;
    public uint cell;           // its feedback cell
    public uint feedback;       // the cell's value: frame << 8 | 255 - the finest mip wanted
    public uint _pad;
}

public struct ProbeGpu
{
    public uint flags;              // PROBE_*
    public uint visible;            // the visible cluster index
    public uint triangle;
    public uint copy;               // VisibleCluster.instance: slot, COPY_FAR on a far step
    public uint64_t word;           // the visibility buffer word
    public uint cluster;            // mesh cluster index, or a producer's cluster
    public uint local;
    public uint section_material;   // VisibleCluster.material
    public uint material;           // the material of the side the pixel shows
    public uint model;
    public uint mesh;
    public uint step;               // the chain step drawn
    public uint section;
    public uint cluster_flags;
    public uint tri_count;
    public uint alpha_first;
    public uint from_group;
    public uint mesh_page;          // the page holding the cluster
    public uint cluster_id;         // its resident cluster id
    public uint64_t page_address;   // mesh.pages[mesh_page] this frame
    public uint page_count;
    public uint pinned_pages;
    public uint level_count;
    public float self_error;        // m, mesh space
    public float parent_error;
    public float self_pixels;       // at the focus, debug scale in
    public float parent_pixels;
    public float merged_pixels;     // the merged stage's error over the model sphere, likewise
    public float placement_scale;
    public float debug_scale;
    public int debug_level;         // -1 none
    public uint seen;               // a view drew the copy last frame
    public float raster_threshold;  // px: the traversal's raster threshold
    public float rt_cached;         // px: the cached cuts' threshold of the slot
    public float rt_gathered;       // px: its gathered cut's
    public uint from_page;          // the page of `from_group`
    public uint64_t from_address;   // its resident address this frame, 0 absent
    public float from_error;        // the from group's error, m
    public float from_pixels;       // px: what its page's request carries
    public uint page_bits;          // PROBE_PAGE_*, the cluster's page then the from page's
    public float page_wanted_error; // px: the wanted error the traversal reported for the from page
    public uint feedback_kept;      // CUT_FEEDBACK
    public uint feedback_requests;  // CUT_REQUESTS
    public uint slot_visibility;    // SLOT_*
    public uint lists;              // PROBE_LIST_*
    public InstanceGpu instance;
    public SlotStateGpu state;
    public SlotCutGpu cut;
    public uint model_cut;          // the model's merged cut this frame
    public uint rt_cut_count;
    public uint fine_cuts;
    public uint admitted;           // CUT_ADMITTED
    public uint nodes_dropped;      // CUT_NODES_DROPPED
    public float rt_cut_error;      // m, mesh space: the chosen cached cut's
    public float rt_cut_pixels;     // px at the focus
    public float held_cut_error;    // the cut the PTLAS instance holds
    public uint64_t rt_cut_blas;
    public uint64_t far_cut_blas;
    public float merge_error;       // m, mesh space
    public float model_pixels;      // px: the nearer step's model's bounding radius at the focus
    public float merge_pixels;      // px: below it a copy traces its model's merged cut
    public float gather_scale;
    // The ray-tracing fields from `cut` on describe the slot's nearer step, as the traversal
    // does, and the far ones its farther step while it fades.
    public uint rt_step;            // the nearer chain step
    public uint rt_model;           // its model
    public uint held_step;          // the step its PTLAS instance holds
    public uint far_step;
    public uint far_model;
    public uint far_cut_count;
    public uint far_held_cut;       // the cut its far part instance holds
    public float far_cut_error;     // m, mesh space: this frame's far cut's
    public float far_cut_pixels;    // px at the focus
    public float far_held_error;
    public float far_distance;      // m from the focus to the far model's bounds
    // The G-buffer.
    public uint4 a;
    public uint4 x;
    public uint4 y;
    public uint2 b;
    public float depth;
    public float distance;          // m along the pixel ray
    public float4 relative;         // the hit, eye-relative render space
    public float4 placement[3];     // rows [L | t] of the drawn placement, t eye-relative
    public float4 base_color;       // w: specular roughness
    public float4 specular_color;   // w: specular weight
    public float4 emission;         // w: metalness
    public float4 normal;           // w: specular IOR
    public float4 geometric_normal; // w: diffuse roughness
    public float4 layers;           // coat, fuzz, subsurface weights, specular anisotropy
    public float4 transmission;     // the material's transmission weight, geometry opacity, alpha cutoff, raster error (m)
    public uint gbuffer_flags;
    public uint material_flags;
    public uint texture_count;
    public uint texture_raises;     // the frame's feedback raises so far
    public uint texture_capacity;
    public uint texture_first;
    public uint frame;
    public float built_error;       // m, mesh space: the largest self error of the built cut it traces
    public ProbeTexture textures[TEXTURE_SLOTS];
    public uint _pad;
}

public struct ProbeRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public RasterGpu* raster;
    public ProbeGpu* out;
    public int4 focus_cell;        // w: 1 with a focus
    public float4 focus_offset;    // w: pixels per radian
    public uint pixel_x;
    public uint pixel_y;
    public float raster_pixels;
    public float rt_pixels;
    public float culled_scale;
    public float gather_scale;
    public float merge_pixels;
    public uint _pad;
}
ORR_ROOT(ProbeRoot);

bool has_focus()
{
    return root.focus_cell.w != 0;
}

// `error` (m, mesh space) over `sphere` placed by `p`, in pixels at the focus: the traversal's
// `focus_error`.
float focus_error(Placement p, float4 sphere, float error)
{
    if (!has_focus())
        return error > 0.0 ? asfloat(0x7f800000u) : 0.0;
    return focus_pixels(p, sphere, error, root.focus_offset.w);
}

uint page_bits(CutGpu* cut, uint entry, bool streams)
{
    uint bit = 1u << (entry % 32);
    uint b = streams ? PROBE_PAGE_STREAMS : 0;
    if (!streams)
        return b;
    if ((cut.touched[entry / 32] & bit) != 0)
        b |= PROBE_PAGE_READ;
    if ((cut.wanted[entry / 32] & bit) != 0)
        b |= PROBE_PAGE_WANTED;
    uint kept = min(cut.counters[CUT_FEEDBACK], cut.feedback_capacity);
    for (uint i = 0; i < kept; i++)
    {
        if (cut.feedback[i].entry == entry)
            b |= PROBE_PAGE_KEPT;
    }
    return b;
}

void probe_cluster(SceneRoot* s, inout ProbeGpu o, VisibleCluster vc)
{
    InstanceGpu inst = copy_record(s, vc.instance);
    uint slot = copy_slot(vc.instance);
    o.instance = s.instances[slot];
    o.state = s.slot_state[slot];
    o.step = copy_step(s, vc.instance);
    if (copy_fading(s, vc.instance))
        o.flags |= PROBE_FADING;
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    o.model = inst.model;
    o.mesh = model.mesh;
    o.page_count = mesh.page_count;
    o.pinned_pages = mesh.pinned_pages;
    o.level_count = mesh.level_count;
    o.merge_error = mesh.merge_error;
    bool staged = mesh_has_merged_stage(mesh);
    if (staged)
        o.flags |= PROBE_MERGED_STAGE;
    if (mesh.edit != nullptr)
        o.flags |= PROBE_EDIT;
    if (mesh.level_count == 0)
        o.flags |= PROBE_DAG;
    Cluster cl = load_cluster(vc.page, vc.local);
    o.section = cl.section;
    o.cluster_flags = cl.flags;
    o.tri_count = cl.tri_count;
    o.alpha_first = cl.alpha_first;
    o.from_group = cl.from_group;
    o.self_error = cl.self_error;
    o.parent_error = cl.parent_error;
    o.mesh_page = cluster_page(mesh, vc.cluster);
    o.page_address = mesh.pages[o.mesh_page];
    o.cluster_id = mesh_cluster_id(mesh, vc.cluster);

    // The thresholds as the traversal's `slot_lod` takes them.
    uint d = mesh.edit != nullptr ? 0 : inst.debug_lod;
    o.debug_level = (d & DEBUG_LOD_LEVEL) != 0 ? int(d & 0xff) : -1;
    o.debug_scale = (d & DEBUG_LOD_LEVEL) == 0 && d != 0 ? 1.0 / f16tof32(d) : 1.0;
    SlotStateGpu st = o.state;
    o.seen = st.last_visible_frame + 1 >= s.frame_index && st.last_visible_frame != 0 ? 1 : 0;
    o.raster_threshold = root.raster_pixels;
    o.rt_cached = root.rt_pixels * (o.seen != 0 ? 1.0 : root.culled_scale);
    o.rt_gathered = o.rt_cached * root.gather_scale;
    o.gather_scale = root.gather_scale;
    o.merge_pixels = root.merge_pixels;

    Placement placed;
    GroupGpu* g = s.groups + inst.group;
    if (has_focus())
        o.flags |= PROBE_FOCUS;
    Placement p = (Placement)0;
    // Every chain step of a copy takes its transform.
    bool at_focus = placement(g, false, inst.group_index, root.focus_cell.xyz, root.focus_offset.xyz, uint(s.origin_cell.w), placed);
    if (at_focus && cluster_placement(s, vc.instance, inst, cl, placed, false, p))
    {
        o.flags |= PROBE_PLACED;
        o.placement_scale = p.scale;
        o.self_pixels = focus_error(p, cl.self_sphere, cl.self_error * o.debug_scale);
        o.parent_pixels = focus_error(p, cl.parent_sphere, cl.parent_error * o.debug_scale);
        float4 sphere = model_sphere(model);
        o.merged_pixels = focus_error(placed, sphere, mesh.merge_error * o.debug_scale);
        if (staged && o.debug_level < 0 && o.merged_pixels <= o.raster_threshold)
            o.flags |= PROBE_MERGED_RASTER;
    }

    CutGpu* cut = s.cut;
    bool streams = mesh.pinned_pages != mesh.page_count;
    o.page_bits = page_bits(cut, mesh.feedback_pages + o.mesh_page, streams && o.mesh_page >= mesh.pinned_pages);
    o.from_page = 0xffffffff;
    if (mesh.level_count == 0 && cl.from_group != CLUSTER_SOURCE && mesh.groups != nullptr)
    {
        o.flags |= PROBE_FROM_GROUP;
        Group from = ((Group*)mesh.groups)[cl.from_group];
        o.from_page = from.page;
        o.from_address = mesh.pages[from.page];
        o.from_error = from.error;
        if ((o.flags & PROBE_PLACED) != 0)
            o.from_pixels = focus_error(p, from.simplified, from.error * o.debug_scale);
        uint entry = mesh.feedback_pages + from.page;
        o.page_bits |= page_bits(cut, entry, streams && from.page >= mesh.pinned_pages) << 4;
        o.page_wanted_error = asfloat(cut.wanted_errors[entry]);
    }
    o.feedback_kept = cut.counters[CUT_FEEDBACK];
    o.feedback_requests = cut.counters[CUT_REQUESTS];
    o.admitted = cut.counters[CUT_ADMITTED];
    o.nodes_dropped = cut.counters[CUT_NODES_DROPPED];

    // The ray-tracing side: this frame's choice for the slot's nearer step and what its PTLAS
    // instance holds, and while it fades its farther step's.
    InstanceGpu base = o.instance;
    uint near = chain_near(st);
    InstanceGpu rt = step_record(s, base, near);
    ModelGpu* rt_model = s.models + rt.model;
    MeshGpu* rt_mesh = s.meshes + rt_model.mesh;
    o.rt_step = near;
    o.rt_model = rt.model;
    if (mesh_has_merged_stage(rt_mesh))
        o.flags |= PROBE_RT_MERGED_STAGE;
    if (rt_mesh.level_count > 0)
        o.flags |= PROBE_RT_LEVELS;
    o.cut = cut.slots[slot];
    o.built_error = asfloat(cut.slots[o.cut.owner].rt_error);
    o.model_cut = cut.model_cut[rt.model];
    o.rt_cut_count = rt_mesh.rt_cut_count;
    o.fine_cuts = rt_mesh.fine_cuts;
    o.model_pixels = asfloat(0x7f800000u);
    if (at_focus && has_focus())
    {
        float4 sphere = model_sphere(rt_model);
        float distance = max(length(place(placed, sphere.xyz)) - sphere.w * placed.scale, FOCUS_NEAR);
        o.model_pixels = sphere.w * placed.scale * root.focus_offset.w / distance;
    }
    if (rt_mesh.rt_cut_count > 0 && o.cut.rt_cut != RT_CUT_BUILT)
    {
        RtCutGpu c = rt_mesh.rt_cuts[o.cut.rt_cut];
        o.rt_cut_error = c.error;
        o.rt_cut_blas = c.blas;
        if (has_focus())
            o.rt_cut_pixels = c.error * o.cut.rt_scale / max(o.cut.rt_distance, 1e-3) * root.focus_offset.w;
    }
    // An instance whose update waits (`rt_age`) can hold a step or a model the copy has since
    // left.
    o.held_step = (st.rt_flags >> RT_STEP_SHIFT) & CHAIN_STEP_MASK;
    if ((st.rt_flags & RT_WRITTEN) != 0 && st.rt_lod != RT_CUT_BUILT)
        o.held_cut_error = s.meshes[s.models[st.rt_model].mesh].rt_cuts[st.rt_lod].error;
    if (st.fade != 0)
    {
        o.far_step = chain_far(st);
        InstanceGpu far = step_record(s, base, o.far_step);
        ModelGpu* far_model = s.models + far.model;
        MeshGpu* far_mesh = s.meshes + far_model.mesh;
        o.far_model = far.model;
        o.far_cut_count = far_mesh.rt_cut_count;
        o.far_held_cut = (st.rt_flags >> RT_FAR_CUT_SHIFT) & 0xff;
        if (at_focus && has_focus())
        {
            float4 sphere = model_sphere(far_model);
            o.far_distance = max(length(place(placed, sphere.xyz)) - sphere.w * placed.scale, 0.0);
        }
        if (far_mesh.rt_cut_count > 0)
        {
            RtCutGpu c = far_mesh.rt_cuts[o.cut.far_cut];
            o.far_cut_error = c.error;
            o.far_cut_blas = c.blas;
            if (has_focus())
                o.far_cut_pixels = c.error * o.cut.rt_scale / max(o.far_distance, 1e-3) * root.focus_offset.w;
        }
        if (st.far_part != 0)
            o.far_held_error = s.meshes[s.models[st.far_model].mesh].rt_cuts[o.far_held_cut].error;
    }
}

// The resolve's tap of slot `slot` of `m` over footprints `uv`, as `material_texture` reads it.
// A streamed texture's tap also gives `probe` the frame's feedback figures.
ProbeTexture probe_texture(MaterialGpu* m, uint slot, UvFootprint uv[2], inout ProbeGpu probe)
{
    ProbeTexture o = (ProbeTexture)0;
    o.slot = slot;
    TextureIndex t = m.textures[slot];
    UvFootprint f = material_footprint(m, slot, material_uv_set(m, slot) == 1 ? uv[1] : uv[0]);
    TextureSamplerGpu smp = m.samplers[slot];
    if ((smp.modes & TEXTURE_TILED) != 0)
        o.flags |= PROBE_TEXTURE_TILED;
    if (!material_reads(m, slot, smp.modes, t, f))
    {
        o.flags |= PROBE_TEXTURE_OUTSIDE;
        o.texture = t.index;
        return o;
    }
    o.texture = t.index;
    uint width;
    uint height;
    t.get<Texture2D<float4>>().GetDimensions(width, height);
    o.size = width | (height << 16);
    float2 place = texture_place(f.uv, smp.modes);
    o.uv = place;
    o.wanted = texture_wanted_lod(smp.modes, f, float2(width, height));
    if (m.texture_streams != nullptr && m.texture_streams[t.index] != nullptr && m.texture_streams[t.index].neural != nullptr)
        o.flags |= PROBE_TEXTURE_NEURAL;
    TextureStreamGpu* stream = texture_stream(m.texture_streams, t);
    if (stream == nullptr)
        return o;
    o.flags |= PROBE_TEXTURE_STREAMED;
    o.tail_mip = stream.tail_mip;
    o.resident = texture_stream_clamp(stream, place, smp.modes, 0, o.wanted);
    uint2 cell = min(uint2(place * float2(stream.cells)), stream.cells - 1);
    o.cell = stream.first_cell + cell.y * stream.cells.x + cell.x;
    TextureFeedbackGpu* fb = stream.feedback;
    o.feedback = fb.cells[o.cell];
    uint raises = fb.counts[fb.slot];
    probe.texture_raises = raises;
    probe.texture_capacity = fb.capacity;
    probe.texture_first = fb.first;
    uint kept = raises > fb.first ? min(raises - fb.first, fb.capacity) : 0;
    for (uint i = 0; i < kept; i++)
    {
        uint2 r = fb.records[i];
        if (r.x == o.cell && r.y == o.feedback)
            o.flags |= PROBE_TEXTURE_RECORDED;
    }
    return o;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void probe(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    ProbeGpu o = (ProbeGpu)0;
    uint2 pixel = uint2(root.pixel_x, root.pixel_y);
    o.frame = s.frame_index;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    uint64_t w = all(pixel < r.gbuffer.extent) ? r.gbuffer.visibility[index] : 0;
    o.word = w;
    if (w == 0)
    {
        *root.out = o;
        return;
    }
    o.flags |= PROBE_GEOMETRY;
    o.visible = visibility_cluster(w);
    o.triangle = visibility_triangle(w);
    VisibleCluster vc = r.gbuffer.clusters[o.visible];
    o.copy = vc.instance;
    o.cluster = vc.cluster;
    o.local = vc.local;
    o.section_material = vc.material;
    uint entry;
    if (visible_producer(s, vc, entry))
        o.flags |= PROBE_PRODUCER;
    else
    {
        o.slot_visibility = r.slot_visibility[copy_slot(vc.instance)];
        probe_cluster(s, o, vc);
    }

    PrimaryHit hit = primary_hit(s, v, vc, o.triangle, float2(pixel) + 0.5);
    o.material = hit.input.material;
    o.distance = hit.distance;
    o.relative = float4(hit.relative, 0.0);
    Placement pl = hit.placement;
    for (uint k = 0; k < 3; k++)
        o.placement[k] = float4(pl.linear[k], pl.translation[k]);
    o.depth = r.depth[index];
    MaterialGpu* m = s.materials + hit.input.material;
    o.material_flags = m.flags;

    GBufferWords words;
    words.a = r.a.get<RWTexture2D<uint4>>()[pixel];
    words.b = r.b.get<RWTexture2D<uint2>>()[pixel];
    uint gflags = words.a.w >> 24;
    words.x = (gflags & GBUFFER_EXTENDED) != 0 && r.x.index != IMAGE_NONE ? r.x.get<RWTexture2D<uint4>>()[pixel] : uint4(0);
    words.y = (gflags & GBUFFER_LAYERS) != 0 && r.y.index != IMAGE_NONE ? r.y.get<RWTexture2D<uint4>>()[pixel] : uint4(0);
    o.a = words.a;
    o.b = words.b;
    o.x = words.x;
    o.y = words.y;
    GSurface g = decode_gsurface(words);
    BsdfParams p = g.bsdf;
    o.gbuffer_flags = g.flags;
    o.base_color = float4(p.base_color, p.specular_roughness);
    o.specular_color = float4(p.specular_color, p.specular_weight);
    o.emission = float4(p.emission, p.metalness);
    o.normal = float4(g.normal, p.specular_ior);
    o.geometric_normal = float4(g.geometric_normal, p.diffuse_roughness);
    o.layers = float4(p.coat_weight, p.fuzz_weight, p.subsurface_weight, p.specular_anisotropy);
    o.transmission = float4(m.transmission_weight, m.geometry_opacity, m.alpha_cutoff, g.raster_error);

    for (uint slot = 0; slot < TEXTURE_SLOTS; slot++)
    {
        if (m.textures[slot].index == IMAGE_NONE || (slot > TEXTURE_TRANSMISSION && (m.flags & MATERIAL_MORE_TEXTURES) == 0))
            continue;
        o.textures[o.texture_count] = probe_texture(m, slot, hit.input.uv, o);
        o.texture_count++;
    }
    *root.out = o;
}

// One thread per entry of the longest raster list: marks the lists holding the probed cluster.
[shader("compute")]
[numthreads(64, 1, 1)]
void lists(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    RasterGpu* r = root.raster;
    ProbeGpu* o = root.out;
    if ((o.flags & PROBE_GEOMETRY) == 0)
        return;
    uint i = grid_index(group) * 64 + thread;
    uint visible = o.visible;
    uint bits = 0;
    if (i < min(r.counts[COUNT_OPAQUE], r.visible_capacity) && r.opaque[i] == visible)
        bits |= PROBE_LIST_OPAQUE << (i >= r.counts[COUNT_OPAQUE_BASE] ? 3 : 0);
    if (i < min(r.counts[COUNT_ALPHA], r.visible_capacity) && r.alpha[i] == visible)
        bits |= PROBE_LIST_ALPHA << (i >= r.counts[COUNT_ALPHA_BASE] ? 3 : 0);
    if (i < min(r.counts[COUNT_SOFTWARE], r.visible_capacity) && r.software[i] == visible)
        bits |= PROBE_LIST_SOFTWARE << (i >= r.counts[COUNT_SOFTWARE_BASE] ? 3 : 0);
    if (bits != 0)
        InterlockedOr(o.lists, bits);
}
