// One path loop for every kind of path: a view's, from the first layer of a traced tile, the
// G-buffer's surface or a thin lens, the world cache's update paths and the caustic photons. A
// vertex runs next-event estimation by resampled importance sampling over the light set's mixture
// and traces one shadow ray for the winner, gathers the caustic photons' light by its diffuse
// lobes, then samples its BSDF, phase function or way out, and its ray goes through the scene's
// PTLAS, seeing opaque and dielectric geometry alike; BSDF rays add the environment, directional
// discs, emission and sphere and rect light proxies they reach, weighted against next-event
// estimation by the balance heuristic with densities taken from the vertex the ray left. Shadow
// rays end at solid glass and at every thin wall but an ideal window, so light that bends on its
// way comes by BSDF rays, or as a caustic by the photons (`trace_path`); a directional light's
// through a plane stack comes by next-event estimation along its bent way too. Hits shade
// through the kernel's linked surface with a `HitContext`. Paths may end in the scene's world
// cache; the cache's update paths trace the same loop and add what each vertex scattered to its
// cell.
// Radiance is written pre-exposed. Each pixel also writes its light feedback, one of the lights
// that added to its path, for the next frame's light selection and local samplers.
//
// A presence layer passes a ray on with probability 1 - opacity; a dielectric picks reflection or
// refraction by Fresnel per path. Nested dielectrics keep an interior list of material ids,
// innermost last: the IOR outside an interface is the innermost other entry's, so glass in water
// refracts against water. Further layer hits are path vertices up to `dielectric_depth`. Segments
// inside an interior take its Beer transmittance; a scattering interior takes one distance sample
// per segment. Where the segment scatters less than once on average, a scatter there shades by
// next-event estimation within the interior, the Henyey-Greenstein phase function and the world
// cache for the rest. Elsewhere it starts a walk, as subsurface
// materials do, that reflects back or refracts out of the surface by Fresnel on the triangle's
// plane; the way out gathers the light it reaches, and outside the reference next-event
// estimation there the lights, taking the radiance inside as uniform. A camera inside a medium starts from `camera_medium`,
// flipped at each surface between the eye and the near plane; media apply gives its camera segment
// from the froxels, and only without them, as under the reference, a distance sample here does.
module path;

import orr_shaders.packing;
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.coherence;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_scene.lights;
import orr_lighting.light_sampling;
import orr_lighting.world_cache;
import orr_lighting.subsurface;
import orr_lighting.shadow;
import orr_lighting.openpbr;
import orr_lighting.interiors;
import orr_lighting.froxels;
import orr_lighting.spectrum;
import orr_lighting.caustics;
import orr_lighting.specialize;

// `PathRoot.mode`: what the radiance image holds.
public static const uint PATH_FULL = 0;
public static const uint PATH_DIRECT = 1;       // light reaching the first vertex directly
public static const uint PATH_INDIRECT = 2;     // everything else, emission seen at x1 left out
public static const uint PATH_NEE_ONLY = 3;     // light through next-event estimation alone
public static const uint PATH_BSDF_ONLY = 4;    // light through BSDF rays alone
public static const uint PATH_RAYS = 5;         // rays traced in r, 1 in g where radiance was not finite
public static const uint PATH_ORIGIN_BIAS = 6;  // x1's ray origin offset in m, in r
public static const uint PATH_CACHE = 7;        // the cache's radiance at x1, as a path ending there adds it
public static const uint PATH_CACHE_LOD = 8;    // x1's cache level in r, 1 in g where a resolved cell answers, a hash of the cell in b
public static const uint PATH_SUBSURFACE = 9;   // light leaving through the path's subsurface walk
public static const uint PATH_CAUSTICS = 10;    // what x1 gathers of the caustic photons' light

// `PathRoot.dielectric_flags`.
public static const uint DIELECTRIC_REFERENCE = 1 << 0;   // the accumulation reference: scatter on every interior segment, no cache
public static const uint DIELECTRIC_DEBUG_LAYER = 1 << 1; // the radiance shows the first layer's shares
public static const uint DIELECTRIC_DISPERSION = 1 << 2;  // dispersive materials split paths by wavelength
public static const uint DIELECTRIC_FROST = 1 << 3;       // the view keeps each pixel's `background`

// `PathRoot.cache_flags`: paths may end in the world cache.
public static const uint CACHE_TERMINATE = 1 << 0;

// `PathRoot.light_flags`: the view dispatch counts each pixel's light feedback into the
// scene's selection.
public static const uint LIGHT_FEEDBACK_COUNT = 1 << 0;

// `PathRoot.sss_flags`: only the first vertex walks.
public static const uint SSS_PRIMARY_ONLY = 1 << 0;
// `PathRoot.sss_flags`: walks reorder lanes every `WALK_POINT_STEPS` steps.
public static const uint SSS_POINTS = 1 << 1;

// Outside the reference, subsurface surfaces whose largest mean free path is below this share
// of the ray cone's width shade their subsurface share as diffuse reflection instead of walking.
public static const float WALK_FOOTPRINT = 0.25;

// Separates a walk's samples from the path's.
public static const uint WALK_STREAM = 0x85EBCA6B;

// Separates the pixel subset's sample from the path's.
public static const uint SUBSET_STREAM = 0x27D4EB2F;

// Separates an update path's samples from the view path's of the same pixel.
public static const uint UPDATE_STREAM = 0x9E3779B9;

// Separates the background layer's samples from the view path's of the same pixel.
public static const uint DOF_STREAM = 0x165667B1;

// Seeds a path's draws of presence layers.
public static const uint LAYER_STREAM = 0x85EBCA6B;

// The stream of a dispersive path's hero wavelength.
public static const uint WAVELENGTH_STREAM = 0xD3A2646C;

// Vertices of an update path that add samples to the cache.
public static const uint UPDATE_VERTICES = 4;

// Segments at least this long add the medium's in-scatter as well as its transmittance.
public static const float INSCATTER_SEGMENT = 500.0;

// Next-event estimation on a metal smoother than this runs with probability roughness /
// `NEE_METAL_ROUGHNESS`; BSDF rays carry the rest.
public static const float NEE_METAL_ROUGHNESS = 0.3;

// The staged paths' memory (`path.stages`): per-path state in words of 16 bytes, each word
// of every path together, and the queues of path slots between stages: a list per queue, and
// its state for each round of stages.
public struct PathStages
{
    public uint capacity;               // path slots
    public uint rounds;                 // rounds of queue state
    public StageQueue* queues;          // STAGE_QUEUES per round
    public uint* lists;                 // a list of `capacity` slots per queue
    public uint4* words;                // a path's state: the words its root's features name (path/stages.slang)
    public uint4* shadows;              // what a deferred shadow ray keeps of its vertex where paths track interiors; null elsewhere
    public uint4* records;              // RECORD_WORDS words per slot: the path records
    // The slots from `update_first` on hold the update paths that ride with a view's: the
    // slice's share of the view's update blocks, a block's path in the slot of its place in
    // that share. Their records, UPDATE_RECORD_WORDS words each, are apart. `capacity` where
    // the memory holds none.
    public uint4* update_records;
    public uint update_first;
    public uint _pad0;
}

// A queue of path slots in one round: the workgroups its stage runs over, which whoever
// appends to the queue counts, and the slots appended.
public struct StageQueue
{
    public uint groups;
    public uint groups_y;
    public uint groups_z;
    public uint count;
}

public struct PathRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public GBufferGpu* gbuffer;
    public LightSetGpu* set;
    public WorldCacheGpu* cache;
    // The view's local light samplers, `local_columns` per row of `LOCAL_TILE` pixel tiles;
    // null for none.
    public LocalLights* local;
    public StorageImageIndex radiance;  // RGBA16F at the G-buffer's extent
    public uint primary_candidates;
    public uint bounce_candidates;
    public uint max_bounces;
    public uint roulette_depth;         // Russian roulette ends paths from this bounce on
    public uint mode;                   // PATH_*
    public uint frame;
    public uint seed;
    public uint segment_steps;
    // Reference thin lens: aperture radius and focus distance in m. With a radius above 0,
    // x1 is traced from the lens through a jittered point of the pixel instead of read from
    // the G-buffer.
    public float aperture_radius;
    public float focus_distance;
    public uint cache_flags;            // CACHE_*
    public uint update_block;           // pixels per side of an update path's block
    // RG32_UINT at the G-buffer's extent: per pixel, its light feedback (a feedback id or
    // NO_LIGHT) and x1's camera distance in m as float bits, 0 without x1.
    public StorageImageIndex feedback;
    public uint local_columns;
    public uint light_flags;            // LIGHT_*
    // The view's layer memory (`traced_tiles`), its froxels this frame (null for none), the
    // layer vertices a path may have, the dielectric whose interior holds the eye (MATERIAL_NONE
    // for none) and DIELECTRIC_* flags.
    public LayerGuide* layers;
    public FroxelVolume* froxels;
    public uint dielectric_depth;
    public uint camera_medium;
    public uint dielectric_flags;
    // Subsurface walks: steps before a walk stops at its cap, the table of what it would yet
    // carry out, and SSS_* flags.
    public uint sss_max_steps;
    public SubsurfaceTableGpu* subsurface_table;
    public uint sss_flags;
    // Share of surface pixels whose path runs, in (0, 1]; each runs with this probability
    // and divides its path's estimate by it.
    public float pixel_fraction;
    // The background layer behind out-of-focus foreground, RGBA16F at the G-buffer's extent;
    // NONE leaves it out. Pixels whose `dof_mask` red channel holds a reverse-Z depth above 0
    // trace a second camera ray from the aperture disc of radius `dof_aperture` m, focused at
    // `*dof_focus` m along the view axis and starting past that depth, and its path.
    public StorageImageIndex dof_background;
    public TextureIndex dof_mask;
    public float* dof_focus;
    public float dof_aperture;
    public uint source;             // PATH_SOURCE_*: the staged paths this root runs
    // The view's overlay front words (`overlay_front_word`), null while off.
    public uint64_t* overlay_front;
    // The caustic photons paths gather and photons deposit in; null for none, which leaves
    // every caustic to BSDF rays.
    public CausticsGpu* caustics;
    // The staged paths' state and queues, and the share of the source's paths this root runs:
    // `slice_count` from `slice_base` on.
    public PathStages* stages;
    public uint slice_base;
    public uint slice_count;
    public uint features;           // FEATURE_* (path/stages.slang)
    // The round a stage runs in, the first round of its pass, where every path of the pass
    // finishes, and the first round of the next pass.
    public uint round;
    public uint round_first;
    public uint round_next;
    public uint _pad0;
    public uint _pad1;
}

// A pixel's front layer over the G-buffer: the top 24 bits of the reverse-Z device depth, so
// the larger word is the nearer layer whatever else it holds, then the 8-bit share of the
// pixel's light the layer itself gives (0 for an overlay fragment, whose share is the overlay's
// coverage) and half floats of the motion; 0 for no layer.
public uint64_t overlay_front_word(float depth, float2 motion, float weight)
{
    uint high = (asuint(depth) & 0xffffff00u) | pack_unorm8(weight);
    return (uint64_t(high) << 32) | uint64_t(f32tof16(motion.x) | (f32tof16(motion.y) << 16));
}

public float overlay_front_depth(uint64_t word)
{
    return asfloat(uint(word >> 32) & 0xffffff00u);
}

public float overlay_front_weight(uint64_t word)
{
    return unpack_unorm8(uint(word >> 32));
}

public float2 overlay_front_motion(uint64_t word)
{
    uint low = uint(word);
    return float2(f16tof32(low & 0xffff), f16tof32(low >> 16));
}

// What the layer guide dispatch leaves the guide resolve of a pixel's first dielectric or
// presence layer.
public struct LayerGuide
{
    public float3 diffuse;
    public float weight;            // of the layer against what the G-buffer holds; 0: no layer
    public float3 specular;
    public float roughness;
    public float3 normal;
    public float depth;             // reverse-Z device depth of the dominant layer
    public float2 motion;           // the dominant layer's, previous uv minus current uv
    public float2 specular_motion;
    public uint dominant;           // 1: `depth` and `motion` replace the G-buffer's
    // Written by the view dispatch: where the camera segment through the linked medium ends,
    // m from the eye to the layer the pixel shades; 0: at the G-buffer's surface or the sky;
    // below 0: nowhere, the path carries the segment through an interior.
    public float medium_end;
    // The interiors at the pixel's near-plane point.
    public EyeInterior near_interior;
    // Written by the view dispatch with DIELECTRIC_FROST, on every pixel whether its tile was
    // traced or not: `guide_luminance` of what the frosted-glass term blurs, the pixel's radiance
    // or, where it has a blur, that of the path from its G-buffer surface.
    public float background;
    // With DIELECTRIC_FROST on a pixel whose dominant layer is rough transmission: half
    // floats of the blur its lobes give the background, as a radius in pixels (low), and of
    // the transmission's `guide_luminance` (high); 0 elsewhere.
    public uint frost;
}

// A view's layer memory at G-buffer extent `extent`, from `LayerGuide*`: a guide per pixel in
// rows of extent.x; then per LAYER_TILE tile, in rows, a word that is nonzero where this frame's
// paths look for layers and the layer guides were written; then the tile indices; then their
// count and the workgroups of the layer guides' dispatch over them.
public uint* traced_tiles(LayerGuide* layers, uint2 extent)
{
    return (uint*)(layers + extent.x * extent.y);
}

public uint* layer_tile_list(LayerGuide* layers, uint2 extent)
{
    return traced_tiles(layers, extent) + layer_tile_count(extent);
}

public uint* layer_tile_args(LayerGuide* layers, uint2 extent)
{
    return layer_tile_list(layers, extent) + layer_tile_count(extent);
}

// The pixel of invocation `lane` (row-major in its tile) of listed tile `entry`.
public uint2 layer_tile_pixel(PathRoot r, uint entry, uint2 lane)
{
    uint2 extent = r.gbuffer.extent;
    uint columns = layer_tile_columns(extent);
    uint index = layer_tile_list(r.layers, extent)[entry];
    return uint2(index % columns, index / columns) * LAYER_TILE + lane;
}

// The pixel of invocation `lane` of an 8 x 4 workgroup of the dispatch over the listed tiles,
// two workgroups to a tile; false past the list or the extent.
public bool layer_group_pixel(PathRoot r, uint3 group, uint2 lane, out uint2 pixel)
{
    uint g = grid_index(group);
    uint2 extent = r.gbuffer.extent;
    pixel = uint2(0);
    if ((g >> 1) >= layer_tile_args(r.layers, extent)[0])
        return false;
    pixel = layer_tile_pixel(r, g >> 1, lane + uint2(0, 4 * (g & 1)));
    return all(pixel < extent);
}

// The guide of `pixel` as this frame left it; no layer where its tile was not traced.
public LayerGuide layer_guide(LayerGuide* layers, uint2 extent, uint2 pixel)
{
    if (traced_tiles(layers, extent)[layer_tile(extent, pixel)] == 0)
        return (LayerGuide)0;
    return layers[pixel.y * extent.x + pixel.x];
}

// A path vertex: its surface, where the path came from and how rays leave it.
public struct PathVertex
{
    public GSurface surface;
    public float3 view;                 // unit, toward where the path came from
    public float origin_error;          // m: what the ray origin offset must cover
    public float3 plane_point;          // a point on the visible triangle's plane
    public float camera_distance;
}

public float3 vertex_origin(PathVertex x, float3 dir)
{
    GSurface s = x.surface;
    return ray_origin(s.position, x.plane_point, s.geometric_normal, dir, gsurface_origin(s), x.origin_error, x.camera_distance);
}

// The vertex of a resolved hit of a ray along `dir`: the shading normal leaned toward the
// ray's origin and the geometric normal facing it.
public PathVertex hit_vertex(ViewConstants* v, SurfaceHit sh, float3 dir)
{
    PathVertex x;
    x.surface = sh.surface;
    x.view = -dir;
    x.surface.normal = lean_normal(x.surface.normal, x.view);
    if (dot(x.surface.geometric_normal, x.view) < 0.0)
        x.surface.geometric_normal = -x.surface.geometric_normal;
    x.origin_error = sh.geometric_error;
    x.plane_point = sh.surface.position - dir * sh.depth_offset;
    x.camera_distance = length(sh.surface.position - v.eye_render.xyz);
    return x;
}

// The error of the cut ray tracing holds for slot copy `copy`, in render space; 0 when the
// copy is not traced and for producer entries, which trace what they raster.
public float traced_error(SceneRoot* s, uint copy)
{
    uint slot = copy_slot(copy);
    if (slot >= s.part_base)
        return 0.0;
    SlotStateGpu st = s.slot_state[slot];
    bool far = (copy & COPY_FAR) != 0;
    if (far ? st.far_part == 0 : (st.rt_flags & RT_WRITTEN) == 0)
        return 0.0;
    InstanceGpu inst = copy_record(s, copy);
    MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
    if (mesh.rt_cut_count == 0)
        return 0.0;
    uint cut = far ? (st.rt_flags >> RT_FAR_CUT_SHIFT) & 0xff : uint(st.rt_lod);
    float error = mesh.rt_cuts[min(cut, mesh.rt_cut_count - 1)].error;
    Placement pl;
    GroupGpu* g = s.groups + inst.group;
    if (!placement(g, false, inst.group_index, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return 0.0;
    return error * pl.scale;
}

// The material of a traced hit, or MATERIAL_NONE where it names none this module resolves.
public uint hit_material(SceneRoot* s, Hit h)
{
    if (!h.hit || h.cluster_id < 0)
        return MATERIAL_NONE;
    return traced_material(s, h.instance_index, uint(h.cluster_id));
}

// The coherence key of a traced hit: hit or miss, its material's cost class, its surface id.
public uint path_key(SceneRoot* s, Hit h)
{
    uint material = hit_material(s, h);
    if (material == MATERIAL_NONE)
        return coherence_key(false, 0, 0);
    MaterialGpu* m = s.materials + material;
    return coherence_key(true, (m.flags & MATERIAL_COST_MASK) >> MATERIAL_COST_SHIFT, m.surface);
}

// Alpha-tested candidates through the linked surface's coverage.
public struct Alpha<S : ISurface> : IAnyHit
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 dir;
    public HitContext context;

    public bool accept(Candidate c)
    {
        return c.cluster_id >= 0 && resolve_alpha<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
    }
}

// Most candidates a ray that resolves particles and light proxies visits.
public static const uint PARTICLE_CANDIDATES = 64;

// What a BSDF ray meets in front of its hit, as `trace_resolve_translucent` layers it: each
// particle is a layer of its emission at its opacity, and each sphere or rect light's proxy a
// layer that only emits, unexposed. Emission is weighed by the balance heuristic against the
// light set's sampling where `pdf` (the ray's density, 0 for none) is above 0; densities take
// the distance and direction from the vertex at `from`. A particle lights its front, and its
// back where it is double-sided or has a back of its own, as light sampling takes it. What
// light sampling draws keeps `own` of its emission; `adds` is false where the ray adds none.
public struct PathEmission : ITranslucent
{
    public SceneRoot* scene;
    public LightSetGpu* set;
    public LocalLights* local;
    public float3 origin;           // the ray's
    public float3 dir;
    public float3 from;             // the vertex's position
    public float3 normal;           // the vertex's shading normal, zero in a medium
    public float pdf;
    public float run;               // the vertex's next-event probability
    public float t_max;             // the ray's hit, or infinity
    public float3 own;
    public bool adds;

    public float4 layer(inout Candidate c)
    {
        if (!c.triangle)
        {
            float3 l;
            float t;
            uint id = c.instance_index - scene.light_base;
            if (!adds || c.class_id != CLASS_LIGHT || id >= scene.light_capacity || !light_proxy_hit(set, id, origin, dir, t_max, from, t, l) || !any(l > 0.0))
                return float4(0.0);
            c.t = t;
            float w = 1.0;
            if (pdf > 0.0)
            {
                float3 toward = normalize(origin + dir * t - from);
                w = mis_balance(pdf, run * nee_local_light_pdf(set, local, normal, id, from, toward));
            }
            return float4(l * own * w, 0.0);
        }
        uint material;
        float3 n;
        float3 e;
        float a;
        if (c.class_id != CLASS_PARTICLE || c.cluster_id < 0
            || !resolve_particle(scene, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, dir, material, n, e, a))
            return float4(0.0);
        a = saturate(a);
        uint flags = scene.materials[material].flags;
        bool back = dot(n, dir) >= 0.0;
        if (!adds || (back && (flags & (MATERIAL_DOUBLE_SIDED | MATERIAL_HAS_BACK)) == 0))
            return float4(0.0, 0.0, 0.0, a);
        float3 w = float3(1.0);
        if ((pdf > 0.0 || any(own != 1.0)) && any(e > 0.0))
        {
            EmitterHit h = emitter_hit(set, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, back && (flags & MATERIAL_HAS_BACK) != 0);
            if (h.density > 0.0)
            {
                w = own;
                if (pdf > 0.0)
                {
                    float3 v = h.point - from;
                    float distance = length(v);
                    float cos_light = distance > 0.0 ? dot(h.normal, v / distance) : 0.0;
                    w *= mis_balance(pdf, run * nee_emissive_pdf(set, local, normal, h, distance, cos_light));
                }
            }
        }
        return float4(e * (a * w), a);
    }

    public bool accept(Candidate c)
    {
        return false;
    }
}

public RayDesc path_ray(float3 origin, float3 dir, float t_max)
{
    RayDesc r;
    r.Origin = origin;
    r.Direction = dir;
    r.TMin = 0.0;
    r.TMax = t_max;
    return r;
}

// Probability that a vertex runs next-event estimation: 1 unless it is a smooth metal or
// transmitting dielectric without directional light, which runs it with probability roughness
// / `NEE_METAL_ROUGHNESS` in the share of its light that such lobes carry.
public float nee_probability(LightSetGpu* set, GSurface s)
{
    float total;
    if (directional_share(set, s.normal, total) > 0.0)
        return 1.0;
    BsdfParams p = s.bsdf;
    float specular = p.metalness + (1.0 - p.metalness) * p.transmission_weight;
    return lerp(1.0, saturate(p.specular_roughness / NEE_METAL_ROUGHNESS), specular);
}

// The local light sampler of `pixel`'s screen tile, or null.
public LocalLights* pixel_local_lights(PathRoot r, uint2 pixel)
{
    if (r.local == nullptr)
        return nullptr;
    uint2 t = (pixel + local_tile_offset(r.frame)) / LOCAL_TILE;
    return r.local + t.y * r.local_columns + t.x;
}

// What next-event estimation weighs light by at a vertex: a surface's BSDF or a medium's
// phase function.
public interface IScatter
{
    // BSDF x cos, or the phase function, toward `wi`; in `pdf` the density the vertex's own
    // sampling draws `wi` with, and in `diffuse` what of the value its diffuse lobes give.
    float3 eval_and_pdf(float3 wi, out float pdf, out float3 diffuse);
    // Whether `eval_and_pdf` gives 0 toward every direction.
    bool specular_only();
}

// The shadow rays of next-event estimation.
public interface IShadow
{
    // What reaches the ray's origin, offset from the vertex's point `start`, from `t_max` along
    // `dir`: 0 or 1 for opaque blockers, times what lies between. `environment`: the light is
    // the environment, whose baked map carries the medium. A `directional` light comes through
    // a plane stack where one holds, whose bent direction and densities are in `stack`;
    // `stack.bent` is `dir` for a straight ray.
    float3 transmittance(float3 start, float3 origin, float3 dir, float t_max, HitContext context, bool environment, bool directional, out StackLight stack);
}

// What next-event estimation draws at a vertex before its shadow ray: the winner, what it gives
// unshadowed along the straight way, and the ray toward it.
public struct NeeDraw
{
    public bool traced;                 // a shadow ray goes to the winner
    public ResolvedLightSample chosen;
    public float3 e;                    // the vertex's BSDF x cos x the winner's radiance
    public float3 l;                    // the winner's radiance
    public float W;                     // its resampling weight
    public float scatter_pdf;           // the vertex's own density of its direction
    public float3 origin;               // the shadow ray's
    public float3 toward;
    public float t_max;
}

// The draws of next-event estimation at `x` (a zero shading normal for a point in a medium),
// weighed by `f`, with local sampler `local` (null for none). It draws the directional lights
// with the vertex's directional share, taking one candidate; otherwise `candidates` draws from
// the rest of the light set's mixture, resampled by unshadowed luminance. Splitting the two
// keeps an occluded sun, whose unshadowed light outweighs every emitter, from winning nearly
// every resampling. The shadow ray goes to the winner, aimed at its point on a finite light, or
// straight for a directional light. With `no_environment` the environment's candidates count
// for nothing. At a `chain` vertex, what a light other than the environment gives keeps `keep`
// of the share outside the vertex's diffuse lobes, the caustic photons' to give. Draws one
// sampler dimension and six per candidate. Where `f` is `specular_only` no candidate has a value,
// so it takes the dimensions and draws none.
public NeeDraw nee_draw<S : ISurface, F : IScatter>(PathRoot r, PathVertex x, F f, uint2 pixel, LocalLights* local, uint candidates, float3 keep, bool chain, inout Sampler rng, bool no_environment)
{
    LightSetGpu* set = r.set;
    NeeDraw d = (NeeDraw)0;
    uint end = rng.dimension + 1 + 6 * candidates;
    if (f.specular_only())
    {
        rng.dimension = end;
        return d;
    }
    NeeVertex v = nee_vertex(set, x.surface.position, x.surface.normal, pixel, local);
    ResolvedLightSample chosen = no_candidate();
    float chosen_target = 0.0;
    float3 chosen_f = float3(0.0);
    float3 chosen_diffuse = float3(0.0);
    float chosen_scatter_pdf = 0.0;
    float sum = 0.0;
    bool directional = rng.next() < v.share;
    uint drawn = directional ? 1 : candidates;
    for (uint i = 0; i < drawn; i++)
    {
        ResolvedLightSample c = directional ? nee_directional_candidate(set, v, rng) : nee_light_candidate<S>(set, v, rng);
        float u = rng.next();
        if (!(c.pdf > 0.0) || (no_environment && c.kind == LIGHT_SAMPLE_ENVIRONMENT))
            continue;
        float scatter_pdf;
        float3 dc;
        float3 fc = f.eval_and_pdf(c.dir, scatter_pdf, dc);
        float target = luminance_rec2020(fc * c.radiance);
        float w = target / c.pdf;
        if (!(w > 0.0) || !isfinite(w))
            continue;
        sum += w;
        if (u * sum < w)
        {
            chosen = c;
            chosen_target = target;
            chosen_f = fc;
            chosen_diffuse = dc;
            chosen_scatter_pdf = scatter_pdf;
        }
    }
    rng.dimension = end;
    if (!(sum > 0.0))
        return d;
    float3 dir = chosen.dir;
    float3 l = chosen.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, chosen.light, dir) : chosen.radiance;
    float3 fe = chosen_f;
    if (chain && chosen.kind != LIGHT_SAMPLE_ENVIRONMENT)
        fe = chosen_diffuse + max(chosen_f - chosen_diffuse, 0.0) * keep;
    float3 e = fe * l;
    if (all(e == 0.0))
        return d;
    float3 origin = vertex_origin(x, dir);
    float t_max = chosen.distance;
    float3 toward = dir;
    // A finite light's ray goes from the offset origin to the point drawn on it, stopping
    // short of it by its rounding.
    if (isfinite(t_max))
    {
        float3 to = chosen.point - origin;
        float length_to = length(to);
        toward = length_to > 0.0 ? to / length_to : dir;
        float magnitude = max(abs(chosen.point.x), max(abs(chosen.point.y), abs(chosen.point.z)));
        t_max = max(0.0, length_to - max(1e-4 * length_to, magnitude * F32_PRECISION));
    }
    d.traced = true;
    d.chosen = chosen;
    d.e = e;
    d.l = l;
    d.W = sum / (float(drawn) * chosen_target);
    d.scatter_pdf = chosen_scatter_pdf;
    d.origin = origin;
    d.toward = toward;
    d.t_max = t_max;
    return d;
}

// What draw `d` gives where its shadow ray found `vis` through plane stack `stack`, before its
// resampling weight: through a stack the light leaves the vertex along its bent direction,
// which BSDF sampling draws at its density there. `scatter_pdf` is the vertex's density of the
// way the light takes.
public float3 nee_bent<F : IScatter>(NeeDraw d, F f, float3 vis, StackLight stack, float3 keep, bool chain, out float scatter_pdf)
{
    scatter_pdf = d.scatter_pdf;
    if (!any(stack.bent != d.toward) || !any(vis > 0.0))
        return d.e;
    float3 db;
    float3 fb = f.eval_and_pdf(stack.bent, scatter_pdf, db);
    scatter_pdf *= stack.density;
    float3 fe = fb;
    if (chain)
        fe = db + max(fb - db, 0.0) * keep;
    return fe * d.l * stack.jacobian;
}

// Next-event estimation at `x`: `nee_draw`, then one shadow ray through `shadow`, for a
// directional light straight or through a plane stack; it is weighted against BSDF sampling
// unless `weight_one`. `light` is the winner's feedback id, NO_LIGHT for a directional winner
// or none. An emitter's coat and fuzz pass its light into the vertex's medium, of IOR
// `exterior_ior`. The shadow ray reports texture feedback with `feedback`
// (`HitContext.feedback`), and counts in `rays`.
public float3 shade_vertex_nee<S : ISurface, F : IScatter, V : IShadow>(PathRoot r, PathVertex x, F f, V shadow, float run, uint2 pixel, LocalLights* local, uint depth, uint candidates, bool weight_one, float exterior_ior, float3 keep, bool chain, bool feedback, inout Sampler rng, inout uint rays, out uint light, bool no_environment = false)
{
    light = NO_LIGHT;
    NeeDraw d = nee_draw<S>(r, x, f, pixel, local, candidates, keep, chain, rng, no_environment);
    if (!d.traced)
        return float3(0.0);
    RayCone cone = { 0.0, 0.0 };
    HitContext context = hit_context(HIT_SECONDARY, depth + 1, cone, 0.0, float2(-1.0), feedback);
    rays++;
    StackLight stack;
    float3 vis = shadow.transmittance(x.surface.position, d.origin, d.toward, d.t_max, context, d.chosen.kind == LIGHT_SAMPLE_ENVIRONMENT, d.chosen.kind == LIGHT_SAMPLE_DIRECTIONAL, stack);
    float scatter_pdf;
    float3 e = nee_bent(d, f, vis, stack, keep, chain, scatter_pdf);
    if (may(MAY_EMISSIVE) && d.chosen.view && any(vis > 0.0))
        vis *= emitter_view_layer<S>(r.set, d.chosen, exterior_ior);
    float mis = 1.0;
    // Every light the tiles hold is one BSDF rays reach too, along the same straight way:
    // emissive entries' surfaces, and sphere and rect lights through their proxies. A
    // directional light is, through a plane stack too.
    if (!weight_one)
        mis = mis_balance(run * d.chosen.pdf, scatter_pdf);
    if (d.chosen.kind != LIGHT_SAMPLE_DIRECTIONAL)
        light = d.chosen.light;
    return e * vis * (d.W * mis / run);
}

// Whether a vertex on `s` at path depth `depth` walks its subsurface share, on a path that
// `walked` already, with a ray cone `cone_width` m wide there. The reference walks every one.
public bool may_walk(PathRoot r, GSurface s, uint depth, bool walked, float cone_width)
{
    BsdfParams p = s.bsdf;
    if (!(p.subsurface_weight > 0.0) || (p.flags & BSDF_THIN_WALLED) != 0)
        return false;
    if ((r.dielectric_flags & DIELECTRIC_REFERENCE) != 0)
        return true;
    if ((r.sss_flags & SSS_PRIMARY_ONLY) != 0 ? depth != 0 : walked)
        return false;
    float3 sigma = bsdf_subsurface_volume(p).extinction;
    return WALK_FOOTPRINT * cone_width * min(sigma.x, min(sigma.y, sigma.z)) <= 1.0;
}

// The parameters a vertex shades with: its subsurface share as diffuse reflection unless it
// `walks`.
public BsdfParams walk_params(BsdfParams p, bool walks)
{
    return walks ? p : bsdf_subsurface_diffuse(p);
}

// Walk steps through the scene: every face of opaque and alpha-tested geometry, forced
// 2-state opacity micromaps, on instance `instance`.
public struct WalkTrace<S : ISurface> : IWalkTrace
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public HitContext context;
    public uint instance;

    public Hit trace(float3 origin, float3 dir, float t_max, out float3 positions[3])
    {
        Alpha<S> a = { scene, view, dir, context };
        return trace_closest(scene.tlas, path_ray(origin, dir, t_max), MASK_OPAQUE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    }

    public WalkMeet meet(Hit h, float3 origin, float3 dir, float3 positions[3], inout Sampler rng)
    {
        return WalkMeet(on_instance(scene, h, instance) ? WALK_BOUNDARY : WALK_NONE);
    }
}

// A sampler of a walk's own, keyed on one dimension of `rng`.
public Sampler walk_sampler(inout Sampler rng)
{
    Sampler walk = rng;
    walk.seed = pcg(hash_combine(rng.seed, rng.dimension) ^ WALK_STREAM);
    walk.dimension = 0;
    rng.next();
    return walk;
}

// A walk's exit: Lambert reflection on the exit's shading normal `n`, times the share of
// the subsurface material `p` that leaves toward each direction, estimated with numbers
// hashed from `seed` and the direction.
public struct WalkExitScatter : IScatter
{
    public BsdfParams p;
    public float3 n;
    public uint seed;

    public float3 eval(float3 wi)
    {
        float c = dot(n, wi);
        if (!(c > 0.0))
            return float3(0.0);
        uint3 h = pcg3d(asuint(wi) ^ seed);
        float3 u = float3(unit_float(h.x), unit_float(h.y), unit_float(h.z));
        return bsdf_walk_exit(p, n, wi, u) * (c * INV_PI);
    }

    public float pdf(float3 wi)
    {
        return cosine_hemisphere_pdf(dot(n, wi));
    }

    public float3 eval_and_pdf(float3 wi, out float density, out float3 diffuse)
    {
        density = pdf(wi);
        diffuse = eval(wi);
        return diffuse;
    }

    public bool specular_only()
    {
        return false;
    }
}

// The subsurface walk of vertex `x` (parameters `p`) entered along `wi`. It starts inside the
// traced surface, past the raster's error and the rounding of `x`'s position, which grows with
// its magnitude and its distance from the eye, and draws from a sampler of its own, keyed on one
// dimension of `rng`.
public Walk walk_from(PathVertex x, BsdfParams p, float3 wi, inout Sampler rng)
{
    GSurface g = x.surface;
    float magnitude = max(x.camera_distance, max(abs(g.position.x), max(abs(g.position.y), abs(g.position.z))));
    float error = gsurface_origin(g) == ORIGIN_DEFAULT ? x.origin_error : 0.0;
    float3 origin = g.position - g.geometric_normal * max(error, magnitude * F32_PRECISION);
    return walk_start(origin, wi, bsdf_subsurface_volume(p), 1.0, walk_sampler(rng));
}

// What traces a walk on instance `instance` from a vertex at path depth `depth`, reporting
// texture feedback with `feedback` (`HitContext.feedback`).
public WalkTrace<S> walk_tracer<S : ISurface>(PathRoot r, uint instance, uint depth, bool feedback)
{
    RayCone cone = { 0.0, 0.0 };
    WalkTrace<S> tracer = { r.scene, r.view, hit_context(HIT_SECONDARY, depth + 1, cone, 0.0, float2(-1.0), feedback), instance };
    return tracer;
}

// How many coherence points a walk has at most, none where nothing reorders or walks.
public uint walk_points(PathRoot r)
{
    return (r.sss_flags & SSS_POINTS) != 0 ? (r.sss_max_steps - 1) / WALK_POINT_STEPS : 0;
}

// Runs `walk` up to its next coherence point, or through its end without points. Outside the
// reference a walk at its cap keeps the subsurface table's share.
public void walk_part<T : IWalkTrace>(PathRoot r, inout Walk walk, T tracer, inout uint rays)
{
    bool share = (r.dielectric_flags & DIELECTRIC_REFERENCE) == 0;
    walk_steps(walk, tracer, (r.sss_flags & SSS_POINTS) != 0 ? WALK_POINT_STEPS : r.sss_max_steps, r.subsurface_table, r.sss_max_steps, share, rays);
}

// The exit vertex of a walk whose last step along `dir` reached `sh`: its normals face out.
public PathVertex exit_vertex(ViewConstants* v, SurfaceHit sh, float3 dir)
{
    PathVertex x;
    x.surface = sh.surface;
    float3 ng = dot(sh.surface.geometric_normal, dir) < 0.0 ? -sh.surface.geometric_normal : sh.surface.geometric_normal;
    x.surface.geometric_normal = ng;
    x.surface.normal = dot(sh.surface.normal, ng) < 0.0 ? -sh.surface.normal : sh.surface.normal;
    x.view = x.surface.normal;
    x.origin_error = sh.geometric_error;
    x.plane_point = sh.surface.position - dir * sh.depth_offset;
    x.camera_distance = length(sh.surface.position - v.eye_render.xyz);
    return x;
}

// A diffuse direction leaving a walk's exit `f`, with its weight. Draws two sampler
// dimensions.
public BsdfSample walk_exit_sample(WalkExitScatter f, inout Sampler rng)
{
    float3 t, b;
    orthonormal_basis(f.n, t, b);
    float3 l = sample_cosine_hemisphere(rng.next2());
    BsdfSample bs = (BsdfSample)0;
    bs.wi = normalize(from_local(l, t, b, f.n));
    bs.pdf = cosine_hemisphere_pdf(l.z);
    bs.weight = bs.pdf > 0.0 ? bsdf_walk_exit(f.p, f.n, bs.wi, rng.next4().xyz) : float3(0.0);
    bs.diffuse = bs.weight;
    bs.lobe = LOBE_REFLECTION | LOBE_DIFFUSE;
    return bs;
}

// The exit scatter of a walk leaving at `x` with parameters `p`; draws one sampler
// dimension for its estimates.
public WalkExitScatter walk_exit(PathVertex x, BsdfParams p, inout Sampler rng)
{
    WalkExitScatter f = { p, x.surface.normal, asuint(rng.next()) };
    return f;
}

// Applies an interior segment of length `t` (+inf: unbounded) to `throughput`: its Beer
// transmittance, or with `scatter` one distance sample from the extinction, picking a channel
// uniformly and weighing by the channels' mean density. True with `at` set where the sample
// scatters before `t`. Draws two sampler dimensions.
public bool interior_segment(InteriorMedium m, float t, bool scatter, inout Sampler rng, inout float3 throughput, out float at)
{
    at = 0.0;
    float u_channel = rng.next();
    float u_distance = rng.next();
    if (!scatter || !any(m.sigma_s > 0.0))
    {
        throughput *= beer(m.sigma_t, t);
        return false;
    }
    float sigma = m.sigma_t[min(uint(u_channel * 3.0), 2u)];
    float sampled = sigma > 0.0 ? -log(1.0 - u_distance) / sigma : positive_infinity();
    if (sampled < t)
    {
        float3 tr = beer(m.sigma_t, sampled);
        float pdf = dot(m.sigma_t * tr, float3(1.0 / 3.0));
        throughput *= pdf > 0.0 ? m.sigma_s * tr / pdf : float3(0.0);
        at = sampled;
        return true;
    }
    float3 tr = beer(m.sigma_t, t);
    float p = dot(select(m.sigma_t > 0.0, tr, float3(1.0)), float3(1.0 / 3.0));
    throughput *= p > 0.0 ? tr / p : float3(0.0);
    return false;
}

public bool is_layer(Hit h)
{
    return may(MAY_LAYERS) && h.hit && (h.class_id == CLASS_DIELECTRIC || h.class_id == CLASS_PRESENCE);
}

// Shadow rays of a vertex: it arrived with interior list `arrived`, on the surface of material
// `surface` (MATERIAL_NONE elsewhere) with outward geometric normal `normal` (zero inside a medium).
// A ray into the surface starts inside its interior.
public struct PathShadow<S : ISurface, M : IMedium> : IShadow
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public uint* overflows;
    public Interior arrived;
    public uint surface;
    public float4 point;            // on `surface`: what its medium reads when a ray enters it here
    public float3 normal;
    // The path splits light by wavelength at dispersive faces, which end plane stacks.
    public bool dispersion;
    // Directional lights' rays look for plane stacks: the scene holds a stack entry.
    public bool bend;

    Interior inside(float3 dir)
    {
        Interior inside = arrived;
        if (surface != MATERIAL_NONE)
            inside.cross(surface, dot(dir, normal) < 0.0, point);
        return inside;
    }

    public float3 transmittance(float3 start, float3 origin, float3 dir, float t_max, HitContext context, bool environment, bool directional, out StackLight stack)
    {
        // Environment light reaches vertices without the medium's transmittance, as BSDF
        // misses see it: the baked map carries the medium.
        if (path_light<S, M>(scene, view, start, origin, dir, t_max, RAY_FLAG_NONE, context, environment, inside(dir), may(MAY_STACK) && directional && bend, dispersion, true, overflows, stack) == STACK_BROKEN)
            return float3(0.0);
        return any(stack.transmittance > 0.0) ? stack.transmittance * exp(-stack.depth) : float3(0.0);
    }
}

// Light from outside as an interior's walk gathers it where it leaves through a surface of
// `eta` times the outside's IOR. The radiance inside is taken as uniform, so what leaves along
// the walk's way out arrives from `wi` by Lambert reflection on outward normal `n` times the
// Fresnel transmission from `wi`, over its mean, which keeps the energy of uniform light.
public struct InteriorExitScatter : IScatter
{
    public float3 n;
    public float eta;
    public float scale;             // 1 / the transmission's cosine-weighted mean

    public float3 eval(float3 wi)
    {
        float c = dot(n, wi);
        if (!(c > 0.0))
            return float3(0.0);
        return float3((1.0 - openpbr_fresnel(eta, c)) * scale * c * INV_PI);
    }

    public float pdf(float3 wi)
    {
        return cosine_hemisphere_pdf(dot(n, wi));
    }

    public float3 eval_and_pdf(float3 wi, out float density, out float3 diffuse)
    {
        density = pdf(wi);
        diffuse = eval(wi);
        return diffuse;
    }

    public bool specular_only()
    {
        return false;
    }
}

public InteriorExitScatter interior_exit(float3 n, float eta)
{
    // The mean, evenly in the cosine squared.
    float sum = 0.0;
    for (uint i = 0; i < 16; i++)
        sum += 1.0 - openpbr_fresnel(eta, sqrt((float(i) + 0.5) / 16.0));
    InteriorExitScatter f = { n, eta, 16.0 / sum };
    return f;
}
// What next-event estimation weighs light by at a path's vertex: its BSDF, or at
// a scatter in an interior the phase function.
public struct VertexScatter : IScatter
{
    public Bsdf b;
    public bool phase;
    public float3 travel;           // a scatter's: the path's direction into it
    public float anisotropy;
    public bool exit;               // a walk's exit, which scatters as `walk`
    public WalkExitScatter walk;
    public bool leave;              // a walk leaving an interior, which scatters as `way_out`
    public InteriorExitScatter way_out;

    public float3 eval_and_pdf(float3 wi, out float density, out float3 diffuse)
    {
        diffuse = float3(0.0);
        if (phase)
        {
            density = hg_phase(anisotropy, dot(travel, wi));
            return float3(density);
        }
        float3 none;
        if (leave)
            return way_out.eval_and_pdf(wi, density, none);
        if (exit)
            return walk.eval_and_pdf(wi, density, diffuse);
        return b.eval_and_pdf(wi, density, diffuse);
    }

    public bool specular_only()
    {
        return !phase && !leave && !exit && b.specular_only();
    }
}
// `LayerVertex.kind`.
public static const uint VERTEX_OPAQUE = 0;
public static const uint VERTEX_LAYER = 1;
public static const uint VERTEX_SCATTER = 2;
public static const uint VERTEX_LEAVE = 3;      // a walk leaving an interior, along a way it chose

// A vertex of a path.
public struct LayerVertex
{
    public PathVertex x;            // a layer's normals keep the material's orientation
    public uint kind;               // VERTEX_*
    public uint section;            // a layer's section material, which owns its interior
    public bool front;              // a layer reached from its front side
    public float anisotropy;        // a scatter's phase function
    public bool chains;             // light leaves it as a caustic by its non-diffuse lobes (`surface_chains`)
    public bool stack;              // a layer on a plane stack face (`stack_face`)
}

// The vertex of a layer hit by a ray along `dir`: the shading normal leaned toward the ray's
// origin on the side the ray arrived from, both normals kept on the material's front side so
// the BSDF sees which side the path is on.
public LayerVertex layer_vertex(ViewConstants* v, SurfaceHit sh, float3 dir)
{
    LayerVertex l;
    l.kind = VERTEX_LAYER;
    l.section = sh.section;
    l.front = dot(sh.surface.geometric_normal, dir) < 0.0;
    l.anisotropy = 0.0;
    l.stack = false;
    PathVertex x;
    x.surface = sh.surface;
    x.view = -dir;
    float side = l.front ? 1.0 : -1.0;
    x.surface.normal = side * lean_normal(side * sh.surface.normal, x.view);
    x.origin_error = sh.geometric_error;
    x.plane_point = sh.surface.position - dir * sh.depth_offset;
    x.camera_distance = length(sh.surface.position - v.eye_render.xyz);
    l.x = x;
    return l;
}

public LayerVertex opaque_vertex(PathVertex x)
{
    LayerVertex l;
    l.x = x;
    l.kind = VERTEX_OPAQUE;
    l.section = MATERIAL_NONE;
    l.front = true;
    l.anisotropy = 0.0;
    l.stack = false;
    return l;
}

// A scatter at `p` in interior `medium`, reached along `dir`.
public LayerVertex scatter_vertex(ViewConstants* v, float3 p, float3 dir, uint medium, float anisotropy)
{
    LayerVertex l;
    l.kind = VERTEX_SCATTER;
    l.section = MATERIAL_NONE;
    l.front = true;
    l.anisotropy = anisotropy;
    l.stack = false;
    l.chains = false;
    PathVertex x = (PathVertex)0;
    x.surface.position = p;
    x.surface.material = medium;
    x.view = -dir;
    x.plane_point = p;
    x.camera_distance = length(p - v.eye_render.xyz);
    l.x = x;
    return l;
}

// A walk leaving through the surface at `p` with outward normal `n`.
public LayerVertex leave_vertex(ViewConstants* v, float3 p, float3 n)
{
    LayerVertex l;
    l.kind = VERTEX_LEAVE;
    l.section = MATERIAL_NONE;
    l.front = true;
    l.anisotropy = 0.0;
    l.stack = false;
    l.chains = false;
    PathVertex x = (PathVertex)0;
    x.surface.position = p;
    x.surface.normal = n;
    x.surface.geometric_normal = n;
    x.surface.material = MATERIAL_NONE;
    x.view = n;
    x.plane_point = p;
    x.camera_distance = length(p - v.eye_render.xyz);
    l.x = x;
    return l;
}

public bool thin_walled(GSurface s)
{
    return (s.bsdf.flags & BSDF_THIN_WALLED) != 0;
}

// The parameters of layer or opaque vertex `l`, its coat, fuzz and film collapsed with `collapse`.
public BsdfParams vertex_params(LayerVertex l, bool collapse)
{
    return collapse ? bsdf_collapse(l.x.surface.bsdf) : l.x.surface.bsdf;
}

// `ior`, the IOR of material `m` at 587.6 nm, at wavelength `nm`, with what `point` reads.
public float dispersed_material_ior(SceneRoot* s, uint m, float ior, float nm, float4 point)
{
    if (m == MATERIAL_NONE)
        return ior;
    MediumParams p = material_medium(s.materials + m, point);
    return dispersed_ior(ior, p.dispersion_scale, p.dispersion_abbe, nm);
}

// Whether layer vertex `l`, reached with `interior`, has a dispersive side.
public bool layer_disperses(SceneRoot* s, LayerVertex l, Interior interior)
{
    uint m = l.section;
    return l.kind == VERTEX_LAYER && (disperses(s, m) || disperses(s, interior.outside(m, m)));
}

// The BSDF of layer or opaque vertex `l`, against the interior list the path arrived with: a
// layer's exterior IOR is the innermost other entry's, an opaque vertex's the innermost entry's,
// so a pool floor meets water. Unless the vertex `walks`, its subsurface share is diffuse
// reflection. A layer takes both IORs at wavelength `nm` where it is above 0, else at 587.6 nm.
public Bsdf prepare_vertex(SceneRoot* s, LayerVertex l, Interior interior, bool collapse, bool walks, bool adjoint, float nm)
{
    BsdfParams p = walk_params(vertex_params(l, collapse), walks);
    uint m = l.section;
    bool layer = l.kind == VERTEX_LAYER;
    uint outside = layer ? interior.outside(m, m) : interior.innermost();
    float exterior = interior_ior(s, outside);
    if (layer && nm > 0.0)
    {
        // A path inside `m` read it where it entered; a path meeting it reads it here.
        float4 entered = interior.point(m);
        float4 point = entered.x != NO_MEDIUM_POINT.x ? entered : l.x.surface.medium_point;
        p.specular_ior = dispersed_material_ior(s, m, p.specular_ior, nm, point);
        exterior = dispersed_material_ior(s, outside, exterior, nm, interior.point(outside));
    }
    return bsdf_prepare(p, l.x.surface.normal, l.x.view, exterior, adjoint);
}

// A path's wavelengths once it meets a dispersive layer; before that its colors are plain
// RGB. From there the path follows the hero's refraction, and every companion's estimate is
// weighed by its own BSDF and density at each direction the hero drew at a dispersive layer,
// over the mean density of all four (the balance heuristic over wavelengths). The path's
// throughput stays the hero's; `transfer` takes its contributions to Rec.2020.
public struct Spectral
{
    public bool on;
    float sample;                   // what draws the wavelengths
    public float4 value;            // per wavelength: its BSDF values over the hero's, multiplied along the path
    public float4 density;          // its densities over the hero's
    public float3x3 transfer;

    public __init()
    {
        on = false;
        sample = 0.0;
        value = float4(1.0);
        density = float4(1.0);
        transfer = float3x3(0.0);
    }

    // Draws the wavelengths from `u`.
    [mutating]
    public void start(float u)
    {
        on = true;
        sample = u;
        update();
    }

    // The hero's wavelength, 0 before the path disperses.
    public float hero()
    {
        return on ? sample_wavelengths(sample).nm.x : 0.0;
    }

    // Whether a companion still carries an estimate.
    public bool companions()
    {
        return on && any(density.yzw > 0.0);
    }

    // Weighs companion `k` by its BSDF x cos `f` and density `pdf` toward the direction the
    // hero drew with BSDF x cos `hero_f` and density `hero_pdf`.
    [mutating]
    public void weigh(uint k, float3 f, float pdf, float3 hero_f, float hero_pdf)
    {
        float3 basis = spectrum_basis(sample_wavelengths(sample).nm[k]);
        float own = dot(f, basis);
        float theirs = dot(hero_f, basis);
        value[k] *= theirs > 0.0 ? own / theirs : 0.0;
        density[k] *= hero_pdf > 0.0 ? pdf / hero_pdf : 0.0;
        if (!(value[k] > 0.0) || !(density[k] > 0.0) || !isfinite(value[k] * density[k]))
        {
            value[k] = 0.0;
            density[k] = 0.0;
        }
    }

    [mutating]
    public void update()
    {
        float mean = dot(density, float4(1.0 / float(WAVELENGTHS)));
        transfer = spectral_transfer(sample_wavelengths(sample), mean > 0.0 ? value / mean : float4(0.0));
    }

    // Wavelength `k`.
    public float wavelength(uint k)
    {
        return sample_wavelengths(sample).nm[k];
    }

    // Contribution `c` in Rec.2020.
    public float3 color(float3 c)
    {
        return on ? mul(transfer, c) : c;
    }
}

// The hero wavelength's sample of pixel `pixel`, a dimension of its own.
public float wavelength_sample(PathRoot r, uint2 pixel)
{
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ LAYER_STREAM ^ WAVELENGTH_STREAM);
    return rng.next();
}

// The shadow of vertex `l` against the interior list it was reached with.
public PathShadow<S, M> vertex_shadow<S : ISurface, M : IMedium>(PathRoot r, LayerVertex l, Interior interior, bool dispersion)
{
    PathShadow<S, M> shadow;
    shadow.dispersion = dispersion;
    shadow.bend = r.set.stack_entries != 0;
    shadow.scene = r.scene;
    shadow.view = r.view;
    shadow.overflows = r.set.shadow_overflows;
    shadow.arrived = interior;
    shadow.surface = MATERIAL_NONE;
    shadow.normal = float3(0.0);
    if (l.kind == VERTEX_LAYER && crosses_interior(r.scene, l.section))
    {
        shadow.surface = l.section;
        shadow.point = l.x.surface.medium_point;
        shadow.normal = l.x.surface.geometric_normal;
    }
    return shadow;
}

// The incident radiance a scatter at `p` takes from the world cache: the mean of its cells
// facing along and against `dir`. Draws eight sampler dimensions.
public float3 interior_ambient(PathRoot r, float3 p, float3 dir, inout Sampler rng)
{
    float3 sum = float3(0.0);
    for (uint k = 0; k < 2; k++)
    {
        CacheCell cell = cache_cell(r.cache, r.scene, p, k == 0 ? dir : -dir, rng.next4().xyz, true);
        float3 cached;
        if (cache_lookup(r.cache, cell.key, cached))
            sum += cached;
    }
    return 0.5 * sum;
}

// Steps of an interior's walk through the scene: every face of opaque geometry and layers.
// Faces of its own material seen from behind reflect it back or let it out by Fresnel on their
// plane, the walk's `ior` times the outside's; any other surface ends the walk there.
public struct InteriorWalk<S : ISurface> : IWalkTrace
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public HitContext context;
    public uint seed;
    public uint material;
    public float ior;

    public Hit trace(float3 origin, float3 dir, float t_max, out float3 positions[3])
    {
        LayerAccept<S> a = { scene, view, dir, context, seed, PRESENCE_DRAWN };
        return trace_closest(scene.tlas, path_ray(origin, dir, t_max), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    }

    public WalkMeet meet(Hit h, float3 origin, float3 dir, float3 positions[3], inout Sampler rng)
    {
        float3 normal;
        uint m;
        if (h.cluster_id < 0 || !hit_geometry(scene, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, normal, m))
            return WalkMeet(WALK_BOUNDARY);
        float c = dot(normal, dir);
        if (m != material || !(c > 0.0))
            return WalkMeet(WALK_BOUNDARY);
        float3 p = origin + dir * h.t;
        float3 way_out = refract(dir, -normal, ior);
        if (rng.next() < openpbr_fresnel(1.0 / ior, c) || all(way_out == 0.0))
        {
            WalkMeet back = WalkMeet(WALK_REFLECT);
            back.dir = reflect(dir, normal);
            back.origin = ray_origin(p, p, normal, back.dir, ORIGIN_DEFAULT, 0.0, length(p - view.eye_render.xyz));
            return back;
        }
        WalkMeet leave = WalkMeet(WALK_LEAVE);
        leave.dir = normalize(way_out);
        leave.origin = p;
        leave.normal = normal;
        return leave;
    }
}

// The walk of interior medium `im` from a scatter at `from` reached along `dir`, its surface
// `ior` times the outside's.
public Walk walk_interior(float3 from, float3 dir, InteriorMedium im, float ior, inout Sampler rng)
{
    SubsurfaceVolume volume;
    volume.extinction = im.sigma_t;
    volume.albedo = im.albedo();
    volume.anisotropy = im.anisotropy;
    Sampler walk = walk_sampler(rng);
    float2 u = walk.next2();
    return walk_start(from, sample_hg(im.anisotropy, dir, u), volume, ior, walk);
}

// What traces a walk in interior `medium`, its surface `ior` times the outside's, reporting
// texture feedback with `feedback` (`HitContext.feedback`).
public InteriorWalk<S> interior_tracer<S : ISurface>(PathRoot r, uint medium, float ior, uint seed, uint depth, bool feedback)
{
    RayCone cone = { 0.0, 0.0 };
    InteriorWalk<S> tracer = { r.scene, r.view, hit_context(HIT_SECONDARY, depth + 1, cone, 0.0, float2(-1.0), feedback), seed, medium, ior };
    return tracer;
}

// What leaves a path's vertex.
public struct LayerRay
{
    public float3 origin;
    public float3 dir;
    public float3 start;            // the point `origin` is offset from, whose segment it is
    public float pdf;               // 0 for an ideal-specular direction
    public float run;               // the vertex's next-event probability
    public float3 normal;           // the vertex's shading normal, zero in a medium
    public float3 from;             // the vertex's position, from which light densities are taken
    public bool diffuse;
    public bool chain;              // left a layer or a scatter
    public bool scatter;            // may scatter in an interior
    public bool local;              // left the first vertex, which reads its tile's local light sampler
    public bool lit;                // the vertex's next-event estimation took its lights but the environment at weight 1
}

// What a path records about its vertices as it traces them.
public interface IPathRecord
{
    // Whether a front-face hit past `scattered` opaque vertices and scatters (x1 is the first)
    // may end the path in the cache, given that the ray reaching it left a diffuse lobe, that
    // its cone is wider than the hit's cell, and that it is the last bounce's.
    bool may_end(uint scattered, bool diffuse, bool wide, bool last);
    // Whether its subsurface vertices may walk; where they do not, their subsurface share is
    // diffuse reflection.
    bool walks();
    // Whether its paths carry flux from a light rather than importance from the camera, so
    // refraction leaves their weight unscaled.
    bool adjoint();
    // Whether it is a caustic photon's: it estimates no light, deposits where it meets a
    // surface after a chain vertex, and ends at its first vertex that is not one.
    bool photon();
    // Whether dispersive materials split its paths by wavelength.
    bool spectral();
    // Whether the ray leaving x1 resolves particles by their emission.
    bool particles();
    // The path reached its first vertex `x`, prepared as `b` (`t` m from the eye), a layer's
    // where `layer` holds; a sky pixel's path reaches none.
    [mutating]
    void arrive(PathRoot r, PathVertex x, Bsdf b, float t, bool layer);
    // Vertex `x`, prepared as `b`, reached with `throughput` once the path's radiance (its
    // emission included) was `radiance`; next-event estimation and a BSDF ray follow it.
    [mutating]
    void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng);
    // A photon of flux `flux` met vertex `l`, prepared as `b`, past a chain vertex.
    [mutating]
    void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng);
    // Light with feedback id `id` added `contribution` of luminance to the path's radiance.
    [mutating]
    void light(uint id, float contribution);
    // The path ended with `radiance` after `layers` layer vertices.
    [mutating]
    void finish(PathRoot r, float3 radiance, uint layers);
}

// A view's path: it ends in the cache after a diffuse bounce, where its cone outgrew the
// cell, or at the last bounce. Its light feedback is one of the lights that added to it,
// chosen with probability proportional to what each added.
public struct ViewPath : IPathRecord
{
    public uint feedback;
    float total;
    uint state;
    public float first_t;               // m from the eye to the first vertex, 0 without one
    public float first_distance;        // that vertex's camera distance, 0 without one
    public bool first_layer;
    public float reflected;             // of a first layer: luminance of its reflection and diffuse lobes
    public float transmitted;           // and of its transmission
    public uint layers;

    // `seed` picks the feedback's random choices, apart from the path's sampler.
    public __init(uint seed)
    {
        feedback = NO_LIGHT;
        total = 0.0;
        state = seed;
        first_t = 0.0;
        first_distance = 0.0;
        first_layer = false;
        reflected = 0.0;
        transmitted = 0.0;
        layers = 0;
    }

    public bool may_end(uint scattered, bool diffuse, bool wide, bool last)
    {
        return scattered >= 1 && (diffuse || wide || last);
    }

    public bool walks()
    {
        return true;
    }

    public bool adjoint()
    {
        return false;
    }

    public bool photon()
    {
        return false;
    }

    public bool spectral()
    {
        return true;
    }

    public bool particles()
    {
        return true;
    }

    [mutating]
    public void arrive(PathRoot r, PathVertex x, Bsdf b, float t, bool layer)
    {
        first_t = t;
        first_distance = x.camera_distance;
        first_layer = layer;
        if (layer)
        {
            reflected = luminance_rec2020(b.albedo().specular + b.albedo().diffuse);
            transmitted = luminance_rec2020(b.albedo().transmission);
        }
    }

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng) {}

    [mutating]
    public void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng) {}

    [mutating]
    public void light(uint id, float contribution)
    {
        if (!(contribution > 0.0) || !isfinite(contribution))
            return;
        total += contribution;
        state = pcg(state);
        if (unit_float(state) * total < contribution)
            feedback = id;
    }

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint count)
    {
        layers = count;
    }
}

// An update path: it resamples the cache from its second opaque vertex on, and adds to the
// cell of each of its first `UPDATE_VERTICES` opaque vertices the light that vertex scattered,
// over its albedo.
public struct CacheUpdate : IPathRecord
{
    uint count;
    uint64_t keys[UPDATE_VERTICES];
    float3 scale[UPDATE_VERTICES];      // 1 / (throughput x albedo) per channel, 0 where the albedo is
    float3 before[UPDATE_VERTICES];     // the path's radiance on arrival

    public __init()
    {
        count = 0;
        for (uint i = 0; i < UPDATE_VERTICES; i++)
        {
            keys[i] = 0;
            scale[i] = float3(0.0);
            before[i] = float3(0.0);
        }
    }

    public bool may_end(uint scattered, bool diffuse, bool wide, bool last)
    {
        return scattered >= 2;
    }

    // The cache holds the zero mean free path limit of subsurface surfaces.
    public bool walks()
    {
        return false;
    }

    public bool adjoint()
    {
        return false;
    }

    public bool photon()
    {
        return false;
    }

    public bool spectral()
    {
        return false;
    }

    public bool particles()
    {
        return false;
    }

    [mutating]
    public void arrive(PathRoot r, PathVertex x, Bsdf b, float t, bool layer) {}

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng)
    {
        float u = rng.next();
        if (count >= UPDATE_VERTICES)
            return;
        float3 albedo = cache_albedo(b.albedo(), x.surface.bsdf) * throughput;
        if (!any(albedo > 1e-6))
            return;
        CacheCell cell = cache_cell(r.cache, r.scene, x.surface.position, x.surface.geometric_normal, float3(0.5, 0.5, u), false);
        keys[count] = cell.key;
        scale[count] = select(albedo > 1e-6, 1.0 / albedo, float3(0.0));
        before[count] = radiance;
        count++;
    }

    [mutating]
    public void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng) {}

    [mutating]
    public void light(uint id, float contribution) {}

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint layers)
    {
        for (uint i = 0; i < count; i++)
            cache_add(r.cache, keys[i], (radiance - before[i]) * scale[i]);
    }
}

// A caustic photon from caster entry `entry`: it deposits its flux in the caustics of the path
// root where it meets a surface with diffuse or subsurface albedo, on the side it arrives from,
// and counts toward its entry's yield when it deposits at all.
public struct CausticPhoton : IPathRecord
{
    uint entry;
    bool deposited;

    public __init(uint from)
    {
        entry = from;
        deposited = false;
    }

    public bool may_end(uint scattered, bool diffuse, bool wide, bool last)
    {
        return false;
    }

    public bool walks()
    {
        return false;
    }

    public bool adjoint()
    {
        return true;
    }

    public bool photon()
    {
        return true;
    }

    public bool spectral()
    {
        return true;
    }

    public bool particles()
    {
        return true;
    }

    [mutating]
    public void arrive(PathRoot r, PathVertex x, Bsdf b, float t, bool layer) {}

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng) {}

    [mutating]
    public void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng)
    {
        float4 u = rng.next4();
        float3 albedo = b.diffuse_reflection() + b.diffuse_transmission() + b.albedo().subsurface;
        if (!any(albedo > 0.0) || !any(flux > 0.0) || !all(isfinite(flux)))
            return;
        GSurface g = l.x.surface;
        float3 n = dot(g.geometric_normal, l.x.view) < 0.0 ? -g.geometric_normal : g.geometric_normal;
        CausticsGpu* c = r.caustics;
        if (c.mode == CAUSTICS_MAP)
            map_deposit(c, g.position, n, flux, l.x.view);
        else
            grid_deposit(c, r.scene, g.position, n, flux, u);
        deposited = true;
    }

    [mutating]
    public void light(uint id, float contribution) {}

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint layers)
    {
        if (deposited)
            InterlockedAdd(r.caustics.source.deposited[entry], 1u);
    }
}

// How a path starts: the interiors around it and the camera ray its first iteration traces.
public struct PathStart
{
    public float3 origin;
    public float3 dir;
    public float t_max;             // m along `dir`
    public float offset;            // m from the eye to `origin`
    public float axis;              // cosine between `dir` and the view axis
    public uint mask;               // 0: no ray, which starts the path at the G-buffer's surface
    public bool gbuffer;            // the ray only looks for layers before the G-buffer's surface
    public Interior interior;
    public float3 throughput;       // what the path carries from its start
    // The camera ray's hit and its corners, given in place of tracing it.
    public bool given;
    public Hit hit;
    public float3 positions[3];
    // A photon's light is a directional light, along `-dir`.
    public bool directional;
}

// The plane stack a path's ray crosses since the vertex that last scattered it, `x`, which
// next-event estimation there reaches a directional light through; for a photon, since its
// light.
public struct PathStack
{
    public bool valid;              // every vertex since `x` is a stack face or a null crossing, in a scene with stack entries
    public uint crossed;            // stack faces so far
    public float3 normal;           // the first face's shading normal
    public float3 origin;           // `x`'s ray origin
    public float3 dir;              // the ray's direction there
    public float3 x_normal;         // `x`'s shading normal, zero in a medium
    public float pdf;               // `x`'s density of `dir`
    public float run;               // `x`'s next-event probability
    public float ior;               // of the medium `dir` leaves `x` into
    public uint depth;              // `x`'s
}

// Whether next-event estimation at a vertex of depth `depth` whose rays leave from `origin`
// reaches a directional light along `dir` through the plane stack `stack` the path crossed: its
// probe meets a face parallel to the stack's first. The probe's alpha tests report texture
// feedback with `feedback` (`HitContext.feedback`).
public bool stack_owns<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, PathStack stack, uint depth, bool dispersion, bool feedback)
{
    if (!stack.valid || stack.crossed == 0)
        return false;
    float3 normal;
    RayCone cone = { 0.0, 0.0 };
    HitContext context = hit_context(HIT_SECONDARY, depth + 1, cone, 0.0, float2(-1.0), feedback);
    return stack_probe<CoverageFeedback<S>>(s, v, origin, dir, RAY_FLAG_NONE, context, dispersion, normal)
        && abs(dot(normal, stack.normal)) >= STACK_PARALLEL;
}

// Iterations of a path: a camera ray, then a vertex and its ray each, and a subsurface walk, or
// under the reference one at every opaque vertex. An interior's walk takes none of its own: it
// runs where a segment scatters, and its way out is a vertex that counts as a bounce.
public uint path_iterations(PathRoot r)
{
    uint walks = (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0 ? r.max_bounces + 1 : 1;
    return r.max_bounces + 3 + walks + r.dielectric_depth;
}

// The path from `start` through linked surface `S` and medium `M`; `C` names the coherence point
// after each trace, and `record` sees its vertices. With `CACHE_TERMINATE`, a front-face hit that
// `record` allows, reached by a ray longer than `CACHE_MIN_RAY_CELLS` edges of its cell, ends in
// its resolved cell unless light leaves it as a caustic. Returns unexposed radiance; `exposed` is
// the G-buffer's emission, which is already pre-exposed, and `sky` the throughput of a camera ray
// that met nothing. `rays` counts every ray traced.
//
// Light the scene's lights send through chain vertices (caster surfaces smoother than
// CHAIN_ROUGHNESS, met by their non-diffuse lobes) to a vertex's diffuse lobes belongs to the
// caustic photons, which the vertex gathers: what a later vertex or ray finds of any light but
// the environment through such a chain keeps only the share of the vertex's BSDF value outside
// its diffuse lobes. A chain vertex with diffuse lobes of its own goes on as a chain or as such a
// vertex, by their shares of its value. A directional light's through a plane stack is next-event
// estimation's (`path_light`) and the BSDF rays', by all of the vertex's lobes: photons from it
// deposit none of it, and a ray that crossed the stack weighs the light against next-event
// estimation at the vertex before the stack. A ray through a thin wall that is an ideal window
// keeps what the vertex before it gave it: that light is straight, and shadow rays take it.
public float3 trace_path<C : ICoherence, S : ISurface, M : IMedium, R : IPathRecord>(PathRoot r, PathStart start, uint2 pixel, inout Sampler rng, inout uint rays, inout R record, out float3 exposed, out float3 sky)
{
    PathResume none = (PathResume)0;
    return resume_path<C, S, M>(r, start, none, pixel, rng, rays, record, exposed, sky);
}

// Where a path whose vertices another kernel shaded so far goes on in `trace_path`'s loop: at
// the trace of iteration `iteration`, whose vertex left ray `ray`. The path's interiors and
// throughput come in its `PathStart`; the rest of what the loop carries is here.
public struct PathResume
{
    public bool on;
    public uint iteration;
    public uint depth;
    public uint opaque;
    public uint layers;
    public uint chains;
    public bool walked;
    public bool dispersive;
    public bool through;
    public float3 radiance;
    public float3 keep;
    public RayCone cone;
    public uint entered;
    public float3 entered_at;
    public float3 entered_normal;
    public LayerRay ray;
    public PathStack stack;
    public Spectral spectral;
}

// `trace_path`, or with `resume.on` the rest of a path from where `resume` says: `start` then
// gives only its interiors and throughput.
public float3 resume_path<C : ICoherence, S : ISurface, M : IMedium, R : IPathRecord>(PathRoot r, PathStart start, PathResume resume, uint2 pixel, inout Sampler rng, inout uint rays, inout R record, out float3 exposed, out float3 sky)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    exposed = float3(0.0);
    sky = float3(0.0);
    bool photon = record.photon();
    bool count_nee = r.mode != PATH_BSDF_ONLY && !photon;
    bool count_bsdf = r.mode != PATH_NEE_ONLY && !photon;
    bool one_technique = r.mode == PATH_NEE_ONLY || r.mode == PATH_BSDF_ONLY;
    bool reference = (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
    bool cache = (r.cache_flags & CACHE_TERMINATE) != 0 && !one_technique && r.mode != PATH_DIRECT;
    bool disperse = record.spectral() && (r.dielectric_flags & DIELECTRIC_DISPERSION) != 0;
    // One technique alone carries caustics itself.
    bool caustics = r.caustics != nullptr && !one_technique && !photon && caustics_hold_light(r.caustics);
    uint seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    // A photon's cone starts at a light, so its footprints say nothing of what the view reads.
    // The coherence points group the lanes that report, in the key's least significant bit, so
    // an implementation that keeps fewer bits drops it first.
    bool feedback = !photon && texture_feedback_drawn(pixel, r.frame);

    Interior interior = start.interior;
    // The froxels hold a camera segment through an interior where this frame has them.
    bool carried = interior.count > 0 && r.froxels == nullptr;
    LayerVertex l = (LayerVertex)0;
    Bsdf b = bsdf_unprepared();
    uint depth = 0;
    RayCone cone = camera_cone(v);
    bool walked = false;
    Spectral spectral = Spectral();
    // Whether `l` is a layer with a dispersive side.
    bool dispersive = false;
    float3 throughput = start.throughput;
    float3 radiance = float3(0.0);
    bool alive = true;
    // Opaque vertices and scatters so far, which the bounce settings count.
    uint opaque = 0;
    // Layer vertices so far, which `dielectric_depth` limits.
    uint layers = 0;
    // What light other than the environment reaching the path through a chain keeps: the
    // share of the last vertex that scattered outside its diffuse lobes, per channel. Set once
    // the path crossed a chain vertex since that vertex.
    float3 keep = float3(1.0);
    bool through = false;
    // A photon's chain vertices so far.
    uint chains = 0;
    PathStack stack = (PathStack)0;
    stack.valid = photon && start.directional && set.stack_entries != 0;
    // The interior the path last entered from outside and where, where a walk in that interior
    // leaves at its cap.
    uint entered = MATERIAL_NONE;
    float3 entered_at = float3(0.0);
    float3 entered_normal = float3(0.0);
    float3 leave_dir = float3(0.0);
    InteriorExitScatter leave_scatter = (InteriorExitScatter)0;
    // A scatter that took the cache's light: it estimates its direct light and ends.
    bool last = false;
    // Set while `l` is a walk's exit, which scatters as `exit_scatter`.
    bool exit = false;
    WalkExitScatter exit_scatter = (WalkExitScatter)0;
    LayerRay ray = (LayerRay)0;
    ray.origin = start.origin;
    ray.start = start.origin;
    ray.dir = start.dir;
    uint points = walk_points(r);
    if (resume.on)
    {
        depth = resume.depth;
        opaque = resume.opaque;
        layers = resume.layers;
        chains = resume.chains;
        walked = resume.walked;
        dispersive = resume.dispersive;
        through = resume.through;
        radiance = resume.radiance;
        keep = resume.keep;
        cone = resume.cone;
        entered = resume.entered;
        entered_at = resume.entered_at;
        entered_normal = resume.entered_normal;
        ray = resume.ray;
        stack = resume.stack;
        spectral = resume.spectral;
    }
    // Set for the first iteration, which traces the camera ray (or takes the hit it was given)
    // and leaves no vertex. Without one the path starts on the G-buffer's surface. Keeping that
    // out of the loop keeps the loop's coherence point out of a branch, which costs far more
    // than the second load.
    bool camera = !resume.on && start.mask != 0;
    if (!camera && !resume.on)
    {
        PathVertex x;
        if (!gbuffer_vertex(s, v, r.gbuffer, pixel, x))
        {
            sky = throughput;
            record.finish(r, radiance, layers);
            return radiance;
        }
        l = opaque_vertex(x);
        l.chains = surface_chains(r, x.surface);
        opaque = 1;
        cone.width = cone_width(cone, x.camera_distance);
        bool walks = record.walks() && may_walk(r, x.surface, 0, walked, cone.width);
        b = prepare_vertex(s, l, interior, false, walks, record.adjoint(), 0.0);
        record.arrive(r, l.x, b, 0.0, false);
        if (r.mode != PATH_INDIRECT && r.mode != PATH_SUBSURFACE)
            exposed = b.emission();
        alive = (x.surface.flags & GBUFFER_UNLIT) == 0;
        if (interior.count == 0)
        {
            alive = alive && runs_path(r, pixel);
            throughput /= r.pixel_fraction;
        }
        if (alive)
            record.vertex(r, l.x, b, throughput, radiance, rng);
    }

    // A walk takes an iteration of its own, and its exit vertex the next one at the same
    // depth.
    for (uint i = resume.on ? resume.iteration : 0; i < path_iterations(r); i++)
    {
        // A resumed path's first iteration starts at its trace: its vertex is shaded.
        bool resumed = resume.on && i == resume.iteration;
        if (!resumed && (!alive || (!camera && l.kind != VERTEX_LAYER && opaque > r.max_bounces + 1)))
            break;
        bool counted = r.mode == PATH_DIRECT ? depth == 0 : r.mode == PATH_INDIRECT ? depth > 0
            : r.mode == PATH_SUBSURFACE ? walked : true;
        Hit h = (Hit)0;
        float3 positions[3] = { float3(0.0), float3(0.0), float3(0.0) };
        // A dispersive layer with live companions adds no direct light: its BSDF differs by
        // wavelength, and its rays reach lights at full weight.
        bool split = dispersive && spectral.companions();
        // Constant false for records that never walk, so their kernels hold no exit code.
        bool at_exit = record.walks() && exit;
        bool walking = false;
        Walk walk = Walk();
        LocalLights* local = nullptr;
        if (depth == 0)
            local = pixel_local_lights(r, pixel);
        if (!camera && !resumed)
        {
            bool medium_vertex = l.kind == VERTEX_SCATTER;
            bool leaving = l.kind == VERTEX_LEAVE;
            bool on_layer = l.kind == VERTEX_LAYER;
            // A photon deposits where it meets a surface after a chain, and goes on only as one.
            // Light that next-event estimation takes through a plane stack is not deposited.
            if (photon)
            {
                if (chains > 0 && !medium_vertex && !leaving && !stack_owns<S>(s, v, vertex_origin(l.x, -start.dir), -start.dir, stack, depth, disperse, feedback))
                    record.deposit(r, l, b, spectral.color(throughput), rng);
                if (!l.chains)
                    break;
            }
            // A chain vertex on a solid's back sees lights from inside it, which BSDF rays carry.
            // Next-event estimation at a walk's way out takes the light inside as uniform, so in
            // the reference the way out's ray alone carries light.
            float run = leaving ? (reference ? 0.0 : 1.0) : medium_vertex || at_exit ? 1.0
                : on_layer && !l.front && l.chains && crosses_interior(s, l.section) ? 0.0 : nee_probability(set, l.x.surface);
            float u_run = rng.next();
            uint candidates = depth == 0 ? r.primary_candidates : r.bounce_candidates;
            if (count_nee && candidates > 0 && u_run < run && !split)
            {
                PathShadow<CoverageFeedback<S>, M> shadow = vertex_shadow<CoverageFeedback<S>, M>(r, l, interior, disperse);
                // Layers and scatters light from both sides.
                PathVertex at = l.x;
                if (on_layer || medium_vertex)
                    at.surface.normal = float3(0.0);
                uint light;
                VertexScatter f = { b, medium_vertex, -l.x.view, l.anisotropy, at_exit, exit_scatter, leaving, leave_scatter };
                // The rays of a walk's way out reach no light this estimate is weighed against.
                float3 e = shade_vertex_nee<S>(r, at, f, shadow, run, pixel, local, depth, candidates, one_technique || leaving, interior_ior(s, interior.innermost()),
                                               one_technique ? float3(1.0) : keep, l.chains && !one_technique, feedback, rng, rays, light, leaving && count_bsdf);
                float3 added = spectral.color(throughput * e);
                if (counted)
                    radiance += added;
                if (light != NO_LIGHT)
                    record.light(light, luminance_rec2020(added));
            }
            // The caustic photons' light at the vertex's diffuse lobes, which is indirect.
            bool gathered = r.mode == PATH_DIRECT ? false : r.mode == PATH_INDIRECT ? true : counted;
            if (caustics && gathered && !medium_vertex && !leaving)
                radiance += spectral.color(throughput * caustic_gather<S>(r, l, b, at_exit, exit_scatter, cone.width, rng));
            if (last)
                alive = false;
            if (alive)
            {
                float3 wi;
                float pdf;
                bool delta = false;
                bool diffuse = false;
                float spread;
                bool crossed = false;
                bool null_crossing = false;
                bool stack_crossing = false;
                float4 u = rng.next4();
                float u_roulette = rng.next();
                float u_role = rng.next();
                uint component = 0;
                float3 hero_f = float3(0.0);
                if (medium_vertex)
                {
                    wi = sample_hg(l.anisotropy, -l.x.view, u.xy);
                    pdf = hg_phase(l.anisotropy, dot(-l.x.view, wi));
                    spread = 1.0;
                    keep = float3(1.0);
                    through = false;
                }
                else if (leaving)
                {
                    wi = leave_dir;
                    pdf = 1.0;
                    delta = true;
                    diffuse = true;
                    spread = 1.0;
                    keep = float3(1.0);
                    through = false;
                }
                else
                {
                    BsdfSample bs;
                    if (at_exit)
                        bs = walk_exit_sample(exit_scatter, rng);
                    else
                        bs = b.sample(u.xyz);
                    alive = bs.pdf > 0.0;
                    wi = bs.wi;
                    pdf = bs.pdf;
                    component = bs.component;
                    hero_f = bs.weight * bs.pdf;
                    delta = (bs.lobe & LOBE_SPECULAR) != 0;
                    diffuse = (bs.lobe & LOBE_DIFFUSE) != 0;
                    float roughness = l.x.surface.bsdf.specular_roughness;
                    spread = delta ? 0.0 : diffuse ? 1.0 : 2.0 * roughness * roughness;
                    // The walk takes the place of an opaque vertex's ray.
                    walking = record.walks() && alive && (bs.lobe & LOBE_VOLUME) != 0;
                    crossed = on_layer && (bs.lobe & LOBE_TRANSMISSION) != 0 && crosses_interior(s, l.section);
                    null_crossing = on_layer && delta && (bs.lobe & LOBE_TRANSMISSION) != 0 && thin_walled(l.x.surface);
                    stack_crossing = crossed && delta && stack.valid && stack.crossed < STACK_CROSSINGS && l.stack
                        && (stack.crossed == 0 || abs(dot(l.x.surface.normal, stack.normal)) >= STACK_PARALLEL);
                    float3 chained = max(bs.weight - bs.diffuse, 0.0);
                    if (null_crossing || walking)
                        throughput *= bs.weight;
                    else if (photon)
                    {
                        throughput *= chained;
                        chains++;
                    }
                    else if (l.chains && !one_technique)
                    {
                        // Its diffuse lobes' share of the value goes on as the vertex's own light, the
                        // rest as a chain.
                        float share = luminance_rec2020(bs.diffuse) / max(luminance_rec2020(bs.weight), 1e-30);
                        if (u_role < share)
                        {
                            throughput *= bs.diffuse / share;
                            keep = float3(0.0);
                            through = false;
                        }
                        else
                        {
                            throughput *= chained / (1.0 - share);
                            through = true;
                        }
                    }
                    else
                    {
                        throughput *= bs.weight;
                        if (!one_technique)
                            keep = select(bs.weight > 0.0, chained / bs.weight, float3(0.0));
                        through = false;
                    }
                }
                if (alive && !walking && !on_layer && opaque > r.roulette_depth)
                {
                    float survive = saturate(max_component(throughput));
                    alive = u_roulette < survive;
                    throughput /= max(survive, 1e-6);
                }
                alive = alive && any(throughput > 0.0) && all(isfinite(throughput));
                if (alive && split)
                {
                    for (uint k = 1; k < WAVELENGTHS; k++)
                    {
                        if (!(spectral.density[k] > 0.0))
                            continue;
                        Bsdf bk = prepare_vertex(s, l, interior, opaque > 0 && !reference, false, record.adjoint(), spectral.wavelength(k));
                        float pk;
                        float3 fk;
                        if (delta)
                            fk = bk.delta_value(component, wi, pk);
                        else
                            fk = bk.eval_and_pdf(wi, pk);
                        spectral.weigh(k, fk, pk, hero_f, pdf);
                    }
                    spectral.update();
                }
                if (alive && walking)
                    walk = walk_from(l.x, vertex_params(l, opaque > 1 && !reference), wi, rng);
                else if (alive)
                {
                    if (crossed)
                    {
                        interior.cross(l.section, l.front, l.x.surface.medium_point);
                        if (l.front)
                        {
                            entered = l.section;
                            entered_at = l.x.surface.position;
                            entered_normal = l.x.surface.geometric_normal;
                        }
                    }
                    ray.origin = vertex_origin(l.x, wi);
                    ray.start = l.x.surface.position;
                    ray.dir = wi;
                    ray.chain = l.kind != VERTEX_OPAQUE;
                    // An ideal window passes the ray on as it reached it.
                    if (!null_crossing)
                    {
                        ray.pdf = delta ? 0.0 : pdf;
                        ray.run = count_nee && !split ? run : 0.0;
                        ray.normal = l.kind == VERTEX_OPAQUE ? l.x.surface.normal : float3(0.0);
                        ray.from = l.x.surface.position;
                        ray.diffuse = diffuse;
                        ray.scatter = record.walks() && (reference || depth == 0);
                        ray.local = depth == 0;
                        ray.lit = leaving && !reference && count_nee && !one_technique;
                        if (stack_crossing)
                        {
                            if (stack.crossed == 0)
                                stack.normal = l.x.surface.normal;
                            stack.crossed++;
                        }
                        else if (photon)
                            stack.valid = false;
                        else
                        {
                            stack.valid = ray.pdf > 0.0 && !one_technique && set.stack_entries != 0;
                            stack.crossed = 0;
                            stack.origin = ray.origin;
                            stack.dir = wi;
                            stack.x_normal = ray.normal;
                            stack.pdf = ray.pdf;
                            stack.run = ray.run;
                            stack.ior = interior_ior(s, interior.innermost());
                            stack.depth = depth;
                        }
                    }
                    cone.spread += spread;
                }
            }
        }
        float t = 0.0;
        uint medium = MATERIAL_NONE;
        bool scatters = false;
        bool interior_walk = false;
        float3 scatter_at = float3(0.0);
        float scatter_anisotropy = 0.0;
        // What owned light the ray finds keeps.
        float3 own = ray.lit ? float3(0.0) : through ? keep : float3(1.0);
        if (alive && !walking)
        {
            if (camera && start.given)
            {
                h = start.hit;
                positions = start.positions;
            }
            else
            {
                uint mask = camera ? start.mask : MASK_OPAQUE | MASK_DIELECTRIC;
                HitContext context = hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, camera ? 0 : depth + 1, cone, 0.0, float2(-1.0), feedback);
                LayerAccept<CoverageFeedback<S>> a = { s, v, ray.dir, context, camera ? seed : pcg(seed ^ (depth + 1)), PRESENCE_DRAWN };
                rays += camera ? 0 : 1;
                RayDesc rd = path_ray(ray.origin, ray.dir, camera ? start.t_max : positive_infinity());
                h = trace_closest(s.tlas, rd, mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
                // Rays leaving a layer or a scatter, and the first vertex's, see particles, and
                // every path ray that adds emission sees the proxies of sphere and rect lights,
                // which cover nothing: a second trace finds them in front of the hit.
                bool adds = count_bsdf && counted;
                bool particles = record.particles() && !camera && (ray.chain || depth == 0) && set.particle_entries > 0;
                bool lights = !camera && adds && set.local_lights > 0;
                if (particles || lights)
                {
                    LocalLights* ray_local = nullptr;
                    if (ray.local)
                        ray_local = pixel_local_lights(r, pixel);
                    RayDesc front = path_ray(ray.origin, ray.dir, h.hit ? h.t : positive_infinity());
                    PathEmission pe = { s, set, ray_local, ray.origin, ray.dir, ray.from, ray.normal, one_technique ? 0.0 : ray.pdf, ray.run, front.TMax, own, adds };
                    // What lies in front of the ray's hit, premultiplied.
                    float4 layers;
                    trace_resolve_translucent(s.tlas, front, (particles ? MASK_PARTICLE : 0) | (lights ? MASK_LIGHT : 0), RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, pe, PARTICLE_CANDIDATES, layers);
                    if (adds)
                        radiance += spectral.color(throughput * layers.rgb);
                    throughput *= 1.0 - layers.a;
                }
            }
            if (!camera)
                depth++;
            t = h.hit ? h.t : camera ? start.t_max : positive_infinity();
            if (!camera || carried)
                medium = interior.innermost();
            if (medium != MATERIAL_NONE)
            {
                InteriorMedium im = interior_medium(s, medium, interior.point(medium));
                scatter_anisotropy = im.anisotropy;
                // The segment runs from the surface point its offset origin left.
                float span = segment_length(ray.start, ray.origin, ray.dir, t);
                // A walk takes a bounce; a ray leaving an opaque vertex takes transmittance only.
                bool walks = record.walks() && ray.chain && (reference || !walked) && opaque < r.max_bounces
                    && max_component(im.sigma_s) * span >= 1.0;
                float scattered;
                scatters = interior_segment(im, span, ray.scatter || walks || (camera && !start.given), rng, throughput, scattered);
                if (scatters)
                {
                    scatter_at = ray.origin + ray.dir * max(scattered - (span - t), 0.0);
                    cone = cone_after_hit(cone, scattered, 0.0);
                    opaque++;
                    if (walks)
                    {
                        interior_walk = true;
                        walked = true;
                        float ior = interior_ior(s, medium) / interior_ior(s, interior.outside(medium, medium));
                        walk = walk_interior(scatter_at, ray.dir, im, ior, rng);
                    }
                }
            }
        }
        if (record.walks())
        {
            // The last part of a walk leads to its lanes' own point, or with an interior's walk
            // that met another surface, to the iteration's main point.
            for (uint p = 0; p <= points && walk.kind == WALK_RUNNING; p++)
            {
                if (walking)
                    walk_part(r, walk, walk_tracer<CoverageFeedback<S>>(r, l.x.surface.instance, depth, feedback), rays);
                else
                    walk_part(r, walk, interior_tracer<CoverageFeedback<S>>(r, medium, walk.ior, pcg(seed ^ (depth + 1)), depth, feedback), rays);
                if (p < points)
                    C.point(uint(walk.kind == WALK_RUNNING), 1);
            }
            if (alive && (walking || interior_walk))
            {
                throughput *= walk.throughput;
                // The reference ends a path whose walk reached its cap, and counts it.
                bool capped = walk.kind == WALK_CAP;
                if (capped && reference)
                    InterlockedAdd(set.reference_counts[0], 1u);
                alive = walk.kind != WALK_NONE && (!capped || (!reference && (walking || entered == medium)));
                if (walking || walk.kind == WALK_BOUNDARY)
                {
                    h = walk.hit;
                    positions = walk.positions;
                }
            }
        }
        if (!alive)
            break;
        exit = false;
        if (walking)
        {
            C.point(path_key(s, h) << 1 | uint(feedback), KEY_BITS + 1);
            walked = true;
            BsdfParams p = vertex_params(l, opaque > 1 && !reference);
            if (walk.kind == WALK_BOUNDARY)
            {
                HitContext context = hit_context(HIT_SECONDARY, depth + 1, cone, h.t, texture_tap(v, pixel, r.frame, r.seed, depth + 1), feedback);
                SurfaceHit sh = resolve_hit<FeedbackSurface<S>>(s, v, walk.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
                alive = sh.valid;
                if (alive)
                    l = opaque_vertex(exit_vertex(v, sh, walk.dir));
                if (alive && sh.surface.bsdf.subsurface_weight > 0.0)
                    p = sh.surface.bsdf;
            }
            else
                l.x.view = l.x.surface.normal;
            // A walk's exit scatters by its own diffuse lobe, never as a chain.
            l.chains = false;
            exit = true;
            exit_scatter = walk_exit(l.x, p, rng);
            continue;
        }
        C.point(path_key(s, h) << 1 | uint(feedback), KEY_BITS + 1);
        // Every lane takes a new vertex past the point. These writes tell the compiler so, which
        // keeps the old vertex out of the state the point holds.
        l = (LayerVertex)0;
        b = bsdf_unprepared();
        exit_scatter = (WalkExitScatter)0;
        leave_scatter = (InteriorExitScatter)0;
        leave_dir = float3(0.0);
        bool first = camera;
        camera = false;

        if (medium != MATERIAL_NONE)
        {
            if (scatters && !interior_walk)
            {
                l = scatter_vertex(v, scatter_at, ray.dir, medium, scatter_anisotropy);
                if (first)
                    record.arrive(r, l.x, b, start.offset + distance(ray.origin, scatter_at), false);
                if (cache && !reference)
                {
                    if (counted)
                        radiance += spectral.color(throughput * interior_ambient(r, l.x.surface.position, ray.dir, rng));
                    last = true;
                }
                continue;
            }
            if (interior_walk)
            {
                if (walk.kind != WALK_BOUNDARY)
                {
                    float3 from = walk.origin;
                    float3 n = walk.normal;
                    leave_dir = walk.dir;
                    if (walk.kind == WALK_CAP)
                    {
                        from = entered_at;
                        n = entered_normal;
                        float3 tangent, bitangent;
                        orthonormal_basis(n, tangent, bitangent);
                        leave_dir = normalize(from_local(sample_cosine_hemisphere(rng.next2()), tangent, bitangent, n));
                    }
                    interior.cross(medium, false);
                    l = leave_vertex(v, from, n);
                    leave_scatter = interior_exit(n, walk.ior);
                    // The way out is a refraction: the walk's Fresnel draw chose it, and it
                    // scales radiance as a BSDF's does.
                    if (!record.adjoint())
                        throughput *= walk.ior * walk.ior;
                    dispersive = false;
                    continue;
                }
                // The walk met another surface inside: it shades as this ray's hit.
                t = h.t;
                ray.dir = walk.dir;
                ray.pdf = 0.0;
                ray.diffuse = true;
                ray.local = false;
            }
        }
        else if (!first)
        {
            if (h.hit && (ray.chain || t >= INSCATTER_SEGMENT))
            {
                float3 through_medium;
                float3 e;
                if (ray.chain)
                    e = froxel_segment_inscatter<M>(r.froxels, v, set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
                else
                    e = segment_inscatter<M>(set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
                if (counted && !photon)
                    radiance += spectral.color(throughput * e);
                throughput *= through_medium;
            }
            else
            {
                // The march's jitter dimension, drawn either way so later dimensions stay put.
                rng.next();
                if (h.hit)
                    throughput *= medium_transmittance<M>(s.medium, ray.origin, ray.dir, t);
            }
        }
        if (!any(throughput > 0.0) || !all(isfinite(throughput)))
        {
            alive = false;
            continue;
        }

        bool weigh = !one_technique && ray.pdf > 0.0;
        LocalLights* ray_local = nullptr;
        if (ray.local)
            ray_local = pixel_local_lights(r, pixel);
        // A pinhole camera ray only looks for layers: without one the path starts on the
        // G-buffer's surface.
        bool gbuffer = first && start.gbuffer && !is_layer(h);
        if (!h.hit && !gbuffer)
        {
            if (first)
                sky = throughput;
            else if (count_bsdf && counted)
            {
                float3 env = environment_map(set.environment, ray.dir);
                float w = weigh ? mis_balance(ray.pdf, ray.run * nee_environment_pdf(set, ray_local, ray.normal, ray.dir)) : 1.0;
                float3 e = env * w;
                record.light(feedback_environment(set), luminance_rec2020(spectral.color(throughput * e)));
                for (uint k = 0; k < set.directional_count; k++)
                {
                    uint id = set.directional_ids[k];
                    float3 d = directional_disc<M>(set, id, ray.origin, ray.dir);
                    if (!any(d > 0.0))
                        continue;
                    // Through a plane stack next-event estimation at the vertex before it reaches
                    // the light too, by all of the vertex's lobes.
                    if (stack_owns<S>(s, v, stack.origin, ray.dir, stack, stack.depth, disperse, feedback))
                    {
                        float jacobian = abs(dot(ray.dir, stack.normal)) / abs(dot(stack.dir, stack.normal));
                        e += d * mis_balance(stack.pdf * jacobian / (stack.ior * stack.ior), stack.run * nee_directional_pdf(set, stack.x_normal, id, ray.dir));
                    }
                    else
                        e += d * own * (weigh ? mis_balance(ray.pdf, ray.run * nee_directional_pdf(set, ray.normal, id, ray.dir)) : 1.0);
                }
                radiance += spectral.color(throughput * e);
            }
            alive = false;
            continue;
        }

        bool layer = false;
        bool front = true;
        // What the light set samples of the hit, and the side it shows.
        EmitterHit emitter = (EmitterHit)0;
        bool swapped = false;
        float cos_light = 0.0;
        if (gbuffer)
        {
            PathVertex x;
            if (!gbuffer_vertex(s, v, r.gbuffer, pixel, x))
            {
                sky = throughput;
                alive = false;
                continue;
            }
            l = opaque_vertex(x);
            l.chains = surface_chains(r, x.surface);
            opaque++;
        }
        else
        {
            float path_t = first ? start.offset + t : t;
            HitContext context = hit_context(first ? HIT_PRIMARY : HIT_SECONDARY, depth, cone, path_t, texture_tap(v, pixel, r.frame, r.seed, depth), feedback);
            SurfaceHit sh = resolve_hit<FeedbackSurface<S>>(s, v, ray.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
            if (!sh.valid)
            {
                alive = false;
                continue;
            }
            if (!first)
                cone = cone_after_hit(cone, t, 0.0);
            // Back faces of single-sided surfaces never emit; double-sided hits face the ray.
            cos_light = dot(sh.surface.geometric_normal, ray.dir);
            front = sh.back_face || cos_light < 0.0;
            layer = is_layer(h);
            if (layer)
            {
                layers++;
                if (layers > r.dielectric_depth && !first)
                {
                    if (cache && counted)
                        radiance += spectral.color(throughput * interior_ambient(r, sh.surface.position, ray.dir, rng));
                    alive = false;
                    continue;
                }
                l = layer_vertex(v, sh, ray.dir);
                StackFace face;
                l.stack = stack.valid && stack_face(s, h, disperse, face);
            }
            else
            {
                l = opaque_vertex(hit_vertex(v, sh, ray.dir));
                opaque++;
            }
            l.chains = surface_chains(r, sh.surface);
            swapped = sh.swapped;
        }
        if (first)
            cone.width = cone_width(cone, l.x.camera_distance);
        dispersive = disperse && layer_disperses(s, l, interior);
        if (dispersive && !spectral.on)
            spectral.start(wavelength_sample(r, pixel));
        // Coat, fuzz and film collapse once the path has scattered, outside the reference.
        uint before = layer ? opaque : opaque - 1;
        bool collapse = before > 0 && !reference;
        bool walks = record.walks() && !layer && may_walk(r, l.x.surface, before, walked, cone.width);
        b = prepare_vertex(s, l, interior, collapse, walks, record.adjoint(), spectral.hero());
        if (first)
            record.arrive(r, l.x, b, gbuffer ? length(l.x.surface.position - v.eye_render.xyz) : start.offset + t, layer);
        // Emission seen from the side that emits, weighed against the light set's sampling of
        // the point, which takes its density from the vertex the ray left; a point the light set
        // never draws takes weight 1, and as no photon leaves it, the whole of its light.
        if (gbuffer)
        {
            if (counted)
                exposed += throughput * b.emission();
        }
        else if ((count_bsdf || (first && !photon)) && counted && front && any(b.emission() > 0.0))
        {
            if (!first && (weigh || any(own != 1.0)))
                emitter = emitter_hit(set, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, swapped);
            float w = 1.0;
            if (weigh && emitter.density > 0.0)
            {
                float3 toward = emitter.point - ray.from;
                float distance = length(toward);
                float cos_to = distance > 0.0 ? dot(emitter.normal, toward / distance) : 0.0;
                w = mis_balance(ray.pdf, ray.run * nee_emissive_pdf(set, ray_local, ray.normal, emitter, distance, cos_to));
            }
            float3 kept = emitter.density > 0.0 ? own : float3(1.0);
            radiance += spectral.color(throughput * b.emission() * (w * kept));
        }
        if (layer)
        {
            alive = (l.x.surface.flags & GBUFFER_UNLIT) == 0;
            continue;
        }
        alive = (l.x.surface.flags & GBUFFER_UNLIT) == 0 && (first || r.mode != PATH_DIRECT);
        // The first vertex of a pixel with no layer runs on the view's share of pixels.
        if (first && interior.count == 0)
        {
            alive = alive && runs_path(r, pixel);
            throughput /= r.pixel_fraction;
        }
        // The cache holds a vertex's light by its albedo, which a caustic through it would count
        // again in the photons a vertex before it gathers.
        if (alive && cache && front && !first && !l.chains)
        {
            CacheCell cell = cache_cell(r.cache, s, l.x.surface.position, l.x.surface.geometric_normal, rng.next4().xyz, true);
            float3 cached;
            if (record.may_end(opaque - 1, ray.diffuse, cone.width > cell.size, opaque == r.max_bounces + 2)
                && t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
            {
                if (counted)
                    radiance += spectral.color(throughput * cache_albedo(b.albedo(), l.x.surface.bsdf) * cached);
                alive = false;
                continue;
            }
        }
        // The last bounce's hit scatters nothing the path estimates.
        if (alive && opaque <= r.max_bounces + 1)
            record.vertex(r, l.x, b, throughput, radiance, rng);
    }
    record.finish(r, radiance, layers);
    return radiance;
}

// The first vertex of `pixel` from the G-buffer; false for sky pixels.
public bool gbuffer_vertex(SceneRoot* s, ViewConstants* v, GBufferGpu* gbuffer, uint2 pixel, out PathVertex x)
{
    x = (PathVertex)0;
    GSurface g = load_gsurface(v, gbuffer, pixel);
    if (g.material == MATERIAL_NONE)
        return false;
    x.surface = g;
    float3 to_eye = v.eye_render.xyz - g.position;
    x.camera_distance = length(to_eye);
    x.view = to_eye / max(x.camera_distance, 1e-30);
    x.origin_error = max(g.raster_error, traced_error(s, g.instance));
    x.plane_point = g.position;
    return true;
}

// A thin lens's camera ray of `pixel`: from a point of the aperture disc of radius `aperture` m
// through a jittered point of the pixel on the focal plane `focus` m along the view axis, starting
// `from` m along that axis. Draws four sampler dimensions.
public PathStart lens_start(PathRoot r, uint2 pixel, float aperture, float focus, float from, uint mask, inout Sampler rng)
{
    ViewConstants* v = r.view;
    float2 uv = (float2(pixel) + rng.next2()) / float2(v.extent);
    float3 through = view_direction(v, uv);
    float3 at_focus = through * (focus / max(-through.z, 1e-6));
    float3 lens = float3(sample_disk_concentric(rng.next2()) * aperture, 0.0);
    float3 view_dir = normalize(at_focus - lens);
    PathStart start = (PathStart)0;
    start.offset = from / max(-view_dir.z, 1e-6);
    start.origin = render_from_view(v, lens + view_dir * start.offset);
    start.dir = render_direction(v, view_dir);
    start.t_max = positive_infinity();
    start.axis = -view_dir.z;
    start.mask = mask;
    start.interior = Interior();
    start.throughput = float3(1.0);
    return start;
}

// The interiors at `origin`, a point of the lens: `camera_medium`, flipped at each surface between
// the eye and it.
public Interior lens_interior<S : ISurface>(PathRoot r, float3 origin)
{
    ViewConstants* v = r.view;
    Interior interior = Interior();
    if (r.camera_medium != MATERIAL_NONE)
        interior.cross(r.camera_medium, true);
    float3 eye = v.eye_render.xyz;
    float t = length(origin - eye);
    surface_crossings<S>(r.scene, v, eye, (origin - eye) / t, t, interior);
    return interior;
}

// The camera ray of `pixel` of view `v` from the near plane: toward the G-buffer `gb`'s
// surface, where the tile may hold a layer before it.
public PathStart pinhole_start(ViewConstants* v, GBufferGpu* gb, uint2 pixel, bool traced)
{
    float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
    PathStart start = (PathStart)0;
    start.dir = render_direction(v, view_direction(v, uv));
    start.origin = render_from_view(v, view_position(v, uv, 1.0));
    start.offset = length(start.origin - v.eye_render.xyz);
    float depth = gb.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    start.t_max = depth > 0.0 ? max((length(view_position(v, uv, depth)) - start.offset) * (1.0 - 1e-5), 0.0) : positive_infinity();
    start.axis = 1.0;
    start.mask = traced ? MASK_OPAQUE | MASK_DIELECTRIC : 0;
    start.gbuffer = true;
    start.interior = Interior();
    start.throughput = float3(1.0);
    return start;
}

// The interiors around `pixel`'s G-buffer surface, which its `pinhole_start` ray `start`
// reaches, from this frame's layer memory `layers` (null for none): on a tile the layer
// guides traced, those at its near-plane point flipped at each surface the ray crosses before
// it; elsewhere none, since no layer lies before the surface and the eye lies in no medium.
public Interior gbuffer_interior<S : ISurface>(SceneRoot* s, ViewConstants* v, LayerGuide* layers, uint2 extent, uint2 pixel, PathStart start)
{
    if (layers == nullptr || traced_tiles(layers, extent)[layer_tile(extent, pixel)] == 0)
        return Interior();
    Interior interior = Interior(layers[pixel.y * extent.x + pixel.x].near_interior);
    if (isfinite(start.t_max))
        surface_crossings<S>(s, v, start.origin, start.dir, start.t_max, interior);
    return interior;
}

// Whether `pixel`'s path runs this frame, with probability `PathRoot.pixel_fraction`.
public bool runs_path(PathRoot r, uint2 pixel)
{
    Sampler subset = Sampler(pixel, r.frame, r.seed ^ SUBSET_STREAM);
    return r.pixel_fraction >= 1.0 || subset.next() < r.pixel_fraction;
}

// A debug view of `pixel`'s G-buffer surface alone.
public void first_vertex_view<S : ISurface>(PathRoot r, uint2 pixel, inout Sampler rng)
{
    ViewConstants* v = r.view;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    PathVertex x1;
    bool surface = gbuffer_vertex(r.scene, v, r.gbuffer, pixel, x1);
    float3 radiance = float3(0.0);
    if (surface && r.mode == PATH_CAUSTICS)
    {
        LayerVertex l = opaque_vertex(x1);
        Bsdf b = bsdf_prepare(x1.surface.bsdf, x1.surface.normal, x1.view);
        RayCone cone = camera_cone(v);
        float width = cone_width(cone, x1.camera_distance);
        if (r.caustics != nullptr)
            radiance = pre_exposure * caustic_gather<S>(r, l, b, false, (WalkExitScatter)0, width, rng);
    }
    else if (surface && r.mode != PATH_ORIGIN_BIAS)
    {
        CacheCell cell = cache_cell(r.cache, r.scene, x1.surface.position, x1.surface.geometric_normal, rng.next4().xyz, true);
        float3 cached;
        bool found = cache_lookup(r.cache, cell.key, cached);
        if (r.mode == PATH_CACHE)
            radiance = pre_exposure * cache_albedo(bsdf_prepare(x1.surface.bsdf, x1.surface.normal, x1.view).albedo(), x1.surface.bsdf) * cached;
        else
            radiance = float3(float(cell.level), found ? 1.0 : 0.0, unit_float(cache_hash(cell.key) * 2654435761u));
    }
    float4 out = float4(r.mode == PATH_CACHE_LOD ? radiance : apply_hdr_ceiling(radiance), surface ? 1.0 : 0.0);
    if (r.mode == PATH_ORIGIN_BIAS)
        out = float4(surface ? length(vertex_origin(x1, x1.surface.normal) - x1.surface.position) : 0.0, 0.0, 0.0, out.a);
    r.radiance.get<RWTexture2D<float4>>()[pixel] = out;
    r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(NO_LIGHT, asuint(surface ? x1.camera_distance : 0.0));
}

// The view paths' body for `pixel`: one path from the first layer of a traced tile or from the
// G-buffer's surface, or from a thin lens. Sky pixels take the linked environment and the
// directional discs. Pixels of tiles the layer guides did not trace start on the G-buffer's
// surface, with no camera ray. With DIELECTRIC_FROST, a pixel with a blur on its dominant layer
// also runs a second path from the G-buffer's surface, for what the frosted-glass term blurs.
public void view_body<C : ICoherence, S : ISurface, E : IEnvironment, M : IMedium>(PathRoot r, uint2 pixel)
{
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    Sampler rng = Sampler(pixel, r.frame, r.seed);
    uint2 extent = r.gbuffer.extent;
    uint index = pixel.y * extent.x + pixel.x;
    bool traced = traced_tiles(r.layers, extent)[layer_tile(extent, pixel)] != 0;
    if (r.mode == PATH_CACHE || r.mode == PATH_CACHE_LOD || r.mode == PATH_ORIGIN_BIAS || r.mode == PATH_CAUSTICS)
    {
        first_vertex_view<S>(r, pixel, rng);
        return;
    }
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    PathStart start;
    if (r.aperture_radius > 0.0)
    {
        start = lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_OPAQUE | MASK_DIELECTRIC, rng);
        start.interior = lens_interior<S>(r, start.origin);
    }
    else
    {
        start = pinhole_start(v, r.gbuffer, pixel, traced);
        if (traced)
            start.interior = Interior(r.layers[index].near_interior);
    }
    bool frost = (r.dielectric_flags & DIELECTRIC_FROST) != 0;
    uint passes = frost && traced && r.layers[index].frost != 0 ? 2 : 1;
    // The froxels hold a camera segment through an interior where this frame has them.
    bool carried = start.interior.count > 0 && r.froxels == nullptr;
    ViewPath view_path = ViewPath(pcg3d(uint3(pixel, r.frame ^ r.seed)).x);
    // What the pixel's own path left its record; the background's path adds nothing to them.
    ViewPath pixel_path = view_path;
    uint rays = 0;
    float3 total = float3(0.0);
    float background = 0.0;
    for (uint k = 0; k < passes; k++)
    {
        float3 exposed;
        float3 sky;
        float3 radiance = trace_path<C, S, M>(r, start, pixel, rng, rays, view_path, exposed, sky);
        float3 pixel_radiance = exposed + pre_exposure * radiance;
        if (any(sky > 0.0) && r.mode != PATH_INDIRECT && r.mode != PATH_SUBSURFACE)
            pixel_radiance += pre_exposure * sky * sky_radiance<E, M>(set, v.eye_render.xyz, start.dir);
        if (k == 0)
        {
            total = pixel_radiance;
            pixel_path = view_path;
            // The background's path starts on the G-buffer's surface, inside what holds it.
            if (passes > 1 && isfinite(start.t_max))
                surface_crossings<S>(r.scene, v, start.origin, start.dir, start.t_max, start.interior);
        }
        background = guide_luminance(pixel_radiance);
        start.mask = 0;
    }
    bool arrived = pixel_path.first_distance > 0.0;
    bool first_layer = pixel_path.first_layer;
    float4 out = float4(apply_hdr_ceiling(total), arrived ? 1.0 : 0.0);
    if (r.mode == PATH_RAYS)
        out = float4(float(rays), all(isfinite(total)) ? 0.0 : 1.0, 0.0, out.a);
    else if ((r.dielectric_flags & DIELECTRIC_DEBUG_LAYER) != 0 && first_layer)
    {
        float sum = max(pixel_path.reflected + pixel_path.transmitted, 1e-6);
        out = float4(pixel_path.reflected / sum, pixel_path.transmitted / sum, float(pixel_path.layers) / float(max(r.dielectric_depth, 1u)), 1.0);
    }
    r.radiance.get<RWTexture2D<float4>>()[pixel] = out;
    if (traced)
        r.layers[index].medium_end = carried ? -1.0 : first_layer ? pixel_path.first_t : 0.0;
    if (frost)
        r.layers[index].background = background;
    uint feedback = pixel_path.feedback;
    r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(feedback, asuint(pixel_path.first_distance));
    if ((r.light_flags & LIGHT_FEEDBACK_COUNT) != 0 && feedback != NO_LIGHT)
        count_feedback(set, feedback);
}

// The background layer's texel of `pixel`: on a marked pixel, the pre-exposed radiance of
// its lens ray past the mask's depth in RGB, the full path at the view path's settings,
// and the reverse-Z depth of its hit in A (0 on a miss); (0, 0, 0, -1) on other pixels.
float4 background_texel<C : ICoherence, S : ISurface, E : IEnvironment, M : IMedium>(PathRoot r, uint2 pixel)
{
    ViewConstants* v = r.view;
    float mask = r.dof_mask.get<Texture2D<float>>().Load(int3(pixel, 0));
    if (!(mask > 0.0))
        return float4(0.0, 0.0, 0.0, -1.0);
    PathRoot b = r;
    b.mode = PATH_FULL;
    b.pixel_fraction = 1.0;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float near = v.clip_from_view[2][3];
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ DOF_STREAM);
    PathStart start = lens_start(b, pixel, r.dof_aperture, *r.dof_focus, near / mask, MASK_OPAQUE | MASK_DIELECTRIC, rng);
    start.interior = Interior();
    uint rays = 0;
    ViewPath record = ViewPath(pcg3d(uint3(pixel, r.frame ^ r.seed ^ DOF_STREAM)).x);
    float3 exposed;
    float3 sky;
    float3 radiance = trace_path<C, S, M>(b, start, pixel, rng, rays, record, exposed, sky);
    if (!(record.first_distance > 0.0))
        return float4(apply_hdr_ceiling(pre_exposure * sky_radiance<E, M>(r.set, v.eye_render.xyz, start.dir)), 0.0);
    return float4(apply_hdr_ceiling(pre_exposure * radiance), near / (start.axis * record.first_t));
}

// The background dispatch's body for `pixel`: writes its texel of the background layer.
public void background_body<C : ICoherence, S : ISurface, E : IEnvironment, M : IMedium>(PathRoot r, uint2 pixel)
{
    r.dof_background.get<RWTexture2D<float4>>()[pixel] = background_texel<C, S, E, M>(r, pixel);
}

// The update path of `block`, a square of `PathRoot.update_block` pixels: one pixel of it,
// rotating through the block over frames, traces a path from its G-buffer surface and adds
// what its vertices scatter to the cache. On a traced tile it first walks the camera ray to
// that surface for the interiors around it.
public void update_body<C : ICoherence, S : ISurface, M : IMedium>(PathRoot r, uint2 block)
{
    uint n = r.update_block;
    uint offset = (r.frame + pcg(block.x | (block.y << 16))) % (n * n);
    uint2 pixel = block * n + uint2(offset % n, offset / n);
    uint2 extent = r.gbuffer.extent;
    if (any(pixel >= extent))
        return;
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ UPDATE_STREAM);
    PathStart start = pinhole_start(r.view, r.gbuffer, pixel, false);
    start.interior = gbuffer_interior<S>(r.scene, r.view, r.layers, extent, pixel, start);
    uint rays = 0;
    CacheUpdate update = CacheUpdate();
    float3 exposed;
    float3 sky;
    trace_path<C, S, M>(r, start, pixel, rng, rays, update, exposed, sky);
}

// Whether light leaves surface `g` as a caustic: smoother than CHAIN_ROUGHNESS here, of a
// section material that casts, and one the caustic photons of `r` start from, which a section
// the cook ranks as a caster holds once its copy's caster entry lists its cluster. A far chain
// step holds no entries; a producer entry's clusters are listed by their material. False
// without photons, whose light BSDF rays then carry.
public bool surface_chains(PathRoot r, GSurface g)
{
    SceneRoot* s = r.scene;
    if (r.caustics == nullptr || !(g.bsdf.specular_roughness < CHAIN_ROUGHNESS) || g.material == MATERIAL_NONE)
        return false;
    PhotonSourceGpu* source = r.caustics.source;
    uint entry;
    if (hit_producer(s, g.instance, entry))
    {
        uint m = traced_material(s, g.instance, g.cluster);
        return m != MATERIAL_NONE && (s.materials[m].flags & MATERIAL_CASTER) != 0 && source.entries[entry].count > 0;
    }
    if ((g.instance & COPY_FAR) != 0)
        return false;
    InstanceGpu inst = copy_record(s, g.instance);
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    if (mesh.casters.count == 0 || inst.first_emissive == NO_EMISSIVE)
        return false;
    uint id = mesh_cluster_id(mesh, g.cluster);
    ClusterRef cr = s.clusters[id];
    Cluster c = load_cluster(cr.page, cr.local);
    uint rank = mesh.casters.ranks[c.section];
    if (rank == NO_RANK || mesh.emissive.count + rank >= inst.emissive_count)
        return false;
    if ((s.materials[section_material(s, inst, model, c.section)].flags & MATERIAL_CASTER) == 0)
        return false;
    EmissiveSelection e = source.entries[inst.first_emissive + mesh.emissive.count + rank];
    uint p = list_position(e, id, g.cluster);
    return p < e.count && list_step(e, p) > 0.0;
}

// What a vertex's diffuse lobes give a reference photon along `wi`: its BSDF's, or a walk
// exit's.
struct PhotonReceiver : IPhotonReceiver
{
    Bsdf b;
    float3 n;                       // the shading normal
    bool exit;
    WalkExitScatter walk;

    float3 diffuse(float3 wi)
    {
        float c = abs(dot(n, wi));
        if (!(c > 0.0))
            return float3(0.0);
        if (exit)
            return walk.eval(wi) / c;
        float3 d;
        b.eval(wi, d);
        return d / c;
    }
}

// What vertex `l`, prepared as `b`, gathers of the caustic photons' light by its diffuse lobes,
// a walk's exit (`exit_scatter`) by its own; `width` m is its footprint. The grid's irradiance
// takes the lobes' albedo, a thin wall's diffuse transmission that of the side behind it; the
// reference's map evaluates them per photon. Draws eight sampler dimensions.
public float3 caustic_gather<S : ISurface>(PathRoot r, LayerVertex l, Bsdf b, bool exit, WalkExitScatter exit_scatter, float width, inout Sampler rng)
{
    CausticsGpu* g = r.caustics;
    float4 u = rng.next4();
    float4 u_back = rng.next4();
    GSurface x = l.x.surface;
    float3 n = dot(x.geometric_normal, l.x.view) < 0.0 ? -x.geometric_normal : x.geometric_normal;
    bool thin = !exit && thin_walled(x);
    if (g.mode == CAUSTICS_MAP)
    {
        PhotonReceiver receiver = { b, x.normal, exit, exit_scatter };
        return map_radiance(g, x.position, n, thin, receiver);
    }
    float3 front = exit ? exit_scatter.eval(exit_scatter.n) * PI : b.diffuse_reflection();
    float3 back = thin ? b.diffuse_transmission() : float3(0.0);
    return grid_gather(g, r.scene, x.position, n, front, back, width, u, u_back);
}

// Separates a photon's samples from every other stream.
public static const uint PHOTON_STREAM = 0xB5297A4D;

// Rows of the photon dispatch's sampler grid.
public static const uint PHOTON_ROW = 4096;

// How photon `index` starts: its caster entry, the first vertex's hit and corners, its flux
// and the interiors around it, the direction it arrives along and the light's feedback id.
public struct PhotonStart
{
    public uint entry;
    public Hit hit;
    public float3 positions[3];
    public float3 flux;
    public Interior interior;
    public float3 dir;
    public float3 point;
    public bool directional;        // the light is a directional light
}

// Photon `index` of a frame whose casters have weight: a caster entry by its weight, a cluster
// of its list by its fixed-point step, a triangle uniform by index and a point uniform in it;
// then one light seen from the point by `owned_light_candidate`, which takes every light but the
// environment, and the shadow ray toward it, which ends at every surface it commits: the
// point must be where the light first meets the scene, since a photon meets every caster
// before it by its own start. A presence layer's point takes its opacity of the flux, and an
// alpha test's hole none. The flux is the light's radiance x the point's cosine to it over
// both densities and the frame's photon count. False for a photon that starts with no flux.
public bool photon_start<S : ISurface, M : IMedium>(PathRoot r, uint index, inout Sampler rng, out PhotonStart p)
{
    p = (PhotonStart)0;
    SceneRoot* s = r.scene;
    CausticsGpu* g = r.caustics;
    PhotonSourceGpu* source = g.source;
    LightSetGpu* set = r.set;
    float pick = rng.next();
    uint2 bits = uint2(rng.next_bits(), rng.next_bits());
    float2 u = rng.next2();
    uint entry = upper_bound(source.cdf, source.entry_count, below(pick, source.total));
    p.entry = entry;
    count_started(source, g, entry);
    EmissiveSelection e = source.entries[entry];
    float chosen = source.cdf[entry] - (entry > 0 ? source.cdf[entry - 1] : 0.0);
    if (e.count == 0)
        return false;
    uint pos = upper_bound(e.cdf, e.count, mulhi(bits.x, 0xffffffffu));
    uint instance_index, cluster_id, triangles, alpha_first;
    if (!list_cluster(s, e, entry, pos, instance_index, cluster_id, triangles, alpha_first) || triangles == 0)
        return false;
    uint triangle = mulhi(bits.y, triangles);
    TracedTriangle t;
    if (!traced_triangle(s, instance_index, cluster_id, triangle, false, t))
        return false;
    float3 n;
    float area = traced_area(t, n);
    if (!(area > 0.0))
        return false;
    float su = sqrt(u.x);
    float2 bary = float2(u.y * su, su - u.y * su);
    float3 y = traced_point(t, bary);
    float density = chosen / source.total * list_step(e, pos) / float(triangles) / area;

    ResolvedLightSample c = owned_light_candidate<S>(set, y, rng);
    float u_hole = rng.next();
    if (!(c.pdf > 0.0))
        return false;
    float cos_y = abs(dot(n, c.dir));
    float3 l = c.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, c.light, c.dir) : c.radiance;
    float3 flux = l * (cos_y / (c.pdf * density * float(g.photon_count)));
    if (!any(flux > 0.0) || !all(isfinite(flux)))
        return false;
    p.interior = Interior(source.interiors[c.light]);
    p.directional = c.kind == LIGHT_SAMPLE_DIRECTIONAL;

    // The first vertex as a ray from the light resolves it.
    p.dir = -c.dir;
    p.point = y;
    Hit h = (Hit)0;
    h.hit = true;
    h.triangle = true;
    h.instance_index = instance_index;
    h.cluster_id = int(cluster_id);
    h.primitive_index = triangle;
    h.barycentrics = bary;
    h.front_face = dot(n, p.dir) < 0.0;
    uint entry_producer;
    if (hit_producer(s, instance_index, entry_producer))
        h.class_id = material_class(s.materials + t.material);
    else
    {
        ClusterRef cr = s.clusters[cluster_id];
        h.class_id = cluster_class(load_cluster(cr.page, cr.local));
    }
    p.hit = h;
    p.positions = t.corners.position;
    RayCone cone = { 0.0, 0.0 };
    HitContext context = hit_context(HIT_SECONDARY, 0, cone, 0.0);
    if (!resolve_alpha<S>(s, r.view, p.dir, instance_index, cluster_id, triangle, bary, context))
        return false;
    if ((s.materials[t.material].flags & MATERIAL_PRESENCE) != 0)
        flux *= saturate(resolve_coverage<S>(s, r.view, p.dir, instance_index, cluster_id, triangle, bary, context));

    float3 origin = ray_origin(y, y, n, c.dir, ORIGIN_DEFAULT, 0.0, length(y - r.view.eye_render.xyz));
    float t_max = c.distance;
    float3 toward = c.dir;
    if (isfinite(t_max))
    {
        float3 to = c.point - origin;
        float length_to = length(to);
        toward = length_to > 0.0 ? to / length_to : c.dir;
        float magnitude = max(abs(c.point.x), max(abs(c.point.y), abs(c.point.z)));
        t_max = max(0.0, length_to - max(1e-4 * length_to, magnitude * F32_PRECISION));
    }
    flux *= visibility<S, M, SHADOW_PHOTON>(s, r.view, origin, toward, t_max, RAY_FLAG_NONE, context, false, p.interior, set.shadow_overflows);
    if (c.view && any(flux > 0.0))
        flux *= emitter_view_layer<S>(set, c, interior_ior(s, p.interior.innermost()));
    p.flux = flux;
    return any(flux > 0.0);
}

// The caustic photon dispatch's body for photon `index`: its start, then the path loop from its
// first vertex, adjoint, depositing in `r.caustics`.
public void photon_body<C : ICoherence, S : ISurface, M : IMedium>(PathRoot r, uint index)
{
    // A frame with no casters starts nothing.
    if (index >= r.caustics.photon_count || !(r.caustics.source.total > 0.0))
        return;
    uint2 pixel = uint2(index % PHOTON_ROW, index / PHOTON_ROW);
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ PHOTON_STREAM);
    PhotonStart p;
    if (!photon_start<S, M>(r, index, rng, p))
        return;
    PathStart start = (PathStart)0;
    start.origin = p.point;
    start.dir = p.dir;
    start.mask = MASK_OPAQUE | MASK_DIELECTRIC;
    start.interior = p.interior;
    start.throughput = p.flux;
    start.given = true;
    start.hit = p.hit;
    start.positions = p.positions;
    start.directional = p.directional;
    uint rays = 0;
    CausticPhoton record = CausticPhoton(p.entry);
    float3 exposed;
    float3 sky;
    trace_path<C, S, M>(r, start, pixel, rng, rays, record, exposed, sky);
}
