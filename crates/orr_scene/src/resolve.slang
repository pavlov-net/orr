// The material resolve. `classify` runs one workgroup per 8 x 8 tile: it writes sky pixels
// and the HZB's finest mip, and lists the tile in every surface bin its pixels use. With
// device-generated commands `sequences` then writes one sequence per bin with tiles.
// `resolve` runs once per bin over the bin's tiles with the bin's surface id uniform: each
// pixel's surface through the linked `Surface` switch, the depth offset applied to depth,
// position and motion, and the G-buffer written.
#include "orr_root.h"
import orr_shaders.abi;
import orr_pack.mesh;
import orr_scene.producer;
import orr_gpu.sequence;
import orr_scene.surface;
import orr_scene.visibility;

// Its taps drive texture streaming and decode the neural sets that stand in for textures.
export static const bool TEXTURE_FEEDBACK = true;
export static const bool NEURAL_TEXTURES = true;

extern struct Surface : ISurface = StandardSurface;

ORR_ROOT(RasterRoot);

groupshared uint bins_used[8];

// Previous uv minus current uv of a direction at infinity.
float2 sky_motion(ViewConstants* v, float3 d)
{
    float4 cur = mul(v.clip_from_view, float4(view_from_relative(v, d), 0.0));
    float4 prev = mul(v.prev_clip_from_view, float4(mul(v.prev_view_from_world_rotation, float4(d, 0.0)), 0.0));
    return motion_between(prev, cur);
}

uint surface_bin(RasterGpu* r, uint surface)
{
    return surface < r.bins ? surface : 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classify(uint group: SV_GroupID, uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    if (t < 8)
        bins_used[t] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint2 pixel = tile_pixel(group, r.tiles_x, t);
    if (all(pixel < r.gbuffer.extent))
    {
        uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
        uint64_t w = r.gbuffer.visibility[index];
        r.hzb[0].get<RWTexture2D<float>>()[pixel] = visibility_depth(w);
        if (w == 0)
        {
            float2 uv = (float2(pixel) + 0.5) / float2(r.gbuffer.extent);
            float3 d = render_direction(v, view_direction(v, uv));
            r.a.get<RWTexture2D<uint4>>()[pixel] = uint4(0);
            r.b.get<RWTexture2D<uint2>>()[pixel] = uint2(0);
            r.motion.get<RWTexture2D<float2>>()[pixel] = sky_motion(v, d);
            r.depth[index] = 0.0;
        }
        else
        {
            MaterialGpu* m = s.materials + r.gbuffer.clusters[visibility_cluster(w)].material;
            uint bin = surface_bin(r, m.surface);
            InterlockedOr(bins_used[bin >> 5], 1u << (bin & 31));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint bin = t; bin < r.bins; bin += 64)
    {
        if ((bins_used[bin >> 5] & (1u << (bin & 31))) != 0)
        {
            uint slot;
            InterlockedAdd(r.bin_args[3 * bin], 1, slot);
            r.bin_tiles[bin * r.tile_capacity + slot] = group;
        }
    }
}

// One thread per bin: a sequence running the resolve set's one pipeline over the bin's
// tiles with the bin's root, for bins with tiles only.
[shader("compute")]
[numthreads(64, 1, 1)]
void sequences(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    uint bin = id.x;
    if (bin >= r.bins)
        return;
    uint tiles = r.bin_args[3 * bin];
    if (tiles == 0)
        return;
    uint slot;
    InterlockedAdd(r.bin_args[3 * r.bins], 1, slot);
    Sequence q;
    q.pipeline = 0;
    q.root = r.bin_roots + uint64_t(bin) * BIN_ROOT_STRIDE;
    q.groups = uint3(tiles, 1, 1);
    store(r.sequences, slot, q);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void resolve(uint group: SV_GroupID, uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    uint bin = root.value;
    uint2 pixel = tile_pixel(r.bin_tiles[bin * r.tile_capacity + group], r.tiles_x, t);
    if (any(pixel >= r.gbuffer.extent))
        return;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    uint64_t w = r.gbuffer.visibility[index];
    if (w == 0)
        return;
    VisibleCluster vc = r.gbuffer.clusters[visibility_cluster(w)];
    MaterialGpu* m = s.materials + vc.material;
    if (surface_bin(r, m.surface) != bin)
        return;

    PrimaryHit hit = primary_hit(s, v, vc, visibility_triangle(w), float2(pixel) + 0.5);
    material_prepare(m, hit.input.uv);
    float offset = 0.0;
    GSurface g = Surface.surface(hit.input, offset);
    offset = max(offset, 0.0);

    float3 relative = hit.relative + hit.direction * offset;
    g.position = relative + v.eye_render.xyz;
    g.raster_error = hit.error;
    g.flags = (g.flags & ~GBUFFER_ORIGIN_MASK) | (hit_origin(m, hit.origin) << GBUFFER_ORIGIN_SHIFT);
    g.instance = vc.instance;
    g.cluster = hit.input.cluster;
    g.primitive = hit.input.primitive;
    g.material = vc.material;
    ExposureState* exposure = v.exposure + v.exposure_slot;
    g.bsdf.emission = apply_hdr_ceiling(g.bsdf.emission * exposure.pre_exposure);

    // Depth and motion of the offset point.
    float3 view_pos = view_from_relative(v, relative);
    float4 clip = mul(v.clip_from_view_jittered, float4(view_pos, 1.0));
    float2 motion = visible_motion(s, v, vc, hit, visibility_triangle(w), relative);

    bool extended = r.x.index != IMAGE_NONE;
    GBufferWords words = encode_gsurface(g, extended);
    r.a.get<RWTexture2D<uint4>>()[pixel] = words.a;
    r.b.get<RWTexture2D<uint2>>()[pixel] = words.b;
    if (extended && ((words.a.w >> 24) & GBUFFER_EXTENDED) != 0)
        r.x.get<RWTexture2D<uint4>>()[pixel] = words.x;
    r.motion.get<RWTexture2D<float2>>()[pixel] = motion;
    r.depth[index] = clip.z / clip.w;
}
