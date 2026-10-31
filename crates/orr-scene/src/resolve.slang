// The material resolve. `classify` runs one workgroup per 8 x 8 tile: it writes sky pixels
// and the HZB's finest mip, and lists the tile in every surface bin its pixels use.
// `resolve` then runs once per bin over the bin's tiles with the bin's surface id uniform:
// each pixel's surface through the linked `Surface` switch, the depth offset applied to
// depth, position and motion, projected-space NDF filtering of the roughness, and the
// G-buffer written.
#include "orr/root.h"
import orr.math;
import orr_pack.mesh;
import producer;
import surface;
import visibility;

extern struct Surface : ISurface = StandardSurface;

ORR_ROOT(RasterRoot);

groupshared uint bins_used[8];
groupshared float3 normals[64];

// Previous uv minus current uv of a direction at infinity.
float2 sky_motion(ViewConstants* v, float3 d)
{
    float4 cur = mul(v.clip_from_view, float4(view_from_relative(v, d), 0.0));
    float4 prev = mul(v.prev_clip_from_view, float4(mul(v.prev_view_from_world_rotation, float4(d, 0.0)), 0.0));
    return (prev.xy / prev.w - cur.xy / cur.w) * 0.5;
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

// Projected-space NDF filtering (Tokuyoshi and Kaplanyan 2021): widens GGX alpha^2 by the
// screen-space variance of the shading normal.
float filtered_roughness(float roughness, float variance)
{
    static const float KAPPA = 0.18;
    float a2 = roughness * roughness * roughness * roughness;
    return sqrt(sqrt(saturate(a2 + min(2.0 * variance, KAPPA))));
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
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    uint64_t w = all(pixel < r.gbuffer.extent) ? r.gbuffer.visibility[index] : 0;
    VisibleCluster vc = r.gbuffer.clusters[w != 0 ? visibility_cluster(w) : 0];
    bool active = w != 0 && surface_bin(r, s.materials[vc.material].surface) == bin;

    GSurface g = (GSurface)0;
    PrimaryHit hit = (PrimaryHit)0;
    float offset = 0.0;
    if (active)
    {
        hit = primary_hit(s, v, vc, visibility_triangle(w), float2(pixel) + 0.5);
        g = Surface.surface(hit.input, offset);
        offset = max(offset, 0.0);
    }

    // Quad neighbors exchange normals; a pixel without a neighbor in its bin takes no
    // variance from that side.
    float3 n = active ? g.normal : float3(0.0);
    normals[t] = n;
    GroupMemoryBarrierWithGroupSync();
    float3 nx = normals[t ^ 1];
    float3 ny = normals[t ^ 2];
    if (!active)
        return;
    float3 dndx = dot(nx, nx) > 0.0 ? nx - n : float3(0.0);
    float3 dndy = dot(ny, ny) > 0.0 ? ny - n : float3(0.0);
    float variance = INV_TWO_PI * (dot(dndx, dndx) + dot(dndy, dndy));
    g.bsdf.specular_roughness = filtered_roughness(g.bsdf.specular_roughness, variance);
    g.bsdf.coat_roughness = filtered_roughness(g.bsdf.coat_roughness, variance);

    MaterialGpu* m = s.materials + vc.material;
    float3 relative = hit.relative + hit.direction * offset;
    g.position = relative + v.eye_render.xyz;
    g.raster_error = hit.error;
    g.flags = (g.flags & ~GBUFFER_ORIGIN_MASK) | (hit_origin(m, hit.origin) << GBUFFER_ORIGIN_SHIFT);
    g.instance = vc.instance;
    g.cluster = vc.cluster;
    g.primitive = visibility_triangle(w);
    g.material = vc.material;
    ExposureState* exposure = v.exposure + v.exposure_slot;
    g.bsdf.emission = apply_hdr_ceiling(g.bsdf.emission * exposure.pre_exposure);

    // Depth and motion of the offset point; its previous position from the previous
    // transform of the same mesh-space point.
    float3 view_pos = view_from_relative(v, relative);
    float4 clip = mul(v.clip_from_view_jittered, float4(view_pos, 1.0));
    float4 current = mul(v.clip_from_view, float4(view_pos, 1.0));
    // A producer batch with previous positions moves the point by its vertices' motion too.
    Placement p = hit.placement;
    float3 object = mul(linear_inverse(p.linear), relative - p.translation);
    Placement prev;
    if (!visible_placement(s, v, vc, true, prev))
        prev = p;
    uint entry;
    if (visible_producer(s, vc, entry) && s.batches[entry].previous != nullptr)
    {
        ProducerBatch pb = s.batches[entry];
        Corners c = producer_triangle(pb, pb.clusters[vc.cluster], visibility_triangle(w), true).corners;
        float2 bc = hit.input.barycentrics;
        object += (1.0 - bc.x - bc.y) * c.position[0] + bc.x * c.position[1] + bc.y * c.position[2] - hit.object;
    }
    float3 prev_relative = place(prev, object) + v.eye_motion.xyz;
    float4 previous = mul(v.prev_clip_from_view, float4(mul(v.prev_view_from_world_rotation, float4(prev_relative, 0.0)), 1.0));
    float2 motion = (previous.xy / previous.w - current.xy / current.w) * 0.5;

    bool extended = r.x.index != TEXTURE_NONE;
    GBufferWords words = encode_gsurface(g, extended);
    r.a.get<RWTexture2D<uint4>>()[pixel] = words.a;
    r.b.get<RWTexture2D<uint2>>()[pixel] = words.b;
    if (extended && ((words.a.w >> 24) & GBUFFER_EXTENDED) != 0)
        r.x.get<RWTexture2D<uint4>>()[pixel] = words.x;
    r.motion.get<RWTexture2D<float2>>()[pixel] = motion;
    r.depth[index] = clip.z / clip.w;
}
