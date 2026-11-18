// What a path of every source takes from its root (`path.slang`): its source and the root's
// predicates, its pixel, sampler and start, and a view path's light feedback.
module common;

import orr_shaders.packing;
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.hits;
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
import orr_lighting.guides;
import orr_lighting.path;
import orr_lighting.specialize;

// `PathRoot.source`: which paths a root runs.
[Host] public static const uint PATH_SOURCE_VIEW = 0;
[Host] public static const uint PATH_SOURCE_BACKGROUND = 1;
[Host] public static const uint PATH_SOURCE_UPDATE = 2;
[Host] public static const uint PATH_SOURCE_PHOTON = 3;

// `r.mode`, which is PATH_FULL in a pipeline that shows no debug view.
public uint path_mode(PathRoot r)
{
    return may(MAY_DEBUG) ? r.mode : PATH_FULL;
}

// Whether the view of `r` has a frosted-glass pass, a thin lens, froxels.
public bool frost_pass(PathRoot r) { return may(MAY_FROST) && (r.dielectric_flags & DIELECTRIC_FROST) != 0; }
public bool thin_lens(PathRoot r) { return may(MAY_LENS) && r.aperture_radius > 0.0; }
public bool has_froxels(PathRoot r) { return may(MAY_FROXELS) && r.froxels != nullptr; }

// Whether a pipeline made for the source of bit `bit` runs no other source's paths, so its root's
// source is known where it is made.
bool sole_source(uint bit)
{
    uint others = (MAY_VIEW | MAY_BACKGROUND | MAY_UPDATE | MAY_PHOTON) & ~bit;
    return PATH_SPECIALIZES && may(bit) && !may_any(others);
}

// Whether the paths of `r` are of `source`, which the pipeline may not run at all, or may run
// alone.
public bool from_view(PathRoot r) { return may(MAY_VIEW) && (sole_source(MAY_VIEW) || r.source == PATH_SOURCE_VIEW); }
public bool from_background(PathRoot r) { return may(MAY_BACKGROUND) && (sole_source(MAY_BACKGROUND) || r.source == PATH_SOURCE_BACKGROUND); }
public bool from_update(PathRoot r) { return may(MAY_UPDATE) && (sole_source(MAY_UPDATE) || r.source == PATH_SOURCE_UPDATE); }
public bool from_photon(PathRoot r) { return may(MAY_PHOTON) && (sole_source(MAY_PHOTON) || r.source == PATH_SOURCE_PHOTON); }

// The pixel of the path of item `index` of the root's source: a photon's place in its rows, an
// update block's pixel, or a view's pixel.
public uint2 item_pixel(PathRoot r, uint index)
{
    if (from_photon(r))
        return uint2(index % PHOTON_ROW, index / PHOTON_ROW);
    if (from_update(r))
        return update_pixel(r, index);
    uint2 extent = r.gbuffer.extent;
    return uint2(index % extent.x, index / extent.x);
}

// The pixel of the update path of the view's `index`th block: one of the block's, rotating
// through it over frames.
public uint2 update_pixel(PathRoot r, uint index)
{
    uint n = r.update_block;
    uint2 blocks = (r.gbuffer.extent + n - 1) / n;
    uint2 block = uint2(index % blocks.x, index / blocks.x);
    uint offset = (r.frame + pcg(block.x | (block.y << 16))) % (n * n);
    return block * n + uint2(offset % n, offset / n);
}

// The stream of the sampler a path of the root's source draws from.
public uint path_stream(PathRoot r)
{
    return r.seed ^ (from_background(r) ? DOF_STREAM : from_update(r) ? UPDATE_STREAM : from_photon(r) ? PHOTON_STREAM : 0u);
}

// The sampler a path of `pixel` of the root's source starts with.
public Sampler path_sampler(PathRoot r, uint2 pixel)
{
    return Sampler(pixel, r.frame, path_stream(r));
}

// Whether `pixel` lies on a tile the view's layer guides traced.
public bool pixel_traced(PathRoot r, uint2 pixel)
{
    uint2 extent = r.gbuffer.extent;
    return traced_tiles(r.layers, extent)[layer_tile(extent, pixel)] != 0;
}

// Whether a view's `pixel` also runs a path for the frosted-glass term, in a second pass.
public bool two_passes(PathRoot r, uint2 pixel)
{
    if (!frost_pass(r) || !pixel_traced(r, pixel))
        return false;
    return r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].frost != 0;
}

// The camera ray a path of `pixel` starts with, as its source makes it: a view's
// on a traced tile or through the lens, the background layer's, or a photon's, whose hit is
// given.
public PathStart camera_start(PathRoot r, uint2 pixel)
{
    if (from_photon(r))
    {
        PathStart given = PathStart();
        given.mask = MASK_OPAQUE | MASK_DIELECTRIC;
        given.given = true;
        return given;
    }
    if (from_background(r))
    {
        float mask = r.dof_mask.get<Texture2D<float>>().Load(int3(pixel, 0));
        float near = r.view.clip_from_view[2][3];
        Sampler rng = Sampler(pixel, r.frame, r.seed ^ DOF_STREAM);
        return lens_start(r, pixel, r.dof_aperture, *r.dof_focus, near / mask, MASK_CAMERA | MASK_DIELECTRIC, rng);
    }
    if (thin_lens(r))
    {
        Sampler rng = Sampler(pixel, r.frame, r.seed);
        return lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_CAMERA | MASK_DIELECTRIC, rng);
    }
    return pinhole_start(r.view, r.gbuffer, pixel, true);
}

// Whether the root's mode is a debug view of the first vertex alone, which a view's start writes.
public bool first_vertex_mode(PathRoot r)
{
    uint mode = path_mode(r);
    return mode == PATH_CACHE || mode == PATH_CACHE_LOD || mode == PATH_ORIGIN_BIAS || mode == PATH_CAUSTICS;
}

// The start of the background layer's path of `pixel` with its sampler `rng`, on a marked pixel:
// its lens ray past the mask's depth. Elsewhere false, with the pixel's texel (0, 0, 0, -1)
// written.
public bool background_path_start(PathRoot r, uint2 pixel, inout Sampler rng, out PathStart start)
{
    start = PathStart();
    float mask = r.dof_mask.get<Texture2D<float>>().Load(int3(pixel, 0));
    if (!(mask > 0.0))
    {
        r.dof_background.get<RWTexture2D<float4>>()[pixel] = float4(0.0, 0.0, 0.0, -1.0);
        return false;
    }
    float near = r.view.clip_from_view[2][3];
    start = lens_start(r, pixel, r.dof_aperture, *r.dof_focus, near / mask, MASK_CAMERA | MASK_DIELECTRIC, rng);
    return true;
}

// What a view path's record keeps of its light feedback.
public struct StoredViewLights
{
    public uint feedback;
    public float total;
    public uint state;
    public float first_distance;
}

// Whether light that added `contribution` of luminance takes part in a view's light feedback.
public bool view_light_counts(float contribution)
{
    return contribution > 0.0 && isfinite(contribution);
}

// A view's light feedback once light `id` added `contribution` of luminance, which
// `view_light_counts`: one of the lights, chosen in proportion to what each added.
public void view_light(inout StoredViewLights p, uint id, float contribution)
{
    p.total += contribution;
    p.state = pcg(p.state);
    if (unit_float(p.state) * p.total < contribution)
        p.feedback = id;
}

// ---------------------------------------------------------------------------------------------
// What a path's shaders derive from it

public bool has(uint flags, uint bit)
{
    return (flags & bit) != 0;
}

// Whether a path with `throughput` goes on: it carries light, and finite light.
public bool carries(float3 throughput)
{
    return any(throughput > 0.0) && all(isfinite(throughput));
}
