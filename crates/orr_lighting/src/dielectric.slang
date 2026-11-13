// Dielectric and presence layers before the view dispatch. The tiles the layers rastered into,
// and the tiles beside them, are the view's traced tiles, whose paths look for a first layer. Per pixel of the traced tiles, a camera ray up to
// the opaque depth finds the first layer, whose guide the guide resolve blends into ray
// reconstruction's guides, and, for the frosted-glass term, how far its lobes blur what lies
// behind it. The guide also names the interior at the pixel's near-plane point.
module dielectric;

import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.path;
import orr_lighting.shadow;
import orr_lighting.interiors;
import orr_lighting.guides;

// `LayerTilesRoot.flags`.
public static const uint LAYER_TILES_EVERYWHERE = 1 << 0; // the eye lies in a medium: trace every tile

// What picks the tiles the view's paths look for layers in.
public struct LayerTilesRoot
{
    public GBufferGpu* gbuffer;
    public LayerGuide* layers;          // the view's layer memory (`traced_tiles`)
    public uint flags;                  // LAYER_TILES_*
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// One thread per tile: a tile is traced when it or one beside it holds a layer, or everywhere
// when a layer may lie before the near plane, the eye lies in a medium or the G-buffer has no
// layer tiles. Traced tiles join the list.
public void layer_tiles_body(LayerTilesRoot r, uint2 tile)
{
    uint2 extent = r.gbuffer.extent;
    uint columns = layer_tile_columns(extent);
    uint count = layer_tile_count(extent);
    uint rows = count / columns;
    if (tile.x >= columns || tile.y >= rows)
        return;
    uint index = tile.y * columns + tile.x;
    uint* marks = r.gbuffer.layer_tiles;
    bool traced = marks == nullptr || (r.flags & LAYER_TILES_EVERYWHERE) != 0 || marks[count] != 0;
    for (int dy = -1; dy <= 1 && !traced; dy++)
    {
        for (int dx = -1; dx <= 1 && !traced; dx++)
        {
            int2 t = int2(tile) + int2(dx, dy);
            if (all(t >= 0) && t.x < int(columns) && t.y < int(rows))
                traced = marks[uint(t.y) * columns + uint(t.x)] != 0;
        }
    }
    traced_tiles(r.layers, extent)[index] = traced ? 1 : 0;
    if (traced)
    {
        uint at;
        InterlockedAdd(layer_tile_args(r.layers, extent)[0], 1, at);
        layer_tile_list(r.layers, extent)[at] = index;
    }
}

// Sizes the layer guides' dispatch over the listed tiles: two 8 x 4 workgroups per tile.
public void layer_args_body(LayerTilesRoot r)
{
    uint* args = layer_tile_args(r.layers, r.gbuffer.extent);
    write_grid(args + 1, 2 * args[0]);
}

// The guides of a pixel whose first layer `l` (prepared as `b`) was hit by the camera ray
// along `dir` with `interior` the eye's list. Presence layers give their own guides at their
// opacity `coverage`. A dielectric gives its `layer_chain`; the dominant one of its reflection
// and its chain gives depth and motion: the hit's own where reflection carries more, the
// virtual point of the chain otherwise. Diffuse albedo holds emission. Specular motion follows
// the mirror reflection's virtual point. Where transmission dominates, `frost` gets the blur
// the chain's rough interfaces give what it reaches, the widths adding in quadrature.
// `front` is the layer's overlay front word where the dominant layer lies behind it, else 0.
public LayerGuide layer_guides<S : ISurfaces>(PathRoot r, LayerVertex l, Bsdf b, Hit h, float3 dir, Interior interior, float coverage, out uint64_t front)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    float exposure = v.exposure[v.exposure_slot].pre_exposure;
    LayerGuide g = (LayerGuide)0;
    float3 lean = l.front ? l.x.surface.normal : -l.x.surface.normal;
    float3 previous;
    if (!hit_previous_position(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, previous))
        previous = l.x.surface.position;
    float own_depth = device_depth(v, l.x.surface.position);
    float2 own_motion = moved_motion(v, l.x.surface.position - v.eye_render.xyz, previous - v.eye_render.xyz, 1.0);
    RayCone cone = camera_cone(v);
    cone.width = cone_width(cone, l.x.camera_distance);

    // The mirror reflection's virtual point.
    float3 reflected = reflect(dir, lean);
    LayerAccept<S> ra = { s, v, reflected, hit_context(HIT_SECONDARY, 1, cone, 0.0), 0, PRESENCE_ANY };
    Hit rh = trace_closest(s.tlas, path_ray(vertex_origin(l.x, reflected), reflected, positive_infinity()), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, ra);
    float3 camera = l.x.surface.position - v.eye_render.xyz;
    g.specular_motion = rh.hit ? reprojected_motion(v, dir * (length(camera) + rh.t), 1.0) : reprojected_motion(v, dir, 0.0);

    if (h.class_id == CLASS_PRESENCE)
    {
        Guides own = surface_guides(l.x.surface, b, exposure);
        g.diffuse = own.diffuse + own.emission;
        g.specular = own.specular;
        g.normal = lean;
        g.roughness = own.roughness;
        g.weight = coverage;
        g.dominant = coverage >= 0.5 ? 1 : 0;
        g.depth = own_depth;
        g.motion = own_motion;
        front = g.dominant == 0 ? overlay_front_word(own_depth, own_motion, coverage) : 0;
        return g;
    }

    LayerChain c = layer_chain<S>(s, v, l, b, dir, interior, r.dielectric_depth, cone, exposure);
    float wr = c.reflected;
    float wt = c.transmitted;
    g.diffuse = c.guides.diffuse + c.guides.emission;
    g.specular = c.guides.specular;
    g.normal = c.guides.normal;
    g.roughness = c.guides.roughness;
    g.weight = 1.0;
    g.dominant = 1;
    front = wr < wt ? overlay_front_word(own_depth, own_motion, wr / (wr + wt)) : 0;
    float traveled = c.traveled;
    if (wr >= wt)
    {
        g.depth = own_depth;
        g.motion = own_motion;
    }
    else if (isfinite(traveled))
    {
        float3 q = dir * (length(camera) + traveled);
        g.depth = device_depth(v, q + v.eye_render.xyz);
        g.motion = reprojected_motion(v, q, 1.0);
    }
    else
    {
        g.depth = 0.0;
        g.motion = reprojected_motion(v, dir, 0.0);
    }
    float3 blur = c.blur;
    if ((r.dielectric_flags & DIELECTRIC_FROST) != 0 && wt > wr && blur.x > 0.0)
    {
        // The blur's radius over the width of a pixel where the chain ends; the angle alone
        // toward the sky.
        float radius = isfinite(traveled)
            ? sqrt(max(traveled * traveled * blur.x - 2.0 * traveled * blur.y + blur.z, 0.0)) / (cone.spread * (length(camera) + traveled))
            : sqrt(blur.x) / cone.spread;
        g.frost = f32tof16(min(radius, FROST_MAX_RADIUS)) | (f32tof16(guide_luminance(c.through)) << 16);
    }
    return g;
}

// The layer guide dispatch's body for `pixel`, in a traced tile: `camera_medium`, flipped at each
// surface between the eye and the near-plane point where a layer may lie there, so a near plane
// straddling a surface works; the camera ray up to the opaque depth; and the guide of its first
// layer. Opaque hits on the way name no layer.
public void layer_guide_body<S : ISurfaces>(PathRoot r, uint2 pixel)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    uint2 extent = r.gbuffer.extent;
    uint index = pixel.y * extent.x + pixel.x;
    PathStart start = pinhole_start(v, r.gbuffer, pixel, true);
    Interior interior = Interior();
    if (r.camera_medium != MATERIAL_NONE)
        interior.cross(r.camera_medium, true);
    uint* marks = r.gbuffer.layer_tiles;
    if (marks == nullptr || marks[layer_tile_count(extent)] != 0 || r.camera_medium != MATERIAL_NONE)
        surface_crossings<S>(s, v, v.eye_render.xyz, start.dir, start.offset, interior);
    RayCone cone = camera_cone(v);
    float3 dir = start.dir;
    HitContext primary = hit_context(HIT_PRIMARY, 0, cone, 0.0);
    LayerAccept<S> a = { s, v, dir, primary, 0, PRESENCE_ANY };
    float3 positions[3];
    // Micromaps forced to two states, as the path's camera ray reads them, so alpha-tested
    // geometry hides a layer from both or neither.
    Hit h = trace_closest(s.tlas, path_ray(start.origin, dir, start.t_max), start.mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    LayerGuide guide = (LayerGuide)0;
    if (is_layer(h) && h.cluster_id >= 0)
    {
        SurfaceHit first = resolve_hit<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, hit_context(HIT_PRIMARY, 0, cone, h.t));
        if (first.valid)
        {
            LayerVertex l = layer_vertex(v, first, dir);
            Bsdf b = prepare_vertex(s, l, interior, false, true, false, 0.0);
            float coverage = h.class_id == CLASS_PRESENCE ? resolve_coverage<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, primary) : 1.0;
            uint64_t front;
            guide = layer_guides<S>(r, l, b, h, dir, interior, coverage, front);
            if (r.overlay_front != nullptr)
                r.overlay_front[index] = front;
        }
    }
    guide.near_interior = EyeInterior(interior);
    r.layers[index] = guide;
}
