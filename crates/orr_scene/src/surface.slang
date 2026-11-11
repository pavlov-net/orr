// The surface link point. A consumer surface is a type implementing `ISurface`, linked at
// build time into every kernel that shades hits: the material resolve for primary hits and
// ray hit shading for every other. `StandardSurface` shades from `MaterialGpu` and serves
// surface id 0 and every id nothing links.
module surface;

import orr_shaders.abi;
import orr_color.primaries;
import orr_shaders.math;
import orr_pack.normal_map;
__exported import orr_scene.gbuffer;
__exported import orr_scene.material;
__exported import orr_scene.tables;
__exported import orr_scene.view;
import orr_scene.scene_may;

// `HitContext.kind`.
public static const uint HIT_PRIMARY = 0;   // the material resolve and the dielectric camera ray's first hit
public static const uint HIT_SECONDARY = 1; // every other ray hit

// What a hit's texture footprint is scaled by: the view's mip bias at primary hits.
public float footprint_bias(uint kind, ViewConstants* v)
{
    return kind == HIT_PRIMARY && v != nullptr ? exp2(v.mip_bias) : 1.0;
}

public struct HitContext
{
    public uint kind;          // HIT_*
    public uint depth;         // path depth, 0 at primary hits
    public float cone_width;   // m: the ray cone's width at the hit
    public float cone_spread;  // rad: the ray cone's spread angle
    public float2 tap;         // the hit's `UvFootprint.tap`: negative x for hardware filtering
    public bool feedback;      // the hit reports the texture levels it reads (`FeedbackSurface`, `CoverageFeedback`)
}

// Everything a surface knows about its hit. Directions and normals are unit vectors in
// render space unless named object.
public struct SurfaceInput
{
    public HitContext context;
    public uint instance;           // instance slot or producer entry
    public uint cluster;            // mesh cluster index; NO_CLUSTER on a drawn triangle
    public uint part;
    public uint primitive;          // triangle in the cluster
    public float2 barycentrics;     // weights of corners 1 and 2
    public float4 vertex_color;     // the mesh's color stream here; 1 without one
    public uint user_tag;           // the copy's or entry's
    public uint cluster_user;       // producer cluster word; 0 on cooked clusters
    public uint3 vertex_user;       // producer vertex words per corner; 0 on cooked clusters
    public uint material;
    public uint surface;            // linked surface id; 0 = StandardSurface
    public UvFootprint uv[2];       // per uv set; producer and patch geometry give both its one uv, but for a subdivision copy holding uv set 1
    public float3 geometric_normal; // the triangle's, turned toward the ray on a back face but for a dielectric's
    public bool back_face;          // the ray met the triangle's back
    public bool swapped;            // `material` is the section material's back
    public float3 normal;           // interpolated vertex normal, unit
    public float3 frame_normal;     // `normal` before normalizing, not always unit
    public float4 tangent;          // xyz as interpolated, not always unit; w bitangent sign
    public float3 coat_frame_normal; // `frame_normal` of the frame the coat follows
    public float4 coat_tangent;     // `tangent` of the frame the coat follows
    public float3 object_position;
    public float3 object_normal;
    public float3x3 object_to_render; // the instance's linear transform
    public float3 position;         // render space
    public int3 cell;               // world position as cell and offset in the cell
    public float3 offset;
    public float3 view_direction;   // from the eye toward the hit
    public SceneRoot* scene;
    public ViewConstants* view;
    public uint64_t data;           // the material's consumer data, else the scene's
}

public interface ISurface
{
    // The shaded surface and a depth offset in m (non-negative, along the view ray) that
    // moves the visible point behind the triangle. Position, the visible triangle and the
    // raster error are the caller's; the rest is the surface's.
    static GSurface surface(SurfaceInput i, out float depth_offset);

    // How much of the triangle at the hit is present, 0 to 1. Without its own, a surface
    // takes the material's alpha test.
    static float coverage(SurfaceInput i)
    {
        return material_coverage(i);
    }

    // Reports to the frame's texture feedback the level each streamed texture `surface` reads
    // at the hit wants (`material_feedback`); `FeedbackSurface` calls it for hits whose context
    // asks. Without its own, a surface reports what `StandardSurface` reads.
    static void feedback(SurfaceInput i)
    {
        surface_feedback(i);
    }

    // Reports to the frame's texture feedback the level each streamed texture `coverage` reads
    // at the hit wants; `CoverageFeedback` calls it for hits whose context asks. Without its
    // own, a surface reports what the material's alpha test reads
    // (`material_opacity_feedback`).
    static void coverage_feedback(SurfaceInput i)
    {
        material_opacity_feedback(i);
    }

    // How much of a presence layer is present at the hit, 0 to 1: its material's opacity, which
    // no linked surface changes, so the switch over linked surfaces forwards none. It is a
    // method so that `CoverageFeedback` can report its read.
    static float presence(SurfaceInput i)
    {
        return material_opacity(i);
    }
}

// Footprint `f` with no extent, read bilinearly from mip 0.
UvFootprint finest_footprint(UvFootprint f)
{
    f.dx = float2(0.0);
    f.dy = float2(0.0);
    f.tap = float2(-1.0);
    return f;
}

// The slot `material_opacity`, `blend_decal` and `material_opacity_feedback` read opacity
// from: the opacity slot where it is bound, else the base color slot, whose alpha they read.
uint opacity_slot(MaterialGpu* m)
{
    return material_bound(m, TEXTURE_OPACITY) ? TEXTURE_OPACITY : TEXTURE_BASE_COLOR;
}

// The material's opacity at the hit: the opacity slot, else the base color slot's alpha,
// times `geometry_opacity`, read bilinearly from mip 0 whatever the footprint; 1 for a
// material that is neither alpha-tested nor a presence layer, whose alpha is not read.
public float material_opacity(SurfaceInput i)
{
    MaterialGpu* m = i.scene.materials + i.material;
    if ((m.flags & (MATERIAL_ALPHA_TESTED | MATERIAL_PRESENCE)) == 0)
        return 1.0;
    UvFootprint finest[2] = { finest_footprint(i.uv[0]), finest_footprint(i.uv[1]) };
    float opacity = m.geometry_opacity * surface_vertex(i, m, TEXTURE_OPACITY).x;
    // One read of whichever slot holds the opacity.
    uint slot = opacity_slot(m);
    if (material_bound(m, slot))
    {
        float4 texel = material_texture(m, slot, finest, float4(1.0));
        opacity *= slot == TEXTURE_OPACITY ? texel.x : texel.w;
    }
    return opacity;
}

// What the vertex color stream multiplies `slot` by at the hit: from its first channel on in
// xyz, or the complement of that for a slot that reads one, and 1 for a slot that reads none.
public float3 surface_vertex(SurfaceInput i, MaterialGpu* m, uint slot)
{
    if (!may2(MAY2_VERTEX_COLORS) || (m.flags & MATERIAL_VERTEX) == 0)
        return float3(1.0);
    uint modes = m.samplers[slot].modes;
    uint first = (modes >> TEXTURE_VERTEX_SHIFT) & TEXTURE_VERTEX_MASK;
    float4 c = i.vertex_color;
    float3 v = first == 0 ? float3(1.0) : first == 1 ? c.xyz : first == 2 ? c.yzw : first == 3 ? float3(c.zw, 1.0) : float3(c.w, 1.0, 1.0);
    uint complement = (modes >> TEXTURE_VERTEX_COMPLEMENT_SHIFT) & TEXTURE_VERTEX_COMPLEMENT_MASK;
    return complement != 0 ? saturate(1.0 - exp2(float(complement - 1)) * v) : v;
}

// Scalar slot `slot` at the hit, times the vertex color stream's value for it.
float surface_scale(SurfaceInput i, MaterialGpu* m, uint slot)
{
    return material_scale(m, slot, i.uv) * surface_vertex(i, m, slot).x;
}

// `material_color` of `slot` at the hit, its factor multiplied by the vertex color stream, which
// is linear Rec.709.
float3 surface_color(SurfaceInput i, MaterialGpu* m, uint slot, float3 factor)
{
    return material_color(m, slot, i.uv, surface_color_factor(i, m, slot, factor));
}

// `factor` multiplied by the vertex color stream's value for color slot `slot`: the factor
// `surface_color` tints by the slot's texel.
float3 surface_color_factor(SurfaceInput i, MaterialGpu* m, uint slot, float3 factor)
{
    if (may2(MAY2_VERTEX_COLORS) && (m.flags & MATERIAL_VERTEX) != 0 && ((m.samplers[slot].modes >> TEXTURE_VERTEX_SHIFT) & TEXTURE_VERTEX_MASK) != 0)
        return mul(REC709_TO_REC2020, mul(REC2020_TO_REC709, factor) * surface_vertex(i, m, slot));
    return factor;
}

// `material_more_color` of `slot`, with `surface_color`'s vertex color.
float3 surface_more_color(SurfaceInput i, MaterialGpu* m, uint slot, float3 factor)
{
    if ((m.flags & (MATERIAL_MORE_TEXTURES | (may2(MAY2_VERTEX_COLORS) ? MATERIAL_VERTEX : 0u))) == 0)
        return factor;
    return surface_color(i, m, slot, factor);
}

// What the surface emits at the hit, in nits: `material_emission_at` with the vertex color
// stream on both of its slots. Past a primary hit it reads with no footprint, as light sampling
// reads an emitter's point, so a ray that meets a light sees what a sample drawn there carries.
public float3 surface_emission(SurfaceInput i, MaterialGpu* m)
{
    if (i.context.kind != HIT_PRIMARY)
    {
        i.uv[0] = finest_footprint(i.uv[0]);
        i.uv[1] = finest_footprint(i.uv[1]);
    }
    return surface_scale(i, m, TEXTURE_EMISSION_LUMINANCE) * surface_color(i, m, TEXTURE_EMISSION, material_emission(m));
}

// The material's alpha test at the hit: 1 where `material_opacity` is at or above the
// cutoff, else 0: the test the cook's opaque triangles and opacity micromaps follow, so the
// raster, ray candidates and micromaps agree.
public float material_coverage(SurfaceInput i)
{
    return material_opacity(i) >= i.scene.materials[i.material].alpha_cutoff ? 1.0 : 0.0;
}

// Reports to the frame's texture feedback what `material_opacity` reads at the hit: the slot
// `opacity_slot` picks, at mip 0.
public void material_opacity_feedback(SurfaceInput i)
{
    MaterialGpu* m = i.scene.materials + i.material;
    if ((m.flags & (MATERIAL_ALPHA_TESTED | MATERIAL_PRESENCE)) == 0)
        return;
    material_feedback(m, opacity_slot(m), i.uv, true);
}

// The shading frame: the vertex normal, turned by the material's normal map in `slot`.
// MikkTSpace's bakers invert this transform: the interpolated normal and tangent as they stand,
// neither unit nor perpendicular. A map of mesh space directions takes the material's
// `object_frame`, then the placement's normal transform, and the side `i.normal` faces where a
// back face turned it.
public float3 material_normal(SurfaceInput i, MaterialGpu* m, uint slot = TEXTURE_NORMAL)
{
    if (!material_bound(m, slot))
        return i.normal;
    return material_texel_normal(i, m, slot, material_texture(m, slot, i.uv, float4(0.0, 0.0, 1.0, 1.0)));
}

// `material_normal` of bound normal slot `slot` from its texel `texel`.
public float3 material_texel_normal(SurfaceInput i, MaterialGpu* m, uint slot, float4 texel)
{
    if ((m.flags & (slot == TEXTURE_NORMAL ? MATERIAL_OBJECT_NORMAL : MATERIAL_OBJECT_COAT_NORMAL)) != 0)
    {
        float3x3 c = normal_matrix(i.object_to_render);
        float side = dot(mul(c, i.object_normal), i.normal) < 0.0 ? -1.0 : 1.0;
        float3 v = object_normal_decode(texel.xyz);
        float3x3 frame = float3x3(m.object_frame[0], m.object_frame[1], m.object_frame[2], m.object_frame[3], m.object_frame[4], m.object_frame[5], m.object_frame[6], m.object_frame[7], m.object_frame[8]);
        float3 baked = v + mul(frame, v);
        return safe_normalize(mul(c, baked) * side, i.normal);
    }
    float3 v = material_normal_frame(m, slot, normal_decode(texel.xy));
    bool coat = slot == TEXTURE_COAT_NORMAL;
    float3 n = coat ? i.coat_frame_normal : i.frame_normal;
    float4 tangent = coat ? i.coat_tangent : i.tangent;
    float3 t = tangent.xyz;
    float3 b = cross(n, t) * tangent.w;
    return safe_normalize(t * v.x + b * v.y + n * v.z, i.normal);
}

// What the decals over a hit write: per parameter the sum of each decal's value times its
// share, blended in id order (higher ids over lower), and how much of the surface's own
// value they cover, 0 to 1.
public struct DecalLayer
{
    public float3 base_color;
    public float base_color_cover;
    public float3 normal;          // render space
    public float normal_cover;
    public float3 emission;        // nits, unexposed
    public float emission_cover;
    public float roughness;
    public float roughness_cover;
    public float metalness;
    public float metalness_cover;
}

void decal_over(inout float3 sum, inout float cover, float3 value, float share)
{
    sum = sum * (1.0 - share) + value * share;
    cover += share - cover * share;
}

void decal_over(inout float sum, inout float cover, float value, float share)
{
    sum = sum * (1.0 - share) + value * share;
    cover += share - cover * share;
}

// Where decal `d` lies over hit `i`, `relative` from the decal's center, with the hit's
// footprint `px`, `py` (m, on the surface): its share before its material's opacity, and the
// footprint of the one uv it projects, which every slot reads. False where it does not cover
// the hit.
bool decal_place(SurfaceInput i, DecalGpu* d, float3 relative, float3 px, float3 py, out float share, out UvFootprint f)
{
    share = 0.0;
    f = (UvFootprint)0;
    float3 axis_x = d.axis_x;
    float3 axis_y = d.axis_y;
    float3 local = float3(dot(relative, axis_x) * d.inv_half_x, dot(relative, axis_y) * d.inv_half_y, dot(relative, d.axis_z) * d.inv_half_z);
    if (any(abs(local) > 1.0))
        return false;
    float facing = dot(i.geometric_normal, d.axis_z);
    share = d.opacity * saturate((facing - d.cos_none) / max(d.cos_full - d.cos_none, 1e-6));
    if (d.depth_fade > 0.0)
        share *= saturate((1.0 - abs(local.z)) / d.depth_fade);
    if (share <= 0.0)
        return false;
    float2 scale = float2(0.5 * d.inv_half_x, -0.5 * d.inv_half_y);
    f.uv = float2(0.5, 0.5) + local.xy * float2(0.5, -0.5);
    f.dx = float2(dot(px, axis_x), dot(px, axis_y)) * scale;
    f.dy = float2(dot(py, axis_x), dot(py, axis_y)) * scale;
    f.tap = i.uv[0].tap;
    f.inward = float2(0.0);
    return true;
}

// Decal `d` blended over `layer` at hit `i`, `relative` from the decal's center, with the
// hit's footprint `px`, `py` (m, on the surface).
void blend_decal(inout DecalLayer layer, SurfaceInput i, DecalGpu* d, float3 relative, float3 px, float3 py)
{
    float share;
    UvFootprint f;
    if (!decal_place(i, d, relative, px, py, share, f))
        return;
    MaterialGpu* m = i.scene.materials + d.material;
    share *= m.geometry_opacity;
    uint c = d.channels;
    bool emits = (c & DECAL_EMISSION) != 0;
    // Past a primary hit a decal's light reads with no footprint, as `surface_emission` does.
    bool primary = i.context.kind == HIT_PRIMARY;
    UvFootprint e = primary ? f : finest_footprint(f);
    uint cover = opacity_slot(m);
    float coverage = 1.0;
    float lit_coverage = 1.0;
    float luminance = 1.0;
    float3 emission = material_emission(m);
    float3 base_color = m.base_color;
    float roughness = 1.0;
    float metalness = 1.0;
    float4 normal_texel = float4(0.0);
    // Every slot the decal reads goes through one texture read, as `StandardSurface` reads its
    // slots: its coverage, its light's coverage, luminance and color, then the channels it
    // writes over the surface, which a decal that covers nothing leaves unread.
    [loop]
    for (uint k = 0; k < 8; k++)
    {
        float lit = share * (primary ? coverage : lit_coverage);
        bool covers = !(share * coverage <= 0.0);
        uint slot;
        bool read;
        switch (k)
        {
        case 0: slot = cover; read = true; break;
        case 1: slot = cover; read = emits && !primary; break;
        case 2: slot = TEXTURE_EMISSION_LUMINANCE; read = emits && lit > 0.0 && (m.flags & MATERIAL_MORE_TEXTURES) != 0; break;
        case 3: slot = TEXTURE_EMISSION; read = emits && lit > 0.0; break;
        case 4: slot = TEXTURE_BASE_COLOR; read = covers && (c & DECAL_BASE_COLOR) != 0; break;
        case 5: slot = TEXTURE_ROUGHNESS; read = covers && (c & DECAL_ROUGHNESS) != 0; break;
        case 6: slot = TEXTURE_METALNESS; read = covers && (c & DECAL_METALNESS) != 0; break;
        default: slot = TEXTURE_NORMAL; read = covers && (c & DECAL_NORMAL) != 0; break;
        }
        if (!read || !material_bound(m, slot))
            continue;
        // A decal projects one uv, which every slot reads.
        UvFootprint g = k >= 1 && k <= 3 ? e : f;
        UvFootprint at[2] = { g, g };
        float4 v = material_texture(m, slot, at, float4(1.0));
        switch (k)
        {
        case 0: coverage = cover == TEXTURE_OPACITY ? v.x : v.w; break;
        case 1: lit_coverage = cover == TEXTURE_OPACITY ? v.x : v.w; break;
        case 2: luminance = v.x; break;
        case 3: emission = material_tint(m, slot, v.rgb, emission); break;
        case 4: base_color = material_tint(m, slot, v.rgb, base_color); break;
        case 5: roughness = v.x; break;
        case 6: metalness = v.x; break;
        default: normal_texel = v; break;
        }
    }
    if (emits)
    {
        float lit = share * (primary ? coverage : lit_coverage);
        if (lit > 0.0)
            decal_over(layer.emission, layer.emission_cover, luminance * emission, lit);
    }
    share *= coverage;
    if (share <= 0.0)
        return;

    if ((c & DECAL_BASE_COLOR) != 0)
        decal_over(layer.base_color, layer.base_color_cover, m.base_weight * base_color, share);
    if ((c & DECAL_ROUGHNESS) != 0)
        decal_over(layer.roughness, layer.roughness_cover, m.specular_roughness * roughness, share);
    if ((c & DECAL_METALNESS) != 0)
        decal_over(layer.metalness, layer.metalness_cover, m.metalness * metalness, share);
    if ((c & DECAL_NORMAL) != 0)
    {
        // The decal's tangent frame on the surface: its x axis, and up its y axis, against
        // its uv's v, as on a mesh whose uv runs like the decal's. A surface past 90 degrees
        // from +z sees the projection mirrored.
        float3 n = i.normal;
        float3 axis_x = d.axis_x;
        float3 axis_y = d.axis_y;
        float3 t = safe_normalize(axis_x - n * dot(n, axis_x), cross(n, axis_y));
        float3 b = cross(n, t);
        b = dot(b, axis_y) < 0.0 ? -b : b;
        float3 normal = n;
        if (material_bound(m, TEXTURE_NORMAL))
        {
            float4 texel = normal_texel;
            if ((m.flags & MATERIAL_OBJECT_NORMAL) != 0)
            {
                // A decal has no mesh: its box stands for the space the map states.
                float3 v = object_normal_decode(texel.xyz);
                normal = axis_x * v.x + axis_y * v.y + d.axis_z * v.z;
            }
            else
            {
                float3 v = material_normal_frame(m, TEXTURE_NORMAL, normal_decode(texel.xy));
                normal = t * v.x + b * v.y + n * v.z;
            }
        }
        decal_over(layer.normal, layer.normal_cover, safe_normalize(normal, n), share);
    }
}

// Reports to the frame's texture feedback what `blend_decal` of decal `d` reads at hit `i`: its
// coverage, then its channels' slots, emission's at mip 0 past a primary hit as
// `surface_emission` reads, with the coverage there too. The coverage is a texel this does not
// read, so where it is 0 the channels report what the blend skips. One read site serves every
// slot, which keeps the code small.
void decal_feedback(SurfaceInput i, DecalGpu* d, float3 relative, float3 px, float3 py)
{
    float share;
    UvFootprint f;
    if (!decal_place(i, d, relative, px, py, share, f))
        return;
    MaterialGpu* m = i.scene.materials + d.material;
    uint c = d.channels;
    bool past = i.context.kind != HIT_PRIMARY;
    bool shown = share * m.geometry_opacity > 0.0;
    // Read 0 is the coverage's, of the slot `opacity_slot` picks.
    static const uint SLOTS[7] = { TEXTURE_OPACITY, TEXTURE_EMISSION, TEXTURE_EMISSION_LUMINANCE, TEXTURE_BASE_COLOR, TEXTURE_ROUGHNESS, TEXTURE_METALNESS, TEXTURE_NORMAL };
    static const uint CHANNELS[7] = { 0, DECAL_EMISSION, DECAL_EMISSION, DECAL_BASE_COLOR, DECAL_ROUGHNESS, DECAL_METALNESS, DECAL_NORMAL };
    for (uint k = 0; k < 7; k++)
    {
        if (k > 0 && !(shown && (c & CHANNELS[k]) != 0))
            continue;
        uint slot = k == 0 ? opacity_slot(m) : SLOTS[k];
        UvFootprint g = past && k <= 2 && (c & DECAL_EMISSION) != 0 ? finest_footprint(f) : f;
        UvFootprint gs[2] = { g, g };
        material_feedback(m, slot, gs);
    }
}

// What takes each decal over a hit in turn (`surface_decals_each`).
interface IDecalVisit
{
    // Decal `d` over hit `i`, `relative` from its center, with the hit's footprint `px`, `py`
    // (m, on the surface).
    [mutating]
    void decal(SurfaceInput i, DecalGpu* d, float3 relative, float3 px, float3 py);
}

struct DecalBlend : IDecalVisit
{
    DecalLayer layer;

    [mutating]
    void decal(SurfaceInput i, DecalGpu* d, float3 relative, float3 px, float3 py)
    {
        blend_decal(layer, i, d, relative, px, py);
    }
}

struct DecalFeedback : IDecalVisit
{
    [mutating]
    void decal(SurfaceInput i, DecalGpu* d, float3 relative, float3 px, float3 py)
    {
        decal_feedback(i, d, relative, px, py);
    }
}

// The decals over hit `i`, each handed to `v`: those of the bin of world space holding it and
// the scene's global ones, merged in id order. Texture footprints follow the hit's ray cone onto
// the triangle's plane, with the view's mip bias at primary hits.
void surface_decals_each<V : IDecalVisit>(SurfaceInput i, inout V v)
{
    SceneRoot* s = i.scene;
    uint globals = s.decal_globals;
    if (!may2(MAY2_DECALS) || (globals == 0 && s.decal_bin_slots == 0))
        return;
    uint cell_log2 = uint(s.origin_cell.w);
    int3 bin = (i.cell << (cell_log2 - s.decal_bin_log2)) + int3(floor(i.offset * exp2(-float(s.decal_bin_log2))));
    uint run = decal_bin_run(s.decal_bins, s.decal_bin_slots, bin);
    uint count = run != DECAL_BIN_EMPTY ? s.decal_refs[run] : 0;
    if (globals == 0 && count == 0)
        return;

    // The footprint: two directions across the ray, the cone's width apart, carried along
    // the ray onto the triangle's plane.
    float3 d = i.view_direction;
    float3 ng = i.geometric_normal;
    float3 x, y;
    orthonormal_basis(d, x, y);
    float width = i.context.cone_width * footprint_bias(i.context.kind, i.view);
    float dn = dot(d, ng);
    dn = abs(dn) < 1e-6 ? (dn < 0.0 ? -1e-6 : 1e-6) : dn;
    float3 px = (x - d * (dot(x, ng) / dn)) * width;
    float3 py = (y - d * (dot(y, ng) / dn)) * width;

    uint a = 0;
    uint b = 0;
    while (a < globals || b < count)
    {
        uint ga = a < globals ? s.decal_refs[a] : 0xffffffff;
        uint gb = b < count ? s.decal_refs[run + 1 + b] : 0xffffffff;
        if (ga < gb)
            a++;
        else
            b++;
        DecalGpu* dc = s.decals + min(ga, gb);
        v.decal(i, dc, view_relative(i.cell, i.offset, dc.cell, dc.offset, cell_log2), px, py);
    }
}

// The decals over hit `i` (`surface_decals_each`) blended in id order.
public DecalLayer surface_decals(SurfaceInput i)
{
    DecalBlend b = { (DecalLayer)0 };
    surface_decals_each(i, b);
    return b.layer;
}

// Emission `emission` with `layer` written over it: what `apply_decals` leaves, and what light
// sampling takes of an emitter's point.
public float3 decal_emission(DecalLayer layer, float3 emission)
{
    return emission * (1.0 - layer.emission_cover) + layer.emission;
}

// `layer` written over a surface's parameters and shading normal.
public void apply_decals(DecalLayer layer, inout BsdfParams p, inout float3 normal)
{
    p.base_color = p.base_color * (1.0 - layer.base_color_cover) + layer.base_color;
    p.specular_roughness = p.specular_roughness * (1.0 - layer.roughness_cover) + layer.roughness;
    p.metalness = p.metalness * (1.0 - layer.metalness_cover) + layer.metalness;
    p.emission = decal_emission(layer, p.emission);
    if (layer.normal_cover > 0.0)
        normal = safe_normalize(normal * (1.0 - layer.normal_cover) + layer.normal, normal);
}

// `BsdfParams.specular_color` of `m` from its specular color `color` and IOR `ior`: past OpenPBR,
// F0, a reflectance that holds to 0..1 in the space it was stated in, Rec.709.
public float3 surface_specular_color(MaterialGpu* m, float3 color, float ior)
{
    if ((m.flags & MATERIAL_FRESNEL_MIX) == 0)
        return color;
    float r = (ior - 1.0) / (ior + 1.0);
    float scale = (m.flags & MATERIAL_F0_STATED) != 0 ? 1.0 : r * r;
    return mul(REC709_TO_REC2020, saturate(scale * mul(REC2020_TO_REC709, color)));
}

// The direction `rotation` turns from `tangent` about `normal`, measured from
// orthonormal_basis(normal). A `tangent` zero or parallel to `normal` takes that basis's tangent
// instead, as OpenPBR allows any unit vector orthogonal to the normal there.
float surface_tangent_angle(float4 tangent, float3 normal, float rotation)
{
    float3 t0, b0;
    orthonormal_basis(normal, t0, b0);
    float3 t = safe_normalize(tangent.xyz - normal * dot(normal, tangent.xyz), t0);
    float3 b = cross(normal, t) * tangent.w;
    float3 d = cos(rotation) * t + sin(rotation) * b;
    return atan2(dot(d, b0), dot(d, t0));
}

// Whether `StandardSurface` reads bound slot `slot` of `m` at the hit's footprint. Opacity is
// the alpha test's, at mip 0 (`material_opacity_feedback`), displacement the tessellation's
// and the medium's slots `material_medium`'s; emission reads at mip 0 past a primary hit
// (`surface_reads_point`).
bool surface_reads(MaterialGpu* m, uint slot, bool primary)
{
    switch (slot)
    {
    case TEXTURE_OPACITY:
    case TEXTURE_DISPLACEMENT:
    case TEXTURE_DISPLACEMENT_U:
    case TEXTURE_DISPLACEMENT_V:
    case TEXTURE_TRANSMISSION_SCATTER:
    case TEXTURE_TRANSMISSION_SCATTER_ANISOTROPY:
    case TEXTURE_DISPERSION_SCALE:
    case TEXTURE_DISPERSION_ABBE:
        return false;
    case TEXTURE_EMISSION:
    case TEXTURE_EMISSION_LUMINANCE:
        return primary;
    case TEXTURE_ANISOTROPY:
    case TEXTURE_TANGENT:
        return (m.flags & MATERIAL_ANISOTROPY) != 0;
    case TEXTURE_COAT_TANGENT:
        return m.coat_weight > 0.0 && m.coat_anisotropy > 0.0;
    case TEXTURE_TRANSMISSION_COLOR:
        // A depth texel of 0 reads it too.
        return (m.flags & MATERIAL_THIN_WALLED) != 0 || m.transmission_depth == 0.0 || material_bound(m, TEXTURE_TRANSMISSION_DEPTH);
    default:
        return true;
    }
}

// Whether a hit on `m` reads bound slot `slot` at mip 0 at its uv: emission past a primary hit
// (`surface_emission`) and with MATERIAL_MEDIUM_TEXTURED, the interior medium's slots
// (`material_medium`, at the point the path enters it). A slot read here wants no coarser mip
// where `surface_reads` reads it too.
bool surface_reads_point(MaterialGpu* m, uint slot, bool primary)
{
    switch (slot)
    {
    case TEXTURE_EMISSION:
    case TEXTURE_EMISSION_LUMINANCE:
        return !primary;
    case TEXTURE_TRANSMISSION_COLOR:
    case TEXTURE_TRANSMISSION_DEPTH:
    case TEXTURE_TRANSMISSION_SCATTER:
    case TEXTURE_TRANSMISSION_SCATTER_ANISOTROPY:
    case TEXTURE_DISPERSION_SCALE:
    case TEXTURE_DISPERSION_ABBE:
        return (m.flags & MATERIAL_MEDIUM_TEXTURED) != 0;
    default:
        return false;
    }
}

// Reports to the frame's texture feedback the level each streamed texture of the hit's
// material and of the decals over it wants where `StandardSurface` reads it, as its taps would
// with `TEXTURE_FEEDBACK`, and those of the medium inside it at the point a path enters it.
// Slots a factor of 0 leaves unread report nothing, and slots a texel of 0 leaves unread report
// all the same.
public void surface_feedback(SurfaceInput i)
{
    MaterialGpu* m = i.scene.materials + i.material;
    bool primary = i.context.kind == HIT_PRIMARY;
    for (uint slot = 0; slot < TEXTURE_SLOTS; slot++)
    {
        if (!material_bound(m, slot))
            continue;
        bool point = surface_reads_point(m, slot, primary);
        if (!point && !surface_reads(m, slot, primary))
            continue;
        UvFootprint uv[2] = { point ? finest_footprint(i.uv[0]) : i.uv[0], point ? finest_footprint(i.uv[1]) : i.uv[1] };
        material_feedback(m, slot, uv);
    }
    DecalFeedback d;
    surface_decals_each(i, d);
}

// `S` whose hits report their texture feedback (`ISurface.feedback`) before they shade, where
// their context asks. A kernel resolves through it only the hits that may report, so the others
// carry none of the code.
public struct FeedbackSurface<S : ISurface> : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        if (may2(MAY2_STREAMS) && i.context.feedback)
            S.feedback(i);
        return S.surface(i, depth_offset);
    }

    public override static float coverage(SurfaceInput i)
    {
        return S.coverage(i);
    }

    public override static void feedback(SurfaceInput i)
    {
        S.feedback(i);
    }

    public override static void coverage_feedback(SurfaceInput i)
    {
        S.coverage_feedback(i);
    }

    public override static float presence(SurfaceInput i)
    {
        return S.presence(i);
    }
}

// `S` whose alpha tests (`ISurface.coverage_feedback`) and presence layers' opacity
// (`material_opacity_feedback`) report their texture feedback where their context asks; its
// hits shade as `S`'s do and report nothing. A kernel tests through it only the candidates and
// pixels that may report, so the others carry none of the code.
public struct CoverageFeedback<S : ISurface> : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        return S.surface(i, depth_offset);
    }

    public override static float coverage(SurfaceInput i)
    {
        if (may2(MAY2_STREAMS) && i.context.feedback)
            S.coverage_feedback(i);
        return S.coverage(i);
    }

    public override static void feedback(SurfaceInput i)
    {
        S.feedback(i);
    }

    public override static void coverage_feedback(SurfaceInput i)
    {
        S.coverage_feedback(i);
    }

    public override static float presence(SurfaceInput i)
    {
        if (may2(MAY2_STREAMS) && i.context.feedback)
            material_opacity_feedback(i);
        return S.presence(i);
    }
}

// OpenPBR from the hit's `MaterialGpu` and its textures, with the decals over the hit
// written over the material's parameters.
public struct StandardSurface : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.0;
        MaterialGpu* m = i.scene.materials + i.material;
        bool more = (m.flags & MATERIAL_MORE_TEXTURES) != 0;
        bool thin = (m.flags & MATERIAL_THIN_WALLED) != 0;
        bool anisotropic = (m.flags & MATERIAL_ANISOTROPY) != 0;
        // Each scalar slot's texel, 1 where unbound as `material_texture` falls back to, and each
        // color slot's color: its factor, tinted by its texel where bound (`material_color`).
        float base_weight = 1.0, metalness = 1.0, specular_ior = 1.0, specular_weight = 1.0;
        float roughness = 1.0, diffuse_roughness = 1.0, transmission = 1.0, transmission_depth = 1.0;
        float subsurface_weight = 1.0, subsurface_radius = 1.0, coat = 1.0, coat_roughness = 1.0;
        float coat_ior = 1.0, coat_darkening = 1.0, coat_anisotropy = 1.0, fuzz_weight = 1.0;
        float fuzz_roughness = 1.0, thin_film_weight = 1.0, thin_film_thickness = 1.0;
        float thin_film_ior = 1.0, emission_luminance = 1.0, specular_anisotropy = 1.0;
        float3 subsurface_radius_scale = float3(1.0);
        float subsurface_anisotropy = 1.0;
        float3 base_color = surface_color_factor(i, m, TEXTURE_BASE_COLOR, m.base_color);
        float3 specular_color = surface_color_factor(i, m, TEXTURE_SPECULAR_COLOR, m.specular_color);
        float3 transmission_color = surface_color_factor(i, m, TEXTURE_TRANSMISSION_COLOR, m.transmission_color);
        float3 subsurface_color = surface_color_factor(i, m, TEXTURE_SUBSURFACE_COLOR, m.subsurface_color);
        float3 coat_color = surface_color_factor(i, m, TEXTURE_COAT_COLOR, m.coat_color);
        float3 fuzz_color = surface_color_factor(i, m, TEXTURE_FUZZ_COLOR, m.fuzz_color);
        float3 emission = surface_color_factor(i, m, TEXTURE_EMISSION, material_emission(m));
        float4 normal = float4(0.0);
        float4 coat_normal = float4(0.0);
        float2 tangent = float2(0.0);
        float2 coat_tangent = float2(0.0);
        // Past a primary hit the emission slots read with no footprint, as `surface_emission` does.
        UvFootprint finest[2] = { finest_footprint(i.uv[0]), finest_footprint(i.uv[1]) };
        bool emission_finest = i.context.kind != HIT_PRIMARY;

        // Every bound slot the surface reads goes through one texture read, so a kernel holds one
        // copy of the texture path per surface it shades. Transmission depth comes first: it says
        // whether the transmission color is read. Slots past TEXTURE_TRANSMISSION bind only with
        // MATERIAL_MORE_TEXTURES, transmission depth aside.
        uint count = more ? TEXTURE_SLOTS : TEXTURE_TRANSMISSION + 2;
        [loop]
        for (uint k = 0; k < count; k++)
        {
            uint slot = k == 0 ? TEXTURE_TRANSMISSION_DEPTH : k - 1 < TEXTURE_TRANSMISSION_DEPTH ? k - 1 : k;
            if (!material_bound(m, slot))
                continue;
            bool read;
            switch (slot)
            {
            case TEXTURE_OPACITY:
            case TEXTURE_DISPLACEMENT:
            case TEXTURE_DISPLACEMENT_U:
            case TEXTURE_DISPLACEMENT_V:
            case TEXTURE_TRANSMISSION_SCATTER:
            case TEXTURE_TRANSMISSION_SCATTER_ANISOTROPY:
            case TEXTURE_DISPERSION_SCALE:
            case TEXTURE_DISPERSION_ABBE:
                read = false;
                break;
            case TEXTURE_TRANSMISSION_COLOR:
                read = thin || m.transmission_depth * transmission_depth == 0.0;
                break;
            case TEXTURE_ANISOTROPY:
            case TEXTURE_TANGENT:
                read = anisotropic;
                break;
            case TEXTURE_COAT_TANGENT:
                read = m.coat_weight * coat * surface_vertex(i, m, TEXTURE_COAT).x > 0.0
                    && m.coat_anisotropy * (coat_anisotropy * surface_vertex(i, m, TEXTURE_COAT_ANISOTROPY).x) > 0.0;
                break;
            default:
                read = true;
                break;
            }
            if (!read)
                continue;
            bool lit = slot == TEXTURE_EMISSION || slot == TEXTURE_EMISSION_LUMINANCE;
            float4 v = material_texture(m, slot, lit && emission_finest ? finest : i.uv, float4(1.0));
            switch (slot)
            {
            case TEXTURE_BASE_WEIGHT: base_weight = v.x; break;
            case TEXTURE_BASE_COLOR: base_color = material_tint(m, slot, v.rgb, base_color); break;
            case TEXTURE_METALNESS: metalness = v.x; break;
            case TEXTURE_SPECULAR_IOR: specular_ior = v.x; break;
            case TEXTURE_SPECULAR_COLOR: specular_color = material_tint(m, slot, v.rgb, specular_color); break;
            case TEXTURE_SPECULAR_WEIGHT: specular_weight = v.x; break;
            case TEXTURE_ROUGHNESS: roughness = v.x; break;
            case TEXTURE_DIFFUSE_ROUGHNESS: diffuse_roughness = v.x; break;
            case TEXTURE_TRANSMISSION: transmission = v.x; break;
            case TEXTURE_TRANSMISSION_DEPTH: transmission_depth = v.x; break;
            case TEXTURE_TRANSMISSION_COLOR: transmission_color = material_tint(m, slot, v.rgb, transmission_color); break;
            case TEXTURE_SUBSURFACE_WEIGHT: subsurface_weight = v.x; break;
            case TEXTURE_SUBSURFACE_COLOR: subsurface_color = material_tint(m, slot, v.rgb, subsurface_color); break;
            case TEXTURE_SUBSURFACE_RADIUS: subsurface_radius = v.x; break;
            case TEXTURE_SUBSURFACE_RADIUS_SCALE: subsurface_radius_scale = v.rgb; break;
            case TEXTURE_SUBSURFACE_ANISOTROPY: subsurface_anisotropy = v.x; break;
            case TEXTURE_COAT: coat = v.x; break;
            case TEXTURE_COAT_COLOR: coat_color = material_tint(m, slot, v.rgb, coat_color); break;
            case TEXTURE_COAT_ROUGHNESS: coat_roughness = v.x; break;
            case TEXTURE_COAT_IOR: coat_ior = v.x; break;
            case TEXTURE_COAT_DARKENING: coat_darkening = v.x; break;
            case TEXTURE_COAT_ANISOTROPY: coat_anisotropy = v.x; break;
            case TEXTURE_FUZZ_WEIGHT: fuzz_weight = v.x; break;
            case TEXTURE_FUZZ_COLOR: fuzz_color = material_tint(m, slot, v.rgb, fuzz_color); break;
            case TEXTURE_FUZZ_ROUGHNESS: fuzz_roughness = v.x; break;
            case TEXTURE_THIN_FILM_WEIGHT: thin_film_weight = v.x; break;
            case TEXTURE_THIN_FILM_THICKNESS: thin_film_thickness = v.x; break;
            case TEXTURE_THIN_FILM_IOR: thin_film_ior = v.x; break;
            case TEXTURE_EMISSION_LUMINANCE: emission_luminance = v.x; break;
            case TEXTURE_EMISSION: emission = material_tint(m, slot, v.rgb, emission); break;
            case TEXTURE_ANISOTROPY: specular_anisotropy = v.x; break;
            case TEXTURE_NORMAL: normal = v; break;
            case TEXTURE_COAT_NORMAL: coat_normal = v; break;
            case TEXTURE_TANGENT: tangent = v.xy; break;
            case TEXTURE_COAT_TANGENT: coat_tangent = v.xy; break;
            default: break;
            }
        }

        BsdfParams p;
        p.base_color = m.base_weight * (base_weight * surface_vertex(i, m, TEXTURE_BASE_WEIGHT).x) * base_color;
        p.metalness = m.metalness * metalness * surface_vertex(i, m, TEXTURE_METALNESS).x;
        p.specular_ior = m.specular_ior * (specular_ior * surface_vertex(i, m, TEXTURE_SPECULAR_IOR).x);
        p.specular_color = surface_specular_color(m, specular_color, p.specular_ior);
        p.specular_weight = m.specular_weight * specular_weight * surface_vertex(i, m, TEXTURE_SPECULAR_WEIGHT).x;
        p.specular_roughness = m.specular_roughness * roughness * surface_vertex(i, m, TEXTURE_ROUGHNESS).x;
        p.diffuse_roughness = m.diffuse_roughness * (diffuse_roughness * surface_vertex(i, m, TEXTURE_DIFFUSE_ROUGHNESS).x);
        p.transmission_weight = m.transmission_weight * transmission * surface_vertex(i, m, TEXTURE_TRANSMISSION).x;
        // OpenPBR: at a depth of 0 the color "acts as a constant (on-surface) transmission tint".
        float depth = m.transmission_depth * transmission_depth;
        p.transmission_color = thin || depth == 0.0 ? transmission_color : float3(1.0);
        p.subsurface_weight = m.subsurface_weight * (subsurface_weight * surface_vertex(i, m, TEXTURE_SUBSURFACE_WEIGHT).x);
        p.subsurface_color = subsurface_color;
        p.subsurface_anisotropy = m.subsurface_scatter_anisotropy * subsurface_anisotropy;
        p.subsurface_radius = m.subsurface_radius * (subsurface_radius * surface_vertex(i, m, TEXTURE_SUBSURFACE_RADIUS).x) * m.subsurface_radius_scale;
        if (more)
            p.subsurface_radius *= subsurface_radius_scale;
        p.coat_weight = m.coat_weight * coat * surface_vertex(i, m, TEXTURE_COAT).x;
        p.coat_color = coat_color;
        p.coat_roughness = m.coat_roughness * (coat_roughness * surface_vertex(i, m, TEXTURE_COAT_ROUGHNESS).x);
        p.coat_ior = m.coat_ior * (coat_ior * surface_vertex(i, m, TEXTURE_COAT_IOR).x);
        p.coat_darkening = m.coat_darkening * (coat_darkening * surface_vertex(i, m, TEXTURE_COAT_DARKENING).x);
        p.fuzz_weight = m.fuzz_weight * (fuzz_weight * surface_vertex(i, m, TEXTURE_FUZZ_WEIGHT).x);
        p.fuzz_color = fuzz_color;
        p.fuzz_roughness = m.fuzz_roughness * (fuzz_roughness * surface_vertex(i, m, TEXTURE_FUZZ_ROUGHNESS).x);
        p.thin_film_weight = m.thin_film_weight * (thin_film_weight * surface_vertex(i, m, TEXTURE_THIN_FILM_WEIGHT).x);
        p.thin_film_thickness = m.thin_film_thickness * (thin_film_thickness * surface_vertex(i, m, TEXTURE_THIN_FILM_THICKNESS).x);
        p.thin_film_ior = m.thin_film_ior * (thin_film_ior * surface_vertex(i, m, TEXTURE_THIN_FILM_IOR).x);
        p.emission = (emission_luminance * surface_vertex(i, m, TEXTURE_EMISSION_LUMINANCE).x) * emission;

        GSurface s = (GSurface)0;
        s.normal = material_bound(m, TEXTURE_NORMAL) ? material_texel_normal(i, m, TEXTURE_NORMAL, normal) : i.normal;
        apply_decals(surface_decals(i), p, s.normal);
        if ((m.flags & MATERIAL_BASE_TINT) != 0)
            p.transmission_color *= p.base_color;
        s.geometric_normal = i.geometric_normal;
        p.specular_anisotropy = 0.0;
        p.tangent_angle = 0.0;
        p.flags = thin ? BSDF_THIN_WALLED : 0;
        if ((m.flags & MATERIAL_FRESNEL_MIX) != 0)
            p.flags |= BSDF_FRESNEL_MIX;
        if ((m.flags & MATERIAL_METAL_FLAT) != 0)
            p.flags |= BSDF_METAL_FLAT;
        if (anisotropic)
        {
            p.specular_anisotropy = m.specular_anisotropy * (specular_anisotropy * surface_vertex(i, m, TEXTURE_ANISOTROPY).x);
            float direction = more && material_bound(m, TEXTURE_TANGENT) ? material_texel_direction(m, TEXTURE_TANGENT, tangent) : 0.0;
            p.tangent_angle = surface_tangent_angle(i.tangent, s.normal, m.specular_rotation + direction);
            p.flags |= BSDF_ANISOTROPY;
        }
        p.coat_normal = (m.flags & MATERIAL_COAT_ON_BASE) != 0 ? s.normal : i.normal;
        if (more && material_bound(m, TEXTURE_COAT_NORMAL))
            p.coat_normal = material_texel_normal(i, m, TEXTURE_COAT_NORMAL, coat_normal);
        p.coat_anisotropy = m.coat_anisotropy * (coat_anisotropy * surface_vertex(i, m, TEXTURE_COAT_ANISOTROPY).x);
        p.coat_tangent_angle = 0.0;
        if (p.coat_weight > 0.0)
        {
            p.flags |= BSDF_COAT;
            if (p.coat_anisotropy > 0.0)
            {
                float direction = more && material_bound(m, TEXTURE_COAT_TANGENT) ? material_texel_direction(m, TEXTURE_COAT_TANGENT, coat_tangent) : 0.0;
                p.coat_tangent_angle = surface_tangent_angle(i.coat_tangent, p.coat_normal, m.coat_rotation + direction);
            }
            if (p.coat_anisotropy > 0.0 || any(p.coat_normal != s.normal))
                p.flags |= BSDF_COAT_FRAME;
        }
        if (p.fuzz_weight > 0.0)
            p.flags |= BSDF_FUZZ;
        if (p.thin_film_weight > 0.0)
            p.flags |= BSDF_THIN_FILM;
        s.bsdf = p;
        s.flags = (m.flags & MATERIAL_UNLIT) != 0 ? GBUFFER_UNLIT : 0;
        return s;
    }

    public override static float coverage(SurfaceInput i)
    {
        return material_coverage(i);
    }
}
