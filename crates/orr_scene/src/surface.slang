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
}

// Everything a surface knows about its hit. Directions and normals are unit vectors in
// render space unless named object.
public struct SurfaceInput
{
    public HitContext context;
    public uint instance;           // instance slot or producer entry
    public uint cluster;            // mesh cluster index
    public uint part;
    public uint primitive;          // triangle in the cluster
    public float2 barycentrics;     // weights of corners 1 and 2
    public float4 vertex_color;     // the mesh's color stream here; 1 without one
    public uint user_tag;           // the copy's or entry's
    public uint cluster_user;       // producer cluster word; 0 on cooked clusters
    public uint3 vertex_user;       // producer vertex words per corner; 0 on cooked clusters
    public uint material;
    public uint surface;            // linked surface id; 0 = StandardSurface
    public UvFootprint uv[2];       // per uv set; producer and patch geometry give both its one uv
    public float3 geometric_normal; // the triangle's front face
    public bool back_face;          // the ray met the back of a double-sided material
    public float3 normal;           // interpolated vertex normal, unit
    public float3 frame_normal;     // `normal` before normalizing, not always unit
    public float4 tangent;          // xyz as interpolated, not always unit; w bitangent sign
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
}

// The material's opacity at the hit: the opacity slot, else the base color slot's alpha,
// times `geometry_opacity`, read bilinearly from mip 0 whatever the footprint; 1 for a
// material that is neither alpha-tested nor a presence layer, whose alpha is not read.
public float material_opacity(SurfaceInput i)
{
    MaterialGpu* m = i.scene.materials + i.material;
    if ((m.flags & (MATERIAL_ALPHA_TESTED | MATERIAL_PRESENCE)) == 0)
        return 1.0;
    UvFootprint finest[2] = {
        { i.uv[0].uv, float2(0.0), float2(0.0), float2(-1.0), i.uv[0].inward },
        { i.uv[1].uv, float2(0.0), float2(0.0), float2(-1.0), i.uv[1].inward },
    };
    float opacity = m.geometry_opacity * surface_vertex(i, m, TEXTURE_OPACITY).x;
    if (material_bound(m, TEXTURE_OPACITY))
        opacity *= material_texture(m, TEXTURE_OPACITY, finest, float4(1.0)).x;
    else if (material_bound(m, TEXTURE_BASE_COLOR))
        opacity *= material_texture(m, TEXTURE_BASE_COLOR, finest, float4(1.0)).w;
    return opacity;
}

// What the vertex color stream multiplies `slot` by at the hit: from its first channel on in
// xyz, or the complement of that for a slot that reads one, and 1 for a slot that reads none.
public float3 surface_vertex(SurfaceInput i, MaterialGpu* m, uint slot)
{
    if ((m.flags & MATERIAL_VERTEX) == 0)
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
    if ((m.flags & MATERIAL_VERTEX) != 0 && ((m.samplers[slot].modes >> TEXTURE_VERTEX_SHIFT) & TEXTURE_VERTEX_MASK) != 0)
        factor = mul(REC709_TO_REC2020, mul(REC2020_TO_REC709, factor) * surface_vertex(i, m, slot));
    return material_color(m, slot, i.uv, factor);
}

// `material_more_color` of `slot`, with `surface_color`'s vertex color.
float3 surface_more_color(SurfaceInput i, MaterialGpu* m, uint slot, float3 factor)
{
    if ((m.flags & (MATERIAL_MORE_TEXTURES | MATERIAL_VERTEX)) == 0)
        return factor;
    return surface_color(i, m, slot, factor);
}

// What the surface emits at the hit, in nits: `material_emission_at` with the vertex color
// stream on both of its slots.
public float3 surface_emission(SurfaceInput i, MaterialGpu* m)
{
    return surface_scale(i, m, TEXTURE_EMISSION_LUMINANCE) * surface_color(i, m, TEXTURE_EMISSION, material_emission(m));
}

// The material's alpha test at the hit: 1 where `material_opacity` is at or above the
// cutoff, else 0: the test the cook's opaque triangles and opacity micromaps follow, so the
// raster, ray candidates and micromaps agree.
public float material_coverage(SurfaceInput i)
{
    return material_opacity(i) >= i.scene.materials[i.material].alpha_cutoff ? 1.0 : 0.0;
}

// The shading frame: the vertex normal, turned by the material's normal map in `slot`.
// MikkTSpace's bakers invert this transform: the interpolated normal and tangent as they stand,
// neither unit nor perpendicular. A map of mesh space directions takes the placement's normal
// transform, and the side `i.normal` faces where a double-sided back face turned it.
public float3 material_normal(SurfaceInput i, MaterialGpu* m, uint slot = TEXTURE_NORMAL)
{
    if (!material_bound(m, slot))
        return i.normal;
    float4 texel = material_texture(m, slot, i.uv, float4(0.5, 0.5, 1.0, 1.0));
    if ((m.flags & (slot == TEXTURE_NORMAL ? MATERIAL_OBJECT_NORMAL : MATERIAL_OBJECT_COAT_NORMAL)) != 0)
    {
        float3x3 c = normal_matrix(i.object_to_render);
        float side = dot(mul(c, i.object_normal), i.normal) < 0.0 ? -1.0 : 1.0;
        return safe_normalize(mul(c, 2.0 * texel.xyz - 1.0) * side, i.normal);
    }
    float3 v = material_normal_frame(m, slot, normal_decode(texel.xy));
    float3 n = i.frame_normal;
    float3 t = i.tangent.xyz;
    float3 b = cross(n, t) * i.tangent.w;
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

// Decal `d` blended over `layer` at hit `i`, `relative` from the decal's center, with the
// hit's footprint `px`, `py` (m, on the surface).
void blend_decal(inout DecalLayer layer, SurfaceInput i, DecalGpu* d, float3 relative, float3 px, float3 py)
{
    float3 axis_x = d.axis_x;
    float3 axis_y = d.axis_y;
    float3 local = float3(dot(relative, axis_x) * d.inv_half_x, dot(relative, axis_y) * d.inv_half_y, dot(relative, d.axis_z) * d.inv_half_z);
    if (any(abs(local) > 1.0))
        return;
    float facing = dot(i.geometric_normal, d.axis_z);
    float share = d.opacity * saturate((facing - d.cos_none) / max(d.cos_full - d.cos_none, 1e-6));
    if (d.depth_fade > 0.0)
        share *= saturate((1.0 - abs(local.z)) / d.depth_fade);
    if (share <= 0.0)
        return;

    MaterialGpu* m = i.scene.materials + d.material;
    float2 scale = float2(0.5 * d.inv_half_x, -0.5 * d.inv_half_y);
    UvFootprint f;
    f.uv = float2(0.5, 0.5) + local.xy * float2(0.5, -0.5);
    f.dx = float2(dot(px, axis_x), dot(px, axis_y)) * scale;
    f.dy = float2(dot(py, axis_x), dot(py, axis_y)) * scale;
    f.tap = i.uv[0].tap;
    f.inward = float2(0.0);
    // A decal projects one uv, which every slot reads.
    UvFootprint fs[2] = { f, f };
    float4 one = float4(1.0);
    share *= m.geometry_opacity;
    if (material_bound(m, TEXTURE_OPACITY))
        share *= material_texture(m, TEXTURE_OPACITY, fs, one).x;
    else if (material_bound(m, TEXTURE_BASE_COLOR))
        share *= material_texture(m, TEXTURE_BASE_COLOR, fs, one).w;
    if (share <= 0.0)
        return;

    uint c = d.channels;
    if ((c & DECAL_BASE_COLOR) != 0)
        decal_over(layer.base_color, layer.base_color_cover, m.base_weight * material_color(m, TEXTURE_BASE_COLOR, fs, m.base_color), share);
    if ((c & DECAL_ROUGHNESS) != 0)
        decal_over(layer.roughness, layer.roughness_cover, m.specular_roughness * material_texture(m, TEXTURE_ROUGHNESS, fs, one).x, share);
    if ((c & DECAL_METALNESS) != 0)
        decal_over(layer.metalness, layer.metalness_cover, m.metalness * material_texture(m, TEXTURE_METALNESS, fs, one).x, share);
    if ((c & DECAL_EMISSION) != 0)
        decal_over(layer.emission, layer.emission_cover, material_emission_at(m, fs), share);
    if ((c & DECAL_NORMAL) != 0)
    {
        // The decal's tangent frame on the surface: its x axis, and up its y axis, against
        // its uv's v, as on a mesh whose uv runs like the decal's. A surface past 90 degrees
        // from +z sees the projection mirrored.
        float3 n = i.normal;
        float3 t = safe_normalize(axis_x - n * dot(n, axis_x), cross(n, axis_y));
        float3 b = cross(n, t);
        b = dot(b, axis_y) < 0.0 ? -b : b;
        float3 normal = n;
        if (material_bound(m, TEXTURE_NORMAL))
        {
            float4 texel = material_texture(m, TEXTURE_NORMAL, fs, float4(0.5, 0.5, 1.0, 1.0));
            if ((m.flags & MATERIAL_OBJECT_NORMAL) != 0)
            {
                // A decal has no mesh: its box stands for the space the map states.
                float3 v = 2.0 * texel.xyz - 1.0;
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

// The decals over hit `i`: those of the bin of world space holding it and the scene's global
// ones, merged in id order. Texture footprints follow the hit's ray cone onto the triangle's
// plane, with the view's mip bias at primary hits.
public DecalLayer surface_decals(SurfaceInput i)
{
    DecalLayer layer = (DecalLayer)0;
    SceneRoot* s = i.scene;
    uint globals = s.decal_globals;
    if (globals == 0 && s.decal_bin_slots == 0)
        return layer;
    uint cell_log2 = uint(s.origin_cell.w);
    int3 bin = (i.cell << (cell_log2 - s.decal_bin_log2)) + int3(floor(i.offset * exp2(-float(s.decal_bin_log2))));
    uint run = decal_bin_run(s.decal_bins, s.decal_bin_slots, bin);
    uint count = run != DECAL_BIN_EMPTY ? s.decal_refs[run] : 0;
    if (globals == 0 && count == 0)
        return layer;

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
        blend_decal(layer, i, dc, view_relative(i.cell, i.offset, dc.cell, dc.offset, cell_log2), px, py);
    }
    return layer;
}

// `layer` written over a surface's parameters and shading normal.
public void apply_decals(DecalLayer layer, inout BsdfParams p, inout float3 normal)
{
    p.base_color = p.base_color * (1.0 - layer.base_color_cover) + layer.base_color;
    p.specular_roughness = p.specular_roughness * (1.0 - layer.roughness_cover) + layer.roughness;
    p.metalness = p.metalness * (1.0 - layer.metalness_cover) + layer.metalness;
    p.emission = p.emission * (1.0 - layer.emission_cover) + layer.emission;
    if (layer.normal_cover > 0.0)
        normal = safe_normalize(normal * (1.0 - layer.normal_cover) + layer.normal, normal);
}

// `BsdfParams.specular_color` of `m` from its specular color `color` and IOR `ior`: past OpenPBR,
// F0, a reflectance that holds to 0..1 in the space it was stated in, Rec.709.
float3 surface_specular_color(MaterialGpu* m, float3 color, float ior)
{
    if ((m.flags & MATERIAL_FRESNEL_MIX) == 0)
        return color;
    float r = (ior - 1.0) / (ior + 1.0);
    float scale = (m.flags & MATERIAL_F0_STATED) != 0 ? 1.0 : r * r;
    return mul(REC709_TO_REC2020, saturate(scale * mul(REC2020_TO_REC709, color)));
}

// The direction `rotation` turns from the mesh tangent about `normal`, measured from
// orthonormal_basis(normal).
float surface_tangent_angle(SurfaceInput i, float3 normal, float rotation)
{
    float3 t = safe_normalize(i.tangent.xyz - normal * dot(normal, i.tangent.xyz), float3(1, 0, 0));
    float3 b = cross(normal, t) * i.tangent.w;
    float3 d = cos(rotation) * t + sin(rotation) * b;
    float3 t0, b0;
    orthonormal_basis(normal, t0, b0);
    return atan2(dot(d, b0), dot(d, t0));
}

// OpenPBR from the hit's `MaterialGpu` and its textures, with the decals over the hit
// written over the material's parameters.
public struct StandardSurface : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.0;
        MaterialGpu* m = i.scene.materials + i.material;
        float4 one = float4(1.0);
        BsdfParams p;
        p.base_color = m.base_weight * surface_scale(i, m, TEXTURE_BASE_WEIGHT) * surface_color(i, m, TEXTURE_BASE_COLOR, m.base_color);
        p.metalness = m.metalness * material_texture(m, TEXTURE_METALNESS, i.uv, one).x * surface_vertex(i, m, TEXTURE_METALNESS).x;
        p.specular_ior = m.specular_ior * surface_scale(i, m, TEXTURE_SPECULAR_IOR);
        p.specular_color = surface_specular_color(m, surface_more_color(i, m, TEXTURE_SPECULAR_COLOR, m.specular_color), p.specular_ior);
        p.specular_weight = m.specular_weight * material_texture(m, TEXTURE_SPECULAR_WEIGHT, i.uv, one).x * surface_vertex(i, m, TEXTURE_SPECULAR_WEIGHT).x;
        p.specular_roughness = m.specular_roughness * material_texture(m, TEXTURE_ROUGHNESS, i.uv, one).x * surface_vertex(i, m, TEXTURE_ROUGHNESS).x;
        p.diffuse_roughness = m.diffuse_roughness * surface_scale(i, m, TEXTURE_DIFFUSE_ROUGHNESS);
        p.transmission_weight = m.transmission_weight * material_texture(m, TEXTURE_TRANSMISSION, i.uv, one).x * surface_vertex(i, m, TEXTURE_TRANSMISSION).x;
        bool thin = (m.flags & MATERIAL_THIN_WALLED) != 0;
        p.transmission_color = thin || m.transmission_depth == 0.0 ? surface_more_color(i, m, TEXTURE_TRANSMISSION_COLOR, m.transmission_color) : float3(1.0);
        p.subsurface_weight = m.subsurface_weight * surface_scale(i, m, TEXTURE_SUBSURFACE_WEIGHT);
        p.subsurface_color = surface_color(i, m, TEXTURE_SUBSURFACE_COLOR, m.subsurface_color);
        p.subsurface_anisotropy = m.subsurface_scatter_anisotropy;
        p.subsurface_radius = m.subsurface_radius * surface_scale(i, m, TEXTURE_SUBSURFACE_RADIUS) * m.subsurface_radius_scale;
        if ((m.flags & MATERIAL_MORE_TEXTURES) != 0)
        {
            p.subsurface_radius *= material_texture(m, TEXTURE_SUBSURFACE_RADIUS_SCALE, i.uv, one).rgb;
            p.subsurface_anisotropy = 2.0 * material_texture(m, TEXTURE_SUBSURFACE_ANISOTROPY, i.uv, float4(0.5 * p.subsurface_anisotropy + 0.5)).x - 1.0;
        }
        p.coat_weight = m.coat_weight * material_texture(m, TEXTURE_COAT, i.uv, one).x * surface_vertex(i, m, TEXTURE_COAT).x;
        p.coat_color = surface_more_color(i, m, TEXTURE_COAT_COLOR, m.coat_color);
        p.coat_roughness = m.coat_roughness * surface_scale(i, m, TEXTURE_COAT_ROUGHNESS);
        p.coat_ior = m.coat_ior * surface_scale(i, m, TEXTURE_COAT_IOR);
        p.coat_darkening = m.coat_darkening * surface_scale(i, m, TEXTURE_COAT_DARKENING);
        p.fuzz_weight = m.fuzz_weight * surface_scale(i, m, TEXTURE_FUZZ_WEIGHT);
        p.fuzz_color = surface_more_color(i, m, TEXTURE_FUZZ_COLOR, m.fuzz_color);
        p.fuzz_roughness = m.fuzz_roughness * surface_scale(i, m, TEXTURE_FUZZ_ROUGHNESS);
        p.thin_film_weight = m.thin_film_weight * surface_scale(i, m, TEXTURE_THIN_FILM_WEIGHT);
        p.thin_film_thickness = m.thin_film_thickness * surface_scale(i, m, TEXTURE_THIN_FILM_THICKNESS);
        p.thin_film_ior = m.thin_film_ior * surface_scale(i, m, TEXTURE_THIN_FILM_IOR);
        p.emission = surface_emission(i, m);

        GSurface s = (GSurface)0;
        s.normal = material_normal(i, m);
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
        if ((m.flags & MATERIAL_ANISOTROPY) != 0)
        {
            p.specular_anisotropy = m.specular_anisotropy * surface_scale(i, m, TEXTURE_ANISOTROPY);
            float rotation = m.specular_rotation + material_direction(m, TEXTURE_TANGENT, i.uv);
            p.tangent_angle = surface_tangent_angle(i, s.normal, rotation);
            p.flags |= BSDF_ANISOTROPY;
        }
        p.coat_normal = (m.flags & MATERIAL_COAT_ON_BASE) != 0 ? s.normal : i.normal;
        if ((m.flags & MATERIAL_MORE_TEXTURES) != 0)
            p.coat_normal = material_bound(m, TEXTURE_COAT_NORMAL) ? material_normal(i, m, TEXTURE_COAT_NORMAL) : p.coat_normal;
        p.coat_anisotropy = m.coat_anisotropy * surface_scale(i, m, TEXTURE_COAT_ANISOTROPY);
        p.coat_tangent_angle = 0.0;
        if (p.coat_weight > 0.0)
        {
            p.flags |= BSDF_COAT;
            if (p.coat_anisotropy > 0.0)
                p.coat_tangent_angle = surface_tangent_angle(i, p.coat_normal, m.coat_rotation + material_direction(m, TEXTURE_COAT_TANGENT, i.uv));
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
