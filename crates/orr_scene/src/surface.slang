// The surface link point. A consumer surface is a type implementing `ISurface`, linked at
// build time into every kernel that shades hits: the material resolve for primary hits and
// ray hit shading for every other. `StandardSurface` shades from `MaterialGpu` and serves
// surface id 0 and every id nothing links.
module surface;

import orr.abi;
import orr.math;
import orr_pack.normal_map;
__exported import gbuffer;
__exported import material;
__exported import tables;
__exported import view;

// `HitContext.kind`.
public static const uint HIT_PRIMARY = 0;   // the material resolve and the dielectric camera ray's first hit
public static const uint HIT_SECONDARY = 1; // every other ray hit

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
    public uint user_tag;           // the copy's or entry's
    public uint cluster_user;       // producer cluster word; 0 on cooked clusters
    public uint3 vertex_user;       // producer vertex words per corner; 0 on cooked clusters
    public uint material;
    public uint surface;            // linked surface id; 0 = StandardSurface
    public UvFootprint uv;
    public float3 geometric_normal; // the triangle's front face
    public float3 normal;           // interpolated vertex normal
    public float4 tangent;          // xyz unit tangent, w bitangent sign
    public float3 object_position;
    public float3 object_normal;
    public float3x3 object_to_render; // the instance's rotation and scale
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
// times `geometry_opacity`, read bilinearly from mip 0 whatever the footprint.
public float material_opacity(SurfaceInput i)
{
    MaterialGpu* m = i.scene.materials + i.material;
    UvFootprint finest = { i.uv.uv, float2(0.0), float2(0.0), float2(-1.0) };
    float opacity = m.geometry_opacity;
    if (material_bound(m, TEXTURE_OPACITY))
        opacity *= material_texture(m, TEXTURE_OPACITY, finest, float4(1.0)).x;
    else if (material_bound(m, TEXTURE_BASE_COLOR))
        opacity *= material_texture(m, TEXTURE_BASE_COLOR, finest, float4(1.0)).w;
    return opacity;
}

// The material's alpha test at the hit: 1 where `material_opacity` is at or above the
// cutoff, else 0: the test the cook's opaque triangles and opacity micromaps follow, so the
// raster, ray candidates and micromaps agree.
public float material_coverage(SurfaceInput i)
{
    return material_opacity(i) >= i.scene.materials[i.material].alpha_cutoff ? 1.0 : 0.0;
}

// The shading frame: the vertex normal and tangent, turned by the material's normal map.
public float3 material_normal(SurfaceInput i, MaterialGpu* m)
{
    float3 n = i.normal;
    if (!material_bound(m, TEXTURE_NORMAL))
        return n;
    float3 v = material_normal_frame(m, TEXTURE_NORMAL, normal_decode(material_texture(m, TEXTURE_NORMAL, i.uv, float4(0.5, 0.5, 1.0, 1.0)).xy));
    float3 t = safe_normalize(i.tangent.xyz - n * dot(n, i.tangent.xyz), float3(1, 0, 0));
    float3 b = cross(n, t) * i.tangent.w;
    return safe_normalize(t * v.x + b * v.y + n * v.z, n);
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
    f.tap = i.uv.tap;
    float4 one = float4(1.0);
    share *= m.geometry_opacity;
    if (material_bound(m, TEXTURE_OPACITY))
        share *= material_texture(m, TEXTURE_OPACITY, f, one).x;
    else if (material_bound(m, TEXTURE_BASE_COLOR))
        share *= material_texture(m, TEXTURE_BASE_COLOR, f, one).w;
    if (share <= 0.0)
        return;

    uint c = d.channels;
    if ((c & DECAL_BASE_COLOR) != 0)
        decal_over(layer.base_color, layer.base_color_cover, m.base_weight * material_color(m, TEXTURE_BASE_COLOR, f, m.base_color), share);
    if ((c & DECAL_ROUGHNESS) != 0)
        decal_over(layer.roughness, layer.roughness_cover, m.specular_roughness * material_texture(m, TEXTURE_ROUGHNESS, f, one).x, share);
    if ((c & DECAL_METALNESS) != 0)
        decal_over(layer.metalness, layer.metalness_cover, m.metalness * material_texture(m, TEXTURE_METALNESS, f, one).x, share);
    if ((c & DECAL_EMISSION) != 0)
        decal_over(layer.emission, layer.emission_cover, material_color(m, TEXTURE_EMISSION, f, material_emission(m)), share);
    if ((c & DECAL_NORMAL) != 0)
    {
        // The decal's tangent frame on the surface: its x axis, and its uv's v (down its y
        // axis) as the bitangent, as on a mesh whose uv runs like the decal's.
        float3 n = i.normal;
        float3 t = safe_normalize(axis_x - n * dot(n, axis_x), cross(n, axis_y));
        float3 b = -cross(n, t);
        float3 v = float3(0.0, 0.0, 1.0);
        if (material_bound(m, TEXTURE_NORMAL))
            v = material_normal_frame(m, TEXTURE_NORMAL, normal_decode(material_texture(m, TEXTURE_NORMAL, f, float4(0.5, 0.5, 1.0, 1.0)).xy));
        decal_over(layer.normal, layer.normal_cover, safe_normalize(t * v.x + b * v.y + n * v.z, n), share);
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
    float width = i.context.cone_width;
    if (i.context.kind == HIT_PRIMARY && i.view != nullptr)
        width *= exp2(i.view.mip_bias);
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
        p.base_color = m.base_weight * material_color(m, TEXTURE_BASE_COLOR, i.uv, m.base_color);
        p.metalness = m.metalness * material_texture(m, TEXTURE_METALNESS, i.uv, one).x;
        p.specular_color = m.specular_color;
        p.specular_weight = m.specular_weight * material_texture(m, TEXTURE_SPECULAR_WEIGHT, i.uv, one).x;
        p.specular_roughness = m.specular_roughness * material_texture(m, TEXTURE_ROUGHNESS, i.uv, one).x;
        p.specular_ior = m.specular_ior;
        p.diffuse_roughness = m.diffuse_roughness;
        p.transmission_weight = m.transmission_weight * material_texture(m, TEXTURE_TRANSMISSION, i.uv, one).x;
        bool thin = (m.flags & MATERIAL_THIN_WALLED) != 0;
        p.transmission_color = thin || m.transmission_depth == 0.0 ? m.transmission_color : float3(1.0);
        p.subsurface_weight = m.subsurface_weight;
        p.subsurface_color = material_color(m, TEXTURE_SUBSURFACE_COLOR, i.uv, m.subsurface_color);
        p.subsurface_anisotropy = m.subsurface_scatter_anisotropy;
        p.subsurface_radius = m.subsurface_radius * m.subsurface_radius_scale;
        p.coat_weight = m.coat_weight * material_texture(m, TEXTURE_COAT, i.uv, one).x;
        p.coat_color = m.coat_color;
        p.coat_roughness = m.coat_roughness;
        p.coat_ior = m.coat_ior;
        p.coat_darkening = 1.0;
        p.fuzz_weight = m.fuzz_weight;
        p.fuzz_color = m.fuzz_color;
        p.fuzz_roughness = m.fuzz_roughness;
        bool film = (m.flags & MATERIAL_THIN_FILM) != 0;
        p.thin_film_weight = film ? 1.0 : 0.0;
        p.thin_film_thickness = m.thin_film_thickness;
        p.thin_film_ior = m.thin_film_ior;
        p.emission = material_color(m, TEXTURE_EMISSION, i.uv, material_emission(m));
        p.scale = float3(material_texture(m, TEXTURE_OCCLUSION, i.uv, one).x);

        GSurface s = (GSurface)0;
        s.normal = material_normal(i, m);
        apply_decals(surface_decals(i), p, s.normal);
        s.geometric_normal = i.geometric_normal;
        p.specular_anisotropy = 0.0;
        p.tangent_angle = 0.0;
        p.flags = thin ? BSDF_THIN_WALLED : 0;
        if ((m.flags & MATERIAL_ANISOTROPY) != 0)
        {
            // The anisotropy direction, turned from the mesh tangent, then measured from
            // orthonormal_basis(n).
            p.specular_anisotropy = m.specular_anisotropy;
            float3 t = safe_normalize(i.tangent.xyz - s.normal * dot(s.normal, i.tangent.xyz), float3(1, 0, 0));
            float3 b = cross(s.normal, t) * i.tangent.w;
            float3 d = cos(m.specular_rotation) * t + sin(m.specular_rotation) * b;
            float3 t0, b0;
            orthonormal_basis(s.normal, t0, b0);
            p.tangent_angle = atan2(dot(d, b0), dot(d, t0));
            p.flags |= BSDF_ANISOTROPY;
        }
        if (p.coat_weight > 0.0)
            p.flags |= BSDF_COAT;
        if (p.fuzz_weight > 0.0)
            p.flags |= BSDF_FUZZ;
        if (film)
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
