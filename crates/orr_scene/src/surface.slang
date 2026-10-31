// The surface link point. A consumer surface is a type implementing `ISurface`, linked at
// build time into every kernel that shades hits: the material resolve for primary hits and
// ray hit shading for every other. `StandardSurface` shades from `MaterialGpu` and serves
// surface id 0 and every id nothing links.
module surface;

import orr.abi;
import orr.math;
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

// The material's alpha test at the hit: 1 at or above the cutoff, else 0. The opacity is the
// opacity slot, else the base color slot's alpha, times `geometry_opacity`, read bilinearly
// from mip 0 whatever the footprint: the test the cook's opaque triangles and opacity
// micromaps follow, so the raster, ray candidates and micromaps agree.
public float material_coverage(SurfaceInput i)
{
    MaterialGpu* m = i.scene.materials + i.material;
    UvFootprint finest = { i.uv.uv, float2(0.0), float2(0.0), float2(-1.0) };
    float opacity = m.geometry_opacity;
    if (material_bound(m, TEXTURE_OPACITY))
        opacity *= material_texture(m, TEXTURE_OPACITY, finest, float4(1.0)).x;
    else if (material_bound(m, TEXTURE_BASE_COLOR))
        opacity *= material_texture(m, TEXTURE_BASE_COLOR, finest, float4(1.0)).w;
    return opacity >= m.alpha_cutoff ? 1.0 : 0.0;
}

// The shading frame: the vertex normal and tangent, turned by the material's normal map.
public float3 material_normal(SurfaceInput i, MaterialGpu* m)
{
    float3 n = i.normal;
    if (!material_bound(m, TEXTURE_NORMAL))
        return n;
    float2 xy = material_texture(m, TEXTURE_NORMAL, i.uv, float4(0.5, 0.5, 1.0, 1.0)).xy * 2.0 - 1.0;
    float z = sqrt(saturate(1.0 - dot(xy, xy)));
    float3 t = safe_normalize(i.tangent.xyz - n * dot(n, i.tangent.xyz), float3(1, 0, 0));
    float3 b = cross(n, t) * i.tangent.w;
    return safe_normalize(t * xy.x + b * xy.y + n * z, n);
}

// OpenPBR from the hit's `MaterialGpu` and its textures.
public struct StandardSurface : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.0;
        MaterialGpu* m = i.scene.materials + i.material;
        float4 one = float4(1.0);
        BsdfParams p;
        float3 base = material_texture(m, TEXTURE_BASE_COLOR, i.uv, one).rgb;
        p.base_color = m.base_weight * m.base_color * base;
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
        p.subsurface_color = m.subsurface_color * material_texture(m, TEXTURE_SUBSURFACE_COLOR, i.uv, one).rgb;
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
        float3 emission = material_texture(m, TEXTURE_EMISSION, i.uv, one).rgb;
        p.emission = material_emission(m) * emission;
        p.scale = float3(material_texture(m, TEXTURE_OCCLUSION, i.uv, one).x);

        GSurface s = (GSurface)0;
        s.normal = material_normal(i, m);
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
