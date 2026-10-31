// The G-buffer: what the material resolve writes per pixel, its packing, and
// `load_gsurface`, which is how every later pass reads it.
//
// Core targets: `a` RGBA32_UINT, `b` RG32_UINT, depth D32 (reverse Z, 0 = no geometry) and
// motion RG16F (previous uv minus current uv, unjittered). The extended target `x`
// (RGBA32_UINT) holds subsurface and lobe scalars where `GBUFFER_EXTENDED` is set.
module gbuffer;

import orr.abi;
import orr.color;
import orr.math;
import orr.packing;
import material;
import tables;
import view;

// `BsdfParams.flags`.
public static const uint BSDF_THIN_WALLED = 1 << 0;
public static const uint BSDF_COAT = 1 << 1;
public static const uint BSDF_FUZZ = 1 << 2;
public static const uint BSDF_THIN_FILM = 1 << 3;
public static const uint BSDF_ANISOTROPY = 1 << 4;
// Set by `bsdf_collapse`.
public static const uint BSDF_COLLAPSED = 1 << 5;

// OpenPBR inputs at a shading point, linear Rec.2020 and absolute units.
public struct BsdfParams
{
    public float3 base_color;           // base_weight x base_color
    public float metalness;
    public float3 specular_color;
    public float specular_weight;
    public float specular_roughness;
    public float specular_anisotropy;
    public float specular_ior;
    public float tangent_angle;         // rad: the anisotropy direction, from orthonormal_basis(n)'s tangent toward its bitangent
    public float diffuse_roughness;
    public float transmission_weight;
    public float3 transmission_color;   // tint applied at the surface: thin-walled or zero depth; 1 when the interior medium carries the color
    public float subsurface_weight;
    public float3 subsurface_color;
    public float subsurface_anisotropy;
    public float3 subsurface_radius;    // m, mean free path per channel (radius x radius_scale)
    public float coat_weight;
    public float3 coat_color;
    public float coat_roughness;
    public float coat_ior;
    public float coat_darkening;
    public float fuzz_weight;
    public float3 fuzz_color;
    public float fuzz_roughness;
    public float thin_film_weight;
    public float thin_film_thickness;   // nm
    public float thin_film_ior;
    public float3 emission;             // any units; `Bsdf.emission` attenuates it by the coat and fuzz
    public float3 scale;                // multiplies every lobe: the directional occlusion, and the layers `bsdf_collapse` removes
    public uint flags;                  // BSDF_*: the lobes that are on and thin-walled geometry
}

// `GSurface.flags`, stored in `a.w`'s top byte.
public static const uint GBUFFER_THIN_TWO_SIDED = 1 << 0;
public static const uint GBUFFER_UNLIT = 1 << 1;
public static const uint GBUFFER_EXTENDED = 1 << 2;
public static const uint GBUFFER_SUBSURFACE = 1 << 3;
public static const uint GBUFFER_LOBE_COLORS = 1 << 4;
public static const uint GBUFFER_ORIGIN_SHIFT = 5;
public static const uint GBUFFER_ORIGIN_MASK = 3 << 5;

// Ray origin policies, in `GSurface.flags` at GBUFFER_ORIGIN_SHIFT.
public static const uint ORIGIN_DEFAULT = 0;
public static const uint ORIGIN_SURFACE = 1;   // raster and traced geometry coincide
public static const uint ORIGIN_TRIANGLE = 2;  // offset from the visible triangle's plane

// One shaded pixel.
public struct GSurface
{
    public BsdfParams bsdf;
    public float3 position;         // render space, any depth offset applied
    public float3 normal;           // unit shading normal, toward the viewer's side
    public float3 geometric_normal; // unit, on the shading normal's side
    public float raster_error;      // m: an upper bound on the raster LOD error; 0 = none
    public uint flags;              // GBUFFER_* and the origin policy
    public uint instance;           // the visible triangle: instance slot,
    public uint cluster;            // mesh cluster index,
    public uint primitive;          // triangle in the cluster
    public uint material;
}

// A cluster the raster drew this view: the visibility buffer names it by its index here.
public struct VisibleCluster
{
    public uint8_t* page;           // its resident page
    public uint instance;
    public uint cluster;            // mesh cluster index
    public uint local;              // index in the page
    public uint material;
}

// Visibility buffer words: reverse-Z f32 depth in the high word, then the visible cluster
// (25 bits) and the triangle (7 bits). 0 = no geometry.
public static const uint VISIBLE_CLUSTER_SHIFT = 7;
public static const uint VISIBLE_CLUSTER_MAX = (1u << 25) - 1;

public uint64_t visibility_word(float depth, uint visible, uint triangle)
{
    return (uint64_t(asuint(depth)) << 32) | uint64_t((visible << VISIBLE_CLUSTER_SHIFT) | triangle);
}

public float visibility_depth(uint64_t w)
{
    return asfloat(uint(w >> 32));
}

public uint visibility_cluster(uint64_t w)
{
    return uint(w) >> VISIBLE_CLUSTER_SHIFT;
}

public uint visibility_triangle(uint64_t w)
{
    return uint(w) & 127;
}

// One view's G-buffer for later passes.
public struct GBufferGpu
{
    public uint64_t* visibility;    // one word per pixel in rows of extent.x
    public VisibleCluster* clusters;
    public TextureIndex depth;      // D32, sampled
    public TextureIndex a;
    public TextureIndex b;
    public TextureIndex x;          // TEXTURE_NONE while no resident material needs it
    public TextureIndex motion;
    public TextureIndex hzb;        // R32F: farthest reverse-Z depth per texel, mip 0 at extent
    public uint2 extent;
    public uint hzb_mips;
    public uint _pad;
}

public struct GBufferWords
{
    public uint4 a;
    public uint2 b;
    public uint4 x;
}

uint unorm8(float v)
{
    return uint(round(saturate(v) * 255.0));
}

float from_unorm8(uint v)
{
    return float(v & 255) / 255.0;
}

uint snorm8(float v)
{
    return uint(int(round(clamp(v, -1.0, 1.0) * 127.0))) & 255;
}

float from_snorm8(uint v)
{
    int s = int(v << 24) >> 24;
    return max(float(s) / 127.0, -1.0);
}

uint rgb8(float3 c)
{
    float3 e = srgb_encode(saturate(c));
    return unorm8(e.x) | (unorm8(e.y) << 8) | (unorm8(e.z) << 16);
}

float3 from_rgb8(uint v)
{
    return srgb_decode(float3(from_unorm8(v), from_unorm8(v >> 8), from_unorm8(v >> 16)));
}

// 8 bits of sqrt(F0) at normal incidence from air: IOR 1 to 3, finest near 1.
uint ior_code(float ior)
{
    float n = max(ior, 1.0);
    return unorm8(2.0 * (n - 1.0) / (n + 1.0));
}

float ior_from_code(uint c)
{
    float r = from_unorm8(c) * 0.5;
    return (1.0 + r) / (1.0 - r);
}

// 5 bits: 0 = none, else the error is at most 2^(e - 21) m.
uint raster_error_code(float error)
{
    if (!(error > 0.0))
        return 0;
    return uint(clamp(ceil(log2(error)) + 21.0, 1.0, 31.0));
}

float raster_error_from_code(uint e)
{
    return e == 0 ? 0.0 : exp2(float(e) - 21.0);
}

// The anisotropy direction is symmetric under a half turn: 8 bits over [0, pi).
uint angle_code(float a)
{
    return uint(round(frac(a / PI) * 256.0)) & 255;
}

float angle_from_code(uint c)
{
    return float(c) / 256.0 * PI;
}

// Whether `s` needs the extended target.
public bool gsurface_extended(GSurface s)
{
    BsdfParams p = s.bsdf;
    return p.subsurface_weight > 0.0 || p.coat_weight > 0.0 || p.fuzz_weight > 0.0 || p.specular_anisotropy > 0.0;
}

// Packs `s`. Emission is stored as given (the resolve pre-exposes it). `extended` false
// drops what only the extended target holds.
public GBufferWords encode_gsurface(GSurface s, bool extended)
{
    BsdfParams p = s.bsdf;
    uint flags = s.flags & (GBUFFER_UNLIT | GBUFFER_ORIGIN_MASK);
    bool thin = (p.flags & BSDF_THIN_WALLED) != 0;
    if (thin)
        flags |= GBUFFER_THIN_TWO_SIDED;
    bool sss = p.subsurface_weight > 0.0;
    if (extended && gsurface_extended(s))
    {
        flags |= GBUFFER_EXTENDED;
        if (sss && !thin)
            flags |= GBUFFER_SUBSURFACE;
        if (!sss && (p.coat_weight > 0.0 || p.fuzz_weight > 0.0))
            flags |= GBUFFER_LOBE_COLORS;
    }
    GBufferWords w;
    w.a.x = rgb8(p.base_color) | (unorm8(p.specular_roughness) << 24);
    w.a.y = oct_encode_snorm16(s.normal);
    w.a.z = pack_rgb9e5(max(p.emission, 0.0));
    w.a.w = unorm8(p.metalness) | (unorm8(p.specular_weight) << 8) | (ior_code(p.specular_ior) << 16) | (flags << 24);
    float3 sc = srgb_encode(saturate(p.specular_color));
    uint3 s7 = uint3(round(sc * 127.0));
    w.b.x = s7.x | (s7.y << 7) | (s7.z << 14) | (raster_error_code(s.raster_error) << 21);
    float occlusion = max(p.scale.x, max(p.scale.y, p.scale.z));
    w.b.y = (oct_encode_snorm8(s.geometric_normal) & 0xffff) | (unorm8(p.diffuse_roughness) << 16) | (unorm8(occlusion) << 24);
    w.x = uint4(0);
    if ((flags & GBUFFER_EXTENDED) != 0)
    {
        if ((flags & GBUFFER_LOBE_COLORS) != 0)
        {
            w.x.x = rgb8(p.coat_color);
            w.x.y = rgb8(p.fuzz_color);
        }
        else
        {
            float3 tint = p.subsurface_weight * p.subsurface_color;
            if (thin)
                tint *= (p.subsurface_anisotropy + 1.0) * 0.5;
            w.x.x = rgb8(tint) | (snorm8(p.subsurface_anisotropy) << 24);
            w.x.y = pack_rgb9e5(max(p.subsurface_radius, 0.0));
        }
        w.x.z = unorm8(p.coat_weight) | (unorm8(p.coat_roughness) << 8) | (ior_code(p.coat_ior) << 16) | (unorm8(p.fuzz_weight) << 24);
        w.x.w = unorm8(p.fuzz_roughness) | (unorm8(p.specular_anisotropy) << 8) | (angle_code(p.tangent_angle) << 16) | (unorm8(p.subsurface_weight) << 24);
    }
    return w;
}

// Unpacks what the words hold. Thin film, coat darkening and, without
// GBUFFER_LOBE_COLORS, the coat and fuzz colors come from `m`, the pixel's material.
// Position and the visible triangle are left 0.
public GSurface decode_gsurface(GBufferWords w, MaterialGpu* m)
{
    GSurface s;
    BsdfParams p;
    uint flags = w.a.w >> 24;
    s.flags = flags;
    p.base_color = from_rgb8(w.a.x);
    p.specular_roughness = from_unorm8(w.a.x >> 24);
    s.normal = oct_decode_snorm16(w.a.y);
    p.emission = unpack_rgb9e5(w.a.z);
    p.metalness = from_unorm8(w.a.w);
    p.specular_weight = from_unorm8(w.a.w >> 8);
    p.specular_ior = ior_from_code(w.a.w >> 16);
    p.specular_color = srgb_decode(float3(uint3(w.b.x, w.b.x >> 7, w.b.x >> 14) & 127) / 127.0);
    s.raster_error = raster_error_from_code((w.b.x >> 21) & 31);
    s.geometric_normal = oct_decode_snorm8(w.b.y & 0xffff);
    p.diffuse_roughness = from_unorm8(w.b.y >> 16);
    p.scale = float3(from_unorm8(w.b.y >> 24));
    p.transmission_weight = 0.0;
    p.transmission_color = float3(1.0);
    bool thin = (flags & GBUFFER_THIN_TWO_SIDED) != 0;
    p.flags = thin ? BSDF_THIN_WALLED : 0;
    p.subsurface_weight = 0.0;
    p.subsurface_color = float3(0.0);
    p.subsurface_anisotropy = 0.0;
    p.subsurface_radius = float3(0.0);
    p.coat_weight = 0.0;
    p.coat_roughness = 0.0;
    p.coat_ior = 1.5;
    p.fuzz_weight = 0.0;
    p.fuzz_roughness = 0.5;
    p.specular_anisotropy = 0.0;
    p.tangent_angle = 0.0;
    p.coat_color = m.coat_color;
    p.fuzz_color = m.fuzz_color;
    p.coat_darkening = 1.0;
    bool film = (m.flags & MATERIAL_THIN_FILM) != 0;
    p.thin_film_weight = film ? 1.0 : 0.0;
    p.thin_film_thickness = m.thin_film_thickness;
    p.thin_film_ior = m.thin_film_ior;
    if (film)
        p.flags |= BSDF_THIN_FILM;
    if ((flags & GBUFFER_EXTENDED) != 0)
    {
        p.coat_weight = from_unorm8(w.x.z);
        p.coat_roughness = from_unorm8(w.x.z >> 8);
        p.coat_ior = ior_from_code(w.x.z >> 16);
        p.fuzz_weight = from_unorm8(w.x.z >> 24);
        p.fuzz_roughness = from_unorm8(w.x.w);
        p.specular_anisotropy = from_unorm8(w.x.w >> 8);
        p.tangent_angle = angle_from_code(w.x.w >> 16);
        p.subsurface_weight = from_unorm8(w.x.w >> 24);
        if ((flags & GBUFFER_LOBE_COLORS) != 0)
        {
            p.coat_color = from_rgb8(w.x.x);
            p.fuzz_color = from_rgb8(w.x.y);
        }
        else
        {
            p.subsurface_anisotropy = from_snorm8(w.x.x >> 24);
            float share = p.subsurface_weight;
            if (thin)
                share *= (p.subsurface_anisotropy + 1.0) * 0.5;
            p.subsurface_color = share > 0.0 ? saturate(from_rgb8(w.x.x) / share) : float3(0.0);
            p.subsurface_radius = unpack_rgb9e5(w.x.y);
        }
        if (p.coat_weight > 0.0)
            p.flags |= BSDF_COAT;
        if (p.fuzz_weight > 0.0)
            p.flags |= BSDF_FUZZ;
        if (p.specular_anisotropy > 0.0)
            p.flags |= BSDF_ANISOTROPY;
    }
    s.bsdf = p;
    s.position = float3(0.0);
    s.instance = 0;
    s.cluster = 0;
    s.primitive = 0;
    s.material = 0;
    return s;
}

// The origin policy of `s`: ORIGIN_*.
public uint gsurface_origin(GSurface s)
{
    return (s.flags & GBUFFER_ORIGIN_MASK) >> GBUFFER_ORIGIN_SHIFT;
}

// `n` turned toward unit `v` until dot(n, v) >= 0.01.
public float3 lean_normal(float3 n, float3 v)
{
    float nv = dot(n, v);
    if (nv >= 0.01)
        return n;
    float3 t = n - nv * v;
    float l = length(t);
    return l > 1e-6 ? 0.01 * v + sqrt(1.0 - 1e-4) * (t / l) : v;
}

// The pixel's surface: render-space position from the depth target, the shading normal
// leaned so that dot(N, V) >= 0.01, the visible triangle and its material. A pixel with no
// geometry (depth 0) returns a surface whose material is MATERIAL_NONE.
public GSurface load_gsurface(SceneRoot* scene, ViewConstants* view, GBufferGpu* g, uint2 pixel)
{
    uint64_t vis = g.visibility[pixel.y * g.extent.x + pixel.x];
    if (vis == 0)
    {
        GSurface none = (GSurface)0;
        none.material = MATERIAL_NONE;
        return none;
    }
    VisibleCluster c = g.clusters[visibility_cluster(vis)];
    GBufferWords w;
    w.a = g.a.get<Texture2D<uint4>>().Load(int3(pixel, 0));
    w.b = g.b.get<Texture2D<uint2>>().Load(int3(pixel, 0));
    w.x = uint4(0);
    uint flags = w.a.w >> 24;
    if ((flags & GBUFFER_EXTENDED) != 0 && g.x.index != TEXTURE_NONE)
        w.x = g.x.get<Texture2D<uint4>>().Load(int3(pixel, 0));
    GSurface s = decode_gsurface(w, scene.materials + c.material);
    float depth = g.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    float2 uv = (float2(pixel) + 0.5) / float2(g.extent);
    s.position = render_from_view(view, view_position(view, uv, depth));
    s.normal = lean_normal(s.normal, normalize(view.eye_render.xyz - s.position));
    s.instance = c.instance;
    s.cluster = c.cluster;
    s.primitive = visibility_triangle(vis);
    s.material = c.material;
    return s;
}
