// The G-buffer: what the material resolve writes per pixel, its packing, and
// `load_gsurface`, which is how every later pass reads it.
//
// Core targets: `a` RGBA32_UINT, `b` RG32_UINT, depth D32 (reverse Z, 0 = no geometry) and
// motion RG16F (previous uv minus current uv, unjittered). The extended target `x`
// (RGBA32_UINT) holds subsurface and lobe scalars where `GBUFFER_EXTENDED` is set, and `y`
// (RGBA32_UINT) the coat, fuzz and thin film beside them where `GBUFFER_LAYERS` is.
module gbuffer;

import orr_shaders.abi;
import orr_color.transfer;
import orr_shaders.math;
import orr_shaders.packing;
import orr_scene.material;
import orr_scene.view;

// `BsdfParams.flags`.
public static const uint BSDF_THIN_WALLED = 1 << 0;
public static const uint BSDF_COAT = 1 << 1;
public static const uint BSDF_FUZZ = 1 << 2;
public static const uint BSDF_THIN_FILM = 1 << 3;
public static const uint BSDF_ANISOTROPY = 1 << 4;
// Set by `bsdf_collapse`.
public static const uint BSDF_COLLAPSED = 1 << 5;
// The coat has a frame of its own: `coat_normal` and `coat_tangent_angle`.
public static const uint BSDF_COAT_FRAME = 1 << 6;
// `specular_color` is F0 at normal incidence, `specular_weight` the reflection at grazing
// incidence and what scales F0 with it, Schlick's curve lies between them, and the layer below
// keeps what the largest channel leaves. `specular_ior` is only the angle of refraction and what
// reflects from inside. Neither reaches a metal, and the coat leaves the base as it is.
public static const uint BSDF_FRESNEL_MIX = 1 << 7;
// A metal reflects its base color at every angle.
public static const uint BSDF_METAL_FLAT = 1 << 8;

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
    public float coat_anisotropy;
    public float coat_tangent_angle;    // rad: as `tangent_angle`, from orthonormal_basis(coat_normal)'s tangent
    public float3 coat_normal;          // unit, with BSDF_COAT_FRAME
    public float fuzz_weight;
    public float3 fuzz_color;
    public float fuzz_roughness;
    public float thin_film_weight;
    public float thin_film_thickness;   // nm
    public float thin_film_ior;
    public float3 emission;             // any units; `Bsdf.emission` attenuates it by the coat and fuzz
    public uint flags;                  // BSDF_*: the lobes that are on and thin-walled geometry
}

// `GSurface.flags`, stored in `a.w`'s top byte.
public static const uint GBUFFER_THIN_TWO_SIDED = 1 << 0;
public static const uint GBUFFER_UNLIT = 1 << 1;
public static const uint GBUFFER_EXTENDED = 1 << 2;
public static const uint GBUFFER_SUBSURFACE = 1 << 3;
public static const uint GBUFFER_LAYERS = 1 << 4;
public static const uint GBUFFER_ORIGIN_SHIFT = 5;
public static const uint GBUFFER_ORIGIN_MASK = 3 << 5;
public static const uint GBUFFER_FRESNEL_MIX = 1 << 7;

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
    public uint cluster;            // mesh cluster index, NO_CLUSTER on a drawn triangle,
    public uint primitive;          // triangle in the cluster
    public uint material;
    public float4 medium_point;     // uv sets 0 and 1, where a medium entered here reads: `material_medium`
}

// A cluster the raster drew this view: the visibility buffer names it by its index here.
public struct VisibleCluster
{
    public uint8_t* page;           // its resident page
    public uint instance;
    public uint cluster;            // mesh cluster index; a drawn record's word
    public uint local;              // index in the page; a producer cluster's VISIBLE_* flags
    public uint material;
}

// `VisibleCluster.local` of producer clusters: both raster paths draw the cluster, the
// software raster its triangles whose screen rect fits `software_limit` and the hardware
// path the others.
public static const uint VISIBLE_BY_SIZE = 1;
// A drawn entry's record (`orr_scene.drawn_raster`): no cluster of its batch.
public static const uint VISIBLE_DRAWN = 2;

// `GSurface.cluster` of a drawn triangle, which no cluster holds.
public static const uint NO_CLUSTER = 0xffffffffu;

// Whether `vc` is a drawn entry's record.
public bool visible_drawn(VisibleCluster vc)
{
    return vc.page == nullptr && (vc.local & VISIBLE_DRAWN) != 0;
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
    public TextureIndex x;          // IMAGE_NONE while no resident material needs it
    public TextureIndex y;          // with `x`
    public TextureIndex motion;
    public TextureIndex hzb;        // R32F: farthest reverse-Z depth per texel, mip 0 at extent
    public uint2 extent;
    public uint hzb_mips;
    // Per LAYER_TILE pixel tile in rows of extent.x / LAYER_TILE rounded up: nonzero where a
    // dielectric or presence cluster rastered in front of the opaque depth. Then one word,
    // nonzero when such a cluster may lie between the eye and the near plane. Null without
    // the layer raster.
    public uint* layer_tiles;
}

// Pixels per side of a `GBufferGpu.layer_tiles` tile.
public static const uint LAYER_TILE = 8;

public uint layer_tile_columns(uint2 extent)
{
    return (extent.x + LAYER_TILE - 1) / LAYER_TILE;
}

public uint layer_tile_count(uint2 extent)
{
    return layer_tile_columns(extent) * ((extent.y + LAYER_TILE - 1) / LAYER_TILE);
}

// The index of the tile holding `pixel`.
public uint layer_tile(uint2 extent, uint2 pixel)
{
    return (pixel.y / LAYER_TILE) * layer_tile_columns(extent) + pixel.x / LAYER_TILE;
}

public struct GBufferWords
{
    public uint4 a;
    public uint2 b;
    public uint4 x;
    public uint4 y;
}

uint rgb8(float3 c)
{
    float3 e = srgb_encode(saturate(c));
    return pack_unorm8(e.x) | (pack_unorm8(e.y) << 8) | (pack_unorm8(e.z) << 16);
}

float3 from_rgb8(uint v)
{
    return srgb_decode(float3(unpack_unorm8(v), unpack_unorm8(v >> 8), unpack_unorm8(v >> 16)));
}

// 8 bits of t = (n - 1) / (n + 1), sqrt(F0) from air with its sign, over every IOR: Fresnel
// reaches 1 toward both 0 and infinity. Codes 16 to 201 hold IOR 1 to 3, finest, and the rest
// IOR 0 to 1 and 3 up. The steps put IOR 1.5, the usual default, exactly on code 90.
uint ior_code(float ior)
{
    float t = 1.0 - 2.0 / (max(ior, 0.0) + 1.0);
    return uint(round(t < 0.0 ? 16.0 * (t + 1.0) : t < 0.5 ? 16.0 + 370.0 * t : 201.0 + 108.0 * (t - 0.5))) & 255;
}

float ior_from_code(uint code)
{
    float c = float(code & 255);
    float t = c < 16.0 ? c / 16.0 - 1.0 : c < 201.0 ? (c - 16.0) / 370.0 : 0.5 + (c - 201.0) / 108.0;
    return (1.0 + t) / max(1.0 - t, 1e-4);
}

// The power of two a specular weight stores under: OpenPBR limits it only by 1 / F0. A weight
// up to 1 has exponent 0.
uint weight_exponent(float weight)
{
    return weight > 1.0 ? uint(min(ceil(log2(weight)), 31.0)) : 0;
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
    return float(c & 255) / 256.0 * PI;
}

// 12 bits per axis of `n` in octahedral form, codes 1 to 4095 for -1 to 1 with 0 on 2048, so
// every axis direction is exact. No axis takes code 0, which leaves 0 to say there is no
// such normal.
uint oct_code(float3 n)
{
    uint2 q = uint2(int2(round(oct_wrap(n) * 2047.0)) + 2048);
    return q.x | (q.y << 12);
}

float3 oct_from_code(uint c)
{
    return oct_unwrap((float2(c & 4095, (c >> 12) & 4095) - 2048.0) / 2047.0);
}

// 8 bits of a GGX roughness's square root, so that steps in alpha = roughness^2 shrink with
// it where the lobe is sharp.
uint roughness_code(float r)
{
    return pack_unorm8(sqrt(saturate(r)));
}

float roughness_from_code(uint c)
{
    float s = unpack_unorm8(c);
    return s * s;
}

// Whether `s` holds a coat, fuzz or thin film.
bool gsurface_layered(GSurface s)
{
    BsdfParams p = s.bsdf;
    return p.coat_weight > 0.0 || p.fuzz_weight > 0.0 || p.thin_film_weight > 0.0;
}

// Whether `s` needs the extended targets.
public bool gsurface_extended(GSurface s)
{
    BsdfParams p = s.bsdf;
    return p.subsurface_weight > 0.0 || gsurface_layered(s) || p.specular_anisotropy > 0.0;
}

// Packs `s`. Emission is stored as given (the resolve pre-exposes it). `extended` false
// drops what only the extended targets hold.
public GBufferWords encode_gsurface(GSurface s, bool extended)
{
    BsdfParams p = s.bsdf;
    uint flags = s.flags & (GBUFFER_UNLIT | GBUFFER_ORIGIN_MASK);
    bool thin = (p.flags & BSDF_THIN_WALLED) != 0;
    if (thin)
        flags |= GBUFFER_THIN_TWO_SIDED;
    bool mix = (p.flags & BSDF_FRESNEL_MIX) != 0;
    if (mix)
        flags |= GBUFFER_FRESNEL_MIX;
    bool sss = p.subsurface_weight > 0.0;
    if (extended && gsurface_extended(s))
    {
        flags |= GBUFFER_EXTENDED;
        if (sss && !thin)
            flags |= GBUFFER_SUBSURFACE;
        if (gsurface_layered(s))
            flags |= GBUFFER_LAYERS;
    }
    GBufferWords w;
    w.a.x = rgb8(p.base_color) | (roughness_code(p.specular_roughness) << 24);
    w.a.y = oct_encode_snorm16(s.normal);
    w.a.z = pack_rgb9e5(max(p.emission, 0.0));
    uint exponent = weight_exponent(p.specular_weight);
    w.a.w = pack_unorm8(p.metalness) | (pack_unorm8(p.specular_weight * exp2(-float(exponent))) << 8) | (ior_code(p.specular_ior) << 16) | (flags << 24);
    // F0 keeps its size in its own byte, so seven bits hold its hue where they have steps.
    float peak = mix ? max_component(p.specular_color) : 1.0;
    float3 hue = peak > 0.0 ? p.specular_color / peak : float3(1.0);
    float3 sc = srgb_encode(saturate(hue));
    uint3 s7 = uint3(round(sc * 127.0));
    uint flat = (p.flags & BSDF_METAL_FLAT) != 0 ? 1u << 31 : 0;
    w.b.x = s7.x | (s7.y << 7) | (s7.z << 14) | (raster_error_code(s.raster_error) << 21) | (exponent << 26) | flat;
    w.b.y = (oct_encode_snorm8(s.geometric_normal) & 0xffff) | (pack_unorm8(p.diffuse_roughness) << 16);
    if (mix)
        w.b.y |= pack_unorm8(sqrt(peak)) << 24;
    w.x = uint4(0);
    w.y = uint4(0);
    if ((flags & GBUFFER_EXTENDED) != 0)
    {
        float3 tint = p.subsurface_weight * p.subsurface_color;
        if (thin)
            tint *= (p.subsurface_anisotropy + 1.0) * 0.5;
        w.x.x = rgb8(tint) | (pack_snorm8(p.subsurface_anisotropy) << 24);
        w.x.y = pack_rgb9e5(max(p.subsurface_radius, 0.0));
        w.x.z = pack_unorm8(p.coat_weight) | (roughness_code(p.coat_roughness) << 8) | (ior_code(p.coat_ior) << 16) | (pack_unorm8(p.fuzz_weight) << 24);
        w.x.w = pack_unorm8(p.fuzz_roughness) | (pack_unorm8(p.specular_anisotropy) << 8) | (angle_code(p.tangent_angle) << 16) | (pack_unorm8(p.subsurface_weight) << 24);
    }
    if ((flags & GBUFFER_LAYERS) != 0)
    {
        w.y.x = rgb8(p.coat_color) | (pack_unorm8(p.coat_darkening) << 24);
        w.y.y = rgb8(p.fuzz_color) | (pack_unorm8(p.coat_anisotropy) << 24);
        if ((p.flags & BSDF_COAT_FRAME) != 0)
            w.y.z = oct_code(p.coat_normal) | (angle_code(p.coat_tangent_angle) << 24);
        // The top half of a float, which holds any thickness a film can state.
        uint thickness = pack_top16(p.thin_film_thickness);
        w.y.w = pack_unorm8(p.thin_film_weight) | (ior_code(p.thin_film_ior) << 8) | (thickness << 16);
    }
    return w;
}

// Unpacks what the words hold. Position, the visible triangle and the material are left 0.
public GSurface decode_gsurface(GBufferWords w)
{
    GSurface s;
    BsdfParams p;
    uint flags = w.a.w >> 24;
    s.flags = flags;
    p.base_color = from_rgb8(w.a.x);
    p.specular_roughness = roughness_from_code(w.a.x >> 24);
    s.normal = oct_decode_snorm16(w.a.y);
    p.emission = unpack_rgb9e5(w.a.z);
    p.metalness = unpack_unorm8(w.a.w);
    p.specular_weight = unpack_unorm8(w.a.w >> 8) * exp2(float((w.b.x >> 26) & 31));
    p.specular_ior = ior_from_code(w.a.w >> 16);
    p.specular_color = srgb_decode(float3(uint3(w.b.x, w.b.x >> 7, w.b.x >> 14) & 127) / 127.0);
    s.raster_error = raster_error_from_code((w.b.x >> 21) & 31);
    s.geometric_normal = oct_decode_snorm8(w.b.y & 0xffff);
    p.diffuse_roughness = unpack_unorm8(w.b.y >> 16);
    if ((flags & GBUFFER_FRESNEL_MIX) != 0)
    {
        float root = unpack_unorm8(w.b.y >> 24);
        p.specular_color *= root * root;
    }
    p.transmission_weight = 0.0;
    p.transmission_color = float3(1.0);
    bool thin = (flags & GBUFFER_THIN_TWO_SIDED) != 0;
    p.flags = thin ? BSDF_THIN_WALLED : 0;
    if ((flags & GBUFFER_FRESNEL_MIX) != 0)
        p.flags |= BSDF_FRESNEL_MIX;
    if ((w.b.x >> 31) != 0)
        p.flags |= BSDF_METAL_FLAT;
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
    p.coat_color = float3(1.0);
    p.fuzz_color = float3(1.0);
    p.coat_darkening = 1.0;
    p.coat_anisotropy = 0.0;
    p.coat_tangent_angle = 0.0;
    p.coat_normal = s.normal;
    p.thin_film_weight = 0.0;
    p.thin_film_thickness = 0.0;
    p.thin_film_ior = 1.4;
    if ((flags & GBUFFER_EXTENDED) != 0)
    {
        p.coat_weight = unpack_unorm8(w.x.z);
        p.coat_roughness = roughness_from_code(w.x.z >> 8);
        p.coat_ior = ior_from_code(w.x.z >> 16);
        p.fuzz_weight = unpack_unorm8(w.x.z >> 24);
        p.fuzz_roughness = unpack_unorm8(w.x.w);
        p.specular_anisotropy = unpack_unorm8(w.x.w >> 8);
        p.tangent_angle = angle_from_code(w.x.w >> 16);
        p.subsurface_weight = unpack_unorm8(w.x.w >> 24);
        p.subsurface_anisotropy = unpack_snorm8(w.x.x >> 24);
        float share = p.subsurface_weight;
        if (thin)
            share *= (p.subsurface_anisotropy + 1.0) * 0.5;
        p.subsurface_color = share > 0.0 ? saturate(from_rgb8(w.x.x) / share) : float3(0.0);
        p.subsurface_radius = unpack_rgb9e5(w.x.y);
        if (p.coat_weight > 0.0)
            p.flags |= BSDF_COAT;
        if (p.fuzz_weight > 0.0)
            p.flags |= BSDF_FUZZ;
        if (p.specular_anisotropy > 0.0)
            p.flags |= BSDF_ANISOTROPY;
    }
    if ((flags & GBUFFER_LAYERS) != 0)
    {
        p.coat_color = from_rgb8(w.y.x);
        p.coat_darkening = unpack_unorm8(w.y.x >> 24);
        p.fuzz_color = from_rgb8(w.y.y);
        p.coat_anisotropy = unpack_unorm8(w.y.y >> 24);
        if ((w.y.z & 0xffffff) != 0)
        {
            p.coat_normal = oct_from_code(w.y.z);
            p.coat_tangent_angle = angle_from_code(w.y.z >> 24);
            p.flags |= BSDF_COAT_FRAME;
        }
        p.thin_film_weight = unpack_unorm8(w.y.w);
        p.thin_film_ior = ior_from_code(w.y.w >> 8);
        p.thin_film_thickness = unpack_top16(w.y.w >> 16);
        if (p.thin_film_weight > 0.0)
            p.flags |= BSDF_THIN_FILM;
    }
    s.bsdf = p;
    s.position = float3(0.0);
    s.instance = 0;
    s.cluster = 0;
    s.primitive = 0;
    s.material = 0;
    s.medium_point = NO_MEDIUM_POINT;
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
public GSurface load_gsurface(ViewConstants* view, GBufferGpu* g, uint2 pixel)
{
    uint64_t vis = g.visibility[pixel.y * g.extent.x + pixel.x];
    if (vis == 0)
    {
        GSurface none = GSurface();
        none.material = MATERIAL_NONE;
        return none;
    }
    VisibleCluster c = g.clusters[visibility_cluster(vis)];
    GBufferWords w;
    w.a = g.a.get<Texture2D<uint4>>().Load(int3(pixel, 0));
    w.b = g.b.get<Texture2D<uint2>>().Load(int3(pixel, 0));
    w.x = uint4(0);
    w.y = uint4(0);
    uint flags = w.a.w >> 24;
    if ((flags & GBUFFER_EXTENDED) != 0 && g.x.index != IMAGE_NONE)
        w.x = g.x.get<Texture2D<uint4>>().Load(int3(pixel, 0));
    if ((flags & GBUFFER_LAYERS) != 0 && g.y.index != IMAGE_NONE)
        w.y = g.y.get<Texture2D<uint4>>().Load(int3(pixel, 0));
    GSurface s = decode_gsurface(w);
    float depth = g.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    float2 uv = (float2(pixel) + 0.5) / float2(g.extent);
    s.position = render_from_view(view, view_position(view, uv, depth));
    s.normal = lean_normal(s.normal, normalize(view.eye_render.xyz - s.position));
    s.instance = c.instance;
    s.cluster = visible_drawn(c) ? NO_CLUSTER : c.cluster;
    s.primitive = visibility_triangle(vis);
    s.material = c.material;
    return s;
}
