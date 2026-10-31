// G-buffer packing: each case's surface through `encode_gsurface` with the extended target,
// the words kept, then `decode_gsurface` back into a case.
#include "orr/root.h"
import gbuffer;
import material;

struct Case
{
    float3 base_color;
    float roughness;
    float3 normal;
    float metalness;
    float3 emission;
    float specular_weight;
    float3 specular_color;
    float ior;
    float3 geometric_normal;
    float diffuse_roughness;
    float3 subsurface_color;
    float occlusion;
    float3 subsurface_radius;
    float raster_error;
    float3 coat_color;
    float coat_weight;
    float3 fuzz_color;
    float coat_roughness;
    float coat_ior;
    float fuzz_weight;
    float fuzz_roughness;
    float anisotropy;
    float tangent_angle;
    float subsurface_weight;
    float subsurface_anisotropy;
    uint flags;        // GBUFFER_UNLIT and the origin policy in, every GSurface flag out
    uint bsdf_flags;   // BSDF_THIN_WALLED in, the decoded BSDF flags out
    uint pad0;
    uint pad1;
    uint pad2;
}

struct Words
{
    uint4 a;
    uint4 x;
    uint2 b;
    uint2 pad;
}

struct Root
{
    Case* cases;
    Words* words;
    Case* decoded;
    MaterialGpu* material;
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
}
ORR_ROOT(Root);

GSurface surface(Case c)
{
    GSurface s = (GSurface)0;
    BsdfParams p = (BsdfParams)0;
    p.base_color = c.base_color;
    p.specular_roughness = c.roughness;
    p.metalness = c.metalness;
    p.emission = c.emission;
    p.specular_weight = c.specular_weight;
    p.specular_color = c.specular_color;
    p.specular_ior = c.ior;
    p.diffuse_roughness = c.diffuse_roughness;
    p.subsurface_color = c.subsurface_color;
    p.scale = float3(c.occlusion);
    p.subsurface_radius = c.subsurface_radius;
    p.coat_color = c.coat_color;
    p.coat_weight = c.coat_weight;
    p.fuzz_color = c.fuzz_color;
    p.coat_roughness = c.coat_roughness;
    p.coat_ior = c.coat_ior;
    p.fuzz_weight = c.fuzz_weight;
    p.fuzz_roughness = c.fuzz_roughness;
    p.specular_anisotropy = c.anisotropy;
    p.tangent_angle = c.tangent_angle;
    p.subsurface_weight = c.subsurface_weight;
    p.subsurface_anisotropy = c.subsurface_anisotropy;
    p.flags = c.bsdf_flags;
    s.bsdf = p;
    s.normal = c.normal;
    s.geometric_normal = c.geometric_normal;
    s.raster_error = c.raster_error;
    s.flags = c.flags;
    return s;
}

Case flatten(GSurface s)
{
    BsdfParams p = s.bsdf;
    Case c = (Case)0;
    c.base_color = p.base_color;
    c.roughness = p.specular_roughness;
    c.metalness = p.metalness;
    c.emission = p.emission;
    c.specular_weight = p.specular_weight;
    c.specular_color = p.specular_color;
    c.ior = p.specular_ior;
    c.diffuse_roughness = p.diffuse_roughness;
    c.subsurface_color = p.subsurface_color;
    c.occlusion = p.scale.x;
    c.subsurface_radius = p.subsurface_radius;
    c.coat_color = p.coat_color;
    c.coat_weight = p.coat_weight;
    c.fuzz_color = p.fuzz_color;
    c.coat_roughness = p.coat_roughness;
    c.coat_ior = p.coat_ior;
    c.fuzz_weight = p.fuzz_weight;
    c.fuzz_roughness = p.fuzz_roughness;
    c.anisotropy = p.specular_anisotropy;
    c.tangent_angle = p.tangent_angle;
    c.subsurface_weight = p.subsurface_weight;
    c.subsurface_anisotropy = p.subsurface_anisotropy;
    c.normal = s.normal;
    c.geometric_normal = s.geometric_normal;
    c.raster_error = s.raster_error;
    c.flags = s.flags;
    c.bsdf_flags = p.flags;
    return c;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void round_trip(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    GBufferWords w = encode_gsurface(surface(root.cases[id.x]), true);
    Words o;
    o.a = w.a;
    o.x = w.x;
    o.b = w.b;
    o.pad = uint2(0);
    root.words[id.x] = o;
    root.decoded[id.x] = flatten(decode_gsurface(w, root.material));
}
