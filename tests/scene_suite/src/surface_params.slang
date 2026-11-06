// `StandardSurface` at one flat, facing point of each material, its parameters kept.
#include "orr_root.h"
import orr_scene.surface;

struct Params
{
    float3 base_color;
    float metalness;
    float3 specular_color;
    float specular_weight;
    float3 transmission_color;
    float specular_ior;
    float3 coat_color;
    float coat_roughness;
    float3 coat_normal;
    float coat_ior;
    float3 fuzz_color;
    float fuzz_weight;
    float3 subsurface_radius;
    float fuzz_roughness;
    float3 emission;
    float thin_film_weight;
    float thin_film_thickness;
    float thin_film_ior;
    float diffuse_roughness;
    float subsurface_weight;
    float subsurface_anisotropy;
    float coat_darkening;
    float coat_anisotropy;
    float specular_anisotropy;
    float tangent_angle;
    float coat_tangent_angle;
    uint flags;
    uint pad;
}

struct Root
{
    SceneRoot* scene;
    Params* out;
    float4 vertex_color;
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    SurfaceInput i = (SurfaceInput)0;
    UvFootprint f = { float2(0.25), float2(0.0), float2(0.0), float2(-1.0), float2(0.0) };
    i.uv[0] = f;
    i.uv[1] = f;
    i.normal = float3(0.0, 0.0, 1.0);
    i.frame_normal = i.normal;
    i.geometric_normal = i.normal;
    i.tangent = float4(1.0, 0.0, 0.0, 1.0);
    i.vertex_color = root.vertex_color;
    i.material = id.x;
    i.scene = root.scene;
    i.view_direction = float3(0.0, 0.0, -1.0);
    float offset;
    GSurface s = StandardSurface.surface(i, offset);
    BsdfParams p = s.bsdf;
    Params o = (Params)0;
    o.base_color = p.base_color;
    o.metalness = p.metalness;
    o.specular_color = p.specular_color;
    o.specular_weight = p.specular_weight;
    o.transmission_color = p.transmission_color;
    o.specular_ior = p.specular_ior;
    o.coat_color = p.coat_color;
    o.coat_roughness = p.coat_roughness;
    o.coat_normal = p.coat_normal;
    o.coat_ior = p.coat_ior;
    o.fuzz_color = p.fuzz_color;
    o.fuzz_weight = p.fuzz_weight;
    o.subsurface_radius = p.subsurface_radius;
    o.fuzz_roughness = p.fuzz_roughness;
    o.emission = p.emission;
    o.thin_film_weight = p.thin_film_weight;
    o.thin_film_thickness = p.thin_film_thickness;
    o.thin_film_ior = p.thin_film_ior;
    o.diffuse_roughness = p.diffuse_roughness;
    o.subsurface_weight = p.subsurface_weight;
    o.subsurface_anisotropy = p.subsurface_anisotropy;
    o.coat_darkening = p.coat_darkening;
    o.coat_anisotropy = p.coat_anisotropy;
    o.specular_anisotropy = p.specular_anisotropy;
    o.tangent_angle = p.tangent_angle;
    o.coat_tangent_angle = p.coat_tangent_angle;
    o.flags = p.flags;
    root.out[id.x] = o;
}
