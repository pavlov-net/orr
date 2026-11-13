// A surface that depends only on where the hit is on the mesh, never on its context or ray.
module steady;

import orr_scene.surface;

public struct Steady : ISurface
{
    public override static bool surface(SurfaceInput i, out GSurface g, out float depth_offset)
    {
        depth_offset = 0.0;
        g = (GSurface)0;
        g.normal = i.normal;
        g.geometric_normal = i.geometric_normal;
        g.bsdf.base_color = frac(float3(i.uv[0].uv * 3.0, dot(i.object_position, float3(0.7, 0.3, 0.5))));
        g.bsdf.specular_roughness = saturate(abs(i.object_normal.y));
        return true;
    }
}
