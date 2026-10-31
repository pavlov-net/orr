// A surface that depends only on where the hit is on the mesh, never on its context or ray.
module steady;

import surface;

public struct Steady : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.0;
        GSurface s = (GSurface)0;
        s.normal = i.normal;
        s.geometric_normal = i.geometric_normal;
        s.bsdf.base_color = frac(float3(i.uv.uv * 3.0, dot(i.object_position, float3(0.7, 0.3, 0.5))));
        s.bsdf.specular_roughness = saturate(abs(i.object_normal.y));
        return s;
    }
}
