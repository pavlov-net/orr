// An unlit surface that shows its hit's path depth: red at depths 0 and 1, green from 2 on.
module context;

import orr_scene.surface;

public static const float3 NEAR = float3(40.0, 4.0, 4.0);
public static const float3 FAR = float3(4.0, 40.0, 4.0);

public struct Context : ISurface
{
    public override static bool surface(SurfaceInput i, out GSurface g, out float depth_offset)
    {
        depth_offset = 0.0;
        g = (GSurface)0;
        g.normal = i.normal;
        g.geometric_normal = i.geometric_normal;
        g.bsdf.specular_ior = 1.5;
        g.bsdf.coat_normal = i.normal;
        g.bsdf.emission = i.context.depth < 2 ? NEAR : FAR;
        g.flags = GBUFFER_UNLIT;
        return true;
    }
}
