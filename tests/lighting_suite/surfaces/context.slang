// An unlit surface that shows its hit's context: red at path depths 0 and 1, green from 2 on,
// and blue where the camera sees the hit (`camera_sees`).
module context;

import orr_scene.surface;

public static const float3 NEAR = float3(40.0, 4.0, 4.0);
public static const float3 FAR = float3(4.0, 40.0, 4.0);
public static const float3 SEEN = float3(0.0, 0.0, 36.0);

public struct Context : ISurface
{
    public override static bool surface(SurfaceInput i, out PlainSurface s, out float depth_offset)
    {
        depth_offset = 0.0;
        s = PlainSurface();
        s.normal = i.normal;
        s.geometric_normal = i.geometric_normal;
        s.specular_ior = 1.5;
        return true;
    }

    public override static void emission(SurfaceInput i, PlainSurface s, inout EmissionLobe lobe)
    {
        lobe.emission = (i.context.depth < 2 ? NEAR : FAR) + (camera_sees(i.context.kind) ? SEEN : float3(0.0));
        lobe.unlit = true;
    }

    public override static void always_shades()
    {
    }
}
