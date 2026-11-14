// A surface that replaces the standard one and leaves coverage to it.
import orr_scene.surface;

public struct Grass : ISurface
{
    override static bool surface(SurfaceInput i, out GSurface g, out float depth_offset)
    {
        depth_offset = i.context.depth == 0 ? 0.5 : 0.0;
        g = { float4(2.0, float(i.context.kind), float(i.context.depth), i.context.cone_spread) };
        return true;
    }
}
