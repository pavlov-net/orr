// A surface that overrides coverage and echoes its hit context.
import surface;

struct GroundBlend : ISurface
{
    static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.125;
        return { float4(1.0, float(i.context.kind), float(i.context.depth), i.context.cone_width) };
    }

    override static float coverage(SurfaceInput i)
    {
        return i.context.kind == HIT_PRIMARY ? 0.75 : 0.5;
    }
}
