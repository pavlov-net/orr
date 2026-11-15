// A surface that replaces the standard one, has its own coverage and echoes its hit context.
import orr_scene.surface;

public struct GroundBlend : ISurface
{
    override static bool surface(SurfaceInput i, out PlainSurface s, out float depth_offset)
    {
        depth_offset = 0.125;
        s = { float4(1.0, float(i.context.kind), float(i.context.depth), i.context.cone_width) };
        return true;
    }

    override static bool coverage(SurfaceInput i, out float coverage)
    {
        coverage = i.context.kind == HIT_PRIMARY ? 0.75 : 0.5;
        return true;
    }

    override static uint coverage_reads(SurfaceInput i)
    {
        return 0;
    }
}
