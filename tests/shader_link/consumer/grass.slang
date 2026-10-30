// A surface without coverage: the interface's body applies.
import surface;

struct Grass : ISurface
{
    static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = i.context.depth == 0 ? 0.5 : 0.0;
        return { float4(2.0, float(i.context.kind), float(i.context.depth), i.context.cone_spread) };
    }
}
