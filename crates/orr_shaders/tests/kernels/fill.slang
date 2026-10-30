[shader("compute")]
[numthreads(64, 1, 1)]
void fill(uint3 id: SV_DispatchThreadID, uniform uint* out, uniform uint count)
{
    if (id.x < count)
        out[id.x] = id.x * 3 + 1;
}
