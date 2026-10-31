// Writes indirect copy lists: page `i` of `dst_pages` from page `page_source[i]` of
// `src_pages`, and tile `i` (x, y, mip, staging slot in `tiles`) from its staging slot.
// Page entries past the page count copy nothing and name the guard page after the last.
#include "orr/root.h"
import copy;

struct Root
{
    uint* page_source;
    uint* tiles;
    uint64_t src_pages;
    uint64_t dst_pages;
    uint64_t page_list;
    uint64_t staging;
    uint64_t tile_list;
    uint pages;
    uint page_entries;
    uint page_bytes;
    uint tile_count;
    uint tile_texels;
    uint tile_bytes;
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void pages(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.page_entries)
        return;
    MemoryCopy c;
    c.src = root.src_pages;
    c.dst = root.dst_pages + uint64_t(root.pages) * root.page_bytes;
    c.bytes = 0;
    if (i < root.pages)
    {
        c.src += uint64_t(root.page_source[i]) * root.page_bytes;
        c.dst = root.dst_pages + uint64_t(i) * root.page_bytes;
        c.bytes = root.page_bytes;
    }
    store(root.page_list, i, c);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tiles(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.tile_count)
        return;
    uint* t = root.tiles + i * 4;
    TextureCopy c;
    c.src = root.staging + uint64_t(t[3]) * root.tile_bytes;
    c.row_texels = root.tile_texels;
    c.mip = t[2];
    c.base_layer = 0;
    c.layers = 1;
    c.offset = uint3(t[0] * root.tile_texels, t[1] * root.tile_texels, 0);
    c.extent = uint3(root.tile_texels, root.tile_texels, 1);
    store(root.tile_list, i, c);
}
