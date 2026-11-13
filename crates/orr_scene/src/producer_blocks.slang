// Producer entries' emitted clusters in blocks of BLOCK_CLUSTERS, one workgroup each, for
// kernels that weigh every cluster of a frame at once and gather per entry after. A layout over
// `entries` producer entries is BLOCKS_FIRST + entries + 1 words: the workgroup grid over its
// blocks (`write_grid`), then from BLOCKS_FIRST each entry's first block, the end last. An
// entry's block j holds its clusters from j * BLOCK_CLUSTERS.
module producer_blocks;

import orr_scene.visibility;

public static const uint BLOCK_CLUSTERS = 256;
public static const uint BLOCKS_FIRST = 4;

public uint cluster_blocks(uint clusters)
{
    return (clusters + BLOCK_CLUSTERS - 1) / BLOCK_CLUSTERS;
}

// Sets entry `entry`'s first block.
public void set_first_block(uint* layout, uint entry, uint first)
{
    layout[BLOCKS_FIRST + entry] = first;
}

// Ends a layout over `entries` entries whose first blocks are written: `total` blocks.
public void close_layout(uint* layout, uint entries, uint total)
{
    layout[BLOCKS_FIRST + entries] = total;
    write_grid(layout, total);
}

// The entry block `b` of a layout over `entries` entries holds, and its place among the
// entry's blocks; false past the last block. An entry with no blocks starts where the next
// one does, so the last entry starting at or before `b` holds it.
public bool block_entry(uint* layout, uint entries, uint b, out uint entry, out uint j)
{
    entry = 0;
    j = 0;
    if (b >= layout[BLOCKS_FIRST + entries])
        return false;
    uint lo = 0;
    uint hi = entries - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1) / 2;
        if (layout[BLOCKS_FIRST + mid] <= b)
            lo = mid;
        else
            hi = mid - 1;
    }
    entry = lo;
    j = b - layout[BLOCKS_FIRST + lo];
    return true;
}

// The blocks of entry `entry`: from `first` to before `end`.
public void entry_blocks(uint* layout, uint entry, out uint first, out uint end)
{
    first = layout[BLOCKS_FIRST + entry];
    end = layout[BLOCKS_FIRST + entry + 1];
}
