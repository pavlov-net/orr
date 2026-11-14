// Fixed-point CDFs over producer entries' clusters, built over every cluster of the frame at
// once in three kernels over a layout of the lists in blocks (`orr_scene.producer_blocks`):
// `weigh_blocks`, one workgroup per block, writes each position's sum within its block and what
// the block adds up to; `place_blocks`, one workgroup per entry, places each block after those
// before it; `finish_blocks`, one workgroup per block, writes the CDF.
//
// A position of no weight takes a step of 0, and the last with weight takes 0xffffffff, so a
// list of no weight at all stays at 0. A float scan's lanes add in different orders, so its
// sums can fall by a rounding past a weight of 0 or a tiny one; a running maximum keeps the CDF
// rising, and a position of no weight takes the one before it. The maximum runs over the float
// sums, which the fixed-point conversion keeps in order, so a block's carry from the blocks
// before it is the very sum the position before it took.
module cluster_lists;

import orr_shaders.sampling;
import orr_scene.producer_blocks;

// Producer entries' lists of clusters, by producer entry.
public interface IClusterLists
{
    // Entry `p`'s CDF, and in `count` the positions it holds; 0 for none.
    public uint* list(uint p, out uint count);
    // The weight of position `c` of entry `p`'s list.
    public float weight(uint p, uint c);
}

// What one block of a list adds up to, and where it lies among the list's blocks.
public struct ListBlock
{
    public float sum;            // its weights' sum
    public uint top;             // the largest of its positions' sums that have weight, as bits
    public uint end;             // one past its last position with weight; 0 for none
    public uint _pad0;
    public float offset;         // the sum of the blocks before it in its list
    public float carry;          // the largest position sum of the blocks before it
    // On a list's first block only: the list's sum and one past its last position with weight.
    public float total;
    public uint list_end;
}

// Marks a position of no weight while the CDF holds sums within blocks; the bits are a NaN,
// which no sum of weights is.
static const uint NO_WEIGHT = 0xffffffffu;

groupshared float sum_scan[BLOCK_CLUSTERS];
groupshared uint max_scan[BLOCK_CLUSTERS];
// The workgroup's largest position sum with weight, as bits, and one past its last position
// with weight.
groupshared uint group_top;
groupshared uint group_end;

// Inclusive sums over the workgroup, the sum before the lane in `before` and the workgroup's in
// `total`.
float scan_sum(uint lane, float v, out float before, out float total)
{
    sum_scan[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < BLOCK_CLUSTERS; off <<= 1)
    {
        float add = lane >= off ? sum_scan[lane - off] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        sum_scan[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    float r = sum_scan[lane];
    before = lane > 0 ? sum_scan[lane - 1] : 0.0;
    total = sum_scan[BLOCK_CLUSTERS - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Inclusive maxima over the workgroup, the maximum before the lane in `before` and the
// workgroup's in `total`.
uint scan_max(uint lane, uint v, out uint before, out uint total)
{
    max_scan[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < BLOCK_CLUSTERS; off <<= 1)
    {
        uint other = lane >= off ? max_scan[lane - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        max_scan[lane] = max(max_scan[lane], other);
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = max_scan[lane];
    before = lane > 0 ? max_scan[lane - 1] : 0;
    total = max_scan[BLOCK_CLUSTERS - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Block `b` of `layout`, over `entries` producer entries: each position's weight, its sum
// within the block in the CDF, or NO_WEIGHT, and what the block adds up to.
public void weigh_blocks<L : IClusterLists>(L lists, uint* layout, ListBlock* blocks, uint entries, uint b, uint lane)
{
    uint p, j;
    if (!block_entry(layout, entries, b, p, j))
        return;
    uint count;
    uint* cdf = lists.list(p, count);
    uint c = j * BLOCK_CLUSTERS + lane;
    float w = c < count ? lists.weight(p, c) : 0.0;
    bool weighted = w > 0.0;
    if (lane == 0)
    {
        group_top = 0;
        group_end = 0;
    }
    float before, sum;
    float s = scan_sum(lane, w, before, sum);
    if (c < count)
        cdf[c] = weighted ? asuint(s) : NO_WEIGHT;
    if (weighted)
    {
        // Sums of weights are not negative, so their bits order as they do.
        InterlockedMax(group_top, asuint(s));
        InterlockedMax(group_end, lane + 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0)
    {
        ListBlock k = ListBlock();
        k.sum = sum;
        k.top = group_top;
        k.end = group_end;
        blocks[b] = k;
    }
}

// Producer entry `p`'s blocks of `layout`: each block's offset and carry, and on its first the
// list's sum and end. Returns the list's sum, 0 for a list of no blocks.
public float place_blocks(uint* layout, ListBlock* blocks, uint p, uint lane)
{
    uint first, end;
    entry_blocks(layout, p, first, end);
    float offset = 0.0;
    uint carry = 0;
    if (lane == 0)
        group_end = 0;
    for (uint base = first; base < end; base += BLOCK_CLUSTERS)
    {
        uint b = base + lane;
        ListBlock k = ListBlock();
        if (b < end)
            k = blocks[b];
        float before, chunk;
        scan_sum(lane, k.sum, before, chunk);
        k.offset = offset + before;
        // The block's largest position sum as the CDF takes it, the offset added.
        uint reach = k.end > 0 ? asuint(k.offset + asfloat(k.top)) : 0;
        uint prior, highest;
        scan_max(lane, reach, prior, highest);
        k.carry = asfloat(max(carry, prior));
        if (k.end > 0)
            InterlockedMax(group_end, (b - first) * BLOCK_CLUSTERS + k.end);
        if (b < end)
        {
            blocks[b].offset = k.offset;
            blocks[b].carry = k.carry;
        }
        offset += chunk;
        carry = max(carry, highest);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && first < end)
    {
        blocks[first].total = offset;
        blocks[first].list_end = group_end;
    }
    return offset;
}

// Block `b` of `layout`, over `entries` producer entries, after `place_blocks`: its positions'
// fixed-point CDF.
public void finish_blocks<L : IClusterLists>(L lists, uint* layout, ListBlock* blocks, uint entries, uint b, uint lane)
{
    uint p, j;
    if (!block_entry(layout, entries, b, p, j))
        return;
    uint count;
    uint* cdf = lists.list(p, count);
    ListBlock k = blocks[b];
    uint first, end;
    entry_blocks(layout, p, first, end);
    ListBlock head = blocks[first];
    uint c = j * BLOCK_CLUSTERS + lane;
    uint bits = c < count ? cdf[c] : NO_WEIGHT;
    uint x = bits != NO_WEIGHT ? asuint(k.offset + asfloat(bits)) : 0;
    uint before, top;
    float y = asfloat(max(asuint(k.carry), scan_max(lane, x, before, top)));
    if (c >= count)
        return;
    if (head.list_end > 0 && c + 1 >= head.list_end)
        cdf[c] = 0xffffffffu;
    else
        cdf[c] = y > 0.0 ? fixed_point_share(y / head.total) : 0;
}
