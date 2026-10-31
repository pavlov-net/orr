// Stored mesh pages (the form orr_pack::mesh::encode_page writes, zstd removed) decoded into
// the resident layout on the GPU, byte for byte what orr_pack::mesh::decode_page writes.
//
// A batch of pages takes two passes with a barrier between them. `page_copy`, one thread
// per resident dword, copies the header, the cluster records and the micromap region and
// zeroes the vertex and triangle regions. `page_decode_cluster`, one workgroup of
// PAGE_CODEC_THREADS threads per cluster, decodes the cluster's vertex stream
// (meshoptimizer's vertex codec, version 1) and its triangle stream (the meshlet
// codec) into its slots. Both return 0 or PAGE_CODEC_* bits; a page that sets any bit in
// either pass is malformed, and the thread or workgroup that found it writes nothing.
//
// The decode keeps its working set in groupshared memory (about 31 KiB) declared here, so
// a kernel that calls `page_decode_cluster` holds it; the module has no bindings.
module page_codec;

import orr_pack.mesh;

public static const uint PAGE_CODEC_THREADS = 256;

// Header, regions, stream table, lengths or the resident capacity.
public static const uint PAGE_CODEC_BAD_PAGE = 1;
// A cluster's slots or stream offsets.
public static const uint PAGE_CODEC_BAD_CLUSTER = 2;
public static const uint PAGE_CODEC_BAD_VERTICES = 4;
// Triangle stream, or a corner outside the cluster.
public static const uint PAGE_CODEC_BAD_TRIANGLES = 8;

// orr_pack::mesh::StoredPage.
public struct StoredPage
{
    public uint8_t* stored;   // 4-aligned
    public uint8_t* resident; // 4-aligned
    public uint stored_len;
    public uint resident_capacity;
}

static const uint HEADER_BYTES = 96;
static const uint CLUSTER_BYTES = 128;
static const uint MAX_VERTICES = 256;
static const uint MAX_VERTEX_BYTES = 8 + 20;
static const uint MAX_CHANNELS = MAX_VERTEX_BYTES / 4;
static const uint MAX_GROUPS = MAX_VERTICES / 16;

// The longest stream either decoder accepts: every byte column as 16 groups of 4-bit values
// that all escape, or every triangle with three extra corners, the odd count's pad included.
static const uint VERTEX_STREAM_MAX = 1 + MAX_CHANNELS + MAX_VERTEX_BYTES * (4 + MAX_GROUPS * 24) + 35;
static const uint TRIANGLE_STREAM_MAX = 256 * 3 + 128 + 16;
static const uint VERTEX_STREAM_WORDS = (VERTEX_STREAM_MAX + 3) / 4;
static const uint TRIANGLE_STREAM_WORDS = (TRIANGLE_STREAM_MAX + 3) / 4;

// Column descriptors from the stream walk.
static const uint COLUMN_ZERO = 0;
static const uint COLUMN_LITERAL = 1;
static const uint COLUMN_GROUPS = 2;

// Triangle corner slots: a corner value, or SLOT_POINTER | the slot it repeats.
static const uint SLOT_POINTER = 0x80000000;

// First byte of a vertex stream: the codec's 0xA0 and version 1.
static const uint VERTEX_CODEC_HEADER = 0xA1;

groupshared uint codec_vstream[VERTEX_STREAM_WORDS];
groupshared uint codec_tstream[TRIANGLE_STREAM_WORDS];
groupshared uint codec_columns[MAX_VERTEX_BYTES];               // kind << 16 | stream offset
groupshared uint codec_groups[MAX_VERTEX_BYTES * MAX_GROUPS];   // bits << 16 | stream offset
groupshared uint codec_bytes[MAX_VERTEX_BYTES * MAX_VERTICES / 4];
groupshared uint codec_channels[MAX_CHANNELS * MAX_VERTICES];
groupshared uint codec_prefix[MAX_VERTICES];                     // extra | next << 16
groupshared uint codec_slots[3 * MAX_VERTICES];
groupshared uint codec_status;

struct PageCheck
{
    PageHeader h;
    uint status;
    uint slots;
    uint tris_len;
    uint streams_end;
}

uint load_u32(uint8_t* p, uint at)
{
    return uint(p[at]) | uint(p[at + 1]) << 8 | uint(p[at + 2]) << 16 | uint(p[at + 3]) << 24;
}

PageCheck check_page(StoredPage p)
{
    PageCheck k;
    k.status = PAGE_CODEC_BAD_PAGE;
    k.slots = 0;
    k.tris_len = 0;
    k.streams_end = 0;
    if (p.stored_len < HEADER_BYTES)
        return k;
    PageHeader h = load_page_header(p.stored);
    k.h = h;
    if (h.cluster_count > (PAGE_MAX - HEADER_BYTES) / CLUSTER_BYTES)
        return k;
    uint n = h.cluster_count;
    bool regions = h.positions_offset == HEADER_BYTES + n * CLUSTER_BYTES
        && h.attrs_offset >= h.positions_offset && h.tris_offset >= h.attrs_offset
        && h.omm_offset >= h.tris_offset && h.len >= h.omm_offset && h.len <= PAGE_MAX
        && h.attr_stride >= 8 && h.attr_stride % 4 == 0 && h.attrs_offset % 4 == 0
        && h.omm_offset % 4 == 0;
    if (!regions)
        return k;
    uint vertex_bytes = h.attrs_offset - h.positions_offset;
    uint slots = vertex_bytes / 6;
    if (slots * 6 != vertex_bytes
        || uint64_t(slots) * uint64_t(h.attr_stride) != uint64_t(h.tris_offset - h.attrs_offset))
        return k;
    if (h.len > p.resident_capacity)
        return k;
    uint table_end = h.positions_offset + (2 * n + 1) * 4;
    if (p.stored_len < table_end)
        return k;
    uint* table = (uint*)(p.stored + h.positions_offset);
    uint streams_end = table[2 * n];
    if (table[0] != table_end || streams_end > p.stored_len
        || p.stored_len - streams_end != h.len - h.omm_offset)
        return k;
    k.status = 0;
    k.slots = slots;
    k.tris_len = h.omm_offset - h.tris_offset;
    k.streams_end = streams_end;
    return k;
}

// Writes resident dword `dword` of the page: records and micromaps copied, the regions
// between them zeroed. Threads past the page's length return 0.
public uint page_copy(StoredPage p, uint dword)
{
    PageCheck k = check_page(p);
    if (k.status != 0)
        return k.status;
    PageHeader h = k.h;
    uint at = dword * 4;
    if (at >= h.len)
        return 0;
    uint v = 0;
    if (at < h.positions_offset)
        v = ((uint*)p.stored)[dword];
    else if (at >= h.omm_offset)
    {
        uint from = k.streams_end + at - h.omm_offset;
        for (uint j = 0; j < 4 && at + j < h.len; j++)
            v |= uint(p.stored[from + j]) << (8 * j);
    }
    if (at + 4 <= h.len)
        ((uint*)p.resident)[dword] = v;
    else
    {
        for (uint j = 0; at + j < h.len; j++)
            p.resident[at + j] = uint8_t(v >> (8 * j));
    }
    return 0;
}

uint vbyte(uint at)
{
    return (codec_vstream[min(at >> 2, VERTEX_STREAM_WORDS - 1)] >> (8 * (at & 3))) & 0xFF;
}

uint vword(uint at)
{
    return vbyte(at) | vbyte(at + 1) << 8 | vbyte(at + 2) << 16 | vbyte(at + 3) << 24;
}

uint tbyte(uint at)
{
    return (codec_tstream[min(at >> 2, TRIANGLE_STREAM_WORDS - 1)] >> (8 * (at & 3))) & 0xFF;
}

// Bytes of a 16-value group of `bits`-bit values: the packed values, then one byte per
// value that holds all ones.
uint group_len(uint at, uint bits)
{
    if (bits == 0)
        return 0;
    if (bits == 8)
        return 16;
    if (bits == 1)
        return 2 + countbits(vbyte(at) | vbyte(at + 1) << 8);
    if (bits == 2)
    {
        uint w = vword(at);
        return 4 + countbits(w & (w >> 1) & 0x55555555);
    }
    uint w0 = vword(at);
    uint w1 = vword(at + 4);
    uint m = 0x11111111;
    return 8 + countbits(w0 & (w0 >> 1) & (w0 >> 2) & (w0 >> 3) & m)
        + countbits(w1 & (w1 >> 1) & (w1 >> 2) & (w1 >> 3) & m);
}

// Packed value `t` of a group: 1-bit values from the low bit up, wider ones from the high
// bits down.
uint group_field(uint at, uint bits, uint t)
{
    uint per = 8 / bits;
    uint i = t % per;
    uint shift = bits == 1 ? i : 8 - bits * (i + 1);
    return (vbyte(at + t / per) >> shift) & ((1u << bits) - 1);
}

uint group_value(uint at, uint bits, uint t)
{
    if (bits == 0)
        return 0;
    if (bits == 8)
        return vbyte(at + t);
    uint all = (1u << bits) - 1;
    uint v = group_field(at, bits, t);
    if (v != all)
        return v;
    uint escapes = 0;
    for (uint u = 0; u < t; u++)
        escapes += group_field(at, bits, u) == all ? 1 : 0;
    return vbyte(at + 2 * bits + escapes);
}

// Value bits of a group: 0, 1, 2 or 4 for a column with control 0, 1, 2, 4 or 8 for
// control 1.
uint group_bits(uint ctrl, uint header_bits)
{
    uint b = header_bits + ctrl;
    return b == 0 ? 0 : 1u << (b - 1);
}

// Walks the vertex stream's structure: header, control bytes, every byte column and group,
// and the tail; fills codec_columns and codec_groups. Returns false where the CPU decoder
// fails.
bool walk_vertices(uint len, uint vn, uint stride)
{
    if (len < 1)
        return false;
    if (vbyte(0) != VERTEX_CODEC_HEADER)
        return false;
    uint tail_size = stride + stride / 4;
    uint tail_pad = max(tail_size, 24u);
    if (len - 1 < tail_pad)
        return false;
    uint at = 1;
    if (vn > 0)
    {
        uint control = at;
        uint control_size = stride / 4;
        if (len - at < control_size)
            return false;
        at += control_size;
        uint groups = (vn + 15) / 16;
        uint header_size = (groups + 3) / 4;
        for (uint k = 0; k < stride; k++)
        {
            uint ctrl = (vbyte(control + k / 4) >> (k % 4 * 2)) & 3;
            if (ctrl == 3)
            {
                if (len - at < vn)
                    return false;
                codec_columns[k] = COLUMN_LITERAL << 16 | at;
                at += vn;
            }
            else if (ctrl == 2)
                codec_columns[k] = COLUMN_ZERO << 16;
            else
            {
                if (len - at < header_size)
                    return false;
                uint header = at;
                at += header_size;
                codec_columns[k] = COLUMN_GROUPS << 16;
                for (uint g = 0; g < groups; g++)
                {
                    if (len - at < 24)
                        return false;
                    uint bits = group_bits(ctrl, (vbyte(header + g / 4) >> (g % 4 * 2)) & 3);
                    codec_groups[k * MAX_GROUPS + g] = bits << 16 | at;
                    at += group_len(at, bits);
                }
            }
            if (k % 4 == 3 && (vbyte(len - tail_size + stride + k / 4) & 3) == 3)
                return false;
        }
    }
    return len - at == tail_pad;
}

uint column_byte(uint k, uint i)
{
    uint d = codec_columns[k];
    uint kind = d >> 16;
    if (kind == COLUMN_ZERO)
        return 0;
    if (kind == COLUMN_LITERAL)
        return vbyte((d & 0xFFFF) + i);
    uint g = codec_groups[k * MAX_GROUPS + i / 16];
    return group_value(g & 0xFFFF, g >> 16, i % 16);
}

uint unzigzag8x4(uint x)
{
    return ((x & 0x01010101) * 0xFF) ^ ((x >> 1) & 0x7F7F7F7F);
}

uint unzigzag16x2(uint x)
{
    return ((x & 0x00010001) * 0xFFFF) ^ ((x >> 1) & 0x7FFF7FFF);
}

uint add8x4(uint a, uint b)
{
    return ((a & 0x7F7F7F7F) + (b & 0x7F7F7F7F)) ^ ((a ^ b) & 0x80808080);
}

uint add16x2(uint a, uint b)
{
    return ((a & 0x7FFF7FFF) + (b & 0x7FFF7FFF)) ^ ((a ^ b) & 0x80008000);
}

// Channel modes from the tail: 0 byte deltas, 1 u16 deltas, 2 xor of u32 rotated by
// `channel >> 4`.
uint channel_mode(uint len, uint stride, uint ch)
{
    return vbyte(len - stride / 4 + ch);
}

uint channel_combine(uint mode, uint a, uint b)
{
    uint m = mode & 3;
    return m == 0 ? add8x4(a, b) : (m == 1 ? add16x2(a, b) : a ^ b);
}

uint channel_delta(uint mode, uint x)
{
    uint m = mode & 3;
    if (m == 0)
        return unzigzag8x4(x);
    if (m == 1)
        return unzigzag16x2(x);
    uint r = (32 - (mode >> 4)) & 31;
    return r == 0 ? x : (x << r) | (x >> (32 - r));
}

// The 4-bit code of triangle `t`, from the codes at the stream's end.
uint triangle_code(uint codes, uint t)
{
    return (tbyte(codes + t / 2) >> (t % 2 * 4)) & 0xF;
}

// Decodes cluster `cluster` of the page into its vertex and triangle slots. Every thread of
// a PAGE_CODEC_THREADS workgroup calls it with its own `lane` and gets the same status.
public uint page_decode_cluster(StoredPage p, uint cluster, uint lane)
{
    PageCheck k = check_page(p);
    if (k.status != 0)
        return k.status;
    PageHeader h = k.h;
    if (cluster >= h.cluster_count)
        return 0;
    Cluster c = load_cluster(p.stored, cluster);
    uint v0 = c.vertex_first;
    uint vn = c.vertex_count;
    uint t0 = c.tri_first;
    uint tn = c.tri_count;
    if (v0 % 2 != 0 || uint64_t(v0) + vn > k.slots || t0 % 4 != 0
        || uint64_t(t0) + tn * 3 > k.tris_len || c.alpha_first > c.tri_count)
        return PAGE_CODEC_BAD_CLUSTER;
    // Clusters take their slots in order, so no two workgroups write the same bytes.
    if (cluster > 0)
    {
        Cluster q = load_cluster(p.stored, cluster - 1);
        if (uint64_t(v0) < uint64_t(q.vertex_first) + (uint(q.vertex_count) + 1) / 2 * 2
            || uint64_t(t0) < uint64_t(q.tri_first) + (uint(q.tri_count) * 3 + 3) / 4 * 4)
            return PAGE_CODEC_BAD_CLUSTER;
    }
    uint* table = (uint*)(p.stored + h.positions_offset);
    uint a = table[2 * cluster];
    uint b = table[2 * cluster + 1];
    uint e = table[2 * cluster + 2];
    uint stride = 8 + h.attr_stride;
    if (!(a <= b && b <= e && e <= k.streams_end) || vn > MAX_VERTICES || stride > MAX_VERTEX_BYTES)
        return PAGE_CODEC_BAD_CLUSTER;
    uint vlen = b - a;
    uint tlen = e - b;
    if (vlen > VERTEX_STREAM_MAX)
        return PAGE_CODEC_BAD_VERTICES;
    if (tlen > TRIANGLE_STREAM_MAX)
        return PAGE_CODEC_BAD_TRIANGLES;

    // A workgroup may decode several clusters in turn: the last call's lanes are done with
    // the shared arrays before this one writes them.
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0)
        codec_status = 0;
    for (uint w = lane; w < VERTEX_STREAM_WORDS; w += PAGE_CODEC_THREADS)
    {
        uint v = 0;
        for (uint j = 0; j < 4 && w * 4 + j < vlen; j++)
            v |= uint(p.stored[a + w * 4 + j]) << (8 * j);
        codec_vstream[w] = v;
    }
    for (uint w = lane; w < TRIANGLE_STREAM_WORDS; w += PAGE_CODEC_THREADS)
    {
        uint v = 0;
        for (uint j = 0; j < 4 && w * 4 + j < tlen; j++)
            v |= uint(p.stored[b + w * 4 + j]) << (8 * j);
        codec_tstream[w] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    // Triangle stream: the codes sit at the end after a gap that pads them to 16 bytes, the
    // extra corners at the start. An odd count's pad code consumes extra corners too.
    uint codes_size = (tn + 1) / 2;
    uint gap = codes_size < 16 ? 16 - codes_size : 0;
    uint codes = tlen - codes_size;
    uint bound = tlen - codes_size - gap;
    if (lane == 0)
    {
        if (!walk_vertices(vlen, vn, stride))
            InterlockedOr(codec_status, PAGE_CODEC_BAD_VERTICES);
        if (tlen < codes_size + gap)
            InterlockedOr(codec_status, PAGE_CODEC_BAD_TRIANGLES);
    }
    uint code = 0;
    uint counts = 0;
    if (lane < (tn + 1) / 2 * 2 && tlen >= codes_size + gap)
    {
        code = triangle_code(codes, lane);
        uint extra = code < 12 ? (code & 1) : code - 12;
        uint next = lane < tn ? (code < 12 ? 1 - (code & 1) : 15 - code) : 0;
        counts = extra | next << 16;
    }
    codec_prefix[lane] = counts;
    GroupMemoryBarrierWithGroupSync();
    if (codec_status != 0)
        return codec_status;

    for (uint task = lane; task < stride * MAX_VERTICES / 4; task += PAGE_CODEC_THREADS)
    {
        uint col = task / (MAX_VERTICES / 4);
        uint i = task % (MAX_VERTICES / 4) * 4;
        uint v = 0;
        for (uint j = 0; j < 4 && i + j < vn; j++)
            v |= column_byte(col, i + j) << (8 * j);
        codec_bytes[task] = v;
    }
    for (uint d = 1; d < MAX_VERTICES; d *= 2)
    {
        uint s = codec_prefix[lane];
        if (lane >= d)
            s += codec_prefix[lane - d];
        GroupMemoryBarrierWithGroupSync();
        codec_prefix[lane] = s;
        GroupMemoryBarrierWithGroupSync();
    }
    uint inclusive = codec_prefix[lane];
    uint last = codec_prefix[MAX_VERTICES - 1];
    if (lane == 0 && (last & 0xFFFF) != bound)
        InterlockedOr(codec_status, PAGE_CODEC_BAD_TRIANGLES);

    uint channels = stride / 4;
    for (uint ch = 0; ch < channels; ch++)
    {
        uint x = 0;
        if (lane < vn)
        {
            uint col = ch * 4;
            uint q = MAX_VERTICES / 4;
            uint at = lane / 4;
            uint shift = lane % 4 * 8;
            for (uint j = 0; j < 4; j++)
                x |= ((codec_bytes[(col + j) * q + at] >> shift) & 0xFF) << (8 * j);
            x = channel_delta(channel_mode(vlen, stride, ch), x);
        }
        codec_channels[ch * MAX_VERTICES + lane] = x;
    }

    // Triangle corners as slots 3t + (a, b, c). Edge reuse names two corners of one of the
    // three triangles before; pointer doubling resolves those chains.
    if (lane < tn)
    {
        uint t = lane;
        uint x = inclusive & 0xFFFF;
        uint next = inclusive >> 16;
        uint extra = code < 12 ? (code & 1) : code - 12;
        x -= extra;
        next -= code < 12 ? 1 - (code & 1) : 15 - code;
        uint sa, sb, sc;
        if (code < 12)
        {
            sc = (code & 1) != 0 ? tbyte(x) : next & 0xFF;
            int j = int(t) - 1 - int(code / 4);
            if (j < 0)
            {
                sa = 0;
                sb = 0;
            }
            else if ((code & 2) != 0)
            {
                sa = SLOT_POINTER | (3 * uint(j) + 2);
                sb = SLOT_POINTER | (3 * uint(j) + 1);
            }
            else
            {
                sa = SLOT_POINTER | (3 * uint(j));
                sb = SLOT_POINTER | (3 * uint(j) + 2);
            }
        }
        else
        {
            uint fa = code > 12 ? 1 : 0;
            uint fb = code > 13 ? 1 : 0;
            uint fc = code > 14 ? 1 : 0;
            sa = fa != 0 ? tbyte(x) : next & 0xFF;
            x += fa;
            next += 1 - fa;
            sb = fb != 0 ? tbyte(x) : next & 0xFF;
            x += fb;
            next += 1 - fb;
            sc = fc != 0 ? tbyte(x) : next & 0xFF;
        }
        codec_slots[3 * t] = sa;
        codec_slots[3 * t + 1] = sb;
        codec_slots[3 * t + 2] = sc;
    }
    GroupMemoryBarrierWithGroupSync();

    uint modes[MAX_CHANNELS];
    for (uint ch = 0; ch < channels; ch++)
        modes[ch] = channel_mode(vlen, stride, ch);
    for (uint d = 1; d < 512; d *= 2)
    {
        uint scan[MAX_CHANNELS];
        for (uint ch = 0; ch < channels; ch++)
        {
            uint s = codec_channels[ch * MAX_VERTICES + lane];
            if (d < MAX_VERTICES && lane >= d)
                s = channel_combine(modes[ch], codec_channels[ch * MAX_VERTICES + lane - d], s);
            scan[ch] = s;
        }
        uint slot[3];
        for (uint s = 0; s < 3; s++)
        {
            uint v = lane < tn ? codec_slots[3 * lane + s] : 0;
            if ((v & SLOT_POINTER) != 0)
                v = codec_slots[v & ~SLOT_POINTER];
            slot[s] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint ch = 0; ch < channels; ch++)
            codec_channels[ch * MAX_VERTICES + lane] = scan[ch];
        if (lane < tn)
        {
            for (uint s = 0; s < 3; s++)
                codec_slots[3 * lane + s] = slot[s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Vertex values continue from the last vertex, which the tail holds.
    uint tail = vlen - (stride + stride / 4);
    for (uint ch = 0; ch < channels; ch++)
    {
        uint i = ch * MAX_VERTICES + lane;
        codec_channels[i] = channel_combine(modes[ch], vword(tail + ch * 4), codec_channels[i]);
    }
    if (lane < tn)
    {
        for (uint s = 0; s < 3; s++)
        {
            if (codec_slots[3 * lane + s] >= vn)
                InterlockedOr(codec_status, PAGE_CODEC_BAD_TRIANGLES);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (codec_status != 0)
        return codec_status;

    // Positions: 6 bytes per vertex from channels 0 and 1; attributes: channels 2 on. An odd
    // count's spare slot is written as zeros.
    uint slots = (vn + 1) / 2 * 2;
    uint* positions = (uint*)(p.resident + h.positions_offset + v0 * 6);
    for (uint d = lane; d < slots * 6 / 4; d += PAGE_CODEC_THREADS)
    {
        uint v = 0;
        for (uint j = 0; j < 4; j++)
        {
            uint q = d * 4 + j;
            uint vertex = q / 6;
            uint r = q % 6;
            if (vertex < vn)
                v |= ((codec_channels[r / 4 * MAX_VERTICES + vertex] >> (r % 4 * 8)) & 0xFF) << (8 * j);
        }
        positions[d] = v;
    }
    uint words = h.attr_stride / 4;
    uint* attrs = (uint*)(p.resident + h.attrs_offset + v0 * h.attr_stride);
    for (uint d = lane; d < slots * words; d += PAGE_CODEC_THREADS)
    {
        uint vertex = d / words;
        attrs[d] = vertex < vn ? codec_channels[(2 + d % words) * MAX_VERTICES + vertex] : 0;
    }
    uint* tris = (uint*)(p.resident + h.tris_offset + t0);
    for (uint d = lane; d < (tn * 3 + 3) / 4; d += PAGE_CODEC_THREADS)
    {
        uint v = 0;
        for (uint j = 0; j < 4 && d * 4 + j < tn * 3; j++)
            v |= codec_slots[d * 4 + j] << (8 * j);
        tris[d] = v;
    }
    return 0;
}
