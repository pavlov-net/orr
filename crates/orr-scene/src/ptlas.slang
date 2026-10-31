// A scene's PTLAS work for one frame. `clear` resets the frame's marks and `recycle` returns
// the part runs released last frame to the free lists; `want` decides each slot's operation
// (a write, a null write, a BLAS update, taking, rewriting or releasing a part run, or
// nothing) from its record, its transform, its pose, its distance from the focus and what
// its PTLAS instances hold, and charges it to the distance bucket of its partition; `cutoff`
// and `admit` spend the frame's budget of rebuilt instances bucket by bucket from the focus
// outward, mandatory work first; `emit` writes the admitted records and ages the rest;
// `translate` writes partition translations and `commands` the build's one command per
// record type. `producer_want` and `producer_emit` do the same for producer entries: an
// entry's instance at `producer_base + id`, in the global partition, is rewritten as
// mandatory work every frame it builds, moves or the render origin moves, and null-written
// when it has nothing to trace.
//
// A traced copy inside its animation range whose cut has animated part clusters holds a
// run of part instances, one per animated part: its body instance traces the cut's body
// BLAS and each part instance the part's BLAS posed by `pose[part]`, rewritten (mandatory
// work) every frame the pose or the copy changes. Other copies trace the rest-pose BLAS of
// the cut. Copies with animated parts stay in their cell partition. Runs come from free
// lists per run class (2^c part instances), filled from a bump pointer; a run released in
// a frame is reused from the next.
#include "orr/root.h"
import orr.abi;
import orr.math;
import orr.trace;
import rtas;
import hit;
import producer;

public static const uint PARTITION_NONE = 0xffffffff;
public static const uint DISTANCE_BUCKETS = 64;
// Frames without a transform change after which a mover returns to its cell partition.
public static const uint MOVER_FRAMES = 30;
public static const float RT_HYSTERESIS = 1.15;

// `PtlasGpu.counters`. Those from COUNTERS_CLEARED on persist across frames.
public static const uint COUNTER_WRITES = 0;
public static const uint COUNTER_UPDATES = 1;
public static const uint COUNTER_TRANSLATES = 2;
public static const uint COUNTER_MANDATORY = 3;   // rebuilt instances of mandatory work
public static const uint COUNTER_REBUILT = 4;     // rebuilt instances admitted, mandatory included
public static const uint COUNTER_DEFERRED = 5;    // operations left for a later frame
public static const uint COUNTER_MAX_AGE = 6;     // frames the oldest deferred operation has waited
public static const uint COUNTER_DROPPED = 7;     // records past their array's capacity
public static const uint COUNTER_CUTOFF = 8;      // the first bucket not admitted whole
public static const uint COUNTER_REMAINING = 9;   // budget left for the cutoff bucket
public static const uint COUNTER_CLAIMED = 10;    // budget the cutoff bucket has claimed
public static const uint COUNTER_BELOW = 11;      // budget the buckets before the cutoff took
public static const uint COUNTER_GLOBAL_CLAIMS = 12;
public static const uint COUNTER_PARTS_REFUSED = 13; // part runs not taken: no room left
public static const uint COUNTERS_CLEARED = 14;
public static const uint COUNTER_GLOBAL = 14;     // copies in the global partition
public static const uint COUNTER_PARTS = 15;      // part instances in held runs
public static const uint COUNTER_PRODUCERS = 16;  // producer instances, all in the global partition
public static const uint COUNTERS = 17;

// `PtlasRoot.flags`.
public static const uint PTLAS_REBASE = 1;        // the render origin moved: mandatory work only
public static const uint PTLAS_FIRST = 2;         // the first build: the persistent state starts at 0

// `PtlasGpu.part_state`.
public static const uint PART_BUMP = 0;           // part instances handed out from the pool
public static const uint PART_RELEASED = 1;       // runs in `part_released`
public static const uint PART_TOPS = 2;           // one free-list size per run class
public static const uint PART_CLASSES = 9;        // runs of 1 to 256 part instances
public static const uint PART_STATE = PART_TOPS + PART_CLASSES;

// `partition_state` bits.
static const uint PS_TOUCHED = 1;
static const uint PS_MANDATORY = 2;
static const uint PS_ADMITTED = 4;

// `PendingGpu.op`.
public static const uint OP_NONE = 0;
public static const uint OP_WRITE = 1;
public static const uint OP_UPDATE = 2;
public static const uint OP_NULL = 4;             // with OP_WRITE: disables the instance
public static const uint OP_MANDATORY = 8;
public static const uint OP_TAKE = 16;            // take a part run and write its parts
public static const uint OP_RELEASE = 32;         // null-write the held part run and release it
public static const uint OP_POSE = 64;            // rewrite the held part run
public static const uint OP_PARTS = OP_TAKE | OP_RELEASE | OP_POSE;

// A cell partition: the world cell its transforms are relative to and the instances it holds.
public struct PartitionGpu
{
    public int3 cell;
    public uint count;
}

// A partition translation to write this frame.
public struct TranslateJob
{
    public float3 translation;
    public uint partition;
}

// One slot's operation this frame.
public struct PendingGpu
{
    public uint op;               // OP_*
    public uint partition;        // the partition written or updated; partition_count: global
    public uint64_t blas;         // the body BLAS with a part run, else the rest-pose BLAS
    public uint64_t rest;         // the rest-pose BLAS
    public uint2 key;
    public uint cut;
    public uint _pad;
}

public struct PtlasGpu
{
    public PartitionGpu* partitions;
    public uint* slot_partition;  // per slot: its cell partition, or PARTITION_NONE
    public PendingGpu* pending;   // per slot
    public uint* partition_state; // per partition, then the global partition
    public uint* buckets;         // DISTANCE_BUCKETS costs in rebuilt instances
    public uint* counters;        // COUNTER_*
    public uint8_t* writes;
    public uint8_t* updates;
    public uint8_t* translates;
    public uint8_t* commands;
    public uint* command_count;
    public uint* part_state;      // PART_*
    public uint* part_free;       // per run class c, from part_stack(c): part_capacity >> c free run starts
    public uint* part_released;   // runs released this frame: first part instance | class << 24
    public uint partition_count;
    public uint write_capacity;
    public uint update_capacity;
    public uint global_capacity;  // copies the global partition takes
}

public struct PtlasRoot
{
    public SceneRoot* scene;
    public PtlasGpu* ptlas;
    public int4 focus_cell;       // w: 1 with a focus
    public float4 focus_offset;   // w: the focus's pixels per radian
    public uint slots;            // slots to walk
    public uint budget;           // rebuilt instances beyond mandatory work
    public uint flags;            // PTLAS_*
    public float error_pixels;
    public TranslateJob* translate_jobs;
    public uint translate_count;
    public uint _pad;
}
ORR_ROOT(PtlasRoot);

uint global_partition()
{
    return root.ptlas.partition_count;
}

uint partition_cost(uint p)
{
    PtlasGpu* t = root.ptlas;
    return p < t.partition_count ? t.partitions[p].count : max(t.counters[COUNTER_GLOBAL] + t.counters[COUNTER_PRODUCERS], 1u);
}

// Integer cell distance of a partition from the focus.
uint partition_bucket(uint p)
{
    PtlasGpu* t = root.ptlas;
    if (p >= t.partition_count || root.focus_cell.w == 0)
        return 0;
    int3 d = abs(t.partitions[p].cell - root.focus_cell.xyz);
    return min(uint(max(d.x, max(d.y, d.z))), DISTANCE_BUCKETS - 1);
}

uint mask_of(uint class_id)
{
    switch (class_id)
    {
    case CLASS_DIELECTRIC:
    case CLASS_PRESENCE:
        return MASK_DIELECTRIC;
    case CLASS_PARTICLE:
        return MASK_PARTICLE;
    case CLASS_MEDIUM:
        return MASK_MEDIUM;
    default:
        return MASK_OPAQUE;
    }
}

uint mix(uint h, uint v)
{
    h ^= v;
    h *= 0x9e3779b1u;
    return h ^ (h >> 15);
}

// A hash of what a write puts in the PTLAS besides the BLAS: transform, model and class.
uint2 instance_key(GroupGpu* g, InstanceGpu inst)
{
    float3x4 m = ((float3x4*)g.current)[inst.group_index];
    uint a = 0x811c9dc5u;
    uint b = 0x01000193u;
    for (uint r = 0; r < 3; r++)
    {
        for (uint c = 0; c < 4; c++)
        {
            a = mix(a, asuint(m[r][c]));
            b = mix(b, asuint(m[r][c]) + 0x632be5abu);
        }
    }
    a = mix(mix(a, inst.model), inst.candidate_class);
    b = mix(mix(b, inst.group), inst.model ^ (uint(inst.candidate_class) << 24));
    return uint2(a, b);
}

bool moved(GroupGpu* g, uint index)
{
    if (g.previous == g.current)
        return false;
    float3x4 a = ((float3x4*)g.current)[index];
    float3x4 b = ((float3x4*)g.previous)[index];
    return any(a[0] != b[0]) || any(a[1] != b[1]) || any(a[2] != b[2]);
}

float projected(RtCutGpu* cuts, uint k, float scale, float distance)
{
    return cuts[k].error * scale / max(distance, 1e-3) * root.focus_offset.w;
}

// The coarsest cut whose error at `distance` stays at or under `limit` pixels.
uint coarsest_cut(MeshGpu* mesh, float scale, float distance, float limit)
{
    for (uint k = mesh.rt_cut_count; k > 1; k--)
    {
        if (projected(mesh.rt_cuts, k - 1, scale, distance) <= limit)
            return k - 1;
    }
    return 0;
}

// The rest-pose and body BLAS of `cut` for the copy's section mask; false when the mask
// names a set whose BLAS are not built.
bool cut_blas(MeshGpu* mesh, ModelGpu* model, InstanceGpu inst, uint cut, out uint64_t rest, out uint64_t body)
{
    RtCutGpu k = mesh.rt_cuts[cut];
    rest = k.blas;
    body = k.body;
    uint mask = (inst.variant >> 16) & ((1u << min(model.section_count, 16u)) - 1);
    if (mask == 0)
        return true;
    for (uint i = 0; i < mesh.mask_count; i++)
    {
        MaskGpu m = mesh.masks[i];
        if (m.mask == mask)
        {
            rest = m.blas[cut];
            body = m.blas[mesh.rt_cut_count + cut];
            return true;
        }
    }
    rest = 0;
    body = 0;
    return false;
}

// Whether any animated part's pose changed since the previous frame.
bool pose_moved(SceneRoot* s, InstanceGpu inst, MeshGpu* mesh)
{
    GroupGpu* g = s.groups + inst.pose_group;
    if (g.previous == g.current)
        return false;
    for (uint j = 0; j < mesh.animated_parts; j++)
    {
        float3x4 a;
        float3x4 b;
        bool shown = part_pose(s, inst, mesh, mesh.first_animated + j, false, a);
        if (shown != part_pose(s, inst, mesh, mesh.first_animated + j, true, b))
            return true;
        if (any(a[0] != b[0]) || any(a[1] != b[1]) || any(a[2] != b[2]))
            return true;
    }
    return false;
}

void charge_write(uint p, bool mandatory)
{
    PtlasGpu* t = root.ptlas;
    uint prev;
    InterlockedOr(t.partition_state[p], PS_TOUCHED | (mandatory ? PS_MANDATORY : 0), prev);
    if ((prev & PS_TOUCHED) != 0)
        return;
    uint cost = partition_cost(p);
    if (mandatory)
        InterlockedAdd(t.counters[COUNTER_MANDATORY], cost);
    else
        InterlockedAdd(t.buckets[partition_bucket(p)], cost);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 id: SV_DispatchThreadID)
{
    PtlasGpu* t = root.ptlas;
    bool first = (root.flags & PTLAS_FIRST) != 0;
    if (id.x <= t.partition_count)
        t.partition_state[id.x] = 0;
    if (id.x < DISTANCE_BUCKETS)
        t.buckets[id.x] = 0;
    if (id.x < (first ? COUNTERS : COUNTERS_CLEARED))
        t.counters[id.x] = 0;
    if (first && id.x < PART_STATE)
        t.part_state[id.x] = 0;
}

// The class of a run of `n` part instances: runs of class c hold 2^c.
uint run_class(uint n)
{
    return n <= 1 ? 0 : firstbithigh(n - 1) + 1;
}

// The first entry of class `c`'s free list in `part_free`.
uint part_stack(uint c)
{
    uint at = 0;
    for (uint k = 0; k < c; k++)
        at += root.scene.part_capacity >> k;
    return at;
}

// One workgroup: puts the runs released last frame on their free lists.
[shader("compute")]
[numthreads(256, 1, 1)]
void recycle(uint t: SV_GroupThreadID)
{
    PtlasGpu* p = root.ptlas;
    if ((root.flags & PTLAS_FIRST) != 0)
        return;
    uint n = p.part_state[PART_RELEASED];
    for (uint i = t; i < n; i += 256)
    {
        uint e = p.part_released[i];
        uint c = e >> 24;
        uint at;
        InterlockedAdd(p.part_state[PART_TOPS + c], 1, at);
        p.part_free[part_stack(c) + at] = e & 0xffffff;
    }
    AllMemoryBarrierWithGroupSync();
    if (t == 0)
        p.part_state[PART_RELEASED] = 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void want(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    PtlasGpu* t = root.ptlas;
    InstanceGpu inst = s.instances[slot];
    SlotStateGpu* st = s.slot_state + slot;
    bool seen = (st.rt_flags & RT_SEEN) != 0;
    bool written = (st.rt_flags & RT_WRITTEN) != 0;
    bool has_focus = root.focus_cell.w != 0;
    uint cell_log2 = uint(s.origin_cell.w);

    PendingGpu op = (PendingGpu)0;
    op.partition = st.rt_partition;
    op.cut = st.rt_lod;
    bool active = (inst.flags & INSTANCE_RAY) != 0;
    GroupGpu* g = s.groups + inst.group;
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    Placement pl = (Placement)0;
    active = active && placement(g, g.current, inst.group_index, root.focus_cell.xyz, root.focus_offset.xyz, cell_log2, pl);
    active = active && mesh.rt_cut_count > 0;
    uint cell = active ? t.slot_partition[slot] : PARTITION_NONE;
    active = active && cell != PARTITION_NONE;

    uint stable = 255;
    bool in_range = false;
    uint64_t body = 0;
    if (active)
    {
        stable = !seen ? 255 : moved(g, inst.group_index) ? 0 : min(uint(st.stable_frames) + 1, 255u);
        float3 center = place(pl, 0.5 * (model.bounds_min + model.bounds_max));
        float radius = 0.5 * length(model.bounds_max - model.bounds_min) * pl.scale;
        float distance = has_focus ? max(length(center) - radius, 0.0) : 0.0;
        float range = float(inst.rt_range) * ((st.rt_flags & RT_IN_RANGE) != 0 ? RT_HYSTERESIS : 1.0);
        in_range = !has_focus || distance <= range;

        uint limit_cut = has_focus ? coarsest_cut(mesh, pl.scale, distance, root.error_pixels) : 0;
        uint cut = limit_cut;
        if (written && has_focus)
        {
            uint current = min(uint(st.rt_lod), mesh.rt_cut_count - 1);
            uint coarser = coarsest_cut(mesh, pl.scale, distance, root.error_pixels / RT_HYSTERESIS);
            if (coarser > current)
                cut = coarser;
            else if (projected(mesh.rt_cuts, current, pl.scale, distance) > root.error_pixels * RT_HYSTERESIS)
                cut = limit_cut;
            else
                cut = current;
        }
        op.cut = cut;
        cut_blas(mesh, model, inst, cut, op.rest, body);
        active = op.rest != 0;
        op.blas = op.rest;
        op.key = instance_key(g, inst);
    }
    st.stable_frames = uint8_t(stable);
    st.rt_flags = (st.rt_flags & ~RT_IN_RANGE) | RT_SEEN | (in_range ? RT_IN_RANGE : 0);

    // A traced copy inside its animation range holds a run while its cut has animated
    // part clusters and a body, and its section mask leaves some part shown.
    uint held = st.part_count;
    uint hidden = (inst.variant >> 16) & 0xffff;
    bool posed = active && in_range && (st.rt_flags & RT_ANIMATED) != 0
        && mesh.rt_cuts[op.cut].parts != nullptr && body != 0
        && (mesh.animated_sections & ~hidden) != 0;
    bool release = held != 0 && (!posed || held != mesh.animated_parts);
    bool take = posed && (held == 0 || release);

    if (!active || !in_range)
    {
        // A removed slot's instances go now: its slot may be handed out again.
        bool mandatory = inst.flags == 0;
        if (written)
            op.op = OP_WRITE | OP_NULL;
        if (release)
            op.op |= OP_RELEASE;
        if (op.op != OP_NONE)
        {
            op.op |= mandatory ? OP_MANDATORY : 0;
            charge_write(op.partition, mandatory);
        }
        t.pending[slot] = op;
        return;
    }

    uint target = cell;
    bool parted = inst.pose_group != 0 && mesh.animated_parts > 0;
    bool in_global = written && (st.rt_flags & RT_GLOBAL) != 0;
    if (!parted && seen && stable < MOVER_FRAMES)
    {
        uint claim = 0;
        if (!in_global)
            InterlockedAdd(t.counters[COUNTER_GLOBAL_CLAIMS], 1, claim);
        if (in_global || t.counters[COUNTER_GLOBAL] + claim < t.global_capacity)
            target = global_partition();
    }
    if (posed)
        op.blas = body;
    // Global instances are in render space, so a rebase rewrites every one of them now.
    bool rebased = (root.flags & PTLAS_REBASE) != 0;
    bool placed = written && st.rt_partition == target && all(st.rt_key == op.key);
    bool update = false;
    if (!placed || (target == global_partition() && rebased))
    {
        bool mandatory = in_global && target == global_partition() && rebased;
        op.op = OP_WRITE | (mandatory ? OP_MANDATORY : 0);
        charge_write(target, mandatory);
        if (written && st.rt_partition != target)
            charge_write(st.rt_partition, false);
    }
    else if (st.rt_blas != op.blas)
    {
        update = true;
    }

    // Part runs: taken and released as the budget admits them; a held run follows its pose
    // and its copy every frame.
    bool repose = posed && !take && (op.op != OP_NONE || update || op.cut != st.rt_lod || pose_moved(s, inst, mesh));
    op.op |= (take ? OP_TAKE : 0) | (release ? OP_RELEASE : 0) | (repose ? OP_POSE : 0);
    if (repose)
        op.op |= OP_MANDATORY;
    if ((op.op & OP_PARTS) != 0)
    {
        // The body's update lands in the partition the parts rebuild.
        charge_write(target, repose);
    }
    else if (update)
    {
        // The same cut at a new address is a rebuilt BLAS: mandatory.
        bool mandatory = op.cut == st.rt_lod;
        op.op = OP_UPDATE | (mandatory ? OP_MANDATORY : 0);
        if (mandatory)
            InterlockedAdd(t.counters[COUNTER_MANDATORY], 1);
        else
            InterlockedAdd(t.buckets[partition_bucket(target)], 1);
    }
    op.partition = target;
    t.pending[slot] = op;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void cutoff(uint3 id: SV_DispatchThreadID)
{
    uint* c = root.ptlas.counters;
    uint* buckets = root.ptlas.buckets;
    uint mandatory = c[COUNTER_MANDATORY];
    bool rebased = (root.flags & PTLAS_REBASE) != 0;
    uint remaining = rebased || root.budget <= mandatory ? 0 : root.budget - mandatory;
    uint below = 0;
    uint cut = DISTANCE_BUCKETS;
    for (uint b = 0; b < DISTANCE_BUCKETS; b++)
    {
        if (buckets[b] > remaining)
        {
            cut = b;
            break;
        }
        remaining -= buckets[b];
        below += buckets[b];
    }
    c[COUNTER_CUTOFF] = cut;
    c[COUNTER_REMAINING] = remaining;
    c[COUNTER_BELOW] = below;
    c[COUNTER_REBUILT] = mandatory + below;
}

// Whether the cutoff bucket can take `cost` more. When nothing before it was admitted, its
// first claim always is, so a bucket larger than the budget still makes progress.
bool claim(uint cost)
{
    uint* c = root.ptlas.counters;
    uint prev;
    InterlockedAdd(c[COUNTER_CLAIMED], cost, prev);
    bool first = prev == 0 && c[COUNTER_BELOW] == 0 && (root.flags & PTLAS_REBASE) == 0;
    bool fits = prev + cost <= c[COUNTER_REMAINING] || first;
    if (fits)
        InterlockedAdd(c[COUNTER_REBUILT], cost);
    return fits;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void admit(uint3 id: SV_DispatchThreadID)
{
    PtlasGpu* t = root.ptlas;
    uint p = id.x;
    if (p > t.partition_count)
        return;
    uint state = t.partition_state[p];
    if ((state & PS_TOUCHED) == 0)
        return;
    uint b = partition_bucket(p);
    uint cut = t.counters[COUNTER_CUTOFF];
    bool admitted = (state & PS_MANDATORY) != 0 || b < cut || (b == cut && claim(partition_cost(p)));
    if (admitted)
        t.partition_state[p] = state | PS_ADMITTED;
}

uint ptlas_partition(uint p)
{
    return p == global_partition() ? PTLAS_GLOBAL_PARTITION : p;
}

// `count` consecutive records of a type's array from `index`; false past its capacity.
bool reserve(uint counter, uint capacity, uint count, out uint index)
{
    InterlockedAdd(root.ptlas.counters[counter], count, index);
    if (index + count > capacity)
    {
        InterlockedAdd(root.ptlas.counters[COUNTER_DROPPED], count);
        return false;
    }
    return true;
}

// The write of an instance over `blas` placed by `pl` with mesh-space bounds `lo`..`hi`.
PtlasWrite instance_write(uint index, uint partition, Placement pl, float3 lo, float3 hi, uint64_t blas, uint class_id)
{
    PtlasWrite w = (PtlasWrite)0;
    w.instance_index = index;
    w.partition = ptlas_partition(partition);
    w.transform = float3x4(
        float4(pl.linear[0], pl.translation.x),
        float4(pl.linear[1], pl.translation.y),
        float4(pl.linear[2], pl.translation.z));
    float3 a = float3(1e30);
    float3 b = float3(-1e30);
    for (uint k = 0; k < 8; k++)
    {
        float3 corner = float3((k & 1) != 0 ? hi.x : lo.x, (k & 2) != 0 ? hi.y : lo.y, (k & 4) != 0 ? hi.z : lo.z);
        float3 p = place(pl, corner);
        a = min(a, p);
        b = max(b, p);
    }
    w.aabb_min = a;
    w.aabb_max = b;
    w.blas = blas;
    w.custom_index = class_id;
    w.mask = mask_of(class_id);
    w.flags = INSTANCE_EXPLICIT_AABB;
    return w;
}

// Null-writes the slot's part run and puts it on the released list.
void release_run(SceneRoot* s, PtlasGpu* t, SlotStateGpu* st, uint slot)
{
    uint n = st.part_count;
    uint index;
    if (reserve(COUNTER_WRITES, t.write_capacity, n, index))
    {
        for (uint j = 0; j < n; j++)
        {
            PtlasWrite w = (PtlasWrite)0;
            w.instance_index = s.part_base + st.part_run + j;
            w.partition = ptlas_partition(st.rt_partition);
            store((uint64_t)t.writes, index + j, w);
        }
    }
    uint at;
    InterlockedAdd(t.part_state[PART_RELEASED], 1, at);
    t.part_released[at] = st.part_run | (run_class(n) << 24);
    InterlockedAdd(t.counters[COUNTER_PARTS], uint(-int(n)));
    st.part_count = 0;
    // An emptied slot's record stays zero.
    if (s.instances[slot].flags != 0)
        s.instances[slot].first_part = NO_PART_RUN;
}

// Takes a run of `n` part instances for the slot, from its class's free list or else the
// pool; false when the pool is full.
bool take_run(SceneRoot* s, PtlasGpu* t, SlotStateGpu* st, uint slot, uint n)
{
    uint c = run_class(n);
    uint top;
    InterlockedAdd(t.part_state[PART_TOPS + c], uint(-1), top);
    uint start;
    if (int(top) > 0)
    {
        start = t.part_free[part_stack(c) + top - 1];
    }
    else
    {
        InterlockedAdd(t.part_state[PART_BUMP], 1u << c, start);
        if (start + (1u << c) > s.part_capacity)
        {
            InterlockedAdd(t.counters[COUNTER_PARTS_REFUSED], 1);
            return false;
        }
    }
    for (uint j = 0; j < n; j++)
        s.part_slots[start + j] = slot;
    st.part_run = start;
    st.part_count = n;
    InterlockedAdd(t.counters[COUNTER_PARTS], n);
    return true;
}

// Writes every part instance of the slot's run: the cut's part BLAS posed by the copy's
// transform and `pose[part]`, null where the part has no clusters at the cut, its sections
// are all hidden or its pose hides it.
void write_parts(SceneRoot* s, PtlasGpu* t, SlotStateGpu* st, InstanceGpu inst, uint cut, uint partition)
{
    uint n = st.part_count;
    uint index;
    if (!reserve(COUNTER_WRITES, t.write_capacity, n, index))
        return;
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    RtCutGpu k = mesh.rt_cuts[cut];
    uint hidden = (inst.variant >> 16) & 0xffff;
    GroupGpu* g = s.groups + inst.group;
    Placement copy;
    bool shown = placement(g, g.current, inst.group_index, t.partitions[partition].cell, float3(0.0), uint(s.origin_cell.w), copy);
    for (uint j = 0; j < n; j++)
    {
        PartGpu part = mesh.parts[j];
        uint64_t blas = k.parts[j];
        float3x4 pose;
        PtlasWrite w = (PtlasWrite)0;
        w.instance_index = s.part_base + st.part_run + j;
        w.partition = ptlas_partition(partition);
        if (shown && blas != 0 && (part.sections & ~hidden) != 0 && part_pose(s, inst, mesh, mesh.first_animated + j, false, pose))
            w = instance_write(w.instance_index, partition, placed_after(copy, pose), part.bounds_min, part.bounds_max, blas, inst.candidate_class);
        store((uint64_t)t.writes, index + j, w);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emit(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    PtlasGpu* t = root.ptlas;
    PendingGpu op = t.pending[slot];
    if (op.op == OP_NONE)
        return;
    SlotStateGpu* st = s.slot_state + slot;
    bool mandatory = (op.op & OP_MANDATORY) != 0;
    bool admitted;
    if ((op.op & (OP_WRITE | OP_PARTS)) != 0)
    {
        admitted = (t.partition_state[op.partition] & PS_ADMITTED) != 0;
    }
    else
    {
        uint b = partition_bucket(op.partition);
        uint cut = t.counters[COUNTER_CUTOFF];
        admitted = mandatory || b < cut || (b == cut && claim(1));
    }
    if (!admitted)
    {
        st.rt_age = st.rt_age + 1;
        InterlockedAdd(t.counters[COUNTER_DEFERRED], 1);
        InterlockedMax(t.counters[COUNTER_MAX_AGE], st.rt_age);
        return;
    }
    st.rt_age = 0;
    st.rt_lod = uint8_t(op.cut);
    InstanceGpu inst = s.instances[slot];

    if ((op.op & OP_RELEASE) != 0)
        release_run(s, t, st, slot);
    bool held = st.part_count != 0;
    if ((op.op & OP_TAKE) != 0)
        held = take_run(s, t, st, slot, s.meshes[s.models[inst.model].mesh].animated_parts);
    uint64_t blas = held ? op.blas : op.rest;
    bool written = (st.rt_flags & RT_WRITTEN) != 0;

    if ((op.op & OP_WRITE) == 0)
    {
        uint index;
        if (written && blas != st.rt_blas && reserve(COUNTER_UPDATES, t.update_capacity, 1, index))
        {
            PtlasUpdate u;
            u.blas = blas;
            u.instance_index = slot;
            u.pad = 0;
            store((uint64_t)t.updates, index, u);
            st.rt_blas = blas;
        }
    }
    else
    {
        uint index;
        if (!reserve(COUNTER_WRITES, t.write_capacity, 1, index))
            return;
        bool was_global = written && (st.rt_flags & RT_GLOBAL) != 0;
        bool disable = (op.op & OP_NULL) != 0;
        bool global = !disable && op.partition == global_partition();
        PtlasWrite w = (PtlasWrite)0;
        w.instance_index = slot;
        w.partition = ptlas_partition(op.partition);
        if (!disable)
        {
            GroupGpu* g = s.groups + inst.group;
            ModelGpu* model = s.models + inst.model;
            int3 cell = global ? s.origin_cell.xyz : t.partitions[op.partition].cell;
            Placement pl;
            placement(g, g.current, inst.group_index, cell, float3(0.0), uint(s.origin_cell.w), pl);
            w = instance_write(slot, op.partition, pl, model.bounds_min, model.bounds_max, blas, inst.candidate_class);
        }
        store((uint64_t)t.writes, index, w);
        if (global && !was_global)
            InterlockedAdd(t.counters[COUNTER_GLOBAL], 1);
        if (was_global && !global)
            InterlockedAdd(t.counters[COUNTER_GLOBAL], uint(-1));
        st.rt_blas = disable ? 0 : blas;
        st.rt_partition = op.partition;
        st.rt_key = op.key;
        st.rt_flags = (st.rt_flags & ~(RT_WRITTEN | RT_GLOBAL)) | (disable ? 0 : RT_WRITTEN) | (global ? RT_GLOBAL : 0);
    }
    if (held && (op.op & (OP_TAKE | OP_POSE)) != 0)
    {
        // A slot handed out again keeps a run of the same size, but its new record does not
        // name it.
        s.instances[slot].first_part = st.part_run;
        write_parts(s, t, st, inst, op.cut, op.partition);
    }
}

// `ProducerStateGpu.rt_flags` of this frame's decision.
static const uint PRODUCER_RT_WRITE = 4;
static const uint PRODUCER_RT_NULL = 8;

// What a producer instance is written with besides its BLAS address: transform and class.
uint2 producer_key(GroupGpu* g, ProducerGpu p)
{
    float3x4 m = ((float3x4*)g.current)[p.group_index];
    uint a = 0x811c9dc5u;
    uint b = 0x01000193u;
    for (uint r = 0; r < 3; r++)
    {
        for (uint c = 0; c < 4; c++)
        {
            a = mix(a, asuint(m[r][c]));
            b = mix(b, asuint(m[r][c]) + 0x632be5abu);
        }
    }
    return uint2(mix(mix(a, p.class_id), p.group), mix(b, p.group_index ^ (p.class_id << 24)));
}

// The render-space placement of producer entry `p`; false when its group hides it.
bool producer_render_placement(SceneRoot* s, ProducerGpu p, out Placement pl)
{
    return producer_placement(s, p, false, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl);
}

// One thread per producer entry: whether its instance is written or null-written this frame.
[shader("compute")]
[numthreads(64, 1, 1)]
void producer_want(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    uint e = id.x;
    if (e >= s.producer_capacity)
        return;
    ProducerGpu p = s.producers[e];
    ProducerStateGpu* st = s.producer_state + e;
    uint flags = st.rt_flags & ~(PRODUCER_RT_WRITE | PRODUCER_RT_NULL);
    bool written = (flags & PRODUCER_RT_WRITTEN) != 0;
    Placement pl;
    float3 lo, hi;
    bool active = (p.flags & INSTANCE_RAY) != 0 && st.blas != 0 && producer_bounds(*st, lo, hi)
        && producer_render_placement(s, p, pl);
    if (active)
    {
        uint2 key = producer_key(s.groups + p.group, p);
        bool rebased = (root.flags & PTLAS_REBASE) != 0;
        if (!written || (flags & PRODUCER_RT_BUILT) != 0 || rebased || any(st.rt_key != key))
            flags |= PRODUCER_RT_WRITE;
    }
    else if (written)
    {
        flags |= PRODUCER_RT_NULL;
    }
    st.rt_flags = flags;
    if ((flags & (PRODUCER_RT_WRITE | PRODUCER_RT_NULL)) != 0)
        charge_write(global_partition(), true);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void producer_emit(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    PtlasGpu* t = root.ptlas;
    uint e = id.x;
    if (e >= s.producer_capacity)
        return;
    ProducerStateGpu* st = s.producer_state + e;
    uint flags = st.rt_flags;
    if ((flags & (PRODUCER_RT_WRITE | PRODUCER_RT_NULL)) == 0)
    {
        st.rt_flags = flags & ~PRODUCER_RT_BUILT;
        return;
    }
    uint index;
    if (!reserve(COUNTER_WRITES, t.write_capacity, 1, index))
        return;
    ProducerGpu p = s.producers[e];
    PtlasWrite w = (PtlasWrite)0;
    w.instance_index = s.producer_base + e;
    w.partition = PTLAS_GLOBAL_PARTITION;
    bool write = (flags & PRODUCER_RT_WRITE) != 0;
    if (write)
    {
        Placement pl;
        float3 lo, hi;
        producer_render_placement(s, p, pl);
        producer_bounds(*st, lo, hi);
        w = instance_write(w.instance_index, global_partition(), pl, lo, hi, st.blas, p.class_id);
        st.rt_key = producer_key(s.groups + p.group, p);
    }
    store((uint64_t)t.writes, index, w);
    bool was = (flags & PRODUCER_RT_WRITTEN) != 0;
    if (write && !was)
        InterlockedAdd(t.counters[COUNTER_PRODUCERS], 1);
    if (!write && was)
        InterlockedAdd(t.counters[COUNTER_PRODUCERS], uint(-1));
    st.rt_flags = write ? PRODUCER_RT_WRITTEN : 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void translate(uint3 id: SV_DispatchThreadID)
{
    PtlasGpu* t = root.ptlas;
    if (id.x >= root.translate_count)
        return;
    TranslateJob j = root.translate_jobs[id.x];
    PtlasTranslate r;
    r.translation = j.translation;
    r.partition = j.partition;
    store((uint64_t)t.translates, id.x, r);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void commands(uint3 id: SV_DispatchThreadID)
{
    PtlasGpu* t = root.ptlas;
    uint* c = t.counters;
    uint n = 0;
    // Pops past an empty free list took from the pool instead; a refused bump keeps none.
    for (uint k = 0; k < PART_CLASSES; k++)
    {
        if (int(t.part_state[PART_TOPS + k]) < 0)
            t.part_state[PART_TOPS + k] = 0;
    }
    t.part_state[PART_BUMP] = min(t.part_state[PART_BUMP], root.scene.part_capacity);
    uint writes = min(c[COUNTER_WRITES], t.write_capacity);
    uint updates = min(c[COUNTER_UPDATES], t.update_capacity);
    c[COUNTER_TRANSLATES] = root.translate_count;
    PtlasCommand cmd;
    if (writes > 0)
    {
        cmd.args = (uint64_t)t.writes;
        cmd.op = PTLAS_WRITE;
        cmd.count = writes;
        store((uint64_t)t.commands, n++, cmd);
    }
    if (updates > 0)
    {
        cmd.args = (uint64_t)t.updates;
        cmd.op = PTLAS_UPDATE;
        cmd.count = updates;
        store((uint64_t)t.commands, n++, cmd);
    }
    if (root.translate_count > 0)
    {
        cmd.args = (uint64_t)t.translates;
        cmd.op = PTLAS_TRANSLATE;
        cmd.count = root.translate_count;
        store((uint64_t)t.commands, n++, cmd);
    }
    *t.command_count = n;
}
