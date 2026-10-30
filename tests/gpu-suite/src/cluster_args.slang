// Writes `n` rtas records of one kind from CPU-written logical records, resolving addresses
// that earlier operations produced on the GPU through `table`, and the record count.
#include "orr/root.h"
import rtas;

static const uint OP_CLAS = 0;
static const uint OP_TEMPLATE = 1;
static const uint OP_INSTANTIATE = 2; // template_address: index into table
static const uint OP_GATHER = 3;      // refs[i] = table[gather[i]]
static const uint OP_BLAS = 4;        // clas_addresses: first index into refs
static const uint OP_CLASSIC = 5;     // blas: index into table
static const uint OP_WRITE = 6;       // blas: index into table, NO_BLAS for none
static const uint OP_UPDATE = 7;      // blas: index into table
static const uint OP_TRANSLATE = 8;
static const uint OP_COMMAND = 9;
static const uint OP_MOVE = 10;       // src: index into table
static const uint OP_POOL = 11;       // bump-allocates sizes[i] into the pool, writes addresses
static const uint64_t NO_BLAS = 0xffffffff;

struct Root
{
    uint64_t dst;
    uint* count;
    uint64_t* table;
    uint64_t* refs;
    uint* gather;
    ClasBuild* clas;
    TemplateBuild* templates;
    Instantiate* instantiate;
    ClusterBlas* blas;
    ClassicInstance* classic;
    PtlasWrite* writes;
    PtlasUpdate* updates;
    PtlasTranslate* translates;
    PtlasCommand* commands;
    Move* moves;
    uint* sizes;
    uint* pool_used; // bytes taken from the pool so far
    uint64_t pool;
    uint n;
    uint op;
    uint64_t pad;
}
ORR_ROOT(Root);

uint64_t lookup(uint64_t index)
{
    return index == NO_BLAS ? 0 : root.table[uint(index)];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i == 0 && root.count != nullptr)
        *root.count = root.n;
    if (i >= root.n)
        return;
    switch (root.op)
    {
    case OP_CLAS:
        store(root.dst, i, root.clas[i]);
        break;
    case OP_TEMPLATE:
        store(root.dst, i, root.templates[i]);
        break;
    case OP_INSTANTIATE:
    {
        Instantiate v = root.instantiate[i];
        v.template_address = lookup(v.template_address);
        store(root.dst, i, v);
        break;
    }
    case OP_GATHER:
        root.refs[i] = root.table[root.gather[i]];
        break;
    case OP_BLAS:
    {
        ClusterBlas v = root.blas[i];
        v.clas_addresses = uint64_t(root.refs) + v.clas_addresses * 8;
        store(root.dst, i, v);
        break;
    }
    case OP_CLASSIC:
    {
        ClassicInstance v = root.classic[i];
        v.blas = lookup(v.blas);
        store(root.dst, i, v);
        break;
    }
    case OP_WRITE:
    {
        PtlasWrite v = root.writes[i];
        v.blas = lookup(v.blas);
        store(root.dst, i, v);
        break;
    }
    case OP_UPDATE:
    {
        PtlasUpdate v = root.updates[i];
        v.blas = lookup(v.blas);
        store(root.dst, i, v);
        break;
    }
    case OP_TRANSLATE:
        store(root.dst, i, root.translates[i]);
        break;
    case OP_COMMAND:
        store(root.dst, i, root.commands[i]);
        break;
    case OP_MOVE:
    {
        Move v = root.moves[i];
        v.src = lookup(v.src);
        store(root.dst, i, v);
        break;
    }
    case OP_POOL:
    {
        uint offset;
        InterlockedAdd(*root.pool_used, (root.sizes[i] + 127) & ~127u, offset);
        root.refs[i] = root.pool + offset;
        break;
    }
    default:
        break;
    }
}
