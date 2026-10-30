// Stores record `i` of every rtas type at index `i` of its own output array, and the
// partition translations of `cells`.
#include "orr/root.h"
import rtas;

struct Translation
{
    int3 cell;
    uint cell_log2;
    int3 origin_cell;
    uint pad;
}

struct Root
{
    ClasBuild* clas;
    TemplateBuild* templates;
    Instantiate* instantiate;
    ClusterBlas* blas;
    Move* moves;
    PtlasWrite* writes;
    PtlasUpdate* updates;
    PtlasTranslate* translates;
    PtlasCommand* commands;
    ClassicInstance* classic;
    Translation* cells;
    float3* translations;
    uint64_t out_clas;
    uint64_t out_templates;
    uint64_t out_instantiate;
    uint64_t out_blas;
    uint64_t out_moves;
    uint64_t out_writes;
    uint64_t out_updates;
    uint64_t out_translates;
    uint64_t out_commands;
    uint64_t out_classic;
    uint records;
    uint cell_count;
    uint64_t pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i < root.records)
    {
        store(root.out_clas, i, root.clas[i]);
        store(root.out_templates, i, root.templates[i]);
        store(root.out_instantiate, i, root.instantiate[i]);
        store(root.out_blas, i, root.blas[i]);
        store(root.out_moves, i, root.moves[i]);
        store(root.out_writes, i, root.writes[i]);
        store(root.out_updates, i, root.updates[i]);
        store(root.out_translates, i, root.translates[i]);
        store(root.out_commands, i, root.commands[i]);
        store(root.out_classic, i, root.classic[i]);
    }
    if (i < root.cell_count)
    {
        Translation t = root.cells[i];
        root.translations[i] = partition_translation(t.cell, t.origin_cell, t.cell_log2);
    }
}
