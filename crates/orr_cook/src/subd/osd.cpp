// A C interface over OpenSubdiv Far: feature-adaptive patch tables with stencils that
// express every patch point in terms of the cage, and limit stencils.

#include <opensubdiv/far/error.h>
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/stencilTable.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/topologyDescriptor.h>

#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace OpenSubdiv;

extern "C" {

struct orr_osd_cage {
    int vertex_count;
    int face_count;
    const int* face_counts;
    const int* face_indices;
    int crease_count;
    const int* crease_pairs;
    const float* crease_sharpness;
    int corner_count;
    const int* corners;
    const float* corner_sharpness;
    int hole_count;
    const int* holes;
    int left_handed;
    // Sdc::Options::VtxBoundaryInterpolation, FVarLinearInterpolation and
    // TriangleSubdivision codes.
    int boundary;
    int uv_linear;
    int triangles;
    // Face-varying channels: each one's value count, and one after another their value index
    // per face corner.
    int fvar_count;
    const int* fvar_values;
    const int* fvar_indices;
    int isolation;
    int smooth_isolation;
};

enum {
    ORR_OSD_STENCIL_SIZES = 0,
    ORR_OSD_STENCIL_INDICES,
    ORR_OSD_STENCIL_WEIGHTS,
    ORR_OSD_PATCH_TYPES,
    ORR_OSD_PATCH_CVS,
    ORR_OSD_PATCH_PARAMS,
    // The same six tables of each face-varying channel follow.
    ORR_OSD_TABLES_PER_CHANNEL
};

// Patch types: 16-point B-spline, 20-point Gregory basis, 4-point bilinear.
enum { ORR_OSD_REGULAR = 0, ORR_OSD_GREGORY = 1, ORR_OSD_LINEAR = 2 };

struct orr_osd {
    std::vector<std::vector<int>> ints;
    std::vector<std::vector<float>> floats;
};

}  // extern "C"

namespace {

thread_local std::string g_error;

void on_error(Far::ErrorType, const char* message) {
    if (g_error.empty()) {
        g_error = message;
    }
}

void on_warning(const char*) {}

using Refiner = Far::TopologyRefiner;

std::unique_ptr<Refiner> refine(const orr_osd_cage& c) {
    Far::SetErrorCallback(on_error);
    Far::SetWarningCallback(on_warning);
    Far::TopologyDescriptor d;
    d.numVertices = c.vertex_count;
    d.numFaces = c.face_count;
    d.numVertsPerFace = c.face_counts;
    d.vertIndicesPerFace = c.face_indices;
    d.numCreases = c.crease_count;
    d.creaseVertexIndexPairs = c.crease_pairs;
    d.creaseWeights = c.crease_sharpness;
    d.numCorners = c.corner_count;
    d.cornerVertexIndices = c.corners;
    d.cornerWeights = c.corner_sharpness;
    d.numHoles = c.hole_count;
    d.holeIndices = c.holes;
    d.isLeftHanded = c.left_handed != 0;
    const int corners = std::accumulate(c.face_counts, c.face_counts + c.face_count, 0);
    std::vector<Far::TopologyDescriptor::FVarChannel> channels(c.fvar_count);
    for (int k = 0; k < c.fvar_count; ++k) {
        channels[k].numValues = c.fvar_values[k];
        channels[k].valueIndices = c.fvar_indices + std::size_t(k) * corners;
    }
    d.numFVarChannels = c.fvar_count;
    d.fvarChannels = channels.data();
    Sdc::Options sdc;
    sdc.SetVtxBoundaryInterpolation(Sdc::Options::VtxBoundaryInterpolation(c.boundary));
    sdc.SetFVarLinearInterpolation(Sdc::Options::FVarLinearInterpolation(c.uv_linear));
    sdc.SetTriangleSubdivision(Sdc::Options::TriangleSubdivision(c.triangles));
    using Factory = Far::TopologyRefinerFactory<Far::TopologyDescriptor>;
    std::unique_ptr<Refiner> r(Factory::Create(d, Factory::Options(Sdc::SCHEME_CATMARK, sdc)));
    if (!r) {
        if (g_error.empty()) {
            g_error = "the cage topology is invalid";
        }
        return nullptr;
    }
    Refiner::AdaptiveOptions a(c.isolation);
    a.SetSecondaryLevel(c.smooth_isolation);
    a.useInfSharpPatch = true;
    a.considerFVarChannels = c.fvar_count > 0;
    r->RefineAdaptive(a);
    return r;
}

// Stencils for every point (cage points as single-weight stencils, refined points, then
// local points) of the vertex data or of face-varying channel `channel`, in terms of the cage.
void stencils(const Refiner& r, const Far::StencilTable* local, int channel, orr_osd& out,
              int sizes) {
    const bool fvar = channel >= 0;
    Far::StencilTableFactory::Options o;
    o.generateIntermediateLevels = true;
    o.generateControlVerts = true;
    o.generateOffsets = true;
    o.interpolationMode = fvar ? Far::StencilTableFactory::INTERPOLATE_FACE_VARYING
                               : Far::StencilTableFactory::INTERPOLATE_VERTEX;
    o.fvarChannel = fvar ? channel : 0;
    std::unique_ptr<const Far::StencilTable> refined(Far::StencilTableFactory::Create(r, o));
    std::unique_ptr<const Far::StencilTable> all;
    if (local && local->GetNumStencils() > 0) {
        all.reset(fvar ? Far::StencilTableFactory::AppendLocalPointStencilTableFaceVarying(
                             r, refined.get(), local, channel)
                       : Far::StencilTableFactory::AppendLocalPointStencilTable(r, refined.get(),
                                                                               local));
    }
    const Far::StencilTable& t = all ? *all : *refined;
    out.ints[sizes] = t.GetSizes();
    out.ints[sizes + 1].assign(t.GetControlIndices().begin(), t.GetControlIndices().end());
    out.floats[sizes + 2] = t.GetWeights();
}

int patch_type(Far::PatchDescriptor::Type t) {
    switch (t) {
    case Far::PatchDescriptor::REGULAR:
        return ORR_OSD_REGULAR;
    case Far::PatchDescriptor::GREGORY_BASIS:
        return ORR_OSD_GREGORY;
    case Far::PatchDescriptor::QUADS:
        return ORR_OSD_LINEAR;
    default:
        return -1;
    }
}

void param(const Far::PatchParam& p, std::vector<int>& out) {
    out.push_back(int(p.field0));
    out.push_back(int(p.field1));
}

bool build(const orr_osd_cage& c, orr_osd& out) {
    std::unique_ptr<Refiner> r = refine(c);
    if (!r) {
        return false;
    }
    const int tables = ORR_OSD_TABLES_PER_CHANNEL * (c.fvar_count + 1);
    out.ints.resize(tables);
    out.floats.resize(tables);
    Far::PatchTableFactory::Options o(c.isolation);
    o.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
    o.useInfSharpPatch = true;
    o.includeBaseLevelIndices = true;
    o.generateVaryingTables = false;
    o.generateFVarTables = c.fvar_count > 0;
    o.includeFVarBaseLevelIndices = true;
    o.generateFVarLegacyLinearPatches = false;
    std::unique_ptr<Far::PatchTable> t(Far::PatchTableFactory::Create(*r, o));
    if (!t) {
        g_error = "OpenSubdiv built no patch table";
        return false;
    }
    stencils(*r, t->GetLocalPointStencilTable(), -1, out, ORR_OSD_STENCIL_SIZES);
    for (int k = 0; k < c.fvar_count; ++k) {
        stencils(*r, t->GetLocalPointFaceVaryingStencilTable(k), k, out,
                 ORR_OSD_TABLES_PER_CHANNEL * (k + 1));
    }
    for (int a = 0; a < t->GetNumPatchArrays(); ++a) {
        int type = patch_type(t->GetPatchArrayDescriptor(a).GetType());
        if (type < 0) {
            g_error = "OpenSubdiv made a patch type the cook does not store";
            return false;
        }
        for (int p = 0; p < t->GetNumPatches(a); ++p) {
            out.ints[ORR_OSD_PATCH_TYPES].push_back(type);
            Far::ConstIndexArray cvs = t->GetPatchVertices(a, p);
            out.ints[ORR_OSD_PATCH_CVS].insert(out.ints[ORR_OSD_PATCH_CVS].end(), cvs.begin(),
                                               cvs.end());
            param(t->GetPatchParam(a, p), out.ints[ORR_OSD_PATCH_PARAMS]);
            for (int k = 0; k < c.fvar_count; ++k) {
                const int base = ORR_OSD_TABLES_PER_CHANNEL * (k + 1);
                Far::PatchParam fp = t->GetPatchFVarPatchParam(a, p, k);
                Far::PatchDescriptor d = fp.IsRegular() ? t->GetFVarPatchDescriptorRegular(k)
                                                        : t->GetFVarPatchDescriptorIrregular(k);
                int fvar_type = patch_type(d.GetType());
                if (fvar_type < 0) {
                    g_error = "OpenSubdiv made a face-varying patch type the cook does not store";
                    return false;
                }
                out.ints[base + ORR_OSD_PATCH_TYPES].push_back(fvar_type);
                Far::ConstIndexArray v = t->GetPatchFVarValues(a, p, k);
                std::vector<int>& cvs_out = out.ints[base + ORR_OSD_PATCH_CVS];
                cvs_out.insert(cvs_out.end(), v.begin(), v.end());
                param(fp, out.ints[base + ORR_OSD_PATCH_PARAMS]);
            }
        }
    }
    return true;
}

void take_error(char* error, int len) {
    if (error && len > 0) {
        std::strncpy(error, g_error.c_str(), size_t(len) - 1);
        error[len - 1] = 0;
    }
    g_error.clear();
}

// A point of N floats, as Far's UpdateValues reads and writes them.
template <int N>
struct Point {
    float v[N];
    void Clear() {
        for (int k = 0; k < N; ++k) v[k] = 0;
    }
    void AddWithWeight(const Point& p, float w) {
        for (int k = 0; k < N; ++k) v[k] += w * p.v[k];
    }
};

// OpenSubdiv's limit of N-wide values of face-varying channel 0 at `at`. It reads no other:
// LimitStencilTableFactory puts the one channel it is asked for in a patch table, then asks
// that table for the channel by its index in the refiner.
template <int N>
bool fvar_limit(const Refiner& r, const Far::LimitStencilTableFactory::LocationArrayVec& at,
                const float* values, float* out) {
    Far::LimitStencilTableFactory::Options o;
    o.generate1stDerivatives = false;
    o.interpolationMode = Far::LimitStencilTableFactory::INTERPOLATE_FACE_VARYING;
    std::unique_ptr<const Far::LimitStencilTable> t(
        Far::LimitStencilTableFactory::Create(r, at, nullptr, nullptr, o));
    if (!t) {
        g_error = "OpenSubdiv made no face-varying limit stencils";
        return false;
    }
    t->UpdateValues(reinterpret_cast<const Point<N>*>(values), reinterpret_cast<Point<N>*>(out));
    return true;
}

}  // namespace

extern "C" {

orr_osd* orr_osd_build(const orr_osd_cage* cage, char* error, int error_len) {
    g_error.clear();
    try {
        std::unique_ptr<orr_osd> out(new orr_osd);
        if (build(*cage, *out)) {
            g_error.clear();
            return out.release();
        }
    } catch (const std::exception& e) {
        g_error = e.what();
    }
    take_error(error, error_len);
    return nullptr;
}

void orr_osd_free(orr_osd* osd) { delete osd; }

// The table's length; its ints or floats in `data`.
int orr_osd_table(const orr_osd* osd, int table, const void** data) {
    if (!osd->floats[table].empty()) {
        *data = osd->floats[table].data();
        return int(osd->floats[table].size());
    }
    *data = osd->ints[table].data();
    return int(osd->ints[table].size());
}

// OpenSubdiv's limit positions, and when given values `fvar_width` wide of channel 0, at `count`
// ptex locations, from its own patch table of the same refinement. Returns 0 on failure.
int orr_osd_limit(const orr_osd_cage* cage, const float* points, const float* fvar,
                  int fvar_width, int count, const int* ptex, const float* s, const float* t,
                  float* out_points, float* out_fvar, char* error, int error_len) {
    g_error.clear();
    try {
        std::unique_ptr<Refiner> r = refine(*cage);
        if (!r) {
            take_error(error, error_len);
            return 0;
        }
        Far::LimitStencilTableFactory::LocationArrayVec at(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            at[i].ptexIdx = ptex[i];
            at[i].numLocations = 1;
            at[i].s = &s[i];
            at[i].t = &t[i];
        }
        Far::LimitStencilTableFactory::Options o;
        o.generate1stDerivatives = false;
        std::unique_ptr<const Far::LimitStencilTable> pos(
            Far::LimitStencilTableFactory::Create(*r, at, nullptr, nullptr, o));
        if (!pos) {
            g_error = "OpenSubdiv made no limit stencils";
            take_error(error, error_len);
            return 0;
        }
        pos->UpdateValues(reinterpret_cast<const Point<3>*>(points),
                         reinterpret_cast<Point<3>*>(out_points));
        if (fvar && !(fvar_width == 2 ? fvar_limit<2>(*r, at, fvar, out_fvar)
                                      : fvar_limit<4>(*r, at, fvar, out_fvar))) {
            take_error(error, error_len);
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_error = e.what();
    }
    take_error(error, error_len);
    return 0;
}

}  // extern "C"
