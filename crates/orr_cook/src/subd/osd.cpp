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
    // Face-varying uvs: value count and one value index per face corner; 0 for none.
    int uv_count;
    const int* uv_indices;
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
    ORR_OSD_UV_STENCIL_SIZES,
    ORR_OSD_UV_STENCIL_INDICES,
    ORR_OSD_UV_STENCIL_WEIGHTS,
    ORR_OSD_UV_PATCH_TYPES,
    ORR_OSD_UV_PATCH_CVS,
    ORR_OSD_UV_PATCH_PARAMS,
    ORR_OSD_TABLES
};

// Patch types: 16-point B-spline, 20-point Gregory basis, 4-point bilinear.
enum { ORR_OSD_REGULAR = 0, ORR_OSD_GREGORY = 1, ORR_OSD_LINEAR = 2 };

struct orr_osd {
    std::vector<int> ints[ORR_OSD_TABLES];
    std::vector<float> floats[ORR_OSD_TABLES];
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
    Far::TopologyDescriptor::FVarChannel uv;
    if (c.uv_count > 0) {
        uv.numValues = c.uv_count;
        uv.valueIndices = c.uv_indices;
        d.numFVarChannels = 1;
        d.fvarChannels = &uv;
    }
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
    a.considerFVarChannels = c.uv_count > 0;
    r->RefineAdaptive(a);
    return r;
}

// Stencils for every point (cage points as single-weight stencils, refined points, then
// local points) of the vertex data or of the uv channel, in terms of the cage.
void stencils(const Refiner& r, const Far::StencilTable* local, bool uv, orr_osd& out,
              int sizes) {
    Far::StencilTableFactory::Options o;
    o.generateIntermediateLevels = true;
    o.generateControlVerts = true;
    o.generateOffsets = true;
    o.interpolationMode = uv ? Far::StencilTableFactory::INTERPOLATE_FACE_VARYING
                             : Far::StencilTableFactory::INTERPOLATE_VERTEX;
    std::unique_ptr<const Far::StencilTable> refined(Far::StencilTableFactory::Create(r, o));
    std::unique_ptr<const Far::StencilTable> all;
    if (local && local->GetNumStencils() > 0) {
        all.reset(uv ? Far::StencilTableFactory::AppendLocalPointStencilTableFaceVarying(
                           r, refined.get(), local)
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
    const bool uv = c.uv_count > 0;
    Far::PatchTableFactory::Options o(c.isolation);
    o.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
    o.useInfSharpPatch = true;
    o.includeBaseLevelIndices = true;
    o.generateVaryingTables = false;
    o.generateFVarTables = uv;
    o.includeFVarBaseLevelIndices = true;
    o.generateFVarLegacyLinearPatches = false;
    std::unique_ptr<Far::PatchTable> t(Far::PatchTableFactory::Create(*r, o));
    if (!t) {
        g_error = "OpenSubdiv built no patch table";
        return false;
    }
    stencils(*r, t->GetLocalPointStencilTable(), false, out, ORR_OSD_STENCIL_SIZES);
    if (uv) {
        stencils(*r, t->GetLocalPointFaceVaryingStencilTable(0), true, out,
                 ORR_OSD_UV_STENCIL_SIZES);
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
            if (!uv) {
                continue;
            }
            Far::PatchParam fp = t->GetPatchFVarPatchParam(a, p, 0);
            Far::PatchDescriptor d = fp.IsRegular() ? t->GetFVarPatchDescriptorRegular(0)
                                                    : t->GetFVarPatchDescriptorIrregular(0);
            int uv_type = patch_type(d.GetType());
            if (uv_type < 0) {
                g_error = "OpenSubdiv made a uv patch type the cook does not store";
                return false;
            }
            out.ints[ORR_OSD_UV_PATCH_TYPES].push_back(uv_type);
            Far::ConstIndexArray v = t->GetPatchFVarValues(a, p, 0);
            out.ints[ORR_OSD_UV_PATCH_CVS].insert(out.ints[ORR_OSD_UV_PATCH_CVS].end(), v.begin(),
                                                  v.end());
            param(fp, out.ints[ORR_OSD_UV_PATCH_PARAMS]);
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

// OpenSubdiv's limit positions (and uvs, when the cage has them) at `count` ptex locations,
// from its own patch table of the same refinement. Returns 0 on failure.
int orr_osd_limit(const orr_osd_cage* cage, const float* points, const float* uvs, int count,
                  const int* ptex, const float* s, const float* t, float* out_points,
                  float* out_uvs, char* error, int error_len) {
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
        if (uvs && cage->uv_count > 0) {
            o.interpolationMode = Far::LimitStencilTableFactory::INTERPOLATE_FACE_VARYING;
            std::unique_ptr<const Far::LimitStencilTable> uv(
                Far::LimitStencilTableFactory::Create(*r, at, nullptr, nullptr, o));
            if (!uv) {
                g_error = "OpenSubdiv made no uv limit stencils";
                take_error(error, error_len);
                return 0;
            }
            uv->UpdateValues(reinterpret_cast<const Point<2>*>(uvs),
                            reinterpret_cast<Point<2>*>(out_uvs));
        }
        return 1;
    } catch (const std::exception& e) {
        g_error = e.what();
    }
    take_error(error, error_len);
    return 0;
}

}  // extern "C"
