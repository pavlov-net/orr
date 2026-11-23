// Manifold next-event estimation in the path tracer (Hanika et al. 2015; Zeltner et al. 2020): a
// directional light reaches a point through the surfaces of solid dielectrics along the chain of
// refractions whose last direction is the light's. A walk leaves the point's ray origin along a
// start direction; the walks' stage traces a face ray to each surface on the way and one past the
// last. A surface's hit resolves it with how its normal and alpha change about the hit
// (`SurfaceChange`: across a triangle of a material that shades by its vertex normals, the
// interpolation of its corners' normals, exact) and calls the face step here, which refracts the
// chain about the surface's microfacet normal there: its shading normal at the point the leg meets
// its plane, tilted by the walk's slope for it at the alpha of its BSDF (`facet`). The step carries
// the chain's differentials with respect to the start direction through the surface. Where the
// chain meets nothing more the walks' stage calls the end step, which compares the chain's last
// direction with the light's: where they agree within MANIFOLD_TOLERANCE the chain is the light's
// way to the point, and the solid angle of the start direction over the light's is 1 / |det J|, J
// the Jacobian of the last direction in the start direction, the generalized geometric term over
// the point's cosine. Where they differ, a chain across one surface starts again where that
// surface's model solves it (`model_aim`), which is exact where the chain meets the triangle the
// model holds, and any other by a Newton step.
//
// The first walk starts from the probe, a face ray along the light's direction bent at the point
// across planes facing its normal (or the light's own): the chain across its surface alone where
// that leaves every interior, by the surface's model, sent again toward where the model puts the
// chain while that lies past its triangle (MANIFOLD_PROBES), else the light's direction bent at the
// point across planes parallel to the probe's microfacet. An estimate weighs the chain of each
// walk likely to converge (`WALK_WEIGH`), and of the converged one where it did not: at each
// surface its BSDF toward the light over the density of the microfacet normal the slope drew, and
// the media along each leg, while the walks' stage traces each leg's shadow ray.
//
// It takes surfaces of every roughness: a mirror refracts about its shading normal and weighs by
// its delta lobe's coefficient, and any other draws its slope from GGX at its BSDF's alpha and
// weighs by the BSDF itself, multiple scattering, coat and anisotropy among it, so the light it
// brings changes smoothly with roughness, across CHAIN_ROUGHNESS too, where only what carries the
// light it does not find changes (the photons below, BSDF rays above).
//
// A walk is a function of its start, the light's direction and its slopes alone, so a ray or a
// photon that crossed the same surfaces tells whether an estimate there owns its light by walking
// from where its track starts with the slopes it crossed at, which stops as soon as a walk would
// start along the ray's or photon's own chain (`check_walk`). An estimate owns its chains whole.
module manifold;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.pipeline;
import orr_shaders.sampling;
import orr_scene.hit;
import orr_scene.lights;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.interiors;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.state;
import orr_lighting.path.vertex;
__exported import orr_lighting.path.facet;

// The interior list `slot`'s walk in manifold words `at` starts in, as `settle_list` left it:
// none, the path's cold list, or the list word's.
public Interior start_list(PathRoot r, uint slot, uint4* at)
{
    uint source = load_coherent(at + MANIFOLD_FROM_WORD).w & MANIFOLD_LIST_MASK;
    if (source == MANIFOLD_LIST_NONE)
        return Interior();
    return load_interior(source == MANIFOLD_LIST_COLD ? path_cold(r, slot) : at + MANIFOLD_LIST_WORD);
}

// The interior list `slot`'s walk starts in, which its manifold words `at` say where to find
// (MANIFOLD_LIST_*): a job's own, across its layer's surface where its light comes through it; the
// path's cold list as it is or out of the interior a walk of the path's goes through; or that list
// back across the surfaces the path's track crossed. One that takes that work it leaves in their
// list word for the walk's steps (`start_list`), which read any other where it is.
public Interior settle_list(PathRoot r, uint slot, uint4* at)
{
    uint4 from = load_coherent(at + MANIFOLD_FROM_WORD);
    uint source = from.w & MANIFOLD_LIST_MASK;
    if (source == MANIFOLD_LIST_NONE)
        return Interior();
    Interior list;
    if (source == MANIFOLD_LIST_JOB || source == MANIFOLD_LIST_PAST)
        list = load_interior(at + MANIFOLD_LIST_WORD);
    else
        list = load_interior(path_cold(r, slot));
    if (source == MANIFOLD_LIST_JOB || source == MANIFOLD_LIST_COLD)
        return list;
    if (source == MANIFOLD_LIST_PAST)
    {
        uint across = load_coherent(at + MANIFOLD_BENT_WORD + 3).w;
        list.cross(across & ~MANIFOLD_ENTERS, (across & MANIFOLD_ENTERS) != 0, asfloat(load_coherent(at + WALK_MEDIUM)));
    }
    else if (source == MANIFOLD_LIST_COLD_OUT)
        list.cross(load_coherent(path_cold(r, slot) + COLD_WALK_WORD).w, false);
    else if (source == MANIFOLD_LIST_TRACK)
        list = track_start(at, list);
    store_list(at + MANIFOLD_LIST_WORD, list);
    store_coherent(at + MANIFOLD_FROM_WORD, uint4(from.xyz, (from.w & ~MANIFOLD_LIST_MASK) | (list.count != 0 ? MANIFOLD_LIST_JOB : MANIFOLD_LIST_NONE)));
    return list;
}

// The sphere, point or rect light whose point the target of the walk in manifold words `at` is,
// as its from word holds it: false for a walk toward a direction.
public bool walk_light(uint4* at, out uint id)
{
    uint bits = load_coherent(at + MANIFOLD_FROM_WORD).w >> MANIFOLD_LIGHT_SHIFT;
    id = bits - 1;
    return bits != 0;
}

// Whether next-event estimation at the point the walk of `slot` starts from, which its from word
// holds, draws the point its target is on a sphere, point or rect light, as a check of whose a
// chain to that point is asks: a point on the light that the straight way from the vertex sees, a
// sphere's within the cone it subtends there and a rect's on its front, where the light's shaping
// sends light that way. Any other target is drawn.
public bool point_drawn(PathRoot r, uint slot)
{
    uint4* at = path_manifold(r, slot);
    return drawn_from(r, at, asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz), asfloat(load_coherent(at + WALK_TARGET).xyz));
}

// Whether an estimate at `x` of the light the walk in manifold words `at` goes toward draws the
// point `y` on it, as `point_drawn` asks.
public bool drawn_from(PathRoot r, uint4* at, float3 x, float3 y)
{
    uint id;
    if (!walk_light(at, id))
        return true;
    LightGpu* l = r.scene.lights + id;
    float3 leaves = normalize(x - y);
    if (!any(light_shaping(l, leaves) > 0.0))
        return false;
    if (l.kind == LIGHT_RECT)
        return dot(cross(l.edge_x, l.edge_y), leaves) > 0.0;
    return l.kind != LIGHT_SPHERE || dot(y - l.position, leaves) > 0.0;
}

// Weighs the converged chain of `slot`'s estimate toward a point on a sphere, point or rect light,
// whose through word holds the chain's density, the solid angle of its start direction over the
// area across its last leg at the point: what the light emits back along that leg, shaped, from
// the side the leg reaches, and for a sphere's or rect's point the area of the light there over
// that across the leg and the solid angle the straight way from the vertex sees it in over that
// area, which turn the pick's density over the solid angle of the straight way, which its job's
// light took, into one over the light's area. A point light's job took its intensity over its
// squared distance, and the chain's density replaces that distance.
public void point_end(PathRoot r, uint slot)
{
    uint4* at = path_manifold(r, slot);
    uint id;
    if (!walk_light(at, id))
        return;
    LightGpu* l = r.scene.lights + id;
    float3 x = asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz);
    float3 y = asfloat(load_coherent(at + WALK_TARGET).xyz);
    float3 leaves = -asfloat(load_coherent(at + WALK_DIR).xyz);
    float3 emitted = l.color;
    float area = 1.0;
    if (l.kind != LIGHT_POINT)
    {
        float r2;
        float3 sphere;
        float3 n = normalize(cross(l.edge_x, l.edge_y));
        if (sphere_light(*l, r2, sphere))
        {
            emitted = sphere;
            n = normalize(y - l.position);
        }
        float3 straight = x - y;
        float d2 = dot(straight, straight);
        float cos_straight = dot(n, straight) * rsqrt(d2);
        float cos_leg = dot(n, leaves);
        area = cos_leg > 0.0 ? cos_leg * d2 / abs(cos_straight) : 0.0;
    }
    uint4* through = at + MANIFOLD_THROUGH_WORD;
    float4 t = asfloat(load_coherent(through));
    store_coherent(through, asuint(float4(t.xyz * emitted * light_shaping(l, leaves), t.w * area)));
}

// Starts the next walk of the walk in manifold words `at` along unit `aim`, as `phase` (WALK_SOLVE
// or WALK_WEIGH) with what `state` says it estimates, inside `list`: its chain from the surface
// point its from word holds, its rays from `origin` off it, the first toward the point `toward`
// where it `aims`, where a surface's model says the chain meets it, so the ray finds that surface
// however far the origin lies off the point; and the code `kept` of the start direction of the
// walk its solve accepted (WALK_ACCEPTED in `state`). False where its walks have run out
// (MANIFOLD_WALKS), but where it walks a `converged` chain again to weigh it.
public bool restart(PathRoot r, uint4* at, float3 origin, float3 aim, uint phase, uint state, Interior list, bool converged = false, float3 toward = float3(0.0), bool aims = false, uint kept = 0)
{
    uint begun = (state >> WALK_BEGUN_SHIFT) & 0xf;
    if (begun >= MANIFOLD_WALKS && !converged)
        return false;
    begun++;
    if (phase == WALK_WEIGH)
        store_coherent(at + MANIFOLD_THROUGH_WORD, asuint(float4(1.0, 1.0, 1.0, 0.0)));
    store_coherent(at + WALK_START, uint4(asuint(origin), phase | (state & WALK_KEPT) | (begun << WALK_BEGUN_SHIFT)));
    uint4 target = load_coherent(at + WALK_TARGET);
    store_coherent(at + WALK_TARGET, uint4(target.xyz, asuint(interior_ior(r.scene, list.innermost()))));
    float3 t, b;
    orthonormal_basis(aim, t, b);
    store_coherent(at + MANIFOLD_BEND_WORD, uint4(asuint(aim), kept));
    store_coherent(at + WALK_POINT, uint4(load_coherent(at + MANIFOLD_FROM_WORD).xyz, 0));
    store_coherent(at + WALK_DIR, uint4(asuint(aim), 0));
    store_differentials(at, float3(0.0), float3(0.0), t, b);
    store_list(at + WALK_LIST, list);
    store_coherent(at + WALK_MODEL, uint4(asuint(toward), aims ? 2 : 0));
    return true;
}

// The chain's differentials in manifold words `at`: its point's along the start direction's basis,
// and its direction's.
void load_differentials(uint4* at, out float3 pu, out float3 pv, out float3 du, out float3 dv)
{
    uint4 a = load_coherent(at + WALK_DIFF);
    uint4 b = load_coherent(at + WALK_DIFF + 1);
    uint4 c = load_coherent(at + WALK_DIFF + 2);
    pu = asfloat(a.xyz);
    pv = asfloat(uint3(a.w, b.x, b.y));
    du = asfloat(uint3(b.z, b.w, c.x));
    dv = asfloat(c.yzw);
}

void store_differentials(uint4* at, float3 pu, float3 pv, float3 du, float3 dv)
{
    store_coherent(at + WALK_DIFF, uint4(asuint(pu), asuint(pv.x)));
    store_coherent(at + WALK_DIFF + 1, uint4(asuint(pv.yz), asuint(du.xy)));
    store_coherent(at + WALK_DIFF + 2, uint4(asuint(du.z), asuint(dv)));
}

// How the microfacet normal `m` of slope `alpha` x `slope` about unit normal `n` turns as the
// normal moves by `dn` and the alpha by `da`.
float3 facet_change(float3 n, float3 m, float alpha, float2 slope, float3 dn, float da)
{
    float h = 1e-3 / max(max(length(dn), abs(da) * length(slope)), 1e-3);
    return (facet(normalize(n + dn * h), (alpha + da * h) * slope) - m) / h;
}

// Where the chain from `from` that crosses the surface of face step `c` alone to `target`, a point
// where `point`, else a direction, on the surface's model about the step's hit (`SurfaceChange`)
// meets the hit's plane, of basis `tx`, `bx` (`orthonormal_basis` of its normal): Newton on the
// plane from `x`, a point on it near the chain, on the generalized half vector (Walter et al. 2007;
// Hanika et al. 2015): the chain refracts there by `eta` (the IORs before and past the surface, as
// a ratio) about the microfacet normal of slope `slope` at the model's alpha, `alpha0` at the hit,
// or about the normal of a `mirror`, where `eta` times the way back toward `from` plus the way on
// toward the target lies along that normal. That constraint stays smooth up to grazing, where the
// refracted direction a step would compare turns past the critical angle within a fraction of a
// degree. False where it meets no such chain within MODEL_STEPS steps.
bool model_aim(ManifoldStep c, float3 tx, float3 bx, float3 from, float3 x, float eta, bool mirror, float alpha0, float2 slope, float3 target, bool point, out float3 root)
{
    root = float3(0.0);
    float3 hit = c.at.xyz;
    float2 uv = float2(dot(x - hit, tx), dot(x - hit, bx));
    // The constraint's error goes in the basis of the microfacet normal where the solve starts,
    // which the surface's model turns little.
    float3 tl = float3(0.0);
    float3 bl = float3(0.0);
    for (uint i = 0; i < MODEL_STEPS; i++)
    {
        float3 q = hit + uv.x * tx + uv.y * bx;
        float3 w = from - q;
        float lw = length(w);
        float3 wi = w / lw;
        float3 iu = (wi * dot(wi, tx) - tx) / lw;
        float3 iv = (wi * dot(wi, bx) - bx) / lw;
        float3 wo = target;
        float3 ou = float3(0.0);
        float3 ov = float3(0.0);
        if (point)
        {
            float3 e = target - q;
            float le = length(e);
            wo = e / le;
            ou = (wo * dot(wo, tx) - tx) / le;
            ov = (wo * dot(wo, bx) - bx) / le;
        }
        float3 raw = c.normal.xyz + c.gradient_t.xyz * uv.x + c.gradient_b.xyz * uv.y;
        float ln = length(raw);
        float3 n = raw / ln;
        float3 nu = (c.gradient_t.xyz - n * dot(n, c.gradient_t.xyz)) / ln;
        float3 nv = (c.gradient_b.xyz - n * dot(n, c.gradient_b.xyz)) / ln;
        float alpha = mirror ? 0.0 : alpha0 + c.gradient_t.w * uv.x + c.gradient_b.w * uv.y;
        float3 mf = facet(n, alpha * slope);
        float3 mu = facet_change(n, mf, alpha, slope, nu, mirror ? 0.0 : c.gradient_t.w);
        float3 mv = facet_change(n, mf, alpha, slope, nv, mirror ? 0.0 : c.gradient_b.w);
        if (i == 0)
            orthonormal_basis(mf, tl, bl);
        // A refraction leaves the microfacet on the side away from the one it arrived on.
        if (!(dot(wi, mf) * dot(wo, mf) < 0.0))
            return false;
        float3 h = eta * wi + wo;
        float3 hu = eta * iu + ou;
        float3 hv = eta * iv + ov;
        float hm = dot(h, mf);
        float3 g = h - mf * hm;
        float3 gu = hu - mu * hm - mf * (dot(hu, mf) + dot(h, mu));
        float3 gv = hv - mv * hm - mf * (dot(hv, mf) + dot(h, mv));
        float2 e = float2(dot(g, tl), dot(g, bl));
        // The chain the constraint holds at q, and how far its direction past the surface lies
        // from the one toward the target.
        float3 dir = -wi;
        float3 mo = dot(dir, mf) < 0.0 ? mf : -mf;
        float ci = -dot(dir, mo);
        float k2 = 1.0 - eta * eta * (1.0 - ci * ci);
        if (k2 > 0.0 && length(eta * dir + (eta * ci - sqrt(k2)) * mo - wo) < MODEL_TOLERANCE)
        {
            root = q;
            return true;
        }
        float j00 = dot(gu, tl);
        float j01 = dot(gv, tl);
        float j10 = dot(gu, bl);
        float j11 = dot(gv, bl);
        float det = j00 * j11 - j01 * j10;
        if (!(abs(det) > 0.0))
            return false;
        float2 step = float2(j11 * e.x - j01 * e.y, j00 * e.y - j10 * e.x) / -det;
        // A step past a quarter of the way to the surface leaves the model.
        float most = MANIFOLD_STEP * lw;
        float moved = length(step);
        if (moved > most)
            step *= most / moved;
        uv += step;
    }
    return false;
}

// The face step of `slot`'s walk at the surface its face ray met, which the surface's hit hands in
// `c`, answering in `c.answer`. A leg meets the surface's plane at a point whose normal and alpha
// the surface's model gives (`SurfaceChange`); the chain refracts there about the microfacet normal
// of the walk's slope for the surface, by the IORs the BSDF bends by, and its differentials follow
// (Igehy's ray differentials, the microfacet normal moving with the normal and the alpha). Past
// the walk's first surface, where the chain leaves every interior, the model gives the start
// direction of the chain that crosses it alone (`model_aim`), which the end step takes. The
// probe's surface gives the first walk's start direction: that one, else, where the chain crosses
// more, the light's direction bent at the start across planes parallel to the probe's microfacet.
// The weighed walk takes the medium of the leg that reached the surface and the surface's BSDF
// toward the refracted direction, a delta lobe's coefficient where it is a mirror, else its value
// over the density of the microfacet normal and the Jacobian of that direction in it (Walter et
// al. 2007).
public void path_manifold_face<M : IMedium>(PathRoot r, uint slot, inout FaceCall call)
{
    ManifoldStep c = face_step_of(call);
    face_step<M>(r, slot, c);
    call.normal = c.answer;
}

void face_step<M : IMedium>(PathRoot r, uint slot, inout ManifoldStep c)
{
    uint4* at = path_manifold(r, slot);
    uint4 start = load_coherent(at + WALK_START);
    uint state = start.w;
    uint phase = state & WALK_PHASE;
    bool probe = phase == WALK_PROBE;
    uint4 slopes = load_coherent(at + WALK_SLOPES);
    uint crossed = probe ? 0 : (state >> WALK_CROSSED_SHIFT) & 0x7;
    c.answer = uint4(STEP_DARK, 0, 0, 0);
    if (crossed == MANIFOLD_CROSSINGS)
        return;
    float3 hit = c.at.xyz;
    uint m = asuint(c.at.w);
    float3 from = asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz);
    uint4 walk_point = probe ? uint4(asuint(from), 0) : load_coherent(at + WALK_POINT);
    float3 point = asfloat(walk_point.xyz);
    float3 dir = asfloat(load_coherent(at + WALK_DIR).xyz);
    float3 pu = float3(0.0);
    float3 pv = float3(0.0);
    float3 du = float3(0.0);
    float3 dv = float3(0.0);
    if (!probe)
        load_differentials(at, pu, pv, du, dv);
    float3 ng = c.geometric.xyz;
    float3 tx, bx;
    orthonormal_basis(ng, tx, bx);
    // The BSDF as a path along the leg meets it, against the media of the walk's list, which a
    // walk takes only where no side of the surface disperses (`walk_crosses`).
    Interior list = probe ? start_list(r, slot, at) : load_interior(at + WALK_LIST);
    uint outer = list.outside(m, m);
    if (path_disperses(r) && disperses(r.scene, outer))
        return;
    float exterior = interior_ior(r.scene, outer);
    Interior past = list;
    float4 medium_point = (r.scene.materials[m].flags & MATERIAL_MEDIUM_TEXTURED) != 0 ? asfloat(load_coherent(at + WALK_MEDIUM)) : NO_MEDIUM_POINT;
    past.cross(m, dot(dir, ng) < 0.0, medium_point);
    uint4 target = load_coherent(at + WALK_TARGET);
    bool point_target = may(MAY_LOCAL_LIGHTS) && (state & WALK_POINT_TARGET) != 0;
    float4 through = float4(1.0, 1.0, 1.0, 0.0);
    if (phase == WALK_WEIGH)
        through = asfloat(load_coherent(at + MANIFOLD_THROUGH_WORD));
    float3 root = float3(0.0);
    float3 aim = float3(0.0);
    bool alone = false;
    // The probe's surface may solve the chain across it alone exactly, within the triangle its
    // model holds, which an estimate then weighs in place: the face step of the walk that starts
    // along that chain, at the same surface, whose first leg no face ray then traces.
    [loop]
    for (uint pass = 0; pass < 2; pass++)
    {
        // The leg's point on the surface's plane: the leg's ray leaves an origin off the point
        // the chain leaves, and misses it by the offset.
        float cosine = dot(dir, ng);
        if (!(abs(cosine) > 0.0))
            return;
        float s = dot(hit - point, ng) / cosine;
        float3 x = point + dir * s;
        pu += du * s - dir * ((dot(pu, ng) + s * dot(du, ng)) / cosine);
        pv += dv * s - dir * ((dot(pv, ng) + s * dot(dv, ng)) / cosine);
        // The normal and the alpha there and along the differentials, by the model.
        float3 shift = x - hit;
        float3 raw = c.normal.xyz + c.gradient_t.xyz * dot(shift, tx) + c.gradient_b.xyz * dot(shift, bx);
        float length_n = length(raw);
        float3 n = raw / length_n;
        float3 gu = c.gradient_t.xyz * dot(pu, tx) + c.gradient_b.xyz * dot(pu, bx);
        float3 gv = c.gradient_t.xyz * dot(pv, tx) + c.gradient_b.xyz * dot(pv, bx);
        float3 nu = (gu - n * dot(n, gu)) / length_n;
        float3 nv = (gv - n * dot(n, gv)) / length_n;
        float3 view = -dir;
        BaseRefraction base = bsdf_base_refraction(c.p, n, view, exterior);
        float alpha = 0.0;
        float au = 0.0;
        float av = 0.0;
        if (!base.mirror)
        {
            alpha = base.alpha_iso + c.gradient_t.w * dot(shift, tx) + c.gradient_b.w * dot(shift, bx);
            au = c.gradient_t.w * dot(pu, tx) + c.gradient_b.w * dot(pu, bx);
            av = c.gradient_t.w * dot(pv, tx) + c.gradient_b.w * dot(pv, bx);
        }
        float2 slope = halves(lane(slopes, crossed));
        float3 mf = facet(n, alpha * slope);
        // Refraction from IOR n1 into n2 at eta = n1 / n2, which the BSDF's eta inverts.
        float eta = 1.0 / base.eta_refraction;
        if (pass == 0)
        {
            // The probe's solve on the model starts where the light's direction bent across the
            // microfacet's plane meets the hit's plane, which no chain leaving the interior
            // meets past its critical angle; a later walk's at the point its leg met it.
            float3 seed = x;
            float3 across;
            if (probe && past.count == 0 && !point_target && stack_direction(asfloat(target.xyz), mf, AMBIENT_IOR * eta, across) && abs(dot(across, ng)) > 0.0)
                seed = from + across * (dot(hit - from, ng) / dot(across, ng));
            alone = crossed == 0 && past.count == 0 && model_aim(c, tx, bx, from, seed, eta, base.mirror, base.alpha_iso, slope, asfloat(target.xyz), point_target, root);
            aim = normalize(root - from);
        }
        // Where the model's chain meets the plane: within the triangle the model is exact across,
        // or past it.
        float2 moved = float2(dot(root - hit, tx), dot(root - hit, bx));
        float4 g = c.bary_change;
        float2 b = c.bary + float2(g.x * moved.x + g.y * moved.y, g.z * moved.x + g.w * moved.y);
        bool exact = alone && any(g != 0.0);
        bool within = b.x >= 0.0 && b.y >= 0.0 && b.x + b.y <= 1.0;
        if (probe)
        {
            // Where the model's chain meets the plane past the triangle the model is exact
            // across, the probe goes again toward it, which finds the surface there.
            uint probes = (state >> WALK_PROBES_SHIFT) & 0x3;
            if (exact && !within && probes + 1 < MANIFOLD_PROBES)
            {
                store_coherent(at + WALK_START, uint4(start.xyz, state + (1u << WALK_PROBES_SHIFT)));
                store_coherent(at + WALK_DIR, uint4(asuint(aim), 0));
                store_coherent(at + WALK_MODEL, uint4(asuint(root), 2));
                c.answer = uint4(STEP_RESTART, start.xyz);
                return;
            }
            // An estimate weighs the first walk, whose start is exact across planes and on an
            // exact model: in place where the model holds the chain's point.
            bool estimates = (state & WALK_ESTIMATES) != 0;
            uint next = estimates ? WALK_WEIGH : WALK_SOLVE;
            if (estimates && exact && within && ((state >> WALK_BEGUN_SHIFT) & 0xf) < MANIFOLD_WALKS)
            {
                probe = false;
                phase = WALK_WEIGH;
                state = WALK_WEIGH | (state & WALK_KEPT) | ((((state >> WALK_BEGUN_SHIFT) & 0xf) + 1) << WALK_BEGUN_SHIFT);
                point = from;
                dir = aim;
                pu = float3(0.0);
                pv = float3(0.0);
                orthonormal_basis(aim, du, dv);
                through = float4(1.0, 1.0, 1.0, 0.0);
                continue;
            }
            float3 light = point_target ? normalize(asfloat(target.xyz) - from) : asfloat(target.xyz);
            if ((alone || stack_direction(light, mf, interior_ior(r.scene, list.innermost()), aim))
                && restart(r, at, asfloat(start.xyz), aim, next, state, list, false, root, alone))
                c.answer = uint4(STEP_RESTART | (next == WALK_WEIGH ? STEP_WEIGHS : 0), start.xyz);
            return;
        }
        if (alone)
            store_coherent(at + WALK_MODEL, uint4(asuint(root), exact && within ? 1 : MODEL_PAST));
        else if (crossed == 0)
            store_coherent(at + WALK_MODEL, uint4(0));
        float side = dot(dir, mf) < 0.0 ? 1.0 : -1.0;
        float3 mo = mf * side;
        float3 mu = facet_change(n, mf, alpha, slope, nu, au) * side;
        float3 mv = facet_change(n, mf, alpha, slope, nv, av) * side;
        float ci = -dot(dir, mo);
        float k2 = 1.0 - eta * eta * (1.0 - ci * ci);
        if (!(k2 > 0.0))
            return;
        float k = sqrt(k2);
        float3 bent = normalize(eta * dir + (eta * ci - k) * mo);
        float ciu = -(dot(du, mo) + dot(dir, mu));
        float civ = -(dot(dv, mo) + dot(dir, mv));
        du = eta * du + (eta * ciu - eta * eta * ci * ciu / k) * mo + (eta * ci - k) * mu;
        dv = eta * dv + (eta * civ - eta * eta * ci * civ / k) * mo + (eta * ci - k) * mv;
        uint answer = STEP_ON | (past.count == 0 ? STEP_OUT : 0);
        if (phase == WALK_WEIGH)
        {
            answer |= STEP_WEIGHS | (pass == 1 ? STEP_INNER : 0);
            // The chain weighed in place leaves the surface out of every interior, and where it
            // converged the end step would answer for it here (`chain_gap`): with the density and
            // the linked medium's transmittance past the surface, which the raygen takes where
            // the leg meets nothing more (STEP_SETTLED). Its leg into the surface lies in an
            // interior, unless its list held none it left by, which the end step answers.
            uint inner = list.innermost();
            bool settles = false;
            float settled_density = 0.0;
            float reach = positive_infinity();
            if (pass == 1 && inner != MATERIAL_NONE)
            {
                ChainGap g = chain_gap(bent, x, asfloat(target.xyz), point_target, pu, pv, du, dv);
                settles = g.converged;
                settled_density = settles ? g.density : 0.0;
                if (point_target)
                    reach = g.reach;
            }
            // The leg's medium: the interior it lies in, or out of every interior the linked
            // medium, whose transmittance one site takes for that leg or a settled chain's out.
            bool froxel = (state & WALK_FROXEL) != 0;
            float3 linked = float3(1.0);
            if ((inner == MATERIAL_NONE || settles) && !froxel)
                linked = medium_transmittance<M>(r.scene.medium, settles ? x : point, settles ? bent : dir, settles ? reach : length(x - point));
            if (inner == MATERIAL_NONE)
                through.xyz *= linked;
            else if (!((state & WALK_OWN_LEG) != 0 && crossed == 0))
                through.xyz *= beer(interior_medium(r.scene, inner, list.point(inner)).sigma_t, length(x - point));
            Bsdf b = bsdf_prepare(c.p, n, view, exterior);
            float3 w;
            if (b.mirror)
            {
                float3 own;
                w = b.refraction(own);
            }
            else
            {
                float er = b.eta_refraction;
                float lm = dot(bent, mf);
                float vm = dot(view, mf);
                float density = facet_density(alpha, slope) * er * er * abs(lm);
                w = density > 0.0 ? b.eval(bent) * ((vm + er * lm) * (vm + er * lm) / density) : float3(0.0);
            }
            through.xyz *= w;
            if (pass == 1)
                store_coherent(at + MANIFOLD_BEND_WORD, uint4(asuint(aim), 0));
            if (settles)
            {
                answer |= STEP_SETTLED;
                store_coherent(at + WALK_MEDIUM, asuint(float4(linked, settled_density)));
            }
            store_coherent(at + MANIFOLD_THROUGH_WORD, asuint(through));
        }
        store_list(at + WALK_LIST, past);
        store_coherent(at + WALK_TARGET, uint4(target.xyz, asuint(interior_ior(r.scene, past.innermost()))));
        store_coherent(at + WALK_START, uint4(start.xyz, (state & ~(0x7u << WALK_CROSSED_SHIFT)) | ((crossed + 1) << WALK_CROSSED_SHIFT)));
        store_coherent(at + WALK_POINT, uint4(asuint(x), crossed == 0 ? asuint(length(x - point)) : walk_point.w));
        store_coherent(at + WALK_DIR, uint4(asuint(bent), 0));
        store_differentials(at, pu, pv, du, dv);
        c.answer = uint4(answer, asuint(ray_origin(x, x, ng, bent, ORIGIN_DEFAULT, c.geometric.w, 0.0)));
        return;
    }
}

// How the chain whose last leg leaves `last` along unit `dir`, with the differentials `pu`, `pv`
// of its point and `du`, `dv` of its direction in its start direction, misses its target
// `target`: a point where `point`, toward which the gap is the leg's from the point's direction
// from `last`, which moves with it, else a direction. The gap goes in the basis of the direction
// wanted (`orthonormal_basis`), with the Jacobian of the last direction there; where the gap lies
// within MANIFOLD_TOLERANCE the chain has converged, at the density of the start direction over
// the light's direction, or over the area across the last leg at the point, where its spread is
// its point's along its differentials, past the point by its direction's.
struct ChainGap
{
    float2 gap;
    float4 jacobian;
    float det;
    bool solvable;
    bool converged;
    float density;
    float reach;
}

ChainGap chain_gap(float3 dir, float3 last, float3 target, bool point, float3 pu, float3 pv, float3 du, float3 dv)
{
    ChainGap g;
    float3 want = target;
    g.reach = 0.0;
    // The gap's change: the leg's direction's less the direction wanted's, which a point moves
    // away from as the leg's start moves across it.
    float3 gu = du;
    float3 gv = dv;
    if (point)
    {
        float3 e = target - last;
        g.reach = length(e);
        want = e / g.reach;
        gu += (pu - want * dot(want, pu)) / g.reach;
        gv += (pv - want * dot(want, pv)) / g.reach;
    }
    float3 tl, bl;
    orthonormal_basis(want, tl, bl);
    float3 gap = dir - want;
    g.gap = float2(dot(gap, tl), dot(gap, bl));
    g.jacobian = float4(dot(gu, tl), dot(gv, tl), dot(gu, bl), dot(gv, bl));
    g.det = g.jacobian.x * g.jacobian.w - g.jacobian.y * g.jacobian.z;
    g.solvable = abs(g.det) > 0.0 && isfinite(g.det);
    g.converged = g.solvable && length(gap) < MANIFOLD_TOLERANCE;
    g.density = 1.0 / abs(g.det);
    if (point)
    {
        float3 yu = pu + du * g.reach - dir * dot(dir, pu + du * g.reach);
        float3 yv = pv + dv * g.reach - dir * dot(dir, pv + dv * g.reach);
        orthonormal_basis(dir, tl, bl);
        g.density = 1.0 / abs(dot(yu, tl) * dot(yv, bl) - dot(yv, tl) * dot(yu, bl));
    }
    return g;
}

// The end step of `slot`'s walk, whose last leg met nothing (`answer`'s first word MET_NONE) or
// something other than a surface a walk crosses (MET_OTHER), or past its probe whose face step
// found no way on (MET_FAILED), answering in `answer`. A probe that met no such surface after its
// model sent it again starts the walk toward where it went; the first, along the light's
// direction bent across planes facing the vertex's normal (`start_probe`), goes again along the
// light's own. A chain that crossed a surface, met nothing more and lies outside every interior
// reaches the light where its direction is the light's (`chain_gap`): a converged walk gives the
// density of the start direction over the light's, and an estimate's walks that converge walk
// again weighed; one that did not moves its start direction by a Newton step, at most
// MANIFOLD_STEP, from the walk the solve accepts, and a walk that fails or comes no nearer steps
// back halfway.
public void path_manifold_end<M : IMedium>(PathRoot r, uint slot, inout uint4 answer)
{
    uint met = answer.x;
    answer = uint4(STEP_DARK, 0, 0, 0);
    uint4* at = path_manifold(r, slot);
    uint4 start = load_coherent(at + WALK_START);
    uint state = start.w;
    uint phase = state & WALK_PHASE;
    uint crossed = (state >> WALK_CROSSED_SHIFT) & 0x7;
    if (phase == WALK_PROBE)
    {
        if (((state >> WALK_PROBES_SHIFT) & 0x3) != 0)
        {
            uint next = (state & WALK_ESTIMATES) != 0 ? WALK_WEIGH : WALK_SOLVE;
            if (restart(r, at, asfloat(start.xyz), asfloat(load_coherent(at + WALK_DIR).xyz), next, state, start_list(r, slot, at)))
                answer = uint4(STEP_RESTART | (next == WALK_WEIGH ? STEP_WEIGHS : 0), start.xyz);
            return;
        }
        if ((state & WALK_PROBED) == 0 || (state & WALK_PROBED_AGAIN) != 0)
            return;
        float3 light = asfloat(load_coherent(at + WALK_TARGET).xyz);
        if (may(MAY_LOCAL_LIGHTS) && (state & WALK_POINT_TARGET) != 0)
            light = normalize(light - asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz));
        store_coherent(at + WALK_START, uint4(start.xyz, state | WALK_PROBED_AGAIN));
        store_coherent(at + WALK_DIR, uint4(asuint(light), 0));
        store_coherent(at + WALK_MODEL, uint4(0));
        answer = uint4(STEP_RESTART, start.xyz);
        return;
    }
    // A light at a point lies in the medium its chain's last leg reached it in, but a direction
    // lies out of every interior: a chain that crossed a surface reaches past it where its last
    // leg lies there, and is whole where that leg met nothing more.
    bool point = may(MAY_LOCAL_LIGHTS) && (state & WALK_POINT_TARGET) != 0;
    float3 origin = asfloat(start.xyz);
    uint4 bend = load_coherent(at + MANIFOLD_BEND_WORD);
    float3 aim = asfloat(bend.xyz);
    Interior list = Interior();
    uint inner = MATERIAL_NONE;
    bool reaches = false;
    ChainGap g = {};
    float3 dir = float3(0.0);
    float3 last = float3(0.0);
    float4 target = float4(0.0);
    if (crossed != 0 && met != MET_FAILED)
    {
        list = load_interior(at + WALK_LIST);
        inner = list.innermost();
        reaches = point || interior_ior(r.scene, inner) == AMBIENT_IOR;
    }
    if (reaches)
    {
        target = asfloat(load_coherent(at + WALK_TARGET));
        dir = asfloat(load_coherent(at + WALK_DIR).xyz);
        last = asfloat(load_coherent(at + WALK_POINT).xyz);
        float3 pu, pv, du, dv;
        load_differentials(at, pu, pv, du, dv);
        g = chain_gap(dir, last, target.xyz, point, pu, pv, du, dv);
    }
    // Something opaque across the last leg of a chain that reaches the light blocks it, and the
    // solve finds no other near it.
    if (met == MET_OTHER && reaches && g.converged)
        return;
    // A walk of a solve that met no surface, met something opaque or found no way on at a
    // surface gives way to one halfway from the start of the walk the solve accepted toward its own
    // (Hanika et al. 2015): its start's step crossed where the chain leaves the surfaces it
    // follows, as a step across an edge of a surface onto another does. No test of the gap
    // shrinking: a check stops where the next start predicts its chain (MANIFOLD_PREDICT), and
    // the estimate it speaks for must take the walks that converge on it.
    if (!(met == MET_NONE && reaches && g.solvable))
    {
        if ((state & WALK_ACCEPTED) == 0)
            return;
        float3 from_aim = oct_decode_snorm16(bend.w);
        float3 half_aim = normalize(from_aim + aim);
        uint next = (state & WALK_ESTIMATES) != 0 && length(half_aim - from_aim) < MANIFOLD_SETTLED ? WALK_WEIGH : WALK_SOLVE;
        if (restart(r, at, origin, half_aim, next, state, start_list(r, slot, at), false, float3(0.0), false, bend.w))
            answer = uint4(STEP_RESTART | (next == WALK_WEIGH ? STEP_WEIGHS : 0), asuint(origin));
        return;
    }
    if (g.converged)
    {
        if (phase == WALK_WEIGH)
        {
            float4 through = asfloat(load_coherent(at + MANIFOLD_THROUGH_WORD));
            if (point && inner != MATERIAL_NONE)
                through.xyz *= beer(interior_medium(r.scene, inner, list.point(inner)).sigma_t, g.reach);
            else if ((state & WALK_FROXEL) == 0)
                through.xyz *= medium_transmittance<M>(r.scene.medium, last, dir, point ? g.reach : positive_infinity());
            store_coherent(at + MANIFOLD_THROUGH_WORD, asuint(float4(through.xyz, g.density)));
            answer.x = STEP_LIT;
            return;
        }
        if ((state & WALK_ESTIMATES) == 0)
        {
            answer.x = STEP_FOUND;
            return;
        }
        // The weighed walk follows the converged chain, which nothing then replaces.
        if (restart(r, at, origin, aim, WALK_WEIGH, state & ~WALK_ACCEPTED, start_list(r, slot, at), true))
            answer = uint4(STEP_RESTART | STEP_WEIGHS, asuint(origin));
        return;
    }
    // The solve accepts the walk. A chain across one surface starts next where that surface's model
    // gives, which is exact across the triangle the walk met, so an estimate weighs that walk; any
    // other moves its start direction by a Newton step, at most MANIFOLD_STEP, and weighs the walk
    // it starts where the step is small enough to leave Newton's quadratic convergence a gap well
    // within tolerance.
    state |= WALK_ACCEPTED;
    uint4 model = load_coherent(at + WALK_MODEL);
    bool past = model.w == MODEL_PAST && (state & WALK_MODEL_PAST) == 0;
    bool modeled = crossed == 1 && (model.w == 1 || past);
    if (modeled && past)
        state |= WALK_MODEL_PAST;
    float3 from = asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz);
    float3 next_aim = normalize(asfloat(model.xyz) - from);
    bool settled = modeled;
    if (!modeled)
    {
        float4 j = g.jacobian;
        float2 e = g.gap;
        float2 step = float2(j.w * e.x - j.y * e.y, j.x * e.y - j.z * e.x) / -g.det;
        float moved = length(step);
        if (moved > MANIFOLD_STEP)
            step *= MANIFOLD_STEP / moved;
        float3 ta, ba;
        orthonormal_basis(aim, ta, ba);
        next_aim = normalize(aim + step.x * ta + step.y * ba);
        settled = moved < MANIFOLD_SETTLED;
    }
    uint next = (state & WALK_ESTIMATES) != 0 && settled ? WALK_WEIGH : WALK_SOLVE;
    if (restart(r, at, origin, next_aim, next, state, start_list(r, slot, at), false, asfloat(model.xyz), modeled, oct_encode_snorm16(aim)))
        answer = uint4(STEP_RESTART | (next == WALK_WEIGH ? STEP_WEIGHS : 0), asuint(origin));
}

// Starts the probe of the solve that `slot`'s manifold words hold (`path_manifold_end`): inside an
// interior the light reaches the vertex bent, at a slant its own direction need not leave the
// interior by, so the probe first goes along the light's direction bent into the interior around
// the vertex across planes facing its normal, as a floor under water faces the water's surface:
// on the light's side, or where the vertex draws its directional lights with no normal, on the side
// the path reached it from, a light behind it mirrored to that side first; where the vertex lies
// in no interior or has no normal, along the light's own, or for an estimate that is aimed
// (WALK_AIMED) toward the point its model word holds. The IOR of the
// interior the walk starts in is the one its job left (MANIFOLD_START_IOR). The probe's first
// face ray leaves from the answer's origin.
public uint4 start_probe(PathRoot r, uint slot)
{
    uint4* at = path_manifold(r, slot);
    uint4 start = load_coherent(at + WALK_START);
    uint state = start.w;
    float3 light = asfloat(load_coherent(at + WALK_TARGET).xyz);
    if (may(MAY_LOCAL_LIGHTS) && (state & WALK_POINT_TARGET) != 0)
        light = normalize(light - asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz));
    float4 normal = asfloat(load_coherent(at + WALK_DIR));
    float ior = asfloat(load_coherent(at + MANIFOLD_START_IOR).w);
    float3 probe = light;
    float3 bent;
    float3 facing = dot(normal.xyz, light) < 0.0 ? -normal.xyz : normal.xyz;
    float3 bending = light;
    if (normal.w == WALK_MIRRORS)
    {
        facing = normal.xyz;
        bending = light - 2.0 * min(dot(light, facing), 0.0) * facing;
    }
    if (normal.w != 0.0 && ior != AMBIENT_IOR && stack_direction(bending, facing, ior, bent))
    {
        probe = bent;
        state |= WALK_PROBED;
    }
    else if ((state & WALK_AIMED) != 0)
        probe = normalize(asfloat(load_coherent(at + WALK_MODEL).xyz) - asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz));
    store_coherent(at + WALK_START, uint4(start.xyz, state));
    store_coherent(at + WALK_DIR, uint4(asuint(probe), 0));
    store_coherent(at + WALK_MODEL, uint4(0));
    return uint4(STEP_RESTART, start.xyz);
}
