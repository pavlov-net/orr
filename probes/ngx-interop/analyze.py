"""Compares the probe's saved images. Run with
`uv run --no-project --with flip-evaluator --with numpy python -I analyze.py <case> <out dir>`.

Cases: interop (bitwise equality across variants), color (Rec.709 vs Rec.2020 RR inputs, FLIP
against the reference), fg (wide-gamut vs clipped PQ backbuffers through FG 2x, inside Rec.709),
dof (with and without the depth-of-field guide), exposure (RR history after steps and ramps).
"""

import csv
import glob
import os
import sys

import flip_evaluator as flip
import numpy as np

M2020_TO_709 = np.array(
    [[1.6604910, -0.5876411, -0.0728499], [-0.1245505, 1.1328999, -0.0083494], [-0.0181508, -0.1005789, 1.1187297]],
    dtype=np.float32,
)
FLIP_RANGE = {"startExposure": -2.0, "stopExposure": 2.0, "numExposures": 5}


def load(path):
    head = np.fromfile(path, dtype=np.uint32, count=3)
    w, h, c = (int(v) for v in head)
    if path.endswith(".u32"):
        return np.fromfile(path, dtype=np.uint32, offset=12).reshape(h, w)
    return np.fromfile(path, dtype=np.float32, offset=12).reshape(h, w, c)


def hdr_flip(ref, test):
    _, mean, _ = flip.evaluate(np.ascontiguousarray(ref[..., :3]), np.ascontiguousarray(test[..., :3]), "HDR", parameters=FLIP_RANGE)
    return mean


def interop(out):
    files = sorted(glob.glob(os.path.join(out, "interop_*.f32")))
    base = load(os.path.join(out, "interop_general_restore_render.f32"))
    ref = load(os.path.join(out, "ref_interop.f32"))
    for f in files:
        img = load(f)
        diff = np.abs(img - base)
        line = f"{os.path.basename(f)}: bit-identical to general_restore_render {np.array_equal(img.view(np.uint32), base.view(np.uint32))}, max abs diff {diff.max():.3g}"
        if "norebind" not in f:
            line += f", FLIP vs final-frame reference {hdr_flip(ref, img):.4f}"
        print(line)


def color(out):
    ref = load(os.path.join(out, "color_reference.f32"))
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, "color_*_s*_*.f32"))):
        name = os.path.basename(f)[:-4]
        _, space, seed, sign = name.split("_")
        img = load(f)[..., :3]
        if space == "2020":
            img = img @ M2020_TO_709.T
        rows.setdefault((space, sign), {})[seed] = img
    for (space, sign), seeds in sorted(rows.items()):
        scores = [hdr_flip(ref, img) for img in seeds.values()]
        print(f"space {space} jitter {sign}: FLIP per seed {['%.4f' % s for s in scores]}, mean {np.mean(scores):.4f}, spread {np.ptp(scores):.4f}")
    for sign in ("std", "flip"):
        a, b = rows.get(("709", sign)), rows.get(("2020", sign))
        if not a or not b:
            continue
        for seed in sorted(set(a) & set(b)):
            d = np.abs(a[seed] - b[seed])
            lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            rel = np.abs(lum(a[seed]) - lum(b[seed])) / np.maximum(lum(a[seed]), 1e-3)
            print(
                f"{sign} {seed}: Rec.2020 output (to 709) vs Rec.709 output: FLIP {hdr_flip(a[seed], b[seed]):.4f}, "
                f"max abs {d.max():.3g}, mean abs {d.mean():.4g}, luminance rel p99 {np.percentile(rel, 99):.4f}, "
                f"mean chroma shift {np.mean(np.abs((a[seed] / np.maximum(lum(a[seed])[..., None], 1e-3)) - (b[seed] / np.maximum(lum(b[seed])[..., None], 1e-3)))):.4f}"
            )
        seeds = sorted(a)
        if len(seeds) >= 2:
            print(f"{sign}: run-to-run (Rec.709, seed {seeds[0]} vs {seeds[1]}): FLIP {hdr_flip(a[seeds[0]], a[seeds[1]]):.4f}")


def pq_decode(words):
    rgb = np.stack([(words >> s) & 0x3FF for s in (0, 10, 20)], axis=-1).astype(np.float64) / 1023.0
    m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
    p = np.power(rgb, 1.0 / m2)
    return (10000.0 * np.power(np.maximum(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1)).astype(np.float32)


def fg(out):
    imgs = {}
    for mode in ("wide", "clip"):
        for kind in ("interp", "truth", "real"):
            p = os.path.join(out, f"fg_{mode}_{kind}.u32")
            if os.path.exists(p):
                lin = pq_decode(load(p)) / 200.0  # 1.0 = 200 nits
                imgs[(mode, kind)] = np.maximum(lin @ M2020_TO_709.T, 0.0)  # inside Rec.709
    if ("wide", "interp") not in imgs or ("clip", "interp") not in imgs:
        print("missing FG outputs")
        return
    wi, ci = imgs[("wide", "interp")], imgs[("clip", "interp")]
    wt, ct = imgs[("wide", "truth")], imgs[("clip", "truth")]
    print(f"truth: wide clipped after decode vs content clipped per sub-sample: FLIP {hdr_flip(ct, wt):.4f} (edge pixels only differ)")
    print(f"FG wide (clipped after) vs FG clipped: FLIP {hdr_flip(ci, wi):.4f}, max abs {np.abs(wi - ci).max():.3g}, mean abs {np.abs(wi - ci).mean():.4g}")
    print(f"FG wide vs mid-frame truth: FLIP {hdr_flip(wt, wi):.4f}; FG clipped vs truth: FLIP {hdr_flip(ct, ci):.4f}")
    rw, rc = imgs[("wide", "real")], imgs[("clip", "real")]
    print(f"real frame vs mid-frame truth (no interpolation, for scale): FLIP {hdr_flip(ct, rc):.4f}")


def dof(out):
    ref = load(os.path.join(out, "dof_reference.f32"))
    h, w = ref.shape[:2]
    region = (slice(int(0.1 * h) + 8, int(0.6 * h) - 8), slice(int(0.55 * w) + 8, int(0.95 * w) - 8))
    imgs = {os.path.basename(f)[4:-4]: load(f) for f in sorted(glob.glob(os.path.join(out, "dof_*_g*.f32")))}
    for name, img in imgs.items():
        print(f"{name}: FLIP whole {hdr_flip(ref, img):.4f}, blurred region {hdr_flip(ref[region], img[region]):.4f}")
    for preset in ("default", "E", "F"):
        a, b = imgs.get(f"{preset}_g0"), imgs.get(f"{preset}_g1")
        if a is None or b is None:
            continue
        d = np.abs(a - b)
        outside = d.copy()
        outside[region] = 0
        print(f"preset {preset}: guide changes output: max abs {d.max():.3g}, mean abs in region {d[region].mean():.4g}, outside {outside.mean():.4g}")


def exposure(out):
    runs = {}
    with open(os.path.join(out, "exposure.csv")) as f:
        for r in csv.DictReader(f):
            runs.setdefault(r["run"], []).append((int(r["frame"]), float(r["exposure"]), float(r["absolute_mean"])))
    steady = np.mean([v for fr, _, v in runs["steady1"] if fr >= 200])
    noise = np.std([np.log2(v / steady) for fr, _, v in runs["steady1"] if fr >= 200])
    s4 = np.mean([v for fr, _, v in runs["steady4"] if fr >= 200])
    print(f"steady absolute mean {steady:.5f} (frame-to-frame sd {noise:.4f} EV); steady at 4x pre-exposure {np.log2(s4 / steady):+.4f} EV")
    for step in ("step_up", "step_down"):
        a = {fr: v for fr, _, v in runs[step]}
        b = {fr: v for fr, _, v in runs[step + "_reset"]}
        dev = [(fr, np.log2(a[fr] / b[fr])) for fr in range(100, 130)]
        settle = next((fr - 100 for fr, d in dev if all(abs(x) < 0.1 for f2, x in dev if f2 >= fr)), None)
        print(f"{step}: EV vs reset run, frames 100..111: {' '.join('%+.3f' % d for _, d in dev[:12])}; within 0.1 EV from frame +{settle}")
        rdev = [np.log2(v / steady) for fr, _, v in runs[step + "_reset"] if 100 <= fr < 112]
        print(f"{step}_reset: EV vs steady, frames 100..111: {' '.join('%+.3f' % d for d in rdev)}")
    for ramp in ("ramp_up", "ramp_down", "ramp_up_slow"):
        dev = [(fr, np.log2(v / steady)) for fr, _, v in runs[ramp]]
        during = [d for fr, d in dev if 110 <= fr < 220]
        after = [d for fr, d in dev if 225 <= fr]
        print(f"{ramp}: EV vs steady during ramp: mean {np.mean(during):+.3f}, max |dev| {np.max(np.abs(during)):.3f}; after ramp mean {np.mean(after):+.3f}")


if __name__ == "__main__":
    {"interop": interop, "color": color, "fg": fg, "dof": dof, "exposure": exposure}[sys.argv[1]](sys.argv[2])
