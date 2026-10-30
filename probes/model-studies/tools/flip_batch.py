# Scores image pairs from a manifest with HDR-FLIP (flip-evaluator).
# Manifest lines: group reference test label. Images: u32 w, u32 h, then f32 RGB rows.
import sys
import numpy as np
import flip_evaluator as flip


def load(path):
    raw = np.fromfile(path, dtype=np.uint8)
    w, h = np.frombuffer(raw[:8].tobytes(), dtype=np.uint32)
    img = np.frombuffer(raw[8:].tobytes(), dtype=np.float32).reshape(h, w, 3)
    return np.ascontiguousarray(np.maximum(img, 0.0))


def main():
    print(f"{'group':16} {'label':16} {'mean':>8} {'p99':>8} {'max':>8}")
    for line in open(sys.argv[1]):
        group, ref, test, label = line.split()
        err, mean, _ = flip.evaluate(load(ref), load(test), "HDR", inputsRGB=False, applyMagma=False)
        err = np.asarray(err)
        print(f"{group:16} {label:16} {mean:8.4f} {np.percentile(err, 99):8.4f} {err.max():8.4f}")


main()
