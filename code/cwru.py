"""CWRU as a control, where the theory predicts the method should not help.

The four CWRU operating conditions differ by 4.4% in shaft speed and by 0 to 3 hp
in load, so there is almost no dilation between them.  If the log axis works by
absorbing a dilation, it should give little or nothing here.

Drive-end accelerometer, 12 kHz, 0.007 inch defects, so the class definitions line
up with JNU.
"""
import hashlib, os
import numpy as np
from scipy.io import loadmat

from features import lfes, ses, order_spectrum, CLASSES
import paths

RAW = paths.CWRU
CACHE = paths.CACHE
FS = 12000.0

# file number -> (class, nominal rpm).  98 and 99 carry no RPM field in the
# distributed .mat; the nominal values from the data centre's own table are used.
FILES = {
    97: ("healthy", 1797), 98: ("healthy", 1772), 99: ("healthy", 1750), 100: ("healthy", 1730),
    105: ("inner", 1797), 106: ("inner", 1772), 107: ("inner", 1750), 108: ("inner", 1730),
    118: ("ball", 1797), 119: ("ball", 1772), 120: ("ball", 1750), 121: ("ball", 1730),
    130: ("outer", 1797), 131: ("outer", 1772), 132: ("outer", 1750), 133: ("outer", 1730),
}
LOADS = [1797, 1772, 1750, 1730]

WIN = 2048                 # 0.171 s, matching the JNU window in seconds
HOP_TRAIN = 512
HOP_EVAL = 1024
SPLIT = (0.70, 0.15, 0.15)


def _de(path, num):
    d = loadmat(path)
    key = [k for k in d if k.endswith("DE_time")]
    return np.asarray(d[key[0]]).ravel().astype(float)


def _windows(x, win, hop):
    if len(x) < win:
        return np.empty((0, win))
    return np.stack([x[s:s + win] for s in np.arange(0, len(x) - win + 1, hop)])


def build(front_end="lfes", nbins=256, fmin=20.0, fmax=2000.0, pad=8, force=False):
    key = hashlib.md5(f"cwru{front_end}{nbins}{fmin}{fmax}{pad}{WIN}".encode()).hexdigest()[:12]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"cwru_{front_end}_{key}.npz")
    if os.path.exists(path) and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}

    fe = {"lfes": lfes, "ses": ses, "order": order_spectrum}[front_end]
    out = {sp: {"X": [], "y": [], "speed": []} for sp in ("train", "val", "test")}
    for num, (cls, rpm) in sorted(FILES.items()):
        raw = _de(os.path.join(RAW, f"{num}.mat"), num)
        n = len(raw)
        a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
        for sp, block in (("train", raw[:a]), ("val", raw[a:b]), ("test", raw[b:])):
            hop = HOP_TRAIN if sp == "train" else HOP_EVAL
            for w in _windows(block, WIN, hop):
                if front_end == "order":
                    v, _ = fe(w, FS, rpm / 60.0, nbins=nbins, pad=pad)
                else:
                    v, _ = fe(w, FS, fmin=fmin, fmax=fmax, nbins=nbins, pad=pad)
                out[sp]["X"].append(v.astype(np.float32))
                out[sp]["y"].append(CLASSES.index(cls))
                out[sp]["speed"].append(rpm)
    packed = {}
    for sp in out:
        packed[f"X_{sp}"] = np.asarray(out[sp]["X"], dtype=np.float32)
        packed[f"y_{sp}"] = np.asarray(out[sp]["y"], dtype=np.int64)
        packed[f"s_{sp}"] = np.asarray(out[sp]["speed"], dtype=np.int64)
    np.savez_compressed(path, **packed)
    return packed


if __name__ == "__main__":
    import sys
    d = build(sys.argv[1] if len(sys.argv) > 1 else "lfes")
    for sp in ("train", "val", "test"):
        print(f"{sp:6s} {d['X_'+sp].shape}  classes={np.bincount(d['y_'+sp])}")
