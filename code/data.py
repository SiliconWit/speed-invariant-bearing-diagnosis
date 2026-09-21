"""JNU bearing data: loading, leak-free splitting, windowing, feature extraction.

Jiangnan University rig, PCB MA352A60 accelerometer, vertical direction, 50 kHz,
at 600, 800 and 1000 rpm.  Four classes: normal, inner race, rolling element and
outer race, cut by wire EDM to 0.3 x 0.05 mm.

The fault recordings are 10 s and the normal recordings 30 s.  The inner-race
recordings come from an NU205 bearing and the other three classes from an N205, so
a classifier could in principle separate that one class on a bearing signature
rather than on fault physics; the speed question is asked within each class.
"""
import hashlib, os
import numpy as np

from features import lfes, ses, order_spectrum, CLASSES
import paths

RAW = paths.JNU
CACHE = paths.CACHE
FS = 50000.0
SPEEDS = [600, 800, 1000]                      # rpm
FR = {s: s / 60.0 for s in SPEEDS}             # shaft frequency, Hz

FILES = {
    ("healthy", 600): "n600_3_2.csv",   ("healthy", 800): "n800_3_2.csv",
    ("healthy", 1000): "n1000_3_2.csv",
    ("inner", 600): "ib600_2.csv",      ("inner", 800): "ib800_2.csv",
    ("inner", 1000): "ib1000_2.csv",
    ("ball", 600): "tb600_2.csv",       ("ball", 800): "tb800_2.csv",
    ("ball", 1000): "tb1000_2.csv",
    ("outer", 600): "ob600_2.csv",      ("outer", 800): "ob800_2.csv",
    ("outer", 1000): "ob1000_2.csv",
}

WIN = 8192                 # 0.164 s; about 8 outer-race impacts at 600 rpm
HOP_TRAIN = 2048
HOP_EVAL = 4096
SPLIT = (0.70, 0.15, 0.15)      # by time, on the continuous recording


def _windows(x, win, hop):
    if len(x) < win:
        return np.empty((0, win))
    starts = np.arange(0, len(x) - win + 1, hop)
    return np.stack([x[s:s + win] for s in starts])


def split_blocks(x):
    """Disjoint time blocks, assigned before any windowing.

    Windowing first and splitting afterwards puts near-identical overlapping
    windows on both sides of the split.  Splitting the continuous recording into
    contiguous blocks first prevents that.
    """
    n = len(x)
    a = int(SPLIT[0] * n)
    b = int((SPLIT[0] + SPLIT[1]) * n)
    return {"train": x[:a], "val": x[a:b], "test": x[b:]}


def build(front_end="lfes", nbins=256, fmin=20.0, fmax=2000.0, pad=8, force=False):
    """Windowed features for every class, speed and split.  Cached on disk."""
    key = hashlib.md5(
        f"{front_end}{nbins}{fmin}{fmax}{pad}{WIN}{HOP_TRAIN}{HOP_EVAL}{SPLIT}".encode()
    ).hexdigest()[:12]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"jnu_{front_end}_{key}.npz")
    if os.path.exists(path) and not force:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    # "order" divides the frequency axis by the known shaft speed, which removes the
    # dilation exactly.  It is the strongest baseline, and it is only available
    # because JNU publishes a nominal speed per recording.
    fe = {"lfes": lfes, "ses": ses, "order": order_spectrum}[front_end]
    out = {sp: {"X": [], "y": [], "speed": []} for sp in ("train", "val", "test")}
    for (cls, rpm), fname in sorted(FILES.items()):
        raw = np.loadtxt(os.path.join(RAW, fname))
        for sp, block in split_blocks(raw).items():
            hop = HOP_TRAIN if sp == "train" else HOP_EVAL
            for w in _windows(block, WIN, hop):
                if front_end == "order":
                    v, _ = fe(w, FS, FR[rpm], nbins=nbins, pad=pad)
                else:
                    v, _ = fe(w, FS, fmin=fmin, fmax=fmax, nbins=nbins, pad=pad)
                out[sp]["X"].append(v.astype(np.float32))
                out[sp]["y"].append(CLASSES.index(cls))
                out[sp]["speed"].append(rpm)
        print(f"  {cls:8s} {rpm:5d} rpm  done", flush=True)

    packed = {}
    for sp in out:
        packed[f"X_{sp}"] = np.asarray(out[sp]["X"], dtype=np.float32)
        packed[f"y_{sp}"] = np.asarray(out[sp]["y"], dtype=np.int64)
        packed[f"s_{sp}"] = np.asarray(out[sp]["speed"], dtype=np.int64)
    np.savez_compressed(path, **packed)
    return packed


if __name__ == "__main__":
    import sys
    fe = sys.argv[1] if len(sys.argv) > 1 else "lfes"
    d = build(fe)
    for sp in ("train", "val", "test"):
        X, y, s = d[f"X_{sp}"], d[f"y_{sp}"], d[f"s_{sp}"]
        print(f"{sp:6s} X{X.shape}  classes={np.bincount(y)}  speeds={np.bincount(s//200)[3:]}")
