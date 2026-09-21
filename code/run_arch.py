"""The architecture control over three axes and eight seeds.

The convolutional encoder varies between seeds, so eight seeds are run and the
analysis reports the median as well as the mean.

    OMP_NUM_THREADS=1 python3 run_arch.py
"""
import itertools, json, os, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
from multiprocessing import Pool

import data, cwru, central, paths
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

OUT = paths.RESULTS
DATASETS = {"jnu": (data, [600, 800, 1000]), "cwru": (cwru, cwru.LOADS)}
# "order" uses the published nominal shaft speed to remove the dilation exactly. It is
# the natural comparison for a method that claims to work without a tachometer.
FRONT_ENDS = ("lfes", "ses", "order")
SEEDS = list(range(8))
NBINS = 256

_C = {}


def _get(mod, fe, tag):
    k = (tag, fe)
    if k not in _C:
        _C[k] = mod.build(fe)
    return _C[k]


def one(job):
    ds, fe, held, seed = job
    mod, conds = DATASETS[ds]
    d = _get(mod, fe, ds)
    X, y, s = d["X_train"], d["y_train"], d["s_train"]
    Xt, yt, st = d["X_test"], d["y_test"], d["s_test"]
    tr = [c for c in conds if c != held]
    a, b, ain = np.isin(s, tr), st == held, np.isin(st, tr)
    m = central.train_central(X, y, a, rounds=150, seed=seed, nbins=NBINS)
    return dict(dataset=ds, front_end=fe, held_out=held, seed=seed,
                cnn_in=central.acc(m, Xt[ain], yt[ain]),
                cnn_out=central.acc(m, Xt[b], yt[b]))


def shallow():
    """Random forest and logistic regression, which have no seed noise worth averaging."""
    rows = []
    for ds, (mod, conds) in DATASETS.items():
        for fe in FRONT_ENDS:
            d = mod.build(fe)
            X, y, s = d["X_train"], d["y_train"], d["s_train"]
            Xt, yt, st = d["X_test"], d["y_test"], d["s_test"]
            for held in conds:
                tr = [c for c in conds if c != held]
                a, b, ain = np.isin(s, tr), st == held, np.isin(st, tr)
                rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                            n_jobs=8).fit(X[a], y[a])
                lr = LogisticRegression(max_iter=4000, C=10.0).fit(X[a], y[a])
                rows.append(dict(dataset=ds, front_end=fe, held_out=held,
                                 rf_in=float(rf.score(Xt[ain], yt[ain])),
                                 rf_out=float(rf.score(Xt[b], yt[b])),
                                 lr_in=float(lr.score(Xt[ain], yt[ain])),
                                 lr_out=float(lr.score(Xt[b], yt[b]))))
                print(f"  {ds} {fe} held={held}: RF {rows[-1]['rf_out']:.3f} "
                      f"LR {rows[-1]['lr_out']:.3f}", flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    sh = shallow()
    jobs = [(ds, fe, held, seed)
            for ds, (_, conds) in DATASETS.items()
            for fe in FRONT_ENDS for held in conds for seed in SEEDS]
    print(f"{len(jobs)} CNN runs", flush=True)
    with Pool(8) as pool:
        deep = []
        for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
            deep.append(r)
            if i % 16 == 0:
                print(f"  [{i}/{len(jobs)}]", flush=True)
    with open(os.path.join(OUT, "arch3.json"), "w") as fh:
        json.dump(dict(shallow=sh, cnn=deep, seeds=len(SEEDS)), fh, indent=1)
    print(f"wrote results/arch3.json in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
