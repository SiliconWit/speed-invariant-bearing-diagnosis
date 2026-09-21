"""The JNU grid: front end x method x held-out speed x seed.

Leave-one-speed-out.  Two of the three speeds are federated across six clients;
the third is never seen in training and is the transfer target.

    OMP_NUM_THREADS=1 python3 run_jnu.py
"""
import itertools, json, os, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)

from multiprocessing import Pool
import data, fed, margin, paths

OUT = paths.RESULTS

FRONT_ENDS = ["lfes", "ses"]
METHODS = ["fedavg", "prox", "proto"]
SPEEDS = [600, 800, 1000]
SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 80

_CACHE = {}


def _get(fe):
    if fe not in _CACHE:
        _CACHE[fe] = data.build(fe)
    return _CACHE[fe]


def one(job):
    fe, method, held, seed = job
    d = _get(fe)
    X, y, s = d["X_train"], d["y_train"], d["s_train"]
    Xv, yv, sv = d["X_test"], d["y_test"], d["s_test"]
    tr = [sp for sp in SPEEDS if sp != held]

    t0 = time.time()
    model, P = fed.train_federated(X, y, s, tr, method=method, rounds=ROUNDS, seed=seed)

    in_sel, out_sel = np.isin(sv, tr), sv == held
    lin_in, np_in, z_in = fed.accuracy(model, Xv[in_sel], yv[in_sel], P)
    lin_out, np_out, z_out = fed.accuracy(model, Xv[out_sel], yv[out_sel], P)

    # geometry from the training-speed embeddings against the held-out speed
    z_tr, _ = fed.embed(model, X[np.isin(s, tr)])
    y_tr = y[np.isin(s, tr)]
    g = margin.geometry(z_tr, y_tr, z_out, yv[out_sel])
    Ptr = g.pop("proto")
    np_in2 = margin.nearest_prototype_acc(z_in, yv[in_sel], Ptr)
    np_out2 = margin.nearest_prototype_acc(z_out, yv[out_sel], Ptr)

    return dict(front_end=fe, method=method, held_out=held, seed=seed,
                rho=max(held, max(tr)) / min(held, min(tr)),
                acc_in_linear=lin_in, acc_out_linear=lin_out,
                acc_in_proto=np_in2, acc_out_proto=np_out2,
                secs=time.time() - t0, **g)


def main():
    os.makedirs(OUT, exist_ok=True)
    for fe in FRONT_ENDS:
        data.build(fe)                        # warm the cache before forking
    jobs = list(itertools.product(FRONT_ENDS, METHODS, SPEEDS, SEEDS))
    print(f"{len(jobs)} runs")
    t0 = time.time()
    with Pool(8) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
            rows.append(r)
            print(f"[{i:3d}/{len(jobs)}] {r['front_end']:5s} {r['method']:7s} "
                  f"held={r['held_out']:4d} seed={r['seed']} "
                  f"out_lin={r['acc_out_linear']:.3f} out_proto={r['acc_out_proto']:.3f} "
                  f"M={r['M']:.2f}  ({r['secs']:.0f}s)", flush=True)
    with open(os.path.join(OUT, "jnu_grid.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote results/jnu_grid.json in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
