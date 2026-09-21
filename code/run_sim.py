"""The controlled experiment: a speed sweep with paired samples.

Real data cannot test the margin condition properly, because there is no way to
observe the same realisation at two speeds, so the displacement has to be
estimated from class means.  In simulation the pairing is exact: generate one
long signal, then read it at two rates.  That makes delta directly measurable
and turns the margin ratio from something estimated into something observed.

The prediction under test is narrow and falsifiable: accuracy should hold while
M(rho) > 1 and start to fall as M crosses 1.

    OMP_NUM_THREADS=1 python3 run_sim.py    # about 30 min on eight cores

Options widen the sweep without changing the default run:

    --seeds N              training seeds 0 to N-1 (default 3)
    --ball-orders A B ...  repeat the sweep with each ball defect order in place of
                           ORDERS["ball"]; writes results/sim_ball_order.json
    --nproc P              worker processes (default 8)
    --out NAME             output file name in the results directory
"""
import argparse, json, os, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
from multiprocessing import Pool

import sim, features, central, margin, paths
from features import CLASSES

OUT = paths.RESULTS

FS, WIN = 25600, 8192
FR0 = 10.0                       # 600 rpm, the training speed
RHOS = [1.0, 1.1, 1.2, 1.35, 1.5, 1.67, 1.85, 2.0, 2.25, 2.5]
N_PER_CLASS = 200
SNR_DB = 5.0
NBINS, FMIN, FMAX, PAD = 256, 20.0, 2000.0, 8


def dilate(x, rho, n_out):
    """y(t) = x(rho t).  The transformation the theory is about."""
    return np.interp(np.arange(n_out) * rho, np.arange(len(x)), x,
                     left=0.0, right=0.0)


def _one_pair(args):
    """One realisation, rendered at the base speed and at every rho, paired."""
    cls, seed, front_end, ball_order = args
    fe = {"lfes": features.lfes, "ses": features.ses}[front_end]
    need = int(WIN * (max(RHOS) + 0.3))
    order = ball_order if cls == "ball" else None
    x = sim.simulate(cls, FR0, fs=FS, n=need, snr_db=SNR_DB, seed=seed, order=order)
    out = {}
    for rho in RHOS:
        seg = x[:WIN] if rho == 1.0 else dilate(x, rho, WIN)
        v, _ = fe(seg, FS, fmin=FMIN, fmax=FMAX, nbins=NBINS, pad=PAD)
        out[rho] = v.astype(np.float32)
    return cls, seed, out


def build(front_end, nproc=8, ball_order=None):
    jobs = [(c, s, front_end, ball_order) for c in CLASSES for s in range(N_PER_CLASS)]
    with Pool(nproc) as pool:
        res = pool.map(_one_pair, jobs)
    X = {rho: [] for rho in RHOS}
    y = []
    for cls, seed, out in res:
        y.append(CLASSES.index(cls))
        for rho in RHOS:
            X[rho].append(out[rho])
    return {rho: np.stack(X[rho]) for rho in RHOS}, np.asarray(y, dtype=np.int64)


# features of the sweep being run, shared with the worker processes by fork
_DATA = {}


def _one_seed(args):
    """Train on the base speed with one seed, then evaluate at every rho."""
    front_end, seed, rounds = args
    X, y = _DATA["X"], _DATA["y"]
    n = len(y)
    rng = np.random.default_rng(100 + seed)
    perm = rng.permutation(n)
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    sel = np.zeros(n, dtype=bool); sel[tr] = True
    model = central.train_central(X[1.0], y, sel, rounds=rounds, seed=seed,
                                  nbins=NBINS)
    z_tr, _ = _embed(model, X[1.0][tr])
    P = margin.prototypes(z_tr, y[tr])
    r = margin.radii(z_tr, y[tr], P)
    Delta = margin.separation(P)
    r_bar = float(np.nanmax(r))
    rows = []
    for rho in RHOS:
        z_a, _ = _embed(model, X[1.0][te])
        z_b, logits = _embed(model, X[rho][te])
        delta = margin.displacement_paired(z_a, z_b)      # exact, paired
        # the estimator available on real data: displacement of the class means
        d_mean, d_max = margin.displacement_means(z_a, y[te], z_b, y[te])
        acc_lin = float((logits.argmax(1) == y[te]).mean())
        acc_np = margin.nearest_prototype_acc(z_b, y[te], P)
        rows.append(dict(front_end=front_end, seed=seed, rho=rho,
                         Delta=Delta, r_bar=r_bar, delta=delta,
                         delta_means=d_mean, delta_means_max=d_max,
                         M=margin.margin_ratio(Delta, r_bar, delta),
                         M_means=margin.margin_ratio(Delta, r_bar, d_mean),
                         acc_linear=acc_lin, acc_proto=acc_np))
        print(f"  {front_end} seed{seed} rho={rho:.2f}  delta={delta:.3f} "
              f"M={rows[-1]['M']:.2f}  acc={acc_lin:.3f}/{acc_np:.3f}", flush=True)
    return rows


def run(front_end, seeds=(0, 1, 2), rounds=150, nproc=8, ball_order=None):
    _DATA["X"], _DATA["y"] = build(front_end, nproc, ball_order)
    jobs = [(front_end, seed, rounds) for seed in seeds]
    with Pool(min(nproc, len(jobs))) as pool:
        res = pool.map(_one_seed, jobs)
    return [row for rows in res for row in rows]


@torch.no_grad()
def _embed(model, X, bs=512):
    model.eval()
    zs, ls = [], []
    for i in range(0, len(X), bs):
        logit, z = model(torch.from_numpy(X[i:i + bs]))
        zs.append(z.numpy()); ls.append(logit.numpy())
    return np.concatenate(zs), np.concatenate(ls)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ball-orders", type=float, nargs="+", default=None)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    name = a.out or ("sim_ball_order.json" if a.ball_orders else "sim_sweep.json")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows = []
    for bo in (a.ball_orders or [None]):
        for fe in ("lfes", "ses"):
            tag = "" if bo is None else f", ball order {bo:.2f}"
            print(f"=== {fe}{tag} ===", flush=True)
            part = run(fe, seeds=tuple(range(a.seeds)), nproc=a.nproc, ball_order=bo)
            if bo is not None:
                for r in part:
                    r["ball_order"] = bo
            rows += part
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {os.path.join(OUT, name)} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
