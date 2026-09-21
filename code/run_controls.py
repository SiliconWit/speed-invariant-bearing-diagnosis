"""Two controls that isolate the mechanism, and the communication accounting.

Control 1, the shift identity.  If a speed change really is a translation on the
log axis, the translation that best realigns two spectra should equal
log(rho)/(bin width in log units), with no fitting.  This is checked directly.

Control 2, the one that matters.  A log axis is only worth anything to a
classifier that can exploit a translation.  Random forests and logistic
regression read individual bins and cannot, so for them the log axis should give
no advantage at all; a convolutional encoder with global pooling should show a
large one.  Running both pins the benefit to the interaction between the axis and
the architecture rather than to the axis alone.

    OMP_NUM_THREADS=1 python3 run_controls.py
"""
import json, os, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(2)

import sim, features, data, cwru, central, margin, paths
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

OUT = paths.RESULTS
FS, WIN = 25600, 8192
NBINS, FMIN, FMAX, PAD = 256, 20.0, 2000.0, 8


def dilate(x, rho, n_out):
    return np.interp(np.arange(n_out) * rho, np.arange(len(x)), x, left=0.0, right=0.0)


def example_pair(rho=1.67, seed=3):
    """One realisation at two speeds, for the left panel of the identity figure."""
    x = sim.simulate("outer", 10.0, fs=FS, n=int(WIN * (rho + 0.3)), snr_db=10, seed=seed)
    a, b = x[:WIN], dilate(x, rho, WIN)
    va, c = features.lfes(a, FS, fmin=FMIN, fmax=FMAX, nbins=NBINS, pad=PAD)
    vb, _ = features.lfes(b, FS, fmin=FMIN, fmax=FMAX, nbins=NBINS, pad=PAD)
    return dict(rho=rho, centres=c.tolist(), a=va.tolist(), b=vb.tolist())


def shift_identity(rhos=(1.1, 1.2, 1.35, 1.5, 1.67, 2.0, 2.5), n_seed=24):
    """Best realigning shift against the value the identity predicts."""
    per_bin = np.log(FMAX / FMIN) / NBINS
    rows = []
    for rho in rhos:
        got, resid, raw = [], [], []
        for seed in range(n_seed):
            x = sim.simulate("outer", 10.0, fs=FS, n=int(WIN * (rho + 0.3)),
                             snr_db=10, seed=seed)
            a, b = x[:WIN], dilate(x, rho, WIN)
            va, _ = features.lfes(a, FS, fmin=FMIN, fmax=FMAX, nbins=NBINS, pad=PAD)
            vb, _ = features.lfes(b, FS, fmin=FMIN, fmax=FMAX, nbins=NBINS, pad=PAD)
            raw.append(float(np.linalg.norm(va - vb)))
            cand = [(float(np.linalg.norm(np.roll(va, k) - vb)), k)
                    for k in range(-90, 91)]
            v, k = min(cand)
            got.append(k); resid.append(v)
        rows.append(dict(rho=rho, shift_measured=float(np.mean(got)),
                         shift_predicted=float(np.log(rho) / per_bin),
                         shift_sd=float(np.std(got)),
                         dist_raw=float(np.mean(raw)),
                         dist_after_shift=float(np.mean(resid))))
        print(f"  rho={rho:.2f}  measured {rows[-1]['shift_measured']:6.1f} "
              f"predicted {rows[-1]['shift_predicted']:6.1f} bins", flush=True)
    return rows


DATASETS = {"jnu": (data, [600, 800, 1000]), "cwru": (cwru, cwru.LOADS)}


def architecture_control(seeds=(0, 1, 2), which="jnu"):
    """Does the log axis help a classifier that cannot use a translation?

    Run on both datasets.  JNU spans a speed ratio of 1.67 and CWRU of 1.044, so
    the theory predicts a large axis effect on the first and none on the second.
    A method that helps on both is not working for the reason claimed.
    """
    mod, conds = DATASETS[which]
    rows = []
    for fe in ("lfes", "ses"):
        d = mod.build(fe)
        X, y, s = d["X_train"], d["y_train"], d["s_train"]
        Xt, yt, st = d["X_test"], d["y_test"], d["s_test"]
        for held in conds:
            tr = [sp for sp in conds if sp != held]
            a, b = np.isin(s, tr), st == held
            ain = np.isin(st, tr)
            rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                        n_jobs=2).fit(X[a], y[a])
            lr = LogisticRegression(max_iter=4000, C=10.0).fit(X[a], y[a])
            cnn_in, cnn_out = [], []
            for seed in seeds:
                m = central.train_central(X, y, a, rounds=150, seed=seed, nbins=NBINS)
                cnn_in.append(central.acc(m, Xt[ain], yt[ain]))
                cnn_out.append(central.acc(m, Xt[b], yt[b]))
            rows.append(dict(dataset=which, front_end=fe, held_out=held,
                             rf_in=float(rf.score(Xt[ain], yt[ain])),
                             rf_out=float(rf.score(Xt[b], yt[b])),
                             lr_in=float(lr.score(Xt[ain], yt[ain])),
                             lr_out=float(lr.score(Xt[b], yt[b])),
                             cnn_in=float(np.mean(cnn_in)),
                             cnn_out=float(np.mean(cnn_out)),
                             cnn_out_sd=float(np.std(cnn_out))))
            print(f"  {which} {fe} held={held}: RF {rows[-1]['rf_out']:.3f}  "
                  f"LR {rows[-1]['lr_out']:.3f}  CNN {rows[-1]['cnn_out']:.3f}", flush=True)
    return rows


def communication(k=64, n_class=4, params=60484, clients=6, rounds=80):
    """Floats on the wire per round, per client, above the model itself."""
    return dict(model_params=params, proto_floats=n_class * k,
                proto_fraction=n_class * k / params,
                prox_extra=params,          # FedProx anchors a full model copy
                adversarial_extra=params,   # a discriminator, order of the encoder
                clients=clients, rounds=rounds)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("shift identity")
    shift = shift_identity()
    print("architecture control: JNU (speed ratio 1.67)")
    arch = architecture_control(which="jnu")
    print("architecture control: CWRU (speed ratio 1.044, predicted null)")
    arch += architecture_control(which="cwru")
    out = dict(shift_identity=shift, example_pair=example_pair(), architecture=arch,
               communication=communication(),
               axis=dict(nbins=NBINS, fmin=FMIN, fmax=FMAX,
                         per_bin_log=float(np.log(FMAX / FMIN) / NBINS)))
    with open(os.path.join(OUT, "controls.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote results/controls.json in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
