"""Turn the raw experiment output into the reported numbers and tables.

Reads results/{jnu_grid,sim_sweep,controls,arch3}.json and writes
results/summary.json.  Every reported number is computed here or in
build_numbers.py from this summary.
"""
import json, os
from collections import defaultdict
import numpy as np

import paths

RES = paths.RESULTS


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def agg(rows, keys, fields):
    """Mean and sd of `fields`, grouped by `keys`."""
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {f: (float(np.mean([r[f] for r in rs])),
                      float(np.std([r[f] for r in rs]))) for f in fields}
        out[k]["n"] = len(rs)
    return out


def main():
    S = {}
    grid = load("jnu_grid.json")
    F = ["acc_in_linear", "acc_out_linear", "acc_in_proto", "acc_out_proto",
         "Delta", "r_bar", "delta_mean", "delta_max", "M"]

    by_fm = agg(grid, ["front_end", "method"], F)
    by_fmh = agg(grid, ["front_end", "method", "held_out"], F)
    S["jnu_by_front_method"] = {"|".join(map(str, k)): v for k, v in by_fm.items()}
    S["jnu_by_front_method_speed"] = {"|".join(map(str, k)): v for k, v in by_fmh.items()}

    # headline: proposed (lfes+proto) against the two obvious controls
    def get(fe, me, field, idx=0):
        return by_fm[(fe, me)][field][idx]

    S["headline"] = dict(
        lfes_proto_out=get("lfes", "proto", "acc_out_proto"),
        lfes_proto_out_sd=get("lfes", "proto", "acc_out_proto", 1),
        lfes_fedavg_out=get("lfes", "fedavg", "acc_out_linear"),
        lfes_fedavg_out_sd=get("lfes", "fedavg", "acc_out_linear", 1),
        ses_fedavg_out=get("ses", "fedavg", "acc_out_linear"),
        ses_fedavg_out_sd=get("ses", "fedavg", "acc_out_linear", 1),
        ses_proto_out=get("ses", "proto", "acc_out_proto"),
        lfes_proto_in=get("lfes", "proto", "acc_in_proto"),
        axis_gain=get("lfes", "fedavg", "acc_out_linear") - get("ses", "fedavg", "acc_out_linear"),
        proto_gain=get("lfes", "proto", "acc_out_proto") - get("lfes", "fedavg", "acc_out_linear"),
    )

    # in-speed accuracy, which is where the linear axis wins and the log axis does not
    S["in_speed"] = {f"{fe}|{me}": by_fm[(fe, me)][
        "acc_in_proto" if me == "proto" else "acc_in_linear"][0]
        for fe in ("lfes", "ses") for me in ("fedavg", "prox", "proto")}

    # the margin ratio on JNU, and how often it exceeds 1
    Ms = [r["M"] for r in grid]
    S["margin_jnu"] = dict(M_min=float(np.min(Ms)), M_max=float(np.max(Ms)),
                           M_mean=float(np.mean(Ms)),
                           frac_above_one=float(np.mean(np.array(Ms) > 1.0)),
                           acc_when_M_below_one=float(np.mean(
                               [r["acc_out_proto"] for r in grid if r["M"] <= 1.0])),
                           acc_when_M_below_one_log=float(np.mean(
                               [r["acc_out_proto"] for r in grid
                                if r["M"] <= 1.0 and r["front_end"] == "lfes"])))
    # does M rank transfers correctly even where it does not gate them?
    lf = [r for r in grid if r["front_end"] == "lfes"]
    m = np.array([r["M"] for r in lf]); a = np.array([r["acc_out_proto"] for r in lf])
    S["margin_jnu"]["spearman_M_acc"] = float(_spearman(m, a))
    # what prototype alignment does to each component of the geometry
    def geo(me):
        r = [x for x in lf if x["method"] == me]
        return {k: float(np.mean([x[k] for x in r]))
                for k in ("Delta", "r_bar", "delta_mean", "M")}
    fa, pr = geo("fedavg"), geo("proto")
    S["proto_geometry"] = dict(fedavg=fa, proto=pr,
                               ratio={k: pr[k] / fa[k] for k in
                                      ("Delta", "r_bar", "delta_mean")})
    S["margin_jnu"]["pearson_M_acc"] = float(np.corrcoef(m, a)[0, 1])

    # simulator sweep
    sim = load("sim_sweep.json")
    by_fr = agg(sim, ["front_end", "rho"],
                ["delta", "M", "acc_linear", "acc_proto", "Delta", "r_bar"])
    S["sim_by_front_rho"] = {"|".join(map(str, k)): v for k, v in by_fr.items()}
    for fe in ("lfes", "ses"):
        rows = sorted([r for r in sim if r["front_end"] == fe], key=lambda r: r["rho"])
        rhos = sorted({r["rho"] for r in rows})
        acc = [np.mean([r["acc_proto"] for r in rows if r["rho"] == x]) for x in rhos]
        MM = [np.mean([r["M"] for r in rows if r["rho"] == x]) for x in rhos]
        dd = [np.mean([r["delta"] for r in rows if r["rho"] == x]) for x in rhos]
        base = acc[0]
        drop90 = next((x for x, a_ in zip(rhos, acc) if a_ < 0.9 * base), None)
        cross = next((x for x, m_ in zip(rhos, MM) if m_ < 1.0), None)
        dm = [np.mean([r["delta_means"] for r in rows if r["rho"] == x]) for x in rhos]
        Mm = [np.mean([r["M_means"] for r in rows if r["rho"] == x]) for x in rhos]
        ratio = [d / m for d, m in zip(dd[1:], dm[1:]) if m > 1e-6]
        S[f"sim_{fe}"] = dict(rhos=rhos, acc=acc, M=MM, delta=dd,
                              delta_means=dm, M_means=Mm,
                              underest_lo=float(min(ratio)), underest_hi=float(max(ratio)),
                              underest_med=float(np.median(ratio)),
                              acc_floor=float(min(acc)), chance=0.25,
                              acc_at_rho1=float(base),
                              rho_acc_drops_10pct=drop90,
                              rho_M_crosses_one=cross,
                              spearman_M_acc=float(_spearman(np.array(MM), np.array(acc))))

    # optional: the sweep over training seeds and ball defect orders
    if os.path.exists(os.path.join(RES, "sim_ball_order.json")):
        S["sim_ball_order"] = _ball_order_summary(load("sim_ball_order.json"))

    C = load("controls.json")
    S["controls"] = C

    # Paired tests over matched (method, held-out speed, seed) runs.
    from scipy import stats as _st
    kf = lambda r: (r["method"], r["held_out"], r["seed"])
    _lf = {kf(r): r for r in grid if r["front_end"] == "lfes"}
    _se = {kf(r): r for r in grid if r["front_end"] == "ses"}
    com = sorted(set(_lf) & set(_se))
    val = lambda r: r["acc_out_proto"] if r["method"] == "proto" else r["acc_out_linear"]
    A = np.array([val(_lf[k]) for k in com]); B = np.array([val(_se[k]) for k in com])
    t, pv = _st.ttest_rel(A, B); _, pw = _st.wilcoxon(A, B)
    ci = _st.t.interval(0.95, len(A) - 1, loc=(A - B).mean(), scale=_st.sem(A - B))
    S["axis_test"] = dict(n=len(com), mean=float((A - B).mean()),
                          sd=float((A - B).std(ddof=1)), t=float(t), p=float(pv),
                          p_wilcoxon=float(pw), wins=int((A > B).sum()),
                          ci_lo=float(ci[0]), ci_hi=float(ci[1]))

    k2 = lambda r: (r["held_out"], r["seed"])
    P1 = {k2(r): r for r in grid if r["front_end"] == "lfes" and r["method"] == "proto"}
    P0 = {k2(r): r for r in grid if r["front_end"] == "lfes" and r["method"] == "fedavg"}
    kk = sorted(set(P1) & set(P0))
    X = np.array([P1[k]["acc_out_proto"] for k in kk])
    Y = np.array([P0[k]["acc_out_linear"] for k in kk])
    t2, p2 = _st.ttest_rel(X, Y)
    S["proto_test"] = dict(n=len(kk), mean=float((X - Y).mean()), t=float(t2),
                           p=float(p2), wins=int((X > Y).sum()))
    # The architecture control comes from arch3.json: eight seeds over three
    # axes, linear, logarithmic, and order normalised against the published shaft
    # speed.  The shift identity and the communication accounting come from
    # controls.json.
    A = load("arch3.json")
    AXES = ("ses", "lfes", "order")
    S["arch_seeds"] = A["seeds"]
    S["axis_by_dataset"] = {}
    for ds in sorted({r["dataset"] for r in A["cnn"]}):
        row = {}
        for key in ("rf_out", "lr_out", "rf_in", "lr_in"):
            d = {fe: float(np.mean([r[key] for r in A["shallow"]
                                    if r["dataset"] == ds and r["front_end"] == fe]))
                 for fe in AXES}
            d["gain"] = d["lfes"] - d["ses"]
            d["vs_order"] = d["lfes"] - d["order"]
            row[key] = d
        for key in ("cnn_out", "cnn_in"):
            d = {}
            for fe in AXES:
                v = np.array([r[key] for r in A["cnn"]
                              if r["dataset"] == ds and r["front_end"] == fe])
                d[fe] = float(np.median(v))
                d[fe + "_mean"] = float(v.mean())
                d[fe + "_sd"] = float(v.std())
            d["gain"] = d["lfes"] - d["ses"]
            d["vs_order"] = d["lfes"] - d["order"]
            row[key] = d
        # per-condition medians, to show the effect is not carried by one split
        row["per_condition"] = {}
        for held in sorted({r["held_out"] for r in A["cnn"] if r["dataset"] == ds}):
            g = {fe: float(np.median([r["cnn_out"] for r in A["cnn"]
                                      if r["dataset"] == ds and r["front_end"] == fe
                                      and r["held_out"] == held]))
                 for fe in AXES}
            row["per_condition"][held] = dict(gain=g["lfes"] - g["ses"],
                                              vs_order=g["lfes"] - g["order"], **g)
        S["axis_by_dataset"][ds] = row
        # Order normalisation: with the shaft speed known, dividing the frequency
        # axis by it removes the dilation exactly.  Pair by held-out condition
        # and seed, as everywhere else.
        byk = {}
        for r in A["cnn"]:
            if r["dataset"] == ds:
                byk.setdefault((r["held_out"], r["seed"]), {})[r["front_end"]] = r["cnn_out"]
        kk = sorted(k for k, v in byk.items() if {"lfes", "order"} <= set(v))
        X = np.array([byk[k]["lfes"] for k in kk])
        Y = np.array([byk[k]["order"] for k in kk])
        t3, p3 = _st.ttest_rel(X, Y)
        w3 = _st.wilcoxon(X, Y) if np.any(X != Y) else None
        S.setdefault("order_test", {})[ds] = dict(
            n=len(kk), mean=float((X - Y).mean()), t=float(t3), p=float(p3),
            p_wilcoxon=float(w3.pvalue) if w3 is not None else float("nan"),
            wins=int((X > Y).sum()))
    # the shift identity, split by how far the ratio is pushed
    sh = C["shift_identity"]
    err = [(r["rho"], abs(r["shift_measured"] - r["shift_predicted"]) / r["shift_predicted"])
           for r in sh]
    S["shift"] = dict(
        max_err_all=float(max(e for _, e in err)),
        max_err_moderate=float(max(e for rho, e in err if rho <= 1.67)),
        worst_rho=float(max(err, key=lambda t: t[1])[0]),
        rho_lo=float(min(r["rho"] for r in sh)), rho_hi=float(max(r["rho"] for r in sh)))
    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    print("wrote results/summary.json")
    return S


def _curve_summary(rows):
    """Seed-mean curves over rho, and the per-seed spread, for one sweep."""
    rhos = sorted({r["rho"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    at = {(r["seed"], r["rho"]): r for r in rows}
    acc = np.array([[at[s, x]["acc_proto"] for x in rhos] for s in seeds])
    M = np.array([[at[s, x]["M"] for x in rhos] for s in seeds])
    dd = np.array([[at[s, x]["delta"] for x in rhos] for s in seeds])
    dm = np.array([[at[s, x]["delta_means"] for x in rhos] for s in seeds])
    acc_m, M_m, dd_m, dm_m = acc.mean(0), M.mean(0), dd.mean(0), dm.mean(0)
    mid = [i for i, x in enumerate(rhos) if 1.0 < x <= 2.0]
    ratio = [d / m for d, m in zip(dd_m[1:], dm_m[1:]) if m > 1e-6]
    per_seed = [float(_spearman(M[i], acc[i])) for i in range(len(seeds))]
    return dict(n_seeds=len(seeds), rhos=rhos,
                acc=acc_m.tolist(), acc_sd=acc.std(0).tolist(), M=M_m.tolist(),
                spearman_M_acc=float(_spearman(M_m, acc_m)),
                spearman_seed=per_seed,
                spearman_seed_mean=float(np.mean(per_seed)),
                spearman_seed_sd=float(np.std(per_seed)),
                spearman_seed_min=float(np.min(per_seed)),
                spearman_seed_max=float(np.max(per_seed)),
                acc_mid_lo=float(acc_m[mid].min()), acc_mid_hi=float(acc_m[mid].max()),
                acc_sd_max=float(acc.std(0).max()),
                acc_floor=float(acc_m.min()),
                M_above_one=int(sum(1 for x, m in zip(rhos, M_m) if x > 1.0 and m > 1.0)),
                underest_lo=float(min(ratio)), underest_hi=float(max(ratio)),
                underest_med=float(np.median(ratio)))


def _ball_order_summary(rows):
    out = {}
    for bo in sorted({r["ball_order"] for r in rows}):
        for fe in sorted({r["front_end"] for r in rows}):
            sel = [r for r in rows if r["ball_order"] == bo and r["front_end"] == fe]
            out[f"{bo:.2f}|{fe}"] = _curve_summary(sel)
    return out


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return np.corrcoef(ra, rb)[0, 1]


if __name__ == "__main__":
    main()
