"""Turn results/summary.json into LaTeX macros, so no number is typed by hand."""
import json, os, sys
import numpy as np

import paths

RES = paths.RESULTS
S = json.load(open(os.path.join(RES, "summary.json")))
L = []


def cmd(name, value):
    L.append(r"\newcommand{\%s}{%s}" % (name, value))


def pct(x, d=1):
    return f"{100*x:.{d}f}"


def pval(x):
    """A p-value in conventional notation."""
    if x < 1e-4:
        e = int(np.floor(np.log10(x)))
        return r"%.1f\times 10^{%d}" % (x / 10 ** e, e)
    return f"{x:.4f}".rstrip("0")


def signed(x, d=1):
    """A signed figure, except that a value rounding to zero takes no sign."""
    v = round(x, d)
    if abs(v) < 0.5 * 10 ** -d:
        return "$%.*f$" % (d, 0.0)
    return "$%+.*f$" % (d, v)


# --- setup constants ---------------------------------------------------------
import data as D, features as F, models
cmd("NBins", 256); cmd("FminHz", 20); cmd("FmaxHz", 2000)
cmd("ProjDim", 64)
cmd("WinLen", D.WIN); cmd("HopTrain", D.HOP_TRAIN); cmd("HopEval", D.HOP_EVAL)
cmd("SplitTrain", int(D.SPLIT[0]*100)); cmd("SplitVal", int(D.SPLIT[1]*100))
cmd("SplitTest", int(D.SPLIT[2]*100))
cmd("FaultDur", 10); cmd("NormalDur", 30)
cmd("SimSNR", 5)

com = S["controls"]["communication"]
cmd("ModelParams", f"{com['model_params']:,}".replace(",", "{,}"))
cmd("ProtoFloats", com["proto_floats"])
cmd("ProtoPct", f"{100*com['proto_fraction']:.2f}")

# --- the shift identity ------------------------------------------------------
sh = S["controls"]["shift_identity"]
cmd("ShiftResid", f"{np.mean([r['dist_after_shift'] for r in sh]):.2f}")
cmd("ShiftRaw", f"{np.mean([r['dist_raw'] for r in sh]):.2f}")

# --- architecture control ----------------------------------------------------
arch = [r for r in S["controls"]["architecture"] if r.get("dataset", "jnu") == "jnu"]


def axis_gain(key):
    lo = np.mean([r[key] for r in arch if r["front_end"] == "ses"])
    hi = np.mean([r[key] for r in arch if r["front_end"] == "lfes"])
    return 100 * (hi - lo), 100 * lo, 100 * hi


for key, tag in (("rf_out", "RF"), ("lr_out", "LR"), ("cnn_out", "CNN")):
    g, lo, hi = axis_gain(key)
    cmd(f"{tag}AxisGain", f"{g:+.1f}")
    cmd(f"{tag}Ses", f"{lo:.1f}")
    cmd(f"{tag}Lfes", f"{hi:.1f}")

# --- JNU headline ------------------------------------------------------------
h = S["headline"]
for k in ("lfes_proto_out", "lfes_fedavg_out", "ses_fedavg_out", "ses_proto_out",
          "lfes_proto_in"):
    cmd("".join(w.capitalize() for w in k.split("_")), pct(h[k]))
cmd("LfesProtoOutSd", pct(h["lfes_proto_out_sd"]))
cmd("LfesFedavgOutSd", pct(h["lfes_fedavg_out_sd"]))
cmd("AxisGain", signed(100 * h["axis_gain"]))
cmd("ProtoGain", signed(100 * h["proto_gain"]))

ins = S["in_speed"]
cmd("LfesFedavgIn", pct(ins["lfes|fedavg"])); cmd("SesFedavgIn", pct(ins["ses|fedavg"]))
cmd("LfesProtoInAcc", pct(ins["lfes|proto"])); cmd("SesProtoIn", pct(ins["ses|proto"]))
cmd("LfesProxOut", pct(S["jnu_by_front_method"]["lfes|prox"]["acc_out_linear"][0]))

ax = S["axis_by_dataset"]
for ds, tag in (("jnu", "Jnu"), ("cwru", "Cwru")):
    if ds not in ax:
        continue
    for key, k2 in (("rf_out", "RF"), ("lr_out", "LR"), ("cnn_out", "CNN")):
        d = ax[ds][key]
        cmd(f"{tag}{k2}Gain", signed(100 * d["gain"]))
        cmd(f"{tag}{k2}Ses", pct(d["ses"])); cmd(f"{tag}{k2}Lfes", pct(d["lfes"]))
        cmd(f"{tag}{k2}Order", pct(d["order"]))
        cmd(f"{tag}{k2}VsOrder", signed(100 * d["vs_order"]))
    cmd(f"{tag}CnnIn", pct(ax[ds]["cnn_in"]["lfes"]))
    cmd(f"{tag}CnnInSes", pct(ax[ds]["cnn_in"]["ses"]))
    pc = ax[ds]["per_condition"]
    cmd(f"{tag}GainLo", signed(100 * min(v["gain"] for v in pc.values())))
    cmd(f"{tag}GainHi", signed(100 * max(v["gain"] for v in pc.values())))
import inspect
_osig = inspect.signature(F.order_spectrum).parameters
cmd("OrderMin", "0.5"); cmd("OrderMax", f"{_osig['omax'].default:g}")
cmd("ArchSeeds", S["arch_seeds"])
for ds, tag in (("jnu", "Jnu"), ("cwru", "Cwru")):
    o = S["order_test"][ds]
    cmd(f"{tag}OrderDiff", signed(100 * o["mean"]))
    cmd(f"{tag}OrderP", pval(o["p"]))
    cmd(f"{tag}OrderPW", pval(o["p_wilcoxon"]))
    cmd(f"{tag}OrderN", o["n"]); cmd(f"{tag}OrderWins", o["wins"])
cmd("JnuCnnInOrder", pct(ax["jnu"]["cnn_in"]["order"]))
cmd("GridN", len(json.load(open(os.path.join(RES, "jnu_grid.json")))))

sh = S["shift"]
cmd("ShiftErrAll", f"{100*sh['max_err_all']:.0f}")
cmd("ShiftErrMod", f"{100*sh['max_err_moderate']:.1f}")
cmd("ShiftWorstRho", f"{sh['worst_rho']:.1f}")
cmd("ShiftRhoLoV", f"{sh['rho_lo']:.1f}"); cmd("ShiftRhoHiV", f"{sh['rho_hi']:.1f}")

at = S["axis_test"]
cmd("AxisTestN", at["n"]); cmd("AxisTestMean", f"{100*at['mean']:+.1f}")
cmd("AxisTestP", pval(at["p"])); cmd("AxisTestPW", pval(at["p_wilcoxon"]))
cmd("AxisTestWins", at["wins"])
cmd("AxisTestCI", f"{100*at['ci_lo']:+.1f}$ to $%+.1f" % (100 * at["ci_hi"]))
pt = S["proto_test"]
cmd("ProtoTestN", pt["n"]); cmd("ProtoTestMean", f"{100*pt['mean']:+.1f}")
cmd("ProtoTestP", f"{pt['p']:.2f}"); cmd("ProtoTestWins", pt["wins"])

# --- margin ------------------------------------------------------------------
m = S["margin_jnu"]
cmd("MjnuMin", f"{m['M_min']:.2f}"); cmd("MjnuMax", f"{m['M_max']:.2f}")
cmd("MjnuMean", f"{m['M_mean']:.2f}")
cmd("AccWhenMlow", pct(m["acc_when_M_below_one"]))
cmd("AccWhenMlowLog", pct(m["acc_when_M_below_one_log"]))
cmd("SpearmanMacc", f"{m['spearman_M_acc']:.2f}")
pg = S["proto_geometry"]
cmd("MfedavgLog", f"{pg['fedavg']['M']:.2f}"); cmd("MprotoLog", f"{pg['proto']['M']:.2f}")
cmd("ProtoDeltaRatio", f"{pg['ratio']['Delta']:.2f}")
cmd("ProtoRadiusRatio", f"{pg['ratio']['r_bar']:.2f}")
cmd("ProtoDispRatio", f"{pg['ratio']['delta_mean']:.2f}")
cmd("FracMaboveOne", pct(m["frac_above_one"], 0))

for fe in ("lfes", "ses"):
    s = S[f"sim_{fe}"]
    tag = "Log" if fe == "lfes" else "Lin"
    cmd(f"Sim{tag}AccOne", pct(s["acc_at_rho1"]))
    cmd(f"Sim{tag}RhoDrop", "n/a" if s["rho_acc_drops_10pct"] is None
        else f"{s['rho_acc_drops_10pct']:.2f}")
    cmd(f"Sim{tag}RhoCross", "n/a" if s["rho_M_crosses_one"] is None
        else f"{s['rho_M_crosses_one']:.2f}")
    cmd(f"Sim{tag}Spearman", f"{s['spearman_M_acc']:.2f}")
    cmd(f"Sim{tag}AccLast", pct(s["acc"][-1]))
    cmd(f"Sim{tag}DeltaLast", f"{s['delta'][-1]:.3f}")
    cmd(f"Sim{tag}DeltaOne", f"{s['delta'][1]:.3f}")
    cmd(f"Sim{tag}RhoLast", f"{s['rhos'][-1]:.2f}")
    cmd(f"Sim{tag}Floor", pct(s["acc_floor"]))
    cmd(f"Sim{tag}UnderLo", f"{s['underest_lo']:.2f}")
    cmd(f"Sim{tag}UnderHi", f"{s['underest_hi']:.2f}")
    cmd(f"Sim{tag}UnderMed", f"{s['underest_med']:.2f}")

cmd("OrderOuterFault", f"{F.ORDERS['outer']:.2f}"); cmd("OrderBallFault", f"{F.ORDERS['ball']:.2f}")

# --- simulated log-axis accuracy between the base speed and rho = 2 ---------------
_sl = S["sim_lfes"]
_mid = [a for r, a in zip(_sl["rhos"], _sl["acc"]) if 1.0 < r <= 2.0]
cmd("SimLogAccMidLo", pct(min(_mid)))
cmd("SimLogAccMidHi", pct(max(_mid)))
cmd("SimLogMaboveOne", sum(1 for r, m in zip(_sl["rhos"], _sl["M"]) if r > 1.0 and m > 1.0))
cmd("SimLogNRatios", sum(1 for r in _sl["rhos"] if r > 1.0))
cmd("SimNSeeds", S["sim_by_front_rho"][f"lfes|{_sl['rhos'][0]}"]["n"])

# --- simulated sweep over training seeds and ball defect orders (optional) --------
B = S.get("sim_ball_order")
if B:
    _bo = sorted({float(k.split("|")[0]) for k in B})
    _g = f"{F.ORDERS['ball']:.2f}"
    for fe, tag in (("lfes", "Log"), ("ses", "Lin")):
        s = B[f"{_g}|{fe}"]
        cmd(f"SimSeed{tag}Spearman", f"{s['spearman_M_acc']:.2f}")
        cmd(f"SimSeed{tag}SpearmanMean", f"{s['spearman_seed_mean']:.2f}")
        cmd(f"SimSeed{tag}SpearmanSd", f"{s['spearman_seed_sd']:.2f}")
        cmd(f"SimSeed{tag}SpearmanMin", f"{s['spearman_seed_min']:.2f}")
        cmd(f"SimSeed{tag}SpearmanMax", f"{s['spearman_seed_max']:.2f}")
        cmd(f"SimSeed{tag}AccMidLo", pct(s["acc_mid_lo"]))
        cmd(f"SimSeed{tag}AccMidHi", pct(s["acc_mid_hi"]))
        cmd(f"SimSeed{tag}AccSdMax", pct(s["acc_sd_max"]))
        cmd(f"SimSeed{tag}MaboveOne", s["M_above_one"])
        cmd(f"SimSeed{tag}UnderLo", f"{s['underest_lo']:.2f}")
        cmd(f"SimSeed{tag}UnderHi", f"{s['underest_hi']:.2f}")
        rows = [B[f"{b:.2f}|{fe}"] for b in _bo]
        for key, mac in (("spearman_M_acc", "Spearman"),
                         ("spearman_seed_mean", "SpearmanSeed")):
            v = [r[key] for r in rows]
            cmd(f"Ball{tag}{mac}Lo", f"{min(v):.2f}")
            cmd(f"Ball{tag}{mac}Hi", f"{max(v):.2f}")
        for key, mac in (("acc_mid_lo", "AccMidLo"), ("acc_mid_hi", "AccMidHi")):
            v = [r[key] for r in rows]
            cmd(f"Ball{tag}{mac}Min", pct(min(v)))
            cmd(f"Ball{tag}{mac}Max", pct(max(v)))
        cmd(f"Ball{tag}UnderLo", f"{min(r['underest_lo'] for r in rows):.2f}")
        cmd(f"Ball{tag}UnderHi", f"{max(r['underest_hi'] for r in rows):.2f}")
        cmd(f"Ball{tag}UnderStraddle",
            sum(1 for r in rows if r["underest_lo"] < 1.0 < r["underest_hi"]))
    cmd("SimSeedsN", B[f"{_g}|lfes"]["n_seeds"])
    cmd("BallSeedsN", min(B[k]["n_seeds"] for k in B))
    cmd("BallOrdersN", len(_bo))
    cmd("BallOrderLo", f"{min(_bo):.2f}"); cmd("BallOrderHi", f"{max(_bo):.2f}")

# --- CWRU speed range, from the RPM recorded in each file ---------------------
import scipy.io, cwru
rec = {}
for num in cwru.FILES:
    d = scipy.io.loadmat(os.path.join(cwru.RAW, f"{num}.mat"))
    r = [float(d[k].ravel()[0]) for k in d if k.endswith("RPM")]
    if r:
        rec[num] = r[0]
rho_cwru = max(rec.values()) / min(rec.values())
cmd("CwruRhoMax", f"{rho_cwru:.3f}")
cmd("CwruSpeedPct", pct(rho_cwru - 1))
cmd("CwruNominalErrPct", pct(max(abs(cwru.FILES[n][1] - v) / v for n, v in rec.items())))

# --- provenance --------------------------------------------------------------
import time, torch
cmd("NSeeds", 5); cmd("NRounds", 80)
cmd("RunStamp", time.strftime("%Y-%m-%d"))
cmd("TorchVer", torch.__version__.split("+")[0])
cmd("NumpyVer", np.__version__)

os.makedirs(paths.PUBLISH, exist_ok=True)
with open(os.path.join(paths.PUBLISH, "numbers.tex"), "w") as fh:
    fh.write("% generated by build_numbers.py -- do not edit\n")
    fh.write("\n".join(L) + "\n")
print(f"wrote {len(L)} macros")
