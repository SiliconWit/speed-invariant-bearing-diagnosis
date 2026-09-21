"""The figures, sized for a 6.5 in text block so nothing is scaled down in the PDF."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAVY, GOLD, SLATE, RED, GREEN = "#0F284D", "#B0892C", "#5A6473", "#8C2F1F", "#3E6B4F"
W, H1, H2, H3 = 6.5, 2.3, 2.2, 2.15

STYLE = {
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "figure.dpi": 200,
}
FE_LABEL = {"lfes": "log axis", "ses": "linear axis",
            "order": "order axis (needs a tachometer)"}
ME_LABEL = {"fedavg": "FedAvg", "prox": "FedProx", "proto": "prototype alignment"}


def use_style():
    plt.rcParams.update(STYLE)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_identity(path, spec_a, spec_b, centres, shift_rows, per_bin):
    """The identity: a speed change is a translation of known size."""
    fig, ax = plt.subplots(1, 2, figsize=(W, H2))
    ax[0].semilogx(centres, spec_a, color=NAVY, lw=1.0, label=r"$\rho=1$")
    ax[0].semilogx(centres, spec_b, color=GOLD, lw=1.0, label=r"$\rho=1.67$")
    ax[0].set_xlabel("envelope frequency (Hz, log axis)")
    ax[0].set_ylabel("normalised amplitude")
    ax[0].set_title("the same realisation at two speeds", pad=4)
    lo, hi = ax[0].get_ylim()
    ax[0].set_ylim(lo, hi + 0.30 * (hi - lo))
    ax[0].legend(frameon=False, loc="upper left", handlelength=1.4,
                 borderaxespad=0.2, fontsize=7)

    rho = np.array([r["rho"] for r in shift_rows])
    got = np.array([r["shift_measured"] for r in shift_rows])
    sd = np.array([r["shift_sd"] for r in shift_rows])
    pred = np.array([r["shift_predicted"] for r in shift_rows])
    ax[1].plot(rho, pred, color=SLATE, ls="--", lw=1.0,
               label=r"$\log\rho\,/\,\Delta u$, predicted")
    ax[1].errorbar(rho, got, yerr=sd, fmt="o", ms=3.2, color=NAVY, lw=0.9,
                   capsize=2, label="measured realignment")
    ax[1].set_xlabel(r"speed ratio $\rho$")
    ax[1].set_ylabel("shift (bins)")
    ax[1].set_title("no fitted parameters", pad=4)
    lo, hi = ax[1].get_ylim()
    ax[1].set_ylim(lo, hi + 0.28 * (hi - lo))
    ax[1].legend(frameon=False, loc="upper left", handlelength=1.6,
                 borderaxespad=0.2, fontsize=7)
    fig.tight_layout(w_pad=1.5)
    _save(fig, path)


def fig_architecture(path, rows):
    """The axis pays off only for the architecture that can use it, and only where
    there is a dilation to use it on."""
    fig, ax = plt.subplots(1, 2, figsize=(W, H2), sharey=True)
    models = [("rf_out", "random\nforest"), ("lr_out", "logistic\nregression"),
              ("cnn_out", "CNN, global\npooling")]
    titles = {"jnu": r"JNU, $\rho_{\max}=1.67$", "cwru": r"CWRU, $\rho_{\max}=1.044$"}
    x = np.arange(len(models))
    # three axes; the third is the baseline that is given the shaft speed
    axes_shown = ("ses", "lfes", "order")
    colour = {"ses": SLATE, "lfes": NAVY, "order": GOLD}
    wdt = 0.26
    for i, ds in enumerate(("jnu", "cwru")):
        a_ = ax[i]
        for j, fe in enumerate(axes_shown):
            r = next(r for r in rows if r["dataset"] == ds and r["front_end"] == fe)
            a_.bar(x + (j - 1) * wdt, [r[k] for k, _ in models], wdt,
                   color=colour[fe], label=FE_LABEL[fe], edgecolor="none")
        a_.set_xticks(x)
        a_.set_xticklabels([m[1] for m in models], fontsize=7)
        a_.axhline(0.25, color=RED, ls=":", lw=0.8)
        a_.set_title(titles[ds], pad=4)
        a_.set_ylim(0, 1.16)
        if i == 0:
            a_.set_ylabel("accuracy at the unseen condition")
            a_.legend(frameon=False, loc="upper left", handlelength=1.1,
                      borderaxespad=0.2, fontsize=6.4, ncol=1)
            a_.text(2.45, 0.27, "chance", fontsize=6.4, color=RED, ha="right")
    fig.tight_layout(w_pad=1.2)
    _save(fig, path)


def fig_jnu(path, summary):
    """The federated grid: front end and method, per held-out speed."""
    fig, ax = plt.subplots(1, 3, figsize=(W, H3), sharey=True)
    speeds = [600, 800, 1000]
    methods = ["fedavg", "prox", "proto"]
    x = np.arange(len(methods))
    wdt = 0.36
    tab = summary["jnu_by_front_method_speed"]
    for i, sp in enumerate(speeds):
        for j, fe in enumerate(("ses", "lfes")):
            mu, sd = [], []
            for me in methods:
                key = f"{fe}|{me}|{sp}"
                fld = "acc_out_proto" if me == "proto" else "acc_out_linear"
                mu.append(tab[key][fld][0]); sd.append(tab[key][fld][1])
            ax[i].bar(x + (j - 0.5) * wdt, mu, wdt, yerr=sd, capsize=2,
                      color=(SLATE if fe == "ses" else NAVY),
                      label=FE_LABEL[fe], edgecolor="none",
                      error_kw=dict(lw=0.8))
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(["FedAvg", "FedProx", "prototype"], rotation=12,
                              ha="right")
        ax[i].set_title(f"held out: {sp} rpm", pad=4)
        ax[i].set_ylim(0, 1.10)
        if i == 0:
            ax[i].set_ylabel("accuracy at the unseen speed")
            ax[i].legend(frameon=False, loc="upper left", handlelength=1.1,
                         borderaxespad=0.2, fontsize=7)
    fig.tight_layout(w_pad=1.0)
    _save(fig, path)


def fig_margin(path, grid, summary):
    """What the margin ratio is made of, and what it does and does not predict."""
    fig, ax = plt.subplots(1, 3, figsize=(W, H3))
    lf = [r for r in grid if r["front_end"] == "lfes"]

    comps = ["Delta", "r_bar", "delta_mean"]
    labels = [r"$\Delta$", r"$\bar{r}$", r"$\delta$"]
    for i, me in enumerate(["fedavg", "proto"]):
        vals = [np.mean([r[c] for r in lf if r["method"] == me]) for c in comps]
        errs = [np.std([r[c] for r in lf if r["method"] == me]) for c in comps]
        ax[0].bar(np.arange(3) + (i - 0.5) * 0.36, vals, 0.36, yerr=errs, capsize=2,
                  color=(SLATE if me == "fedavg" else NAVY), edgecolor="none",
                  label=ME_LABEL[me], error_kw=dict(lw=0.8))
    ax[0].set_xticks(np.arange(3)); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("distance on the unit sphere")
    ax[0].set_title("geometry of the embedding", pad=4)
    lo, hi = ax[0].get_ylim(); ax[0].set_ylim(lo, hi + 0.30 * (hi - lo))
    ax[0].legend(frameon=False, loc="upper right", handlelength=1.1,
                 borderaxespad=0.2, fontsize=6.6)

    for me, col, mk in (("fedavg", SLATE, "o"), ("prox", GOLD, "s"), ("proto", NAVY, "^")):
        sub = [r for r in lf if r["method"] == me]
        ax[1].scatter([r["M"] for r in sub], [r["acc_out_proto"] for r in sub],
                      s=14, color=col, marker=mk, label=ME_LABEL[me], alpha=0.85)
    ax[1].axvline(1.0, color=RED, ls="--", lw=1.0)
    ax[1].set_xlabel(r"margin ratio $M$")
    ax[1].set_ylabel("accuracy, unseen speed")
    rho = summary["margin_jnu"]["spearman_M_acc"]
    ax[1].set_title(f"JNU: $M<1$ throughout, $\\rho_s={rho:.2f}$", pad=4)
    ax[1].set_xlim(0, 1.35); ax[1].set_ylim(0, 1.05)
    ax[1].text(1.04, 0.46, "condition\nholds", fontsize=6.2, color=RED)
    ax[1].legend(frameon=False, loc="lower right", handlelength=1.1,
                 borderaxespad=0.2, fontsize=6.2)

    # rho = 1 gives M of order 25 and squashes everything else; it is the
    # trained condition and carries no information about transfer.
    for fe, col, mk, lab in (("lfes", NAVY, "o", "log axis"),
                             ("ses", SLATE, "s", "linear axis")):
        sp = summary[f"sim_{fe}"]
        M = [m for m, r in zip(sp["M"], sp["rhos"]) if r > 1.0]
        A = [a for a, r in zip(sp["acc"], sp["rhos"]) if r > 1.0]
        ax[2].plot(M, A, color=col, marker=mk, ms=3, ls="none", label=lab)
    ax[2].axvline(1.0, color=RED, ls="--", lw=1.0)
    ax[2].set_xlabel(r"margin ratio $M$")
    ax[2].set_ylabel("accuracy")
    ax[2].set_title(r"simulation, $\delta$ measured exactly", pad=4)
    ax[2].set_ylim(0, 1.05)
    ax[2].legend(frameon=False, loc="upper left", handlelength=1.1,
                 borderaxespad=0.2, fontsize=6.6)
    fig.tight_layout(w_pad=1.6)
    _save(fig, path)


def fig_sim(path, summary):
    """The controlled sweep: displacement, margin and accuracy against speed ratio."""
    fig, ax = plt.subplots(1, 3, figsize=(W, H3))
    for fe, col, mk, ls in (("lfes", NAVY, "o", "-"), ("ses", SLATE, "s", "--")):
        s = summary[f"sim_{fe}"]
        ax[0].plot(s["rhos"], s["delta"], color=col, marker=mk, ms=3, ls=ls,
                   label=FE_LABEL[fe] + ", paired")
        ax[0].plot(s["rhos"], s["delta_means"], color=col, marker=mk, ms=2.2, ls=":",
                   alpha=0.65, label=FE_LABEL[fe] + ", class means")
        ax[1].semilogy(s["rhos"], s["M"], color=col, marker=mk, ms=3, ls=ls,
                       label=FE_LABEL[fe])
        ax[2].plot(s["rhos"], s["acc"], color=col, marker=mk, ms=3, ls=ls,
                   label=FE_LABEL[fe])
    ax[0].set_ylabel(r"displacement $\delta$")
    ax[0].set_title("speed-induced displacement", pad=4)
    ax[0].legend(frameon=False, loc="lower center", handlelength=1.4, ncol=2,
                 borderaxespad=0.2, fontsize=5.4, labelspacing=0.25,
                 columnspacing=0.8)
    ax[1].axhline(1.0, color=RED, ls="--", lw=1.0)
    ax[1].set_ylabel(r"margin ratio $M$")
    ax[1].set_title("margin ratio, log scale", pad=4)
    ax[2].axhline(0.25, color=RED, ls=":", lw=0.9)
    ax[2].text(1.62, 0.32, "chance", fontsize=6.2, color=RED, ha="left")
    ax[2].set_ylabel("accuracy")
    ax[2].set_title("classification accuracy", pad=4)
    ax[2].set_ylim(0, 1.05)
    for a_ in ax[1:]:
        a_.legend(frameon=False, loc="upper right", handlelength=1.3,
                  borderaxespad=0.2, fontsize=6.6)
    for a_ in ax:
        a_.set_xlabel(r"speed ratio $\rho$")
    hi = ax[0].get_ylim()[1]
    ax[0].set_ylim(-0.55 * hi, hi * 1.06)
    ax[0].set_yticks([t for t in ax[0].get_yticks() if t >= 0])
    fig.tight_layout(w_pad=1.5)
    _save(fig, path)
