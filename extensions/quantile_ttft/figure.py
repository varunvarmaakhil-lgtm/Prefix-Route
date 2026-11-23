import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate as E

C = {"A": "#2a78d6", "CQR": "#eb6834", "CQR+online": "#1baf7a", "Hybrid": "#6250d6"}
NAME = {"A": "Analytic estimator + conformal", "CQR": "Conformalised quantile regression",
        "CQR+online": "CQR + online recalibration",
        "Hybrid": "Hybrid: QR on residual of analytic estimate"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

res = pd.read_csv("results_intervals.csv")
agg = res.groupby(["regime", "method"]).mean(numeric_only=True)
regs = E.REGIMES
labels = ["in-dist.", "90% load", "5 s tools,\n0.25x cache", "synthetic", "8 replicas", "chat"]
meths = ["A", "CQR", "CQR+online"]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
x = np.arange(len(regs))
w = 0.26
for k, m in enumerate(meths):
    cov = [agg.loc[(r, m), "coverage"] for r in regs]
    ax[0].plot(x + (k - 1) * w, cov, "o", ms=8, color=C[m], mec="#fcfcfb", mew=1.5, label=NAME[m], zorder=3)
    ax[0].vlines(x + (k - 1) * w, 0.8, cov, color=C[m], lw=2, zorder=2)
ax[0].axhline(0.8, color=INK, lw=1, ls="--")
ax[0].text(-0.45, 0.865, "dashed = target 80%", ha="left", va="bottom", color=INK, fontsize=8)
ax[0].set_ylim(0.5, 0.9)
ax[0].set_ylabel("coverage of 80% interval")
ax[0].set_title("a  Coverage under shift", loc="left", color=INK, fontsize=10)
for k, m in enumerate(["A", "CQR", "Hybrid"]):
    err = [100 * agg.loc[(r, m), "mape_median"] for r in regs]
    ax[1].bar(x + (k - 1) * w, err, w * 0.92, color=C[m], label=NAME[m] if m == "Hybrid" else None)
    if m in ("A", "Hybrid"):
        for xi, e in zip(x + (k - 1) * w, err):
            if xi > 4.5:
                ax[1].text(xi, e + 0.3, f"{e:.1f}", ha="center", fontsize=7.5, color=INK2)
ax[1].set_ylabel("median |error| of point prediction (%)")
ax[1].set_title("b  Point accuracy (median prediction)", loc="left", color=INK, fontsize=10)
for a in ax[:2]:
    a.set_xticks(x, labels, fontsize=8)
    a.axvspan(4.5, 5.5, color="#f3f2ef", zorder=-1)
    a.yaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)

# c: rolling coverage on chat (seed 3)
tr = pd.concat([E.load("base", 0), E.load("base", 1)])
ms = E.fit_qr(tr[tr.warm])
cal = E.load("base", 2)
cal = cal[cal.warm]
clo, _, chi = E.qr_predict(ms, cal)
yc = np.log(cal.ttft.values)
q = E.conformal_offset(np.maximum(clo - yc, yc - chi))
lr = np.log(cal.ttft / cal.est)
a_lo, a_hi = np.quantile(lr, 0.1), np.quantile(lr, 0.9)
df = E.load("chat", 3)
y = df.ttft.values
lo, md, hi = E.qr_predict(ms, df)
le = np.log(df.est.values)
ivs = {"A": (le + a_lo, le + a_hi), "CQR": (lo - q, hi + q), "CQR+online": E.online(df, lo, hi, q)}
for m in meths:
    l, h = ivs[m]
    cov = ((y >= np.exp(l)) & (y <= np.exp(h))).astype(float)
    roll = pd.Series(cov).rolling(1000, min_periods=1000).mean()
    ax[2].plot(np.arange(len(roll)) / 1000, roll, color=C[m], lw=2)
ax[2].axhline(0.8, color=INK, lw=1, ls="--")
ax[2].set_ylim(0.4, 1.0)
ax[2].set_xlabel("requests served (thousands)")
ax[2].set_ylabel("rolling coverage (1,000 requests)")
ax[2].set_title("c  Chat traffic: online fix restores coverage", loc="left", color=INK, fontsize=10)
ax[2].yaxis.grid(True, color=GRID, lw=0.6)
h0, l0 = ax[0].get_legend_handles_labels()
h1, l1 = ax[1].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=4, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("fig_intervals.png", dpi=200)
print("ok")
