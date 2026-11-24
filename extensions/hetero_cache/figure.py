"""Figure for the heterogeneous-cluster study -> results/fig_hetero.png   (python figure.py)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"evict_aware": "#eb6834", "evict_aware_het": "#2a78d6", "oracle_het": "#1baf7a", "2xA100_only": "#52514e"}
NAME = {"evict_aware": "Eviction-aware router, assumes identical GPUs (paper)",
        "evict_aware_het": "Same router, per-GPU cost model + cache size",
        "oracle_het": "Clairvoyant reference (per-GPU)", "2xA100_only": "2x A100 only (no L4s)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

d = pd.read_csv("results/het_all.csv")
d["grp"] = d.group.str.replace("_fastonly", "")
x = d[d.grp == "load"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for p in ["evict_aware", "evict_aware_het", "oracle_het", "2xA100_only"]:
    g = x[x.policy == p].groupby("util")
    for k, (col, scale) in enumerate([("ttft_p95", 1e3), ("session_time_mean", 1.0)]):
        m, lo, hi = g[col].mean() * scale, g[col].min() * scale, g[col].max() * scale
        ax[k].errorbar(m.index, m.values, yerr=[m - lo, hi - m], color=C[p], lw=2, marker="o", ms=5,
                       mec="#fcfcfb", mew=1.2, capsize=0, elinewidth=1, label=NAME[p])
ax[0].set_yscale("log")
ax[0].set_ylabel("p95 time to first token (ms, log)")
ax[0].set_title("a  Tail TTFT vs load", loc="left", color=INK, fontsize=10)
ax[1].set_ylabel("mean session completion time (s)")
ax[1].set_ylim(4, 16)
ax[1].set_title("b  End-to-end: agentic session time vs load", loc="left", color=INK, fontsize=10)
ax[1].text(0.7, 15.4, "2x A100 only leaves the chart:\n59-70 s at 2.5-3.0x (unstable)", fontsize=8,
           color=INK2, va="top")
for a in ax:
    a.set_xlabel("offered load (x nominal capacity of 2x A100 + 2x L4)")
    a.set_xticks([0.7, 1.1, 1.6, 2.0, 2.5, 3.0])
    a.yaxis.grid(True, color=GRID, lw=0.6, which="both")
    a.set_axisbelow(True)
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig("results/fig_hetero.png", dpi=200)
print("ok")
