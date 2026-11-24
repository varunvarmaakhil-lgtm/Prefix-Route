"""Figure for the KV-tiering study -> results/fig_tiering.png   (python figure.py)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tier import KV_BYTES_PER_TOKEN  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
BREAK_EVEN_GBPS = KV_BYTES_PER_TOKEN / 139.4e-6 / 1e9      # A100 recompute cost per token (see README)

d = pd.read_csv("results/tier_all.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.1, 1.25, 0.9]})

# a: hit rate vs total cache capacity (multiples of the Little's-law reusable working set)
c = d[d.group == "capacity"].copy()
c["cap_ws"] = c.cap_frac * (1 + c.k)
g = c.groupby("cap_ws")
for col, color, lab in (("hit_rate", BLUE, "total prefix hit rate"), ("t2_hit_rate", ORANGE, "served from tier 2")):
    m = g[col].mean()
    ax[0].plot(m.index, m.values, color=color, lw=2, marker="o", ms=5, mec="#fcfcfb", mew=1.2, label=lab)
ax[0].axvline(1.0, color=GREY, lw=1, ls="--")
ax[0].text(1.08, 0.42, "1x Little's-law\nworking set", fontsize=8, color=INK2)
ax[0].set_xscale("log")
ax[0].set_xticks([0.1, 0.3, 1, 3, 6.5], ["0.1x\n(HBM only)", "0.3x", "1x", "3x", "6.5x"])
ax[0].set_ylim(0, 1)
ax[0].set_xlabel("HBM + tier-2 capacity (x reusable working set)")
ax[0].set_ylabel("share of prompt tokens")
ax[0].set_title("a  Hit rate needs 3-6x the working set", loc="left", color=INK, fontsize=10)
ax[0].legend(frameon=False, fontsize=8, loc="upper left")

# b: p95 TTFT vs tier bandwidth, k = 16
b = d[d.group == "bandwidth"].groupby("gbps")["ttft_p95"]
m, lo, hi = b.mean() * 1e3, b.min() * 1e3, b.max() * 1e3
ax[1].errorbar(m.index, m.values, yerr=[m - lo, hi - m], color=BLUE, lw=2, marker="o", ms=5, mec="#fcfcfb",
               mew=1.2, elinewidth=1, capsize=0, label="HBM + 16x tier 2")
base = d[(d.group == "capacity") & (d.k == 0)].ttft_p95.mean() * 1e3
ax[1].axhline(base, color=ORANGE, lw=1.5, label="HBM only (recompute misses)")
ax[1].axvline(BREAK_EVEN_GBPS, color=GREY, lw=1, ls="--")
ax[1].text(BREAK_EVEN_GBPS * 1.1, 700, f"analytic break-even\n{BREAK_EVEN_GBPS:.2f} GB/s", fontsize=8, color=INK2)
for x, lab in ((64, "CXL"), (25, "DRAM\n(PCIe 4)"), (7, "NVMe"), (1.5, "")):
    if lab:
        ax[1].text(x, 85, lab, ha="center", fontsize=7.5, color=INK2)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_ylim(60, 1500)
ax[1].set_xlabel("tier-2 bandwidth per replica (GB/s, log)")
ax[1].set_ylabel("p95 time to first token (ms, log)")
ax[1].set_title("b  Bandwidth matters only below ~1 GB/s", loc="left", color=INK, fontsize=10)
ax[1].legend(frameon=False, fontsize=8, loc="upper right")

# c: tool latency
t = d[d.group == "tool"].groupby(["tool_median_s", "k"])["ttft_p95"].mean().unstack() * 1e3
x = np.arange(len(t))
ax[2].bar(x - 0.2, t[0].values, 0.38, color=ORANGE, label="HBM only")
ax[2].bar(x + 0.2, t[16].values, 0.38, color=BLUE, label="+ tier 2")
for i, (a0, a1) in enumerate(zip(t[0].values, t[16].values)):
    ax[2].text(i + 0.2, a1 + 4, f"-{100 * (1 - a1 / a0):.0f}%", ha="center", fontsize=8, color=INK2)
ax[2].set_xticks(x, [f"{v:.0f} s" for v in t.index])
ax[2].set_xlabel("median tool-call latency (tier 2: 16x, 25 GB/s)")
ax[2].set_ylabel("p95 time to first token (ms)")
ax[2].set_title("c  Longer tool waits need more tier", loc="left", color=INK, fontsize=10)
ax[2].set_ylim(0, 290)
ax[2].legend(frameon=False, fontsize=8, loc="upper left")
for a in ax:
    a.yaxis.grid(True, color=GRID, lw=0.6, which="major")
    a.set_axisbelow(True)
fig.tight_layout()
fig.savefig("results/fig_tiering.png", dpi=200)
print("ok", round(BREAK_EVEN_GBPS, 3))
