"""Tool-latency x KV-capacity sweep (Task 1): when does the policy ranking change?

For each workload, capacity factor kappa and median tool latency, paired (same-seed) comparisons of p95 TTFT:
eviction-aware vs cache-aware, vs session affinity, and cache-aware vs session affinity. The crossover latency is
the smallest swept latency from which eviction-aware beats the best eviction-oblivious locality policy (per seed,
min of cache-aware and session affinity) by >= MARGIN with a 95% paired CI excluding zero, at that latency and all
larger ones. Also reports Kendall tau of each latency's policy ranking against the 50 ms ranking.

    python analysis/latency.py [--raw results/raw_v2] [--tag ""]
Writes results/tables/latency*.csv|json, paper/figs/fig_latency.pdf, and appends macros to paper/macros.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from make_tables import load_cells, T95, _style, COLORS, MARKERS, POL, POL_LABEL  # noqa: E402

MARGIN = 5.0     # percent
LOCALITY = ["session_affinity", "cache_aware"]


def paired(piv, a, b):
    """Percent by which a's p95 is below b's (positive = a better), with 95% t-CI over seeds."""
    rel = 100 * (piv[b] - piv[a]) / piv[b]
    rel = rel.dropna()
    n = len(rel)
    ci = T95.get(n - 1, 1.96) * rel.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return float(rel.mean()), float(ci), int((rel > 0).sum()), n


def analyse(df):
    rows, ranks = [], []
    for (wl, cf, lat), g in df.groupby(["workload", "cap_frac", "tool_median_s"]):
        piv = g.pivot_table(index="seed", columns="policy", values="ttft_p95")
        if "evict_aware" not in piv:
            continue
        best_obl = piv[[p for p in LOCALITY if p in piv]].min(axis=1)
        piv = piv.assign(best_oblivious=best_obl)
        rec = {"workload": wl, "cap_frac": cf, "tool_s": lat}
        for a, b in [("evict_aware", "best_oblivious"), ("evict_aware", "cache_aware"),
                     ("evict_aware", "session_affinity"), ("cache_aware", "session_affinity"),
                     ("oracle", "evict_aware")]:
            if a in piv and b in piv:
                m, ci, w, n = paired(piv, a, b)
                rec[f"{a}_vs_{b}"] = m
                rec[f"{a}_vs_{b}_ci"] = ci
                rec[f"{a}_vs_{b}_wins"] = f"{w}/{n}"
        for p in piv.columns:
            if p in POL:
                rec[f"p95_{p}"] = float(piv[p].mean())
                rec[f"rel_{p}"] = float((piv[p] / piv["oracle"]).mean()) if "oracle" in piv else np.nan
        rows.append(rec)
        ranks.append({"workload": wl, "cap_frac": cf, "tool_s": lat,
                      "rank": list(piv[[p for p in POL if p in piv]].mean().sort_values().index)})
    t = pd.DataFrame(rows).sort_values(["workload", "cap_frac", "tool_s"])
    # Kendall tau vs the 50 ms ranking (same workload, kappa)
    taus = []
    for (wl, cf), g in t.groupby(["workload", "cap_frac"]):
        base = g[g.tool_s == g.tool_s.min()].iloc[0]
        pols = [p for p in POL if f"p95_{p}" in g and p != "prefix_hash"]
        for _, r in g.iterrows():
            tau = kendalltau([base[f"p95_{p}"] for p in pols], [r[f"p95_{p}"] for p in pols]).statistic
            taus.append({"workload": wl, "cap_frac": cf, "tool_s": r.tool_s, "tau_vs_50ms": float(tau)})
    t = t.merge(pd.DataFrame(taus), on=["workload", "cap_frac", "tool_s"])
    # crossover
    cross = {}
    for (wl, cf), g in t.groupby(["workload", "cap_frac"]):
        g = g.sort_values("tool_s")
        ok = [(r["evict_aware_vs_best_oblivious"] - r["evict_aware_vs_best_oblivious_ci"] > 0 and
               r["evict_aware_vs_best_oblivious"] >= MARGIN) for _, r in g.iterrows()]
        lat = list(g.tool_s)
        x = None
        for i in range(len(ok)):
            if all(ok[i:]):
                x = lat[i]
                break
        cross[f"{wl}|{cf}"] = x
    return t, ranks, cross


def fig(t, cross, path):
    plt = _style()
    wls = [w for w in ["agentic", "synth"] if w in set(t.workload)]
    kappas = [k for k in [0.25, 2.0] if k in set(t.cap_frac)]
    fig, axs = plt.subplots(1, len(wls) * len(kappas), figsize=(7.16, 1.85), sharey=True)
    axs = np.atleast_1d(axs)
    show = ["session_affinity", "cache_aware", "evict_aware"]
    k = 0
    for wl in wls:
        for cf in kappas:
            ax = axs[k]
            k += 1
            g = t[(t.workload == wl) & (t.cap_frac == cf)].sort_values("tool_s")
            for p in show:
                ax.plot(g.tool_s, g[f"rel_{p}"], marker=MARKERS[POL.index(p)], ms=3, color=COLORS[POL.index(p)],
                        label=POL_LABEL[p])
            ax.axhline(1.0, color="#8a897f", lw=0.6, ls=":")
            x = cross.get(f"{wl}|{cf}")
            if x is not None:
                ax.axvline(x, color="#4a3aa7", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_xticks([0.05, 0.2, 0.5, 1, 2, 5])
            ax.set_xticklabels([".05", ".2", ".5", "1", "2", "5"])
            ax.set_xlabel("median tool latency (s)")
            ax.set_title(f"{'Agentic' if wl == 'agentic' else 'Synthetic'}, $\\kappa$={cf:g}", fontsize=7)
    axs[0].set_ylabel("p95 TTFT / clairvoyant")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw_v2"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--headroom", type=float, default=2.0)
    a = ap.parse_args()
    df = load_cells(a.raw)
    if "tool_median_s" not in df:
        print("no tool-latency cells")
        return
    df = df[df.tool_median_s.notna() & (df.headroom == a.headroom) & (df.N == 4) & (df.util == 0.7)]
    t, ranks, cross = analyse(df)
    tag = a.tag or ("" if a.headroom == 2.0 else "hr1")
    tables = os.path.join(ROOT, "results", "tables")
    t.to_csv(os.path.join(tables, f"latency{tag}.csv"), index=False)
    json.dump({"crossover_s": cross, "margin_pct": MARGIN, "ranks": ranks},
              open(os.path.join(tables, f"latency{tag}.json"), "w"), indent=1, default=str)
    lines = []

    def mac(k, v):
        lines.append(f"\\expandafter\\def\\csname pr@{k}\\endcsname{{{v}}}")

    if "rel_least_loaded" in t:
        mac(f"lat{tag}llmin", f"{t.rel_least_loaded.min():.1f}")
        mac(f"lat{tag}llmax", f"{t.rel_least_loaded.max():.1f}")
    for key, x in cross.items():
        wl, cf = key.split("|")
        nm = f"lat{tag}{wl}k{cf.replace('.', 'p')}"
        mac(nm + "cross", "none" if x is None else (f"{x * 1000:.0f}~ms" if x < 1 else f"{x:g}~s"))
    for _, r in t.iterrows():
        nm = f"lat{tag}{r.workload}k{str(r.cap_frac).replace('.', 'p')}t{str(r.tool_s).replace('.', 'p')}"
        for col, short in [("evict_aware_vs_best_oblivious", "eb"), ("evict_aware_vs_cache_aware", "ec"),
                           ("evict_aware_vs_session_affinity", "es"), ("cache_aware_vs_session_affinity", "cs")]:
            if col in r and np.isfinite(r[col]):
                mac(nm + short, f"{r[col]:.0f}")
                mac(nm + short + "ci", f"{r[col + '_ci']:.0f}")
        mac(nm + "tau", f"{r.tau_vs_50ms:.2f}")
        for p in ["session_affinity", "cache_aware", "evict_aware", "least_loaded", "oracle"]:
            if f"p95_{p}" in r and np.isfinite(r[f"p95_{p}"]):
                mac(nm + p.replace("_", ""), f"{1000 * r[f'p95_{p}']:.0f}")
    with open(os.path.join(ROOT, "paper", "macros.tex"), "a") as fh:
        fh.write(f"% tool-latency sweep (analysis/latency.py {tag})\n" + "\n".join(lines) + "\n")
    if not tag:
        fig(t, cross, os.path.join(ROOT, "paper", "figs", "fig_latency.pdf"))
    pd.set_option("display.width", 250)
    cols = ["workload", "cap_frac", "tool_s", "evict_aware_vs_best_oblivious", "evict_aware_vs_best_oblivious_ci",
            "evict_aware_vs_cache_aware", "evict_aware_vs_session_affinity", "cache_aware_vs_session_affinity",
            "tau_vs_50ms"]
    print(t[[c for c in cols if c in t]].round(1).to_string())
    print("crossover:", cross)


if __name__ == "__main__":
    main()
