"""Miss-cost decomposition (Task 3): does eviction-aware routing make misses cheaper, or keep them from delaying
other requests?

Per request (prefixroute/metrics/stats.py: miss_decomposition): recomputed = previously computed prefix tokens
that were not reused; class hit (< 1 block) / cheap miss (<= 1024 tokens) / expensive miss; delay = TTFT minus
the request's own isolated prefill time (time lost to other requests). Aggregated per policy over seeds, with
paired (same-seed) differences to eviction-aware.

    python analysis/mechanism.py [--raw results/raw_v2]
Writes results/tables/mechanism.csv, paper/tables_mechanism.tex, and appends macros to paper/macros.tex.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from make_tables import load_cells, T95, POL_LABEL  # noqa: E402

POLS = ["least_loaded", "session_affinity", "cache_aware", "evict_aware", "oracle"]
METRICS = ["ttft_p95", "hit_rate", "frac_hit", "frac_cheap", "frac_exp", "recomp_per_miss_mean",
           "recomp_per_miss_p95", "ttft_p95_hit", "ttft_p95_exp", "delay_mean", "delay_p95", "delay_share_tail",
           "tail_share_hit", "tail_share_cheap", "tail_share_exp", "delay_mean_hit", "delay_mean_exp"]


def pdiff(piv, a, b):
    d = (piv[a] - piv[b]).dropna()
    n = len(d)
    ci = T95.get(n - 1, 1.96) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return float(d.mean()), float(ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw_v2"))
    a = ap.parse_args()
    df = load_cells(a.raw)
    df = df[df.tool_median_s.notna() & (df.headroom == 2.0) & (df.N == 4) & (df.util == 0.7)]
    rows, lines = [], []

    def mac(k, v):
        lines.append(f"\\expandafter\\def\\csname pr@{k}\\endcsname{{{v}}}")

    for (wl, cf, lat), g in df.groupby(["workload", "cap_frac", "tool_median_s"]):
        g = g[g.policy.isin(POLS)]
        for p in POLS:
            x = g[g.policy == p]
            if x.empty:
                continue
            r = {"workload": wl, "cap_frac": cf, "tool_s": lat, "policy": p}
            r.update({m: float(x[m].mean()) for m in METRICS if m in x})
            rows.append(r)
        # paired differences eviction-aware minus cache-aware / session affinity
        for other in ("cache_aware", "session_affinity"):
            for m in ("frac_exp", "recomp_per_miss_mean", "hit_rate", "delay_mean", "delay_p95", "ttft_p95_hit"):
                piv = g.pivot_table(index="seed", columns="policy", values=m)
                if "evict_aware" in piv and other in piv:
                    d, ci = pdiff(piv, "evict_aware", other)
                    rows.append({"workload": wl, "cap_frac": cf, "tool_s": lat, "policy": f"diff_ea_{other}",
                                 m: d, m + "_ci": ci})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(ROOT, "results", "tables", "mechanism.csv"), index=False)
    # macros + a compact table for the headline regime (kappa=0.25, 5 s tools) per workload
    for (wl, cf, lat), g in t[~t.policy.str.startswith("diff")].groupby(["workload", "cap_frac", "tool_s"]):
        nm = f"mech{wl}k{str(cf).replace('.', 'p')}t{str(lat).replace('.', 'p')}"
        for _, r in g.iterrows():
            p = r.policy.replace("_", "")
            for m, fmt, scale in [("hit_rate", "{:.1f}", 100), ("frac_exp", "{:.1f}", 100),
                                  ("frac_cheap", "{:.1f}", 100), ("recomp_per_miss_mean", "{:.0f}", 1),
                                  ("recomp_per_miss_p95", "{:.0f}", 1), ("ttft_p95_hit", "{:.0f}", 1000),
                                  ("ttft_p95_exp", "{:.0f}", 1000), ("delay_mean", "{:.0f}", 1000),
                                  ("delay_p95", "{:.0f}", 1000), ("tail_share_exp", "{:.0f}", 100),
                                  ("delay_share_tail", "{:.0f}", 100), ("ttft_p95", "{:.0f}", 1000)]:
                if m in r and np.isfinite(r[m]):
                    mac(f"{nm}{p}{m.replace('_', '')}", fmt.format(scale * r[m]))
    for _, r in t[t.policy.str.startswith("diff")].iterrows():
        nm = f"mech{r.workload}k{str(r.cap_frac).replace('.', 'p')}t{str(r.tool_s).replace('.', 'p')}{r.policy.replace('_', '')}"
        for m, fmt, scale in [("frac_exp", "{:+.1f}", 100), ("recomp_per_miss_mean", "{:+.0f}", 1),
                              ("hit_rate", "{:+.1f}", 100), ("delay_mean", "{:+.0f}", 1000),
                              ("delay_p95", "{:+.0f}", 1000), ("ttft_p95_hit", "{:+.0f}", 1000)]:
            if m in r and pd.notna(r.get(m)):
                mac(f"{nm}{m.replace('_', '')}", fmt.format(scale * r[m]))
                mac(f"{nm}{m.replace('_', '')}ci", "{:.1f}".format(scale * r[m + "_ci"]) if scale == 100
                    else "{:.0f}".format(scale * r[m + "_ci"]))
    with open(os.path.join(ROOT, "paper", "macros.tex"), "a") as fh:
        fh.write("% miss-cost decomposition (analysis/mechanism.py)\n" + "\n".join(lines) + "\n")
    write_tex(t, os.path.join(ROOT, "paper", "tables_mechanism.tex"))
    pd.set_option("display.width", 250)
    sel = t[(t.cap_frac.isin([0.25, 2.0])) & (t.tool_s.isin([0.05, 5.0]))]
    cols = ["workload", "cap_frac", "tool_s", "policy", "ttft_p95", "hit_rate", "frac_exp", "recomp_per_miss_mean",
            "ttft_p95_hit", "delay_mean", "delay_p95", "tail_share_exp", "delay_share_tail"]
    print(sel[cols].round(3).to_string())


def write_tex(t, path, wl_list=("agentic", "synth"), cf=0.25, lat=5.0):
    cols = [("hit_rate", "hit\\%", 100, "{:.0f}"), ("frac_exp", "exp.\\,miss\\%", 100, "{:.1f}"),
            ("recomp_per_miss_mean", "recomp./miss", 1, "{:.0f}"), ("delay_mean", "delay (ms)", 1000, "{:.0f}"),
            ("ttft_p95_hit", "hit p95 (ms)", 1000, "{:.0f}"), ("tail_share_exp", "tail exp.\\%", 100, "{:.0f}"),
            ("ttft_p95", "p95 (ms)", 1000, "{:.0f}")]
    lines = ["\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule",
             "& Policy & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for wl in wl_list:
        g = t[(t.workload == wl) & (t.cap_frac == cf) & (t.tool_s == lat) & (~t.policy.str.startswith("diff"))]
        for i, p in enumerate(POLS):
            r = g[g.policy == p]
            if r.empty:
                continue
            r = r.iloc[0]
            lab = ("\\multirow{5}{*}{\\rotatebox{90}{" + ("Agentic" if wl == "agentic" else "Synthetic") + "}}") if i == 0 else ""
            lines.append(lab + " & " + POL_LABEL[p] + " & " +
                         " & ".join(f.format(s * r[m]) for m, _, s, f in cols) + " \\\\")
        lines.append("\\midrule" if wl != wl_list[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
