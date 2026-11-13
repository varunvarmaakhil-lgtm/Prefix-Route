"""Paired (same seed = same sessions and arrivals) comparisons of p95 TTFT between policies under KV pressure.
Writes results/tables/paired.csv and appends macros to paper/macros.tex (run after make_tables.py)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from make_tables import load_cells, T95  # noqa: E402


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "raw")
    df = load_cells(raw)
    df = df[df.tool_median_s.isna()]
    df = df[(df.N == 4) & (df.util == 0.7) & (df.think_scale == 100.0) & (df.growth_scale == 1.0)]
    rows, lines = [], []
    for (wl, hr, cf), g in df.groupby(["workload", "headroom", "cap_frac"]):
        piv = g.pivot_table(index="seed", columns="policy", values="ttft_p95")
        for a, b in [("evict_aware", "cache_aware"), ("evict_aware", "session_affinity"),
                     ("cache_aware", "session_affinity")]:
            if a in piv and b in piv:
                d = (piv[b] - piv[a]).dropna() * 1000        # ms saved by a relative to b
                n = len(d)
                ci = T95.get(n - 1, 1.96) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                rel = 100 * (piv[b] - piv[a]).dropna().sum() / piv[b].dropna().sum()
                rows.append({"workload": wl, "headroom": hr, "cap_frac": cf, "a": a, "b": b, "n": n,
                             "diff_ms": d.mean(), "ci_ms": ci, "rel_pct": rel, "wins": int((d > 0).sum()),
                             "significant": bool(d.mean() - ci > 0 or d.mean() + ci < 0)})
                key = (f"pair{wl}hr{int(hr)}k{str(cf).replace('.', 'p')}"
                       f"{a.replace('_', '')}vs{b.replace('_', '')}")
                lines.append(f"\\expandafter\\def\\csname pr@{key}diff\\endcsname{{{d.mean():.0f}}}")
                lines.append(f"\\expandafter\\def\\csname pr@{key}ci\\endcsname{{{ci:.0f}}}")
                lines.append(f"\\expandafter\\def\\csname pr@{key}rel\\endcsname{{{rel:.0f}}}")
                lines.append(f"\\expandafter\\def\\csname pr@{key}wins\\endcsname{{{int((d > 0).sum())}/{n}}}")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(ROOT, "results", "tables", "paired.csv"), index=False)
    with open(os.path.join(ROOT, "paper", "macros.tex"), "a") as fh:
        fh.write("% paired comparisons (analysis/paired.py)\n" + "\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(t[t.cap_frac.isin([0.25, 2.0])].round(1).to_string())


if __name__ == "__main__":
    main()
