"""Simulator bias by regime: replay every measured C1 benchmark setting through the simulator and compare TTFT.

Each setting (server config, concurrency c, prompt length L, hit fraction h) is reproduced exactly: one replica
with the engine's reported KV capacity, a warm-up request carrying the shared prefix of h*L tokens, then c
closed-loop clients issuing the same number of requests (unique suffixes, 64 output tokens). Residual =
simulated - measured median TTFT. The miss penalty is TTFT(h=0) / TTFT(h=0.9) at fixed (c, L); if the simulator
understates it, simulated advantages of miss-aware routing are conservative.
Writes results/tables/sim_bias.csv, results/tables/sim_bias_summary.json, paper/figs/fig_sim_bias.pdf.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prefixroute.measure.bench import n_requests, OUT_TOKENS  # noqa: E402
from prefixroute.policies.routers import POLICIES  # noqa: E402
from prefixroute.sim.costmodel import CostModel  # noqa: E402
from prefixroute.sim.engine import Cluster  # noqa: E402
from prefixroute.workload.sessions import Session  # noqa: E402

B = 16


def replay(cost, cap_tokens, c, L, h, caching=True):
    n_pre = int(round(h * L))
    ss = []
    t0 = 0.0
    if n_pre > 0 and caching:
        ss.append(Session(10_000, 0, n_pre, 0.0, [(n_pre + 8, 1, 0.0)], fresh_per_turn=True))
        t0 = cost.prefill_time(n_pre + 8) + 0.05
    per = max(1, n_requests(c) // c)
    for w in range(c):
        ss.append(Session(w, 0, n_pre, t0, [(L, OUT_TOKENS, 0.0)] * per, fresh_per_turn=True))
    cl = Cluster(1, int(cap_tokens // B), cost, POLICIES["round_robin"](), B=B, caching=caching)
    done = [r for r in cl.run(ss) if r.sess.sid != 10_000]
    return float(np.median([r.t_first - r.arrival for r in done]))


def main(cost_name="costmodel.json", suffix=""):
    mdir = os.path.join(ROOT, "results", "measure")
    cm = json.load(open(os.path.join(mdir, cost_name)))
    cost = CostModel.load(os.path.join(mdir, cost_name))
    caps = cm["kv_capacity_tokens"]
    s = pd.read_parquet(os.path.join(mdir, "curves.parquet"))
    rows = []
    for _, r in s.iterrows():
        cap = caps.get(f"server_{r.tag}.log", caps.get("server_on.log"))
        sim = replay(cost, cap, int(r.c), int(r.L), float(r.h), caching=(r.tag != "nocache"))
        rows.append({"tag": r.tag, "c": int(r.c), "L": int(r.L), "h": float(r.h), "meas_ms": 1000 * r.ttft_p50,
                     "sim_ms": 1000 * sim})
    d = pd.DataFrame(rows)
    d["resid_ms"] = d.sim_ms - d.meas_ms
    d["resid_pct"] = 100 * d.resid_ms / d.meas_ms
    d["log_ratio"] = np.log(d.sim_ms / d.meas_ms)
    tables = os.path.join(ROOT, "results", "tables")
    d.to_csv(os.path.join(tables, f"sim_bias{suffix}.csv"), index=False)
    on = d[d.tag == "on"]
    # median % residual by hit fraction and by concurrency (caching on)
    by_h = on.groupby("h").resid_pct.median()
    by_c = on.groupby("c").resid_pct.median()
    grid = on.pivot_table(index="c", columns="h", values="resid_pct", aggfunc="median")
    # miss penalty TTFT(h=0)/TTFT(h=0.9), measured vs simulated
    pen = []
    for (c, L), g in on.groupby(["c", "L"]):
        g0, g9 = g[g.h == 0.0], g[g.h == 0.9]
        if len(g0) and len(g9):
            pen.append({"c": c, "L": L, "meas": g0.meas_ms.iloc[0] / g9.meas_ms.iloc[0],
                        "sim": g0.sim_ms.iloc[0] / g9.sim_ms.iloc[0]})
    pen = pd.DataFrame(pen)
    pen["ratio"] = pen.sim / pen.meas
    pen.to_csv(os.path.join(tables, f"sim_bias_penalty{suffix}.csv"), index=False)
    from scipy.stats import spearmanr
    rho_all = spearmanr(on.meas_ms, on.sim_ms).statistic
    # within each (c, L): does the simulator order the hit fractions like the engine?
    rhos = [spearmanr(g.meas_ms, g.sim_ms).statistic for _, g in on.groupby(["c", "L"]) if len(g) > 2]
    summ = {"n_settings": int(len(d)), "resid_pct_median_on": float(on.resid_pct.median()),
            "resid_pct_by_h": {str(k): float(v) for k, v in by_h.items()},
            "resid_pct_by_c": {str(k): float(v) for k, v in by_c.items()},
            "resid_pct_c1": float(on[on.c == 1].resid_pct.median()),
            "resid_pct_c1_h0": float(on[(on.c == 1) & (on.h == 0)].resid_pct.median()),
            "resid_pct_c1_h90": float(on[(on.c == 1) & (on.h == 0.9)].resid_pct.median()),
            "resid_pct_hi_c_h0": float(on[(on.c >= 16) & (on.h == 0)].resid_pct.median()),
            "resid_pct_hi_c_h90": float(on[(on.c >= 16) & (on.h == 0.9)].resid_pct.median()),
            "penalty_ratio_median": float(pen.ratio.median()),
            "penalty_ratio_c1": float(pen[pen.c == 1].ratio.median()),
            "penalty_ratio_hi_c": float(pen[pen.c >= 16].ratio.median()),
            "spearman_all": float(rho_all), "spearman_within_median": float(np.median(rhos)),
            "grid_resid_pct": {str(c): {str(h): float(v) for h, v in row.items()} for c, row in grid.iterrows()}}
    json.dump(summ, open(os.path.join(tables, f"sim_bias_summary{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "grid_resid_pct"}, indent=1))
    print(grid.round(0))
    print(pen.round(2))
    if not suffix:
        fig(on, os.path.join(ROOT, "paper", "figs", "fig_sim_bias.pdf"))


def fig(on, path):
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from make_tables import _style, COLORS, MARKERS
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    for i, c in enumerate(sorted(on.c.unique())):
        g = on[on.c == c].groupby("h").resid_pct.median()
        ax.plot(g.index, g.values, marker=MARKERS[i], ms=3, color=COLORS[i], label=f"c={c}")
    ax.axhline(0, color="#8a897f", lw=0.6)
    ax.set_xlabel("prefix hit fraction")
    ax.set_ylabel("sim - measured TTFT (%)")
    ax.legend(frameon=False, ncol=4, fontsize=5.5, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
    main("costmodel_cachedcost.json", "_cachedcost")
