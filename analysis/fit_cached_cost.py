"""Fit the per-cached-token cost term e (sensitivity model) and write results/measure/costmodel_cachedcost.json.

The base simulator treats cached prompt tokens as free. Replaying the isolated (c=1) near-full-hit benchmark settings
(h >= 0.9) shows the engine charges a small cost per cached token; e is the least-squares slope through the origin of
(measured - simulated TTFT) against cached tokens. Requires results/tables/sim_bias.csv (analysis/sim_bias.py, base
model run first).
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    d = pd.read_csv(os.path.join(ROOT, "results", "tables", "sim_bias.csv"))
    x = d[(d.tag == "on") & (d.c == 1) & (d.h >= 0.9)].copy()
    x["cached"] = (x.h * x.L // 16 * 16).clip(upper=x.L - 1)
    gap = (x.meas_ms - x.sim_ms) / 1000
    e = float((gap * x.cached).sum() / (x.cached ** 2).sum())
    src = os.path.join(ROOT, "results", "measure", "costmodel.json")
    cm = json.load(open(src))
    cm["e"] = e
    cm["source"] = "A100-measured+cached-token-cost"
    cm["e_fit"] = {"n": int(len(x)), "points": x[["L", "h", "meas_ms", "sim_ms"]].round(2).to_dict("records")}
    json.dump(cm, open(os.path.join(ROOT, "results", "measure", "costmodel_cachedcost.json"), "w"), indent=1)
    print(f"e = {e:.4g} s per cached token ({len(x)} points)")


if __name__ == "__main__":
    main()
