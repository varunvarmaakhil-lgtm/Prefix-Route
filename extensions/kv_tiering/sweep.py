"""All cells of the KV-tiering study -> results/tier_all.csv   (python sweep.py, ~10 min on 2 cores)

Regime: agentic trace, 4 replicas, 70% load, HBM cache = 0.1 x working set, 30 s median tool latency
(sessions idle long enough for their prefixes to be evicted from HBM).
  capacity : tier-2 size k x HBM cache share, k = 0..64, at 25 GB/s (host DRAM over PCIe 4.0 x16)
  bandwidth: k = 16, tier bandwidth 64 .. 0.1 GB/s (CXL ... remote/slow storage)
  tool     : k = 0 vs k = 16 at 25 GB/s, for 10 / 30 / 60 s tools
"""
import multiprocessing as mp
import os

import pandas as pd

import tier

SEEDS = (0, 1, 2)
R = dict(cap_frac=0.1, tool_median_s=30.0)


def cells():
    out = [dict(group="capacity", seed=s, k=k, gbps=25.0, **R) for k in (0, 1, 2, 4, 8, 16, 32, 64) for s in SEEDS]
    out += [dict(group="bandwidth", seed=s, k=16, gbps=g, **R)
            for g in (64.0, 25.0, 12.5, 7.0, 3.0, 1.5, 0.75, 0.4, 0.2, 0.1) for s in SEEDS]
    out += [dict(group="tool", seed=s, k=k, gbps=25.0, cap_frac=0.1, tool_median_s=t)
            for t in (10.0, 30.0, 60.0) for k in (0, 16) for s in SEEDS]
    return out


def one(c):
    g = c.pop("group")
    r = tier.run(c)
    r["group"] = g
    return r


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    with mp.get_context("fork").Pool(os.cpu_count() or 2) as pool:
        df = pd.DataFrame(pool.map(one, cells(), chunksize=1))
    df.to_csv("results/tier_all.csv", index=False)
    print(len(df), "cells")
