"""Every cell of the heterogeneous-cluster study -> results/het_all.csv   (python sweep.py, ~25 min on 2 cores)

Mixed cluster: 2x A100 (measured cost model) + 2x L4-class (roofline-scaled, 0.53x KV memory).
  load      : cache_frac 1.0, default tool latency, nominal load 0.7 .. 3.0
  pressure  : cache_frac 0.25, 5 s tools (eviction-heavy), nominal load 0.5 .. 1.1
  speed     : slow replicas uniformly 1.5x / 2x / 3x slower, same memory, load 0.9
  memory    : slow replicas same speed, 0.5x memory, load 0.9
  *_fastonly: the 2 A100s alone (same per-GPU memory) at the same arrival rate
"""
import multiprocessing as mp
import os

import pandas as pd

import het

ALL = ["least_loaded", "least_loaded_speed", "cache_aware", "evict_aware", "evict_aware_het", "oracle_het"]
P3 = ["evict_aware", "evict_aware_het", "oracle_het"]
SEEDS = (0, 1, 2)
PRESS = dict(tool_median_s=5.0, cap_frac=0.25)


def cells():
    out = []
    for u in (0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0):
        pols = ALL if u <= 1.3 else P3
        out += [dict(group="load", policy=p, seed=s, n_slow=2, slow=("l4",), util=u) for p in pols for s in SEEDS]
    for u in (0.5, 0.7, 0.9, 1.1):
        pols = ALL if u == 0.9 else P3
        out += [dict(group="pressure", policy=p, seed=s, n_slow=2, slow=("l4",), util=u, **PRESS)
                for p in pols for s in SEEDS]
    for x in (1.5, 2.0, 3.0):
        out += [dict(group="speed", policy=p, seed=s, n_slow=2, slow=("uniform", x), util=0.9) for p in P3 for s in SEEDS]
    out += [dict(group="memory", policy=p, seed=s, n_slow=2, slow=("mem", 0.5), util=0.9) for p in P3 for s in SEEDS]
    return out


def one(c):
    g = c.pop("group")
    r = het.run(c)
    r["group"] = g
    return r


def fast_only(args):
    row, extra, g = args
    c = dict(policy="evict_aware", seed=int(row["seed"]), n_slow=0, slow=("l4",), N=2, cache_N=4,
             util=row["util"] * row["n_eff"] / 2, **extra)
    r = het.run(c)
    r.update(group=g, util=row["util"], policy="2xA100_only")
    return r


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    with mp.get_context("fork").Pool(os.cpu_count() or 2) as pool:
        df = pd.DataFrame(pool.map(one, cells(), chunksize=1))
        ref = df[df.policy == "evict_aware"]
        jobs = [(r, {}, "load_fastonly") for _, r in ref[ref.group == "load"].iterrows()]
        jobs += [(r, PRESS, "pressure_fastonly") for _, r in ref[ref.group == "pressure"].iterrows()]
        df2 = pd.DataFrame(pool.map(fast_only, jobs, chunksize=1))
    out = pd.concat([df, df2], ignore_index=True)
    out.to_csv("results/het_all.csv", index=False)
    print(len(out), "cells")
