"""Cache-aware LLM routing on a heterogeneous GPU cluster (PrefixRoute extension).

Replicas differ in speed (per-replica cost model) and in KV-cache capacity. We ask whether the eviction-aware
router of the paper -- which assumes identical replicas -- still works, and whether giving it each replica's own
cost model and cache size ("heterogeneity-aware") is enough to recover.

Slow GPU (L4-class) = the measured A100 cost model scaled by public roofline ratios, NOT measured:
  compute-bound terms (b, dp)  x 312/121 TFLOP/s    = 2.58
  memory-bound term  (dd)      x 1555/300 GB/s      = 5.18
  fixed term a: weight read 5.5 GB at 300 GB/s + the A100's non-weight overhead
  KV memory: (0.9*24 - 5.5) / (0.9*40 - 5.5) GB     = 0.53 of the A100's cacheable capacity
"""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quantile_ttft"))
import shim  # noqa: E402,F401  (repo on sys.path + parquet fallback)

from prefixroute import run_sim as RS  # noqa: E402
from prefixroute.metrics.stats import cell_metrics  # noqa: E402
from prefixroute.policies.routers import (POLICIES, CacheAware, Clairvoyant,  # noqa: E402,F401
                                          EvictionAware, LeastLoaded)
from prefixroute.sim.costmodel import CostModel, blocks  # noqa: E402
from prefixroute.sim.engine import Cluster  # noqa: E402

COST = os.path.join(RS.ROOT, "results", "measure", "costmodel.json")
A100_WEIGHT_READ = 5.5 / 1555.0          # s, weights streamed once per iteration
L4 = {"fc": 312 / 121, "fm": 1555 / 300, "mem": (0.9 * 24 - 5.5) / (0.9 * 40 - 5.5), "bw": 300.0}


def slow_cost(base: CostModel, fc, fm, bw_gbs):
    overhead = base.a - A100_WEIGHT_READ
    return replace(base, a=overhead + 5.5 / bw_gbs, b=base.b * fc, dp=base.dp * fc, dd=base.dd * fm,
                   source=f"roofline-scaled(fc={fc:.2f},fm={fm:.2f})")


def uniform_slow(base: CostModel, s):
    return replace(base, a=base.a * s, b=base.b * s, c=base.c * s, dp=base.dp * s, dd=base.dd * s,
                   source=f"uniform x{s}")


class HetCluster(Cluster):
    """Cluster whose replicas have their own cost model and KV capacity. `cl.cost` stays the reference
    (fast) model, so a heterogeneity-blind router keeps assuming every replica is a fast one."""

    def __init__(self, costs, caps, policy, B=32):
        super().__init__(len(costs), caps[0], costs[0], policy, B=B)
        self.base_cost = costs[0]
        for rep, c, k in zip(self.reps, costs, caps):
            rep.cost, rep.cap = c, k

    def _start_iteration(self, rep):
        self.cost = rep.cost
        try:
            super()._start_iteration(rep)
        finally:
            self.cost = self.base_cost


class HetEvictionAware(EvictionAware):
    """The paper's eviction-aware router, given each replica's own cost model and cache capacity."""
    name = "evict_aware_het"

    def reset(self, cl):
        super().reset(cl)
        self.caps = [rep.cap for rep in cl.reps]

    def est_ttft_j(self, cl, j, *a):
        saved = cl.cost
        cl.cost = cl.reps[j].cost
        try:
            return self.est_ttft(cl, *a)
        finally:
            cl.cost = saved

    def route(self, r, cl):
        B = cl.B
        best, bc = 0, float("inf")
        for j in range(cl.n):
            m = min(self._match(j, r.ids) * B, r.prompt - 1)
            t = self.est_ttft_j(cl, j, self.pend[j], self.dec[j], self.ctx[j], r.prompt - m, m)
            if t < bc - 1e-12:
                best, bc = j, t
        return best

    def _touch(self, j, ids):
        s = self.shadow[j]
        for b in ids:
            if b in s:
                s.move_to_end(b)
            else:
                s[b] = None
        while len(s) > self.caps[j]:
            s.popitem(last=False)


class HetOracle(Clairvoyant):
    """Clairvoyant reference with per-replica cost models (true cache contents, decode state, headroom)."""
    name = "oracle_het"

    def route(self, r, cl):
        saved = cl.cost
        best, bc = 0, float("inf")
        try:
            for j in range(cl.n):
                cl.cost = cl.reps[j].cost
                t = self._score(r, cl, j)
                if t < bc - 1e-12:
                    best, bc = j, t
        finally:
            cl.cost = saved
        return best

    def _score(self, r, cl, j):
        import math
        rep, B = cl.reps[j], cl.B
        m = min(rep.match(r.ids) * B, r.prompt - 1)
        dec = [q for q in rep.running if q.decoding]
        ctx = sum(q.prompt + q.out - q.dec_left for q in dec)
        t = self.est_ttft(cl, self.pend[j], len(dec), ctx, r.prompt - m, m)
        need = math.ceil((r.prompt + r.out) / B) - m
        if dec and need > rep.cap - rep.used() + len(rep.lru):
            t += min(q.dec_left for q in dec) * cl.cost.decode_iter(len(dec), ctx / len(dec))
        return t


class SpeedWeightedLeastLoaded(LeastLoaded):
    """Least outstanding requests, normalised by replica speed (a common heterogeneity-aware baseline)."""
    name = "least_loaded_speed"

    def reset(self, cl):
        super().reset(cl)
        ref = cl.reps[0].cost.prefill_time(2048)
        self.speed = [ref / rep.cost.prefill_time(2048) for rep in cl.reps]

    def route(self, r, cl):
        return min(range(cl.n), key=lambda j: ((cl.outstanding[j] + 1) / self.speed[j], j))


for P in (HetEvictionAware, HetOracle, SpeedWeightedLeastLoaded):
    POLICIES[P.name] = P

BASE = {"N": 4, "cap_frac": 1.0, "util": 0.7, "B": 32, "headroom": 2.0, "slo_s": 1.0, "think_scale": 1.0,
        "growth_scale": 1.0, "turns_scale": 1.0, "workload": "agentic", "n_sessions": 5000}


def run(cell):
    """cell: BASE overrides + 'policy', 'seed', 'n_slow', and 'slow' = ('l4',) | ('uniform', s) | ('mem', ratio)."""
    c = dict(BASE)
    c.update(cell)
    base = CostModel.load(COST)
    kind = c["slow"][0]
    if kind == "l4":
        sc, mem = slow_cost(base, L4["fc"], L4["fm"], L4["bw"]), L4["mem"]
    elif kind == "uniform":
        sc, mem = uniform_slow(base, c["slow"][1]), 1.0
    elif kind == "mem":
        sc, mem = base, c["slow"][1]
    else:
        raise ValueError(kind)
    N, ns = c["N"], c["n_slow"]
    costs = [base] * (N - ns) + [sc] * ns
    # arrival rate: util relative to the cluster's real capacity (fast-replica equivalents)
    ss = RS.build_sessions(dict(c, N=N), base)
    d_fast = sum(base.nominal_demand(p, o) for s in ss for (p, o, _) in s.turns)
    d_slow = sum(sc.nominal_demand(p, o) for s in ss for (p, o, _) in s.turns)
    n_eff = (N - ns) + ns * d_fast / d_slow
    c2 = dict(c, N=n_eff)
    ss = RS.build_sessions(c2, base)
    B = c["B"]
    ws, fly = RS.working_set_blocks(ss, base, B)
    max_req = max(blocks(p + o, B) for s in ss for (p, o, _) in s.turns)
    nc = c.get("cache_N", N)           # replicas the memory budget is defined over (same physical GPU size)
    head = int(c["headroom"] * fly / nc) + max_req
    cache = c["cap_frac"] * ws / nc
    caps = [head + int(cache)] * (N - ns) + [head + int(mem * cache)] * ns
    pol = POLICIES[c["policy"]](seed=c["seed"])
    cl = HetCluster(costs, caps, pol, B=B)
    done = cl.run(ss)
    m = cell_metrics(done, cl, c["slo_s"])
    n_warm = [0] * N
    t0 = min(r.arrival for r in done) + 0.1 * (max(r.arrival for r in done) - min(r.arrival for r in done))
    for r in done:
        if r.arrival >= t0:
            n_warm[r.replica] += 1
    out = {k: c[k] for k in ("policy", "seed", "n_slow", "util", "workload")}
    out.update(slow="/".join(map(str, c["slow"])), tool_median_s=c.get("tool_median_s"), cap_frac=c["cap_frac"],
               n_eff=n_eff, slow_share=sum(n_warm[N - ns:]) / max(1, sum(n_warm)),
               **{k: m[k] for k in ("ttft_p50", "ttft_p95", "ttft_p99", "ttft_mean", "e2e_p95", "e2e_mean", "session_time_mean",
                                   "session_time_p95", "thpt_req_s", "hit_rate", "n_arrived", "n_done")})
    return out
