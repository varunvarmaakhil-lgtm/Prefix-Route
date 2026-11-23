"""Run PrefixRoute cells under the eviction-aware router and log, for every request, the features the router
can observe at routing time (for the replica it chose) plus the realised TTFT.

    python collect.py            # writes logs/<regime>__seed<k>.csv
"""
import shim  # noqa: F401  (repo path + parquet fallback; import before prefixroute)

import os
import sys
from collections import deque

import numpy as np
import pandas as pd

from prefixroute import run_sim as RS
from prefixroute.policies.routers import EvictionAware, POLICIES
from prefixroute.sim.costmodel import CostModel, blocks
from prefixroute.sim.engine import Cluster

COST = os.path.join(RS.ROOT, "results", "measure", "costmodel.json")
BASE = {"N": 4, "cap_frac": 1.0, "util": 0.7, "B": 32, "headroom": 2.0, "slo_s": 1.0,
        "think_scale": 1.0, "growth_scale": 1.0, "turns_scale": 1.0, "workload": "agentic",
        "n_sessions": 5000, "policy": "evict_aware"}

# regime name -> overrides of BASE.  "base" is the training regime.
REGIMES = {
    "base":       {},
    "util0.9":    {"util": 0.9},
    "slowtool":   {"tool_median_s": 5.0, "cap_frac": 0.25},
    "synth":      {"workload": "synth"},
    "N8":         {"N": 8},
    "chat":       {"workload": "chat", "n_sessions": 12000},
}


class LoggingEvictAware(EvictionAware):
    """EvictionAware with identical routing decisions, plus a per-request feature log."""
    name = "evict_aware_log"

    def reset(self, cl):
        super().reset(cl)
        self.tick = [0] * cl.n
        self.stamp = [dict() for _ in range(cl.n)]
        self.last_done = {}
        self.recent = deque()
        self.log = {}

    def _touch(self, j, ids):
        super()._touch(j, ids)
        st = self.stamp[j]
        for b in ids:
            self.tick[j] += 1
            st[b] = self.tick[j]
        if len(st) > 4 * self.cap:  # drop stamps of blocks no longer in the shadow cache
            sh = self.shadow[j]
            self.stamp[j] = {b: t for b, t in st.items() if b in sh}

    def route(self, r, cl):
        j = super().route(r, cl)
        B = cl.B
        mb = self._match(j, r.ids)
        m = min(mb * B, r.prompt - 1)
        est = self.est_ttft(cl, self.pend[j], self.dec[j], self.ctx[j], r.prompt - m, m)
        if mb > 0:
            age = (self.tick[j] - self.stamp[j].get(r.ids[mb - 1], self.tick[j])) / self.cap
        else:
            age = -1.0
        now = cl.now
        self.recent.append(now)
        while self.recent and self.recent[0] < now - 5.0:
            self.recent.popleft()
        prev = self.last_done.get(r.sess.sid)
        self.log[id(r)] = {
            "arrival": r.arrival, "turn": r.turn, "prompt": r.prompt, "match": m,
            "new": r.prompt - m, "match_frac": m / r.prompt,
            "pend": self.pend[j], "dec": self.dec[j], "ctx": self.ctx[j], "outst": cl.outstanding[j],
            "est": est, "age_ratio": age, "shadow_fill": len(self.shadow[j]) / self.cap,
            "gap": (now - prev) if prev is not None else -1.0,
            "pend_all": sum(self.pend), "dec_all": sum(self.dec), "outst_all": sum(cl.outstanding),
            "rate5": len(self.recent) / 5.0, "n_rep": cl.n,
        }
        return j

    def on_first_token(self, r, i, cl):
        super().on_first_token(r, i, cl)
        rec = self.log.get(id(r))
        if rec is not None and "ttft" not in rec:
            rec["t_first"] = cl.now
            rec["ttft"] = cl.now - r.arrival
            rec["cached_true"] = r.cached

    def on_complete(self, r, i, cl):
        super().on_complete(r, i, cl)
        self.last_done[r.sess.sid] = cl.now


POLICIES[LoggingEvictAware.name] = LoggingEvictAware


def run(regime, seed, out_dir="logs"):
    cell = dict(BASE)
    cell.update(REGIMES[regime])
    cell["seed"] = seed
    cost = CostModel.load(COST)
    B = cell["B"]
    ss = RS.build_sessions(cell, cost)
    ws, fly = RS.working_set_blocks(ss, cost, B)
    max_req = max(blocks(p + o, B) for s in ss for (p, o, _) in s.turns)
    head = int(cell["headroom"] * fly / cell["N"]) + max_req
    cap = head + int(cell["cap_frac"] * ws / cell["N"])
    pol = LoggingEvictAware(seed=seed)
    cl = Cluster(cell["N"], cap, cost, pol, B=B)
    done = cl.run(ss)
    df = pd.DataFrame(list(pol.log.values())).sort_values("arrival").reset_index(drop=True)
    assert len(df) == len(done) and df.ttft.notna().all()
    t0 = df.arrival.min() + 0.1 * (df.arrival.max() - df.arrival.min())     # same warm-up as the paper
    df["warm"] = df.arrival >= t0
    df["regime"], df["seed"] = regime, seed
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{regime}__seed{seed}.csv"), index=False)
    w = df[df.warm]
    return regime, seed, len(df), float(np.quantile(w.ttft, 0.95))


if __name__ == "__main__":
    import multiprocessing as mp
    jobs = [("base", s) for s in range(5)] + [(g, s) for g in REGIMES if g != "base" for s in (3, 4)]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for res in pool.starmap(run, jobs):
            print(res, flush=True)
