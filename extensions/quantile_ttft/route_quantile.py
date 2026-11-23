"""Does routing on a predicted TTFT quantile beat the eviction-aware router? Paired comparison on held-out seeds.

QuantRoute(q): for each candidate replica, build the same router-observable features as collect.py and send the
request to the replica with the smallest predicted q-quantile of TTFT (QR model trained on the base regime).
"""
import shim  # noqa: F401

import multiprocessing as mp

import numpy as np
import pandas as pd

import collect as C
import evaluate as E
from prefixroute import run_sim as RS
from prefixroute.sim.costmodel import CostModel, blocks
from prefixroute.sim.engine import Cluster

MODELS = None


class QuantRoute(C.LoggingEvictAware):
    name = "quant_route"

    def __init__(self, seed=0, q=0.9, **kw):
        super().__init__(seed=seed, **kw)
        self.q = q

    def route(self, r, cl):
        B = cl.B
        now = cl.now
        prev = self.last_done.get(r.sess.sid)
        rate = (len(self.recent) + 1) / 5.0
        rows = []
        for j in range(cl.n):
            mb = self._match(j, r.ids)
            m = min(mb * B, r.prompt - 1)
            est = self.est_ttft(cl, self.pend[j], self.dec[j], self.ctx[j], r.prompt - m, m)
            age = ((self.tick[j] - self.stamp[j].get(r.ids[mb - 1], self.tick[j])) / self.cap) if mb else -1.0
            rows.append([r.turn, r.prompt, m, r.prompt - m, m / r.prompt, self.pend[j], self.dec[j], self.ctx[j],
                         cl.outstanding[j], est, age, len(self.shadow[j]) / self.cap,
                         (now - prev) if prev is not None else -1.0, sum(self.pend), sum(self.dec),
                         sum(cl.outstanding), rate])
        pred = MODELS[self.q].predict(np.asarray(rows, dtype=float))
        j = int(np.argmin(pred))
        # keep the logging state consistent with the parent (age stamps, arrival window)
        self.recent.append(now)
        while self.recent and self.recent[0] < now - 5.0:
            self.recent.popleft()
        return j


def run(regime, seed, policy):
    cell = dict(C.BASE)
    cell.update(C.REGIMES[regime])
    cell["seed"] = seed
    cost = CostModel.load(C.COST)
    B = cell["B"]
    ss = RS.build_sessions(cell, cost)
    ws, fly = RS.working_set_blocks(ss, cost, B)
    max_req = max(blocks(p + o, B) for s in ss for (p, o, _) in s.turns)
    cap = int(cell["headroom"] * fly / cell["N"]) + max_req + int(cell["cap_frac"] * ws / cell["N"])
    pol = QuantRoute(seed=seed, q=policy[1]) if policy[0] == "quant" else C.LoggingEvictAware(seed=seed)
    cl = Cluster(cell["N"], cap, cost, pol, B=B)
    done = cl.run(ss)
    from prefixroute.metrics.stats import cell_metrics
    m = cell_metrics(done, cl, 1.0)
    return {"regime": regime, "seed": seed, "policy": f"{policy[0]}{policy[1] or ''}",
            "ttft_p95": m["ttft_p95"], "ttft_p99": m["ttft_p99"], "ttft_mean": m["ttft_mean"],
            "thpt": m["thpt_req_s"], "hit_rate": m["hit_rate"]}


if __name__ == "__main__":
    tr = pd.concat([E.load("base", 0), E.load("base", 1)])
    tr = tr[tr.warm]
    MODELS = E.fit_qr(tr)
    jobs = [(g, s, p) for g in ("base", "util0.9", "slowtool") for s in (3, 4)
            for p in (("evict", None), ("quant", 0.5), ("quant", 0.9))]
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.starmap(run, jobs)
    df = pd.DataFrame(res)
    df.to_csv("results_routing.csv", index=False)
    piv = df.pivot_table(index=["regime", "seed"], columns="policy", values=["ttft_p95", "ttft_mean"])
    print((piv * 1e3).round(2).to_string())
