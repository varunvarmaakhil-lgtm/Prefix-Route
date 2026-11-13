"""Single-replica simulator validation against the real engine.

Replays the measured multi-turn agentic session set (results/measure/agentic_on_r0.jsonl: same arrival times,
recorded prompt/output token counts, same tool gaps) through the simulator with N=1, the measured cost model
and the engine's reported KV capacity, and compares TTFT p50/p95 and prefix hit rate.
Writes results/measure/validation.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prefixroute.policies.routers import POLICIES  # noqa: E402
from prefixroute.sim.costmodel import CostModel  # noqa: E402
from prefixroute.sim.engine import Cluster  # noqa: E402
from prefixroute.workload.sessions import Session  # noqa: E402


def main(mdir=os.path.join(ROOT, "results", "measure"), tag="on", cost_name="costmodel.json", suffix=""):
    rec_p = os.path.join(mdir, f"agentic_{tag}_r0.jsonl")
    ses_p = os.path.join(mdir, "agentic_sessions_seed11.json")
    cm_p = os.path.join(mdir, cost_name)
    if not (os.path.exists(rec_p) and os.path.exists(ses_p) and os.path.exists(cm_p)):
        print("validation skipped: missing", [p for p in (rec_p, ses_p, cm_p) if not os.path.exists(p)])
        return
    recs = {}
    for line in open(rec_p):
        r = json.loads(line)
        if r.get("error") is None and r.get("prompt_tokens"):
            recs[(r["sid"], r["turn"])] = r
    spec = json.load(open(ses_p))
    cost = CostModel.load(cm_p)
    caps = json.load(open(cm_p)).get("kv_capacity_tokens", {})
    cap_tokens = caps.get(f"server_{tag}.log") or max(caps.values(), default=400000)
    B = 16
    # template prefix = shortest first-turn prompt of the template minus the question (~30 tokens)
    first = {}
    for s in spec:
        r = recs.get((s["sid"], 0))
        if r:
            first.setdefault(s["tpl"], []).append(r["prompt_tokens"])
    sys_tok = {k: max(0, min(v) - 30) for k, v in first.items()}
    sessions = []
    for s in spec:
        turns = []
        for t in range(len(s["out_tokens"])):
            r = recs.get((s["sid"], t))
            if r is None:
                break
            think = s["think_s"][t] if t < len(s["think_s"]) else 0.0
            turns.append((int(r["prompt_tokens"]), int(max(1, r["n_out"] or s["out_tokens"][t])), float(think)))
        if turns:
            sessions.append(Session(s["sid"], s["tpl"], sys_tok.get(s["tpl"], 0), float(s["arrival"]), turns,
                                    append_output=False))
    cl = Cluster(1, int(cap_tokens // B), cost, POLICIES["round_robin"](), B=B)
    done = cl.run(sessions)
    sim_ttft = np.array([r.t_first - r.arrival for r in done])
    sim_hit = sum(r.cached for r in done) / sum(r.prompt for r in done)
    real = [recs[(r.sess.sid, r.turn)] for r in done]
    real_ttft = np.array([r["ttft"] for r in real])
    real_hit = (sum(r["cached_tokens"] or 0 for r in real) / sum(r["prompt_tokens"] for r in real)
                if all(r.get("cached_tokens") is not None for r in real) else None)
    out = {"n_requests": len(done), "cap_tokens": cap_tokens,
           "real_ttft_p50": float(np.median(real_ttft)), "sim_ttft_p50": float(np.median(sim_ttft)),
           "real_ttft_p95": float(np.quantile(real_ttft, 0.95)), "sim_ttft_p95": float(np.quantile(sim_ttft, 0.95)),
           "real_hit": real_hit, "sim_hit": float(sim_hit)}
    out["ttft_p50_err_pct"] = 100 * (out["sim_ttft_p50"] / out["real_ttft_p50"] - 1)
    out["ttft_p95_err_pct"] = 100 * (out["sim_ttft_p95"] / out["real_ttft_p95"] - 1)
    out["hit_err_pp"] = 100 * (sim_hit - real_hit) if real_hit is not None else None
    # per-request correlation of TTFT
    out["ttft_spearman"] = float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(real_ttft, sim_ttft).statistic)
    json.dump(out, open(os.path.join(mdir, f"validation{suffix}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
    main(cost_name="costmodel_cachedcost.json", suffix="_cachedcost")
