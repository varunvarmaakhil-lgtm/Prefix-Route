"""Run simulation cells (one JSON per cell, atomic write + .done marker, skipped when already done).

    python -m prefixroute.run_sim --exp configs/experiments.json --groups main cap --jobs 60
    python -m prefixroute.run_sim --list            # print cell counts per group
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

from .metrics.stats import cell_metrics
from .policies.routers import POLICIES
from .sim.costmodel import CostModel, blocks
from .sim.engine import Cluster
from .workload import sessions as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@functools.lru_cache(maxsize=4)
def _agentic(path):
    return W.AgenticTrace(path)


@functools.lru_cache(maxsize=4)
def _chat(path):
    return W.ChatTrace(path)


def working_set_blocks(sessions, cost, B):
    """Nominal (unloaded, no-reuse) timeline footprints, in blocks, time-averaged over the trace:
      ws_reuse  - reusable state: shared templates + contexts of sessions *between* turns (waiting on a
                  tool), i.e. what must stay cached for the next turn to hit;
      inflight  - contexts of requests being served (needed to run the batch at all)."""
    reuse_s, fly_s, t_min, t_max = 0.0, 0.0, float("inf"), 0.0
    tpl = {}
    for s in sessions:
        t = s.arrival
        t_min = min(t_min, t)
        for k, (p, o, th) in enumerate(s.turns):
            dur = cost.prefill_time(p) + o * cost.decode_iter(16, p + o / 2)
            fly_s += blocks(p + o, B) * dur
            if k + 1 < len(s.turns):
                reuse_s += (blocks(p + o, B) - s.sys_tokens // B) * th
                t += th
            t += dur
        t_max = max(t_max, t)
        tpl[s.tpl] = max(tpl.get(s.tpl, 0), s.sys_tokens // B)
    span = max(t_max - t_min, 1e-9)
    return reuse_s / span + sum(tpl.values()), fly_s / span


@functools.lru_cache(maxsize=2)
def base_tool_median(wl):
    """Median tool latency (s) between consecutive turns in the unscaled workload (recorded against a mock server)."""
    tr = _agentic(os.path.join(ROOT, "data", "agentic_trace.parquet"))
    if wl == "agentic":
        d = tr.df
        return float(d[d.turn < d.n_turns - 1].tool_ms.median()) / 1000.0
    return float(np.exp(tr.fit_synth()["think_ln"][0]))


def think_scale(cell):
    """tool_median_s (absolute median tool latency) takes precedence over the relative think_scale."""
    if cell.get("tool_median_s") is not None and cell["workload"] != "chat":
        return cell["tool_median_s"] / base_tool_median(cell["workload"])
    return cell.get("think_scale", 1.0)


def build_sessions(cell, cost):
    rng = np.random.default_rng(cell["seed"])
    data = os.path.join(ROOT, "data")
    wl = cell["workload"]
    n = cell["n_sessions"]
    if wl == "chat":
        ss = _chat(os.path.join(data, "azure", "AzureLLMInferenceTrace_conv.csv")).sample(n, rng)
        W.set_arrivals(ss, cost, cell["N"], cell["util"], rng, keep_trace_times=True)
    elif wl == "agentic":
        ss = _agentic(os.path.join(data, "agentic_trace.parquet")).sample(n, rng)
        ts = think_scale(cell)
        for s in ss:
            s.turns = [(p, o, th * ts) for (p, o, th) in s.turns]
        W.set_arrivals(ss, cost, cell["N"], cell["util"], rng, cv=cell.get("cv", 1.0))
    else:
        p = _agentic(os.path.join(data, "agentic_trace.parquet")).fit_synth()
        ss = W.synth_sessions(n, rng, p, cell.get("turns_scale", 1.0), cell.get("growth_scale", 1.0),
                              think_scale(cell))
        W.set_arrivals(ss, cost, cell["N"], cell["util"], rng, cv=cell.get("cv", 1.0))
    return ss


def run_cell(cell, cost_path, out_dir):
    cid = cell_id(cell)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, cid + ".json")
    if os.path.exists(path + ".done"):
        return cid, "skip"
    t0 = time.time()
    cost = CostModel.load(cost_path)
    B = cell.get("B", 32)
    ss = build_sessions(cell, cost)
    ws, fly = working_set_blocks(ss, cost, B)
    max_req = max(blocks(p + o, B) for s in ss for (p, o, _) in s.turns)
    # per-replica KV capacity = headroom for the running batch + cap_frac x (reusable working set / N)
    head = int(cell.get("headroom", 1.0) * fly / cell["N"]) + max_req
    cap = head + int(cell["cap_frac"] * ws / cell["N"])
    clamped = False
    pol = POLICIES[cell["policy"]](seed=cell["seed"], **cell.get("policy_kw", {}))
    cl = Cluster(cell["N"], cap, cost, pol, B=B, caching=cell.get("caching", True))
    done = cl.run(ss)
    m = cell_metrics(done, cl, cell.get("slo_s", 1.0))
    assert m["n_done"] == cl.n_arrived == sum(len(s.turns) for s in ss), "requests lost"
    rec = {"cell": cell, "cell_id": cid, "metrics": m, "cap_blocks": cap, "cap_clamped": clamped,
           "ws_blocks": ws, "inflight_blocks": fly, "headroom_blocks": head, "max_req_blocks": max_req, "cost_source": cost.source,
           "cost": cost.to_dict(), "wall_s": time.time() - t0}
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)
    open(path + ".done", "w").close()
    return cid, f"{time.time() - t0:.1f}s"


def guarded(c, cost_path, out_dir, deadline=None):
    if deadline and time.time() > deadline:
        return cell_id(c), "deadline-skip"
    try:
        return run_cell(c, cost_path, out_dir)
    except Exception as e:  # keep the sweep going; the failure is logged
        return cell_id(c), f"FAIL {type(e).__name__}: {e}"


def cell_id(cell):
    keys = ("workload", "policy", "N", "cap_frac", "util", "seed")
    base = "__".join(f"{k}={cell[k]}" for k in keys)
    extra = {k: v for k, v in cell.items() if k not in keys + ("group",)}
    h = hashlib.sha1(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:8]
    return f"{base}__{h}"


def expand(exp):
    """experiments.json: {"defaults": {...}, "groups": {name: {"grid": {k: [..]}, ...fixed}}}"""
    d = exp["defaults"]
    cells = []
    for g, spec in exp["groups"].items():
        grid = spec.get("grid", {})
        fixed = {k: v for k, v in spec.items() if k != "grid"}
        keys = list(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            c = dict(d)
            c.update(fixed)
            c.update(dict(zip(keys, combo)))
            c["group"] = g
            if isinstance(c.get("n_sessions"), dict):
                c["n_sessions"] = c["n_sessions"][c["workload"].split("_")[0]]
            cells.append(c)
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=os.path.join(ROOT, "configs", "experiments.json"))
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--cost", default=os.path.join(ROOT, "results", "measure", "costmodel.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--deadline", type=float, default=None, help="unix time after which no new cell starts")
    a = ap.parse_args(argv)
    exp = json.load(open(a.exp))
    cells = [c for c in expand(exp) if not a.groups or c["group"] in a.groups]
    # dedupe (groups may share cells, e.g. the main configuration)
    uniq = {}
    for c in cells:
        uniq.setdefault(cell_id(c), c)
    cells = list(uniq.values())
    if a.list:
        from collections import Counter
        print(Counter(c["group"] for c in cells), len(cells))
        return
    os.makedirs(a.out, exist_ok=True)
    todo = [c for c in cells if not os.path.exists(os.path.join(a.out, cell_id(c) + ".json.done"))]
    print(f"{len(cells)} cells, {len(todo)} to run, jobs={a.jobs}, cost={a.cost}", flush=True)
    from joblib import Parallel, delayed
    from prefixroute import run_sim as RS    # importable module, so workers unpickle by reference

    # heavy cells (large N, oracle) first for better packing
    todo.sort(key=lambda c: (-c["N"], c["policy"] != "oracle"))
    for cid, st in Parallel(n_jobs=a.jobs, return_as="generator_unordered", verbose=0)(
            delayed(RS.guarded)(c, a.cost, a.out, a.deadline) for c in todo):
        print(st.ljust(12), cid, flush=True)


if __name__ == "__main__":
    sys.exit(main())
