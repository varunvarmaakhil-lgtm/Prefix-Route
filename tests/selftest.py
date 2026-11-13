"""Simulator / pipeline self-test (handoff A4).   python tests/selftest.py

Invariants on toy traces: no request served before arrival; replica KV capacity never exceeded; hit rate 0
when nothing is reusable (or caching is off); near-perfect reuse under the clairvoyant policy when
co-location is possible; total served == total arrived; block accounting returns to zero. Plus: the cost
model fit recovers known coefficients, and an end-to-end pass through the analysis on a small grid.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prefixroute.policies.routers import POLICIES  # noqa: E402
from prefixroute.sim.costmodel import CostModel  # noqa: E402
from prefixroute.sim import engine as E  # noqa: E402
from prefixroute.workload.sessions import Session  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class CheckedReplica(E.Replica):
    max_used = 0

    def try_admit(self, r, now):
        ok = super().try_admit(r, now)
        CheckedReplica.max_used = max(CheckedReplica.max_used, self.used() / self.cap)
        assert self.used() <= self.cap, "capacity exceeded"
        assert self.private >= 0
        return ok

    def release(self, r):
        super().release(r)
        assert self.used() <= self.cap and self.private >= 0


def run(sessions, policy, N=4, cap=100000, caching=True, B=16):
    E.Replica = CheckedReplica
    cost = CostModel()
    cl = E.Cluster(N, cap, cost, POLICIES[policy](seed=0), B=B, caching=caching)
    done = cl.run(sessions)
    E.Replica = CheckedReplica.__mro__[1]
    return cl, done


def toy_unique(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(0.05, n))
    return [Session(i, i, 0, float(t[i]), [(int(rng.integers(64, 2000)), int(rng.integers(1, 64)), 0.0)])
            for i in range(n)]


def toy_identical(n=200, L=1024, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(0.05, n))
    return [Session(i, 0, L, float(t[i]), [(L, 16, 0.0)]) for i in range(n)]


def toy_growing(n=60, turns=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    t = 0.0
    for i in range(n):
        t += rng.exponential(0.3)
        p, turns_l = 512, []
        for k in range(turns):
            o = 32
            turns_l.append((p, o, 0.2))
            p += o + 256
        out.append(Session(i, i % 3, 256, t, turns_l))
    return out


def main():
    print("[1] invariants on toy traces, every policy")
    for pol in POLICIES:
        for name, ss in [("unique", toy_unique()), ("identical", toy_identical()), ("growing", toy_growing())]:
            n_req = sum(len(s.turns) for s in ss)
            cl, done = run(ss, pol, cap=4000)
            check(len(done) == n_req == cl.n_arrived, f"{pol}/{name}: served {len(done)} == arrived {cl.n_arrived}")
            check(all(r.admit_t >= r.arrival - 1e-12 and r.t_first >= r.arrival and r.t_done >= r.t_first
                      for r in done), f"{pol}/{name}: nothing served before arrival")
            check(all(rp.private == 0 and not rp.ref for rp in cl.reps), f"{pol}/{name}: block accounting drains")
            # closed-loop turns: turn k+1 arrives after turn k completes
            by = {}
            for r in done:
                by.setdefault(r.sess.sid, []).append(r)
            ok = all(b.arrival >= a.t_done - 1e-9 for rs in by.values()
                     for a, b in zip(sorted(rs, key=lambda r: r.turn), sorted(rs, key=lambda r: r.turn)[1:]))
            check(ok, f"{pol}/{name}: session turns are closed-loop")
            if name == "unique":
                hit = sum(r.cached for r in done)
                check(hit == 0, f"{pol}/unique: hit rate 0 when nothing is shared ({hit} cached tokens)")
    check(CheckedReplica.max_used <= 1.0, f"capacity never exceeded (max used fraction {CheckedReplica.max_used:.3f})")

    print("[2] hit rate 0 with prefix caching off")
    for pol in ("round_robin", "cache_aware", "oracle"):
        cl, done = run(toy_identical(), pol, caching=False)
        check(sum(r.cached for r in done) == 0, f"{pol}: no hits with caching off")

    print("[3] perfect co-location -> reuse efficiency ~1")
    ss = toy_identical(n=300, L=1024)
    cl, done = run(ss, "oracle", N=4, cap=100000)
    reusable = sum(r.prompt - 1 for r in done) - (1024 - 1) * len({r.replica for r in done})
    eff = sum(r.cached for r in done) / reusable
    check(eff > 0.99, f"oracle, identical prompts: reuse efficiency {eff:.3f} (cached / reusable)")
    ss = toy_growing()
    cl, done = run(ss, "session_affinity", N=4, cap=100000)
    later = [r for r in done if r.turn > 0]
    exp = [((r.sess.turns[r.turn - 1][0] + r.sess.turns[r.turn - 1][1]) // 16) * 16 for r in later]
    check(all(r.cached >= e for r, e in zip(later, exp)),
          "session affinity, growing prefixes, ample cache: each turn reuses the previous prompt+output")

    print("[4] eviction under pressure")
    cl, done = run(toy_growing(n=120), "round_robin", N=2, cap=300)
    check(sum(rp.evictions for rp in cl.reps) > 0, "small cache evicts")
    cl2, done2 = run(toy_growing(n=120), "round_robin", N=2, cap=100000)
    h1 = sum(r.cached for r in done) / sum(r.prompt for r in done)
    h2 = sum(r.cached for r in done2) / sum(r.prompt for r in done2)
    check(h1 < h2, f"hit rate falls with capacity ({h1:.3f} < {h2:.3f})")

    print("[5] single request TTFT equals the cost model's isolated prefill time")
    s = [Session(0, 0, 0, 0.0, [(5000, 4, 0.0)])]
    cl, done = run(s, "round_robin", N=1)
    check(abs(done[0].t_first - CostModel().prefill_time(5000)) < 1e-9, "sim TTFT == CostModel.prefill_time")

    print("[6] cost-model fit recovers known coefficients (noise-free synthetic measurements)")
    import pandas as pd
    from prefixroute.measure.fit import mechanistic_fit
    gt = CostModel(a=0.01, b=5e-5, c=2e-4, dp=3e-6, dd=4e-5)
    rows = []
    for L in (1024, 4096, 8192, 16384):
        for h in (0, 0.25, 0.5, 0.75, 0.9, 1.0):
            cached = min(int(L * h) // 16 * 16, L - 1)
            rows.append({"c": 1, "L": L, "h": h, "caching": True, "cached_tokens": cached,
                         "ttft": gt.prefill_time(L - cached, cached), "itl_p50": None})
        for c in (1, 4, 16, 32):
            rows.append({"c": c, "L": L, "h": 0.0, "caching": True, "cached_tokens": 0, "ttft": None,
                         "itl_p50": gt.decode_iter(c, L + 32)})
    df = pd.DataFrame(rows)
    df = df[(df.ttft.notna()) | (df.itl_p50.notna())]
    f = mechanistic_fit(df, 8192)
    rel = {k: abs(f[k] - getattr(gt, k)) / getattr(gt, k) for k in ("a", "b", "c", "dp", "dd")}
    check(max(rel.values()) < 0.05, f"fit relative errors {json.dumps({k: round(v, 4) for k, v in rel.items()})}")

    print("[7] end-to-end: small grid -> run_sim -> analysis")
    out = os.path.join(ROOT, ".selftest")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    exp = {"defaults": {"N": 2, "cap_frac": 1.0, "util": 0.6, "B": 32, "headroom": 2.0, "slo_s": 1.0,
                        "n_sessions": {"chat": 400, "agentic": 150, "synth": 150}},
           "groups": {"main": {"grid": {"workload": ["chat", "agentic", "synth"],
                                        "policy": ["least_loaded", "cache_aware", "oracle"], "seed": [0, 1]}}}}
    json.dump(exp, open(os.path.join(out, "exp.json"), "w"))
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-m", "prefixroute.run_sim", "--exp", os.path.join(out, "exp.json"),
                        "--jobs", "4", "--cost", "none", "--out", os.path.join(out, "raw")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    n_done = len([f for f in os.listdir(os.path.join(out, "raw")) if f.endswith(".done")])
    check(r.returncode == 0 and n_done == 18, f"run_sim small grid: {n_done}/18 cells ({r.stderr[-300:]})")
    r = subprocess.run([sys.executable, "-m", "prefixroute.run_sim", "--exp", os.path.join(out, "exp.json"),
                        "--jobs", "2", "--cost", "none", "--out", os.path.join(out, "raw")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    check("0 to run" in r.stdout, "resubmission skips finished cells")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "make_tables.py"), "--raw",
                        os.path.join(out, "raw"), "--out", out, "--measure", os.path.join(out, "nomeasure")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    check(r.returncode == 0 and os.path.exists(os.path.join(out, "macros.tex")),
          f"analysis/make_tables.py end-to-end ({r.stderr[-400:]})")

    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
