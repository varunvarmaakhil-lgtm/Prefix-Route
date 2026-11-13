"""Per-cell metrics from completed requests."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, stat, n_boot=200, seed=0, alpha=0.05):
    x = np.asarray(x)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    vals = stat(x[idx], axis=1)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def cell_metrics(done, cluster, slo_s, warmup_frac=0.1):
    reqs = sorted(done, key=lambda r: r.arrival)
    t_arr = np.array([r.arrival for r in reqs])
    t0 = t_arr.min() + warmup_frac * (t_arr.max() - t_arr.min())
    keep = [r for r in reqs if r.arrival >= t0]
    ttft = np.array([r.t_first - r.arrival for r in keep])
    e2e = np.array([r.t_done - r.arrival for r in keep])
    prompt = np.array([r.prompt for r in keep], dtype=float)
    cached = np.array([r.cached for r in keep], dtype=float)
    outt = np.array([r.out for r in keep], dtype=float)
    span = max(r.t_done for r in keep) - min(r.arrival for r in keep)
    reps = cluster.reps
    work = np.array([rp.busy_time for rp in reps])
    nreq = np.array([rp.n_done for rp in reps], dtype=float)
    q95 = lambda a, axis=None: np.quantile(a, 0.95, axis=axis)
    m = {
        "n_req": len(keep), "n_arrived": cluster.n_arrived, "n_done": len(done),
        "ttft_p50": float(np.quantile(ttft, 0.5)), "ttft_p95": float(q95(ttft)),
        "ttft_p99": float(np.quantile(ttft, 0.99)), "ttft_mean": float(ttft.mean()),
        "ttft_p95_ci": bootstrap_ci(ttft, q95),
        "e2e_p95": float(q95(e2e)), "e2e_mean": float(e2e.mean()),
        "hit_rate": float(cached.sum() / prompt.sum()),
        "hit_frac_mean": float((cached / prompt).mean()),
        "thpt_req_s": len(keep) / span, "thpt_tok_s": float(outt.sum() / span),
        "prefill_tokens": int(sum(rp.prefill_tokens for rp in reps)),
        "evictions": int(sum(rp.evictions for rp in reps)),
        "imbalance_cv_work": float(work.std() / work.mean()) if work.mean() > 0 else 0.0,
        "imbalance_cv_req": float(nreq.std() / nreq.mean()) if nreq.mean() > 0 else 0.0,
        "slo_viol": float((ttft > slo_s).mean()), "slo_s": slo_s,
    }
    # agentic: per-session completion time (first arrival -> last done) for sessions fully after warm-up
    sess = {}
    for r in reqs:
        a, d = sess.get(r.sess.sid, (r.arrival, r.t_done))
        sess[r.sess.sid] = (min(a, r.arrival), max(d, r.t_done))
    st = np.array([d - a for (a, d) in sess.values() if a >= t0])
    m["session_time_mean"] = float(st.mean()) if len(st) else float("nan")
    m["session_time_p95"] = float(np.quantile(st, 0.95)) if len(st) else float("nan")
    m.update(miss_decomposition(keep, cluster.B, cluster.cost))
    return m


EXPENSIVE_TOKENS = 1024   # a miss that recomputes more than this many previously-computed prefix tokens


def miss_decomposition(reqs, B, cost=None):
    """Classify each request by how much *previously computed* prefix it had to recompute.

    reusable = the prefix that existed before this request (turn 0: the shared template; turn t>0: the previous
    turn's prompt + output, i.e. what an ideal cache would still hold), floored to whole blocks.
    recomputed = reusable - cached tokens (evicted, or placed on a replica that never held it).
    hit: recomputed < 1 block; cheap miss: <= EXPENSIVE_TOKENS; expensive miss: more.
    """
    rec, ttft, wait, cls, delay = [], [], [], [], []
    for r in reqs:
        s = r.sess
        if r.turn == 0:
            reusable = min(s.sys_tokens, r.prompt)
        else:
            pp, po, _ = s.turns[r.turn - 1]
            reusable = min(pp + po if s.append_output else pp, r.prompt)
        reusable = (reusable // B) * B
        x = max(0, reusable - r.cached)
        rec.append(x)
        ttft.append(r.t_first - r.arrival)
        wait.append((r.admit_t or r.arrival) - r.arrival)
        # delay caused by other requests = TTFT minus this request's own isolated (unloaded) prefill time
        own = cost.prefill_time(r.prompt - r.cached, r.cached) if cost is not None else 0.0
        delay.append(max(0.0, (r.t_first - r.arrival) - own))
        cls.append(0 if x < B else (1 if x <= EXPENSIVE_TOKENS else 2))
    rec, ttft, wait, cls, delay = map(np.asarray, (rec, ttft, wait, cls, delay))
    tail = ttft >= np.quantile(ttft, 0.95)
    miss = cls > 0
    out = {"frac_hit": float((cls == 0).mean()), "frac_cheap": float((cls == 1).mean()),
           "frac_exp": float((cls == 2).mean()),
           "recomp_per_req": float(rec.mean()),
           "recomp_per_miss_mean": float(rec[miss].mean()) if miss.any() else 0.0,
           "recomp_per_miss_p95": float(np.quantile(rec[miss], 0.95)) if miss.any() else 0.0,
           "wait_mean": float(wait.mean()), "wait_p95": float(np.quantile(wait, 0.95)),
           "delay_mean": float(delay.mean()), "delay_p95": float(np.quantile(delay, 0.95)),
           "delay_share_tail": float(delay[tail].sum() / max(1e-12, ttft[tail].sum()))}
    for k, name in enumerate(("hit", "cheap", "exp")):
        sel = cls == k
        out[f"ttft_mean_{name}"] = float(ttft[sel].mean()) if sel.any() else float("nan")
        out[f"ttft_p95_{name}"] = float(np.quantile(ttft[sel], 0.95)) if sel.any() else float("nan")
        out[f"tail_share_{name}"] = float((sel & tail).sum() / max(1, tail.sum()))
        out[f"delay_mean_{name}"] = float(delay[sel].mean()) if sel.any() else float("nan")
    return out
