"""Turn C1 measurements into (i) results/measure/curves.parquet, (ii) an empirical TTFT model
TTFT = f(prompt_len, hit_fraction, concurrency) with R^2 and residuals, and (iii) the mechanistic
per-iteration CostModel used by the simulator (results/measure/costmodel.json).

    python -m prefixroute.measure.fit --dir results/measure
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd


def load_sweep(d):
    rows = []
    for path in glob.glob(os.path.join(d, "sweep*.jsonl")):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                mb, ma = r.get("metrics_before", {}), r.get("metrics_after", {})
                dlt = {k: ma.get(k, 0.0) - mb.get(k, 0.0) for k in ("vllm:gpu_prefix_cache_hits_total",
                       "vllm:gpu_prefix_cache_queries_total", "vllm:num_preemptions_total")}
                q = dlt["vllm:gpu_prefix_cache_queries_total"]
                ctr_hit = dlt["vllm:gpu_prefix_cache_hits_total"] / q if q > 0 else np.nan
                nominal_q = len(r["ttft"]) * (r["L"] // 16 - 1)       # block queries if nothing is recomputed
                for k, t in enumerate(r["ttft"]):
                    rows.append({"tag": r["tag"], "c": r["c"], "L": r["L"], "h": r["h"], "req": k, "ttft": t,
                                 "e2e": r["e2e"][k], "itl_mean": r["itl_mean"][k], "itl_p50": r["itl_p50"][k],
                                 "cached_tokens": r["cached_tokens"][k], "prompt_tokens": r["prompt_tokens"][k],
                                 "n_out": r["n_out"][k], "kv_usage_max": r.get("kv_usage_max"),
                                 "thpt_tok_s": r.get("thpt_tok_s"), "wall_s": r["wall_s"],
                                 "ctr_hit": ctr_hit, "preemptions": dlt["vllm:num_preemptions_total"],
                                 "query_inflation": q / nominal_q if nominal_q > 0 else np.nan})
    df = pd.DataFrame(rows)
    if len(df):
        df["caching"] = ~df.tag.str.contains("nocache")
    return df


def summarise(df):
    g = df.groupby(["tag", "caching", "c", "L", "h"])
    s = g.agg(n=("ttft", "size"), ttft_p50=("ttft", "median"), ttft_mean=("ttft", "mean"),
              ttft_p95=("ttft", lambda x: np.quantile(x, 0.95)), itl_p50=("itl_p50", "median"),
              e2e_p50=("e2e", "median"), thpt_tok_s=("thpt_tok_s", "first"), kv_usage_max=("kv_usage_max", "first"),
              ctr_hit=("ctr_hit", "first"), preemptions=("preemptions", "first"),
              query_inflation=("query_inflation", "first"),
              cached_frac=("cached_tokens", lambda x: np.nan if x.isna().all() else np.nanmean(x.astype(float))))
    s = s.reset_index()
    s["cached_frac"] = s.cached_frac / s.L
    return s


def empirical_fit(df):
    """log TTFT ~ log(new tokens) + log(L) + log(c) + interactions, on per-request data (caching on)."""
    d = df[df.caching & df.ttft.notna()].copy()
    if len(d) < 10:
        return None
    new = np.maximum(d.L * (1 - d.h), 16)
    X = np.column_stack([np.ones(len(d)), np.log(new), np.log(d.L), np.log(d.c), np.log(new) * np.log(d.c),
                         (new / 1e4) ** 2])
    y = np.log(d.ttft.values)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2_log = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    lin_pred, lin_y = np.exp(pred), d.ttft.values
    r2_lin = 1 - ((lin_y - lin_pred) ** 2).sum() / ((lin_y - lin_y.mean()) ** 2).sum()
    return {"form": "log TTFT = b0 + b1 log(n_new) + b2 log(L) + b3 log(c) + b4 log(n_new)log(c) + b5 (n_new/1e4)^2",
            "coef": coef.tolist(), "r2_log": float(r2_log), "r2_linear": float(r2_lin), "n": int(len(d)),
            "mape": float(np.mean(np.abs(lin_pred - lin_y) / lin_y)),
            "residuals": {"L": d.L.tolist(), "h": d.h.tolist(), "c": d.c.tolist(), "resid_log": (y - pred).tolist()}}


def mechanistic_fit(df, chunk):
    """Fit a, b, c, dp, dd of T_iter = a + b P + c D + dp A_p + dd A_d on (i) isolated TTFT (c=1, caching on and off) and
    (ii) decode ITL at concurrency c (steady state, median gap)."""
    rows_X, rows_y, w = [], [], []
    iso = df[(df.c == 1) & df.ttft.notna()]
    for _, r in iso.iterrows():
        cached = (r.cached_tokens if r.caching and not pd.isna(r.cached_tokens)
                  else (int(r.L * r.h) // 16 * 16 if r.caching else 0))
        cached = min(cached, r.L - 1)
        n_new, ctx = r.L - cached, cached
        n_it, P, A = 0, 0, 0.0
        while n_new > 0:
            ch = min(chunk, n_new)
            n_it += 1
            P += ch
            A += ch * (ctx + ch / 2) / 1e3
            ctx += ch
            n_new -= ch
        rows_X.append([n_it, P, 0.0, A, 0.0])
        rows_y.append(r.ttft)
        w.append(1.0 / max(r.ttft, 1e-3))          # relative error weighting
    dec = df[df.itl_p50.notna()]
    for (c, L, h, caching), g in dec.groupby(["c", "L", "h", "caching"]):
        itl = float(np.median(g.itl_p50))
        ctx = L + 32
        rows_X.append([1, 0, c, 0.0, c * ctx / 1e3])
        rows_y.append(itl)
        w.append(3.0 / max(itl, 1e-4))
    X, y, w = np.array(rows_X), np.array(rows_y), np.array(w)
    from scipy.optimize import nnls
    coef, _ = nnls(X * w[:, None], y * w)
    pred = X @ coef
    rel = np.abs(pred - y) / y
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n_iso = len(iso)
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "dp": float(coef[3]), "dd": float(coef[4]),
            "r2": float(r2), "mape_ttft": float(rel[:n_iso].mean()) if n_iso else None,
            "mape_itl": float(rel[n_iso:].mean()) if len(rel) > n_iso else None,
            "n_ttft": int(n_iso), "n_itl": int(len(rel) - n_iso)}


def kv_capacity_tokens(d):
    """Parse the engine's KV capacity from the server logs (V1: 'GPU KV cache size: N tokens')."""
    caps = {}
    for p in glob.glob(os.path.join(d, "server_*.log")):
        txt = open(p, errors="ignore").read()
        m = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", txt)
        if m:
            caps[os.path.basename(p)] = int(m[-1].replace(",", ""))
        else:
            m = re.findall(r"# GPU blocks:\s*(\d+)", txt)
            if m:
                caps[os.path.basename(p)] = int(m[-1]) * 16
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/measure")
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    df = load_sweep(a.dir)
    if df.empty:
        print("no sweep data")
        return
    df.to_parquet(os.path.join(a.dir, "curves_requests.parquet"), index=False)
    s = summarise(df)
    s.to_parquet(os.path.join(a.dir, "curves.parquet"), index=False)
    main_df = df[~df.tag.str.contains("pressure")]
    emp = empirical_fit(main_df)
    mech = mechanistic_fit(main_df, a.chunk)
    caps = kv_capacity_tokens(a.dir)
    cm = {k: mech[k] for k in ("a", "b", "c", "dp", "dd")}
    cm.update({"chunk": a.chunk, "max_seqs": 256, "source": "A100-measured",
               "fit": mech, "kv_capacity_tokens": caps})
    with open(os.path.join(a.dir, "costmodel.json"), "w") as fh:
        json.dump(cm, fh, indent=1)
    with open(os.path.join(a.dir, "fit_empirical.json"), "w") as fh:
        json.dump(emp, fh)
    print(json.dumps({"mech": mech, "emp_r2_log": emp and emp["r2_log"], "emp_r2_lin": emp and emp["r2_linear"],
                      "kv": caps}, indent=1))


if __name__ == "__main__":
    main()
