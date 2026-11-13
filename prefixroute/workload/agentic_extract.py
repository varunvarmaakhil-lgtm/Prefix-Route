"""Extract a numeric agentic workload descriptor from the companion FHIR-agent run logs.

Only numeric fields are read and written (steps, token totals, per-tool-call output size and latency).
No prompt, answer, question or clinical content leaves this function. Output: data/agentic_trace.parquet
with one row per (session, turn).

The companion logs record per-session totals (tokens_in = sum of prompt tokens over all LLM turns) and,
per tool call, the step index, the size of the tool output in characters and the tool latency. Because the
agent loop only ever *appends* to its message list, the prompt of turn t is the prompt of turn t-1 plus the
model's previous output plus the tool results (plus a small chat-template overhead):

    P_t = P_1 + sum_{j<t} (o_j + g + c_j / k)

with P_1 the initial prompt (system + tool schemas + question), o_j the output tokens of turn j, c_j the tool
output characters appended after turn j, k characters per token and g per-turn template overhead. Summing over
the S turns of a session gives a linear model for the logged tokens_in, fitted per (model, agent, config) cell
by least squares on P_1, 1/k and g. The fit quality is written to data/agentic_fit.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

NUMERIC_CALL_FIELDS = ("step", "out_chars", "ms")


def _load_cell(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            calls = [{k: c.get(k) for k in NUMERIC_CALL_FIELDS} for c in r.get("calls", [])]
            rows.append({"steps": int(r["steps"]), "tokens_in": int(r["tokens_in"]),
                         "tokens_out": int(r["tokens_out"]), "llm_ms": float(r.get("llm_ms") or 0.0),
                         "agent": str(r.get("agent")), "error": r.get("error") is not None, "calls": calls})
    return rows


def _per_step(calls, S):
    c = np.zeros(S)
    ms = np.zeros(S)
    for cl in calls:
        s = int(cl["step"] or 0)
        if 1 <= s <= S:
            c[s - 1] += cl["out_chars"] or 0
            ms[s - 1] += cl["ms"] or 0.0
    return c, ms


def fit_cell(rows):
    X, y = [], []
    for r in rows:
        S = r["steps"]
        o = r["tokens_out"] / S
        c, _ = _per_step(r["calls"], S)
        w = np.array([S - t for t in range(1, S + 1)], dtype=float)  # weight of increment after turn t
        w_inc = w[:S - 1] if S > 1 else np.zeros(0)
        X.append([S, w_inc.sum(), (w_inc * c[:S - 1]).sum()])
        y.append(r["tokens_in"] - o * w_inc.sum())
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    p1, g, inv_k = coef
    fallback = False
    if not (p1 > 0 and 0 <= g < 200 and 1 / 8 < inv_k < 1 / 1.5):
        # implausible fit: fix k and g, refit P_1 only
        fallback = True
        inv_k, g = 1 / 3.5, 12.0
        p1 = float(np.mean((y - X[:, 1] * g - X[:, 2] * inv_k) / X[:, 0]))
    pred = X @ np.array([p1, g, inv_k])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return {"p1": float(p1), "g": float(g), "chars_per_token": float(1 / inv_k),
            "r2": 1 - ss_res / ss_tot, "n": len(rows), "fallback": fallback,
            "mape": float(np.mean(np.abs(y - pred) / np.maximum(y, 1)))}


def extract(raw_dir, out_parquet, fit_json, models=("qwen14b_awq", "qwen7b")):
    turns, fits = [], {}
    sid = 0
    agent_codes = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        cell = os.path.basename(path)[:-6]
        model = cell.split("__")[0]
        if model not in models:
            continue
        rows = [r for r in _load_cell(path) if not r["error"] and r["steps"] >= 1]
        if len(rows) < 30:
            continue
        f = fit_cell(rows)
        fits[cell] = f
        for r in rows:
            S = r["steps"]
            c, ms = _per_step(r["calls"], S)
            o = r["tokens_out"] / S
            code = agent_codes.setdefault(r["agent"].split("-")[0], len(agent_codes))
            p = f["p1"]
            for t in range(S):
                prev = 0.0 if t == 0 else p
                if t > 0:
                    p = p + o + f["g"] + c[t - 1] / f["chars_per_token"]
                turns.append({"session": sid, "turn": t, "n_turns": S, "template": code,
                              "cell_idx": len(fits) - 1, "prompt_tokens": int(round(p)),
                              "shared_prefix_tokens": int(round(prev)), "output_tokens": int(round(o)),
                              "tool_chars": int(c[t]), "tool_ms": float(ms[t]),
                              "llm_ms_mean": r["llm_ms"] / S})
            sid += 1
    df = pd.DataFrame(turns)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    with open(fit_json, "w") as fh:
        json.dump({"cells": fits, "agent_template_codes": agent_codes, "n_sessions": sid,
                   "n_turns": len(df)}, fh, indent=1)
    return df, fits


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "path/to/agent_run_logs"
    df, fits = extract(raw, "data/agentic_trace.parquet", "data/agentic_fit.json")
    r2 = [f["r2"] for f in fits.values()]
    print(f"sessions={df.session.nunique()} turns={len(df)} cells={len(fits)} "
          f"R2 median={np.median(r2):.3f} min={np.min(r2):.3f} fallback={sum(f['fallback'] for f in fits.values())}")
    print(df.groupby("turn").prompt_tokens.describe().head(12).round(0).to_string())
