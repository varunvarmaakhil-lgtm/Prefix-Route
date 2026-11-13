"""Output-correctness check for prefix caching (greedy decoding, temperature 0).

Comparisons (token-by-token on the streamed tokens):
  probe:   within a server run, cold vs warm vs warm2 vs concurrent; across runs, caching on vs off.
  agentic: multi-turn growing-prefix replay under concurrency; run-vs-run on the same configuration gives the
           batch-nondeterminism baseline, which on-vs-off and pressure-vs-off are compared against.
A caching bug shows up as on-vs-off divergence clearly above the off-vs-off baseline and/or degenerate text.
Writes results/measure/correctness.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def degenerate(text):
    """Heuristic flag for corrupted generations: long single-token loops or replacement/control characters."""
    if re.search(r"(\S+)(\s*\1){7,}", text or ""):
        return True
    return bool(re.search(r"[�\x00-\x08\x0e-\x1f]", text or ""))


def compare(A, B):
    keys = sorted(set(A) & set(B))
    if not keys:
        return None
    divs = [first_div(A[k]["tokens"], B[k]["tokens"]) for k in keys]
    same = [d is None for d in divs]
    pos = [d for d in divs if d is not None]
    return {"n": len(keys), "exact_match": float(np.mean(same)),
            "first_div_median": float(np.median(pos)) if pos else None,
            "first_div_min": int(min(pos)) if pos else None,
            "degenerate_a": float(np.mean([degenerate(A[k]["text"]) for k in keys])),
            "degenerate_b": float(np.mean([degenerate(B[k]["text"]) for k in keys]))}


def load(path, keyf):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("error") is None:
                out.setdefault(r.get("tag"), {})
                out[r["tag"]].setdefault(r.get("phase", r.get("repeat")), {})[keyf(r)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/measure")
    a = ap.parse_args()
    res = {"probe": {}, "agentic": {}}
    P = load(os.path.join(a.dir, "probe.jsonl"), lambda r: r["i"])
    for tag, ph in P.items():
        for x, y in [("cold", "warm"), ("warm", "warm2"), ("cold", "concurrent")]:
            if x in ph and y in ph:
                res["probe"][f"{tag}:{x}-vs-{y}"] = compare(ph[x], ph[y])
    if "on" in P and "nocache" in P:
        for x in ("cold", "warm", "concurrent"):
            if x in P["on"] and x in P["nocache"]:
                res["probe"][f"on-vs-nocache:{x}"] = compare(P["on"][x], P["nocache"][x])
    G = defaultdict(dict)
    for p in glob.glob(os.path.join(a.dir, "agentic_*_r*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r.get("error") is None:
                G[(r["tag"], r["repeat"])][(r["sid"], r["turn"])] = r
    pairs = [(("on", 0), ("on", 1)), (("nocache", 0), ("nocache", 1)), (("on", 0), ("nocache", 0)),
             (("on", 1), ("nocache", 1)), (("pressure", 0), ("nocache", 0)), (("pressure", 0), ("on", 0))]
    for x, y in pairs:
        if x in G and y in G:
            c = compare(G[x], G[y])
            if c:
                # divergence by turn index (later turns reuse more cached prefix)
                by_turn = defaultdict(list)
                for k in set(G[x]) & set(G[y]):
                    by_turn[k[1]].append(first_div(G[x][k]["tokens"], G[y][k]["tokens"]) is None)
                c["exact_by_turn"] = {int(t): float(np.mean(v)) for t, v in sorted(by_turn.items()) if len(v) >= 5}
            res["agentic"][f"{x[0]}{x[1]}-vs-{y[0]}{y[1]}"] = c
    cached = {tag: float(np.nanmean([r["cached_tokens"] / r["prompt_tokens"] for r in d.values()
                                     if r.get("cached_tokens") is not None and r.get("prompt_tokens")]))
              for (tag, rep), d in G.items() if rep == 0}
    res["agentic_cached_frac"] = cached
    json.dump(res, open(os.path.join(a.dir, "correctness.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
