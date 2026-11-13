"""Per-iteration latency model of one serving replica, calibrated from the A100 measurements (C1).

An engine iteration (vLLM V1-style continuous batching with chunked prefill) processes P prefill tokens
spread over some prefill chunks and D decode sequences. Its duration is modelled as

    T_iter = a + b * P + c * D + dp * A_p + dd * A_d

where A_p = sum over prefill chunks of chunk * (ctx_before + chunk/2) / 1e3 (causal attention work of the
chunk against its context, compute-bound) and A_d = sum over decode sequences of their context / 1e3
(KV-cache reads, memory-bound).

Defaults are a first-principles roofline estimate for Qwen2.5-7B (AWQ) on one A100-40GB, used only when no
measurement exists (source = "roofline-estimate"): a = weight read 5.5 GB / 1.55 TB/s + ~2.5 ms scheduling;
b = 2 x 7.6e9 FLOP/token at 50% of 312 TFLOP/s; c = per-sequence sampling/bookkeeping; dp = 4 x 28 layers x
3584 x 1e3 FLOP per k-token^2 at 50% peak; dd = 57 KB of KV per context token x 1e3 / 1.55 TB/s.
Coefficients come from `results/measure/costmodel.json`, written by `prefixroute.measure.fit`.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict


@dataclass
class CostModel:
    a: float = 0.006         # s, fixed per-iteration cost
    b: float = 1.0e-4        # s per prefill token
    c: float = 5.0e-5        # s per decode sequence
    dp: float = 2.6e-6       # s per k-token^2 of prefill attention
    dd: float = 3.8e-5       # s per k-token of decode context
    e: float = 0.0           # s per *cached* prompt token when a request starts (sensitivity term; 0 = free hits)
    chunk: int = 8192        # max_num_batched_tokens (prefill token budget per iteration)
    max_seqs: int = 256      # max_num_seqs
    source: str = "roofline-estimate"

    @classmethod
    def load(cls, path):
        if path and os.path.exists(path):
            with open(path) as fh:
                d = json.load(fh)
            keep = {k: d[k] for k in ("a", "b", "c", "dp", "dd", "e", "chunk", "max_seqs", "source") if k in d}
            return cls(**keep)
        return cls()

    def to_dict(self):
        return asdict(self)

    def iter_time(self, P, D, Ap, Ad=0.0):
        return self.a + self.b * P + self.c * D + self.dp * Ap + self.dd * Ad

    def prefill_time(self, n_new, n_cached=0):
        """Isolated TTFT of one request (no other load): chunked prefill of n_new tokens after n_cached."""
        if self.e:
            return self._prefill_time(n_new, n_cached) + self.e * n_cached
        return self._prefill_time(n_new, n_cached)

    def _prefill_time(self, n_new, n_cached=0):
        if n_new <= 0:
            return self.iter_time(1, 0, (n_cached + 1) / 1e3)
        t, ctx, left = 0.0, n_cached, n_new
        while left > 0:
            ch = min(self.chunk, left)
            t += self.iter_time(ch, 0, ch * (ctx + ch / 2) / 1e3)
            ctx += ch
            left -= ch
        return t

    def decode_iter(self, batch, ctx):
        return self.iter_time(0, batch, 0.0, batch * ctx / 1e3)

    def nominal_demand(self, prompt, out, batch=16):
        """Service demand (s) of one request with no reuse: isolated prefill + amortised decode at `batch`."""
        return self.prefill_time(prompt, 0) + out * self.decode_iter(batch, prompt + out / 2) / batch


def blocks(n_tokens, B):
    return int(math.ceil(n_tokens / B))
