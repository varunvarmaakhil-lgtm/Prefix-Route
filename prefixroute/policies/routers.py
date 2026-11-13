"""Request routing policies. A policy sees only what a real router sees (the request's prompt tokens, its
session id, dispatch/first-token/completion events) unless it is marked as the clairvoyant reference.

Interface: reset(cluster); route(req, cluster) -> replica; on_dispatch / on_first_token / on_complete hooks.
"""
from __future__ import annotations

import bisect
import math
from collections import OrderedDict

import numpy as np

M64 = (1 << 64) - 1


def mix64(x):
    """splitmix64 finaliser: deterministic, well-spread 64-bit hash of an int."""
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


class Policy:
    name = "base"

    def __init__(self, seed=0, **kw):
        self.seed = seed
        self.kw = kw

    def reset(self, cl):
        self.rng = np.random.default_rng(self.seed + 7919)

    def route(self, r, cl):
        raise NotImplementedError

    def on_dispatch(self, r, i, cl):
        pass

    def on_first_token(self, r, i, cl):
        pass

    def on_complete(self, r, i, cl):
        pass

    def least_loaded(self, cl, cands=None):
        cands = range(cl.n) if cands is None else cands
        return min(cands, key=lambda j: (cl.outstanding[j], j))


class RoundRobin(Policy):
    name = "round_robin"

    def reset(self, cl):
        super().reset(cl)
        self.k = 0

    def route(self, r, cl):
        self.k += 1
        return (self.k - 1) % cl.n


class Random(Policy):
    name = "random"

    def route(self, r, cl):
        return int(self.rng.integers(cl.n))


class LeastLoaded(Policy):
    name = "least_loaded"

    def route(self, r, cl):
        return self.least_loaded(cl)


class SessionAffinity(Policy):
    """Sticky by session id (hash), as in session-id routing of production routers."""
    name = "session_affinity"

    def route(self, r, cl):
        return mix64(r.sess.sid + 1) % cl.n


class ConsistentRing:
    def __init__(self, n, vnodes=64):
        pts = sorted((mix64((j << 16) + v), j) for j in range(n) for v in range(vnodes))
        self.keys = [p for p, _ in pts]
        self.owner = [j for _, j in pts]

    def lookup(self, h):
        k = bisect.bisect(self.keys, h) % len(self.keys)
        return self.owner[k]


class PrefixHash(Policy):
    """Consistent hashing on the leading `prefix_tokens` of the prompt (block granularity)."""
    name = "prefix_hash"

    def reset(self, cl):
        super().reset(cl)
        self.ring = ConsistentRing(cl.n)
        self.nb = max(1, self.kw.get("prefix_tokens", 256) // cl.B)

    def route(self, r, cl):
        h = 0x12345
        for b in r.ids[:self.nb]:
            h = mix64(h ^ b)
        if not r.ids:
            h = mix64(r.sess.sid)
        return self.ring.lookup(h)


class CacheAware(Policy):
    """Longest-matching-prefix routing with load-balance fallback, modelled on the SGLang router:
    the router keeps a per-replica index of prompts it has sent (it does not observe engine evictions).
    If load is imbalanced (max-min > abs and max > rel*min) route to the least loaded replica; otherwise
    route to the longest match if its match ratio exceeds `threshold`, else to the replica with the
    smallest index (least cached state)."""
    name = "cache_aware"

    def reset(self, cl):
        super().reset(cl)
        self.tree = [set() for _ in range(cl.n)]
        self.th = self.kw.get("threshold", 0.5)
        self.babs = self.kw.get("balance_abs", 32)
        self.brel = self.kw.get("balance_rel", 1.5)

    def _match(self, j, ids):
        s = self.tree[j]
        n = 0
        for b in ids:
            if b in s:
                n += 1
            else:
                break
        return n

    def route(self, r, cl):
        loads = cl.outstanding
        mx, mn = max(loads), min(loads)
        if mx - mn > self.babs and mx > self.brel * mn:
            return self.least_loaded(cl)
        ms = [self._match(j, r.ids) for j in range(cl.n)]
        best = max(range(cl.n), key=lambda j: (ms[j], -loads[j], -j))
        if r.ids and ms[best] / len(r.ids) > self.th:
            return best
        return min(range(cl.n), key=lambda j: (len(self.tree[j]), loads[j], j))

    def on_dispatch(self, r, i, cl):
        self.tree[i].update(r.ids)


class _Estimator(Policy):
    """TTFT estimate for placing r on replica j: FCFS prefill of the replica's pending uncached tokens plus
    r's own uncached tokens, under the calibrated per-iteration cost model."""

    def est_ttft(self, cl, pend_tokens, n_dec, ctx_dec, n_new, n_cached):
        c = cl.cost
        tot = pend_tokens + n_new
        n_it = max(1, math.ceil(tot / c.chunk))
        return (n_it * (c.a + c.c * n_dec + c.dd * ctx_dec / 1e3) + c.b * tot
                + c.dp * n_new * (n_cached + n_new / 2) / 1e3)


class EvictionAware(_Estimator):
    """Cache-aware routing that (i) mirrors each replica's LRU cache with a bounded shadow cache of the
    known KV capacity, so matches that have probably been evicted are discounted, and (ii) trades cache
    reuse against queueing with an explicit TTFT estimate. Uses only router-observable information."""
    name = "evict_aware"

    def reset(self, cl):
        super().reset(cl)
        self.cap = cl.reps[0].cap
        self.shadow = [OrderedDict() for _ in range(cl.n)]
        self.pend = [0] * cl.n
        self.dec = [0] * cl.n
        self.ctx = [0] * cl.n
        self.est_new = {}

    def _match(self, j, ids):
        s = self.shadow[j]
        n = 0
        for b in ids:
            if b in s:
                n += 1
            else:
                break
        return n

    def route(self, r, cl):
        B = cl.B
        best, bc = 0, float("inf")
        for j in range(cl.n):
            m = min(self._match(j, r.ids) * B, r.prompt - 1)
            t = self.est_ttft(cl, self.pend[j], self.dec[j], self.ctx[j], r.prompt - m, m)
            if t < bc - 1e-12:
                best, bc = j, t
        return best

    def _touch(self, j, ids):
        s = self.shadow[j]
        for b in ids:
            if b in s:
                s.move_to_end(b)
            else:
                s[b] = None
        while len(s) > self.cap:
            s.popitem(last=False)

    def on_dispatch(self, r, i, cl):
        m = min(self._match(i, r.ids) * cl.B, r.prompt - 1)
        self.est_new[id(r)] = r.prompt - m
        self.pend[i] += r.prompt - m
        self._touch(i, r.ids)

    def on_first_token(self, r, i, cl):
        self.pend[i] -= self.est_new.pop(id(r), 0)
        self.dec[i] += 1
        self.ctx[i] += r.prompt + r.out // 2

    def on_complete(self, r, i, cl):
        from ..sim.engine import block_ids
        if id(r) in self.est_new:       # completed without a separate first-token event
            self.on_first_token(r, i, cl)
        self.dec[i] -= 1
        self.ctx[i] -= r.prompt + r.out // 2
        self._touch(i, block_ids(r.sess, r.prompt + r.out, cl.B, r.turn))


class Clairvoyant(_Estimator):
    """Reference policy with perfect knowledge of every replica's true cache contents, decode state and
    memory headroom (not observable by a real router); places each request where its TTFT estimate is
    smallest. It is a greedy per-request reference, not a proven optimum."""
    name = "oracle"

    def reset(self, cl):
        super().reset(cl)
        self.pend = [0] * cl.n
        self.est_new = {}

    def route(self, r, cl):
        B = cl.B
        best, bc = 0, float("inf")
        for j, rep in enumerate(cl.reps):
            m = min(rep.match(r.ids) * B, r.prompt - 1)
            dec = [q for q in rep.running if q.decoding]
            ctx = sum(q.prompt + q.out - q.dec_left for q in dec)
            t = self.est_ttft(cl, self.pend[j], len(dec), ctx, r.prompt - m, m)
            need = math.ceil((r.prompt + r.out) / B) - m
            if dec and need > rep.cap - rep.used() + len(rep.lru):
                # cannot be admitted now: waits at least until the earliest decode finishes
                t += min(q.dec_left for q in dec) * cl.cost.decode_iter(len(dec), ctx / len(dec))
            if t < bc - 1e-12:
                best, bc = j, t
        return best

    def on_dispatch(self, r, i, cl):
        m = min(cl.reps[i].match(r.ids) * cl.B, r.prompt - 1)
        self.est_new[id(r)] = r.prompt - m
        self.pend[i] += r.prompt - m

    def on_first_token(self, r, i, cl):
        self.pend[i] -= self.est_new.pop(id(r), 0)


POLICIES = {p.name: p for p in (RoundRobin, Random, LeastLoaded, SessionAffinity, PrefixHash, CacheAware,
                                EvictionAware, Clairvoyant)}
