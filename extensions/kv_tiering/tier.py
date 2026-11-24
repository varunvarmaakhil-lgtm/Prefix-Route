"""A second, larger and slower KV-cache tier for PrefixRoute (density vs. latency for agentic LLM serving).

Each replica keeps its GPU (HBM) prefix cache as in the paper. Blocks evicted from HBM are demoted into a tier-2 LRU
of `cap2` blocks (host DRAM over PCIe, CXL memory, SSD, ...); blocks evicted from tier 2 are dropped. A request whose
prefix is found in tier 2 promotes those blocks back into HBM and pays a transfer cost of `fetch_s_per_token` seconds
per promoted token, charged synchronously to the iteration that starts its prefill (a conservative model: no overlap
of transfer with compute). Prefix tokens found nowhere are recomputed, as in the paper.

Per-token transfer cost = KV bytes per token / tier bandwidth. Qwen2.5-7B: 28 layers x 2 (K,V) x 4 KV heads x 128 x
2 bytes = 57,344 B/token.
"""
import math
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "quantile_ttft"))
import shim  # noqa: E402,F401  (repo on sys.path + parquet fallback)

from prefixroute import run_sim as RS  # noqa: E402
from prefixroute.metrics.stats import cell_metrics  # noqa: E402
from prefixroute.policies.routers import POLICIES, EvictionAware  # noqa: E402
from prefixroute.sim.costmodel import CostModel, blocks  # noqa: E402
from prefixroute.sim.engine import Cluster, Replica, Req  # noqa: E402,F401

COST = os.path.join(RS.ROOT, "results", "measure", "costmodel.json")
KV_BYTES_PER_TOKEN = 28 * 2 * 4 * 128 * 2          # 57,344


def fetch_cost(gb_per_s):
    return KV_BYTES_PER_TOKEN / (gb_per_s * 1e9)


class TieredReplica(Replica):
    def __init__(self, idx, cap_blocks, cost, B, cap2, fetch):
        super().__init__(idx, cap_blocks, cost, B, True)
        self.cap2, self.fetch = cap2, fetch
        self.t2 = OrderedDict()
        self.t2_hits_tokens = 0
        self.demotions = 0
        self.fetch_pending = {}          # id(req) -> tier-2 tokens still to be transferred (Req uses __slots__)

    def match(self, ids):
        n = 0
        ref, lru, t2 = self.ref, self.lru, self.t2
        for b in ids:
            if b in ref or b in lru or b in t2:
                n += 1
            else:
                break
        return n

    def _demote(self, b):
        if self.cap2 <= 0:
            return
        self.t2[b] = None
        self.t2.move_to_end(b)
        self.demotions += 1
        while len(self.t2) > self.cap2:
            self.t2.popitem(last=False)

    def try_admit(self, r, now):
        m = self.match(r.ids)
        pre = r.ids[:m]
        in_t2 = [b for b in pre if b in self.t2]
        m2 = len(in_t2)
        need = int(math.ceil((r.prompt + r.out) / self.B)) - m
        in_lru = sum(1 for b in pre if b in self.lru)
        if need + m2 > self.cap - self.used() + len(self.lru) - in_lru:
            return False
        for b in pre:
            if b in self.t2:
                del self.t2[b]
                self.ref[b] = self.ref.get(b, 0) + 1      # promoted into HBM, pinned
            else:
                self._pin(b)
        short = need - (self.cap - self.used())
        while short > 0:
            b, _ = self.lru.popitem(last=False)
            self.evictions += 1
            self._demote(b)
            short -= 1
        self.private += need
        r.priv = need
        r.pinned = list(pre)
        r.cached = min(m * self.B, r.prompt - 1)
        ft = min(m2 * self.B, r.cached)
        if ft:
            self.fetch_pending[id(r)] = ft
        self.t2_hits_tokens += ft
        r.pf_done = r.cached
        r.admit_t = now
        self.prompt_tokens += r.prompt
        self.cached_tokens += r.cached
        return True


class TieredCluster(Cluster):
    def __init__(self, n, cap_blocks, cost, policy, B, cap2, fetch):
        super().__init__(n, cap_blocks, cost, policy, B=B)
        self.reps = [TieredReplica(i, cap_blocks, cost, B, cap2, fetch) for i in range(n)]

    def _start_iteration(self, rep):
        """Same as Cluster._start_iteration, plus the tier-2 transfer time of requests starting prefill."""
        cost = self.cost
        while rep.waiting and len(rep.running) < cost.max_seqs:
            r = rep.waiting[0]
            if not rep.try_admit(r, self.now):
                if not rep.running:
                    raise RuntimeError("request cannot fit replica cache")
                break
            rep.waiting.popleft()
            rep.running.append(r)
        if not rep.running:
            rep.busy_until = None
            rep.ff = None
            return
        dec = [r for r in rep.running if r.decoding]
        pre = [r for r in rep.running if not r.decoding]
        D = len(dec)
        Ad = sum(r.prompt + r.out - r.dec_left for r in dec) / 1e3
        rep.epoch += 1
        if not pre and not rep.waiting:
            n_it = min(r.dec_left for r in dec)
            dt = cost.iter_time(0, D, 0.0, Ad + D * n_it / 2e3)
            rep.ff = (self.now, n_it, dt)
            rep.busy_until = self.now + n_it * dt
            rep.busy_time += n_it * dt
            rep.plan = (dec, [])
            self.push(rep.busy_until, 1, rep.idx, rep.epoch)
            return
        budget = cost.chunk - D
        P, Ap, E, F = 0, 0.0, 0, 0
        plan = []
        for r in pre:
            if budget <= 0:
                break
            ch = min(budget, r.prompt - r.pf_done)
            Ap += ch * (r.pf_done + ch / 2) / 1e3
            if r.pf_done == r.cached:
                E += r.cached
                F += rep.fetch_pending.pop(id(r), 0)   # charged once
            plan.append((r, ch))
            P += ch
            budget -= ch
        dt = cost.iter_time(P, D, Ap, Ad) + cost.e * E + rep.fetch * F
        rep.ff = None
        rep.busy_until = self.now + dt
        rep.busy_time += dt
        rep.plan = (dec, plan)
        self.push(rep.busy_until, 1, rep.idx, rep.epoch)


class TierAwareEvict(EvictionAware):
    """The paper's eviction-aware router with its shadow cache sized to HBM + tier 2."""
    name = "evict_aware_tier"

    def reset(self, cl):
        super().reset(cl)
        self.cap = cl.reps[0].cap + cl.reps[0].cap2


POLICIES[TierAwareEvict.name] = TierAwareEvict

BASE = {"N": 4, "cap_frac": 0.25, "util": 0.7, "B": 32, "headroom": 2.0, "slo_s": 1.0, "think_scale": 1.0,
        "growth_scale": 1.0, "turns_scale": 1.0, "workload": "agentic", "n_sessions": 5000,
        "policy": "evict_aware_tier"}


def run(cell):
    """cell: BASE overrides + seed, k (tier-2 size as a multiple of the HBM cache share), gbps (tier bandwidth)."""
    c = dict(BASE)
    c.update(cell)
    cost = CostModel.load(COST)
    B = c["B"]
    ss = RS.build_sessions(c, cost)
    ws, fly = RS.working_set_blocks(ss, cost, B)
    max_req = max(blocks(p + o, B) for s in ss for (p, o, _) in s.turns)
    head = int(c["headroom"] * fly / c["N"]) + max_req
    cache = int(c["cap_frac"] * ws / c["N"])
    cap2 = int(c["k"] * cache)
    fetch = fetch_cost(c["gbps"]) if c["gbps"] else 0.0
    pol = POLICIES[c["policy"]](seed=c["seed"])
    cl = TieredCluster(c["N"], head + cache, cost, pol, B, cap2, fetch)
    done = cl.run(ss)
    m = cell_metrics(done, cl, c["slo_s"])
    t2 = sum(rp.t2_hits_tokens for rp in cl.reps)
    pt = sum(rp.prompt_tokens for rp in cl.reps)
    out = {k: c.get(k) for k in ("seed", "k", "gbps", "cap_frac", "util", "tool_median_s", "policy")}
    out.update(fetch_us_per_token=fetch * 1e6, t2_hit_rate=t2 / pt if pt else 0.0,
               **{k: m[k] for k in ("ttft_p50", "ttft_p95", "ttft_p99", "ttft_mean", "e2e_p95", "e2e_mean",
                                    "session_time_mean", "thpt_req_s", "hit_rate", "evictions", "prefill_tokens")})
    return out
