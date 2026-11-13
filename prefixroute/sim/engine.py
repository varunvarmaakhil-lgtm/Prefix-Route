"""Discrete-event simulator of N serving replicas behind a router.

Each replica models a vLLM-V1-like engine:
  * FCFS waiting queue; admission reserves KV blocks for prompt + output (no preemption); head-of-line
    blocking when the KV cache cannot fit the head request;
  * automatic prefix caching at block granularity: the longest run of leading full blocks already resident
    (pinned by running requests or evictable) is reused; unreferenced blocks are evicted LRU;
  * iterations with chunked prefill: all decoding sequences take one token, the remaining token budget goes
    to prefill chunks in FCFS order; iteration duration from the calibrated CostModel;
  * decode-only stretches are fast-forwarded and cut short when a new request arrives.
Block ids are derived from a request's lineage, so shared prefixes map to identical ids.
"""
from __future__ import annotations

import heapq
import math
from collections import OrderedDict, deque

from .costmodel import CostModel

TPL_BASE = 1 << 40
SESS_BASE = 1 << 62


def block_ids(sess, n_tokens, B, turn=0):
    """Ids of the full blocks covering the first n_tokens of the session lineage. Sessions with
    fresh_per_turn (benchmark replay: independent requests sharing only the template) get a new lineage per turn."""
    nb = n_tokens // B
    k = min(nb, sess.sys_tokens // B)      # blocks entirely inside the shared template prefix
    tb = (sess.tpl + 1) * TPL_BASE
    lin = sess.sid * 4096 + turn if getattr(sess, "fresh_per_turn", False) else sess.sid
    sb = SESS_BASE + (lin << 20)
    return [tb + i for i in range(k)] + [sb + i for i in range(k, nb)]


class Req:
    __slots__ = ("sess", "turn", "prompt", "out", "think", "arrival", "replica", "ids", "cached",
                 "pf_done", "dec_left", "t_first", "t_done", "priv", "pinned", "admit_t", "decoding")

    def __init__(self, sess, turn, t, B):
        p, o, th = sess.turns[turn]
        self.sess, self.turn, self.prompt, self.out, self.think = sess, turn, p, o, th
        self.arrival = t
        self.ids = block_ids(sess, p, B, turn)
        self.cached = 0
        self.pf_done = 0
        self.dec_left = o
        self.t_first = self.t_done = self.admit_t = None
        self.replica = -1
        self.priv = 0
        self.pinned = []
        self.decoding = False


class Replica:
    def __init__(self, idx, cap_blocks, cost: CostModel, B, caching=True):
        self.idx, self.cap, self.cost, self.B, self.caching = idx, cap_blocks, cost, B, caching
        self.waiting = deque()
        self.running = []            # admitted, prefilling or decoding
        self.ref = {}                # block id -> refcount (resident, pinned by running requests)
        self.lru = OrderedDict()     # block id -> None (resident, evictable), least recently used first
        self.private = 0             # reserved blocks that are not shared cache entries
        self.busy_until = None
        self.epoch = 0
        self.ff = None               # decode fast-forward: (t_start, n_iters, dt)
        self.plan = ([], [])
        self.n_done = 0
        self.prefill_tokens = 0
        self.cached_tokens = 0
        self.prompt_tokens = 0
        self.evictions = 0
        self.busy_time = 0.0

    def used(self):
        return len(self.ref) + len(self.lru) + self.private

    def resident(self, b):
        return b in self.ref or b in self.lru

    def match(self, ids):
        if not self.caching:
            return 0
        n = 0
        ref, lru = self.ref, self.lru
        for b in ids:
            if b in ref or b in lru:
                n += 1
            else:
                break
        return n

    def _pin(self, b):
        if b in self.lru:
            del self.lru[b]
            self.ref[b] = 1
        else:
            self.ref[b] = self.ref.get(b, 0) + 1

    def try_admit(self, r: Req, now):
        m = self.match(r.ids)
        need = int(math.ceil((r.prompt + r.out) / self.B)) - m
        in_lru = sum(1 for b in r.ids[:m] if b in self.lru)
        if need > self.cap - self.used() + len(self.lru) - in_lru:
            return False
        for b in r.ids[:m]:
            self._pin(b)
        short = need - (self.cap - self.used())
        while short > 0:
            self.lru.popitem(last=False)
            self.evictions += 1
            short -= 1
        self.private += need
        r.priv = need
        r.pinned = list(r.ids[:m])
        r.cached = min(m * self.B, r.prompt - 1)   # a full hit still recomputes the last token
        r.pf_done = r.cached
        r.admit_t = now
        self.prompt_tokens += r.prompt
        self.cached_tokens += r.cached
        return True

    def finish_prefill(self, r: Req):
        """The request's computed full prompt blocks become shared cache entries (pinned by r)."""
        if not self.caching:
            return
        for b in r.ids[len(r.pinned):]:
            self._pin(b)
            r.pinned.append(b)
            self.private -= 1
            r.priv -= 1

    def release(self, r: Req):
        """Request done: prompt + output full blocks stay cached (evictable, most recent); rest is freed."""
        if self.caching and r.sess.append_output:
            full = block_ids(r.sess, r.prompt + r.out, self.B, r.turn)
            for b in full[len(r.ids):]:
                if not self.resident(b):
                    self.lru[b] = None
                    self.private -= 1
                    r.priv -= 1
                else:
                    self.lru.move_to_end(b) if b in self.lru else None
        self.private -= r.priv
        r.priv = 0
        for b in r.pinned:
            c = self.ref[b]
            if c <= 1:
                del self.ref[b]
                self.lru[b] = None
            else:
                self.ref[b] = c - 1
        r.pinned = []


class Cluster:
    def __init__(self, n, cap_blocks, cost: CostModel, policy, B=32, caching=True):
        self.cost, self.B = cost, B
        self.reps = [Replica(i, cap_blocks, cost, B, caching) for i in range(n)]
        self.n = n
        self.policy = policy
        self.events = []
        self.seq = 0
        self.now = 0.0
        self.done = []
        self.outstanding = [0] * n
        self.n_arrived = 0

    def push(self, t, kind, obj, epoch=0):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, obj, epoch))

    def run(self, sessions):
        self.policy.reset(self)
        for s in sessions:
            self.push(s.arrival, 0, (s, 0))
        while self.events:
            t, _, kind, obj, ep = heapq.heappop(self.events)
            self.now = t
            if kind == 0:
                s, turn = obj
                r = Req(s, turn, t, self.B)
                self.n_arrived += 1
                i = self.policy.route(r, self)
                r.replica = i
                self.outstanding[i] += 1
                self.policy.on_dispatch(r, i, self)
                rep = self.reps[i]
                rep.waiting.append(r)
                self._kick(rep)
            else:
                rep = self.reps[obj]
                if ep == rep.epoch:
                    self._iteration_end(rep)
        return self.done

    def _kick(self, rep):
        if rep.busy_until is None:
            self._start_iteration(rep)
        elif rep.ff is not None:
            t0, n_it, dt = rep.ff
            k = int((self.now - t0) / dt) + 1        # iterations completed at the next boundary
            if k < n_it:
                rep.busy_time -= (n_it - k) * dt
                rep.ff = (t0, k, dt)
                rep.busy_until = t0 + k * dt
                rep.epoch += 1
                self.push(rep.busy_until, 1, rep.idx, rep.epoch)

    def _start_iteration(self, rep: Replica):
        cost = self.cost
        while rep.waiting and len(rep.running) < cost.max_seqs:
            r = rep.waiting[0]
            if not rep.try_admit(r, self.now):
                if not rep.running:
                    raise RuntimeError(f"request {r.prompt}+{r.out} tokens cannot fit replica cache "
                                       f"({rep.cap} blocks of {self.B})")
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
        P, Ap, E = 0, 0.0, 0
        plan = []
        for r in pre:
            if budget <= 0:
                break
            ch = min(budget, r.prompt - r.pf_done)
            Ap += ch * (r.pf_done + ch / 2) / 1e3
            if r.pf_done == r.cached:          # first chunk: the cached prefix is loaded/attached now
                E += r.cached
            plan.append((r, ch))
            P += ch
            budget -= ch
        dt = cost.iter_time(P, D, Ap, Ad) + cost.e * E
        rep.ff = None
        rep.busy_until = self.now + dt
        rep.busy_time += dt
        rep.plan = (dec, plan)
        self.push(rep.busy_until, 1, rep.idx, rep.epoch)

    def _iteration_end(self, rep: Replica):
        dec, plan = rep.plan
        n_it = rep.ff[1] if rep.ff is not None else 1
        now = self.now
        finished = []
        for r in dec:
            r.dec_left -= n_it
            if r.dec_left <= 0:
                finished.append(r)
        for r, ch in plan:
            r.pf_done += ch
            rep.prefill_tokens += ch
            if r.pf_done >= r.prompt:
                r.t_first = now
                r.decoding = True
                self.policy.on_first_token(r, rep.idx, self)
                r.dec_left -= 1                   # the prefill iteration emits the first token
                rep.finish_prefill(r)
                if r.dec_left <= 0:
                    finished.append(r)
        for r in finished:
            r.t_done = now
            rep.running.remove(r)
            rep.release(r)
            rep.n_done += 1
            self.outstanding[rep.idx] -= 1
            self.done.append(r)
            self.policy.on_complete(r, rep.idx, self)
            s = r.sess
            if r.turn + 1 < len(s.turns):
                self.push(now + r.think, 0, (s, r.turn + 1))
        rep.ff = None
        self._start_iteration(rep)
