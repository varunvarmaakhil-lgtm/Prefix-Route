"""Unified replayable workload: a list of sessions, each a sequence of turns issued closed-loop.

A session has a template (shared system prompt of `sys_tokens` tokens, identical across all sessions of the
template) and a private token lineage. Turn t's prompt is the first `prompt` tokens of the lineage; the model's
output is appended to the lineage, so for agentic sessions each prompt is a strict prefix-extension of the
previous one (growing prefix). A chat request is a one-turn session: template prefix + unique content.

Turn t+1 is issued `think` seconds after turn t completes (tool latency for agents).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_PROMPT = 16384
MAX_OUT = 2048


@dataclass
class Session:
    sid: int
    tpl: int
    sys_tokens: int
    arrival: float
    turns: list = field(default_factory=list)   # [(prompt_tokens, output_tokens, think_s_after)]
    append_output: bool = True                   # False: next prompt continues after the prompt, not the output
    fresh_per_turn: bool = False                 # True: turns are independent requests sharing only the template


# ---------------------------------------------------------------- arrival processes
def arrivals(n, rate, rng, cv=1.0):
    """n arrival times with mean rate `rate`; cv=1 Poisson, cv>1 bursty (gamma inter-arrivals)."""
    if cv == 1.0:
        gaps = rng.exponential(1.0 / rate, n)
    else:
        k = 1.0 / cv ** 2
        gaps = rng.gamma(k, 1.0 / (rate * k), n)
    return np.cumsum(gaps) - gaps[0]


# ---------------------------------------------------------------- agentic (companion trace descriptor)
class AgenticTrace:
    QUESTION_TOKENS = 80   # per-session part of the first prompt (question); the rest is the template prefix

    def __init__(self, path):
        df = pd.read_parquet(path)
        self.df = df
        first = df[df.turn == 0]
        self.sys_tokens = (first.groupby("template").prompt_tokens.median() - self.QUESTION_TOKENS
                           ).clip(lower=0).round().astype(int).to_dict()
        self.groups = {s: g.sort_values("turn") for s, g in df.groupby("session")}
        self.ids = np.array(sorted(self.groups))

    def sample(self, n, rng):
        out = []
        for i, s in enumerate(rng.choice(self.ids, n, replace=True)):
            g = self.groups[s]
            tpl = int(g.template.iloc[0])
            turns = [(int(min(p, MAX_PROMPT)), int(max(1, min(o, MAX_OUT))), float(tm) / 1000.0)
                     for p, o, tm in zip(g.prompt_tokens, g.output_tokens, g.tool_ms)]
            out.append(Session(i, tpl, int(self.sys_tokens[tpl]), 0.0, turns))
        return out

    def fit_synth(self):
        """Distribution parameters for the synthetic generator, fitted to this trace."""
        df = self.df
        first = df[df.turn == 0]
        nt = df.groupby("session").n_turns.first().values
        grow = []
        for g in self.groups.values():
            p = g.prompt_tokens.values
            o = g.output_tokens.values
            if len(p) > 1:
                grow.extend(p[1:] - p[:-1] - o[:-1])        # tool + template overhead tokens per turn
        grow = np.maximum(np.array(grow, dtype=float), 1.0)
        out = np.maximum(df.output_tokens.values.astype(float), 1.0)
        think = np.maximum(df[df.turn < df.n_turns - 1].tool_ms.values / 1000.0, 1e-3)
        ln = lambda x: (float(np.mean(np.log(x))), float(np.std(np.log(x))))
        return {"turns_mean": float(nt.mean()), "turns_p": float(1.0 / nt.mean()),
                "tool_ln": ln(grow), "out_ln": ln(out), "think_ln": ln(think),
                "n_templates": int(first.template.nunique()),
                "sys_tokens": [int(v) for v in self.sys_tokens.values()],
                "question_tokens": self.QUESTION_TOKENS}


# ---------------------------------------------------------------- synthetic agentic generator
def synth_sessions(n, rng, p, turns_scale=1.0, growth_scale=1.0, think_scale=1.0):
    """Sessions drawn from fitted distributions; scales sweep regimes the trace does not cover."""
    out = []
    mean_turns = max(1.0, p["turns_mean"] * turns_scale)
    sys_list = p["sys_tokens"]
    for i in range(n):
        T = int(rng.geometric(1.0 / mean_turns))
        tpl = int(rng.integers(len(sys_list)))
        prompt = sys_list[tpl] + p["question_tokens"]
        turns = []
        for t in range(T):
            o = int(np.clip(rng.lognormal(*p["out_ln"]), 1, MAX_OUT))
            think = float(rng.lognormal(*p["think_ln"])) * think_scale
            turns.append((int(min(prompt, MAX_PROMPT)), o, think))
            prompt = prompt + o + int(rng.lognormal(*p["tool_ln"]) * growth_scale)
            if prompt > MAX_PROMPT:
                break
        out.append(Session(i, tpl, sys_list[tpl], 0.0, turns))
    return out


# ---------------------------------------------------------------- chat (Azure LLM inference trace)
class ChatTrace:
    N_TEMPLATES = 16
    ZIPF_S = 1.1
    SYS_MEDIAN = 600      # tokens; system-prompt length per template (lognormal, sigma 0.6)

    def __init__(self, path):
        df = pd.read_csv(path)
        df["t"] = pd.to_datetime(df.TIMESTAMP).astype("int64") / 1e9
        df["t"] -= df.t.min()
        self.df = df.sort_values("t").reset_index(drop=True)

    def sample(self, n, rng):
        df = self.df
        start = int(rng.integers(0, max(1, len(df) - n)))
        w = df.iloc[start:start + n]
        tr = np.random.default_rng(12345)       # template library is fixed across seeds
        sys_len = np.round(tr.lognormal(np.log(self.SYS_MEDIAN), 0.6, self.N_TEMPLATES)).astype(int)
        pz = 1.0 / np.arange(1, self.N_TEMPLATES + 1) ** self.ZIPF_S
        pz /= pz.sum()
        tpls = rng.choice(self.N_TEMPLATES, len(w), p=pz)
        t0 = float(w.t.iloc[0])
        out = []
        for i, (t, ctx, gen, tpl) in enumerate(zip(w.t, w.ContextTokens, w.GeneratedTokens, tpls)):
            prompt = int(np.clip(ctx, 1, MAX_PROMPT))
            sys_t = int(min(sys_len[tpl], int(0.9 * prompt)))
            out.append(Session(i, int(tpl), sys_t, float(t) - t0,
                               [(prompt, int(np.clip(gen, 1, MAX_OUT)), 0.0)]))
        return out


# ---------------------------------------------------------------- utilisation targeting
def set_arrivals(sessions, cost, n_replicas, util, rng, keep_trace_times=False, cv=1.0):
    """Assign session arrival times so that nominal (no-reuse) utilisation equals `util`."""
    demand = sum(cost.nominal_demand(p, o) for s in sessions for (p, o, _) in s.turns)
    n = len(sessions)
    if keep_trace_times:
        span = max(s.arrival for s in sessions) or 1.0
        target_span = demand / (util * n_replicas)
        scale = target_span / span
        for s in sessions:
            s.arrival *= scale
    else:
        rate = n * util * n_replicas / demand
        for s, a in zip(sessions, arrivals(n, rate, rng, cv)):
            s.arrival = float(a)
    return demand


def load_workload(name, n, rng, data_dir, cost, n_replicas, util, synth_params=None, **kw):
    if name == "chat":
        ss = ChatTrace(os.path.join(data_dir, "azure", "AzureLLMInferenceTrace_conv.csv")).sample(n, rng)
        set_arrivals(ss, cost, n_replicas, util, rng, keep_trace_times=True)
    elif name == "agentic":
        ss = AgenticTrace(os.path.join(data_dir, "agentic_trace.parquet")).sample(n, rng)
        set_arrivals(ss, cost, n_replicas, util, rng, cv=kw.get("cv", 1.0))
    elif name.startswith("synth"):
        ss = synth_sessions(n, rng, synth_params, kw.get("turns_scale", 1.0), kw.get("growth_scale", 1.0),
                            kw.get("think_scale", 1.0))
        set_arrivals(ss, cost, n_replicas, util, rng, cv=kw.get("cv", 1.0))
    else:
        raise ValueError(name)
    return ss
