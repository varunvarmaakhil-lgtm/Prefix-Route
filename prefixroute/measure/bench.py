"""C1 measurement client for a running vLLM OpenAI-compatible server (one A100).

Subcommands (all append to JSONL files under --out, resumable: finished settings are skipped):

  sweep     TTFT/ITL/E2E vs prefix hit fraction x prompt length x concurrency (prompts as token ids,
            controlled shared prefix + unique suffix, greedy, fixed output length with ignore_eos)
  probe     correctness: identical prompts at temperature 0, cold vs warm cache (and, across two server
            runs, caching on vs off) -> compare completions token by token
  agentic   multi-turn growing-prefix replay (sessions with Poisson arrivals, tool gaps), used both for the
            correctness check under concurrent multi-turn load and to validate the simulator on 1 replica

The server is started by cluster/measure.sbatch; the label (--tag) records its configuration.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time

import aiohttp
import numpy as np

HIT_FRACS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
LENGTHS = [1024, 4096, 8192, 16384]
CONCS = [1, 4, 16, 32]
OUT_TOKENS = 64
VOCAB_LO, VOCAB_HI = 1000, 30000   # ordinary BPE tokens, avoids special tokens


def n_requests(c):
    return {1: 8, 4: 16, 16: 48, 32: 64}.get(c, 2 * c)


# ------------------------------------------------------------------ server helpers
class Server:
    def __init__(self, base, model):
        self.base, self.model = base.rstrip("/"), model

    async def wait_ready(self, session, timeout=1800):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                async with session.get(self.base + "/health") as r:
                    if r.status == 200:
                        return True
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(5)
        raise TimeoutError("server not ready")

    async def metrics(self, session):
        try:
            async with session.get(self.base + "/metrics") as r:
                txt = await r.text()
        except aiohttp.ClientError:
            return {}
        out = {}
        for line in txt.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
            if m and any(k in m.group(1) for k in ("cache_usage", "prefix_cache", "num_requests",
                                                    "num_preemptions")):
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        return out

    async def reset_cache(self, session):
        try:
            async with session.post(self.base + "/reset_prefix_cache") as r:
                return r.status == 200
        except aiohttp.ClientError:
            return False

    async def complete(self, session, prompt_ids=None, prompt_text=None, max_tokens=OUT_TOKENS,
                       ignore_eos=True):
        """Streamed greedy completion. Returns timing, token ids, text and engine-reported cached tokens."""
        body = {"model": self.model, "prompt": prompt_ids if prompt_ids is not None else prompt_text,
                "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": ignore_eos,
                "stream_options": {"include_usage": True}, "logprobs": 1}
        t0 = time.perf_counter()
        stamps, texts, toks, usage, err = [], [], [], None, None
        try:
            async with session.post(self.base + "/v1/completions", json=body) as r:
                if r.status != 200:
                    err = f"HTTP {r.status}: {(await r.text())[:300]}"
                else:
                    async for raw in r.content:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        j = json.loads(data)
                        if j.get("usage"):
                            usage = j["usage"]
                        for ch in j.get("choices", []):
                            lp = ch.get("logprobs") or {}
                            n_new = len(lp.get("tokens") or []) or (1 if ch.get("text") else 0)
                            if n_new:
                                now = time.perf_counter()
                                stamps.extend([now] * n_new)
                                texts.append(ch.get("text", ""))
                                toks.extend(lp.get("tokens") or [])
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            err = f"{type(e).__name__}: {e}"
        t_end = time.perf_counter()
        ttft = (stamps[0] - t0) if stamps else None
        gaps = np.diff(sorted(set(stamps))) if len(set(stamps)) > 1 else np.array([])
        cached = None
        if usage and usage.get("prompt_tokens_details"):
            cached = usage["prompt_tokens_details"].get("cached_tokens")
        return {"t_send": t0, "ttft": ttft, "e2e": t_end - t0, "itl_mean": float(gaps.mean()) if len(gaps) else None,
                "itl_p50": float(np.median(gaps)) if len(gaps) else None, "n_out": len(stamps),
                "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": cached,
                "text": "".join(texts), "tokens": toks, "error": err}


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return keys


def append(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------------------------------------------------ sweep
async def sweep(a):
    srv = Server(a.url, a.model)
    path = os.path.join(a.out, "sweep.jsonl")
    done = done_keys(path)
    deadline = time.time() + a.budget_min * 60
    to = aiohttp.ClientTimeout(total=900)
    rng_master = random.Random(a.seed)
    # priority order: the c=1 curves first (the core figure), then higher concurrency
    settings = [(c, L, h) for c in a.concs for L in a.lengths for h in a.hits]
    async with aiohttp.ClientSession(timeout=to, connector=aiohttp.TCPConnector(limit=0)) as s:
        await srv.wait_ready(s)
        for (c, L, h) in settings:
            key = f"{a.tag}|c={c}|L={L}|h={h}"
            if key in done:
                continue
            if time.time() > deadline:
                append(os.path.join(a.out, "skipped.jsonl"), {"key": key, "reason": "budget"})
                continue
            rng = random.Random(f"{a.seed}|{key}")
            n_pre = int(round(h * L))
            prefix = [rng.randint(VOCAB_LO, VOCAB_HI) for _ in range(n_pre)]
            await srv.reset_cache(s)
            if n_pre > 0:   # warm the shared prefix (its own unique tail so it is not a full-prompt match)
                await srv.complete(s, prefix + [rng.randint(VOCAB_LO, VOCAB_HI) for _ in range(8)], max_tokens=1)
            n_total = n_requests(c)
            queue = list(range(n_total))
            m_before = await srv.metrics(s)
            samples = []
            stop = asyncio.Event()

            async def sampler():
                while not stop.is_set():
                    samples.append(await srv.metrics(s))
                    try:
                        await asyncio.wait_for(stop.wait(), 0.5)
                    except asyncio.TimeoutError:
                        pass

            async def worker(w):
                res = []
                while queue:
                    i = queue.pop()
                    suf = [rng.randint(VOCAB_LO, VOCAB_HI) for _ in range(L - n_pre)]
                    res.append(await srv.complete(s, prefix + suf))
                return res

            st = asyncio.create_task(sampler())
            t0 = time.perf_counter()
            results = sum(await asyncio.gather(*[worker(w) for w in range(c)]), [])
            wall = time.perf_counter() - t0
            stop.set()
            await st
            m_after = await srv.metrics(s)
            ok = [r for r in results if r["error"] is None and r["ttft"] is not None]
            usage = [x.get("vllm:gpu_cache_usage_perc", x.get("vllm:kv_cache_usage_perc", np.nan)) for x in samples]
            rec = {"key": key, "tag": a.tag, "c": c, "L": L, "h": h, "n": len(results), "n_ok": len(ok),
                   "wall_s": wall,
                   "ttft": [r["ttft"] for r in ok], "e2e": [r["e2e"] for r in ok],
                   "itl_mean": [r["itl_mean"] for r in ok], "itl_p50": [r["itl_p50"] for r in ok],
                   "cached_tokens": [r["cached_tokens"] for r in ok], "prompt_tokens": [r["prompt_tokens"] for r in ok],
                   "n_out": [r["n_out"] for r in ok],
                   "thpt_tok_s": sum(r["n_out"] for r in ok) / wall if wall > 0 else None,
                   "kv_usage_max": float(np.nanmax(usage)) if usage else None,
                   "kv_usage_mean": float(np.nanmean(usage)) if usage else None,
                   "metrics_before": m_before, "metrics_after": m_after,
                   "errors": [r["error"] for r in results if r["error"]][:5]}
            append(path, rec)
            print(f"{key} n_ok={len(ok)} ttft_p50={np.median(rec['ttft']) if ok else None} wall={wall:.1f}s", flush=True)


# ------------------------------------------------------------------ correctness probes
PROBE_TOPICS = ["a distributed key-value store", "a railway timetable", "photosynthesis", "binary search trees",
                "the water cycle", "TCP congestion control", "a bakery's weekly orders", "compiler register allocation"]


def probe_prompts():
    shared = ("You are a careful technical assistant. Answer precisely and concisely. " * 40).strip()
    out = []
    for i, t in enumerate(PROBE_TOPICS):
        body = f"\n\nTask {i}: Write a short, factual explanation of {t}, in plain prose, as numbered points."
        out.append(shared + body)
    return out


async def probe(a):
    srv = Server(a.url, a.model)
    path = os.path.join(a.out, "probe.jsonl")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        await srv.wait_ready(s)
        await srv.reset_cache(s)
        prompts = probe_prompts()
        for rep in ("cold", "warm", "warm2"):
            for i, p in enumerate(prompts):
                r = await srv.complete(s, prompt_text=p, max_tokens=96, ignore_eos=False)
                append(path, {"key": f"{a.tag}|probe|{rep}|{i}", "tag": a.tag, "phase": rep, "i": i,
                              "tokens": r["tokens"], "text": r["text"], "cached_tokens": r["cached_tokens"],
                              "ttft": r["ttft"], "error": r["error"]})
        # the same probes issued concurrently (batch composition differs from the sequential runs)
        rs = await asyncio.gather(*[srv.complete(s, prompt_text=p, max_tokens=96, ignore_eos=False) for p in prompts])
        for i, r in enumerate(rs):
            append(path, {"key": f"{a.tag}|probe|concurrent|{i}", "tag": a.tag, "phase": "concurrent", "i": i,
                          "tokens": r["tokens"], "text": r["text"], "cached_tokens": r["cached_tokens"],
                          "ttft": r["ttft"], "error": r["error"]})
    print("probe done", a.tag, flush=True)


# ------------------------------------------------------------------ agentic multi-turn replay
TOOL_WORDS = ["status", "ok", "id", "value", "unit", "date", "count", "items", "code", "name", "system", "entry"]


def make_sessions(n, seed, spec):
    """Synthetic multi-turn sessions (text): shared system prompt per template, question, then per turn the
    model's own output + a synthetic JSON-like tool result appended (growing prefix)."""
    rng = random.Random(seed)
    sys_prompts = [("You are an agent that answers questions by calling tools. Template %d. " % k) * 60
                   for k in range(4)]
    out = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(spec["rate"])
        T = min(spec["max_turns"], 1 + int(rng.expovariate(1.0 / (spec["turns_mean"] - 1))))
        tools = []
        for _ in range(T - 1):
            k = max(5, int(rng.lognormvariate(spec["tool_ln"][0], spec["tool_ln"][1]) / 3))
            tools.append("{" + ", ".join(f'"{rng.choice(TOOL_WORDS)}": {rng.randint(0, 9999)}' for _ in range(k)) + "}")
        out.append({"sid": i, "tpl": i % 4, "arrival": t, "system": sys_prompts[i % 4],
                    "question": f"Question {i}: summarise record set {rng.randint(0, 10**6)} and report the count.",
                    "tools": tools, "think_s": [rng.lognormvariate(spec["think_ln"][0], spec["think_ln"][1])
                                                for _ in range(T - 1)],
                    "out_tokens": [max(8, min(256, int(rng.lognormvariate(spec["out_ln"][0], spec["out_ln"][1]))))
                                   for _ in range(T)]})
    return out


async def agentic(a):
    srv = Server(a.url, a.model)
    path = os.path.join(a.out, f"agentic_{a.tag}_r{a.repeat}.jsonl")
    spec = json.load(open(a.spec))
    sessions = make_sessions(a.n_sessions, a.seed, spec)
    with open(os.path.join(a.out, f"agentic_sessions_seed{a.seed}.json"), "w") as fh:
        json.dump(sessions, fh)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        await srv.wait_ready(s)
        await srv.reset_cache(s)
        t_start = time.perf_counter()

        async def run_session(ss):
            await asyncio.sleep(max(0.0, ss["arrival"] - (time.perf_counter() - t_start)))
            ctx = ss["system"] + "\n\n" + ss["question"]
            for turn in range(len(ss["out_tokens"])):
                r = await srv.complete(s, prompt_text=ctx, max_tokens=ss["out_tokens"][turn], ignore_eos=True)
                append(path, {"key": f"{a.tag}|{a.repeat}|{ss['sid']}|{turn}", "tag": a.tag, "repeat": a.repeat,
                              "sid": ss["sid"], "turn": turn, "t_arrive": r["t_send"] - t_start, "ttft": r["ttft"],
                              "e2e": r["e2e"], "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"],
                              "n_out": r["n_out"], "tokens": r["tokens"], "text": r["text"], "error": r["error"]})
                if turn + 1 < len(ss["out_tokens"]):
                    # growth uses the *reference* continuation so that prompts are identical across runs even if
                    # a run's output diverges; the model's own text is recorded for the correctness comparison
                    ctx = ctx + " " + f"[assistant turn {turn}]" + " " + ss["tools"][turn]
                    await asyncio.sleep(ss["think_s"][turn])

        await asyncio.gather(*[run_session(ss) for ss in sessions])
    print("agentic done", a.tag, a.repeat, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "probe", "agentic"])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/measure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=120)
    ap.add_argument("--hits", type=float, nargs="*", default=HIT_FRACS)
    ap.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    ap.add_argument("--concs", type=int, nargs="*", default=CONCS)
    ap.add_argument("--spec", default="configs/agentic_replay.json")
    ap.add_argument("--n-sessions", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    asyncio.run({"sweep": sweep, "probe": probe, "agentic": agentic}[a.cmd](a))


if __name__ == "__main__":
    main()
