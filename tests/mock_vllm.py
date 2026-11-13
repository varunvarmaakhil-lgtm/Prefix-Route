"""Minimal stand-in for a vLLM OpenAI server, for testing bench.py / fit.py without a GPU.

Serves /health, /metrics, /reset_prefix_cache and streaming /v1/completions. It runs requests through an
iteration loop with the same CostModel form the simulator uses (known ground-truth coefficients) and a
block-level prefix cache, so fit.py should recover the coefficients approximately.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from aiohttp import web

GT = {"a": 0.010, "b": 5.0e-5, "c": 2.0e-4, "dp": 3.0e-6, "dd": 4.0e-5}
CHUNK = 8192
BLOCK = 16


class Engine:
    def __init__(self, caching=True, cap_blocks=40000):
        self.caching, self.cap = caching, cap_blocks
        self.cache = {}              # block key -> last use
        self.tick = 0
        self.reqs = []
        self.wake = asyncio.Event()
        self.hits = self.queries = 0

    def keys(self, toks):
        out, h = [], 0
        for i in range(len(toks) // BLOCK):
            h = hash((h, tuple(toks[i * BLOCK:(i + 1) * BLOCK])))
            out.append(h)
        return out

    async def loop(self):
        while True:
            if not self.reqs:
                self.wake.clear()
                await self.wake.wait()
            dec = [r for r in self.reqs if r["pf_left"] == 0]
            pre = [r for r in self.reqs if r["pf_left"] > 0]
            budget = CHUNK - len(dec)
            P, A, Ad = 0, 0.0, sum(r["ctx"] for r in dec) / 1e3
            plan = []
            for r in pre:
                if budget <= 0:
                    break
                ch = min(budget, r["pf_left"])
                A += ch * (r["ctx"] + ch / 2) / 1e3
                plan.append((r, ch))
                P += ch
                budget -= ch
            dt = GT["a"] + GT["b"] * P + GT["c"] * len(dec) + GT["dp"] * A + GT["dd"] * Ad
            await asyncio.sleep(dt)
            for r in dec:
                r["ctx"] += 1
                r["q"].put_nowait("tok")
                r["left"] -= 1
            for r, ch in plan:
                r["pf_left"] -= ch
                r["ctx"] += ch
                if r["pf_left"] == 0:
                    r["q"].put_nowait("tok")
                    r["left"] -= 1
                    self.tick += 1
                    for k in r["keys"]:
                        self.cache[k] = self.tick
                    while len(self.cache) > self.cap:
                        del self.cache[min(self.cache, key=self.cache.get)]
            for r in [r for r in self.reqs if r["left"] <= 0]:
                self.reqs.remove(r)
                r["q"].put_nowait(None)

    def submit(self, toks, n_out):
        ks = self.keys(toks)
        m = 0
        if self.caching:
            for k in ks:
                if k in self.cache:
                    m += 1
                else:
                    break
        cached = min(m * BLOCK, len(toks) - 1) if self.caching else 0
        self.hits += cached
        self.queries += len(toks)
        q = asyncio.Queue()
        self.reqs.append({"pf_left": len(toks) - cached, "ctx": cached, "left": n_out, "q": q, "keys": ks})
        self.wake.set()
        return q, cached


def make_app(caching, cap):
    eng = Engine(caching, cap)
    app = web.Application()

    async def start(app_):
        app_["loop"] = asyncio.create_task(eng.loop())

    app.on_startup.append(start)

    async def health(req):
        return web.Response(text="ok")

    async def metrics(req):
        usage = len(eng.cache) / eng.cap
        return web.Response(text=f"vllm:gpu_cache_usage_perc {usage}\nvllm:gpu_prefix_cache_hits_total {eng.hits}\n"
                                 f"vllm:gpu_prefix_cache_queries_total {eng.queries}\n")

    async def reset(req):
        eng.cache.clear()
        return web.Response(text="ok")

    async def completions(req):
        body = await req.json()
        p = body["prompt"]
        toks = p if isinstance(p, list) else [hash(w) % 30000 for w in p.split()]
        n_out = int(body.get("max_tokens", 16))
        q, cached = eng.submit(toks, n_out)
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        await resp.prepare(req)
        i = 0
        while True:
            x = await q.get()
            if x is None:
                break
            chunk = {"choices": [{"text": f" t{i}", "logprobs": {"tokens": [f" t{i}"]}}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            i += 1
        usage = {"prompt_tokens": len(toks), "completion_tokens": i,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/reset_prefix_cache", reset)
    app.router.add_post("/v1/completions", completions)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--nocache", action="store_true")
    ap.add_argument("--cap", type=int, default=40000)
    a = ap.parse_args()
    web.run_app(make_app(not a.nocache, a.cap), host="127.0.0.1", port=a.port, print=None)
