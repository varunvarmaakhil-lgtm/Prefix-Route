"""One-command reproduction of the PrefixRoute paper (no HPC needed).

    python reproduce.py check                 # dependencies
    python reproduce.py quick                 # ~5 min: self-test + main simulation grid, verified against reference
    python reproduce.py all [--jobs N]        # everything from the shipped measurements (~1-2 h on 8-20 cores)
    python reproduce.py analysis --use-reference   # tables/figures/numbers from the shipped reference cells (~2 min)

Stages (run individually or via `all`, in this order):
  selftest   simulator invariants and pipeline end-to-end test (tests/selftest.py)
  fit        measured TTFT curves + cost model + correctness from the measurement logs (results/measure)
  bias       simulator-vs-engine replay of all measured settings, cached-token cost fit, multi-turn validation
  sim        all simulation cells (configs/experiments_v2.json -> results/raw_v2;
             configs/experiments_sens.json with the cached-token cost -> results/raw_v2_sens)
  analysis   every table, figure and macro of the paper (paper/macros.tex, paper/figs, results/tables)
  paper      LaTeX PDF of the camera-ready paper (needs pdflatex)
  verify     compare regenerated simulation cells and paper numbers with the shipped reference
The GPU measurement itself (results/measure) needs a Linux machine with an NVIDIA GPU: scripts/measure_local.sh.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
REF = os.path.join(ROOT, "reference")


def run(args, **kw):
    print("  $", " ".join(str(a) for a in args), flush=True)
    t = time.time()
    r = subprocess.run([str(a) for a in args], cwd=ROOT, **kw)
    if r.returncode != 0:
        sys.exit(f"step failed ({r.returncode}): {' '.join(map(str, args))}")
    print(f"    ({time.time() - t:.0f}s)", flush=True)


def py(*args, quiet=False):
    kw = {"stdout": subprocess.DEVNULL} if quiet else {}
    run([PY, *args], **kw)


def stage_check(a):
    missing = []
    for mod in ("numpy", "pandas", "pyarrow", "scipy", "joblib", "matplotlib", "aiohttp"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    print("python", sys.version.split()[0], "| missing:", missing or "none")
    print("pdflatex:", shutil.which("pdflatex") or "not found (paper stage will be skipped)")
    if missing:
        sys.exit("install with: pip install -r requirements-cpu.txt")


def stage_selftest(a):
    py("tests/selftest.py")


def stage_fit(a):
    py("-m", "prefixroute.measure.fit", "--dir", a.measure_dir, "--chunk", "8192")
    py("-m", "prefixroute.measure.correctness", "--dir", a.measure_dir, quiet=True)


def stage_bias(a):
    code = ("import sys; sys.path.insert(0, 'analysis'); import sim_bias as s; s.main('costmodel.json', '')")
    py("-c", code, quiet=True)
    py("analysis/fit_cached_cost.py")
    code = ("import sys; sys.path.insert(0, 'analysis'); import sim_bias as s; "
            "s.main('costmodel_cachedcost.json', '_cachedcost')")
    py("-c", code, quiet=True)
    py("analysis/validate_sim.py", quiet=True)


def stage_sim(a):
    jobs = str(a.jobs)
    groups = ["--groups", *a.groups] if a.groups else []
    py("-m", "prefixroute.run_sim", "--exp", "configs/experiments_v2.json", "--jobs", jobs, *groups,
       "--cost", os.path.join(a.measure_dir, "costmodel.json"), "--out", "results/raw_v2", quiet=True)
    if not a.groups:
        py("-m", "prefixroute.run_sim", "--exp", "configs/experiments_sens.json", "--jobs", jobs,
           "--cost", os.path.join(a.measure_dir, "costmodel_cachedcost.json"), "--out", "results/raw_v2_sens",
           quiet=True)


def stage_analysis(a):
    if a.use_reference:
        for d in ("raw_v2", "raw_v2_sens"):
            dst = os.path.join(ROOT, "results", d)
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(dst)
            with zipfile.ZipFile(os.path.join(REF, d + ".zip")) as z:
                z.extractall(dst)
        print("  using the shipped reference simulation cells")
    os.makedirs(os.path.join(ROOT, "results", "tables"), exist_ok=True)
    py("analysis/make_tables.py", "--raw", "results/raw_v2", quiet=True)
    py("analysis/paired.py", "results/raw_v2", quiet=True)
    py("analysis/latency.py", "--raw", "results/raw_v2", quiet=True)
    py("analysis/latency.py", "--raw", "results/raw_v2", "--headroom", "1.0", quiet=True)
    if os.path.isdir(os.path.join(ROOT, "results", "raw_v2_sens")):
        py("analysis/latency.py", "--raw", "results/raw_v2_sens", "--tag", "sens", quiet=True)
    py("analysis/mechanism.py", "--raw", "results/raw_v2", quiet=True)
    py("analysis/check_macros.py")


def stage_paper(a):
    if not shutil.which("pdflatex"):
        print("  pdflatex not found: skipping")
        return
    pdir = os.path.join(ROOT, "paper")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=pdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  ->", os.path.join(pdir, "main.pdf"))


def macros(path):
    out = {}
    for m in re.finditer(r"\\csname pr@([A-Za-z0-9]+)\\endcsname\{(.*)\}\s*$", open(path, encoding="utf-8").read(), re.M):
        out[m.group(1)] = m.group(2)
    return out


def stage_verify(a):
    ok = True
    for d in ("raw_v2", "raw_v2_sens"):
        new = sorted(glob.glob(os.path.join(ROOT, "results", d, "*.json")))
        same = diff = 0
        bad = []
        refz = zipfile.ZipFile(os.path.join(REF, d + ".zip"))
        names = set(refz.namelist())
        for p in new:
            name = os.path.basename(p)
            if name not in names:
                continue
            if json.load(open(p))["metrics"] == json.loads(refz.read(name))["metrics"]:
                same += 1
            else:
                diff += 1
                bad.append(os.path.basename(p))
        print(f"  {d}: {same} cells identical to reference, {diff} different" + (f" e.g. {bad[:3]}" if bad else ""))
        ok &= diff == 0
    if getattr(a, "cells_only", False):
        print("REPRODUCED (simulation cells)" if ok else "MISMATCH (see above)")
        return
    newm = macros(os.path.join(ROOT, "paper", "macros.tex"))
    refm = macros(os.path.join(REF, "macros.tex"))
    common = [k for k in refm if k in newm]
    differ = [k for k in common if newm[k] != refm[k]]
    print(f"  paper numbers: {len(common)}/{len(refm)} regenerated, {len(differ)} differ"
          + (f" e.g. {[(k, refm[k], newm[k]) for k in differ[:5]]}" if differ else ""))
    ok &= not differ
    print("REPRODUCED" if ok else "MISMATCH (see above)")


STAGES = {"check": stage_check, "selftest": stage_selftest, "fit": stage_fit, "bias": stage_bias,
          "sim": stage_sim, "analysis": stage_analysis, "paper": stage_paper,
          "verify": stage_verify}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "all", "quick"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--groups", nargs="*", help="sim: only these experiment groups (see configs/experiments_v2.json)")
    ap.add_argument("--measure-dir", default="results/measure")
    ap.add_argument("--use-reference", action="store_true", help="analysis: use the shipped reference cells")
    a = ap.parse_args()
    if a.stage == "all":
        order = ["check", "selftest", "fit", "bias", "sim", "analysis", "paper", "verify"]
    elif a.stage == "quick":
        a.groups = ["main"]
        a.cells_only = True
        order = ["check", "selftest", "fit", "bias", "sim", "verify"]
    else:
        order = [a.stage]
    for d in ("results/tables", "results/logs", "paper/figs"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    for s in order:
        print(f"== {s}", flush=True)
        STAGES[s](a)


if __name__ == "__main__":
    main()
