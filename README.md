# PrefixRoute: eviction-aware routing for agentic LLM serving

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24190142.svg)](https://doi.org/10.5281/zenodo.24190142)

Code and data for the PCDS 2026 paper

> **Growing Prefixes Meet Finite Caches: When Does Eviction-Aware Routing Pay Off for Agentic LLM Serving?**
> Akhil Varun Varma Kalidindi, Manda Siddharth, Rishi Kamishetty, Bilakanti Mukesh Reddy, Lakshmi Pavan Rangisetti,
> Veeraiah Talagondapati. Dept. of Computer Science and Engineering, École Centrale School of Engineering,
> Mahindra University. PCDS 2026.

Cache-aware routers send LLM requests that share a prompt prefix to the replica that already caches it. Agentic
workloads have prompts that grow with every tool call, and idle sessions waiting on tools hold reusable KV-cache
that may be evicted. This repository contains:

- a **measurement client** for vLLM (TTFT vs. prefix-hit curves, multi-turn agentic replay, correctness check),
  with the raw measurements from one NVIDIA A100 (MIG 4g.20gb, vLLM 0.8.5.post1, Qwen2.5-7B-Instruct-AWQ);
- a calibrated, iteration-level **multi-replica simulator** of the vLLM V1 engine (prefix cache with LRU eviction);
- **eight routing policies**: round-robin, random, least-loaded, session affinity, prefix hash, cache-aware
  (SGLang-router style), eviction-aware, and a clairvoyant reference;
- the **analysis** scripts that produce the tables, figures and numbers, plus the simulation cells behind them.

## Install

Python 3.10–3.13, any OS. No GPU is needed except to re-measure.

```bash
git clone https://github.com/varunvarmaakhil-lgtm/Prefix-Route.git
cd Prefix-Route
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
python reproduce.py check
```

On Windows, clone to a short path (e.g. `C:\prefixroute`): simulation cell files have long names.

## Reproduce the tables and figures

| Command | What it does | Time (8–20 cores) |
|---|---|---|
| `python reproduce.py quick` | Self-test, cost-model fit from the measurement logs, simulator-bias replay, main simulation grid (120 cells), verified against the reference | ~5 min |
| `python reproduce.py analysis --use-reference` | All tables, figures and numbers from the shipped simulation cells | ~2 min |
| `python reproduce.py all` | Everything from the measurement logs: 4,630 simulation cells, analysis, LaTeX paper, verification | ~1–2 h |

The last stage, `verify`, prints `REPRODUCED` when every regenerated simulation cell and number equals the
shipped reference in `reference/`; the simulator is deterministic per seed. Outputs go to `results/tables/`
(CSV/JSON), `paper/figs/` (figures) and `paper/macros.tex` (numbers). Stages, options and GPU re-measurement
are described in [REPRODUCE.md](REPRODUCE.md).

### What v1.0 covers

| Paper item | Where |
|---|---|
| Fig. 1, measured TTFT vs. prefix hit and concurrency | `results/measure/`, `prefixroute/measure/fit.py` |
| Table I, policies at N = 4, κ = 1, 70% load | `main` group, `results/tables/main.csv` |
| Replica count N ∈ {2, 8, 16}, load, session-shape sweeps | `nrep`, `util`, `synth_*` groups |
| Tool-latency × capacity sweep, τ = 0.05–5 s, κ ∈ {0.25, 0.5, 1, 2} | `lat`, `lat_hr1` groups, `analysis/latency.py` |
| Table III, miss-cost decomposition | `analysis/mechanism.py` |
| Mean / p95 end-to-end latency per policy | `e2e_mean`, `e2e_p95` in every simulation cell |
| Simulator validation and bias vs. the engine | `analysis/sim_bias.py`, `analysis/validate_sim.py` |

**Not yet included.** The camera-ready revision experiments are not in v1.0: the extended crossover sweep
(τ = 10–60 s, κ = 0.25 / 2 and fixed absolute capacity C; Fig. 2 and Table II), the ablations, the ρ-fixed check,
the 85%-load runs, the ITL columns of Table IV, and the correctness / McNemar replay. They will be added in a
later release. The camera-ready figures themselves are in `paper/figs/`.

## Repository layout

```
prefixroute/measure    vLLM benchmark client, cost-model fit, correctness check
prefixroute/workload   chat (Azure trace), agentic (numeric descriptor) and synthetic sessions
prefixroute/sim        iteration-level replica model and per-iteration cost model
prefixroute/policies   the eight routing policies
prefixroute/metrics    per-cell metrics incl. miss-cost decomposition
prefixroute/run_sim.py parallel, resumable grid runner
configs/               experiment grids (experiments_v2.json = paper grid), pinned model
analysis/              tables, figures, paired statistics, simulator validation
data/                  workload inputs (see Data sources)
results/measure/       raw A100 measurements and fitted cost model
reference/             shipped simulation cells (raw_v2*.zip), tables and numbers used by `verify`
paper/                 camera-ready LaTeX source and figures
scripts/, cluster/     GPU measurement on a local machine / on a Slurm cluster
tests/                 self-test and a mock vLLM server (tests the client without a GPU)
```

## Data sources

- **Chat workload:** Azure LLM Inference Trace 2023 (conversation and code), from the
  [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset). It contains timestamps and token counts
  only; see that repository for its licence. Prefix structure is synthesised (`prefixroute/workload/sessions.py`).
- **Agentic workload:** `data/agentic_trace.parquet`, a numeric descriptor (per-turn prompt, shared-prefix and
  output token counts, tool-output sizes and tool latencies) extracted from the authors' companion agent runs by
  `prefixroute/workload/agentic_extract.py`. It contains **no prompts, answers or clinical content**. The original
  run logs are not distributed.
- **Synthetic agentic workload:** generated from parameters in `configs/`.
- **Measurements:** `results/measure/`, collected with `scripts/measure_local.sh` / `cluster/measure.sbatch`. The
  agentic replay prompts are synthetic filler text with numeric tool outputs. The vLLM server logs are reduced to the
  KV-capacity line the analysis reads.
- **Model:** Qwen2.5-7B-Instruct-AWQ at revision `b25037543e9394b818fdfca67ab2a00ecc7dd641` (not included; download
  instructions in REPRODUCE.md).

## Citation

If you use this code, please cite the paper (see [CITATION.cff](CITATION.cff); GitHub's "Cite this repository"
button gives BibTeX). The archived code and data have the DOI
[10.5281/zenodo.24190142](https://doi.org/10.5281/zenodo.24190142) (all versions; v1.0 is
[10.5281/zenodo.24190143](https://doi.org/10.5281/zenodo.24190143)).

## License

Code: [MIT](LICENSE). Third-party data in `data/azure/` remains under its original licence.

## Extension: calibrated TTFT intervals under distribution shift

Quantile and conformal TTFT prediction at routing time, tested under load, tool-latency, cluster-size and traffic shifts. See [`extensions/quantile_ttft/`](extensions/quantile_ttft/).

## Extension: calibrated TTFT intervals under distribution shift

Quantile and conformal TTFT prediction at routing time, tested under load, tool-latency, cluster-size and traffic shifts. See [`extensions/quantile_ttft/`](extensions/quantile_ttft/).
