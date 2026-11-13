# Reproducing PrefixRoute

*Growing Prefixes Meet Finite Caches: When Does Eviction-Aware Routing Pay Off for Agentic LLM Serving?*
(PCDS 2026). Everything below runs on an ordinary computer, with no HPC or Slurm. Only the GPU measurement
(step 4) needs a Linux machine with an NVIDIA GPU, and its outputs are included, so every other step works from them.

## 1. Setup (any OS, Python 3.10–3.13)
On Windows, extract to a short path (e.g. `C:\prefixroute`): simulation cells have long file names, and Windows
limits paths to 260 characters unless long paths are enabled.
```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
python reproduce.py check
```

## 2. Reproduce
| Command | What it does | Time (8–20 cores) |
|---|---|---|
| `python reproduce.py quick` | Self-test (117 checks), refits the cost model from the measurement logs, replays the simulator bias, re-simulates the main grid (120 cells), then checks every regenerated cell against the reference | ~5 min |
| `python reproduce.py analysis --use-reference` | Rebuilds all tables, figures and the 102 paper numbers from the shipped simulation cells | ~2 min |
| `python reproduce.py all` | Everything from the measurement logs: 4,130 + 500 simulation cells, analysis, LaTeX paper, verification | ~1–2 h |

If a run is interrupted (sleep, closed laptop), run the same command again: finished simulation cells are kept and skipped.

`verify` (the last stage of `quick` and `all`) prints `REPRODUCED` when every regenerated simulation cell and every
paper number equals the shipped reference. The simulator is deterministic per seed, so they should match exactly.

Each stage can also be run on its own: `python reproduce.py <stage>`, where stage is one of `selftest`, `fit`,
`bias`, `sim`, `analysis`, `paper` or `verify`. Useful options:
- `--jobs N`: number of parallel processes for the simulation.
- `--groups main lat ...`: simulate only some experiment groups (see `configs/experiments_v2.json`).

Outputs:
- `paper/macros.tex`: every generated number (compared with `reference/macros.tex` by `verify`).
- `paper/figs/`: figures.
- `results/tables/`: CSV/JSON tables, plus the logs of every analysis step.
- `paper/main.pdf`: the camera-ready LaTeX paper (needs `pdflatex`).

## 3. What each part is
| Path | Contents |
|---|---|
| `prefixroute/measure/` | vLLM benchmark client (`bench.py`), cost-model fit (`fit.py`), correctness check, manifest |
| `prefixroute/sim/` | iteration-level replica model (`engine.py`) and per-iteration cost model (`costmodel.py`) |
| `prefixroute/policies/routers.py` | the 8 routing policies |
| `prefixroute/workload/` | chat (Azure trace), agentic (trace descriptor) and synthetic workloads; `agentic_extract.py` built the descriptor |
| `prefixroute/metrics/stats.py` | per-cell metrics, including the miss-cost / interference decomposition |
| `prefixroute/run_sim.py` | grid runner (parallel, resumable: finished cells are skipped) |
| `configs/` | `experiments_v2.json` (paper grid), `experiments_sens.json` (cached-token-cost sensitivity), `model.json`, `agentic_replay.json` |
| `analysis/` | `make_tables.py`, `paired.py`, `latency.py`, `mechanism.py`, `sim_bias.py`, `fit_cached_cost.py`, `validate_sim.py`, `check_macros.py` |
| `data/` | `agentic_trace.parquet` (numeric agentic workload descriptor), `agentic_fit.json`, `azure/` (Azure LLM inference trace 2023) |
| `results/measure/` | raw GPU measurement logs (sweep, probes, multi-turn replays) and the fitted cost model; `server_*.log` keep only the KV-capacity line that the analysis reads |
| `reference/` | `raw_v2.zip` and `raw_v2_sens.zip` (the 4,630 simulation cells), tables and `macros.tex` behind the submitted paper, for verification |
| `paper/` | camera-ready LaTeX source (`main.tex`, `figs/`) |
| `scripts/measure_local.sh` | GPU measurement on a local Linux machine |
| `cluster/` | the Slurm scripts used for the original run (not needed without an HPC) |
| `tests/` | self-test and a mock vLLM server used to test the measurement client without a GPU |

## 4. Re-measuring on a GPU (optional)
The paper's measurements ran on an NVIDIA A100 MIG 4g.20gb slice with vLLM 0.8.5.post1 and
Qwen2.5-7B-Instruct-AWQ @ `b25037543e9394b818fdfca67ab2a00ecc7dd641`. To measure your own GPU:
```bash
python -m venv .venv-gpu && . .venv-gpu/bin/activate && pip install -r requirements.lock
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
    --local-dir models/qwen2.5-7b-instruct-awq
bash scripts/measure_local.sh                              # about 2.5-4 h; writes results/measure_new
python reproduce.py all --measure-dir results/measure_new  # paper numbers for your GPU
```
With different hardware the absolute latencies change, so `verify` will report differences. The measurement client
can also be tested without a GPU against `tests/mock_vllm.py`.

## 5. Not included
- **The companion project's raw agent logs.** `prefixroute/workload/agentic_extract.py` built
  `data/agentic_trace.parquet` from them. The parquet itself is included; it holds only numeric per-turn token counts
  and tool latencies.
- **Model weights.** Download them as shown above.
- **Data licence.** The Azure trace is redistributed from https://github.com/Azure/AzurePublicDataset; see the
  licence there.
