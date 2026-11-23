# Calibrated TTFT intervals at routing time, under distribution shift

Extension of PrefixRoute (PCDS 2026). Place this folder at `extensions/quantile_ttft/` in the repo.

```bash
cd extensions/quantile_ttft
export OMP_NUM_THREADS=1        # avoids an OpenMP deadlock in forked workers
python collect.py               # 15 simulation runs -> logs/*.csv   (~1 min on 2 cores)
python evaluate.py              # intervals + metrics -> results_intervals.csv, fit_info.json
python figure.py                # -> fig_intervals.png
python route_quantile.py        # routing on predicted quantiles -> results_routing.csv (~6 min)
```

- `collect.py` - eviction-aware router instrumented to log 17 router-observable features + realised TTFT
  (routing decisions unchanged; base seed 0 reproduces the reference p95 of 56.879 ms).
- `evaluate.py` - analytic+conformal, quantile regression, CQR, hybrid (residual on the analytic estimate),
  and online quantile-tracking recalibration; coverage, width, Winkler score, median error.
- `route_quantile.py` - routes to the replica with the lowest predicted q50 / q90 TTFT.
- `parquet_lite.py`, `shim.py` - pure-Python Parquet reader used only when pyarrow is unavailable.

Train: agentic, N=4, util 0.7, cap_frac 1.0, seeds 0-1 (fit) + 2 (calibration). Test: seeds 3-4 of that regime
and of: util 0.9; 5 s tools + 0.25x cache; synthetic sessions; N=8; chat traffic.
