# Cache-aware LLM routing on a heterogeneous GPU cluster

Extension of PrefixRoute (PCDS 2026). Mixed cluster: 2x A100 (measured cost model) + 2x L4-class GPUs
(roofline-scaled from the A100 measurements: 2.58x slower compute, 5.18x lower bandwidth, 0.53x KV memory).
**The L4 model is an assumption, not a measurement.**

```bash
conda activate prefixroute
cd extensions/hetero_cache
export OMP_NUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
python sweep.py      # 225 simulation cells -> results/het_all.csv  (~10 min on 2 cores, longer on 1)
python figure.py     # -> results/fig_hetero.png
```

- `het.py` - `HetCluster` (per-replica cost model and KV capacity), `evict_aware_het` (the paper's router given
  per-GPU cost models and cache sizes), `oracle_het` (clairvoyant reference), `least_loaded_speed`.
  With identical replicas, `evict_aware_het` reproduces the paper's reference cell exactly (p95 56.879 ms).
- `sweep.py` - load sweep (0.7-3.0x), KV-cache-pressure regime (cache_frac 0.25, 5 s tools), speed-only and
  memory-only ablations, and a "2x A100 only" baseline with the same per-GPU memory at the same arrival rate.
- Load is "nominal": the no-reuse demand of the trace relative to the mixed cluster's capacity. With ~92% prefix
  reuse, real utilisation is much lower than the nominal figure.
