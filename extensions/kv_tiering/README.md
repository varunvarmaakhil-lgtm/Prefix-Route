# A second KV-cache tier for agentic LLM serving: capacity vs. bandwidth

Extension of PrefixRoute (PCDS 2026). Each replica's GPU (HBM) prefix cache gets a larger, slower tier-2 LRU
(host DRAM over PCIe, CXL memory, NVMe SSD, ...). Blocks evicted from HBM are demoted to tier 2; a prefix found
in tier 2 is promoted back and pays `57,344 B/token / bandwidth` (Qwen2.5-7B KV size), charged synchronously to
the iteration that starts its prefill (no compute/transfer overlap: conservative). Tier bandwidths are nominal
figures, not measurements; transfers from several requests do not contend for a shared link.

```bash
conda activate prefixroute
cd extensions/kv_tiering
export OMP_NUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
python sweep.py      # 72 simulation cells -> results/tier_all.csv  (~8 min on 2 cores)
python figure.py     # -> results/fig_tiering.png
```

- `tier.py` - `TieredReplica`, `TieredCluster`, and `evict_aware_tier` (the paper's router with its shadow cache
  sized to HBM + tier 2). With tier 2 disabled and the paper's router, it reproduces the reference cell exactly
  (p95 56.879 ms).
- Regime: agentic trace, 4 replicas, 70% load, HBM cache = 0.1x the reusable working set, 30 s median tool latency.
- Analytic break-even: fetching beats recomputing when bandwidth > 57,344 B / (recompute cost per token).
  With the A100 cost model and the trace's reusable prefixes, recompute costs 139.4 us/token -> 0.41 GB/s.
