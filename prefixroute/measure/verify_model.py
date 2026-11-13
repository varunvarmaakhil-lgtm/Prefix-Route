"""Verify a downloaded model directory: all shards present, AWQ quantisation config. Exit 1 on failure."""
import json
import os
import sys

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
shards = [f for f in os.listdir(d) if f.endswith(".safetensors")]
idx_path = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx_path):
    need = set(json.load(open(idx_path))["weight_map"].values())
    missing = [f for f in need if f not in shards]
    assert not missing, f"missing shards {missing}"
assert shards, "no safetensors"
q = cfg.get("quantization_config", {})
assert q.get("quant_method") == "awq", q
print("model verified:", cfg["architectures"], cfg["num_hidden_layers"], "layers,", len(shards), "shards")
