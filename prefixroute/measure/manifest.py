"""Write MANIFEST.json: model revision + shard checksums, data checksums + row counts, lock checksum, versions."""
import glob
import hashlib
import json
import os
import subprocess

import pandas as pd


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.load(open("configs/model.json"))
    d = os.path.join(os.environ.get("MODELS_DIR", "models"), m["key"])
    man = {"model": {**m, "files": {os.path.basename(p): sha(p) for p in sorted(glob.glob(d + "/*.safetensors"))}},
           "data": {}, "lock_sha256": sha(os.environ.get("LOCK_FILE", "requirements.lock"))}
    for p in ["data/agentic_trace.parquet", "data/azure/AzureLLMInferenceTrace_conv.csv",
              "data/azure/AzureLLMInferenceTrace_code.csv"]:
        if os.path.exists(p):
            n = len(pd.read_parquet(p)) if p.endswith(".parquet") else sum(1 for _ in open(p)) - 1
            man["data"][p] = {"sha256": sha(p), "rows": n}
    try:
        out = subprocess.check_output(["uv", "pip", "freeze"], text=True)
        man["versions"] = [l for l in out.splitlines()
                           if l.split("==")[0] in ("vllm", "torch", "transformers", "xformers", "flashinfer-python")]
    except Exception as e:  # noqa: BLE001
        man["versions"] = str(e)
    json.dump(man, open("MANIFEST.json", "w"), indent=1)
    print("MANIFEST.json written:", len(man["model"]["files"]), "model files,", len(man["data"]), "data files")


if __name__ == "__main__":
    main()
