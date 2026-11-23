"""Put the PrefixRoute repo on sys.path. If pyarrow is missing, read agentic_trace.parquet with the bundled
pure-Python reader (verified to reproduce the paper's reference cells bit for bit)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
try:
    import pyarrow  # noqa: F401
except ImportError:
    import pandas as pd
    from parquet_lite import read_parquet
    _orig = pd.read_parquet

    def _rp(path, *a, **k):
        if str(path).endswith(".parquet"):
            return pd.DataFrame(read_parquet(str(path)))
        return _orig(path, *a, **k)
    pd.read_parquet = _rp
