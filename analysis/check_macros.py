"""List \\pr{...} names used in paper/main.tex that analysis/make_tables.py did not generate (they print as ??)."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
mac = open(os.path.join(ROOT, "paper", "macros.tex"), encoding="utf-8").read()
used = set(re.findall(r"\\pr\{([A-Za-z0-9]+)\}", tex))
defined = set(re.findall(r"pr@([A-Za-z0-9]+)", mac))
missing = sorted(used - defined)
print(f"{len(used)} macros used, {len(missing)} undefined: {missing}")
sys.exit(1 if missing else 0)
