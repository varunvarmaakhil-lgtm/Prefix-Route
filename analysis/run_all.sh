#!/usr/bin/env bash
# Regenerate every table, figure and macro of the paper from results/ (v2: results/raw_v2). Order matters:
# make_tables.py rewrites paper/macros.tex; the later scripts append to it.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=${RAW:-results/raw_v2}
python analysis/sim_bias.py > results/tables/sim_bias.log
python analysis/validate_sim.py > results/tables/validate_sim.log
python analysis/make_tables.py --raw "$RAW"
python analysis/paired.py "$RAW"
python analysis/latency.py --raw "$RAW" > results/tables/latency.log
python analysis/latency.py --raw "$RAW" --headroom 1.0 > results/tables/latency_hr1.log
[[ -d results/raw_v2_sens ]] && python analysis/latency.py --raw results/raw_v2_sens --tag sens > results/tables/latency_sens.log
python analysis/mechanism.py --raw "$RAW" > results/tables/mechanism.log
python analysis/check_macros.py
