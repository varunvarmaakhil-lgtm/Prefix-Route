#!/usr/bin/env bash
#SBATCH --job-name=pr-sim
#SBATCH --output=results/logs/%x-%j.out
# C2 simulation (CPU) + analysis. Cells skip when .done exists; DEADLINE (unix s) stops new cells.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster/cluster.env
source "$VENV_DIR/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -f results/measure/costmodel.json ]] || python -m prefixroute.measure.fit --dir results/measure || true
[[ -f results/measure/costmodel.json ]] || echo "WARNING: no measured cost model; simulation uses the roofline estimate"
for g in ${SIM_GROUPS:-main cap cap_slowtool cap_headroom1 nrep util synth_turns synth_growth synth_think nocache}; do
  python -m prefixroute.run_sim --groups "$g" --jobs "${SLURM_CPUS_PER_TASK:-$SIM_CPUS}" ${DEADLINE:+--deadline $DEADLINE}
done
python analysis/validate_sim.py || true
python analysis/make_tables.py && python analysis/paired.py || true
echo "sim done $(date -Is)"
