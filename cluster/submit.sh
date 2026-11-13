#!/usr/bin/env bash
# Submit the night: GPU measurement, then CPU simulation + analysis (afterany).
#   bash cluster/submit.sh [all|measure|sim]
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env
mkdir -p results/logs
WHAT=${1:-all}
A=()
[[ -n "$SLURM_ACCOUNT" ]] && A=(-A "$SLURM_ACCOUNT")
MJ=""
if [[ $WHAT == all || $WHAT == measure ]]; then
  MJ=$(sbatch --parsable "${A[@]}" ${SLURM_GPU_PARTITION:+-p $SLURM_GPU_PARTITION} --gres="$GPU_GRES" \
       -c "$MEASURE_CPUS" --mem=48G -t $(( MEASURE_BUDGET_MIN + 20 )) cluster/measure.sbatch)
  echo "measure job $MJ"
fi
if [[ $WHAT == all || $WHAT == sim ]]; then
  DL=$(( $(date +%s) + ${SIM_HOURS:-10} * 3600 ))
  SJ=$(sbatch --parsable "${A[@]}" ${SLURM_CPU_PARTITION:+-p $SLURM_CPU_PARTITION} -c "$SIM_CPUS" --mem=96G -t 08:00:00 \
       ${MJ:+--dependency=afterany:$MJ} --export=ALL,DEADLINE=$DL cluster/sim.sbatch)
  echo "sim job $SJ"
fi
squeue -u "$USER"
