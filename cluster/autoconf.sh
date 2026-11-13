#!/usr/bin/env bash
# Pick the Slurm account and GPU/CPU partitions this user may use; write cluster/slurm.env. Idempotent.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ASSOC=$(sacctmgr -nP show assoc user=$USER format=account,partition 2>/dev/null)
ACC=$(echo "$ASSOC" | cut -d'|' -f1 | grep -v '^$' | head -1)
PARTS=$(echo "$ASSOC" | cut -d'|' -f2 | grep -v '^$' | tr '\n' ' ')
ALL=$(sinfo -h -o "%P" | tr -d '*' | sort -u)
pick() {  # prefix -> first partition matching gpu_/cpu_ that we may use (association-restricted or allowed)
  local kind=$1 p
  for p in $ALL; do
    [[ "$p" == ${kind}* ]] || continue
    if [[ -n "$PARTS" ]]; then [[ " $PARTS " == *" $p "* ]] && { echo "$p"; return; }
    else
      # no partition restriction in the association: prefer the one named after the account (e.g. gpu_student)
      [[ -n "$ACC" && "$p" == *"$ACC"* ]] && { echo "$p"; return; }
    fi
  done
  for p in $ALL; do [[ "$p" == ${kind}* ]] && { echo "$p"; return; }; done
}
GP=$(pick gpu); CP=$(pick cpu)
cat > cluster/slurm.env <<EOT
export SLURM_ACCOUNT="${ACC}"
export SLURM_GPU_PARTITION="${GP}"
export SLURM_CPU_PARTITION="${CP}"
EOT
cat cluster/slurm.env
