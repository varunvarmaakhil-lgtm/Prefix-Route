#!/usr/bin/env bash
# Read-only cluster probe (login node) + one short GPU job. Output -> notes/cluster_env_raw.txt
#   bash cluster/probe.sh
set -u
cd "$(dirname "$0")/.."
source cluster/cluster.env
OUT=notes/cluster_env_raw.txt; mkdir -p notes
{
sec() { echo; echo "===== $* ====="; }
sec host; hostname; date -Is; whoami; grep PRETTY /etc/os-release
sec slurm; sinfo -o "%P %a %l %D %G %c %m" 2>&1 | head -30
sacctmgr -nP show assoc user=$USER format=account,partition,qos,grptres,maxtres,maxjobs 2>&1 | head
sacctmgr -nP show qos format=name,maxwall,maxtrespu,maxjobspu 2>/dev/null | head
squeue -u $USER 2>&1 | head
sec filesystem; echo "HOME=$HOME"; df -h $HOME /tmp 2>/dev/null; (quota -s 2>/dev/null || true) | head; du -sh $HOME 2>/dev/null
sec python; command -v python3 && python3 --version; command -v uv || echo "uv: absent"
sec internet; for u in https://pypi.org/simple/ https://huggingface.co; do curl -sS -m 10 -o /dev/null -w "$u %{http_code}\n" $u || echo "$u FAIL"; done
sec "gpu job"
bash cluster/autoconf.sh >/dev/null 2>&1; source cluster/slurm.env
timeout 900 srun ${SLURM_GPU_PARTITION:+-p $SLURM_GPU_PARTITION} ${SLURM_ACCOUNT:+-A $SLURM_ACCOUNT} --gres=$GPU_GRES -t 00:05:00 -c 2 --mem=8G bash -c '
  hostname; nvidia-smi --query-gpu=name,memory.total,driver_version,mig.mode.current --format=csv; nvidia-smi | grep "CUDA Version";
  curl -sS -m 10 -o /dev/null -w "compute-node pypi %{http_code}\n" https://pypi.org/simple/ || echo "compute-node internet: NONE";
  nproc; free -g | head -2; df -h /tmp | tail -1' 2>&1 || echo "srun GPU probe failed"
echo; echo "probe done $(date -Is)"
} 2>&1 | tee "$OUT"
