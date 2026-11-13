#!/usr/bin/env bash
# Workstation-side launcher. Usage: bash cluster/launch.sh <ssh-alias> <step>
#   push     scp bundle + unpack into ~/prefixroute (code/data only; results never overwritten)
#   probe    run cluster/probe.sh (read-only + 5-min GPU job) and autoconf the Slurm partitions
#   setup    start cluster/setup.sh in a detached tmux session on the login node
#   status   tail setup log / squeue / measurement progress
#   submit   bash cluster/submit.sh all
#   pull     rsync/scp results back into ./results
set -euo pipefail
H=${1:?ssh alias}; STEP=${2:?step}
cd "$(dirname "$0")/.."
case $STEP in
  push)
    bash cluster/make_bundle.sh >/dev/null
    scp dist/prefixroute_bundle.tar.gz "$H":~/
    ssh "$H" 'mkdir -p ~/prefixroute && tar -xzf ~/prefixroute_bundle.tar.gz -C ~ && mkdir -p ~/prefixroute/results/logs && ls ~/prefixroute' ;;
  probe)
    ssh "$H" 'cd ~/prefixroute && bash cluster/probe.sh >/dev/null 2>&1; bash cluster/autoconf.sh; tail -25 notes/cluster_env_raw.txt' ;;
  setup)
    ssh "$H" 'cd ~/prefixroute && if command -v tmux >/dev/null; then (tmux kill-session -t prsetup 2>/dev/null || true); tmux new -d -s prsetup "bash cluster/setup.sh 2>&1 | tee results/logs/setup.log"; else nohup bash cluster/setup.sh > results/logs/setup.log 2>&1 < /dev/null & fi; echo started' ;;
  status)
    ssh "$H" 'cd ~/prefixroute && tail -5 results/logs/setup.log 2>/dev/null; squeue -u $USER; ls results/measure 2>/dev/null | head -30; tail -3 results/measure/sweep.jsonl 2>/dev/null | cut -c1-120; tail -3 results/logs/pr-*.out 2>/dev/null' ;;
  submit)
    ssh "$H" 'cd ~/prefixroute && bash cluster/submit.sh all' ;;
  pull)
    mkdir -p results
    scp -r "$H":~/prefixroute/results/measure results/ || true
    ssh "$H" 'cd ~/prefixroute && tar -czf /tmp/$USER-raw.tgz results/raw results/tables 2>/dev/null' && scp "$H":/tmp/*-raw.tgz dist/ && tar -xzf dist/*-raw.tgz || true
    scp -r "$H":~/prefixroute/results/logs results/cluster_logs || true
    scp "$H":~/prefixroute/MANIFEST.json . || true
    scp "$H":~/prefixroute/notes/cluster_env_raw.txt "$H":~/prefixroute/notes/storage.md notes/ || true ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
