#!/usr/bin/env bash
#SBATCH --job-name=pr-measure
#SBATCH --output=results/logs/%x-%j.out
# C1 measurement on one A100, hard-capped at MEASURE_BUDGET_MIN. Three server configurations in priority order:
#   on (prefix caching) -> nocache -> pressure (caching on, KV cache forced small). Resumable: finished
#   settings are skipped on resubmission.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster/cluster.env
source "$VENV_DIR/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_SERVER_DEV_MODE=1          # enables /reset_prefix_cache
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OUT=results/measure
mkdir -p "$OUT" results/logs
T_END=$(( $(date +%s) + MEASURE_BUDGET_MIN * 60 ))
left_min() { echo $(( (T_END - $(date +%s)) / 60 )); }
eval "$(python -m prefixroute.measure.modelvars)"
MDIR="$MODELS_DIR/$KEY"
[[ -f "$MDIR/.complete" ]] || { echo "model missing: run cluster/setup.sh"; exit 3; }
LOCAL="${SLURM_TMPDIR:-/tmp}/$USER-pr"
mkdir -p "$LOCAL" && rsync -a --exclude .cache "$MDIR/" "$LOCAL/$KEY/" && MDIR="$LOCAL/$KEY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" | tee "$OUT/versions.txt"
PORT=$(python -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1])")
VPID=""

start_server() {  # tag, extra args...
  local tag=$1; shift
  vllm serve "$MDIR" --served-model-name "$KEY" --host 127.0.0.1 --port "$PORT" --max-model-len "$MAXLEN" \
    --max-num-batched-tokens "$MNBT" --gpu-memory-utilization "$GPUUTIL" --quantization "$QUANT" --seed 0 \
    --enable-prompt-tokens-details --disable-log-requests --generation-config vllm "$@" > "$OUT/server_$tag.log" 2>&1 &
  VPID=$!
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "[$(date +%T)] server $tag up"; return 0; fi
    if ! kill -0 "$VPID" 2>/dev/null; then echo "server $tag died"; tail -40 "$OUT/server_$tag.log"; VPID=""; return 1; fi
    sleep 5
  done
  return 1
}
stop_server() {
  if [[ -n "$VPID" ]]; then kill "$VPID" 2>/dev/null; sleep 8; kill -9 "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null; fi
  VPID=""
}
trap stop_server EXIT

B="python -m prefixroute.measure.bench --url http://127.0.0.1:$PORT --model $KEY --out $OUT"
phase() {  # tag budget_min server-args...
  local tag=$1 budget=$2; shift 2
  local l; l=$(left_min)
  if (( l < 5 )); then echo "skip phase $tag: out of time"; return; fi
  (( budget > l )) && budget=$l
  local p_end=$(( $(date +%s) + budget * 60 ))
  echo "[$(date +%T)] phase $tag budget ${budget} min"
  if ! start_server "$tag" "$@" ${EXTRA_VLLM_ARGS:-}; then
    echo "retrying $tag with --enforce-eager"; EXTRA_VLLM_ARGS="--enforce-eager"
    start_server "$tag" "$@" $EXTRA_VLLM_ARGS || return
  fi
  $B probe --tag "$tag"
  local sb=$(( budget - 15 )); (( sb < 5 )) && sb=5
  if [[ $tag == pressure ]]; then
    $B sweep --tag "$tag" --budget-min "$sb" --lengths 4096 --concs 1 16 32
  else
    $B sweep --tag "$tag" --budget-min "$sb"
  fi
  for r in 0 1; do
    (( $(date +%s) + 240 < p_end )) || break
    timeout 600 $B agentic --tag "$tag" --repeat "$r" --n-sessions "${AGENTIC_SESSIONS:-300}" --seed 11
  done
  stop_server
}
phase on       110 --enable-prefix-caching
phase nocache   85 --no-enable-prefix-caching
phase pressure  35 --enable-prefix-caching --num-gpu-blocks-override "$PBLOCKS"
python -m prefixroute.measure.fit --dir "$OUT" --chunk "$MNBT" | tee "$OUT/fit.txt"
python -m prefixroute.measure.correctness --dir "$OUT" | tee "$OUT/correctness.txt"
echo "measure done $(date -Is)"
