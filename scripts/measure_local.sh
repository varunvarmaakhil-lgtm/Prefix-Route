#!/usr/bin/env bash
# C1 measurement on a local Linux machine with one NVIDIA GPU (no Slurm needed). Same procedure as
# cluster/measure.sbatch: three vLLM server configurations (prefix caching on / off / forced-small KV cache), each
# with correctness probes, the TTFT sweep and the multi-turn agentic replay; then the cost-model fit.
#
#   python -m venv .venv-gpu && . .venv-gpu/bin/activate && pip install -r requirements.lock   # vLLM 0.8.5.post1
#   huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
#       --local-dir models/qwen2.5-7b-instruct-awq
#   bash scripts/measure_local.sh                    # writes results/measure_new (shipped data stay untouched)
#
# Env: MODEL_DIR (default models/qwen2.5-7b-instruct-awq), OUT (default results/measure_new),
#      BUDGET_MIN (default 235), AGENTIC_SESSIONS (default 300).
# Then reproduce the paper from the new data:  python reproduce.py all --measure-dir results/measure_new
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_SERVER_DEV_MODE=1
eval "$(python -m prefixroute.measure.modelvars)"
MDIR="${MODEL_DIR:-models/$KEY}"
OUT="${OUT:-results/measure_new}"
[[ -f "$MDIR/config.json" ]] || { echo "model not found in $MDIR (see header for the download command)"; exit 3; }
mkdir -p "$OUT"
T_END=$(( $(date +%s) + ${BUDGET_MIN:-235} * 60 ))
left_min() { echo $(( (T_END - $(date +%s)) / 60 )); }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" | tee "$OUT/versions.txt"
PORT=$(python -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1])")
VPID=""
start_server() {
  local tag=$1; shift
  vllm serve "$MDIR" --served-model-name "$KEY" --host 127.0.0.1 --port "$PORT" --max-model-len "$MAXLEN" \
    --max-num-batched-tokens "$MNBT" --gpu-memory-utilization "$GPUUTIL" --quantization "$QUANT" --seed 0 \
    --enable-prompt-tokens-details --disable-log-requests --generation-config vllm "$@" > "$OUT/server_$tag.log" 2>&1 &
  VPID=$!
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "[$(date +%T)] server $tag up"; return 0; }
    kill -0 "$VPID" 2>/dev/null || { echo "server $tag died"; tail -40 "$OUT/server_$tag.log"; VPID=""; return 1; }
    sleep 5
  done
  return 1
}
stop_server() { [[ -n "$VPID" ]] && { kill "$VPID" 2>/dev/null; sleep 8; kill -9 "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null; }; VPID=""; }
trap stop_server EXIT
B="python -m prefixroute.measure.bench --url http://127.0.0.1:$PORT --model $KEY --out $OUT"
phase() {
  local tag=$1 budget=$2; shift 2
  local l; l=$(left_min); (( l < 5 )) && { echo "skip $tag: out of time"; return; }
  (( budget > l )) && budget=$l
  local p_end=$(( $(date +%s) + budget * 60 ))
  start_server "$tag" "$@" || { echo "retrying $tag with --enforce-eager"; start_server "$tag" "$@" --enforce-eager || return; }
  $B probe --tag "$tag"
  local sb=$(( budget - 15 )); (( sb < 5 )) && sb=5
  if [[ $tag == pressure ]]; then $B sweep --tag "$tag" --budget-min "$sb" --lengths 4096 --concs 1 16 32
  else $B sweep --tag "$tag" --budget-min "$sb"; fi
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
echo "measurement done -> $OUT"
