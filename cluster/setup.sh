#!/usr/bin/env bash
# One-time setup on the LOGIN node (network I/O only): venv + pinned wheels, model at a pinned revision,
# verification, MANIFEST.json. Idempotent/resumable.
#   tmux new -s setup 'bash cluster/setup.sh 2>&1 | tee results/logs/setup.log'
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env
mkdir -p "$MODELS_DIR" results/logs notes
log() { echo "[setup $(date +%H:%M:%S)] $*"; }
store() {
  echo "$(date -Is) $*: dir $(du -sh "$PR_ROOT" 2>/dev/null | cut -f1), free $(df -h --output=avail "$PR_ROOT" | tail -1 | tr -d ' ')" >> notes/storage.md
}
store "setup start"

# ---- venv (uv-managed CPython 3.12) + pinned lock -------------------------------------------------
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[[ -x "$VENV_DIR/bin/python" ]] || uv venv -p 3.12 "$VENV_DIR"
source "$VENV_DIR/bin/activate"
SHA=$(sha256sum "$LOCK_FILE" | cut -c1-64)
if [[ "$(cat "$VENV_DIR/.lock_sha" 2>/dev/null)" != "$SHA" ]]; then
  log "installing $LOCK_FILE"
  uv pip install -r "$LOCK_FILE"
  echo "$SHA" > "$VENV_DIR/.lock_sha"
fi
rm -rf "$UV_CACHE_DIR"          # wheel cache is off-quota and deleted (50 GB cap)
python -c "import aiohttp, numpy, pandas, pyarrow, joblib, scipy, matplotlib; print('imports ok')"
store "after venv"

# ---- model (pinned revision) --------------------------------------------------------------------------
read -r KEY REPO REV <<<"$(python -c "import json;m=json.load(open('configs/model.json'));print(m['key'],m['repo'],m['revision'])")"
DEST="$MODELS_DIR/$KEY"
if [[ ! -f "$DEST/.complete" ]]; then
  for i in 1 2 3 4 5; do
    hf download "$REPO" --revision "$REV" --local-dir "$DEST" && break
    log "download attempt $i failed"; sleep 30
    [[ $i == 5 ]] && exit 2
  done
  python -m prefixroute.measure.verify_model "$DEST"
  touch "$DEST/.complete"
fi
store "after model"

python -m prefixroute.measure.manifest
bash cluster/autoconf.sh
log "setup complete. Next: bash cluster/submit.sh"
