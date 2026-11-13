#!/usr/bin/env bash
# Workstation: pack code + small data for scp (no venv, no results).   bash cluster/make_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p prefixroute/dist
tar --exclude='__pycache__' --exclude='*.pyc' \
    -czf prefixroute/dist/prefixroute_bundle.tar.gz \
    prefixroute/prefixroute prefixroute/configs prefixroute/cluster prefixroute/analysis prefixroute/tests \
    prefixroute/data/agentic_trace.parquet prefixroute/data/agentic_fit.json prefixroute/data/azure \
    prefixroute/requirements.lock prefixroute/requirements-cu13.lock prefixroute/notes
sha256sum prefixroute/dist/prefixroute_bundle.tar.gz | tee prefixroute/dist/prefixroute_bundle.tar.gz.sha256
ls -la prefixroute/dist
