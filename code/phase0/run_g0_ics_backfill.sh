#!/usr/bin/env bash
# Background backfill for remaining G0 global ICs (skips already staged).
set -euo pipefail
cd ~/fourcastnet
source ~/fcn3-venv/bin/activate
mkdir -p logs data/g0_ics
LOG=logs/g0_ics_backfill.log
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting --all (resume/skip existing)" | tee -a "$LOG"
# Do NOT touch ERA5 regional backfill PID
python -u code/phase0/stage_g0_ics.py --all >> "$LOG" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done exit=$?" | tee -a "$LOG"
