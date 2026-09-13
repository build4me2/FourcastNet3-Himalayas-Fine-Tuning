# Tier-A v1.2b — iterate FAIL v1.2 (NO diffusion)

**Status:** TRAIN (Manisha greenlit). Parent FAIL: `docs/research/TIER_A_V1_2_CALL.md`.
**Arch:** same ElevCondResidualUNet as v1.1/v1.2 (`base=16`, `dropout=0.20`).
**Knobs:** `w_120=2.0` (was 1.5), `early_stop_patience=80` (was 50).
**PASS lock:** `n_eligible_saves >= 1` with `composite_eligible` reload; unconstrained fallback ⇒ auto FAIL.

## Gates (expand — Leonard HOLDOUT_EXPAND_CALL.md)

| Gate | Bar |
|------|-----|
| val t2m pooled RMSE | strictly **< 1.987014 K** |
| test t2m pooled RMSE | strictly **< 1.897298 K** |
| **hard** val +120 h t2m RMSE | **≤ expand-val raw 2.131516 K** |
| eligible composite ckpt | **n_eligible_saves ≥ 1** |

## Do not overwrite

`v0/`, `v0_1/`, `v1/`, `v1_1/`, `v1_1_expand/`, **`v1_2/`** (FAIL freeze), holdouts, G0. Leave ERA5 PID **611595** alone.

## Quick start (Spark)

```bash
cd ~/fourcastnet
~/fcn3-venv/bin/python code/tier_a/v1_2b/train.py --config configs/tier_a_v1_2b.yaml --dry-run
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_2b/train.py \
  --config configs/tier_a_v1_2b.yaml --train --allow-gpu --epochs 400
```

Outputs: `runs/phase0/tier_a/v1_2b/{tier_a_v1_2b_results.json,tier_a_v1_2b_metrics.csv,best_residual.pt}`.
