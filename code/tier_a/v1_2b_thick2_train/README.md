# Tier-A v1.2b thick-2 RETRAIN (NO diffusion)

**Status:** SCAFFOLD / Leonard FREEZE 2026-09-14 / Manisha greenlit GPU after dry-run.
**Recipe:** `docs/research/TIER_A_V1_2B_THICK2_TRAIN_RECIPE.md`
**Arch:** ElevCondResidualUNet (`base=16`, `dropout=0.20`) · `w_120=2.0` · patience=80 · seed=42 · epochs=400
**Data:** `holdout_expand=v2` train **12** / val **16** / test **16** · `bars_set=thick2`
**PASS lock:** all three thick-2 gates + `n_eligible_saves ≥ 1` (`composite_eligible`); unconstrained fallback ⇒ auto FAIL.

## Gates (thick-2 — living)

| Gate | Bar |
|------|-----|
| val t2m pooled RMSE | strictly **< 1.988588 K** |
| test t2m pooled RMSE | strictly **< 1.918383 K** |
| **hard** val +120 h t2m RMSE | **≤ thick-2 raw 2.178842 K** |
| eligible composite ckpt | **n_eligible_saves ≥ 1** |

## Do not overwrite

`v1_2b/` (INTERIM PASS weights), `v1_2b_thick2/` (ZS), earlier frozen dirs, holdouts, G0. Leave ERA5 PID **611595** alone. No diffusion.

## Quick start (Spark)

```bash
cd ~/fourcastnet
~/fcn3-venv/bin/python code/tier_a/v1_2b_thick2_train/train.py \
  --config configs/tier_a_v1_2b_thick2_train.yaml --dry-run
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_2b_thick2_train/train.py \
  --config configs/tier_a_v1_2b_thick2_train.yaml --train --allow-gpu --epochs 400
```

Outputs: `runs/phase0/tier_a/v1_2b_thick2_train/{tier_a_v1_2b_thick2_train_results.json,best_residual.pt}`.
