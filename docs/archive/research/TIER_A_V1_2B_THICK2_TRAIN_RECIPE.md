# Tier-A v1.2b thick-2 retrain recipe (Leonard) — 2026-09-14

**Manisha greenlight (via Howard):** retrain v1.2b on thick-2 · **not** diffusion.  
**Bars (living):** `HOLDOUT_THICK2_CALL.md` — val **< 1.988588** / test **< 1.918383** / +120h **≤ 2.178842**  
**Years:** hard-locked · `provisional_years=false` · `year_split_frozen=true`  
**Pairs:** `holdout_expand=v2` · train 12 / val 16 / test 16  
**Prior ZS:** `v1_2b_thick2/` INTERIM PASS CONFIRM (historical once retrain lands)

## Freeze — train recipe

| Knob | v1.2b (expand train) | **v1.2b thick-2 retrain** |
| --- | --- | --- |
| Model | ElevCondResidualUNet base=16, depth=3, dropout=0.20 | **Same** |
| Params | ~168k | **Same** |
| FCN3 | FROZEN | **Same** |
| `w_120` | 2.0 | **Keep 2.0** |
| lead weights | 1 / 1 / 2.0 | **Same** |
| weight_decay | 1e-3 | **Same** |
| elev_weight_boost | 1.5 | **Same** |
| channel weights | t2m=2, u/v=0.5 | **Same** |
| ckpt | composite; reject if val +120h > raw | **Same** · raw = **thick-2** 2.178842 |
| patience | 80 | **Keep 80** |
| max epochs | 400 | **Keep 400** |
| seed | 42 (unless logged otherwise) | **Keep; document in JSON** |
| Train ICs | ic01–08 (8) | **ic01–08 + ic33–36 (12)** |
| Val (early-stop) | expand val 12 | **thick-2 val 16** |
| Test | expand test 12 | **thick-2 test 16** |

### Paths (LOCKED)
- **Out dir:** `runs/phase0/tier_a/v1_2b_thick2_train/`
- **Config:** `configs/tier_a_v1_2b_thick2_train.yaml`
- **Code:** reuse `code/tier_a/v1_2b/` (or thin copy under `code/tier_a/v1_2b_thick2_train/` if cleaner) — **do not** overwrite `v1_2b/`, `v1_2b_thick2/` (ZS), or earlier frozen dirs
- **Results:** `.../tier_a_v1_2b_thick2_train_results.json` + `best_residual.pt`
- **Pairs:** `runs/phase0/tier0_holdout_expand_v2/pairs/` (or manifest-equivalent)

### Changes vs expand-era v1.2b train
1. Train/val/test IC counts **12/16/16** (not 8/12/12).
2. Composite raw +120h reference = **2.178842** (not 2.131516).
3. Beat-this = **thick-2** bars (not expand-v1).
4. Labels: `holdout_expand=v2`, `provisional_years=false`, `year_split_frozen=true`, `bars_set=thick2`.

## PASS / FAIL (LOCKED)

INTERIM PASS requires **all**:
1. Val pooled t2m RMSE **< 1.988588**
2. Test pooled t2m RMSE **< 1.918383**
3. Val +120 h t2m RMSE **≤ 2.178842**
4. **`n_eligible_saves ≥ 1`** and reload = `composite_eligible`  
   — unconstrained fallback ⇒ **automatic FAIL** (same as v1.2b rule)

Report per-lead + elev bands. ZS thick-2 numbers stay as baseline comparison in the call.

### After results
- Leonard writes `TIER_A_V1_2B_THICK2_TRAIN_CALL.md` PASS/FAIL.
- If PASS: freeze this dir as **new living** residual (ZS demoted to historical).
- If FAIL: iterate knobs only with Leonard call — **still no diffusion** unless Manisha orders.

## Eng order (Howard)
1. CPU scaffold + dry-run (echo thick-2 bars / raw +120h).
2. GPU `--train` when dry-run OK.
3. Ping Leonard with results JSON path.
4. **Nepal ERA5 CDS crop still filling — leave download alone (check `ps` / `logs/era5_pull.log` for live PID).** No diffusion.

## Non-goals
- Overwriting `v1_2b/` or `v1_2b_thick2/`
- CorrDiff / Tier-B / G1 claims
