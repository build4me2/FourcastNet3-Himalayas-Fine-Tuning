# Tier-A v1.3 joint call (Leonard) — 2026-09-15

**Artifacts (Spark)**
- `~/fourcastnet/runs/phase0/tier_a/v1_3_joint/tier_a_v1_3_joint_results.json`
- `best_residual.pt` · `living_wind_baseline.json` · `v1_3_joint_wind_score.json`
- Recipe: `TIER_A_V1_3_JOINT_RECIPE.md`

**Arch:** ElevCondResidualUNet · ~168k · `channel_weights=[1.5,1.5,1.5]` · FCN3 **FROZEN** · parallel residual (not leftover-diff)

## Call

| Gate | Result |
| --- | --- |
| **A** thick-2 t2m floors + eligible | **PASS** — val **1.770** / test **1.775** (16-IC) / +120h **1.982** · **46** eligible · `composite_eligible` |
| **B** protect living t2m (±0.02 K) | **PASS** — val Δ **+0.010** / test Δ **−0.005** (16-IC pooled vs living baseline) |
| **C** wind-vector lead-mean vs living+raw | **PASS** — val **0.69560** (< 0.69910 / 0.71518) · test **0.73877** (< 0.74493 / 0.76276) |
| **INTERIM PASS** | **YES — CONFIRM** |
| **Promote living residual?** | **YES** |

## Living residual (LOCKED after this call)

| Role | Path |
| --- | --- |
| **NEW living** | `runs/phase0/tier_a/v1_3_joint/` |
| Prior living | `v1_2b_thick2_train/` → **historical** (still frozen; md5 unchanged through train) |
| v1-diff null | `v1_diff/` → keep |

## Headline (protocol-locked thick-2 12/16/16)

| | Prior living | **v1.3 joint** |
| --- | ---: | ---: |
| Val t2m pooled | 1.760 | **1.770** |
| Test t2m pooled (16-IC) | 1.780 | **1.775** |
| Val +120 h t2m | 1.962 | **1.982** |
| Val wind-vector lead-mean | 0.69910 | **0.69560** |
| Test wind-vector lead-mean | 0.74493 | **0.73877** |

Winds are the promote story (small but strict lift). t2m stays inside the protect band (val slight regress allowed; test slight improve).

## WARN — results.json test echo

`tier_a_v1_3_joint_results.json` lists **17** `test_ids` (extra **`ic45`**) and `metrics.test` n=39627.  
Config + wind baseline/score use the **locked 16** thick-2 test ICs (n=37296).  

**Claims / promote numbers use the 16-IC wind_score + abc_gates path.**  
Howard: patch results JSON echo to match config `test_ids` (16 only) — **no retrain**; do not let `ic45` leak into future runs.

## Next eng (Howard)

1. Mirror this call to Spark `docs/research/TIER_A_V1_3_JOINT_CALL.md`.
2. Echo **new living** = `v1_3_joint/` in status docs; keep prior living frozen.
3. Fix results `test_ids` / optional 16-IC test re-score headline only.
4. Idle on next model unless Manisha assigns (deferred: parallel-mean diff / multi-seed / write-up).
5. Nepal ERA5 CDS crop still filling — leave download alone (check `ps` / `logs/era5_pull.log` for live PID).

## Non-claims
- Not G1 / IMDAA / CorrDiff / FCN3 FT  
- interim_era5 · years hard-locked · thick-2  
- Wind lift is modest — publishable as joint-balance success, not a winds breakthrough  
