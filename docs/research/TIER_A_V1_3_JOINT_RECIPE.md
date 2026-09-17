# Tier-A v1.3 joint (winds+t2m residual) recipe (Leonard) — 2026-09-15

**Manisha assign (via Howard):** different Phase-3 fork — **not** leftover-target v1-diff (NULL).  
**Living residual (canonical):** `runs/phase0/tier_a/v1_2b_thick2_train/` · t2m **1.760 / 1.780 / +120h 1.962**  
**Protocol:** thick-2 12/16/16 · year hard-lock · bars `HOLDOUT_THICK2_CALL.md`  
**CDS PID 611595:** leave alone. **No train until this recipe lands** (now frozen).

## Why this fork (not the others)

| Candidate | Decision |
| --- | --- |
| Leftover-target diffusion (ERA5−(FCN3+residual)) | **Killed** — `TIER_A_V1_DIFF_CALL.md` NULL |
| Parallel-mean diffusion (ERA5−FCN3, residual out of target) | **Deferred** — higher risk / new stack after NULL |
| Multi-seed uncertainty table | **Deferred** — characterization, not a skill fork |
| **Joint winds+t2m residual upgrade** | **SELECTED** — living residual is strong on t2m; channel weights were `t2m=2, u/v=0.5`; G0 cares about winds; reuses ElevCondResidualUNet |

**One-sentence method:** Same frozen-FCN3 residual UNet family as v1.2b, rebalanced for **joint t2m + u10m + v10m**, with winds as a **hard promote gate** while protecting living t2m skill.

## 1. Scope

| In | Out |
| --- | --- |
| ElevCondResidualUNet residual on Nepal 21×37 | Diffusion / CorrDiff / leftover-target |
| Channels t2m + u10m + v10m (joint loss) | FCN3 weight FT / Tier-B |
| Thick-2 ICs / pairs / leads +24/+72/+120 | IMDAA / G1 claims |
| Score + gate **winds** (new) | Overwriting living residual dir |
| | Blocking on CDS |

## 2. What stays frozen (LOCKED)

| Component | Status |
| --- | --- |
| FCN3 checkpoint | **FROZEN** |
| Living residual `v1_2b_thick2_train/` | **FROZEN** — do not overwrite; compare against |
| `v1_diff/` null artifact | Keep; do not delete |
| Thick-2 bars + year hard-lock | **FROZEN** |
| G0 / Tier-0 thick-2 / prior Tier-A dirs | **FROZEN** |

## 3. Architecture / train knobs

| Knob | Living v1.2b thick-2 | **v1.3 joint** |
| --- | --- | --- |
| Model | ElevCondResidualUNet base=16, depth=3, dropout=0.20 | **Same** (no capacity bump on first pass) |
| Out channels | 3 (t2m,u,v) | **Same** |
| `channel_weights` | `[2.0, 0.5, 0.5]` | **`[1.5, 1.5, 1.5]`** (joint) |
| `w_120` / lead weights | 2.0 · 1/1/2.0 | **Same** |
| weight_decay / elev_boost | 1e-3 / 1.5 | **Same** |
| ckpt composite | reject if val +120h t2m > raw | **Same** · raw = **2.178842** |
| patience / max epochs / seed | 80 / 400 / 42 | **Same; log seed in JSON** |
| Target | ERA5 interim − FCN3 | **Same** (full residual, not leftover-of-living) |

Do **not** condition on / subtract the living residual mean — this is a **parallel residual retrain**, not diffusion leftover.

## 4. Data / pairs

- Manifest: thick-2 `holdout_expand=v2` (12/16/16).
- Leads: +24/+72/+120 h.
- Labels: `provisional_years=false` · `year_split_frozen=true` · `claim_level=interim_era5` · `g1_claimable=false` · `bars_set=thick2` · `diffusion=false`.

## 5. Pre-train freeze (CPU — required)

Before `--train`, score on thick-2 val/test:

1. **Raw FCN3** u10m / v10m / wind-vector RMSE (pooled over leads 24/72/120; same pooling as t2m living).
2. **Living residual** ckpt on the same wind metrics (t2m already known).

Write `runs/phase0/tier_a/v1_3_joint/living_wind_baseline.json` with exact numbers.  
**Wind-vector RMSE** = sqrt((u_err^2 + v_err^2)/2) grid-pooled, then lead-mean (document in JSON).

Those living wind numbers become the **vs-living wind bars** for PASS. Ping Leonard if pooling definition is ambiguous; otherwise proceed.

## 6. PASS / FAIL (LOCKED)

### A. Absolute floors (must clear — else FAIL)

| Gate | Rule |
| --- | --- |
| Val pooled t2m | **< 1.988588** |
| Test pooled t2m | **< 1.918383** |
| Val +120 h t2m | **≤ 2.178842** |
| Eligible ckpt | `n_eligible_saves ≥ 1` · reload=`composite_eligible` (unconstrained fallback ⇒ **auto FAIL**) |

### B. Protect living t2m (no material regress — else FAIL as upgrade)

| Gate | Rule |
| --- | --- |
| Val t2m vs living | **≤ 1.759883 + 0.020** (≤ **1.779883**) |
| Test t2m vs living | **≤ 1.7797 + 0.020** (≤ **1.7997**) |

Prefer strict improve on t2m; small regress within 0.02 K allowed if winds lift is clear.

### C. Promote on winds (required for INTERIM PASS as product upgrade)

| Gate | Rule |
| --- | --- |
| Val wind-vector | **strictly <** living val wind-vector (from §5 baseline) |
| Test wind-vector | **strictly <** living test wind-vector |
| Also beat raw | Val and test wind-vector **strictly <** raw FCN3 |

If A+B pass but **C fails** → **FAIL as upgrade** · label `joint_improves_winds=false` · living residual stays canonical.  
If A+B+C pass → **INTERIM PASS** · may promote `v1_3_joint/` as new living residual (Leonard call).

### Secondary (report only)
- Per-lead / elev bands for t2m and u/v  
- Component u10m / v10m RMSE (honesty)  
- Test +120 h t2m vs raw (not a hard gate)

### Kill / iterate
- NaNs / OOM → shrink batch/accum; do not touch FCN3.  
- Pass A but fail B → channel rebalance too aggressive; stop and call Leonard (do not silently retune).  
- Pass A+B, fail C → null on winds; living stays; **no auto diffusion**.  
- No Tier-B.

## 7. Dirs (LOCKED)

| Item | Path |
| --- | --- |
| Out | `runs/phase0/tier_a/v1_3_joint/` |
| Config | `configs/tier_a_v1_3_joint.yaml` |
| Code | reuse `code/tier_a/v1_2b/` or thin `code/tier_a/v1_3_joint/` |
| Results | `.../tier_a_v1_3_joint_results.json` + `best_residual.pt` |
| Baseline | `.../living_wind_baseline.json` |
| Forbidden overwrite | `v1_2b_thick2_train/`, `v1_2b_thick2/`, `v1_2b/`, `v1_diff/`, G0, tier0 thick-2 |

## 8. Eng order (Howard)

1. **CPU:** config echoing bars + living path; wire wind metrics; write `living_wind_baseline.json`; dry-run forward — **no `--train`**.  
2. Ping Leonard with baseline path (quick confirm bars look sane).  
3. **GPU `--train`** only after Leonard OK on baseline (or if baseline matches recipe pooling and Manisha already greenlit train — default: **wait for Leonard baseline ACK**).  
4. Ping Leonard with results JSON for PASS/FAIL call.  
5. **CDS 611595 leave alone.**

## 9. Deferred forks (do not start without Manisha)

- Parallel-mean diffusion: target ERA5−FCN3; living residual **not** in target/conditioner.  
- Multi-seed (3–5) uncertainty table on living residual.  
- Write-up / pause Phase 3 (residual-only story).

## 10. Non-goals
- Leftover-target diffusion scaling  
- CorrDiff NVIDIA parity  
- Replacing living residual without wind lift  
- FCN3 weight FT / precip / G1  
