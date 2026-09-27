# FINAL residual v0 call (Leonard) — 2026-09-25

**Artifacts (Spark `~/fourcastnet`)**
- `runs/phase0/final_eval/final_residual_v0/final_residual_v0_results.json`
- `final_residual_v0_metrics.csv`
- `best_residual.pt` · MD5 **`586ab17b843757bb84e66e2e3af8dc01`**
- Recipe: `FINAL_TRAIN_RECIPE.md` · Bars: `FINAL_EVAL_BARS_CALL.md` · Protocol: `FINAL_EVAL_PROTOCOL.md`

**Run:** `status=train_ok` · early-stop ep **135** · best eligible ep **55** · `n_eligible_saves=19` · reload=`composite_eligible` · wall ~**399 s** · created **2026-09-25 ~16:25 PT**

**Arch:** ElevCondResidualUNet · `channel_weights=[1.5,1.5,1.5]` · `w_120=2.0` · FCN3 **FROZEN** · parallel residual (not leftover-diff) · years 1980–2019 / 2020–2021 / 2022–2025 · ICs 320/64/64

## Call

| Gate | Result |
| --- | --- |
| **A** vs raw FCN3 (+ eligible) | **PASS** — val t2m **1.819863** < 2.099229 · test **1.856390** < 2.182565 · val +120h **2.010414** ≤ 2.248841 · val WV **0.800354** < 0.890307 · test WV **0.825004** < 0.916351 · **19** eligible · `composite_eligible` |
| **B** vs Tier-0 t2m | **PASS** — val **1.819863** < 2.039476 · test **1.856390** < 2.104525 |
| **C** vs living on FINAL | **PASS** — val t2m **1.819863** < 1.982226 · test **1.856390** < 1.987826 · val WV **0.800354** < 0.845186 · test WV **0.825004** < 0.876129 |
| **`final_g1_candidate_pass`** | **YES — CONFIRM** (A ∧ B ∧ C) |
| **Promote living residual?** | **YES** |
| **`g1_claimable`** | **false** (unchanged — needs Manisha + explicit flip; do not auto-set) |

Independent verify used `metrics.*.rmse_tier_a` / `wind_vector.*.lead_mean` against `FINAL_EVAL_BARS_CALL.md` floats (not the JSON’s pre-echoed `pass` flags). IC id lists **byte-equal** to locked `final_eval_protocol_ics.json` (file MD5 `7de70ec8…`); living `v1_3_joint` MD5 **`c81a5a4c4f07ca0a52bb0bffee0045fa`** unchanged (`freeze_md5_unchanged=true`).

## Living residual (LOCKED after this call)

| Role | Path |
| --- | --- |
| **NEW living (FINAL protocol)** | `runs/phase0/final_eval/final_residual_v0/` |
| Prior interim living | `runs/phase0/tier_a/v1_3_joint/` → **historical interim** (still frozen; do not overwrite) |
| Thick-2 history | unchanged · report-only bridge |

## Headline (FINAL protocol 320/64/64 · gate leads 24/72/120)

| | Living `v1_3_joint` on FINAL | **final_residual_v0** | Δ |
| --- | ---: | ---: | ---: |
| Val t2m pooled | 1.982226 | **1.819863** | **−0.162** |
| Test t2m pooled | 1.987826 | **1.856390** | **−0.131** |
| Val +120 h t2m | 2.204377 | **2.010414** | **−0.194** |
| Val wind-vector lead-mean | 0.845186 | **0.800354** | **−0.045** |
| Test wind-vector lead-mean | 0.876129 | **0.825004** | **−0.051** |

Clear lift on both t2m and winds vs living scored on the same FINAL ICs — not a borderline ε call.

## Thick-2 legacy bridge — FILLED (report-only · continuity · not FAIL)

Sibling `runs/phase0/final_eval/final_residual_v0/thick2_bridge_results.json` (md5 ckpt **`586ab17b843757bb84e66e2e3af8dc01`**).

| | Interim living `v1_3_joint` (thick-2) | **final_residual_v0** on thick-2 | Δ |
| --- | ---: | ---: | ---: |
| Val t2m pooled | 1.770 | **1.883** | +0.113 |
| Test t2m pooled | 1.775 | **1.875** | +0.100 |

**Continuity / honesty only** — not a G1/FINAL promote path and **not** a FAIL against FINAL A∧B∧C (different protocol). `g1_claimable` remains **false**. Did not overwrite `v1_3_joint/` weights.

## Next eng (Howard)

1. GitHub + paper sync of FINAL residual v0 PASS + living pointer (this package).
2. Keep `v1_3_joint/` frozen historical; **`g1_claimable=false`** everywhere.
3. Do **not** flip `g1_claimable` until Manisha + Leonard explicit unlock.
4. Idle on next model / iterate unless Manisha assigns.

## Non-claims

- Not global WB2 / FCN3 SOTA · not ops replacement · not CorrDiff parity · no precip · no stations  
- `g1_claimable` still **false** until separate unlock  
- Interim thick-2 / `interim_era5` bars unchanged  
