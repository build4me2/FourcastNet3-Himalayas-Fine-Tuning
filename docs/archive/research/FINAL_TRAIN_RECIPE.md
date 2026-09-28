# FINAL residual train recipe (Leonard) — 2026-09-22

**Manisha unlock:** FINAL residual train GO.  
**Protocol:** `FINAL_EVAL_PROTOCOL.md` + `final_eval_protocol_ics.json`  
**Bars:** `FINAL_EVAL_BARS_CALL.md` (A∧B∧C from measured `final_baselines.json`)  
**Architecture baseline:** living **`v1_3_joint`** hyperparams (frozen FCN3 + ElevCondResidualUNet)  
**Do not overwrite:** `runs/phase0/tier_a/v1_3_joint/` (interim living) until Leonard PASS promote

## 1. Method (one sentence)

Parallel residual retrain on FINAL years/ICs: same ElevCondResidualUNet as `v1_3_joint`, joint channel weights, truth = audited CDS ERA5 crop; score and gate with FINAL A∧B∧C (no ε_protect).

## 2. Frozen locks

| Component | Status |
| --- | --- |
| FCN3 weights | **FROZEN** |
| Interim living `v1_3_joint/` | **FROZEN** compare — do not overwrite |
| FINAL protocol years/ICs/hashes | **FROZEN** |
| FINAL bars A∧B∧C | **FROZEN** |
| Thick-2 interim bars / years | **FROZEN** (legacy bridge only) |
| Diffusion / FCN3 FT | **OUT** |

## 3. Architecture / train knobs (= v1.3 joint)

| Knob | Value |
| --- | --- |
| Model | ElevCondResidualUNet · base=16 · depth=3 · dropout=0.20 |
| Channels out | 3 (t2m, u10m, v10m) |
| `channel_weights` | **`[1.5, 1.5, 1.5]`** |
| `w_120` / lead weights | **2.0** · 1 / 1 / 2.0 on gate leads |
| weight_decay / elev_boost | 1e-3 / 1.5 |
| ckpt | composite; **reject** if val +120 h t2m **>** raw A3 (**2.2488412332039327**) |
| patience / max epochs / seed | **80** / **400** / **42** (log seed) |
| Target | ERA5 − FCN3 (full residual; **not** leftover-of-living) |
| Gate leads in loss | +24 / +72 / +120 h |
| Report leads at score | +24 / +48 / +72 / +96 / +120 h |

No capacity bump on first FINAL pass unless this run fails C and Manisha assigns iterate.

## 4. Data

- Years: train **1980–2019** / val **2020–2021** / test **2022–2025** (2026 excluded)
- ICs: 320 / 64 / 64 — hash-verify against `final_eval_protocol_ics.json` on load
- Box 26–31°N / 80–89°E · CDS ERA5 truth
- Labels: `claim_level=final_candidate` · **`g1_claimable=false`** until promote · `diffusion=false` · `bars_set=final_eval`

## 5. Dirs (LOCKED)

| Item | Path |
| --- | --- |
| Out | `runs/phase0/final_eval/final_residual_v0/` |
| Config | `configs/final_residual_v0.yaml` |
| Code | reuse `code/tier_a/v1_3_joint/` or thin `code/final_eval/final_residual_v0/` |
| Results | `.../final_residual_v0_results.json` + `best_residual.pt` |
| Forbidden overwrite | `tier_a/v1_3_joint/`, thick-2 living history, `final_baselines.json`, protocol IC JSON |

## 6. PASS / FAIL

Score with same pooling as `final_baselines.json`.  
**PASS = A ∧ B ∧ C** per `FINAL_EVAL_BARS_CALL.md` (exact floats).  
Also run **legacy thick-2 bridge** re-score (report-only).

On PASS: Leonard may promote to new living / consider `g1_claimable` — **not automatic**.  
On FAIL: stop; iterate only with Leonard + Manisha.

## 7. Eng order (Howard)

1. Echo this recipe in config; hash-verify ICs.  
2. Dry-run / smoke OK → **GPU `--train`**.  
3. Score FINAL protocol + thick-2 bridge → ping Leonard with results JSON.  
4. Do **not** overwrite `v1_3_joint/` until Leonard promote call.

## 8. Non-goals

- Leftover diffusion · CorrDiff · precip · FCN3 weight FT · silently widening bars  
