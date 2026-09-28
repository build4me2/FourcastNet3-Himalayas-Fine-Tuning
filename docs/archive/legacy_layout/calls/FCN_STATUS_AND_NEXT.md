# FCN3 Phase 0 — status & next (Leonard) — 2026-09-27 afternoon PT

**Manisha:** work on FourCastNet (Howard = FCN-only; TTA parked).

## Frozen state (do not overwrite)

| Layer | Status | Path / bar |
| --- | --- | --- |
| G0 verifying | **PASS claimable** | `g0_verifying_results.json`; adapters ≤+5% CRPS; PSD hard &lt;0.25 |
| Tier-0 thin | historical | val 1.948661 / test 1.850338 |
| Tier-0 expand-v1 | historical | val 1.987014 / test 1.897298 / +120h ≤ 2.131516 |
| Tier-0 **thick-2** | historical beat-this (interim) | val **&lt; 1.988588** / test **&lt; 1.918383** / +120h ≤ **2.178842** |
| Tier-A v0 | best **test pooled** champ (1.733 thin) | `tier_a/v0/` |
| Tier-A v1.1 | first +120h-compliant PASS (thin) | `tier_a/v1_1/` |
| Tier-A v1.1 expand ZS | INTERIM PASS (expand-v1 hist.) | val 1.808 / test 1.826 / +120h 1.998 |
| Tier-A v1.2b | INTERIM PASS (expand train) → **historical** | `tier_a/v1_2b/` · still frozen |
| Tier-A v1.2b **thick-2 ZS** | INTERIM PASS → **historical** (ZS demoted) | val 1.829 / test 1.897 / +120h 2.015 · `tier_a/v1_2b_thick2/` |
| Tier-A v1.2b **thick-2 retrain** | INTERIM PASS → **historical** | val **1.760** / test **1.780** / +120h **1.962** · `tier_a/v1_2b_thick2_train/` |
| Tier-A **v1.3 joint** | INTERIM PASS → **historical interim** (still frozen) | val **1.770** / test **1.775** (16-IC) / +120h **1.982** · WV **0.69560 / 0.73877** · `tier_a/v1_3_joint/` · md5 `c81a5a4c…` |
| Tier-A **v1-diff** | **NULL — do not promote** | `tier_a/v1_diff/` · keep as null artifact |
| **FINAL residual v0** | **PASS A∧B∧C · PROMOTE · living residual (FINAL protocol)** | val t2m **1.8199** / test **1.8564** / +120h **2.0104** · WV **0.8004 / 0.8250** · `final_eval/final_residual_v0/` · md5 `586ab17b…` |
| Years (FINAL) | 1980–2019 / 2020–2021 / 2022–2025 · ICs 320/64/64 | `FINAL_EVAL_PROTOCOL.md` |
| Labels | living FINAL **`g1_claimable=true`** (unlock 2026-09-27); interim historical stays `interim_era5` | flip only via Leonard call |

Calls: `FINAL_RESIDUAL_V0_CALL.md` (living promote), `TIER_A_V1_3_JOINT_CALL.md` (interim historical), `TIER_A_V1_DIFF_CALL.md` (NULL), `FINAL_EVAL_BARS_CALL.md`, `FINAL_EVAL_PROTOCOL.md`, `FINAL_TRAIN_RECIPE.md`.

## Living residual (LOCKED)

| Role | Path |
| --- | --- |
| **Living (FINAL protocol / canonical)** | `runs/phase0/final_eval/final_residual_v0/` (`best_residual.pt` · md5 `586ab17b843757bb84e66e2e3af8dc01`) |
| **HF model (public)** | https://huggingface.co/build4me2/fcn3-nepal-final-residual-v0 |
| Prior interim living (historical) | `runs/phase0/tier_a/v1_3_joint/` · **still frozen** · md5 `c81a5a4c4f07ca0a52bb0bffee0045fa` |
| Prior thick-2 living (historical) | `runs/phase0/tier_a/v1_2b_thick2_train/` · still frozen |
| v1-diff (null artifact) | `runs/phase0/tier_a/v1_diff/` · **do not delete** · do not promote |
| Prior expand-train v1.2b | `v1_2b/` → historical (still frozen) |
| Thick-2 ZS | `v1_2b_thick2/` → historical (ZS demoted) |
| v1.2 FAIL | `v1_2/` → keep |

Do **not** overwrite living residual (`final_residual_v0/`), interim historical (`v1_3_joint/`), prior living (`v1_2b_thick2_train/`), or v1_diff results.

## Recommended next (default)

**Idle pending Manisha.** Living residual = `final_residual_v0/`.  
**`g1_claimable=true`** — Leonard unlock 2026-09-27 (`G1_CLAIMABLE_UNLOCK_CALL.md`). Allowed: regional FINAL G1 candidate clearing A∧B∧C. Hard non-claims: no global SOTA/ops/CorrDiff/precip/stations; thick-2 report-only; HF: https://huggingface.co/build4me2/fcn3-nepal-final-residual-v0 .  
Interim `v1_3_joint/` preserved as historical.  
Optional non-blocking: thick-2 bridge score (WARN-missing from results).  
No GPU / no next model unless assigned.

### FINAL residual v0 promote (headline — FINAL protocol)
- t2m: val **1.8199** / test **1.8564** / +120h **2.0104** · eligible **19** · `composite_eligible`
- wind-vector lead-mean: val **0.8004** / test **0.8250**
- Gates: **A∧B∧C PASS** per `FINAL_RESIDUAL_V0_CALL.md` · `final_g1_candidate_pass=YES` · **`g1_claimable=true`** (unlock call)

### Interim v1.3 joint (historical — 16-IC locked claims)
- t2m: val **1.770** / test **1.775** / +120h **1.982** · WV **0.69560 / 0.73877**
- Still frozen; do not overwrite weights

### v1-diff null (headline)
- one_step 0-noise: val ~1.7598 / test ~1.7799 / +120h ~1.962 (≈ prior living; Δ ~1e-5)
- ens K=4: **worse** · `diff_improves_residual=false`

## Alternatives (if Manisha overrides)

| Option | When |
| --- | --- |
| **Thick-2 bridge score** | Optional continuity table; non-blocking |
| **Different Phase-3 fork** | Explicit assign — not more leftover-target scaling |
| **Pause eng** | Status-only; Howard idle on FCN (current default) |

## Non-goals until ordered
- TTA harness (Howard not assigned)
- FCN3 weight FT / Tier-B
- IMDAA / global SOTA / ops / CorrDiff / precip / stations (HF published for regional FINAL G1 candidate only — https://huggingface.co/build4me2/fcn3-nepal-final-residual-v0)
- Scaling leftover-target diffusion further (STOP)
- Overwriting frozen artifacts (incl. living `final_residual_v0/`, interim `v1_3_joint/`, prior `v1_2b_thick2_train/`, null `v1_diff/`)
- Inventing new bars

## Next eng (Howard)
1. Mirror Leonard `FINAL_RESIDUAL_V0_CALL.md` · echo living = `final_residual_v0/` in status docs — **this pass**.
2. Keep `v1_3_joint/` frozen historical; living FINAL **`g1_claimable=true`** per unlock; interim configs stay false.
3. Optional: thick-2 bridge when convenient.
4. Idle pending Manisha.
