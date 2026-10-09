# 04 — Results (measured outcomes)

**Consolidation date:** 2026-09-27 PT  
**Living run:** `runs/phase0/final_eval/final_residual_v0/`  
**Sources:** `FINAL_RESIDUAL_V0_CALL.md`, `final_residual_v0_results.json`, `FINAL_EVAL_BARS_CALL.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`, `HF_RELEASE_FINAL_RESIDUAL_V0.md`, `STATUS_WHAT_WORKS_WHAT_FAILED.md`, `FCN_STATUS_AND_NEXT.md`, `TIER_A_V1_3_JOINT_CALL.md`.

## 1. Living product (FINAL protocol)

| Field | Value |
| --- | --- |
| Protocol | FINAL · ICs **320/64/64** · years **1980–2019 / 2020–2021 / 2022–2025** · leads **24/72/120** |
| Architecture | Frozen FCN3 + ElevCondResidualUNet parallel residual |
| Checkpoint MD5 | **`586ab17b843757bb84e66e2e3af8dc01`** |
| HF | https://huggingface.co/build4me2/fcn3-himalayas-final-residual-v0 (Apache-2.0; `base_model: nvidia/fourcastnet3`, adapter) |
| `final_g1_candidate_pass` | **YES** (A ∧ B ∧ C) |
| `g1_claimable` | **`true`** (Leonard unlock 2026-09-27) |

(source: `FINAL_RESIDUAL_V0_CALL.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`, `HF_RELEASE_FINAL_RESIDUAL_V0.md`; Spark `md5sum` 2026-09-27)

## 2. Exact headlines (FINAL)

| Metric | Exact | Display roundings used in status |
| --- | ---: | ---: |
| Val t2m pooled | **1.8198625586531565** | 1.819863 / 1.8199 |
| Test t2m pooled | **1.8563898906179215** | 1.856390 / 1.8564 |
| Val +120 h t2m | **2.010413966886577** | 2.010414 / 2.0104 |
| Val WV lead-mean | **0.8003535884784455** | 0.800354 / 0.8004 |
| Test WV lead-mean | **0.8250037602833951** | 0.825004 / 0.8250 |

(source: `FINAL_RESIDUAL_V0_CALL.md`, `final_residual_v0_results.json`)

### Lift vs prior living scored on FINAL ICs

| | Living `v1_3_joint` on FINAL | **final_residual_v0** | Δ |
| --- | ---: | ---: | ---: |
| Val t2m pooled | 1.982226 | **1.819863** | **−0.162** |
| Test t2m pooled | 1.987826 | **1.856390** | **−0.131** |
| Val +120 h t2m | 2.204377 | **2.010414** | **−0.194** |
| Val WV | 0.845186 | **0.800354** | **−0.045** |
| Test WV | 0.876129 | **0.825004** | **−0.051** |

(source: `FINAL_RESIDUAL_V0_CALL.md`)

## 3. Gate table (A ∧ B ∧ C)

| Gate | Result | Key comparisons (call) |
| --- | --- | --- |
| **A** vs raw FCN3 (+ eligible) | **PASS** | val t2m **1.819863** < 2.099229 · test **1.856390** < 2.182565 · val +120h **2.010414** ≤ 2.248841 · val WV **0.800354** < 0.890307 · test WV **0.825004** < 0.916351 · **19** eligible · `composite_eligible` |
| **B** vs Tier-0 t2m | **PASS** | val **1.819863** < 2.039476 · test **1.856390** < 2.104525 |
| **C** vs living on FINAL | **PASS** | val t2m **1.819863** < 1.982226 · test **1.856390** < 1.987826 · val WV **0.800354** < 0.845186 · test WV **0.825004** < 0.876129 |

Authoritative bar floats (full precision) are in archived `FINAL_EVAL_BARS_CALL.md` §1–2.  
(source: `FINAL_RESIDUAL_V0_CALL.md`, `FINAL_EVAL_BARS_CALL.md`)

## 4. Allowed claim language

OK (with protocol citation):

1. **G1 candidate (regional FINAL):** On the locked FINAL eval protocol, `final_residual_v0` clears A∧B∧C vs raw FCN3, Tier-0 t2m, and prior living residual scored on the same FINAL ICs.
2. Headline numbers above (FINAL protocol only).
3. Method label: frozen FCN3 + ElevCondResidualUNet residual (joint t2m+u10m+v10m), Nepal box 26–31°N / 80–89°E (central Himalaya only), CDS ERA5 truth.

(source: `G1_CLAIMABLE_UNLOCK_CALL.md`)

## 5. Hard non-claims

- Not global FCN3 / WeatherBench-2 SOTA  
- Not operational NWP replacement  
- Not CorrDiff / diffusion parity  
- No precip · no station / IMDAA claim  
- Not a license to overwrite `v1_3_joint/` or retrain under softer bars  
- Thick-2 bridge is **report-only** — not a second G1 path  

(source: `G1_CLAIMABLE_UNLOCK_CALL.md`, `FINAL_RESIDUAL_V0_CALL.md`)

## 6. Thick-2 legacy bridge (report-only · continuity)

Same ckpt md5 **`586ab17b…`** scored on thick-2 protocol:

| | Interim living `v1_3_joint` (thick-2) | **final_residual_v0** on thick-2 | Δ |
| --- | ---: | ---: | ---: |
| Val t2m pooled | 1.770 | **1.883** | +0.113 |
| Test t2m pooled | 1.775 | **1.875** | +0.100 |

STATUS echo (more digits): val **1.883181** / test **1.875323**; WV val **0.701070** / test **0.741207**.  
**Not** a FAIL against FINAL A∧B∧C (different protocol). Did not overwrite `v1_3_joint/` weights.  
(source: `FINAL_RESIDUAL_V0_CALL.md`, `STATUS.md`)

## 7. Failures / nulls preserved

| Run | Verdict | Notes |
| --- | --- | --- |
| Tier-A **v1.2** | **FAIL** | +120 h gate miss; keep historical |
| Tier-A **v1-diff** | **NULL** | Leftover-target diffusion ≈ living residual (Δ ~1e-5); ensemble worse; **do not promote** |
| Earlier Tier-0 / Tier-A expands | Historical beat-this records | See `docs/05_IMPLEMENTATION_HISTORY.md` |

(source: `STATUS_WHAT_WORKS_WHAT_FAILED.md`, `FCN_STATUS_AND_NEXT.md`)

## 8. Interim historical headlines (not FINAL G1)

`v1_3_joint/` on thick-2 16-IC locked claims: val t2m **1.770** / test **1.775** / +120h **1.982** · WV **0.69560 / 0.73877** · md5 **`c81a5a4c4f07ca0a52bb0bffee0045fa`**.  
(source: `TIER_A_V1_3_JOINT_CALL.md`, `LIVING_INDEX.md`)

## 9. Doc conflicts affecting results narrative

- **[CONFLICT:]** `STATUS_WHAT_WORKS_WHAT_FAILED.md` still says “no HF” while `HF_RELEASE_FINAL_RESIDUAL_V0.md` / `LIVING_INDEX.md` publish the HF URL — living results doc follows the HF release note.
- **[MISSING:]** G2 calibration claim for living FINAL not asserted in unlock/call cards.
