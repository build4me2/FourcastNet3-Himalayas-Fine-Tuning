# 01 — Plan (goals, pathway, gates, locks)

**Consolidation date:** 2026-09-27 PT  
**Living residual:** `runs/phase0/final_eval/final_residual_v0/`  
**Sources:** `FINETUNE_PATHWAY.md`, `FINETUNE_METHOD_DESIGN.md`, `FINAL_EVAL_PROTOCOL.md`, `FINAL_EVAL_BARS_CALL.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`, `PROJECT_SCOPE_AND_HISTORY.md`, `REFERENCE.md` (archived).

## 1. Goal (locked)

Regional **probabilistic** skill over Nepal / HKH / adjacent South Asian orography — elevated t2m and near-surface winds first — while **preserving** FCN3 quality invariants (probabilistic 6 h spherical ensembles, 72-channel state, anisotropic + spectral operators, spatial + spectral CRPS discipline).  
(source: `FINETUNE_METHOD_DESIGN.md`, `REFERENCE.md`)

## 2. Non-goals

| Non-goal | Why |
| --- | --- |
| Full FCN3 multi-step + large-ensemble weight FT on 2 Sparks | Infeasible at NVIDIA Stage1/2/FT scale |
| MSE-only dynamics FT / replace FCN3 with pure MSE UNet | Breaks Lead probabilistic picture |
| Label Tier-A as “FCN3 weight fine-tune” | G3 honesty unless Tier-B passes |
| Idea1 glacial-lake / GEE / wipe archaeology in FCN claim | Out of this tree’s narrative boundary |
| Global WB2/FCN3 SOTA or ops replacement claims | Explicit hard non-claims |

(source: `FINETUNE_METHOD_DESIGN.md`, `PROJECT_SCOPE_AND_HISTORY.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`)

## 3. Method family — RRCA-FD

**RRCA-FD** = Regionally Reweighted CRPS Adapter + Frozen Diagnostic.

| Tier | Role | What moves |
| --- | --- | --- |
| **Tier-0** | Elevation-aware bias baseline | No FCN3 weights |
| **Tier-A (Plan A)** | Frozen FCN3 + regional residual / diagnostic | Residual UNet (± diffusion experiments) |
| **Tier-B (Plan B)** | PEFT / LoRA on FCN3 | Gated; not the living FINAL path |

Living FINAL product is **Tier-A style**: frozen FCN3 + **ElevCondResidualUNet** joint residual on t2m+u10m+v10m.  
(source: `FINETUNE_METHOD_DESIGN.md`, `FINAL_RESIDUAL_V0_CALL.md`)

## 4. Quality invariants (I1–I6) and gates (G0–G3)

Invariants I1–I6 (probabilistic map, one-step HMM members, 72-ch + aux, anisotropic+spectral arch, dual CRPS, separate weights/bias/ERA5 inheritance) are **hard design gates**.  
(source: `FINETUNE_METHOD_DESIGN.md`, `REFERENCE.md`)

| Gate | Design pass rule | Living outcome (summary) |
| --- | --- | --- |
| **G0** | ≤~5% relative CRPS degrade vs base @ +15 d; no spectral collapse | Verifying path **PASS claimable** (adapters ≤~5% CRPS; PSD floor held) |
| **G1** | Beat frozen FCN3 crop **and** Tier-0 on regional metrics | FINAL residual v0 clears A∧B∧C on FINAL protocol; **`g1_claimable=true`** (regional FINAL G1 candidate) |
| **G2** | Calibration SSR≈1 at +24–120 h | **[MISSING:]** dedicated G2 claim not asserted for living FINAL in call cards |
| **G3** | Honest labeling | Living labeled as frozen FCN3 + ElevCond residual — not “FCN3 weight FT” |

(source: `FINETUNE_PATHWAY.md`, `FCN_STATUS_AND_NEXT.md`, `FINAL_RESIDUAL_V0_CALL.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`)

## 5. FINAL evaluation contract (LOCKED)

| Item | Value |
| --- | --- |
| Box | 26–31°N / 80–89°E |
| Years | Train **1980–2019** / Val **2020–2021** / Test **2022–2025** |
| ICs | **320 / 64 / 64** (`final_eval_protocol_ics.json`) |
| Gate leads | +24 / +72 / +120 h |
| Product | `final_g1_candidate_pass = A ∧ B ∧ C` |

Bars are frozen from measured `final_baselines.json` only — see `docs/04_RESULTS.md`.  
(source: `FINAL_EVAL_PROTOCOL.md`, `FINAL_EVAL_BARS_CALL.md`)

### Interim protocol (historical — do not confuse with FINAL)

Thick-2 / year-hard-lock: train **2018–2021** / val **2022** / test **2023–2024**; `claim_level=interim_era5`. Used for earlier Tier-A living (`v1_3_joint/`) and report-only thick-2 bridge.  
(source: `YEAR_HARD_LOCK.md`, `TIER_A_V1_3_JOINT_CALL.md`)

## 6. Scope boundary

**In scope:** FCN3 Nepal residual fine-tune claim tree (this repo / Spark `~/fourcastnet`).  
**Out of scope for FCN narrative:** Idea1 glacial-lake / GEE / pre-wipe Spark archaeology (2026-09-10).  
(source: `PROJECT_SCOPE_AND_HISTORY.md`)  
**[CONFLICT:]** that scope file still points living → `v1_3_joint/` and `g1_claimable=false` (stale vs 2026-09-27 living). Living plan uses FINAL residual v0 + unlock call.

## 7. Plan vs outcome (pathway)

| Plan (design) | Outcome (2026-09-27) |
| --- | --- |
| Bring up Phase 0 → Tier-0 → Tier-A on Spark | Done through FINAL residual v0 |
| Freeze box + years | Box + FINAL years locked; interim years separately locked |
| FINAL suite after ERA5 audit | Protocol + bars frozen; residual trained; A∧B∧C PASS |
| Promote + claim carefully | Living promoted; `g1_claimable=true`; HF published for residual weights |

Stale rows inside `FINETUNE_PATHWAY.md` “tracking dashboard” still say training/eval not started — **ignore those rows** relative to the living header and call cards.  
(source: `FINETUNE_PATHWAY.md` header vs dashboard; `LEDGER_CONFLICTS.md`)
