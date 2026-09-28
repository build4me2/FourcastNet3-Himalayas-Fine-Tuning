# LEDGER — Plan / pathway / gates
Date extracted: 2026-09-27 PT

## Goal
- Regional probabilistic skill over Nepal/HKH/South Asian orography.
- Preserve FCN3 invariants (6h ensembles, 72-ch, anisotropic+spectral, dual CRPS).
- (source: FINETUNE_METHOD_DESIGN.md, REFERENCE.md, FINETUNE_PATHWAY.md)

## Non-goals
- Full FCN3 multi-step + large-ensemble weight FT on 2 Sparks.
- MSE-only FT / replace FCN3 with pure MSE UNet.
- Claim "FCN3 weight FT" for Tier-A only (G3).
- Idea1 glacial-lake/GEE (out of FCN claim tree).
- (source: FINETUNE_METHOD_DESIGN.md, PROJECT_SCOPE_AND_HISTORY.md)

## Method family
- RRCA-FD: Regionally Reweighted CRPS Adapter + Frozen Diagnostic.
- Tier-0: frozen FCN3 + elevation-aware bias.
- Tier-A (Plan A): frozen FCN3 + regional residual/diagnostic.
- Tier-B (Plan B gated): LoRA / PEFT — not default.
- (source: FINETUNE_METHOD_DESIGN.md)

## Gates (design)
| Gate | Pass rule (design) |
| G0 | Global CRPS/SSR + spectra @ +15d: ≤~5% relative CRPS degrade; no spectral collapse |
| G1 | Nepal skill beat frozen FCN3 crop and Tier-0; elev-banded t2m improves |
| G2 | Calibration SSR≈1 at +24–120h |
| G3 | Honest labeling |
(source: FINETUNE_PATHWAY.md, REFERENCE.md)

## FINAL protocol locks (outcome living)
- Box 26–31N / 80–89E
- Years train 1980–2019 / val 2020–2021 / test 2022–2025
- ICs 320/64/64; gate leads 24/72/120
- (source: FINAL_EVAL_PROTOCOL.md)

## Interim year hard-lock (historical; not FINAL)
- Train 2018–2021 / val 2022 / test 2023–2024
- claim_level=interim_era5
- (source: YEAR_HARD_LOCK.md)

## Living claim state (2026-09-27)
- Living: runs/phase0/final_eval/final_residual_v0/
- PASS A∧B∧C; g1_claimable=true
- HF: https://huggingface.co/build4me2/fcn3-nepal-final-residual-v0
- (source: FINAL_RESIDUAL_V0_CALL.md, G1_CLAIMABLE_UNLOCK_CALL.md, LIVING_INDEX.md, HF_RELEASE_FINAL_RESIDUAL_V0.md)
