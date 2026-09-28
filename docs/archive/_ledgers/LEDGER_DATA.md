# LEDGER — Data
Date extracted: 2026-09-27 PT

## Domain box
- 26–31°N / 80–89°E (CDS 31/80/26/89)
- (source: FINAL_EVAL_PROTOCOL.md, G1_CLAIMABLE_UNLOCK_CALL.md, REFERENCE.md)

## Truth / IC sources
- Truth: audited CDS ERA5 crop for gated years.
- ICs: global ARCO (not Nepal CDS crop as IC source).
- Nepal CDS crop = archive for later, not IC source.
- (source: FINAL_EVAL_PROTOCOL.md, STATUS_WHAT_WORKS_WHAT_FAILED.md)

## ERA5 coverage audit (2026-09-22 UTC stamp in file)
- Window 1980-01 → 2026-09
- Months scanned 561; Present 560; Missing 0; Partial 1; Holes 1
- Frontier contiguous from start: 2026-08
- Soft hole 2026-09 tip-partial excluded from gated years
- (source: ERA5_COVERAGE_AUDIT.md, FINAL_EVAL_PROTOCOL.md)

## FINAL year split (LOCKED)
| Split | Years | N ICs |
| Train | 1980–2019 | 320 |
| Val | 2020–2021 | 64 |
| Test | 2022–2025 | 64 |
| Excluded gated | 2026 | ungated tip / report-only |
- IC hashes in FINAL_EVAL_PROTOCOL.md / final_eval_protocol_ics.json
- file MD5 of ICs json (from results freeze): 7de70ec8a0b5b1e7f77ae21660751a51
- (source: FINAL_EVAL_PROTOCOL.md, FINAL_RESIDUAL_V0_CALL.md, final_residual_v0_results.json)

## Interim thick-2 (historical)
- Holdout 12/16/16; years 2018–21 / 2022 / 2023–24
- (source: YEAR_HARD_LOCK.md, TIER_A_V1_3_JOINT_CALL.md)

## Primary variables (regional residual)
- t2m, u10m, v10m (joint ElevCond residual)
- No precip claim in living FINAL
- (source: FINAL_RESIDUAL_V0_CALL.md, ERA5_COVERAGE_AUDIT.md)
