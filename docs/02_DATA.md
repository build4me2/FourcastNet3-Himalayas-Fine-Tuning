# 02 — Data (box, ERA5, years, ICs)

**Consolidation date:** 2026-09-27 PT  
**Sources:** `FINAL_EVAL_PROTOCOL.md`, `ERA5_COVERAGE_AUDIT.md`, `YEAR_HARD_LOCK.md`, `STATUS_WHAT_WORKS_WHAT_FAILED.md`, `final_eval_protocol_ics.json`, `FINAL_RESIDUAL_V0_CALL.md`.

## 1. Geographic box (LOCKED)

| Field | Value |
| --- | --- |
| Lat / lon | **26–31°N / 80–89°E** |
| CDS order note | `31/80/26/89` in crop configs |
| Focus variables (living residual) | t2m, u10m, v10m |

(source: `FINAL_EVAL_PROTOCOL.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`, `ERA5_COVERAGE_AUDIT.md`)

## 2. Truth vs initial conditions

| Role | Source | Note |
| --- | --- | --- |
| **Truth (eval / train targets)** | Audited **CDS ERA5** Nepal crop | Gated years hole-free through 2025 per protocol |
| **ICs** | **Global ARCO** | Nepal CDS crop is **not** the IC source |
| Nepal CDS crop | Archive for later | Explicitly not IC source |

(source: `FINAL_EVAL_PROTOCOL.md`, `STATUS_WHAT_WORKS_WHAT_FAILED.md`)

## 3. ERA5 coverage audit

Audit stamp in file: **2026-09-22T01:12:14.392779+00:00** (UTC).  
Raw dir (Spark): `~/fourcastnet/data/era5/raw`. Window **1980-01 → 2026-09**.

| Metric | Value |
| --- | ---: |
| Months scanned | 561 |
| Present | 560 |
| Missing | 0 |
| Partial | 1 |
| Holes (non-present) | 1 |
| Frontier contiguous from start | **2026-08** |
| Last present month (any) | 2026-08 |

Protocol consequence: **2026** is excluded from gated FINAL splits (Sep tip-partial); contiguous full months through **2026-08** used as audit basis for freezing FINAL years.  
(source: `ERA5_COVERAGE_AUDIT.md`, `FINAL_EVAL_PROTOCOL.md`)

Surface diagnostic map in audit: t2m←`t2m`; u10m←`u10`; v10m←`v10`.  
(source: `ERA5_COVERAGE_AUDIT.md`)

## 4. FINAL year split (LOCKED integers)

| Split | Years | Calendar span |
| --- | --- | --- |
| **Train** | **1980–2019** | 40 years |
| **Val** | **2020–2021** | 2 years |
| **Test** | **2022–2025** | 4 years |
| **Excluded from gated splits** | **2026** | Ungated tip / report-only if used later |

Membership rule: IC **calendar year** determines split. No Dec(Y−1) pulled into year Y.  
(source: `FINAL_EVAL_PROTOCOL.md`)

## 5. FINAL IC recipe and counts (LOCKED)

| Knob | Value |
| --- | --- |
| Init hours | 00 and 12 UTC |
| Min separation | ≥ 5 days within each split |
| Counts | Train **320** · Val **64** · Test **64** |
| Season balance | Prefer-balanced DJF/MAM/JJA/SON |

### IC list hashes (SHA-256 of sorted IDs + trailing newline)

| Split | N | sha256 |
| --- | ---: | --- |
| Train | 320 | `e119c288bfca9d170a1692cbc566948e3d21797d727b4b060f8c418c6ce9d86d` |
| Val | 64 | `42bffc2e3cad252a0b8217e6a2f8919e4b681aa17554a2c8591bee04a4b4c403` |
| Test | 64 | `bd50f2fb85f25316fe0a90d1b92867fcb98bc4d5bc46c497169890eef4204828` |

Full ID lists: `final_eval_protocol_ics.json` (Spark / local protocol artifact; not in git).  
Freeze echo MD5 of that JSON from FINAL residual results: **`7de70ec8a0b5b1e7f77ae21660751a51`**.  
(source: `FINAL_EVAL_PROTOCOL.md`, `FINAL_RESIDUAL_V0_CALL.md`, `final_residual_v0_results.json`)

## 6. Interim thick-2 / year hard-lock (historical)

| Split | Years | Role |
| --- | --- | --- |
| Train | **2018–2021** | Interim thick-2 era |
| Val | **2022** | Interim |
| Test | **2023–2024** | Interim |

Labels: `provisional_years=false`, `year_split_frozen=true`, `claim_level=interim_era5`.  
Holdout IC staging referred as thick-2 **12 / 16 / 16**.  
This is **not** the FINAL year lock — keep distinct in all claims.  
(source: `YEAR_HARD_LOCK.md`, `FCN_STATUS_AND_NEXT.md`, `TIER_A_V1_3_JOINT_CALL.md`)

## 7. Other streams (intent / status)

| Stream | Role | Living FINAL status |
| --- | --- | --- |
| DEM elevation | Elev conditioning / bands | Used in ElevCond residual |
| IMDAA | Optional high-res target | **Not** used in living FINAL claim |
| IMERG / stations | Precip / obs verify | **No** precip/station claim |

(source: `REFERENCE.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`)

## 8. Missing data notes

- **[MISSING:]** IMDAA/station skill tables for living FINAL (explicitly non-claimed).
- **[MISSING:]** Re-print of full 320/64/64 IC ID lists in this file — authoritative lists remain in archived `final_eval_protocol_ics.json`.
