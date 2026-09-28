# FINAL eval suite recipe (Leonard) — 2026-09-16

**Lit base:** `EVAL_BENCHMARKS_AND_FINAL_SUITE.md` (Sheldon)  
**Call:** `FINAL_EVAL_SUITE_CALL.md` (this freeze)  
**Living interim (unchanged):** `runs/phase0/tier_a/v1_3_joint/` · thick-2 · `interim_era5` · `g1_claimable=false`  
**Status:** Protocol + hardness **FROZEN**. Numeric beat-this **NOT** invented — freeze from `final_baselines.json` after archive audit. **No train** until Manisha unlocks after protocol + baselines land.

## 0. What this is

Firm **FINAL / G1-candidate** evaluation for the completely finetuned Nepal FCN3 residual (RRCA-FD) **after** full CDS ERA5 crop. Tests whether we got the model we wanted — not soft interim bars.

Interim thick-2 stays frozen forever as historical. FINAL is a **new protocol**, not a quiet rewrite of thick-2.

## 1. Interim → FINAL mapping (LOCKED)

| Role | Decision |
| --- | --- |
| Thick-2 beat-this (1.988588 / 1.918383 / 2.178842) | **KEEP** — never weaken; `interim_era5` only |
| Year hard-lock 2018–21 / 2022 / 2023–24 | **KEEP** for interim + **legacy bridge** re-scores |
| Living `v1_3_joint/` headlines | **KEEP** as interim promote record |
| Claim protocol for publishable / G1 | **REPLACE** with this FINAL suite |
| IC density / season balance | **TIGHTEN** (hard floors below) |
| Numeric bars | **RE-MEASURE** on FINAL protocol (raw / Tier-0 / living) then freeze |

**Legacy bridge (required every FINAL ckpt):** re-score on thick-2 12/16/16 with locked wind-vector def → continuity table vs living 1.770 / 1.775 / 1.982 and WV 0.69560 / 0.73877. Bridge is **report + honesty**, not a second promote path.

## 2. Prerequisites (hard — block FINAL train)

1. **Archive audit PASS** — CDS Nepal crop complete for intended FINAL years; no holes on t2m / u10m / v10m (document coverage JSON).  
2. **`FINAL_EVAL_PROTOCOL.md`** written with concrete years + IC list hashes (this recipe’s structure + measured years).  
3. **`final_baselines.json`** scored for raw FCN3, Tier-0, living `v1_3_joint` on that protocol.  
4. Leonard **`FINAL_EVAL_BARS_CALL.md`** freezes floats from that JSON (strict inequalities).  
5. Manisha greenlight for GPU FINAL train.

Until (1)–(4): `g1_claimable=false`; no FINAL train.

## 3. Domain / truth / pooling (LOCKED)

| Item | Spec |
| --- | --- |
| Box | 26–31°N / 80–89°E |
| Truth | Full CDS ERA5 crop (not interim ARCO-only) once audit PASS |
| Pool | Same grid-pool as living residual JSON; document `n` cells × ICs |
| Elev bands | Interim bins; **report-only** (not hard gate unless later promote) |
| Primary pool | Unweighted box (continuity); lat-weight = sensitivity column |
| Wind-vector | \(\sqrt{\mathrm{mean}((u_{\mathrm{err}}^2+v_{\mathrm{err}}^2)/2)}\) grid-pooled → **lead-mean** over gate leads |

## 4. Variables & metrics

### P0 — hard gates (required)

| Variable | Metric | Gate role |
| --- | --- | --- |
| **t2m** | Box-pooled RMSE (K) | Val primary / test secondary |
| **t2m** | Val +120 h RMSE | Hard vs measured raw +120 h |
| **Wind-vector** | Locked lead-mean WV RMSE | Val **and** test promote vs raw + living |

Also report: bias; u10m / v10m component RMSE; per-lead {24,72,120} and report leads.

### P1 — report (not first G1 blocker)

| Metric | When |
| --- | --- |
| Elev-band RMSE | Always |
| Full lead curve | +24 / +48 / +72 / +96 / +120 h |
| ACC(t2m), ACC(winds) | If regional climato built |
| Residual PSD / spectra vs ERA5 & raw FCN3 | Diagnostic (blur honesty) |

### P2 — deferred (separate unlock)

Precip SEEPS/CSI · stations / IMDAA · CRPS/spread–skill (unless multi-member path returns) · extremes quantile/Brier as **secondary** only once IC density supports it.

## 5. Leads (LOCKED)

| Set | Leads |
| --- | --- |
| **Gate** | +24 / +72 / +120 h |
| **Report** | +24 / +48 / +72 / +96 / +120 h |

## 6. Year protocol (structure FROZEN; years after audit)

| Split | Structure rule | Interim (legacy only) |
| --- | --- | --- |
| Train | Longest contiguous verified CDS years **before** val; never touch val/test years | 2018–2021 |
| Val | ≥ **1** full held-out calendar year | 2022 |
| Test | ≥ **2** held-out years; prefer distinct monsoon / ENSO flavor if archive allows | 2023–2024 |

**Rules:** Audit before flipping. No silent train→val leakage. Exact year integers land in `FINAL_EVAL_PROTOCOL.md` after audit — not invented here.

## 7. IC density & seasons (HARD floors)

| Split | Min ICs | Season floor | Spacing / init |
| --- | --- | --- | --- |
| Train | ≥ **48** | Prefer balanced; not gated | Prefer 00 & 12 UTC; ≥ **5 d** separation |
| Val | ≥ **64** | ≥ **8** ICs each in DJF / MAM / JJA / **SON** | Same |
| Test | ≥ **64** | ≥ **8** ICs each in DJF / MAM / JJA / **SON** | Same |

Fixes interim test SON soft miss (was 3). If GPU forces a temporary lower floor, Leonard must amend this recipe in writing — **no silent shrink**.

IC lists hashed (sha256 of sorted IC IDs) in protocol file before train.

## 8. Baselines (always re-score on FINAL protocol)

| Baseline | Role |
| --- | --- |
| Raw FCN3 | Absolute floor |
| Tier-0 | Lightweight adapter reference |
| Living `v1_3_joint` | Continuity / promote bar |
| Optional | Persistence / climatology (sanity) |

Same ICs, leads, pooling, wind def for all.

## 9. Pass / fail rule schema (numeric TBD from JSON)

After `final_baselines.json` exists, Leonard freezes floats. **Schema (LOCKED now):**

### A — Absolute vs raw FCN3 (hard FAIL if any miss)

1. Val t2m pooled **strictly <** measured raw val t2m  
2. Test t2m pooled **strictly <** measured raw test t2m  
3. Val +120 h t2m **≤** measured raw val +120 h  
4. Val & test wind-vector **strictly <** measured raw WV  
5. `n_eligible_saves ≥ 1` · reload=`composite_eligible` (reject if val +120 h > raw)

### B — vs Tier-0 (hard)

6. Val t2m **strictly <** measured Tier-0 val t2m  
7. Test t2m **strictly <** measured Tier-0 test t2m  

### C — vs living interim on FINAL protocol (hard for G1 promote)

8. Val t2m **strictly <** living re-score val t2m  
9. Test t2m **strictly <** living re-score test t2m  
10. Val & test wind-vector **strictly <** living re-score WV  

**No ε_protect on FINAL G1** (firmer than interim v1.3 ±0.02 K). Joint trade that worsens t2m to buy winds ⇒ **FAIL as G1 promote** (may still be an interesting interim variant under a separate label).

### Mechanical product

`final_g1_candidate_pass = A ∧ B ∧ C`  
Only then may Leonard consider `g1_claimable=true` (still requires Manisha + explicit call).

### Soft / report fails (do not alone kill G1 if A–C pass)

- Seasonal RMSE imbalance (report)  
- Spectra blur WARN (document; hard FAIL only if Leonard later sets PSD floors)  
- Extremes / precip absent (expected)

## 10. Explicit FAIL conditions (summary)

- Any A/B/C miss  
- Unconstrained +120 h fallback  
- Protocol / IC hash mismatch vs frozen protocol  
- Archive holes discovered post-hoc on gated years  
- Relabeling interim thick-2 runs as FINAL / G1  
- Claiming global WB2 / CorrDiff / precip / stations / ops replacement

## 11. Non-claims (REQUIRED in every FINAL call / README)

- Not global WeatherBench 2 / FCN3 leaderboard SOTA  
- Not operational NWP replacement over Nepal  
- Not km-scale CorrDiff / RCM downscaling  
- Not precip skill (until precip unlock)  
- Not IMDAA / station-verified until wired  
- Not calibrated probabilistic ensembles (mean residual)  
- Interim thick-2 / `v1_3_joint` remain `interim_era5` — **do not** relabel as FINAL

## 12. Eng order (Howard) — no train yet

1. Mirror this recipe + call to Spark `docs/research/`.  
2. Continue CDS; leave alone except coverage logging toward archive audit.  
3. When CDS complete for candidate years → archive coverage JSON → ping Leonard.  
4. Leonard writes `FINAL_EVAL_PROTOCOL.md` (concrete years + IC lists).  
5. Howard scores `final_baselines.json` (raw / Tier-0 / living) — **CPU/GPU eval only**.  
6. Leonard `FINAL_EVAL_BARS_CALL.md` freezes floats.  
7. **Only then** Manisha may unlock FINAL residual train.

## 13. Paths (LOCKED names)

| Doc / artifact | Path |
| --- | --- |
| Lit | `~/Desktop/Research/fourcastnet3/EVAL_BENCHMARKS_AND_FINAL_SUITE.md` |
| This recipe | `.../FINAL_EVAL_SUITE_RECIPE.md` |
| Freeze call | `.../FINAL_EVAL_SUITE_CALL.md` |
| Later protocol | `.../FINAL_EVAL_PROTOCOL.md` (after audit) |
| Later bars | `.../FINAL_EVAL_BARS_CALL.md` (from JSON) |
| Spark mirror | `~/fourcastnet/docs/research/` same basenames |
| Baselines | `runs/phase0/final_eval/final_baselines.json` (when scored) |

