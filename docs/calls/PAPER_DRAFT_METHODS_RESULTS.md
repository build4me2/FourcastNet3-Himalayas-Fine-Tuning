# Regional FCN3 residual fine-tune over Nepal — Methods & Results (draft)

**Authors:** Manisha Chand (lead) · eng notes from Spark FCN runs (Howard)  
**Date:** 2026-09-21 PT · **Claim level:** `interim_era5` · **`g1_claimable=false`**  
**Living residual (LOCKED):** `runs/phase0/tier_a/v1_3_joint/best_residual.pt`  
(promoted 2026-09-15; prior `v1_2b_thick2_train/` historical / frozen)  
**Sources:** `FINAL_EVAL_SUITE_CALL.md`, `FINAL_EVAL_SUITE_RECIPE.md`, `EVAL_BENCHMARKS_AND_FINAL_SUITE.md`, `TIER_A_V1_3_JOINT_CALL.md`, `TIER_A_V1_3_JOINT_RECIPE.md`, `tier_a_v1_3_joint_results.json`, `tier_a_v1_3_joint_results_claim_16ic.json`, `living_wind_baseline.json`, `v1_3_joint_wind_score.json`, `LIVING_RESIDUAL_WINDS_ELEV_TABLE.md`, `TIER_A_V1_2B_THICK2_TRAIN_CALL.md`, `TIER_A_V1_DIFF_CALL.md`, `HOLDOUT_THICK2_CALL.md`, `YEAR_HARD_LOCK.md`.

> Stanford-plain draft for **methods + results** only. Not a full paper.  
> All numeric claims are measured from existing artifacts — **no invented metrics**.  
> Protocol claims use **16 thick-2 test ICs** (`claim_16ic` / wind_score); do not use main-JSON `metrics.test` when it includes `ic45`.

---

## 0. Intro

Global ML weather models such as FourCastNet 3 (FCN3) are strong at planetary scale, but **regional** skill over complex terrain still needs a local correction that does not require full-weight fine-tuning. This draft records an **interim** Nepal-box residual program: freeze the FCN3 backbone (~711M), train a small elev-conditioned residual UNet on FCN3 ens-mean crops, and verify against **ERA5 interim** (`interim_era5`).

| | |
| --- | --- |
| Domain | **26–31°N, 80–89°E** (Nepal / High Himalaya crop) |
| Diagnostics | t2m, u10m, v10m at +24 / +72 / +120 h |
| Living product | `runs/phase0/tier_a/v1_3_joint/` (promoted 2026-09-15) |
| Claim level | `interim_era5` · **`g1_claimable=false`** |

ICs remain **global ARCO** crops. The Nepal CDS ERA5 crop is an archive still filling for later FINAL work — not the IC source. Headlines below are thick-2 interim promote numbers only; they are **not** FINAL / G1.

---

## 1. Problem

FourCastNet 3 (FCN3) is a global probabilistic weather model. We target **regional residual skill** over a Nepal / High Himalaya box, without fine-tuning FCN3 weights (not Spark-feasible at full scale).

| | |
| --- | --- |
| Domain | **26–31°N, 80–89°E** (Nepal crop) |
| Diagnostic vars (Tier-A) | 2 m temperature (**t2m**), 10 m winds (**u10m**, **v10m**) |
| Leads | +24 / +72 / +120 h |
| Approach | Freeze FCN3; train a small elev-conditioned regional residual on FCN3 ens-mean crops |
| Verification | **ERA5 interim** (`interim_era5`) |

ICs are global ARCO crops. A Nepal CDS ERA5 crop archive is **still filling** — leave the CDS download process alone (check `ps` / `logs/era5_pull.log` for live PID; do not hardcode) — and is **not** the IC source. This draft does **not** claim G1 / IMDAA / published operational skill.

---

## 2. Method — frozen FCN3 + living residual (`v1_3_joint`)

**Name.** RRCA-FD Plan A — *Regionally Reweighted CRPS Adapter + Frozen Diagnostic* (residual mean stage). Living product = **v1.3 joint** (parallel residual retrain with balanced channel weights).

| Component | Living `v1_3_joint` |
| --- | --- |
| FCN3 (~711M) | **FROZEN** (Earth2Studio inference; ens-mean crops cached) |
| Residual | **ElevCondResidualUNet** (~168k params; base=16, depth=3, dropout=0.20) |
| Input | pred t2m/u10m/v10m + elev_norm + lead_norm (5 ch) |
| Output | residual Δ on (t2m, u10m, v10m) toward ERA5 interim |
| Loss | elev-weighted, lead-weighted MSE; `w_120=2.0`; elev boost 1.5 |
| Channel weights | **`[1.5, 1.5, 1.5]`** (joint t2m+winds; prior living used `[2.0, 0.5, 0.5]`) |
| Checkpoint | composite: reject if val +120 h t2m RMSE > raw; else best lead-mean |
| Seed / epochs | 42 · max 400 · patience 80 · **46** eligible saves · reload=`composite_eligible` |

This is a **parallel residual** (target = ERA5 − FCN3), **not** leftover-of-living diffusion.

### 2.1 Protocol locks

| Lock | Value |
| --- | --- |
| Holdout thick-2 | train **12** / val **16** / test **16** ICs (`holdout_expand=v2`) |
| Year hard-lock | train **2018–2021** / val **2022** / test **2023–2024** (`provisional_years=false`, `year_split_frozen=true`) |
| Beat-this (frozen thick-2) | val t2m **< 1.988588** · test t2m **< 1.918383** · val +120 h t2m **≤ 2.178842** |
| Eligible ckpt | `n_eligible_saves ≥ 1`, reload=`composite_eligible` (unconstrained fallback ⇒ FAIL) |
| Claim label | `interim_era5` · `g1_claimable=false` |

### 2.2 Promote gates (v1.3 ABC)

| Gate | Rule (measured) |
| --- | --- |
| **A** Absolute floors | Clear thick-2 t2m bars + eligible composite ckpt |
| **B** Protect prior living t2m | ≤ living ± 0.02 K (val ≤ 1.779883 · test ≤ 1.7997) |
| **C** Promote on winds | wind-vector lead-mean **strictly <** prior living **and** raw FCN3 (val & test) |

Leonard call 2026-09-15: **A+B+C PASS → INTERIM PASS CONFIRM → promote** `v1_3_joint/` as living residual.

---

## 3. Results — living `v1_3_joint/` (headline)

Authoritative promote numbers: `tier_a_v1_3_joint_results_claim_16ic.json` + `v1_3_joint_wind_score.json` (16-IC locked).

### 3.1 Pooled t2m + wind-vector

| Metric | Prior living `v1_2b_thick2_train` | **Living `v1_3_joint`** |
| --- | ---: | ---: |
| Val t2m pooled (K) | 1.760 | **1.770** |
| Test t2m pooled (K, 16-IC) | 1.780 | **1.775** |
| Val +120 h t2m (K) | 1.962 | **1.982** |
| Val wind-vector lead-mean | 0.69910 | **0.69560** |
| Test wind-vector lead-mean | 0.74493 | **0.73877** |

Exact floats (living): val t2m **1.770077** · test t2m **1.774864** · val +120 h **1.982401** · val WV **0.69560168** · test WV **0.73876564**.  
n (val/test grid-pooled): **37296** each.

**Gate echo (call):** A PASS · B PASS (val Δ **+0.010** / test Δ **−0.005** vs prior living) · C PASS · eligible **46** · `composite_eligible`.

Winds are the promote story (modest but strict lift vs prior living and raw). t2m stays inside the protect band.

### 3.2 Absolute floors vs thick-2 beat-this

| Gate | Living | Bar | Result |
| --- | ---: | ---: | --- |
| Val pooled t2m | 1.770077 | < 1.988588 | PASS |
| Test pooled t2m (16-IC) | 1.774864 | < 1.918383 | PASS |
| Val +120 h t2m | 1.982401 | ≤ 2.178842 | PASS (Δ −0.196 vs raw) |

### 3.3 t2m by lead (h) — living `v1_3_joint` (16-IC)

From `v1_3_joint_wind_score.json` / results val:

| Split | Lead | RMSE raw | RMSE residual | n |
| --- | ---: | ---: | ---: | ---: |
| val | 24 | 1.696995 | 1.456261 | 12432 |
| val | 72 | 2.138514 | 1.830002 | 12432 |
| val | 120 | 2.178842 | **1.982401** | 12432 |
| test | 24 | 1.945683 | 1.629914 | 12432 |
| test | 72 | 2.242590 | 1.909474 | 12432 |
| test | 120 | 1.832432 | 1.774180 | 12432 |

**Honesty:** test +120 h residual (1.774) **beats** raw (1.832) on the 16-IC wind_score path. Hard gate remains **val** +120 h only.

### 3.4 t2m by elevation band — living val

Elevation bins (m): [0, 1500, 3000, 4500, 9000]. From `tier_a_v1_3_joint_results.json` metrics.val (n=37296):

| Bin | Label | RMSE raw | RMSE residual | n |
| ---: | --- | ---: | ---: | ---: |
| 0 | [0,1500)m | 1.399552 | 1.239915 | 15456 |
| 1 | [1500,3000)m | 1.822012 | 1.679697 | 2928 |
| 2 | [3000,4500)m | 2.309077 | 2.067816 | 3072 |
| 3 | [4500,9000]m | 2.452855 | 2.127509 | 15840 |

Residual RMSE rises with elevation (val ~1.24 → ~2.13 K from bin0→bin3), consistent with orographic hardness.

### 3.5 Component winds (grid-pooled, living)

| Split | Var | RMSE raw | RMSE living residual | n |
| --- | --- | ---: | ---: | ---: |
| val | u10m | 0.737046 | 0.727939 | 37296 |
| val | v10m | 0.696005 | 0.665488 | 37296 |
| test | u10m | 0.833946 | 0.805408 | 37296 |
| test | v10m | 0.699712 | 0.675783 | 37296 |

Wind-vector definition (recipe §5 / JSON): per-lead `sqrt(mean((u_err²+v_err²)/2))` over valid elev cells; headline = **lead-mean** over {24,72,120}.

---


## 4.0 Related work (cousins — no invented metrics)

We situate this interim residual work against three public cousins; citations and
non-claims follow `EVAL_BENCHMARKS_AND_FINAL_SUITE.md` (not a leaderboard entry).

| Cousin | What we borrow | What we do **not** claim |
| --- | --- | --- |
| **WeatherBench 2** (Rasp et al., arXiv:2308.15560) | Year-holdout discipline; lead-time honesty; one-year / extremes caveats | Global WB2 SOTA; medium-range leaderboard parity; “better than IFS” from a Nepal-box ERA5 RMSE |
| **FourCastNet 3 / FCN3** | Frozen global backbone; ens-mean crops as prior | FCN3 weight fine-tune; global FCN3 scorecard replacement |
| **CorrDiff** (NVIDIA; UNet mean + diffusion residual on RCM) | Residual-corrector framing (coarse prior + local Δ) | Km-scale CorrDiff / RCM downscaling parity; precip / CRPS calibration claims |

Methodologically the closest cousin is CorrDiff’s residual-corrector idea, but the
problem differs: we correct a frozen global FCN3 prior toward **ERA5 interim** on
the Nepal box at FCN3 resolution — not generative km-scale downscaling. WB2 supplies
the honesty template (year splits, lead curves, extremes thinness), not a score we
report. Full lit tables and gap list: `docs/research/EVAL_BENCHMARKS_AND_FINAL_SUITE.md`.

## 4. Contrasts

### 4.1 Prior living — `v1_2b_thick2_train` (historical)

Same protocol (thick-2 12/16/16, year hard-lock). Channel weights `[2.0, 0.5, 0.5]` (t2m-heavy). INTERIM PASS → demoted to historical on v1.3 promote; **weights still frozen**.

| Metric | Prior living |
| --- | ---: |
| Val / test t2m | **1.760 / 1.780** |
| Val +120 h t2m | **1.962** |
| Val / test wind-vector | **0.69910 / 0.74493** |

Elev/lead honesty tables for this checkpoint: `LIVING_RESIDUAL_WINDS_ELEV_TABLE.md` (post-hoc frozen eval; MD5 `aa0ed74a…` unchanged).

Notable honesty from prior living: test +120 h t2m residual **1.838** slightly **worse** than raw **1.832** (not a hard gate).

### 4.2 v1-diff — NULL (not promoted)

Leftover-target EDM on ERA5−(FCN3+living residual): one-step decode ≈ prior living (Δ val ~ **−1.7×10⁻⁵**); `diff_improves_residual=false`; ens-mean K=4 worse. Product call **`NULL_NO_MATERIAL_LIFT`**. Dir `v1_diff/` kept as null artifact.

### 4.3 Other historical contrasts

| Variant | Verdict | Key measured numbers |
| --- | --- | --- |
| **v1.2** (expand-era) | **FAIL** | val +120 h t2m **2.132014** > raw **2.131516** |
| Thick-2 ZS `v1_2b_thick2` | historical PASS | val 1.829 / test 1.897 / +120 h 2.015 |

---

## 5. Honesty / non-claims

- Label stays **`interim_era5`**. **`g1_claimable=false`** — no G1 / IMDAA / Canvas / published skill claim.
- **Interim thick-2 ≠ FINAL G1.** Living headlines above are the thick-2 promote record only; the FINAL suite is a **new protocol** (denser ICs, post-archive years) — do not relabel these floats as FINAL or G1.
- **ERA5-as-truth caveats.** Verification is against ERA5 reanalysis, not stations or IMDAA. Himalayan t2m / 10 m winds have representativeness limits; short year splits are honest for interim but thin for ENSO / extreme-monsoon claims (`EVAL_BENCHMARKS_AND_FINAL_SUITE.md` § gaps).
- Not FCN3 weight fine-tune; not CorrDiff NVIDIA parity; not precip; not leftover-target diffusion success.
- Wind lift is **modest** — joint-balance success, **not** a winds breakthrough.
- Main `tier_a_v1_3_joint_results.json` historically echoed **17** test ICs (`ic45`); protocol claims use **16-IC** sidecar / wind_score only.
- CDS Nepal crop is an **archive for later**, still filling — leave the CDS download process on Spark alone; check `ps` / `logs/era5_pull.log` for live PID.
- **FINAL suite frozen as protocol+hardness only** (`FINAL_EVAL_SUITE_CALL.md`): measure `final_baselines.json` before any bars.
- Do **not** overwrite living weights `runs/phase0/tier_a/v1_3_joint/best_residual.pt` or prior living `v1_2b_thick2_train/`.



---

## 5b. FINAL gate / non-claims (protocol freeze — no new bars)

**Citation:** `FINAL_EVAL_SUITE_CALL.md` · `FINAL_EVAL_SUITE_RECIPE.md` · lit base `EVAL_BENCHMARKS_AND_FINAL_SUITE.md`.

Living `v1_3_joint/` headlines above remain the **interim** promote record (`interim_era5`, thick-2 12/16/16). They are **not** FINAL G1 — interim thick-2 skill does not transfer by renaming. The FINAL suite is a **new protocol** after full CDS ERA5 crop + archive audit — denser ICs (val/test ≥64; ≥8/season incl. SON), gate leads +24/+72/+120, report leads through +120, locked wind-vector def. Until that protocol lands, treat living 1.770 / 1.775 / 1.982 (and WV 0.69560 / 0.73877) as **continuity reference only**.

**ERA5-as-truth (again):** FINAL gates, when frozen, will still be ERA5-verified regional RMSE — not station/ops replacement. Extremes and one-year caveats from WB2 apply by analogy; we do not invent absolute K bars here.

| Rule | Status |
| --- | --- |
| Measure baselines before bars | **required** — score `runs/phase0/final_eval/final_baselines.json` (raw FCN3 / Tier-0 / living) on FINAL ICs **before** any beat-this floats |
| Invent absolute K bars now | **forbidden** |
| Relabel living interim as FINAL / G1 | **forbidden** |
| FINAL / G1 train | **NO-GO** until audit + `FINAL_EVAL_PROTOCOL.md` + baselines JSON + `FINAL_EVAL_BARS_CALL.md` + Manisha |
| `g1_claimable` | **false** until explicit later call |

Scaffold (CPU, no train): `code/final_eval/` + `configs/final_eval_baselines.yaml`. Thick-2 legacy bridge re-score is report/honesty only.

**Non-claims (FINAL call):** not global WB2/FCN3 SOTA · not ops NWP replacement · not CorrDiff parity · not precip · not stations · interim thick-2 ≠ FINAL/G1 · not IMDAA-verified.



## 7. Discussion (interim honesty)

**What this interim proves.** Under a locked thick-2 protocol (12/16/16 ICs, year hard-lock 2018–21 / 2022 / 2023–24), a frozen-backbone elev-conditioned residual can clear absolute t2m floors, protect prior living t2m within ±0.02 K, and modestly improve wind-vector lead-mean vs prior living and raw FCN3. Living headlines: t2m **1.770 / 1.775 / +120h 1.982**; WV **0.69560 / 0.73877**. Leftover-target diffusion (`v1_diff`) did **not** lift the residual — product call NULL.

**What it does not prove.** These bars are **not** FINAL G1, not IMDAA/station-verified, not precip, not CorrDiff/km-scale parity, and not an ops NWP replacement. ERA5-as-truth over the Himalaya has representativeness limits; short year splits are thin for ENSO / extreme-monsoon claims. Wind lift is joint-balance success, not a winds breakthrough.

**FINAL gate ahead.** After the CDS Nepal crop completes and archive audit PASSes, Leonard freezes `FINAL_EVAL_PROTOCOL.md`, baselines are scored into `final_baselines.json` **before** any beat-this floats, then bars land only on Manisha call. Relabeling living interim as FINAL / G1 is forbidden. Until then: idle / writeup / doc hygiene; **no GPU train** unless assigned.


## 8. Figures package stub

Planned figures for a later methods/results writeup. **Paths TBD** under `docs/figures/` — placeholder only; **no plotted numbers invented here**.

| ID | Figure | Intent | Path (TBD) |
| --- | --- | --- | --- |
| F1 | Lead curves (t2m) | Living vs prior living vs raw at +24/+72/+120 | `docs/figures/f1_t2m_lead_curves.(png|pdf)` |
| F2 | Elev-band skill | Residual vs raw by elevation band (honesty) | `docs/figures/f2_elev_bands.(png|pdf)` |
| F3 | Winds table graphic | Wind-vector lead-mean val/test (living vs prior vs raw) | `docs/figures/f3_winds_table.(png|pdf)` |
| F4 | Protocol diagram | Frozen FCN3 → residual → thick-2 / year locks → FINAL gate | `docs/figures/f4_protocol_diagram.(png|pdf)` |

See `docs/figures/README.md`. Source tables for real numbers: `tier_a_v1_3_joint_results_claim_16ic.json`, `v1_3_joint_wind_score.json`, `LIVING_RESIDUAL_WINDS_ELEV_TABLE.md`.

## 6. Artifact index

| Role | Path |
| --- | --- |
| Living weights | `runs/phase0/tier_a/v1_3_joint/best_residual.pt` |
| Living results | `.../v1_3_joint/tier_a_v1_3_joint_results.json` |
| 16-IC claim sidecar | `.../tier_a_v1_3_joint_results_claim_16ic.json` |
| Wind baseline (prior living) | `.../living_wind_baseline.json` |
| Wind score (living v1.3) | `.../v1_3_joint_wind_score.json` |
| Config / code | `configs/tier_a_v1_3_joint.yaml` · `code/tier_a/v1_3_joint/` |
| Living promote call | `docs/research/TIER_A_V1_3_JOINT_CALL.md` |
| Recipe | `docs/research/TIER_A_V1_3_JOINT_RECIPE.md` |
| Prior living winds elev/lead | `docs/research/LIVING_RESIDUAL_WINDS_ELEV_TABLE.md` |
| Year hard-lock | `docs/research/YEAR_HARD_LOCK.md` |
| Status | `docs/research/FCN_STATUS_AND_NEXT.md` |
| v1-diff null | `docs/research/TIER_A_V1_DIFF_CALL.md` |
| Thick-2 bars | `docs/research/HOLDOUT_THICK2_CALL.md` |
| FINAL recipe / call | `docs/research/FINAL_EVAL_SUITE_RECIPE.md` · `FINAL_EVAL_SUITE_CALL.md` |
| FINAL baselines scaffold | `code/final_eval/` · `runs/phase0/final_eval/final_baselines.json` |
| ERA5 coverage audit | `data/era5/coverage_audit.json` · `docs/research/ERA5_COVERAGE_AUDIT.md` |
