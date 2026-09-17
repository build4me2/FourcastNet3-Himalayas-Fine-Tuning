# Weather-ML Evaluation Benchmarks & FINAL Eval Suite Proposal
## Nepal FourCastNet3 Residual Fine-Tune (RRCA-FD)

**Author:** research note for Manisha (via Sheldon)  
**Date:** 2026-09-16 PT  
**Repo:** https://github.com/build4me2/FourcastNet3-Himalayas-Fine-Tuning  
**Spark tree:** `~/fourcastnet`  
**Method:** RRCA-FD — frozen FCN3 backbone + residual UNet; Nepal box **26–31°N / 80–89°E**  
**Living residual (INTERIM PASS PROMOTE):** `runs/phase0/tier_a/v1_3_joint/`  
**Status:** No GPU / no new training. CDS ERA5 crop still backfilling (~1980→tip). This note proposes a **FINAL** unlock **after** full ERA5 — it does **not** reopen frozen interim locks.

---

## 0. Locked interim context (do not contradict)

| Lock | Value | Source |
| --- | --- | --- |
| Holdout thick-2 | train **12** / val **16** / test **16** ICs (`holdout_expand=v2`) | `docs/calls/HOLDOUT_THICK2_CALL.md` |
| Year hard-lock | train **2018–2021** / val **2022** / test **2023–2024** | `docs/04-gates/YEAR_HARD_LOCK.md` |
| Claim labels | `claim_level=interim_era5` · `g1_claimable=false` | thick-2 + year-lock calls |
| Interim beat-this (frozen) | val t2m **< 1.988588** · test t2m **< 1.918383** · val +120h **≤ 2.178842** | HOLDOUT_THICK2_CALL |
| Living headlines (16-IC) | t2m RMSE val **1.770** / test **1.775** / +120h **1.982**; wind-vector **0.69560** / **0.73877** | TIER_A_V1_3_JOINT_CALL + user lock |
| Wind-vector def | \(\sqrt{\mathrm{mean}((u_{\mathrm{err}}^2+v_{\mathrm{err}}^2)/2)}\) grid-pooled, then lead-mean over {24,72,120} | v1_3 joint recipe |

**Rule:** Interim bars stay frozen. FINAL is a **new protocol unlock** once the year archive + IC density support publishable claims. Do **not** invent FINAL numeric pass bars here — Leonard freezes them from **measured** baselines after the archive lands.

---

## 1. Benchmarks & eval protocols in the literature

### 1.1 WeatherBench (WB1)

| Item | Detail |
| --- | --- |
| Paper | Rasp et al., *WeatherBench: A Benchmark Dataset for Data-Driven Weather Forecasting*, JAMES 2020 |
| URL | https://doi.org/10.1029/2020MS002203 · dataset: https://github.com/pangeo-data/WeatherBench |
| Setup | ERA5 coarsened (typically 5.625° / 2.8125°); medium-range leads; train/val/test year splits |
| Metrics | **RMSE**, **ACC** (anomaly correlation) vs climatology; latitude weighting |
| Claim vs measure | Standardized comparison for early DL weather models at coarse resolution — not regional orographic skill |

### 1.2 WeatherBench 2 (WB2)

| Item | Detail |
| --- | --- |
| Paper | Rasp et al., *WeatherBench 2*, arXiv:2308.15560 |
| URL | https://arxiv.org/abs/2308.15560 · https://github.com/google-research/weatherbench2 |
| Protocol | Eval year **2020**; ICs at **00/12 UTC** (full year); regrid (often 1.5°) for fair compare; 6 h steps |
| Deterministic | Latitude-weighted **RMSE**, **ACC** (DoY/hour clim., ~1990–2019), **bias / RMSB** |
| Probabilistic | **CRPS**, spread–skill |
| Precip | Prefer **SEEPS** over raw RMSE (skew / intermittency) |
| Extremes honesty | One year is thin for hurricane/heatwave extremes (WB2 states this) |

**Headline inflation pattern:** “beats IFS HRES on RMSE/ACC” is usually **global, latitude-weighted, reanalysis-verified**, often on a **single OOS year** — not regional station skill, Himalayan orography, or multi-year extremes.

### 1.3 FourCastNet / FourCastNet 3

| Model | Primary source | Eval emphasis |
| --- | --- | --- |
| FourCastNet | Pathak et al., arXiv:2202.11214 — https://arxiv.org/abs/2202.11214 | ACC/RMSE at 0.25°; short-lead IFS compare; extremes / TC / ARs as case studies |
| **FourCastNet 3** | Kurth et al., arXiv:2507.12144 — https://arxiv.org/abs/2507.12144 · NVIDIA: https://research.nvidia.com/publication/2025-07_fourcastnet-3-geometric-approach-probabilistic-machine-learning-weather | **CRPS**, ens-mean RMSE, **spread–skill**, **rank histograms** over **12-hourly ICs in 2020**; angular/zonal **power spectra**; physical-consistency; stability to ~60 d |

**FCN3 claim vs measure:** Global probabilistic skill on 2020 scorecards — **not** a Nepal residual-adapter claim. Spectra/rank-hist matter for generative FCN3; our residual UNet is currently **deterministic mean correction**, so CRPS is optional later (ensemble path), not required for RRCA-FD mean skill.

### 1.4 Pangu-Weather

| Item | Detail |
| --- | --- |
| Paper | Bi et al., arXiv:2211.02556 — https://arxiv.org/abs/2211.02556 |
| Metrics | RMSE & ACC by lead on WeatherBench-style vars; HRES baselines (**HRES-vs-ERA5 vs HRES-fc0 matters**) |
| Claim vs measure | Fast deterministic global skill on selected targets; weak on calibrated ensembles |

### 1.5 GraphCast

| Item | Detail |
| --- | --- |
| Paper | Lam et al., arXiv:2212.12794 — https://arxiv.org/abs/2212.12794 |
| Metrics | RMSE & ACC vs HRES-fc0 / ERA5; large var×level×lead scorecards; spectral / blurring discussion in supplements |
| Claim vs measure | Outperforms HRES on majority of targets; still **global deterministic** |

### 1.6 GenCast (probabilistic)

| Item | Detail |
| --- | --- |
| Paper | Price et al., arXiv:2312.15796 — https://arxiv.org/abs/2312.15796 · Nature: https://www.nature.com/articles/s41586-024-08252-9 |
| Metrics | **CRPS** scorecards; calibration; extreme exceedance (Brier / relative economic value); spectra |
| Claim vs measure | Beats ENS on vast majority of CRPS targets; precip often caveated |

### 1.7 CorrDiff (regional generative downscaling)

| Item | Detail |
| --- | --- |
| Paper | Mardani et al., arXiv:2309.15214 — https://arxiv.org/abs/2309.15214 · Comm. Earth Environ. https://doi.org/10.1038/s43247-025-02042-5 |
| Setup | Taiwan ~2 km RCM target ← ~25 km ERA5; **UNet mean + diffusion residual** |
| Metrics | **MAE**, **CRPS** (32 members), spectra/distributions, case studies; calibration still hard |
| Eval sample | **205** random OOS times in **2021** (single year) |

**Closest methodological cousin** (frozen coarse prior + residual corrector), but CorrDiff is **km-scale downscaling to RCM**, not frozen-global-backbone residual on ERA5 at FCN3 resolution. Do **not** claim CorrDiff parity.

### 1.8 Other regional / fine-tune evaluations

| Work | URL | Takeaway |
| --- | --- | --- |
| MENA localized adaptation (ClimaX + LoRA) | https://arxiv.org/abs/2409.07585 | Regional LoRA ≈ full FT; report regional RMSE vs global model on same box |
| GraphCast FT on Canadian analyses | AMS AIES (AIES-D-24-0101) | Operational analysis adaptation; warn multi-step smoothing |
| GraphCast precip over China | https://pmc.ncbi.nlm.nih.gov/articles/PMC12038038/ | Station/categorical scores — relevant **later** for precip |
| Extremes: IFS vs Pangu vs GraphCast | https://gmd.copernicus.org/articles/17/7915/2024/ | Tail RMSE beyond quantiles; regional scorecards — honesty template |
| Lagged-ensemble probabilistic benchmark | https://doi.org/10.1029/2024GL113656 | Deterministic winners ≠ ensemble CRPS winners |

### 1.9 Metric cheat-sheet

| Metric | Role | Nepal RRCA-FD use |
| --- | --- | --- |
| **RMSE** (box-pooled) | Deterministic accuracy | **Core** t2m / winds |
| **Wind-vector RMSE** | Joint u/v | **Core** (locked def) |
| **ACC** | Pattern vs climatology | FINAL optional if regional climato built |
| **Bias / elev-band RMSE** | Systematic / orographic | Report (elev bands already in interim) |
| **CRPS / spread–skill** | Probabilistic | Only if multi-member path returns |
| **Energy / angular spectra** | Blurring vs noise | Diagnostic over complex terrain |
| **Extremes (quantile RMSE, Brier)** | Tails | FINAL secondary; needs denser ICs |
| **SEEPS / CSI / stations** | Precip / obs truth | Later unlock — not first FINAL gate |

---

## 2. What papers claim vs what they measure — failure modes

### 2.1 Claim inflation patterns

| Claim language | Often actually measured | Failure mode |
| --- | --- | --- |
| “Better than IFS” | ERA5-verified RMSE, one year, global mean | Regionally worse (orography); baseline mismatch |
| “SOTA weather AI” | WB2 2020 scorecard | Single-year; no stations; precip weak |
| “Skillful to 10 days” | Global Z500 ACC still > ~0.6 | Surface/winds die earlier; regional ACC collapses sooner |
| “Improves extremes” | Few case studies | Underpowered; no multi-year extreme climatology |
| “Regional FT wins” | In-sample or adjacent-year ICs; thin seasons | Overfit monsoon; fails next season |

### 2.2 Failure modes especially relevant to us

1. **In-sample / leakage ICs** — overlap train years or reuse across val/test.  
2. **Short year splits** — 2018–21 / 2022 / 2023–24 is honest for *interim* but thin for ENSO / extreme monsoon claims.  
3. **Single-season underpower** — thick-2 already flagged soft test SON miss (SON=3, wanted ≥4).  
4. **Reanalysis-as-truth** — ERA5 ≠ stations; Himalayan t2m/winds have representativeness issues.  
5. **Baseline mismatch** — beat raw FCN3 on Nepal crop is the right bar; “beats GraphCast globally” is out of scope.  
6. **Deterministic vs probabilistic confusion** — residual RMSE ≠ CRPS/calibration.  
7. **Precip RMSE** — favors smooth forecasts; use SEEPS/CSI later.  
8. **Thin IC density** — 16 test ICs cannot support publishable extreme or seasonal-stratified claims.

**Honesty templates to copy:** WB2 one-year caveat; GenCast precip exclusion; CorrDiff calibration warning; GMD extremes quantile-only scoring.

---

## 3. What transfers to a regional Nepal fine-tune (frozen global backbone)

### 3.1 Can claim (when FINAL archive + density unlock)

| Claim class | OK if… |
| --- | --- |
| **Regional ERA5 skill** on Nepal box for **t2m** + **10 m wind-vector** | Strict year holdout; denser ICs; beat raw FCN3 + Tier-0 + living interim on locked metrics |
| **Lead-time honesty through +120 h** | Report full lead curve; no cherry-pick |
| **Residual adapter improves frozen backbone locally** | Backbone frozen documented; residual-only trainable params |
| **Orographic stratification** | Elev-band RMSE tables |
| **Seasonal stratification** | Enough ICs per DJF/MAM/JJA/SON on val *and* test |

### 3.2 Cannot claim (even after FINAL)

| Non-claim | Why |
| --- | --- |
| Global FCN3 / WB2 leaderboard parity | We do not retrain or re-score the globe |
| Operational NWP replacement over Nepal | No station/ops assimilation loop; ERA5 targets |
| Km-scale CorrDiff / RCM downscaling | Different problem |
| Calibrated probabilistic ensembles | Mean residual unless members added |
| Precip skill (until precip suite) | Out of current Tier-A scope |
| `g1_claimable=true` / IMDAA / published G1 | Locked false until Leonard unlocks after archive |
| Multi-decadal climate attribution | Wrong tool |

### 3.3 Transfer table

| Literature practice | Transfer? | How we use it |
| --- | --- | --- |
| WB2 RMSE by lead | **Yes** | Box-pooled RMSE on Nepal crop |
| WB2 ACC | **Partial** | Need regional climato; FINAL optional |
| WB2 CRPS | **Later** | If multi-member FCN3 / diffusion returns |
| FCN3 spectra | **Diagnostic** | Residual PSD vs ERA5 / vs raw FCN3 |
| GraphCast scorecards | **Yes (narrow)** | t2m + wind-vector × lead |
| CorrDiff MAE/CRPS | **Analog only** | Residual-corrector framing; not km-scale claim |
| Station CSI/POD | **Later** | Precip / DHM stations |
| Extremes quantile RMSE | **FINAL secondary** | Needs denser ICs + multi-year test |

---

## 4. Draft FINAL eval suite (completely finetuned / full-archive model)

> **Philosophy:** FINAL = new unlock after CDS ERA5 year archive + IC density support claims.  
> **Do not invent numeric pass bars.** Leonard freezes bars from **measured** baselines (raw FCN3, Tier-0, living `v1_3_joint`) on the FINAL protocol, then sets beat-this with an explicit margin rule.

### 4.1 Variables

| Priority | Variable | Metric |
| --- | --- | --- |
| **P0** | **t2m** | RMSE (K), bias; elev-band RMSE |
| **P0** | **Wind-vector** (u10, v10) | Locked wind-vector RMSE; also report component RMSE |
| **P1** | ACC(t2m), ACC(u10/v10) | If regional climato available |
| **P2 (later)** | Total precip | SEEPS / CSI / frequency bias — **not** first FINAL gate |
| **P2 (diag)** | Spectra / PSD | Residual vs ERA5 relative spectral error |

### 4.2 Lead times

| Set | Leads | Role |
| --- | --- | --- |
| **Gate leads** | **+24 / +72 / +120 h** | Continuity with interim; +120 h honesty |
| **Report leads** | +24 / +48 / +72 / +96 / +120 h | Full lead curve |
| Optional | +6 / +12 h | Only if budget allows |

### 4.3 Domain & pooling

- **Box:** 26–31°N / 80–89°E (unchanged).  
- **Pooling:** Same grid-pool definition as living residual JSON (document `n` cells × ICs).  
- **Elev bands:** Keep interim bins; report-only unless Leonard promotes one band to gate.  
- **Primary bar:** unweighted box pool (continuity with interim); lat-weight as sensitivity.

### 4.4 Year splits (once full archive lands)

Interim hard-lock stays for historical continuity. FINAL proposes a **wider** split only after archive completeness is verified:

| Split | Interim (locked) | FINAL structure (years TBD by Leonard after archive audit) |
| --- | --- | --- |
| Train | 2018–2021 | Longer contiguous train once pre-2018 CDS crop is verified |
| Val | 2022 | ≥1 full held-out year |
| Test | 2023–2024 | ≥2 held-out years spanning distinct monsoon / ENSO flavors if available |

**Rules:** (1) Audit CDS completeness before flipping claim labels. (2) Never silently extend train into val/test. (3) Keep a **legacy thick-2 re-score** of the FINAL ckpt for apples-to-apples vs `v1_3_joint`. (4) Flip `g1_claimable` only by explicit Leonard call.

### 4.5 IC density / seasons

| Aspect | Interim thick-2 | FINAL target design |
| --- | --- | --- |
| Counts | 12 / 16 / 16 | **Much denser** — WB2-like regionally (e.g. ≥50–100 ICs/split or 00/12 UTC across held-out years; Leonard sets GPU-affordable floor) |
| Seasons | Soft SON miss | **Hard recipe:** min IC floor per DJF/MAM/JJA/SON on val *and* test |
| Init hours | Document current | Prefer **00 and 12 UTC** if FCN3 ICs allow |
| Spacing | Sparse | Min separation (e.g. ≥5–7 days) to reduce IC autocorrelation |

### 4.6 Baselines to beat (always re-score on FINAL protocol)

| Baseline | Role |
| --- | --- |
| **Raw FCN3** (frozen backbone, no residual) | Absolute floor — must beat |
| **Tier-0 adapter** | Lightweight adapter reference |
| **Living interim `v1_3_joint`** | Continuity — FINAL must not regress without justification |
| Optional | Persistence / climatology (sanity) |

Identical ICs, leads, pooling, wind-vector definition for all baselines.

### 4.7 How Leonard freezes FINAL pass/fail bars (no invented numbers)

1. **Freeze FINAL protocol** (years, IC list hashes, leads, pooling, wind def) in `FINAL_EVAL_PROTOCOL.md` **before** training the completely finetuned model.  
2. **Score baselines** (raw FCN3, Tier-0, living `v1_3_joint`) → `final_baselines.json` with exact floats.  
3. **Set beat-this from measurements**, e.g.:  
   - Primary: val t2m RMSE **strictly <** measured raw FCN3 val (and ≤ living ± ε_protect if protecting interim).  
   - Secondary: test t2m similarly.  
   - +120 h hard: val +120 h t2m **≤** measured raw FCN3 val +120 h (thick-2 spirit).  
   - Winds: val & test wind-vector **strictly <** both raw and living (promote), *or* document deliberate t2m-only FINAL if winds deferred.  
4. **Margin rule:** Prefer **strict inequality vs measured baseline** (as thick-2) rather than inventing absolute K thresholds. If a % margin is desired, compute it from the JSON.  
5. **Eligible ckpt rule:** Keep interim spirit (`composite_eligible`; reject if +120 h worse than raw).  
6. **Publish gate:** upgrade claim labels only when archive complete, IC floors met, FINAL gates pass, and Leonard sets `g1_claimable` explicitly.

### 4.8 Pass/fail hardness philosophy

| Layer | Hardness |
| --- | --- |
| Protocol locks | Years, IC IDs, metric defs — hard frozen before train |
| Absolute vs raw | Hard fail if not better than raw FCN3 on primary metrics |
| Protect living | Soft band (e.g. ±0.02 K style) only if joint upgrade; else strict improve |
| Winds promote | Hard if winds in product claim; else report-only |
| Seasonal IC floors | Hard recipe for counts; soft for per-season RMSE unless powered |
| Extremes / precip / stations | Report-only until separate unlock |
| Statistics | Optional bootstrap / paired-IC tests for “significant” paper language |

### 4.9 Non-claims for FINAL paper/README

- Not global WB2 SOTA.  
- Not operational forecast replacement.  
- Not km-scale downscaling.  
- Not precip skill (until precip unlock).  
- Not IMDAA/station-verified until that truth is wired.  
- Interim thick-2 numbers remain `interim_era5` historical — do not relabel as FINAL.

---

## 5. Explicit gaps: FINAL suite vs current interim thick-2

| Dimension | Interim thick-2 (now) | FINAL suite (proposed) | Gap |
| --- | --- | --- | --- |
| IC density | 12 / 16 / 16 | Dense (WB2-like regional) | **Large** |
| Years | 2018–21 / 2022 / 2023–24 | Longer train + multi-year test after CDS | **Archive-dependent** |
| Claim level | `interim_era5` · `g1_claimable=false` | Publishable only after Leonard unlock | **Policy** |
| Leads reported | Gate 24/72/120 | Full curve through +120 h | **Modest** |
| Variables | t2m + winds | Same P0; precip later | Precip **deferred** |
| Truth | ERA5 interim / ARCO | Full CDS ERA5 crop + optional stations | **Provenance** |
| Probabilistic | Members mostly unused for residual gates | Optional CRPS if ensembles | **Method** |
| Extremes | Not gated | Secondary quantile/case suite | **Missing** |
| Spectra | G0 PSD historically; not living gate | Diagnostic residual spectra | **Underused** |
| Baselines | Raw + living beat-this frozen | Re-freeze on FINAL protocol | Must **re-measure** |
| SON balance | Soft miss (test SON=3) | Hard seasonal IC floors | **Recipe** |
| Training | Hold on new GPU train | FINAL train only after protocol freeze | Process |

**Bottom line:** Living `v1_3_joint` is an honest **interim** win on a thin but locked protocol. It is **not** FINAL G1. FINAL needs archive + density + re-frozen bars.

---

## 6. Alignment with locked project language

This note **does not** change:

- Thick-2 beat-this floats (**1.988588** / **1.918383** / **2.178842**).  
- Year hard-lock 2018–21 / 2022 / 2023–24.  
- `claim_level=interim_era5` · `g1_claimable=false`.  
- Living promote of `v1_3_joint/` or its 16-IC headlines.  
- Hold on new training / GPU work.

It **does** propose FINAL as a **new unlock** after full ERA5, with bars frozen from measured baselines — same *style* as HOLDOUT_THICK2_CALL (measure → freeze → gate).

---

## 7. Key citations used

1. Rasp et al., WeatherBench, JAMES 2020 — https://doi.org/10.1029/2020MS002203  
2. Rasp et al., WeatherBench 2, arXiv:2308.15560 — https://arxiv.org/abs/2308.15560  
3. WeatherBench2 code — https://github.com/google-research/weatherbench2  
4. Pathak et al., FourCastNet, arXiv:2202.11214 — https://arxiv.org/abs/2202.11214  
5. Kurth et al., FourCastNet 3, arXiv:2507.12144 — https://arxiv.org/abs/2507.12144  
6. NVIDIA FCN3 publication card — https://research.nvidia.com/publication/2025-07_fourcastnet-3-geometric-approach-probabilistic-machine-learning-weather  
7. Bi et al., Pangu-Weather, arXiv:2211.02556 — https://arxiv.org/abs/2211.02556  
8. Lam et al., GraphCast, arXiv:2212.12794 — https://arxiv.org/abs/2212.12794  
9. Price et al., GenCast, arXiv:2312.15796 — https://arxiv.org/abs/2312.15796  
10. Mardani et al., CorrDiff, arXiv:2309.15214 — https://arxiv.org/abs/2309.15214 · https://doi.org/10.1038/s43247-025-02042-5  
11. MENA LoRA adaptation, arXiv:2409.07585 — https://arxiv.org/abs/2409.07585  
12. GMD extremes (IFS/Pangu/GraphCast) — https://gmd.copernicus.org/articles/17/7915/2024/  
13. Lagged-ensemble probabilistic benchmark — https://doi.org/10.1029/2024GL113656  
14. Project locks: `docs/calls/HOLDOUT_THICK2_CALL.md`, `docs/04-gates/YEAR_HARD_LOCK.md`, `docs/calls/TIER_A_V1_3_JOINT_CALL.md`

---

## For Leonard: FINAL_EVAL_CALL checklist

- [ ] **Archive audit:** CDS ERA5 Nepal crop complete for intended FINAL years (no holes on t2m/u10/v10).  
- [ ] **Protocol freeze first:** Write `FINAL_EVAL_PROTOCOL.md` (years, IC list hash, leads, pooling, wind-vector def) **before** any FINAL train.  
- [ ] **Baseline measure:** Score raw FCN3, Tier-0, living `v1_3_joint` on FINAL protocol → `final_baselines.json`.  
- [ ] **Freeze beat-this from JSON only** (strict < measured baselines; document ε_protect if any) — **no invented K bars**.  
- [ ] **IC density + season floors:** Min ICs per DJF/MAM/JJA/SON on val and test; fix interim SON soft miss.  
- [ ] **Gates:** t2m pooled (val primary / test secondary) + val +120 h hard + wind-vector promote (or explicit defer).  
- [ ] **Legacy bridge:** Re-score FINAL ckpt on thick-2 interim protocol for continuity table.  
- [ ] **Labels:** Keep `g1_claimable=false` until explicit promote; do not relabel interim runs as FINAL.  
- [ ] **Non-claims block** in call: no global SOTA, no precip, no stations, no CorrDiff parity.  
- [ ] **Only then:** Unlock FINAL train / completely finetuned residual (still no contradiction of interim locks).

---

*End of research note.*
*Paths: manii `/home/manisha/Desktop/Research/fourcastnet3/EVAL_BENCHMARKS_AND_FINAL_SUITE.md` · box `/workspace/fourcastnet3-eval/EVAL_BENCHMARKS_AND_FINAL_SUITE.md`.*
