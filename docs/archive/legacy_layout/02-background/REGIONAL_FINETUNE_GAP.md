# Regional Fine-Tune Gap Brief: FCN3 / Global MLWX → Nepal–Himalaya–South Asia

**For:** Manisha Chand  
**Date context:** September 2026  
**Scope:** Credible research-gap assessment for fine-tuning FourCastNet 3 (FCN3; arXiv:2507.12144) or similar global ML weather models for better skill over Nepal / Himalaya / South Asian mountainous regions (Nepal–India–China border and neighbors).  
**Methods:** WebSearch + WebFetch / ar5iv HTML extraction of primary papers and NVIDIA Earth2 docs. **No invented papers.**

---

## Verdict

**YES — credible research gap (partially crowded on *evaluation*, open on *adaptation*).**

| Layer | Status |
| --- | --- |
| Global CRPS / WeatherBench-style skill for FCN3 & peers | **Crowded** — strong global benchmarks |
| Observation-based South Asian monsoon evaluation | **Partially crowded** — MAUSAM (2025), GraphCast-ISM (2026), Bangladesh GraphCast study; still thin on **Nepal / High Himalaya elevation bands** |
| Public **FCN3** Himalaya / South Asia **LoRA / regional fine-tune** | **Open** — no published FCN3 regional adapter found; NVIDIA states no released fine-tuned FCN3 variants (as of Earth2Studio issue discussion) |
| Regional PEFT of *other* global weather FMs | **Emerging** — MENA–ClimaX LoRA; FengWu-GHR LoRA (lead-time, not Himalaya); GraphCast→Canadian analysis fine-tune; WeatherPEFT |
| km-scale regional generative / CAM emulation | **Crowded elsewhere** (StormCast-CONUS, CorrDiff-Taiwan) — **no public Himalaya analogue** |

**One-line gap statement:**  
> Global MLWX / FCN3 look strong on global CRPS, but Himalaya T2M, orographic precip, and elevation-dependent wind/humidity remain underevaluated against stations/DEM; **there is no public FCN3 (or Makani) Himalaya LoRA / domain-adaptive recipe**, while 0.25° physics and FCN3’s lack of prognostic precipitation hard-cap what fine-tuning alone can fix.

---

## 1. Documented weaknesses (complex terrain / tropics / monsoon / Himalaya)

### 1.1 South Asia monsoon — observation-focused (strongest recent evidence)

**MAUSAM** (arXiv:2509.01879) evaluates Seven AIWP systems including **FourCastNet, FourCastNet-SFNO, Pangu, GraphCast, Aurora, AIFS, GenCast** against IMD stations, rain gauges, and INSAT during South Asian monsoon:

- Errors vs **observations are 15–45% larger** than vs ERA5 — reanalysis-centric benchmarks **overstate** skill.
- Systematic **T2M** pattern at peak monsoon: cold bias over Indo-Gangetic Plain, warm bias over Western Ghats (~1–2 K); FourCastNet variants among **larger** regional errors; GraphCast / AIFS / GenCast smaller.
- **Northeast India / Meghalaya / eastern Arunachal**: wet bias and high inter-model precip disagreement near Himalayan foothills / eastern terrain.
- **Extreme precip tails** systematically underpredicted; light–moderate overpredicted; GraphCast especially weak on tails.
- **Mesoscale kinetic energy / humidity spectra** underestimated vs ERA5 (except GenCast closer); 0.25° / 6-hourly cadence misses high-frequency local extremes.
- Cloud-cover errors (AIFS) increase with lead over **Himalayan foothills**, Western Ghats, and NE India high-precip zones.
- Explicit framing: tropics + topography + monsoon moist convection as a stress test for global AIWP.

### 1.2 GraphCast Indian Summer Monsoon (orographic)

**GraphCast Skill and Systematic Biases in Indian Summer Monsoon Forecasts** (arXiv:2607.11905):

- Captures broad ISM rainfall pattern including orographic maxima (Western Ghats, NE India).
- **Undershoots orographic peak rainfall ~15–20%** vs IMERG at +24 h; bias **worsens to ~25–30%** by +72 h over Western Ghats.
- Domain wet bias + **suppressed rainfall variance** (power-spectrum variance ratio ≈ 0.14 vs IMERG); extremes underrepresented; deficient lower-tropospheric diabatic heating (Q1).

### 1.3 Bangladesh / eastern hills (neighboring complex terrain)

**PLOS Climate** GraphCast Bangladesh study ([doi:10.1371/journal.pclm.0000791](https://doi.org/10.1371/journal.pclm.0000791)):

- Competitive routine precip skill vs ECMWF/GFS, but **weaker over coastal and southeastern hilly regions**; extremes detection limited (CSI/FAR at high thresholds).

### 1.4 Classical NWP / climate orography lessons (transferable ceiling)

Not AI-specific, but establish physical constraints that also bind 0.25° MLWX:

- Coarse grids → **excessive moisture transport through Himalayas → wet bias over Tibetan Plateau**; finer orography (~10 km or better) reduces bias (e.g. Lin et al. 2018; HR vs LR CMIP-class comparisons, AGU 2024).
- Fine terrain complexity changes cloud/precip partitioning on southern TP slopes (2025 modeling studies).

### 1.5 FCN / FCN3-specific notes

- **FCN3 paper** (arXiv:2507.12144): reports **global** CRPS / ensemble-mean RMSE / SSR for 2020; architecture explicitly mentions anisotropic local filters for **blocked flow around topographic features** and uses **orography + land–sea mask as auxiliary inputs**. Does **not** publish Himalaya / South Asia regional CRPS scorecards or monsoon case studies.
- **Precipitation is not a prognostic FCN3 output**; authors state plans to add precip as a **diagnostic** later. (Original FCN1 used a separate precip head; FCN3’s 72 channels are u/v/t/z/q levels + surface winds/T2M/MSL/TCWV.)
- MAUSAM still evaluates **older FCN / FCN-SFNO**, not FCN3 specifically — so FCN3 Himalaya skill is **underevaluated**, not proven weak or strong.

---

## 2. Published regional fine-tuning / transfer / LoRA / adapters

| Work | What was adapted | Region / task | Method | Relevance to Nepal |
| --- | --- | --- | --- | --- |
| Munir et al., arXiv:2409.07585 | ClimaX | **MENA** limited-area | LoRA / PEFT | Closest published **regional PEFT** template; not mountains/monsoon |
| FengWu-GHR, arXiv:2402.00059 | FengWu → ~0.09° | Global high-res | **LoRA per lead step** + transfer from low-res meta-model | LoRA recipe for weather FMs; **not** Himalaya domain |
| Husain et al., arXiv:2408.14587 | GraphCast 37-level 0.25° | Global, new analysis (ECCC GDPS) | Full / efficient fine-tune on ~2 yr operational analysis | Shows IC-distribution shift fine-tune; **not** regional crop |
| WeatherPEFT, arXiv:2509.22020 / ICLR 2026 | Weather foundation models | Heterogeneous downstream tasks | Task-adaptive PEFT (SFAS + prompting) | Method paper; not Himalaya |
| StormCast, arXiv:2408.10958 | Regional CAM emulator | **CONUS** 3 km HRRR | Diffusion + regression, ERA5/GFS conditioning | NVIDIA regional stack exists — **CONUS only** |
| CorrDiff, arXiv:2309.15214 | Downscaling + precip synthesis | **Taiwan** ~2 km | Residual diffusion | Template for FCN3→km precip **diagnostic** over mountains |
| NVIDIA Earth2Studio issue #879 | FCN2/FCN3/Atlas | IC shift (EC HRES t0) | Advice: Makani fine-tune or bias-correct diagnostic; **no released FT weights** | Confirms FCN3 FT is DIY |

**Not found (as of Sep 2026 search):** public paper or checkpoint for **FCN3 Himalaya LoRA**, **FCN3 Nepal crop fine-tune**, or **GraphCast/Pangu Himalaya adapter**.

---

## 3. Does FCN3 paper / Earth2Studio discuss regional FT recipes, regional CRPS, blind spots?

| Question | Finding |
| --- | --- |
| Regional fine-tune recipes in FCN3 paper? | **No.** Only **global** curriculum FT on ERA5 2012–2016 (distribution drift / medium-range), with spatial model-parallelism in **Makani**. |
| Regional CRPS / Himalaya scorecards? | **No.** Scores are globally averaged (2020 ICs); spectral fidelity case study is extratropical storm Dennis. |
| Known blind spots called out by authors? | Precip not included yet; future DA uncertainty; emphasis that pointwise CRPS alone is incomplete. **No monsoon/Himalaya blind-spot section.** |
| Earth2Studio | **Inference** toolkit (FCN3 load + GFS/IFS workflows). Training recipes live in **Makani** / PhysicsNeMo. Docs advertise regional workflows (StormCast-CONUS, CorrDiff) separately from FCN3. |
| Blind spots from *ecosystem* evidence | (i) No precip channel; (ii) 0.25° orography too smooth for peaks; (iii) ERA5-trained → obs gap in South Asia; (iv) spherical global model — naive lat–lon crop FT may break boundary/spectral assumptions unless carefully designed. |

---

## 4. Data available for Nepal-region fine-tuning

### 4.1 Matches FCN3’s 72-channel, 0.25°, 6-hourly ERA5 setup

| Source | Match | Notes |
| --- | --- | --- |
| **ERA5** (CDS) | **Native match** | Same variables FCN3 trains on; crop South Asia / Himalaya box for reweighted loss. ~0.25°. **Cannot resolve ridge–valley precip.** |
| **ERA5 orography / LSM** | Already FCN3 auxiliaries | Still ~25–30 km effective terrain |
| **GFS / IFS open data** | Inference ICs via Earth2Studio | Useful for ops eval; distribution shift vs ERA5 → may need FT (cf. GraphCast operational / Earth2 #879) |

### 4.2 Higher-value regional data (better for mountains; **not** drop-in FCN3 channels)

| Source | Res / cadence | Coverage | Use |
| --- | --- | --- | --- |
| **IMDAA** (NCMRWF; Rani et al. 2021) | ~**0.12° (~12 km)**, hourly / 3-hourly; 1979–2020 | **30°E–120°E, 15°S–45°N** — includes Nepal & Himalayas | Best regional reanalysis for monsoon orography; channel remap + regrid needed; access via https://rds.ncmrwf.gov.in/ |
| **IMD** gridded precip / stations | Daily / gauge | India-focused; NE India foothills relevant | Obs loss / verification (as in MAUSAM) |
| **NCMRWF** operational analyses / forecasts | Regional NWP | South Asia | Distillation targets (strategy d) |
| **DHM Nepal** AWS / stations | Point, irregular | Nepal | Sparse; elevation-stratified verification; assimilation-style loss (e) |
| **IMERG / GPM** | ~0.1°, 30 min–daily | Global incl. Himalaya | Precip diagnostic target; known mountain gauge/radar issues |
| **HIWAT** WRF ensemble archive (NASA SERVIR / ICIMOD) | ~4 km CAM | Nepal, Bangladesh, NE India (pre/wet monsoon archive) | Distillation / CorrDiff-style HR targets |
| **High-res DEM** (SRTM, Copernicus DEM, GMTED2010) | 30–90 m → aggregate to model grid | Global | Conditioning / subgrid orography stats; **not** a substitute for resolving peaks at 0.25° |
| **HAR / HAR v2** | ~10 km TP/Himalaya | High Asia | Climate-oriented regional analysis alternative |

### 4.3 Practical mismatch summary

- **Drop-in continued FT:** ERA5 crop of the 72 FCN3 channels.  
- **Mountain-relevant FT / diagnostics:** IMDAA (≈2× resolution), IMERG/IMD/DHM precip, DEM-derived subgrid orography, HIWAT/WRF for km fields.  
- **Hard constraint:** Even perfect ERA5 FT at 0.25° **cannot** resolve individual Himalayan peaks or valley winds; gains will be **bias / monsoon regime / foothill-band** skill, not true peak-resolving precip.

---

## 5. Concrete gap statement (crowded vs open)

### Crowded
- Global deterministic & probabilistic benchmarks (WeatherBench 2, FCN3 paper CRPS).
- Generic “AI beats IFS on global RMSE” narratives.
- South Asia **monsoon evaluation** of GraphCast / AIFS / GenCast / FCN (MAUSAM; GraphCast-ISM; Bangladesh).
- Regional AI NWP / downscaling **in other domains** (CONUS StormCast, Taiwan CorrDiff, MENA ClimaX LoRA).
- Classical Himalaya orography–precip bias literature in physics NWP.

### Open (actionable for Manisha)
1. **No public FCN3 Himalaya / Nepal LoRA or domain-adaptive Makani recipe.**
2. **FCN3 not yet in MAUSAM-style obs verification** over South Asia / elevation bands.
3. **Elevation-stratified skill** (foothills vs High Himalaya vs TP) for T2M, 10 m wind, humidity, TCWV — largely missing for FCN3.
4. **Joint story:** FCN3 backbone + **precip diagnostic / CorrDiff-like** head trained on IMDAA/IMERG/HIWAT over Nepal box.
5. **Monsoon-season multi-step rollout FT** with **regionally reweighted CRPS** (vs global CRPS only).
6. Transfer of PEFT methods proven on ClimaX/FengWu to **spherical-conv FCN3** (non-trivial — architecture differs).

---

## 6. Feasible fine-tune strategies (ranked)

| Rank | Strategy | Feasibility | Expected gain | Needs Leonard / heavy GPU? | Needs Howard / domain partner? |
| --- | --- | --- | --- | --- | --- |
| **1 — (a)** Continued ERA5 fine-tune on lat–lon crop (e.g. 20–40°N, 70–95°E) with **spatially reweighted CRPS/MSE** (upweight high-orography / monsoon months) | High — closest to FCN3 curriculum; Makani supports CRPS | Bias reduction on T2M/winds/humidity in foothills; limited precip | **Yes (Leonard)** — multi-GPU Makani; start from NGC/HF Apache-2.0 weights | Light — box definition, elevation mask |
| **2 — (b)** Multi-step autoregressive rollout FT focused on **JJAS monsoon** seasons | Medium–High — mirrors FCN3 stage-2/3 | Better medium-range monsoon stability | **Yes (Leonard)** — memory-heavy AR FT | Monsoon IC selection, verification seasons |
| **3 — (c)** Add **high-res orography / DEM-derived** conditioning (subgrid std, slope, anisotropy; replace/augment ERA5 orography aux) | Medium — FCN3 already has orography aux; encoder changes | Marginal at 0.25°; better static bias | Medium GPU; careful channel surgery | **Yes (Howard)** — DEM processing, physical priors |
| **4 — Hybrid diagnostic** CorrDiff / precip head on IMDAA–IMERG–HIWAT conditioned on FCN3 state | High scientific value; separate from backbone FT | **Addresses precip gap** FCN3 lacks | Separate training run (Leonard or PhysicsNeMo) | **Yes (Howard)** — precip truth, mountain gauge QC |
| **5 — (d)** Distill from regional NWP (IMDAA / NCMRWF / HIWAT WRF) | Medium — label shift, license/access | Teaches finer orographic precip patterns into diagnostic or LoRA | Leonard for student training | **Yes (Howard + NCMRWF/ICIMOD access)** |
| **6 — (e)** Station / obs assimilation-style loss (DHM + IMD gauges; StormCast-SDA-like) | Lower near-term — sparse, noisy, elevation bias | Direct obs skill; publication angle vs ERA5-only | Research-heavy | **Yes (Howard + DHM data MoUs)** |

**Recommended path (phased):**  
**(a)+(b)** on ERA5 crop with orography-weighted CRPS → publish **FCN3 Himalaya verification + adapter** → add **precip diagnostic (CorrDiff-class)** with IMDAA/IMERG → later **(e)** if station access materializes.  
Treat **(c)** as ablation, not the main bet — 0.25° remains the bottleneck.

---

## 7. Risks

| Risk | Detail |
| --- | --- |
| **Catastrophic forgetting** | Region-only FT can degrade global / extratropical skill; mitigate with LoRA / small LR / mix global+regional batches / keep frozen backbone. |
| **0.25° cannot resolve peaks** | Everest-scale terrain & valley circulations unresolved; market as **regional bias correction / foothill monsoon skill**, not “peak-resolving NWP.” |
| **Precip not in FCN3 outputs** | Backbone FT alone does **not** fix rainfall products; need diagnostic head or external downscaler. |
| **ERA5 mountain biases inherited** | FT on ERA5 may entrench wet-TP / smoothed-orography errors; prefer IMDAA/obs for diagnostic targets. |
| **Spherical architecture vs lat–lon crop** | FCN3 is global spherical-conv / SHT-aware; naive patch training may need padded global inputs with masked loss rather than true limited-area model. |
| **License / weights** | FCN3 weights + Makani: **Apache 2.0** (NGC / Hugging Face `nvidia/fourcastnet3`). ERA5: CDS license constraints for redistribution. IMDAA/HIWAT: registration / attribution; CorrDiff Taiwan data historically **CC BY-NC-ND** — check before commercial. |
| **Compute** | FCN3 training used up to 1024 H100s; regional LoRA / short FT is far cheaper but still multi-GPU and non-trivial to configure in Makani. |
| **Evaluation trap** | Improving vs ERA5 crop ≠ improving vs DHM/IMD stations (MAUSAM lesson). |

---

## Recommended approach (executive)

1. **Claim the gap:** “First FCN3 domain-adaptive / LoRA study with elevation-stratified, observation-based verification over Nepal–Himalaya–South Asia.”  
2. **Do not claim** km-scale precip skill from 0.25° FT alone.  
3. **Phase 1 (Leonard-led):** Makani continued FT / LoRA on ERA5 with Himalaya+monsoon **loss reweighting**; Earth2Studio inference baselines vs FCN3 vanilla.  
4. **Phase 2 (Howard+Leonard):** Precip **diagnostic** (CorrDiff- or FCN1-style) using IMDAA + IMERG + optional HIWAT; DEM subgrid features.  
5. **Phase 3:** Station-aware loss / SDA if DHM data secured.  
6. **Baselines to beat:** vanilla FCN3, GraphCast/AIFS (from MAUSAM), persistence/climatology, and IFS/HRES where available — report **obs and ERA5** both.

---

## Reading list (real URLs / arXiv only)

### Core model & tooling
- FCN3 paper: https://arxiv.org/abs/2507.12144  
- FCN3 HTML: https://ar5iv.labs.arxiv.org/html/2507.12144  
- NVIDIA research page: https://research.nvidia.com/publication/2025-07_fourcastnet-3-geometric-approach-probabilistic-machine-learning-weather  
- NVIDIA tech blog: https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/  
- Makani (training): https://github.com/NVIDIA/makani/  
- Earth2Studio: https://github.com/NVIDIA/earth2studio/  
- HF weights (Apache-2.0): https://huggingface.co/nvidia/fourcastnet3  
- NGC model card: https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3  
- Fine-tune discussion (no released FT models): https://github.com/NVIDIA/earth2studio/issues/879  

### South Asia / monsoon / terrain evaluation
- MAUSAM: https://arxiv.org/abs/2509.01879 — HTML: https://ar5iv.labs.arxiv.org/html/2509.01879  
- GraphCast ISM biases: https://arxiv.org/abs/2607.11905 — HTML: https://ar5iv.labs.arxiv.org/html/2607.11905  
- GraphCast Bangladesh (PLOS Climate): https://doi.org/10.1371/journal.pclm.0000791  
- EGU abstract (GraphCast/FuXi ISM 2023): https://doi.org/10.5194/egusphere-egu26-7223  

### Regional / PEFT / high-res adaptation
- MENA ClimaX LoRA: https://arxiv.org/abs/2409.07585 — code: https://github.com/akhtarvision/weather-regional  
- FengWu-GHR (LoRA lead-time FT): https://arxiv.org/abs/2402.00059  
- GraphCast → Canadian analysis FT: https://arxiv.org/abs/2408.14587  
- WeatherPEFT: https://arxiv.org/abs/2509.22020  
- StormCast: https://arxiv.org/abs/2408.10958  
- CorrDiff: https://arxiv.org/abs/2309.15214 — Nature CEE version: https://www.nature.com/articles/s43247-025-02042-5  
- Earth2 CorrDiff example: https://nvidia.github.io/earth2studio/examples/03_downscaling/01_corrdiff_inference.html  

### Regional data & orography context
- IMDAA (JCLI): https://doi.org/10.1175/jcli-d-20-0412.1  
- IMDAA portal: https://rds.ncmrwf.gov.in/  
- IMDAA config note: https://rmets.onlinelibrary.wiley.com/doi/10.1002/asl.808  
- HIWAT HKH thunderstorms (BAMS): https://doi.org/10.1175/bams-d-21-0260.1  
- HIWAT archive (Data): https://www.mdpi.com/2306-5729/10/7/112  
- WeatherBench 2: https://sites.research.google/weatherbench/  
- WB2 regions API (tropics/extratropics; no built-in Himalaya mask): https://weatherbench2.readthedocs.io/en/latest/api.html  

---

## Uncertainties / caveats

1. **FCN3-specific Himalaya skill is unknown** — MAUSAM covers FCN/FCN-SFNO, not FCN3; do not extrapolate FCN blow-up or bias magnitudes to FCN3 without a new eval.  
2. arXiv:2607.11905 is a **2026** preprint; treat results as credible but not yet journal-final.  
3. Searches may miss non-indexed local reports (DHM, ICIMOD tech notes) or Chinese-language TP AIWX papers.  
4. “Howard / Leonard” roles assumed from project context (domain/data vs GPU/Makani); adjust ownership if staffing differs.  
5. IMDAA channel set and vertical levels **do not automatically match** FCN3’s 13 pressure levels × 5 vars + 7 surface — remapping effort is non-trivial.  
6. Political/border data access (China side TP stations) may limit fully cross-border verification.  
7. WebFetch to arxiv.org PDF timed out in this session; FCN3/MAUSAM content was verified via **ar5iv HTML** + search snippets — citations above point to canonical abs/HTML URLs.

---

*End of brief. File: `REGIONAL_FINETUNE_GAP.md`*
