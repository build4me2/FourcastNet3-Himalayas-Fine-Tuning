# Survey: Regional Fine-Tunes of FourCastNet-Family Models

**Prepared for:** Manisha Chand  
**Survey date:** September 2026  
**Scope:** Which models/projects have fine-tuned a FourCastNet-family base (FourCastNet / FCN1 / FourCastNet-v2 / SFNO / FourCastNet3 / FCN3 / NVIDIA Earth-2 FourCastNet checkpoints) for a **specific geographic region**.  
**Method:** Extensive WebSearch + WebFetch of papers, arXiv, NVIDIA blogs/docs, Earth2Studio/Makani/PhysicsNeMo materials. **No papers invented; uncertain matches flagged.**

---

## Executive finding (one paragraph)

**True weight-level fine-tunes of released NVIDIA FourCastNet / SFNO / FCN3 checkpoints for a geographic region are essentially absent in the public literature as of Sep 2026.** What exists instead is: (A) a small set of **FCN-architecture** limited-area models trained on regional reanalysis (Europe/CERRA) with nesting against a global FCN; (B) a richer **NVIDIA Earth-2 stack** where pretrained FourCastNet/SFNO stays global and **CorrDiff / StormCast** (or other downscalers) are trained regionally and conditioned on FCN-class synoptic fields; (C) many **non-FCN** regional PEFT/stretched-grid results (ClimaX, Aurora, GraphCast-like, Pangu-driven WRF) useful only as comparison. **No public FCN3 regional fine-tune was found.**

---

## Taxonomy used in this survey

| Tier | Definition |
|------|------------|
| **A — True FCN / SFNO / FCN3 regional lineage** | Uses FourCastNet/AFNO/SFNO/FCN3 architecture and/or NVIDIA FCN checkpoints for a limited area. Prefer weight FT from a published global checkpoint; also include architecture-port + regional training when clearly FCN-family. |
| **B — Closely related (NVIDIA Earth-2 stack)** | CorrDiff / StormCast / Makani FT / Earth2Studio pipelines that **keep FCN global** and adapt regionally via generative downscaling, CAM emulation, or diagnostic heads conditioned on FCN/ERA5/GFS. |
| **C — Other global MLWX regional PEFT (not-FCN)** | ClimaX, GraphCast, FengWu, Pangu, Aurora, AIFS-like stretched grids, etc. **Labeled not-FCN; comparison only.** |

---

## A) True FCN / SFNO / FCN3 regional fine-tunes

### A0. Negative result (important)

| Claim | Status (Sep 2026) |
|-------|-------------------|
| Public **FCN3** regional FT (any region) | **None found** |
| Public **SFNO / FourCastNet-v2** weight FT for a cropped region (LoRA or full) | **None found** with published regional metrics |
| Public LoRA/PEFT on NVIDIA Earth-2 FCN checkpoints for India / China / Europe / US / MENA / Himalaya | **None found** |
| Earth2Studio / Makani docs for “regional FCN3 FT recipe with released weights” | Training infrastructure exists (Makani); **no published regional FT paper/checkpoint** |
| GitHub issue on EC-hres-t0 FT for FCN2/FCN3 | NVIDIA reply (Nick Geneva): **no plans to release FT models**; suggested diagnostic bias-corrector instead ([earth2studio#879](https://github.com/NVIDIA/earth2studio/issues/879)) |

FCN3’s own paper describes a **global** curriculum fine-tune on ERA5 2012–2016 (distribution drift), **not** a geographic regionalization ([arXiv:2507.12144](https://arxiv.org/abs/2507.12144)).

---

### A1. Nested regional FourCastNet (Europe / CERRA) — **primary true-lineage case**

| Field | Detail |
|-------|--------|
| **Name / papers** | Hamer, Sleeman, Halem — *Towards a Dynamic Data Driven AI Regional Weather Forecast Model* (DDDAS / Springer chapter, 2024); master’s thesis *A Nested Fully Data Driven AFNO-Based Regional Weather Forecast Model* (UMBC, 2024); AGU 2024 abstract *Nested Regional Weather Modeling using FourCastNet* |
| **Region** | Continental Europe (CERRA domain), ~5.5 km |
| **Base** | NVIDIA **FourCastNet v1 (AFNO)** architecture; nesting uses NVIDIA’s **global FCN** trained on ERA5 as boundary driver |
| **Fine-tune data** | CERRA reanalysis, 3-hourly; experiments with **1 / 3 / 5 years** (2013–2017 train, 2018 test); 5 variables × 4 pressure levels (T, Z, U, V, RH at 50/500/850/1000 hPa); grid truncated to 1024×1024 |
| **Method** | **Architecture port + full training on regional domain** (not classic weight transfer of the full NVIDIA 40-yr ERA5 checkpoint into a cropped globe). Autoregressive AFNO; **nesting / boundary forcing**: after each step, replace N-pixel border with interpolated global FCN; ablations replacing only geopotential |
| **Metrics / results** | 5-day forecasts in &lt;10 s (4×V100) / &lt;1 min (laptop GPU). **5-year CERRA training beats 3-year** on almost all 20 var×level RMSE curves. Regional model beats ERA5-interpolated global FCN over the domain for most variables; **global better on geopotential**. Nesting **mixed** when replacing all edge variables; **geopotential-only nesting** improves Z and often helps other variables |
| **Year** | 2023–2024 (AGU23 poster → 2024 papers/thesis/AGU24) |
| **URLs** | [UMBC paper PDF](https://ebiquity.umbc.edu/get/a/publication/1453.pdf); [UMBC thesis DOI](https://doi.org/10.13016/m2ksah-dwhr); [AGU ADS](https://ui.adsabs.harvard.edu/abs/2024AGUFMA31A..167H/abstract); [ESSOAr AGU23](https://doi.org/10.22541/essoar.171285742.22398551/v1) |
| **Uncertainty flag** | **Strong FCN-lineage match**, but this is **regional training of FCN architecture + nesting**, **not** a published LoRA/full FT of the official NGC SFNO/FCN3 weights onto a geographic crop. Authors also planned HRRR/CONUS extension (AGU23); **no public CONUS FCN FT results found**. |

---

### A2. Near-misses inside FCN lineage (global FT, not regional)

| Name | Why listed | Why **not** regional FT |
|------|------------|-------------------------|
| FourCastNeXt (Guo et al., 2024) | Multi-step FT of FCN-style training for limited compute | **Global** ERA5; polar behavior discussion only ([arXiv:2401.05584](https://arxiv.org/abs/2401.05584)) |
| SDSU / “Democracy of AI NWP” (Khadir & Stevenson, 2025) | Continued FT of FourCastNet backbone 18 epochs | **Global**; plateaued without new data ([arXiv:2504.17028](https://arxiv.org/abs/2504.17028)) |
| FCN3 curriculum FT stage | Official NVIDIA FCN3 | **Global** near-term ERA5 years only |
| NOAA-EMC FourCastNetv2 ops | SFNO inference with GDAS ICs | Operational **global** inference, not regional FT |

---

## B) Closely related — NVIDIA stack conditioned on / paired with FCN

These do **not** fine-tune the FCN weights for a region; they train **separate** regional models that consume FCN (or ERA5/GFS synoptic) fields.

### B1. CorrDiff — Taiwan (km-scale generative downscaling)

| Field | Detail |
|-------|--------|
| **Name** | *Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling* (CorrDiff); Communications Earth & Environment version also circulating 2025 |
| **Region** | Taiwan + surrounding ocean (~116.4–125.6°E, 19.5–27.8°N) |
| **Base checkpoint** | **Not an FCN weight FT.** Conditioning: **ERA5 25 km** (paper explicitly says inputs can come from FourCastNet / GFS). Target: CWA radar-assimilating **WRF ~2 km** |
| **Fine-tune data** | CWA-WRF hourly 2018–2020 train / 2021 test; 12 ERA5 in-channels → 4 out (t2m, u10, v10, max radar reflectivity); ~24k training images |
| **Method** | Two-step: **UNet regression (mean) + residual diffusion (EDM)**; channel synthesis (radar); ~80M UNet; 16×8 H100 for ~7 days |
| **Metrics** | Best CRPS vs UNet/RF/ERA5 interp (e.g. radar CRPS 1.90 vs UNet MAE 2.51); restores spectra/PDFs; sharpens fronts; partially intensifies typhoons; ensemble under-dispersive; ≥22× faster / ~1300× more energy-efficient than CWA-WRF on cited hardware |
| **Year** | 2023 preprint → 2024/2025 journal |
| **URL** | [arXiv:2309.15214](https://arxiv.org/abs/2309.15214) |

### B2. CorrDiff — UAE nested (Earth-2 production pipeline)

| Field | Detail |
|-------|--------|
| **Name** | G42 / Inception / Space42 + NVIDIA Earth-2 — UAE regional AI weather |
| **Region** | UAE country-wide **2 km**; Abu Dhabi hyperlocal **200 m** |
| **Base** | **Pretrained NVIDIA FourCastNet (SFNO)** in Earth2Studio for global 0.25° / 6 h; **two custom CorrDiff** models; temporal interpolation to 1 h |
| **Fine-tune data** | ERA5 0.25° inputs; WRF targets driven by GFS/GDAS, **5 years 2019–2023**, hourly; fog index + custom precip handling |
| **Method** | Nested CorrDiff (2 km → 200 m); Earth2Studio Enterprise orchestration; diagnostic fog index |
| **Metrics / results** | 1 day nested forecast ≈ **170 GPU-seconds** on H100 vs **960 CPU-core-hours** WRF at 200 m; qualitative skill on sea breeze, fog, Apr 2024 extreme rain structure |
| **Year** | NVIDIA Technical Blog **19 Mar 2025** |
| **URL** | [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-earth-2-powers-regional-ai-weather-forecasting-in-the-united-arab-emirates/) |
| **Flag** | Strongest **operational** FCN+regional example; CorrDiff FT is regional, **FCN weights not regionally fine-tuned**. |

### B3. CorrDiff — CONUS / GEFS–HRRR (PhysicsNeMo recipe)

| Field | Detail |
|-------|--------|
| **Name** | NVIDIA PhysicsNeMo CorrDiff examples (`gefs_hrrr`, `hrrr_mini`) |
| **Region** | Continental US (HRRR / GEFS–HRRR pairing) |
| **Base** | Downscaling conditioned on coarse GEFS-like fields; pretrained CONUS CorrDiff checkpoints noted for **Earth2Studio inference** (NVAIE); training usually from scratch in current `train.py` |
| **Method** | Same CorrDiff two-step (regression + diffusion / patched diffusion) |
| **Year** | Ongoing PhysicsNeMo docs (2024–2026) |
| **URL** | [PhysicsNeMo CorrDiff docs](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/corrdiff/README.html) |
| **Flag** | Recipe/checkpoint ecosystem; **not** a peer-reviewed FCN weight FT paper. |

### B4. StormCast — Central US CAM emulation

| Field | Detail |
|-------|--------|
| **Name** | *Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling* (StormCast); Science Advances |
| **Region** | Central US box ~1536×1920 km at **HRRR 3 km** |
| **Base** | Separate diffusion CAM emulator; **synoptic conditioning** from ERA5 (train) / **GFS** (forecast); paper explicitly lists **FourCastNet** as an admissible synoptic driver |
| **Fine-tune data** | HRRR v4 after Jul 2018; ~3.5 yr train through Dec 2021; 99 mesoscale + 26 synoptic channels; 1 h step |
| **Method** | CorrDiff-style **regression + residual diffusion**, **autoregressive** 1 h stepping (not pure downscaling) |
| **Metrics** | Competitive FSS/RMSE vs HRRR for radar/winds out to ~1–6 h with 5-member PMM; realistic updrafts/cold pools; under-dispersive ensembles |
| **Year** | arXiv 2024 ([2408.10958](https://arxiv.org/abs/2408.10958)); Sci. Adv. published form |
| **URL** | [arXiv:2408.10958](https://arxiv.org/abs/2408.10958); [doi:10.1126/sciadv.adv0423](https://doi.org/10.1126/sciadv.adv0423) |

### B5. FourCastNet + E-TEPS — Italy early-warning downscaling

| Field | Detail |
|-------|--------|
| **Name** | Shafei et al. — FourCastNet + Elevation-integrated TEmperature and Precipitation SRGAN (E-TEPS) |
| **Region** | Italy / Central Italy extremes (Emilia-Romagna 2023, Marche 2022 floods) |
| **Base** | **FourCastNet global** inference; separate **SRGAN** downscaler (~3 km) with elevation |
| **Fine-tune data** | CMCC high-res regional data; ERA5 in fine-tuning stage of downscaler |
| **Method** | Crop FCN outputs to Italy + elevation-conditioned SRGAN (diagnostic / downscaling head, **not** FCN backbone FT) |
| **Metrics** | MAE/RMSE/correlation vs CMCC; end-to-end &lt;1 min; topography-related caveats |
| **Year** | 2024 preprint |
| **URL** | [doi:10.20944/preprints202408.1322.v1](https://doi.org/10.20944/preprints202408.1322.v1) |
| **Flag** | FCN used as **frozen global driver**; regional adaptation is the downscaler. |

### B6. Makani / Earth2Studio — capability without published regional FT

| Field | Detail |
|-------|--------|
| **Makani** | Official training stack for FCN1 / SFNO / HENS-SFNO / **FCN3**; YAML-configurable FT, multi-step, losses, channel weights ([NVIDIA/makani](https://github.com/NVIDIA/makani)) |
| **Earth2Studio** | Inference orchestration: FCN3, SFNO, CorrDiff, StormCast, HENS recipes with **regional crop boxes for I/O** (not weight FT) |
| **Published regional Makani FT of FCN3/SFNO** | **None found** with metrics as of Sep 2026 |

---

## C) Other global MLWX regional PEFT — **not-FCN** (comparison only)

| Name | Region | Base (not FCN) | Method | Key result | Year / URL |
|------|--------|----------------|--------|------------|------------|
| Munir et al. *Efficient Localized Adaptation… MENA* | MENA (ERA5 crop) | **ClimaX** | Full FT vs **LoRA / resLoRA / GLoRA** | Regional ≫ global on MENA; LoRA ≈ full FT (e.g. t2m ACC 0.823 vs 0.816) with **~85% fewer params** (16.2M vs 108M) | 2024 [arXiv:2409.07585](https://arxiv.org/abs/2409.07585) |
| WeatherPEFT (Cao et al., ICLR 2026) | Regional precip (ERA5-CH / China-focused task among others) | **Aurora** (main), Prithvi-WxC; FCN only as from-scratch baseline | Task-adaptive PEFT (TADP + SFAS) | Matches full FT with far fewer trainable params on downscaling / regional precip / post-processing | 2025–2026 [arXiv:2509.22020](https://arxiv.org/abs/2509.22020) |
| Baño-Medina et al. Western US ARs | Western US 6 km (stretched grid) | **Graph-transformer / AIFS-like**, CW3E+ERA5 | Stretched-grid autoregressive (multi-stage train) | Competitive with West-WRF 9 km; better extremes/ARs than coarse global AI/IFS | 2025 [npj Clim. Atmos. Sci.](https://doi.org/10.1038/s41612-025-01265-9); related Nipen stretched-grid [arXiv:2409.02891](https://arxiv.org/abs/2409.02891) |
| IndiaWeatherBench | India / IMDAA | UNet, Transformers, **GraphCast-style**, etc. | Regional benchmark from scratch / adapted globals | Standardized India regional MLWX testbed | 2025 [arXiv:2509.00653](https://arxiv.org/abs/2509.00653) |
| Pangu-driven WRF North China | North China extreme precip | **Pangu-Weather** → regional WRF | AI-driven dynamical downscaling (not PEFT of Pangu) | Improves disastrous precip forecasts | 2024 [IOPscience](https://iopscience.iop.org/article/10.1088/1748-9326/ad41f0) |
| Monsoon onset benchmarking India | India | AIFS, FuXi, GraphCast, GenCast, NeuralGCM (eval only) | Decision-oriented verification | AIWP useful ~15 d; localization noted as advantage | EGU26 abstract |

**Explicitly not FCN:** MENA LoRA paper cites FourCastNet only as related work; experiments are on **ClimaX**.

---

## Region × approach matrix (quick view)

| Region | True FCN FT / FCN-arch regional | NVIDIA stack (B) | Notable not-FCN (C) |
|--------|----------------------------------|------------------|---------------------|
| **Europe** | Hamer nested FCN–CERRA 5.5 km | — | — |
| **Taiwan** | — | CorrDiff 2 km | — |
| **UAE / MENA** | — | CorrDiff 2 km + 200 m + SFNO | ClimaX LoRA MENA |
| **CONUS / Central US** | Planned HRRR FCN (no public results) | StormCast 3 km; CorrDiff GEFS–HRRR | — |
| **Western US** | — | — | Stretched-grid AR model 6 km |
| **Italy** | — | FCN + E-TEPS ~3 km | — |
| **India** | **None found** | — | IndiaWeatherBench; monsoon eval |
| **China / East Asia** | **None found** | — | WeatherPEFT regional precip; Pangu+WRF |
| **Himalaya** | **None found** | — | — |
| **Global only** | FCN3 / SFNO / FourCastNeXt / HENS | — | — |

---

## Explicit search coverage (requested queries)

| Query theme | Outcome |
|-------------|---------|
| “FourCastNet fine-tune” | Mostly **global** multi-step FT (original FCN, FourCastNeXt, FCN3 curriculum); regional hits → Hamer nested CERRA |
| “FourCastNet regional” | Hamer/UMBC nested Europe; AGU abstracts; UAE blog (FCN+CorrDiff) |
| “SFNO fine-tune” | Global Makani/NIM finetuned SFNO checkpoint (`earth2-sfno-era5-73ch:…-finetuned`); **not regional** |
| “FCN3 fine-tune” | Global only; Earth2Studio issue denies public IC-domain FT releases |
| “Makani fine-tune regional” | Framework capable; **no published regional FCN3/SFNO FT study** |
| India / China / Europe / US / MENA / Himalaya FCN adaptation | Europe: Hamer; MENA: **ClimaX not FCN**; UAE: CorrDiff+SFNO; US: StormCast/CorrDiff; India/China/Himalaya FCN FT: **none** |
| Earth2Studio fine-tune | Inference + CorrDiff packaging; FT deferred to PhysicsNeMo/Makani |

---

## What has been MOST EFFECTIVE? (evidence ranking)

**Evidence for true FCN regional weight FT is thin — say so explicitly.** Ranking below mixes (i) published FCN-lineage regional results and (ii) carefully labeled analogy from the adjacent NVIDIA stack and non-FCN PEFT.

### Ranked by published regional skill / operational maturity

1. **Most effective in practice for km-scale regional products in the FCN ecosystem: CorrDiff-style generative residual diffusion nested under a frozen global FCN/SFNO**  
   - **Evidence:** Taiwan CorrDiff CRPS/spectra/case studies ([arXiv:2309.15214](https://arxiv.org/abs/2309.15214)); UAE production pipeline with SFNO + nested 2 km/200 m CorrDiff and large wall-clock wins ([NVIDIA blog 2025](https://developer.nvidia.com/blog/nvidia-earth-2-powers-regional-ai-weather-forecasting-in-the-united-arab-emirates/)).  
   - **Why it works:** Keeps spherical/global FCN intact; puts capacity into learning unresolved scales, topography, and new channels (radar/fog/precip).

2. **Best for convection-allowing *autoregressive* regional emulation: StormCast (regression + residual diffusion, synoptic-conditioned)**  
   - **Evidence:** Competitive HRRR-relative FSS/RMSE 1–6 h over Central US ([arXiv:2408.10958](https://arxiv.org/abs/2408.10958)).  
   - **Analogy to FCN:** Uses FCN-class synoptic drivers; does not FT FCN weights.

3. **Best published *FCN-architecture* limited-area forecast: Hamer nested AFNO on CERRA**  
   - **Evidence:** Clear RMSE gains from more regional years; selective geopotential nesting helps ([UMBC PDF](https://ebiquity.umbc.edu/get/a/publication/1453.pdf)).  
   - **Caveat:** Nesting all variables is mixed; not a release-weight FT of SFNO/FCN3; modest variable set / short train eras.

4. **Most transferable PEFT lesson (not-FCN, by analogy): LoRA / task-adaptive PEFT on transformer WFMs**  
   - **Evidence:** MENA ClimaX LoRA ≈ full FT at ~15% params ([arXiv:2409.07585](https://arxiv.org/abs/2409.07585)); WeatherPEFT ≈ full FT on Aurora regional tasks ([arXiv:2509.22020](https://arxiv.org/abs/2509.22020)).  
   - **Analogy caution:** **No equivalent published LoRA study on SFNO/FCN3.** AFNO/SFNO operator structure may need different adapter placement than ViT/ClimaX.

5. **Stretched-grid high-res globals (not-FCN)** can beat uniform coarse AI on regional precip/ARs (Western US) but are a different design axis than FCN FT.

### Practical recommendation for an FCN-family regional project (evidence-based)

| Goal | Prefer | Avoid assuming |
|------|--------|----------------|
| Km-scale precip/radar/fog over a country | **Frozen SFNO/FCN3 + CorrDiff (PhysicsNeMo)** | That FCN3 regional LoRA already exists |
| Storm-scale 1–12 h CONUS | **StormCast-class** CAM emulator | Pure global FCN crop |
| Medium-range regional dynamics on regional reanalysis | **FCN-arch limited-area + selective nesting** (Hamer-style) | Blind full-border replacement |
| Parameter-efficient localization | Try **LoRA/WeatherPEFT-style** adapters — but treat as **R&D** on SFNO/FCN3 | Direct transfer of ClimaX LoRA results |

---

## Gaps & open opportunities (Sep 2026)

1. **No public FCN3 regional FT** (confirmed).  
2. **No public SFNO LoRA/crop FT** with WeatherBench-style regional scores.  
3. **India / Himalaya / China** lack FCN-lineage regional FT papers (rich not-FCN activity instead).  
4. Hamer CONUS/HRRR FCN plan **not evidenced** as a completed public release.  
5. Makani makes regional FT *possible*; community results have not yet appeared for FCN3.

---

## Search / provenance notes

- Primary tools: WebSearch + WebFetch (arXiv HTML, NVIDIA blogs, UMBC PDF, PhysicsNeMo/Earth2Studio docs).  
- Prefer peer-reviewed / arXiv / official NVIDIA over secondary blogs when conflicting.  
- Preprints and blog case studies labeled as such.  
- “Uncertain match” used when FCN is only a conditioning input or citation rather than the fine-tuned backbone.

---

*End of survey.*
