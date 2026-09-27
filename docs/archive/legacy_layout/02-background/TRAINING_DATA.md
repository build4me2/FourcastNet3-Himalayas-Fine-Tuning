# FourCastNet 3 (FCN3) — Training Data Report

**Prepared for:** Manisha Chand  
**Research date:** 2026-09-09  
**Primary sources preferred:** arXiv:2507.12144, NVIDIA NGC / Hugging Face model cards, Earth2Studio, Makani

---

## Executive summary

**FourCastNet 3 (FCN3 / FourCastNet3 / NVIDIA FourCastNet v3)** is trained on **ECMWF ERA5 reanalysis** only (plus static/auxiliary fields and stochastic noise). The released Earth-2 checkpoint forecasts **72 channels** on a **0.25° equirectangular 721×1440** grid with a **6-hour** model timestep. Training uses a **curriculum** on hourly-then-6-hourly ERA5, with final fine-tuning on **2012–2016**.

**Important:** “FourCastNet3” is **not** the same as **FourCastNet v2 / SFNO**. SFNO is a separate lineage (often marketed as FourCastNet2). See §9–10.

---

## 0. Naming / which release this applies to

| Name | What it is | Training-data note |
|------|------------|-------------------|
| **FourCastNet 3 / FCN3** (Bonev et al., 2025) | Probabilistic spherical CNN / neural-operator model; NGC `earth-2/fourcastnet3`, HF `nvidia/fourcastnet3`, Earth2Studio `FCN3` | **This report.** ERA5; **72** prognostic channels; paper curriculum 1980–2016 (+ fine-tune 2012–2016). |
| **FourCastNet / FCN1** (Pathak et al., 2022) | AFNO deterministic model | ERA5; **20** variables; typically 1979–2015 train. |
| **FourCastNet v2 / SFNO** (Bonev et al., 2023; Earth2Studio `SFNO`) | Spherical FNO; sometimes called “FourCastNet2” | ERA5; **73** variables (includes **surface pressure `sp`**); SFNO paper: 1979–2015 train / 2016–17 val / 2018 test. |
| **Huge Ensembles SFNO** (Mahesh et al., 2024) | Large SFNO ensembles | Related to SFNO lineage, **not** FCN3. |

**Released FCN3 artifacts (same model card content):**

- NGC: [Earth-2 FourCastNet 3](https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3) — Model Version **v1**, package **0.1.0**, released **2025-07-18**
- Hugging Face: [nvidia/fourcastnet3](https://huggingface.co/nvidia/fourcastnet3)
- Earth2Studio package pin (as of docs): `hf://nvidia/fourcastnet3@76ef0c60237e458b33196ba027134e27f3fc4538`
- Training framework: [NVIDIA/makani](https://github.com/NVIDIA/makani)

**Uncertainty:** HF `config.json` experiment paths still use names like `fcn2_sc2_...` / `fourcastnet3.yaml`, but `"nettype": "FCN3"`. Treat as FCN3 training config naming leftovers, not a second public FCN3 data recipe.

---

## 1. Exact dataset name(s)

| Role | Dataset | Source |
|------|---------|--------|
| **Primary training / targets** | **ERA5** (ECMWF Reanalysis v5) — single levels + pressure levels | Hersbach et al. 2020; CDS |
| CDS links cited by paper | [reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) (paper also implies pressure-level products for `z/t/u/v/q`) | arXiv:2507.12144 Data availability |
| **Static / auxiliary inputs (not learned targets)** | Land–sea mask, orography, solar **cosine zenith angle** | Paper Table 1 / §C.1; HF config `add_landmask`, `add_orography`, `add_zenith` |
| **Stochastic inputs** | 8 spherical diffusion noise channels | Paper §B.7 / Table 1 |
| **Other datasets?** | **None documented** for FCN3 training (no IFS analysis, no HRES, no synthetic climate runs as training targets) | — |

**Uncertainty:** Model card describes ERA5 generically (“30 km grid … 137 levels”) — that is **full ERA5 product metadata**, not the **72-channel / selected-level subset** used by FCN3.

---

## 2. Spatial resolution / grid

| Item | Value | Source |
|------|-------|--------|
| **Native I/O grid** | **0.25°** equirectangular **lat–lon**, **721 × 1440** | Paper §3–4, Appendix C; NGC/HF; Earth2Studio |
| **Latitude** | 90° → −90° (721 points, 0.25° spacing) | HF `config.json` `lat`; Earth2Studio `FCN3` |
| **Longitude** | 0° → 359.75° (1440 points, endpoint excluded) | Same |
| **Internal latent grid** | Downsampled to **360 × 720 Gaussian** grid inside encoder | Paper §3 / Appendix C |
| **Approx. physical scale** | ~25–31 km (model cards say “~30 km”) | NGC/HF |

Earth2Studio text says “south-pole excluding” for the equirectangular layout; the published coordinate vectors still use the standard **721×1440** ERA5 lat–lon mesh (poles present as grid rows). Flag as **docs wording vs standard ERA5 grid**.

---

## 3. Temporal coverage — training vs validation / test

### 3.1 Paper (authoritative for how FCN3 was trained) — arXiv:2507.12144 Appendix E

| Split | Years | Notes |
|-------|-------|--------|
| **Training (pre-train)** | **1980–2016** | Stage 1: **hourly** samples; Stage 2: **6-hourly** ICs |
| **Fine-tuning** | **2012–2016** (subset of train era) | 6-hourly; 8-step autoregressive |
| **Test** | **2017** | Paper §E.4 |
| **Out-of-sample / validation** | **2018–2021** | Metrics in paper mainly on **2020** (12-hourly ICs) |

Main text also states: “FCN3 is trained on historic atmospheric ERA5 reanalysis data ranging from **1980 to 2016**.”

**Internal paper inconsistencies (flag):**

- Table 3 Stage 1: “1-hourly, **1980–2016**, 332,800 samples”
- Body of §E.2: “hourly … ranging from **1979 to 2016**, which amounts to … 332,800 samples”
- Prefer **1980–2016** (Table 3 + main text + §E.4 split), and treat **1979** as a likely slip unless clarified in a later revision.

### 3.2 NGC / Hugging Face model card (differs from paper)

| Split | Years (model card) |
|-------|--------------------|
| Training | **1980–2015** |
| Testing | **2016–2017** |
| Evaluation | **2018–2019** |
| Partition | “training 95%, testing 2.5%, validation 2.5%”; **110,960** “data points” |

**Flag — conflict:** Model card years/partitions **do not match** Appendix E (train through **2016**, test **2017**, OOS **2018–2021**, eval emphasis **2020**). Prefer the **paper** for scientific training protocol; treat the model card as a **simplified / possibly copy-pasted** datasheet unless NVIDIA updates it.

---

## 4. Temporal resolution and forecast lead / timestep

| Item | Value | Source |
|------|-------|--------|
| **ERA5 native** | Hourly | ERA5; paper |
| **Model forecast Δt** | **6 hours** | Paper; Earth2Studio; NGC (“60-day … 6-hourly”) |
| **Stage 1 pairs** | From **hourly** ERA5, build **6 h** input→target pairs starting at **every UTC hour** | Main text §4; Appendix E |
| **Stage 2 / fine-tune ICs** | **00:00, 06:00, 12:00, 18:00 UTC** only | Appendix E.2–E.3 |
| **Autoregressive depth in training** | Stage 1: **1** step; Stage 2: **4** steps; Fine-tune: **8** steps | Table 3 |
| **Inference rollouts** | Demonstrated to **15 days**, **30 days**, up to **~60 days** | Paper abstract / results |

HF config: `"dhours": 6` (model step), with dataset description for the on-disk corpus: *“ERA5 data at 6 hourly frequency with snapshots at 0000, 0600, 1200, 1800 UTC”* (that describes the **6-hourly shard**; Stage 1 additionally used hourly data per the paper).

---

## 5. Full list of atmospheric / surface VARIABLES (channels)

### 5.1 Prognostic I/O — **72 channels**

**Surface / near-surface (7):**

| Code | Description | Unit (Earth2Studio) | Norm (paper Table 4) |
|------|-------------|---------------------|----------------------|
| `u10m` | 10 m U wind | m s⁻¹ | z-score |
| `v10m` | 10 m V wind | m s⁻¹ | z-score |
| `u100m` | 100 m U wind | m s⁻¹ | z-score |
| `v100m` | 100 m V wind | m s⁻¹ | z-score |
| `t2m` | 2 m temperature | K | z-score |
| `msl` | Mean sea-level pressure | Pa | z-score |
| `tcwv` | Total column water vapour | kg m⁻² | **min/max** |

**Pressure-level fields (5 vars × 13 levels = 65):**  
Levels (**hPa**): **50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000**

| Field | Codes | Unit | Norm |
|-------|-------|------|------|
| Geopotential | `z50` … `z1000` | m² s⁻² | z-score |
| Temperature | `t50` … `t1000` | K | z-score |
| U wind | `u50` … `u1000` | m s⁻¹ | z-score |
| V wind | `v50` … `v1000` | m s⁻¹ | z-score |
| Specific humidity | `q50` … `q1000` | kg kg⁻¹ | **min/max** |

**Ordered channel list** (NGC / HF / Earth2Studio):

`u10m, v10m, u100m, v100m, t2m, msl, tcwv, u50, u100, u150, u200, u250, u300, u400, u500, u600, u700, u850, u925, u1000, v50, v100, v150, v200, v250, v300, v400, v500, v600, v700, v850, v925, v1000, z50, z100, z150, z200, z250, z300, z400, z500, z600, z700, z850, z925, z1000, t50, t100, t150, t200, t250, t300, t400, t500, t600, t700, t850, t925, t1000, q50, q100, q150, q200, q250, q300, q400, q500, q600, q700, q850, q925, q1000`

### 5.2 Auxiliary / conditioning (inputs only)

From paper Table 1 and HF `aux_channel_names`:

- Land–sea mask (land / sea)
- Orography
- Cosine solar zenith angle (`xzen`)
- **8** noise channels (`xnoise0`…`xnoise7`) from spherical diffusion processes

### 5.3 Present in on-disk ERA5 pack but **not** FCN3 prognostic channels

HF `data_channel_names` includes **`sp`** (surface pressure), **`sst`**, **`tp`** (total precipitation).  
`in_channels` / `out_channels` **skip `sp`** (index 5) and do not use `sst`/`tp`. Paper: precipitation planned as future diagnostic, **not** in current training outputs.

**vs SFNO 73-ch:** SFNO keeps **`sp`** → **73** channels; FCN3 drops it → **72**.

---

## 6. Number of samples / timesteps

### From paper Table 3 (best published training counts)

| Stage | Data frequency | Years | **Samples** | Grad steps |
|-------|----------------|-------|-------------|------------|
| Pre-train 1 | Hourly → 6 h pairs | 1980–2016 | **332,800** | 208,320 |
| Pre-train 2 | 6-hourly | 1980–2016 | **55,460** (text once says ~55,400) | 5,040 |
| Fine-tune | 6-hourly | 2012–2016 | **5,840** | 4,380 |

Batch / ensemble (Table 3): Stage1 BS=16, ens=16; Stage2 BS=32, ens=2; Fine-tune BS=4, ens=4.

### From NGC/HF model card

- **Total size:** **110,960** “data points”
- Split: 95% / 2.5% / 2.5%

**Flag:** **110,960 is not reconciled** with Table 3 (332,800 / 55,460 / 5,840). Do **not** treat 110,960 as the paper’s hourly training-set size without further confirmation from NVIDIA.

### Rough geometry of one sample

One atmospheric state ≈ **72 × 721 × 1440** floats (~**39.5 TB** for the training subset cited in §E.4).

---

## 7. Normalization / preprocessing

Documented in paper §E.4, Table 4, §C.8, and HF `config.json` / Earth2Studio comments:

1. **Z-score** for most channels (surface winds, `t2m`, `msl`, and pressure-level `z/t/u/v`).
2. **Min–max** for water channels: **`tcwv` and all `q*`** (needed for positive soft-clamp output).
3. Stats are **spatially averaged over the sphere**, then averaged over the **training set** (means/stds/mins/maxs).
4. **Winds:** `u`/`v` treated as **zero mean**; scaled by **std of wind speed magnitude** (preserves direction).
5. **Loss channel weights** `w_c` (Table 4): e.g. `t2m` = 1.0; many surface fields 0.1; pressure-level weights ∝ `p · 10⁻³`.
6. Additional **temporal difference weighting** `w_{Δt,c}` from climatological std of **1-hourly** differences (GraphCast-style; paper §E.1). HF: `"temp_diff_normalization": true`.
7. **No layer normalization** inside the network (architectural choice).
8. **Water output:** smooth **softclamp / spline** positivity constraint (`clamp_water: true`).
9. **Predicts next state directly** (not tendency / residual prediction).
10. Artifacts on HF/NGC package: `global_means.npy`, `global_stds.npy`, `mins.npy`, `maxs.npy`, `time_means.npy`, `time_diff_*.npy`, `orography.nc`, `land_mask.nc`.

---

## 8. Differences vs FourCastNet / FourCastNet2 training data

| | **FCN1 (AFNO)** | **FCN2 / SFNO** | **FCN3** |
|--|-----------------|-----------------|----------|
| Paper | Pathak et al. 2022, [arXiv:2202.11214](https://arxiv.org/abs/2202.11214) | Bonev et al. 2023, [arXiv:2306.03838](https://arxiv.org/abs/2306.03838) | Bonev et al. 2025, [arXiv:2507.12144](https://arxiv.org/abs/2507.12144) |
| Dataset | ERA5 | ERA5 | ERA5 |
| Channels | **20** (few levels; RH not q; includes `sp`) | **73** (includes **`sp`**) | **72** (**no `sp`**) |
| Humidity | Relative humidity at selected levels | Specific humidity `q` @ 13 levels | Specific humidity `q` @ 13 levels |
| Near-surface winds | 10 m | 10 m + **100 m** | 10 m + **100 m** |
| Grid | 0.25°, ~721×1440 (some docs 720×1440) | 0.25°, 721×1440 | 0.25°, 721×1440 |
| Train years (typical) | **1979–2015** (paper); some follow-ups use through 2017 | **1979–2015** train; 2016–17 val; 2018 test | Paper: **1980–2016** (+ FT **2012–2016**) |
| Temporal sampling | 6-hourly (00/06/12/18) | 6-hourly | Stage1 **hourly pairs**; then 6-hourly |
| Task | Deterministic | Deterministic SFNO | **Probabilistic** (ensemble CRPS + spectral CRPS) |
| Norm | Global z-score files | center/scale tensors | z-score + **minmax for water**; wind-magnitude std |

---

## 9. Citations (real URLs / arXiv IDs only)

1. **FCN3 paper:** Bonev, Kurth, Mahesh, et al. (2025). *FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale.* **arXiv:2507.12144** — https://arxiv.org/abs/2507.12144 — HTML: https://ar5iv.labs.arxiv.org/html/2507.12144  
2. **ERA5:** Hersbach et al. (2020). *The ERA5 global reanalysis.* QJRMS. (cited as [40] in FCN3 paper)  
3. **CDS ERA5 single levels:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels  
4. **NGC model card:** https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3  
5. **Hugging Face model card / config:** https://huggingface.co/nvidia/fourcastnet3 — raw config: https://huggingface.co/nvidia/fourcastnet3/raw/main/config.json  
6. **Earth2Studio FCN3:** https://nvidia.github.io/earth2studio/modules/generated/models/px/earth2studio.models.px.FCN3.html — scorecard: https://nvidia.github.io/earth2studio/main/scorecard/generated/fcn3/  
7. **Makani training code:** https://github.com/NVIDIA/makani  
8. **torch-harmonics:** https://github.com/NVIDIA/torch-harmonics  
9. **FCN1:** Pathak et al. (2022). **arXiv:2202.11214** — https://arxiv.org/abs/2202.11214  
10. **SFNO / FCN2 architecture paper:** Bonev et al. (2023). **arXiv:2306.03838** — https://arxiv.org/abs/2306.03838  
11. **NVIDIA Earth-2 open models blog** (mentions FourCastNet3; little training-data detail): https://blogs.nvidia.com/blog/nvidia-earth-2-open-models/

---

## 10. Uncertainties checklist

| Issue | Severity | Guidance |
|-------|----------|----------|
| Model card years (1980–2015 / 2016–17 / 2018–19) vs paper (1980–2016 / 2017 / 2018–2021, eval 2020) | **High** | Prefer **paper Appendix E** for training protocol |
| Model card **110,960** vs Table 3 **332,800 / 55,460 / 5,840** | **High** | Prefer **Table 3** |
| 1979 vs 1980 in §E.2 body vs Table 3 | Medium | Prefer **1980** |
| Whether Stage-1 hourly files are redistributed publicly | Medium | Paper cites CDS; 39.5 TB subset; Makani expects `/train` shards |
| “FourCastNet3” vs leftover `fcn2_*` config names | Low | Checkpoint is FCN3 (`nettype: FCN3`) |
| Precipitation / SST in H5 metadata but not trained outputs | Low | Confirmed unused as FCN3 channels |

---

## Quick reference (copy box)

```
Dataset:     ERA5 (ECMWF) + land/sea mask, orography, cos zenith, 8 noise chans
Grid:        0.25° equirectangular 721×1440 (latent 360×720 Gaussian)
Channels:    72 (7 surface + 5 fields × 13 levels); NO surface pressure
Levels:      50,100,150,200,250,300,400,500,600,700,850,925,1000 hPa
Fields:      u,v,z,t,q + u10m,v10m,u100m,v100m,t2m,msl,tcwv
Δt:          6 h forecast step
Train years: 1980–2016 (paper); fine-tune 2012–2016
Samples:     ~332.8k hourly (stage1); ~55.5k 6-h (stage2); 5.84k fine-tune
Norm:        z-score (most); minmax (q, tcwv); wind via |V| std
```
