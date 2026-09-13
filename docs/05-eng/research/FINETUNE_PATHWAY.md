# FCN3 Regional Fine-Tune — Complete Pathway (Reference & Tracking)

**Project:** FourCastNet3 → Nepal / South Asian mountain skill  
**Method family:** RRCA-FD (Regionally Reweighted CRPS Adapter + Frozen Diagnostic)  
**Hardware:** 2× DGX Spark (128 GB unified each)  
**Owner (research):** Sheldon · **Experiments:** Leonard · **Eng (later):** Howard · **Compute ops:** Raj  
**Last updated:** 2026-09-10  
**Status legend:** ✅ done · 🔄 in progress · ⏳ next · 🔒 blocked · ❌ not started · — N/A

**Companion docs (same folder):**
- `TRAINING_DATA.md` — what FCN3 trained on
- `TRAINING_METHOD.md` — Lead’s picture of how FCN3 was built
- `REGIONAL_FINETUNE_GAP.md` — why regional adaptation is open
- `FCN_REGIONAL_FINETUNES.md` — who fine-tuned FCN-family elsewhere
- `SPARK_FINETUNE_PLAN.md` — what 2 Sparks can / can’t do
- `FINETUNE_METHOD_DESIGN.md` — RRCA-FD technique (canonical method)

---

## Tracking dashboard

| Step | Title | Status | Notes |
|------|--------|--------|-------|
| **1** | Define task & scope | ✅ largely done | Goals, invariants, metrics drafted; freeze years/box still open |
| **2** | Select pre-trained model | ✅ done | FCN3 (`nvidia/fourcastnet3`), Apache-2.0 |
| **3** | Prepare & format data | 🔄 partial | Sources identified; crops/splits/masks not locked on disk |
| **4** | Choose FT technique | ✅ done | RRCA-FD Plan A primary; Plan B gated |
| **5** | Set up training & run | ❌ not started | Waiting on Phase 0 bring-up |
| **6** | Evaluate & validate | ⏳ gates defined | G0–G3 written; no runs yet |

**Overall:** Design / research phase complete → **next = Phase 0 bring-up** (inference + data lock).

---

## 1. Define the Task and Scope

> LLM guides say: identify goal, success metrics, budget/compute. Same here — adapted to weather ML.

### 1.1 Goal (what we are teaching)

| Dimension | Decision |
|-----------|----------|
| **Not** | Generic “instruction following” or chat format |
| **Yes** | **Domain / regional adaptation:** better probabilistic skill over Nepal–HKH / South Asian orography |
| **Preserve** | Lead’s FCN3 picture: probabilistic 6 h spherical ensembles, 72-ch state, anisotropic+spectral ops, spatial+spectral CRPS |
| **Not copying** | NVIDIA Stage1→2→FT curriculum replay on Sparks |

**Status:** ✅ goal locked in conversation + `FINETUNE_METHOD_DESIGN.md`

### 1.2 Success metrics (when is training “done”)

Hard gates from RRCA-FD (all required):

| Gate | Metric | Pass rule | Status |
|------|--------|-----------|--------|
| **G0** | Global CRPS/SSR + spectra @ +15 d | ≤~5% relative CRPS degrade vs base; no spectral collapse | ❌ not run |
| **G1** | Nepal t2m / winds / moisture / precip CSI | Beat frozen FCN3 crop **and** Tier-0 bias baseline; elevation-banded t2m improves | ❌ not run |
| **G2** | Calibration (SSR, ranks) over region | Spread–skill ≈ 1 at +24–120 h | ❌ not run |
| **G3** | Honest labeling | Tier-A ≠ claim “FCN3 weight FT” unless Tier-B passes | ✅ policy set |

**Optional stretch:** monsoon case studies (2–3 extremes) documented.

**Status:** ✅ metrics defined · ❌ thresholds not yet measured on hardware

### 1.3 Budget and compute

| Resource | Plan |
|----------|------|
| **Hardware** | 2× DGX Spark, 128 GB UMA each; RoCE between nodes (~200 Gb/s) |
| **Feasible** | FCN3 **inference**; Tier-A CorrDiff/StormCast-lite train; Tier-0 bias |
| **Not feasible here** | Full FCN3 multi-step + large-ensemble weight FT (needs domain-parallel cluster) |
| **PEFT?** | Tier-B LoRA = optional / gated (activation-bound, unproven on FCN3) |
| **Escalate** | Raj / larger GPU only if Tier-B needs AR+ens domain parallel |

**Status:** ✅ assessed (`SPARK_FINETUNE_PLAN.md`)

### Step 1 checklist

- [x] Goal written
- [x] Invariants (I1–I6) written
- [x] Success gates G0–G3 written
- [x] Compute envelope decided
- [ ] Freeze lat–lon box + train/val/test years (decision still open)
- [ ] Assign primary owner for Phase 0 kickoff

---

## 2. Select a Pre-Trained Model

### 2.1 Base model choice

| Field | Choice |
|-------|--------|
| **Model** | **FourCastNet 3 (FCN3)** |
| **Checkpoint** | Hugging Face `nvidia/fourcastnet3` / NGC Earth-2 FourCastNet3 |
| **Why** | Probabilistic, spherical geometry, dual CRPS, open weights, Earth2Studio path |
| **Not choosing as backbone** | Pure MSE UNet-only (breaks Lead picture); FCN1 deterministic AFNO as sole model |

**Status:** ✅ selected

### 2.2 License

| Item | Status |
|------|--------|
| FCN3 weights + Makani | **Apache-2.0** (commercial OK with attribution) |
| ERA5 | CDS license — use OK; redistribution constraints apply |
| IMDAA / HIWAT / IMERG | Registration / attribution — **verify before ops/commercial** |

**Status:** ✅ FCN3 license OK · ⏳ regional data licenses to confirm when access opens

### Step 2 checklist

- [x] Base model chosen (FCN3)
- [x] License checked (Apache-2.0)
- [x] Inference stack identified (Earth2Studio)
- [ ] Confirm Spark CUDA / torch-harmonics / Earth2Studio install works (Phase 0)

---

## 3. Prepare and Format the Data

> Analog to LLM “collect / clean / template / split” — for FCN3: channels, grids, masks, crops, targets.

### 3.1 Collect data

| Stream | Role | Status |
|--------|------|--------|
| ERA5 (72 FCN3 channels + aux) | ICs / Tier-B labels / spectral anchor | ⏳ need download/crop pipeline |
| DEM-derived elevation / slope mask | Regional loss weights \(w_\mathcal{R}\) | ❌ not built |
| IMDAA (~12 km) | Tier-A high-res target | 🔒 access TBD |
| IMERG | Precip verify / diagnostic | ⏳ |
| HIWAT subset (optional) | km CAM target | 🔒 optional |
| Station / DHM (optional later) | Obs-aware verify | 🔒 later |

### 3.2 Clean and preprocess

| Task | Status |
|------|--------|
| Remap IMDAA ↔ FCN3 levels/vars | ❌ |
| Time align 6 h vs hourly products | ❌ |
| QC mountain precip (IMERG vs gauges) | ❌ |
| Build static \(w_\mathcal{R}\) mask | ❌ |

### 3.3 Apply “template” (FCN3 I/O contract)

| Requirement | Spec | Status |
|-------------|------|--------|
| Grid | 0.25°, 721×1440 global (crop after or mask loss) | ✅ known |
| Channels | 72 prognostics + land/sea/orography/coszen + noise | ✅ known |
| Norm | Use released `global_means/stds`, mins/maxs | ⏳ load from HF package |
| Δt | 6 h | ✅ |
| Ensemble noise | FCN3 HMM sampler at inference | ✅ keep frozen |

### 3.4 Split the dataset

| Split | Provisional suggestion | Status |
|-------|------------------------|--------|
| Train | e.g. 2018–2021 (monsoon-aware sampling) | ⏳ **not frozen** |
| Val | e.g. 2022 | ⏳ |
| Test / cases | Held monsoon extremes + elevation bands | ⏳ |
| Global probe | Fixed non-Asia ICs for G0 | ⏳ |

### Step 3 checklist

- [x] Data sources identified
- [ ] ERA5 Nepal/HKH crop on disk
- [ ] Train/val/test years frozen
- [ ] \(w_\mathcal{R}\) mask built
- [ ] IMDAA/IMERG access confirmed
- [ ] Norm tensors loaded from FCN3 package

---

## 4. Choose the Fine-Tuning Technique

> Full FT vs PEFT — for us: hybrid RRCA-FD.

### 4.1 Options considered

| Technique | Verdict |
|-----------|---------|
| **Full FCN3 FT** (all ~711M, AR+ens) | ❌ not on 2 Sparks; not our method copy of NVIDIA |
| **PEFT / LoRA on FCN3** | ⏳ Tier-B gated research |
| **Frozen backbone + regional diagnostic** | ✅ **Plan A (primary)** |
| **MSE-only FT** | ❌ forbidden (breaks Lead picture) |

### 4.2 Chosen technique: RRCA-FD

| Tier | What trains | When |
|------|-------------|------|
| **Tier-0** | Elevation-aware bias baseline | First |
| **Tier-A** | CorrDiff/StormCast-lite (or precip head) conditioned on frozen FCN3 | Primary |
| **Tier-B** | LoRA on local DISCO + regional CRPS + spectral anchor + global mix | Only if A works and memory allows |

**Loss discipline:** spatial CRPS (regionally reweighted) + spectral CRPS anchor (λ≈0.1); never MSE-only for dynamics claims.

**Status:** ✅ technique chosen (`FINETUNE_METHOD_DESIGN.md`)

### Step 4 checklist

- [x] Full vs PEFT vs diagnostic decided
- [x] Plan A / Plan B ordered
- [x] Forbidden techniques listed
- [ ] Implement Tier-0 trainer stub
- [ ] Implement Tier-A recipe (PhysicsNeMo / custom)

---

## 5. Set Up Training and Run Iterations

> Hyperparameters, “tokenize,” execute, monitor — mapped to weather tensors.

### 5.1 Configure hyperparameters (defaults)

| Knob | Tier-A | Tier-B |
|------|--------|--------|
| Precision | BF16 | BF16 AMP |
| Batch / GPU | 1 + grad accum 16–64 | 1 |
| AR depth | N/A (frame/downscale) | **1 only** on Spark |
| Train ens M | per CorrDiff recipe | 1–2 |
| LR | AdamW + cosine (UNet-scale) | Adam ~1e-5–1e-6 on adapters |
| Spectral λ | optional regional FFT small | **0.1** start |
| Global mix π | — | 0.3–0.5 |

### 5.2 “Tokenize” analog

Convert ERA5 / IMDAA fields → normalized tensors matching FCN3 channel order; write Zarr/NVMe crops; cache FCN3 ensemble outputs from Spark-1.

**Status:** ❌ not started

### 5.3 Execute training (phased)

| Phase | Work | Status |
|-------|------|--------|
| **0** | Spark bring-up, FCN3 inference smoke | ❌ |
| **1** | Tier-0 bias probe | ❌ |
| **2** | Tier-A regression | ❌ |
| **3** | Tier-A diffusion / precip | ❌ |
| **4** | G0 invariant suite | ❌ |
| **5** | Tier-B LoRA smoke (optional) | ❌ |
| **6** | Cases + write-up | ❌ |

**Monitor:** val CRPS/RMSE, PSD, GPU mem, never celebrate regional RMSE if G0 fails.

### Step 5 checklist

- [ ] Environment: Earth2Studio + PhysicsNeMo (or equivalent) on Spark
- [ ] Dual-Spark split: Spark-1 inference cache / Spark-2 train
- [ ] Logging (metrics + mem)
- [ ] First Tier-0 run complete

---

## 6. Evaluate and Validate

### 6.1 Test performance

| Suite | Against | Status |
|-------|---------|--------|
| G0 global probe | Base FCN3 | ❌ |
| G1 regional | Frozen FCN3 + Tier-0 | ❌ |
| G2 calibration | Ideal SSR/ranks | ❌ |
| Case studies | Monsoon / elevation bias | ❌ |

### 6.2 Iterate

| If… | Then… |
|-----|-------|
| Beats Tier-0 on t2m/winds, precip weak | Stay at regression; add precip diagnostic data |
| Diffusion unstable | Kill Phase 3; keep regression |
| Tier-B OOM or G0 fail | Discard adapters; Spark story = Tier-A |
| Need true FCN3 weight FT | Escalate to Raj (cluster domain parallel) |

### Step 6 checklist

- [ ] Evaluation scripts frozen
- [ ] Baseline tables filled
- [ ] Decision: Spark-complete (A) vs escalate (B+)

---

## Pathway map (one page)

```
[1 Scope ✅] → [2 FCN3 ✅] → [3 Data 🔄] → [4 RRCA-FD ✅]
                                              ↓
                                    [5 Train ❌] → Tier-0 → Tier-A → (Tier-B?)
                                              ↓
                                    [6 Eval gates G0–G3]
                                              ↓
                         Pass A → ship diagnostic story
                         Need weight FT → Raj/cluster
```

---

## Where we are (plain English)

You’re **past research/design**, **before experimental bring-up**.

**Done:** problem framing, Lead invariants, model choice, licenses for FCN3, technique (RRCA-FD), compute honesty, eval gates, literature context.

**Not done:** actual ERA5/IMDAA pipelines on disk, Spark software smoke test, any training run, any score vs baseline.

**Immediate next (Step 3 + Phase 0):**
1. Freeze lat–lon box + years  
2. Pull ERA5 crop + FCN3 norm files  
3. Run frozen FCN3 inference on Spark  
4. Build Tier-0 bias baseline  

---

## Decision log

| Date | Decision | By |
|------|----------|-----|
| 2026-09-09 | Research FCN3 training data | Manisha → Sheldon |
| 2026-09-09 | Regional FT gap for Nepal/HKH | Manisha → Sheldon |
| 2026-09-09 | Lead-level TRAINING_METHOD | Manisha → Sheldon |
| 2026-09-09 | Survey FCN regional FTs | Manisha → Sheldon |
| 2026-09-09 | 2× Spark feasibility plan | Manisha → Sheldon |
| 2026-09-10 | Custom method RRCA-FD (not NVIDIA copy); preserve Lead picture | Manisha → Sheldon |
| 2026-09-10 | Pathway tracking doc (this file) | Manisha → Sheldon |

---

## Open decisions (need Manisha)

1. Exact geographic box (narrow Nepal vs wider HKH/monsoon)?  
2. Train/val/test years?  
3. Start Phase 0 on Spark now (loop Raj for env)?  
4. Is precip in v1 success criteria or t2m/winds-first?

---

*Living document — update the dashboard when a checklist item flips.*
