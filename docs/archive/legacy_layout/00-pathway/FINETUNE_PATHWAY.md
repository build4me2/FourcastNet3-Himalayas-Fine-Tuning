# FCN3 Regional Fine-Tune — Complete Pathway (Reference & Tracking)

**Project:** FourCastNet3 → Nepal / South Asian mountain skill  
**Method family:** RRCA-FD (Regionally Reweighted CRPS Adapter + Frozen Diagnostic)  
**Hardware:** 2× DGX Spark (128 GB unified each)  
**Owner (research):** Sheldon · **Experiments:** Leonard · **Eng (later):** Howard · **Compute ops:** Raj  
**Last updated:** 2026-09-11 (Tier-A v1.1 interim PASS; G0 PASS; years unlock pending)
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
| **1** | Define task & scope | ✅ largely done | Goals locked; box + t2m/winds-first locked; years still open |
| **2** | Select pre-trained model | ✅ done | FCN3 (`nvidia/fourcastnet3`), Apache-2.0 |
| **3** | Prepare & format data | 🔄 in progress | Max-through-latest pull; 80/20 policy locked; year hard-lock still open |
| **4** | Choose FT technique | ✅ done | RRCA-FD Plan A primary; Plan B gated |
| **5** | Set up training & run | 🔄 Phase 0–A | Tier-A v1.1 interim PASS; G0 verifying PASS; diffusion held |
| **6** | Evaluate & validate | 🔄 partial | G0 verifying PASS; Tier-0 beat-this + v0 test champ frozen; G1 regional ongoing |

**Overall:** Tier-A **v1.1 interim PASS** frozen (v0 test champ); **G0 verifying PASS**; Tier-0 beat-this frozen. **Next unlock:** Manisha **year hard-lock** / more ICs before diffusion. Sheldon = docs.

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

**Locked policy (Manisha, 2026-09-10):** pull **as much data as possible through the latest date available** for each source (ERA5 FCN3 channels + aux for Nepal/HKH crop; IMDAA/IMERG/etc. when accessible). Do **not** artificially truncate the end date. **Owner: Howard** (acquisition + eng). Sheldon = pathway/docs only unless asked.


**Latest available (research check 2026-09-10 PT — for Howard’s pull):**
| Product | Through | Note |
|---------|---------|------|
| ERA5 single-levels (CDS, ERA5T tip) | **2026-09-05** | ~5-day latency |
| ERA5 pressure-levels (CDS) | **2026-09-04** | |
| ERA5 final (validated) | ~**2026-06** | last ~2–3 months still ERA5T |
| IMERG Final V07 | **2025-09-30** | NRT Early/Late after that |
| IMDAA reanalysis | **1979–2020** | post-2020 = IMDAA-Like (separate) |

**Pull guidance:** Nepal/HKH **area crop** (e.g. 26–31°N, 80–89°E), 6-hourly, 72 FCN3 channels — **~15 GB** float32 for 1980→2026-09 vs multi-TB if global. Prefer final ERA5 through ~2026-06; optionally append ERA5T to tip if true max calendar coverage. CDS creds required on Spark.


| Stream | Role | Status |
|--------|------|--------|
| ERA5 (72 FCN3 channels + aux) | ICs / Tier-B labels / spectral anchor | ⏳ Howard — through **latest CDS/ERA5 available** |
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

**Locked policy (Manisha, 2026-09-10):** after all usable samples are accumulated, split **~80% train / ~20% test**, with **random selection** (fixed seed for reproducibility). Test must **not be tiny** — target **≥20%** of the pool (never <15% without an explicit exception). Prefer a further **train→val** carve from the 80% (e.g. 70% train / 10% val / 20% test) so early stopping does not peek at test.

| Split | Ratio | Selection | Status |
|-------|-------|-----------|--------|
| **Train** | ~80% of pool (or ~70% if val carved) | Random from eligible ICs/samples | ⏳ policy locked; not executed |
| **Val** (recommended) | ~10% carved from train pool | Random, same seed family | ⏳ |
| **Test** | **~20%** (floor **15%**) | Random holdout; **never** used for training or HP search | ⏳ |
| **Global probe (G0)** | Fixed non-Asia cases | Separate from regional 80/20; not counted in the 20% | ⏳ |

**How to do “random” safely for weather (important):**  
Do **not** shuffle individual adjacent 6 h frames independently (autocorrelation → leakage). Randomize at a **block** level, then assign whole blocks to train or test:

1. Form units = **calendar days** or **3–7 day blocks** (or whole monsoon seasons if N is small).  
2. Shuffle block IDs with a fixed seed (e.g. `seed=42`).  
3. Assign first ~80% of blocks → train(+val), last ~20% → test.  
4. Optionally stratify so monsoon / pre-monsoon / winter each appear in both sides (better than pure luck on season mix).  
5. Write `splits.json` with seed, block size, counts, and hashes — Howard/Leonard use this as source of truth.

**Minimum test size:** if the total pool is small, keep **20%** and add samples rather than shrink test below 15%. If still < ~500 test ICs at 6 h, flag for Manisha before training.

**Status:** ✅ ratio + random policy locked in pathway · ❌ not applied on disk yet

### Step 3 checklist

- [x] Data sources identified
- [x] Split policy: **80/20**, random block-level, test ≥15–20%
- [ ] ERA5 Nepal/HKH crop on disk (**through latest available** — Howard)
- [ ] Execute split → `splits.json` (seeded)
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
| **0** | Spark bring-up, FCN3 inference smoke | ✅ / 🔄 G0 PASS |
| **1** | Tier-0 bias probe | ✅ beat-this frozen |
| **2** | Tier-A regression | ✅ v1.1 interim PASS (v0 test champ) |
| **3** | Tier-A diffusion / precip | 🔒 wait year hard-lock / more ICs |
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
| 2026-09-11 | Box **26–31N 80–89E**; v1 = **t2m/winds-first** (precip later); Tier-0 ERA5 interim OK; years freeze later (suggest 2018–21 / 2022 / 2023–24) | Manisha → Howard (Sheldon pathway) |

---

## Open decisions (need Manisha)

1. ~~Exact geographic box?~~ → **locked:** **26–31°N, 80–89°E**
2. Train/val/test years? → **still open — BLOCKS diffusion**; Howard suggest 2018–21 / 2022 / 2023–24
3. ~~Start Phase 0?~~ → **yes** (resumed; G0 GPU running)
4. ~~Precip in v1?~~ → **locked:** **t2m/winds-first**; precip later
5. ~~80/20 split?~~ → **locked:** 80/20 random block-level; test ≥15–20%
6. ~~Data end date?~~ → **locked:** through **latest available** per source (Howard pulls)
7. ~~Tier-0 labels?~~ → **locked:** ERA5 interim OK until IMDAA/obs


---

*Living document — update the dashboard when a checklist item flips.*
