# Actionable Plan: Fine-Tune Toward Nepal / South Asian Mountain Weather Skill on 2× DGX Spark

**For:** Manisha Chand  
**Date:** September 2026  
**Hardware:** 2× NVIDIA DGX Spark (GB10 / Grace Blackwell), 128 GB unified memory each  
**Base interest:** FourCastNet3 / FCN family regional adaptation  
**Method:** WebSearch + WebFetch of NVIDIA docs, FCN3 paper/blog, PhysicsNeMo (CorrDiff/StormCast), DGX Spark clustering guides; aligned with prior briefs (`TRAINING_DATA.md`, `FCN_REGIONAL_FINETUNES.md`, `TRAINING_METHOD.md`)

---

## Honest one-liner

**Full FCN3 multi-step / ensemble fine-tuning is NOT feasible on 2× Sparks.** Use Sparks for (1) FCN3/SFNO **inference**, (2) **CorrDiff / StormCast-scale** regional diagnostic or downscaling training on a Nepal box, and/or (3) a **small limited-area FCN-arch** model. Escalate to Raj / cluster only when you need FCN3 weight FT with AR + ensemble domain parallel.

---

## 1. What DGX Spark actually is

### Product framing

NVIDIA DGX Spark is a **desktop / personal AI supercomputer** built around the **GB10 Grace Blackwell Superchip** (CPU + GPU on one package with coherent unified memory). It is **not** a mini-H100/B200 with HBM: capacity is high; **bandwidth is LPDDR-class**.

Official product page: [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)  
Clustering / ConnectX-7: [DGX Spark User Guide — ConnectX-7 Networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)  
Playbook: [Connect Two Sparks](https://build.nvidia.com/playbooks/connect-two-sparks/stacked-sparks)  
Sync Cluster Assistant: [NVIDIA Sync — Cluster Assistant](https://docs.nvidia.com/sync/0.97.6/cluster-assistant.html)

### Specs (cite NVIDIA)

| Item | Spec (NVIDIA product page / docs) |
|------|-----------------------------------|
| SoC | **GB10** Grace Blackwell Superchip |
| CPU | **20-core Arm** (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | Blackwell architecture; 5th-gen Tensor Cores; 4th-gen RT |
| Memory | **128 GB LPDDR5x coherent unified** system memory (CPU+GPU same pool) |
| Memory interface / BW | 256-bit; **273 GB/s** |
| AI performance claim | **Up to 1 PFLOP FP4** (theoretical, **with sparsity**) |
| Storage | 4 TB NVMe M.2 (self-encrypting) |
| NIC | **ConnectX-7 @ 200 Gbps** (+ 10 GbE RJ-45, Wi‑Fi 7) |
| Power | PSU 240 W; GB10 TDP **140 W** |
| Size / weight | 150×150×50.5 mm; 1.2 kg |
| OS | NVIDIA DGX OS |
| On-chip CPU↔GPU | **NVLink-C2C** (coherent fabric inside the Superchip) |

NVIDIA’s own workload framing on the product page:

- **Fine-tune** models up to ~**70B** parameters (LLM-oriented marketing)
- **Inference** models up to ~**200B** (single Spark) / higher when Sparks are clustered
- ConnectX networking: product page currently says connect **up to four** Sparks for larger models (up to hundreds of B params in LLM framing)

### Multi-Spark networking (critical for weather FT expectations)

| Fact | Detail |
|------|--------|
| **Between Sparks** | **Not NVLink / NVSwitch.** Inter-node = **ConnectX-7 Ethernet / RoCEv2** at up to **200 Gb/s** per QSFP port |
| **Inside one Spark** | **NVLink-C2C** only joins Grace CPU ↔ Blackwell GPU on the same SoC |
| **2-node wiring** | One approved **QSFP** DAC between matching rear ports (NVIDIA lists Amphenol NJAAKK-N911 / Luxshare LMTQF022-SD-R) |
| **Config path** | Prefer **NVIDIA Sync Cluster Assistant**; or manual playbook (IPs on CX-7 ifaces, SSH, NCCL) |
| **Scale-out** | Direct cable: 2–3 Sparks; **4 Sparks need a switch** (per Sync / User Guide) |
| **What you get** | Data / tensor parallel over **NCCL + RoCE**, ~25 GB/s unidirectional peak — useful, **not** H100 NVLink domain-decomp bandwidth |

### Bandwidth reality (why this matters more than “128 GB”)

H100 HBM3 is on the order of **~3 TB/s**. Spark’s unified pool is **273 GB/s** (~**11–12× slower**). Weather grids (721×1440 × dozens of channels) are **activation- and bandwidth-heavy**. Fitting weights in 128 GB does **not** imply H100-like training throughput or that FCN3 AR training will be comfortable.

Hot Chips / third-party teardown notes (e.g. GB10 ~48 SMs, SM121, no HBM) reinforce: Spark is a **capacity-rich, bandwidth-modest** personal box — excellent for local agents / midsize FT / research iteration, poor substitute for Eos-scale FCN3 training.

---

## 2. Memory reality check vs FCN3 / CorrDiff / StormCast

### FCN3 facts that constrain you

Sources: [arXiv:2507.12144](https://arxiv.org/abs/2507.12144), [NVIDIA blog](https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/), [HF `nvidia/fourcastnet3`](https://huggingface.co/nvidia/fourcastnet3), [NGC Earth-2 FourCastNet3](https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3/-), [Makani](https://github.com/NVIDIA/makani)

| Fact | Number / claim |
|------|----------------|
| Parameters | **~710.9M** (~711M) |
| Grid | **0.25°**, **721×1440**, **72** prognostic channels + aux/noise |
| Timestep | **6 h** |
| Paper training | Stage 1: **1024× H100**, batch 16, ensemble 16, ~78 h; Stage 2 AR: **512× A100**; final FT: **256× H100**, 8 h |
| Domain parallel | Paper: **single model instance does not fit on 80 GB**; **4-way** spatial split in pretrain → **16-way** in AR fine-tune |
| Inference | **Single H100**: 15-day ~1 min; 60-day &lt;4 min (bf16) |
| NIM / deploy guidance | Non-optimized FCN3 often cited needing **~60 GB** GPU memory class |

Weight math alone is misleading: BF16 weights ≈ **1.4 GB**. The killers are **global activations**, **ensemble members**, **autoregressive graph**, **optimizer states**, and **spectral / spherical ops**. That is why NVIDIA used domain + batch + ensemble parallelism on ≥256–1024 GPUs.

### Feasibility matrix on **2× 128 GB Spark**

| Workload | Feasible? | Why |
|----------|-----------|-----|
| **FCN3 inference** (Earth2Studio, bf16, modest ensemble) | **YES — primary use of Spark #1** | ~60 GB class fits in 128 GB UMA; expect **slower than H100** due to 273 GB/s BW; validate SM121 / torch-harmonics CUDA builds early |
| **FCN3 full multi-step + ensemble FT** (paper-style) | **NO** | Needs **16-way** domain parallel on 80 GB H100s for AR FT; 2 Sparks ⇒ at best **2-way** over 200 GbE — orders of magnitude short |
| **FCN3 1-step LoRA / PEFT / partial FT** | **MARGINAL / RESEARCH** | Adapter params fit; **full global forward activations still dominate**. Regional loss masking does not shrink spherical graph. No public FCN3 LoRA recipe (see `FCN_REGIONAL_FINETUNES.md`) |
| **SFNO / FCN2 lighter FT** | **MARGINAL** | Smaller than FCN3 AR+ens, still global 0.25°; try only after inference baseline; prefer frozen SFNO + diagnostic |
| **CorrDiff regional** (Nepal box, patch / mini UNet) | **YES — recommended train path** | PhysicsNeMo: single GPU OK if memory managed; `batch_size_per_gpu`↓ + grad accum; CorrDiff-Mini ~**10 h on A100s** educational path; Taiwan/custom recipes exist. Note: CorrDiff example marked **deprecated** in favor of unified **StormCast** recipe — prefer StormCast for new work |
| **StormCast-scale** (full CONUS paper: ~120 h × 64 H100) | **Partial YES** | Full paper scale **no**. **Lite / custom Nepal domain**, smaller channel set, BF16 UNet, batch=1 + accum, shorter training: **yes** on 1–2 Sparks |
| **Hamer-style limited-area AFNO** from scratch | **YES** | Europe/CERRA trained on **4× V100 32 GB**; Spark 128 GB is **more capacity** than that setup |
| **Distill / bias-correct / post-process only** | **YES — always do this** | Cheap baselines; NVIDIA staff have suggested diagnostic bias-correctors over FT releases ([earth2studio#879](https://github.com/NVIDIA/earth2studio/issues/879)) |

### Bottom line

| Question | Answer |
|----------|--------|
| Can 2×128 GB do **full FCN3 FT**? | **No** |
| Can they do **LoRA of FCN3** usefully? | **Unproven; activation memory still global** — treat as optional Phase-2 experiment, not Plan A |
| Inference-only FCN3? | **Yes** |
| CorrDiff / StormCast-class regional train? | **Yes (sized down)** |
| StormCast paper-scale? | **No** — lite / Nepal-cropped yes |

---

## 3. Ranked strategy portfolio (THIS hardware)

Order = **expected skill-per-GPU-hour × risk**, for Nepal / HKH orographic weather — not for matching GenCast CRPS globally.

### Rank 1 — **Strategy A: Frozen FCN3 (or SFNO) inference + train CorrDiff / precip diagnostic on Nepal box**

**What:** Keep NVIDIA FCN3 weights frozen. Generate global 0.25° / 6 h ensembles via Earth2Studio. Train a **regional** residual UNet + diffusion (CorrDiff / StormCast *downscaling* mode) or a lighter precip/orographic diagnostic head on a Nepal–HKH box.

**Why #1 on Sparks:** Matches how Earth-2 actually ships regional skill (Taiwan CorrDiff, UAE FCN+CorrDiff, CONUS GEFS–HRRR). Matches prior survey finding: **no public FCN3 regional weight FT**. Fits memory. Dual-Spark: Spark‑1 = inference + data prep; Spark‑2 = diffusion train (or DDP across both).

**Data stack (access varies):**

| Dataset | Role | Access notes |
|---------|------|--------------|
| **ERA5** (crop South Asia) | IC / FCN3 input; low-res conditioner | CDS; assume crop possible |
| **IMDAA** (~12 km, South Asia) | High-res regional reanalysis target / conditioner | [rds.ncmrwf.gov.in](https://rds.ncmrwf.gov.in/) — register + request |
| **IMERG** | Precip / extreme-rain target or diagnostic | GPM / NASA Earthdata |
| **HIWAT** | CAM ensemble archive over Nepal–Bangladesh–NE India (2017–2022) | [NASA Earthdata HIWAT](https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-hiwat-1), DOI [10.5067/MODEL/HIWAT/DATA101](https://doi.org/10.5067/MODEL/HIWAT/DATA101); large (~hundreds of TB archive — **subset aggressively**) |

**Concrete recipe sketch:**

1. Define box e.g. **78–90°E, 26–31°N** (Nepal + near-field) or wider HKH for monsoon dynamics.
2. Align ERA5/FCN3 fields → IMDAA or HIWAT native grid (bilinear + orography invariants).
3. Train **regression UNet** (deterministic residual) then **diffusion** on residuals (PhysicsNeMo StormCast/CorrDiff custom dataset).
4. Prefer **patch diffusion** if box is large; set patch ≥ residual autocorrelation length.
5. Score vs IMERG / station / IMDAA holdouts: RMSE, CSI for heavy precip, CRPS if ensembled.

**Risk:** HIWAT volume; IMDAA registration latency; precip channel gaps in FCN3 (see §6).

---

### Rank 2 — **Strategy D: Distill / bias-correct / post-process only**

**What:** Frozen FCN3 → Nepal crop → learn **per-variable, elevation-aware bias** (linear / small MLP / UNet) vs IMDAA or IMERG. Optional: quantile mapping for precip-related fields (tcwv, humidity levels).

**Why #2:** Days–weeks, not months. Establishes **whether** regional error is correctable before expensive diffusion. Provides a strong baseline that A must beat.

**When to skip ahead:** If Week 1–2 shows large structured orographic bias that a linear corrector cannot fix — still keep D as published baseline.

---

### Rank 3 — **Strategy C: Small FCN-arch limited-area model (Hamer-style) on regional reanalysis**

**What:** Port AFNO / small SFNO-like stack to a **limited domain** (IMDAA 12 km or ERA5 crop nested). Train from scratch or warm-start embeddings; optional **nesting** from frozen global FCN3 at boundaries (Hamer et al. Europe/CERRA on 4×V100).

**Why #3:** Proven on hardware weaker than Spark; gives a true regional dynamical emulator; publication-friendly “FCN-family regional” story without claiming FCN3 weight FT.

**Caveats:** Not using the **711M FCN3** weights; nesting quality mixed in Hamer (geopotential-only nesting often best). Orography must be first-class (static channels + maybe elevation-conditioned loss).

References: [UMBC PDF](https://ebiquity.umbc.edu/get/a/publication/1453.pdf); [ESSOAr](https://doi.org/10.22541/essoar.171285742.22398551/v1); FourCastNeXt limited-compute tricks: [arXiv:2401.05584](https://arxiv.org/abs/2401.05584).

---

### Rank 4 — **Strategy B: LoRA / PEFT / partial FT of FCN3 or SFNO**

**What:** Adapter layers / last-block FT / frozen encoder + regional lat–lon **masked loss** with Himalaya upweight; 1-step only; ens=1; heavy checkpointing; microbatch 1; optional 2-Spark domain parallel via Makani `h_parallel_size`/`w_parallel_size`.

**Why last:** Paper FCN3 **does not fit** one 80 GB GPU without domain parallel; LoRA saves **optimizer/adapter** memory, not **spherical activation** memory. No public FCN3 LoRA. High catastrophic-forgetting risk outside Nepal mask. Treat as **ablation** after A+D baselines exist — or escalate to Raj for real domain-parallel FT.

**If you still try on Spark:** SFNO first (simpler), 1-step CRPS/MSE, ens=1, 2-way spatial parallel across Sparks, stop if OOM or global skill collapses.

---

## 4. Concrete efficiency tactics

### Precision

| Mode | Use on Spark |
|------|----------------|
| **BF16 AMP** | Default for FCN3 inference (Earth2Studio) and UNet CorrDiff/StormCast train |
| **FP8 / NVFP4** | Inference-oriented marketing (1 PFLOP sparse FP4). **Do not** bank on FP4 for FCN3/CorrDiff training quality until you verify PhysicsNeMo + SM121 kernels |
| **FP32** | StormCast docs: prefer FP32 for **DiT**; BF16 OK for **UNets** |

### Batch / memory

- **batch_size_per_gpu = 1** (or 2), **gradient accumulation** to effective 16–64 (CorrDiff / StormCast Hydra knobs).
- **Gradient checkpointing** / `use_checkpointing` where available.
- **Frozen encoder / frozen FCN3**: never backprop through global prognostic for Strategy A.
- **Channel subset**: start with surface + mid-levels relevant to monsoon/orography (`t2m`, `u10/v10`, `msl`, `tcwv`, `z/t/q` at 850/500); expand later.
- **6 h single-step first**; multi-step AR only after 1-step loss plateaus.
- **Reduced ensemble**: train ens=1–2; infer larger ens from FCN3 HMM noise without FT.
- **Regional masked loss** only helps if the forward pass is already regional (Strategy C) or you accept full global forward cost (Strategy B).
- **Patch size** (diffusion): ≥ residual autocorrelation length; smaller patches save memory.

### Dual-Spark usage

1. **Cable:** 1× approved QSFP between matching CX-7 ports ([User Guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html); [Connect Two Sparks playbook](https://build.nvidia.com/playbooks/connect-two-sparks/stacked-sparks)).
2. **Software:** NVIDIA Sync Cluster Assistant → SSH + CX-7 IPs + NCCL; verify with `ibdev2netdev`, `nccl-tests`.
3. **Workload split (recommended):**
   - **Heterogeneous:** Spark‑1 continuous FCN3 inference / ERA5 staging; Spark‑2 diffusion training.
   - **Homogeneous DDP:** `torchrun` 2 processes for CorrDiff/StormCast data parallel (easy win).
   - **Domain parallel:** only if **one sample** OOMs at batch=1 (`training.domain_parallel_size=2` in StormCast) — expect RoCE latency hit vs NVLink.

### Software stack pins

- Earth2Studio + FCN3 package (NGC/HF)
- Makani (custom FCN3 train — cluster only)
- PhysicsNeMo **StormCast** example (prefer over deprecated CorrDiff folder for new projects): [StormCast README](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/stormcast/README.html)
- Verify **CUDA / PyTorch / SM121** wheels on DGX OS before multi-day runs

### Data efficiency

- Crop ERA5 / IMDAA to box + halo early; don’t stream global zarr every step.
- Cache normalized tensors on NVMe (4 TB helps).
- HIWAT: download **probability products / selected members / monsoon months only**, not the full 349 TB archive.

---

## 5. Phased plan

### Week 1–2 — Verification baseline (both Sparks)

| Day focus | Deliverable |
|-----------|-------------|
| HW bring-up | `nvidia-smi`, unified memory visibility, BF16 matmul smoke; **QSFP link + NCCL test** between Sparks |
| FCN3 inference | Earth2Studio 4-member, 16-step (4-day) over Nepal box; wall-clock vs H100 expectations; peak mem |
| Data pipes | ERA5 crop; start IMDAA registration; IMERG subset; HIWAT catalog sample |
| Baseline scores | Cropped FCN3 vs ERA5/IMDAA: RMSE/ACC for z500, t2m, winds; **precip proxies** (tcwv) vs IMERG |
| Tiny corrector | Strategy D: elevation-binned bias on t2m / winds — 1–2 day train |

**Go / no-go:** If FCN3 inference OOMs or torch-harmonics CUDA fails on SM121 → fix env before any train. If inference works but is &gt;5–10× slower than H100, still OK for research; plan ensemble generation offline overnight.

### Weeks 3–8 — Primary path (Strategy A + D)

1. Lock **Nepal/HKH box**, variables, train/val years (e.g. train 2018–2021, val 2022, monsoon-focused metrics).
2. Build PhysicsNeMo **custom dataset** (ERA5 or FCN3 forecast as `background`; IMDAA/HIWAT/IMERG as `state`).
3. Train **regression UNet** on Spark‑2 (BF16, batch=1, accum).
4. Train **diffusion** on residuals; start `model_size: mini` then `normal`.
5. Keep D bias-corrector as baseline; A must beat D on CSI/CRPS for heavy rain and mountain t2m.
6. Optional parallel track: **Strategy C** small AFNO on IMDAA if nesting paper story is needed.

### Weeks 9–12 — Harden & decide escalate

- Multi-step regional rollout; calendar monsoon case studies (e.g. extreme precip events).
- Ablate FCN3-conditioned vs ERA5-conditioned downscaler.
- **Only then** consider Strategy B LoRA on Spark or write Raj proposal for FCN3 domain-parallel FT.

### Success metrics (actionable)

| Metric | Target sense |
|--------|----------------|
| t2m / 10 m wind RMSE over Nepal vs IMDAA | Beat cropped FCN3 and beat Strategy D |
| Heavy precip CSI / ETS (IMERG or gauge) | Clear lift in orographic bands |
| CRPS (if diffusion ens) | Competitive vs regression-only |
| Spread–skill | Not wildly miscalibrated |
| Latency | Regional 24–48 h product in minutes on 1 Spark |
| Stability | No NaNs; spectra not washed out at 24–48 h |

### When to escalate to Raj / bigger GPU cluster

Escalate when **any** of these are true:

1. You need **FCN3 weight-level** FT with AR≥4 and ens≥8 (paper regime).
2. Domain parallel needs **≫2** spatial shards (activations still OOM on 2×128 GB).
3. StormCast-quality training needs **multi-week** wall time even with lite config — H100/B200 node-hours dominate.
4. You must train on **HIWAT-scale** domains at full CAM channel counts with long diffusion schedules.
5. Production SLA requires H100-class throughput for large ensembles.

**Keep Sparks** for: inference, data prep, CorrDiff/StormCast lite, demos, LoRA ablations.

---

## 6. Risks (be explicit)

| Risk | Severity | Mitigation |
|------|----------|------------|
| **OOM** on FCN3 train / large diffusion | High for B; med for A | Batch=1, checkpointing, patches, freeze FCN3, domain_parallel=2 max; accept “no” for full FCN3 FT |
| **Catastrophic forgetting** (Strategy B) | High | Regional mask alone insufficient; prefer A/C; if B, early-stop on global probes |
| **0.25° ceiling** | Structural | FCN3 cannot resolve steep Himalayan valleys; **must** downscale (IMDAA 12 km / HIWAT ~km) for mountain precip skill |
| **Precip absence / weakness** | Structural | FCN3 prognostics emphasize dynamics/moisture state (**tcwv**, q) not dedicated surface precip like IMERG; train **precip diagnostic or downscaler** explicitly |
| **Bandwidth starvation** | Med–High | 273 GB/s → slow epochs; cache on NVMe; don’t expect Eos throughput |
| **SM121 software gaps** | Med | Validate PhysicsNeMo / torch-harmonics / NCCL on DGX OS in Week 1 |
| **Data access latency** | Med | Start IMDAA RDS registration Day 1; HIWAT subset only; ERA5 crop first |
| **CorrDiff deprecation** | Low–Med | New work on **StormCast** recipe (downscaling conditions = CorrDiff setting) |
| **Overclaiming “FCN3 fine-tune”** | Reputational | Publish as **FCN3-conditioned regional downscaler** or **limited-area FCN-arch**, not “FCN3 FT on Spark” unless weights actually update |

---

## Recommended primary path (decision)

```
Week 1–2:  FCN3 inference + D bias baseline + data access
    │
    ▼
PRIMARY:   A — Frozen FCN3 → StormCast/CorrDiff-style Nepal downscaler
    │         (+ keep D as paper baseline)
    │
OPTIONAL:  C — Small limited-area AFNO on IMDAA (parallel if student capacity)
    │
DEFER:     B — LoRA/partial FCN3 FT (only after A beats D, or on Raj)
    │
NEVER ON 2 SPARKS: paper-scale FCN3 multi-step ensemble FT
```

---

## Key URLs (real)

| Resource | URL |
|----------|-----|
| DGX Spark product | https://www.nvidia.com/en-us/products/workstations/dgx-spark/ |
| DGX Spark ConnectX-7 clustering | https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html |
| Connect Two Sparks playbook | https://build.nvidia.com/playbooks/connect-two-sparks/stacked-sparks |
| NVIDIA Sync Cluster Assistant | https://docs.nvidia.com/sync/0.97.6/cluster-assistant.html |
| FCN3 paper | https://arxiv.org/abs/2507.12144 |
| FCN3 NVIDIA blog | https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/ |
| FCN3 HF | https://huggingface.co/nvidia/fourcastnet3 |
| FCN3 NGC | https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3/- |
| Makani | https://github.com/NVIDIA/makani |
| PhysicsNeMo StormCast | https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/stormcast/README.html |
| PhysicsNeMo CorrDiff (legacy) | https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/corrdiff/README.html |
| UAE FCN+CorrDiff blog | https://developer.nvidia.com/blog/nvidia-earth-2-powers-regional-ai-weather-forecasting-in-the-united-arab-emirates/ |
| CorrDiff paper | https://arxiv.org/abs/2309.15214 |
| Hamer regional FCN | https://ebiquity.umbc.edu/get/a/publication/1453.pdf |
| FourCastNeXt | https://arxiv.org/pdf/2401.05584 |
| IMDAA RDS | https://rds.ncmrwf.gov.in/ |
| HIWAT Earthdata | https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-hiwat-1 |
| HIWAT Data journal | https://doi.org/10.3390/data10070112 |

---

*End of plan — Sep 2026. Prefer evidence over Spark marketing: 128 GB unified ≠ 1024×H100 FCN3 training.*
