# Research Method Brief: Fine-Tuning Technique for FourCastNet3 → Nepal / South Asian Mountain Skill

**For:** Manisha Chand (research lead)  
**Date:** 10 Sep 2026 (PT)  
**Hardware target:** 2× NVIDIA DGX Spark (GB10, 128 GB unified each; ConnectX-7 RoCE ~200 Gb/s between nodes)  
**Base model:** FourCastNet 3 (FCN3; Bonev, Kurth et al., [arXiv:2507.12144](https://arxiv.org/abs/2507.12144))  
**Companion plans:** `SPARK_FINETUNE_PLAN.md`, `TRAINING_METHOD.md`, `REGIONAL_FINETUNE_GAP.md`, `FCN_REGIONAL_FINETUNES.md`  
**Constraint (non-negotiable):** Full FCN3 multi-step + large-ensemble fine-tune (NVIDIA Stage 1/2/FT scale) is **not** feasible on 2 Sparks. This brief designs a **technique** that preserves FCN3’s critical probabilistic/geometric properties while staying honest about freeze vs update sets.

---

## 1. Restated goals + non-goals

### Goals

1. **Regional probabilistic skill** over Nepal / High Himalaya / adjacent South Asian orography (elevated T2M, orographic moisture / precip proxies, near-surface winds, mid-level thermodynamics), scored against **observations / regional reanalysis**, not only ERA5 self-consistency.
2. **Preserve FCN3 quality invariants** listed in §2 (HMM one-step ensembles, dual CRPS discipline, anisotropic spherical operator, 72-channel hydrostatic state + auxiliaries).
3. **Spark-honest training:** specify exactly what updates on-Spark vs what stays frozen; escalate to cluster only when weight-level AR+ensemble FT is required.
4. **Publishable method delta** vs copying NVIDIA’s global Stage1→Stage2→FT curriculum: regional objective, PEFT/diagnostic adapters, spectral anchoring against forgetting, precip/orography pathway that FCN3 does not natively provide.

### Non-goals

- Matching GenCast / IFS-ENS **global** CRPS.
- Claiming “FCN3 fine-tuned on 2 Sparks” if only a diagnostic/downscaler trains.
- Replacing FCN3 with a pure MSE UNet as the **only** model.
- Resolving ridge–valley precip at 0.25° by weight FT alone (structural ceiling; needs downscaler / regional reanalysis target).
- Reproducing NVIDIA’s 1024×H100 / 16-way domain-parallel AR curriculum on desktop hardware.
- Inventing precip as a fake FCN3 prognostic without an explicit diagnostic head / downscaler.

---

## 2. What we refuse to break (quality invariants / regression gates)

These are **hard gates**. Any candidate method that violates them fails, even if Nepal RMSE improves.

| ID | Invariant | Why it matters | Regression probe |
|----|-----------|----------------|------------------|
| **I1** | Probabilistic **6 h map**; ensembles stay **sharp for weeks** | MSE-only → blur / wrong spectra (FCN3 problem statement; Pathak FCN1) | Angular/zonal PSD vs ERA5 at +15 d / +30 d on global + extratropical storm cases |
| **I2** | **One-step-per-member** via HMM spherical multi-scale noise (not iterative diffusion for the global step) | Inference cost + FCN3 identity ([NVIDIA blog](https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/)) | Wall-clock for 50-member 15-day; member generation = 1 forward / step |
| **I3** | **72-channel** hydrostatic state + orography / landmask / cos-zenith auxiliaries | Markov state + geography; coszen carries diurnal cycle | Channel presence checklist; t2m diurnal amplitude vs ERA5 over Nepal |
| **I4** | **Anisotropic local + global spectral** spherical architecture (DISCO local : spectral global ≈ 4:1) | Orographic blocking / fronts need oriented local filters; SFNO-only isotropy insufficient | Do **not** swap backbone for MSE UNet as sole dynamics model |
| **I5** | **Spatial CRPS + spectral CRPS** (or principled equivalent) | Pointwise CRPS alone admits spatially scrambled ensembles (Bonev et al.; Kochkov NeuralGCM spectral CRPS; Bris stretched-grid spectral fCRPS) | Spectral CRPS / PSD match; SSR≈1; rank histograms not U-shaped from spatial scramble |
| **I6** | Explicit separation of **(a) NN weights**, **(b) forecast systematic bias**, **(c) ERA5 inheritance** | Avoid confusing FT gains with reanalysis echo or post-hoc bias fix (FCN3 §F.6; MAUSAM obs gap) | Report all three: Δθ (or adapter), bias fields \(b(x)\), and skill vs ERA5 **and** vs IMDAA/IMERG/stations |

**Gate rule:** Nepal skill claims require I1–I6 green. Regional CRPS win with spectral collapse or MSE-blurred members = **fail**.

---

## 3. Proposed method family

### Name

**RRCA-FD — Regionally Reweighted CRPS Adapter + Frozen Diagnostic**  
*(hybrid; two tiers share one evaluation contract)*

| Tier | Short name | Role on 2 Sparks | What moves |
|------|------------|------------------|------------|
| **Tier-0 (always)** | Frozen FCN3 inference + elevation-aware bias probe | Days | No FCN3 weights |
| **Tier-A (primary)** | **Frozen FCN3 + CRPS-preserving regional diagnostic / downscaler** | Weeks–months; **Plan A** | Regional UNet±diffusion (CorrDiff/StormCast-class); FCN3 **frozen** |
| **Tier-B (secondary / research)** | **Regionally Reweighted CRPS-LoRA + Spectral Anchor** | Marginal; ablation after A; or escalate | Low-rank adapters on selected FCN3 blocks; backbone mostly frozen; **1-step**, tiny ens |

**One-sentence pitch.** Keep the pretrained spherical probabilistic operator intact as the global prior; learn **regional residual skill** with proper scores and spectral anchors; only lightly adapt FCN3 parameters if activation memory and forgetting probes allow — never by replaying NVIDIA’s full Stage1/2/FT stack on Sparks.

### Why hybrid (not LoRA-only, not diagnostic-only)

- **Public gap:** no published FCN3 Himalaya / South Asia weight FT (`FCN_REGIONAL_FINETUNES.md`; Earth2Studio #879: no released FT variants; prefer diagnostic bias-correctors).
- **Hardware:** FCN3 AR+ens FT needed **16-way** domain parallel on 80 GB H100s (paper); 2 Sparks ≈ 2-way over RoCE — activation-bound, not weight-bound (`SPARK_FINETUNE_PLAN.md`).
- **Physics ceiling:** 0.25° cannot resolve Himalayan valleys; mountain precip skill needs IMDAA (~12 km) / HIWAT / IMERG targets via **downscaler** (CorrDiff [arXiv:2309.15214](https://arxiv.org/abs/2309.15214); StormCast).
- **PEFT precedent exists off-FCN:** MENA ClimaX LoRA ([Munir et al., arXiv:2409.07585](https://arxiv.org/abs/2409.07585)); FengWu-GHR per-lead LoRA ([arXiv:2402.00059](https://arxiv.org/abs/2402.00059)); WeatherPEFT task-adaptive PEFT ([Cao et al., arXiv:2509.22020](https://arxiv.org/abs/2509.22020), ICLR 2026). These **motivate** Tier-B but do **not** prove FCN3 LoRA fits Spark activations.
- **Regional CRPS + spectral precedent:** Bris stretched-grid CRPS with \(\lambda_r,\lambda_f\) ([arXiv:2511.23043](https://arxiv.org/abs/2511.23043)); AIFS-CRPS afCRPS ([arXiv:2412.15832](https://arxiv.org/abs/2412.15832)); NeuralGCM grid+spectral CRPS (Kochkov et al., Nature 2024); FCN3 dual CRPS (Bonev et al.).

### What we refuse as “the method”

| Anti-pattern | Why refused |
|--------------|-------------|
| Copy Stage1 (ens=16, 1-step, 1024 GPU) → Stage2 (AR=4) → FT (AR=8, 16-way parallel) on Sparks | Infeasible; not a research delta |
| MSE-only fine-tune of FCN3 or MSE UNet replacement | Breaks I1, I5 |
| Regional lat–lon crop FT that drops spherical globals / HMM noise | Breaks I2, I4 |
| Claim success from ERA5-only regional RMSE | Ignores MAUSAM obs gap; conflates I6(c) |

---

## 4. Algorithm sketch

### 4.1 Freeze / update sets

#### Tier-A (primary — Spark-feasible)

| Component | Status | Notes |
|-----------|--------|-------|
| FCN3 encoder / local DISCO / global spectral / decoder (~711M) | **FROZEN** | Inference only (Earth2Studio); bf16 |
| HMM spherical multi-scale noise sampler | **FROZEN hyperparameters** | Still used at inference for ensemble members |
| Auxiliaries (orography, LSM, coszen) | **Fixed fields / formula** | May add **higher-res DEM stats** as extra conditioner to diagnostic only |
| Regional regression UNet (mean residual) | **TRAIN** | CorrDiff step-1 / StormCast regression |
| Regional diffusion corrector (residual) | **TRAIN** | CorrDiff step-2; patch diffusion if memory tight |
| Optional precip / orographic diagnostic head | **TRAIN** | Maps FCN3 state (+tcwv, q levels) → IMERG/IMDAA precip or HIWAT fields |
| Linear / elevation-binned bias (Tier-0) | **TRAIN** | Baseline that A must beat |

#### Tier-B (secondary — marginal)

| Component | Status | Notes |
|-----------|--------|-------|
| FCN3 majority weights | **FROZEN** | |
| LoRA / DoRA on **local DISCO MLPs + residual scales** (prefer local anisotropic path for orography) | **TRAIN** | Inspired by Hu et al. LoRA; weather PEFT evidence that vanilla LoRA can underperform WeatherPEFT — treat LoRA as baseline adapter, SFAS/Fisher selection as ablation ([WeatherPEFT](https://arxiv.org/abs/2509.22020)) |
| Global spectral filter coeffs | **FROZEN or very low LR / EWC-penalized** | Protect I5 spectra; global modes ≈ climate prior |
| Encoder/decoder channel-grouped filters | **FROZEN initially** | Variables have incompatible spectra (FCN3 design) |
| Noise process \(k_T,\lambda,\sigma\) | **FROZEN** | Do not re-fit noise to regional MSE |
| Optional Fisher-masked selective FT (SFAS-style) | **Ablation** | Update only high-Fisher params for regional CRPS |

**Honest memory note:** LoRA shrinks **optimizer/adapter** state; it does **not** shrink **global 721×1440×72 activations**. Tier-B remains activation-limited on Spark even at ens=1, AR=1.

### 4.2 Loss terms

Denote:
- \(\mathcal{R}\) = Nepal/HKH region (e.g. 78–90°E, 26–31°N, or wider monsoon box with foothills)
- \(M\) = train ensemble size (tiny on Spark)
- Base frozen FCN3 predictive ensemble \(\{u^{(m)}\}\); adapted \(\{\tilde u^{(m)}\}\) if Tier-B

**Composite objective (Tier-B sketch):**

\[
\begin{aligned}
\mathcal{L}
&= \underbrace{\lambda_{\mathrm{reg}}\,\mathrm{CRPS}_{\mathrm{spatial}}\bigl(\{\tilde u^{(m)}\},\,y;\,w_{\mathcal{R}}\bigr)}_{\text{regionally reweighted spatial CRPS}}
+ \underbrace{\lambda_{\mathrm{spec}}\,\mathrm{CRPS}_{\mathrm{spectral}}\bigl(\{\tilde u^{(m)}\},\,y;\,\mathcal{S}\bigr)}_{\text{spectral CRPS anchor}} \\
&\quad + \underbrace{\lambda_{\mathrm{glob}}\,\mathrm{CRPS}_{\mathrm{spatial}}\bigl(\{\tilde u^{(m)}\},\,y;\,w_{\mathrm{sub}}\bigr)}_{\text{global / subsampled sphere mix}}
+ \underbrace{\lambda_{\mathrm{KD}}\,D_{\mathrm{KL}}\bigl(p_{\tilde\theta}\,\|\,p_{\theta_0}\bigr)}_{\text{optional KL / moment match to base ens}} \\
&\quad + \underbrace{\lambda_{\mathrm{EWC}}\sum_i F_i(\theta_i-\theta_{0,i})^2}_{\text{optional EWC on Fisher-critical weights}}.
\end{aligned}
\]

| Term | Definition / precedent | Spark default |
|------|------------------------|---------------|
| **Regional spatial CRPS** | Area-weighted CRPS with \(w_{\mathcal{R}}(x)\uparrow\) over high elevation / steep slope / monsoon precip max; cf. Bris \(\lambda_r\) regional CRPS ([arXiv:2511.23043](https://arxiv.org/abs/2511.23043)); CRPS-LAM regional framing ([arXiv:2510.09484](https://arxiv.org/abs/2510.09484)) | \(\lambda_{\mathrm{reg}}=1\); elevation-aware \(w_{\mathcal{R}}\) |
| **Spectral CRPS anchor** | CRPS on SH coeffs (multiplicity-aware as in FCN3; or NeuralGCM spectral CRPS up to \(\ell_{\max}\); Bris \(\lambda_f\approx0.1\)) on **full sphere or latitude-band subsample** | \(\lambda_{\mathrm{spec}}=0.1\) starting point; **do not** raise until spatial coherence checked (Bris: \(\lambda_f=0.5\) hurt) |
| **Global mix batches** | Fraction \(\pi_{\mathrm{glob}}\) of steps use near-uniform sphere weights / non-Asia ICs | \(\pi_{\mathrm{glob}}\in[0.2,0.5]\) — anti-forgetting without full Stage replay |
| **KL / ensemble KD** | Match base FCN3 member marginals or pairwise spread outside \(\mathcal{R}\); LwF-style distillation analogue | Optional; start \(\lambda_{\mathrm{KD}}=0\), enable if global SSR/CRPS regresses |
| **EWC** | Kirkpatrick et al. 2017 Fisher penalty toward \(\theta_0\) | Optional on Tier-B; Fisher estimated on small global CRPS probe set |

**Tier-A losses (diagnostic / downscaler):**

- Regression: MSE or CRPS on residual mean toward IMDAA / HIWAT / IMERG (variable-dependent).
- Diffusion: EDM / CorrDiff residual likelihood (Mardani et al., [arXiv:2309.15214](https://arxiv.org/abs/2309.15214)).
- Optional **spectral loss on regional patch FFT** (Nordhagen/Bris-style) with small \(\lambda_f\); CRPS-LAM notes spectral add-ons can artifact — ablate.
- **Never** backprop Tier-A losses into FCN3.

**Uncertainty flags:** Exact FCN3 \(\lambda_{\mathrm{spectral}}\) not boldly printed in paper Table 3; Makani/HF configs use relative spectral weight **0.1** — treat as practical prior, not gospel. Fair vs almost-fair CRPS (AIFS-CRPS \(\alpha\approx0.95\)): prefer **afCRPS / skillspread** at \(M=2\) to avoid fair-CRPS degeneracy in low precision.

### 4.3 Data mix

| Stream | Role | Cadence / notes |
|--------|------|-----------------|
| **ERA5** global (or South Asia crop + halo) | FCN3 IC; Tier-B labels; spectral anchor | 6 h; FCN3-matched 72 channels |
| **Regional weight mask** | From DEM slope/elevation + climatological precip | Static + optional monsoon-season boost |
| **IMDAA** (~12 km) | Tier-A high-res target / conditioner | Register [rds.ncmrwf.gov.in](https://rds.ncmrwf.gov.in/); remap channels |
| **IMERG** | Precip diagnostic / verification | Mountain gauge issues → dual verify with stations where possible |
| **HIWAT** subset | Optional km CAM target (monsoon months, few members) | Do **not** pull full archive |
| **Global probe set** | Forgetting / I5 gates (extratropics + tropics non-Asia) | Fixed 2020-style ICs |

**Mix schedule (Tier-B):** each step sample with prob \(\pi_{\mathrm{reg}}\) a Nepal-upweighted batch and with \(\pi_{\mathrm{glob}}=1-\pi_{\mathrm{reg}}\) a global/subsampled batch. Start \(\pi_{\mathrm{reg}}=0.6\).

### 4.4 Noise handling

- **Keep** FCN3 spherical multi-scale HMM noise at inference (I2).
- **Train Tier-B** with \(M\in\{1,2\}\) members; noise-centering (\(\pm z\)) only if memory allows (FCN3 late FT trick).
- **Do not** replace HMM with IC jitter alone.
- **Tier-A** may use its own diffusion stochasticity for **downscaled** fields; global FCN3 members remain HMM-generated.
- Report separately: (i) FCN3 ensemble CRPS on 0.25° Nepal crop; (ii) downscaler ensemble CRPS on IMDAA/IMERG grid.

---

## 5. Spark-feasible training recipe

### Hardware reality (one table)

| Workload | On 2× Spark? | Update set |
|----------|--------------|------------|
| FCN3 inference (bf16, modest ens) | **YES** | none |
| FCN3 paper AR=8, ens≥4, domain∥=16 FT | **NO** | — |
| Tier-A CorrDiff/StormCast-lite Nepal box | **YES** | diagnostic only |
| Tier-B 1-step LoRA, ens=1–2, AR=1 | **MARGINAL** | adapters only |
| Tier-B AR≥4 + ens≥8 | **Escalate to cluster** | — |

### Precision / batch / AR / ens

| Knob | Tier-A (diagnostic) | Tier-B (adapter) |
|------|---------------------|------------------|
| Precision | BF16 UNet; FP32 if DiT | BF16 AMP; watch SHT numerics |
| Batch / GPU | 1 (+ grad accum → 16–64) | 1 |
| AR depth | N/A for single-frame downscale; multi-frame only after 1-step OK | **1 only** on Spark |
| Train ens \(M\) | Diffusion samples at infer; train per CorrDiff recipe | **1–2** |
| Infer ens | FCN3 HMM 20–50; downscaler 4–16 | Same FCN3 HMM |
| Nodes | Spark-1: FCN3 inference + cache; Spark-2: train **or** DDP×2 over RoCE | Prefer **1 node** until NCCL proven; 2-way domain∥ last resort |
| Checkpointing | Yes | Aggressive activation checkpointing |
| Optimizer | AdamW; cosine | Adam, LR ≪ FCN3 Stage1 (e.g. \(10^{-5}\)–\(10^{-6}\) on adapters) |

### Dual-Spark split (recommended)

```
Spark-1: Earth2Studio FCN3 → write Nepal-crop ensemble zarr/NVMe
Spark-2: Tier-0 bias + Tier-A regression → diffusion
Optional: torchrun DDP×2 for Tier-A only (data parallel)
Tier-B: attempt on Spark-2 alone; if OOM at batch=1 → stop, escalate (don’t fake with MSE crop)
```

**Bandwidth honesty:** unified memory **273 GB/s** ≪ H100 HBM; expect slow epochs. Cache cropped tensors on 4 TB NVMe. RoCE helps DDP allreduce for small adapters/UNets; it does **not** recreate NVLink domain-decomp for FCN3 AR.

---

## 6. Evaluation gates before claiming success

Claim “regional FCN3 adaptation success” only if **all** pass:

### Gate G0 — Integrity (I1–I6)

| Check | Pass criterion |
|-------|----------------|
| Global CRPS / SSR (2020-style probe, key vars z500, t850, t2m, winds) | No worse than base FCN3 beyond pre-registered ε (suggest ≤5% relative CRPS degrade) |
| Spectra (+15 d) | No MSE-blur collapse; no HF blow-up vs base |
| HMM one-step identity | Still 1 forward/member/step |
| Architecture | Anisotropic local+global still in loop for global state |

### Gate G1 — Nepal / HKH skill

| Metric | Vs baselines |
|--------|----------------|
| t2m, 10 m wind RMSE/ACC | Beat **cropped frozen FCN3** and beat **Tier-0 bias** |
| Elevation-banded t2m bias | Reduce high-elevation systematic error |
| Moisture: tcwv / q850 | Improve vs IMDAA where available |
| Precip (diagnostic): CSI/ETS heavy rain, CRPS | Beat FCN3-proxies and Tier-0; report IMERG **and** gauge/IMDAA if possible |
| Orographic spatial structure | Visual + spectrum over southern slopes / foothills |

### Gate G2 — Calibration

- Spread–skill ≈ 1 over \(\mathcal{R}\) at +24–120 h (not wildly over/under-dispersive).
- Rank histograms not dominated by bias (separate I6(b) correction if needed — and **label it as bias**, not weight skill).

### Gate G3 — Honesty labeling

| If you trained… | You may claim… |
|-----------------|----------------|
| Tier-A only | “FCN3-conditioned CRPS-preserving regional diagnostic / downscaler” |
| Tier-B adapters + gates pass | “Regionally reweighted CRPS-LoRA on FCN3 with spectral anchor” |
| Neither weight nor proper-score diagnostic | Do **not** claim FCN3 fine-tune |

**Obs gap reminder (MAUSAM, arXiv:2509.01879):** skill vs ERA5 overstates South Asian monsoon skill vs stations — G1 must include non-ERA5 targets.

---

## 7. Phased experiments (Leonard-ready)

Ordered for evidence density; each phase has a kill criterion.

### Phase 0 — Bring-up (Week 1–2)

- QSFP RoCE + NCCL smoke; BF16 matmul; Earth2Studio FCN3 4-member × 16-step Nepal crop.
- Lock box, years (e.g. train 2018–2021, val 2022, monsoon-heavy metrics), channel list.
- **Kill:** FCN3 inference broken on SM121 / OOM → fix env; no training.

### Phase 1 — Tier-0 bias probe (Week 2–3)

- Elevation-binned / small UNet bias on t2m, winds, tcwv vs IMDAA/ERA5 crop.
- Deliverable: bias maps + RMSE table (I6(b) baseline).
- **Kill:** none (always publish as baseline).

### Phase 2 — Tier-A regression (Weeks 3–5)

- Frozen FCN3 (or ERA5) → IMDAA residual UNet; BF16; batch=1+accum.
- Ablate conditioner: ERA5 vs FCN3 forecast (distribution shift).
- **Kill:** cannot beat Tier-0 on t2m/winds → debug data alignment before diffusion.

### Phase 3 — Tier-A diffusion / precip diagnostic (Weeks 5–8)

- CorrDiff/StormCast-lite residual diffusion; patch size ≥ residual autocorrelation.
- Score CRPS/CSI vs IMERG/IMDAA; spectra of downscaled fields.
- **Go:** A beats Tier-0 on mountain t2m **and** heavy precip CSI.
- **Kill:** diffusion unstable / no precip lift → stay at regression + report; still valid paper as frozen-FCN3 diagnostic.

### Phase 4 — Forgetting & invariant suite (parallel Weeks 6–8)

- Run G0 global probes after any adapter attempt; PSD; SSR.
- **Kill for Tier-B:** any I1/I5 fail → discard adapters, keep Tier-A.

### Phase 5 — Tier-B LoRA smoke (Weeks 9–10, optional)

- AR=1, M=1–2, LoRA on local blocks only; \(\lambda_{\mathrm{spec}}=0.1\); \(\pi_{\mathrm{glob}}\geq0.3\).
- Compare: LoRA vs BitFit/bias-only vs WeatherPEFT-style Fisher mask (if implementable).
- **Kill:** OOM at batch=1 **or** G0 fail → stop; write Raj brief for domain-parallel FT.

### Phase 6 — Case studies & write-up (Weeks 10–12)

- 2–3 monsoon / extreme precip / high-elevation cold-bias cases.
- Ablation table (§8); clear Tier-A vs Tier-B labeling.
- Decision: Spark-complete story (A) vs cluster escalation (B+).

### Leonard handoff packet (minimal)

1. This method brief + `SPARK_FINETUNE_PLAN.md` feasibility table.  
2. Phase 0 mem/wall-clock numbers.  
3. Tier-0 vs Tier-A metric table on locked box.  
4. G0 invariant plots (spectra, SSR).  
5. Explicit sentence: *what was frozen*.

---

## 8. Risks + ablations

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Activation OOM on Tier-B | High | Cap AR=1,M≤2; freeze globals; else abandon B |
| Catastrophic forgetting outside Asia | High | Global mix + spectral anchor + EWC/KD; G0 hard gate |
| Spectral artifacts from large \(\lambda_{\mathrm{spec}}\) / regional FFT | Med | Start 0.1; Bris/CRPS-LAM caution |
| Obs–reanalysis gap | High | Dual verify; don’t declare victory on ERA5 |
| Precip not in FCN3 prognostics | Structural | Tier-A diagnostic mandatory for precip claims |
| 0.25° orography ceiling | Structural | Downscale; DEM as conditioner not magic |
| RoCE DDP inefficiency | Med | Prefer heterogeneous split over forced 2-node FCN3 parallel |
| Overclaiming “FCN3 FT” | Reputational | Naming rules in G3 |
| Vanilla LoRA weak on weather FMs | Med | WeatherPEFT evidence; ablate Fisher/selective PEFT |
| Software: torch-harmonics / PhysicsNeMo on SM121 | Med | Phase 0 validation |

### Ablations (pre-register)

1. Tier-0 vs Tier-A regression vs Tier-A+diffusion.  
2. Conditioner: ERA5 vs frozen FCN3 forecast.  
3. Loss: MSE residual vs spatial CRPS vs spatial+spectral (regional patch).  
4. Regional weight: uniform crop vs elevation/slope upweight vs precip-climatology upweight.  
5. Tier-B: no adapter / LoRA local-only / LoRA+spectral anchor / +EWC / +global mix off.  
6. \(\lambda_{\mathrm{spec}}\in\{0,0.1,0.5\}\) (expect 0.5 risky).  
7. \(\pi_{\mathrm{glob}}\in\{0,0.3,0.5\}\).  
8. Ensemble size at train \(M=1\) vs \(2\) (calibration sensitivity).

---

## 9. Why this is different from copying NVIDIA FT

| NVIDIA FCN3 curriculum FT | RRCA-FD (this brief) |
|---------------------------|----------------------|
| Global ERA5 2012–2016 distribution-drift FT | **Geographic** Nepal/HKH skill objective |
| Continues full ~711M training with AR=8, ens=4, 16-way domain∥, 256×H100 | **Frozen** backbone on Spark; train diagnostic and/or **≪1%** adapters |
| Spatial+spectral CRPS on full globe, uniform mission | **Regionally reweighted** CRPS + spectral **anchor** to protect globe |
| No precip prognostic; no Himalaya scorecard | Explicit **precip/orography diagnostic** path (CorrDiff/StormCast-class) |
| Same Stage1→2→FT ladder | **Different ladder:** bias probe → frozen-conditioned CRPS diagnostic → optional PEFT; kill switches |
| Success = global medium-range CRPS / spectra | Success = **G0 invariants held** + **G1 regional obs-aware skill** |
| Engineering: Eos-scale Makani parallel | Engineering: 2×Spark capacity/BW honesty; escalate when AR+ens FT needed |

**Research claim (accurate):** a Spark-feasible, CRPS-and-spectrum-preserving **adaptation protocol** for FCN3 toward South Asian mountain weather — not a miniature replay of NVIDIA’s pretraining fine-tune stage.

---

## Key citations (real papers / docs only)

| Topic | Citation |
|-------|----------|
| FCN3 architecture, HMM noise, dual CRPS | Bonev, Kurth et al., [arXiv:2507.12144](https://arxiv.org/abs/2507.12144) |
| Spectral + grid CRPS (GCM) | Kochkov et al., NeuralGCM, Nature 2024 / [arXiv:2311.07222](https://arxiv.org/abs/2311.07222) |
| afCRPS training | Lang et al., AIFS-CRPS, [arXiv:2412.15832](https://arxiv.org/abs/2412.15832) |
| Regional + spectral CRPS, \(\lambda_r,\lambda_f\) | Bris stretched-grid, [arXiv:2511.23043](https://arxiv.org/abs/2511.23043) |
| Regional CRPS LAM | CRPS-LAM, [arXiv:2510.09484](https://arxiv.org/abs/2510.09484) |
| CorrDiff under frozen coarse model | Mardani et al., [arXiv:2309.15214](https://arxiv.org/abs/2309.15214); PhysicsNeMo CorrDiff/StormCast docs |
| LoRA | Hu et al., [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| Regional weather LoRA (ClimaX/MENA) | Munir et al., [arXiv:2409.07585](https://arxiv.org/abs/2409.07585) |
| Per-lead LoRA (FengWu-GHR) | [arXiv:2402.00059](https://arxiv.org/abs/2402.00059) |
| WeatherPEFT (task-adaptive PEFT) | Cao et al., [arXiv:2509.22020](https://arxiv.org/abs/2509.22020) |
| EWC | Kirkpatrick et al., PNAS 2017 |
| South Asia AIWP vs observations | MAUSAM, [arXiv:2509.01879](https://arxiv.org/abs/2509.01879) |
| Nested regional FCN-arch (Europe) | Hamer et al., UMBC / ESSOAr nested AFNO |
| DGX Spark clustering | [NVIDIA DGX Spark ConnectX-7 docs](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) |

---

## Uncertainties (explicit)

1. **No public FCN3 LoRA recipe** — Tier-B is hypothesis-generating; WeatherPEFT suggests vanilla LoRA may be weak on WFMs.  
2. **Activation fit of Tier-B on Spark unverified** until Phase 5 smoke.  
3. **Optimal \(\lambda_{\mathrm{spec}},\pi_{\mathrm{glob}},w_{\mathcal{R}}\)** unknown for Himalaya; borrow Bris/FCN3 priors then sweep.  
4. **IMDAA access latency** and HIWAT subset size may gate Phase 2–3.  
5. **Precip skill ceiling** without km targets may be modest; still valuable if t2m/orographic moisture improve under G0.  
6. FCN3 paper internal inconsistency on embedding width (641 vs ~677) irrelevant to method but flag if matching configs.

---

*End of method design — Sep 2026. Prefer Tier-A on Sparks; treat Tier-B as gated research; never equate desktop PEFT with NVIDIA Stage FT.*
