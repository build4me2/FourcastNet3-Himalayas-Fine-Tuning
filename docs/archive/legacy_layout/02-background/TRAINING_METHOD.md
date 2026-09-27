# FourCastNet 3 (FCN3): Lead-Level Technical Briefing — Creation & Training

**Audience:** Manisha Chand (research lead)  
**Primary source:** Bonev, Kurth et al., *FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale*, [arXiv:2507.12144](https://arxiv.org/abs/2507.12144) ([HTML](https://ar5iv.labs.arxiv.org/html/2507.12144))  
**Supporting:** [HF model card](https://huggingface.co/nvidia/fourcastnet3), [NGC / Earth-2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/fourcastnet3), [NVIDIA blog](https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/), [Makani](https://github.com/NVIDIA/makani), [torch-harmonics](https://github.com/NVIDIA/torch-harmonics)  
**Light contrast:** FCN1 [arXiv:2202.11214](https://arxiv.org/abs/2202.11214); SFNO [arXiv:2306.03838](https://arxiv.org/abs/2306.03838)

---

## 1. Problem they’re solving

**Deterministic MLWX vs probabilistic ensembles.**  
A deterministic map \(u_{n+1}=F_\theta(u_n,t_n)\) trained with MSE/RMSE learns a conditional *mean*. For a chaotic atmosphere that is under-determined at 6 h from a single reanalysis state, the minimizer of squared error is a blurred “ensemble average”: skillful on RMSE, unusable as a weather scenario (Pathak et al. FCN1; GraphCast-class; WeatherBench-2 discussion). Probabilistic MLWX instead aims to sample from \(p(u_{n+1}\mid u_n,t_n)\). GenCast does this with iterative denoising diffusion—excellent medium-range CRPS, expensive inference and imperfect spectra. FCN3’s bet: **one-step ensemble generation** via a hidden-Markov conditioning noise, trained end-to-end with a proper scoring rule, so 15-day / 60-day large ensembles stay cheap (paper: ~60 s for 15 days on H100; <4 min for 60 days).

**Why geometry matters.**  
Atmosphere lives on \(S^2\). Lat–lon CNNs / AFNO (FCN1) treat the grid as a rectangle: polar anisotropy, discontinuous longitude wrap, and translation-equivariant filters that are *wrong* symmetries for rotating-sphere dynamics. Graph neural nets approximate locality but do not bake in \(SO(3)/SO(2)\) structure. Spherical spectral methods (SFNO) fix global isotropy via SHT but force **radially symmetric** kernels—poor for orographic blocking, tilted isentropic flow, zonal preference. FCN3’s problem statement is therefore dual: (i) sample physically realistic, well-calibrated ensembles; (ii) do so with operators that respect spherical topology, multi-scale locality, and anisotropic morphology—without spectral blow-up on long rollouts.

---

## 2. Core idea in one page

**Pitch.** FCN3 is a **spherical neural operator** (encoder → latent on a Gaussian grid → stack of local/global spherical ConvNeXt-style blocks → decoder) that advances the ERA5 state by \(\Delta t=6\,\mathrm{h}\). Stochasticity is not IC jitter and not a reverse diffusion sampler: at each step the network is conditioned on a latent field \(z_n\) drawn from a **spherical diffusion process** (Palmer-style spectral AR(1) noise at multiple length scales). Autoregressive composition yields a hidden Markov model; different \(z\) trajectories → ensemble members in one forward pass each.

**Anisotropic filters for topography / fronts.**  
- **Global blocks:** spectral convolution via the spherical convolution theorem—isotropic filters \(\widehat{k}_\ell^0\), SHT multiply, like IFS pseudo-spectral ops (SFNO DNA).  
- **Local blocks (majority):** DISCO group convolutions (Ocampo et al.; Liu-Schiaffini et al.)—compactly supported kernels expanded in **Morlet-like wavelets on a disk**, so filters can be elongated / oriented (zonal vs meridional, topographic wrap-around). Ratio found best: **4 local : 1 global**.

**What “probabilistic” means here.**  
Not a parametric density head. Training minimizes **ensemble CRPS** of the predictive marginals vs the single verifying ERA5 field, in **physical space and in spherical-harmonic space**. Proper scoring on marginals alone can be gamed by spatially scrambled ensembles; spectral CRPS (coefficients weighted by multiplicity, all wavelengths) forces spatial correlation structure. Calibration target: spread–skill ≈ 1, flat rank histograms—even when inference ensemble size (50) exceeds training ensemble size (2–16).

**Vs FCN1 / SFNO (clarifying delta).**  
FCN1 = adaptive FNO on lat–lon, deterministic, tendency-style AFNO stack. SFNO = spherical spectral neural operator for stable dynamics, still not the FCN3 probabilistic + local DISCO + spectral-CRPS package. FCN3 = SFNO-grade geometry + local anisotropic integral kernels + HMM noise + dual CRPS + domain-decomposed training at ≥1024 GPUs.

---

## 3. Physics / atmosphere intuition

**72 fields as a dynamical state.**  
Seven surface prognostics (\(u_{10},v_{10},u_{100},v_{100},t_{2m},\mathrm{msl},\mathrm{tcwv}\)) plus five 3D fields (\(z,t,u,v,q\)) on 13 pressure levels ≈ a coarse discretisation of the dry + moisture hydrostatic atmosphere: mass (via \(z\), msl), thermodynamics (\(t\)), horizontal momentum (\(u,v\)), moisture (\(q\), tcwv). This is the Markov state the operator learns to advance—not full GCM physics, but enough to carry synoptic and many mesoscale signatures at 0.25°.

**6 h \(\Delta t\).**  
Large enough to skip acoustic/gravity CFL pain of classical NWP; small enough that the learned map remains locally Lipschitz and rollouts can reach medium-range (15 d = 60 steps) and subseasonal (60 d = 240 steps). Stage-1 still *samples* hourly ERA5 starts so every UTC hour teaches the same 6 h map (data augmentation in time).

**Orography / land–sea mask / cosine zenith.**  
Static geography breaks spherical homogeneity: blocked flow, land–sea contrast, boundary-layer drag proxies. Cosine solar zenith is the only cheap, analytic diurnal forcing—without it, \(t_{2m}\) / PBL-sensitive channels lose the day–night cycle. These auxiliaries are **inputs**, not predicted; recomputed or held fixed each step.

**Why CRPS vs MSE.**  
MSE → mean → blur → wrong spectra. CRPS is a proper score for the predictive CDF: skill *and* spread. Ensemble spread must track error growth (chaos + analysis uncertainty + model error). Over-dispersion early / under-dispersion mid-range then relaxation (paper Fig. 3) is the calibration story they optimize toward.

**Ensemble spread.**  
Spread is generated by the latent spherical diffusion \(z_n\) (8 channels, distinct \(k_T\) length scales, \(\lambda=1\), \(\sigma=1\)), not by perturbing ERA5 ICs. Inference needs no external perturbation scheme (Earth2Studio uses `Zero` IC perturbation). Noise-centering in late training (paired \(\pm z\)) improves both FT and inference.

---

## 4. Architecture logic (conceptual)

```
[ERA5 state 721×1440] + [land, sea, orography, coszen] + [8 noise fields]
        │
        ▼  grouped local DISCO encode (no channel mixing; atm encoder shared across levels)
[latent on 360×720 Gaussian grid, ~677 emb. channels]   ← Table 2; Fig. 9 caption says 641 — paper inconsistency
        │
        ▼  8 local + 2 global ConvNeXt-style spherical NO blocks
           (concat state + aux/noise each block → spherical conv → GeLU → pointwise MLP → scaled residual)
        │
        ▼  bilinear upsample to 721×1440 Gaussian + grouped local decode
        │
        ▼  softclamp spline on water channels (q, tcwv) → next state (direct, not tendency)
```

**Noise channels.** Sampled from spectral AR process  
\(z_n=\phi z_{n-1}+\sum_{\ell,m}\sigma_\ell\eta_\ell Y_\ell^m\), \(\phi=e^{-\lambda}\), \(\sigma_\ell\propto e^{-\frac12 k_T\ell(\ell+1)}\)—multi-scale red noise on the sphere (Palmer et al. 2009).

**What weights learn vs what’s fixed.**

| Learned (~711M params) | Fixed / non-learned |
|---|---|
| DISCO basis coeffs / spectral filter coeffs, MLPs, residual scales | Grid geometry, quadrature weights, SHT basis |
| Encoder/decoder grouped filters | Land/sea/orography fields; coszen formula |
| Output softclamp shape is fixed spline | Norm stats (z-score or min/max per Table 4); channel & \(\Delta t\) loss weights |
| | Noise process hyperparameters (\(k_T\) ladder, \(\lambda,\sigma\)) |

**Deliberate omissions / choices.** No LayerNorm (absolute magnitudes matter; He-style init + careful residual scaling instead). Predict **full next state**, not tendency—avoids Euler-only inductive bias and residual high-frequency pass-through that blows rollouts. Encoder/decoder **do not mix channels**—variables have incompatible spectra.

---

## 5. Training curriculum (Table 3 — the “fine-tuning” they mean)

ERA5 1980–2016 (hourly used in Stage 1; 6-hourly IC subsets later). Curriculum = three regimes that trade **short-lead skill / large-batch CRPS** for **autoregressive stability** and **recent-climate alignment**.

| | **Stage 1 (pretrain)** | **Stage 2 (pretrain)** | **Fine-tune (FT)** |
|---|---|---|---|
| **Why it exists** | Learn accurate 6 h stochastic map on maximal data; large ensemble stabilizes *biased* CRPS | Teach multi-step consistency; switch to *fair* CRPS | Suppress AR error accumulation; adapt to near-term climate; enable noise-centering |
| **Data** | 1-hourly pairs, 1980–2016 (~332.8k samples) | 6-hourly ICs 00/06/12/18Z, 1980–2016 (~55.5k) | 6-hourly, **2012–2016 only** (~5.8k) |
| **Objective** | nodal + spectral **CRPS** (biased / skillspread) | nodal + spectral **fCRPS** | nodal + spectral **fCRPS** |
| **AR steps** | 1 | 4 | **8** |
| **Batch / ens** | 16 / **16** | 32 / **2** | 4 / **4** |
| **Model-parallel** | lat×lon = 2×2 | 2×4 | **4×4** (16-way; AR memory) |
| **GPUs / time** | 1024×H100 / ~78 h | 512×A100 / ~15 h | 256×H100 / ~8 h |
| **Steps** | 208,320 | 5,040 | 4,380 |
| **LR schedule** | constant \(5\times10^{-4}\) | start \(4\times10^{-4}\), halve every 840 steps | start \(4\times10^{-6}\), halv every 1,095 steps |

**Logic chain.** Stage 1: fair CRPS is unstable early (two-member ambiguity → unbounded spread; also noted by AIFS-CRPS). Large \(N_\mathrm{ens}=16\) + biased CRPS compensates. Stage 2: fewer members, fair CRPS, 4-step BPTT-through-time on the sphere. FT: longer rollout, tiny LR, recent years only (distribution drift), noise-centering on. Spatial domain decomposition (Makani) is what makes AR+ensemble+0.25° fit in 80 GB VRAM at all.

---

## 6. Loss & optimization

**Composite objective** (Eq. 87):

\[
\mathcal{L}_\mathrm{ens}=\sum_n\sum_c w_c\,w_{\Delta t,c}\,w_n\Bigl(\mathcal{L}_\mathrm{spatial}+\lambda_\mathrm{spectral}\mathcal{L}_\mathrm{spectral}\Bigr).
\]

- \(\mathcal{L}_\mathrm{spatial}\): sphere-averaged pointwise ensemble CRPS (quadrature on \(S^2\)).  
- \(\mathcal{L}_\mathrm{spectral}\): CRPS on each SH coefficient \(\hat u_\ell^m\) across the ensemble (all \(\ell\) up to grid Nyquist; multiplicity weighting in the design narrative). Fixes the “shuffle ensemble members per gridpoint” degeneracy of marginal CRPS.  
- **\(\lambda_\mathrm{spectral}\):** not printed as a number in the main Table 3 text; released Makani/HF training config uses spectral `relative_weight: 0.1` vs spatial `1.0` — treat **0.1 as the practical value**, flag if reproducing from paper alone.

**Channel weights \(w_c\)** (Table 4): surface winds/msl/tcwv \(0.1\), \(t_{2m}=1.0\); upper-air \(w_c=p\cdot10^{-3}\) (pressure in hPa)—GraphCast-style emphasis on tropospheric levels that dominate synoptic skill.  
**Temp-diff weights \(w_{\Delta t,c}\):** inverse of climatological spatial std of 1-hourly increments (text); Eq. 88 rendering on ar5iv looks like an expectation of the raw difference (missing \(\mathrm{Std}\)) — **follow the prose / GraphCast convention; verify against Makani `temp_diff_normalization: true`**.  
**Lead weights \(w_n\):** average over AR steps when \(N_\mathrm{times}>1\) (exact schedule of \(w_n\) underspecified in prose—uncertainty).

**CRPS variant.** Stage 1: biased skillspread CRPS. Stage 2/FT: fair CRPS (unbiased spread term). HF config: `crps_type: skillspread` for both loss heads.

**Optimizer.** Adam, all stages; AMP **bf16** throughout (paper E.3). HF/Makani extras (not all in Table 3): \(\beta_1=0.9\), \(\beta_2=0.95\), max grad norm 32. LR as in Table 3 above.

---

## 7. What was really significant / novel (honest, sourced)

| Claim | Evidence / caveat |
|---|---|
| **Medium-range CRPS ≈ GenCast, ≫ IFS-ENS, at 6 h and 0.25°** | Paper Fig. 3 / App. F; WB2 protocol on 2020. Hardware/resolution caveats acknowledged. |
| **One-step ensemble member (no diffusion sampler)** | HMM + CRPS vs GenCast iterative denoise → ~8× faster than GenCast, ~60× vs IFS-ENS (their numbers). |
| **Stable, non-blurry, non-noisy spectra to 60 days** | Angular/zonal PSD vs ERA5; contrast GraphCast/NeuralGCM blur and GenCast/AIFS-CRPS HF buildup (Fig. 4–5, F.7). **This is the headline scientific win.** |
| **Local anisotropic spherical CNNs + spectral globals** | Beyond SFNO isotropy; beyond FCN1 lat–lon AFNO. |
| **Spectral CRPS (full band, multiplicity-aware)** | Addresses known failure of pointwise CRPS; improves on NeuralGCM’s low-pass spectral loss. |
| **Domain-decomposed model∥ + ensemble∥ + batch∥ to 1024 GPUs** | Engineering novelty enabling the model size × AR × ensemble product; open in Makani. |
| **Not novel** | CRPS training itself (AIFS-CRPS, NeuralGCM); spherical FNOs (SFNO); ERA5 0.25° 6 h framing (GraphCast-class). Skill parity with GenCast is “rival,” not clear dominate. |

---

## 8. Weights & “bias” — three different things

**(a) Neural network parameters \(\theta\).**  
~710.9M convolutional / MLP weights. Optimized by Adam on \(\mathcal{L}_\mathrm{ens}\). Initialization: He-style variance control because LayerNorm is absent. Checkpoints on NGC/HF; inference prefers bf16 AMP.

**(b) Forecast systematic bias.**  
\(b(x,t_n)=\mathbb{E}_{t_i,e}[u_e-u^*]\) (Eq. 91). Paper F.6: mild cold bias in t850, low equatorial z500 tendency; otherwise fairly uniform. Rank histograms already hinted temperature bias. CRPS *penalizes* mis-centered CDFs but does not include an explicit bias-correction head; FT on 2012–2016 is the main distributional steer. Residual bias remains a post-processing / DA topic.

**(c) ERA5 climate bias inheritance.**  
Training targets are reanalysis, not truth. Model inherits ERA5’s assimilation climate (moisture, orography-related precipitation proxies via tcwv/q, tropical biases, etc.). No mechanism in FCN3 “debias” ERA5 toward observations; FT recent years only tracks **ERA5’s** recent climate. For Nepal work: expect orographic moisture and monsoon biases characteristic of ERA5 at 0.25°, not station climatology.

---

## 9. Practical lead checklist (if rebuilding)

1. **Grid & operator class** — equiangular I/O 721×1440; latent Gaussian 360×720; commit to DISCO local + spectral global (not pure transformer/GNN) if spectra/stability are requirements.  
2. **Channels & auxiliaries** — 72 prognostics; land/sea/orography/coszen; decide precip diagnostic later (not in FCN3). Fix norm stats (z-score vs min/max per Table 4).  
3. **Stochastic mechanism** — multi-scale spherical diffusion noise (8 ch) vs diffusion-model sampler vs IC perturbation; prefer HMM if inference cost matters.  
4. **Loss** — spatial CRPS + spectral CRPS (\(\lambda\sim0.1\)); \(w_c\), \(w_{\Delta t,c}\); biased→fair CRPS schedule; never MSE-only if you care about sharpness.  
5. **Curriculum** — (i) single-step large-ens biased CRPS on hourly; (ii) 4-step fair CRPS; (iii) 8-step FT on recent years + noise-centering. Match model-parallelism to AR depth.  
6. **Predict state, not tendency**; no LayerNorm; water softclamp; 4:1 local:global.  
7. **Eval beyond RMSE** — CRPS, SSR, rank histograms, **angular PSD at long lead**, case studies (e.g. storm landfall). Score regional (Nepal) separately—global mean CRPS can hide orographic failure.  
8. **Systems** — need Makani-class domain∥ for full 0.25° rebuild; or accept smaller latent / fewer AR steps on fewer GPUs.

---

## 10. Open limitations that matter for later Nepal work

- **No precipitation output.** Paper explicitly defers precip to a future diagnostic. Monsoon / landslide / hydro use-cases need a precip head or coupled diagnostic—tcwv/q are proxies only.  
- **0.25° (~25–30 km).** Central Himalaya orography is badly under-resolved; anisotropic filters help morphology *in principle* but cannot invent valley-scale flow. Expect smoothed windward/leeward contrasts.  
- **Global loss.** \(w_c\) and sphere-averaged CRPS optimize Europe–Atlantic–Pacific skill that dominates WB2; Nepal/Himalaya is a tiny measure contribution—regional fine-tuning or reweighted loss is almost certainly required.  
- **ERA5 inheritance** — moisture and mountain climate biases pass through (Section 8c).  
- **No DA uncertainty** — ensembles reflect learned stochastic dynamics + noise process, not analysis-error covariances (future work in conclusions).  
- **Calibration quirks** — slightly over-dispersive ≤24 h, then under-dispersive before flattening; local reliability in complex terrain unproven.  
- **Spectral CRPS / Eq. 88 / embedding dim** — reproduce with Makani configs; do not trust ar5iv math alone for every index.

---

## Key citations (real URLs only)

- FCN3 paper: https://arxiv.org/abs/2507.12144 · https://ar5iv.labs.arxiv.org/html/2507.12144  
- FCN1: https://arxiv.org/abs/2202.11214  
- SFNO: https://arxiv.org/abs/2306.03838  
- GraphCast (channel \(\Delta t\) weighting precedent): https://arxiv.org/abs/2212.12794  
- GenCast: https://www.nature.com/articles/s41586-024-08252-9 (also arXiv:2312.15796)  
- AIFS-CRPS: https://arxiv.org/abs/2412.15832  
- NeuralGCM: https://arxiv.org/abs/2311.07222  
- DISCO convolutions: https://arxiv.org/abs/2209.13603 · localized NO kernels https://arxiv.org/abs/2402.16845  
- Code: https://github.com/NVIDIA/makani · https://github.com/NVIDIA/torch-harmonics  
- Weights: https://huggingface.co/nvidia/fourcastnet3 · NGC Earth-2 fourcastnet3  
- Blog: https://developer.nvidia.com/blog/fourcastnet-3-enables-fast-and-accurate-large-ensemble-weather-forecasting-with-scalable-geometric-ml/

*Uncertainties flagged inline: \(\lambda_\mathrm{spectral}\) numeric (use HF 0.1), Eq. 88 std vs \(\mathbb{E}\) rendering, latent channel count 641 vs 677, exact \(w_n\) schedule.*
