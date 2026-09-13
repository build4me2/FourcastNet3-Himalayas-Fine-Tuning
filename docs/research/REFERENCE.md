# FCN3 Regional Fine-Tune — Working Reference (Howard)

**Last updated:** 2026-09-10 (PT)  
**Project root (Spark):** `~/fourcastnet`  
**Research canon (manii):** `~/Desktop/Research/fourcastnet3/` (mirrored under `docs/research/`)  
**Method:** RRCA-FD (Regionally Reweighted CRPS Adapter + Frozen Diagnostic)  
**Hardware:** 2× DGX Spark (128 GB UMA each)

## Doc precedence (Sheldon)

1. `docs/research/FINETUNE_PATHWAY.md` — lifecycle + status (start here)
2. `docs/research/FINETUNE_METHOD_DESIGN.md` — RRCA-FD technique
3. `docs/research/SPARK_FINETUNE_PLAN.md` — what 2 Sparks can / cannot do
4. Background only (do not contradict 1–3): `TRAINING_METHOD.md`, `TRAINING_DATA.md`, `REGIONAL_FINETUNE_GAP.md`, `FCN_REGIONAL_FINETUNES.md`

If docs conflict: PATHWAY + METHOD_DESIGN + SPARK_PLAN win. Ignore inference-frontier leftovers.

## Goal (locked)

Regional **probabilistic** skill over Nepal / HKH / South Asian orography — not chat FT, not NVIDIA Stage1→2→FT curriculum replay on Sparks.

Preserve FCN3 Lead picture: probabilistic 6 h spherical ensembles, 72-ch state, anisotropic+spectral ops, spatial+spectral CRPS.

## Non-goals

- Full FCN3 multi-step + large-ensemble weight FT on 2 Sparks (infeasible)
- MSE-only FT or replacing FCN3 with a pure MSE UNet as sole dynamics
- Claiming "FCN3 weight FT" if only a diagnostic trains (G3 honesty)
- Idea 1/2 glacial-lake / GLOF work (wiped; do not resurrect)

## Quality invariants (I1–I6) — hard gates

| ID | Invariant |
|----|-----------|
| I1 | Probabilistic 6 h map; ensembles stay sharp (no MSE blur) |
| I2 | One-step-per-member via HMM spherical multi-scale noise |
| I3 | 72-channel hydrostatic state + orography / LSM / coszen |
| I4 | Anisotropic local DISCO + global spectral spherical arch |
| I5 | Spatial CRPS + spectral CRPS (or principled equivalent) |
| I6 | Separate (a) NN/adapter weights, (b) systematic bias, (c) ERA5 inheritance |

## Method tiers (RRCA-FD)

| Tier | What | Status intent |
|------|------|---------------|
| **Tier-0** | Frozen FCN3 + elevation-aware bias baseline | First runnable |
| **Tier-A (Plan A primary)** | Frozen FCN3 + CorrDiff/StormCast-lite regional diagnostic / downscaler | Primary Spark story |
| **Tier-B (Plan B gated)** | LoRA on local DISCO + regional CRPS + spectral anchor | Only if A works + memory allows |

Loss discipline: spatially reweighted CRPS + spectral CRPS anchor (λ≈0.1); never MSE-only for dynamics claims.

## Eval gates

| Gate | Pass rule |
|------|-----------|
| G0 | Global CRPS/SSR + spectra @ +15 d: ≤~5% relative CRPS degrade vs base; no spectral collapse |
| G1 | Nepal t2m/winds/moisture/(precip): beat frozen FCN3 crop **and** Tier-0; elevation-banded t2m improves |
| G2 | Calibration SSR≈1 at +24–120 h over region |
| G3 | Honest labeling: Tier-A ≠ "FCN3 weight FT" unless Tier-B passes |

## Base model / data facts

| Item | Value |
|------|-------|
| Model | FourCastNet 3 (`nvidia/fourcastnet3`), ~711M, Apache-2.0 |
| Checkpoint on Spark | `~/fourcastnet/models/fourcastnet3/training_checkpoints/best_ckpt_mp0.tar` (~2.7G) |
| Norms | `global_means.npy`, `global_stds.npy`, mins/maxs in model dir |
| Grid | 0.25°, 721×1440, Δt = 6 h |
| Channels | 72 prognostics + land/sea/orography/coszen + noise |
| Training data (upstream FCN3) | ERA5 only (paper: train ~1980–2016; FT 2012–2016; prefer paper over model-card year slips) |

### Regional data streams (for our FT)

| Stream | Role | Access |
|--------|------|--------|
| ERA5 (72-ch + aux) | ICs / Tier-B labels / spectral anchor | CDS — pipeline TBD |
| DEM elevation/slope | Regional loss weights \(w_\mathcal{R}\) | Build |
| IMDAA (~12 km) | Tier-A high-res target | Access TBD |
| IMERG | Precip verify / diagnostic | TBD |
| HIWAT | Optional km CAM | Optional |
| Stations / DHM | Later obs-aware verify | Later |

## Spark honesty

- Feasible: FCN3 **inference**; Tier-0 bias; Tier-A CorrDiff/StormCast-lite train
- Not feasible here: full FCN3 AR + large-ens weight FT (needs domain-parallel cluster)
- Inter-Spark = RoCE ~200 Gb/s (not NVLink); memory is capacity-rich, bandwidth-modest (~273 GB/s)

## Pathway status (as of research pack 2026-09-10)

| Step | Status |
|------|--------|
| 1 Scope | ✅ largely done (box/years still open) |
| 2 Pretrained model | ✅ FCN3 |
| 3 Data prepare | 🔄 sources ID'd; crops/splits/masks not on disk |
| 4 FT technique | ✅ RRCA-FD |
| 5 Train | ❌ waiting Phase 0 |
| 6 Eval | ⏳ gates defined; no runs |

**Immediate next (Phase 0):** freeze lat–lon box + years → ERA5 crop + norms → frozen FCN3 inference smoke → Tier-0 bias baseline.

## Open decisions (Manisha — do not invent)

1. Exact geographic box (narrow Nepal vs wider HKH/monsoon)?
2. Train / val / test years?
3. Start Phase 0 on Spark now (loop Raj)?
4. Precip in v1 success criteria, or t2m/winds-first?

## Directory layout (Spark `~/fourcastnet`)

```
models/fourcastnet3/     # kept checkpoint + norms
venv → ~/fcn3-venv       # FCN3 / Earth2 stack (rebuilding fresh)
venv-idea1/              # analysis env (rebuilding; name legacy)
docs/
  research/              # Sheldon pack mirror
  REFERENCE.md           # this file
  CLAUDE.md              # resume + work log
  PROGRESS.md            # fine-tune progress for paper
code/ configs/ data/ runs/ logs/
```

## Coordination

| Who | Role |
|-----|------|
| Sheldon | Research docs |
| Howard | Eng / implement |
| Raj | Spark schedule — send command+cwd+success |
| Leonard | Experiment design / validation gates |
| Manisha | Next steps, open locks, approvals |

## Where to start (eng checklist)

1. Confirm fresh `fcn3-venv` loads torch + FCN3 checkpoint / Earth2Studio smoke
2. Lock box + years with Manisha
3. ERA5 crop pipeline for box (CDS)
4. Build \(w_\mathcal{R}\) elevation mask
5. Tier-0 bias probe scripts + baselines
6. Log every measured number in `PROGRESS.md` (paper trail)

## Locked data policy (2026-09-10, Manisha via Sheldon)

- Geo box: **26–31°N, 80–89°E** (CDS area 31/80/26/89)
- Years: **1980 → latest CDS available** (do not truncate end; ERA5T tip ~2026-09)
- Split: **80/20 random block-level** train/test (test ≥15–20%); not adjacent-frame shuffle
- Pull owner: **Howard**; Sheldon docs-only
- Aux: IMERG Final→2025-09-30; IMDAA→2020 (pull when able)
- CDS: `~/.cdsapirc` present on Spark

## Gate recipes

- `docs/research/GATE_RECIPE_TIER0_G0.md` — Leonard freeze 2026-09-11 (Tier-0 + G0 IC protocol). Random IC / Nepal-crop-as-IC forbidden for gates.
- `data/g0_ics/` + `g0_ic_manifest.json` — global ERA5 ICs for G0 (ARCO); stage via `code/phase0/stage_g0_ics.py`.

## Decision locks (Manisha 2026-09-11 ~10:50 PT)
- G0 full GPU: **greenlit** — running `g0_base.py --full --allow-gpu` (PID check logs/g0_base_full.nohup.out)
- precip_in_v1: **false** — t2m/winds-first; precip stretch only
- box: **26–31N, 80–89E** confirmed
- Tier-0 target: **ERA5 interim OK** (label target=ERA5_interim)
- years: freeze later; provisional suggest train 2018–2021 / val 2022 / test 2023–2024 (G0 not blocked)

## Gate / Phase 0 pointers (2026-09-11)

- Leonard G0 call: `docs/research/G0_BASE_CALL.md` — finite PASS; claimable G0 **NOT YET**
- Gate recipe: `docs/research/GATE_RECIPE_TIER0_G0.md`
- G0 results: `runs/phase0/g0/g0_base_results.json`
- Tier-0 real (ERA5_interim): `runs/phase0/tier0/tier0_metrics.json` + `tier0_bias_maps.nc`
- Verifying-ERA5 hook (CRPS/SSR/PSD): `code/phase0/g0_verifying_era5.py`

## Tier-A v0 (Phase 0→1)

| Item | Value |
|------|-------|
| Status | Scaffold + smoke OK (2026-09-11) |
| Method | RRCA-FD Plan A — frozen FCN3 + TinyElevResidualUNet residual |
| Code | `code/tier_a/` |
| Config | `configs/tier_a_v0.yaml` |
| Pairs | `runs/phase0/tier0_holdout/pairs/` (train 2018–21 / val 2022 / test 2023–24) |
| Outputs | `runs/phase0/tier_a/v0/` |
| Beat-this (frozen) | val t2m < **1.948661** K; test < **1.850338** K |
| Labels | `provisional_years=true`; `claim_level=interim_era5`; `g1_claimable=false` |
| Do not overwrite | `runs/phase0/g0/g0_verifying_results.json`, `runs/phase0/tier0_holdout/` |
| Smoke (CPU 30 ep) | val 1.902 / test 1.769 K (pipeline proof; not published PASS) |

## Tier-A v1-reg (Phase 0→1)

| Item | Path |
|------|------|
| Call | `docs/research/TIER_A_V1_SKETCH_CALL.md` |
| Note | `docs/research/TIER_A_V1_NOTE.md` |
| Code | `code/tier_a/v1/` |
| Config | `configs/tier_a_v1.yaml` |
| Outputs | `runs/phase0/tier_a/v1/` |
| Results | `runs/phase0/tier_a/v1/tier_a_v1_results.json` |
| Cut | v1-reg only (no diffusion); FCN3 frozen |
| Mechanical | val PASS 1.796 / +120h PASS 2.018 / test FAIL 1.950 → interim FAIL |

