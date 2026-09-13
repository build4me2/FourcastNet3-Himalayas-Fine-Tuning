# Fine-Tune Progress Log (paper trail)

**Project:** FCN3 → Nepal/HKH regional skill (RRCA-FD)  
**Rule:** Append only. Measured numbers only. No speculative claims. Separate eras/runs — never splice.

## Run registry

| Run ID | Date | Tier | Config | Status | Notes |
|--------|------|------|--------|--------|-------|
| — | — | — | — | — | No training/eval runs yet |

## Phase 0 — Bring-up

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| Fresh project dirs | OK | 2026-09-10 | `~/fourcastnet/{docs,code,data,configs,runs,logs}` |
| Research pack mirrored | OK | 2026-09-10 | 8 md files → `docs/research/` |
| Checkpoint present | OK | 2026-09-10 | `best_ckpt_mp0.tar` ~2.7G |
| `fcn3-venv` torch import | PENDING | 2026-09-10 | Reinstall in progress |
| Earth2Studio / FCN3 inference smoke | OK | 2026-09-11 | 4×16 frozen; crop (4,17,72,21,37); rollout 335.22s; peak CUDA 52.9GB |
| ERA5 regional crop on disk | NOT RUN | — | Needs box/years lock |
| Tier-0 bias baseline | CPU scaffold OK | 2026-09-11 | synthetic smoke not_skill; skill blocked on global ERA5 ICs |
| G0 global ERA5 ICs (≥8) | IN PROGRESS | 2026-09-11 | ARCO direct; smoke 2/8 on disk; backfill ic03–ic08 running; manifest `data/g0_ics/g0_ic_manifest.json` |
| G0 base Path A runner | FULL RUNNING (OOM fixed) | 2026-09-11 | preview_only storage; smoke_ok; full PID 654878 past m0 step5; `logs/g0_base_full.nohup.out` |

## Metrics tables (fill when measured)

### G0 — Global probe

| Lead | Metric | Base FCN3 | Candidate | Δ rel | Pass? |
|------|--------|-----------|-----------|-------|-------|
| +15 d | CRPS | — | — | — | — |
| +15 d | SSR | — | — | — | — |
| +15 d | Spectra | — | — | — | — |

### G1 — Regional (Nepal/HKH)

| Var | Lead | Frozen FCN3 | Tier-0 | Tier-A | Notes |
|-----|------|-------------|-------|--------|-------|
| t2m | — | — | — | — | elevation bands TBD |
| winds | — | — | — | — | — |
| moisture | — | — | — | — | — |
| precip | — | — | — | — | only if in v1 |

### G2 — Calibration

| Lead | SSR | Rank hist notes |
|------|-----|-----------------|
| — | — | — |

## Decisions affecting numbers

| Date | Decision | By | Impact |
|------|----------|-----|--------|
| 2026-09-10 | RRCA-FD Plan A primary; Plan B gated | Manisha→Sheldon | Method freeze |
| 2026-09-10 | Fresh start; Idea1 wiped | Manisha | Clean slate |
| TBD | Geo box | Manisha | Data crop |
| TBD | Years | Manisha | Splits |
| TBD | Precip in v1? | Manisha | G1 scope |

## Paper notes (facts only)

- Research/design phase complete as of PATHWAY 2026-09-10; experimental bring-up not started.
- Full FCN3 AR+ensemble weight FT assessed **not feasible** on 2× Spark (see SPARK_FINETUNE_PLAN).
- Public gap: no published FCN3 Himalaya weight FT (see REGIONAL_FINETUNE_GAP / FCN_REGIONAL_FINETUNES).

## Locked data policy (2026-09-10, Manisha via Sheldon)

- Geo box: **26–31°N, 80–89°E** (CDS area 31/80/26/89)
- Years: **1980 → latest CDS available** (do not truncate end; ERA5T tip ~2026-09)
- Split: **80/20 random block-level** train/test (test ≥15–20%); not adjacent-frame shuffle
- Pull owner: **Howard**; Sheldon docs-only
- Aux: IMERG Final→2025-09-30; IMDAA→2020 (pull when able)
- CDS: `~/.cdsapirc` present on Spark


## Append — 2026-09-10 PT (Howard Phase 0 stubs)

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| Phase 0 code stubs on Spark | OK | 2026-09-10 | `code/phase0/*.py`, configs YAML |
| `load_norms.py` artifact presence | OK | 2026-09-10 | All required files exist; ckpt 2843497639 bytes |
| `load_norms.py` numpy array shapes | SKIP | 2026-09-10 | `No module named numpy` in fcn3-venv |
| `env_smoke.py` torch import | FAIL | 2026-09-10 | torch install still in progress (uv cu130) |
| `inference_smoke.py` dry-run | BLOCKED | 2026-09-10 | Needs pyyaml + Manisha locks; no invent |
| GPU visible (`nvidia-smi -L`) | OK | 2026-09-10 | NVIDIA GB10 (earlier session) |

No G0/G1/G2 numbers yet — no inference/train runs.

## Append — config lock sync

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| Config YAML mirrors Manisha locks | OK | 2026-09-10 | box 26–31N/80–89E; 1980→latest; 80/20 block |
| precip_in_v1 | OPEN | 2026-09-10 | still null in config |

## 2026-09-10 RESUME Phase 0
Old stop lifted. Incomplete smoke file may exist (era5_single_2020-01.nc ~1.6M). Re-verify after Raj smoke.

### ERA5 smoke PASSED (2026-09-10 ~18:04 PT)
| File | Size | Notes |
|------|------|-------|
| era5_single_2020-01.nc | 1.59 MB | u10/v10/u100/v100/t2m/msl/tcwv |
| era5_pressure_2020-01.nc | 12.05 MB | u/v/z/t/q × 13 levels |
Full 1980→latest backfill launched by Raj (skip existing).

### Status 2026-09-11 morning (~07:18 PT)
- Env: torch 2.14.0+cu130, CUDA True; PHASE0_ENV_OK (env_smoke + load_norms)
- ERA5 backfill PID 611595 still running (~13h): through 1983-12 complete; retrieving 1984-01 single
- Files: 49 single + 49 pressure (~624 MB raw); target ~561 months 1980-01→2026-09
- Inference smoke: not run yet (next Phase 0 step)

## Append — 2026-09-11 ~08:34 PT (Howard — Phase 0 FCN3 inference smoke)

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| `inference_smoke.py` dry-run | OK | 2026-09-11 | ok=true; torch 2.14.0+cu130; earth2studio+makani |
| Local package load | OK | 2026-09-11 | `Package(~/fourcastnet/models/fourcastnet3)` → FCN3; n_params=380870790; 72 vars |
| Frozen 4×16 rollout | OK | 2026-09-11 | exit 0, ok=true; device cuda; no train/backprop |
| Crop artifacts | OK | 2026-09-11 | `data/cache/phase0_infer/`: 4× member*_crop.pt + preview.pt, ensemble_crop.npz, smoke_report.json |
| Ensemble shape | OK | 2026-09-11 | (4, 17, 72, 21, 37) = members × (IC+16 steps) × vars × Nepal crop |
| Rollout wall | 335.22 s | 2026-09-11 | members sequential ~82–85 s each; model load 28.47 s |
| Peak CUDA alloc | 52.939 GB | 2026-09-11 | reserved 69.407 GB; rss_gb 9.674 |
| IC source | Random | 2026-09-11 | ERA5 raw crops are Nepal 21×37 only; FCN3 needs global 721×1440 |
| torch-harmonics | 0.8.1 + disco CUDA | 2026-09-11 | rebuilt with FORCE_CUDA_EXTENSION (sm_121); without it ~747 s/step |
| ERA5 backfill PID 611595 | untouched | 2026-09-11 | still running after smoke |
| precip_in_v1 | OPEN | 2026-09-11 | not invented; not a smoke blocker |

Measured command: `~/fcn3-venv/bin/python code/phase0/inference_smoke.py` from `~/fourcastnet`.

## Append — 2026-09-11 ~08:42 PT (Howard — Tier-0 CPU scaffolding)

Aligned to Leonard freeze `docs/research/GATE_RECIPE_TIER0_G0.md` (mirrored from manii).

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| Gate recipe mirrored | OK | 2026-09-11 | `docs/research/GATE_RECIPE_TIER0_G0.md` |
| `tier0_bias.py --help` | OK | 2026-09-11 | exit 0 |
| `tier0_bias.py --dry-run` | OK | 2026-09-11 | ok=true; oro crop 21×37; elev 25–5632 m; bins n=[322,61,64,330] |
| `tier0_bias.py --smoke` | OK | 2026-09-11 | CPU-only; synthetic residual; not_skill=true; wall 0.117 s |
| Elevation mask `w_R` | OK | 2026-09-11 | `data/masks/w_R_elevation.nc` |
| Tier-0 artifacts | OK | 2026-09-11 | `runs/phase0/tier0/{tier0_bias_maps.nc,tier0_metrics.csv,tier0_metrics.json,README.md}` |
| GPU / FCN3 in Tier-0 | NOT USED | 2026-09-11 | Manisha/Raj holding GPU; scaffold is CPU-only |
| G0 / Tier-0 skill numbers | BLOCKED | 2026-09-11 | Need ≥8 global ERA5 ICs + `g0_base_results.json`; Random IC forbidden for gates |
| Nepal crop as IC | FORBIDDEN | 2026-09-11 | Per GATE_RECIPE §0; not used |
| ERA5 backfill PID 611595 | untouched | 2026-09-11 | still running |
| precip_in_v1 | OPEN | 2026-09-11 | not invented |

**Honest limitation:** smoke fit is synthetic elevation-correlated residual plumbing only — **not** a skill claim. Do not copy synthetic RMSE into G1 tables.

Measured commands (from `~/fourcastnet`):
- `~/fcn3-venv/bin/python code/phase0/tier0_bias.py --dry-run`
- `~/fcn3-venv/bin/python code/phase0/tier0_bias.py --smoke --allow-random-ic-plumbing`


## Append — 2026-09-11 PT (Howard G0 IC staging — GATE_RECIPE §3 step 1)

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| `code/phase0/stage_g0_ics.py` + `arco_direct_fetch.py` | OK | 2026-09-11 | FCN3 72-ch order from local `config.json`; ARCO gs zarr; retries |
| `configs/g0_ics.yaml` | OK | 2026-09-11 | N=8 mix: 4 non-Asia + 2 Asia-ex-Nepal + 2 monsoon-adj; 4 DJF + 4 JJA; all 00 UTC |
| Smoke ICs on disk | OK | 2026-09-11 | `ic_20180115T0000_global.npy`, `ic_20190712T0000_global.npy` (each ~286 MiB; shape 72×721×1440 float32) |
| `g0_ic_manifest.json` | OK | 2026-09-11 | dates, region rationale, source URL, sha256, sizes |
| Remaining 6 ICs | LAUNCHED | 2026-09-11 | `nohup code/phase0/run_g0_ics_backfill.sh`; log `logs/g0_ics_backfill.log` |
| CDS competition | AVOIDED | 2026-09-11 | ARCO only; regional backfill PID 611595 left running |
| GPU / G0 rollout | NOT RUN | 2026-09-11 | Manisha GPU widget skipped → treat as NOT approved |

**Source:** `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (ERA5 final ARCO).  
**Forbidden honored:** no Nepal crop-as-IC; no Random IC for gate artifacts; no GPU G0.

**Next (when GPU greenlit):**
```bash
# after backfill completes (staged==8 in manifest):
source ~/fcn3-venv/bin/activate
cd ~/fourcastnet
# G0 base path A — implement/run only after Manisha GPU OK
# python code/phase0/g0_base.py --manifest data/g0_ics/g0_ic_manifest.json
```


## Append — 2026-09-11 ~09:24 PT (Howard — G0 base Path A runner scaffolding)

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| `code/phase0/g0_base.py` + `configs/g0_base.yaml` | OK | 2026-09-11 | Path A: global IC → global rollout → CRPS/SSR/PSD stubs |
| `g0_base.py --dry-run` | OK | 2026-09-11 | exit 0; ok=true; status=dry_run_ready; gpu_used=false |
| Staged ICs verified | OK | 2026-09-11 | 3/8 on disk at dry-run (ic01–ic03); shapes 72×721×1440 float32 |
| Wall estimate full G0 | ~2.67 h | 2026-09-11 | 8 IC × 4 members × 60 steps × ~5 s/step + ~30 s load (disco basis) |
| `--full` / `--smoke` without `--allow-gpu` | REFUSED | 2026-09-11 | exit 3; status=gpu_not_allowed |
| Full G0 GPU run | NOT RUN | 2026-09-11 | Manisha GPU not approved; scaffolding only |
| G0 IC backfill PID 641936 | untouched | 2026-09-11 | still running |
| ERA5 regional PID 611595 | untouched | 2026-09-11 | still running |
| Plan artifact | OK | 2026-09-11 | `runs/phase0/g0/g0_base_dry_run.json` |

**Full G0 not executed.** Results target when greenlit: `runs/phase0/g0/g0_base_results.json`.

Measured command (from `~/fourcastnet`):
```bash
~/fcn3-venv/bin/python code/phase0/g0_base.py --dry-run
```

## Decision locks (Manisha 2026-09-11 ~10:50 PT)
- G0 full GPU: **greenlit** — running `g0_base.py --full --allow-gpu` (PID check logs/g0_base_full.nohup.out)
- precip_in_v1: **false** — t2m/winds-first; precip stretch only
- box: **26–31N, 80–89E** confirmed
- Tier-0 target: **ERA5 interim OK** (label target=ERA5_interim)
- years: freeze later; provisional suggest train 2018–2021 / val 2022 / test 2023–2024 (G0 not blocked)


## Append — 2026-09-11 ~11:27 PT (Howard — G0 base OOM fix + full relaunch)

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| Root cause | OOM | 2026-09-11 | Full (S,V,H,W) CPU storage ≈18 GB/member ×4 ≈73 GB + model on 128 GB UMA; PID 650049 died after ic01 start |
| Fix | preview_only | 2026-09-11 | Keep preview_variables only; discard full field each step; metrics on (M,S,Vp,H,W) |
| Traceback path | OK | 2026-09-11 | GPU failures → `logs/g0_base_full.traceback.txt` + JSON flush |
| `--smoke --allow-gpu` | OK | 2026-09-11 | exit 0; smoke_ok; RSS~9.7 GB; CUDA alloc~53 GB |
| `--full --allow-gpu` relaunch | RUNNING | 2026-09-11 | PID **654878**; preview_only; M=4; past ic01 member0 step5+ |
| ERA5 regional PID 611595 | untouched | 2026-09-11 | still running |
| Only one g0_base | YES | 2026-09-11 | single python process |
| Log | `logs/g0_base_full.nohup.out` | 2026-09-11 | PYTHONUNBUFFERED; progress every 5 steps |
| Partial checkpoints | `runs/phase0/g0/partial/` | 2026-09-11 | per-IC JSON |

### 2026-09-11 ~14:15 PT — G0 full DONE
- `runs/phase0/g0/g0_base_results.json` status=full_ok; 8/8 ICs; all_finite=true; preview_only storage
- Peak CUDA alloc ~53GB; g0_pass_claimable=false (CRPS/SSR/PSD stubs — need verifying ERA5)
- Next: Tier-0 real fit (global-IC→crop, target=ERA5_interim); wire real G0 metrics later

## Append — 2026-09-11 ~15:19 PT (Howard — Tier-0 REAL / ERA5_interim)

Path B: global ERA5 IC → frozen FCN3 → Nepal crop 26–31N 80–89E vs ARCO ERA5 at +24/+72/+120 h.
Leonard `G0_BASE_CALL.md` already on disk: finite PASS; claimable G0 NOT YET.

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| Approach | **B** | 2026-09-11 | G0 preview tensors were discarded (`preview_only`); Path A crop-from-saved-preview blocked |
| `tier0_era5_targets.py` | OK | 2026-09-11 | 8 IC × 3 leads × t2m/u10m/v10m/tcwv; ARCO crop; wall **238.5 s**; CDS unused |
| ic03 ARCO vs local CDS 2020-01 | OK | 2026-09-11 | max\|Δt2m\|≈0.0008 K (float32 noise) |
| `tier0_crop_rollout.py --allow-gpu` | OK | 2026-09-11 | 8 IC × 2 mem × 20 steps; crop 21×37; wall **1656.6 s**; CUDA alloc 53.2 GB; PID 663758; one GPU process |
| `tier0_bias.py --real` | OK | 2026-09-11 | status=`real_ok`; claim_level=`interim_era5`; not_skill=`false`; g1_claimable=`false` |
| Elevation bins n_grid | [322, 61, 64, 330] | 2026-09-11 | same `w_R_elevation.nc` |
| t2m RMSE pooled (8 IC × 3 leads, n=18648) | raw **2.028 K** → lin **1.978 K** | 2026-09-11 | valid land points |
| t2m RMSE +24 / +72 / +120 h | 1.759 / 2.136 / 2.164 → lin 1.575 / 1.937 / 2.095 | 2026-09-11 | n=6216/lead |
| t2m high-elev [4500,9000] m pooled a_const | **−0.666 K** | 2026-09-11 | FCN3 warmer than ERA5; n=7920 |
| t2m lowland [0,1500) m pooled a_const | **+0.062 K** | 2026-09-11 | n=7728 |
| Year split | NOT frozen | 2026-09-11 | all 8 ICs are 2018–2021; suggested later 2018–21 / 2022 / 2023–24 |
| ERA5 regional PID 611595 | untouched | 2026-09-11 | still running |
| G0 CRPS/SSR/PSD | still stubs | 2026-09-11 | hook: `code/phase0/g0_verifying_era5.py` (plan only; preview tensors not saved) |

**Artifacts**
- `runs/phase0/tier0/tier0_bias_maps.nc`
- `runs/phase0/tier0/tier0_metrics.csv`
- `runs/phase0/tier0/tier0_metrics.json`
- `runs/phase0/tier0/pairs/forecast/` + `pairs/targets/`
- smoke CPU run preserved under `runs/phase0/tier0/smoke_cpu_20260911/`

**Honesty:** ERA5-as-target only. Cannot claim G1. Cannot claim G0 PASS (CRPS still stub). Year-split not frozen — fit used all staged ICs (train-era years only; no 2022+ holdout).

Measured commands (from `~/fourcastnet`):
```bash
~/fcn3-venv/bin/python code/phase0/tier0_era5_targets.py
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/phase0/tier0_crop_rollout.py --allow-gpu
~/fcn3-venv/bin/python code/phase0/tier0_bias.py --real
~/fcn3-venv/bin/python code/phase0/g0_verifying_era5.py --dry-run
```


## Append — 2026-09-11 PT (G0 verifying CRPS/SSR/PSD — Leonard #1)

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| `g0_verifying_era5.py` streaming scorer | OK | 2026-09-11 | Fair CRPS + SSR + zonal PSD; lead-only preview (no 73GB) |
| Smoke verifying | OK | 2026-09-11 | 1 IC (ic01) × 2 mem × 20 steps @ +120 h; `runs/phase0/g0/g0_verifying_smoke_results.json` |
| Smoke CRPS midlat +120 h | MEASURED | 2026-09-11 | t2m 0.692; z500 152.9; t850 0.895; u10m 1.173; v10m 1.218; tcwv 1.701 |
| Smoke SSR midlat +120 h | MEASURED | 2026-09-11 | t2m 0.536; z500 0.587; t850 0.612 (sane; not collapsed) |
| Smoke PSD ratio synoptic | MEASURED | 2026-09-11 | t2m 0.84; z500 0.87; t850 0.78 (sane band 0.25–4) |
| Smoke peak mem | MEASURED | 2026-09-11 | CUDA alloc 52.9 GB; reserved 69.4 GB; RSS 9.8 GB |
| Smoke wall | MEASURED | 2026-09-11 | fetch 106 s (6-var ARCO); rollout 211 s |
| Full verifying | OK full_ok | 2026-09-11 | 8/8 ICs × 4 mem × 60 steps; leads 120/240/360; rollout 9971 s; `g0_verifying_results.json` |
| `g0_pass_claimable` | TRUE | 2026-09-11 | Real CRPS/SSR/PSD; all_finite; SSR/PSD sane; protocol complete @ +360 h |
| ERA5 regional PID 611595 | UNTOUCHED | 2026-09-11 | Still alive |
| Provisional years (Leonard) | NOTED | 2026-09-11 | train 2018–2021 / val 2022 / test 2023–2024; Tier-0 bar t2m RMSE < 1.978 K (holdout re-score required) |

### G0 — Global probe (partial; smoke only)

| Lead | Metric | Base FCN3 | Candidate | Δ rel | Pass? |
|------|--------|-----------|-----------|-------|-------|
| +5 d | CRPS t2m midlat | 0.692 | — | — | smoke only |
| +5 d | SSR t2m midlat | 0.536 | — | — | sane |
| +5 d | PSD ratio t2m | 0.84 | — | — | sane |
| +15 d | CRPS | — | — | — | full running |
| +15 d | SSR | — | — | — | full running |
| +15 d | Spectra | — | — | — | full running |

**Honesty:** Leonard ~5% CRPS rule is adapter-vs-this-baseline. Base claimable only after full 8-IC +15 d real metrics + SSR/PSD sane. Stub `g0_base_results.json` kept; sidecar + `g0_verifying_results.json` are the skill paths.


## Append — 2026-09-11 PT (G0 verifying FULL complete)

| Check | Result | Date | Notes |
|-------|--------|------|-------|
| Full verifying | OK | 2026-09-11 | status=`full_ok`; `runs/phase0/g0/g0_verifying_results.json` |
| Verifying ERA5 frames | 24/24 | 2026-09-11 | `data/g0_verify/*_era5_preview.npz` ARCO |
| Rollout wall | 9971.17 s | 2026-09-11 | ~2.77 h; load 28.74 s; peak CUDA alloc 52.9 GB |
| +15 d CRPS midlat (8 IC mean) | MEASURED | 2026-09-11 | t2m 1.277; z500 440.1; t850 1.971; u10m 2.312; v10m 2.345; tcwv 2.983 |
| +15 d SSR midlat | MEASURED | 2026-09-11 | t2m 0.666; z500 0.732; t850 0.787 (sane; no collapse) |
| +15 d PSD synoptic ratio | MEASURED | 2026-09-11 | t2m 0.712; z500 0.438; t850 0.515; winds ~0.38–0.41 (within 0.25–4 sane band; borderline blur on winds/z) |
| +5 d CRPS midlat | MEASURED | 2026-09-11 | t2m 0.673; z500 157.1; t850 0.893 |
| +10 d CRPS midlat | MEASURED | 2026-09-11 | t2m 1.054; z500 324.3; t850 1.584 |
| g0_pass_claimable | TRUE | 2026-09-11 | Base GATE §1.5; sidecar `g0_base_results_verifying_sidecar.json`; stub base JSON preserved |
| ERA5 regional 611595 | UNTOUCHED | 2026-09-11 | Alive through full run |

### G0 — Global probe (FULL base; midlat CRPS)

| Lead | Metric | Base FCN3 | Candidate | Δ rel | Pass? |
|------|--------|-----------|-----------|-------|-------|
| +5 d | CRPS t2m | 0.673 | — | — | real |
| +10 d | CRPS t2m | 1.054 | — | — | real |
| +15 d | CRPS t2m | 1.277 | — | — | real / claimable baseline |
| +15 d | CRPS z500 | 440.1 | — | — | real |
| +15 d | CRPS t850 | 1.971 | — | — | real |
| +15 d | SSR t2m | 0.666 | — | — | sane |
| +15 d | PSD t2m | 0.712 | — | — | sane |
| +15 d | PSD z500 | 0.438 | — | — | sane (borderline) |

**Leonard ~5% rule:** applies to future adapters vs this file — not to base itself. Base claimable = true.


## Append — 2026-09-11 ~20:15 PT (Tier-0 HOLDOUT — provisional years)

Leonard priority #2 after G0 verifying DONE / G0_VERIFYING_CALL.md PASS claimable YES.
No Tier-A. provisional_years=true (Manisha has not hard-locked).

| Check | Result | Notes |
|-------|--------|-------|
| Stage train ICs 2018–2021 | OK | Reused 8 G0 globals via hardlink -> data/tier0_ics/ |
| Stage val 2022 (ic09–ic12) | OK | ARCO global 721x1440x72 |
| Stage test 2023–2024 (ic13–ic16) | OK | ARCO; 16/16 staged |
| Manifest split | OK | tier0_ic_manifest.json train=8/val=4/test=4; provisional_years=true; year_split_frozen=true |
| ERA5 targets val+test | OK | ARCO crop; CDS unused; PID 611595 untouched |
| Crop rollout val+test | OK | 8 IC x 2 mem x 20 steps; crop_rollout_ok; GPU one job |
| Train-only elev-binned fit | OK | Fit on ic01–ic08 only; maps tier0_bias_maps_train_only.nc |
| Val score (apply train coefs) | MEASURED | t2m pooled raw 1.949 -> lin 1.949 K |
| Test score | MEASURED | t2m pooled raw 1.893 -> lin 1.850 K |
| Train self (sanity) | MATCH | 2.028 -> 1.978 K (= prior all-train table) |
| claim_level | interim_era5 | g1_claimable=false |
| Beat-this (updated) | val lin 1.949 K | Primary=val; test secondary 1.850; prior 1.978 optimistic |
| ERA5 regional PID 611595 | UNTOUCHED | Alive through holdout |
| Tier-A | NOT STARTED | Per steering |

### Holdout t2m pooled (train-only fit applied)

| Split | n_IC | RMSE raw | RMSE lin | Notes |
|-------|-----:|---------:|---------:|-------|
| train_self | 8 | 2.028 | 1.978 | same ICs as fit |
| val 2022 | 4 | 1.949 | 1.949 | primary beat-this |
| test 2023–24 | 4 | 1.893 | 1.850 | secondary |

**Honesty:** Val elevation-band biases differ from train (lowland +0.75 K vs train +0.06 K) — linear correction barely moves pooled val RMSE. Small N (4+4). Years provisional.

**Artifacts**
- data/tier0_ics/tier0_ic_manifest.json + 16 global npy
- runs/phase0/tier0_holdout/ metrics/maps/pairs
- configs: tier0_holdout_ics.yaml, tier0_bias.yaml locks provisional
- code: stage_tier0_holdout_ics.py, tier0_holdout.py; crop/targets --splits/--ids

## Append — 2026-09-11 ~20:25 PT (Tier-A v0 scaffold + smoke)

Leonard TIER0_HOLDOUT_CALL.md: **Tier-A eng GO**. RRCA-FD Plan A (frozen FCN3 + residual diagnostic).

| Check | Result | Notes |
|-------|--------|-------|
| Scaffold `code/tier_a/` | OK | dataset + TinyElevResidualUNet + train.py |
| Config beat-this echo | OK | val <1.948661 / test <1.850338 in JSON |
| Dry-run | OK | pairs + elev + splits resolve |
| Smoke train (CPU, 30 ep) | OK | ~15 s; params=35507; n_train=72 |
| Val t2m pooled RMSE (Tier-A) | **1.902 K** | bar 1.948661 — smoke interim beat |
| Test t2m pooled RMSE (Tier-A) | **1.769 K** | bar 1.850338 — smoke interim beat |
| claim_level | interim_era5 | g1_claimable=false; provisional_years=true |
| G0 adapter | N/A | FCN3 frozen; baseline path echoed |
| Forbidden overwrites | OK | g0_verifying + tier0_holdout untouched |
| ERA5 PID 611595 | UNTOUCHED | still alive |
| CorrDiff | NOT STARTED | optional after v0 |

**Honesty:** Smoke on 4+4 holdout ICs; small N; ERA5_interim only; not G1; not FCN3 weight FT; not dynamics/CRPS claim. Treat smoke gate as pipeline proof — longer train + Leonard PASS call still required for published interim beat.

**Artifacts**
- `configs/tier_a_v0.yaml`
- `code/tier_a/{train,dataset,model}.py`
- `runs/phase0/tier_a/v0/tier_a_v0_results.json` (+ ckpt, csv, dry_run)

### Tier-A v0 300-ep train (CPU, finished ~20:24 PT)

- Reloaded best-val ckpt (early ~ep35; later epochs overfit val→~2.39).
- Final: val t2m **1.892 K**, test **1.733 K** vs bars 1.948661 / 1.850338.
- elapsed ~147 s CPU. Log: `logs/tier_a_v0_train300.nohup.out`.
- Still interim_era5 / provisional_years / g1_claimable=false — ping Leonard for PASS call; not auto-published.


## Append — 2026-09-11 ~20:34 PT (Leonard Tier-A v0 call + v0.1 lead-balanced)

### Leonard `TIER_A_V0_CALL.md` (v0)

| Gate | Result |
|------|--------|
| Val / test beat-this | PASS 1.892 / 1.733 |
| **Tier-A v0 INTERIM PASS** | **YES** |
| G0 adapter | N/A (FCN3 frozen) |
| Published / G1 | NO |

**WARN:** val +120 h raw 2.220 → v0 **2.382** (regress); pooled win short-lead heavy. **FREEZE** `runs/phase0/tier_a/v0/` — do not overwrite.

### Tier-A v0.1 (lead-balanced / multi-lead loss)

Same splits/bars; FCN3 frozen; outputs only under `runs/phase0/tier_a/v0_1/`.

| Check | Result | Notes |
|-------|--------|-------|
| Loss | equal-weight multi-lead elev-weighted MSE | +24/+72/+120 each 1/3; `ckpt_select=lead_mean` |
| Train | OK CPU 300 ep ~156 s | params=35507; n_train=72 |
| Val pooled | **1.889 K** | bar <1.948661 ✓ (vs v0 1.892) |
| Test pooled | **1.786 K** | bar <1.850338 ✓ (vs v0 1.733; still pass) |
| Val +120 h | **2.354** | vs raw 2.220 (still regress); vs v0 2.382 (slight improve) |
| Test +120 h | **1.832** | slight improve vs raw 1.844 |
| Mechanical interim gates | val+test PASS | claim_level=interim_era5; g1_claimable=false |
| Forbidden | OK | v0 / g0_verifying / tier0_holdout / ERA5 PID 611595 untouched |

**Honesty:** Lead-balance partially mitigates +120 h WARN vs v0 but does **not** remove val +120 h regression vs raw. Small-N; not G1; not CorrDiff; not FCN3 FT.

**Artifacts**
- `configs/tier_a_v0_1.yaml`
- `docs/research/TIER_A_V0_1_NOTE.md`
- `runs/phase0/tier_a/v0_1/tier_a_v0_1_results.json` (+ ckpt, csv)
- Log: `logs/tier_a_v0_1_train300.nohup.out`


## Append — 2026-09-11 ~20:46 PT (Leonard Tier-A v1-reg)

Leonard `TIER_A_V1_SKETCH_CALL.md`: **v1-reg first, NO diffusion**. FCN3 frozen. New dirs only.

| Check | Result | Notes |
|-------|--------|-------|
| Code `code/tier_a/v1/` | OK | ElevCondResidualUNet base32/depth3/FiLM; 665411 params |
| Config w_120 | **3.0** | ≥1 documented; w_24=w_72=1; composite reject +120h regress |
| Train | OK CPU early-stop ep180 / best ep100 ~268 s | train ICs only; seed 42 |
| Val pooled | **1.796 K** | bar <1.948661 **PASS** (vs v0.1 1.889) |
| Test pooled | **1.950 K** | bar <1.850338 **FAIL** (raw 1.893; overfit) |
| Val +120 h | **2.018 K** | raw 2.220; **PASS** hard rule (−0.202) |
| Mechanical interim | **FAIL** | val_pass=true, test_pass=false, val_120h_le_raw=true |
| claim_level | interim_era5 | g1_claimable=false; provisional_years=true |
| Diffusion / FCN3 FT | NOT STARTED | per freeze |
| Forbidden | OK | v0 / v0.1 / g0_verifying / tier0_holdout / ERA5 PID 611595 untouched |

**Honesty:** +120 h hard gate now operational and this cut meets it, but larger residual overfits 4 val ICs — test regresses vs raw. Small-N; not G1; not CorrDiff; not FCN3 FT.

**Artifacts**
- `configs/tier_a_v1.yaml`
- `code/tier_a/v1/{model,dataset,train}.py`
- `docs/research/TIER_A_V1_NOTE.md`
- `runs/phase0/tier_a/v1/tier_a_v1_results.json` (+ ckpt, csv)
- Log: `logs/tier_a_v1_train400.nohup.out`

## Append — 2026-09-11 ~20:57 PT (Leonard Tier-A v1.1)

Leonard `TIER_A_V1_CALL.md`: v1-reg **FAIL** (test 1.950). Freeze `v1/`. Iterate **v1.1** smaller/regularized. **NO diffusion**.

| Check | Result | Notes |
|-------|--------|-------|
| Code `code/tier_a/v1_1/` | OK | ElevCondResidualUNet base16/depth3/dropout0.20; 167843 params |
| Config w_120 | **1.5** | ≥1; w_24=w_72=1; composite reject +120h regress; wd=1e-3; patience 50 |
| Train | OK CPU early-stop ep283 / best ep233 ~252 s | 14 eligible saves; train ICs only; seed 42 |
| Val pooled | **1.716 K** | bar <1.948661 **PASS** (vs v1 1.796) |
| Test pooled | **1.812 K** | bar <1.850338 **PASS** (vs v1 1.950 FAIL; raw 1.893; v0 still 1.733 champ) |
| Val +120 h | **2.015 K** | raw 2.220; **PASS** hard rule (−0.205) |
| Mechanical interim | **PASS** | val_pass=true, test_pass=true, val_120h_le_raw=true |
| claim_level | interim_era5 | g1_claimable=false; provisional_years=true |
| Diffusion / FCN3 FT | NOT STARTED | per freeze |
| Forbidden | OK | v0 / v0.1 / **v1** / g0_verifying / tier0_holdout / ERA5 PID 611595 untouched |

**Honesty:** Smaller/regularized residual is the first cut to meet all three v1-era gates. Test pooled still worse than v0/v0.1; test +120 h still > raw (2.000 vs 1.844). Small-N; not G1; not CorrDiff; not FCN3 FT.

**Artifacts**
- `configs/tier_a_v1_1.yaml`
- `code/tier_a/v1_1/{model,dataset,train}.py`
- `docs/research/TIER_A_V1_1_NOTE.md`
- `runs/phase0/tier_a/v1_1/tier_a_v1_1_results.json` (+ ckpt, csv)
- Log: `logs/tier_a_v1_1_train400.nohup.out`

## Append — 2026-09-12 ~11:05 PT (Holdout IC expand v1)

**Manisha chose expand holdout** (thicken val/test). Leonard froze recipe: `docs/research/HOLDOUT_IC_EXPAND_RECIPE.md`.

| Check | Result | Notes |
|-------|--------|-------|
| Decision | EXPAND | No year hard-lock; no diffusion; years still provisional 2018–2021 / 2022 / 2023–2024 |
| Recipe | holdout_expand:v1 | train 8 keep / val 12 / test 12; add ic17–ic32 (16 new) |
| Config | UPDATED | `configs/tier0_holdout_ics.yaml` ic01–ic32; backup `.bak_pre_expand` |
| Seasons | DJF+MAM+JJA+SON | ≥3 per season in val and test after expand |
| Staging | **DONE** | 16 new `.npy` (~286MB/299013248B each); log `logs/tier0_holdout_expand_ics.log`; existing skipped; aiohttp loop-close noise after ok=true (benign) |
| Manifest target | 8/12/12 | `data/tier0_ics/tier0_ic_manifest.json` + `holdout_expand: v1` |
| ERA5 1980 backfill | UNTOUCHED | PID 611595 left running |
| Re-train / GPU eval | NOT STARTED | this task = stage + docs only |

**Concrete new ICs (00Z):**  
Val: ic17 2022-02-15 DJF N_Atlantic; ic18 2022-03-15 MAM East_Asia; ic19 2022-04-15 MAM SH_midlat; ic20 2022-05-15 MAM Bay_of_Bengal_premonsoon; ic21 2022-08-15 JJA West_Pacific_typhoon; ic22 2022-09-15 SON Bay_of_Bengal_retreat; ic23 2022-10-15 SON N_Pacific; ic24 2022-11-15 SON East_Asia.  
Test: ic25 2023-03-15 MAM N_Atlantic; ic26 2023-05-15 MAM Bay_of_Bengal_premonsoon; ic27 2023-09-15 SON West_Pacific; ic28 2023-11-15 SON NE_India_Myanmar_fringe; ic29 2024-02-15 DJF SH_midlat; ic30 2024-04-15 MAM East_Asia; ic31 2024-07-12 JJA N_Pacific; ic32 2024-10-15 SON Bay_of_Bengal_retreat.

**Next (parent phase):** pairs for new ICs → Tier-0 re-fit/score on expand → Tier-A v1.1 zero-shot re-score → Leonard freeze new beat-this. No diffusion; no year hard-lock; 1980 ERA5 crop untouched.

### Holdout expand staging COMPLETE — 2026-09-12 ~12:27 PT

- Manifest `holdout_expand: v1` · listed/staged **32** · by_split **train8 / val12 / test12**
- Seasons: val & test each **DJF3 / MAM3 / JJA3 / SON3** (thin set lacked MAM/SON — fixed)
- All ic17–ic32 `.npy` under `data/tier0_ics/`; no ±1 day shifts needed
- ERA5 crop PID **611595** untouched; no re-train / no GPU eval this task
- **Next (parent):** pairs for new ICs → Tier-0 re-fit train ic01–ic08 + score expand val/test → write `runs/phase0/tier0_holdout_expand/tier0_holdout_expand_metrics.json` → Tier-A v1.1 zero-shot on expand (frozen `v1_1/best_residual.pt`) → Leonard freeze new bars


## Append — 2026-09-12 ~12:33 PT (Holdout expand pairs + Tier-A zero-shot scaffolding)

| Check | Result | Notes |
|-------|--------|-------|
| Thin pairs hardlink | OK | ic01–ic16 → `runs/phase0/tier0_holdout_expand/pairs/` (inode-shared) |
| Expand yaml | OK | `configs/tier_a_v1_1_expand.yaml` |
| train.py --eval-only | OK | CLI: `--eval-only --pairs-dir --out-dir --ckpt`; allows `v1_1_expand/` writes; thin `v1_1/` frozen |
| ERA5 targets NEW | RUNNING | ic17–ic32 CPU/ARCO; log `logs/tier0_expand_targets.log`; manifest `data/tier0_ics/tier0_ic_manifest.json` |
| GPU crop rollout | **HELD** | Wait for duplexchat PID **729380** to exit; do not kill; ERA5 **611595** untouched |
| Tier-0 expand fit | NOT STARTED | Needs 32 forecast+target npz |
| Tier-A v1.1 zero-shot | NOT STARTED | Frozen `v1_1/best_residual.pt` only after pairs+Tier-0 |

**Next when GPU idle:** `tier0_crop_rollout.py --ids ic17..ic32 --pairs-dir .../tier0_holdout_expand/pairs --allow-gpu` → verify 32+32 → Tier-0 re-fit → `--eval-only` → Leonard (parent pings).


## Append — 2026-09-12 ~13:53 PT (Holdout expand pairs + Tier-0 + v1.1 zero-shot COMPLETE)

| Check | Result | Notes |
|-------|--------|-------|
| Thin pairs hardlink | OK | ic01–ic16 inode-shared into expand pairs |
| Crop rollout NEW | OK | ic17–ic32; log `logs/tier0_expand_crop_rollout.log`; PID 744654 |
| ERA5 targets NEW | OK | CPU/ARCO; log `logs/tier0_expand_targets.log`; 32 targets |
| Pairs verify | OK | **32** forecast + **32** targets |
| Tier-0 expand fit | OK | train ic01–ic08 only; score val12+test12 |
| Tier-0 val lin | **1.987014** | n=27972; raw 1.999 |
| Tier-0 test lin | **1.897298** | n=27972; raw 1.994 |
| Metrics JSON | OK | `tier0_holdout_expand_metrics.json` (+ script name `tier0_holdout_metrics.json`) |
| train.py --eval-only | OK | zero_shot; frozen ckpt; no optimize |
| Tier-A v1.1 expand val pooled | **1.807716** | thin bar <1.948661 PASS |
| Tier-A v1.1 expand test pooled | **1.826052** | thin bar <1.850338 PASS |
| Val +120 h | **1.998044** | raw expand 2.132; thin raw-bar 2.220; le thin-raw PASS |
| Test +120 h | **1.818926** | raw 1.733 (tier_a > raw at +120h on expand test) |
| Mechanical vs thin bars | interim PASS | Leonard must freeze **new** expand beat-this; claim_level=interim_era5 |
| Diffusion / retrain | NOT DONE | frozen ckpt only |
| Forbidden | OK | thin `tier0_holdout/` (mtime Sep 11 20:15) + `v1_1/best_residual.pt` (Sep 11 20:56) + ERA5 **611595** untouched |

**Per-lead t2m RMSE (v1.1 zero-shot expand):**  
Val 24/72/120h: 1.538 / 1.857 / 1.998  
Test 24/72/120h: 1.657 / 1.987 / 1.819  

**Artifacts**
- `runs/phase0/tier0_holdout_expand/tier0_holdout_expand_metrics.json`
- `runs/phase0/tier_a/v1_1_expand/tier_a_v1_1_expand_results.json` (`zero_shot=true`, `holdout_expand=v1`)
- `configs/tier_a_v1_1_expand.yaml`
- Logs: `logs/tier0_expand_crop_rollout.log`, `logs/tier0_expand_targets.log`, `logs/tier0_expand_fit.log`, `logs/tier_a_v1_1_expand_eval.log`

**Next (superseded):** Leonard froze expand beat-this — see append 14:00 PT. No diffusion. No year hard-lock. No new Tier-A train unless Manisha orders.


## Append — 2026-09-12 ~14:00 PT (Leonard HOLDOUT_EXPAND_CALL — beat-this FROZEN)

**Call doc:** `docs/research/HOLDOUT_EXPAND_CALL.md` (authoritative).

| Layer | Result |
|-------|--------|
| Expand protocol 8/12/12 | **PASS** |
| New beat-this | **FROZEN** — primary val **< 1.987014**; secondary test **< 1.897298**; val +120h ≤ expand-val raw **2.131516** |
| v1.1 INTERIM PASS (thick / zero-shot) | **CONFIRM YES** — val **1.808** / test **1.826** / +120h **1.998** |
| Thin bars 1.948661 / 1.850338 / 2.220 | **historical only** (v0/v0.1/v1/v1.1 thin) |
| New Tier-A train / diffusion | **NO** unless Manisha orders |
| claim_level | `interim_era5`; `g1_claimable=false`; `provisional_years=true` |

**Eng (no train / no GPU):**
- `configs/tier_a_v1_1_expand.yaml` beat_this → expand bars; `holdout_expand: v1`
- `code/tier_a/v1_1/train.py` defaults to expand bars when expand variant/paths/`holdout_expand=v1`; gate note uses live bars; `thin_beat_this_bars` stays thin
- `runs/phase0/tier_a/v1_1_expand/tier_a_v1_1_expand_results.json` `beat_this` patched in-place to expand bars (+ `thin_beat_this_bars` preserved)
- Backups: `*.bak_pre_expand_bars`
- Thin `tier0_holdout/` + `tier_a/v1_1/` + ERA5 PID **611595** untouched

**Next:** No new Tier-A train; no diffusion until Manisha asks. Living beat-this = expand bars above.
