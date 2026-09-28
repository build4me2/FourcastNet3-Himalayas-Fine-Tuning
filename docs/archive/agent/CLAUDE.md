# CLAUDE.md — FCN3 Regional Fine-Tune (agent resume)

**Owner eng:** Howard  
**Research:** Sheldon · **Compute:** Raj · **Design/tests:** Leonard  
**User:** Manisha Chand  
**Updated:** 2026-09-27 13:45 PT (FINAL residual v0 living; g1_claimable=true via G1_CLAIMABLE_UNLOCK_CALL)


## Unlock B applied (2026-09-13)

- Years **HARD-LOCKED**: train 2018–2021 / val 2022 / test 2023–2024.
- Labels: `provisional_years=false`, `year_split_frozen=true` (hard).
- `claim_level` stays `interim_era5`; `g1_claimable=false`.
- Expand bars unchanged. **No Tier-A retrain** for label flip.
- Canonical: `docs/research/FCN_NEXT_UNLOCK.md`, `docs/research/YEAR_HARD_LOCK.md`.
- Next: Unlock **A** thicken ICs (after B); no diffusion / new arch until A+B.
- ERA5 Nepal CDS PID **611595** must stay running.

## Mission

Implement RRCA-FD regional fine-tune of FCN3 for Nepal/HKH skill on 2× Spark. Continuously document for paper. Follow `docs/research/FINETUNE_PATHWAY.md`. Keep `docs/REFERENCE.md` accurate. Append measured results to `docs/PROGRESS.md` only.

## Hard rules

- Never resurrect Idea 1/2 lake/GLOF engineering
- Never delete `models/fourcastnet3/`
- Never invent geo box / years / precip policy — ask Manisha
- Never claim weight-FT if only Tier-A diagnostic trains (G3)
- No Canvas submit / no email send without Manisha approval
- No destructive commands on laptop manii unless she asks
- Ask Manisha directly for blockers (not via Bernadette)

## Canon docs

See `docs/REFERENCE.md` + `docs/research/FINETUNE_PATHWAY.md`.

## Current state

| Item | State |
|------|-------|
| Old Idea1/2 code/data/docs | Wiped from Spark (2026-09-10) |
| Checkpoint | Kept: `models/fourcastnet3/training_checkpoints/best_ckpt_mp0.tar` |
| Envs | `~/fcn3-venv` OK (torch 2.14.0+cu130, earth2studio, makani, torch-harmonics 0.8.1+disco CUDA) |
| Project dirs | Created: docs/, code/, data/, configs/, runs/, logs/ |
| Research mirror | `docs/research/*.md` copied from manii pack |
| REFERENCE / PROGRESS | Created this session |
| Phase 0 inference smoke | OK 2026-09-11 — 4×16 frozen local package; artifacts under data/cache/phase0_infer/ |
| Tier-0 bias (CPU scaffold) | OK 2026-09-11 — `code/phase0/tier0_bias.py`; w_R mask + synthetic smoke; **not_skill**; GPU held |
| Gate recipe Tier-0/G0 | Leonard freeze mirrored → `docs/research/GATE_RECIPE_TIER0_G0.md` |
| G0 global ERA5 ICs | IN PROGRESS 2026-09-11 — script `code/phase0/stage_g0_ics.py`; smoke 2/8 staged under `data/g0_ics/`; backfill for remaining launched (ARCO; no CDS) |
| G0 base path A runner | FULL RUNNING 2026-09-11 — OOM fixed (preview_only); smoke_ok; PID 654878 past m0 step5 |
| Training runs | None |

## Work log

### 2026-09-10

- Manisha ordered fresh start: read all `Desktop/Research/fourcastnet3` docs, new dir, follow PATHWAY, CLAUDE + continuous reference, document FT for paper; wipe earlier work.
- Confirmed wipe: Spark `~/fourcastnet` had only models + venvs; no Idea1 code/data.
- Sheldon: pack current; precedence PATHWAY > METHOD > SPARK_PLAN; open locks listed.
- Raj: standing by for command+cwd+success (no lake jobs).
- Leonard: standing by for design/gates ping.
- Created fresh tree under `~/fourcastnet`; mirrored research md; wrote REFERENCE.md, CLAUDE.md, PROGRESS.md.
- Env: `uv pip install torch==2.14.0+cu130` running into fresh `fcn3-venv` (torch not importable mid-install).

## Next human actions (Manisha)

1. Lock geographic box
2. Lock train/val/test years
3. Confirm precip-in-v1 vs t2m/winds-first
4. Confirm Phase 0 kickoff (Raj loop)

## Next eng actions (unblocked / partial)

1. ~~Env + inference smoke~~ DONE
2. ~~Tier-0 CPU scaffold~~ DONE (`tier0_bias.py` --dry-run/--smoke; w_R elevation mask)
3. ~~G0 IC staging script~~ DONE — `code/phase0/stage_g0_ics.py` + `configs/g0_ics.yaml`; ARCO direct zarr; smoke 2 staged; backfill →8 running (`logs/g0_ics_backfill.log`)
4. **G0 base runner ready (dry-run OK).** Blocked on Manisha GPU greenlight + staged≥8 for `--full --allow-gpu` → `runs/phase0/g0/g0_base_results.json`
5. After G0 base: Tier-0 fit on global-IC→crop forecasts (ERA5 target OK if labeled `target=ERA5_interim`)
6. Do **not** use Random IC or Nepal crop-as-IC for any gate number (GATE_RECIPE §0)
7. precip_in_v1 still open — do not invent

## Time-wasters to avoid

- Rebuilding Idea1 lake pipelines
- Full FCN3 AR+ens FT attempts on Spark
- Inventing box/years
- Claiming success from ERA5-only regional RMSE without G0/I1–I6

## Locked data policy (2026-09-10, Manisha via Sheldon)

- Geo box: **26–31°N, 80–89°E** (CDS area 31/80/26/89)
- Years: **1980 → latest CDS available** (do not truncate end; ERA5T tip ~2026-09)
- Split: **80/20 random block-level** train/test (test ≥15–20%); not adjacent-frame shuffle
- Pull owner: **Howard**; Sheldon docs-only
- Aux: IMERG Final→2025-09-30; IMDAA→2020 (pull when able)
- CDS: `~/.cdsapirc` present on Spark

- 2026-09-10 17:51: Manisha/Sheldon locked max-through-latest + box 26–31N 80–89E + 80/20 block split; ERA5 crop downloader kicked off.

### 2026-09-10 (Howard executor — Phase 0 stubs)

- Added `code/phase0/{env_smoke,load_norms,inference_smoke,tier0_bias_stub}.py` + `code/README.md`.
- Added `configs/phase0_nepal_box.yaml`, `configs/tier0_bias.yaml` (locks false; provisional box/years only).
- Added `data/README.md` placeholders (era5/imdaa/imerg/masks/cache empty).
- Ran `load_norms.py`: OK — norms, orography, land_mask, config.json (`nettype=FCN3`, `dhours=6`), `best_ckpt_mp0.tar` ~2.7G present. numpy not yet in `fcn3-venv` (shape load skipped).
- Ran `env_smoke.py`: FAIL torch missing — `uv pip install torch==2.14.0+cu130` still running into `~/fcn3-venv` (observed PID mid-session). Separate `venv-data` pip install (cdsapi/netCDF4/xarray/numpy/pyyaml) also mid-flight.
- Inference smoke not executed for real: blocked on Manisha locks (geo box, years, precip-in-v1) + Earth2Studio/torch not ready. Raj handoff when GPU train/infer starts.
- Did **not** overwrite REFERENCE/CLAUDE/PROGRESS/README bodies beyond this append.

### 2026-09-10 (Howard — config lock sync)

- Synced `configs/phase0_nepal_box.yaml` + `tier0_bias.yaml` to Manisha locks already recorded above: box **26–31°N, 80–89°E**, years **1980→latest CDS**, 80/20 block split. `precip_in_v1` still null.
- Prior stub note “blocked on Manisha locks” for box/years is **obsolete** for those two; remaining blockers = torch/Earth2Studio env + ERA5 crop on disk + precip policy.
- `load_norms` still the only measured Phase 0 script success; torch install still mid-flight at last check.

### 2026-09-10 ~18:01 PT — RESUME
- Manisha (via Raj): continue Phase 0; old full stop lifted for regional FT.
- Raj owns: torch uv + numpy/pyyaml/earth2studio + env_smoke + load_norms + ERA5 2020-01 smoke restart.
- Howard: no double-start; full 1980→latest backfill after smoke OK.
- Leftover incomplete?: data/era5/raw/era5_single_2020-01.nc ~1.6M from killed smoke — re-pull/verify.

- ERA5 smoke PASSED 2020-01 (single 1.59MB + pressure 12.05MB); full backfill launched by Raj.

### 2026-09-11 ~08:34 PT (Howard — Phase 0 inference smoke)

- Replaced stub `code/phase0/inference_smoke.py` with real Earth2Studio FCN3 path.
- Installed `makani` from NVIDIA git pin b38fcb27 + rebuilt `torch-harmonics==0.8.1` with `FORCE_CUDA_EXTENSION=1` / `TORCH_CUDA_ARCH_LIST=12.1` (disco CUDA True). Without disco CUDA: ~747 s/step / ~77–99 GB; with: ~5 s/step / ~53 GB alloc.
- Local package load OK: `~/fourcastnet/models/fourcastnet3` + `best_ckpt_mp0.tar` (no HF download).
- Ran frozen 4-member × 16-step sequential rollout; Nepal crop 26–31N, 80–89E → 21×37.
- Measured: load 28.47 s; rollout 335.22 s; peak CUDA alloc 52.939 GB; ensemble shape (4, 17, 72, 21, 37); ok=true exit 0.
- IC: `earth2studio.data.Random` (ERA5 regional crops cannot fill 721×1440).
- ERA5 backfill PID 611595 left untouched.
- `precip_in_v1` still open — not invented.

### 2026-09-11 ~08:42 PT (Howard — Tier-0 CPU scaffolding)

- Mirrored Leonard `GATE_RECIPE_TIER0_G0.md` → `docs/research/`.
- Replaced stub with CPU-only `code/phase0/tier0_bias.py` + updated `configs/tier0_bias.yaml` (Leonard elev bins 0/1.5/3/4.5/9 km).
- Built Nepal oro crop + `data/masks/w_R_elevation.nc`; smoke fit synthetic residual → `runs/phase0/tier0/` (not_skill=true).
- No GPU / no FCN3 inference in this step (Manisha/Raj holding GPU).
- ERA5 PID 611595 untouched. precip_in_v1 still null.
- Skill path blocked until global ERA5 ICs + G0 base results exist.

### 2026-09-11 ~09:24 PT (Howard — G0 base Path A scaffolding)

- Added `code/phase0/g0_base.py` + `configs/g0_base.yaml` (GATE_RECIPE §1 path A).
- Modes: `--dry-run` (default safe), `--smoke` / `--full` require `--allow-gpu`.
- Reuses inference_smoke Package+FCN3.load_model pattern; IC from staged global npy (not Random, not Nepal crop).
- `--dry-run` exit 0; verified 3 staged ICs; wall est full G0 ~2.67 h @ ~5 s/step disco.
- Wrote `runs/phase0/g0/g0_base_dry_run.json`. Did **not** run full GPU G0.
- PIDs 641936 (G0 IC backfill) and 611595 (ERA5 regional) left untouched.

## Decision locks (Manisha 2026-09-11 ~10:50 PT)
- G0 full GPU: **greenlit** — running `g0_base.py --full --allow-gpu` (PID check logs/g0_base_full.nohup.out)
- precip_in_v1: **false** — t2m/winds-first; precip stretch only
- box: **26–31N, 80–89E** confirmed
- Tier-0 target: **ERA5 interim OK** (label target=ERA5_interim)
- years: freeze later; provisional suggest train 2018–2021 / val 2022 / test 2023–2024 (G0 not blocked)

### 2026-09-11 ~11:25 PT (Howard — G0 OOM fix + full relaunch)

- **Root cause:** first full run (PID 650049) OOM after `ic01 start` — stored full (S,V,H,W)
  frames ≈ 18 GB/member × 4 ≈ 73 GB + ~53–69 GB model on 128 GB Spark UMA.
- **Fix:** `g0_base.py` keeps only `preview_variables` on CPU; discards full field each step;
  `metrics_stubs_global` on (M,S,Vp,H,W); GPU try/except → `logs/g0_base_full.traceback.txt`;
  progress every 5 steps; partial JSON under `runs/phase0/g0/partial/`.
- **Config:** `configs/g0_base.yaml` documents memory; M=4 kept (safe with preview_only).
- **Smoke:** `--smoke --allow-gpu` exit 0, `smoke_ok`, RSS~9.7 GB.
- **Full relaunch:** PID **654878**, `PYTHONUNBUFFERED=1`, nohup → `logs/g0_base_full.nohup.out`;
  alive past ic01 member0 step 5+. ERA5 PID 611595 untouched. One g0_base only.

### 2026-09-11 ~14:15 PT — G0 full DONE
- `runs/phase0/g0/g0_base_results.json` status=full_ok; 8/8 ICs; all_finite=true; preview_only storage
- Peak CUDA alloc ~53GB; g0_pass_claimable=false (CRPS/SSR/PSD stubs — need verifying ERA5)
- Next: Tier-0 real fit (global-IC→crop, target=ERA5_interim); wire real G0 metrics later

### 2026-09-11 ~15:19 PT (Howard — Tier-0 real)

- Path A (reuse G0 preview npz) **blocked**: G0 `storage_mode=preview_only` discarded tensors.
- Path B ran: global staged ICs → frozen FCN3 → Nepal 21×37 crops at +24/+72/+120 h (2 members, seed 333/334).
- Targets: ARCO ERA5 crop (not CDS). ic03 overlaps local `era5_single_2020-01.nc` (max |Δt2m|≈8e-4 K).
- Fit: elevation-binned linear; `claim_level=interim_era5`; `not_skill=false`; `g1_claimable=false`; years **not** frozen.
- t2m pooled RMSE 2.028 → 1.978 K after lin; high-elev (>4.5 km) mean bias −0.666 K.
- ERA5 PID 611595 untouched. GPU idle after crop.
- `docs/research/G0_BASE_CALL.md` present (finite PASS; claimable G0 NOT YET).
- Hook for verifying ERA5 CRPS/SSR/PSD: `code/phase0/g0_verifying_era5.py` (plan; needs G0 re-roll streaming scores — preview ensembles not on disk).

## Next eng actions
1. Ping Leonard with Tier-0 paths (maps/csv/json) — numeric ε freeze if wanted.
2. After Tier-0: wire G0 verifying ERA5 (`g0_verifying_era5.py --fetch-global` + stream-score re-roll). Do not claim G0 PASS from stubs.
3. Year-split still open (Manisha). No 2022+ ICs staged.
4. Leave CDS regional backfill (611595) running.

## Current — 2026-09-11 PT (G0 verifying)

- Leonard priority #1: real G0 CRPS/SSR/PSD vs ERA5 (blocks claimable G0).
- `code/phase0/g0_verifying_era5.py`: streaming lead-only re-roll + fair CRPS/SSR/zonal PSD.
- Smoke OK then FULL OK: 8/8 @ +120/240/360; +15 d CRPS midlat t2m=1.277 z500=440.1 t850=1.971; SSR t2m=0.666; PSD t2m=0.712.
- Artifacts: `g0_verifying_results.json` + sidecar; smoke kept as `g0_verifying_smoke_results.json`.
- `g0_pass_claimable`=**true** after full 8 IC @ +360 h (real CRPS/SSR/PSD; see `g0_verifying_results.json`).
- Provisional years: train 2018–2021 / val 2022 / test 2023–2024.
- Tier-0 interim beat-this: t2m RMSE < 1.978 K (holdout re-score required).
- ERA5 regional PID 611595 untouched.

## Current — 2026-09-11 PT evening (G0 verifying FULL)

- Leonard #1 DONE for base claimable metrics path.
- `g0_verifying_results.json` status=full_ok; g0_pass_claimable=true; crps_ssr_psd=real.
- +15 d midlat CRPS: t2m=1.277, z500=440.1, t850=1.971 (8-IC mean).
- SSR sane (~0.67–0.79); PSD synoptic ratios in sane band (winds/z500 ~0.38–0.44 borderline).
- Stub `g0_base_results.json` unchanged (plumbing); sidecar points to verifying results.
- ERA5 regional PID 611595 untouched.
- Next: ping Leonard with paths; adapters compared with ≤~5% CRPS degrade vs this baseline.



### 2026-09-11 ~20:15 PT (Tier-0 holdout provisional years)

- Staged val 2022 + test 2023–2024 global ERA5 ICs (ARCO) under data/tier0_ics/; train = hardlink reuse of G0 8 ICs.
- provisional_years=true, year_split_frozen=true (provisional only — Manisha hard-lock TBD).
- Re-fit Tier-0 elev-binned bias on train ICs only; scored val/test with frozen coefs.
- Val t2m lin RMSE 1.949 K (primary beat-this); test 1.850 K; train-self still 1.978.
- claim_level=interim_era5; g1_claimable=false. No Tier-A.
- ERA5 regional PID 611595 untouched.
- Leonard paths: runs/phase0/tier0_holdout/, data/tier0_ics/tier0_ic_manifest.json.

### 2026-09-11 ~20:25 PT (Tier-A v0 scaffold + smoke)

- Leonard Tier-A eng **GO** (TIER0_HOLDOUT_CALL.md). Plan A: frozen FCN3 + tiny residual UNet.
- Created `code/tier_a/` + `configs/tier_a_v0.yaml`; outputs under `runs/phase0/tier_a/v0/` only.
- Smoke CPU 30 ep: val t2m 1.902 / test 1.769 vs bars 1.948661 / 1.850338 (interim; small-N).
- Echoed beat-this + provisional_years=true + g1_claimable=false in result JSON.
- Did **not** overwrite g0_verifying or tier0_holdout; ERA5 PID 611595 untouched.
- Next: optional longer `--train --allow-gpu --epochs 300` (Raj); CorrDiff later.


### 2026-09-11 ~20:34 PT (Tier-A v0 INTERIM PASS + v0.1)

- Leonard `TIER_A_V0_CALL.md`: v0 INTERIM PASS **YES**; **FREEZE** `runs/phase0/tier_a/v0/`. WARN val +120h 2.220→2.382.
- Implemented v0.1 lead-balanced / equal-weight multi-lead loss (`configs/tier_a_v0_1.yaml`); train shared `code/tier_a/train.py`.
- Metrics on disk: `runs/phase0/tier_a/v0_1/tier_a_v0_1_results.json` — val 1.889 / test 1.786 (bars pass); val +120h **2.354** (still regress vs raw; better than v0).
- Note: `docs/research/TIER_A_V0_1_NOTE.md`. G0 adapter N/A. ERA5 PID 611595 / g0_verifying / tier0_holdout / v0 untouched.
- Next: ping Leonard with v0.1 paths; optional elev-band aux or RRCA-FD/CorrDiff v1 under new variant id.

### 2026-09-11 ~20:46 PT (Tier-A v1-reg — NO diffusion)

- Leonard `TIER_A_V1_SKETCH_CALL.md`: implement **v1-reg first**; no v1-diff; freeze v0/v0.1.
- New package `code/tier_a/v1/` (ElevCondResidualUNet 665k, FiLM lead+elev). PhysicsNeMo 3D UNet not used.
- Config `configs/tier_a_v1.yaml`: **w_120=3.0 ≥ 1**; composite ckpt rejects val +120h > raw 2.219971.
- Train CPU 400-ep cap, early-stop ep 180, best eligible ep 100, ~268 s. Outputs **only** `runs/phase0/tier_a/v1/`.
- Gates: val pooled **1.796 PASS** (<1.948661); val +120h **2.018 PASS** (≤2.220, −0.202 vs raw); test **1.950 FAIL** (<1.850338). `tier_a_interim_pass=false`.
- Note: `docs/research/TIER_A_V1_NOTE.md`. FCN3 frozen. No diffusion. ERA5 PID 611595 / g0_verifying / tier0_holdout / v0 / v0.1 untouched (md5).
- Next: ping Leonard — FAIL on test; iterate capacity/reg before any v1-diff.

### 2026-09-11 ~20:57 PT (Tier-A v1.1 — NO diffusion)

- Leonard `TIER_A_V1_CALL.md`: v1-reg **FAIL** (test 1.950 overfit). Freeze `v1/`. Iterate **v1.1** smaller/regularized; no v1-diff.
- New package `code/tier_a/v1_1/` (ElevCondResidualUNet base16 / dropout 0.20 / 167843 params). PhysicsNeMo 3D UNet not used.
- Config `configs/tier_a_v1_1.yaml`: **w_120=1.5 ≥ 1**; wd=1e-3; patience 50; composite ckpt rejects val +120h > raw 2.219971.
- Train CPU 400-ep cap, early-stop ep 283, best eligible ep 233, ~252 s. Outputs **only** `runs/phase0/tier_a/v1_1/`.
- Gates: val pooled **1.716 PASS** (<1.948661); test **1.812 PASS** (<1.850338); val +120h **2.015 PASS** (≤2.220, −0.205 vs raw). `tier_a_interim_pass=true`.
- Note: `docs/research/TIER_A_V1_1_NOTE.md`. FCN3 frozen. No diffusion. ERA5 PID 611595 / g0_verifying / tier0_holdout / v0 / v0.1 / **v1** untouched (md5).
- Next: ping Leonard — INTERIM PASS (all three gates). Do not start v1-diff. v0 remains best test-pooled (1.733).

### 2026-09-12 ~11:05 PT (Holdout IC expand v1 — Manisha / Leonard)

- **Manisha decision:** expand holdout ICs (thicken val/test). **No** year hard-lock. **No** diffusion.
- **Recipe (Leonard frozen):** `docs/research/HOLDOUT_IC_EXPAND_RECIPE.md` → `holdout_expand: v1`.
- **Years still provisional:** train 2018–2021 / val 2022 / test 2023–2024.
- **Counts:** keep train **8**; val **4→12**; test **4→12** (total 16→32). IDs ic01–ic16 kept; append **ic17–ic32**.
- **Seasons:** thin set was DJF+JJA only; expand adds **MAM+SON** (≥3/season in val and test).
- Config: `configs/tier0_holdout_ics.yaml` (+ `.bak_pre_expand`). Staging: `code/phase0/stage_tier0_holdout_ics.py --splits val,test` (no `--force`).
- Log: `logs/tier0_holdout_expand_ics.log` (**staging COMPLETE** 2026-09-12). Manifest: `data/tier0_ics/tier0_ic_manifest.json` with `holdout_expand: v1`; counts 8/12/12.
- **1980→latest ERA5 crop download untouched** (PID 611595). Do not disturb.
- **Next (parent / later phase — NOT this task):** build pairs for new ICs only; Tier-0 re-fit train ic01–ic08 + score full expand val/test; Tier-A v1.1 zero-shot re-score on expand (no retrain). No new Tier-A train until Leonard calls.


## Append — 2026-09-12 ~13:53 PT (Holdout expand COMPLETE)

**DONE (Leonard order):** pairs ic17–ic32 → Tier-0 re-fit train-only → Tier-A v1.1 **zero-shot** (frozen `v1_1/best_residual.pt`). No retrain. No diffusion. ERA5 PID **611595** untouched. Thin `tier0_holdout/` + `tier_a/v1_1/` untouched.

| Artifact | Path |
|----------|------|
| Expand pairs | `runs/phase0/tier0_holdout_expand/pairs/` (32 forecast + 32 targets) |
| Tier-0 metrics | `runs/phase0/tier0_holdout_expand/tier0_holdout_expand_metrics.json` |
| Tier-A zero-shot | `runs/phase0/tier_a/v1_1_expand/tier_a_v1_1_expand_results.json` |
| Config | `configs/tier_a_v1_1_expand.yaml` |
| CLI | `train.py --eval-only --pairs-dir --out-dir --ckpt` |

**Headline (expand 12+12):** Tier-0 val lin **1.987** / test lin **1.897**. v1.1 zero-shot val pooled **1.808** / test pooled **1.826** / val+120h **1.998** (vs new expand bars 1.987/1.897 / +120h 2.132). See Leonard `HOLDOUT_EXPAND_CALL.md` — beat-this **FROZEN**; v1.1 thick INTERIM PASS **CONFIRM YES**.


## Append — 2026-09-12 ~14:00 PT (Leonard HOLDOUT_EXPAND_CALL — FROZEN)

**Authoritative call:** `docs/research/HOLDOUT_EXPAND_CALL.md`

| Item | Call |
|------|------|
| Expand protocol (8/12/12) | **PASS** |
| **New beat-this** | **FROZEN** — supersedes thin for all future Tier-A |
| Primary | val strictly **< 1.987014 K** |
| Secondary | test strictly **< 1.897298 K** |
| +120 h hard (v1.x) | val +120h ≤ **expand-val raw 2.131516** (not thin 2.220) |
| **v1.1 INTERIM PASS on thick set** | **CONFIRM YES** (zero-shot 1.808 / 1.826 / +120h 1.998) |
| Thin bars (1.948661 / 1.850338 / 2.220) | **historical only** |
| New Tier-A train / diffusion | **NO** unless Manisha orders |

Eng: templates/JSON patched to echo expand bars (`configs/tier_a_v1_1_expand.yaml`, `train.py` defaults when expand, `tier_a_v1_1_expand_results.json`). Thin artifacts untouched. ERA5 PID 611595 untouched. No retrain.


## Append — 2026-09-12 ~21:30 PT (Tier-A v1.2 CPU scaffold)

**Leonard:** CPU scaffold only. **Manisha has NOT confirmed v1.2 train.** No `--train`, no `--allow-gpu`. ERA5 PID **611595** untouched.

| Item | Path / note |
|------|-------------|
| Code | `code/tier_a/v1_2/{train,model,dataset,README}.py` (from v1.1; same arch base16 / drop0.20 / w_120=1.5 composite) |
| Config | `configs/tier_a_v1_2.yaml` — `variant: v1_2`; expand pairs; expand beat-this |
| pairs_dir | `runs/phase0/tier0_holdout_expand/pairs` |
| Beat-this | val **< 1.987014**; test **< 1.897298**; +120h **≤ 2.131516** |
| Splits | train ic01–08; val ic09–12+ic17–24; test expand 12 |
| out_dir | `runs/phase0/tier_a/v1_2/` only |
| Dry-run | **OK** → `runs/phase0/tier_a/v1_2/tier_a_v1_2_dry_run.json` |
| Frozen | `v0/` `v0_1/` `v1/` `v1_1/` `v1_1_expand/` untouched |

**Ready-for-train (ONLY after Manisha confirms):**
`PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_2/train.py --config configs/tier_a_v1_2.yaml --train --epochs 400`



## Append — 2026-09-12 ~21:42 PT (Tier-A v1.2b GPU train — INTERIM PASS)

**Leonard** `TIER_A_V1_2_CALL.md`: v1.2 **FAIL** (val +120h 2.132014 / n_eligible_saves=0 / unconstrained fallback). Iterate **v1.2b** (no diffusion).

| Knob | v1.2 | **v1.2b** |
|------|------|-----------|
| w_120 | 1.5 | **2.0** |
| early_stop_patience | 50 | **80** |
| PASS lock | composite | **n_eligible_saves ≥ 1** required; unconstrained fallback ⇒ auto FAIL |

| Item | Path / note |
|------|-------------|
| Code | `code/tier_a/v1_2b/` (from v1.2; same ElevCondResidualUNet base16 / drop0.20) |
| Config | `configs/tier_a_v1_2b.yaml` |
| out_dir | `runs/phase0/tier_a/v1_2b/` only |
| Train | GPU `--allow-gpu` 400-ep cap; early-stop ep **284**; best eligible ep **204**; **n_eligible_saves=17**; reload=`composite_eligible`; ~24 s |
| PID | 775901 (finished) |
| Results | `runs/phase0/tier_a/v1_2b/tier_a_v1_2b_results.json` |

**Gates (expand bars):**
| Gate | Result |
|------|--------|
| Val pooled < 1.987014 | **PASS** — **1.809** |
| Test pooled < 1.897298 | **PASS** — **1.876** |
| Val +120h ≤ 2.131516 | **PASS** — **1.998** (Δ −0.134 K) |
| n_eligible_saves ≥ 1 | **PASS** — **17** |
| **tier_a_interim_pass** | **true** |

Frozen untouched: `v0/` `v0_1/` `v1/` `v1_1/` `v1_1_expand/` **`v1_2/`** (md5 unchanged). ERA5 PID **611595** untouched. No diffusion.

## Unlock A (thick-2) — COMPLETE (2026-09-13 ~14:13 PT)

- Counts: **train 12 / val 16 / test 16** (`holdout_expand: v2`). ICs **ic33–ic44** staged ARCO global.
- Pairs: `runs/phase0/tier0_holdout_expand_v2/pairs/` (44 forecast + 44 targets).
- Tier-0: `runs/phase0/tier0_holdout_expand_v2/tier0_holdout_expand_v2_metrics.json` (val lin **1.988588** / test lin **1.918383** / val+120h raw **2.178842**).
- v1.2b zero-shot: `runs/phase0/tier_a/v1_2b_thick2/tier_a_v1_2b_thick2_results.json` (val **1.829** / test **1.897** / +120h **2.015**; frozen ckpt untouched).
- OLD expand metrics + `tier_a/v1_2b/` weights frozen. ERA5 PID **611595** untouched. No diffusion / no new Tier-A train.
- **Next:** Leonard freezes thick-2 bars + PASS call.


## Append — 2026-09-13 ~14:20 PT (thick-2 call CONFIRM YES — idle)

**Leonard** `HOLDOUT_THICK2_CALL.md`: Unlock A complete. Living beat-this = **thick-2** (supersedes expand-v1).

| Item | Value |
|------|-------|
| Set | train **12** / val **16** / test **16** · holdout_expand **v2** · years HARD-locked |
| Primary val | strictly **< 1.988588** |
| Secondary test | strictly **< 1.918383** |
| +120h hard | ≤ **2.178842** (thick-2 val raw) |
| Expand-v1 | historical only (1.987014 / 1.897298 / 2.131516) |
| v1.2b thick-2 ZS | **INTERIM PASS CONFIRM YES** — 1.829 / 1.897 / 2.015 |
| Soft note | test SON=3 (not FAIL) |
| Eng | echo thick-2 bars in living configs/JSON; **idle** on new arch/diffusion/train until Manisha |
| CDS | PID **611595** leave alone |



## Append — 2026-09-14 ~10:19 PT (v1.2b thick-2 retrain — freeze + GPU)

**Leonard** `TIER_A_V1_2B_THICK2_TRAIN_RECIPE.md` FROZEN. Manisha greenlit. CPU dry-run **OK** → GPU `--train --allow-gpu --epochs 400`.

| Item | Value |
|------|-------|
| Config | `configs/tier_a_v1_2b_thick2_train.yaml` |
| Code | `code/tier_a/v1_2b_thick2_train/` |
| Out | `runs/phase0/tier_a/v1_2b_thick2_train/` |
| Arch | ElevCondResidualUNet base=16 drop=0.20 · w_120=2.0 · patience=80 · seed=42 |
| Data | holdout_expand=v2 · train12/val16/test16 · bars_set=thick2 |
| Bars | val **&lt; 1.988588** / test **&lt; 1.918383** / +120h **≤ 2.178842** |
| PASS | all three + n_eligible_saves≥1 (fallback=auto FAIL) |
| Frozen | `v1_2b/` · `v1_2b_thick2/` ZS · no overwrite |
| CDS | PID **611595** leave alone · no diffusion |

**Leonard ping path:** `runs/phase0/tier_a/v1_2b_thick2_train/tier_a_v1_2b_thick2_train_results.json`


## Append — 2026-09-14 ~10:21 PT (thick-2 retrain DONE — ping Leonard)

GPU train finished. Mechanical **tier_a_interim_pass=true** (val 1.760 / test 1.780 / +120h 1.962 / n_eligible=52 / composite_eligible). Formal call: Leonard `TIER_A_V1_2B_THICK2_TRAIN_CALL.md`.

**Results JSON:** `runs/phase0/tier_a/v1_2b_thick2_train/tier_a_v1_2b_thick2_train_results.json`  
Frozen `v1_2b/` + `v1_2b_thick2/` md5 unchanged. CDS **611595** untouched. No diffusion.


## Append — 2026-09-14 ~10:22 PT (thick-2 retrain INTERIM PASS CONFIRM — living residual)

Leonard `TIER_A_V1_2B_THICK2_TRAIN_CALL.md` mirrored. **INTERIM PASS CONFIRM YES**.

| Item | Value |
|------|-------|
| Living residual | `runs/phase0/tier_a/v1_2b_thick2_train/` (`best_residual.pt`) |
| Call | val **1.760** / test **1.780** / +120h **1.962** / eligible **52** |
| ZS demoted | `v1_2b_thick2/` → historical (still frozen) |
| Prior expand v1.2b | `v1_2b/` → historical (still frozen) |
| Eng | **idle** pending Manisha · **no auto diffusion** · no GPU |
| CDS | PID **611595** leave alone |



## Append — 2026-09-14 ~12:32 PT (v1-diff CorrDiff-lite — NULL vs living)

| Item | Value |
|------|-------|
| Recipe | `docs/research/TIER_A_V1_DIFF_RECIPE.md` |
| Out | `runs/phase0/tier_a/v1_diff/` |
| Arch | custom 2D **EDM** `ResidualEDM2D` (~111k) · not NeMo |
| Dry-run | EXIT 0 |
| Product | **NULL** · `diff_improves_residual=false` |
| Living | `v1_2b_thick2_train/` canonical · md5 unchanged |
| CDS | PID **611595** alive · untouched |
| Next | do **not** scale diffusion; await Leonard |


## Append — 2026-09-14 ~12:34 PT (v1-diff NULL call mirrored — idle)

Leonard `TIER_A_V1_DIFF_CALL.md` on Spark (fuller local). **Product NULL** — do not promote; stop leftover-target diffusion scaling.

| Item | Value |
|------|-------|
| Living residual | `runs/phase0/tier_a/v1_2b_thick2_train/` **canonical** |
| v1-diff | `runs/phase0/tier_a/v1_diff/` null artifact (keep) |
| Status | `docs/research/FCN_STATUS_AND_NEXT.md` — idle pending Manisha |
| Eng | **no GPU** · different Phase-3 fork only if assigned |
| CDS | PID **611595** leave alone |


## Append — 2026-09-15 ~22:10 PT (v1.3 joint winds+t2m — GPU train / ABC PASS)

Leonard ACK → GPU `--train --allow-gpu`. Parallel residual (not diffusion).

| Item | Value |
|------|-------|
| Out | `runs/phase0/tier_a/v1_3_joint/` |
| ckpt | `best_residual.pt` · results JSON with `abc_gates` |
| t2m val/test | 1.77008 / 1.77136 |
| wind-vector val/test | 0.69560 / 0.73877 (beat living + raw) |
| n_eligible | 46 · composite_eligible |
| mechanical upgrade | **PASS** (A+B+C) — await Leonard promote |
| Living | `v1_2b_thick2_train/` still canonical · md5 `aa0ed74a…` |
| CDS | PID **611595** alive · untouched |


## Append — 2026-09-15 ~22:14 PT (v1.3 joint INTERIM PASS CONFIRM — PROMOTE living)

Leonard `TIER_A_V1_3_JOINT_CALL.md` mirrored. **INTERIM PASS CONFIRM YES · PROMOTE**.

| Item | Value |
|------|-------|
| **NEW living residual** | `runs/phase0/tier_a/v1_3_joint/` (`best_residual.pt`) |
| Prior living | `v1_2b_thick2_train/` → historical (frozen; md5 `aa0ed74a…` unchanged) |
| Call headlines (16-IC) | val **1.770** / test **1.775** / +120h **1.982** · wind **0.69560** / **0.73877** |
| results patch | drop ic45 from `test_ids` echo; sidecar claim JSON; no retrain |
| Eng | **idle** pending Manisha · no GPU · CDS **611595** leave alone |

## Append — 2026-09-25 evening PT (FINAL residual v0 PASS — PROMOTE living)

Leonard `FINAL_RESIDUAL_V0_CALL.md` on Spark (frozen; do not rewrite). **PASS A∧B∧C · PROMOTE**.

| Item | Value |
|------|-------|
| **NEW living residual (FINAL protocol)** | `runs/phase0/final_eval/final_residual_v0/` (`best_residual.pt` · md5 `586ab17b843757bb84e66e2e3af8dc01`) |
| Prior interim living | `runs/phase0/tier_a/v1_3_joint/` → **historical interim** (frozen; md5 `c81a5a4c4f07ca0a52bb0bffee0045fa` unchanged) |
| Headlines (FINAL 320/64/64) | val t2m **1.8199** / test **1.8564** / +120h **2.0104** · WV **0.8004 / 0.8250** |
| Claim | PASS A∧B∧C · `final_g1_candidate_pass=YES` · **`g1_claimable=true`** (Leonard unlock 2026-09-27 · regional FINAL G1 candidate; hard non-claims apply) |
| Thick-2 bridge | WARN-missing from results · non-blocking |
| Eng | status-echo only this pass · no GPU · no weight writes · idle pending Manisha |
