# 03 — Method and code

**Consolidation date:** 2026-09-27 PT  
**Sources:** `FINETUNE_METHOD_DESIGN.md`, `FINAL_TRAIN_RECIPE.md`, `FINAL_RESIDUAL_V0_CALL.md`, `FINAL_EVAL_BARS_CALL.md`, `FINAL_EVAL_PROTOCOL.md`, run `STATUS.md`, Spark tree listing.

## 1. Living method (FINAL residual v0)

| Field | Value |
| --- | --- |
| Backbone | **Frozen** FourCastNet3 |
| Residual | **ElevCondResidualUNet** (elevation-conditioned) |
| Channels / joint | t2m + u10m + v10m |
| Residual style | **Parallel residual** (explicitly **not** leftover-diff) |
| `channel_weights` | `[1.5, 1.5, 1.5]` |
| `w_120` | `2.0` |
| Other knobs (STATUS echo) | base=16 · depth=3 · dropout=0.20 · elev_boost=1.5 · weight_decay=1e-3 · patience 80 · epochs 400 · seed 42 |

(source: `FINAL_RESIDUAL_V0_CALL.md`, `runs/phase0/final_eval/final_residual_v0/STATUS.md`)

### What this is / is not

- **Is:** Tier-A-style regional residual on top of frozen FCN3 (RRCA-FD Plan A family).
- **Is not:** FCN3 weight fine-tune; not CorrDiff parity; not leftover-target diffusion (that path was NULL — see history).

(source: `FINETUNE_METHOD_DESIGN.md`, `STATUS_WHAT_WORKS_WHAT_FAILED.md`)

## 2. Training run outcome (distinguish from recipe plan)

| Field | Outcome |
| --- | --- |
| `status` | `train_ok` |
| Early-stop epoch | **135** |
| Best eligible epoch | **55** |
| `n_eligible_saves` | **19** |
| Reload | `composite_eligible` |
| Wall clock | ~**399 s** |
| Created | **2026-09-25 ~16:25 PT** |
| Checkpoint | `best_residual.pt` |
| MD5 | **`586ab17b843757bb84e66e2e3af8dc01`** |

Composite reject rule (STATUS): reject if val +120 h > raw A3 **2.2488412332039327**.  
(source: `FINAL_RESIDUAL_V0_CALL.md`, `STATUS.md`)

## 3. Evaluation method

### Metrics (locked with bars)

| Metric | Definition |
| --- | --- |
| t2m pooled | Grid-pooled t2m RMSE over gate leads {24,72,120} |
| t2m +120 h | Per-lead +120 h grid-pooled t2m RMSE |
| Wind-vector (WV) | Lead-mean of per-lead √(mean((u_err²+v_err²)/2)) over {24,72,120} |

(source: `FINAL_EVAL_BARS_CALL.md`)

### Product

```text
final_g1_candidate_pass = A ∧ B ∧ C
```

- **A** vs raw FCN3 (incl. eligible ckpt discipline)  
- **B** vs Tier-0 t2m  
- **C** vs prior living residual scored on the **same FINAL ICs**

Independent verify used `metrics.*.rmse_tier_a` / `wind_vector.*.lead_mean` against bars floats (not only JSON pre-echoed pass flags). IC id lists byte-equal to locked protocol JSON.  
(source: `FINAL_RESIDUAL_V0_CALL.md`, `FINAL_EVAL_BARS_CALL.md`)

## 4. Code map (Spark `~/fourcastnet`)

| Path | Role |
| --- | --- |
| `code/phase0/` | Env / norms / inference smoke / G0 helpers |
| `code/tier_a/v0` … `v1_3_joint/` | Historical Tier-A train/eval trees |
| `code/tier_a/v1_diff/` | Leftover-diff NULL experiment (keep) |
| `code/final_eval/final_residual_v0/` | Living FINAL residual train/eval |
| `code/final_eval/tests/` | FINAL scorer unit tests |
| `configs/final_residual_v0.yaml` | Living FINAL train config |
| `configs/phase0_nepal_box.yaml` | Box lock |
| `configs/tier_a_*.yaml` | Historical Tier-A configs |
| `scripts/` | Pipeline helpers (e.g. FINAL residual supervisor) |

(source: Spark directory listing 2026-09-27)

## 5. Artifacts (weights not in git)

| Artifact | Location |
| --- | --- |
| Living weights | `runs/phase0/final_eval/final_residual_v0/best_residual.pt` |
| Results JSON | `.../final_residual_v0_results.json` |
| Metrics CSV | `.../final_residual_v0_metrics.csv` |
| Thick-2 bridge JSON | `.../thick2_bridge_results.json` (report-only) |
| Public HF | https://huggingface.co/build4me2/fcn3-himalayas-final-residual-v0 (Apache-2.0; `base_model: nvidia/fourcastnet3`, adapter) |

Prior interim weights: `runs/phase0/tier_a/v1_3_joint/` · md5 **`c81a5a4c4f07ca0a52bb0bffee0045fa`** · **do not overwrite**.  
(source: `LIVING_INDEX.md`, `HF_RELEASE_FINAL_RESIDUAL_V0.md`, `FINAL_RESIDUAL_V0_CALL.md`)

## 6. Recipe references

Authoritative training/eval details for the living FINAL residual are summarized in this file and in `docs/04_RESULTS.md` / `docs/05_IMPLEMENTATION_HISTORY.md`. Historical call cards (`FINAL_TRAIN_RECIPE`, `FINAL_EVAL_PROTOCOL`, `FINAL_RESIDUAL_V0_CALL`, `TIER_A_V1_3_JOINT_RECIPE`, etc.) were removed from git after consolidation; keep local/Spark backups if needed for provenance.

## 7. One-command runner (`fcn3_himalayas/`, release v0.1.0)

Installable package (`pyproject.toml`, console script `fcn3-himalayas`, also `python -m fcn3_himalayas`). Install + command: see README **Quickstart**.

```bash
fcn3-himalayas forecast --init 2024-07-01T00 --lead 120 --out out.nc
```

**Pipeline — reuses the FINAL eval logic, not a re-implementation:**

| Step | Runner (`fcn3_himalayas/pipeline.py`) | Eval source it mirrors |
| --- | --- | --- |
| FCN3 load | Earth2Studio `FCN3.load_model(FCN3.load_default_package())` → `hf://nvidia/fourcastnet3` (pinned revision inside Earth2Studio) | `code/phase0/g0_base.py::load_fcn3` (same class; eval used a local copy of the same package) |
| IC | `earth2studio.data.ARCO` → 72-channel global field (1,1,1,72,721,1440), lat 90→−90, lon 0–359.75 | `code/data/stage_g0_ics.py` (FINAL ICs staged from ARCO); `g0_base.py::build_ic_tensor` |
| Rollout | `set_rng(seed=333, reset=True)`, `create_iterator`, 6 h/step, single member | `code/phase0/tier0_crop_rollout.py` (FINAL pairs: 1 member, seed 333) |
| Crop | inclusive 26–31°N, 80–89°E → 21×37 | `code/phase0/inference_smoke.py::crop_nepal` |
| Adapter input | `[t2m, u10m, v10m]` raw physical units + `elev_norm` (box elevation z-scored over the box) + `lead_norm=(h−24)/96` | `code/tier_a/v1_3_joint/dataset.py` |
| Correction | `corrected = raw + ElevCondResidualUNet(x)` (adapter run on CPU fp32) | `code/tier_a/v1_3_joint/train.py` eval loop |

`fcn3_himalayas/adapter_model.py` is a verbatim copy of `code/tier_a/v1_3_joint/model.py`; `fcn3_himalayas/data/box_elevation.npz` is `elevation_m` from `data/masks/w_R_elevation.nc` (derived from FCN3 `orography.nc`).

### 7.1 Spark verification (2026-10-09, FINAL **test**-split IC `ic_20230110_00`, leads 24–120 h)

- **End-to-end default path** (`fcn3-himalayas forecast --init 2023-01-10T00 --lead 120`, FCN3 from `hf://nvidia/fourcastnet3`, IC from ARCO, adapter from HF): **PASS**. The ARCO IC is byte-identical to the staged FINAL-eval IC (max abs diff 0.0).
- **Parity vs FINAL eval pairs:** raw FCN3 crop identical (max abs diff 0 for t2m/u10m/v10m at every lead); corrected max abs diff ≤ 3e-5 K (t2m), 0 m/s (winds) vs the eval adapter path.
- **Same IC vs ERA5 truth (single IC, illustrative only):** t2m RMSE raw→corrected 2.132→2.012 (+24 h), 2.241→1.777 (+72 h), 3.099→2.440 K (+120 h), identical to the eval pipeline.
- **Cost (GB10, shared with another job):** total ≈ 6.7–7 min wall (FCN3 load ≈ 2 min, ARCO IC ≈ 1.3 min, 20 FCN3 steps ≈ 3.5 min); peak CUDA allocation 49.0 GiB; host RSS ≈ 9.9 GB.
- **Colab T4 (16 GB):** does not fit at the measured 49 GiB peak; reduced-memory modes were not tested.
- **Caveat — fresh venv:** `pip install` into a clean venv succeeded, but pip's `torch-harmonics` wheel has no CUDA DISCO extension (Earth2Studio warns "FCN3 run on GPU/CUDA will be slower"); on Spark that FCN3 load stalled for >20 min and was stopped. The verified run used the project env that has the extension. Build it with `FORCE_CUDA_EXTENSION=1 pip install --no-build-isolation torch-harmonics`.
- **Caveat — ARCO in-process:** the IC is fetched *before* FCN3 is loaded; fetching after the load hung on Spark (gcsfs/aiohttp).

No change to FINAL numbers, bars, protocol or weights.
