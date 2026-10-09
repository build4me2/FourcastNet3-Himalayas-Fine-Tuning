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
